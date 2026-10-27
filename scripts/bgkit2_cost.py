#!/usr/bin/env python
"""End-to-end cost of reading a long context: bgkit (encoder pass + a decoder prefill over the
budget) against the original decoder prefilling the whole context. Uses the released model
(release/huggingface/modeling_bgkit.py, or the Hub repo) so the numbers describe what a user runs.

For each context length L (a real long text cut to L tokens) and budget B, per question:
  bgkit   encoder seconds (chunks of chunk_tokens, compressed to ~B vectors in all), decoder
          prefill seconds over the slotted prompt, seconds per generated token, peak CUDA memory
  full    the original decoder (LFM2.5-350M) prefilling the whole context as text, seconds per
          generated token, peak CUDA memory
Medians over --repeats runs after one warm-up. Also reports the analytic forward FLOPs
(2 * params * tokens) of each path.

    python scripts/bgkit2_cost.py --model gabrielpickard/bgkit-350m --base LiquidAI/LFM2.5-350M \\
        --text-items .../longmemeval_s.jsonl --lengths 4096,16384,32768,65536,110000 \\
        --budgets 1024,2048 --out results/cost.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

SYSTEM_QA = "You answer questions about the provided context. Be brief: give only the answer."
QUESTION = "What did the user say about their plans for the weekend?"


def _sync() -> None:
    torch.cuda.synchronize()


def _peak_reset() -> int:
    _sync()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


def _peak(base: int) -> float:
    _sync()
    return (torch.cuda.max_memory_allocated() - base) / 2**30


@torch.no_grad()
def decode_rate(model, x: torch.Tensor, n_new: int) -> tuple[float, float]:
    """(prefill seconds, seconds per generated token) for inputs_embeds ``x`` under greedy decoding
    with the KV cache: one forward over x, then n_new single-token steps."""
    _sync()
    t0 = time.perf_counter()
    out = model(inputs_embeds=x, use_cache=True, logits_to_keep=1)
    _sync()
    t_pre = time.perf_counter() - t0
    past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
    emb = model.get_input_embeddings()
    t1 = time.perf_counter()
    for _ in range(n_new):
        out = model(inputs_embeds=emb(nxt), past_key_values=past, use_cache=True)
        past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
    _sync()
    return t_pre, (time.perf_counter() - t1) / n_new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gabrielpickard/bgkit-350m")
    ap.add_argument("--base", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--text-items", required=True, help="benchmark jsonl; the longest item's text is cut")
    ap.add_argument("--lengths", default="4096,16384,32768,65536,110000")
    ap.add_argument("--budgets", default="1024,2048")
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cuda-fraction", type=float, default=0.25, help="the GPU may be shared")
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(args.cuda_fraction)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "release/huggingface"))
    from modeling_bgkit import Bgkit
    from transformers import AutoModelForCausalLM

    bg = Bgkit.from_pretrained(args.model, device="cuda")
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to("cuda").eval()
    tok, dev = bg.tok, bg.device
    n_enc = sum(p.numel() for p in bg.encoder.parameters())
    n_dec = sum(p.numel() for p in bg.decoder.parameters())
    items = [json.loads(ln) for ln in open(args.text_items)]
    text = "\n\n".join(max(items, key=lambda it: sum(map(len, it["segments"])))["segments"])
    all_ids = tok(text, add_special_tokens=False)["input_ids"]
    lengths = [n for n in map(int, args.lengths.split(",")) if n <= len(all_ids)]
    rows = []

    def run_bgkit(ids: list[int], budget: int) -> dict:
        b0 = _peak_reset()
        _sync()
        t0 = time.perf_counter()
        chunks = bg._chunks([tok.decode(ids)])
        total = sum(len(c) for c in chunks)
        ratio = min(max(total / budget, 1.0), bg.cfg["max_compression"])
        slots = [bg._encode(c, QUESTION, bg.encoder.ratio_for(len(c), max(4, round(len(c) / ratio))))
                 for c in chunks]
        _sync()
        t_enc = time.perf_counter() - t0
        msgs = [{"role": "system", "content": SYSTEM_QA},
                {"role": "user", "content": "Context:\n" + "\n".join([bg.sentinel] * len(slots))
                 + f"\n\nQuestion: {QUESTION}"}]
        p_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        p_ids = p_ids if isinstance(p_ids, list) else p_ids["input_ids"]
        emb = bg.decoder.get_input_embeddings()
        cuts = [i for i, t in enumerate(p_ids) if t == bg.sentinel_id]
        parts, start = [], 0
        for c, s in zip(cuts, slots, strict=True):
            parts += [emb(torch.tensor(p_ids[start:c], device=dev)), s.to(dev, emb.weight.dtype)]
            start = c + 1
        parts.append(emb(torch.tensor(p_ids[start:], device=dev)))
        x = torch.cat(parts)[None]
        t_pre, t_tok = decode_rate(bg.decoder, x, args.new_tokens)
        return {"enc_s": t_enc, "prefill_s": t_pre, "per_token_s": t_tok, "positions": x.shape[1],
                "peak_gb": _peak(b0)}

    def run_full(ids: list[int]) -> dict:
        b0 = _peak_reset()
        msgs = [{"role": "system", "content": SYSTEM_QA},
                {"role": "user", "content": "Context:\n" + tok.decode(ids) + f"\n\nQuestion: {QUESTION}"}]
        p_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        p_ids = p_ids if isinstance(p_ids, list) else p_ids["input_ids"]
        with torch.no_grad():
            x = base.get_input_embeddings()(torch.tensor(p_ids, device=dev))[None]
        t_pre, t_tok = decode_rate(base, x, args.new_tokens)
        return {"enc_s": 0.0, "prefill_s": t_pre, "per_token_s": t_tok, "positions": x.shape[1],
                "peak_gb": _peak(b0)}

    def median(fn, *a) -> dict:
        fn(*a)  # warm-up
        runs = [fn(*a) for _ in range(args.repeats)]
        return {k: statistics.median(r[k] for r in runs) for k in runs[0]}

    with torch.autocast("cuda", dtype=torch.bfloat16):
        for n in lengths:
            ids = all_ids[:n]
            r = median(run_full, ids)
            r.update(arm="full", length=n, budget=None, flops=2 * n_dec * r["positions"])
            rows.append(r)
            print(json.dumps(r), flush=True)
            for b in map(int, args.budgets.split(",")):
                if b >= n:
                    continue
                r = median(run_bgkit, ids, b)
                r.update(arm="bgkit", length=n, budget=b,
                         flops=2 * n_enc * n + 2 * n_dec * r["positions"])
                rows.append(r)
                print(json.dumps(r), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"encoder_params": n_enc, "decoder_params": n_dec,
                                          "new_tokens": args.new_tokens, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
