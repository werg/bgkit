#!/usr/bin/env python
"""LongLLMLingua competitor for scripts/bgkit2_bench.py: question-aware token pruning of each item's
context to the reader's budget (Jiang et al., 2024), with the method's recommended settings
(question-conditioned coarse ranking of passages, dynamic per-passage ratios, condition_compare,
passage reordering off so the document order is kept). The scoring model is a small causal LM
(default Qwen3-1.7B, larger than the bgkit encoder). Output: the summaries jsonl schema
(``id``, ``budget``, ``summary``, cost fields), read by bgkit2_bench.py via
``--precomputed lingua=<path>``. The result is cut to the budget in the READER's tokens, so the
arm never sees more decoder positions than the others.

Needs ``llmlingua`` with transformers 4.x (its iterative compression indexes tuple KV caches), so
it runs in its own environment, apart from bgkit2's transformers 5.

    python scripts/bgkit2_bench_lingua.py --items .../hotpot.jsonl --budgets 256,512,1024 \\
        --out .../summaries/hotpot.lingua.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def passages(tok, segments: list[str], max_tokens: int) -> list[str]:
    """Segments split into pieces of <= max_tokens scorer tokens (the coarse-ranking units)."""
    out = []
    for seg in segments:
        ids = tok(seg, add_special_tokens=False)["input_ids"]
        out.extend(tok.decode(ids[i : i + max_tokens]) for i in range(0, len(ids), max_tokens))
    # stripped: llmlingua tokenises "<prefix>\n\n<passage>" and asserts the token boundary falls
    # at the join, which fails when a passage's leading whitespace merges with the newlines
    return [p.strip() for p in out if p.strip()]


def legacy_cache_adapter(model) -> None:
    """llmlingua trims and re-feeds the KV cache as nested (key, value) lists; current models only
    take Cache objects. Convert at the model boundary in both directions; the algorithm is untouched."""
    from transformers.cache_utils import Cache, DynamicCache

    fwd = model.forward

    def forward(*a, past_key_values=None, **kw):
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(tuple(tuple(kv) for kv in past_key_values))
        out = fwd(*a, past_key_values=past_key_values, **kw)
        if isinstance(getattr(out, "past_key_values", None), Cache):
            out.past_key_values = out.past_key_values.to_legacy_cache()
        return out

    model.forward = forward


def question(item: dict) -> str:
    q = item["question"]
    if item.get("options"):
        q += "\n" + "\n".join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(item["options"]))
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--budgets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--reader", default="LiquidAI/LFM2.5-350M", help="tokenizer the budget counts in")
    ap.add_argument("--passage-tokens", type=int, default=512)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from llmlingua import PromptCompressor
    from tokenizers import Tokenizer

    comp = PromptCompressor(model_name=args.model, device_map="cuda")
    legacy_cache_adapter(comp.model)
    # the reader's tokenizer through `tokenizers` directly: its config targets transformers 5,
    # while llmlingua needs transformers 4 (tuple KV caches)
    rtok = Tokenizer.from_file(hf_hub_download(args.reader, "tokenizer.json"))

    def reader_ids(text: str) -> list[int]:
        return rtok.encode(text, add_special_tokens=False).ids
    items = [json.loads(ln) for ln in open(args.items)]
    if args.n:
        items = items[: args.n]
    budgets = [int(b) for b in args.budgets.split(",")]
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {(d["id"], d["budget"]) for d in map(json.loads, out.read_text().splitlines()) if d}
    with out.open("a") as fh:
        for n, it in enumerate(items):
            ctx = passages(comp.tokenizer, it["segments"], args.passage_tokens)
            full = "\n\n".join(it["segments"])
            ctx_tokens = len(reader_ids(full))
            for b in budgets:
                if (it["id"], b) in done:
                    continue
                t0 = time.time()
                if ctx_tokens <= b:
                    text = full
                else:
                    # the target is in the scorer's tokens and only approximately met: shrink it
                    # until the result fits the budget in the reader's tokens, so the final cut
                    # below (which would drop the last passages) is rarely needed
                    target = b
                    for _ in range(4):
                        res = comp.compress_prompt(
                            ctx, question=question(it), target_token=target,
                            rank_method="longllmlingua", condition_in_question="after_condition",
                            reorder_context="original", dynamic_context_compression_ratio=0.3,
                            condition_compare=True, context_budget="+100",
                            use_sentence_level_filter=False,
                        )
                        # the question is appended by compress_prompt; the reader gets it separately
                        text = res["compressed_prompt"].removesuffix(question(it)).rstrip()
                        n_out = len(reader_ids(text))
                        if n_out <= b:
                            break
                        target = max(16, int(target * b / n_out * 0.95))
                ids = reader_ids(text)
                if len(ids) > b:
                    text = rtok.decode(ids[:b])
                    ids = ids[:b]
                fh.write(json.dumps({"id": it["id"], "budget": b, "summary": text,
                                     "summary_tokens": len(ids), "ctx_tokens": ctx_tokens,
                                     "in_tokens": ctx_tokens, "out_tokens": 0,
                                     "seconds": round(time.time() - t0, 2)}) + "\n")
                fh.flush()
            if n % 10 == 0:
                print(f"[{n + 1}/{len(items)}]", flush=True)


if __name__ == "__main__":
    main()
