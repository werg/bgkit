#!/usr/bin/env python
"""Is the fine-tuned decoder still a language model? (2026-08-25)

The plain-text BABILong probe put the wide-net decoder at 0.000 against the
stock model's 0.56 — but that comparison prompts a chat-tuned, heavily
fine-tuned model with a RAW text template, which is itself out of
distribution for it. This probe removes prompting from the question
entirely: plain next-token cross-entropy on held-out natural text.

Perplexity needs no template, no instruction-following and no task ability,
so a large gap is unambiguous evidence that the token-level language model
was damaged — not that the prompt was unfamiliar. It is the format-free
counterpart to the BABILong probe.

Motivation: v6's decoder differs from pristine Qwen3.5 mostly in ONE place —
``embed_tokens`` / ``lm_head`` (tied) have row-wise cosine 0.56 to pristine
and 0.80x the row norm, while the backbone layers moved a median of 0.08
relative. The spliced reps meanwhile sit at ~218x the (now shrunken)
embedding norm, against ~38x documented for this lineage. The hypothesis is
that the projection output drifted up in norm and the decoder absorbed the
mismatch by rewriting its token geometry — the representation-interface
failure class, silently paid for in general language ability.

Usage (GPU container):
    python scripts/probe_decoder_language_health.py \
      --checkpoint /workspace/checkpoints_fast/<v6> \
      --tokens-dir /workspace/data/fineweb_edu_v1/tokens --n-docs 64
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


def mean_ce(model, chunks: list[torch.Tensor]) -> float:
    total_nll = 0.0
    total_tok = 0
    with torch.no_grad():
        for ids in chunks:
            ids = ids.unsqueeze(0).cuda()
            out = model(input_ids=ids)
            logits = out.logits[:, :-1].float()
            targets = ids[:, 1:]
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                reduction="sum",
            )
            total_nll += float(nll.item())
            total_tok += int(targets.numel())
    return total_nll / max(total_tok, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint", required=True, action="append",
        help="repeat to compare several checkpoints in one pass (lineage sweep)",
    )
    ap.add_argument("--family", default="qwen35")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--tokens-dir", required=True, help="token shard dir (held-out text)")
    ap.add_argument("--n-docs", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    shards = sorted(Path(args.tokens_dir).glob("shard_*.parquet"))
    if not shards:
        raise SystemExit(f"no shards under {args.tokens_dir}")
    # Take from the LAST shard: the wide-net runs never touched this corpus,
    # but using the tail keeps it clearly disjoint from any earlier sampling.
    rows = pq.read_table(shards[-1], columns=["token_ids"]).to_pylist()
    chunks: list[torch.Tensor] = []
    for r in rows:
        arr = np.asarray(r["token_ids"], dtype=np.int64)
        if arr.size < args.seq_len:
            continue
        chunks.append(torch.from_numpy(arr[: args.seq_len]))
        if len(chunks) >= args.n_docs:
            break
    print(f"{len(chunks)} chunks x {args.seq_len} tokens", flush=True)

    results: dict[str, float] = {}
    for arm in ["stock", *args.checkpoint]:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, trust_remote_code=True
        )
        if arm != "stock":
            # Two on-disk layouts: the joint `model.pt` of the Phase-2 trainers
            # and the per-model `decoder_qwen.pt` of the summarization base.
            root = Path(arm)
            joint = root / "model.pt"
            solo = root / "decoder_qwen.pt"
            if joint.exists():
                sd = torch.load(
                    str(joint), map_location="cpu", mmap=True, weights_only=True)
                prefix = f"decoders.{args.family}.backbone."
                trimmed = {
                    k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)
                }
            elif solo.exists():
                sd = torch.load(
                    str(solo), map_location="cpu", mmap=True, weights_only=True)
                if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
                    sd = sd["model"]
                for pre in ("backbone.", "decoder.backbone.", ""):
                    cand = {
                        k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)
                    } if pre else dict(sd)
                    if "model.embed_tokens.weight" in cand:
                        trimmed = cand
                        break
                else:
                    raise SystemExit(f"cannot locate decoder tensors in {solo}")
            else:
                raise SystemExit(f"no model.pt or decoder_qwen.pt under {root}")
            missing, unexpected = model.load_state_dict(trimmed, strict=False)
            print(
                f"loaded {len(trimmed)} tensors "
                f"(missing={len(missing)}, unexpected={len(unexpected)})",
                flush=True,
            )
            if missing or unexpected:
                raise SystemExit("checkpoint/model key mismatch — probe would be invalid")
        model = model.cuda().eval()
        ce = mean_ce(model, chunks)
        label = arm if arm == "stock" else Path(arm).name
        results[f"{label}/ce"] = ce
        results[f"{label}/ppl"] = math.exp(min(ce, 20.0))
        print(f"{label}: CE {ce:.4f}  PPL {results[f'{label}/ppl']:.2f}", flush=True)
        del model
        torch.cuda.empty_cache()

    print("SUMMARY", json.dumps(results, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
