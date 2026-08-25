#!/usr/bin/env python
"""Did the decoder's operating point move to the reps' scale? (2026-08-25)

Established, in order:

- Wide-net Phase-2 training takes the decoder from PPL 31 (summarization
  base) to 671 (v6) to 2585 (v7) on held-out plain text.
- It is NOT the token embedding: base, v6 and v7 all carry essentially the
  same one (norm/pristine 0.798/0.797/0.797, cosine 0.557/0.561/0.561) and
  the base scores 31 with it. The rotation happened in Phase 1 and is benign.
- It is NOT raw weight distance: git-repro moved the backbone FURTHER from
  the same base (median drift 0.069 vs wide-net's 0.046) over 8700 steps and
  stayed at PPL 64.

What separates the two lineages is the SCALE of what the decoder reads. The
spliced reps sit at ~218x the token-embedding norm in wide-net (guard:
embed_norm_mean 0.5, rep_norm_mean 109) against ~4x in git-repro. If the
decoder re-centred on a residual stream whose informative positions are two
orders of magnitude larger than token positions, then ordinary text — where
every position is uniformly small — is the out-of-distribution input, and
that is a representation-interface fault, not forgetting.

This probe tests it directly: multiply the input embeddings by k and
re-measure plain-text CE. A deep minimum at k >> 1 means the decoder is
readable but mis-scaled — repairable by anchoring the projection output to
the embedding manifold (Phase 1 does exactly this in Step 2.5) rather than
by retraining. A flat curve means the damage is real and structural.

Usage (GPU container):
    python scripts/probe_decoder_input_scale.py \
      --checkpoint /workspace/checkpoints_fast/<v6> \
      --tokens-dir /workspace/data/fineweb_edu_v1/tokens
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


@torch.no_grad()
def ce_at_scale(model, chunks: list[torch.Tensor], scale: float) -> float:
    """Plain next-token CE with input embeddings multiplied by ``scale``.

    Feeds ``inputs_embeds`` so only the INPUT side is rescaled — the LM head
    still reads the untouched final hidden state, so this isolates "what
    magnitude does the stack expect to read" from any output-side effect.
    """
    embed = model.get_input_embeddings()
    total_nll = 0.0
    total_tok = 0
    for ids in chunks:
        ids = ids.unsqueeze(0).cuda()
        embeds = embed(ids) * scale
        logits = model(inputs_embeds=embeds).logits[:, :-1].float()
        targets = ids[:, 1:]
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum",
        )
        total_nll += float(nll.item())
        total_tok += int(targets.numel())
    return total_nll / max(total_tok, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, action="append")
    ap.add_argument("--family", default="qwen35")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--tokens-dir", required=True)
    ap.add_argument("--n-docs", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument(
        "--scales", nargs="+", type=float,
        default=[1.0, 4.0, 16.0, 40.0, 100.0, 218.0, 500.0],
    )
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    from bgkit.training.lm_health import load_decoder_tensors

    shards = sorted(Path(args.tokens_dir).glob("shard_*.parquet"))
    rows = pq.read_table(shards[-1], columns=["token_ids"]).to_pylist()
    chunks: list[torch.Tensor] = []
    for r in rows:
        arr = np.asarray(r["token_ids"], dtype=np.int64)
        if arr.size >= args.seq_len:
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
            trimmed = load_decoder_tensors(arm, args.family)
            missing, unexpected = model.load_state_dict(trimmed, strict=False)
            if missing or unexpected:
                raise SystemExit("checkpoint/model key mismatch — probe would be invalid")
        model = model.cuda().eval()
        label = arm if arm == "stock" else Path(arm).name[:44]
        best = (None, float("inf"))
        for s in args.scales:
            ce = ce_at_scale(model, chunks, s)
            ppl = math.exp(min(ce, 20.0))
            results[f"{label}/scale_{s:g}/ppl"] = ppl
            if ce < best[1]:
                best = (s, ce)
            print(f"  {label:46s} scale {s:7.1f}  PPL {ppl:10.2f}", flush=True)
        print(f"  -> {label}: best scale {best[0]:g} (PPL {math.exp(min(best[1], 20)):.2f})",
              flush=True)
        results[f"{label}/best_scale"] = float(best[0])
        del model
        torch.cuda.empty_cache()

    print("SUMMARY", json.dumps(results, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
