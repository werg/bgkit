#!/usr/bin/env python
"""Where does the decoder's language damage live — embedding or backbone?

Established 2026-08-25 by ``probe_decoder_language_health.py`` (plain
next-token CE on held-out FineWeb-Edu, no prompting):

    stock Qwen3.5-0.8B                    PPL   15.1
    summarization base (step 51945)       PPL   33.2
    wide-net v6   (2629 steps)            PPL  670.7
    wide-net v7   (+999 steps at 6x)      PPL 2585.2

So the base was healthy and Phase-2 wide-net training destroyed the decoder,
monotonically. The weights say the change is concentrated in ONE place:
``embed_tokens`` / ``lm_head`` (tied) sit at row-wise cosine 0.56 to pristine
and 0.80x the row norm, while the 318 backbone tensors moved a median of
0.079 relative.

This probe swaps the two halves to find which one carries the damage:

    stock embed + ckpt backbone   — recovers? then the embedding is the fault
    ckpt  embed + stock backbone  — recovers? then the backbone is the fault

The answer decides the fix. If it is the embedding, freezing (or heavily
down-LR-ing) ``embed_tokens``/``lm_head`` and restoring the Phase-1 Step-2.5
projection embed-anchor repair in Phase 2 is the targeted repair. If it is
the backbone, the training mix itself needs general-text replay.

Note the halves are NOT independent — a backbone adapted to a rotated
embedding may score badly with the pristine one purely from mismatch. Read
the two numbers together: only a half that recovers CLOSE to the base's 33
implicates the other half cleanly.

Usage (GPU container):
    python scripts/probe_decoder_damage_locus.py \
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

from bgkit.training.checkpointing import normalize_model_state

EMBED_KEYS = ("model.embed_tokens.weight", "lm_head.weight")


def load_decoder_tensors(root: Path, family: str) -> dict[str, torch.Tensor]:
    """Decoder state dict from any of the three on-disk layouts we ship."""
    joint = root / "model.pt"
    solo = root / "decoder_qwen.pt"
    src = joint if joint.exists() else solo
    if not src.exists():
        raise SystemExit(f"no model.pt or decoder_qwen.pt under {root}")
    sd = torch.load(str(src), map_location="cpu", mmap=True, weights_only=True)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd.get("model"), dict):
        sd = normalize_model_state(sd)["model"]
    for prefix in (f"decoders.{family}.backbone.", "decoder.backbone.", "backbone.", ""):
        cand = (
            {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if prefix else dict(sd)
        )
        if "model.embed_tokens.weight" in cand:
            return cand
    raise SystemExit(f"cannot locate decoder tensors in {src}")


def mean_ce(model, chunks: list[torch.Tensor]) -> float:
    total_nll = 0.0
    total_tok = 0
    with torch.no_grad():
        for ids in chunks:
            ids = ids.unsqueeze(0).cuda()
            logits = model(input_ids=ids).logits[:, :-1].float()
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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--family", default="qwen35")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--tokens-dir", required=True)
    ap.add_argument("--n-docs", type=int, default=48)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

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

    ckpt = load_decoder_tensors(Path(args.checkpoint), args.family)
    results: dict[str, float] = {}

    for arm in ("stock", "checkpoint", "stock_embed+ckpt_backbone", "ckpt_embed+stock_backbone"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, trust_remote_code=True
        )
        pristine = {k: v.clone() for k, v in model.state_dict().items() if k in EMBED_KEYS}
        if arm != "stock":
            load = dict(ckpt)
            if arm == "stock_embed+ckpt_backbone":
                load.update(pristine)
            elif arm == "ckpt_embed+stock_backbone":
                load = {k: v for k, v in ckpt.items() if k in EMBED_KEYS}
            _missing, unexpected = model.load_state_dict(load, strict=False)
            if unexpected:
                raise SystemExit(f"unexpected keys for {arm}: {list(unexpected)[:3]}")
        model = model.cuda().eval()
        ce = mean_ce(model, chunks)
        results[f"{arm}/ce"] = ce
        results[f"{arm}/ppl"] = math.exp(min(ce, 20.0))
        print(f"{arm:28s} CE {ce:7.4f}  PPL {results[f'{arm}/ppl']:10.2f}", flush=True)
        del model
        torch.cuda.empty_cache()

    print("SUMMARY", json.dumps(results, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
