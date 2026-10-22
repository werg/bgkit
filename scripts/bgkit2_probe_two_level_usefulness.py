#!/usr/bin/env python
"""Are LEVEL-1 reps USABLE, or merely distinct?

The identity probe (bgkit2_probe_two_level.py) could not tell a trained bridge
from a random one: pretrained 0.938, random 0.938 on real code. A random
projection preserves document identity, so identity is the wrong question.

This asks the question that matters: given only the level-1 (repo-level) reps,
how well does the decoder predict the actual CONTENT of a member file? Three
arms, and the floor is not optional -- without it a low CE says nothing:

  zeroed      reps present but zeroed   (the floor: what the prompt alone buys)
  random      level-1 reps, RANDOM bridge
  pretrained  level-1 reps, M1's bridge

If pretrained ~ random, Stage A bought nothing usable and Stage B has no
foundation. If both ~ zeroed, the second level carries nothing at all and the
problem is upstream of the bridge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--experiment", required=True,
                    help="experiment the checkpoint was trained with: the models are built and "
                    "loaded by the trainer's code (bgkit2.training.standalone.load_models), so an "
                    "encoder-adapter checkpoint loads WITH its adapter and nothing is dropped")
    ap.add_argument("--cuda-fraction", type=float, default=0.5,
                    help="per-process CUDA cap: a runaway raises here, not a host OOM")
    ap.add_argument("--corpus", default="/home/werg/bgkit-data-nvme/bgkit2/corpus/code_eval")
    ap.add_argument("--docs", type=int, default=32)
    ap.add_argument("--files-per-doc", type=int, default=4)
    ap.add_argument("--file-tokens", type=int, default=256)
    ap.add_argument("--target-tokens", type=int, default=64)
    ap.add_argument("--ratio-lower", type=float, default=0.25)
    ap.add_argument("--ratio-upper", type=float, default=0.25)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from bgkit2.data.token_store import TokenStore
    from bgkit2.models.multilevel import encode_two_level
    from bgkit2.training.standalone import load_models

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        from bgkit_core.host_memory_guard import arm, cap_cuda

        arm(10.0)
        cap_cuda(args.cuda_fraction)
    # the run's own models, loaded strictly (adapters included) by the trainer's
    # code -- the old hand-built encoder + strict=False dropped adapter tensors
    core = load_models(args.experiment, args.checkpoint, device=dev)
    enc, dec = core.encoder, core.decoder
    trained = {k: v.clone() for k, v in enc.repro.state_dict().items()}

    store = TokenStore(args.corpus)
    n = args.docs * args.files_per_doc
    need = args.file_tokens + args.target_tokens
    rows = [torch.as_tensor(store[i][:need], dtype=torch.long) for i in range(n * 3)]
    rows = [r for r in rows if r.numel() >= need][:n]
    if len(rows) < n:
        raise SystemExit(f"need {n} docs of >= {need} tokens; got {len(rows)}")
    full = torch.stack(rows).to(dev)
    ids, tgt = full[:, : args.file_tokens], full[:, args.file_tokens :]
    mask = torch.ones_like(ids, dtype=torch.bool)
    sizes = [args.files_per_doc] * args.docs
    # Predict the continuation of each document's FIRST file from the repo-level reps.
    first = torch.arange(0, n, args.files_per_doc, device=dev)
    prefix = [torch.tensor([], dtype=torch.long, device=dev) for _ in range(args.docs)]
    suffix = [tgt[i] for i in first.tolist()]

    def ce(reps, rep_mask) -> float:
        with torch.no_grad(), core.autocast():
            b = dec.build_batch(prefix, reps, rep_mask, suffix)
            return float(dec.forward(b))

    out: dict[str, float] = {}
    for label in ("pretrained", "random"):
        if label == "random":
            for m in enc.repro.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
        # a group of files is a small tree: the trainer's tree role rule
        with torch.no_grad(), enc.role(core.tree_role(max(sizes))), core.autocast():
            r = encode_two_level(enc, ids, mask, ratio_lower=args.ratio_lower,
                                 ratio_upper=args.ratio_upper, group_sizes=sizes)
        out[label] = ce(r.top.reps, r.top.rep_mask)
        if label == "pretrained":
            out["zeroed"] = ce(torch.zeros_like(r.top.reps), r.top.rep_mask)
        print(f"{label:>11}: CE {out[label]:.4f}")
    enc.repro.load_state_dict(trained)
    print(f"{'zeroed':>11}: CE {out['zeroed']:.4f}   (floor)")
    out["gain_pretrained"] = out["zeroed"] - out["pretrained"]
    out["gain_random"] = out["zeroed"] - out["random"]
    print(f"gain over floor — pretrained {out['gain_pretrained']:+.4f} nats, "
          f"random {out['gain_random']:+.4f} nats")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
