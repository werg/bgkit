#!/usr/bin/env python
"""Stage D's constraint, measured: encode a WHOLE repository as a frozen tree.

The multi-level design exists because a repo does not fit one encoder pass, and
its end state keeps the encoder tree frozen and out of the optimizer. Nothing
had measured the inference-side footprint of that tree. This encodes N files of
T tokens at depth 1/2/3 under no_grad and reports peak CUDA memory, wall time,
and root rep count, so the depth cap and file cap are numbers, not guesses.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--experiment", required=True,
                    help="experiment the checkpoint was trained with: the models are built and "
                    "loaded by the trainer's code (bgkit2.training.standalone.load_models), so an "
                    "encoder-adapter checkpoint loads WITH its adapter and nothing is dropped")
    ap.add_argument("--cuda-fraction", type=float, default=0.5,
                    help="per-process CUDA cap: a runaway raises here, not a host OOM")
    ap.add_argument("--files", type=int, nargs="+", default=[24, 96, 200])
    ap.add_argument("--file-tokens", type=int, default=512)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--ratio-leaf", type=float, default=0.125)
    ap.add_argument("--ratio-node", type=float, default=0.25)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from bgkit2.models.multilevel import encode_tree
    from bgkit2.training.standalone import load_models
    from bgkit_core.host_memory_guard import arm, cap_cuda

    arm(10.0)
    cap_cuda(args.cuda_fraction)
    dev = "cuda"
    # the run's own models (a hand-built encoder loaded strict=False silently
    # dropped adapter tensors, 2026-09-18 / 2026-09-22)
    core = load_models(args.experiment, args.checkpoint, device=dev)
    enc = core.encoder
    for p in enc.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(core.cfg2.model.encoder_id)
    enc_tok = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731

    g = torch.Generator().manual_seed(0)
    rows = []
    for n in args.files:
        # a plausible tree: ~sqrt(n) top-level dirs, each with a sub-dir
        paths = [f"pkg{i % max(1, int(n ** 0.5))}/sub{i % 3}/mod{i}.py" for i in range(n)]
        files = [torch.randint(3, 60000, (args.file_tokens,), generator=g) for _ in range(n)]
        for d in args.depths:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            # the trainer's role rule: an adapter checkpoint encodes small trees
            # through its adapter, large ones on the frozen base path
            with torch.no_grad(), enc.role(core.tree_role(n)), core.autocast():
                out = encode_tree(enc, files, paths, tok_encode=enc_tok,
                                  ratio_leaf=args.ratio_leaf, ratio_node=args.ratio_node,
                                  max_depth=d, device=dev)
            torch.cuda.synchronize()
            r = {
                "files": n, "tokens": n * args.file_tokens, "depth": d,
                "realised_depth": out.depth, "nodes": out.n_nodes,
                "root_reps": int(out.root.k.sum()),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "secs": round(time.perf_counter() - t0, 2),
            }
            rows.append(r)
            print(json.dumps(r), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
