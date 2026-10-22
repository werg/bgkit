#!/usr/bin/env python
"""Navigation from the ROOT encoding of a repo tree, with real controls.

Replaces bgkit2_probe_two_level_navigation.py, whose "+2.694 vs +1.368" result
had three defects: 58% of its questions stated the answer path, n was 24, and
its "random bridge" was an MLP re-initialised at std 0.02 -- a near-zero map,
not a random projection. This version:

  - reads repo_tree_v3 (0 leaky questions, verified; paths kept; sub-trees
    around the evidence file), encoding the repo with encode_tree so the
    decoder reads the ROOT and answers with a path;
  - runs FOUR arms per checkpoint on the same trees:
      zeroed      root reps present but zeroed        (floor)
      identity    no bridge: survivors fed straight to the next level
      random      a properly SCALED random projection (orthogonal, unit gain)
      trained     the checkpoint's repro head
  - n defaults to 120 trees.

CE is on the answer path's tokens given the question. Lower is better; report
gains over the zeroed floor, and read trained against BOTH controls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _orthogonal_bridge(d: int, device, dtype):
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(1)))
    q = q.to(device=device, dtype=dtype)

    def f(x: torch.Tensor) -> torch.Tensor:
        return x @ q.to(x.dtype)

    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="label=<experiment>:<checkpoint path> ...")
    ap.add_argument(
        "--store", default="/home/werg/bgkit-data-nvme/bgkit2/tasks/repo_tree_v3.parquet"
    )
    ap.add_argument("--split", default="eval", help="eval (repo-disjoint) or all")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--max-files", type=int, default=24)
    ap.add_argument("--ratio-leaf", type=float, default=0.125)
    ap.add_argument("--ratio-node", type=float, default=0.25)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--file-tokens", type=int, default=512, help="per-file cap (v4: 4096)")
    ap.add_argument("--leaf-chunk", type=int, default=16, help="max files per encoder call")
    ap.add_argument("--leaf-chunk-tokens", type=int, default=32768,
                    help="max padded tokens (files x longest) per encoder call")
    ap.add_argument("--node-max-tokens", type=int, default=8192,
                    help="directory inputs above this are split into virtual sub-directories")
    ap.add_argument("--cuda-fraction", type=float, default=0.5,
                    help="per-process CUDA cap: a runaway raises here, not a host OOM")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from bgkit2.models.multilevel import encode_tree
    from bgkit_core.host_memory_guard import arm, cap_cuda

    arm(10.0)            # the container's sitecustomize arms it too; idempotent
    cap_cuda(args.cuda_fraction)
    dev = "cuda"
    enc_tok = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-Encoder-350M")
    dec_tok = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-350M")
    tok_encode = lambda s: enc_tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731

    from bgkit2.data.repo_tree import RepoTreeDataset

    ds = RepoTreeDataset(args.store, args.split, max_files=args.max_files,
                         file_tokens=args.file_tokens)
    # Rows are grouped by repo (8 questions each): the first N rows are a
    # handful of repos. Take an even stride over the split so every repo --
    # the large ones in particular -- is represented.
    n = min(args.n, len(ds))
    pick = sorted({round(j * (len(ds) - 1) / max(n - 1, 1)) for j in range(n)})
    items = [ds[i] for i in pick]
    rows = [
        {"question": it.question, "answer": it.answer, "paths": it.paths,
         "file_ids": it.file_ids, "repo": it.repo}
        for it in items
    ]
    if not rows:
        raise SystemExit("no rows")
    # leak guard, again: the probe must not be able to copy the answer
    for r in rows:
        q, a = r["question"], r["answer"]
        base = a.rsplit("/", 1)[-1]
        assert a not in q and base not in q and base.rsplit(".", 1)[0] not in q, r["answer"]
    print(f"{len(rows)} clean trees from {args.store} [{args.split}]")

    def load(spec_path: str):
        """``<experiment>:<checkpoint>`` -- the models are built from the RUN'S
        OWN config by the trainer's code (bgkit2.training.standalone); a probe
        that hand-builds them measures a different model (2026-09-18)."""
        from bgkit2.training.standalone import load_models

        experiment, path = spec_path.split(":", 1)
        core = load_models(experiment, path, device=dev)
        return core.encoder, core.decoder, core

    results: dict[str, dict[str, float]] = {}
    for spec in args.checkpoints:
        label, path = spec.split("=", 1)
        enc, dec, core = load(path)
        d = enc.backbone.hidden_size
        arms = {
            "trained": None,
            "identity": (lambda x: x),
            "random": _orthogonal_bridge(d, dev, torch.float32),  # cast inside for bf16
        }
        sums = {k: 0.0 for k in [*arms, "zeroed", "noslot"]}
        n_tok = 0
        by_size: dict[str, dict[str, float]] = {}  # bucket -> arm sums + tokens
        n_files_tot = 0
        for ri, r in enumerate(rows):
            n_files_tot += len(r["paths"])
            if ri % 5 == 0:
                print(f"[{label}] tree {ri}/{len(rows)} files={len(r['paths'])} "
                      f"cuda_max={torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
            torch.cuda.empty_cache()
            files = [f.long() for f in r["file_ids"]]  # capped by the dataset
            paths = list(r["paths"])
            q_ids = dec_tok(f"{r['question']}\nWhich file? ", add_special_tokens=False)["input_ids"]
            prefix = [torch.tensor(q_ids, device=dev)]
            suffix = [torch.tensor(dec_tok(r["answer"], add_special_tokens=False)["input_ids"],
                                   device=dev)]
            nt = int(suffix[0].numel())
            n_tok += nt
            nf = len(paths)
            bucket = "<=24" if nf <= 24 else "25-100" if nf <= 100 else ">100"
            bs = by_size.setdefault(bucket, {"tok": 0.0, "trees": 0.0})
            bs["tok"] += nt
            bs["trees"] += 1
            # the run's own rule (an encoder-adapter checkpoint: small trees
            # through the adapter, the rest on the frozen base encoder)
            with torch.no_grad(), enc.role(core.tree_role(len(paths))), \
                    torch.autocast("cuda", dtype=torch.bfloat16):
                for arm_name, br in arms.items():
                    out = encode_tree(enc, files, paths, tok_encode=tok_encode,
                                      ratio_leaf=args.ratio_leaf, ratio_node=args.ratio_node,
                                      max_depth=args.max_depth, device=dev, bridge=br,
                                      leaf_chunk=args.leaf_chunk, grad_levels=0,
                                      leaf_chunk_tokens=args.leaf_chunk_tokens,
                                      node_max_tokens=args.node_max_tokens)
                    reps, mask = out.root.reps, out.root.rep_mask
                    sb = dec.build_batch(prefix, reps, mask, suffix)
                    ce_arm = float(dec.forward(sb)) * nt
                    sums[arm_name] += ce_arm
                    bs[arm_name] = bs.get(arm_name, 0.0) + ce_arm
                    if arm_name == "trained":
                        zb = dec.build_batch(prefix, torch.zeros_like(reps), mask, suffix)
                        ce_z = float(dec.forward(zb)) * nt
                        sums["zeroed"] += ce_z
                        bs["zeroed"] = bs.get("zeroed", 0.0) + ce_z
                        # NO SLOT at all: the question alone. Zero vectors are
                        # out-of-distribution for the decoder and can score
                        # WORSE than no slot, inflating "gain over zeroed"
                        # (2026-09-18: a random rotation between levels still
                        # "gained" 8 nats over zeroed). This is the honest floor.
                        nb = dec.build_batch(prefix, None, None, suffix)
                        ce_n = float(dec.forward(nb)) * nt
                        sums["noslot"] += ce_n
                        bs["noslot"] = bs.get("noslot", 0.0) + ce_n
        ce = {k: v / n_tok for k, v in sums.items()}
        gains = {k: ce["zeroed"] - ce[k] for k in arms}
        gains_noslot = {k: ce["noslot"] - ce[k] for k in arms}
        sized = {
            b: {"trees": int(v["trees"]),
                **{f"gain_{a}": (v["zeroed"] - v[a]) / v["tok"] for a in arms},
                "ce_identity": v["identity"] / v["tok"], "ce_zeroed": v["zeroed"] / v["tok"],
                "ce_noslot": v["noslot"] / v["tok"]}
            for b, v in by_size.items()
        }
        results[label] = {"ce": ce, "gain_over_zeroed": gains, "gain_over_noslot": gains_noslot,
                          "files_per_tree": n_files_tot / len(rows), "by_repo_size": sized}
        print(f"== {label}")
        print(f"   zeroed floor CE {ce['zeroed']:.4f}   NO-SLOT floor CE {ce['noslot']:.4f}")
        for k in ("identity", "random", "trained"):
            print(f"   {k:>9}: CE {ce[k]:.4f}   gain vs zeroed {gains[k]:+.4f}   "
                  f"vs no-slot {gains_noslot[k]:+.4f}")
        for b, v in sorted(sized.items()):
            print(f"   size {b:>6}: trees {v['trees']:3d}  identity CE {v['ce_identity']:.3f}  "
                  f"no-slot CE {v['ce_noslot']:.3f}  zeroed CE {v['ce_zeroed']:.3f}")
        del enc, dec
        torch.cuda.empty_cache()
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
