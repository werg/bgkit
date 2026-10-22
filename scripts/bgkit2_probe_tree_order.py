#!/usr/bin/env python
"""Is the tree-navigation gain a shortcut on FILE ORDER?

repo_tree's builder puts the evidence file FIRST. Every file's path is a header
at level 0, and positions survive the recursion (rotary), so "emit the first
file's header" answers the question with no navigation. This scores one
checkpoint on the repo-disjoint eval trees under (a) builder order (answer
first) and (b) a fixed per-row shuffle. A large drop from (a) to (b) is the
shortcut's share of the gain; (b) is the number that stands.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    ap.add_argument(
        "--store", default="/home/werg/bgkit-data-nvme/bgkit2/tasks/repo_tree_v3.parquet"
    )
    ap.add_argument("--n", type=int, default=162)
    ap.add_argument("--bridge", default="repro", choices=["repro", "identity"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from bgkit2.data.repo_tree import RepoTreeDataset
    from bgkit2.models.multilevel import encode_tree
    from bgkit2.training.standalone import load_models
    from bgkit_core.host_memory_guard import arm, cap_cuda

    arm(10.0)
    cap_cuda(args.cuda_fraction)
    dev = "cuda"
    # the run's own models, loaded strictly (adapters included) by the trainer's code
    core = load_models(args.experiment, args.checkpoint, device=dev)
    enc, dec = core.encoder, core.decoder
    enc_tok = AutoTokenizer.from_pretrained(core.cfg2.model.encoder_id)
    dec_tok = AutoTokenizer.from_pretrained(core.cfg2.model.decoder_id)
    tok_encode = lambda s: enc_tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    bridge = (lambda x: x) if args.bridge == "identity" else None

    def score(ds) -> tuple[float, float]:
        tot = tot0 = 0.0
        n_tok = 0
        for i in range(min(args.n, len(ds))):
            t = ds[i]
            files = [f.to(dev) for f in t.file_ids]
            with torch.no_grad(), core.autocast():
                with enc.role(core.tree_role(len(t.paths))):
                    out = encode_tree(enc, files, t.paths, tok_encode=tok_encode,
                                      ratio_leaf=0.125, ratio_node=0.25, max_depth=3, device=dev,
                                      bridge=bridge)
                q = dec_tok(f"{t.question}\nWhich file? ", add_special_tokens=False)["input_ids"]
                a = dec_tok(t.answer, add_special_tokens=False)["input_ids"]
                pre = [torch.tensor(q, device=dev)]
                suf = [torch.tensor(a, device=dev)]
                reps, mask = out.root.reps, out.root.rep_mask
                ce = float(dec.forward(dec.build_batch(pre, reps, mask, suf)))
                ce0 = float(dec.forward(dec.build_batch(pre, torch.zeros_like(reps), mask, suf)))
            tot += ce * len(a)
            tot0 += ce0 * len(a)
            n_tok += len(a)
        return tot / max(n_tok, 1), tot0 / max(n_tok, 1)

    res = {}
    for label, shuf in (("answer_first", False), ("shuffled", True)):
        ds = RepoTreeDataset(args.store, "eval", max_files=24, file_tokens=512, shuffle=shuf)
        ce, ce0 = score(ds)
        res[label] = {"nav_ce": ce, "zeroed": ce0, "gain": ce0 - ce}
        print(f"{label:>13}: nav CE {ce:.3f}  zeroed {ce0:.3f}  gain {ce0 - ce:+.3f}")
    res["shortcut_share_nats"] = res["answer_first"]["gain"] - res["shuffled"]["gain"]
    print(f"order shortcut worth {res['shortcut_share_nats']:+.3f} nats of the gain")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
