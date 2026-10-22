"""PATH WALK eval of a trained adapter on ANY tree store + node cache, outside
training: the trainer's own ``evaluate_tree`` (step accuracy by depth, greedy
full walks, CE against both floors) on a store the run never saw -- e.g. the
whole-file (uncut) caches or the large-repo tier.

    docker compose ... run --rm --no-deps train-bgkit2-p28 python scripts/bgkit2_eval_walk.py \\
        --experiment bgkit2_w2_path_walk_all_repos --checkpoint /workspace/checkpoints/<w2 final> \\
        --store /workspace/data/bgkit2_walk/repo_walk_large.parquet \\
        --cache /workspace/checkpoints/tree_caches/repo_walk_large.nodes --max-files 2000 \\
        --out /workspace/checkpoints/eval_reports/walk_large.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--max-files", type=int, default=400)
    ap.add_argument("--max-items", type=int, default=512, help="walk steps scored (stride sample)")
    ap.add_argument("--max-walks", type=int, default=256, help="greedy full walks")
    ap.add_argument("--cuda-fraction", type=float, default=0.5)
    ap.add_argument("--tag", default="tree")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from bgkit2.data.repo_tree import RepoTreeDataset
    from bgkit2.models.loading import load_tokenizer
    from bgkit2.training.bgkit2_trainer import BgKIT2Trainer
    from bgkit2.training.standalone import load_models
    from bgkit_core.host_memory_guard import arm, cap_cuda

    arm(10.0)
    cap_cuda(args.cuda_fraction)
    t0 = time.time()
    core = load_models(args.experiment, args.checkpoint)
    core.tok = load_tokenizer(core.cfg2.model.decoder_id)
    from bgkit2.data.templates import set_no_think_template
    if core.cfg2.model.decoder_no_think:
        set_no_think_template(core.tok)
    core.load_tree_root_cache(args.cache)
    meta = core.tree_node_cache.meta
    depth = int(meta.get("max_depth", core.cfg2.multilevel.max_depth))
    ds = RepoTreeDataset(args.store, args.split, max_files=args.max_files,
                         file_tokens=int(meta.get("file_tokens", 4096)), shuffle=False,
                         walk_depth=depth, load_files=False)
    n_all = len(ds)
    n = ds.restrict_to(core.tree_node_cache.repos)
    print(json.dumps({"store": args.store, "items": n, "of": n_all,
                      "repos": len({r["repo"] for r in ds.rows}), "cache_meta": meta,
                      "load_s": round(time.time() - t0, 1)}), flush=True)
    if n == 0:
        raise SystemExit("no cached repo in this split")
    core.evaluate_tree = types.MethodType(BgKIT2Trainer.evaluate_tree, core)
    core._evaluate_full_walks = types.MethodType(BgKIT2Trainer._evaluate_full_walks, core)
    m = core.evaluate_tree(max_items=args.max_items, ds=ds, tag=args.tag,
                           max_walks=args.max_walks)
    rec = {"experiment": args.experiment, "checkpoint": args.checkpoint, "store": args.store,
           "cache": args.cache, "items": n, "seconds": round(time.time() - t0, 1),
           "metrics": {k: round(float(v), 4) for k, v in sorted(m.items())}}
    print(json.dumps(rec, indent=1), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
