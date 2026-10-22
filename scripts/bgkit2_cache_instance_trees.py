#!/usr/bin/env python
"""All-nodes tree caches of SWE-style instances AT THEIR base_commit (GPU; 2026-09-22).

The per-commit counterpart of scripts/bgkit2_cache_repo_trees.py: same frozen tree encoder, same
encode_tree settings (whole files: --leaf-window / --leaf-max-reps), same shard layout
(part-XXXX.safetensors + index.jsonl + meta.json, read by bgkit2.data.node_cache.NodeCache), but
the input is a MANIFEST of (repo, base_commit) trees (scripts/bgkit2_swe_tree_survey.py writes
it) and every tree is keyed ``<owner>/<name>@<sha12>`` (bgkit2.agent.swe_trees.tree_key). Files
are read from the git object database at the commit (no checkout, nothing from the future) with
the tree stores' filter. Next to the shards, ``trees.jsonl`` records each tree's key, repo,
commit, the instance ids that use it and its file paths -- what
bgkit2.agent.overview.instance_overview_lookup needs to find a tree by instance and to build
the ``browse`` listings.

Leaf reuse: a repo's trees are encoded in commit-time order and a file whose (path, blob) was
already encoded for an earlier commit reuses that leaf (its encoding depends only on its path
header and its tokens); only the directory levels are re-run. SWE-bench Lite has 300 instances
on 12 repos, so most leaves repeat.

Guarded like every standalone GPU script (feedback_standalone_gpu_scripts_must_be_guarded):
host-memory guard, cap_cuda, bounded encoder inputs (--leaf-chunk-tokens, --node-max-tokens),
a per-tree token budget (--max-tree-tokens), progress lines per tree. Resumable: re-running skips
every key already in index.jsonl or skipped.jsonl.

    # in the GPU container (train-bgkit2-p28 service), ONLY when no training runs:
    python scripts/bgkit2_cache_instance_trees.py --experiment bgkit2_m3_tree_nav_nobridge \\
        --checkpoint /workspace/checkpoints/<M3-nobridge ckpt> \\
        --manifest /workspace/bgkit2_data/swe_trees/lite/manifest.jsonl \\
        --out /workspace/checkpoints/tree_caches/swe_lite.nodes
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgkit2.agent.swe_trees import TIERS, TreeJob, read_manifest

TIER_NAMES = [t for t, _ in TIERS] + ["unsupported"]


def map_path(p: str, maps: list[tuple[str, str]]) -> str:
    """Host path -> container path (the manifest is written on the host)."""
    if os.path.exists(p):
        return p
    for src, dst in maps:
        if p.startswith(src):
            q = dst + p[len(src):]
            if os.path.exists(q):
                return q
    return p


def order_jobs(jobs: list[TreeJob], maps) -> list[TreeJob]:
    """Grouped by repo (largest group first), commit-time order inside a repo:
    consecutive commits share the most leaves."""
    import pygit2

    by_repo: dict[str, list[TreeJob]] = defaultdict(list)
    for j in jobs:
        by_repo[j.repo].append(j)
    out: list[TreeJob] = []
    for _repo, js in sorted(by_repo.items(), key=lambda kv: -len(kv[1])):
        try:
            r = pygit2.Repository(map_path(js[0].repo_dir, maps))
            t = {j.key: r.revparse_single(j.commit).peel(pygit2.Commit).commit_time for j in js}
            js = sorted(js, key=lambda j: t[j.key])
        except Exception:  # an unreadable mirror fails per tree below, with its message
            pass
        out.extend(js)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--experiment", required=True,
                    help="experiment the ENCODER checkpoint was trained with")
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="shard directory (resumable)")
    ap.add_argument("--tiers", default="live,normal,large",
                    help="which size tiers to build (bgkit2.agent.swe_trees.TIERS, by kept files)")
    ap.add_argument("--max-files", type=int, default=2000, help="hard cap on kept files per tree")
    ap.add_argument("--max-tree-tokens", type=int, default=12_000_000,
                    help="skip a tree whose kept files exceed this many tokens (walk_large's cap)")
    ap.add_argument("--file-tokens", type=int, default=131072)
    ap.add_argument("--ratio-leaf", type=float, default=0.125)
    ap.add_argument("--ratio-node", type=float, default=0.25)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--leaf-chunk", type=int, default=16)
    ap.add_argument("--leaf-chunk-tokens", type=int, default=65536)
    ap.add_argument("--node-max-tokens", type=int, default=8192)
    ap.add_argument("--leaf-window", type=int, default=4096)
    ap.add_argument("--leaf-max-reps", type=int, default=512)
    ap.add_argument("--bridge", default="identity", choices=["identity", "repro"])
    ap.add_argument("--leaf-reuse-gb", type=float, default=8.0,
                    help="host RAM for reused leaf encodings (LRU, one repo at a time)")
    ap.add_argument("--no-leaf-reuse", action="store_true",
                    help="encode every tree whole, without the reuse path (parity control: cache a "
                    "few trees both ways into two dirs and compare their node reps)")
    ap.add_argument("--shard-repos", type=int, default=25, help="trees per shard")
    ap.add_argument("--cuda-fraction", type=float, default=0.5)
    ap.add_argument("--host-floor-gb", type=float, default=10.0)
    ap.add_argument("--path-map", nargs="*",
                    default=["/mnt/external/bgkit-data=/workspace/data",
                             "/home/werg/bgkit-data-nvme/bgkit2=/workspace/bgkit2_data"],
                    help="HOST=CONTAINER prefixes for the manifest's repo_dir")
    ap.add_argument("--tokenizer", default="LiquidAI/LFM2.5-Encoder-350M")
    ap.add_argument("--max-trees", type=int, default=0,
                    help="cache at most this many trees IN TOTAL (already cached ones count), "
                    "taken in manifest order -- the survey writes the generator's task order, so "
                    "a prefix is the sample the generator will run first. 0 = all")
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N trees")
    args = ap.parse_args()
    maps = [tuple(m.split("=", 1)) for m in args.path_map or []]
    want_tiers = {t.strip() for t in args.tiers.split(",") if t.strip()}
    if bad := want_tiers - set(TIER_NAMES):
        raise SystemExit(f"unknown tiers {bad}; choose from {TIER_NAMES}")

    import numpy as np
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    from bgkit2.agent.swe_trees import commit_python_sources, tier_of
    from bgkit2.models.multilevel import encode_tree
    from bgkit2.training.standalone import load_models
    from bgkit_core.host_memory_guard import arm, cap_cuda

    arm(args.host_floor_gb)
    cap_cuda(args.cuda_fraction)
    dev = "cuda"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path, trees_path = out_dir / "index.jsonl", out_dir / "trees.jsonl"
    skip_path = out_dir / "skipped.jsonl"
    done: set[str] = set()
    n_shard = 0
    if idx_path.exists():
        for ln in idx_path.open():
            d = json.loads(ln)
            done.add(d["repo"])
            n_shard = max(n_shard, int(d["shard"][5:9]) + 1)
    skipped: set[str] = set()
    if skip_path.exists():
        skipped = {json.loads(ln)["key"] for ln in skip_path.open() if ln.strip()}
    jobs: list[TreeJob] = []
    seen: set[str] = set()
    for m in args.manifest:
        for j in read_manifest(m):
            if j.key not in seen:
                seen.add(j.key)
                jobs.append(j)
    todo = [j for j in jobs if j.key not in done and j.key not in skipped
            and (not j.tier or j.tier in want_tiers)]
    if args.max_trees:
        todo = todo[: max(0, args.max_trees - len(done))]
    print(f"{len(jobs)} trees in the manifest(s); {len(done)} cached, {len(skipped)} skipped "
          f"earlier; {len(todo)} to build (tiers {sorted(want_tiers)}), next shard {n_shard}",
          flush=True)
    if not todo:
        return
    todo = order_jobs(todo, maps)

    enc_tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def tok_encode(s: str) -> list[int]:
        return enc_tok(s, add_special_tokens=False)["input_ids"]

    core = load_models(args.experiment, args.checkpoint, device=dev)
    enc = core.encoder
    del core.decoder
    # An encoder-adapter checkpoint carries two tree encoders; the caches hold the
    # FROZEN base one (what the trainer and the student server read as cached nodes).
    role = (lambda: enc.role("base")) if hasattr(enc, "role") else contextlib.nullcontext
    bridge = (lambda x: x) if args.bridge == "identity" else None
    kw = dict(tok_encode=tok_encode, ratio_leaf=args.ratio_leaf, ratio_node=args.ratio_node,
              device=dev, bridge=bridge, leaf_chunk=args.leaf_chunk, grad_levels=0,
              leaf_chunk_tokens=args.leaf_chunk_tokens, node_max_tokens=args.node_max_tokens,
              leaf_window=args.leaf_window, leaf_max_reps=args.leaf_max_reps)
    meta = {"checkpoint": args.checkpoint, "experiment": args.experiment,
            "manifests": args.manifest, "keyed_by": "repo@base_commit[:12]",
            "max_files": args.max_files, "max_tree_tokens": args.max_tree_tokens,
            "file_tokens": args.file_tokens, "ratio_leaf": args.ratio_leaf,
            "ratio_node": args.ratio_node, "max_depth": args.max_depth, "bridge": args.bridge,
            "node_max_tokens": args.node_max_tokens, "all_nodes": True,
            "leaf_window": args.leaf_window, "leaf_max_reps": args.leaf_max_reps,
            "node_inputs": False, "variants": 0}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    shard: dict[str, torch.Tensor] = {}
    pending: list[tuple[dict, dict]] = []  # (index line, trees line) of the filling shard

    def flush() -> None:
        nonlocal shard, pending, n_shard
        if not pending:
            return
        name = f"part-{n_shard:04d}.safetensors"
        save_file(shard, str(out_dir / (name + ".tmp")))
        os.replace(out_dir / (name + ".tmp"), out_dir / name)
        # trees.jsonl first: a tree listed there without an index line is ignored
        # by the lookup (and rebuilt on resume); the reverse would be a cached tree
        # the lookup cannot find
        with trees_path.open("a") as fh:
            for _i, t in pending:
                fh.write(json.dumps(t) + "\n")
        with idx_path.open("a") as fh:
            for i, _t in pending:
                fh.write(json.dumps({**i, "shard": name}) + "\n")
        n_shard += 1
        shard, pending = {}, []

    def skip(j: TreeJob, why: str, persist: bool = False, **extra) -> None:
        # only PERMANENT reasons are persisted (a resume never retries them); a tier /
        # token skip depends on this run's flags and is re-decided by the next run
        if persist:
            with skip_path.open("a") as fh:
                fh.write(json.dumps({"key": j.key, "repo": j.repo, "why": why, **extra}) + "\n")
        print(f"SKIP {j.key}: {why} {extra}", flush=True)

    leaf_lru: OrderedDict[tuple[str, str], torch.Tensor] = OrderedDict()  # (path, oid) -> (k, D)
    lru_bytes = 0
    cur_repo = ""
    t0 = time.time()
    tot_tok = new_tok = n_done = 0
    for j in todo:
        if args.limit and n_done >= args.limit:
            break
        if j.repo != cur_repo:  # leaves never repeat across repos: free the host RAM
            leaf_lru.clear()
            lru_bytes, cur_repo = 0, j.repo
        try:
            src = commit_python_sources(map_path(j.repo_dir, maps), j.commit)
            files = list(src) if src is not None else []
        except Exception as exc:  # a missing mirror / commit: record, go on
            skip(j, "unreadable", error=str(exc)[:200])
            continue
        n_kept = len(files)
        if tier_of(n_kept) not in want_tiers or n_kept > args.max_files:
            skip(j, "tier", n_files=n_kept, tier=tier_of(n_kept))
            continue
        if n_kept < 2:
            skip(j, "too_few_files", persist=True, n_files=n_kept)
            continue
        paths = [p for p, _o, _t in files]
        ids = [np.asarray(tok_encode(t)[: args.file_tokens], dtype=np.int64) for _p, _o, t in files]
        n_tok = int(sum(x.shape[0] for x in ids))
        if n_tok > args.max_tree_tokens:
            skip(j, "tokens", n_files=n_kept, tokens=n_tok)
            continue
        keys = [(p, o) for p, o, _t in files]
        new = [i for i, k in enumerate(keys) if k not in leaf_lru]
        t_tree = time.time()
        files_t = [torch.from_numpy(x).to(dev) for x in ids]
        with torch.no_grad(), role(), torch.autocast("cuda", dtype=torch.bfloat16):
            if args.no_leaf_reuse:  # the plain encode, as bgkit2_cache_repo_trees.py does it
                o = encode_tree(enc, files_t, paths, max_depth=args.max_depth, **kw)
            else:
                if new:
                    # leaves of the files not seen at an earlier commit: a flat (depth-1)
                    # pass -- only its leaf encodings are kept
                    lo = encode_tree(enc, [files_t[i] for i in new], [paths[i] for i in new],
                                     max_depth=1, **kw)
                    for r, i in enumerate(new):
                        leaf = lo.leaf_reps[r][lo.leaf_mask[r]].to(torch.bfloat16).cpu()
                        leaf_lru[keys[i]] = leaf
                        lru_bytes += leaf.numel() * leaf.element_size()
                    del lo
                got = [leaf_lru[k] for k in keys]
                for k in keys:
                    leaf_lru.move_to_end(k)
                kmax = max(1, max(int(x.shape[0]) for x in got))
                lr = torch.zeros(len(got), kmax, int(got[0].shape[-1]), dtype=torch.bfloat16,
                                 device=dev)
                lm = torch.zeros(len(got), kmax, dtype=torch.bool, device=dev)
                for f, x in enumerate(got):
                    lr[f, : x.shape[0]] = x.to(dev)
                    lm[f, : x.shape[0]] = True
                o = encode_tree(enc, files_t, paths, max_depth=args.max_depth, leaves=(lr, lm),
                                **kw)
                del lr, lm
        while lru_bytes > args.leaf_reuse_gb * 1e9 and len(leaf_lru) > len(keys):
            _k, old = leaf_lru.popitem(last=False)
            lru_bytes -= old.numel() * old.element_size()
        nodes = []
        for node, (nr, nm) in (o.nodes or {}).items():
            shard[f"{j.key}//{node}/reps"] = nr[: int(nm.sum())].to(torch.bfloat16).cpu()
            nodes.append(node)
        n_new_tok = int(sum(ids[i].shape[0] for i in new))
        pending.append((
            {"repo": j.key, "nodes": nodes, "variants": []},
            {"key": j.key, "repo": j.repo, "commit": j.commit, "instances": j.instances,
             "n_files": n_kept, "tokens": n_tok, "paths": paths},
        ))
        del o, files_t
        torch.cuda.empty_cache()
        n_done += 1
        tot_tok += n_tok
        new_tok += n_new_tok
        el = time.time() - t0
        print(f"[{n_done}/{len(todo)}] {j.key} files={n_kept} new={len(new)} tok={n_tok} "
              f"new_tok={n_new_tok} nodes={len(nodes)} {time.time() - t_tree:.1f}s | {el:.0f}s "
              f"{new_tok / max(el, 1):.0f} new tok/s, reuse {1 - new_tok / max(tot_tok, 1):.2f} "
              f"lru={lru_bytes / 1e9:.1f}GB "
              f"cuda_max={torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
        if len(pending) >= args.shard_repos:
            flush()
    flush()
    print(json.dumps({**meta, "trees": n_done, "already_done": len(done), "tokens": tot_tok,
                      "encoded_leaf_tokens": new_tok, "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
