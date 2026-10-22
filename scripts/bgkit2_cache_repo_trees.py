#!/usr/bin/env python
"""Encode every repo of a tree store ONCE with a frozen encoder and cache the
tree's node encodings for the decoder.

Stage D keeps the encoder frozen (user 2026-09-17), so a repo's tree encoding
is a fixed function of the repo: compute it here and the trainer's tree batches
cost nothing per step. Re-run when the encoder checkpoint, the store, or the
tree ratios/depth change (all recorded in the metadata / index).

Outputs
  --out FILE.safetensors          one file (small stores). Root only
                                  ("<repo>/reps") or, with --all-nodes, every
                                  directory ("<repo>//<dir>/reps", "" = root).
  --out DIR (with --sharded)      part-XXXX.safetensors + index.jsonl, written
                                  every --shard-repos repos and RESUMABLE: the
                                  all-repos store is an ~18 h job and ~100 GB.
                                  Read lazily by bgkit2.data.node_cache.NodeCache.
  --variants N                    N extra forms per repo: a random --variant-keep
                                  fraction of its files, directory levels re-run
                                  on the full pass's leaves (no file re-encoded).
                                  Keys "<repo>@v<k>//<dir>/reps"; the kept file
                                  indices go to the index. Breaks the fixed
                                  "fingerprint" a cached encoding otherwise is.

    python scripts/bgkit2_cache_repo_trees.py --experiment <exp of the ENCODER ckpt> \\
        --checkpoint /workspace/checkpoints/<enc> --store <store>.parquet --all-nodes \\
        --sharded --out /workspace/checkpoints/tree_caches/<name>.nodes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def iter_repos(store: str, order: str = "store", skip: set[str] | None = None):
    """(repo, paths, file_ids) over the store's repo table, shard by shard --
    in the store's order, or in a fixed pseudo-random order (``hash``) read
    row by row so memory stays bounded. The train/eval split hashes the same
    key (sha1 of the repo name, eval = the lowest 2%), so in ``hash`` order the
    HELD-OUT repos come first: any prefix of the cache holds the complete eval
    set, followed by a uniform sample of the training repos."""
    import hashlib

    import pyarrow.parquet as pq

    side = store + ".repos.parquet"
    if order == "hash" and os.path.exists(side):
        shards = (sorted(str(p) for p in Path(side).glob("*.parquet"))
                  if os.path.isdir(side) else [side])
        where = []  # (hash, shard, row)
        for sh in shards:
            names = pq.read_table(sh, columns=["repo"]).column("repo").to_pylist()
            for i, name in enumerate(names):
                if skip and name in skip:
                    continue
                where.append((hashlib.sha1(name.encode()).hexdigest(), sh, i))
        where.sort()
        tables: dict[str, object] = {}
        for _h, sh, i in where:
            t = tables.get(sh)
            if t is None:
                if len(tables) >= 3:
                    tables.pop(next(iter(tables)))
                t = tables[sh] = pq.read_table(sh)
            row = t.slice(i, 1).to_pydict()
            yield row["repo"][0], list(row["paths"][0]), row["file_ids"][0]
        return
    if not os.path.exists(side):  # v3-style store: files ride in the question rows
        seen = set()
        t = pq.read_table(store, columns=["repo", "paths", "file_ids"])
        for repo, paths, ids in zip(t.column("repo").to_pylist(), t.column("paths").to_pylist(),
                                    t.column("file_ids").to_pylist(), strict=True):
            if repo not in seen:
                seen.add(repo)
                yield repo, list(paths), ids
        return
    shards = sorted(str(p) for p in Path(side).glob("*.parquet")) if os.path.isdir(side) else [side]
    for sh in shards:
        pf = pq.ParquetFile(sh)
        for batch in pf.iter_batches(batch_size=8):
            d = batch.to_pydict()
            for repo, paths, ids in zip(d["repo"], d["paths"], d["file_ids"], strict=True):
                yield repo, list(paths), ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--experiment", required=True,
                    help="experiment the ENCODER checkpoint was trained with (model built from it)")
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-files", type=int, default=400)
    ap.add_argument("--file-tokens", type=int, default=4096)
    ap.add_argument("--ratio-leaf", type=float, default=0.125)
    ap.add_argument("--ratio-node", type=float, default=0.25)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--leaf-chunk", type=int, default=16)
    ap.add_argument("--leaf-chunk-tokens", type=int, default=32768)
    ap.add_argument("--node-max-tokens", type=int, default=8192,
                    help="directory inputs above this are split into virtual sub-directories")
    ap.add_argument("--leaf-window", type=int, default=0,
                    help="WHOLE FILES: encode a file longer than this in windows of this many "
                         "tokens (one leaf per file); use with a store built at a large "
                         "--file-tokens and pass the same large --file-tokens here. 0 = off")
    ap.add_argument("--leaf-max-reps", type=int, default=0,
                    help="bound a leaf: long files compress harder (ratio = min(ratio-leaf, "
                         "cap / tokens)) instead of being cut. 0 = off")
    ap.add_argument("--node-inputs", action="store_true",
                    help="all-nodes caches: ALSO store each node's encoder INPUT "
                         "(<repo>//<node>/input = [header | children's survivors]) for the "
                         "QUESTION-CONDITIONED read-out (one prompted encoder pass per browsed "
                         "node). ~4x the size of the reps; use a small --shard-repos")
    ap.add_argument("--cuda-fraction", type=float, default=0.5)
    ap.add_argument("--bridge", default="identity", choices=["identity", "repro"])
    ap.add_argument("--all-nodes", action="store_true",
                    help="cache EVERY directory node (for the path walk); default: root only")
    ap.add_argument("--sharded", action="store_true", help="--out is a resumable shard DIRECTORY")
    ap.add_argument("--shard-repos", type=int, default=100)
    ap.add_argument("--variants", type=int, default=0)
    ap.add_argument("--variant-keep", type=float, default=0.7)
    ap.add_argument("--order", default="store", choices=["store", "hash"],
                    help="'hash': a fixed pseudo-random order (sha1 of the repo name), so ANY "
                    "prefix of the cache is a uniform sample of the store across languages and "
                    "sizes. The store's own order is by language then stars: its first 600 "
                    "repos were 596 Python (2026-09-19).")
    ap.add_argument("--max-repos", type=int, default=0,
                    help="stop once this many repos are cached IN TOTAL (already-cached ones "
                    "count). Size the cache to what training will read: ~1-1.5 passes.")
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N repos")
    args = ap.parse_args()
    if args.sharded and not args.all_nodes:
        raise SystemExit("--sharded caches are all-nodes caches; pass --all-nodes")
    if args.node_inputs and (not args.all_nodes or args.variants):
        raise SystemExit("--node-inputs needs --all-nodes and no --variants")
    if args.variants and not args.sharded:
        raise SystemExit("--variants needs --sharded (the kept files are recorded in the index)")

    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    from bgkit2.models.multilevel import encode_tree
    from bgkit2.training.standalone import load_models
    from bgkit_core.host_memory_guard import arm, cap_cuda

    arm(10.0)
    cap_cuda(args.cuda_fraction)
    dev = "cuda"
    enc_tok = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-Encoder-350M")
    tok_encode = lambda s: enc_tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    core = load_models(args.experiment, args.checkpoint, device=dev)
    enc = core.encoder
    del core.decoder
    bridge = (lambda x: x) if args.bridge == "identity" else None
    kw = dict(tok_encode=tok_encode, ratio_leaf=args.ratio_leaf, ratio_node=args.ratio_node,
              max_depth=args.max_depth, device=dev, bridge=bridge, leaf_chunk=args.leaf_chunk,
              grad_levels=0, leaf_chunk_tokens=args.leaf_chunk_tokens,
              node_max_tokens=args.node_max_tokens, leaf_window=args.leaf_window,
              leaf_max_reps=args.leaf_max_reps, keep_inputs=bool(args.node_inputs))
    meta = {"checkpoint": args.checkpoint, "store": args.store, "max_files": args.max_files,
            "file_tokens": args.file_tokens, "ratio_leaf": args.ratio_leaf,
            "ratio_node": args.ratio_node, "max_depth": args.max_depth, "bridge": args.bridge,
            "node_max_tokens": args.node_max_tokens, "all_nodes": bool(args.all_nodes),
            "leaf_window": args.leaf_window, "leaf_max_reps": args.leaf_max_reps,
            "node_inputs": bool(args.node_inputs),
            "variants": args.variants, "variant_keep": args.variant_keep}

    done: set[str] = set()
    n_shard = 0
    if args.sharded:
        os.makedirs(args.out, exist_ok=True)
        idx_path = os.path.join(args.out, "index.jsonl")
        if os.path.exists(idx_path):
            with open(idx_path) as fh:
                for ln in fh:
                    d = json.loads(ln)
                    done.add(d["repo"])
                    n_shard = max(n_shard, int(d["shard"][5:9]) + 1)
        with open(os.path.join(args.out, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        print(f"resuming: {len(done)} repos already cached, next shard {n_shard}", flush=True)

    out: dict[str, torch.Tensor] = {}
    pending: list[dict] = []  # index lines of the shard being filled
    t0, n_files, n = time.time(), 0, 0

    def flush() -> None:
        nonlocal out, pending, n_shard
        if not args.sharded or not pending:
            return
        name = f"part-{n_shard:04d}.safetensors"
        save_file(out, os.path.join(args.out, name + ".tmp"))
        os.replace(os.path.join(args.out, name + ".tmp"), os.path.join(args.out, name))
        with open(os.path.join(args.out, "index.jsonl"), "a") as fh:
            for d in pending:
                fh.write(json.dumps({**d, "shard": name}) + "\n")
        n_shard += 1
        out, pending = {}, []

    for repo, paths, ids in iter_repos(args.store, order=args.order, skip=done):
        if repo in done:
            continue
        if args.limit and n >= args.limit:
            break
        if args.max_repos and len(done) + n >= args.max_repos:
            break
        paths = paths[: args.max_files]
        files = [torch.tensor(np.asarray(x[: args.file_tokens], dtype=np.int64), device=dev)
                 for x in ids[: args.max_files]]
        n_files += len(files)
        # role base: the FROZEN tree encoder (an encoder-adapter checkpoint's
        # adapter is off here -- the trainer reads these nodes as base output)
        with torch.no_grad(), enc.role("base"), torch.autocast("cuda", dtype=torch.bfloat16):
            o = encode_tree(enc, files, paths, **kw)
        line = {"repo": repo, "nodes": [], "variants": []}
        if args.all_nodes:
            for node, (nr, nm) in (o.nodes or {}).items():
                out[f"{repo}//{node}/reps"] = nr[: int(nm.sum())].to(torch.bfloat16).cpu()
                line["nodes"].append(node)
                if args.node_inputs:
                    out[f"{repo}//{node}/input"] = o.node_inputs[node].to(torch.bfloat16).cpu()
        else:
            k = int(o.root.rep_mask[0].sum())
            out[f"{repo}/reps"] = o.root.reps[0, :k].to(torch.bfloat16).cpu()
            out[f"{repo}/mask"] = torch.ones(k, dtype=torch.bool)
        for v in range(args.variants):
            rng = np.random.default_rng([abs(hash(repo)) % (2**31), v])
            keep = sorted(rng.permutation(len(paths))[: max(2, int(len(paths) * args.variant_keep))]
                          .tolist())
            with torch.no_grad(), enc.role("base"), torch.autocast("cuda", dtype=torch.bfloat16):
                ov = encode_tree(enc, [files[i] for i in keep], [paths[i] for i in keep],
                                 leaves=(o.leaf_reps[keep], o.leaf_mask[keep]), **kw)
            vname = f"{repo}@v{v + 1}"
            for node, (nr, nm) in (ov.nodes or {}).items():
                out[f"{vname}//{node}/reps"] = nr[: int(nm.sum())].to(torch.bfloat16).cpu()
            line["variants"].append({"name": vname, "keep": keep, "nodes": list(ov.nodes or {})})
        pending.append(line)
        n += 1
        del o
        torch.cuda.empty_cache()
        if n % 25 == 1:
            el = time.time() - t0
            print(f"[{n}] {repo} files={len(files)} nodes={len(line['nodes'])} "
                  f"{el:.0f}s ({n_files / max(el, 1):.1f} files/s) "
                  f"cuda_max={torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
        if args.sharded and len(pending) >= args.shard_repos:
            flush()
    flush()
    if not args.sharded:
        save_file(out, args.out, metadata={k: str(v) for k, v in meta.items()})
    print(json.dumps({**meta, "repos": n, "already_done": len(done),
                      "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
