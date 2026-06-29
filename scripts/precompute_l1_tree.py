#!/usr/bin/env python
"""Offline bottom-up encode of a browse tree into a cached L1 subtree-summary.

Deep semantic browse trees (pubmedqa MeSH, KILT DBpedia, git_history
org→repo→year) need every node to carry a dense L1 summary of its entire
subtree so the live search path can read a node's gist without re-encoding
the whole subtree. Those summaries are built BOTTOM-UP via recursive L1:

    leaf node      ── L0 survivors of its articles
                      bridged L0-output → L1-input via ``l0.auto_reproduce``
                      ──► encode_node ──► node rep (L1-output)

    interior node  ── children's cached node reps (L1-output)
                      bridged L1-output → L1-input via ``l1_auto_reproduce``
                      ──► encode_node ──► node rep (L1-output)

Each node rep is the recursive-L1 survivor set in L1-OUTPUT (pre-norm) space,
so a parent re-bridges it via ``encoder.l1_auto_reproduce`` and calls
``encode_node`` again. This mirrors the way leaf L0 survivors are re-bridged
via ``encoder.l0.auto_reproduce``.

LOCKED DESIGN: cached node reps are QUERY-AGNOSTIC — computed with no query
conditioning. Query-conditioning is re-applied later, only on the live search
path + final retrieval. ``encode_node`` therefore takes no query here.

Output layout (identical to the L0 cache, see :mod:`bgkit.data.l0_cache`)::

    {output_dir}/{dataset}/shard_NNNN/{survivors,offsets}.npy
    {output_dir}/{dataset}/index.parquet     # node_id → (shard_id, row_index)
    {output_dir}/{dataset}/cache_manifest.json

The manifest additionally pins the source L0-cache hash + the ``l1l1_bridge``
weights hash so a stale tree (rebuilt L0 cache, or retrained recursive bridge)
is detected.

Idempotent: nodes already present in ``index.parquet`` are skipped (their reps
are loaded from disk so parents can still consume them).

CPU/GPU: mirrors ``scripts/precompute_l0_subset.py`` — runs on whatever device
is available; the full encoder (L0 + L1 + both bridges) is loaded, not the
L0-only subset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bgkit.data.browse_tree import BrowseTree
from bgkit.data.l0_cache import (
    L0Cache,
    SurvivorBlockCache,
    SurvivorBlockCacheWriter,
    tensor_state_sha256,
    update_dataset_index,
    write_l1_tree_cache_manifest,
)
from bgkit.utils.deltanet_patch import (
    patch_fused_rms_norm_gated_for_sm121,
    patch_gated_delta_rule_numerics,
)
from bgkit.utils.triton_alloc_patch import patch_triton_allocator
from bgkit.utils.triton_patch import patch_triton_autotuner

patch_triton_allocator()
patch_triton_autotuner()
patch_gated_delta_rule_numerics()
patch_fused_rms_norm_gated_for_sm121()


# ---------------------------------------------------------------------------
# Encoder loading (full encoder — L0 + L1 + both bridges)
# ---------------------------------------------------------------------------


def _extract_encoder_state(state: dict) -> dict:
    """Pull the un-prefixed encoder state dict out of a checkpoint payload.

    Same convention as ``scripts/precompute_l0_subset._load_encoder``: new
    layout checkpoints store ``encoder`` directly with un-prefixed keys;
    older single-file checkpoints nest everything under ``model`` with an
    ``encoder.`` prefix.
    """
    encoder_state = state.get("encoder")
    if encoder_state is None:
        model_state = state.get("model", {})
        encoder_state = {
            k.replace("encoder.", "", 1): v
            for k, v in model_state.items()
            if k.startswith("encoder.")
        }
    return dict(encoder_state)


def _load_full_encoder(
    phase1_checkpoint: Path,
    stage_a_checkpoint: Path | None,
    encoder_name: str = "Qwen/Qwen3.5-0.8B-Base",
    hidden_dim: int = 1024,
) -> torch.nn.Module:
    """Load the FULL encoder (L0 + L1 + projection + both bridges).

    Unlike ``precompute_l0_subset._load_encoder`` (which loads ``encoder.l0``
    only) the recursive-L1 tree encode needs ``encoder.l1``, the L0→L1 bridge
    (``encoder.l0.auto_reproduce``) AND the L1→L1 recursive bridge
    (``encoder.l1_auto_reproduce`` / ``encoder.l1l1_bridge``). Stage A's L0
    weights are merged on top of the Phase 1 base when provided, matching the
    L0 cache so leaf survivors line up.
    """
    from bgkit.models.encoder import BgKITEncoder
    from bgkit.training.checkpointing import load_checkpoint

    _meta, state = load_checkpoint(phase1_checkpoint)
    encoder_state = _extract_encoder_state(state)

    if stage_a_checkpoint is not None:
        _meta_a, state_a = load_checkpoint(stage_a_checkpoint)
        enc_a = _extract_encoder_state(state_a)
        l0_state_a = {
            k: v
            for k, v in enc_a.items()
            if k.startswith("l0.") or k.startswith("l0_task_prompts.")
        }
        if not l0_state_a:
            raise RuntimeError(
                f"Stage A checkpoint {stage_a_checkpoint} has no "
                "encoder.l0.* keys to merge."
            )
        encoder_state.update(l0_state_a)

    encoder = BgKITEncoder.from_pretrained_with_state_dict(
        encoder_name, encoder_state, hidden_dim=hidden_dim,
    )
    return encoder


# ---------------------------------------------------------------------------
# Retention table (per browse-tree depth)
# ---------------------------------------------------------------------------


class DepthRetention:
    """Per-depth L1 retention ratios for tree encoding.

    JSON shape: ``{"default": 0.15, "0": 0.5, "1": 0.3, ...}`` where integer
    string keys are tree depths (root = depth 0). ``default`` applies to any
    depth not explicitly listed.
    """

    def __init__(self, raw: dict):
        self._default = float(raw.get("default", 0.15))
        self._by_depth: dict[int, float] = {}
        for k, v in raw.items():
            if k == "default":
                continue
            try:
                self._by_depth[int(k)] = float(v)
            except (TypeError, ValueError):
                continue

    @classmethod
    def load(cls, path: Path) -> DepthRetention:
        return cls(json.loads(path.read_text()))

    def for_depth(self, depth: int) -> float:
        return self._by_depth.get(int(depth), self._default)

    def as_dict(self) -> dict:
        d: dict[str, float] = {"default": self._default}
        d.update({str(k): v for k, v in sorted(self._by_depth.items())})
        return d


# ---------------------------------------------------------------------------
# Tree walk helpers
# ---------------------------------------------------------------------------


def topo_postorder(tree: BrowseTree) -> list[str]:
    """Bottom-up topological order: children before parents, from ``root``.

    Iterative post-order DFS over the ``children`` edges. Article/leaf-tag
    nodes (no children) come first; ``root`` comes last.
    """
    order: list[str] = []
    visited: set[str] = set()
    stack: list[tuple[str, bool]] = [("root", False)]
    while stack:
        nid, processed = stack.pop()
        if nid in visited:
            continue
        if processed:
            visited.add(nid)
            order.append(nid)
            continue
        stack.append((nid, True))
        for cid in tree.get(nid).children:
            if cid in tree and cid not in visited:
                stack.append((cid, False))
    return order


def _next_shard_index(cache_dir: Path, dataset: str) -> int:
    """Smallest shard index not yet on disk for ``dataset`` (resume-safe)."""
    ds_dir = cache_dir / dataset
    if not ds_dir.exists():
        return 0
    used = [
        int(p.name.split("_")[1])
        for p in ds_dir.glob("shard_*")
        if p.is_dir() and p.name.split("_")[1].isdigit()
    ]
    return (max(used) + 1) if used else 0


# ---------------------------------------------------------------------------
# Per-node encode
# ---------------------------------------------------------------------------


def _leaf_article_ids(tree: BrowseTree, node_id: str) -> list[str]:
    node = tree.get(node_id)
    if node.is_article:
        return [node.id]
    if node.is_leaf_tag:
        return list(node.articles)
    return []


def _encode_node_reps(
    *,
    encoder: torch.nn.Module,
    tree: BrowseTree,
    dataset: str,
    node_id: str,
    target_ratio: float,
    l0_cache: L0Cache,
    reps: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Encode one node into its L1-output (pre-norm) survivor set.

    Returns ``None`` when the node has no usable input (e.g. all its articles
    are missing from the L0 cache, or it is an empty interior node).
    """
    node = tree.get(node_id)
    is_leaf = node.is_article or node.is_leaf_tag

    pieces: list[torch.Tensor] = []
    if is_leaf:
        for aid in _leaf_article_ids(tree, node_id):
            if not l0_cache.has(dataset, aid):
                continue
            surv = l0_cache.get(dataset, aid).to(device=device, dtype=dtype)
            if surv.numel() == 0:
                continue
            # L0-output (pre-norm) → L1-input space.
            pieces.append(encoder.l0.auto_reproduce(surv))
    else:
        for cid in node.children:
            child_rep = reps.get(cid)
            if child_rep is None or child_rep.numel() == 0:
                continue
            child_rep = child_rep.to(device=device, dtype=dtype)
            # L1-output (pre-norm) → L1-input space.
            pieces.append(encoder.l1_auto_reproduce(child_rep))

    if not pieces:
        return None

    children_reps_l1in = torch.cat(pieces, dim=0)
    children_cu = torch.tensor(
        [0, int(children_reps_l1in.shape[0])],
        dtype=torch.int32,
        device=device,
    )
    with torch.no_grad():
        node_surv, _node_cu = encoder.encode_node(
            children_reps_l1in=children_reps_l1in,
            children_cu_seqlens=children_cu,
            target_ratio=target_ratio,
        )
    return node_surv.detach()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def precompute_tree(
    *,
    tree: BrowseTree,
    dataset: str,
    encoder: torch.nn.Module,
    l0_cache: L0Cache,
    retention: DepthRetention,
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    shard_size: int,
) -> tuple[int, int]:
    """Walk ``tree`` bottom-up and write each node's L1 summary to the cache.

    Returns ``(n_encoded, n_skipped)``.
    """
    order = topo_postorder(tree)

    # Resume: anything already indexed is reused (loaded into ``reps`` so its
    # parents can consume it) instead of re-encoded.
    existing = SurvivorBlockCache(output_dir) if (output_dir / dataset).exists() else None
    already: set[str] = (
        set(existing.node_ids(dataset)) if existing is not None else set()
    )

    reps: dict[str, torch.Tensor] = {}
    shard_idx = _next_shard_index(output_dir, dataset)
    writer = SurvivorBlockCacheWriter(output_dir, dataset, f"shard_{shard_idx:04d}")
    index_rows: list[tuple[str, int]] = []
    count_in_shard = 0
    n_encoded = 0
    n_skipped = 0

    def _flush_shard() -> None:
        nonlocal shard_idx, writer, index_rows, count_in_shard
        n, rows = writer.finalize()
        if n:
            update_dataset_index(
                output_dir, dataset, f"shard_{shard_idx:04d}", rows,
                id_column="node_id",
            )
        shard_idx += 1
        writer = SurvivorBlockCacheWriter(
            output_dir, dataset, f"shard_{shard_idx:04d}",
        )
        index_rows = []
        count_in_shard = 0

    for node_id in order:
        depth = len(tree.path_to(node_id)) - 1
        ratio = retention.for_depth(depth)

        if node_id in already and existing is not None:
            # Reuse the previously-cached rep so parents can bridge it.
            reps[node_id] = existing.get(dataset, node_id).to(
                device=device, dtype=dtype,
            )
            n_skipped += 1
            continue

        rep = _encode_node_reps(
            encoder=encoder,
            tree=tree,
            dataset=dataset,
            node_id=node_id,
            target_ratio=ratio,
            l0_cache=l0_cache,
            reps=reps,
            device=device,
            dtype=dtype,
        )
        if rep is None or rep.numel() == 0:
            n_skipped += 1
            continue

        reps[node_id] = rep
        writer.add(node_id, rep)
        index_rows.append((node_id, count_in_shard))
        count_in_shard += 1
        n_encoded += 1
        if count_in_shard >= shard_size:
            _flush_shard()

    # Final partial shard.
    n, rows = writer.finalize()
    if n:
        update_dataset_index(
            output_dir, dataset, f"shard_{shard_idx:04d}", rows,
            id_column="node_id",
        )
    return n_encoded, n_skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--browse-tree",
        required=True,
        type=Path,
        help="Path to the dataset's browse tree parquet.",
    )
    parser.add_argument(
        "--l0-cache-dir",
        required=True,
        type=Path,
        help="Root of the L0 survivor cache (leaf survivors).",
    )
    parser.add_argument(
        "--phase1-checkpoint",
        "--checkpoint",
        dest="phase1_checkpoint",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--stage-a-checkpoint",
        required=False,
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        type=Path,
        help="Root of the L1-tree cache (default: $DATA_DIR/l1_tree_cache_kb).",
    )
    parser.add_argument("--retention-json", required=True, type=Path)
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument(
        "--encoder-name", default="Qwen/Qwen3.5-0.8B-Base",
    )
    parser.add_argument("--hidden-dim", type=int, default=1024)
    args = parser.parse_args()

    if args.output_dir is None:
        from bgkit.env import get_data_dir

        args.output_dir = get_data_dir() / "l1_tree_cache_kb"

    retention = DepthRetention.load(args.retention_json)
    tree = BrowseTree.load(args.browse_tree, dataset=args.dataset)
    l0_cache = L0Cache(str(args.l0_cache_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = _load_full_encoder(
        args.phase1_checkpoint,
        args.stage_a_checkpoint,
        encoder_name=args.encoder_name,
        hidden_dim=args.hidden_dim,
    )
    encoder.to(device).eval()
    dtype = next(encoder.parameters()).dtype

    n_encoded, n_skipped = precompute_tree(
        tree=tree,
        dataset=args.dataset,
        encoder=encoder,
        l0_cache=l0_cache,
        retention=retention,
        output_dir=args.output_dir,
        device=device,
        dtype=dtype,
        shard_size=args.shard_size,
    )

    # Record provenance + staleness fingerprints.
    bridge_sha = tensor_state_sha256(encoder.l1l1_bridge.state_dict())
    write_l1_tree_cache_manifest(
        args.output_dir,
        args.dataset,
        phase1_checkpoint=args.phase1_checkpoint,
        stage_a_checkpoint=args.stage_a_checkpoint,
        source_l0_cache_dir=args.l0_cache_dir,
        l1l1_bridge_sha=bridge_sha,
        retention=retention.as_dict(),
        extra={
            "n_nodes_encoded": n_encoded,
            "n_nodes_skipped": n_skipped,
            "n_tree_nodes": len(tree),
        },
    )

    print(
        f"dataset={args.dataset} — encoded {n_encoded} nodes, "
        f"skipped {n_skipped} (of {len(tree)} tree nodes)"
    )


if __name__ == "__main__":
    main()
