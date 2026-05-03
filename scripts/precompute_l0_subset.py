#!/usr/bin/env python
"""Batch-encode L0 survivors for a fixed subset of articles.

Input:
    --articles: JSONL with ``{"dataset": ..., "article_id": ...}`` rows,
                typically produced by :mod:`scripts.build_trajectory_set`.
                ``article_id`` must match ``document_id`` in the dataset's
                ``metadata.parquet`` under ``--mmap-dir``.
    --mmap-dir: root of the Phase 2 mmap layout
                (default ``$DATA_DIR/mmap/phase2``). Each subdirectory is one
                dataset: ``{mmap_dir}/{dataset}/tokens.npy`` etc.
    --phase1-checkpoint: Phase 1 checkpoint that provides the encoder base
                         weights (frozen throughout KB training).
    --stage-a-checkpoint: Optional Stage A checkpoint. When provided, the
                          ``encoder.l0.*`` weights from Stage A are merged
                          on top of Phase 1's base so the cache reflects
                          Stage A's text-adapted L0 behavior. Omit to
                          encode with bare Phase 1 weights (bootstrap
                          before Stage A).
    --output-dir: root directory for the :class:`L0Cache` layout.
    --retention-json: JSON file mapping dataset name → retention ratio.

Loads ONLY ``encoder.l0`` (skips L1 + projection_block weights — they're
constructed but never consulted) via
:meth:`BgKITEncoder.load_l0_only`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from bgkit.data.article_token_store import ArticleTokenStore
from bgkit.data.l0_cache import (
    L0CacheWriter,
    update_dataset_index,
    write_cache_manifest,
)
from bgkit.utils.packing import position_ids_from_cu


def _load_encoder(
    phase1_checkpoint: Path,
    stage_a_checkpoint: Path | None,
) -> torch.nn.Module:
    """Load Phase 1 ``encoder.l0`` and optionally overlay Stage A's L0.

    With L0 LoRA dropped (Phase 2 KB now trains ``encoder.l0`` weights
    directly in Stage A), Stage A's L0 weights are merged on top of the
    Phase 1 base. Without ``stage_a_checkpoint`` the cache reflects bare
    Phase 1 L0 (bootstrap pre-compute).
    """
    from bgkit.models.encoder import BgKITEncoder
    from bgkit.training.checkpointing import load_checkpoint

    _meta, state = load_checkpoint(phase1_checkpoint)
    model_state = state.get("model", {})
    encoder_state = {
        k.replace("encoder.", "", 1): v
        for k, v in model_state.items()
        if k.startswith("encoder.")
    }

    if stage_a_checkpoint is not None:
        _meta_a, state_a = load_checkpoint(stage_a_checkpoint)
        model_state_a = state_a.get("model", {})
        l0_state_a = {
            k.replace("encoder.", "", 1): v
            for k, v in model_state_a.items()
            if k.startswith("encoder.l0.")
        }
        if not l0_state_a:
            raise RuntimeError(
                f"Stage A checkpoint {stage_a_checkpoint} has no "
                "encoder.l0.* keys to merge."
            )
        encoder_state.update(l0_state_a)

    encoder = BgKITEncoder.load_l0_only(
        "Qwen/Qwen3.5-0.8B-Base", encoder_state, hidden_dim=1024,
    )
    return encoder


def _pack_batch(
    tokens_list: list[torch.Tensor],
    embed_tokens: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate per-article token tensors into a packed batch.

    Returns ``(input_embeds, cu_seqlens, position_ids)`` where
    ``input_embeds`` is ``(N, D)`` bf16-able, ``cu_seqlens`` is ``(B+1,)``
    int32, and ``position_ids`` is ``(N,)`` int64.
    """
    lengths = torch.tensor(
        [int(t.size(0)) for t in tokens_list], dtype=torch.int32,
    )
    cu_seqlens = torch.zeros(len(tokens_list) + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    flat_tokens = torch.cat(
        [t.to(device=device, dtype=torch.long) for t in tokens_list], dim=0,
    )
    input_embeds = embed_tokens(flat_tokens)
    cu_seqlens = cu_seqlens.to(device)
    position_ids = position_ids_from_cu(cu_seqlens, int(flat_tokens.size(0)))
    return input_embeds, cu_seqlens, position_ids


def _encode_and_write_batch(
    *,
    pending_ids: list[str],
    dataset: str,
    token_store: ArticleTokenStore,
    encoder: torch.nn.Module,
    embed_tokens: torch.nn.Module,
    device: torch.device,
    retention_ratio: float,
    writer: L0CacheWriter,
    index_rows: list[tuple[str, int]],
) -> int:
    """Encode + write a batch. Returns the number of articles actually
    written to the shard (excludes missing articles and zero-survivor rows).

    Uses the packed encoder forward: all articles in the batch are packed
    into one flat ``(N, D)`` buffer with per-article ``cu_seqlens``; the
    encoder's varlen attention keeps articles from attending across their
    boundaries. Survivors come out flat; ``survivor_cu_seqlens`` marks the
    per-article boundaries for the on-disk cache.
    """
    if not pending_ids:
        return 0
    present = [aid for aid in pending_ids if token_store.has(dataset, aid)]
    if not present:
        return 0
    # Pull raw per-article token tensors (skip the padded batch API).
    tokens_list = [token_store.get(dataset, aid) for aid in present]
    with torch.no_grad():
        input_embeds, cu_seqlens, position_ids = _pack_batch(
            tokens_list, embed_tokens, device,
        )
        out = encoder.l0(
            content_embeddings=input_embeds,
            content_cu_seqlens=cu_seqlens,
            content_position_ids=position_ids,
            target_ratio=retention_ratio,
        )
        survivors_flat = out.survivor_embeddings.cpu().float().numpy()
        survivor_cu = out.survivor_cu_seqlens.cpu().to(torch.int64).numpy()
    n_written = 0
    for i, aid in enumerate(present):
        start = int(survivor_cu[i])
        end = int(survivor_cu[i + 1])
        n = end - start
        if n == 0:
            continue
        row = survivors_flat[start:end]
        writer.add(aid, row)
        index_rows.append((aid, len(index_rows)))
        n_written += 1
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", required=True, type=Path)
    parser.add_argument(
        "--mmap-dir",
        required=False,
        type=Path,
        help="Phase 2 mmap root (default: $DATA_DIR/mmap/phase2)",
    )
    parser.add_argument(
        "--phase1-checkpoint",
        "--checkpoint",
        dest="phase1_checkpoint",
        required=True,
        type=Path,
        help="Phase 1 checkpoint providing the frozen encoder base.",
    )
    parser.add_argument(
        "--stage-a-checkpoint",
        required=False,
        type=Path,
        default=None,
        help="Stage A KRKBTrainer checkpoint providing trained encoder.l0.* "
             "weights. Omit for bootstrap pre-compute before Stage A.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank — recorded in the cache manifest only (Phase 2 KB "
             "now trains encoder.l0 directly without an L0 LoRA wrapper).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help="LoRA alpha — recorded in the cache manifest only.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--retention-json", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=8192)
    args = parser.parse_args()

    if args.mmap_dir is None:
        from bgkit.env import DATA_DIR

        args.mmap_dir = Path(DATA_DIR) / "mmap" / "phase2"

    retention = {k: float(v) for k, v in json.loads(args.retention_json.read_text()).items()}
    by_dataset: dict[str, list[str]] = defaultdict(list)
    with args.articles.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            by_dataset[str(row["dataset"])].append(str(row["article_id"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = _load_encoder(
        args.phase1_checkpoint,
        args.stage_a_checkpoint,
    )
    encoder.to(device).eval()
    embed_tokens = encoder.l0.backbone.get_input_embeddings()
    token_store = ArticleTokenStore(args.mmap_dir)

    for dataset, article_ids in by_dataset.items():
        ratio = retention.get(dataset, 0.10)
        shard_idx = 0
        writer = L0CacheWriter(args.output_dir, dataset, f"shard_{shard_idx:04d}")
        count_in_shard = 0
        index_rows: list[tuple[str, int]] = []
        pending: list[str] = []

        for aid in article_ids:
            pending.append(aid)
            if len(pending) >= args.batch_size:
                written = _encode_and_write_batch(
                    pending_ids=pending,
                    dataset=dataset,
                    token_store=token_store,
                    encoder=encoder,
                    embed_tokens=embed_tokens,
                    device=device,
                    retention_ratio=ratio,
                    writer=writer,
                    index_rows=index_rows,
                )
                pending.clear()
                count_in_shard += written
                if count_in_shard >= args.shard_size:
                    _, shard_index = writer.finalize()
                    update_dataset_index(
                        args.output_dir,
                        dataset,
                        f"shard_{shard_idx:04d}",
                        shard_index,
                    )
                    shard_idx += 1
                    writer = L0CacheWriter(
                        args.output_dir, dataset, f"shard_{shard_idx:04d}",
                    )
                    count_in_shard = 0
                    index_rows = []
        if pending:
            _encode_and_write_batch(
                pending_ids=pending,
                dataset=dataset,
                token_store=token_store,
                encoder=encoder,
                embed_tokens=embed_tokens,
                device=device,
                retention_ratio=ratio,
                writer=writer,
                index_rows=index_rows,
            )
        n, shard_index = writer.finalize()
        if n:
            update_dataset_index(
                args.output_dir,
                dataset,
                f"shard_{shard_idx:04d}",
                shard_index,
            )

        # Record provenance so KRKBTrainer.setup can verify the cache
        # was built against the same checkpoints + LoRA shape it's about
        # to load. Without this manifest the Stage A → Stage B handoff
        # is silently corruptible.
        write_cache_manifest(
            args.output_dir,
            dataset,
            phase1_checkpoint=args.phase1_checkpoint,
            stage_a_checkpoint=args.stage_a_checkpoint,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            retention=ratio,
            extra={
                "shard_count": shard_idx + 1,
                "article_count": len(article_ids),
            },
        )

        print(f"dataset={dataset} — wrote {len(article_ids)} articles across "
              f"{shard_idx + 1} shards")


if __name__ == "__main__":
    main()
