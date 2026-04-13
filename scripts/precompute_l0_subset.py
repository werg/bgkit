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
                          L0 LoRA adapter weights from it are loaded and
                          activated so the cache reflects Stage A's
                          text-adapted L0 behavior. Omit to encode with
                          bare Phase 1 weights (bootstrap before Stage A).
    --output-dir: root directory for the :class:`L0Cache` layout.
    --retention-json: JSON file mapping dataset name → retention ratio.

This script reuses the canonical Phase 2 token store via
:class:`bgkit.data.article_token_store.ArticleTokenStore`. There is no
sidecar ``{dataset}_tokens.parquet`` — tokens come from the same mmap
files that the single-doc Phase 2 trainers consume.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from bgkit.data.article_token_store import ArticleTokenStore
from bgkit.data.l0_cache import (
    L0CacheWriter,
    update_dataset_index,
    write_cache_manifest,
)
from bgkit.models.lora_encoder import DEFAULT_LORA_TARGETS, LoRARouter


def _load_encoder_and_lora(
    phase1_checkpoint: Path,
    stage_a_checkpoint: Path | None,
    lora_rank: int,
    lora_alpha: float | None,
) -> tuple[torch.nn.Module, torch.nn.Module, LoRARouter | None]:
    """Load Phase 1 encoder + ICE, optionally install and load a Stage A LoRA.

    When ``stage_a_checkpoint`` is provided we install the same LoRA router
    configuration that ``KRKBTrainer._install_lora`` uses, then load the
    Stage A model state dict into the encoder so the LoRA adapters receive
    their trained weights. Without this the ``lora_level="l0"`` hint at
    encoder forward time is a no-op — the router returns None and the
    bare Phase 1 weights run, silently discarding Stage A's training.
    """
    from bgkit.models.encoder import BgKITEncoder
    from bgkit.models.ice import ICE
    from bgkit.training.checkpointing import load_checkpoint

    _meta, state = load_checkpoint(phase1_checkpoint)
    model_state = state.get("model", {})
    encoder_state = {
        k.replace("encoder.", "", 1): v
        for k, v in model_state.items()
        if k.startswith("encoder.")
    }
    ice_state = {
        k.replace("ice.", "", 1): v
        for k, v in model_state.items()
        if k.startswith("ice.")
    }
    encoder = BgKITEncoder.from_pretrained_with_state_dict(
        "Qwen/Qwen3.5-0.8B-Base", encoder_state, hidden_dim=1024,
    )
    ice = ICE(input_dim=1024, hidden_dim=128, num_layers=3)
    if ice_state:
        ice.load_state_dict(ice_state, strict=False)

    router: LoRARouter | None = None
    if stage_a_checkpoint is not None:
        # Install LoRA router with the same target modules and level ranks
        # that the KRKBTrainer uses. Adapters are zero-initialized here;
        # the Stage A state dict load below populates them.
        router = LoRARouter.install(
            encoder,
            target_names=DEFAULT_LORA_TARGETS,
            levels={"l0": lora_rank, "l1": lora_rank},
            alpha=alpha_or_default(lora_alpha, lora_rank),
            dropout=0.0,
        )
        LoRARouter.bind(router)

        _meta_a, state_a = load_checkpoint(stage_a_checkpoint)
        model_state_a = state_a.get("model", {})
        encoder_state_a = {
            k.replace("encoder.", "", 1): v
            for k, v in model_state_a.items()
            if k.startswith("encoder.")
        }
        if not encoder_state_a:
            raise RuntimeError(
                f"Stage A checkpoint {stage_a_checkpoint} has no encoder.* "
                "keys — cannot load L0 LoRA weights."
            )
        # strict=False because Phase 1 base keys are already loaded and the
        # checkpoint may also contain decoder/ice keys we're ignoring here.
        encoder.load_state_dict(encoder_state_a, strict=False)
        # Sanity check: at least some lora_A/lora_B params should have loaded.
        loaded_lora = sum(
            1 for k in encoder_state_a
            if ".adapters.l0." in k or ".adapters.l1." in k
        )
        if loaded_lora == 0:
            raise RuntimeError(
                f"Stage A checkpoint {stage_a_checkpoint} contains no LoRA "
                "adapter keys (looked for '.adapters.l0.' / '.adapters.l1.'). "
                "Was this checkpoint saved by KRKBTrainer?"
            )

    return encoder, ice, router


def alpha_or_default(alpha: float | None, rank: int) -> float:
    return float(alpha) if alpha is not None else 2.0 * rank


def _encode_and_write_batch(
    *,
    pending_ids: list[str],
    dataset: str,
    token_store: ArticleTokenStore,
    encoder: torch.nn.Module,
    ice: torch.nn.Module,
    embed_tokens: torch.nn.Module,
    device: torch.device,
    retention_ratio: float,
    writer: L0CacheWriter,
    index_rows: list[tuple[str, int]],
) -> int:
    """Encode + write a batch. Returns the number of articles actually
    written to the shard (excludes missing articles and zero-survivor rows).
    Callers use this to advance the per-shard counter — counting against
    the requested batch size instead would close shards early when many
    articles are absent from the token store.
    """
    if not pending_ids:
        return 0
    present = [aid for aid in pending_ids if token_store.has(dataset, aid)]
    if not present:
        return 0
    tokens, mask = token_store.get_batch(dataset, present)
    tokens = tokens.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        input_embeds = embed_tokens(tokens)
        scores = ice(input_embeds)
        survivor_mask = torch.zeros_like(mask, dtype=torch.bool)
        for i in range(tokens.size(0)):
            length = int(mask[i].sum())
            keep = max(1, math.ceil(length * retention_ratio))
            _, topk = torch.topk(scores[i, :length], min(keep, length))
            survivor_mask[i, topk] = True
        out = encoder(
            input_embeddings=input_embeds,
            survivor_mask=survivor_mask,
            attention_mask=mask,
            lora_level="l0",
        )
        survivors = out.survivor_embeddings.cpu().float().numpy()
        survivor_mask_cpu = out.survivor_attention_mask.cpu().numpy()
    n_written = 0
    for i, aid in enumerate(present):
        n = int(survivor_mask_cpu[i].sum())
        if n == 0:
            continue
        row = survivors[i, :n]
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
        help="Stage A KRKBTrainer checkpoint providing trained L0 LoRA weights. "
             "Omit for bootstrap pre-compute before Stage A has trained.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank — must match the rank used during Stage A training.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help="LoRA alpha. Defaults to 2*rank (matches Stage A default).",
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
    encoder, ice, router = _load_encoder_and_lora(
        args.phase1_checkpoint,
        args.stage_a_checkpoint,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    encoder.to(device).eval()
    ice.to(device).eval()
    if router is not None:
        router.to(device)
    embed_tokens = encoder.compressor.backbone.get_input_embeddings()
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
                    ice=ice,
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
                ice=ice,
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
