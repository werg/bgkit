"""Re-anchor encoder.l0.auto_repro_head to the survivor-subset distribution.

After the L0/L1 split rebuild, auto_repro_head's input distribution shifted:
Joint Block trained it on FULL-content L0 outputs (no compression, no
survive_embedding); now in Step 5 it sees SURVIVOR-SUBSET L0 outputs that
have survive_embedding's signal baked in via blocks 2-5. The new L0→L1
bridge requires a high-quality auto_repro_head — it must invert L0's
encoding back to encoder.embed_tokens(content_ids) at SURVIVOR positions.

This script does a focused MSE+cosine re-train, freezing everything except
``encoder.l0.auto_repro_head``. Direct supervision against
``encoder.l0.backbone.embed_tokens(content_ids)[at survivor positions]``.
No decoder loop — clean per-position gradient signal.

Usage::

    python scripts/repair_auto_repro_head.py [--source SPLIT_CKPT] [--steps N]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from bgkit.data.collators import collate_compression
from bgkit.data.datasets.commit_encoding_dataset import CommitEncodingDataset
from bgkit.data.samplers import PackedTokenBudgetSampler
from bgkit.env import get_checkpoint_dir, get_data_dir
from bgkit.models.encoder import BgKITEncoder
from bgkit.training.checkpoint_registry import CheckpointRegistry, RegistryEntry
from bgkit.training.checkpointing import load_checkpoint
from bgkit.utils.attention_backend import resolve_attention_implementation
from bgkit.utils.packing import position_ids_from_cu

logger = logging.getLogger("repair_auto_repro_head")


def _build_encoder(state_dicts: dict, attn_impl: str) -> BgKITEncoder:
    """Construct encoder from a split-layout state dict."""
    return BgKITEncoder.from_pretrained_with_state_dict(
        "Qwen/Qwen3.5-0.8B-Base",
        state_dicts["encoder"],
        hidden_dim=1024,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )


def _freeze_all_except_auto_repro(encoder: BgKITEncoder) -> tuple[int, int]:
    """Freeze every encoder param except encoder.l0.auto_repro_head.

    Returns (trainable_count, total_count) for sanity logging.
    """
    encoder.requires_grad_(False)
    for p in encoder.l0.auto_repro_head.parameters():
        p.requires_grad_(True)
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder.parameters())
    return trainable, total


def _l0_forward_at_ratio(
    encoder: BgKITEncoder,
    batch: dict,
    target_ratio_l0: float,
    device: torch.device,
):
    """Run L0 forward (no_grad on backbone — only auto_repro is trainable).

    Returns (auto_repro_pred, target_emb, survivor_mask, content_ids):
        auto_repro_pred: (N_surv, D) — auto_repro_head(self.norm(survivor_embeddings))
        target_emb: (N_surv, D) — embed_tokens(content_ids)[at survivors]
        survivor_mask: (N_content,) bool
        content_ids: (N_content,) int — for diagnostics
    """
    file_ids = batch["content_token_ids"].to(device)
    cu_file = batch["cu_file_seqlens"].to(device)
    content_position_ids = batch["content_position_ids"].to(device)
    prompt_ids = batch["prompt_token_ids"].to(device)
    prompt_cu = batch["prompt_cu_seqlens"].to(device)
    prompt_position_ids = position_ids_from_cu(prompt_cu, int(prompt_ids.shape[0]))

    embed = encoder.l0.backbone.get_input_embeddings()
    # Backbone forward + survivor selection: no grad needed for backbone
    # (only the auto_repro_head is trainable; the survivor_embeddings carry
    # the training signal for the head).
    with torch.no_grad():
        encoder.l0.eval()
        content_emb = embed(file_ids)
        prompt_emb = embed(prompt_ids)
        l0_out = encoder.l0(
            content_embeddings=content_emb,
            content_cu_seqlens=cu_file,
            content_position_ids=content_position_ids,
            prompt_embeddings=prompt_emb,
            prompt_cu_seqlens=prompt_cu,
            prompt_position_ids=prompt_position_ids,
            target_ratio=target_ratio_l0,
        )
        survivors = l0_out.survivor_embeddings  # (N_surv, D), pre-norm
        survivor_mask = l0_out.survivor_mask  # (N_content,) bool
        # Targets: input embeddings at survivor positions
        target_emb = embed(file_ids[survivor_mask]).detach()

    encoder.l0.auto_repro_head.train()
    auto_repro_pred = encoder.l0.auto_reproduce(survivors)  # applies l0.norm internally

    return auto_repro_pred, target_emb, survivor_mask, file_ids


def _losses(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """MSE + cosine + log-norm-ratio losses."""
    pred_f = pred.float()
    target_f = target.float()
    mse = F.mse_loss(pred_f, target_f)
    cos = 1.0 - F.cosine_similarity(pred_f, target_f, dim=-1).mean()
    pred_norm = pred_f.norm(dim=-1).mean().clamp(min=1e-6)
    tgt_norm = target_f.norm(dim=-1).mean().clamp(min=1e-6)
    norm_log = (pred_norm.log() - tgt_norm.log()).pow(2)
    return {"mse": mse, "cos": cos, "norm_log": norm_log}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-name",
        default=None,
        help="Source split-layout checkpoint name. Default: latest phase1_step4_split.",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-ratio-l0", type=float, default=0.20,
                        help="L0 selection ratio during training. 0.20 matches Step 4.")
    parser.add_argument("--max-batch-tokens", type=int, default=4096)
    # Cosine-dominant weighting: MSE alone collapses prediction magnitude
    # to zero (trivially reduces MSE + norm_log) without aligning
    # directions. Cosine must dominate to drive directional alignment;
    # norm_log holds magnitude in the right neighborhood; MSE is a small
    # fine-tune signal once direction is in place.
    parser.add_argument("--mse-weight", type=float, default=0.1)
    parser.add_argument("--cos-weight", type=float, default=5.0)
    parser.add_argument("--norm-weight", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-suffix", default="autorepro_repaired")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    checkpoint_dir = get_checkpoint_dir()
    registry = CheckpointRegistry(checkpoint_dir)
    registry.backfill(checkpoint_dir)

    if args.source_name is None:
        latest = registry.latest(phase="phase1_step4_split")
        if latest is None:
            logger.error("No phase1_step4_split checkpoint found.")
            return 1
        src_name = latest.name
    else:
        src_name = args.source_name

    src_path = checkpoint_dir / src_name
    logger.info("loading source: %s", src_path)
    src_metadata, state_dicts = load_checkpoint(src_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn_impl = resolve_attention_implementation("auto")
    encoder = _build_encoder(state_dicts, attn_impl)
    encoder.to(device)

    trainable, total = _freeze_all_except_auto_repro(encoder)
    logger.info("trainable params: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    # Dataset: commit_encoding (matches Step 5 distribution)
    data_dir = get_data_dir()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    from bgkit.data.chat_template import TOOL_CONFIGS
    import json
    variant_path = Path("configs/prompt_variants/commit_encoding.json")
    if variant_path.exists():
        variants = json.loads(variant_path.read_text())
    else:
        variants = [{
            "system_prompt": "You are an AI assistant with access to the bgkit_reproduce_commit tool.",
            "user_prompt": "Reproduce the commit",
            "compression_prompt": "Reproduce the complete commit from compressed context",
            "response_prefix": "Here is the reconstructed commit:",
        }]
    dataset = CommitEncodingDataset(
        data_dir=str(data_dir / "processed/commit_encoding"),
        tokenizer=tokenizer,
        variant_bank=variants,
        config=TOOL_CONFIGS["commit_encoding"],
        max_diff_tokens_per_file=4096,
        max_files_per_commit=16,
        max_message_tokens=256,
        seed=42,
    )
    # Filter to small samples for speed
    indices = list(range(len(dataset)))
    train_indices = indices[: min(len(indices), 5000)]
    train_subset = Subset(dataset, train_indices)
    lengths = [dataset.token_length(i) for i in train_indices]
    import numpy as np
    sampler = PackedTokenBudgetSampler(
        train_subset,
        lengths=np.array(lengths, dtype=np.int64),
        max_batch_tokens=args.max_batch_tokens,
        shuffle=True,
        seed=42,
    )
    dataloader = DataLoader(
        train_subset,
        batch_sampler=sampler,
        collate_fn=collate_compression,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        [p for p in encoder.parameters() if p.requires_grad],
        lr=args.lr,
    )

    step = 0
    iter_dl = iter(dataloader)
    while step < args.steps:
        try:
            batch = next(iter_dl)
        except StopIteration:
            iter_dl = iter(dataloader)
            batch = next(iter_dl)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            auto_repro_pred, target_emb, survivor_mask, content_ids = _l0_forward_at_ratio(
                encoder, batch, args.target_ratio_l0, device,
            )
            losses = _losses(auto_repro_pred, target_emb)
            total_loss = (
                args.mse_weight * losses["mse"]
                + args.cos_weight * losses["cos"]
                + args.norm_weight * losses["norm_log"]
            )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % args.log_every == 0:
            with torch.no_grad():
                cos_sim = F.cosine_similarity(
                    auto_repro_pred.float(), target_emb.float(), dim=-1,
                ).mean().item()
                pred_norm = auto_repro_pred.float().norm(dim=-1).mean().item()
                tgt_norm = target_emb.float().norm(dim=-1).mean().item()
            logger.info(
                "step=%d loss=%.4f mse=%.4f cos_loss=%.4f norm_log=%.4f "
                "cos_sim=%.4f pred_norm=%.2f tgt_norm=%.2f n_surv=%d",
                step, total_loss.item(),
                losses["mse"].item(), losses["cos"].item(), losses["norm_log"].item(),
                cos_sim, pred_norm, tgt_norm, int(survivor_mask.sum().item()),
            )
        step += 1

    # Save updated encoder back into the source checkpoint.
    out_dir = checkpoint_dir / f"{src_name}_{args.save_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    enc_sd = encoder.state_dict()
    torch.save(enc_sd, out_dir / "encoder.pt")
    # Carry through other artifacts.
    for fname in ("decoder.pt", "decoder_merged.pt", "l0.pt", "l1.pt", "projection_block.pt"):
        src_file = src_path / fname
        if src_file.is_file():
            import shutil
            shutil.copy(src_file, out_dir / fname)
    # Metadata.
    import json
    metadata = {
        "phase": "phase1_step4_split",
        "step": getattr(src_metadata, "step", 0),
        "epoch": 0,
        "parent_checkpoint": src_name,
        "metrics": getattr(src_metadata, "metrics", None),
        "schedule_params": None,
        "training_state": None,
        "optimizer_type": "adamw",
        "note": (
            f"Re-anchored encoder.l0.auto_repro_head via {args.steps} steps of "
            f"focused MSE+cosine training at target_ratio_l0={args.target_ratio_l0}. "
            "Direct supervision against encoder.embed_tokens(content_ids)[at survivors] "
            "to fix the survivor-subset + survive_embedding distribution shift "
            "introduced by the L0/L1 split rebuild. All other params frozen."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    entry = RegistryEntry(
        name=out_dir.name,
        phase="phase1_step4_split",
        step=int(getattr(src_metadata, "step", 0) or 0),
        epoch=0,
        timestamp=datetime.now(UTC).isoformat(),
        metrics=getattr(src_metadata, "metrics", None) or {},
        parent_checkpoint=src_name,
        notes="auto_repro_head re-anchored; everything else unchanged",
        tags=["autorepro_repaired"],
    )
    registry.register(entry)
    logger.info("wrote repaired checkpoint: %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
