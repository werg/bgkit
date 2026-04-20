#!/usr/bin/env python
"""Ablation analyzer: with-vs-without-survivors per-token CE, by position.

Decisive test for the question "is the decoder using survivor embeddings,
or is the loss floor ~1.0 mostly autoregressive?". Runs the same eval
samples through the decoder twice on the same forward template:

  Pass A: survivor embeddings as produced by the encoder
  Pass B: survivor embeddings replaced with zeros (same shape, same mask;
          decoder still attends to the splice positions, but they carry
          no information)

For each supervised content token, dumps both losses + position-from-splice
+ position-in-content. Output parquet supports computing
``Δ = loss_without − loss_with`` by position bucket — a large, persistent Δ
at depth confirms survivors are doing real work; a small Δ confirms the
decoder is autoregressing past the splice.

Usage (compose service ``analyze-phase1-step3-survivor-ablation``):
    python scripts/analyze_step3_survivor_ablation.py \\
        +experiment=phase1_step3 \\
        +analyze.checkpoint=/workspace/checkpoints/phase1_step3_legacy_step1500_20260417_042208 \\
        +analyze.output_dir=/workspace/data/diagnostics/ablation_step3_step1500 \\
        +analyze.max_samples=500 \\
        ++compute.attention_implementation=sdpa \\
        ++training.max_batch_tokens=8192
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import pandas as pd
import structlog
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from bgkit.utils.diagnostic_harness import (
    apply_diagnostic_patches,
    prepare_diagnostic_trainer,
)

logger = structlog.get_logger()


def _compute_per_token_ce(
    hidden_states: torch.Tensor,
    lm_head: torch.nn.Module,
    target_ids: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Per-token next-token CE without materialising ``(B, S, V)`` logits."""
    shift_h = hidden_states[:, :-1, :]
    shift_t = target_ids[:, 1:]
    b, s_minus_1, _ = shift_h.shape
    out = shift_h.new_zeros(b, s_minus_1, dtype=torch.float32)
    lm_head_weight = lm_head.weight
    lm_head_bias = getattr(lm_head, "bias", None)
    for start in range(0, s_minus_1, chunk_size):
        end = min(start + chunk_size, s_minus_1)
        h_chunk = shift_h[:, start:end, :].contiguous()
        t_chunk = shift_t[:, start:end].contiguous()
        logits = F.linear(h_chunk, lm_head_weight, lm_head_bias)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            t_chunk.reshape(-1),
            reduction="none",
        ).view(b, end - start)
        out[:, start:end] = ce.float()
    return out


def _decoder_forward_per_token_ce(
    trainer, batch, survivors: torch.Tensor, survivor_mask: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Run decoder with given survivors, return per-token CE + masks.

    Returns ``(per_token_ce, shift_mask, max_pre, max_survivors)``.
    """
    device = trainer.device
    token_ids = batch["token_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    loss_mask = batch["loss_mask"].to(device)
    splice_start = batch["bgkit_splice_start"].to(device)
    splice_len = batch["bgkit_splice_len"].to(device)

    with torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda",
    ):
        out = trainer.decoder.forward_with_single_splice(
            survivor_embeddings=survivors,
            survivor_attention_mask=survivor_mask,
            token_ids=token_ids,
            token_attention_mask=attention_mask,
            splice_starts=splice_start,
            splice_lengths=splice_len,
            loss_mask=loss_mask,
            return_hidden_states=True,
        )
    per_token_ce = _compute_per_token_ce(
        out.hidden_states, out.lm_head, out.token_ids, chunk_size=chunk_size,
    )
    shift_mask = (
        out.attention_mask[:, 1:].bool() & out.loss_mask[:, 1:].bool()
    )
    max_pre = int(splice_start.max().item())
    max_survivors = int(survivors.size(1))
    return per_token_ce, shift_mask, max_pre, max_survivors


def _run_analysis(trainer, cfg: DictConfig) -> None:
    encoder = trainer.encoder
    decoder = trainer.decoder
    eval_dl = trainer.eval_dataloader
    max_samples = int(cfg.analyze.get("max_samples", 500))
    chunk_size = int(cfg.analyze.get("ce_chunk_size", 256))

    output_dir = Path(str(cfg.analyze.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "ablation_per_token.parquet"
    summary_path = output_dir / "summary.json"

    encoder.eval()
    decoder.eval()

    rows: list[dict] = []
    sample_idx = 0
    sum_with = 0.0
    sum_without = 0.0
    n_supervised = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dl):
            if sample_idx >= max_samples:
                break

            languages = batch.get("languages", None)
            content_mask = batch["content_attention_mask"].to(trainer.device)

            enc_out = trainer._compute_survivors(batch)
            survivors_real = enc_out.survivor_embeddings
            survivor_mask = enc_out.survivor_attention_mask

            # Pass A: real survivors
            ce_with, shift_mask, max_pre, max_survivors = _decoder_forward_per_token_ce(
                trainer, batch, survivors_real, survivor_mask, chunk_size,
            )

            # Pass B: zeroed survivors (same shape and mask — decoder still
            # attends to the splice positions, but they carry no signal).
            survivors_zero = torch.zeros_like(survivors_real)
            ce_without, _, _, _ = _decoder_forward_per_token_ce(
                trainer, batch, survivors_zero, survivor_mask, chunk_size,
            )

            splice_end_global = max_pre + max_survivors
            batch_size = ce_with.size(0)
            n_survivors_per_sample = survivor_mask.sum(dim=1).tolist()
            content_len_per_sample = content_mask.sum(dim=1).tolist()

            for b in range(batch_size):
                if sample_idx >= max_samples:
                    break
                mask_b = shift_mask[b]
                valid_positions = mask_b.nonzero(as_tuple=False).squeeze(-1)
                if valid_positions.numel() == 0:
                    sample_idx += 1
                    continue

                first_sup = int(valid_positions[0].item())
                lang = (
                    languages[b] if languages is not None and b < len(languages) else ""
                )
                n_surv = int(n_survivors_per_sample[b])
                content_len = int(content_len_per_sample[b])

                tgt_positions = valid_positions + 1  # next-token positions
                pos_from_splice_end = (
                    (tgt_positions - splice_end_global).cpu().tolist()
                )
                pos_in_content = (
                    (tgt_positions - (first_sup + 1)).cpu().tolist()
                )

                ce_with_vals = ce_with[b, valid_positions].cpu().tolist()
                ce_without_vals = ce_without[b, valid_positions].cpu().tolist()

                for pos_fs, pos_ic, lw, lwo in zip(
                    pos_from_splice_end,
                    pos_in_content,
                    ce_with_vals,
                    ce_without_vals,
                    strict=True,
                ):
                    rows.append({
                        "sample_idx": sample_idx,
                        "pos_from_splice_end": int(pos_fs),
                        "pos_in_content": int(pos_ic),
                        "loss_with": float(lw),
                        "loss_without": float(lwo),
                        "delta": float(lwo) - float(lw),
                        "language": lang,
                        "n_survivors": n_surv,
                        "content_len": content_len,
                    })

                sum_with += sum(ce_with_vals)
                sum_without += sum(ce_without_vals)
                n_supervised += len(ce_with_vals)
                sample_idx += 1

            if batch_idx % 5 == 0:
                logger.info(
                    "ablation_progress",
                    samples=sample_idx,
                    rows=len(rows),
                    mean_with=sum_with / max(n_supervised, 1),
                    mean_without=sum_without / max(n_supervised, 1),
                    mean_delta=(sum_without - sum_with) / max(n_supervised, 1),
                )

    if not rows:
        raise RuntimeError("No supervised tokens captured.")

    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)
    logger.info("wrote_parquet", path=str(parquet_path), rows=len(df))

    # Position-bucket breakdown — the decisive view.
    bins = [-1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 10**9]
    labels = ["0-4", "4-8", "8-16", "16-32", "32-64", "64-128",
              "128-256", "256-512", "512-1024", "1024+"]
    df_pic = df.assign(
        bucket=pd.cut(df["pos_in_content"], bins=bins, labels=labels),
    )
    by_pic = (
        df_pic.groupby("bucket", observed=True)
        .agg(
            count=("loss_with", "count"),
            mean_with=("loss_with", "mean"),
            mean_without=("loss_without", "mean"),
            mean_delta=("delta", "mean"),
        )
        .round(4)
        .to_dict()
    )

    summary = {
        "checkpoint": str(cfg.analyze.checkpoint),
        "samples_analyzed": sample_idx,
        "supervised_tokens": int(n_supervised),
        "overall": {
            "mean_loss_with_survivors": sum_with / max(n_supervised, 1),
            "mean_loss_without_survivors": sum_without / max(n_supervised, 1),
            "mean_delta": (sum_without - sum_with) / max(n_supervised, 1),
        },
        "by_pos_in_content": by_pic,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote_summary", path=str(summary_path))

    print("\n=== Survivor ablation summary ===")
    print(f"Checkpoint: {cfg.analyze.checkpoint}")
    print(f"Samples: {sample_idx}, supervised tokens: {n_supervised}")
    print(f"Mean loss WITH survivors:    {summary['overall']['mean_loss_with_survivors']:.4f}")
    print(f"Mean loss WITHOUT survivors: {summary['overall']['mean_loss_without_survivors']:.4f}")
    print(f"Mean Δ (without − with):     {summary['overall']['mean_delta']:+.4f}")
    print()
    print("By pos_in_content bucket:")
    print(f"  {'bucket':>10}  {'count':>8}  {'with':>8}  {'without':>8}  {'Δ':>8}")
    for label in labels:
        if label not in by_pic.get("count", {}):
            continue
        c = by_pic["count"][label]
        w = by_pic["mean_with"][label]
        wo = by_pic["mean_without"][label]
        d = by_pic["mean_delta"][label]
        print(f"  {label:>10}  {c:>8}  {w:>8.3f}  {wo:>8.3f}  {d:>+8.3f}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    apply_diagnostic_patches()
    from bgkit.training.phase1.decoder_init import DecoderInitTrainer

    trainer = prepare_diagnostic_trainer(
        cfg,
        trainer_cls=DecoderInitTrainer,
        expected_phases=("phase1_step3",),
    )
    _run_analysis(trainer, cfg)


if __name__ == "__main__":
    main()
