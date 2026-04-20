#!/usr/bin/env python
"""Survivor-behaviour analyzer for Phase 1 Step 3.

Companion to ``analyze_step3_loss.py``. Answers three questions about
compression selection before we flip ``floor_post_warmup`` config:

1. Counterfactual: does forcing ``min_per_sample=1`` materially reduce loss
   on zero-survivor samples, or does it just pick noise?
2. Logit-distribution: for zero-survivor samples, is ``max(logits_for_op)``
   just barely below θ (calibration issue) or far below (head is confidently
   silent)?
3. Content: what *are* the zero-survivor samples? Legitimate short code,
   boilerplate, or OOD content?

Runs eval twice on the same samples: once with the config's
``floor_post_warmup`` (0 for Step 3 at step 1500), once forcing
``min_per_sample=1``. Writes per-sample summary parquet plus a
zero-survivor-detail JSON that includes decoded content.

Usage (inside the analyze-phase1-step3-survivors compose service):
    python scripts/analyze_step3_survivors.py \
        +experiment=phase1_step3 \
        +analyze.checkpoint=/workspace/checkpoints/phase1_step3_legacy_step1500_20260417_042208 \
        +analyze.output_dir=/workspace/data/diagnostics/survivors_step1500 \
        +analyze.max_samples=500 \
        ++compute.attention_implementation=sdpa \
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


def _run_encoder_with_floor(
    trainer, batch: dict, min_per_sample: int,
):
    """Mirror of DecoderInitTrainer._compute_survivors with an explicit floor.

    Keeping the call shape byte-identical to training — only the
    ``min_per_sample`` argument is overridden.
    """
    device = trainer.device
    content_token_ids = batch["content_token_ids"].to(device)
    content_attention_mask = batch["content_attention_mask"].to(device)
    compression_prompt_ids = batch["compression_prompt_ids"].to(device)
    compression_prompt_mask = batch["compression_prompt_mask"].to(device)
    target_ratio = (
        trainer._current_target_ratio() if trainer._compression_active else None
    )
    bgkit_embed = trainer.encoder.compressor.backbone.get_input_embeddings()
    return trainer.encoder(
        input_embeddings=bgkit_embed(content_token_ids),
        attention_mask=content_attention_mask,
        prompt_embeddings=bgkit_embed(compression_prompt_ids),
        prompt_attention_mask=compression_prompt_mask,
        target_ratio=target_ratio,
        level="l0",
        min_per_sample=min_per_sample,
    )


def _compute_per_token_ce(
    hidden_states: torch.Tensor,
    lm_head: torch.nn.Module,
    target_ids: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
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


def _per_sample_loss(
    trainer, batch: dict, enc_out,
) -> torch.Tensor:
    """Return a length-B tensor of per-sample mean CE over supervised tokens."""
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
            survivor_embeddings=enc_out.survivor_embeddings,
            survivor_attention_mask=enc_out.survivor_attention_mask,
            token_ids=token_ids,
            token_attention_mask=attention_mask,
            splice_starts=splice_start,
            splice_lengths=splice_len,
            loss_mask=loss_mask,
            return_hidden_states=True,
        )
    per_token_ce = _compute_per_token_ce(
        out.hidden_states, out.lm_head, out.token_ids, chunk_size=256,
    )
    shift_mask = (
        out.attention_mask[:, 1:].bool() & out.loss_mask[:, 1:].bool()
    )
    # Per-sample sum of CE over supervised tokens / count.
    mass = (per_token_ce * shift_mask.float()).sum(dim=1)
    count = shift_mask.float().sum(dim=1).clamp(min=1)
    return (mass / count).detach().cpu()


def _run_analysis(trainer, cfg: DictConfig) -> None:
    encoder = trainer.encoder
    decoder = trainer.decoder
    tokenizer = trainer.tokenizer
    eval_dl = trainer.eval_dataloader
    max_samples = int(cfg.analyze.get("max_samples", 500))

    output_dir = Path(str(cfg.analyze.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = output_dir / "per_sample.parquet"
    detail_path = output_dir / "zero_survivor_detail.json"
    summary_path = output_dir / "summary.json"

    encoder.eval()
    decoder.eval()

    rows: list[dict] = []
    zero_detail: list[dict] = []
    sample_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dl):
            if sample_idx >= max_samples:
                break

            languages = batch.get("languages", None)
            content_token_ids = batch["content_token_ids"]  # (B, L_padded) long
            content_attention_mask = batch["content_attention_mask"]  # (B, L)

            # --- pass A: default (min_per_sample = floor_post_warmup = 0 here)
            enc_a = _run_encoder_with_floor(
                trainer, batch, min_per_sample=trainer._floor_post_warmup,
            )
            loss_a = _per_sample_loss(trainer, batch, enc_a)

            # --- pass B: forced floor = 1
            enc_b = _run_encoder_with_floor(trainer, batch, min_per_sample=1)
            loss_b = _per_sample_loss(trainer, batch, enc_b)

            batch_size = content_token_ids.size(0)
            theta_val = (
                float(enc_a.theta_tensor.item())
                if enc_a.theta_tensor is not None
                else float("nan")
            )

            logits = enc_a.logits_for_op  # (B, L_content)
            n_survivors_a = enc_a.survivor_attention_mask.sum(dim=1).cpu()
            n_survivors_b = enc_b.survivor_attention_mask.sum(dim=1).cpu()

            content_lens = content_attention_mask.sum(dim=1).cpu()

            for b in range(batch_size):
                if sample_idx >= max_samples:
                    break
                valid_len = int(content_lens[b].item())
                lang = (
                    languages[b] if languages is not None and b < len(languages) else ""
                )
                content_len = int(content_lens[b].item())

                if logits is not None and valid_len > 0:
                    sample_logits = logits[b, :valid_len].float().cpu()
                    max_logit = float(sample_logits.max().item())
                    mean_logit = float(sample_logits.mean().item())
                    std_logit = float(sample_logits.std(unbiased=False).item())
                    p90 = float(torch.quantile(sample_logits, 0.9).item())
                    p99 = (
                        float(torch.quantile(sample_logits, 0.99).item())
                        if valid_len >= 100 else max_logit
                    )
                else:
                    sample_logits = None
                    max_logit = mean_logit = std_logit = p90 = p99 = float("nan")

                n_a = int(n_survivors_a[b].item())
                n_b = int(n_survivors_b[b].item())
                la = float(loss_a[b].item())
                lb = float(loss_b[b].item())

                rows.append({
                    "sample_idx": sample_idx,
                    "language": lang,
                    "content_len": content_len,
                    "theta": theta_val,
                    "n_survivors_min0": n_a,
                    "n_survivors_min1": n_b,
                    "loss_min0": la,
                    "loss_min1": lb,
                    "loss_delta": la - lb,
                    "max_logit": max_logit,
                    "mean_logit": mean_logit,
                    "std_logit": std_logit,
                    "p90_logit": p90,
                    "p99_logit": p99,
                    "max_minus_theta": max_logit - theta_val,
                    "mean_minus_theta": mean_logit - theta_val,
                })

                # For zero-survivor samples under pass A, dump content +
                # full logits for qualitative inspection.
                if n_a == 0 and sample_logits is not None:
                    raw_content = content_token_ids[b, :valid_len].tolist()
                    content_text = tokenizer.decode(
                        raw_content, skip_special_tokens=False,
                    )
                    zero_detail.append({
                        "sample_idx": sample_idx,
                        "language": lang,
                        "content_len": content_len,
                        "theta": theta_val,
                        "loss_min0": la,
                        "loss_min1": lb,
                        "loss_delta": la - lb,
                        "n_survivors_min1": n_b,
                        "max_logit": max_logit,
                        "mean_logit": mean_logit,
                        "std_logit": std_logit,
                        "logits": sample_logits.tolist(),
                        "content_preview": content_text[:1200],
                        "content_full": content_text,
                    })

                sample_idx += 1

            if batch_idx % 5 == 0:
                logger.info(
                    "survivors_progress",
                    batch=batch_idx,
                    samples=sample_idx,
                    theta=theta_val,
                    n_zero_survivors_so_far=sum(
                        1 for r in rows if r["n_survivors_min0"] == 0
                    ),
                )

    if not rows:
        raise RuntimeError("No samples processed.")

    df = pd.DataFrame(rows)
    df.to_parquet(per_sample_path, index=False)
    logger.info("wrote_per_sample", path=str(per_sample_path), rows=len(df))

    # Content-weighted means (matches trainer eval/loss semantics)
    w = df["content_len"].clip(lower=1).astype("float64")
    weighted_mean_a = float((df["loss_min0"] * w).sum() / w.sum())
    weighted_mean_b = float((df["loss_min1"] * w).sum() / w.sum())

    zero_mask = df["n_survivors_min0"] == 0
    low_mask = df["n_survivors_min0"].between(1, 5)

    summary = {
        "checkpoint": str(cfg.analyze.checkpoint),
        "samples_analyzed": len(df),
        "theta": float(df["theta"].iloc[0]),
        "counterfactual": {
            "content_weighted_mean_loss_min0": weighted_mean_a,
            "content_weighted_mean_loss_min1": weighted_mean_b,
            "delta_min0_minus_min1": weighted_mean_a - weighted_mean_b,
        },
        "zero_survivor": {
            "n_samples": int(zero_mask.sum()),
            "mean_loss_min0": (
                float(df.loc[zero_mask, "loss_min0"].mean())
                if zero_mask.any() else None
            ),
            "mean_loss_min1": (
                float(df.loc[zero_mask, "loss_min1"].mean())
                if zero_mask.any() else None
            ),
            "mean_max_minus_theta": (
                float(df.loc[zero_mask, "max_minus_theta"].mean())
                if zero_mask.any() else None
            ),
            "median_max_minus_theta": (
                float(df.loc[zero_mask, "max_minus_theta"].median())
                if zero_mask.any() else None
            ),
        },
        "low_survivor_1to5": {
            "n_samples": int(low_mask.sum()),
            "mean_loss_min0": (
                float(df.loc[low_mask, "loss_min0"].mean())
                if low_mask.any() else None
            ),
            "mean_loss_min1": (
                float(df.loc[low_mask, "loss_min1"].mean())
                if low_mask.any() else None
            ),
        },
        "by_n_survivors_min0_bucket": (
            df.assign(
                bucket=pd.cut(
                    df["n_survivors_min0"],
                    bins=[-1, 0, 5, 20, 50, 150, 10**9],
                    labels=["0", "1-5", "6-20", "21-50", "51-150", "150+"],
                ),
            )
            .groupby("bucket", observed=True)
            .agg(
                n=("sample_idx", "count"),
                mean_loss_min0=("loss_min0", "mean"),
                mean_loss_min1=("loss_min1", "mean"),
                mean_delta=("loss_delta", "mean"),
                mean_max_minus_theta=("max_minus_theta", "mean"),
            )
            .round(4)
            .to_dict()
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote_summary", path=str(summary_path))

    detail_path.write_text(json.dumps(zero_detail, indent=2, default=str))
    logger.info("wrote_zero_detail", path=str(detail_path), count=len(zero_detail))

    print("\n=== Survivor analysis summary ===")
    print(f"Checkpoint: {cfg.analyze.checkpoint}")
    print(f"Samples analyzed: {len(df)}")
    print(f"θ = {df['theta'].iloc[0]:.4f}")
    print()
    print("Content-weighted mean loss:")
    print(f"  min_per_sample=0 (current):  {weighted_mean_a:.4f}")
    print(f"  min_per_sample=1 (forced):   {weighted_mean_b:.4f}")
    print(f"  delta:                       {weighted_mean_a - weighted_mean_b:+.4f}")
    print()
    print(f"Zero-survivor samples: {int(zero_mask.sum())}")
    if zero_mask.any():
        print(f"  mean loss with min=0: {df.loc[zero_mask, 'loss_min0'].mean():.3f}")
        print(f"  mean loss with min=1: {df.loc[zero_mask, 'loss_min1'].mean():.3f}")
        print(f"  mean(max_logit - θ):  {df.loc[zero_mask, 'max_minus_theta'].mean():.4f}")


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
