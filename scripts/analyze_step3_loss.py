#!/usr/bin/env python
"""Per-token reconstruction-loss analyzer for Phase 1 Step 3 reconstruction checkpoints.

Loads a completed Step 3 checkpoint, runs the standard training-time eval
loop with ``return_hidden_states=True`` on the decoder, and dumps per-token
CE to a parquet file plus a summary JSON. Designed to run concurrently with
ongoing training — shares GPU time but does not touch the training process
or its checkpoint.

Usage (inside Docker, via the ``analyze-phase1-step3-loss`` compose service
or ad-hoc):

    python scripts/analyze_step3_loss.py \\
        +experiment=phase1_step3 \\
        +analyze.checkpoint=/workspace/checkpoints/phase1_step3_legacy_step1500_20260417_042208 \\
        +analyze.output_dir=/workspace/data/diagnostics/analyze_step3_step1500 \\
        +analyze.max_samples=500

Outputs under ``analyze.output_dir``:
    - per_token_losses.parquet
    - summary.json
    - top_hard_tokens.csv  (top 100 worst tokens by mean loss)

The parquet has one row per *supervised* (loss_mask=1) content token with
columns: sample_idx, pos_global, pos_from_splice_end, pos_in_content,
token_id, token_str, loss, language, seq_len, n_survivors.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import pandas as pd
import structlog
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from bgkit.utils.deltanet_patch import patch_gated_delta_rule_numerics
from bgkit.utils.logging import setup_logging
from bgkit.utils.reproducibility import set_seed
from bgkit.utils.triton_alloc_patch import patch_triton_allocator
from bgkit.utils.triton_patch import patch_triton_autotuner

logger = structlog.get_logger()


def _compute_per_token_ce(
    hidden_states: torch.Tensor,
    lm_head: torch.nn.Module,
    target_ids: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Per-token next-token CE without materialising ``(B, S, V)`` logits.

    Returns shape ``(B, S-1)`` with the CE for predicting ``target_ids[:, i+1]``
    from ``hidden_states[:, i, :]``. Runs in the same dtype as ``hidden_states``
    so aggregate means are directly comparable with the trainer's reported
    eval/loss (both bf16 under the Step 3 config).
    """
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


def _run_analysis(trainer, cfg: DictConfig) -> None:
    device = trainer.device
    encoder = trainer.encoder
    decoder = trainer.decoder
    tokenizer = trainer.tokenizer
    eval_dl = trainer.eval_dataloader
    max_samples = int(cfg.analyze.get("max_samples", 500))
    chunk_size = int(cfg.analyze.get("ce_chunk_size", 256))

    output_dir = Path(str(cfg.analyze.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "per_token_losses.parquet"
    summary_path = output_dir / "summary.json"
    hard_path = output_dir / "top_hard_tokens.csv"

    encoder.eval()
    decoder.eval()

    sample_idx = 0
    all_rows: list[dict] = []
    total_loss_weighted = 0.0
    total_supervised = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dl):
            if sample_idx >= max_samples:
                break

            token_ids = batch["token_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            splice_start = batch["bgkit_splice_start"].to(device)
            splice_len = batch["bgkit_splice_len"].to(device)
            content_mask = batch["content_attention_mask"].to(device)
            languages = batch.get("languages", None)

            enc_out = trainer._compute_survivors(batch)

            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda",
            ):
                out = decoder.forward_with_single_splice(
                    survivor_embeddings=enc_out.survivor_embeddings,
                    survivor_attention_mask=enc_out.survivor_attention_mask,
                    token_ids=token_ids,
                    token_attention_mask=attention_mask,
                    splice_starts=splice_start,
                    splice_lengths=splice_len,
                    loss_mask=loss_mask,
                    return_hidden_states=True,
                )

            hidden = out.hidden_states
            token_ids_full = out.token_ids
            loss_mask_full = out.loss_mask
            attn_full = out.attention_mask
            lm_head = out.lm_head

            per_token_ce = _compute_per_token_ce(
                hidden, lm_head, token_ids_full, chunk_size=chunk_size,
            )

            shift_mask = (
                attn_full[:, 1:].bool() & loss_mask_full[:, 1:].bool()
            )

            batch_size = token_ids_full.size(0)
            max_pre = int(splice_start.max().item())
            max_survivors = int(enc_out.survivor_embeddings.size(1))
            splice_end_global = max_pre + max_survivors

            per_sample_survivors = enc_out.survivor_attention_mask.sum(dim=1).tolist()
            per_sample_content_len = content_mask.sum(dim=1).tolist()

            for b in range(batch_size):
                if sample_idx >= max_samples:
                    break
                mask_b = shift_mask[b]
                valid_positions = mask_b.nonzero(as_tuple=False).squeeze(-1)
                if valid_positions.numel() == 0:
                    sample_idx += 1
                    continue

                first_sup = int(valid_positions[0].item())
                sample_seq_len = int(attn_full[b].sum().item())
                sample_lang = (
                    languages[b] if languages is not None and b < len(languages) else ""
                )
                sample_n_surv = int(per_sample_survivors[b])
                sample_content_len = int(per_sample_content_len[b])

                pos_tensor = valid_positions
                ce_values = per_token_ce[b, pos_tensor].detach().cpu().tolist()
                tgt_positions = pos_tensor + 1
                tgt_ids = token_ids_full[b, tgt_positions].detach().cpu().tolist()

                pos_from_splice_end = (tgt_positions - splice_end_global).detach().cpu().tolist()
                pos_in_content = (tgt_positions - (first_sup + 1)).detach().cpu().tolist()
                token_strs = tokenizer.convert_ids_to_tokens(tgt_ids)

                for p, pos_fs, pos_ic, tid, tstr, ce in zip(
                    tgt_positions.cpu().tolist(),
                    pos_from_splice_end,
                    pos_in_content,
                    tgt_ids,
                    token_strs,
                    ce_values,
                    strict=True,
                ):
                    all_rows.append({
                        "sample_idx": sample_idx,
                        "pos_global": int(p),
                        "pos_from_splice_end": int(pos_fs),
                        "pos_in_content": int(pos_ic),
                        "token_id": int(tid),
                        "token_str": tstr,
                        "loss": float(ce),
                        "language": sample_lang,
                        "seq_len": sample_seq_len,
                        "n_survivors": sample_n_surv,
                        "content_len": sample_content_len,
                    })

                total_loss_weighted += sum(ce_values)
                total_supervised += len(ce_values)
                sample_idx += 1

            if batch_idx % 5 == 0:
                logger.info(
                    "analyze_progress",
                    samples=sample_idx,
                    rows=len(all_rows),
                    running_loss=total_loss_weighted / max(total_supervised, 1),
                    trainer_scalar_loss=float(out.loss.item()),
                )

    if not all_rows:
        raise RuntimeError("No supervised tokens captured — empty eval?")

    df = pd.DataFrame(all_rows)
    df.to_parquet(parquet_path, index=False)
    logger.info("wrote_parquet", path=str(parquet_path), rows=len(df))

    mean_loss = float(df["loss"].mean())
    percentiles = {
        f"p{p}": float(df["loss"].quantile(p / 100))
        for p in (10, 50, 75, 90, 95, 99)
    }
    by_pos_bucket = (
        df.assign(
            bucket=pd.cut(
                df["pos_from_splice_end"],
                bins=[-1e9, 0, 4, 16, 64, 256, 1e9],
                labels=["pre_or_splice", "0_4", "4_16", "16_64", "64_256", "256+"],
            ),
        )
        .groupby("bucket", observed=True)["loss"]
        .agg(["count", "mean"])
        .to_dict()
    )
    by_lang = (
        df.groupby("language")["loss"]
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
        .head(30)
        .to_dict()
    )

    summary = {
        "checkpoint": str(cfg.analyze.checkpoint),
        "samples_analyzed": sample_idx,
        "supervised_tokens": int(total_supervised),
        "mean_loss": mean_loss,
        "percentiles": percentiles,
        "by_pos_from_splice_end": by_pos_bucket,
        "by_language_top30": by_lang,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote_summary", path=str(summary_path), mean_loss=mean_loss)

    top_hard = (
        df.groupby(["token_id", "token_str"])["loss"]
        .agg(["count", "mean"])
        .query("count >= 5")
        .sort_values("mean", ascending=False)
        .head(100)
        .reset_index()
    )
    top_hard.to_csv(hard_path, index=False)
    logger.info("wrote_top_hard", path=str(hard_path), rows=len(top_hard))

    print("\n=== Analysis summary ===")
    print(f"Checkpoint: {cfg.analyze.checkpoint}")
    print(f"Samples analyzed: {sample_idx}")
    print(f"Supervised tokens: {total_supervised}")
    print(f"Mean per-token loss: {mean_loss:.4f}")
    print(f"Percentiles: {percentiles}")
    print("\nOutputs:")
    print(f"  {parquet_path}")
    print(f"  {summary_path}")
    print(f"  {hard_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    patch_triton_allocator()
    patch_triton_autotuner()
    patch_gated_delta_rule_numerics()
    setup_logging()
    set_seed(cfg.seed)

    if not cfg.get("analyze", None) or not cfg.analyze.get("checkpoint", None):
        raise ValueError(
            "Pass +analyze.checkpoint=<path> +analyze.output_dir=<path>",
        )
    if not cfg.analyze.get("output_dir", None):
        raise ValueError("Pass +analyze.output_dir=<path>")

    phase = cfg.training.get("phase", None)
    if phase != "phase1_step3":
        raise ValueError(
            f"analyze_step3_loss expects phase=phase1_step3, got {phase}. "
            "Use +experiment=phase1_step3.",
        )

    print(OmegaConf.to_yaml(cfg))

    from bgkit.training.phase1.decoder_init import DecoderInitTrainer

    trainer = DecoderInitTrainer(cfg)
    # setup() is normally called inside train(); we call it manually for
    # eval-only use. Builds encoder (Step 2 weights via bgkit_checkpoint=auto),
    # decoder (Qwen3.5 base + LoRA), dataset, optimizer.
    trainer.setup()

    checkpoint_path = Path(str(cfg.analyze.checkpoint))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    logger.info("loading_step3_checkpoint_override", path=str(checkpoint_path))
    trainer.load_checkpoint(checkpoint_path)

    _run_analysis(trainer, cfg)


if __name__ == "__main__":
    main()
