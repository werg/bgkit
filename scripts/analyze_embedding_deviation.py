#!/usr/bin/env python
"""Embedding-space deviation diagnostic.

Answers: "how far does the encoder's projected output at no-compression
deviate from the decoder's token-embedding manifold?"

Motivated by the suspicion that Phase 1 Step 1/2 never explicitly taught
the projection block to reproduce *decoder* embeddings, so the decoder
may fail to recognise what the encoder produces. The only training
signal linking proj → decoder-embed space is end-to-end CE through the
decoder stack (Step 1) or proj-MSE student→teacher (Step 2) — neither
anchors to ``decoder.embed_tokens`` directly.

For each eval sample, we:

1. Run encoder at ``target_ratio=None`` (no compression) -> ``proj``
   shape ``(B, L_content, D)`` aligned 1:1 with content token ids.
2. Compare ``proj[b, i]`` against ``decoder.embed_tokens(content_ids[b, i])``:
   MSE, cosine, norm ratio. Baselines vs encoder.embed_tokens and
   enc_emb↔dec_emb (to quantify base vs instruct vocab drift).
3. Top-K (1/5/50) nearest-neighbour retrieval of ``proj[b, i]`` against
   the full decoder vocab — does the closest vocab embedding match
   the actual token?
4. Reconstruction-loss comparison (decoder forward with splice) under
   three survivor sources:
      A. proj at ratio=None (length = L_content)
      B. decoder.embed_tokens(content_ids) directly (identity upper bound)
      C. normal compressed survivors at ratio=0.1 (training regime)
   Per-position CE, bucketed by ``pos_in_content``.

Usage (compose service ``analyze-phase1-step4-embedding-deviation``):

    python scripts/analyze_embedding_deviation.py \\
        +experiment=phase1_step4 \\
        +analyze.checkpoint=/workspace/checkpoints/phase1_step4_step2000_20260417_083031 \\
        +analyze.output_dir=/workspace/data/diagnostics/embdev_step4_step2000 \\
        +analyze.max_samples=200 \\
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


# ------------------------------------------------------------------
# Encoder helpers
# ------------------------------------------------------------------


def _run_encoder(trainer, batch: dict, target_ratio: float | None):
    """Run the encoder with an explicit ``target_ratio`` override.

    Mirrors ``DecoderInitTrainer._compute_survivors`` but lets the caller
    force ``target_ratio=None`` (no compression) or a fixed ratio for
    comparison, regardless of the trainer's curriculum state.
    """
    device = trainer.device
    content_token_ids = batch["content_token_ids"].to(device)
    content_attention_mask = batch["content_attention_mask"].to(device)
    compression_prompt_ids = batch["compression_prompt_ids"].to(device)
    compression_prompt_mask = batch["compression_prompt_mask"].to(device)
    bgkit_embed = trainer.encoder.compressor.backbone.get_input_embeddings()
    return trainer.encoder(
        input_embeddings=bgkit_embed(content_token_ids),
        attention_mask=content_attention_mask,
        prompt_embeddings=bgkit_embed(compression_prompt_ids),
        prompt_attention_mask=compression_prompt_mask,
        target_ratio=target_ratio,
        level="l0",
        min_per_sample=0,
    )


# ------------------------------------------------------------------
# Geometric metrics vs decoder / encoder embed_tokens
# ------------------------------------------------------------------


def _pairwise_geom_metrics(
    query: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Per-(b,i) MSE, cosine, norm ratio between ``query`` and ``target``.

    ``query``/``target`` shape ``(B, L, D)``; ``mask`` ``(B, L)`` bool.
    Returns the per-position means over valid positions.
    """
    q = query.float()
    t = target.float()

    mse = (q - t).pow(2).mean(dim=-1)  # (B, L)
    cos = F.cosine_similarity(q, t, dim=-1)  # (B, L)
    q_norm = q.norm(dim=-1)
    t_norm = t.norm(dim=-1).clamp(min=1e-6)
    norm_ratio = q_norm / t_norm  # (B, L)

    m = mask.to(dtype=torch.bool)
    n = m.sum().clamp(min=1)
    return {
        "mse": float((mse * m).sum().item() / n.item()),
        "cos": float((cos * m).sum().item() / n.item()),
        "norm_ratio": float((norm_ratio * m).sum().item() / n.item()),
        "q_norm": float((q_norm * m).sum().item() / n.item()),
        "t_norm": float((t_norm * m).sum().item() / n.item()),
    }


def _topk_nn_accuracy(
    query: torch.Tensor,
    vocab: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 50),
    chunk_size: int = 128,
) -> dict[str, float]:
    """Top-K NN retrieval of ``query`` against ``vocab``, matched to ``target_ids``.

    Args:
        query: (B, L, D) projected vectors.
        vocab: (V, D) embedding table (decoder).
        target_ids: (B, L) long — the "correct" vocab id at each position.
        mask: (B, L) bool.
    """
    B, L, D = query.shape
    flat_q = query.reshape(B * L, D).float()
    flat_t = target_ids.reshape(B * L).long()
    flat_m = mask.reshape(B * L).to(dtype=torch.bool)

    # Restrict to valid positions to save memory
    valid_idx = flat_m.nonzero(as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        return {f"top{k}": 0.0 for k in ks} | {"n": 0}
    q_valid = flat_q[valid_idx]
    t_valid = flat_t[valid_idx]

    q_norm = F.normalize(q_valid, dim=-1)
    v_norm = F.normalize(vocab.float(), dim=-1)  # (V, D)

    max_k = max(ks)
    hits = {k: 0 for k in ks}
    for start in range(0, q_norm.size(0), chunk_size):
        end = min(start + chunk_size, q_norm.size(0))
        sims = q_norm[start:end] @ v_norm.T  # (chunk, V)
        topk_ids = sims.topk(max_k, dim=-1).indices  # (chunk, max_k)
        tgt = t_valid[start:end].unsqueeze(-1)  # (chunk, 1)
        match = (topk_ids == tgt)  # (chunk, max_k)
        for k in ks:
            hits[k] += int(match[:, :k].any(dim=-1).sum().item())

    n = int(valid_idx.numel())
    return {f"top{k}": hits[k] / n for k in ks} | {"n": n}


# ------------------------------------------------------------------
# Decoder reconstruction CE
# ------------------------------------------------------------------


def _per_token_ce(
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


def _decoder_forward_ce(
    trainer, batch: dict, survivors: torch.Tensor, survivor_mask: torch.Tensor,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Run decoder with supplied survivors, return (per_token_ce, shift_mask, max_pre, max_surv)."""
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
    per_token_ce = _per_token_ce(
        out.hidden_states, out.lm_head, out.token_ids, chunk_size=chunk_size,
    )
    shift_mask = (
        out.attention_mask[:, 1:].bool() & out.loss_mask[:, 1:].bool()
    )
    max_pre = int(splice_start.max().item())
    max_surv = int(survivors.size(1))
    return per_token_ce, shift_mask, max_pre, max_surv


# ------------------------------------------------------------------
# Main analysis loop
# ------------------------------------------------------------------


def _decoder_embed_fn(trainer) -> torch.nn.Module:
    """Return the decoder's input-token embedding module (handles LoRA wrap)."""
    inner, _lm_head = trainer.decoder._get_inner_model_and_head()
    return inner.get_input_embeddings()


def _run_analysis(trainer, cfg: DictConfig) -> None:
    encoder = trainer.encoder
    decoder = trainer.decoder
    eval_dl = trainer.eval_dataloader
    device = trainer.device
    max_samples = int(cfg.analyze.get("max_samples", 200))
    chunk_size = int(cfg.analyze.get("ce_chunk_size", 256))

    output_dir = Path(str(cfg.analyze.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    per_pos_ce_path = output_dir / "per_pos_ce.parquet"
    summary_path = output_dir / "summary.json"

    encoder.eval()
    decoder.eval()

    dec_embed_mod = _decoder_embed_fn(trainer)
    dec_vocab = dec_embed_mod.weight.detach()  # (V, D)
    enc_embed_mod = encoder.compressor.backbone.get_input_embeddings()
    enc_vocab = enc_embed_mod.weight.detach()

    # Running accumulators for geometric metrics
    geom_acc: dict[str, dict[str, float]] = {
        "proj_nocomp_vs_dec": {"mse": 0.0, "cos": 0.0, "norm_ratio": 0.0,
                               "q_norm": 0.0, "t_norm": 0.0, "n": 0.0},
        "proj_nocomp_vs_enc": {"mse": 0.0, "cos": 0.0, "norm_ratio": 0.0,
                               "q_norm": 0.0, "t_norm": 0.0, "n": 0.0},
        "enc_emb_vs_dec":     {"mse": 0.0, "cos": 0.0, "norm_ratio": 0.0,
                               "q_norm": 0.0, "t_norm": 0.0, "n": 0.0},
        "proj_comp_vs_dec":   {"mse": 0.0, "cos": 0.0, "norm_ratio": 0.0,
                               "q_norm": 0.0, "t_norm": 0.0, "n": 0.0},
    }
    nn_acc: dict[str, dict[str, float]] = {
        "proj_nocomp_vs_dec": {"top1": 0.0, "top5": 0.0, "top50": 0.0, "n": 0.0},
        "enc_emb_vs_dec":     {"top1": 0.0, "top5": 0.0, "top50": 0.0, "n": 0.0},
        "proj_comp_vs_dec":   {"top1": 0.0, "top5": 0.0, "top50": 0.0, "n": 0.0},
    }

    def _accumulate(acc: dict[str, float], batch_stats: dict[str, float], n_pos: int) -> None:
        # batch_stats values are already per-position means over n_pos positions.
        for k in ("mse", "cos", "norm_ratio", "q_norm", "t_norm"):
            acc[k] += batch_stats[k] * n_pos
        acc["n"] += n_pos

    def _accumulate_nn(acc: dict[str, float], stats: dict[str, float]) -> None:
        n = stats["n"]
        for k in ("top1", "top5", "top50"):
            acc[k] += stats[k] * n
        acc["n"] += n

    rows: list[dict] = []
    sample_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dl):
            if sample_idx >= max_samples:
                break

            content_token_ids = batch["content_token_ids"].to(device)
            content_attention_mask = batch["content_attention_mask"].to(device).bool()
            B, L_content = content_token_ids.shape
            languages = batch.get("languages", None)

            # --- Encoder forward at ratio=None (uncompressed) ---
            enc_nocomp = _run_encoder(trainer, batch, target_ratio=None)
            proj_nocomp = enc_nocomp.survivor_embeddings  # (B, L_content, D)
            proj_mask = enc_nocomp.survivor_attention_mask.bool()  # (B, L_content)

            # Shape check — at ratio=None encoder returns content-sliced proj
            if proj_nocomp.shape[:2] != (B, L_content):
                raise RuntimeError(
                    f"expected uncompressed proj shape ({B},{L_content},D); "
                    f"got {tuple(proj_nocomp.shape)}"
                )

            # --- Encoder forward at ratio=0.1 (compressed, training regime) ---
            enc_comp = _run_encoder(trainer, batch, target_ratio=0.1)
            proj_comp = enc_comp.survivor_embeddings  # (B, K, D)
            proj_comp_mask = enc_comp.survivor_attention_mask.bool()  # (B, K)
            comp_survivor_mask_content = enc_comp.survivor_mask  # (B, L_content) bool

            # --- Decoder + encoder embedding lookups ---
            dec_emb = dec_embed_mod(content_token_ids).detach()
            enc_emb = enc_embed_mod(content_token_ids).detach()

            # --- Geometric metrics (no-compression branch) ---
            g_proj_dec = _pairwise_geom_metrics(proj_nocomp, dec_emb, proj_mask)
            g_proj_enc = _pairwise_geom_metrics(proj_nocomp, enc_emb, proj_mask)
            g_enc_dec = _pairwise_geom_metrics(enc_emb, dec_emb, content_attention_mask)
            n_valid = int(proj_mask.sum().item())
            n_valid_ed = int(content_attention_mask.sum().item())
            _accumulate(geom_acc["proj_nocomp_vs_dec"], g_proj_dec, n_valid)
            _accumulate(geom_acc["proj_nocomp_vs_enc"], g_proj_enc, n_valid)
            _accumulate(geom_acc["enc_emb_vs_dec"], g_enc_dec, n_valid_ed)

            # --- Geometric metrics for compressed proj against the content
            # tokens it corresponds to. Select content ids at survivor positions.
            # comp_survivor_mask_content: (B, L_content) -> pad to match (B, K).
            if comp_survivor_mask_content is not None:
                comp_tgt_ids = []
                for b in range(B):
                    ids_b = content_token_ids[b][comp_survivor_mask_content[b]]
                    comp_tgt_ids.append(ids_b)
                max_k = proj_comp.size(1)
                comp_tgt_padded = torch.zeros(B, max_k, dtype=torch.long, device=device)
                for b, ids_b in enumerate(comp_tgt_ids):
                    comp_tgt_padded[b, : ids_b.size(0)] = ids_b
                comp_dec_emb = dec_embed_mod(comp_tgt_padded).detach()
                g_comp = _pairwise_geom_metrics(proj_comp, comp_dec_emb, proj_comp_mask)
                _accumulate(
                    geom_acc["proj_comp_vs_dec"], g_comp,
                    int(proj_comp_mask.sum().item()),
                )

            # --- Top-K NN retrieval (sample-rate to keep it cheap) ---
            # Use every position in the first few batches, then subsample.
            nn_subsample_pos = int(cfg.analyze.get("nn_subsample_pos", 512))
            if n_valid > 0:
                q_flat = proj_nocomp[proj_mask]           # (N, D)
                t_flat = content_token_ids[proj_mask]     # (N,)
                if q_flat.size(0) > nn_subsample_pos:
                    perm = torch.randperm(q_flat.size(0), device=device)[:nn_subsample_pos]
                    q_flat = q_flat[perm]
                    t_flat = t_flat[perm]
                nn_stats = _topk_nn_accuracy(
                    q_flat.unsqueeze(0),
                    dec_vocab,
                    t_flat.unsqueeze(0),
                    torch.ones_like(t_flat, dtype=torch.bool).unsqueeze(0),
                )
                _accumulate_nn(nn_acc["proj_nocomp_vs_dec"], nn_stats)

                # enc_emb baseline — what the decoder *would* see if we passed
                # encoder embeddings through unmodified.
                q_flat_e = enc_emb[proj_mask]
                t_flat_e = content_token_ids[proj_mask]
                if q_flat_e.size(0) > nn_subsample_pos:
                    perm_e = torch.randperm(q_flat_e.size(0), device=device)[:nn_subsample_pos]
                    q_flat_e = q_flat_e[perm_e]
                    t_flat_e = t_flat_e[perm_e]
                nn_stats_e = _topk_nn_accuracy(
                    q_flat_e.unsqueeze(0),
                    dec_vocab,
                    t_flat_e.unsqueeze(0),
                    torch.ones_like(t_flat_e, dtype=torch.bool).unsqueeze(0),
                )
                _accumulate_nn(nn_acc["enc_emb_vs_dec"], nn_stats_e)

                if proj_comp_mask.any():
                    q_flat_c = proj_comp[proj_comp_mask]
                    t_flat_c = comp_tgt_padded[proj_comp_mask]
                    if q_flat_c.size(0) > nn_subsample_pos:
                        perm_c = torch.randperm(q_flat_c.size(0), device=device)[:nn_subsample_pos]
                        q_flat_c = q_flat_c[perm_c]
                        t_flat_c = t_flat_c[perm_c]
                    nn_stats_c = _topk_nn_accuracy(
                        q_flat_c.unsqueeze(0),
                        dec_vocab,
                        t_flat_c.unsqueeze(0),
                        torch.ones_like(t_flat_c, dtype=torch.bool).unsqueeze(0),
                    )
                    _accumulate_nn(nn_acc["proj_comp_vs_dec"], nn_stats_c)

            # --- Decoder reconstruction CE under 3 survivor sources ---
            # A: proj at ratio=None (L_content survivors)
            ce_A, shift_mask, max_pre, max_surv_A = _decoder_forward_ce(
                trainer, batch, proj_nocomp, proj_mask, chunk_size,
            )
            # B: decoder embedding table (identity upper bound)
            ce_B, _, _, max_surv_B = _decoder_forward_ce(
                trainer, batch, dec_emb, content_attention_mask, chunk_size,
            )
            # C: normal compressed survivors
            ce_C, _, _, max_surv_C = _decoder_forward_ce(
                trainer, batch, proj_comp, proj_comp_mask, chunk_size,
            )

            # per-token rows for position-bucket breakdown
            # all three CE tensors share the same (B, S-1) layout for the
            # supervised content positions; shift_mask gates valid positions.
            for b in range(B):
                if sample_idx >= max_samples:
                    break
                mb = shift_mask[b]
                valid = mb.nonzero(as_tuple=False).squeeze(-1)
                if valid.numel() == 0:
                    sample_idx += 1
                    continue
                first_sup = int(valid[0].item())
                lang = (
                    languages[b] if languages is not None and b < len(languages) else ""
                )
                # A and C can differ in splice-end because their survivor
                # counts differ. We position by `pos_in_content` = target-pos −
                # (first supervised + 1), which is ratio-invariant. Separate
                # shift_masks would be needed for exact splice_end alignment
                # under C, but pos_in_content tracks the content offset
                # regardless of survivor-span length.
                ce_A_b = ce_A[b, valid].cpu().tolist()
                ce_B_b = ce_B[b, valid].cpu().tolist()
                # For C the valid positions may differ because the survivor
                # span length differs. Recompute with shift_mask per pass is
                # overkill; instead reuse A's valid set (they all supervise
                # the same content tokens, just at different absolute seq
                # positions). We'll report A vs B as the primary comparison
                # and put C's overall summary from its own shift_mask below.
                tgt_positions = (valid + 1).cpu().tolist()
                for pos_idx, (ap, bp) in enumerate(zip(tgt_positions, ce_A_b, strict=True)):
                    pass
                pos_in_content = [int(t - (first_sup + 1)) for t in tgt_positions]
                for pic, la, lb in zip(
                    pos_in_content, ce_A_b, ce_B_b, strict=True,
                ):
                    rows.append({
                        "sample_idx": sample_idx,
                        "pos_in_content": pic,
                        "loss_proj_nocomp": float(la),
                        "loss_decoder_embed": float(lb),
                        "language": lang,
                    })
                sample_idx += 1

            # Overall means
            if batch_idx % 2 == 0:
                def _mean_over(t: torch.Tensor, m: torch.Tensor) -> float:
                    m_f = m.float()
                    return float((t * m_f).sum().item() / m_f.sum().clamp(min=1).item())

                # Per-pass shift masks for fair overall comparison.
                mean_A = _mean_over(ce_A, shift_mask)

                # Rebuild shift_mask for B and C using their respective
                # attention & loss masks returned by the forward.
                _, shift_mask_B, _, _ = _decoder_forward_ce(
                    trainer, batch, dec_emb, content_attention_mask, chunk_size,
                )
                mean_B = _mean_over(ce_B, shift_mask_B)

                _, shift_mask_C, _, _ = _decoder_forward_ce(
                    trainer, batch, proj_comp, proj_comp_mask, chunk_size,
                )
                mean_C = _mean_over(ce_C, shift_mask_C)

                logger.info(
                    "embdev_progress",
                    batch=batch_idx,
                    samples=sample_idx,
                    ce_proj_nocomp=mean_A,
                    ce_decoder_embed=mean_B,
                    ce_proj_comp=mean_C,
                    cos_proj_vs_dec=(
                        geom_acc["proj_nocomp_vs_dec"]["cos"]
                        / max(geom_acc["proj_nocomp_vs_dec"]["n"], 1.0)
                    ),
                    cos_enc_vs_dec=(
                        geom_acc["enc_emb_vs_dec"]["cos"]
                        / max(geom_acc["enc_emb_vs_dec"]["n"], 1.0)
                    ),
                )

    if not rows:
        raise RuntimeError("no valid content positions accumulated — empty eval set?")

    df = pd.DataFrame(rows)
    df.to_parquet(per_pos_ce_path, index=False)
    logger.info("wrote_per_pos_ce", path=str(per_pos_ce_path), rows=len(df))

    def _finalize_geom(d: dict[str, float]) -> dict[str, float]:
        n = max(d["n"], 1.0)
        return {k: v / n for k, v in d.items() if k != "n"} | {"n_positions": d["n"]}

    def _finalize_nn(d: dict[str, float]) -> dict[str, float]:
        n = max(d["n"], 1.0)
        return {k: v / n for k, v in d.items() if k != "n"} | {"n_positions": d["n"]}

    geom_final = {name: _finalize_geom(acc) for name, acc in geom_acc.items()}
    nn_final = {name: _finalize_nn(acc) for name, acc in nn_acc.items()}

    # Position-bucket breakdown for the reconstruction comparison.
    bins = [-1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 10**9]
    labels = [
        "0-4", "4-8", "8-16", "16-32", "32-64", "64-128",
        "128-256", "256-512", "512-1024", "1024+",
    ]
    df["bucket"] = pd.cut(df["pos_in_content"], bins=bins, labels=labels)
    by_pic = (
        df.groupby("bucket", observed=True)
        .agg(
            count=("loss_proj_nocomp", "count"),
            mean_proj=("loss_proj_nocomp", "mean"),
            mean_dec_embed=("loss_decoder_embed", "mean"),
        )
        .assign(delta_proj_minus_dec=lambda x: x["mean_proj"] - x["mean_dec_embed"])
        .round(4)
        .to_dict()
    )

    summary = {
        "checkpoint": str(cfg.analyze.checkpoint),
        "samples_analyzed": sample_idx,
        "geometric": geom_final,
        "top_k_nn_accuracy": nn_final,
        "reconstruction_ce_by_pos_in_content": by_pic,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote_summary", path=str(summary_path))

    # Console print — headline numbers.
    print("\n=== Embedding deviation summary ===")
    print(f"Checkpoint: {cfg.analyze.checkpoint}")
    print(f"Samples analyzed: {sample_idx}")
    print()
    print("Geometric (per-position mean over valid content positions):")
    for name, m in geom_final.items():
        print(
            f"  {name:>24s}  "
            f"mse={m['mse']:7.3f}  cos={m['cos']:+.4f}  "
            f"|q|/|t|={m['norm_ratio']:.3f}  "
            f"|q|={m['q_norm']:.3f}  |t|={m['t_norm']:.3f}"
        )
    print()
    print("Top-K NN accuracy (proj → decoder vocab; correct = actual content token):")
    for name, m in nn_final.items():
        print(
            f"  {name:>24s}  top1={m['top1']:.4f}  "
            f"top5={m['top5']:.4f}  top50={m['top50']:.4f}"
        )
    print()
    print("Reconstruction CE by pos_in_content (proj at ratio=None vs decoder-embed identity):")
    print(f"  {'bucket':>10}  {'count':>8}  {'proj':>8}  {'dec_emb':>8}  {'Δ':>8}")
    for label in labels:
        if label not in by_pic.get("count", {}):
            continue
        c = by_pic["count"][label]
        mp = by_pic["mean_proj"][label]
        md = by_pic["mean_dec_embed"][label]
        d = by_pic["delta_proj_minus_dec"][label]
        print(f"  {label:>10}  {c:>8}  {mp:>8.3f}  {md:>8.3f}  {d:>+8.3f}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    apply_diagnostic_patches()
    from bgkit.training.phase1.decoder_init import DecoderInitTrainer

    trainer = prepare_diagnostic_trainer(
        cfg,
        trainer_cls=DecoderInitTrainer,
        expected_phases=("phase1_step3", "phase1_step4"),
    )
    _run_analysis(trainer, cfg)


if __name__ == "__main__":
    main()
