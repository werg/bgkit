#!/usr/bin/env python
"""DECISIVE test: is the encode->splice->decode machinery intact / reps load-bearing?

Reuses SummarizationRoundRobinTrainer.setup() so we test the EXACT current-code
path (splice via forward_with_single_splice, raw inputs_embeds operating point,
per-family projection routing) with the checkpoint the trainer saved.

The summarization checkpoint's decoder was trained on the SUMMARY task (target =
abstract/summary given compressed source reps). So the fair "known-good" test is:
  - reps present  -> summary CE should be LOW and MUCH better than zeroed
  - zeroed reps   -> lower bound (LM prior + prompt only)
  - noise reps    -> control (matched per-vector norm)
  - text oracle   -> full uncompressed source tokens in the slot (qwen only,
                     source is qwen-tokenized) == upper bound.
rep_gain = ce_zeroed - ce_reps is the LOAD-BEARING measure.

Swap the checkpoint via `step1_checkpoint=<dir>` on the CLI:
  - matched 51945 dir  -> enc51945 + dec51945 (known-good, machinery ground truth)
  - hybrid dir         -> enc9164  + dec51945 (isolates the git-repro encoder)

Run (trainer stopped, GPU free):
  docker compose --env-file .env -f docker/docker-compose.yaml run --rm --no-deps \
    train-phase2-kb-git-repro-fullbackprop \
    python /workspace/scripts/diag_verbatim_decisive.py \
    +experiment=phase1_summarization_round_robin \
    step1_checkpoint=/workspace/checkpoints/<dir> \
    training.max_total_source_tokens=3072
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import structlog
import torch
from omegaconf import DictConfig

from bgkit.training.phase1.summarization_round_robin import (
    SummarizationRoundRobinTrainer,
)
from bgkit.utils.logging import setup_logging

logger = structlog.get_logger()

# (l0_ratio, l1_ratio_or_None).  0.999 ~ near-lossless (all L0 survive).
# 0.316 x 0.316 ~ 0.10 end-to-end == the summarization curriculum END operating
# point the step-51945 checkpoint was trained at (most in-distribution).
RATIOS = (
    (0.999, None),
    (0.5, None),
    (0.316, None),
    (0.1, None),
    (0.5, 0.5),
    (0.316, 0.316),
)

N_SAMPLES = int(os.environ.get("DIAG_N_SAMPLES", "6"))


def build_loss_mask(prefix_ids, per_group, suffix_masks, device):
    full = []
    for i, (pre, sm) in enumerate(zip(prefix_ids, suffix_masks, strict=True)):
        n_pre = int(pre.shape[0])
        n_surv = int(per_group[i])
        full.append(
            torch.cat(
                [
                    torch.zeros(n_pre, dtype=torch.bool, device=device),
                    torch.zeros(n_surv, dtype=torch.bool, device=device),
                    sm.to(device),
                ]
            )
        )
    return torch.cat(full)


def ce_with_slot(trainer, decoder, survivors, group_cu, prefix_ids, suffix_ids, loss_mask):
    out = decoder.forward_with_single_splice(
        survivor_embeddings=survivors,
        survivor_cu_seqlens=group_cu,
        prefix_ids=prefix_ids,
        suffix_ids=suffix_ids,
        loss_mask=loss_mask,
    )
    loss = out.loss if hasattr(out, "loss") else out
    return float(loss.item())


@torch.no_grad()
def run_family(trainer, family, samples, report):
    trainer.encoder.set_active_decoder_family(family)
    decoder = trainer.decoder_qwen if family == "qwen35" else trainer.decoder_falcon
    dev = trainer.device
    embed_dec = decoder._get_inner_model_and_head()[0].get_input_embeddings()
    embed_norm_mean = float(embed_dec.weight.detach().float().norm(dim=-1).mean())

    for si, batch in enumerate(samples):
        prefix_ids, suffix_ids, suffix_masks, comp = trainer._build_chat_inputs(family, batch)
        row = {
            "family": family,
            "sample": si,
            "dataset": batch["dataset_names"][0],
            "group_id": str(batch["group_ids"][0])[:60],
            "summary_tokens": int(sum(int(m.sum()) for m in suffix_masks)),
            "embed_norm_mean": round(embed_norm_mean, 3),
            "ratios": {},
        }

        # ---- text oracle (qwen only): full uncompressed source in the slot ----
        if family == "qwen35":
            src = torch.cat(
                [torch.as_tensor(d, dtype=torch.long) for d in batch["source_docs"][0]]
            ).to(dev)
            oracle_emb = embed_dec(src).to(dtype=torch.bfloat16)
            gc = torch.tensor([0, int(src.shape[0])], dtype=torch.int32, device=dev)
            lm = build_loss_mask(prefix_ids, [int(src.shape[0])], suffix_masks, dev)
            row["text_oracle_ce"] = round(
                ce_with_slot(trainer, decoder, oracle_emb, gc, prefix_ids, suffix_ids, lm),
                4,
            )
            row["source_tokens"] = int(src.shape[0])

        for r0, r1 in RATIOS:
            trainer._target_ratio_start = trainer._target_ratio_end = r0
            trainer.global_step = 0
            if r1 is None:
                trainer._l1_introduction_step = 10**9
            else:
                trainer._l1_introduction_step = 0
                trainer._target_ratio_l1_start = trainer._target_ratio_l1_end = r1
            enc_out, group_cu, per_group, _, _, _ = trainer._encode_batch(batch, comp)
            survivors = enc_out.survivor_embeddings
            K = int(survivors.shape[0])
            lm = build_loss_mask(prefix_ids, per_group, suffix_masks, dev)

            ce_reps = ce_with_slot(
                trainer, decoder, survivors, group_cu, prefix_ids, suffix_ids, lm
            )
            ce_zero = ce_with_slot(
                trainer, decoder, torch.zeros_like(survivors), group_cu,
                prefix_ids, suffix_ids, lm,
            )
            # noise matched to per-vector norm of the real reps
            if K > 0:
                vnorm = survivors.detach().float().norm(dim=-1, keepdim=True)
                noise = torch.randn_like(survivors.float())
                noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-6) * vnorm
                noise = noise.to(survivors.dtype)
                surv_norm_mean = float(vnorm.mean())
            else:
                noise = survivors
                surv_norm_mean = 0.0
            ce_noise = ce_with_slot(
                trainer, decoder, noise, group_cu, prefix_ids, suffix_ids, lm
            )

            # RESCALE PROBE: put reps back onto the decoder's trained operating
            # norm (matched-51945 norm_ratio: ~4.2 qwen, ~1.0 falcon). If gain
            # RECOVERS -> the drift is norm-only (trivial rescale fix). If it
            # stays dead -> the projection direction ALSO drifted off-manifold
            # (needs projection repair / retraining, not a rescale).
            ce_rescaled = None
            if K > 0:
                target_nr = 4.2 if family == "qwen35" else 1.0
                tgt = target_nr * embed_norm_mean
                cur = survivors.detach().float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
                rescaled = (survivors.float() * (tgt / cur)).to(survivors.dtype)
                ce_rescaled = ce_with_slot(
                    trainer, decoder, rescaled, group_cu, prefix_ids, suffix_ids, lm
                )

            row["ratios"][f"l0={r0},l1={r1}"] = {
                "K": K,
                "ce_reps": round(ce_reps, 4),
                "ce_zeroed": round(ce_zero, 4),
                "ce_noise": round(ce_noise, 4),
                "ce_reps_rescaled": round(ce_rescaled, 4) if ce_rescaled is not None else None,
                "rep_gain_vs_zeroed": round(ce_zero - ce_reps, 4),
                "rescale_gain_vs_zeroed": (
                    round(ce_zero - ce_rescaled, 4) if ce_rescaled is not None else None
                ),
                "surv_norm_mean": round(surv_norm_mean, 3),
                "norm_ratio_surv_over_embed": round(surv_norm_mean / max(embed_norm_mean, 1e-6), 3),
            }
            enc_out.release()
        report["rows"].append(row)
        print("\n" + "=" * 100)
        print(json.dumps(row, indent=2, default=str))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    assert cfg.training.phase == "phase1_summarization_round_robin", cfg.training.phase
    trainer = SummarizationRoundRobinTrainer(cfg)
    trainer.setup()
    trainer.encoder.eval()
    trainer.decoder_qwen.eval()
    trainer.decoder_falcon.eval()

    # Grab N single-sample eval batches (B=1 for clean per-sample CE + memory).
    samples = []
    for flat_i in trainer._eval_flat_idx[: N_SAMPLES * 3]:
        b = trainer._collate([int(flat_i)])
        # skip pathologically long arxiv to bound memory/time
        n_src = sum(len(d) for d in b["source_docs"][0])
        if n_src > int(cfg.training.get("max_total_source_tokens", 3072)):
            continue
        samples.append(b)
        if len(samples) >= N_SAMPLES:
            break
    logger.info("samples_collected", n=len(samples),
                datasets=[s["dataset_names"][0] for s in samples])

    families = [
        f.strip()
        for f in os.environ.get("DIAG_FAMILIES", "qwen35,falcon_h1").split(",")
        if f.strip()
    ]
    report = {
        "step1_checkpoint": str(cfg.get("step1_checkpoint")),
        "n_samples": len(samples),
        "rows": [],
    }
    for family in families:
        run_family(trainer, family, samples, report)

    # aggregate mean rep_gain per (family, ratio)
    agg = {}
    for r in report["rows"]:
        for rk, e in r["ratios"].items():
            key = f'{r["family"]}/{rk}'
            a = agg.setdefault(key, {"reps": [], "zero": [], "noise": [], "gain": [],
                                     "rgain": [], "nr": [], "K": []})
            a["reps"].append(e["ce_reps"])
            a["zero"].append(e["ce_zeroed"])
            a["noise"].append(e["ce_noise"])
            a["gain"].append(e["rep_gain_vs_zeroed"])
            if e.get("rescale_gain_vs_zeroed") is not None:
                a["rgain"].append(e["rescale_gain_vs_zeroed"])
            a["nr"].append(e["norm_ratio_surv_over_embed"])
            a["K"].append(e["K"])
    print("\n" + "#" * 100)
    print("AGGREGATE  ckpt=", report["step1_checkpoint"])
    for key, a in sorted(agg.items()):
        n = len(a["gain"])
        m = {
            "mean_ce_reps": round(sum(a["reps"]) / n, 4),
            "mean_ce_zeroed": round(sum(a["zero"]) / n, 4),
            "mean_ce_noise": round(sum(a["noise"]) / n, 4),
            "mean_rep_gain": round(sum(a["gain"]) / n, 4),
            "mean_rescale_gain": (
                round(sum(a["rgain"]) / len(a["rgain"]), 4) if a["rgain"] else None
            ),
            "mean_norm_ratio": round(sum(a["nr"]) / n, 3),
            "mean_K": round(sum(a["K"]) / n, 1),
        }
        report.setdefault("agg", {})[key] = m
        print(f"  [{key:28s}] {json.dumps(m)}")
    # text-oracle summary
    to = [r["text_oracle_ce"] for r in report["rows"] if "text_oracle_ce" in r]
    if to:
        report["text_oracle_qwen_mean_ce"] = round(sum(to) / len(to), 4)
        print(f"\n  qwen text-oracle (full source in slot) mean CE = {report['text_oracle_qwen_mean_ce']}")

    tag = Path(str(cfg.get("step1_checkpoint"))).name[:40]
    out = Path(f"/workspace/checkpoints/diag_verbatim_decisive_{tag}.json")
    try:
        out.write_text(json.dumps(report, indent=2, default=str))
        logger.info("report_written", path=str(out))
    except Exception as exc:  # noqa: BLE001
        logger.warning("report_write_failed", err=str(exc))


if __name__ == "__main__":
    main()
