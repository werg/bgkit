#!/usr/bin/env python
"""KB-scale Phase 2 trajectory evaluation harness.

Loads a :class:`bgkit.training.phase2.kr_kb_trainer.KRKBTrainer` from a
config (via Hydra) plus an optional checkpoint, runs the KB trajectory
metrics defined in :mod:`bgkit.eval.kb_trajectory_eval` over the eval
split, and writes a JSON report with aggregate metrics and optional
per-sample breakdown.

Metrics (see :mod:`bgkit.eval.kb_trajectory_eval` for full semantics):

- ``free_running/*``: autonomous route completion, surfaced-ID validity,
  evidence recall, and answer F1 without future teacher context.
- ``trajectory_step_accuracy`` / ``tool_call_id_accuracy`` /
  ``answer_token_f1``: exact teacher-forced diagnostics.
- ``eval/loss``, ``eval/n_samples``, ``eval/tokens_per_sample``: kept
  for backward compatibility with the existing trainer eval report.

Usage::

    python scripts/eval_phase2_kb.py \\
        training=phase2_kb_stage_a \\
        +eval.checkpoint=checkpoints/phase2_kb_best \\
        +eval.max_samples=256 \\
        +eval.per_sample=true
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import structlog
import torch
from omegaconf import DictConfig

from bgkit.eval.kb_trajectory_eval import (
    evaluate_free_running_sample,
    evaluate_sample,
)
from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
from bgkit.utils.logging import setup_logging

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(
    per_sample: list[dict],
) -> dict[str, float]:
    """Roll up per-sample metric dicts into aggregate KB-trajectory metrics.

    ``trajectory_step_accuracy`` is micro-averaged across tokens (sum
    of correct / sum of total) so longer trajectories get proportional
    weight. ``answer_token_f1`` is macro-averaged across samples (every
    sample gets one vote). Tool-call ID accuracy is micro-averaged by
    turn count (per bgkit call), not by sample — a sample with 5 bgkit
    calls counts 5x in the average, which is the correct semantics for
    ``did the model emit the right tag at every call site``.
    """
    if not per_sample:
        return {
            "kb/trajectory_step_accuracy": 0.0,
            "kb/tool_call_id_accuracy/bgkit": 0.0,
            "kb/tool_call_id_accuracy/overall": 0.0,
            "kb/answer_token_f1": 0.0,
            "kb/n_samples": 0.0,
            "kb/n_samples_with_answer": 0.0,
            "kb/n_bgkit_calls": 0.0,
            "kb/free_running/route_exact": 0.0,
            "kb/free_running/valid_navigation": 0.0,
            "kb/free_running/evidence_recall": 0.0,
            "kb/free_running/answer_token_f1": 0.0,
            "kb/free_running/answer_exact_match": 0.0,
        }

    total_tokens = 0
    correct_tokens = 0
    bgkit_sum = 0.0
    bgkit_n = 0
    f1_sum = 0.0
    f1_n = 0
    free_rows = 0
    free_route_exact = 0.0
    free_valid = 0.0
    free_evidence = 0.0
    free_answer_f1 = 0.0
    free_answer_exact = 0.0

    for row in per_sample:
        correct_tokens += int(row.get("trajectory_correct_tokens", 0))
        total_tokens += int(row.get("trajectory_total_tokens", 0))
        tc = row.get("tool_call_id_accuracy", {}) or {}
        n_bg = int(tc.get("n_bgkit", 0))
        if n_bg:
            bgkit_sum += float(tc.get("bgkit", 0.0)) * n_bg
            bgkit_n += n_bg
        # Macro F1: only count samples that actually had an answer turn
        # (indicated by non-empty gold_answer in the evaluate_sample
        # bundle — if the gold is empty we shouldn't penalize the run).
        if row.get("gold_answer"):
            f1_sum += float(row.get("answer_token_f1", 0.0))
            f1_n += 1
        free = row.get("free_running")
        if isinstance(free, dict):
            free_rows += 1
            free_route_exact += float(free.get("route_exact", 0.0))
            free_valid += float(free.get("valid_navigation", 0.0))
            free_evidence += float(free.get("evidence_recall", 0.0))
            free_answer_f1 += float(free.get("answer_token_f1", 0.0))
            free_answer_exact += float(free.get("answer_exact_match", 0.0))

    step_acc = correct_tokens / total_tokens if total_tokens else 0.0
    bgkit_acc = bgkit_sum / bgkit_n if bgkit_n else 0.0
    f1 = f1_sum / f1_n if f1_n else 0.0

    return {
        "kb/trajectory_step_accuracy": step_acc,
        "kb/tool_call_id_accuracy/bgkit": bgkit_acc,
        "kb/tool_call_id_accuracy/overall": bgkit_acc,
        "kb/answer_token_f1": f1,
        "kb/n_samples": float(len(per_sample)),
        "kb/n_samples_with_answer": float(f1_n),
        "kb/n_bgkit_calls": float(bgkit_n),
        "kb/free_running/route_exact": (
            free_route_exact / free_rows if free_rows else 0.0
        ),
        "kb/free_running/valid_navigation": (
            free_valid / free_rows if free_rows else 0.0
        ),
        "kb/free_running/evidence_recall": (
            free_evidence / free_rows if free_rows else 0.0
        ),
        "kb/free_running/answer_token_f1": (
            free_answer_f1 / free_rows if free_rows else 0.0
        ),
        "kb/free_running/answer_exact_match": (
            free_answer_exact / free_rows if free_rows else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    if cfg.training.phase != "phase2_kb":
        raise ValueError(
            "eval_phase2_kb.py expects a Phase 2 KB-scale training config "
            f"(training.phase=phase2_kb), got {cfg.training.phase!r}"
        )

    trainer = KRKBTrainer(cfg)
    trainer.setup()

    eval_cfg = cfg.get("eval", {}) or {}
    checkpoint_path = eval_cfg.get("checkpoint")
    if checkpoint_path:
        logger.info("loading_checkpoint", path=str(checkpoint_path))
        trainer.load_checkpoint(Path(str(checkpoint_path)))
        logger.info("checkpoint_loaded", step=trainer.global_step)

    max_samples = int(eval_cfg.get("max_samples", 256))
    collect_per_sample = bool(eval_cfg.get("per_sample", False))
    free_running = bool(eval_cfg.get("free_running", True))
    max_tool_calls = int(eval_cfg.get("max_tool_calls", 16))
    max_new_tokens = int(eval_cfg.get("max_new_tokens", 8192))
    ablation = str(eval_cfg.get("ablation", "") or "")
    if ablation:
        # e.g. +eval.ablation=oracle_span — applies to BOTH the trainer metric
        # pass and the per-sample teacher-forced + free-running loop below.
        trainer.set_ablation_mode(ablation)
        logger.info("kb_eval_ablation_mode", mode=ablation)

    # Trainer-side loss metric (backward-compatible baseline). Done
    # before the KB metrics so a failure in the KB path doesn't hide
    # the loss number entirely.
    logger.info("kb_trainer_evaluate")
    trainer_metrics = trainer.evaluate()

    # KB trajectory metrics. We iterate the eval dataloader directly so
    # we can cap samples and optionally record per-sample rows without
    # reimplementing trainer internals.
    trainer.model.eval()
    per_sample_rows: list[dict] = []
    samples_seen = 0
    with torch.no_grad():
        for batch in trainer.eval_dataloader:
            if samples_seen >= max_samples:
                break
            for sample in batch:
                if samples_seen >= max_samples:
                    break
                if getattr(trainer, "_round_robin", False):
                    # Same family policy as the trainer's eval pass (follows
                    # the training mix: Qwen-only runs evaluate Qwen only).
                    trainer._set_active_decoder(trainer._eval_family_for_index(samples_seen))
                # Teacher-forced scoring must use the training-mode decoder
                # forward (Falcon-H1's eval-mode Mixer path is numerically
                # different); generation below keeps eval mode for the cache.
                with trainer._teacher_forced_decoders():
                    result = evaluate_sample(trainer, sample)
                free_result = (
                    evaluate_free_running_sample(
                        trainer,
                        sample,
                        max_tool_calls=max_tool_calls,
                        max_new_tokens=max_new_tokens,
                    )
                    if free_running else None
                )
                row = {
                    "dataset": getattr(sample, "dataset_name", ""),
                    "decoder_family": getattr(trainer, "_decoder_family", ""),
                    **result,
                }
                if free_result is not None:
                    row["free_running"] = free_result
                per_sample_rows.append(row)
                samples_seen += 1
    clear_tree = getattr(trainer, "_clear_eval_shared_tree", None)
    if callable(clear_tree):
        clear_tree()
    trainer.model.train()

    aggregate = _aggregate(per_sample_rows)

    # Per-dataset breakdown
    per_dataset: dict[str, list[dict]] = {}
    for row in per_sample_rows:
        ds = str(row.get("dataset", "unknown"))
        per_dataset.setdefault(ds, []).append(row)
    per_dataset_metrics: dict[str, dict[str, float]] = {
        ds: _aggregate(rows) for ds, rows in per_dataset.items()
    }
    per_family: dict[str, list[dict]] = {}
    for row in per_sample_rows:
        family = str(row.get("decoder_family", "unknown"))
        per_family.setdefault(family, []).append(row)
    per_family_metrics = {
        family: _aggregate(rows) for family, rows in per_family.items()
    }

    # Keep backward-compat keys from trainer.evaluate()
    report: dict = {
        "trainer_metrics": dict(trainer_metrics),
        "kb_metrics": aggregate,
        "per_dataset": per_dataset_metrics,
        "per_decoder_family": per_family_metrics,
        "num_samples_evaluated": len(per_sample_rows),
        "ablation": ablation or None,
    }
    # Ensure the required backward-compat keys show up at the top-level
    # aggregate view too.
    for key in ("eval/loss", "eval/n_samples", "eval/tokens_per_sample"):
        if key in trainer_metrics:
            report.setdefault("eval", {})[key] = trainer_metrics[key]

    if collect_per_sample:
        # Drop decoded text from the JSON report when it's long to keep
        # file sizes manageable; keep the first 200 chars for spot checks.
        trimmed: list[dict] = []
        for row in per_sample_rows:
            r = dict(row)
            for fld in ("pred_answer", "gold_answer"):
                v = r.get(fld, "")
                if isinstance(v, str) and len(v) > 200:
                    r[fld] = v[:200] + "..."
            free = r.get("free_running")
            if isinstance(free, dict):
                free = dict(free)
                for fld in ("pred_answer", "gold_answer"):
                    value = free.get(fld, "")
                    if isinstance(value, str) and len(value) > 200:
                        free[fld] = value[:200] + "..."
                r["free_running"] = free
            trimmed.append(r)
        report["per_sample"] = trimmed

    output_dir = Path(eval_cfg.get("output_dir", "eval_reports"))
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = str(cfg.training.get("stage", "?")).upper()
    suffix = f"_{ablation}" if ablation else ""
    report_path = output_dir / f"eval_phase2_kb_stage_{stage}{suffix}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info(
        "kb_eval_report_written",
        path=str(report_path),
        n_samples=len(per_sample_rows),
        **{k: v for k, v in aggregate.items() if not k.endswith("_calls")},
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
