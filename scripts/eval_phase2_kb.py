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
  ``answer_token_f1`` / ``answer_exact_match``: exact teacher-forced
  diagnostics.
- ``ceiling``: with ``+eval.ablation_sweep`` covering the reps arm, a floor
  (``zeroed``) and a ceiling (``full_text``), the report carries
  ``fraction_captured`` per dataset -- how much of the information the model
  could read it actually got from the reps. A reps score alone does not say
  that, which is how a pooled token-F1 of 0.386 passed for a working model
  against a 0.752 ceiling and a 0.386 floor.
- ``eval/loss``, ``eval/n_samples``, ``eval/tokens_per_sample``: kept
  for backward compatibility with the existing trainer eval report.

Usage::

    python scripts/eval_phase2_kb.py \\
        training=phase2_kb_stage_a \\
        +eval.checkpoint=checkpoints/phase2_kb_best \\
        +eval.max_samples=256 \\
        +eval.per_sample=true

    # floor / reps / ceiling in one pass over the same samples:
    python scripts/eval_phase2_kb.py \\
        +experiment=phase2_kb_widenet_v8 \\
        +eval.checkpoint=... \\
        "+eval.ablation_sweep=[none,zeroed,full_text]"
"""

from __future__ import annotations

import json
import os
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


def _collect_eval_samples(trainer: KRKBTrainer, max_samples: int) -> list:
    """Materialise the eval samples once, so every ablation arm sees them."""
    samples: list = []
    for batch in trainer.eval_dataloader:
        for sample in batch:
            if len(samples) >= max_samples:
                return samples
            samples.append(sample)
    return samples


def _score_samples(
    trainer: KRKBTrainer,
    samples: list,
    *,
    free_running_limit: int,
    max_tool_calls: int,
    max_new_tokens: int,
    force_first_call: bool,
) -> list[dict]:
    """Teacher-forced (and optionally free-running) scoring of ``samples``.

    The decoder family per sample follows the TRAINING mix, and is keyed off
    the sample's index in the list rather than a running counter, so an arm
    that scores fewer samples cannot silently shift the family assignment
    relative to another arm.
    """
    trainer.model.eval()
    rows: list[dict] = []
    free_run_done = 0
    with torch.no_grad():
        for index, sample in enumerate(samples):
            if getattr(trainer, "_round_robin", False):
                # Same family policy as the trainer's eval pass (follows
                # the training mix: Qwen-only runs evaluate Qwen only).
                trainer._set_active_decoder(trainer._eval_family_for_index(index))
            # Teacher-forced scoring must use the training-mode decoder
            # forward (Falcon-H1's eval-mode Mixer path is numerically
            # different); generation below keeps eval mode for the cache.
            with trainer._teacher_forced_decoders():
                result = evaluate_sample(trainer, sample)
            free_result = None
            if free_run_done < free_running_limit:
                free_result = evaluate_free_running_sample(
                    trainer,
                    sample,
                    max_tool_calls=max_tool_calls,
                    max_new_tokens=max_new_tokens,
                    force_first_call=force_first_call,
                )
                free_run_done += 1
            row = {
                "dataset": getattr(sample, "dataset_name", ""),
                "decoder_family": getattr(trainer, "_decoder_family", ""),
                # Keep the prompt question with the prediction: the report
                # is then self-contained for spot checks, and external
                # scorers that need it (BABILong's ``compare_answers``
                # excludes labels already named in the question) don't have
                # to re-join against the trajectory parquet.
                "question": getattr(sample, "question", ""),
                **result,
            }
            if free_result is not None:
                row["free_running"] = free_result
            rows.append(row)
    clear_tree = getattr(trainer, "_clear_eval_shared_tree", None)
    if callable(clear_tree):
        clear_tree()
    trainer.model.train()
    return rows


# Metrics the ceiling table is computed for. Both are per-sample averages, so
# a floor/ceiling ratio over them is meaningful; micro-averaged counters
# (tool-call accuracy) are not comparable this way and are left out.
_CEILING_METRICS = ("kb/answer_token_f1", "kb/answer_exact_match")


def _ceiling_table(
    arms: dict[str, list[dict]],
    floor: str = "zeroed",
    ceiling: str = "full_text",
    reps: str = "none",
) -> dict:
    """How much of the readable information do the reps actually deliver?

    ``fraction_captured = (reps - floor) / (ceiling - floor)`` per metric,
    overall and per dataset. The floor arm is the model answering with no
    information in the splice; the ceiling arm is the same model handed the
    raw document. Without both, a rep score is uninterpretable: v8's pooled
    token-F1 of 0.386 reads as a mediocre model until the floor (0.386) and
    ceiling (0.752) put the fraction captured at ~0.

    Returns ``{}`` unless all three arms ran. A non-positive ceiling-minus-
    floor gap yields ``None`` for that metric rather than a ratio -- there is
    no headroom to capture a fraction of, and dividing by it manufactures
    numbers of either sign.
    """
    if not {floor, ceiling, reps} <= set(arms):
        return {}

    def _split(rows: list[dict]) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {"__all__": rows}
        for row in rows:
            out.setdefault(str(row.get("dataset", "unknown")), []).append(row)
        return out

    parts = {name: _split(rows) for name, rows in arms.items()}
    table: dict[str, dict] = {}
    for group in parts[reps]:
        agg = {
            name: _aggregate(parts[name].get(group, [])) for name in (floor, ceiling, reps)
        }
        entry: dict[str, dict] = {"n_samples": len(parts[reps].get(group, []))}
        for metric in _CEILING_METRICS:
            lo = agg[floor].get(metric)
            hi = agg[ceiling].get(metric)
            mid = agg[reps].get(metric)
            if lo is None or hi is None or mid is None:
                continue
            headroom = hi - lo
            entry[metric] = {
                floor: lo,
                reps: mid,
                ceiling: hi,
                "headroom": headroom,
                "fraction_captured": (
                    (mid - lo) / headroom if headroom > 1e-9 else None
                ),
            }
        key = "overall" if group == "__all__" else group
        table[key] = entry
    return table


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
            "kb/answer_exact_match": 0.0,
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
    em_sum = 0.0

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
            em_sum += float(row.get("answer_exact_match", 0.0))
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
    em = em_sum / f1_n if f1_n else 0.0

    return {
        "kb/trajectory_step_accuracy": step_acc,
        "kb/tool_call_id_accuracy/bgkit": bgkit_acc,
        "kb/tool_call_id_accuracy/overall": bgkit_acc,
        "kb/answer_token_f1": f1,
        "kb/answer_exact_match": em,
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
        # WEIGHTS ONLY. trainer.load_checkpoint is the RESUME path: it restores
        # optimizer state and therefore refuses on an optimizer-type mismatch
        # ("saved with 'muon' but current config uses 'adamw'"). That guard is
        # correct for resuming training and wrong for evaluation — an eval
        # needs no optimizer, and refusing here makes whole lineages
        # unevaluable under a config that merely chose a different optimizer.
        # Hit 2026-08-30 evaluating the Muon-trained summarization base under
        # the AdamW wide-net config, which is exactly the cross-lineage
        # comparison an eval script exists to make.
        from bgkit.training.checkpointing import (
            load_checkpoint as _load_ckpt,
        )
        from bgkit.training.checkpointing import (
            normalize_model_state as _norm_state,
        )
        _meta, _state = _load_ckpt(Path(str(checkpoint_path)))
        _state = _norm_state(_state)
        try:
            trainer._restore_model_state(_state)
        except RuntimeError as exc:
            # CROSS-LINEAGE EVAL. A Phase-1 checkpoint predates Phase-2-only
            # modules (e.g. encoder.l1l1_bridge, added for the recursive-L1
            # tree), so a STRICT load fails on keys that simply did not exist
            # when it was saved. The trainer's guard is right to stay strict
            # for RESUMING training; for evaluation, refusing makes every
            # earlier lineage unevaluable under the current config.
            #
            # Non-strict, but NOT silent: every missing/unexpected key is
            # logged at WARNING. A module left at initialisation is only
            # harmless if it is off the eval path — say which ones they are
            # rather than letting an untrained module run unnoticed.
            missing, unexpected = trainer.model.load_state_dict(
                _state["model"], strict=False,
            )
            logger.warning(
                "checkpoint_loaded_non_strict",
                reason=str(exc)[:200],
                missing=list(missing)[:20],
                n_missing=len(missing),
                unexpected=list(unexpected)[:20],
                n_unexpected=len(unexpected),
                hint="listed params are at INIT, not trained — verify they are "
                     "off the eval path before trusting the numbers",
            )
        trainer.global_step = int(getattr(_meta, "step", 0) or 0)
        logger.info(
            "checkpoint_loaded", step=trainer.global_step, weights_only=True,
        )

    max_samples = int(eval_cfg.get("max_samples", 256))
    collect_per_sample = bool(eval_cfg.get("per_sample", False))
    # HOW MANY samples get free-running generation, not whether.
    #
    # Free-running is by far the most expensive thing this script does:
    # autoregressive greedy decode, one tiny GPU kernel per token, so it is
    # CPU-launch-bound and shows up as ~88% GPU "utilization" at ~35W with one
    # core pinned — indistinguishable from a hang until you read a stack.
    #
    # It used to be an all-or-nothing bool read from ``eval.free_running``
    # (default True), while the in-training knob for the same behaviour is
    # ``training.eval_free_running_samples``. Two names for one thing, and the
    # script silently ignored the documented one: on 2026-08-30 three eval runs
    # were launched with ``training.eval_free_running_samples=0`` and every one
    # of them free-ran anyway, costing hours and two wrong hang diagnoses.
    #
    # Resolution order, most specific first, with the result LOGGED because a
    # silently-ignored config key is what caused the incident:
    #   1. eval.free_running_samples: N   (explicit cap for this script)
    #   2. training.eval_free_running_samples: N  (the shared, documented key)
    #   3. eval.free_running: bool        (legacy; True -> all, False -> none)
    step_cfg = getattr(trainer, "step_cfg", {}) or {}
    _explicit = eval_cfg.get("free_running_samples", None)
    _shared = step_cfg.get("eval_free_running_samples", None)
    if _explicit is not None:
        free_running_limit, _src = int(_explicit), "eval.free_running_samples"
    elif _shared is not None:
        free_running_limit = int(_shared)
        _src = "training.eval_free_running_samples"
    else:
        free_running_limit = max_samples if bool(
            eval_cfg.get("free_running", True)
        ) else 0
        _src = "eval.free_running"
    free_running_limit = max(0, min(free_running_limit, max_samples))
    logger.info(
        "eval_free_running_resolved",
        free_running_samples=free_running_limit,
        source=_src,
        max_samples=max_samples,
        note="0 disables generation; teacher-forced scoring is unaffected",
    )
    max_tool_calls = int(eval_cfg.get("max_tool_calls", 16))
    max_new_tokens = int(eval_cfg.get("max_new_tokens", 8192))
    # Seed the gold retrieval turn instead of making the model emit it:
    # for a benchmark arm whose baselines are simply handed their context,
    # charging ours for retrieval too makes the comparison unfair (and an
    # out-of-distribution id format masks the capability entirely).
    force_first_call = bool(eval_cfg.get("force_first_call", False))
    ablation = str(eval_cfg.get("ablation", "") or "")
    # An ablation SWEEP scores the same samples under several modes in one
    # process. Reporting a rep-dependence number needs at least a floor and a
    # ceiling beside it -- the widenet runs spent weeks on a "reps" number
    # with neither, and a pooled 0.386 looked like a bad-but-working model
    # until the full-text arm put its ceiling at 0.752 (2026-08-30). Running
    # the arms here rather than as separate invocations is what guarantees
    # they saw the same documents.
    # "none" is spelled out rather than left empty: an empty element inside a
    # Hydra list literal is fragile to quote and easy to lose silently, and a
    # sweep that quietly dropped its reps arm would report a ceiling table
    # against the wrong baseline.
    sweep = [
        "" if str(m or "").lower() in ("", "none") else str(m)
        for m in (eval_cfg.get("ablation_sweep") or [])
    ]
    modes = sweep or [ablation]
    if ablation:
        # e.g. +eval.ablation=oracle_span — applies to BOTH the trainer metric
        # pass and the per-sample teacher-forced + free-running loop below.
        trainer.set_ablation_mode(ablation)
        logger.info("kb_eval_ablation_mode", mode=ablation)

    # Trainer-side loss metric (backward-compatible baseline). Done
    # before the KB metrics so a failure in the KB path doesn't hide
    # the loss number entirely. Primary arm only: pooled loss is owned by the
    # biggest-target samples and has repeatedly failed to track rep
    # dependence, so paying for it once per arm buys a misleading number.
    logger.info("kb_trainer_evaluate")
    trainer_metrics = trainer.evaluate()

    # KB trajectory metrics, scored over a FIXED sample list. Pulling the
    # samples out of the dataloader once is what makes a sweep a controlled
    # comparison: every arm must see the same documents, or its ceiling is
    # partly a difference in the sample draw.
    eval_samples = _collect_eval_samples(trainer, max_samples)
    primary = modes[0]
    arms: dict[str, list[dict]] = {}
    for arm_index, mode in enumerate(modes):
        trainer.set_ablation_mode(mode or None)
        if len(modes) > 1:
            logger.info(
                "kb_eval_ablation_arm",
                mode=mode or "none", arm=arm_index + 1, of=len(modes),
            )
        arms[mode or "none"] = _score_samples(
            trainer,
            eval_samples,
            # Generation runs on the primary arm only: it costs orders of
            # magnitude more than teacher forcing and the ceiling table is
            # built from the teacher-forced metrics.
            free_running_limit=free_running_limit if arm_index == 0 else 0,
            max_tool_calls=max_tool_calls,
            max_new_tokens=max_new_tokens,
            force_first_call=force_first_call,
        )
    trainer.set_ablation_mode(primary or None)
    per_sample_rows = arms[primary or "none"]

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
    if len(modes) > 1:
        report["ablation_sweep"] = {
            name: {
                "kb_metrics": _aggregate(rows),
                "per_dataset": {
                    ds: _aggregate(
                        [r for r in rows if str(r.get("dataset", "unknown")) == ds]
                    )
                    for ds in {str(r.get("dataset", "unknown")) for r in rows}
                },
            }
            for name, rows in arms.items()
        }
        ceiling = _ceiling_table(arms)
        if ceiling:
            report["ceiling"] = ceiling
            for group, entry in sorted(ceiling.items()):
                for metric, cells in entry.items():
                    if metric == "n_samples":
                        continue
                    logger.info(
                        "kb_eval_ceiling",
                        group=group,
                        metric=metric,
                        n=entry["n_samples"],
                        **{k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in cells.items()},
                    )
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

    # Default the report under CHECKPOINT_DIR, which is bind-mounted, NOT a
    # container-relative path. ``docker compose run --rm`` deletes the
    # container filesystem on exit, so the previous default ("eval_reports")
    # wrote the report into a directory that ceased to exist moments later —
    # on 2026-08-30 a completed 128-sample eval's report was destroyed this
    # way and survived only because the container log happened to be teed to
    # a file. A result you cannot read is a result you did not produce.
    _default_out = os.environ.get("CHECKPOINT_DIR", "")
    output_dir = Path(
        eval_cfg.get("output_dir", None)
        or (Path(_default_out) / "eval_reports" if _default_out else "eval_reports")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = str(cfg.training.get("stage", "?")).upper()
    suffix = "_sweep" if len(modes) > 1 else (f"_{ablation}" if ablation else "")
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
