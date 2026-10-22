"""Domain transfer analysis: Phase 1 -> Phase 2 benefit measurement.

Compares Phase 2 Step 1 training from:
  (a) Phase 1 Step 5 checkpoint (code compression pre-training)
  (b) Raw Qwen3.5-0.8B (no Phase 1)

Measures whether code compression pre-training helps knowledge retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DomainTransferResult:
    """Comparison between Phase-1-initialized and raw-initialized models."""

    phase1_init_metrics: dict[str, float]
    raw_init_metrics: dict[str, float]
    improvement: dict[str, float]


def compute_domain_transfer(
    phase1_metrics: dict[str, float],
    raw_metrics: dict[str, float],
) -> DomainTransferResult:
    """Compute domain transfer benefit.

    Args:
        phase1_metrics: Eval metrics from model initialized with Phase 1 checkpoint.
        raw_metrics: Eval metrics from model initialized with raw Qwen3.5-0.8B.

    Returns:
        DomainTransferResult with per-metric improvement.
    """
    improvement = {}
    all_keys = set(phase1_metrics) | set(raw_metrics)
    for key in all_keys:
        p1_val = phase1_metrics.get(key, 0.0)
        raw_val = raw_metrics.get(key, 0.0)
        if raw_val != 0:
            improvement[key] = (p1_val - raw_val) / abs(raw_val)
        else:
            improvement[key] = p1_val - raw_val

    return DomainTransferResult(
        phase1_init_metrics=dict(phase1_metrics),
        raw_init_metrics=dict(raw_metrics),
        improvement=improvement,
    )


def compute_compression_pareto(
    eval_fn,
    ratios: list[float] | None = None,
) -> list[dict[str, float]]:
    """Evaluate quality at multiple retention ratios for Pareto frontier.

    Uses pre-computed L0 cache's sub-selection capability.

    Args:
        eval_fn: Function(retention_ratio) -> dict of benchmark metrics.
        ratios: Retention ratios to sweep.

    Returns:
        List of {ratio, metric1, metric2, ...} dicts.
    """
    if ratios is None:
        ratios = [0.50, 0.10, 0.05, 0.02, 0.01]

    points = []
    for ratio in ratios:
        metrics = eval_fn(ratio)
        points.append({"retention_ratio": ratio, **metrics})
    return points
