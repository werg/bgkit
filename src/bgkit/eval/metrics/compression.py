"""Compression ratio vs quality curves."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class CompressionQualityPoint:
    """A single point on the compression-quality curve."""

    compression_ratio: float  # e.g., 0.1 = 10% survivors
    reconstruction_loss: float
    parse_success_rate: float
    description_quality: float


def compute_compression_curve(
    eval_fn: Callable[[float], dict[str, float]],
    ratios: list[float] | None = None,
) -> list[CompressionQualityPoint]:
    """Evaluate quality across a range of compression ratios.

    The ``eval_fn`` is provided by the caller and handles setting the survivor
    ratio, running eval, and returning a metrics dict with keys
    ``reconstruction_loss``, ``parse_success_rate``, ``description_quality``.

    Args:
        eval_fn: Function(ratio) -> dict of metrics.
        ratios: Compression ratios to evaluate. Defaults to
            [0.05, 0.1, 0.15, 0.2, 0.3, 0.5].

    Returns:
        List of quality points, one per ratio.
    """
    if ratios is None:
        ratios = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

    points = []
    for ratio in ratios:
        metrics = eval_fn(ratio)
        points.append(
            CompressionQualityPoint(
                compression_ratio=ratio,
                reconstruction_loss=metrics.get("reconstruction_loss", float("nan")),
                parse_success_rate=metrics.get("parse_success_rate", float("nan")),
                description_quality=metrics.get("description_quality", float("nan")),
            )
        )
    return points
