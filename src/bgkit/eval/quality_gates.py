"""Phase transition quality gate checks.

Before proceeding from Phase 1 to Phase 2, verify:
- Decoder reconstruction loss at target compression ratios
- Reconstructed code parses successfully
- Frozen decoder (Qwen3.5-0.8B) reproduces text from projected survivors (3a)
- Frozen decoder (Qwen3.5-0.8B) generates coherent descriptions (3b)
- Quality across compression ratio range
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

logger = structlog.get_logger()


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""

    gate_name: str
    passed: bool
    skipped: bool
    metrics: dict[str, float]
    message: str


def check_phase1_gates(
    step: str = "all",
    reconstruction_loss: float | None = None,
    parse_success_rate: float | None = None,
    text_repro_loss: float | None = None,
    description_quality: float | None = None,
    max_reconstruction_loss: float = 2.0,
    min_parse_success_rate: float = 0.8,
    max_text_repro_loss: float = 3.0,
    min_description_quality: float = 0.5,
) -> list[QualityGateResult]:
    """Check Phase 1 quality gates, step-aware.

    Gates only check metrics available at the given step:
    - ``"1"`` / ``"2"``: Check reconstruction_loss and parse_success_rate only.
    - ``"3a"``: Add text_repro_loss.
    - ``"3b"`` / ``"all"``: Check all four gates.

    For steps ``"1"`` and ``"2"``, metrics not active at that step are skipped.
    For steps ``"3a"``, ``"3b"``, and ``"all"``, active metrics that are missing
    (``None``) will **fail** — callers must provide all metrics required by the
    step or use an earlier step.

    Args:
        step: Training step ("1", "2", "3a", "3b", "all").
        reconstruction_loss: Decoder reconstruction loss.
        parse_success_rate: Fraction of generated code that parses.
        text_repro_loss: Decoder text reproduction loss.
        description_quality: Description generation quality score.
        max_reconstruction_loss: Threshold for reconstruction loss gate.
        min_parse_success_rate: Threshold for parse success rate gate.
        max_text_repro_loss: Threshold for text reproduction loss gate.
        min_description_quality: Threshold for description quality gate.

    Returns:
        List of gate results. All non-skipped gates must pass to proceed.
    """
    # Define which gates are active at each step
    active_gates: dict[str, set[str]] = {
        "1": {"reconstruction_loss", "parse_success_rate"},
        "2": {"reconstruction_loss", "parse_success_rate"},
        "3a": {"reconstruction_loss", "parse_success_rate", "text_reproduction"},
        "3b": {
            "reconstruction_loss", "parse_success_rate",
            "text_reproduction", "description_quality",
        },
        "all": {
            "reconstruction_loss", "parse_success_rate",
            "text_reproduction", "description_quality",
        },
    }

    if step not in active_gates:
        raise ValueError(f"Unknown step: {step!r}. Expected one of {list(active_gates.keys())}")

    active = active_gates[step]

    gate_defs = [
        (
            "reconstruction_loss",
            reconstruction_loss is not None and reconstruction_loss <= max_reconstruction_loss,
            reconstruction_loss is not None,
            {"reconstruction_loss": reconstruction_loss} if reconstruction_loss is not None else {},
            (
                f"Reconstruction loss {reconstruction_loss:.3f}"
                f" vs threshold {max_reconstruction_loss}"
            ) if reconstruction_loss is not None else "No reconstruction_loss provided",
        ),
        (
            "parse_success_rate",
            parse_success_rate is not None and parse_success_rate >= min_parse_success_rate,
            parse_success_rate is not None,
            (
                {"parse_success_rate": parse_success_rate}
                if parse_success_rate is not None else {}
            ),
            (
                f"Parse rate {parse_success_rate:.1%}"
                f" vs threshold {min_parse_success_rate:.1%}"
            ) if parse_success_rate is not None else "No parse_success_rate provided",
        ),
        (
            "text_reproduction",
            text_repro_loss is not None and text_repro_loss <= max_text_repro_loss,
            text_repro_loss is not None,
            {"text_repro_loss": text_repro_loss} if text_repro_loss is not None else {},
            (
                f"Text repro loss {text_repro_loss:.3f} vs threshold {max_text_repro_loss}"
            ) if text_repro_loss is not None else "No text_repro_loss provided",
        ),
        (
            "description_quality",
            (
                description_quality is not None
                and description_quality >= min_description_quality
            ),
            description_quality is not None,
            (
                {"description_quality": description_quality}
                if description_quality is not None else {}
            ),
            (
                f"Description quality {description_quality:.3f}"
                f" vs threshold {min_description_quality}"
            ) if description_quality is not None else "No description_quality provided",
        ),
    ]

    gates = []
    for name, passed, has_value, metrics, message in gate_defs:
        is_active = name in active
        # Active gates with missing values: fail (not skip) for "all" and "3b",
        # skip for earlier steps where the metric genuinely doesn't exist yet.
        if is_active and not has_value and step in ("all", "3a", "3b"):
            gates.append(QualityGateResult(
                gate_name=name,
                passed=False,
                skipped=False,
                metrics={},
                message=f"MISSING — required at step {step!r} but not provided",
            ))
        else:
            skipped = not is_active or not has_value
            gates.append(QualityGateResult(
                gate_name=name,
                passed=passed if not skipped else True,
                skipped=skipped,
                metrics=metrics,
                message=message,
            ))

    for gate in gates:
        if gate.skipped:
            logger.info(f"Quality gate {gate.gate_name}: SKIPPED")
        else:
            status = "PASSED" if gate.passed else "FAILED"
            logger.info(f"Quality gate {gate.gate_name}: {status} - {gate.message}")

    return gates
