"""Phase transition quality gate checks.

Before proceeding from Phase 1 to Phase 2, verify:
- Decoder reconstruction loss at target compression ratios
- Reconstructed code parses successfully
- Frozen target LLM reproduces text from projected survivors (3a)
- Frozen target LLM generates coherent descriptions (3b)
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
    metrics: dict[str, float]
    message: str


def check_phase1_gates(
    reconstruction_loss: float,
    parse_success_rate: float,
    text_repro_loss: float,
    description_quality: float,
    max_reconstruction_loss: float = 2.0,
    min_parse_rate: float = 0.8,
    max_repro_loss: float = 3.0,
    min_description_quality: float = 0.5,
) -> list[QualityGateResult]:
    """Check all Phase 1 quality gates.

    Returns:
        List of gate results. All must pass to proceed to Phase 2.
    """
    gates = [
        QualityGateResult(
            gate_name="reconstruction_loss",
            passed=reconstruction_loss <= max_reconstruction_loss,
            metrics={"reconstruction_loss": reconstruction_loss},
            message=(
                f"Reconstruction loss {reconstruction_loss:.3f}"
                f" vs threshold {max_reconstruction_loss}"
            ),
        ),
        QualityGateResult(
            gate_name="parse_success_rate",
            passed=parse_success_rate >= min_parse_rate,
            metrics={"parse_success_rate": parse_success_rate},
            message=f"Parse rate {parse_success_rate:.1%} vs threshold {min_parse_rate:.1%}",
        ),
        QualityGateResult(
            gate_name="text_reproduction",
            passed=text_repro_loss <= max_repro_loss,
            metrics={"text_repro_loss": text_repro_loss},
            message=f"Text repro loss {text_repro_loss:.3f} vs threshold {max_repro_loss}",
        ),
        QualityGateResult(
            gate_name="description_quality",
            passed=description_quality >= min_description_quality,
            metrics={"description_quality": description_quality},
            message=(
                f"Description quality {description_quality:.3f}"
                f" vs threshold {min_description_quality}"
            ),
        ),
    ]
    for gate in gates:
        status = "PASSED" if gate.passed else "FAILED"
        logger.info(f"Quality gate {gate.gate_name}: {status} - {gate.message}")
    return gates
