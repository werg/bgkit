"""The pin rewrite must refuse rather than corrupt.

On 2026-08-31 this logic, then a heredoc inside restart-train.sh, destroyed
docker-compose.yaml. The guard was ``if old not in s: refuse`` -- which does
not catch an EMPTY pin, because ``"" in s`` is always true. The fall-through
was ``s.replace("", new)``, inserting the checkpoint name between every
character: 56226 "occurrences" across 71 services, recovered only because the
file was committed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "repin_compose",
    Path(__file__).resolve().parents[3] / "scripts" / "repin_compose.py",
)
repin_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(repin_mod)
repin = repin_mod.repin
Refused = repin_mod.RepinRefusedError

COMPOSE = "services:\n  a:\n    command: [--resume=ckpt_step10]\n  b:\n    command: [x]\n"


def test_an_empty_old_pin_is_refused_not_expanded():
    """THE bug. Every character position matches an empty pattern."""
    with pytest.raises(Refused, match="current pin is empty"):
        repin(COMPOSE, "", "ckpt_step20")
    with pytest.raises(Refused, match="current pin is empty"):
        repin(COMPOSE, "   ", "ckpt_step20")


def test_the_corruption_shape_is_what_would_have_happened():
    """Pins the failure mode itself, so nobody reintroduces it thinking an
    empty pattern is harmless."""
    assert COMPOSE.count("") == len(COMPOSE) + 1
    assert len(COMPOSE.replace("", "X")) > len(COMPOSE) * 2


def test_an_empty_new_pin_is_refused():
    with pytest.raises(Refused, match="new pin is empty"):
        repin(COMPOSE, "ckpt_step10", "")


def test_a_pin_not_present_is_refused_rather_than_silently_doing_nothing():
    """Silently doing nothing would start the run on a stale pin, which is
    the exact class of loss this script exists to prevent."""
    with pytest.raises(Refused, match="not found verbatim"):
        repin(COMPOSE, "ckpt_step99", "ckpt_step20")


def test_an_unchanged_pin_is_a_no_op_not_an_error():
    text, n = repin(COMPOSE, "ckpt_step10", "ckpt_step10")
    assert n == 0
    assert text == COMPOSE


def test_a_real_swap_replaces_only_the_pin():
    text, n = repin(COMPOSE, "ckpt_step10", "ckpt_step20")
    assert n == 1
    assert "ckpt_step20" in text
    assert "ckpt_step10" not in text
    # Nothing else moved.
    assert text.count("\n") == COMPOSE.count("\n")
    assert "services:" in text and "  b:" in text


def test_every_occurrence_of_the_same_pin_is_swapped():
    doubled = COMPOSE + "  c:\n    command: [--resume=ckpt_step10]\n"
    text, n = repin(doubled, "ckpt_step10", "ckpt_step20")
    assert n == 2
    assert "ckpt_step10" not in text
