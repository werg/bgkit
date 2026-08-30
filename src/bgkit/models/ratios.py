"""Explicit spellings for retention-ratio intent.

THE COLLISION THIS RETIRES. ``target_ratio=None`` meant three different things
in three places:

    level_compressor   "compression disabled, keep everything"
    encoder.forward    "SKIP this level entirely (and its bridge)"
    _run_l1_batch      "UNSET — resolve from config, or sample one"

The second and third are near-opposites, and the third shadowed the second:
because ``_run_l1_batch`` treated ``None`` as "resolve", passing ``None`` there
could never mean "skip", so the L0-only configuration was UNREACHABLE from the
Phase-2 trainer for its entire existence — while every positive rep measurement
on the base was taken in exactly that configuration, via ``encoder.forward``
where ``None`` does mean skip.

Nothing detected this. Both call sites were individually correct, well-tested
and well-commented; they simply disagreed about what a shared sentinel meant.
Distinct meanings need distinct spellings.
"""

from __future__ import annotations

from typing import Final


class _Skip:
    """Sentinel: skip this compression level entirely."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "SKIP_LEVEL"

    def __bool__(self) -> bool:
        # Guards against `if ratio:` accidentally treating SKIP as falsy-absent.
        raise TypeError(
            "SKIP_LEVEL has no truth value — compare with `is SKIP_LEVEL`"
        )


SKIP_LEVEL: Final = _Skip()


def is_skip(value: object) -> bool:
    """True when the caller explicitly asked to skip the level."""
    return value is SKIP_LEVEL


def resolve_ratio(value: object, default: float | None) -> float | None:
    """Resolve a ratio argument to a float, ``None`` (no compression), or raise.

    ``SKIP_LEVEL`` must be handled by the CALLER before reaching here — it is a
    control-flow decision (run the level or not), not a ratio. Raising rather
    than coercing keeps the two meanings from re-merging.
    """
    if is_skip(value):
        raise ValueError(
            "SKIP_LEVEL is a control-flow signal, not a ratio; branch on "
            "is_skip(...) before resolving"
        )
    if value is None:
        return default
    return float(value)
