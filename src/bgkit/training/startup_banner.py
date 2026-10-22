"""Loud, visible startup banner so the operator can verify checkpoint sources.

Wires up at the end of every trainer's ``setup()``. The 'silent pristine encoder'
class of bug — where a config key was set but the trainer didn't read it, and
training quietly ran on random HF weights — is invisible in normal log noise.
This banner shows every checkpoint actually loaded (and crucially, every
component that was NOT loaded), bracketed by a hard-to-miss header and footer.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()


def log_startup_banner(
    *,
    phase: str,
    run_name: str,
    encoder_source: str | None,
    decoder_source: str | None,
    optimizer_state_source: str | None,
    extras: dict[str, str | int | float | None] | None = None,
    notes: list[str] | None = None,
) -> None:
    """Emit a multi-line, visually-distinct banner describing checkpoint sources.

    Pass ``None`` for any source that loaded FRESH (HF pristine, random init,
    etc.) so the banner can flag it loudly. A trainer that meant to cold-start
    a component should still report that explicitly via ``notes`` —
    "decoder cold-start (intentional, no Qwen ckpt exists for current encoder)".

    ``extras`` carries any phase-specific values worth surfacing: active
    decoder family, projection anchor count, target ratio, teacher checkpoint
    for distillation runs, etc.
    """
    bar = "=" * 78
    hdr = "    !!!  STARTUP CHECKPOINT SUMMARY  !!!"
    lines: list[str] = [
        "",
        bar,
        hdr,
        bar,
        f"  phase     : {phase}",
        f"  run_name  : {run_name}",
        "",
        "  Checkpoint sources actually loaded into the trainer:",
        f"    encoder         : {_fmt(encoder_source)}",
        f"    decoder         : {_fmt(decoder_source)}",
        f"    optimizer_state : {_fmt(optimizer_state_source)}",
    ]
    if extras:
        lines.append("")
        lines.append("  Phase-specific:")
        for k in sorted(extras.keys()):
            lines.append(f"    {k:<20s}: {extras[k]}")
    if notes:
        lines.append("")
        lines.append("  Notes:")
        for note in notes:
            lines.append(f"    - {note}")
    lines.extend([
        "",
        "  >>> PLEASE CHECK IF THESE ARE THE CHECKPOINTS YOU WERE EXPECTING <<<",
        "  >>> Anything marked '** FRESH / RANDOM INIT **' is a cold start. <<<",
        bar,
        "",
    ])
    # One print call so the lines stay contiguous in container logs even
    # under interleaved output from other threads.
    print("\n".join(lines), flush=True)
    # Also emit a structured event so the same data is grep-able and shows
    # up in wandb's run config under a stable key.
    logger.info(
        "startup_checkpoint_summary",
        phase=phase,
        run_name=run_name,
        encoder_source=encoder_source,
        decoder_source=decoder_source,
        optimizer_state_source=optimizer_state_source,
        extras=extras or {},
        notes=notes or [],
    )


def _fmt(source: str | None) -> str:
    if source is None or source == "":
        return "** FRESH / RANDOM INIT **"
    return source
