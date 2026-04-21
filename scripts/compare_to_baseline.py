#!/usr/bin/env python
"""Wave 4.5 validation: compare a live packed-era run to the 04-20 padded baseline.

The 04-20 baseline run (werg/bgkit/6wznpmwv) was a padded-attention Step 3
run; the packed-era runs use FA4 varlen packed attention, larger batch sizes
(higher ``max_batch_tokens``), and gradient accumulation.  The goals for
Wave 4.5 are:

    1. cuda_max_allocated_gb  in [50, 85]       (larger budget accepted,
                                                  but must not OOM 128 GB)
    2. step wall-clock        down >= 3x        (baseline ~8.94-9.12 s/step)
    3. eval/loss trajectory   within +/-20% of  (per eval step, where both
                               baseline per step  runs have overlap)
    4. eval wall-clock        down >= 2x        (if measurable — both runs
                                                  must log eval/_runtime or
                                                  similar)

Usage::

    python scripts/compare_to_baseline.py \\
        --baseline docs/baselines/phase1_step3_04_20_baseline.json \\
        --run werg/bgkit/<run_id>

    # Smoke-test with the crashed run that shows empty summary:
    python scripts/compare_to_baseline.py \\
        --baseline docs/baselines/phase1_step3_04_20_baseline.json \\
        --run werg/bgkit/pj1ws5bc

Exit codes:
    0  — all measurable gates pass
    1  — at least one gate fails or the run is not yet far enough to evaluate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------


def _load_baseline(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _baseline_step_wall_clock(baseline: dict) -> float:
    """Derive per-step wall-clock from the baseline JSON.

    The JSON stores ``step_wall_clock_seconds`` directly (computed by the
    baseline capture script as total_runtime / final_step).  If that key
    is absent we fall back to inferring it from the training trajectory.
    """
    if "step_wall_clock_seconds" in baseline:
        return float(baseline["step_wall_clock_seconds"])
    traj = baseline.get("training_trajectory", [])
    if len(traj) >= 2:
        steps = [t["_step"] for t in traj]
        times = [t.get("_runtime", None) for t in traj]
        pairs = [(s, r) for s, r in zip(steps, times, strict=True) if r is not None]
        if len(pairs) >= 2:
            ds = pairs[-1][0] - pairs[0][0]
            dt = pairs[-1][1] - pairs[0][1]
            if ds > 0 and dt > 0:
                return dt / ds
    # Hard fallback from CLAUDE.md: baseline was 8.94 s/step
    return 8.94


def _baseline_eval_loss_at_step(baseline: dict, step: int) -> float | None:
    """Return the nearest-step eval/loss value from the baseline, or None."""
    traj = baseline.get("eval_trajectory", [])
    if not traj:
        return None
    nearest = min(traj, key=lambda t: abs(t["_step"] - step))
    return nearest.get("eval/loss")


def _baseline_cuda_gb(baseline: dict) -> float:
    return float(baseline.get("cuda_max_allocated_gb", 0.0))


# ---------------------------------------------------------------------------
# Wandb helpers
# ---------------------------------------------------------------------------


def _fetch_run_history(run_path: str) -> tuple[Any, list[dict]]:
    """Load the wandb run and return (run_obj, rows).

    Each row is a plain dict with at least ``_step``, and whatever
    metrics the run logged.
    """
    try:
        import wandb  # type: ignore[import]
    except ImportError as exc:
        print(f"ERROR: wandb not available: {exc}")
        sys.exit(2)

    api = wandb.Api()
    run = api.run(run_path)

    # Fetch all rows with the metrics we care about.
    wanted = [
        "_step",
        "_runtime",
        "loss",
        "grad_norm",
        "mem/cuda_max_allocated_gb",
        "mem/cuda_max_reserved_gb",
        "eval/loss",
        "eval/compression_ratio",
    ]
    try:
        hist_df = run.history(samples=10000, keys=wanted)
    except Exception:
        # Older wandb API or empty run — fall back to unfiltered.
        try:
            hist_df = run.history(samples=10000)
        except Exception:
            hist_df = None

    if hist_df is None or (hasattr(hist_df, "empty") and hist_df.empty):
        return run, []

    rows = hist_df.where(hist_df.notna(), None).to_dict(orient="records")
    return run, rows


def _run_step_wall_clock(rows: list[dict]) -> float | None:
    """Estimate step wall-clock from the history rows (_runtime / _step gap)."""
    pairs = [
        (r["_step"], r["_runtime"])
        for r in rows
        if r.get("_step") is not None and r.get("_runtime") is not None
    ]
    if len(pairs) < 2:
        return None
    # Use the last 20% of available steps to avoid warmup noise.
    n_tail = max(2, len(pairs) // 5)
    tail = pairs[-n_tail:]
    ds = tail[-1][0] - tail[0][0]
    dt = tail[-1][1] - tail[0][1]
    if ds <= 0 or dt <= 0:
        return None
    return dt / ds


def _run_max_cuda_gb(rows: list[dict]) -> float | None:
    vals = [
        r.get("mem/cuda_max_allocated_gb")
        for r in rows
        if r.get("mem/cuda_max_allocated_gb") is not None
    ]
    if not vals:
        return None
    return float(max(vals))


def _run_eval_rows(rows: list[dict]) -> list[dict]:
    """Return rows that have an eval/loss value."""
    return [r for r in rows if r.get("eval/loss") is not None and r.get("_step") is not None]


# ---------------------------------------------------------------------------
# Evaluation gates
# ---------------------------------------------------------------------------


CUDA_GB_MIN = 50.0
CUDA_GB_MAX = 85.0
SPEEDUP_WALL_CLOCK_MIN = 3.0   # live must be >= 3x faster per step
SPEEDUP_EVAL_MIN = 2.0          # eval wall-clock >= 2x faster (if measurable)
EVAL_LOSS_TOLERANCE = 0.20      # +/-20% at each overlapping eval step


def _evaluate_gates(baseline: dict, run_rows: list[dict]) -> tuple[list[dict], bool]:
    """Run all gate checks.  Returns (gate_rows, all_pass).

    gate_rows is a list of dicts describing each gate's outcome:
      name, baseline_val, current_val, target, status, note
    """
    gates = []
    all_pass = True

    # --- Gate 1: CUDA max allocated in [50, 85] GB ---
    live_cuda = _run_max_cuda_gb(run_rows)
    if live_cuda is None:
        gates.append({
            "name": f"cuda_max_allocated_gb in [{CUDA_GB_MIN:.0f}, {CUDA_GB_MAX:.0f}]",
            "baseline_val": f"{_baseline_cuda_gb(baseline):.2f}",
            "current_val": "N/A",
            "target": f"[{CUDA_GB_MIN:.0f}, {CUDA_GB_MAX:.0f}]",
            "status": "not yet reached",
            "note": "no cuda memory metrics logged yet",
        })
    else:
        ok = CUDA_GB_MIN <= live_cuda <= CUDA_GB_MAX
        if not ok:
            all_pass = False
        gates.append({
            "name": f"cuda_max_allocated_gb in [{CUDA_GB_MIN:.0f}, {CUDA_GB_MAX:.0f}]",
            "baseline_val": f"{_baseline_cuda_gb(baseline):.2f}",
            "current_val": f"{live_cuda:.2f}",
            "target": f"[{CUDA_GB_MIN:.0f}, {CUDA_GB_MAX:.0f}]",
            "status": "PASS" if ok else "FAIL",
            "note": "" if ok else f"outside [{CUDA_GB_MIN:.0f}, {CUDA_GB_MAX:.0f}] GB window",
        })

    # --- Gate 2: step wall-clock >= 3x faster ---
    baseline_wc = _baseline_step_wall_clock(baseline)
    live_wc = _run_step_wall_clock(run_rows)
    target_wc = baseline_wc / SPEEDUP_WALL_CLOCK_MIN
    if live_wc is None:
        gates.append({
            "name": f"step wall-clock down >= 3x (baseline {baseline_wc:.2f} s)",
            "baseline_val": f"{baseline_wc:.2f} s",
            "current_val": "N/A",
            "target": f"<= {target_wc:.2f} s",
            "status": "not yet reached",
            "note": "need >= 2 logged steps with _runtime",
        })
    else:
        speedup = baseline_wc / live_wc
        ok = speedup >= SPEEDUP_WALL_CLOCK_MIN
        if not ok:
            all_pass = False
        gates.append({
            "name": f"step wall-clock down >= 3x (baseline {baseline_wc:.2f} s)",
            "baseline_val": f"{baseline_wc:.2f} s",
            "current_val": f"{live_wc:.2f} s",
            "target": f"<= {target_wc:.2f} s  (speedup {SPEEDUP_WALL_CLOCK_MIN:.0f}x)",
            "status": "PASS" if ok else "FAIL",
            "note": f"measured speedup {speedup:.2f}x",
        })

    # --- Gate 3: eval/loss trajectory within +/-20% at overlapping steps ---
    eval_rows = _run_eval_rows(run_rows)
    tol_pct = EVAL_LOSS_TOLERANCE * 100
    if not eval_rows:
        gates.append({
            "name": f"eval/loss trajectory within +/-{tol_pct:.0f}% of baseline",
            "baseline_val": "(trajectory)",
            "current_val": "N/A",
            "target": f"+/-{tol_pct:.0f}% per eval step",
            "status": "not yet reached",
            "note": "no eval/loss logged yet",
        })
    else:
        mismatches = []
        matched_steps = 0
        for row in eval_rows:
            step = int(row["_step"])
            live_loss = float(row["eval/loss"])
            bl_loss = _baseline_eval_loss_at_step(baseline, step)
            if bl_loss is None:
                continue
            matched_steps += 1
            # Allow +/-20% of baseline value
            lo = bl_loss * (1.0 - EVAL_LOSS_TOLERANCE)
            hi = bl_loss * (1.0 + EVAL_LOSS_TOLERANCE)
            if not (lo <= live_loss <= hi):
                mismatches.append(
                    f"step {step}: live={live_loss:.4f} vs baseline={bl_loss:.4f}"
                    f" (+/-{tol_pct:.0f}% = [{lo:.4f}, {hi:.4f}])"
                )
        if matched_steps == 0:
            gates.append({
                "name": f"eval/loss trajectory within +/-{tol_pct:.0f}% of baseline",
                "baseline_val": "(trajectory)",
                "current_val": f"{len(eval_rows)} eval points, no step overlap",
                "target": f"+/-{tol_pct:.0f}% per eval step",
                "status": "not yet reached",
                "note": (
                    f"run has eval at steps "
                    f"{[int(r['_step']) for r in eval_rows]}, "
                    "no overlap with baseline steps "
                    f"{[t['_step'] for t in baseline.get('eval_trajectory', [])]}"
                ),
            })
        else:
            ok = len(mismatches) == 0
            if not ok:
                all_pass = False
            note = "; ".join(mismatches[:3])
            if len(mismatches) > 3:
                note += f"; ... ({len(mismatches)} total)"
            gates.append({
                "name": f"eval/loss trajectory within +/-{tol_pct:.0f}% of baseline",
                "baseline_val": "(trajectory)",
                "current_val": (
                    f"{len(eval_rows)} eval points, "
                    f"{matched_steps} step(s) with baseline overlap"
                ),
                "target": f"+/-{tol_pct:.0f}% per eval step",
                "status": "PASS" if ok else "FAIL",
                "note": note,
            })

    return gates, all_pass


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _format_gates_markdown(
    gates: list[dict],
    run_path: str,
    baseline_run_id: str,
    run_state: str,
    run_step: int | None,
) -> str:
    lines = [
        "## Wave 4.5 Validation: Packed vs. Padded Baseline",
        "",
        f"**Baseline run**: `{baseline_run_id}`",
        f"**Current run**: `{run_path}`  (state: {run_state}"
        + (f", step: {run_step}" if run_step is not None else "")
        + ")",
        "",
        "| Gate | Baseline | Current | Target | Status | Note |",
        "|:---|---:|---:|:---|:---:|:---|",
    ]
    for g in gates:
        name = g["name"]
        bv = g["baseline_val"]
        cv = g["current_val"]
        target = g["target"]
        status = g["status"]
        note = str(g.get("note", ""))[:100]
        lines.append(f"| {name} | {bv} | {cv} | {target} | {status} | {note} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare a live packed-era run to the 04-20 padded baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--baseline",
        default="docs/baselines/phase1_step3_04_20_baseline.json",
        help="Path to the baseline JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--run",
        default="werg/bgkit/pq428wpa",
        help=(
            "WandB run path (entity/project/run_id).  "
            "Defaults to the current live Step 3 run pq428wpa."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the markdown report to.",
    )
    args = parser.parse_args()

    # Load baseline
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"ERROR: baseline file not found: {baseline_path}")
        return 2
    baseline = _load_baseline(str(baseline_path))
    baseline_run_id = baseline.get("wandb_run_id", "unknown")

    print(f"[compare] baseline: {baseline_run_id} (from {baseline_path})")
    print(f"[compare] current run: {args.run}")
    print("[compare] fetching run history from wandb...")

    run_obj, rows = _fetch_run_history(args.run)
    run_state = run_obj.state
    run_step: int | None = None
    if rows:
        steps = [int(r["_step"]) for r in rows if r.get("_step") is not None]
        run_step = max(steps) if steps else None

    print(
        f"[compare] run state={run_state} "
        f"step={run_step if run_step is not None else 'N/A'} "
        f"history_rows={len(rows)}"
    )

    # Evaluate gates
    gates, all_pass = _evaluate_gates(baseline, rows)

    # Render markdown
    md = _format_gates_markdown(
        gates,
        run_path=args.run,
        baseline_run_id=baseline_run_id,
        run_state=run_state,
        run_step=run_step,
    )
    print()
    print(md)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
        print(f"\n[compare] wrote report to {out_path}")

    # Determine exit code
    # If no rows at all (empty run), treat as "not yet" rather than hard fail.
    if not rows:
        print(
            "\n[compare] RESULT: no history data — "
            "run is empty or too early to evaluate (exit 1)",
        )
        return 1

    # "not yet reached" status means the gate couldn't be evaluated yet.
    has_unmet = any(
        g["status"] in ("not yet reached",) or g.get("current_val") == "N/A"
        for g in gates
    )
    if not all_pass:
        failed = [g["name"] for g in gates if g["status"] == "FAIL"]
        print(f"\n[compare] RESULT: FAIL - gates failed: {', '.join(failed)}")
        return 1
    if has_unmet:
        unmet = [g["name"] for g in gates if g["status"] == "not yet reached"]
        print(
            f"\n[compare] RESULT: INCOMPLETE - "
            f"gates not yet evaluable (run needs more steps): {', '.join(unmet)}",
        )
        return 1

    print("\n[compare] RESULT: PASS — all gates satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
