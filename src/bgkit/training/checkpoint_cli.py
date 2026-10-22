"""CLI for checkpoint registry inspection and management.

Usage:
    bgkit-ckpt list [--phase X] [--status X] [--tag X]
    bgkit-ckpt show <name>
    bgkit-ckpt annotate <name> [--notes "..."] [--tag X]
    bgkit-ckpt best --phase X --metric X [--higher-is-better]
    bgkit-ckpt backfill

Environment Variables (set in .env):
    CHECKPOINT_DIR - Path to checkpoint directory (required)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bgkit.env import get_checkpoint_dir
from bgkit.training.checkpoint_registry import CheckpointRegistry


def _get_checkpoint_dir() -> Path:
    return get_checkpoint_dir()


def cmd_list(args: argparse.Namespace) -> None:
    registry = CheckpointRegistry(_get_checkpoint_dir())
    tags = [args.tag] if args.tag else None
    entries = registry.list_entries(phase=args.phase, status=args.status, tags=tags)
    if not entries:
        print("No checkpoints found.")
        return

    # Tabular output
    print(f"{'Name':<45} {'Phase':<20} {'Step':>8} {'Status':<12} {'Metric':>12} {'Tags'}")
    print("-" * 110)
    for e in entries:
        metric_str = ""
        if e.metrics:
            # Show first metric value
            key, val = next(iter(e.metrics.items()))
            metric_str = f"{key}={val:.4f}"
        tags_str = ", ".join(e.tags) if e.tags else ""
        print(f"{e.name:<45} {e.phase:<20} {e.step:>8} {e.status:<12} {metric_str:>12} {tags_str}")


def cmd_show(args: argparse.Namespace) -> None:
    registry = CheckpointRegistry(_get_checkpoint_dir())
    entry = registry.get(args.name)
    if entry is None:
        print(f"Checkpoint '{args.name}' not found in registry.", file=sys.stderr)
        sys.exit(1)
    from dataclasses import asdict

    print(json.dumps(asdict(entry), indent=2))


def cmd_annotate(args: argparse.Namespace) -> None:
    registry = CheckpointRegistry(_get_checkpoint_dir())
    tags = args.tag if args.tag else None
    ok = registry.annotate(args.name, notes=args.notes, tags=tags)
    if not ok:
        print(f"Checkpoint '{args.name}' not found in registry.", file=sys.stderr)
        sys.exit(1)
    print(f"Updated '{args.name}'.")


def cmd_best(args: argparse.Namespace) -> None:
    registry = CheckpointRegistry(_get_checkpoint_dir())
    entry = registry.best(
        phase=args.phase,
        metric=args.metric,
        lower_is_better=not args.higher_is_better,
    )
    if entry is None:
        print("No matching checkpoint found.", file=sys.stderr)
        sys.exit(1)
    val = entry.metrics[args.metric] if entry.metrics else "N/A"
    print(f"{entry.name}  ({args.metric}={val})")


def cmd_backfill(args: argparse.Namespace) -> None:
    checkpoint_dir = _get_checkpoint_dir()
    registry = CheckpointRegistry(checkpoint_dir)
    count = registry.backfill(checkpoint_dir)
    print(f"Backfilled {count} checkpoint(s) into registry.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bgkit-ckpt",
        description="Checkpoint registry management",
    )
    sub = parser.add_subparsers(dest="command")

    # list
    p_list = sub.add_parser("list", help="List checkpoints")
    p_list.add_argument("--phase", help="Filter by training phase")
    p_list.add_argument("--status", help="Filter by status (completed/interrupted/pruned)")
    p_list.add_argument("--tag", help="Filter by tag")

    # show
    p_show = sub.add_parser("show", help="Show full checkpoint details")
    p_show.add_argument("name", help="Checkpoint directory name")

    # annotate
    p_annotate = sub.add_parser("annotate", help="Add notes/tags to a checkpoint")
    p_annotate.add_argument("name", help="Checkpoint directory name")
    p_annotate.add_argument("--notes", help="Free-text notes")
    p_annotate.add_argument("--tag", action="append", help="Tag to add (repeatable)")

    # best
    p_best = sub.add_parser("best", help="Find best checkpoint by metric")
    p_best.add_argument("--phase", required=True, help="Training phase")
    p_best.add_argument("--metric", required=True, help="Metric key (e.g. eval/mse)")
    p_best.add_argument(
        "--higher-is-better", action="store_true", help="Higher metric values are better"
    )

    # backfill
    sub.add_parser("backfill", help="Populate registry from on-disk checkpoints")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "list": cmd_list,
        "show": cmd_show,
        "annotate": cmd_annotate,
        "best": cmd_best,
        "backfill": cmd_backfill,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
