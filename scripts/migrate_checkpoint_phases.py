#!/usr/bin/env python
"""Migrate checkpoint directories and registry from old phase names to new.

Rename mapping:
  phase1_step1a  -> phase1_step2
  commit_encoding -> phase1_step3
  phase1_step2   -> phase1_step4

This script:
1. Renames checkpoint directories (phase prefix in name)
2. Updates metadata.json phase fields inside each checkpoint
3. Updates registry.json entries

Safe to run multiple times (idempotent — skips already-migrated names).

Usage:
  python scripts/migrate_checkpoint_phases.py [--checkpoint-dir CHECKPOINT_DIR]
  python scripts/migrate_checkpoint_phases.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Phase rename map. Ordered by longest prefix first to avoid partial matches.
# History of renames:
#   v1: phase1_step1a -> phase1_step2, commit_encoding -> phase1_step3,
#       phase1_step2 (old compression) -> phase1_step4
#   v2: phase1_step3 (commit_encoding) -> phase1_step4,
#       phase1_step4 (compression) -> phase1_step5
RENAME_MAP = {
    "phase1_step4": "phase1_step5",  # old compression -> step 5
    "phase1_step3": "phase1_step4",  # old commit_encoding -> step 4
    # Earlier renames (already applied, kept for completeness):
    # "phase1_step1a": "phase1_step2",
    # "commit_encoding": "phase1_step3",
}


def migrate_checkpoint_dir(ckpt_path: Path, dry_run: bool = False) -> Path | None:
    """Rename a checkpoint directory and update its metadata.json.

    Returns the new path if renamed, None if skipped.
    """
    name = ckpt_path.name
    new_name = name
    matched_old = None
    for old, new in RENAME_MAP.items():
        if name.startswith(old + "_"):
            new_name = new + name[len(old):]
            matched_old = old
            break

    if matched_old is None:
        return None

    new_path = ckpt_path.parent / new_name
    if new_path.exists():
        print(f"  SKIP (target exists): {name} -> {new_name}")
        return None

    # Update metadata.json inside the checkpoint
    meta_path = ckpt_path / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("phase") == matched_old:
            meta["phase"] = RENAME_MAP[matched_old]
            if not dry_run:
                meta_path.write_text(json.dumps(meta, indent=2))
            print(f"  metadata.json: phase {matched_old} -> {RENAME_MAP[matched_old]}")

    # Rename the directory
    if not dry_run:
        ckpt_path.rename(new_path)
    print(f"  RENAME: {name} -> {new_name}")
    return new_path


def migrate_registry(checkpoint_dir: Path, dry_run: bool = False) -> int:
    """Update phase names in registry.json. Returns count of entries updated."""
    registry_path = checkpoint_dir / "registry.json"
    if not registry_path.exists():
        print("  No registry.json found, skipping.")
        return 0

    data = json.loads(registry_path.read_text())
    entries = data.get("entries", [])
    count = 0

    for entry in entries:
        old_phase = entry.get("phase", "")
        if old_phase in RENAME_MAP:
            entry["phase"] = RENAME_MAP[old_phase]
            count += 1

        # Also update the name field if it starts with an old phase prefix
        old_name = entry.get("name", "")
        for old, new in RENAME_MAP.items():
            if old_name.startswith(old + "_"):
                entry["name"] = new + old_name[len(old):]
                break

    if count > 0 and not dry_run:
        registry_path.write_text(json.dumps(data, indent=2))

    return count


def main():
    parser = argparse.ArgumentParser(description="Migrate checkpoint phase names")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Checkpoint directory (default: from .env CHECKPOINT_DIR)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would change")
    args = parser.parse_args()

    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        # Try to read from .env
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("CHECKPOINT_DIR="):
                    checkpoint_dir = Path(line.split("=", 1)[1].strip().strip('"'))
                    break
            else:
                checkpoint_dir = Path("checkpoints")
        else:
            checkpoint_dir = Path("checkpoints")

    if not checkpoint_dir.exists():
        print(f"Checkpoint directory does not exist: {checkpoint_dir}")
        sys.exit(1)

    prefix = "[DRY RUN] " if args.dry_run else ""
    print(f"{prefix}Migrating checkpoints in: {checkpoint_dir}")
    print(f"  Rename map: {RENAME_MAP}")
    print()

    # Step 1: Rename checkpoint directories (process phase1_step2->step4 FIRST
    # to avoid collision with phase1_step1a->step2)
    dirs = sorted(checkpoint_dir.iterdir())
    renamed = 0
    for d in dirs:
        if not d.is_dir() or d.name.startswith("."):
            continue
        result = migrate_checkpoint_dir(d, dry_run=args.dry_run)
        if result is not None:
            renamed += 1

    print(f"\n{prefix}Renamed {renamed} checkpoint directories.")

    # Step 2: Update registry.json
    print(f"\n{prefix}Updating registry.json...")
    count = migrate_registry(checkpoint_dir, dry_run=args.dry_run)
    print(f"{prefix}Updated {count} registry entries.")

    # Step 3: Update .last_checkpoint if it exists
    last_ckpt = checkpoint_dir / ".last_checkpoint"
    if last_ckpt.exists():
        content = last_ckpt.read_text().strip()
        for old, new in RENAME_MAP.items():
            if old in content:
                new_content = content.replace(old + "_", new + "_")
                if new_content != content:
                    if not args.dry_run:
                        last_ckpt.write_text(new_content)
                    print(f"\n{prefix}Updated .last_checkpoint: {content} -> {new_content}")
                    break

    print(f"\n{prefix}Migration complete.")


if __name__ == "__main__":
    main()
