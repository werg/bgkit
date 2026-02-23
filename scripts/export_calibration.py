#!/usr/bin/env python3
"""Export calibrator snapshots from a training checkpoint to YAML.

Usage:
    python scripts/export_calibration.py <checkpoint_dir> [-o output.yaml]

Loads L0 and L1 calibrator state from a checkpoint and writes static
CDF snapshots that can be used at inference time via
``threshold_from_snapshot()``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

from bgkit.data.threshold_calibrator import ThresholdCalibrator, threshold_from_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Export calibrator snapshots from checkpoint")
    parser.add_argument("checkpoint_dir", type=Path, help="Path to checkpoint directory")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output YAML path")
    parser.add_argument(
        "--ratio", type=float, default=0.15,
        help="Example target ratio for threshold computation (default: 0.15)",
    )
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir
    if not ckpt_dir.exists():
        print(f"Error: checkpoint directory not found: {ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    snapshots: dict[str, dict] = {}

    for name in ("l0_calibrator", "l1_calibrator"):
        pt_path = ckpt_dir / f"{name}.pt"
        if not pt_path.exists():
            print(f"Warning: {pt_path} not found, skipping {name}", file=sys.stderr)
            continue

        state = torch.load(pt_path, map_location="cpu", weights_only=True)
        calibrator = ThresholdCalibrator()
        calibrator.load_state_dict(state)
        snap = calibrator.snapshot()
        snapshots[name] = snap

        # Show example threshold
        threshold = threshold_from_snapshot(snap, args.ratio)
        print(f"{name}: ratio={args.ratio} -> threshold={threshold:.4f}")

    if not snapshots:
        print("No calibrators found in checkpoint", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or (ckpt_dir / "calibration_snapshots.yaml")
    with open(output_path, "w") as f:
        yaml.dump(snapshots, f, default_flow_style=False)
    print(f"Snapshots written to {output_path}")


if __name__ == "__main__":
    main()
