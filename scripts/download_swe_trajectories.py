#!/usr/bin/env python3
"""Download SWE-bench trajectory datasets from HuggingFace.

Downloads and joins trajectories with task definitions for base_commit info.

Datasets:
- nebius/SWE-rebench-openhands-trajectories (67K)
- nebius/SWE-agent-trajectories (80K)
- nvidia/Nemotron-SWE-v1 (59K)
- SWE-bench/SWE-smith-trajectories (5K)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


_TRAJECTORY_DATASETS = {
    "openhands": {
        "trajectory_id": "nebius/SWE-rebench-openhands-trajectories",
        "task_id": "nebius/SWE-rebench",
    },
    "swe_agent": {
        "trajectory_id": "nebius/SWE-agent-trajectories",
    },
    "nemotron": {
        "trajectory_id": "nvidia/Nemotron-SWE-v1",
    },
    "swe_smith": {
        "trajectory_id": "SWE-bench/SWE-smith-trajectories",
    },
}


def download_dataset(name: str, output_dir: Path) -> None:
    """Download a trajectory dataset and save as JSONL."""
    from datasets import load_dataset

    spec = _TRAJECTORY_DATASETS[name]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}_trajectories.jsonl"

    if output_path.exists():
        print(f"Already exists: {output_path}")
        return

    print(f"Downloading {spec['trajectory_id']}...")
    try:
        ds = load_dataset(spec["trajectory_id"], split="train")
    except Exception as exc:
        print(f"Warning: Failed to download {name}: {exc}", file=sys.stderr)
        return

    # Load task definitions for base_commit joining
    task_map = {}
    if "task_id" in spec:
        try:
            tasks = load_dataset(spec["task_id"], split="test")
            for task in tasks:
                instance_id = task.get("instance_id")
                if instance_id:
                    task_map[instance_id] = {
                        "base_commit": task.get("base_commit"),
                        "repo": task.get("repo"),
                        "version": task.get("version"),
                    }
        except Exception as exc:
            print(f"Warning: Could not load tasks for {name}: {exc}", file=sys.stderr)

    count = 0
    with output_path.open("w") as f:
        for row in ds:
            record = dict(row)
            # Join with task definitions
            instance_id = record.get("instance_id")
            if instance_id and instance_id in task_map:
                record.update(task_map[instance_id])
            f.write(json.dumps(record, default=str) + "\n")
            count += 1

    print(f"Saved {count} trajectories to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(_TRAJECTORY_DATASETS),
        default=["openhands"],
        help="Datasets to download",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for name in args.datasets:
        download_dataset(name, args.output_dir)


if __name__ == "__main__":
    main()
