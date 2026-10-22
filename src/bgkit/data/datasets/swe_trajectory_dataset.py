"""SWE-bench trajectory dataset for Phase 3 distillation."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class SWETrajectoryDataset(Dataset):
    """Loads filtered SWE-bench trajectories for distillation training.

    Each item returns:
    - issue_text: the SWE-bench issue description
    - filtered_trajectory: the filtered teacher trajectory
    - model_patch: the gold patch (for evaluation)
    - instance_id: SWE-bench instance identifier
    - repo: repository name
    - base_commit: commit SHA at which to check out the repo
    """

    def __init__(
        self,
        data_dir: str,
        *,
        tokenizer=None,
        max_trajectory_tokens: int = 32768,
        max_issue_tokens: int = 2048,
        require_resolved: bool = True,
    ):
        self._data_dir = Path(data_dir)
        self._tokenizer = tokenizer
        self._max_trajectory_tokens = max_trajectory_tokens
        self._max_issue_tokens = max_issue_tokens

        # Load all trajectory JSONL files
        self._records = []
        for jsonl_path in sorted(self._data_dir.glob("*_filtered.jsonl")):
            with jsonl_path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if require_resolved and not record.get("resolved", False):
                        continue
                    self._records.append(record)

        # Fallback: try non-filtered files
        if not self._records:
            for jsonl_path in sorted(self._data_dir.glob("*.jsonl")):
                with jsonl_path.open() as f:
                    for line in f:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        if require_resolved and not record.get("resolved", False):
                            continue
                        self._records.append(record)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        record = self._records[idx]

        # Extract issue/problem statement
        issue_text = (
            record.get("problem_statement")
            or record.get("issue_text")
            or record.get("input", "")
        )

        # Get filtered trajectory (or original)
        trajectory = (
            record.get("filtered_trajectory")
            or record.get("trajectory")
            or record.get("messages", [])
        )

        # Format trajectory as text
        trajectory_text = self._format_trajectory(trajectory)

        result = {
            "issue_text": str(issue_text),
            "trajectory_text": trajectory_text,
            "model_patch": str(record.get("model_patch", "")),
            "instance_id": str(record.get("instance_id", f"idx_{idx}")),
            "repo": str(record.get("repo", "")),
            "base_commit": str(record.get("base_commit", "")),
            # Optional wall-clock ordering for leakage-safe prior-session
            # context. Commit SHAs are identifiers, not chronological values.
            "trajectory_timestamp": (
                record.get("trajectory_timestamp")
                or record.get("created_at")
                or record.get("timestamp")
            ),
        }

        # Tokenize if tokenizer available
        if self._tokenizer:
            result["issue_token_ids"] = torch.tensor(
                self._tokenizer.encode(
                    issue_text, add_special_tokens=False,
                )[:self._max_issue_tokens],
                dtype=torch.long,
            )
            result["trajectory_token_ids"] = torch.tensor(
                self._tokenizer.encode(
                    trajectory_text, add_special_tokens=False,
                )[:self._max_trajectory_tokens],
                dtype=torch.long,
            )

        return result

    def _format_trajectory(self, trajectory: list[dict]) -> str:
        """Format trajectory messages as a single text string."""
        parts = []
        for msg in trajectory:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        item.get("text", str(item))
                        if isinstance(item, dict) else str(item)
                        for item in content
                    )
                parts.append(f"[{role}]\n{content}")
            else:
                parts.append(str(msg))
        return "\n\n".join(parts)
