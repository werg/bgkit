"""SWE-bench evaluation integration.

Wraps the swebench harness for evaluation and provides
metrics computation for BgKIT-specific analysis.
"""

from __future__ import annotations

import json


def load_predictions(predictions_path: str) -> list[dict]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(predictions_path) as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions


def compute_resolution_rate(
    resolved_ids: list[str],
    total_ids: list[str],
) -> dict[str, float]:
    """Compute SWE-bench resolution metrics.

    Args:
        resolved_ids: Instance IDs that were resolved.
        total_ids: All instance IDs attempted.

    Returns:
        Dict with resolution rate and counts.
    """
    total = len(total_ids)
    resolved = len(resolved_ids)
    return {
        "resolution_rate": resolved / max(total, 1),
        "resolved_count": resolved,
        "total_count": total,
    }


def compute_trajectory_efficiency(
    predictions: list[dict],
    teacher_stats: dict[str, dict] | None = None,
) -> dict[str, float]:
    """Compute trajectory efficiency metrics.

    Measured during generation, not evaluation:
    - Avg turns per instance
    - File reads count
    - Time-to-first-edit
    """
    if not predictions:
        return {}

    total_turns = 0
    total_reads = 0
    total_first_edit = 0
    count = 0

    for pred in predictions:
        trajectory = pred.get("trajectory", [])
        if not trajectory:
            continue

        turns = len(trajectory)
        reads = sum(1 for msg in trajectory if _is_file_read(msg))

        first_edit = turns  # Default: never edited
        for i, msg in enumerate(trajectory):
            if _is_edit(msg):
                first_edit = i + 1
                break

        total_turns += turns
        total_reads += reads
        total_first_edit += first_edit
        count += 1

    if count == 0:
        return {}

    result = {
        "avg_turns": total_turns / count,
        "avg_file_reads": total_reads / count,
        "avg_time_to_first_edit": total_first_edit / count,
    }

    # Compare with teacher if available
    if teacher_stats:
        teacher_turns = teacher_stats.get("avg_turns", 0)
        if teacher_turns > 0:
            result["turns_ratio_vs_teacher"] = result["avg_turns"] / teacher_turns

    return result


def _is_file_read(msg: dict) -> bool:
    """Check if a trajectory message is a file read."""
    content = str(msg.get("content", ""))
    if any(cmd in content for cmd in ["cat ", "read_file", "open_file", "view "]):
        return True
    for call in msg.get("tool_calls", []):
        fn = call.get("function", {}).get("name", "")
        if fn in ("read_file", "open_file", "view_file"):
            return True
    return False


def _is_edit(msg: dict) -> bool:
    """Check if a trajectory message is a file edit."""
    content = str(msg.get("content", ""))
    if any(cmd in content for cmd in ["edit_file", "write_file", "apply_patch"]):
        return True
    for call in msg.get("tool_calls", []):
        fn = call.get("function", {}).get("name", "")
        if fn in ("edit_file", "write_file", "apply_patch", "str_replace_editor"):
            return True
    return False


def format_predictions_jsonl(
    predictions: list[dict],
    model_name: str = "bgkit",
    output_path: str | None = None,
) -> list[dict]:
    """Format predictions for swebench harness.

    Each prediction needs: instance_id, model_name_or_path, model_patch
    """
    formatted = []
    for pred in predictions:
        entry = {
            "instance_id": pred["instance_id"],
            "model_name_or_path": model_name,
            "model_patch": pred.get("model_patch", ""),
        }
        formatted.append(entry)

    if output_path:
        with open(output_path, "w") as f:
            for entry in formatted:
                f.write(json.dumps(entry) + "\n")

    return formatted
