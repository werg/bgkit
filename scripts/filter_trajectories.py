#!/usr/bin/env python3
"""Filter SWE-bench trajectories for distillation training.

Filtering pipeline:
1. Parse model_patch -> set of edited file paths
2. Iterate through trajectory messages:
   - Remove messages from exploration subagents
   - Classify file reads: edit-target (keep) vs non-edit (drop with probability p)
   - Keep all non-read actions
3. Store filtered trajectory alongside original
4. Record statistics

Usage:
    python scripts/filter_trajectories.py \
        --input trajectories/openhands_trajectories.jsonl \
        --output trajectories/openhands_filtered.jsonl \
        --drop-probability 0.8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from random import Random

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


def _extract_edited_files(model_patch: str) -> set[str]:
    """Extract file paths from a unified diff patch."""
    files = set()
    for line in model_patch.splitlines():
        if line.startswith("diff --git"):
            # Extract b/path from "diff --git a/path b/path"
            parts = line.split()
            if len(parts) >= 4:
                path = parts[3]
                if path.startswith("b/"):
                    path = path[2:]
                files.add(path)
        elif line.startswith("+++"):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            if path != "/dev/null":
                files.add(path)
    return files


def _is_subagent_message(message: dict) -> bool:
    """Detect messages from exploration subagents."""
    content = str(message.get("content", ""))

    # Heuristic: subagent messages often have specific markers
    if "subagent" in content.lower():
        return True
    if message.get("agent_type") == "exploration":
        return True
    if message.get("is_subagent", False):
        return True
    return False


def _is_file_read(message: dict) -> tuple[bool, str | None]:
    """Check if a message is a file read action and extract the file path."""
    content = str(message.get("content", ""))

    # Common file read patterns in agent trajectories
    patterns = [
        r"cat\s+([^\s|>]+)",
        r"open_file\s+([^\s]+)",
        r"view\s+([^\s]+)",
        r"read_file\s+([^\s]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return True, match.group(1)

    # Tool call patterns
    if message.get("tool_calls"):
        for call in message.get("tool_calls", []):
            fn = call.get("function", {})
            if fn.get("name") in ("read_file", "open_file", "view_file"):
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        continue
                path = args.get("path") or args.get("file_path") or args.get("filename")
                if path:
                    return True, str(path)

    return False, None


# ---------------------------------------------------------------------------
# Tool-call normalization: convert various teacher formats to Qwen3.5 JSON
# ---------------------------------------------------------------------------

# Mapping from various teacher tool names to our canonical names
_TOOL_NAME_MAP = {
    # OpenHands / OpenDevin names
    "open_file": "read_file",
    "view_file": "read_file",
    "goto": "read_file",
    "scroll_down": "read_file",
    "scroll_up": "read_file",
    "create_file": "edit_file",
    "insert_content_at_line": "edit_file",
    "replace_content_at_line": "edit_file",
    "str_replace_editor": "edit_file",
    # SWE-agent names
    "find_file": "list_files",
    "search_dir": "list_files",
    "search_file": "list_files",
    # Passthrough
    "read_file": "read_file",
    "edit_file": "edit_file",
    "list_files": "list_files",
    "run_command": "run_command",
    "done": "done",
}

# Argument key normalization: map various teacher arg keys to canonical keys
_ARG_KEY_MAP = {
    "file_path": "path",
    "filename": "path",
    "file_name": "path",
    "file": "path",
    "old_str": "old",
    "new_str": "new",
    "old_string": "old",
    "new_string": "new",
    "cmd": "command",
}


def _normalize_tool_call(message: dict) -> dict | None:
    """Convert a teacher tool-call message to Qwen3.5 JSON format.

    Handles:
    - OpenHands function_call JSON (tool_calls list with function.name + arguments)
    - SWE-agent custom format (action/action_input in content or structured fields)
    - Raw content with tool-call patterns

    Returns:
        A dict with {"name": "...", "arguments": {...}} in canonical format,
        or None if the message does not contain a recognizable tool call.
    """
    # --- Format 1: OpenAI-style tool_calls list (OpenHands, etc.) ---
    if message.get("tool_calls"):
        results = []
        for call in message["tool_calls"]:
            fn = call.get("function", {})
            raw_name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            canonical_name = _TOOL_NAME_MAP.get(raw_name, raw_name)
            canonical_args = {
                _ARG_KEY_MAP.get(k, k): v for k, v in args.items()
            }
            results.append({"name": canonical_name, "arguments": canonical_args})
        # Return first tool call (most trajectories have one per message)
        return results[0] if results else None

    # --- Format 2: function_call dict (older OpenAI format) ---
    fn_call = message.get("function_call")
    if fn_call and isinstance(fn_call, dict):
        raw_name = fn_call.get("name", "")
        args = fn_call.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        canonical_name = _TOOL_NAME_MAP.get(raw_name, raw_name)
        canonical_args = {_ARG_KEY_MAP.get(k, k): v for k, v in args.items()}
        return {"name": canonical_name, "arguments": canonical_args}

    # --- Format 3: SWE-agent structured fields ---
    action = message.get("action")
    if action and isinstance(action, str):
        action_input = message.get("action_input", message.get("args", ""))
        if isinstance(action_input, str):
            try:
                action_input = json.loads(action_input)
            except json.JSONDecodeError:
                action_input = {"command": action_input} if action_input else {}
        canonical_name = _TOOL_NAME_MAP.get(action, action)
        canonical_args = {
            _ARG_KEY_MAP.get(k, k): v
            for k, v in (action_input if isinstance(action_input, dict) else {}).items()
        }
        return {"name": canonical_name, "arguments": canonical_args}

    # --- Format 4: content-embedded tool calls ---
    content = str(message.get("content", ""))
    # Try JSON inside <tool_call> tags (already in target format)
    tc_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
    if tc_match:
        try:
            parsed = json.loads(tc_match.group(1))
            raw_name = parsed.get("name", "")
            args = parsed.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            canonical_name = _TOOL_NAME_MAP.get(raw_name, raw_name)
            canonical_args = {_ARG_KEY_MAP.get(k, k): v for k, v in args.items()}
            return {"name": canonical_name, "arguments": canonical_args}
        except json.JSONDecodeError:
            pass

    # Try Python-style tool calls: name(key="value")
    py_match = re.search(r"<tool_call>\s*(\w+)\(([^)]*)\)\s*</tool_call>", content, re.DOTALL)
    if py_match:
        raw_name = py_match.group(1)
        args_str = py_match.group(2)
        kv_re = re.compile(
            r'(\w+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\')',
        )
        args = {}
        for kv in kv_re.finditer(args_str):
            key = kv.group(1)
            value = kv.group(2) if kv.group(2) is not None else kv.group(3)
            value = value.replace('\\"', '"').replace("\\'", "'").replace("\\n", "\n")
            args[key] = value
        canonical_name = _TOOL_NAME_MAP.get(raw_name, raw_name)
        canonical_args = {_ARG_KEY_MAP.get(k, k): v for k, v in args.items()}
        return {"name": canonical_name, "arguments": canonical_args}

    return None


def _reformat_message_tool_calls(message: dict) -> dict:
    """Reformat a message's tool calls into Qwen3.5 JSON format.

    If the message contains a tool call in any recognized format, replaces
    the content (or structured fields) with the canonical Qwen3.5 format:
        <tool_call>
        {"name": "...", "arguments": {...}}
        </tool_call>

    Non-tool-call messages are returned unchanged.
    """
    normalized = _normalize_tool_call(message)
    if normalized is None:
        return message

    # Build a new message with Qwen3.5 format in content
    new_msg = dict(message)
    tc_json = json.dumps(normalized, ensure_ascii=False)
    new_msg["content"] = f"<tool_call>\n{tc_json}\n</tool_call>"
    # Remove legacy structured tool-call fields
    new_msg.pop("tool_calls", None)
    new_msg.pop("function_call", None)
    new_msg.pop("action", None)
    new_msg.pop("action_input", None)
    new_msg.pop("args", None)
    return new_msg


def filter_trajectory(
    trajectory: list[dict],
    edited_files: set[str],
    drop_probability: float = 0.8,
    rng: Random | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Filter a single trajectory and normalize tool calls to Qwen3.5 format.

    Returns:
        (filtered_messages, statistics)
    """
    if rng is None:
        rng = Random(42)

    stats = {
        "original_turns": len(trajectory),
        "subagent_removed": 0,
        "edit_target_reads_kept": 0,
        "non_edit_reads_kept": 0,
        "non_edit_reads_dropped": 0,
        "non_read_kept": 0,
        "tool_calls_normalized": 0,
    }

    filtered = []
    for message in trajectory:
        # Remove subagent messages
        if _is_subagent_message(message):
            stats["subagent_removed"] += 1
            continue

        # Classify file reads
        is_read, file_path = _is_file_read(message)
        if is_read and file_path:
            # Check if this is a read of an edit-target file
            is_edit_target = any(
                file_path.endswith(edited) or edited.endswith(file_path)
                for edited in edited_files
            )
            if is_edit_target:
                normalized = _reformat_message_tool_calls(message)
                if normalized is not message:
                    stats["tool_calls_normalized"] += 1
                filtered.append(normalized)
                stats["edit_target_reads_kept"] += 1
            elif rng.random() > drop_probability:
                normalized = _reformat_message_tool_calls(message)
                if normalized is not message:
                    stats["tool_calls_normalized"] += 1
                filtered.append(normalized)
                stats["non_edit_reads_kept"] += 1
            else:
                stats["non_edit_reads_dropped"] += 1
        else:
            # Normalize tool calls in non-read messages too
            normalized = _reformat_message_tool_calls(message)
            if normalized is not message:
                stats["tool_calls_normalized"] += 1
            filtered.append(normalized)
            stats["non_read_kept"] += 1

    stats["filtered_turns"] = len(filtered)
    return filtered, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drop-probability", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_stats = {
        "total_trajectories": 0,
        "resolved_trajectories": 0,
        "total_original_turns": 0,
        "total_filtered_turns": 0,
        "total_subagent_removed": 0,
        "total_edit_target_reads": 0,
        "total_non_edit_reads_dropped": 0,
    }

    with args.input.open() as fin, args.output.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)

            model_patch = record.get("model_patch", "")
            edited_files = _extract_edited_files(model_patch)

            # Extract trajectory messages
            trajectory = record.get("trajectory", record.get("messages", []))
            if not isinstance(trajectory, list):
                trajectory = []

            filtered, stats = filter_trajectory(
                trajectory, edited_files, args.drop_probability, rng,
            )

            record["filtered_trajectory"] = filtered
            record["filter_stats"] = stats
            fout.write(json.dumps(record, default=str) + "\n")

            total_stats["total_trajectories"] += 1
            if record.get("resolved", False):
                total_stats["resolved_trajectories"] += 1
            total_stats["total_original_turns"] += stats["original_turns"]
            total_stats["total_filtered_turns"] += stats["filtered_turns"]
            total_stats["total_subagent_removed"] += stats["subagent_removed"]
            total_stats["total_edit_target_reads"] += stats["edit_target_reads_kept"]
            total_stats["total_non_edit_reads_dropped"] += stats["non_edit_reads_dropped"]

    print(json.dumps(total_stats, indent=2))


if __name__ == "__main__":
    main()
