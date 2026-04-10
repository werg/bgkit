"""Tests for trajectory filtering from filter_trajectories.py."""

from __future__ import annotations

import sys
from pathlib import Path
from random import Random

import pytest

_scripts = str(Path(__file__).resolve().parents[3] / "scripts")
_src = str(Path(__file__).resolve().parents[3] / "src")
for p in (_scripts, _src):
    if p not in sys.path:
        sys.path.insert(0, p)

from filter_trajectories import (
    _extract_edited_files,
    _is_file_read,
    _is_subagent_message,
    filter_trajectory,
)


# ===================================================================
# _extract_edited_files
# ===================================================================


class TestExtractEditedFiles:
    def test_parses_diff_header(self):
        patch = (
            "diff --git a/src/foo.py b/src/foo.py\n"
            "--- a/src/foo.py\n"
            "+++ b/src/foo.py\n"
            "@@ -1,3 +1,3 @@\n"
            "-old line\n"
            "+new line\n"
        )
        files = _extract_edited_files(patch)
        assert "src/foo.py" in files

    def test_parses_multiple_files(self):
        patch = (
            "diff --git a/src/foo.py b/src/foo.py\n"
            "+++ b/src/foo.py\n"
            "diff --git a/src/bar.py b/src/bar.py\n"
            "+++ b/src/bar.py\n"
        )
        files = _extract_edited_files(patch)
        assert files == {"src/foo.py", "src/bar.py"}

    def test_new_file(self):
        patch = (
            "diff --git a/new_file.py b/new_file.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/new_file.py\n"
        )
        files = _extract_edited_files(patch)
        assert "new_file.py" in files
        assert "/dev/null" not in files

    def test_empty_patch(self):
        assert _extract_edited_files("") == set()


# ===================================================================
# _is_file_read
# ===================================================================


class TestIsFileRead:
    def test_cat_pattern(self):
        msg = {"content": "cat src/main.py"}
        is_read, path = _is_file_read(msg)
        assert is_read
        assert path == "src/main.py"

    def test_open_file_pattern(self):
        msg = {"content": "open_file src/utils.py"}
        is_read, path = _is_file_read(msg)
        assert is_read
        assert path == "src/utils.py"

    def test_view_pattern(self):
        msg = {"content": "view tests/test_foo.py"}
        is_read, path = _is_file_read(msg)
        assert is_read
        assert path == "tests/test_foo.py"

    def test_read_file_pattern(self):
        msg = {"content": "read_file config.yaml"}
        is_read, path = _is_file_read(msg)
        assert is_read
        assert path == "config.yaml"

    def test_tool_call_pattern(self):
        msg = {
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "src/auth.py"},
                },
            }],
        }
        is_read, path = _is_file_read(msg)
        assert is_read
        assert path == "src/auth.py"

    def test_non_read_message(self):
        msg = {"content": "Let me think about this problem."}
        is_read, path = _is_file_read(msg)
        assert not is_read
        assert path is None


# ===================================================================
# filter_trajectory
# ===================================================================


class TestFilterTrajectory:
    def test_keeps_edit_target_reads(self):
        trajectory = [
            {"content": "cat src/auth.py", "role": "assistant"},
            {"content": "file contents here", "role": "tool"},
        ]
        edited_files = {"src/auth.py"}
        filtered, stats = filter_trajectory(trajectory, edited_files)
        assert len(filtered) == 2
        assert stats["edit_target_reads_kept"] == 1

    def test_drops_non_edit_reads_with_probability(self):
        trajectory = [
            {"content": "cat src/unrelated.py", "role": "assistant"},
        ]
        edited_files = {"src/auth.py"}
        # With drop_probability=1.0, all non-edit reads should be dropped
        filtered, stats = filter_trajectory(
            trajectory, edited_files, drop_probability=1.0,
        )
        assert len(filtered) == 0
        assert stats["non_edit_reads_dropped"] == 1

    def test_keeps_non_edit_reads_sometimes(self):
        # Use a seeded RNG that will keep a message (rng.random() > 0.5 means keep)
        trajectory = [
            {"content": f"cat src/file{i}.py", "role": "assistant"}
            for i in range(100)
        ]
        edited_files = set()  # no edit targets
        filtered, stats = filter_trajectory(
            trajectory, edited_files, drop_probability=0.5, rng=Random(42),
        )
        # With 0.5 probability, approximately half should be kept
        assert 20 < stats["non_edit_reads_kept"] < 80
        assert stats["non_edit_reads_dropped"] > 0

    def test_removes_subagent_messages(self):
        trajectory = [
            {"content": "subagent exploration started", "role": "assistant"},
            {"content": "I will edit the file", "role": "assistant"},
        ]
        edited_files = set()
        filtered, stats = filter_trajectory(trajectory, edited_files)
        assert stats["subagent_removed"] == 1
        assert len(filtered) == 1

    def test_keeps_non_read_actions(self):
        trajectory = [
            {"content": "I will apply this fix now", "role": "assistant"},
            {"content": "edit done", "role": "tool"},
        ]
        edited_files = set()
        filtered, stats = filter_trajectory(trajectory, edited_files)
        assert len(filtered) == 2
        assert stats["non_read_kept"] == 2

    def test_statistics_computed_correctly(self):
        trajectory = [
            {"content": "cat src/auth.py", "role": "assistant"},     # edit target read
            {"content": "cat src/other.py", "role": "assistant"},    # non-edit read
            {"content": "I will fix this", "role": "assistant"},     # non-read
            {"content": "subagent checking", "role": "assistant"},   # subagent
        ]
        edited_files = {"src/auth.py"}
        filtered, stats = filter_trajectory(
            trajectory, edited_files, drop_probability=0.0,
        )
        assert stats["original_turns"] == 4
        assert stats["edit_target_reads_kept"] == 1
        assert stats["subagent_removed"] == 1
        assert stats["non_read_kept"] == 1
        assert stats["filtered_turns"] == 3
