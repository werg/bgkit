"""Tests for tool-call parsing and execution from eval_swebench.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the scripts directory's src path is importable.
_scripts = str(Path(__file__).resolve().parents[3] / "scripts")
_src = str(Path(__file__).resolve().parents[3] / "src")
for p in (_scripts, _src):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_swebench import _execute_tool, _extract_diff, _parse_tool_calls


# ===================================================================
# _parse_tool_calls
# ===================================================================


class TestParseToolCalls:
    def test_basic_read_file(self):
        text = (
            'Some reasoning here.\n'
            '<tool_call>\n'
            '{"name": "read_file", "arguments": {"path": "src/foo.py"}}\n'
            '</tool_call>'
        )
        calls = _parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "read_file"
        assert calls[0]["arguments"]["path"] == "src/foo.py"

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>\n'
            'Some analysis.\n'
            '<tool_call>{"name": "edit_file", "arguments": '
            '{"path": "a.py", "old": "x", "new": "y"}}</tool_call>'
        )
        calls = _parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["name"] == "read_file"
        assert calls[1]["name"] == "edit_file"

    def test_escaped_quotes_in_arguments(self):
        text = (
            '<tool_call>\n'
            '{"name": "edit_file", "arguments": '
            '{"path": "test.py", "old": "x = \\"hello\\"", "new": "x = \\"world\\""}}\n'
            '</tool_call>'
        )
        calls = _parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["arguments"]["old"] == 'x = "hello"'

    def test_newlines_in_values(self):
        text = (
            '<tool_call>\n'
            '{"name": "edit_file", "arguments": '
            '{"path": "test.py", "old": "line1\\nline2", "new": "line1\\nline2\\nline3"}}\n'
            '</tool_call>'
        )
        calls = _parse_tool_calls(text)
        assert len(calls) == 1
        assert "line1\nline2" in calls[0]["arguments"]["old"]

    def test_invalid_json_skipped(self):
        text = (
            '<tool_call>not valid json</tool_call>\n'
            '<tool_call>{"name": "done", "arguments": {}}</tool_call>'
        )
        calls = _parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "done"

    def test_missing_name_skipped(self):
        text = '<tool_call>{"arguments": {"path": "foo.py"}}</tool_call>'
        calls = _parse_tool_calls(text)
        assert len(calls) == 0

    def test_string_arguments_parsed(self):
        text = (
            '<tool_call>\n'
            '{"name": "read_file", "arguments": "{\\"path\\": \\"foo.py\\"}"}\n'
            '</tool_call>'
        )
        calls = _parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["arguments"]["path"] == "foo.py"


# ===================================================================
# _execute_tool
# ===================================================================


class TestExecuteTool:
    def test_read_file_reads_content(self, tmp_path):
        test_file = tmp_path / "hello.py"
        test_file.write_text("print('hello world')")
        result = _execute_tool(
            {"name": "read_file", "arguments": {"path": "hello.py"}},
            tmp_path,
        )
        assert "print('hello world')" in result

    def test_read_file_missing_returns_error(self, tmp_path):
        result = _execute_tool(
            {"name": "read_file", "arguments": {"path": "nonexistent.py"}},
            tmp_path,
        )
        assert "Error" in result

    def test_edit_file_modifies_content(self, tmp_path):
        test_file = tmp_path / "code.py"
        test_file.write_text("x = 1\ny = 2\n")
        result = _execute_tool(
            {"name": "edit_file", "arguments": {
                "path": "code.py",
                "old": "x = 1",
                "new": "x = 42",
            }},
            tmp_path,
        )
        assert "Edited" in result
        assert test_file.read_text() == "x = 42\ny = 2\n"

    def test_edit_file_creates_new(self, tmp_path):
        result = _execute_tool(
            {"name": "edit_file", "arguments": {
                "path": "new_file.py",
                "new": "# new content",
            }},
            tmp_path,
        )
        assert "Created" in result
        assert (tmp_path / "new_file.py").read_text() == "# new content"

    def test_edit_file_old_not_found(self, tmp_path):
        test_file = tmp_path / "code.py"
        test_file.write_text("x = 1")
        result = _execute_tool(
            {"name": "edit_file", "arguments": {
                "path": "code.py",
                "old": "nonexistent string",
                "new": "replacement",
            }},
            tmp_path,
        )
        assert "Error" in result

    def test_done_returns_sentinel(self, tmp_path):
        result = _execute_tool(
            {"name": "done", "arguments": {}},
            tmp_path,
        )
        assert result == "__DONE__"

    def test_unknown_tool_returns_error(self, tmp_path):
        result = _execute_tool(
            {"name": "unknown_tool", "arguments": {}},
            tmp_path,
        )
        assert "Error" in result
        assert "unknown" in result


# ===================================================================
# _extract_diff
# ===================================================================


class TestExtractDiff:
    def test_extract_diff_from_modified_file(self, tmp_path):
        """Creates a git repo, modifies a file, and checks diff output."""
        import subprocess

        subprocess.run(
            ["git", "init"], cwd=str(tmp_path), capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=str(tmp_path), capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=str(tmp_path), capture_output=True,
        )

        test_file = tmp_path / "code.py"
        test_file.write_text("x = 1\n")
        subprocess.run(
            ["git", "add", "code.py"], cwd=str(tmp_path), capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )
        test_file.write_text("x = 42\n")

        diff = _extract_diff(tmp_path)
        assert "diff --git" in diff
        assert "-x = 1" in diff
        assert "+x = 42" in diff
