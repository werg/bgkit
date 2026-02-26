"""Tests for description prompt builders in generate_descriptions.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts/ to path so we can import generate_descriptions
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from generate_descriptions import (
    MAX_CONTENT_CHARS,
    MAX_DESC_CHARS_IN_PROMPT,
    _truncate_desc,
    build_file_prompt,
    build_module_prompt,
    build_repo_prompt,
)


class TestBuildFilePrompt:
    def test_includes_content(self):
        prompt = build_file_prompt("src/main.py", "print('hello')", "Python")
        assert "print('hello')" in prompt

    def test_truncates_long_content(self):
        long_content = "x" * (MAX_CONTENT_CHARS + 1000)
        prompt = build_file_prompt("big.py", long_content, "Python")
        assert "... (truncated)" in prompt
        # Original content should be truncated to MAX_CONTENT_CHARS
        assert "x" * (MAX_CONTENT_CHARS + 1) not in prompt

    def test_includes_language(self):
        prompt = build_file_prompt("lib.rs", "fn main() {}", "Rust")
        assert "(Rust)" in prompt

    def test_no_language_annotation_when_none(self):
        prompt = build_file_prompt("README", "hello", None)
        assert "README\n" in prompt
        assert "()" not in prompt

    def test_asks_for_exports_and_deps(self):
        prompt = build_file_prompt("src/utils.py", "def foo(): pass", "Python")
        assert "Key exports" in prompt
        assert "Dependencies" in prompt
        assert "Purpose" in prompt
        assert "Role" in prompt

    def test_asks_for_na_on_missing(self):
        prompt = build_file_prompt("config.json", '{"key": "val"}', None)
        assert "N/A" in prompt


class TestBuildModulePrompt:
    def test_includes_file_descriptions(self):
        file_descs = [
            {"file_path": "src/a.py", "description": "Module A does X"},
            {"file_path": "src/b.py", "description": "Module B does Y"},
        ]
        prompt = build_module_prompt("src", file_descs)
        assert "src/a.py" in prompt
        assert "Module A does X" in prompt
        assert "src/b.py" in prompt

    def test_includes_skeleton(self):
        file_descs = [{"file_path": "src/a.py", "description": "desc"}]
        prompt = build_module_prompt("src", file_descs, skeleton_text="class Foo:\n  pass")
        assert "class Foo:" in prompt
        assert "Structural skeleton:" in prompt

    def test_no_skeleton_when_none(self):
        file_descs = [{"file_path": "src/a.py", "description": "desc"}]
        prompt = build_module_prompt("src", file_descs, skeleton_text=None)
        assert "Structural skeleton:" not in prompt

    def test_asks_for_structured_sections(self):
        file_descs = [{"file_path": "src/a.py", "description": "desc"}]
        prompt = build_module_prompt("src", file_descs)
        assert "Purpose" in prompt
        assert "Public API" in prompt
        assert "Internal structure" in prompt
        assert "External dependencies" in prompt

    def test_truncates_long_descriptions(self):
        long_desc = "A" * 300
        file_descs = [{"file_path": "src/a.py", "description": long_desc}]
        prompt = build_module_prompt("src", file_descs)
        assert long_desc not in prompt
        assert "A" * MAX_DESC_CHARS_IN_PROMPT + "..." in prompt


class TestBuildRepoPrompt:
    def test_includes_modules_and_files(self):
        module_descs = [
            {"module_path": "src/core", "description": "Core module"},
        ]
        file_descs = [
            {"file_path": "README.md", "description": "Project readme"},
        ]
        prompt = build_repo_prompt("owner/repo", module_descs, file_descs)
        assert "src/core" in prompt
        assert "Core module" in prompt
        assert "README.md" in prompt
        assert "Project readme" in prompt

    def test_asks_for_structured_sections(self):
        prompt = build_repo_prompt("owner/repo", [], [])
        assert "Purpose" in prompt
        assert "Architecture" in prompt
        assert "Technology stack" in prompt
        assert "Entry points" in prompt

    def test_handles_empty_descriptions(self):
        prompt = build_repo_prompt("owner/repo", [], [])
        assert "Modules:" not in prompt
        assert "Key files:" not in prompt

    def test_truncates_long_descriptions(self):
        long_desc = "B" * 300
        file_descs = [{"file_path": "main.py", "description": long_desc}]
        prompt = build_repo_prompt("owner/repo", [], file_descs)
        assert long_desc not in prompt
        assert "B" * MAX_DESC_CHARS_IN_PROMPT + "..." in prompt


class TestTruncateDesc:
    def test_short_desc_unchanged(self):
        assert _truncate_desc("short") == "short"

    def test_long_desc_truncated(self):
        long = "x" * 300
        result = _truncate_desc(long)
        assert len(result) == MAX_DESC_CHARS_IN_PROMPT + 3  # +3 for "..."
        assert result.endswith("...")

    def test_exact_limit_unchanged(self):
        exact = "y" * MAX_DESC_CHARS_IN_PROMPT
        assert _truncate_desc(exact) == exact
