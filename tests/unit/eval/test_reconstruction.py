"""Tests for reconstruction metrics: code extraction and parse success rate."""

from __future__ import annotations

import pytest

from bgkit.eval.metrics.reconstruction import (
    extract_code_from_chat_response,
    parse_success_rate,
)


class TestExtractCodeFromChatResponse:
    def test_extracts_from_code_fence(self):
        text = '```python\nprint("hello")\n```'
        assert extract_code_from_chat_response(text) == 'print("hello")'

    def test_extracts_from_full_chat_response(self):
        text = (
            "<think>\n\n</think>\n\n"
            "Here are the contents of `foo.py`:\n\n"
            '```python\nx = 1\ny = 2\n```'
        )
        assert extract_code_from_chat_response(text) == "x = 1\ny = 2"

    def test_extracts_from_fence_without_language(self):
        text = "```\nsome code\n```"
        assert extract_code_from_chat_response(text) == "some code"

    def test_fallback_no_fence(self):
        """Returns full text when no code fence is found."""
        text = "just raw code\nx = 1"
        assert extract_code_from_chat_response(text) == text

    def test_multiline_code(self):
        text = "```python\ndef foo():\n    return 42\n\nfoo()\n```"
        assert extract_code_from_chat_response(text) == "def foo():\n    return 42\n\nfoo()"

    def test_ignores_text_outside_fence(self):
        text = "Some prefix text\n```python\nx = 1\n```\nSome suffix"
        assert extract_code_from_chat_response(text) == "x = 1"

    def test_last_fence_wins_with_inner_fences(self):
        """Content containing inner code fences: last fence is correct."""
        text = (
            "<think>\n\n</think>\n\n"
            "Here are the contents of `README.md`:\n\n"
            "```markdown\n"
            "# My Project\n\n"
            "Example:\n\n"
            "```python\n"
            "x = 1\n"
            "```\n\n"
            "More text\n"
            "```"
        )
        result = extract_code_from_chat_response(text)
        # The last fence should capture "More text" (the outermost closing)
        assert "More text" in result

    def test_multiple_fences_last_one_extracted(self):
        """Multiple fence blocks: last one is the generated content."""
        text = (
            "```python\nfirst_block = True\n```\n"
            "Some text between\n"
            "```python\nsecond_block = True\n```"
        )
        assert extract_code_from_chat_response(text) == "second_block = True"


class TestParseSuccessRate:
    def test_all_valid_python(self):
        code = ["x = 1", "def foo(): pass", "import os"]
        assert parse_success_rate(code) == 1.0

    def test_all_invalid(self):
        code = ["def :", "x =", "if"]
        assert parse_success_rate(code) == 0.0

    def test_mixed(self):
        code = ["x = 1", "def :", "y = 2", "if"]
        assert parse_success_rate(code) == 0.5

    def test_empty_list(self):
        assert parse_success_rate([]) == 0.0

    def test_chat_formatted_input(self):
        """Should handle chat-formatted decoder output with chat_formatted=True."""
        code = [
            "<think>\n\n</think>\n\nHere are the contents of `foo.py`:\n\n"
            "```python\nx = 1\n```",
            "<think>\n\n</think>\n\nHere are the contents of `bar.py`:\n\n"
            "```python\ndef :\n```",
        ]
        assert parse_success_rate(code, chat_formatted=True) == 0.5

    def test_raw_code_not_stripped(self):
        """Raw code with backticks should NOT be stripped by default."""
        # Markdown file containing inner fences — should be parsed as-is
        code = ["# Title\n\n```python\nx = 1\n```\n\nMore text"]
        # Without chat_formatted, the full string is passed to the parser
        # This is valid Markdown content, not chat output
        result = parse_success_rate(code, language="Markdown")
        # We just verify it doesn't crash; actual parse result depends on grammar
        assert isinstance(result, float)

    def test_per_sample_languages_python(self):
        """Per-sample language labels route to correct parser."""
        code = ["x = 1", "y = 2"]
        languages = ["Python", "Python"]
        assert parse_success_rate(code, languages=languages) == 1.0

    def test_unparseable_language_excluded(self):
        """Languages without parsers are excluded from denominator."""
        code = ["x = 1", "some nim code"]
        languages = ["Python", "Nim"]
        # Only Python sample evaluated (1/1 = 1.0), Nim excluded
        assert parse_success_rate(code, languages=languages) == 1.0

    def test_all_unparseable_returns_zero(self):
        """If all samples are unparseable languages, return 0.0."""
        code = ["some code"]
        languages = ["Nim"]
        assert parse_success_rate(code, languages=languages) == 0.0

    def test_lowercase_python_with_languages(self):
        """Lowercase 'python' should match Python parser when passed via languages."""
        code = ["x = 1"]
        languages = ["python"]
        assert parse_success_rate(code, languages=languages) == 1.0

    def test_lowercase_language_labels(self):
        """Lowercase language labels should match title-cased lookup table."""
        code = ["x = 1", "some nim code"]
        languages = ["python", "nim"]
        # Python parses, Nim excluded from denominator -> 1.0
        assert parse_success_rate(code, languages=languages) == 1.0

    def test_languages_length_mismatch_raises(self):
        """Mismatched languages/code lengths should raise ValueError."""
        with pytest.raises(ValueError, match="languages length"):
            parse_success_rate(["x = 1"], languages=["Python", "Go"])


class TestParseSuccessRateTreeSitter:
    """Tests for tree-sitter based parsing of non-Python languages."""

    ts = pytest.importorskip("tree_sitter_language_pack")

    def test_valid_javascript(self):
        code = ["function foo() { return 1; }"]
        languages = ["JavaScript"]
        assert parse_success_rate(code, languages=languages) == 1.0

    def test_invalid_javascript(self):
        code = ["function { { { !!!"]
        languages = ["JavaScript"]
        assert parse_success_rate(code, languages=languages) == 0.0

    def test_valid_go(self):
        code = ['package main\n\nfunc main() {\n\tprintln("hello")\n}']
        languages = ["Go"]
        assert parse_success_rate(code, languages=languages) == 1.0

    def test_mixed_languages(self):
        """Mixed Python + JS with per-sample languages."""
        code = [
            "x = 1",  # valid Python
            "function foo() { return 1; }",  # valid JS
            "def :",  # invalid Python
        ]
        languages = ["Python", "JavaScript", "Python"]
        # 2 out of 3 parse successfully
        assert abs(parse_success_rate(code, languages=languages) - 2.0 / 3.0) < 1e-6

    def test_valid_json(self):
        code = ['{"key": "value", "num": 42}']
        languages = ["JSON"]
        assert parse_success_rate(code, languages=languages) == 1.0
