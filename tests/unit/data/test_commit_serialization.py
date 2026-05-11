"""Tests for commit serialization."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bgkit.data.commit_extraction import ExtractedCommit
from bgkit.data.commit_serialization import serialize_and_tokenize_commit, serialize_commit


def _make_commit(**overrides) -> ExtractedCommit:
    defaults = {
        "repo_path": "/tmp/test-repo",
        "sha": "a" * 40,
        "parent_sha": "b" * 40,
        "message": "Fix bug in parser",
        "diff_paths": ["src/parser.py"],
        "diff_hunks": [["+    return True\n"]],
        "author": "dev",
        "timestamp": 1700000000,
        "additions": 1,
        "deletions": 0,
        "is_cross_file": False,
    }
    defaults.update(overrides)
    return ExtractedCommit(**defaults)


class TestSerializeCommit:
    def test_basic_format(self):
        commit = _make_commit()
        text = serialize_commit(commit)

        assert text.startswith("<commit>")
        assert text.endswith("</commit>")
        assert "<message>" in text
        assert "</message>" in text
        assert "<files>" in text
        assert "</files>" in text
        assert "<diff>" in text
        assert "</diff>" in text

    def test_message_included(self):
        commit = _make_commit(message="Add new feature")
        text = serialize_commit(commit)
        assert "Add new feature" in text

    def test_paths_included(self):
        commit = _make_commit(
            diff_paths=["src/a.py", "src/b.py"],
            diff_hunks=[["+line1\n"], ["+line2\n"]],
            is_cross_file=True,
        )
        text = serialize_commit(commit)
        assert "src/a.py" in text
        assert "src/b.py" in text

    def test_multi_file_hunk_grouping(self):
        commit = _make_commit(
            diff_paths=["file1.py", "file2.py"],
            diff_hunks=[
                ["+added to file1\n", "-removed from file1\n"],
                ["+added to file2\n"],
            ],
            is_cross_file=True,
        )
        text = serialize_commit(commit)

        # Hunks should appear under their respective file headers
        lines = text.split("\n")
        file1_idx = next(i for i, line in enumerate(lines) if line == "--- file1.py")
        file2_idx = next(i for i, line in enumerate(lines) if line == "--- file2.py")

        # file1 hunks appear between file1 header and file2 header
        between = "\n".join(lines[file1_idx + 1 : file2_idx])
        assert "+added to file1" in between
        assert "-removed from file1" in between

        # file2 hunks appear after file2 header
        after = "\n".join(lines[file2_idx + 1 :])
        assert "+added to file2" in after

    def test_empty_hunks(self):
        commit = _make_commit(
            diff_paths=["renamed.py"],
            diff_hunks=[[]],  # file renamed with no content changes
        )
        text = serialize_commit(commit)
        assert "--- renamed.py" in text

    def test_roundtrip_structure(self):
        """Verify the document structure is consistent."""
        commit = _make_commit(
            diff_paths=["a.py", "b.py"],
            diff_hunks=[["+x\n"], ["-y\n"]],
            is_cross_file=True,
        )
        text = serialize_commit(commit)
        # Each section should appear exactly once
        assert text.count("<commit>") == 1
        assert text.count("</commit>") == 1
        assert text.count("<message>") == 1
        assert text.count("</message>") == 1
        assert text.count("<files>") == 1
        assert text.count("</files>") == 1
        assert text.count("<diff>") == 1
        assert text.count("</diff>") == 1


def _make_mock_tokenizer():
    """Create a mock tokenizer that assigns one token per word."""
    tok = MagicMock()
    tok.encode.side_effect = lambda text, **kw: list(range(len(text.split())))
    tok.decode.side_effect = lambda ids: " ".join(f"t{i}" for i in ids)
    return tok


class TestSerializeAndTokenize:
    @pytest.fixture()
    def tokenizer(self):
        return _make_mock_tokenizer()

    def test_returns_tokens_under_limit(self, tokenizer):
        commit = _make_commit()
        result = serialize_and_tokenize_commit(commit, tokenizer, max_tokens=4096)
        assert result is not None
        assert isinstance(result, list)
        assert all(isinstance(t, int) for t in result)

    def test_returns_none_over_limit(self, tokenizer):
        # Create a commit with a very long message that will produce many tokens
        commit = _make_commit(message="x " * 5000)
        result = serialize_and_tokenize_commit(commit, tokenizer, max_tokens=100)
        assert result is None

    def test_token_count_matches_encode(self, tokenizer):
        commit = _make_commit(message="Refactor database layer")
        token_ids = serialize_and_tokenize_commit(commit, tokenizer, max_tokens=4096)
        assert token_ids is not None

        # Verify encode was called with the serialized text
        text = serialize_commit(commit)
        expected = tokenizer.encode(text)
        assert token_ids == expected

    def test_skips_pathological_long_line_before_tokenization(self, tokenizer):
        commit = _make_commit(diff_hunks=[["+" + ("A" * 40000) + "\n"]])
        result = serialize_and_tokenize_commit(commit, tokenizer, max_tokens=4096)
        assert result is None
        tokenizer.encode.assert_not_called()

    def test_skips_base64ish_payload_before_tokenization(self, tokenizer):
        commit = _make_commit(diff_hunks=[["+" + ("A" * 9000) + "\n"]])
        result = serialize_and_tokenize_commit(commit, tokenizer, max_tokens=4096)
        assert result is None
        tokenizer.encode.assert_not_called()


class TestSerializerWithDivergentCounts:
    def test_mismatched_paths_and_hunks_raises(self):
        commit = _make_commit(
            diff_paths=["a.py", "b.py"],
            diff_hunks=[["+line\n"]],  # only 1 entry for 2 paths
        )
        with pytest.raises(ValueError):
            serialize_commit(commit)
