"""Tests for SWETrajectoryDataset."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.datasets.swe_trajectory_dataset import SWETrajectoryDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_trajectory_jsonl(path, records: list[dict], suffix: str = "_filtered.jsonl"):
    """Write records to a JSONL file in the given directory."""
    path.mkdir(parents=True, exist_ok=True)
    fpath = path / f"trajectories{suffix}"
    with fpath.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path


def _make_record(
    instance_id: str = "test__1",
    resolved: bool = True,
    trajectory: list[dict] | None = None,
    model_patch: str = "diff --git a/f.py b/f.py\n+fix",
    repo: str = "owner/repo",
    base_commit: str = "abc123",
    issue: str = "Fix the bug in module X",
) -> dict:
    if trajectory is None:
        trajectory = [
            {"role": "user", "content": issue},
            {"role": "assistant", "content": "I will read the file first."},
            {"role": "tool", "content": "file contents"},
        ]
    return {
        "instance_id": instance_id,
        "resolved": resolved,
        "filtered_trajectory": trajectory,
        "model_patch": model_patch,
        "repo": repo,
        "base_commit": base_commit,
        "problem_statement": issue,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSWETrajectoryDatasetLoading:
    def test_loads_from_filtered_jsonl(self, tmp_path):
        records = [_make_record("t1"), _make_record("t2")]
        _write_trajectory_jsonl(tmp_path, records)
        ds = SWETrajectoryDataset(str(tmp_path))
        assert len(ds) == 2

    def test_require_resolved_filters_unresolved(self, tmp_path):
        records = [
            _make_record("t1", resolved=True),
            _make_record("t2", resolved=False),
        ]
        _write_trajectory_jsonl(tmp_path, records)
        ds = SWETrajectoryDataset(str(tmp_path), require_resolved=True)
        assert len(ds) == 1

    def test_require_resolved_false_includes_all(self, tmp_path):
        records = [
            _make_record("t1", resolved=True),
            _make_record("t2", resolved=False),
        ]
        _write_trajectory_jsonl(tmp_path, records)
        ds = SWETrajectoryDataset(str(tmp_path), require_resolved=False)
        assert len(ds) == 2

    def test_fallback_to_non_filtered_files(self, tmp_path):
        records = [_make_record("t1")]
        _write_trajectory_jsonl(tmp_path, records, suffix=".jsonl")
        ds = SWETrajectoryDataset(str(tmp_path))
        assert len(ds) == 1

    def test_empty_directory(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        ds = SWETrajectoryDataset(str(tmp_path))
        assert len(ds) == 0


class TestSWETrajectoryDatasetGetitem:
    def test_getitem_returns_expected_keys(self, tmp_path):
        _write_trajectory_jsonl(tmp_path, [_make_record()])
        ds = SWETrajectoryDataset(str(tmp_path))
        item = ds[0]
        assert "issue_text" in item
        assert "trajectory_text" in item
        assert "model_patch" in item
        assert "instance_id" in item
        assert "repo" in item
        assert "base_commit" in item

    def test_issue_text_from_problem_statement(self, tmp_path):
        _write_trajectory_jsonl(
            tmp_path, [_make_record(issue="Fix critical bug")],
        )
        ds = SWETrajectoryDataset(str(tmp_path))
        assert ds[0]["issue_text"] == "Fix critical bug"

    def test_trajectory_text_contains_roles(self, tmp_path):
        trajectory = [
            {"role": "user", "content": "Please fix it"},
            {"role": "assistant", "content": "I will read the file"},
        ]
        _write_trajectory_jsonl(
            tmp_path, [_make_record(trajectory=trajectory)],
        )
        ds = SWETrajectoryDataset(str(tmp_path))
        text = ds[0]["trajectory_text"]
        assert "[user]" in text
        assert "[assistant]" in text
        assert "Please fix it" in text


class TestFormatTrajectory:
    def test_dict_messages(self, tmp_path):
        _write_trajectory_jsonl(tmp_path, [_make_record()])
        ds = SWETrajectoryDataset(str(tmp_path))
        trajectory = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        text = ds._format_trajectory(trajectory)
        assert "[user]" in text
        assert "hello" in text
        assert "[assistant]" in text
        assert "world" in text

    def test_string_messages(self, tmp_path):
        _write_trajectory_jsonl(tmp_path, [_make_record()])
        ds = SWETrajectoryDataset(str(tmp_path))
        trajectory = ["plain string message", "another string"]
        text = ds._format_trajectory(trajectory)
        assert "plain string message" in text
        assert "another string" in text

    def test_list_content_in_dict(self, tmp_path):
        _write_trajectory_jsonl(tmp_path, [_make_record()])
        ds = SWETrajectoryDataset(str(tmp_path))
        trajectory = [{
            "role": "assistant",
            "content": [
                {"text": "part one"},
                {"text": "part two"},
            ],
        }]
        text = ds._format_trajectory(trajectory)
        assert "part one" in text
        assert "part two" in text


class TestSWETrajectoryTokenization:
    def test_tokenization_with_mock_tokenizer(self, tmp_path):
        _write_trajectory_jsonl(tmp_path, [_make_record()])

        class MockTokenizer:
            def encode(self, text, add_special_tokens=True):
                return list(range(min(len(text.split()), 100)))

        ds = SWETrajectoryDataset(
            str(tmp_path),
            tokenizer=MockTokenizer(),
            max_issue_tokens=10,
            max_trajectory_tokens=20,
        )
        item = ds[0]
        assert "issue_token_ids" in item
        assert "trajectory_token_ids" in item
        assert isinstance(item["issue_token_ids"], torch.Tensor)
        assert item["issue_token_ids"].dtype == torch.long
        assert len(item["issue_token_ids"]) <= 10
        assert len(item["trajectory_token_ids"]) <= 20
