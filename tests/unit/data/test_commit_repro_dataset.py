"""Tests for CommitReproDataset (mmap-based)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bgkit.data.datasets.commit_repro_dataset import CommitReproDataset


@pytest.fixture
def commit_data_dir(tmp_path: Path, create_mmap_artifacts) -> Path:
    """Create a small mmap commit dataset."""
    d = tmp_path / "commits"
    d.mkdir()
    create_mmap_artifacts(
        d,
        token_lists=[
            list(range(10)),        # commit 0: 10 tokens
            list(range(20, 25)),    # commit 1: 5 tokens
            list(range(100, 103)),  # commit 2: 3 tokens
        ],
    )
    return d


class TestCommitReproDataset:
    def test_len(self, tmp_path: Path, create_mmap_artifacts):
        d = tmp_path / "data"
        d.mkdir()
        create_mmap_artifacts(d, token_lists=[[1, 2, 3], [4, 5, 6, 7, 8], [10, 20]])
        ds = CommitReproDataset(str(d))
        assert len(ds) == 3

    def test_getitem(self, commit_data_dir: Path):
        ds = CommitReproDataset(str(commit_data_dir))
        sample = ds[0]

        assert "token_ids" in sample
        assert isinstance(sample["token_ids"], torch.Tensor)
        assert sample["token_ids"].dtype == torch.int64
        assert len(sample["token_ids"]) == 10

    def test_missing_files(self, tmp_path: Path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(FileNotFoundError, match="Missing mmap artifacts"):
            CommitReproDataset(str(d))

    def test_all_items_accessible(self, commit_data_dir: Path):
        ds = CommitReproDataset(str(commit_data_dir))
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["token_ids"].ndim == 1
            assert sample["token_ids"].shape[0] > 0

    def test_max_seq_len_truncates(self, tmp_path: Path, create_mmap_artifacts):
        d = tmp_path / "trunc"
        d.mkdir()
        create_mmap_artifacts(d, token_lists=[list(range(100))])
        ds = CommitReproDataset(str(d), max_seq_len=10)
        sample = ds[0]
        assert sample["token_ids"].shape[0] == 10
        assert torch.equal(sample["token_ids"], torch.arange(10, dtype=torch.int64))

    def test_multi_worker_dataloader(self, commit_data_dir: Path):
        """Verify dataset works with multi-worker DataLoader."""
        ds = CommitReproDataset(str(commit_data_dir))
        loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=2)
        samples = list(loader)
        assert len(samples) == 3

    def test_zero_length_commits_skipped(self, tmp_path: Path, create_mmap_artifacts):
        d = tmp_path / "zeros"
        d.mkdir()
        create_mmap_artifacts(d, token_lists=[[], [1, 2, 3], []])
        ds = CommitReproDataset(str(d))
        assert len(ds) == 1
        assert torch.equal(ds[0]["token_ids"], torch.tensor([1, 2, 3], dtype=torch.int64))

    def test_pickle_roundtrip(self, commit_data_dir: Path):
        ds = CommitReproDataset(str(commit_data_dir))
        data = pickle.dumps(ds)
        # Pickle should be small (no mmap data)
        assert len(data) < 100_000

        ds2 = pickle.loads(data)
        assert len(ds2) == len(ds)
        assert torch.equal(ds2[0]["token_ids"], ds[0]["token_ids"])

    def test_manifest_validation(self, tmp_path: Path):
        d = tmp_path / "bad_manifest"
        d.mkdir()
        np.save(d / "tokens.npy", np.array([1], dtype=np.int32))
        np.save(d / "offsets.npy", np.array([0, 1], dtype=np.int64))
        (d / "manifest.json").write_text(json.dumps({"schema_version": 99}))
        with pytest.raises(ValueError, match="Unsupported manifest schema version"):
            CommitReproDataset(str(d))

    def test_manifest_row_count_validation(self, tmp_path: Path):
        d = tmp_path / "stale"
        d.mkdir()
        np.save(d / "tokens.npy", np.array([1, 2, 3], dtype=np.int32))
        np.save(d / "offsets.npy", np.array([0, 3], dtype=np.int64))
        (d / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "row_count": 99})
        )
        with pytest.raises(ValueError, match="Manifest row_count"):
            CommitReproDataset(str(d))

    def test_manifest_total_tokens_validation(self, tmp_path: Path):
        d = tmp_path / "stale_tokens"
        d.mkdir()
        np.save(d / "tokens.npy", np.array([1, 2, 3], dtype=np.int32))
        np.save(d / "offsets.npy", np.array([0, 3], dtype=np.int64))
        (d / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "row_count": 1, "total_tokens": 999})
        )
        with pytest.raises(ValueError, match="Manifest total_tokens"):
            CommitReproDataset(str(d))

    def test_lengths_property(self, tmp_path: Path, create_mmap_artifacts):
        d = tmp_path / "lengths"
        d.mkdir()
        create_mmap_artifacts(
            d,
            token_lists=[
                list(range(20)),   # 20 tokens, truncated to 10
                list(range(5)),    # 5 tokens, not truncated
                [],                # 0 tokens, skipped
                list(range(3)),    # 3 tokens, not truncated
            ],
        )
        ds = CommitReproDataset(str(d), max_seq_len=10)
        np.testing.assert_array_equal(ds.lengths, [10, 5, 3])


class TestGoldenOutput:
    """Golden-output test: exact expected values for known commit sequences."""

    @pytest.fixture
    def golden_dir(self, tmp_path: Path, create_mmap_artifacts) -> Path:
        d = tmp_path / "golden"
        d.mkdir()
        create_mmap_artifacts(
            d,
            token_lists=[
                [10, 20, 30, 40, 50],  # commit 0: 5 tokens
                [100, 200, 300],        # commit 1: 3 tokens
                [],                     # commit 2: empty, skipped
                [999],                  # commit 3: 1 token
            ],
        )
        return d

    def test_golden_values(self, golden_dir: Path):
        ds = CommitReproDataset(str(golden_dir), max_seq_len=4096)

        # 3 valid commits (empty one skipped)
        assert len(ds) == 3
        np.testing.assert_array_equal(ds.lengths, [5, 3, 1])

        expected = [
            [10, 20, 30, 40, 50],
            [100, 200, 300],
            [999],
        ]

        for i, exp_tokens in enumerate(expected):
            sample = ds[i]
            assert torch.equal(
                sample["token_ids"],
                torch.tensor(exp_tokens, dtype=torch.int64),
            ), f"Token mismatch at commit {i}"
