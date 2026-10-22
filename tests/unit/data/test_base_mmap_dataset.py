"""Tests for BaseMmapDataset and standalone validation functions."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bgkit.data.datasets.base_mmap_dataset import (
    BaseMmapDataset,
    check_required_files,
    load_and_validate_manifest,
    validate_manifest_counts,
)


class TestBaseMmapDataset:
    def test_init_and_len(self, tmp_path, create_mmap_artifacts):
        d = tmp_path / "data"
        d.mkdir()
        create_mmap_artifacts(d, [[1, 2, 3], [4, 5]])
        ds = BaseMmapDataset(str(d))
        assert len(ds) == 2

    def test_getitem(self, tmp_path, create_mmap_artifacts):
        d = tmp_path / "data"
        d.mkdir()
        create_mmap_artifacts(d, [[10, 20, 30]])
        ds = BaseMmapDataset(str(d))
        sample = ds[0]
        assert torch.equal(sample["token_ids"], torch.tensor([10, 20, 30], dtype=torch.int64))

    def test_lengths_truncated(self, tmp_path, create_mmap_artifacts):
        d = tmp_path / "data"
        d.mkdir()
        create_mmap_artifacts(d, [list(range(20)), list(range(3))])
        ds = BaseMmapDataset(str(d), max_seq_len=10)
        np.testing.assert_array_equal(ds.lengths, [10, 3])

    def test_getitem_truncated(self, tmp_path, create_mmap_artifacts):
        d = tmp_path / "data"
        d.mkdir()
        create_mmap_artifacts(d, [list(range(100))])
        ds = BaseMmapDataset(str(d), max_seq_len=5)
        sample = ds[0]
        assert sample["token_ids"].shape == (5,)
        assert torch.equal(sample["token_ids"], torch.arange(5, dtype=torch.int64))

    def test_zero_length_filtered(self, tmp_path, create_mmap_artifacts):
        d = tmp_path / "data"
        d.mkdir()
        create_mmap_artifacts(d, [[], [1, 2, 3], []])
        ds = BaseMmapDataset(str(d))
        assert len(ds) == 1
        assert torch.equal(ds[0]["token_ids"], torch.tensor([1, 2, 3], dtype=torch.int64))

    def test_pickle_roundtrip(self, tmp_path, create_mmap_artifacts):
        d = tmp_path / "data"
        d.mkdir()
        create_mmap_artifacts(d, [[1, 2], [3, 4, 5]])
        ds = BaseMmapDataset(str(d))
        data = pickle.dumps(ds)
        assert len(data) < 100_000  # no mmap data in pickle

        ds2 = pickle.loads(data)
        assert len(ds2) == 2
        assert torch.equal(ds2[0]["token_ids"], ds[0]["token_ids"])

    def test_missing_files_raises(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(FileNotFoundError, match="Missing mmap artifacts"):
            BaseMmapDataset(str(d))

    def test_bad_schema_version(self, tmp_path):
        d = tmp_path / "bad"
        d.mkdir()
        np.save(d / "tokens.npy", np.array([1], dtype=np.int32))
        np.save(d / "offsets.npy", np.array([0, 1], dtype=np.int64))
        (d / "manifest.json").write_text(json.dumps({"schema_version": 99}))
        with pytest.raises(ValueError, match="Unsupported manifest schema version"):
            BaseMmapDataset(str(d))

    def test_manifest_row_count_mismatch(self, tmp_path):
        d = tmp_path / "stale"
        d.mkdir()
        np.save(d / "tokens.npy", np.array([1, 2], dtype=np.int32))
        np.save(d / "offsets.npy", np.array([0, 2], dtype=np.int64))
        (d / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "row_count": 99})
        )
        with pytest.raises(ValueError, match="Manifest row_count"):
            BaseMmapDataset(str(d))

    def test_manifest_total_tokens_mismatch(self, tmp_path):
        d = tmp_path / "stale"
        d.mkdir()
        np.save(d / "tokens.npy", np.array([1, 2], dtype=np.int32))
        np.save(d / "offsets.npy", np.array([0, 2], dtype=np.int64))
        (d / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "row_count": 1, "total_tokens": 999})
        )
        with pytest.raises(ValueError, match="Manifest total_tokens"):
            BaseMmapDataset(str(d))


class TestStandaloneFunctions:
    def test_check_required_files_passes(self, tmp_path):
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.txt").touch()
        check_required_files(tmp_path, ["a.txt", "b.txt"], "hint")

    def test_check_required_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Missing mmap artifacts"):
            check_required_files(tmp_path, ["missing.npy"], "hint")

    def test_load_and_validate_manifest(self, tmp_path):
        (tmp_path / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "row_count": 5})
        )
        m = load_and_validate_manifest(tmp_path)
        assert m["row_count"] == 5

    def test_load_and_validate_manifest_bad_version(self, tmp_path):
        (tmp_path / "manifest.json").write_text(
            json.dumps({"schema_version": 2})
        )
        with pytest.raises(ValueError, match="Unsupported"):
            load_and_validate_manifest(tmp_path)

    def test_validate_manifest_counts_passes(self):
        offsets = np.array([0, 3, 7], dtype=np.int64)
        tokens = np.zeros(7, dtype=np.int32)
        manifest = {"row_count": 2, "total_tokens": 7}
        validate_manifest_counts(manifest, offsets, tokens)  # should not raise

    def test_validate_manifest_counts_row_mismatch(self):
        offsets = np.array([0, 3], dtype=np.int64)
        tokens = np.zeros(3, dtype=np.int32)
        with pytest.raises(ValueError, match="Manifest row_count"):
            validate_manifest_counts({"row_count": 99}, offsets, tokens)
