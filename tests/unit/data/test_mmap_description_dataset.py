"""Tests for MmapDescriptionDataset (file-level 3-tuple key lookup)."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.datasets.description_dataset import MmapDescriptionDataset


class TestMmapDescriptionDataset:
    def test_basic_load_and_lookup(self, tmp_path, create_mmap_artifacts):
        d = tmp_path / "desc"
        d.mkdir()
        create_mmap_artifacts(
            d,
            token_lists=[[1, 2, 3, 4, 5], [10, 20, 30]],
            metadata_columns={
                "repo_path": ["owner/repo1", "owner/repo2"],
                "file_path": ["a.py", "b.py"],
                "commit_sha": ["abc123", "def456"],
            },
        )
        ds = MmapDescriptionDataset(str(d), max_seq_len=2048)
        assert len(ds) == 2

        assert ds.has_key("owner/repo1", "a.py", "abc123")
        assert not ds.has_key("owner/repo1", "b.py", "abc123")
        assert ds.lookup_index("owner/repo2", "b.py", "def456") == 1
        assert ds.lookup_index("nonexistent", "x", "y") is None

    def test_get_by_key(self, tmp_path, create_mmap_artifacts):
        d = tmp_path / "desc"
        d.mkdir()
        create_mmap_artifacts(
            d,
            token_lists=[[10, 20, 30]],
            metadata_columns={
                "repo_path": ["owner/repo"],
                "file_path": ["main.py"],
                "commit_sha": ["sha1"],
            },
        )
        ds = MmapDescriptionDataset(str(d))
        sample = ds.get_by_key("owner/repo", "main.py", "sha1")
        assert sample is not None
        assert sample["token_ids"].shape == (3,)

    def test_missing_key_returns_none(self, tmp_path, create_mmap_artifacts):
        d = tmp_path / "desc"
        d.mkdir()
        create_mmap_artifacts(
            d,
            token_lists=[[1, 2]],
            metadata_columns={
                "repo_path": ["owner/repo"],
                "file_path": ["a.py"],
                "commit_sha": ["abc"],
            },
        )
        ds = MmapDescriptionDataset(str(d))
        assert ds.get_by_key("nonexistent", "x.py", "zzz") is None

    def test_truncation(self, tmp_path, create_mmap_artifacts):
        d = tmp_path / "desc"
        d.mkdir()
        create_mmap_artifacts(
            d,
            token_lists=[list(range(100))],
            metadata_columns={
                "repo_path": ["owner/repo"],
                "file_path": ["a.py"],
                "commit_sha": ["abc"],
            },
        )
        ds = MmapDescriptionDataset(str(d), max_seq_len=10)
        sample = ds[0]
        assert sample["token_ids"].shape == (10,)

    def test_pickle_roundtrip(self, tmp_path, create_mmap_artifacts):
        d = tmp_path / "desc"
        d.mkdir()
        create_mmap_artifacts(
            d,
            token_lists=[[5, 10, 15]],
            metadata_columns={
                "repo_path": ["owner/repo"],
                "file_path": ["a.py"],
                "commit_sha": ["abc"],
            },
        )
        ds = MmapDescriptionDataset(str(d))
        data = pickle.dumps(ds)
        ds2 = pickle.loads(data)
        assert len(ds2) == 1
        sample = ds2.get_by_key("owner/repo", "a.py", "abc")
        assert sample is not None
        assert sample["token_ids"].shape == (3,)
