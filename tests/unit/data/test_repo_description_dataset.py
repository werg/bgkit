"""Tests for MmapRepoDescriptionDataset."""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.datasets.description_dataset import MmapRepoDescriptionDataset


class TestMmapRepoDescriptionDataset:
    def test_basic_load_and_lookup(self, tmp_path, create_mmap_artifacts):
        """Synthetic repo-scope mmap, verify 2-tuple key lookup."""
        create_mmap_artifacts(
            tmp_path,
            token_lists=[list(range(1, 11)), list(range(1, 16))],
            metadata_columns={
                "repo_path": ["owner/repo1", "owner/repo2"],
                "commit_sha": ["abc123", "def456"],
            },
        )

        ds = MmapRepoDescriptionDataset(str(tmp_path), max_seq_len=2048)
        assert len(ds) == 2

        # Key lookup
        assert ds.has_key("owner/repo1", "abc123")
        assert ds.has_key("owner/repo2", "def456")
        assert not ds.has_key("owner/repo1", "def456")

        # Index lookup
        assert ds.lookup_index("owner/repo1", "abc123") == 0
        assert ds.lookup_index("owner/repo2", "def456") == 1
        assert ds.lookup_index("nonexistent", "xxx") is None

        # get_by_key
        sample = ds.get_by_key("owner/repo1", "abc123")
        assert sample is not None
        assert sample["token_ids"].shape == (10,)

        sample2 = ds.get_by_key("owner/repo2", "def456")
        assert sample2 is not None
        assert sample2["token_ids"].shape == (15,)

    def test_missing_key_returns_none(self, tmp_path, create_mmap_artifacts):
        """Absent key should return None."""
        create_mmap_artifacts(
            tmp_path,
            token_lists=[list(range(1, 11))],
            metadata_columns={
                "repo_path": ["owner/repo1"],
                "commit_sha": ["abc123"],
            },
        )

        ds = MmapRepoDescriptionDataset(str(tmp_path), max_seq_len=2048)
        assert ds.get_by_key("nonexistent/repo", "xyz") is None

    def test_getitem(self, tmp_path, create_mmap_artifacts):
        """Direct index access should return token_ids dict."""
        create_mmap_artifacts(
            tmp_path,
            token_lists=[list(range(1, 9))],
            metadata_columns={
                "repo_path": ["owner/repo1"],
                "commit_sha": ["abc123"],
            },
        )

        ds = MmapRepoDescriptionDataset(str(tmp_path), max_seq_len=2048)
        sample = ds[0]
        assert "token_ids" in sample
        assert sample["token_ids"].shape == (8,)

    def test_max_seq_len_truncation(self, tmp_path, create_mmap_artifacts):
        """Tokens longer than max_seq_len should be truncated."""
        create_mmap_artifacts(
            tmp_path,
            token_lists=[list(range(1, 101))],
            metadata_columns={
                "repo_path": ["owner/repo1"],
                "commit_sha": ["abc123"],
            },
        )

        ds = MmapRepoDescriptionDataset(str(tmp_path), max_seq_len=50)
        sample = ds[0]
        assert sample["token_ids"].shape == (50,)

    def test_pickle_roundtrip(self, tmp_path, create_mmap_artifacts):
        """Dataset should survive pickle (for DataLoader workers)."""
        import pickle

        create_mmap_artifacts(
            tmp_path,
            token_lists=[list(range(1, 11))],
            metadata_columns={
                "repo_path": ["owner/repo1"],
                "commit_sha": ["abc123"],
            },
        )

        ds = MmapRepoDescriptionDataset(str(tmp_path), max_seq_len=2048)
        state = pickle.dumps(ds)
        ds2 = pickle.loads(state)

        assert len(ds2) == 1
        sample = ds2.get_by_key("owner/repo1", "abc123")
        assert sample is not None
        assert sample["token_ids"].shape == (10,)

    def test_missing_files_raises(self, tmp_path):
        """Missing artifacts should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Missing mmap artifacts"):
            MmapRepoDescriptionDataset(str(tmp_path))
