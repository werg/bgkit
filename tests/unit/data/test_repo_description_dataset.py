"""Tests for MmapRepoDescriptionDataset."""
from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from bgkit.data.datasets.description_dataset import MmapRepoDescriptionDataset


def _create_repo_desc_artifacts(
    tmpdir,
    repo_paths: list[str],
    commit_shas: list[str],
    token_lengths: list[int],
):
    """Create synthetic mmap artifacts for repo-scope descriptions."""
    n = len(repo_paths)
    assert len(commit_shas) == n
    assert len(token_lengths) == n

    # Build tokens and offsets
    offsets = [0]
    all_tokens = []
    for length in token_lengths:
        tokens = list(range(1, length + 1))
        all_tokens.extend(tokens)
        offsets.append(offsets[-1] + length)

    np.save(tmpdir / "tokens.npy", np.array(all_tokens, dtype=np.int32))
    np.save(tmpdir / "offsets.npy", np.array(offsets, dtype=np.int64))

    manifest = {
        "schema_version": 1,
        "row_count": n,
        "total_tokens": len(all_tokens),
    }
    (tmpdir / "manifest.json").write_text(json.dumps(manifest))

    table = pa.table({
        "repo_path": repo_paths,
        "commit_sha": commit_shas,
    })
    pq.write_table(table, tmpdir / "metadata.parquet")


class TestMmapRepoDescriptionDataset:
    def test_basic_load_and_lookup(self, tmp_path):
        """Synthetic repo-scope mmap, verify 2-tuple key lookup."""
        _create_repo_desc_artifacts(
            tmp_path,
            repo_paths=["owner/repo1", "owner/repo2"],
            commit_shas=["abc123", "def456"],
            token_lengths=[10, 15],
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

    def test_missing_key_returns_none(self, tmp_path):
        """Absent key should return None."""
        _create_repo_desc_artifacts(
            tmp_path,
            repo_paths=["owner/repo1"],
            commit_shas=["abc123"],
            token_lengths=[10],
        )

        ds = MmapRepoDescriptionDataset(str(tmp_path), max_seq_len=2048)
        assert ds.get_by_key("nonexistent/repo", "xyz") is None

    def test_getitem(self, tmp_path):
        """Direct index access should return token_ids dict."""
        _create_repo_desc_artifacts(
            tmp_path,
            repo_paths=["owner/repo1"],
            commit_shas=["abc123"],
            token_lengths=[8],
        )

        ds = MmapRepoDescriptionDataset(str(tmp_path), max_seq_len=2048)
        sample = ds[0]
        assert "token_ids" in sample
        assert sample["token_ids"].shape == (8,)

    def test_max_seq_len_truncation(self, tmp_path):
        """Tokens longer than max_seq_len should be truncated."""
        _create_repo_desc_artifacts(
            tmp_path,
            repo_paths=["owner/repo1"],
            commit_shas=["abc123"],
            token_lengths=[100],
        )

        ds = MmapRepoDescriptionDataset(str(tmp_path), max_seq_len=50)
        sample = ds[0]
        assert sample["token_ids"].shape == (50,)

    def test_pickle_roundtrip(self, tmp_path):
        """Dataset should survive pickle (for DataLoader workers)."""
        import pickle

        _create_repo_desc_artifacts(
            tmp_path,
            repo_paths=["owner/repo1"],
            commit_shas=["abc123"],
            token_lengths=[10],
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
