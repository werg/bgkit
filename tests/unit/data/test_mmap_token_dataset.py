"""Tests for MmapTokenDataset."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from bgkit.data.datasets.mmap_token_dataset import MmapTokenDataset


def _create_mmap_data(
    data_dir: Path,
    file_token_ids: list[list[int]],
    file_paths: list[str],
    languages: list[str],
) -> None:
    """Helper to write mmap artifacts from known data."""
    all_tokens = []
    offsets = [0]
    for tids in file_token_ids:
        all_tokens.extend(tids)
        offsets.append(len(all_tokens))

    np.save(data_dir / "tokens.npy", np.array(all_tokens, dtype=np.int32))
    np.save(data_dir / "offsets.npy", np.array(offsets, dtype=np.int64))

    meta = pa.table({
        "file_path": pa.array(file_paths, type=pa.string()),
        "language": pa.array(languages, type=pa.string()),
    })
    pq.write_table(meta, data_dir / "metadata.parquet")

    manifest = {
        "schema_version": 1,
        "row_count": len(file_token_ids),
        "total_tokens": len(all_tokens),
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest))


@pytest.fixture
def token_data_dir(tmp_path: Path) -> Path:
    """Create a small mmap token dataset."""
    d = tmp_path / "tokens"
    d.mkdir()
    _create_mmap_data(
        d,
        file_token_ids=[
            list(range(10)),       # 10 tokens
            list(range(20, 25)),   # 5 tokens
            list(range(100, 103)), # 3 tokens
        ],
        file_paths=["a.py", "b.js", "c.rs"],
        languages=["python", "javascript", "rust"],
    )
    return d


class TestMmapTokenDataset:
    def test_len(self, token_data_dir: Path):
        ds = MmapTokenDataset(str(token_data_dir), max_seq_len=8192)
        # 3 files, all fit in one chunk
        assert len(ds) == 3

    def test_getitem_returns_expected_keys(self, token_data_dir: Path):
        ds = MmapTokenDataset(str(token_data_dir), max_seq_len=8192)
        sample = ds[0]
        assert set(sample.keys()) == {"token_ids", "file_path", "language"}

    def test_getitem_token_ids(self, token_data_dir: Path):
        ds = MmapTokenDataset(str(token_data_dir), max_seq_len=8192)
        sample = ds[0]
        assert sample["token_ids"].dtype == torch.int64
        assert torch.equal(sample["token_ids"], torch.arange(10, dtype=torch.int64))

    def test_getitem_metadata(self, token_data_dir: Path):
        ds = MmapTokenDataset(str(token_data_dir), max_seq_len=8192)
        assert ds[0]["file_path"] == "a.py"
        assert ds[0]["language"] == "python"
        assert ds[1]["file_path"] == "b.js"
        assert ds[2]["language"] == "rust"

    def test_getitem_without_metadata(self, token_data_dir: Path):
        ds = MmapTokenDataset(str(token_data_dir), max_seq_len=8192, include_metadata=False)
        sample = ds[0]
        assert set(sample.keys()) == {"token_ids"}
        assert sample["token_ids"].dtype == torch.int64

    def test_lengths(self, token_data_dir: Path):
        ds = MmapTokenDataset(str(token_data_dir), max_seq_len=8192)
        np.testing.assert_array_equal(ds.lengths, [10, 5, 3])

    def test_chunking(self, tmp_path: Path):
        """Files longer than max_seq_len should be split into chunks."""
        d = tmp_path / "chunked"
        d.mkdir()
        _create_mmap_data(
            d,
            file_token_ids=[list(range(25))],
            file_paths=["long.py"],
            languages=["python"],
        )
        ds = MmapTokenDataset(str(d), max_seq_len=10)
        assert len(ds) == 3  # 10 + 10 + 5
        np.testing.assert_array_equal(ds.lengths, [10, 10, 5])

        # Verify chunk contents
        assert torch.equal(ds[0]["token_ids"], torch.arange(0, 10, dtype=torch.int64))
        assert torch.equal(ds[1]["token_ids"], torch.arange(10, 20, dtype=torch.int64))
        assert torch.equal(ds[2]["token_ids"], torch.arange(20, 25, dtype=torch.int64))

        # All chunks share the same file metadata
        for i in range(3):
            assert ds[i]["file_path"] == "long.py"
            assert ds[i]["language"] == "python"

    def test_zero_length_file_skipped(self, tmp_path: Path):
        """Files with 0 tokens should produce no chunks."""
        d = tmp_path / "zero"
        d.mkdir()
        _create_mmap_data(
            d,
            file_token_ids=[[], [1, 2, 3], []],
            file_paths=["empty1.py", "real.py", "empty2.py"],
            languages=["python", "python", "python"],
        )
        ds = MmapTokenDataset(str(d), max_seq_len=8192)
        assert len(ds) == 1
        assert ds[0]["file_path"] == "real.py"
        assert torch.equal(ds[0]["token_ids"], torch.tensor([1, 2, 3], dtype=torch.int64))

    def test_missing_files_error(self, tmp_path: Path):
        d = tmp_path / "missing"
        d.mkdir()
        with pytest.raises(FileNotFoundError, match="Missing mmap artifacts"):
            MmapTokenDataset(str(d))

    def test_missing_metadata_allowed_when_disabled(self, tmp_path: Path):
        d = tmp_path / "no_meta"
        d.mkdir()
        np.save(d / "tokens.npy", np.array([1, 2, 3], dtype=np.int32))
        np.save(d / "offsets.npy", np.array([0, 3], dtype=np.int64))
        (d / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "row_count": 1, "total_tokens": 3})
        )
        ds = MmapTokenDataset(str(d), include_metadata=False)
        assert len(ds) == 1
        assert set(ds[0].keys()) == {"token_ids"}

    def test_invalid_manifest(self, tmp_path: Path):
        d = tmp_path / "bad_manifest"
        d.mkdir()
        np.save(d / "tokens.npy", np.array([1], dtype=np.int32))
        np.save(d / "offsets.npy", np.array([0, 1], dtype=np.int64))
        pq.write_table(
            pa.table({"file_path": ["x"], "language": ["py"]}),
            d / "metadata.parquet",
        )
        (d / "manifest.json").write_text(json.dumps({"schema_version": 99}))
        with pytest.raises(ValueError, match="Unsupported manifest schema version"):
            MmapTokenDataset(str(d))

    def test_manifest_row_count_validation(self, tmp_path: Path):
        """Stale manifest row_count should raise ValueError."""
        d = tmp_path / "stale"
        d.mkdir()
        np.save(d / "tokens.npy", np.array([1, 2, 3], dtype=np.int32))
        np.save(d / "offsets.npy", np.array([0, 3], dtype=np.int64))
        pq.write_table(
            pa.table({"file_path": ["x"], "language": ["py"]}),
            d / "metadata.parquet",
        )
        (d / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "row_count": 99})
        )
        with pytest.raises(ValueError, match="Manifest row_count"):
            MmapTokenDataset(str(d))

    def test_manifest_total_tokens_validation(self, tmp_path: Path):
        """Stale manifest total_tokens should raise ValueError."""
        d = tmp_path / "stale_tokens"
        d.mkdir()
        np.save(d / "tokens.npy", np.array([1, 2, 3], dtype=np.int32))
        np.save(d / "offsets.npy", np.array([0, 3], dtype=np.int64))
        pq.write_table(
            pa.table({"file_path": ["x"], "language": ["py"]}),
            d / "metadata.parquet",
        )
        (d / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "row_count": 1, "total_tokens": 999})
        )
        with pytest.raises(ValueError, match="Manifest total_tokens"):
            MmapTokenDataset(str(d))

    def test_pickle_roundtrip(self, token_data_dir: Path):
        ds = MmapTokenDataset(str(token_data_dir), max_seq_len=8192)
        data = pickle.dumps(ds)
        # Pickle should be small (no mmap data)
        assert len(data) < 100_000

        ds2 = pickle.loads(data)
        assert len(ds2) == len(ds)
        sample = ds2[0]
        assert torch.equal(sample["token_ids"], ds[0]["token_ids"])

    def test_multi_worker_dataloader(self, token_data_dir: Path):
        """Verify dataset works with multi-worker DataLoader."""
        ds = MmapTokenDataset(str(token_data_dir), max_seq_len=8192)
        loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=2)
        samples = list(loader)
        assert len(samples) == 3

    def test_all_items_accessible(self, token_data_dir: Path):
        ds = MmapTokenDataset(str(token_data_dir), max_seq_len=8192)
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["token_ids"].ndim == 1


class TestGoldenOutput:
    """Golden-output fixture test: exact expected values for a known dataset."""

    @pytest.fixture
    def golden_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "golden"
        d.mkdir()
        _create_mmap_data(
            d,
            file_token_ids=[
                [10, 20, 30, 40, 50],       # file 0: 5 tokens, 1 chunk
                [100, 200, 300],             # file 1: 3 tokens, 1 chunk
                [1, 2, 3, 4, 5, 6, 7, 8],   # file 2: 8 tokens, 2 chunks @ max_seq_len=5
                [],                          # file 3: empty, skipped
                [999],                       # file 4: 1 token, 1 chunk
            ],
            file_paths=["f0.py", "f1.js", "f2.rs", "f3.go", "f4.c"],
            languages=["python", "javascript", "rust", "go", "c"],
        )
        return d

    def test_golden_values(self, golden_dir: Path):
        ds = MmapTokenDataset(str(golden_dir), max_seq_len=5)

        # Expected chunks:
        # chunk 0: file 0, [10,20,30,40,50], len=5
        # chunk 1: file 1, [100,200,300], len=3
        # chunk 2: file 2, [1,2,3,4,5], len=5
        # chunk 3: file 2, [6,7,8], len=3
        # chunk 4: file 4, [999], len=1
        assert len(ds) == 5
        np.testing.assert_array_equal(ds.lengths, [5, 3, 5, 3, 1])

        expected = [
            {"token_ids": [10, 20, 30, 40, 50], "file_path": "f0.py", "language": "python"},
            {"token_ids": [100, 200, 300], "file_path": "f1.js", "language": "javascript"},
            {"token_ids": [1, 2, 3, 4, 5], "file_path": "f2.rs", "language": "rust"},
            {"token_ids": [6, 7, 8], "file_path": "f2.rs", "language": "rust"},
            {"token_ids": [999], "file_path": "f4.c", "language": "c"},
        ]

        for i, exp in enumerate(expected):
            sample = ds[i]
            assert torch.equal(
                sample["token_ids"],
                torch.tensor(exp["token_ids"], dtype=torch.int64),
            ), f"Token mismatch at chunk {i}"
            assert sample["file_path"] == exp["file_path"], f"file_path mismatch at chunk {i}"
            assert sample["language"] == exp["language"], f"language mismatch at chunk {i}"
