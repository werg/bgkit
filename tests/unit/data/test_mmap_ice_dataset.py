"""Tests for MmapICEDataset."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from bgkit.data.datasets.mmap_ice_dataset import MmapICEDataset


def _create_ice_data(
    data_dir: Path,
    file_token_ids: list[list[int]],
    file_ce_values: list[list[float]],
) -> None:
    """Helper to write mmap ICE artifacts from known data."""
    all_tokens = []
    all_ce = []
    offsets = [0]
    ce_offsets = [0]
    for tids, cev in zip(file_token_ids, file_ce_values):
        all_tokens.extend(tids)
        offsets.append(len(all_tokens))
        all_ce.extend(cev)
        ce_offsets.append(len(all_ce))

    np.save(data_dir / "tokens.npy", np.array(all_tokens, dtype=np.int32))
    np.save(data_dir / "offsets.npy", np.array(offsets, dtype=np.int64))
    np.save(data_dir / "ce_values.npy", np.array(all_ce, dtype=np.float32))
    np.save(data_dir / "ce_offsets.npy", np.array(ce_offsets, dtype=np.int64))

    manifest = {
        "schema_version": 1,
        "row_count": len(file_token_ids),
        "total_tokens": len(all_tokens),
        "total_ce_values": len(all_ce),
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest))


@pytest.fixture
def ice_data_dir(tmp_path: Path) -> Path:
    """Create a small mmap ICE dataset (5 files)."""
    d = tmp_path / "ice"
    d.mkdir()
    _create_ice_data(
        d,
        file_token_ids=[
            list(range(10 + i * 5)) for i in range(5)
        ],
        file_ce_values=[
            [0.5 * j for j in range(9 + i * 5)] for i in range(5)
        ],
    )
    return d


class TestMmapICEDataset:
    def test_len(self, ice_data_dir: Path):
        ds = MmapICEDataset(str(ice_data_dir))
        assert len(ds) == 5

    def test_getitem_returns_dict(self, ice_data_dir: Path):
        ds = MmapICEDataset(str(ice_data_dir))
        sample = ds[0]
        assert isinstance(sample, dict)
        assert "token_ids" in sample
        assert "ce_values" in sample

    def test_getitem_tensor_types(self, ice_data_dir: Path):
        ds = MmapICEDataset(str(ice_data_dir))
        sample = ds[0]
        assert isinstance(sample["token_ids"], torch.Tensor)
        assert isinstance(sample["ce_values"], torch.Tensor)
        assert sample["token_ids"].dtype == torch.int64
        assert sample["ce_values"].dtype == torch.float32

    def test_getitem_shapes(self, ice_data_dir: Path):
        ds = MmapICEDataset(str(ice_data_dir))
        sample = ds[0]
        # First row: 10 token_ids, 9 ce_values
        assert sample["token_ids"].shape == (10,)
        assert sample["ce_values"].shape == (9,)

    def test_all_items_accessible(self, ice_data_dir: Path):
        ds = MmapICEDataset(str(ice_data_dir))
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["token_ids"].ndim == 1
            assert sample["ce_values"].ndim == 1

    def test_ce_length_is_tokens_minus_one(self, ice_data_dir: Path):
        """CE values should have length = token length - 1 per sample."""
        ds = MmapICEDataset(str(ice_data_dir))
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["ce_values"].shape[0] == sample["token_ids"].shape[0] - 1

    def test_missing_files(self, tmp_path: Path):
        d = tmp_path / "missing"
        d.mkdir()
        with pytest.raises(FileNotFoundError, match="Missing mmap artifacts"):
            MmapICEDataset(str(d))

    def test_invalid_manifest(self, tmp_path: Path):
        d = tmp_path / "bad"
        d.mkdir()
        np.save(d / "tokens.npy", np.array([1], dtype=np.int32))
        np.save(d / "offsets.npy", np.array([0, 1], dtype=np.int64))
        np.save(d / "ce_values.npy", np.array([], dtype=np.float32))
        np.save(d / "ce_offsets.npy", np.array([0, 0], dtype=np.int64))
        (d / "manifest.json").write_text(json.dumps({"schema_version": 99}))
        with pytest.raises(ValueError, match="Unsupported manifest schema version"):
            MmapICEDataset(str(d))

    def test_ce_alignment_validation(self, tmp_path: Path):
        """Mismatched CE/token lengths should raise ValueError at load time."""
        d = tmp_path / "misaligned"
        d.mkdir()
        # 3 tokens but 3 CE values (should be 2)
        np.save(d / "tokens.npy", np.array([1, 2, 3], dtype=np.int32))
        np.save(d / "offsets.npy", np.array([0, 3], dtype=np.int64))
        np.save(d / "ce_values.npy", np.array([0.1, 0.2, 0.3], dtype=np.float32))
        np.save(d / "ce_offsets.npy", np.array([0, 3], dtype=np.int64))
        (d / "manifest.json").write_text(json.dumps({"schema_version": 1}))
        with pytest.raises(ValueError, match="CE/token alignment error"):
            MmapICEDataset(str(d))

    def test_manifest_row_count_validation(self, tmp_path: Path):
        """Stale manifest row_count should raise ValueError."""
        d = tmp_path / "stale"
        d.mkdir()
        np.save(d / "tokens.npy", np.array([1, 2, 3], dtype=np.int32))
        np.save(d / "offsets.npy", np.array([0, 3], dtype=np.int64))
        np.save(d / "ce_values.npy", np.array([0.1, 0.2], dtype=np.float32))
        np.save(d / "ce_offsets.npy", np.array([0, 2], dtype=np.int64))
        (d / "manifest.json").write_text(json.dumps({"schema_version": 1, "row_count": 5}))
        with pytest.raises(ValueError, match="Manifest row_count"):
            MmapICEDataset(str(d))

    def test_multi_file_data(self, tmp_path: Path):
        """Verify correct indexing across multiple files."""
        d = tmp_path / "multi"
        d.mkdir()
        _create_ice_data(
            d,
            file_token_ids=[[1, 2, 3], [10, 20, 30, 40], [100, 200]],
            file_ce_values=[[0.1, 0.2], [0.3, 0.4, 0.5], [0.6]],
        )
        ds = MmapICEDataset(str(d))
        assert len(ds) == 3

        s0 = ds[0]
        assert torch.equal(s0["token_ids"], torch.tensor([1, 2, 3], dtype=torch.int64))
        assert torch.allclose(s0["ce_values"], torch.tensor([0.1, 0.2], dtype=torch.float32))

        s1 = ds[1]
        assert torch.equal(s1["token_ids"], torch.tensor([10, 20, 30, 40], dtype=torch.int64))

        s2 = ds[2]
        assert torch.equal(s2["token_ids"], torch.tensor([100, 200], dtype=torch.int64))

    def test_zero_length_file_skipped(self, tmp_path: Path):
        """Files with 0 tokens are excluded."""
        d = tmp_path / "zero"
        d.mkdir()
        _create_ice_data(
            d,
            file_token_ids=[[], [1, 2, 3], []],
            file_ce_values=[[], [0.1, 0.2], []],
        )
        ds = MmapICEDataset(str(d))
        assert len(ds) == 1
        assert torch.equal(ds[0]["token_ids"], torch.tensor([1, 2, 3], dtype=torch.int64))

    def test_lengths_property(self, ice_data_dir: Path):
        ds = MmapICEDataset(str(ice_data_dir))
        expected = [10, 15, 20, 25, 30]
        np.testing.assert_array_equal(ds.lengths, expected)

    def test_pickle_roundtrip(self, ice_data_dir: Path):
        ds = MmapICEDataset(str(ice_data_dir))
        data = pickle.dumps(ds)
        assert len(data) < 100_000

        ds2 = pickle.loads(data)
        assert len(ds2) == len(ds)
        sample = ds2[0]
        assert torch.equal(sample["token_ids"], ds[0]["token_ids"])

    def test_multi_worker_dataloader(self, ice_data_dir: Path):
        ds = MmapICEDataset(str(ice_data_dir))
        loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=2)
        samples = list(loader)
        assert len(samples) == 5
