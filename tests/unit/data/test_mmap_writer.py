"""Tests for mmap_writer utility functions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bgkit.data.mmap_writer import (
    arrow_to_numpy,
    build_csr_offsets,
    collect_jsonl_files,
    infer_repo_path,
    write_mmap_artifacts,
)


class TestBuildCsrOffsets:
    def test_basic(self):
        lengths = np.array([3, 5, 2], dtype=np.int64)
        offsets = build_csr_offsets(lengths)
        np.testing.assert_array_equal(offsets, [0, 3, 8, 10])

    def test_empty(self):
        lengths = np.array([], dtype=np.int64)
        offsets = build_csr_offsets(lengths)
        np.testing.assert_array_equal(offsets, [0])

    def test_single(self):
        lengths = np.array([7], dtype=np.int64)
        offsets = build_csr_offsets(lengths)
        np.testing.assert_array_equal(offsets, [0, 7])

    def test_dtype(self):
        offsets = build_csr_offsets(np.array([1, 2], dtype=np.int64))
        assert offsets.dtype == np.int64


class TestWriteMmapArtifacts:
    def test_basic(self, tmp_path):
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        offsets = np.array([0, 3, 5], dtype=np.int64)

        manifest = write_mmap_artifacts(tmp_path / "out", tokens, offsets)

        assert (tmp_path / "out" / "tokens.npy").exists()
        assert (tmp_path / "out" / "offsets.npy").exists()
        assert (tmp_path / "out" / "manifest.json").exists()

        assert manifest["schema_version"] == 1
        assert manifest["row_count"] == 2
        assert manifest["total_tokens"] == 5
        assert "offsets_sha256" in manifest
        assert "conversion_timestamp" in manifest

    def test_with_manifest_extra(self, tmp_path):
        tokens = np.array([1], dtype=np.int32)
        offsets = np.array([0, 1], dtype=np.int64)

        manifest = write_mmap_artifacts(
            tmp_path, tokens, offsets,
            manifest_extra={"source_shard_count": 5, "custom_key": "val"},
        )

        assert manifest["source_shard_count"] == 5
        assert manifest["custom_key"] == "val"

    def test_with_extra_arrays(self, tmp_path):
        tokens = np.array([1, 2, 3], dtype=np.int32)
        offsets = np.array([0, 3], dtype=np.int64)
        ce_values = np.array([0.1, 0.2], dtype=np.float32)
        ce_offsets = np.array([0, 2], dtype=np.int64)

        write_mmap_artifacts(
            tmp_path, tokens, offsets,
            extra_arrays={"ce_values.npy": ce_values, "ce_offsets.npy": ce_offsets},
        )

        loaded_ce = np.load(tmp_path / "ce_values.npy")
        assert loaded_ce.dtype == np.float32
        np.testing.assert_allclose(loaded_ce, [0.1, 0.2])

        loaded_ce_off = np.load(tmp_path / "ce_offsets.npy")
        assert loaded_ce_off.dtype == np.int64

    def test_with_metadata_table(self, tmp_path):
        tokens = np.array([1, 2, 3, 4], dtype=np.int32)
        offsets = np.array([0, 2, 4], dtype=np.int64)
        meta = pa.table({
            "file_path": ["a.py", "b.py"],
            "language": ["python", "python"],
        })

        write_mmap_artifacts(tmp_path, tokens, offsets, metadata_table=meta)

        loaded_meta = pq.read_table(tmp_path / "metadata.parquet")
        assert loaded_meta.num_rows == 2
        assert "file_path" in loaded_meta.column_names

    def test_creates_dirs(self, tmp_path):
        tokens = np.array([1], dtype=np.int32)
        offsets = np.array([0, 1], dtype=np.int64)
        write_mmap_artifacts(tmp_path / "nested" / "dir", tokens, offsets)
        assert (tmp_path / "nested" / "dir" / "tokens.npy").exists()


class TestArrowToNumpy:
    def test_plain_array(self):
        arr = pa.array([1, 2, 3], type=pa.int32())
        result = arrow_to_numpy(arr)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_chunked_array(self):
        arr = pa.chunked_array([
            pa.array([1, 2], type=pa.int32()),
            pa.array([3, 4], type=pa.int32()),
        ])
        result = arrow_to_numpy(arr)
        np.testing.assert_array_equal(result, [1, 2, 3, 4])

    def test_dtype_cast(self):
        arr = pa.array([1.5, 2.5], type=pa.float64())
        result = arrow_to_numpy(arr, dtype=np.float32)
        assert result.dtype == np.float32


class TestCollectJsonlFiles:
    def test_finds_nested_jsonl(self, tmp_path):
        (tmp_path / "owner").mkdir()
        (tmp_path / "owner" / "repo1.jsonl").touch()
        (tmp_path / "owner" / "repo2.jsonl").touch()
        files = collect_jsonl_files(tmp_path)
        assert len(files) == 2

    def test_excludes_tmp(self, tmp_path):
        (tmp_path / "data.jsonl").touch()
        (tmp_path / "data.jsonl.tmp").touch()
        files = collect_jsonl_files(tmp_path)
        assert len(files) == 1
        assert files[0].name == "data.jsonl"

    def test_empty_dir(self, tmp_path):
        assert collect_jsonl_files(tmp_path) == []


class TestInferRepoPath:
    def test_basic(self, tmp_path):
        jsonl_path = tmp_path / "owner" / "repo.jsonl"
        assert infer_repo_path(jsonl_path, tmp_path) == "owner/repo"

    def test_nested(self, tmp_path):
        jsonl_path = tmp_path / "a" / "b.jsonl"
        assert infer_repo_path(jsonl_path, tmp_path) == "a/b"
