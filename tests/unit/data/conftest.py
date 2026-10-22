"""Shared test fixtures for mmap dataset tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def create_mmap_artifacts():
    """Factory fixture to write mmap artifacts from known token data.

    Returns a callable: create(data_dir, token_lists, metadata_columns=None) -> Path
    """

    def _create(
        data_dir: Path,
        token_lists: list[list[int]],
        metadata_columns: dict[str, list] | None = None,
    ) -> Path:
        all_tokens = []
        offsets = [0]
        for tids in token_lists:
            all_tokens.extend(tids)
            offsets.append(len(all_tokens))

        np.save(data_dir / "tokens.npy", np.array(all_tokens, dtype=np.int32))
        np.save(data_dir / "offsets.npy", np.array(offsets, dtype=np.int64))

        manifest = {
            "schema_version": 1,
            "row_count": len(token_lists),
            "total_tokens": len(all_tokens),
        }
        (data_dir / "manifest.json").write_text(json.dumps(manifest))

        if metadata_columns is not None:
            arrow_cols: dict[str, pa.Array] = {}
            for col_name, values in metadata_columns.items():
                if col_name == "prompt_version":
                    arrow_cols[col_name] = pa.array(values, type=pa.int32())
                else:
                    arrow_cols[col_name] = pa.array(values, type=pa.string())
            pq.write_table(pa.table(arrow_cols), data_dir / "metadata.parquet")

        return data_dir

    return _create


@pytest.fixture
def create_ice_artifacts(create_mmap_artifacts):
    """Factory fixture to write ICE mmap artifacts (tokens + CE values).

    Returns a callable: create(data_dir, file_token_ids, file_ce_values) -> Path
    """

    def _create(
        data_dir: Path,
        file_token_ids: list[list[int]],
        file_ce_values: list[list[float]],
    ) -> Path:
        all_tokens = []
        all_ce = []
        offsets = [0]
        ce_offsets = [0]
        for tids, cev in zip(file_token_ids, file_ce_values, strict=False):
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

        return data_dir

    return _create
