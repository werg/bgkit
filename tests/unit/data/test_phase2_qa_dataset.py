"""Tests for the generic Phase 2 QA mmap dataset and collator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from bgkit.data.collators import collate_qa
from bgkit.data.datasets.phase2_qa_dataset import Phase2QADataset


def _write_phase2_artifacts(base: Path) -> Path:
    base.mkdir()
    np.save(base / "tokens.npy", np.array([11, 12, 13, 21, 22], dtype=np.int32))
    np.save(base / "offsets.npy", np.array([0, 3, 5], dtype=np.int64))
    np.save(base / "question_tokens.npy", np.array([31, 32, 41], dtype=np.int32))
    np.save(base / "question_offsets.npy", np.array([0, 2, 3], dtype=np.int64))
    np.save(base / "answer_tokens.npy", np.array([51, 52, 53, 61], dtype=np.int32))
    np.save(base / "answer_offsets.npy", np.array([0, 3, 4], dtype=np.int64))
    (base / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "row_count": 2,
        "total_tokens": 5,
        "dataset_name": "pubmedqa",
    }))
    pq.write_table(
        pa.table({
            "id": pa.array(["q1", "q2"]),
            "document_id": pa.array(["d1", "d2"]),
            "language": pa.array(["text", "text"]),
            "tag_list_json": pa.array(['["a/b", "c"]', '["a/d"]']),
        }),
        base / "metadata.parquet",
    )
    return base


def test_phase2_dataset_loads_and_builds_targets(tmp_path):
    ds = Phase2QADataset(str(_write_phase2_artifacts(tmp_path / "qa")))
    sample = ds[0]
    assert sample.dataset_name == "pubmedqa"
    assert sample.document_id == "d1"
    assert sample.question_token_ids.tolist() == [31, 32]
    assert sample.answer_token_ids.tolist() == [51, 52, 53]
    assert sample.target_token_ids.tolist() == [31, 32, 51, 52, 53]
    assert sample.target_loss_mask.tolist() == [False, False, True, True, True]
    assert sample.tags == ["a/b", "c"]


def test_collate_qa_pads_question_answer_and_target(tmp_path):
    ds = Phase2QADataset(str(_write_phase2_artifacts(tmp_path / "qa")))
    batch = collate_qa([ds[0], ds[1]])
    assert batch["content_token_ids"].shape == (2, 3)
    assert batch["question_token_ids"].shape == (2, 2)
    assert batch["answer_token_ids"].shape == (2, 3)
    assert batch["target_token_ids"].shape == (2, 5)
    assert batch["target_loss_mask"][1].tolist() == [False, True, False, False, False]
