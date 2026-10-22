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


def test_collate_qa_packed_shapes(tmp_path):
    """Packed collator produces flat (N,) tensors with cu_seqlens, no padding."""
    ds = Phase2QADataset(str(_write_phase2_artifacts(tmp_path / "qa")))
    batch = collate_qa([ds[0], ds[1]])

    # content: sample 0 has 3 tokens, sample 1 has 2 tokens → N=5
    N_content = batch["content_cu_seqlens"][-1].item()
    assert batch["content_token_ids"].shape == (N_content,)
    assert N_content == 3 + 2

    # question: sample 0 has 2 tokens, sample 1 has 1 token → N=3
    N_q = batch["question_cu_seqlens"][-1].item()
    assert batch["question_token_ids"].shape == (N_q,)
    assert N_q == 2 + 1

    # answer: sample 0 has 3 tokens, sample 1 has 1 token → N=4
    N_a = batch["answer_cu_seqlens"][-1].item()
    assert batch["answer_token_ids"].shape == (N_a,)
    assert N_a == 3 + 1

    # target: sample 0 has 5 tokens, sample 1 has 2 tokens → N=7
    N_t = batch["target_cu_seqlens"][-1].item()
    assert batch["target_token_ids"].shape == (N_t,)
    assert batch["target_loss_mask"].shape == (N_t,)

    # cu_seqlens invariants
    cu = batch["content_cu_seqlens"]
    assert cu[0].item() == 0
    assert cu.shape[0] == 3  # B+1 = 2+1
    assert "attention_mask" not in batch
