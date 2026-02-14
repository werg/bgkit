"""Tests for ICE label generation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from bgkit.data.ice_label_generator import _batched_ce_inference, generate_ce_for_sequence


class TinyMockModel(nn.Module):
    """Tiny causal LM mock that returns random logits."""

    def __init__(self, vocab_size: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        # Need at least one parameter so next(model.parameters()) works
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, attention_mask=None):
        batch, seq_len = input_ids.shape
        logits = torch.randn(batch, seq_len, self.vocab_size, device=input_ids.device)
        result = MagicMock()
        result.logits = logits
        return result


class TestGenerateCEForSequence:
    def test_basic_shape(self):
        model = TinyMockModel(vocab_size=100)
        token_ids = list(range(50))

        ce = generate_ce_for_sequence(
            token_ids, model, max_seq_len=128, device=torch.device("cpu")
        )

        assert isinstance(ce, np.ndarray)
        assert ce.dtype == np.float16
        assert len(ce) == len(token_ids) - 1

    def test_empty_sequence(self):
        model = TinyMockModel()
        ce = generate_ce_for_sequence([], model, device=torch.device("cpu"))
        assert len(ce) == 0

    def test_single_token(self):
        model = TinyMockModel()
        ce = generate_ce_for_sequence([42], model, device=torch.device("cpu"))
        assert len(ce) == 0

    def test_two_tokens(self):
        model = TinyMockModel()
        ce = generate_ce_for_sequence([1, 2], model, device=torch.device("cpu"))
        assert len(ce) == 1

    def test_ce_values_non_negative(self):
        model = TinyMockModel(vocab_size=100)
        token_ids = list(range(20))

        ce = generate_ce_for_sequence(
            token_ids, model, max_seq_len=128, device=torch.device("cpu")
        )

        assert np.all(ce >= 0)

    def test_chunking_long_sequence(self):
        model = TinyMockModel(vocab_size=256)
        token_ids = list(range(200))

        ce = generate_ce_for_sequence(
            token_ids, model, max_seq_len=64, device=torch.device("cpu")
        )

        assert isinstance(ce, np.ndarray)
        assert len(ce) > 0


class TestBatchedCEInference:
    def test_basic_batch(self):
        model = TinyMockModel(vocab_size=100)
        sequences = [list(range(10)), list(range(20)), list(range(15))]

        results = _batched_ce_inference(
            sequences, model, torch.device("cpu"), max_batch_tokens=4096
        )

        assert len(results) == 3
        assert len(results[0]) == 9   # 10 - 1
        assert len(results[1]) == 19  # 20 - 1
        assert len(results[2]) == 14  # 15 - 1

    def test_preserves_order(self):
        model = TinyMockModel(vocab_size=100)
        # Varying lengths — batching sorts internally but must restore order
        sequences = [list(range(30)), list(range(5)), list(range(50)), list(range(10))]

        results = _batched_ce_inference(
            sequences, model, torch.device("cpu"), max_batch_tokens=4096
        )

        assert len(results) == 4
        assert len(results[0]) == 29
        assert len(results[1]) == 4
        assert len(results[2]) == 49
        assert len(results[3]) == 9

    def test_small_batch_budget_forces_splits(self):
        model = TinyMockModel(vocab_size=100)
        sequences = [list(range(20)), list(range(20)), list(range(20))]

        # Budget of 30 means only 1 seq of len 20 per batch
        results = _batched_ce_inference(
            sequences, model, torch.device("cpu"), max_batch_tokens=30
        )

        assert len(results) == 3
        for r in results:
            assert len(r) == 19

    def test_empty_input(self):
        model = TinyMockModel()
        results = _batched_ce_inference([], model, torch.device("cpu"))
        assert results == []

    def test_single_token_sequences_skipped(self):
        model = TinyMockModel(vocab_size=100)
        sequences = [[42], list(range(10)), [7]]

        results = _batched_ce_inference(
            sequences, model, torch.device("cpu"), max_batch_tokens=4096
        )

        assert len(results) == 3
        assert len(results[0]) == 0   # single token -> empty
        assert len(results[1]) == 9
        assert len(results[2]) == 0

    def test_ce_values_non_negative(self):
        model = TinyMockModel(vocab_size=100)
        sequences = [list(range(10)), list(range(20))]

        results = _batched_ce_inference(
            sequences, model, torch.device("cpu"), max_batch_tokens=4096
        )

        for r in results:
            assert np.all(r >= 0)

    def test_all_same_length(self):
        model = TinyMockModel(vocab_size=100)
        sequences = [list(range(15))] * 5

        results = _batched_ce_inference(
            sequences, model, torch.device("cpu"), max_batch_tokens=4096
        )

        assert len(results) == 5
        for r in results:
            assert len(r) == 14


class TestOutputFormat:
    def test_ce_label_shard_schema(self, tmp_path: Path):
        """Verify the expected output Parquet schema."""
        table = pa.table({
            "repo_path": pa.array(["repo/a"], type=pa.string()),
            "file_path": pa.array(["main.py"], type=pa.string()),
            "language": pa.array(["Python"], type=pa.string()),
            "chunk_idx": pa.array([0], type=pa.int32()),
            "token_ids": pa.array(
                [np.array([1, 2, 3], dtype=np.int32)], type=pa.list_(pa.int32())
            ),
            "ce_values": pa.array(
                [np.array([0.5, 1.2], dtype=np.float16)], type=pa.list_(pa.float16())
            ),
        })

        shard_path = tmp_path / "shard_00000.parquet"
        pq.write_table(table, shard_path)

        loaded = pq.read_table(shard_path)
        assert loaded.num_rows == 1
        assert set(loaded.column_names) == {
            "repo_path", "file_path", "language", "chunk_idx", "token_ids", "ce_values"
        }
        assert loaded.column("token_ids")[0].as_py() == [1, 2, 3]
