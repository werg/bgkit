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

from bgkit.data.ice_label_generator import generate_ce_for_sequence


class TinyMockModel(nn.Module):
    """Tiny causal LM mock that returns random logits."""

    def __init__(self, vocab_size: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        # Need at least one parameter so next(model.parameters()) works
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor):
        batch, seq_len = input_ids.shape
        logits = torch.randn(batch, seq_len, self.vocab_size, device=input_ids.device)
        result = MagicMock()
        result.logits = logits
        return result


class TestGenerateCEForSequence:
    def test_basic_shape(self):
        model = TinyMockModel(vocab_size=100)
        token_ids = list(range(50))

        ce = generate_ce_for_sequence(token_ids, model, max_seq_len=128, device=torch.device("cpu"))

        assert isinstance(ce, np.ndarray)
        assert ce.dtype == np.float16
        # CE has len(token_ids) - 1 values (shifted prediction)
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

        ce = generate_ce_for_sequence(token_ids, model, max_seq_len=128, device=torch.device("cpu"))

        # Cross-entropy should be non-negative
        assert np.all(ce >= 0)

    def test_chunking_long_sequence(self):
        model = TinyMockModel(vocab_size=256)
        token_ids = list(range(200))

        ce = generate_ce_for_sequence(
            token_ids, model, max_seq_len=64, device=torch.device("cpu")
        )

        assert isinstance(ce, np.ndarray)
        # With chunking, total CE values should cover all chunks
        # Each chunk produces chunk_len - 1 CE values
        assert len(ce) > 0


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
