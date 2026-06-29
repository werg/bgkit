"""Unit test for the recursive-L1 entry point ``BgKITEncoder.encode_node``.

Uses a tiny stub L1 so the test runs on CPU with no real backbone. Verifies:
- right output shape/space (L1-output survivors, hidden_dim wide),
- input-dim assertion,
- ``pinned_id_embeddings`` wired as a per-node 1-token L1 prompt,
- ``projection_blocks`` is never consulted (stub has none — any touch would
  raise AttributeError).
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from bgkit.models.encoder import BgKITEncoder


class _StubLevelOutput:
    def __init__(self, survivor_embeddings, survivor_cu_seqlens):
        self.survivor_embeddings = survivor_embeddings
        self.survivor_cu_seqlens = survivor_cu_seqlens


class _StubL1:
    """Records the call kwargs and returns the first row of each segment as
    that segment's single survivor (a stand-in for compression)."""

    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
        self.last_call: dict = {}

    def __call__(self, **kwargs):
        self.last_call = kwargs
        content = kwargs["content_embeddings"]
        cu = kwargs["content_cu_seqlens"].to(torch.int64)
        # One survivor per segment: the segment's first row.
        starts = cu[:-1]
        survivors = content[starts]
        out_cu = torch.arange(
            0, starts.numel() + 1, dtype=torch.int32, device=content.device,
        )
        return _StubLevelOutput(survivors, out_cu)


def _stub_encoder(hidden_dim: int):
    return types.SimpleNamespace(l1=_StubL1(hidden_dim))


def test_encode_node_shape_and_space():
    hidden = 8
    enc = _stub_encoder(hidden)
    # Two nodes: 3 children rows then 2 children rows.
    reps = torch.arange(5 * hidden, dtype=torch.float32).reshape(5, hidden)
    cu = torch.tensor([0, 3, 5], dtype=torch.int32)

    surv, surv_cu = BgKITEncoder.encode_node(enc, reps, cu, target_ratio=0.5)

    # One survivor per node, hidden_dim wide (L1-output space).
    assert surv.shape == (2, hidden)
    assert surv_cu.tolist() == [0, 1, 2]
    # No prompt was supplied.
    assert enc.l1.last_call["prompt_embeddings"] is None
    # The survivor is the first row of each segment (rows 0 and 3).
    assert torch.allclose(surv[0], reps[0])
    assert torch.allclose(surv[1], reps[3])


def test_encode_node_rejects_wrong_input_dim():
    enc = _stub_encoder(8)
    reps = torch.zeros(3, 7)  # wrong width
    cu = torch.tensor([0, 3], dtype=torch.int32)
    with pytest.raises(ValueError, match="L1 input dim"):
        BgKITEncoder.encode_node(enc, reps, cu, target_ratio=0.5)


def test_encode_node_pinned_id_embeddings_wired_as_prompt():
    hidden = 8
    enc = _stub_encoder(hidden)
    reps = torch.zeros(4, hidden)
    cu = torch.tensor([0, 2, 4], dtype=torch.int32)  # 2 nodes
    pinned = torch.ones(2, hidden)  # one id embedding per node

    BgKITEncoder.encode_node(
        enc, reps, cu, target_ratio=0.5, pinned_id_embeddings=pinned,
    )
    call = enc.l1.last_call
    assert call["prompt_embeddings"] is not None
    assert call["prompt_embeddings"].shape == (2, hidden)
    # One prompt token per node.
    assert call["prompt_cu_seqlens"].tolist() == [0, 1, 2]


def test_encode_node_rejects_bad_pinned_shape():
    enc = _stub_encoder(8)
    reps = torch.zeros(4, 8)
    cu = torch.tensor([0, 2, 4], dtype=torch.int32)  # 2 nodes
    bad = torch.ones(3, 8)  # 3 != 2 nodes
    with pytest.raises(ValueError, match="pinned_id_embeddings"):
        BgKITEncoder.encode_node(
            enc, reps, cu, target_ratio=0.5, pinned_id_embeddings=bad,
        )
