"""Tests for ICETeacher.unload(): semantics + idempotency."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.models.ice import ICE  # noqa: E402
from bgkit.models.ice_teacher import ICETeacher  # noqa: E402


def _make_teacher(tmp_path):
    """Build a tiny ICETeacher backed by a freshly-initialized ICE checkpoint."""
    ice = ICE(input_dim=16, hidden_dim=8, num_layers=2, kernel_size=3, dropout=0.0)
    ckpt = tmp_path / "ice.pt"
    torch.save(ice.state_dict(), ckpt)
    embed = torch.nn.Embedding(num_embeddings=32, embedding_dim=16)
    return ICETeacher(
        ckpt, embed,
        input_dim=16, hidden_dim=8, num_layers=2, kernel_size=3,
    )


def test_score_works_before_unload(tmp_path):
    teacher = _make_teacher(tmp_path)
    token_ids = torch.tensor([[1, 2, 3, 4]])
    attn = torch.tensor([[1, 1, 1, 1]])
    scores = teacher.score(token_ids, attn)
    assert scores.shape == (1, 4)
    assert teacher.is_loaded


def test_score_raises_after_unload(tmp_path):
    teacher = _make_teacher(tmp_path)
    teacher.unload()
    assert not teacher.is_loaded
    with pytest.raises(RuntimeError, match="ICE unloaded"):
        teacher.score(torch.tensor([[1, 2]]), torch.tensor([[1, 1]]))


def test_teacher_mask_raises_after_unload(tmp_path):
    teacher = _make_teacher(tmp_path)
    teacher.unload()
    with pytest.raises(RuntimeError, match="ICE unloaded"):
        teacher.teacher_mask(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 1, 1]]),
            target_ratio=0.5,
        )


def test_unload_is_idempotent(tmp_path):
    teacher = _make_teacher(tmp_path)
    teacher.unload()
    teacher.unload()  # no-op, no exception
    assert not teacher.is_loaded


def test_unload_releases_ice_attribute(tmp_path):
    teacher = _make_teacher(tmp_path)
    assert teacher.ice is not None
    # ICE is a registered submodule; its parameters count.
    n_params_before = sum(p.numel() for p in teacher.parameters())
    assert n_params_before > 0

    teacher.unload()
    assert teacher.ice is None
    # After unload, ICE's parameters should no longer be tracked.
    assert "ice" not in teacher._modules
    n_params_after = sum(p.numel() for p in teacher.parameters())
    assert n_params_after < n_params_before
