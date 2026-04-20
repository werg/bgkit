"""Tests for ICETeacher: packed API semantics + unload idempotency."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.models.ice import ICE
from bgkit.models.ice_teacher import ICETeacher


def _make_teacher(tmp_path):
    """Build a tiny ICETeacher backed by a freshly-initialized ICE checkpoint."""
    ice = ICE(input_dim=16, hidden_dim=8, num_layers=2, kernel_size=3, dropout=0.0)
    ckpt = tmp_path / "ice.pt"
    torch.save(ice.state_dict(), ckpt)
    embed = torch.nn.Embedding(num_embeddings=32, embedding_dim=16)
    return ICETeacher(
        ckpt,
        embed,
        input_dim=16,
        hidden_dim=8,
        num_layers=2,
        kernel_size=3,
    )


def _packed_single(ids_list: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor(ids_list, dtype=torch.long)
    cu = torch.tensor([0, len(ids_list)], dtype=torch.int32)
    return ids, cu


def test_score_works_before_unload(tmp_path):
    teacher = _make_teacher(tmp_path)
    ids, cu = _packed_single([1, 2, 3, 4])
    scores = teacher.score(ids, cu)
    assert scores.shape == (4,)
    assert teacher.is_loaded


def test_score_raises_after_unload(tmp_path):
    teacher = _make_teacher(tmp_path)
    teacher.unload()
    assert not teacher.is_loaded
    ids, cu = _packed_single([1, 2])
    with pytest.raises(RuntimeError, match="ICE unloaded"):
        teacher.score(ids, cu)


def test_teacher_mask_raises_after_unload(tmp_path):
    teacher = _make_teacher(tmp_path)
    teacher.unload()
    ids, cu = _packed_single([1, 2, 3])
    with pytest.raises(RuntimeError, match="ICE unloaded"):
        teacher.teacher_mask(ids, cu, target_ratio=0.5)


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


# ---------------------------------------------------------------------------
# Packed parity: per-segment vs batched packed call
# ---------------------------------------------------------------------------


def test_teacher_mask_packed_matches_per_sample_calls(tmp_path):
    """Packed multi-segment call should equal concatenated per-segment calls."""
    teacher = _make_teacher(tmp_path)
    torch.manual_seed(17)
    lengths = [4, 7, 3, 10]
    per_seg_ids: list[torch.Tensor] = [
        torch.randint(0, 32, (seg_len,), dtype=torch.long) for seg_len in lengths
    ]
    # Packed form.
    packed_ids = torch.cat(per_seg_ids, dim=0)
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    for i, seg_len in enumerate(lengths):
        cu[i + 1] = cu[i] + seg_len

    packed_mask = teacher.teacher_mask(packed_ids, cu, target_ratio=0.5)
    assert packed_mask.shape == (sum(lengths),)

    # Per-sample reference: each segment on its own with cu=[0, L].
    ref_parts: list[torch.Tensor] = []
    for seg_ids in per_seg_ids:
        seg_cu = torch.tensor([0, seg_ids.numel()], dtype=torch.int32)
        ref_parts.append(teacher.teacher_mask(seg_ids, seg_cu, target_ratio=0.5))
    ref_mask = torch.cat(ref_parts, dim=0)

    assert torch.equal(packed_mask, ref_mask)


def test_score_packed_matches_per_sample_calls(tmp_path):
    teacher = _make_teacher(tmp_path)
    torch.manual_seed(23)
    lengths = [5, 2, 9]
    per_seg_ids = [torch.randint(0, 32, (seg_len,), dtype=torch.long) for seg_len in lengths]
    packed_ids = torch.cat(per_seg_ids, dim=0)
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    for i, seg_len in enumerate(lengths):
        cu[i + 1] = cu[i] + seg_len

    packed_scores = teacher.score(packed_ids, cu)
    ref = torch.cat([
        teacher.score(ids, torch.tensor([0, ids.numel()], dtype=torch.int32))
        for ids in per_seg_ids
    ])
    # Exact equality is fine — ICE is deterministic and no inter-segment mixing.
    assert torch.allclose(packed_scores, ref, atol=1e-6)


def test_teacher_mask_respects_per_segment_top_k(tmp_path):
    """Number of survivors per segment should be ceil(L*ratio), min 1."""
    import math

    teacher = _make_teacher(tmp_path)
    lengths = [10, 20, 3]
    per_seg_ids = [torch.randint(0, 32, (seg_len,), dtype=torch.long) for seg_len in lengths]
    packed_ids = torch.cat(per_seg_ids, dim=0)
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    for i, seg_len in enumerate(lengths):
        cu[i + 1] = cu[i] + seg_len

    ratio = 0.25
    mask = teacher.teacher_mask(packed_ids, cu, target_ratio=ratio)
    for i, seg_len in enumerate(lengths):
        start = int(cu[i].item())
        end = int(cu[i + 1].item())
        seg_count = int(mask[start:end].sum().item())
        expected = max(1, min(seg_len, math.ceil(seg_len * ratio)))
        assert seg_count == expected, (
            f"segment {i}: expected {expected} survivors, got {seg_count}"
        )
