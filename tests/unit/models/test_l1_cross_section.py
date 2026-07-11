"""Unit tests for the L1 cross-section interaction rework (2026-06-11).

L1 must merge each sample's sections into ONE segment (separators between, prompt
once) so it attends across sections — not compress each section in isolation.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.models.encoder import BgKITEncoder
from bgkit.models.level_compressor import _build_combined_pack


def _merge(section_sep, embeds, survivor_cu, group_cu):
    """Call BgKITEncoder._merge_sections_for_l1 with a minimal stub holder."""

    class _Stub:
        pass

    stub = _Stub()
    stub.section_separator_embedding = section_sep
    return BgKITEncoder._merge_sections_for_l1(stub, embeds, survivor_cu, group_cu)


def test_merge_sections_regroups_to_per_sample_no_data_change():
    D = 4
    sep = torch.zeros(D)  # unused now (no separator tokens)
    # 3 sections: lengths 2, 3, 1. embeds rows tagged 0..5 in col 0.
    embeds = torch.zeros(6, D)
    embeds[:, 0] = torch.arange(6, dtype=torch.float32)
    survivor_cu = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
    # sample 0 = sections [0,2) ; sample 1 = section [2,3)
    group_cu = torch.tensor([0, 2, 3], dtype=torch.int32)

    merged, merged_cu, merged_pos, selectable = _merge(sep, embeds, survivor_cu, group_cu)

    # Pure cu regroup: data unchanged, merged_cu = survivor_cu[group_cu].
    assert torch.equal(merged, embeds)
    assert selectable is None
    assert merged_cu.tolist() == [0, 5, 6]  # sample0 = 2+3 survivors, sample1 = 1
    assert merged_cu.dtype == torch.int32
    # length preserved == survivor count (alignment invariant)
    assert merged.shape[0] == int(survivor_cu[-1])
    # per-sample position ids restart at 0
    assert merged_pos[:5].tolist() == [0, 1, 2, 3, 4]
    assert merged_pos[5].item() == 0


def test_merge_zero_survivor_sample_gives_zero_length_segment():
    D = 4
    sep = torch.zeros(D)
    # sample 0's 2 sections both empty; sample 1 has 2 survivors.
    embeds = torch.zeros(2, D)
    embeds[:, 0] = torch.tensor([10.0, 11.0])
    survivor_cu = torch.tensor([0, 0, 0, 2], dtype=torch.int32)
    group_cu = torch.tensor([0, 2, 3], dtype=torch.int32)  # sample0 = secs[0,2) = empty

    merged, merged_cu, _, selectable = _merge(sep, embeds, survivor_cu, group_cu)
    # sample0 → zero-length segment (guarded by min-survivors aux loss upstream)
    assert merged_cu.tolist() == [0, 0, 2]
    assert merged.shape[0] == 2
    assert selectable is None


def test_combined_pack_selectable_mask_excludes_section_separators():
    D = 3
    # one sample: content has 4 positions, position 1 is a (section) separator.
    content = torch.randn(4, D)
    content_cu = torch.tensor([0, 4], dtype=torch.int32)
    content_pos = torch.arange(4, dtype=torch.int64)
    prompt = torch.randn(2, D)
    prompt_cu = torch.tensor([0, 2], dtype=torch.int32)
    prompt_pos = torch.arange(2, dtype=torch.int64)
    prompt_sep = torch.zeros(D)
    selectable = torch.tensor([True, False, True, True])  # idx1 = section sep

    x, cu, pos, mask, mx = _build_combined_pack(
        content, content_cu, content_pos, prompt, prompt_cu, prompt_pos,
        prompt_sep, content_selectable_mask=selectable,
    )
    # combined = [prompt(2) | prompt_sep(1) | content(4)] = 7
    assert x.shape[0] == 7
    # mask: prompt+sep False (3), then content selectable [T,F,T,T]
    assert mask.tolist() == [False, False, False, True, False, True, True]


def test_combined_pack_default_mask_all_content_selectable():
    D = 3
    content = torch.randn(3, D)
    content_cu = torch.tensor([0, 3], dtype=torch.int32)
    content_pos = torch.arange(3, dtype=torch.int64)
    prompt = torch.randn(1, D)
    prompt_cu = torch.tensor([0, 1], dtype=torch.int32)
    prompt_pos = torch.arange(1, dtype=torch.int64)
    x, cu, pos, mask, mx = _build_combined_pack(
        content, content_cu, content_pos, prompt, prompt_cu, prompt_pos, torch.zeros(D),
    )
    # [prompt(1)|sep(1)|content(3)] = 5; content all True
    assert mask.tolist() == [False, False, True, True, True]


def test_combined_pack_vectorized_multi_sample_preserves_order_and_gradients():
    hidden_dim = 2
    content = torch.arange(10, dtype=torch.float32).reshape(5, hidden_dim).requires_grad_()
    prompt = torch.arange(6, dtype=torch.float32).reshape(3, hidden_dim).requires_grad_()
    separator = torch.tensor([-1.0, -1.0], requires_grad=True)
    content_cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    prompt_cu = torch.tensor([0, 1, 3], dtype=torch.int32)

    packed, cu, pos, mask, max_len = _build_combined_pack(
        content,
        content_cu,
        torch.tensor([0, 1, 0, 1, 2]),
        prompt,
        prompt_cu,
        torch.tensor([0, 0, 1]),
        separator,
    )

    expected = torch.cat([
        prompt[:1], separator[None], content[:2],
        prompt[1:], separator[None], content[2:],
    ])
    assert torch.equal(packed, expected)
    assert cu.tolist() == [0, 4, 10]
    assert pos.tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 4, 5]
    assert mask.tolist() == [False, False, True, True, False, False, False, True, True, True]
    assert max_len == 6

    packed.sum().backward()
    assert torch.equal(content.grad, torch.ones_like(content))
    assert torch.equal(prompt.grad, torch.ones_like(prompt))
    assert torch.equal(separator.grad, torch.tensor([2.0, 2.0]))
