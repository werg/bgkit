"""Unit tests for ``LevelCompressor`` (head fires at block 1 hook,
``survive_embedding`` propagates the decision to subsequent blocks)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from bgkit.models.level_compressor import (
    LevelCompressor,
    LevelOutput,
    _gather_survivors_packed,
)


# ----------------------------- helpers -----------------------------


class _StubBackbone(nn.Module):
    """Stand-in for PrunedBidirectionalQwen35.

    Two-block layout so we can register a hook at block 1 (the same
    relative position as the real pruned backbone). Each block is a
    learnable Linear; the hook receives the block-1 output, optionally
    modifies it (``survive_embedding`` scatter), and the modified hidden
    feeds block 2's input. After block 2, a final norm is applied.

    Captures the hidden-state values at hook entry / exit so tests can
    assert downstream blocks consumed the modified activations.
    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.block_0 = nn.Linear(hidden_dim, hidden_dim)
        self.block_1 = nn.Linear(hidden_dim, hidden_dim)
        self.block_2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.captured_pre_hook: torch.Tensor | None = None
        self.captured_post_hook: torch.Tensor | None = None

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_ids: torch.Tensor,
        layer_hooks: dict | None = None,
        **_kw,
    ):
        from transformers.modeling_outputs import BaseModelOutputWithPast
        h = self.block_0(inputs_embeds)
        h = self.block_1(h)
        self.captured_pre_hook = h.detach().clone()
        if layer_hooks and 1 in layer_hooks:
            h = layer_hooks[1](h)
            self.captured_post_hook = h.detach().clone()
        h = self.block_2(h)
        h = self.norm(h)
        return BaseModelOutputWithPast(last_hidden_state=h, hidden_states=None)


def _make_packed_content(B: int, lengths: list[int], D: int = 16, requires_grad: bool = True):
    """Build a ``(N, D)`` packed content tensor with cu_seqlens + position_ids."""
    assert len(lengths) == B
    N = sum(lengths)
    content = torch.randn(N, D, requires_grad=requires_grad)
    cu = torch.zeros(B + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
    pos = torch.cat([torch.arange(L, dtype=torch.int64) for L in lengths])
    return content, cu, pos


def _make_lc(backbone, **kw):
    """Construct a LevelCompressor with the test stub's hook layout
    (hook fires at block 1 — same as the pruned-backbone default)."""
    kw.setdefault("hidden_dim", 16)
    kw.setdefault("head_layer_index", 1)
    return LevelCompressor(backbone=backbone, **kw)


# ----------------------------- tests -----------------------------


def test_construct_l0_with_prompt_and_auto_repro():
    """L0 has prompt_separator_embedding + auto_repro_head + survive_embedding."""
    backbone = _StubBackbone(hidden_dim=16)
    lc = _make_lc(
        backbone=backbone,
        hidden_dim=16,
        survivorship_inner_dim=8,
        with_prompt=True,
        with_auto_repro=True,
    )
    assert lc.prompt_separator_embedding is not None
    assert lc.auto_repro_head is not None
    assert lc.head is not None
    assert lc.threshold is not None
    assert lc.survive_embedding is not None
    assert lc.survive_embedding.shape == (16,)


def test_construct_l1_no_prompt_no_auto_repro():
    """L1 has no prompt separator and no auto_repro_head, but has survive_embedding."""
    backbone = _StubBackbone(hidden_dim=16)
    lc = _make_lc(
        backbone=backbone,
        hidden_dim=16,
        with_prompt=False,
        with_auto_repro=False,
    )
    assert lc.prompt_separator_embedding is None
    assert lc.auto_repro_head is None
    assert lc.survive_embedding is not None


def test_forward_no_compression_returns_all_positions():
    """When target_ratio=None, survivors == all content positions, hook NOT called."""
    backbone = _StubBackbone(hidden_dim=16)
    lc = _make_lc(
        backbone=backbone, hidden_dim=16,
        with_prompt=False,
    )

    content, cu, pos = _make_packed_content(B=2, lengths=[5, 3])
    out = lc(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio=None,
    )

    assert out.survivor_embeddings.shape == (8, 16)  # all positions survive
    assert out.survivor_mask.all().item()
    assert out.survivor_counts.tolist() == [5, 3]
    assert out.base_raw is None  # head not consulted
    # Hook not installed under compression_off, so backbone's post-hook
    # capture remains its pre-hook value (the hook was never called).
    assert backbone.captured_post_hook is None


def test_forward_with_compression_runs_head_at_block1_hook():
    """Compression on: head fires at block 1, survive_embedding scattered, downstream blocks see modified hidden."""
    backbone = _StubBackbone(hidden_dim=16)
    lc = _make_lc(
        backbone=backbone, hidden_dim=16,
        survivorship_inner_dim=8,
        with_prompt=False,
        threshold_controller_cfg={"init_target_ratio": 0.5},
    )

    content, cu, pos = _make_packed_content(B=2, lengths=[10, 6])
    out = lc(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio=0.5,
    )

    assert out.base_raw is not None
    assert out.base_raw.shape == (16,)  # all content positions
    assert out.logits_for_op is not None
    assert out.survive_probs is not None
    assert out.survivor_mask.shape == (16,)
    assert out.survivor_embeddings.shape == (out.survivor_mask.sum().item(), 16)
    assert out.survivor_counts.sum().item() == out.survivor_mask.sum().item()
    # cu_seqlens monotone increasing
    cu_list = out.survivor_cu_seqlens.tolist()
    assert all(cu_list[i] <= cu_list[i + 1] for i in range(len(cu_list) - 1))
    # Hook fired — pre/post differ at surviving positions where survive_embedding
    # was added (assuming at least one survivor).
    assert backbone.captured_pre_hook is not None
    assert backbone.captured_post_hook is not None
    diff = (backbone.captured_post_hook - backbone.captured_pre_hook).abs().sum().item()
    assert diff > 0, "block-1 hook should have modified the hidden state"


def test_survive_embedding_scattered_only_at_survivors():
    """survive_embedding is added at surviving content positions only."""
    backbone = _StubBackbone(hidden_dim=16)
    lc = _make_lc(
        backbone=backbone, hidden_dim=16,
        survivorship_inner_dim=8,
        with_prompt=False,
        threshold_controller_cfg={"init_target_ratio": 0.5},
    )
    # Force survive_embedding to a recognizable sentinel.
    with torch.no_grad():
        lc.survive_embedding.fill_(7.0)

    content, cu, pos = _make_packed_content(B=1, lengths=[10])
    out = lc(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio=0.5,
    )
    pre = backbone.captured_pre_hook
    post = backbone.captured_post_hook
    delta = post - pre
    surv_mask = out.survivor_mask
    # At surviving positions, delta should be ~7.0 in every dim (LayerNorm-free
    # scatter; the stub has no normalization between block 1 and the hook).
    assert torch.allclose(
        delta[surv_mask],
        torch.full_like(delta[surv_mask], 7.0),
        atol=1e-5,
    )
    # At non-surviving positions, delta should be exactly zero.
    if (~surv_mask).any():
        assert torch.allclose(
            delta[~surv_mask],
            torch.zeros_like(delta[~surv_mask]),
            atol=1e-6,
        )


def test_forward_with_prompt_and_separator():
    """L0 with prompt: combined pack = prompt + separator + content; head fires on content only."""
    backbone = _StubBackbone(hidden_dim=16)
    lc = _make_lc(
        backbone=backbone, hidden_dim=16,
        with_prompt=True,
        threshold_controller_cfg={"init_target_ratio": 0.5},
    )

    content, content_cu, content_pos = _make_packed_content(B=2, lengths=[10, 6])
    prompt, prompt_cu, prompt_pos = _make_packed_content(B=2, lengths=[3, 4])

    out = lc(
        content_embeddings=content,
        content_cu_seqlens=content_cu,
        content_position_ids=content_pos,
        prompt_embeddings=prompt,
        prompt_cu_seqlens=prompt_cu,
        prompt_position_ids=prompt_pos,
        target_ratio=0.5,
    )

    # base_raw is over CONTENT positions only — prompt and separator ignored
    assert out.base_raw.shape == (16,)
    assert out.content_embeddings.shape == (16, 16)


def test_utility_grad_capture_populates_static_fields():
    """utility_grad_active=True captures the head-input values + a detached
    head re-application (``base_raw_for_util``) into the LevelOutput.

    Note: the ``register_hook`` on hook-time ``content_hidden`` is wired but
    does NOT receive a meaningful gradient when survivors are extracted
    POST-norm at the last block — content_hidden (block 1 output, content
    positions) is upstream of the head only, and the head's output is not
    in the loss path. Trainer code consults ``post_head_content_values``
    and ``base_raw_for_util`` directly. This matches the pre-rebuild
    behavior.
    """
    backbone = _StubBackbone(hidden_dim=16)
    lc = _make_lc(
        backbone=backbone, hidden_dim=16,
        with_prompt=False,
        threshold_controller_cfg={"init_target_ratio": 0.5},
    )

    content, cu, pos = _make_packed_content(B=1, lengths=[8], requires_grad=True)
    out = lc(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio=0.5,
        utility_grad_active=True,
    )

    assert out.base_raw_for_util is not None
    assert out.base_raw_for_util.shape == (8,)
    assert out.post_head_content_values is not None
    assert out.post_head_content_values.shape == (8, 16)
    # post_head_content_values is detached (just captured values for the
    # post-backward utility-grad BCE loss).
    assert not out.post_head_content_values.requires_grad


def test_auto_reproduce_l0_only():
    """auto_reproduce works on L0 (with_auto_repro=True), errors on L1."""
    backbone_l0 = _StubBackbone(hidden_dim=16)
    backbone_l1 = _StubBackbone(hidden_dim=16)
    l0 = _make_lc(
        backbone=backbone_l0, hidden_dim=16,
        with_prompt=True, with_auto_repro=True,
    )
    l1 = _make_lc(
        backbone=backbone_l1, hidden_dim=16,
        with_prompt=False, with_auto_repro=False,
    )

    embeddings = torch.randn(5, 16)
    out_l0 = l0.auto_reproduce(embeddings)
    assert out_l0.shape == (5, 16)

    with pytest.raises(RuntimeError, match="auto_reproduce"):
        l1.auto_reproduce(embeddings)


def test_l1_init_from_l0_clone_evolves_independently():
    """L1 can be init by deepcopy of L0's backbone state and then diverge under training."""
    import copy

    backbone_l0 = _StubBackbone(hidden_dim=16)
    backbone_l1 = _StubBackbone(hidden_dim=16)
    # Init L1 from L0 clone
    backbone_l1.load_state_dict(copy.deepcopy(backbone_l0.state_dict()))

    l0 = _make_lc(backbone=backbone_l0, hidden_dim=16, with_prompt=False)
    l1 = _make_lc(backbone=backbone_l1, hidden_dim=16, with_prompt=False)

    # Confirm initial parity
    for (n0, p0), (n1, p1) in zip(
        l0.backbone.named_parameters(), l1.backbone.named_parameters(),
    ):
        assert torch.allclose(p0, p1), f"init mismatch on {n0}"

    # Train L1 only — L0 should be unchanged
    content, cu, pos = _make_packed_content(B=1, lengths=[6], requires_grad=False)
    out = l1(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio=0.5,
    )
    out.survivor_embeddings.sum().backward()
    optim = torch.optim.SGD(l1.parameters(), lr=0.1)
    optim.step()

    # L1 parameters should differ somewhere now
    diffs = []
    for (n0, p0), (n1, p1) in zip(
        l0.backbone.named_parameters(), l1.backbone.named_parameters(),
    ):
        diffs.append((p0 - p1).abs().max().item())
    assert max(diffs) > 0, "L1 should have diverged from L0 after a training step"


def test_gather_survivors_packed():
    """_gather_survivors_packed correctly subsets and rebuilds cu_seqlens."""
    content = torch.arange(20, dtype=torch.float32).unsqueeze(-1)  # (20, 1)
    cu = torch.tensor([0, 8, 14, 20], dtype=torch.int32)  # 3 samples: lengths 8, 6, 6
    # Survive: indices 0..3 in sample 0, 8..9 in sample 1, 14..19 in sample 2
    mask = torch.zeros(20, dtype=torch.bool)
    mask[0:4] = True
    mask[8:10] = True
    mask[14:20] = True

    surv, surv_cu, counts = _gather_survivors_packed(content, mask, cu)
    assert surv.shape == (12, 1)
    assert counts.tolist() == [4, 2, 6]
    assert surv_cu.tolist() == [0, 4, 6, 12]
    # First 4 surv are 0..3, next 2 are 8..9, last 6 are 14..19
    assert surv.squeeze().tolist() == [0, 1, 2, 3, 8, 9, 14, 15, 16, 17, 18, 19]


def test_pinned_positions_force_survival():
    """Pinned positions appear in survivors even when head wants to drop them."""
    backbone = _StubBackbone(hidden_dim=16)
    lc = _make_lc(
        backbone=backbone, hidden_dim=16,
        with_prompt=False,
        threshold_controller_cfg={"init_target_ratio": 0.05},  # very aggressive
    )
    content, cu, pos = _make_packed_content(B=1, lengths=[10])
    pinned = torch.zeros(10, dtype=torch.bool)
    pinned[3] = True
    pinned[7] = True

    out = lc(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio=0.05,
        pinned_positions=pinned,
    )
    assert out.survivor_mask[3].item()
    assert out.survivor_mask[7].item()


def test_compression_off_skips_head_and_survive_embedding():
    """target_ratio >= 0.999 also skips the head and survive_embedding scatter."""
    backbone = _StubBackbone(hidden_dim=16)
    lc = _make_lc(
        backbone=backbone, hidden_dim=16,
        with_prompt=False,
    )
    content, cu, pos = _make_packed_content(B=1, lengths=[5])
    out = lc(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio=1.0,
    )
    assert out.base_raw is None
    assert out.survivor_mask.all().item()
    # Hook was not installed — no post-hook capture.
    assert backbone.captured_post_hook is None
