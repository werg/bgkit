"""Tests for PrunedBidirectionalQwen35.forward_from_block.

Verifies that:
- ``forward_from_block(start_block=0)`` is bit-identical to ``forward()``.
- Resuming from an interior block on a captured intermediate hidden state
  reproduces the same output as a single full forward.
- ``apply_final_norm=False`` yields the pre-norm hidden state.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from bgkit.models.components.mlp_only_layer import MLPOnlyLayer  # noqa: E402
from bgkit.models.components.residual_conv1d import ResidualConv1d  # noqa: E402
from bgkit.models.pruned_qwen35 import (  # noqa: E402
    PrunedBidirectionalQwen35,
    PrunedBlock,
)

HIDDEN_DIM = 64
BATCH = 2
SEQ_LEN = 16


class MockMLP(nn.Module):
    def __init__(self, dim=HIDDEN_DIM):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class MockFullAttentionLayer(nn.Module):
    def __init__(self, dim=HIDDEN_DIM):
        super().__init__()
        self.self_attn = nn.Linear(dim, dim)
        self.input_layernorm = nn.LayerNorm(dim)
        self.post_attention_layernorm = nn.LayerNorm(dim)
        self.mlp = MockMLP(dim)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        return self.self_attn(hidden_states) + hidden_states


class MockRotaryEmb(nn.Module):
    def forward(self, x, position_ids):
        b, seq, d = x.shape
        cos = torch.ones(b, seq, d, device=x.device, dtype=x.dtype)
        sin = torch.zeros(b, seq, d, device=x.device, dtype=x.dtype)
        return cos, sin


def _make_pruned_model(num_blocks: int = 4) -> PrunedBidirectionalQwen35:
    """Build a small pruned model with ``num_blocks`` complete blocks."""
    blocks = []
    for _ in range(num_blocks):
        conv = ResidualConv1d(HIDDEN_DIM, kernel_size=4)
        mlp1 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), MockMLP())
        mlp2 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), MockMLP())
        attn = MockFullAttentionLayer()
        blocks.append(PrunedBlock(conv, mlp1, mlp2, attn))
    return PrunedBidirectionalQwen35(
        embed_tokens=nn.Embedding(1000, HIDDEN_DIM),
        norm=nn.LayerNorm(HIDDEN_DIM),
        rotary_emb=MockRotaryEmb(),
        blocks=nn.ModuleList(blocks),
        bidi_warmup_steps=0,
    )


@pytest.fixture
def pruned():
    torch.manual_seed(0)
    model = _make_pruned_model(num_blocks=4)
    model.eval()
    return model


@pytest.fixture
def inputs():
    torch.manual_seed(1)
    embeds = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
    attn = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
    attn[1, 12:] = 0  # padding on second sample
    return embeds, attn


def test_start_block_zero_matches_forward(pruned, inputs):
    embeds, attn = inputs
    with torch.no_grad():
        ref = pruned(inputs_embeds=embeds, attention_mask=attn).last_hidden_state
        new = pruned.forward_from_block(
            hidden=embeds,
            start_block=0,
            attention_mask=attn,
        ).last_hidden_state
    assert torch.allclose(ref, new, atol=0, rtol=0)


def test_resume_from_intermediate_block(pruned, inputs):
    """Capture hidden after block k-1, resume from block k, expect same output."""
    embeds, attn = inputs
    captured: dict[int, torch.Tensor] = {}

    def hook_after_block_1(h):
        captured[1] = h.clone()
        return h

    with torch.no_grad():
        ref = pruned(
            inputs_embeds=embeds,
            attention_mask=attn,
            layer_hooks={1: hook_after_block_1},
        ).last_hidden_state
        # Resume from block 2 using the captured state — should match ref.
        resumed = pruned.forward_from_block(
            hidden=captured[1],
            start_block=2,
            attention_mask=attn,
        ).last_hidden_state
    assert torch.allclose(ref, resumed, atol=1e-6, rtol=1e-6)


def test_apply_final_norm_false_skips_norm(pruned, inputs):
    embeds, attn = inputs
    with torch.no_grad():
        normed = pruned.forward_from_block(
            hidden=embeds, start_block=0, attention_mask=attn,
            apply_final_norm=True,
        ).last_hidden_state
        unnormed = pruned.forward_from_block(
            hidden=embeds, start_block=0, attention_mask=attn,
            apply_final_norm=False,
        ).last_hidden_state
        # Manually applying norm should bring them into agreement.
        manual = pruned.norm(unnormed)
    assert not torch.allclose(normed, unnormed)
    assert torch.allclose(normed, manual, atol=1e-6)


def test_layer_hook_at_correct_index_when_resumed(pruned, inputs):
    """Hooks fire at absolute block indices, not offsets within the slice."""
    embeds, attn = inputs
    captured = {}

    def make_hook(i):
        def h(x):
            captured[i] = x.clone()
            return x
        return h

    with torch.no_grad():
        pruned.forward_from_block(
            hidden=embeds,
            start_block=2,
            attention_mask=attn,
            layer_hooks={2: make_hook(2), 3: make_hook(3)},
        )
    assert 2 in captured
    assert 3 in captured


def test_invalid_start_block_raises(pruned, inputs):
    embeds, attn = inputs
    with pytest.raises(ValueError):
        pruned.forward_from_block(
            hidden=embeds, start_block=-1, attention_mask=attn,
        )
    with pytest.raises(ValueError):
        pruned.forward_from_block(
            hidden=embeds, start_block=99, attention_mask=attn,
        )


def test_start_block_at_end_returns_normed_input(pruned, inputs):
    """start_block == len(blocks) means no blocks run; just final norm."""
    embeds, attn = inputs
    n_blocks = len(pruned.blocks)
    with torch.no_grad():
        out = pruned.forward_from_block(
            hidden=embeds, start_block=n_blocks, attention_mask=attn,
        ).last_hidden_state
        manual = pruned.norm(embeds)
    assert torch.allclose(out, manual, atol=1e-6)


def _make_pruned_with_tail() -> PrunedBidirectionalQwen35:
    """Build a pruned model with 5 complete + 1 tail block, mirroring the
    real pruned backbone layout (where block 5 has no FullAttn)."""
    torch.manual_seed(0)
    blocks = []
    for i in range(6):
        conv = ResidualConv1d(HIDDEN_DIM, kernel_size=4)
        mlp1 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), MockMLP())
        mlp2 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), MockMLP())
        attn = MockFullAttentionLayer() if i < 5 else None
        from bgkit.models.pruned_qwen35 import PrunedBlock
        blocks.append(PrunedBlock(conv, mlp1, mlp2, attn))
    return PrunedBidirectionalQwen35(
        embed_tokens=nn.Embedding(1000, HIDDEN_DIM),
        norm=nn.LayerNorm(HIDDEN_DIM),
        rotary_emb=MockRotaryEmb(),
        blocks=nn.ModuleList(blocks),
        bidi_warmup_steps=0,
    )


def test_forward_from_block_crosses_tail_boundary():
    """Resuming at block 2 must correctly pass through all subsequent
    blocks including the tail (no FullAttn) at index 5."""
    model = _make_pruned_with_tail()
    model.eval()
    torch.manual_seed(1)
    embeds = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
    attn = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)

    captured: dict[int, torch.Tensor] = {}

    def hook_at_1(h):
        captured[1] = h.clone()
        return h

    with torch.no_grad():
        ref = model(
            inputs_embeds=embeds,
            attention_mask=attn,
            layer_hooks={1: hook_at_1},
        ).last_hidden_state
        resumed = model.forward_from_block(
            hidden=captured[1],
            start_block=2,
            attention_mask=attn,
        ).last_hidden_state
    assert torch.allclose(ref, resumed, atol=1e-6)


def test_forward_from_block_resume_at_tail_only():
    """start_block = 5 (tail block, no attention) should also work."""
    model = _make_pruned_with_tail()
    model.eval()
    torch.manual_seed(2)
    embeds = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
    attn = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
    captured: dict[int, torch.Tensor] = {}

    def hook_at_4(h):
        captured[4] = h.clone()
        return h

    with torch.no_grad():
        ref = model(
            inputs_embeds=embeds, attention_mask=attn,
            layer_hooks={4: hook_at_4},
        ).last_hidden_state
        resumed = model.forward_from_block(
            hidden=captured[4], start_block=5, attention_mask=attn,
        ).last_hidden_state
    assert torch.allclose(ref, resumed, atol=1e-6)
