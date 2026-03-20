"""Tests for PrunedBidirectionalQwen35: pruned encoder backbone."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.components.mlp_only_layer import MLPOnlyLayer
from bgkit.models.components.residual_conv1d import ResidualConv1d
from bgkit.models.pruned_qwen35 import PrunedBidirectionalQwen35, PrunedBlock

# ---------------------------------------------------------------------------
# Mock components (same as test_bidirectional_qwen35.py)
# ---------------------------------------------------------------------------

HIDDEN_DIM = 64


class MockLinearAttn(nn.Module):
    def __init__(self, dim=HIDDEN_DIM):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.conv1d = nn.Conv1d(dim, dim, 4, groups=dim, bias=False, padding=3)
        self.causal_conv1d_fn = None
        self.causal_conv1d_update = None


class MockMLP(nn.Module):
    def __init__(self, dim=HIDDEN_DIM):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class MockDeltaNetLayer(nn.Module):
    def __init__(self, dim=HIDDEN_DIM):
        super().__init__()
        self.linear_attn = MockLinearAttn(dim)
        self.input_layernorm = nn.LayerNorm(dim)
        self.post_attention_layernorm = nn.LayerNorm(dim)
        self.mlp = MockMLP(dim)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None):
        return self.linear_attn.proj(hidden_states) + hidden_states


class MockFullAttentionLayer(nn.Module):
    def __init__(self, dim=HIDDEN_DIM):
        super().__init__()
        self.self_attn = nn.Linear(dim, dim)
        self.input_layernorm = nn.LayerNorm(dim)
        self.post_attention_layernorm = nn.LayerNorm(dim)
        self.mlp = MockMLP(dim)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None):
        return self.self_attn(hidden_states) + hidden_states


class MockRotaryEmb(nn.Module):
    def __init__(self, dim=HIDDEN_DIM):
        super().__init__()
        self.dim = dim

    def forward(self, x, position_ids):
        b, seq, d = x.shape
        cos = torch.ones(b, seq, d, device=x.device, dtype=x.dtype)
        sin = torch.zeros(b, seq, d, device=x.device, dtype=x.dtype)
        return cos, sin


class MockConfig:
    hidden_size = HIDDEN_DIM
    num_hidden_layers = 8
    model_type = "qwen3_5_text"


def _make_mock_base_model(num_groups=2):
    """Create mock base model with [D,D,D,A] x num_groups layers."""
    model = nn.Module()
    model.embed_tokens = nn.Embedding(1000, HIDDEN_DIM)
    model.norm = nn.LayerNorm(HIDDEN_DIM)
    model.rotary_emb = MockRotaryEmb()
    model.config = MockConfig()

    layers = []
    for _ in range(num_groups):
        layers.append(MockDeltaNetLayer())
        layers.append(MockDeltaNetLayer())
        layers.append(MockDeltaNetLayer())
        layers.append(MockFullAttentionLayer())
    model.layers = nn.ModuleList(layers)
    return model


def _make_mock_bidi_model(num_groups=2):
    """Create a mock BidirectionalQwen35-like object for from_unpruned()."""
    from bgkit.models.bidirectional_qwen35 import BidirectionalQwen35

    base = _make_mock_base_model(num_groups)
    return BidirectionalQwen35(base, bidi_warmup_steps=0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH = 2
SEQ_LEN = 16


@pytest.fixture
def bidi_model():
    """8-layer [D,D,D,A]x2 mock bidi model."""
    return _make_mock_bidi_model(num_groups=2)


@pytest.fixture
def pruned_model(bidi_model):
    """Pruned model from 8-layer mock (produces 2 blocks: 1 complete + 1 tail)."""
    # 8 layers = [D,D,D,A, D,D,D,A]
    # from_unpruned expects 24 layers for 5 complete + 1 tail, but we can test
    # the mechanism with a smaller model using from_text_model with customization.
    # Instead, let's build a pruned model manually for unit testing.
    conv = ResidualConv1d(HIDDEN_DIM, kernel_size=4)
    mlp1 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), MockMLP())
    mlp2 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), MockMLP())
    attn = MockFullAttentionLayer()
    block_complete = PrunedBlock(conv, mlp1, mlp2, attn)

    conv2 = ResidualConv1d(HIDDEN_DIM, kernel_size=4)
    mlp3 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), MockMLP())
    mlp4 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), MockMLP())
    block_tail = PrunedBlock(conv2, mlp3, mlp4, None)

    return PrunedBidirectionalQwen35(
        embed_tokens=nn.Embedding(1000, HIDDEN_DIM),
        norm=nn.LayerNorm(HIDDEN_DIM),
        rotary_emb=MockRotaryEmb(),
        blocks=nn.ModuleList([block_complete, block_tail]),
        bidi_warmup_steps=0,
    )


# ---------------------------------------------------------------------------
# Tests: PrunedBlock
# ---------------------------------------------------------------------------

class TestPrunedBlock:
    def test_complete_block_has_attention(self):
        conv = ResidualConv1d(HIDDEN_DIM, kernel_size=4)
        mlp1 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), MockMLP())
        mlp2 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), MockMLP())
        attn = MockFullAttentionLayer()
        block = PrunedBlock(conv, mlp1, mlp2, attn)
        assert block.has_attention is True

    def test_tail_block_no_attention(self):
        conv = ResidualConv1d(HIDDEN_DIM, kernel_size=4)
        mlp1 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), MockMLP())
        mlp2 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), MockMLP())
        block = PrunedBlock(conv, mlp1, mlp2, None)
        assert block.has_attention is False

    def test_forward_shape(self):
        conv = ResidualConv1d(HIDDEN_DIM, kernel_size=4)
        mlp1 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), MockMLP())
        mlp2 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), MockMLP())
        attn = MockFullAttentionLayer()
        block = PrunedBlock(conv, mlp1, mlp2, attn)

        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = block(x, position_embeddings=(torch.ones(BATCH, SEQ_LEN, HIDDEN_DIM),
                                             torch.zeros(BATCH, SEQ_LEN, HIDDEN_DIM)))
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)


# ---------------------------------------------------------------------------
# Tests: PrunedBidirectionalQwen35
# ---------------------------------------------------------------------------

class TestPrunedBidirectionalQwen35:
    def test_forward_shape(self, pruned_model):
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.bool)
        out = pruned_model(x, attention_mask=mask)
        assert out.last_hidden_state.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_forward_no_mask(self, pruned_model):
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = pruned_model(x)
        assert out.last_hidden_state.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_return_intermediates(self, pruned_model):
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = pruned_model(x, return_intermediates=True)
        assert out.hidden_states is not None
        # 2 blocks = 2 intermediates
        assert len(out.hidden_states) == 2
        for h in out.hidden_states:
            assert h.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_no_intermediates_by_default(self, pruned_model):
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = pruned_model(x, return_intermediates=False)
        assert out.hidden_states is None

    def test_bidi_warmup(self, pruned_model):
        pruned_model.bidi_warmup_steps = 100
        pruned_model._step.fill_(0)
        assert pruned_model.bidi_alpha == 0.0

        pruned_model._step.fill_(50)
        assert pruned_model.bidi_alpha == 0.5

        pruned_model._step.fill_(100)
        assert pruned_model.bidi_alpha == 1.0

    def test_bidi_immediate(self, pruned_model):
        pruned_model.bidi_warmup_steps = 0
        assert pruned_model.bidi_alpha == 1.0

    def test_step_bidi_warmup(self, pruned_model):
        pruned_model.bidi_warmup_steps = 10
        pruned_model._step.fill_(0)
        pruned_model.step_bidi_warmup()
        assert pruned_model._step.item() == 1


class TestFreezeStage:
    def test_stage_0_only_conv(self, pruned_model):
        pruned_model.freeze_stage(0)
        for block in pruned_model.blocks:
            # Conv should be trainable
            for p in block.conv.parameters():
                assert p.requires_grad
            # MLPs should be frozen
            for p in block.mlp_retrained.parameters():
                assert not p.requires_grad
            for p in block.mlp_frozen.parameters():
                assert not p.requires_grad

    def test_stage_1_conv_and_retrained(self, pruned_model):
        pruned_model.freeze_stage(1)
        for block in pruned_model.blocks:
            for p in block.conv.parameters():
                assert p.requires_grad
            for p in block.mlp_retrained.parameters():
                assert p.requires_grad
            for p in block.mlp_frozen.parameters():
                assert not p.requires_grad

    def test_stage_2_all_mlps(self, pruned_model):
        pruned_model.freeze_stage(2)
        for block in pruned_model.blocks:
            for p in block.conv.parameters():
                assert p.requires_grad
            for p in block.mlp_retrained.parameters():
                assert p.requires_grad
            for p in block.mlp_frozen.parameters():
                assert p.requires_grad
            # FullAttn still frozen
            if block.has_attention:
                for p in block.full_attn_layer.parameters():
                    assert not p.requires_grad

    def test_stage_3_everything(self, pruned_model):
        pruned_model.freeze_stage(3)
        for p in pruned_model.parameters():
            assert p.requires_grad


class TestStateDict:
    def test_roundtrip(self, pruned_model):
        """State dict save/load roundtrip."""
        state = pruned_model.state_dict()
        # Create a fresh model with same structure
        conv = ResidualConv1d(HIDDEN_DIM, kernel_size=4)
        mlp1 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), MockMLP())
        mlp2 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), MockMLP())
        attn = MockFullAttentionLayer()
        block_complete = PrunedBlock(conv, mlp1, mlp2, attn)

        conv2 = ResidualConv1d(HIDDEN_DIM, kernel_size=4)
        mlp3 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), MockMLP())
        mlp4 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), MockMLP())
        block_tail = PrunedBlock(conv2, mlp3, mlp4, None)

        new_model = PrunedBidirectionalQwen35(
            embed_tokens=nn.Embedding(1000, HIDDEN_DIM),
            norm=nn.LayerNorm(HIDDEN_DIM),
            rotary_emb=MockRotaryEmb(),
            blocks=nn.ModuleList([block_complete, block_tail]),
            bidi_warmup_steps=0,
        )
        new_model.load_state_dict(state)

        # Verify outputs match
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        pruned_model.eval()
        new_model.eval()
        with torch.no_grad():
            out1 = pruned_model(x).last_hidden_state
            out2 = new_model(x).last_hidden_state
        torch.testing.assert_close(out1, out2)


class TestFromUnpruned:
    """Test from_unpruned with a full 24-layer mock model."""

    @pytest.fixture
    def full_bidi_model(self):
        return _make_mock_bidi_model(num_groups=6)

    def test_from_unpruned_block_count(self, full_bidi_model):
        pruned = PrunedBidirectionalQwen35.from_unpruned(
            full_bidi_model, hidden_dim=HIDDEN_DIM, conv_kernel_size=4,
        )
        # 5 complete blocks + 1 tail = 6
        assert len(pruned.blocks) == 6

    def test_complete_blocks_have_attention(self, full_bidi_model):
        pruned = PrunedBidirectionalQwen35.from_unpruned(
            full_bidi_model, hidden_dim=HIDDEN_DIM, conv_kernel_size=4,
        )
        for block in pruned.blocks[:5]:
            assert block.has_attention is True

    def test_tail_block_no_attention(self, full_bidi_model):
        pruned = PrunedBidirectionalQwen35.from_unpruned(
            full_bidi_model, hidden_dim=HIDDEN_DIM, conv_kernel_size=4,
        )
        assert pruned.blocks[5].has_attention is False

    def test_forward_after_pruning(self, full_bidi_model):
        pruned = PrunedBidirectionalQwen35.from_unpruned(
            full_bidi_model, hidden_dim=HIDDEN_DIM, conv_kernel_size=4,
        )
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = pruned(x)
        assert out.last_hidden_state.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_mlp_weights_from_correct_layers(self, full_bidi_model):
        """Verify MLP weights come from D1 (retrained) and D2 (frozen) layers."""
        # Record original D1 and D2 MLP weights for first block
        d1_weight = full_bidi_model.layers[1].mlp.linear.weight.data.clone()
        d2_weight = full_bidi_model.layers[2].mlp.linear.weight.data.clone()

        pruned = PrunedBidirectionalQwen35.from_unpruned(
            full_bidi_model, hidden_dim=HIDDEN_DIM, conv_kernel_size=4,
        )

        # First block's retrained MLP should have D1's weights
        torch.testing.assert_close(
            pruned.blocks[0].mlp_retrained.mlp.linear.weight.data, d1_weight,
        )
        # First block's frozen MLP should have D2's weights
        torch.testing.assert_close(
            pruned.blocks[0].mlp_frozen.mlp.linear.weight.data, d2_weight,
        )

    def test_conv1d_fresh_init(self, full_bidi_model):
        """Conv1d should have fresh (not zero) weights."""
        pruned = PrunedBidirectionalQwen35.from_unpruned(
            full_bidi_model, hidden_dim=HIDDEN_DIM, conv_kernel_size=4,
        )
        for block in pruned.blocks:
            assert not torch.all(block.conv.conv.weight == 0)

    def test_embed_tokens_preserved(self, full_bidi_model):
        orig_weight = full_bidi_model.embed_tokens.weight.data.clone()
        pruned = PrunedBidirectionalQwen35.from_unpruned(
            full_bidi_model, hidden_dim=HIDDEN_DIM, conv_kernel_size=4,
        )
        torch.testing.assert_close(pruned.embed_tokens.weight.data, orig_weight)

    def test_config_preserved(self, full_bidi_model):
        pruned = PrunedBidirectionalQwen35.from_unpruned(
            full_bidi_model, hidden_dim=HIDDEN_DIM, conv_kernel_size=4,
        )
        assert pruned.config is not None
        assert pruned.config.hidden_size == HIDDEN_DIM


class TestFromTextModel:
    """Test from_text_model (fresh construction from raw HF layers)."""

    @pytest.fixture
    def text_model_24(self):
        """24-layer mock text model (before projection layer is popped)."""
        return _make_mock_base_model(num_groups=6)

    def test_from_text_model_block_count(self, text_model_24):
        # Pop last layer (projection) before calling from_text_model
        del text_model_24.layers[-1]
        pruned = PrunedBidirectionalQwen35.from_text_model(
            text_model_24, hidden_dim=HIDDEN_DIM, conv_kernel_size=4,
        )
        assert len(pruned.blocks) == 6

    def test_from_text_model_forward(self, text_model_24):
        del text_model_24.layers[-1]
        pruned = PrunedBidirectionalQwen35.from_text_model(
            text_model_24, hidden_dim=HIDDEN_DIM, conv_kernel_size=4,
        )
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = pruned(x)
        assert out.last_hidden_state.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_from_text_model_config(self, text_model_24):
        del text_model_24.layers[-1]
        pruned = PrunedBidirectionalQwen35.from_text_model(
            text_model_24, hidden_dim=HIDDEN_DIM, conv_kernel_size=4,
        )
        assert pruned.config is not None
        assert pruned.config.model_type == "qwen3_5_text"

    def test_from_text_model_too_few_layers(self):
        """Should raise if fewer than 23 layers."""
        model = _make_mock_base_model(num_groups=2)  # 8 layers
        with pytest.raises(ValueError, match="Expected at least 23"):
            PrunedBidirectionalQwen35.from_text_model(
                model, hidden_dim=HIDDEN_DIM, conv_kernel_size=4,
            )

    def test_state_dict_compatible_with_from_unpruned(self, text_model_24):
        """State dict from from_text_model should have the same keys as from_unpruned."""
        from bgkit.models.bidirectional_qwen35 import BidirectionalQwen35

        # Path A: from_text_model (with norm set to Identity, like _from_pretrained_pruned)
        model_a_src = _make_mock_base_model(num_groups=6)
        del model_a_src.layers[-1]
        pruned_a = PrunedBidirectionalQwen35.from_text_model(
            model_a_src, hidden_dim=HIDDEN_DIM, conv_kernel_size=4,
        )
        pruned_a.norm = nn.Identity()  # matches _from_pretrained_pruned behavior

        # Path B: from_unpruned (norm already Identity because BidirectionalQwen35
        # wrapping + _set_norm_to_identity is what from_pretrained does)
        model_b_src = _make_mock_base_model(num_groups=6)
        bidi = BidirectionalQwen35(model_b_src, bidi_warmup_steps=0)
        # Simulate what from_pretrained does: pop last layer, set norm to Identity
        del bidi.layers[-1]
        bidi.norm = nn.Identity()
        pruned_b = PrunedBidirectionalQwen35.from_unpruned(
            bidi, hidden_dim=HIDDEN_DIM, conv_kernel_size=4,
        )

        keys_a = set(pruned_a.state_dict().keys())
        keys_b = set(pruned_b.state_dict().keys())
        assert keys_a == keys_b, (
            f"Key mismatch:\n  A-B: {keys_a - keys_b}\n  B-A: {keys_b - keys_a}"
        )
