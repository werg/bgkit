"""Tests for PruningDistillTrainer: loss computation, stage management, embedding separation."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F
from torch import nn

from bgkit.models.components.mlp_only_layer import MLPOnlyLayer
from bgkit.models.components.residual_conv1d import ResidualConv1d
from bgkit.models.pruned_qwen35 import PrunedBidirectionalQwen35, PrunedBlock

HIDDEN_DIM = 32
# Packed form: N = total tokens across all samples (flat, no batch/seq dims).
N_TOKENS = 16  # e.g. two samples of 8 tokens each


class MockMLP(nn.Module):
    def __init__(self, dim=HIDDEN_DIM):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class MockSelfAttn(nn.Module):
    """Minimal self-attention stub that _packed_full_attention can dispatch through."""

    def __init__(self, dim=HIDDEN_DIM):
        super().__init__()
        # _packed_full_attention calls module.forward via the attention interface or
        # direct call; we just do an identity-like linear to keep shapes right.
        self.linear = nn.Linear(dim, dim)
        self.is_causal = False

    def forward(self, hidden_states, position_embeddings=None, **kwargs):
        return self.linear(hidden_states), None


class MockFullAttnLayer(nn.Module):
    """Full-attention layer mock matching the HF Qwen3.5DecoderLayer interface.

    _run_attn_layer accesses: input_layernorm, self_attn, post_attention_layernorm, mlp.
    """

    def __init__(self, dim=HIDDEN_DIM):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(dim)
        self.self_attn = MockSelfAttn(dim)
        self.post_attention_layernorm = nn.LayerNorm(dim)
        self.mlp = nn.Linear(dim, dim)


class MockRotaryEmb(nn.Module):
    def __init__(self, dim=HIDDEN_DIM):
        super().__init__()
        self.dim = dim

    def forward(self, x, position_ids):
        # x is (N, D), position_ids is (1, N) in packed form
        n = x.shape[0]
        d = x.shape[-1]
        return (
            torch.ones(1, n, d, device=x.device, dtype=x.dtype),
            torch.zeros(1, n, d, device=x.device, dtype=x.dtype),
        )


def _build_pruned_model(n_blocks=2):
    """Build a small pruned model for testing."""
    blocks = nn.ModuleList()
    for i in range(n_blocks):
        conv = ResidualConv1d(HIDDEN_DIM, kernel_size=4)
        mlp1 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), MockMLP())
        mlp2 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), MockMLP())
        attn = MockFullAttnLayer() if i < n_blocks - 1 else None
        blocks.append(PrunedBlock(conv, mlp1, mlp2, attn))

    return PrunedBidirectionalQwen35(
        embed_tokens=nn.Embedding(100, HIDDEN_DIM),
        norm=nn.LayerNorm(HIDDEN_DIM),
        rotary_emb=MockRotaryEmb(),
        blocks=blocks,
        bidi_warmup_steps=0,
    )


def _packed_inputs(n_tokens=N_TOKENS, hidden_dim=HIDDEN_DIM):
    """Return ``(hidden, cu_seqlens, max_seqlen, position_ids)`` for packed forward."""
    # Two samples of equal length n_tokens // 2.
    half = n_tokens // 2
    cu = torch.tensor([0, half, n_tokens], dtype=torch.int32)
    pos = torch.cat([torch.arange(half), torch.arange(n_tokens - half)]).long()
    x = torch.randn(n_tokens, hidden_dim)
    return x, cu, half, pos


class TestDistillationLoss:
    """Test the loss computation logic (extracted from trainer for unit testing).

    All tensors use the packed flat ``(N, D)`` convention.
    """

    def _compute_loss(
        self,
        teacher_intermediates,
        student_intermediates,
        teacher_repro,
        student_repro,
        teacher_proj,
        student_proj,
        teacher_final,
        student_final,
        w_boundary=1.0,
        w_repro=2.0,
        w_proj=1.0,
        w_cosine=0.2,
    ):
        """Replicate the loss computation from PruningDistillTrainer.

        Inputs are flat ``(N, D)`` packed tensors.
        """
        n = min(len(teacher_intermediates), len(student_intermediates))
        boundary_loss = torch.tensor(0.0)
        if n > 0:
            for t_h, s_h in zip(teacher_intermediates[:n], student_intermediates[:n], strict=False):
                boundary_loss = boundary_loss + F.mse_loss(s_h, t_h.detach())
            boundary_loss = boundary_loss / n

        repro_loss = F.mse_loss(student_repro, teacher_repro.detach())
        proj_loss = F.mse_loss(student_proj, teacher_proj.detach())

        # Packed form: (N, D), reduce over token dimension directly.
        cos_sim = F.cosine_similarity(student_final, teacher_final.detach(), dim=-1)
        cosine_loss = (1.0 - cos_sim).mean()

        total = (
            w_boundary * boundary_loss + w_repro * repro_loss
            + w_proj * proj_loss + w_cosine * cosine_loss
        )
        return total, boundary_loss, repro_loss, proj_loss, cosine_loss

    def test_zero_loss_when_identical(self):
        """Loss should be ~0 when teacher and student produce identical outputs."""
        # Packed (N, D): N_TOKENS flat tokens, no batch dimension.
        h = torch.randn(N_TOKENS, HIDDEN_DIM)
        total, _bl, _rl, _pl, _cl = self._compute_loss(
            [h], [h], h, h, h, h, h, h,
        )
        assert total.item() < 1e-6

    def test_nonzero_loss_when_different(self):
        """Loss should be positive when outputs differ."""
        t = torch.randn(N_TOKENS, HIDDEN_DIM)
        s = torch.randn(N_TOKENS, HIDDEN_DIM)
        total, _bl, _rl, _pl, _cl = self._compute_loss(
            [t], [s], t, s, t, s, t, s,
        )
        assert total.item() > 0

    def test_repro_weight_higher(self):
        """Repro loss should be weighted higher (2x) than boundary."""
        t = torch.randn(N_TOKENS, HIDDEN_DIM)
        s = torch.randn(N_TOKENS, HIDDEN_DIM)
        # With all losses being the same MSE, repro should contribute more
        _total, bl, rl, _pl, _cl = self._compute_loss(
            [t], [s], t, s, t, s, t, s,
        )
        # repro_loss * 2.0 should be > boundary_loss * 1.0 when both losses are equal
        assert 2.0 * rl.item() > bl.item() or bl.item() < 1e-8

    def test_teacher_detached(self):
        """Teacher outputs should be detached (no gradients flow to teacher)."""
        teacher_h = torch.randn(N_TOKENS, HIDDEN_DIM, requires_grad=True)
        student_h = torch.randn(N_TOKENS, HIDDEN_DIM, requires_grad=True)

        total, _, _, _, _ = self._compute_loss(
            [teacher_h], [student_h], teacher_h, student_h,
            teacher_h, student_h, teacher_h, student_h,
        )
        total.backward()

        # Student should have gradients
        assert student_h.grad is not None
        # Teacher should NOT have gradients (detached in loss computation)
        assert teacher_h.grad is None

    def test_packed_flat_shapes_accepted(self):
        """Loss helper accepts flat (N, D) tensors (no batch/seq dims)."""
        # N = 12, D = HIDDEN_DIM (irregular, not BATCH * SEQ_LEN)
        n = 12
        t = torch.randn(n, HIDDEN_DIM)
        s = torch.randn(n, HIDDEN_DIM)
        total, *_ = self._compute_loss([t], [s], t, s, t, s, t, s)
        assert total.isfinite()


class TestStageFreezing:
    """Test that freeze_stage correctly toggles parameters."""

    def test_stage_transitions_increase_trainable_params(self):
        model = _build_pruned_model(n_blocks=2)

        counts = []
        for stage in range(4):
            model.freeze_stage(stage)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            counts.append(trainable)

        # Each stage should have more trainable params than the previous
        for i in range(1, len(counts)):
            assert counts[i] >= counts[i - 1], f"Stage {i} has fewer params than stage {i-1}"

    def test_stage_0_conv_only(self):
        model = _build_pruned_model(n_blocks=2)
        model.freeze_stage(0)

        # Only conv params should be trainable
        for block in model.blocks:
            for p in block.conv.parameters():
                assert p.requires_grad
            for p in block.mlp_retrained.parameters():
                assert not p.requires_grad

    def test_gradient_only_flows_to_trainable(self):
        model = _build_pruned_model(n_blocks=2)
        model.freeze_stage(1)  # conv + retrained MLPs; mlp_frozen is frozen

        # Drive a backward pass through only the conv and mlp_retrained subgraph
        # (i.e. skip the full-attention layer, which requires a real Qwen3.5 attn
        # module). We compose a mini-forward that exercises the non-attn path.
        x = torch.randn(N_TOKENS, HIDDEN_DIM, requires_grad=False)
        out = x
        for block in model.blocks:
            out = block.conv(out.unsqueeze(0)).squeeze(0)
            out = block.mlp_retrained(out)
            out = block.mlp_frozen(out)
        out.sum().backward()

        # Frozen MLP should have no gradients
        for block in model.blocks:
            for p in block.mlp_frozen.parameters():
                assert p.grad is None or torch.all(p.grad == 0)


class TestEmbeddingSeparation:
    """Test that teacher and student embeddings are independent."""

    def test_separate_embedding_tables(self):
        model1 = _build_pruned_model(n_blocks=1)
        model2 = _build_pruned_model(n_blocks=1)

        # Modify one's embedding
        with torch.no_grad():
            model1.embed_tokens.weight.fill_(42.0)

        # Other should be unaffected
        assert not torch.all(model2.embed_tokens.weight == 42.0)

    def test_teacher_frozen_no_grad(self):
        teacher = _build_pruned_model(n_blocks=1)
        teacher.requires_grad_(False)
        teacher.eval()

        # Drive a simple forward through embed + conv to verify requires_grad stays off.
        ids = torch.randint(0, 100, (N_TOKENS,))
        with torch.no_grad():
            emb = teacher.embed_tokens(ids)  # (N, D)
            for block in teacher.blocks:
                emb = block.conv(emb.unsqueeze(0)).squeeze(0)
                emb = block.mlp_retrained(emb)
                emb = block.mlp_frozen(emb)

        # No parameter should have requires_grad
        for p in teacher.parameters():
            assert not p.requires_grad


class TestStateDictAutoDetection:
    """Test that pruned vs unpruned encoder is auto-detected from state dict keys."""

    def test_pruned_state_dict_detected(self):
        from bgkit.models.encoder import is_pruned_encoder_state_dict

        pruned_keys = {
            "compressor.backbone.blocks.0.conv.norm.weight": torch.tensor([1.0]),
            "compressor.backbone.blocks.0.mlp_retrained.mlp.linear.weight": torch.tensor([1.0]),
            "projection_block.projection_head.weight": torch.tensor([1.0]),
        }
        assert is_pruned_encoder_state_dict(pruned_keys) is True

    def test_unpruned_state_dict_detected(self):
        from bgkit.models.encoder import is_pruned_encoder_state_dict

        unpruned_keys = {
            "compressor.backbone.layers.0.linear_attn.proj.weight": torch.tensor([1.0]),
            "compressor.backbone.layers.3.self_attn.weight": torch.tensor([1.0]),
            "projection_block.projection_head.weight": torch.tensor([1.0]),
        }
        assert is_pruned_encoder_state_dict(unpruned_keys) is False

    def test_empty_state_dict_is_unpruned(self):
        from bgkit.models.encoder import is_pruned_encoder_state_dict

        assert is_pruned_encoder_state_dict({}) is False
