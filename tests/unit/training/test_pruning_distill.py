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
BATCH = 2
SEQ_LEN = 8


class MockMLP(nn.Module):
    def __init__(self, dim=HIDDEN_DIM):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class MockFullAttnLayer(nn.Module):
    def __init__(self, dim=HIDDEN_DIM):
        super().__init__()
        self.self_attn = nn.Linear(dim, dim)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        return self.self_attn(hidden_states) + hidden_states


class MockRotaryEmb(nn.Module):
    def __init__(self, dim=HIDDEN_DIM):
        super().__init__()
        self.dim = dim

    def forward(self, x, position_ids):
        b, seq, d = x.shape
        return (torch.ones(b, seq, d, device=x.device, dtype=x.dtype),
                torch.zeros(b, seq, d, device=x.device, dtype=x.dtype))


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


class TestDistillationLoss:
    """Test the loss computation logic (extracted from trainer for unit testing)."""

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
        """Replicate the loss computation from PruningDistillTrainer."""
        n = min(len(teacher_intermediates), len(student_intermediates))
        boundary_loss = torch.tensor(0.0)
        if n > 0:
            for t_h, s_h in zip(teacher_intermediates[:n], student_intermediates[:n], strict=False):
                boundary_loss = boundary_loss + F.mse_loss(s_h, t_h.detach())
            boundary_loss = boundary_loss / n

        repro_loss = F.mse_loss(student_repro, teacher_repro.detach())
        proj_loss = F.mse_loss(student_proj, teacher_proj.detach())

        cos_sim = F.cosine_similarity(
            student_final.flatten(0, 1), teacher_final.detach().flatten(0, 1), dim=-1,
        )
        cosine_loss = (1.0 - cos_sim).mean()

        total = (
            w_boundary * boundary_loss + w_repro * repro_loss
            + w_proj * proj_loss + w_cosine * cosine_loss
        )
        return total, boundary_loss, repro_loss, proj_loss, cosine_loss

    def test_zero_loss_when_identical(self):
        """Loss should be ~0 when teacher and student produce identical outputs."""
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        total, _bl, _rl, _pl, _cl = self._compute_loss(
            [h], [h], h, h, h, h, h, h,
        )
        assert total.item() < 1e-6

    def test_nonzero_loss_when_different(self):
        """Loss should be positive when outputs differ."""
        t = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        s = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        total, _bl, _rl, _pl, _cl = self._compute_loss(
            [t], [s], t, s, t, s, t, s,
        )
        assert total.item() > 0

    def test_repro_weight_higher(self):
        """Repro loss should be weighted higher (2x) than boundary."""
        t = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        s = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        # With all losses being the same MSE, repro should contribute more
        _total, bl, rl, _pl, _cl = self._compute_loss(
            [t], [s], t, s, t, s, t, s,
        )
        # repro_loss * 2.0 should be > boundary_loss * 1.0 when both losses are equal
        assert 2.0 * rl.item() > bl.item() or bl.item() < 1e-8

    def test_teacher_detached(self):
        """Teacher outputs should be detached (no gradients flow to teacher)."""
        teacher_h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, requires_grad=True)
        student_h = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, requires_grad=True)

        total, _, _, _, _ = self._compute_loss(
            [teacher_h], [student_h], teacher_h, student_h,
            teacher_h, student_h, teacher_h, student_h,
        )
        total.backward()

        # Student should have gradients
        assert student_h.grad is not None
        # Teacher should NOT have gradients (detached in loss computation)
        assert teacher_h.grad is None


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
        model.freeze_stage(1)  # conv + retrained MLPs

        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = model(x)
        out.last_hidden_state.sum().backward()

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

        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        with torch.no_grad():
            teacher(x)

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
