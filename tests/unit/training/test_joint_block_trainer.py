"""Tests for JointBlockTrainer: train_step, freeze, evaluate, checkpoint (packed).

CPU-tensor mocks go through the CPU-SDPA branch of ``_packed_full_attention``
(see ``bgkit.models.bidirectional_qwen35._cpu_sdpa_packed``). GPU production
paths always route through FA4.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.bgkit_compressor import BgKITCompressor
from bgkit.models.encoder import BgKITEncoder, _resolve_layers
from bgkit.models.projection_block import ProjectionBlock
from bgkit.training.joint_block_trainer import JointBlockTrainer, joint_block_collate_fn

# ---------------------------------------------------------------------------
# Mock components — packed FA4 API
# ---------------------------------------------------------------------------


class _Output:
    def __init__(self, last_hidden_state, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class MockBackbone(nn.Module):
    """Minimal backbone that accepts the packed (cu_seqlens / position_ids) API."""

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norm = nn.Identity()
        self.rotary_emb = MockRotaryEmb(hidden_dim)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        inputs_embeds=None,
        cu_seqlens=None,
        max_seqlen=None,
        position_ids=None,
        return_intermediates=False,
        layer_hooks=None,
        **kwargs,
    ):
        """Accept packed args; ignore cu_seqlens/position_ids for mock purposes."""
        x = inputs_embeds  # (N_total, D) packed
        all_hidden = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if return_intermediates:
                all_hidden.append(x)
            if layer_hooks and i in layer_hooks:
                x = layer_hooks[i](x)
        x = self.norm(x)
        hidden_states = all_hidden if return_intermediates else None
        return _Output(last_hidden_state=x, hidden_states=hidden_states)


class MockSelfAttn(nn.Module):
    """Mock self-attention that skips real attention — just projects through o_proj."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        # _packed_full_attention infers heads from proj weight shapes.
        # We use head_dim=8, n_heads=8 so hidden_dim=64.
        head_dim = 8
        n_heads = hidden_dim // head_dim
        # q_proj doubles output for gate split
        self.q_proj = nn.Linear(hidden_dim, n_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5


class MockTransformerLayer(nn.Module):
    """Mock Qwen3.5-style decoder layer used by ProjectionBlock."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = MockSelfAttn(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))


class MockRotaryEmb(nn.Module):
    """Minimal rotary embedding mock returning unit cos/sin for packed (N, D) input.

    apply_rotary_pos_emb splits q/k along the last dim into rot and pass halves,
    so the rotary_dim must equal head_dim // 2.  For MockSelfAttn: head_dim=8,
    so rotary_dim=4.
    """

    HEAD_DIM = 8

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.dim = hidden_dim
        # Must match head_dim // 2 in MockSelfAttn.
        self.rotary_dim = self.HEAD_DIM // 2

    def forward(self, x, position_ids):
        # x: (N, D) packed; position_ids: (1, N) — return (1, num_tokens, rotary_dim)
        num_tokens = position_ids.shape[-1]
        cos = torch.ones(1, num_tokens, self.rotary_dim, device=x.device, dtype=x.dtype)
        sin = torch.zeros(1, num_tokens, self.rotary_dim, device=x.device, dtype=x.dtype)
        return cos, sin


def _make_encoder(hidden_dim: int = 64, num_layers: int = 3) -> BgKITEncoder:
    backbone = MockBackbone(vocab_size=1000, hidden_dim=hidden_dim, num_layers=num_layers)
    compressor_norm = nn.LayerNorm(hidden_dim)
    compressor = BgKITCompressor(
        backbone, compressor_norm, hidden_dim=hidden_dim, survivorship_inner_dim=8
    )

    proj_layer = MockTransformerLayer(hidden_dim)
    proj_norm = nn.LayerNorm(hidden_dim)
    rotary = MockRotaryEmb(hidden_dim)
    projection_block = ProjectionBlock(proj_layer, proj_norm, rotary, hidden_dim=hidden_dim)

    return BgKITEncoder(compressor, projection_block)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def trainer():
    """Create a JointBlockTrainer with mock components (no setup() call)."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "training": {
            "phase": "joint_block_pretrain",
            "max_steps": 100,
            "lr": 1e-3,
            "warmup_steps": 10,
            "eval_every": 50,
            "save_every": 0,
            "w_repro": 1.0,
            "w_proj": 1.0,
        },
        "wandb": {"enabled": False},
    })

    hidden_dim = 64
    t = JointBlockTrainer(cfg)
    t.device = torch.device("cpu")

    t.encoder = _make_encoder(hidden_dim=hidden_dim, num_layers=3)
    t.model = t.encoder
    t.w_repro = 1.0
    t.w_proj = 1.0

    # Decoder embeddings (frozen reference target)
    t.decoder_embed = nn.Embedding(1000, hidden_dim)
    t.decoder_embed.requires_grad_(False)

    # Freeze everything, then unfreeze targets
    t.encoder.requires_grad_(False)

    # Unfreeze penultimate layer (last in compressor backbone)
    compressor_layers = _resolve_layers(t.encoder.compressor.backbone)
    compressor_layers[-1].requires_grad_(True)

    # Unfreeze compressor norm + auto_repro_head + projection block
    t.encoder.compressor.norm.requires_grad_(True)
    t.encoder.compressor.auto_repro_head.requires_grad_(True)
    t.encoder.projection_block.requires_grad_(True)

    trainable_params = [p for p in t.encoder.parameters() if p.requires_grad]
    t.optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    return t


def _make_batch(batch_size: int = 2, seq_len: int = 20):
    """Create a collated packed batch of random token IDs."""
    samples = [{"token_ids": torch.randint(0, 1000, (seq_len,))} for _ in range(batch_size)]
    return joint_block_collate_fn(samples)


def _make_variable_batch(lengths: list[int]):
    """Create a packed batch with explicitly varying sequence lengths."""
    samples = [{"token_ids": torch.randint(0, 1000, (L,))} for L in lengths]
    return joint_block_collate_fn(samples)


# ---------------------------------------------------------------------------
# Train step tests
# ---------------------------------------------------------------------------


class TestJointBlockTrainStep:
    def test_returns_expected_metrics(self, trainer):
        """train_step should return all expected metric keys."""
        metrics = trainer.train_step(_make_batch())
        assert "loss" in metrics
        assert "loss_repro" in metrics
        assert "loss_proj" in metrics
        assert "cosine_sim_repro" in metrics
        assert "cosine_sim_proj" in metrics
        assert "grad_norm" in metrics

    def test_loss_is_finite(self, trainer):
        """Loss should be a finite number."""
        metrics = trainer.train_step(_make_batch())
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_both_losses_positive(self, trainer):
        """Both loss components should be positive."""
        metrics = trainer.train_step(_make_batch())
        assert metrics["loss_repro"] > 0
        assert metrics["loss_proj"] > 0

    def test_loss_decreases_over_steps(self, trainer):
        """Loss should decrease over multiple training steps on the same data."""
        torch.manual_seed(42)
        batch = _make_batch(batch_size=4, seq_len=30)

        losses = []
        for _ in range(50):
            metrics = trainer.train_step(batch)
            losses.append(metrics["loss"])

        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, f"Loss didn't decrease: {early_avg:.4f} -> {late_avg:.4f}"


# ---------------------------------------------------------------------------
# Freeze tests
# ---------------------------------------------------------------------------


class TestJointBlockFreeze:
    def test_only_target_components_unfrozen(self, trainer):
        """Only penultimate layer, norm, auto_repro_head, and projection block trainable."""
        # Compressor embedding should be frozen
        for p in trainer.encoder.compressor.backbone.embed_tokens.parameters():
            assert not p.requires_grad, "Embedding should be frozen"

        # Earlier layers should be frozen
        compressor_layers = _resolve_layers(trainer.encoder.compressor.backbone)
        for i in range(len(compressor_layers) - 1):
            for p in compressor_layers[i].parameters():
                assert not p.requires_grad, f"Layer {i} should be frozen"

        # Penultimate layer (last in compressor) should be unfrozen
        for p in compressor_layers[-1].parameters():
            assert p.requires_grad, "Penultimate layer should be trainable"

        # Compressor norm should be unfrozen
        for p in trainer.encoder.compressor.norm.parameters():
            assert p.requires_grad, "Compressor norm should be trainable"

        # Auto-repro head should be unfrozen
        for p in trainer.encoder.compressor.auto_repro_head.parameters():
            assert p.requires_grad, "Auto-repro head should be trainable"

        # Projection block should be unfrozen
        for p in trainer.encoder.projection_block.parameters():
            assert p.requires_grad, "Projection block should be trainable"

        # Survive embedding should be frozen
        assert not trainer.encoder.compressor.survive_embedding.requires_grad


# ---------------------------------------------------------------------------
# Evaluate tests
# ---------------------------------------------------------------------------


class TestJointBlockEvaluate:
    def test_returns_expected_metrics(self, trainer):
        """evaluate() should return mse and cosine_sim for both objectives."""
        from torch.utils.data import DataLoader

        eval_data = [
            {"token_ids": torch.randint(0, 1000, (10,))},
            {"token_ids": torch.randint(0, 1000, (8,))},
        ]
        trainer.eval_dataloader = DataLoader(
            eval_data, batch_size=2, collate_fn=joint_block_collate_fn,
        )

        metrics = trainer.evaluate()
        assert "mse_repro" in metrics
        assert "mse_proj" in metrics
        assert "cosine_sim_repro" in metrics
        assert "cosine_sim_proj" in metrics


# ---------------------------------------------------------------------------
# ChatML prefix tests
# ---------------------------------------------------------------------------


def _set_chatml_prefix(trainer, prefix_len: int = 3):
    """Pre-compute and cache prompt embeddings on a trainer, mirroring setup()."""
    prefix_ids = torch.randint(0, 1000, (prefix_len,), dtype=torch.long)
    with torch.no_grad():
        trainer._prompt_embeddings = trainer._get_input_embeddings(prefix_ids)
        trainer._prompt_len = prefix_len


class TestJointBlockChatMLPrefix:
    def test_train_step_with_chatml_prefix(self, trainer):
        """train_step with ChatML prefix should produce finite loss without shape mismatch."""
        _set_chatml_prefix(trainer)

        batch = _make_batch(batch_size=2, seq_len=20)
        metrics = trainer.train_step(batch)
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert "loss_repro" in metrics
        assert "loss_proj" in metrics
        assert "cosine_sim_repro" in metrics
        assert "cosine_sim_proj" in metrics

    def test_evaluate_with_chatml_prefix(self, trainer):
        """evaluate() with ChatML prefix should return finite metrics."""
        from torch.utils.data import DataLoader

        _set_chatml_prefix(trainer)

        eval_data = [
            {"token_ids": torch.randint(0, 1000, (10,))},
            {"token_ids": torch.randint(0, 1000, (8,))},
        ]
        trainer.eval_dataloader = DataLoader(
            eval_data, batch_size=2, collate_fn=joint_block_collate_fn,
        )

        metrics = trainer.evaluate()
        assert "mse_repro" in metrics
        assert "mse_proj" in metrics
        assert "cosine_sim_repro" in metrics
        assert "cosine_sim_proj" in metrics
        for k, v in metrics.items():
            assert torch.isfinite(torch.tensor(v)), f"{k} is not finite: {v}"

    def test_loss_decreases_with_chatml_prefix(self, trainer):
        """Loss should decrease over steps even with ChatML prefix."""
        torch.manual_seed(42)
        _set_chatml_prefix(trainer)

        batch = _make_batch(batch_size=4, seq_len=30)
        losses = []
        for _ in range(50):
            metrics = trainer.train_step(batch)
            losses.append(metrics["loss"])

        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, f"Loss didn't decrease: {early_avg:.4f} -> {late_avg:.4f}"


# ---------------------------------------------------------------------------
# Collation tests (packed semantics)
# ---------------------------------------------------------------------------


class TestJointBlockCollateFn:
    def test_packed_output_keys(self):
        """Collated batch should contain packed keys, no attention_mask."""
        batch = [
            {"token_ids": torch.tensor([1, 2, 3])},
            {"token_ids": torch.tensor([4, 5, 6, 7, 8])},
        ]
        out = joint_block_collate_fn(batch)
        assert "input_ids" in out
        assert "position_ids" in out
        assert "cu_seqlens" in out
        assert "max_seqlen" in out
        assert "attention_mask" not in out
        assert "token_ids" not in out

    def test_flat_concatenation(self):
        """input_ids should be flat concatenation of all samples."""
        s1 = torch.tensor([1, 2, 3])
        s2 = torch.tensor([4, 5, 6, 7, 8])
        batch = [{"token_ids": s1}, {"token_ids": s2}]
        out = joint_block_collate_fn(batch)
        assert out["input_ids"].shape == (8,)
        assert out["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_cu_seqlens_correct(self):
        """cu_seqlens should encode sample boundaries correctly."""
        batch = [
            {"token_ids": torch.tensor([1, 2, 3])},
            {"token_ids": torch.tensor([4, 5])},
            {"token_ids": torch.tensor([6, 7, 8, 9])},
        ]
        out = joint_block_collate_fn(batch)
        assert out["cu_seqlens"].tolist() == [0, 3, 5, 9]
        assert out["cu_seqlens"].dtype == torch.int32

    def test_position_ids_per_sample_restart(self):
        """position_ids should restart at 0 for each sample."""
        batch = [
            {"token_ids": torch.tensor([10, 20, 30])},
            {"token_ids": torch.tensor([40, 50])},
        ]
        out = joint_block_collate_fn(batch)
        assert out["position_ids"].tolist() == [0, 1, 2, 0, 1]
        assert out["position_ids"].dtype == torch.int64

    def test_max_seqlen_is_max_length(self):
        """max_seqlen should equal the length of the longest sample."""
        batch = [
            {"token_ids": torch.tensor([1, 2, 3])},
            {"token_ids": torch.tensor([4, 5, 6, 7, 8])},
        ]
        out = joint_block_collate_fn(batch)
        assert out["max_seqlen"] == 5

    def test_variable_length_samples(self):
        """Variable-length samples should be packed without padding."""
        lengths = [3, 7, 1, 5]
        samples = [{"token_ids": torch.randint(0, 100, (L,))} for L in lengths]
        out = joint_block_collate_fn(samples)
        assert out["input_ids"].shape == (sum(lengths),)
        assert out["cu_seqlens"][-1].item() == sum(lengths)
        assert out["max_seqlen"] == max(lengths)
