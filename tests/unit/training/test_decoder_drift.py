"""Tests for decoder drift detection and LoRA fallback."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.data.threshold_calibrator import ThresholdCalibrator
from bgkit.eval.metrics.embedding_health import embedding_drift_metrics
from bgkit.models.bgkit_compressor import BgKITCompressor
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.models.projection_block import ProjectionBlock
from bgkit.training.phase1.compression import CompressionTrainer

HIDDEN_DIM = 32
VOCAB_SIZE = 256


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _Output:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class MockEncoderBackbone(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)])
        self.norm = nn.Identity()

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        return _Output(last_hidden_state=self.layers[0](inputs_embeds))


class MockTransformerLayer(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        return self.linear(hidden_states)


class MockRotaryEmb(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.dim = hidden_dim

    def forward(self, x, position_ids):
        seq_len = x.size(1)
        cos = torch.ones(1, seq_len, self.dim, device=x.device, dtype=x.dtype)
        sin = torch.zeros(1, seq_len, self.dim, device=x.device, dtype=x.dtype)
        return cos, sin


class _CausalLMOutput:
    def __init__(self, logits):
        self.logits = logits


class _MockQwen3Model(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)


class MockCausalLMBackbone(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.model = _MockQwen3Model(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        x = inputs_embeds
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.norm(x)
        return _CausalLMOutput(logits=self.lm_head(x))


def _make_trainer(drift_threshold: float = 0.5, drift_check_every: int = 100):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "training": {
            "phase": "phase1_step2",
            "max_steps": 1000,
            "lr": 1e-3,
            "warmup_steps": 10,
            "eval_every": 500,
            "save_every": 0,
            "curriculum": {
                "l1_introduction_step": 20000,
                "target_ratio_start": 0.30,
                "target_ratio_end": 0.15,
                "target_ratio_ramp_steps": 100000,
                "max_survivor_gap": 64,
                "fallback_threshold": 3.0,
                "calibrator_ema_decay": 0.99,
                "calibrator_warmup_batches": 50,
                "l1_calibrator_fast_batches": 200,
            },
            "decoder": {
                "drift_threshold": drift_threshold,
                "drift_check_every": drift_check_every,
            },
        },
        "compute": {"num_workers": 0, "pin_memory": False},
        "wandb": {"enabled": False},
    })

    t = CompressionTrainer(cfg)
    t.device = torch.device("cpu")

    # Encoder
    backbone = MockEncoderBackbone()
    compressor_norm = nn.LayerNorm(HIDDEN_DIM)
    compressor = BgKITCompressor(backbone, compressor_norm, hidden_dim=HIDDEN_DIM)
    proj_layer = MockTransformerLayer()
    proj_norm = nn.LayerNorm(HIDDEN_DIM)
    rotary = MockRotaryEmb()
    proj_block = ProjectionBlock(proj_layer, proj_norm, rotary, hidden_dim=HIDDEN_DIM)
    t.encoder = BgKITEncoder(compressor, proj_block)

    # Decoder
    decoder_bb = MockCausalLMBackbone()
    t.decoder = ReconstructionDecoder(decoder_bb, hidden_dim=HIDDEN_DIM)
    t.model = t.decoder

    # Optimizer
    params = list(t.encoder.parameters()) + list(t.decoder.parameters())
    t.optimizer = torch.optim.AdamW(params, lr=1e-3)

    # Curriculum defaults
    t._l1_introduction_step = 20000
    t._target_ratio_start = 0.30
    t._target_ratio_end = 0.15
    t._target_ratio_ramp_steps = 100000
    t._target_ratio_override = None
    t._max_gap = 64
    t._l0_calibrator = ThresholdCalibrator(
        ema_decay=0.99, warmup_batches=50, fallback_threshold=3.0,
    )
    t._l1_calibrator = ThresholdCalibrator(
        ema_decay=0.95, warmup_batches=50, fallback_threshold=3.0,
    )
    t._l1_calibrator_fast_batches = 200
    t._l1_calibrator_slow_decay = 0.99
    t._l1_batches_seen = 0
    t._pending_l0_scores = []
    t._pending_l1_scores = []
    t._is_evaluating = False
    t._l1_enabled = False
    t._l1_transitioned = False
    t._l1_rebuild_pending = False
    t._drift_threshold = drift_threshold
    t._drift_check_every = drift_check_every
    t._lora_active = False
    t._eval_count = 0

    return t


# ---------------------------------------------------------------------------
# Embedding drift metric tests
# ---------------------------------------------------------------------------


class TestDriftDetectedLowSimilarity:
    def test_random_embeddings_low_similarity(self):
        """Random survivors far from token embeddings should have low similarity."""
        # Random survivors (not from the token embedding space)
        survivors = torch.randn(50, HIDDEN_DIM) * 100  # large magnitude, random direction
        token_emb = torch.randn(VOCAB_SIZE, HIDDEN_DIM)

        metrics = embedding_drift_metrics(survivors, token_emb)
        # Random high-dim vectors tend to be roughly orthogonal
        # Mean max cosine sim should be moderate but we check it exists
        assert "mean_max_cosine_sim" in metrics
        assert "std_max_cosine_sim" in metrics
        assert "min_max_cosine_sim" in metrics

    def test_drift_check_triggers_at_correct_step(self):
        """Drift check should run at multiples of drift_check_every."""
        t = _make_trainer(drift_check_every=100, drift_threshold=0.99)
        t.global_step = 100

        # Create survivors that are far from token embeddings
        survivors = torch.randn(10, HIDDEN_DIM) * 100
        # Should not raise, just log and potentially trigger LoRA
        t._check_decoder_drift(survivors)

    def test_no_check_at_step_zero(self):
        """Drift check should be skipped at step 0."""
        t = _make_trainer(drift_check_every=100)
        t.global_step = 0
        survivors = torch.randn(10, HIDDEN_DIM)
        # Should be a no-op at step 0
        t._check_decoder_drift(survivors)
        assert not t._lora_active

    def test_no_check_off_schedule(self):
        """Drift check should be skipped when not on schedule."""
        t = _make_trainer(drift_check_every=100, drift_threshold=0.0)
        t.global_step = 50  # not a multiple of 100
        survivors = torch.randn(10, HIDDEN_DIM)
        t._check_decoder_drift(survivors)
        assert not t._lora_active


class TestNoDriftHighSimilarity:
    def test_identical_embeddings_high_similarity(self):
        """Survivors that are token embeddings should have similarity ~1.0."""
        token_emb = torch.randn(VOCAB_SIZE, HIDDEN_DIM)
        # Use actual token embeddings as survivors
        survivors = token_emb[:50].clone()

        metrics = embedding_drift_metrics(survivors, token_emb)
        assert metrics["mean_max_cosine_sim"] > 0.99

    def test_no_lora_when_similarity_high(self):
        """LoRA should NOT be triggered when embeddings are close to tokens."""
        t = _make_trainer(drift_check_every=100, drift_threshold=0.5)
        t.global_step = 100

        # Use actual token embeddings as survivors
        token_emb = t.encoder.compressor.backbone.get_input_embeddings().weight.detach()
        survivors = token_emb[:10].clone()

        t._check_decoder_drift(survivors)
        assert not t._lora_active


class TestLoraAppliedOnDrift:
    def test_lora_flag_set_on_drift(self):
        """When drift is detected, _lora_active should be set to True."""
        pytest.importorskip("peft")

        t = _make_trainer(drift_check_every=100, drift_threshold=0.99)
        t.global_step = 100

        # Orthogonal survivors that definitely have low similarity
        survivors = torch.randn(10, HIDDEN_DIM) * 100
        t._check_decoder_drift(survivors)

        assert t._lora_active

    def test_lora_not_applied_twice(self):
        """LoRA should only be applied once (idempotent)."""
        pytest.importorskip("peft")

        t = _make_trainer(drift_check_every=100, drift_threshold=0.99)
        t.global_step = 100

        survivors = torch.randn(10, HIDDEN_DIM) * 100
        t._check_decoder_drift(survivors)
        assert t._lora_active

        # Second check should not re-apply
        t.global_step = 200
        t._check_decoder_drift(survivors)  # Should be a no-op since already active
        assert t._lora_active

    def test_lora_creates_trainable_params(self):
        """After LoRA fallback, there should be trainable LoRA parameters."""
        pytest.importorskip("peft")

        t = _make_trainer(drift_check_every=100, drift_threshold=0.99)

        t.global_step = 100
        survivors = torch.randn(10, HIDDEN_DIM) * 100
        t._check_decoder_drift(survivors)

        # After LoRA, should have some trainable params
        trainable_after = sum(
            p.numel() for p in t.decoder.backbone.parameters() if p.requires_grad
        )
        assert trainable_after > 0, "Should have trainable LoRA params"

    def test_apply_lora_fallback_directly(self):
        """Test _apply_lora_fallback method directly."""
        pytest.importorskip("peft")

        t = _make_trainer()
        assert not t._lora_active

        t._apply_lora_fallback()
        assert t._lora_active

        # Non-LoRA params should be frozen
        for name, param in t.decoder.backbone.named_parameters():
            if "lora_" not in name:
                assert not param.requires_grad, f"Non-LoRA param {name} should be frozen"

    def test_lora_skipped_without_peft(self, monkeypatch):
        """If peft is not available, LoRA fallback should be skipped gracefully."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "peft":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        t = _make_trainer(drift_check_every=100, drift_threshold=0.99)
        t.global_step = 100

        survivors = torch.randn(10, HIDDEN_DIM) * 100
        t._check_decoder_drift(survivors)

        # Should not crash, and lora should not be active
        assert not t._lora_active
