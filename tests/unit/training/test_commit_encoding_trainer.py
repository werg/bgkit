"""Tests for CommitEncodingTrainer: setup, forward_backward, checkpoint round-trip."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pathlib import Path
from unittest.mock import patch

from torch import nn

from bgkit.data.collators import collate_compression
from bgkit.data.datasets.compression_dataset import RepoCompressionSample
from bgkit.data.threshold_calibrator import ThresholdCalibrator
from bgkit.models.bgkit_compressor import BgKITCompressor
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.models.ice import ICE
from bgkit.models.projection_block import ProjectionBlock
from bgkit.training.phase1.commit_encoding import CommitEncodingTrainer

# ---------------------------------------------------------------------------
# Mock backbones (reuse patterns from test_compression_trainer)
# ---------------------------------------------------------------------------

HIDDEN_DIM = 32
VOCAB_SIZE = 256
SEQ_LEN = 16


class _Output:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class MockEncoderBackbone(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)])
        self.norm = nn.Identity()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        x = self.layers[0](inputs_embeds)
        x = self.norm(x)
        return _Output(last_hidden_state=x)


class MockTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        return self.linear(hidden_states)


class MockRotaryEmb(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
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


class _ModelOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _MockQwen3Model(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return _ModelOutput(last_hidden_state=x)


class MockCausalLMBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = 2,
    ):
        super().__init__()
        self.model = _MockQwen3Model(vocab_size, hidden_dim, num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        x = inputs_embeds
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.norm(x)
        logits = self.lm_head(x)
        return _CausalLMOutput(logits=logits)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_encoder(hidden_dim: int = HIDDEN_DIM) -> BgKITEncoder:
    backbone = MockEncoderBackbone(hidden_dim=hidden_dim)
    compressor_norm = nn.LayerNorm(hidden_dim)
    compressor = BgKITCompressor(backbone, compressor_norm, hidden_dim=hidden_dim)

    proj_layer = MockTransformerLayer(hidden_dim)
    proj_norm = nn.LayerNorm(hidden_dim)
    rotary = MockRotaryEmb(hidden_dim)
    projection_block = ProjectionBlock(proj_layer, proj_norm, rotary, hidden_dim=hidden_dim)

    return BgKITEncoder(compressor, projection_block)


def _make_commit_batch(
    batch_size: int = 2,
    n_files: int = 3,
    file_len: int = 8,
    target_len: int = SEQ_LEN,
    prompt_len: int = 4,
) -> dict:
    """Create a collated commit encoding batch from RepoCompressionSamples."""
    samples = []
    for _ in range(batch_size):
        file_ids = [torch.randint(0, VOCAB_SIZE, (file_len,)) for _ in range(n_files)]
        file_masks = [torch.ones(file_len, dtype=torch.bool) for _ in range(n_files)]
        sample = RepoCompressionSample(
            objective="commit_encoding",
            file_token_ids=file_ids,
            file_attention_masks=file_masks,
            compression_ratio=0.0,
            compression_level=1,
            target_token_ids=torch.randint(0, VOCAB_SIZE, (target_len,)),
            target_attention_mask=torch.ones(target_len, dtype=torch.bool),
            target_loss_mask=torch.ones(target_len, dtype=torch.long),
            prefix_ids=torch.randint(0, VOCAB_SIZE, (3,)),
            compression_prompt_ids=torch.randint(0, VOCAB_SIZE, (prompt_len,)),
        )
        samples.append(sample)
    return collate_compression(samples)


@pytest.fixture()
def trainer():
    """Create a CommitEncodingTrainer with mock components (bypasses setup())."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "training": {
            "phase": "phase1_step4",
            "max_steps": 100,
            "lr": 1e-3,
            "warmup_steps": 10,
            "eval_every": 50,
            "save_every": 0,
            "curriculum": {
                "target_ratio_start": 0.50,
                "target_ratio_end": 0.20,
                "target_ratio_ramp_steps": 30000,
                "max_survivor_gap": 64,
                "fallback_threshold": 3.0,
                "calibrator_ema_decay": 0.99,
                "calibrator_warmup_batches": 50,
                "l1_calibrator_fast_batches": 200,
            },
        },
        "compute": {"num_workers": 0, "pin_memory": False},
        "wandb": {"enabled": False},
    })

    t = CommitEncodingTrainer(cfg)
    t.device = torch.device("cpu")

    # Encoder (trainable)
    t.encoder = _make_mock_encoder(hidden_dim=HIDDEN_DIM)
    t.encoder.requires_grad_(True)
    t.encoder.train()

    # Decoder (trainable)
    decoder_backbone = MockCausalLMBackbone(hidden_dim=HIDDEN_DIM)
    t.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=HIDDEN_DIM)
    t.model = t.decoder

    # Frozen ICE model
    t.ice_model = ICE(input_dim=HIDDEN_DIM, hidden_dim=16, num_layers=2, kernel_size=3)
    t.ice_model.eval()
    t.ice_model.requires_grad_(False)

    # Optimizer
    params = list(t.encoder.parameters()) + list(t.decoder.parameters())
    t.optimizer = torch.optim.AdamW(params, lr=1e-3)

    # Curriculum state
    t._target_ratio_start = 0.50
    t._target_ratio_end = 0.20
    t._target_ratio_ramp_steps = 30000
    t._target_ratio_override = None
    t._max_gap = 64

    # Calibrators (both active from step 0)
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
    t._profile_enabled = False

    return t


# ---------------------------------------------------------------------------
# Setup tests
# ---------------------------------------------------------------------------


class TestSetup:
    def test_encoder_exists_and_trainable(self, trainer):
        assert isinstance(trainer.encoder, BgKITEncoder)
        assert any(p.requires_grad for p in trainer.encoder.parameters())

    def test_decoder_exists_and_trainable(self, trainer):
        assert isinstance(trainer.decoder, ReconstructionDecoder)
        assert any(p.requires_grad for p in trainer.decoder.parameters())

    def test_ice_frozen(self, trainer):
        assert isinstance(trainer.ice_model, ICE)
        assert not any(p.requires_grad for p in trainer.ice_model.parameters())

    def test_both_calibrators_exist(self, trainer):
        assert isinstance(trainer._l0_calibrator, ThresholdCalibrator)
        assert isinstance(trainer._l1_calibrator, ThresholdCalibrator)


# ---------------------------------------------------------------------------
# Forward/backward tests
# ---------------------------------------------------------------------------


class TestForwardBackward:
    def test_returns_loss_and_metrics(self, trainer):
        batch = _make_commit_batch(batch_size=1, n_files=2, file_len=8)
        metrics = trainer._forward_backward(batch)
        assert "loss" in metrics
        assert "target_ratio" in metrics
        assert "actual_ratio" in metrics
        assert "calibrated_threshold_l0" in metrics
        assert "calibrated_threshold_l1" in metrics
        assert metrics["loss"] > 0

    def test_loss_is_finite(self, trainer):
        batch = _make_commit_batch(batch_size=2, n_files=3)
        metrics = trainer._forward_backward(batch)
        assert not torch.isnan(torch.tensor(metrics["loss"]))
        assert not torch.isinf(torch.tensor(metrics["loss"]))

    def test_gradients_flow_to_decoder(self, trainer):
        batch = _make_commit_batch(batch_size=1, n_files=2, file_len=8)
        trainer.optimizer.zero_grad()
        trainer._forward_backward(batch)
        # Decoder should receive gradients from reconstruction loss
        decoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in trainer.decoder.parameters()
            if p.requires_grad
        )
        assert decoder_has_grad, "Decoder should receive gradients"

    def test_multi_file_l0_compression(self, trainer):
        """L0 should process each file via batched _compress_l0_batched."""
        batch = _make_commit_batch(batch_size=1, n_files=4, file_len=8)
        file_ids = batch["file_token_ids"]
        file_masks = batch["file_attention_masks"]
        prompt_ids = batch["compression_prompt_ids"]
        prompt_mask = batch["compression_prompt_mask"]
        bgkit_embed = trainer.encoder.compressor.backbone.get_input_embeddings()
        prompt_emb = bgkit_embed(prompt_ids[0:1])

        l0_surv = trainer._compress_l0_batched(
            file_ids[0], file_masks[0], 4,
            prompt_emb, prompt_mask[0:1], bgkit_embed,
        )
        assert l0_surv.ndim == 2  # (total_survivors, hidden_dim)
        assert l0_surv.size(0) > 0, "Should have at least some survivors"

    def test_calibrator_scores_not_doubled_by_checkpoint(self, trainer):
        """Scores should be buffered exactly once per file, not doubled.

        _score_and_select lives OUTSIDE the activation checkpoint boundary.
        If it were inside, checkpoint recomputation would double-append
        scores to _pending_l0_scores. This test catches that regression.
        """
        trainer._pending_l0_scores.clear()
        n_files = 4
        batch = _make_commit_batch(batch_size=1, n_files=n_files, file_len=8)
        file_ids = batch["file_token_ids"]
        file_masks = batch["file_attention_masks"]
        prompt_ids = batch["compression_prompt_ids"]
        prompt_mask = batch["compression_prompt_mask"]
        bgkit_embed = trainer.encoder.compressor.backbone.get_input_embeddings()
        prompt_emb = bgkit_embed(prompt_ids[0:1])

        trainer._compress_l0_batched(
            file_ids[0], file_masks[0], n_files,
            prompt_emb, prompt_mask[0:1], bgkit_embed,
        )

        # L0 scores: one per file
        n_l0_appends = len(trainer._pending_l0_scores)
        assert n_l0_appends == n_files, (
            f"Expected {n_files} L0 score appends (one per file), "
            f"got {n_l0_appends} — checkpoint may be re-executing side effects"
        )


# ---------------------------------------------------------------------------
# Curriculum tests
# ---------------------------------------------------------------------------


class TestCurriculum:
    def test_target_ratio_at_start(self, trainer):
        trainer.global_step = 0
        assert trainer._current_target_ratio() == 0.50

    def test_target_ratio_at_end(self, trainer):
        trainer.global_step = 30000
        assert trainer._current_target_ratio() == 0.20

    def test_target_ratio_override(self, trainer):
        trainer._target_ratio_override = 0.35
        trainer.global_step = 15000
        assert trainer._current_target_ratio() == 0.35

    def test_calibrator_flush(self, trainer):
        """Flushing pending scores should update calibrators."""
        trainer._pending_l0_scores = [torch.randn(100)]
        trainer._pending_l1_scores = [torch.randn(50)]
        trainer._flush_calibrator_scores()
        assert len(trainer._pending_l0_scores) == 0
        assert len(trainer._pending_l1_scores) == 0
        assert trainer._l1_batches_seen == 1


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_save_produces_multi_artifact(self, trainer, tmp_path):
        """save_checkpoint() should create encoder.pt, decoder.pt, etc."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        trainer.global_step = 42
        trainer.epoch = 1
        trainer._training_state = {}
        trainer._last_checkpoint_path = None
        trainer._schedule_params = None
        trainer._optimizer_type = "adamw"

        ckpt_path = trainer.save_checkpoint(ckpt_dir, metrics={"loss": 1.5})
        assert ckpt_path.exists()

        # Verify multi-artifact format
        assert (ckpt_path / "encoder.pt").exists()
        assert (ckpt_path / "decoder.pt").exists()
        assert (ckpt_path / "optimizer.pt").exists()
        assert (ckpt_path / "l0_calibrator.pt").exists()
        assert (ckpt_path / "l1_calibrator.pt").exists()
        assert (ckpt_path / "metadata.json").exists()

    def test_checkpoint_roundtrip(self, trainer, tmp_path):
        """Save and reload should restore all state."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        trainer.global_step = 100
        trainer.epoch = 3
        trainer._training_state = {}
        trainer._last_checkpoint_path = None
        trainer._schedule_params = None
        trainer._optimizer_type = "adamw"
        trainer._l1_batches_seen = 42

        ckpt_path = trainer.save_checkpoint(ckpt_dir, metrics={"loss": 2.0})

        # Reset state
        trainer.global_step = 0
        trainer.epoch = 0
        trainer._l1_batches_seen = 0

        trainer.load_checkpoint(ckpt_path)
        assert trainer.global_step == 100
        assert trainer.epoch == 3
        assert trainer._l1_batches_seen == 42

    def test_checkpoint_loadable_by_compression_trainer(self, trainer, tmp_path):
        """Commit encoding checkpoints must be consumable by CompressionTrainer (Step 2)."""
        from bgkit.training.checkpointing import load_checkpoint

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        trainer.global_step = 50
        trainer.epoch = 1
        trainer._training_state = {}
        trainer._last_checkpoint_path = None
        trainer._schedule_params = None
        trainer._optimizer_type = "adamw"

        ckpt_path = trainer.save_checkpoint(ckpt_dir, metrics={"loss": 1.0})

        # Load as CompressionTrainer would (separate encoder/decoder state dicts)
        metadata, state_dicts = load_checkpoint(ckpt_path)
        assert metadata.phase == "phase1_step4"
        assert "encoder" in state_dicts
        assert "decoder" in state_dicts

        # Verify state_dicts can be loaded into fresh models
        new_encoder = _make_mock_encoder(hidden_dim=HIDDEN_DIM)
        new_encoder.load_state_dict(state_dicts["encoder"])

        decoder_backbone = MockCausalLMBackbone(hidden_dim=HIDDEN_DIM)
        new_decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=HIDDEN_DIM)
        new_decoder.load_state_dict(state_dicts["decoder"])


# ---------------------------------------------------------------------------
# Post-step tests
# ---------------------------------------------------------------------------


class TestPostStep:
    def test_post_step_calls_bidi_warmup(self, trainer):
        """_post_step should call encoder.step_bidi_warmup()."""
        called = [False]
        original = trainer.encoder.step_bidi_warmup

        def mock_step():
            called[0] = True

        trainer.encoder.step_bidi_warmup = mock_step
        trainer._post_step(0)
        assert called[0], "_post_step should call encoder.step_bidi_warmup()"
        trainer.encoder.step_bidi_warmup = original


# ---------------------------------------------------------------------------
# Live config tests
# ---------------------------------------------------------------------------


class TestLiveConfig:
    def test_target_ratio_override(self, trainer):
        trainer.apply_live_config({"target_ratio": 0.4})
        assert trainer._target_ratio_override == 0.4

    def test_target_ratio_override_none_clears(self, trainer):
        trainer._target_ratio_override = 0.4
        trainer.apply_live_config({"target_ratio": None})
        assert trainer._target_ratio_override is None

    def test_max_survivor_gap(self, trainer):
        trainer.apply_live_config({"max_survivor_gap": 32})
        assert trainer._max_gap == 32


# ---------------------------------------------------------------------------
# Resolve step1 checkpoint tests
# ---------------------------------------------------------------------------


class TestResolveStep1:
    def test_auto_resolves_phase1_step1(self, trainer):
        """auto should resolve from phase1_step1."""
        from omegaconf import OmegaConf

        trainer.cfg = OmegaConf.create({
            **OmegaConf.to_container(trainer.cfg),
            "step1_checkpoint": "auto",
            "checkpoint_dir": "/tmp/nonexistent",
        })

        with patch("bgkit.training.phase1.commit_encoding.resolve_checkpoint") as mock_resolve:
            mock_resolve.return_value = Path("/checkpoints/phase1_step1_step100")
            result = trainer._resolve_step1_checkpoint()
            assert result == "/checkpoints/phase1_step1_step100"
            mock_resolve.assert_called_once()

    def test_explicit_path_passthrough(self, trainer):
        from omegaconf import OmegaConf

        trainer.cfg = OmegaConf.create({
            **OmegaConf.to_container(trainer.cfg),
            "step1_checkpoint": "/my/checkpoint",
        })
        result = trainer._resolve_step1_checkpoint()
        assert result == "/my/checkpoint"
        assert trainer._input_sources["step1"] == "checkpoint"

    def test_none_returns_none(self, trainer):
        from omegaconf import OmegaConf

        trainer.cfg = OmegaConf.create({
            **OmegaConf.to_container(trainer.cfg),
            "step1_checkpoint": None,
        })
        result = trainer._resolve_step1_checkpoint()
        assert result is None
