"""Tests for CompressionTrainer: setup, train_step, checkpoint round-trip."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pathlib import Path

from torch import nn

from bgkit.data.collators import collate_compression
from bgkit.data.datasets.compression_dataset import (
    FileCompressionSample,
    RepoCompressionSample,
)
from bgkit.data.threshold_calibrator import ThresholdCalibrator
from bgkit.models.bgkit_compressor import BgKITCompressor
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.models.ice import ICE
from bgkit.models.projection_block import ProjectionBlock
from bgkit.training.phase1.compression import CompressionTrainer

# ---------------------------------------------------------------------------
# Mock backbones (same patterns as test_decoder_init_trainer)
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


class _MockQwen3Model(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)


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


def _make_file_batch(
    batch_size: int = 2,
    content_len: int = SEQ_LEN,
    target_len: int = SEQ_LEN,
    prompt_len: int = 4,
) -> dict:
    """Create a collated file compression batch from FileCompressionSamples."""
    samples = []
    for _ in range(batch_size):
        sample = FileCompressionSample(
            objective="data_reconstruction",
            content_token_ids=torch.randint(0, VOCAB_SIZE, (content_len,)),
            content_attention_mask=torch.ones(content_len, dtype=torch.bool),
            compression_ratio=0.2,
            compression_level=0,
            target_token_ids=torch.randint(0, VOCAB_SIZE, (target_len,)),
            target_attention_mask=torch.ones(target_len, dtype=torch.bool),
            target_loss_mask=torch.ones(target_len, dtype=torch.long),
            prefix_ids=torch.randint(0, VOCAB_SIZE, (3,)),
            compression_prompt_ids=torch.randint(0, VOCAB_SIZE, (prompt_len,)),
        )
        samples.append(sample)
    return collate_compression(samples)


def _make_repo_batch(
    batch_size: int = 2,
    n_files: int = 2,
    file_len: int = 8,
    target_len: int = SEQ_LEN,
    prompt_len: int = 4,
) -> dict:
    """Create a collated repo compression batch from RepoCompressionSamples."""
    samples = []
    for _ in range(batch_size):
        file_ids = [torch.randint(0, VOCAB_SIZE, (file_len,)) for _ in range(n_files)]
        file_masks = [torch.ones(file_len, dtype=torch.bool) for _ in range(n_files)]
        sample = RepoCompressionSample(
            objective="commit_reproduction",
            file_token_ids=file_ids,
            file_attention_masks=file_masks,
            compression_ratio=0.2,
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
    """Create a CompressionTrainer with mock components (bypasses setup())."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "training": {
            "phase": "phase1_step2",
            "max_steps": 100,
            "lr": 1e-3,
            "warmup_steps": 10,
            "eval_every": 50,
            "save_every": 0,
            "curriculum": {
                "l1_introduction_step": 20,
                "target_ratio_start": 0.30,
                "target_ratio_end": 0.15,
                "target_ratio_ramp_steps": 100,
                "max_survivor_gap": 64,
                "fallback_threshold": 3.0,
                "calibrator_ema_decay": 0.99,
                "calibrator_warmup_batches": 50,
                "l1_calibrator_fast_batches": 200,
            },
            "decoder": {
                "drift_threshold": 0.5,
                "drift_check_every": 5000,
            },
        },
        "compute": {"num_workers": 0, "pin_memory": False},
        "wandb": {"enabled": False},
    })

    t = CompressionTrainer(cfg)
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
    t._l1_introduction_step = 20
    t._target_ratio_start = 0.30
    t._target_ratio_end = 0.15
    t._target_ratio_ramp_steps = 100
    t._target_ratio_override = None
    t._max_gap = 64

    # Calibrators
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
    t._drift_threshold = 0.5
    t._drift_check_every = 5000
    t._lora_active = False
    t._eval_count = 0

    return t


# ---------------------------------------------------------------------------
# Setup tests
# ---------------------------------------------------------------------------


class TestSetupCreatesComponents:
    def test_encoder_exists(self, trainer):
        assert trainer.encoder is not None
        assert isinstance(trainer.encoder, BgKITEncoder)

    def test_decoder_exists(self, trainer):
        assert trainer.decoder is not None
        assert isinstance(trainer.decoder, ReconstructionDecoder)

    def test_optimizer_exists(self, trainer):
        assert trainer.optimizer is not None

    def test_encoder_is_trainable(self, trainer):
        has_trainable = any(p.requires_grad for p in trainer.encoder.parameters())
        assert has_trainable, "Encoder should be trainable in Step 2"

    def test_decoder_is_trainable(self, trainer):
        has_trainable = any(p.requires_grad for p in trainer.decoder.parameters())
        assert has_trainable, "Decoder should be trainable"

    def test_ice_model_exists_and_frozen(self, trainer):
        assert trainer.ice_model is not None
        assert isinstance(trainer.ice_model, ICE)
        assert not any(p.requires_grad for p in trainer.ice_model.parameters())


# ---------------------------------------------------------------------------
# Train step tests (file samples)
# ---------------------------------------------------------------------------


class TestTrainStepFileSample:
    def test_returns_expected_metrics(self, trainer):
        batch = _make_file_batch()
        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert "grad_norm" in metrics
        assert "target_ratio" in metrics
        assert "calibrated_threshold_l0" in metrics
        assert metrics["sample_type"] == "file"

    def test_loss_is_finite(self, trainer):
        batch = _make_file_batch()
        metrics = trainer.train_step(batch)
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_encoder_is_trainable(self, trainer):
        """In Step 2 encoder should be trainable (requires_grad=True) and in optimizer."""
        trainable_enc = [
            p for p in trainer.encoder.parameters() if p.requires_grad
        ]
        assert len(trainable_enc) > 0, "Encoder should have trainable params"

        opt_params = {
            id(p) for g in trainer.optimizer.param_groups for p in g["params"]
        }
        for p in trainable_enc:
            assert id(p) in opt_params, "Trainable encoder param not in optimizer"

    def test_decoder_gets_gradients(self, trainer):
        batch = _make_file_batch()
        trainer.train_step(batch)
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in trainer.decoder.parameters()
        )
        assert has_grad, "Decoder should have gradients"

    def test_loss_decreases(self, trainer):
        torch.manual_seed(42)
        batch = _make_file_batch(batch_size=4)
        losses = []
        for _ in range(30):
            metrics = trainer.train_step(batch)
            losses.append(metrics["loss"])
        early = sum(losses[:5]) / 5
        late = sum(losses[-5:]) / 5
        assert late < early, f"Loss didn't decrease: {early:.4f} -> {late:.4f}"


# ---------------------------------------------------------------------------
# Train step tests (repo samples)
# ---------------------------------------------------------------------------


def _make_direct_l1_batch(
    batch_size: int = 2,
    file_len: int = 12,
    target_len: int = SEQ_LEN,
    prompt_len: int = 4,
) -> dict:
    """Create a collated repo batch with direct_l1=True (single file per sample)."""
    samples = []
    for _ in range(batch_size):
        sample = RepoCompressionSample(
            objective="commit_reproduction",
            file_token_ids=[torch.randint(0, VOCAB_SIZE, (file_len,))],
            file_attention_masks=[torch.ones(file_len, dtype=torch.bool)],
            compression_ratio=0.2,
            compression_level=1,
            target_token_ids=torch.randint(0, VOCAB_SIZE, (target_len,)),
            target_attention_mask=torch.ones(target_len, dtype=torch.bool),
            target_loss_mask=torch.ones(target_len, dtype=torch.long),
            prefix_ids=torch.randint(0, VOCAB_SIZE, (3,)),
            compression_prompt_ids=torch.randint(0, VOCAB_SIZE, (prompt_len,)),
            direct_l1=True,
        )
        samples.append(sample)
    return collate_compression(samples)


class TestTrainStepRepoSample:
    def test_returns_expected_metrics(self, trainer):
        batch = _make_repo_batch()
        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert "grad_norm" in metrics
        assert metrics["sample_type"] == "repo"

    def test_loss_is_finite(self, trainer):
        batch = _make_repo_batch()
        metrics = trainer.train_step(batch)
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_single_file_repo(self, trainer):
        """Repo sample with single file should work (commit reproduction pattern)."""
        batch = _make_repo_batch(n_files=1, file_len=12)
        metrics = trainer.train_step(batch)
        assert torch.isfinite(torch.tensor(metrics["loss"]))


# ---------------------------------------------------------------------------
# Curriculum tests
# ---------------------------------------------------------------------------


class TestRatioTargeting:
    def test_ratio_starts_at_start(self, trainer):
        trainer.global_step = 0
        ratio = trainer._current_target_ratio()
        assert abs(ratio - 0.30) < 1e-6

    def test_ratio_ramps(self, trainer):
        trainer.global_step = 50
        ratio = trainer._current_target_ratio()
        expected = 0.30 + 0.5 * (0.15 - 0.30)
        assert abs(ratio - expected) < 1e-6

    def test_ratio_ends(self, trainer):
        trainer.global_step = 100
        ratio = trainer._current_target_ratio()
        assert abs(ratio - 0.15) < 1e-6

    def test_ratio_override(self, trainer):
        trainer._target_ratio_override = 0.20
        trainer.global_step = 50
        assert trainer._current_target_ratio() == 0.20

    def test_ratio_override_none_returns_to_ramp(self, trainer):
        trainer._target_ratio_override = None
        trainer.global_step = 50
        expected = 0.30 + 0.5 * (0.15 - 0.30)
        assert abs(trainer._current_target_ratio() - expected) < 1e-6


# ---------------------------------------------------------------------------
# L1 rebuild tests
# ---------------------------------------------------------------------------


class TestL1RebuildPending:
    def test_l1_not_enabled_before_step(self, trainer):
        trainer.global_step = 10
        trainer._check_l1_introduction()
        assert not trainer._l1_enabled
        assert not trainer._l1_rebuild_pending

    def test_l1_enabled_at_step(self, trainer):
        trainer.global_step = 20
        trainer._check_l1_introduction()
        assert trainer._l1_enabled
        assert trainer._l1_rebuild_pending

    def test_l1_only_triggers_once(self, trainer):
        trainer.global_step = 20
        trainer._check_l1_introduction()
        assert trainer._l1_rebuild_pending

        # Simulate rebuild
        trainer._l1_rebuild_pending = False

        # Next step should not re-trigger
        trainer.global_step = 21
        trainer._check_l1_introduction()
        assert not trainer._l1_rebuild_pending


# ---------------------------------------------------------------------------
# Checkpoint round-trip tests
# ---------------------------------------------------------------------------


class TestCheckpointRoundtrip:
    def test_save_load_roundtrip(self, trainer, tmp_path):
        # Do a train step
        trainer.train_step(_make_file_batch())
        trainer.global_step = 42
        trainer.epoch = 2
        trainer._l1_enabled = True
        trainer._l1_transitioned = True
        trainer._l1_batches_seen = 5
        trainer._target_ratio_override = 0.20

        # Save
        ckpt_path = trainer.save_checkpoint(tmp_path)
        assert Path(ckpt_path).exists()
        assert (Path(ckpt_path) / "metadata.json").exists()
        assert (Path(ckpt_path) / "encoder.pt").exists()
        assert (Path(ckpt_path) / "decoder.pt").exists()
        assert (Path(ckpt_path) / "optimizer.pt").exists()
        assert (Path(ckpt_path) / "l0_calibrator.pt").exists()
        assert (Path(ckpt_path) / "l1_calibrator.pt").exists()

        # Mutate state
        trainer.global_step = 0
        trainer.epoch = 0
        trainer._l1_enabled = False
        trainer._l1_batches_seen = 0
        trainer._target_ratio_override = None

        # Load
        trainer.load_checkpoint(Path(ckpt_path))
        assert trainer.global_step == 42
        assert trainer.epoch == 2
        assert trainer._l1_enabled is True
        assert trainer._l1_transitioned is True
        assert trainer._l1_batches_seen == 5
        assert trainer._target_ratio_override == 0.20

    def test_encoder_weights_restored(self, trainer, tmp_path):
        trainer.train_step(_make_file_batch())
        trainer.global_step = 10

        ckpt_path = trainer.save_checkpoint(tmp_path)

        # Zero out encoder
        with torch.no_grad():
            for p in trainer.encoder.parameters():
                p.zero_()

        trainer.load_checkpoint(Path(ckpt_path))

        has_nonzero = any(
            p.abs().sum() > 0 for p in trainer.encoder.parameters()
        )
        assert has_nonzero, "Encoder params should be restored"

    def test_decoder_weights_restored(self, trainer, tmp_path):
        trainer.train_step(_make_file_batch())
        trainer.global_step = 10

        ckpt_path = trainer.save_checkpoint(tmp_path)

        with torch.no_grad():
            for p in trainer.decoder.parameters():
                p.zero_()

        trainer.load_checkpoint(Path(ckpt_path))

        has_nonzero = any(
            p.abs().sum() > 0 for p in trainer.decoder.parameters()
        )
        assert has_nonzero, "Decoder params should be restored"


# ---------------------------------------------------------------------------
# Direct L1 tests
# ---------------------------------------------------------------------------


class TestDirectL1:
    def test_direct_l1_dispatches(self, trainer):
        """Batch with all direct_l1=True should produce valid metrics."""
        batch = _make_direct_l1_batch()
        assert batch["direct_l1"].all()

        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert metrics["sample_type"] == "repo"

    def test_direct_l1_returns_valid_embeddings(self, trainer):
        """direct_l1 batch should return embeddings with correct shapes."""
        batch = _make_direct_l1_batch(batch_size=2, file_len=12)
        survivors, survivor_mask = trainer._compress_repo_batch(batch)

        assert survivors.dim() == 3  # (B, num_survivors, D)
        assert survivors.size(0) == 2
        assert survivors.size(-1) == HIDDEN_DIM
        assert survivor_mask.dim() == 2
        assert survivor_mask.size(0) == 2

    def test_direct_l1_uses_single_direct_l1(self, trainer):
        """direct_l1 samples should call _compress_single_direct_l1."""
        batch = _make_direct_l1_batch()

        original = trainer._compress_single_direct_l1
        call_count = [0]

        def tracking(*args, **kwargs):
            call_count[0] += 1
            return original(*args, **kwargs)

        trainer._compress_single_direct_l1 = tracking
        trainer._compress_repo_batch(batch)
        assert call_count[0] == batch["file_token_ids"].size(0)

    def test_non_direct_l1_does_not_dispatch(self, trainer):
        """Regular repo batch should NOT call _compress_single_direct_l1."""
        batch = _make_repo_batch(n_files=2)
        assert not batch["direct_l1"].any()

        call_count = [0]
        original = trainer._compress_single_direct_l1

        def tracking(*args, **kwargs):
            call_count[0] += 1
            return original(*args, **kwargs)

        trainer._compress_single_direct_l1 = tracking
        trainer._compress_repo_batch(batch)
        assert call_count[0] == 0

    def test_compress_repo_batch_dispatches_direct_l1(self, trainer):
        """_compress_repo_batch handles direct_l1 batches end-to-end."""
        batch = _make_direct_l1_batch()
        survivors, survivor_mask = trainer._compress_repo_batch(batch)
        assert survivors.dim() == 3
        assert survivor_mask.dim() == 2

    def test_direct_l1_single_encoder_pass(self, trainer):
        """direct_l1 samples should get exactly one encoder call, not two."""
        batch = _make_direct_l1_batch(batch_size=2)

        encoder_call_count = [0]
        original_encoder = trainer.encoder.forward

        def counting_encoder(*args, **kwargs):
            encoder_call_count[0] += 1
            return original_encoder(*args, **kwargs)

        trainer.encoder.forward = counting_encoder
        trainer._compress_repo_batch(batch)
        # One encoder call per direct_l1 sample, no extra L1 pass
        assert encoder_call_count[0] == 2

    def test_direct_l1_rejects_multi_file(self, trainer):
        """direct_l1=True with file_count > 1 should raise ValueError."""
        # Create a sample with direct_l1=True but 2 files
        sample = RepoCompressionSample(
            objective="commit_reproduction",
            file_token_ids=[
                torch.randint(0, VOCAB_SIZE, (12,)),
                torch.randint(0, VOCAB_SIZE, (12,)),
            ],
            file_attention_masks=[
                torch.ones(12, dtype=torch.bool),
                torch.ones(12, dtype=torch.bool),
            ],
            compression_ratio=0.2,
            compression_level=1,
            target_token_ids=torch.randint(0, VOCAB_SIZE, (SEQ_LEN,)),
            target_attention_mask=torch.ones(SEQ_LEN, dtype=torch.bool),
            target_loss_mask=torch.ones(SEQ_LEN, dtype=torch.long),
            prefix_ids=torch.randint(0, VOCAB_SIZE, (3,)),
            compression_prompt_ids=torch.randint(0, VOCAB_SIZE, (4,)),
            direct_l1=True,
        )
        batch = collate_compression([sample])
        with pytest.raises(ValueError, match=r"direct_l1 sample.*file_count=2"):
            trainer._compress_repo_batch(batch)


# ---------------------------------------------------------------------------
# ICE scoring and threshold selection tests
# ---------------------------------------------------------------------------


class TestScoreAndSelect:
    def test_returns_bool_mask(self, trainer):
        """_score_and_select should return a boolean mask."""
        embeddings = torch.randn(2, SEQ_LEN, HIDDEN_DIM)
        mask = torch.ones(2, SEQ_LEN, dtype=torch.bool)
        result = trainer._score_and_select(embeddings, mask)
        assert result.dtype == torch.bool
        assert result.shape == (2, SEQ_LEN)

    def test_handles_partial_mask(self, trainer):
        """Padding positions should not be selected."""
        embeddings = torch.randn(1, 20, HIDDEN_DIM)
        mask = torch.zeros(1, 20, dtype=torch.bool)
        mask[0, :10] = True  # only first 10 are valid
        result = trainer._score_and_select(embeddings, mask)
        # No padding positions should be selected
        assert result[0, 10:].sum() == 0

    def test_ice_model_called(self, trainer):
        """ICE model should be called during scoring."""
        call_count = [0]
        original_forward = trainer.ice_model.forward

        def tracking_forward(*args, **kwargs):
            call_count[0] += 1
            return original_forward(*args, **kwargs)

        trainer.ice_model.forward = tracking_forward
        embeddings = torch.randn(1, SEQ_LEN, HIDDEN_DIM)
        mask = torch.ones(1, SEQ_LEN, dtype=torch.bool)
        trainer._score_and_select(embeddings, mask)
        assert call_count[0] == 1

    def test_buffers_scores_during_training(self, trainer):
        """Should buffer L0 scores for deferred calibrator update."""
        trainer._is_evaluating = False
        embeddings = torch.randn(1, SEQ_LEN, HIDDEN_DIM)
        mask = torch.ones(1, SEQ_LEN, dtype=torch.bool)
        trainer._score_and_select(embeddings, mask, level="l0")
        assert len(trainer._pending_l0_scores) == 1

    def test_does_not_buffer_during_eval(self, trainer):
        """Should NOT buffer scores when evaluating."""
        trainer._is_evaluating = True
        embeddings = torch.randn(1, SEQ_LEN, HIDDEN_DIM)
        mask = torch.ones(1, SEQ_LEN, dtype=torch.bool)
        trainer._score_and_select(embeddings, mask, level="l0")
        assert len(trainer._pending_l0_scores) == 0

    def test_level_l1_buffers_to_l1(self, trainer):
        """level='l1' should buffer to L1 pending scores."""
        trainer._is_evaluating = False
        embeddings = torch.randn(1, SEQ_LEN, HIDDEN_DIM)
        mask = torch.ones(1, SEQ_LEN, dtype=torch.bool)
        trainer._score_and_select(embeddings, mask, level="l1")
        assert len(trainer._pending_l1_scores) == 1
        assert len(trainer._pending_l0_scores) == 0


class TestL0ToL1AutoRepro:
    def test_auto_reproduce_called_in_repo_batch(self, trainer):
        """_compress_repo_batch should call auto_reproduce for L0→L1 handoff."""
        call_count = [0]
        original = trainer.encoder.auto_reproduce

        def tracking_auto_repro(x):
            call_count[0] += 1
            return original(x)

        trainer.encoder.auto_reproduce = tracking_auto_repro
        batch = _make_repo_batch(batch_size=1, n_files=2, file_len=8)
        trainer._compress_repo_batch(batch)
        # Should be called once per file
        assert call_count[0] == 2

    def test_l1_uses_ice_scoring(self, trainer):
        """L1 stage should also call _score_and_select (for L1 survivor selection)."""
        call_count = [0]
        original = trainer._score_and_select

        def tracking_score(emb, mask, level="l0"):
            call_count[0] += 1
            return original(emb, mask, level=level)

        trainer._score_and_select = tracking_score
        batch = _make_repo_batch(batch_size=1, n_files=2, file_len=8)
        trainer._compress_repo_batch(batch)
        # Called once per file (L0) + once for L1 = 3
        assert call_count[0] == 3
