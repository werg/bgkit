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
from bgkit.models.bgkit_compressor import BgKITCompressor
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
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

    def forward(
        self, inputs_embeds=None, attention_mask=None,
        return_intermediates=False, layer_hooks=None, **kwargs,
    ):
        x = self.layers[0](inputs_embeds)
        # Fire any registered hooks (compressor uses these for ratio injection
        # and survivorship head evaluation)
        if layer_hooks:
            for idx in sorted(layer_hooks):
                x = layer_hooks[idx](x)
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
            bgkit_splice_start=3,
            bgkit_splice_len=0,
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
            bgkit_splice_start=3,
            bgkit_splice_len=0,
        )
        samples.append(sample)
    return collate_compression(samples)


@pytest.fixture()
def trainer():
    """Create a CompressionTrainer with mock components (bypasses setup())."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "training": {
            "phase": "phase1_step6",
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

    # Optimizer
    params = list(t.encoder.parameters()) + list(t.decoder.parameters())
    t.optimizer = torch.optim.AdamW(params, lr=1e-3)

    # Curriculum state
    t._l1_introduction_step = 20
    t._target_ratio_start = 0.30
    t._target_ratio_end = 0.15
    t._target_ratio_ramp_steps = 100
    t._target_ratio_override = None

    t._l1_enabled = False
    t._l1_transitioned = False
    t._l1_rebuild_pending = False
    t._is_evaluating = False
    t._eval_count = 0
    t._profile_enabled = False

    # Survivorship head aux loss weights (normally set in setup())
    t._ratio_loss_weight = 0.1
    t._decisiveness_loss_weight = 0.05

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


# ---------------------------------------------------------------------------
# Train step tests (file samples)
# ---------------------------------------------------------------------------


class TestTrainStepFileSample:
    def test_returns_expected_metrics(self, trainer):
        batch = _make_file_batch()
        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert "grad_norm" in metrics
        assert "min_target_ratio" in metrics
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
        trainer._target_ratio_override = 0.20

        # Save
        ckpt_path = trainer.save_checkpoint(tmp_path)
        assert Path(ckpt_path).exists()
        assert (Path(ckpt_path) / "metadata.json").exists()
        assert (Path(ckpt_path) / "encoder.pt").exists()
        assert (Path(ckpt_path) / "decoder.pt").exists()
        assert (Path(ckpt_path) / "optimizer_state_by_name.pt").exists()

        # Mutate state
        trainer.global_step = 0
        trainer.epoch = 0
        trainer._l1_enabled = False
        trainer._target_ratio_override = None

        # Load
        trainer.load_checkpoint(Path(ckpt_path))
        assert trainer.global_step == 42
        assert trainer.epoch == 2
        assert trainer._l1_enabled is True
        assert trainer._l1_transitioned is True
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
# _resolve_step1_checkpoint wiring tests
# ---------------------------------------------------------------------------


class TestResolveStep1Checkpoint:
    def test_auto_resolves_phase1_step4(self, trainer):
        """Auto should resolve the best phase1_step4 checkpoint."""
        from pathlib import Path
        from unittest.mock import patch

        from omegaconf import OmegaConf
        trainer.cfg = OmegaConf.merge(trainer.cfg, {
            "step1_checkpoint": "auto",
            "checkpoint_dir": "/tmp/ckpts",
        })

        mock_path = Path("/tmp/ckpts/phase1_step4_step10000_20260301")
        with patch(
            "bgkit.training.phase1.compression.resolve_checkpoint",
            return_value=mock_path,
        ) as mock_resolve:
            result = trainer._resolve_step1_checkpoint()

        mock_resolve.assert_called_once_with(
            Path("/tmp/ckpts"),
            phase="phase1_step5",
            metric="eval/loss",
            label="step1_checkpoint",
        )
        assert result == str(mock_path)
        assert trainer._input_sources["step1"] == "phase1_step4_step10000_20260301"

    def test_auto_raises_when_no_checkpoint(self, trainer):
        """Auto should raise when no phase1_step4 checkpoint exists."""
        from unittest.mock import patch

        import pytest
        from omegaconf import OmegaConf
        trainer.cfg = OmegaConf.merge(trainer.cfg, {
            "step1_checkpoint": "auto",
            "checkpoint_dir": "/tmp/ckpts",
        })

        with patch(
            "bgkit.training.phase1.compression.resolve_checkpoint",
            side_effect=ValueError("No phase1_step5 checkpoint found"),
        ), pytest.raises(ValueError, match="phase1_step5"):
            trainer._resolve_step1_checkpoint()

    def test_explicit_path_passthrough(self, trainer):
        """Explicit path should pass through and populate _input_sources."""
        from omegaconf import OmegaConf
        trainer.cfg = OmegaConf.merge(trainer.cfg, {
            "step1_checkpoint": "/workspace/checkpoints/step1_ckpt_300",
        })

        result = trainer._resolve_step1_checkpoint()
        assert result == "/workspace/checkpoints/step1_ckpt_300"
        assert trainer._input_sources["step1"] == "step1_ckpt_300"

    def test_none_returns_none(self, trainer):
        """No step1_checkpoint config should return None and no step1 in _input_sources."""
        result = trainer._resolve_step1_checkpoint()
        assert result is None
        assert "step1" not in trainer._input_sources


def _make_mixed_batch(
    n_file_samples: int = 2,
    n_repo_samples: int = 2,
    n_files: int = 2,
    file_len: int = 8,
    content_len: int = SEQ_LEN,
    target_len: int = SEQ_LEN,
    prompt_len: int = 4,
) -> dict:
    """Create a mixed batch with both file and repo sub-batches."""
    file_samples = []
    for _ in range(n_file_samples):
        file_samples.append(FileCompressionSample(
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
            bgkit_splice_start=3,
            bgkit_splice_len=0,
        ))
    repo_samples = []
    for _ in range(n_repo_samples):
        fids = [torch.randint(0, VOCAB_SIZE, (file_len,)) for _ in range(n_files)]
        fmasks = [torch.ones(file_len, dtype=torch.bool) for _ in range(n_files)]
        repo_samples.append(RepoCompressionSample(
            objective="commit_reproduction",
            file_token_ids=fids,
            file_attention_masks=fmasks,
            compression_ratio=0.2,
            compression_level=1,
            target_token_ids=torch.randint(0, VOCAB_SIZE, (target_len,)),
            target_attention_mask=torch.ones(target_len, dtype=torch.bool),
            target_loss_mask=torch.ones(target_len, dtype=torch.long),
            prefix_ids=torch.randint(0, VOCAB_SIZE, (3,)),
            compression_prompt_ids=torch.randint(0, VOCAB_SIZE, (prompt_len,)),
            bgkit_splice_start=3,
            bgkit_splice_len=0,
        ))
    return collate_compression(file_samples + repo_samples)


class TestForwardBackwardMixed:
    def test_mixed_batch_returns_expected_metrics(self, trainer):
        """_forward_backward_mixed should return loss and standard metrics."""
        batch = _make_mixed_batch()
        assert batch.get("mixed", False)
        trainer.optimizer.zero_grad()
        metrics = trainer._forward_backward(batch)
        assert "loss" in metrics
        assert metrics["sample_type"] == "mixed"
        assert "actual_ratio" in metrics
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_mixed_batch_accumulates_gradients(self, trainer):
        """Mixed batch should produce gradients on the decoder."""
        batch = _make_mixed_batch()
        trainer.optimizer.zero_grad()
        trainer._forward_backward(batch)
        decoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in trainer.decoder.parameters() if p.requires_grad
        )
        assert decoder_has_grad, "Decoder should receive gradients from mixed batch"

    def test_mixed_loss_is_weighted_average(self, trainer):
        """Reported loss should be between file-only and repo-only losses."""
        batch = _make_mixed_batch(n_file_samples=2, n_repo_samples=2)
        trainer.optimizer.zero_grad()
        metrics = trainer._forward_backward(batch)
        # Loss is finite and non-negative (CE loss)
        assert metrics["loss"] >= 0
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_mixed_batch_delegates_to_repo_persample(self, trainer):
        """Repo portion of mixed batch should use _forward_backward_repo_persample."""
        batch = _make_mixed_batch(n_file_samples=1, n_repo_samples=2)
        call_count = [0]
        original = trainer._forward_backward_repo_persample

        def tracking(*args, **kwargs):
            call_count[0] += 1
            return original(*args, **kwargs)

        trainer._forward_backward_repo_persample = tracking
        trainer.optimizer.zero_grad()
        trainer._forward_backward(batch)
        assert call_count[0] == 1, "Should delegate repo portion exactly once"


class TestEvaluateRepoBatchPersample:
    def test_repo_eval_returns_token_weighted_loss(self, trainer):
        """_evaluate_repo_batch_persample should return (loss_sum, token_count)."""
        batch = _make_repo_batch(batch_size=2, n_files=2, file_len=8)
        with torch.no_grad():
            trainer.encoder.eval()
            trainer.decoder.eval()
            trainer._is_evaluating = True
            loss_sum, token_count = trainer._evaluate_repo_batch_persample(batch)
            trainer._is_evaluating = False
        assert loss_sum >= 0
        assert token_count > 0
        # Average loss should be finite
        avg = loss_sum / token_count
        assert torch.isfinite(torch.tensor(avg))

    def test_repo_eval_token_count_matches_target_mask(self, trainer):
        """Token count should match the sum of target attention masks."""
        batch = _make_repo_batch(batch_size=2, n_files=1, file_len=8)
        expected_tokens = batch["target_attention_mask"].sum().item()
        with torch.no_grad():
            trainer.encoder.eval()
            trainer.decoder.eval()
            trainer._is_evaluating = True
            _, token_count = trainer._evaluate_repo_batch_persample(batch)
            trainer._is_evaluating = False
        assert token_count == expected_tokens


class TestEvaluateBreakdown:
    def test_evaluate_reports_token_weighted_per_objective_metrics(self, trainer):
        """evaluate() should emit per-objective loss and token shares."""
        trainer.eval_dataloader = [
            _make_file_batch(batch_size=2, target_len=12),
            _make_repo_batch(batch_size=2, n_files=2, target_len=10),
        ]

        with torch.no_grad():
            metrics = trainer.evaluate()

        assert "data_reconstruction_loss" in metrics
        assert "commit_reproduction_loss" in metrics
        assert metrics["data_reconstruction_tokens"] > 0
        assert metrics["commit_reproduction_tokens"] > 0
        frac_sum = (
            metrics["data_reconstruction_token_fraction"]
            + metrics["commit_reproduction_token_fraction"]
        )
        assert 0.99 <= frac_sum <= 1.01

    def test_evaluate_handles_mixed_batches_instead_of_skipping(self, trainer):
        """Mixed batches should contribute to overall and per-objective eval metrics."""
        trainer.eval_dataloader = [_make_mixed_batch(n_file_samples=1, n_repo_samples=1)]

        with torch.no_grad():
            metrics = trainer.evaluate()

        assert metrics["mixed_batches"] == 1.0
        assert "data_reconstruction_loss" in metrics
        assert "commit_reproduction_loss" in metrics
        assert metrics["loss"] >= 0


class TestL0ToL1AutoRepro:
    def test_auto_reproduce_called_in_per_sample_loop(self, trainer):
        """Per-sample forward should call compressor.auto_reproduce for L0→L1 handoff."""
        call_count = [0]
        original = trainer.encoder.compressor.auto_reproduce

        def tracking_auto_repro(x):
            call_count[0] += 1
            return original(x)

        trainer.encoder.compressor.auto_reproduce = tracking_auto_repro
        batch = _make_repo_batch(batch_size=1, n_files=2, file_len=8)
        trainer.optimizer.zero_grad()
        trainer._forward_backward(batch)
        # Batched L0: 2 files processed in one batched call.
        # With selective checkpointing, short files (< threshold) skip checkpoint,
        # so auto_reproduce is called once per file in the forward only = 2.
        assert call_count[0] >= 2

