"""Tests for CommitEncodingTrainer: ratio curriculum, live-config, resolve wiring.

Under the FA4 packed-attention migration the forward passes require a
Qwen3.5-shaped encoder + decoder. The unit tests here cover only the
pure-Python control plane; forward-pass tests are skipped and covered by
integration tests + live runs.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pathlib import Path
from unittest.mock import patch

from torch import nn

from bgkit.models.bgkit_compressor import BgKITCompressor
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.models.projection_block import ProjectionBlock
from bgkit.training.phase1.commit_encoding import CommitEncodingTrainer

# ---------------------------------------------------------------------------
# Minimal mocks — no forward pass exercised.
# ---------------------------------------------------------------------------

HIDDEN_DIM = 32
VOCAB_SIZE = 256


class _Output:
    def __init__(self, last_hidden_state, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class MockEncoderBackbone(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)])
        self.norm = nn.Identity()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds=None, **kwargs):
        return _Output(last_hidden_state=self.layers[0](inputs_embeds))


class MockSelfAttn(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        head_dim = 8
        n_heads = hidden_dim // head_dim
        self.q_proj = nn.Linear(hidden_dim, n_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5


class MockTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = MockSelfAttn(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))


class MockRotaryEmb(nn.Module):
    HEAD_DIM = 8

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.dim = hidden_dim
        self.rotary_dim = self.HEAD_DIM // 2

    def forward(self, x, position_ids):
        num_tokens = position_ids.shape[-1]
        cos = torch.ones(1, num_tokens, self.rotary_dim, device=x.device, dtype=x.dtype)
        sin = torch.zeros(1, num_tokens, self.rotary_dim, device=x.device, dtype=x.dtype)
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

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens


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


def _make_mock_encoder(hidden_dim: int = HIDDEN_DIM) -> BgKITEncoder:
    backbone = MockEncoderBackbone(hidden_dim=hidden_dim)
    compressor_norm = nn.LayerNorm(hidden_dim)
    compressor = BgKITCompressor(backbone, compressor_norm, hidden_dim=hidden_dim)

    proj_layer = MockTransformerLayer(hidden_dim)
    proj_norm = nn.LayerNorm(hidden_dim)
    rotary = MockRotaryEmb(hidden_dim)
    projection_block = ProjectionBlock(proj_layer, proj_norm, rotary, hidden_dim=hidden_dim)

    return BgKITEncoder(compressor, projection_block)


@pytest.fixture()
def trainer():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "training": {
            "phase": "phase1_step5",
            "max_steps": 100,
            "lr": 1e-3,
            "warmup_steps": 10,
            "eval_every": 50,
            "save_every": 0,
            "curriculum": {
                "target_ratio_start": 0.50,
                "target_ratio_end": 0.20,
                "target_ratio_ramp_steps": 30000,
            },
        },
        "compute": {"num_workers": 0, "pin_memory": False},
        "wandb": {"enabled": False},
    })

    t = CommitEncodingTrainer(cfg)
    t.device = torch.device("cpu")

    t.encoder = _make_mock_encoder(hidden_dim=HIDDEN_DIM)
    t.encoder.requires_grad_(True)
    t.encoder.train()

    decoder_backbone = MockCausalLMBackbone(hidden_dim=HIDDEN_DIM)
    t.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=HIDDEN_DIM)
    t.model = t.decoder

    params = list(t.encoder.parameters()) + list(t.decoder.parameters())
    t.optimizer = torch.optim.AdamW(params, lr=1e-3)

    t._target_ratio_start = 0.50
    t._target_ratio_end = 0.20
    t._target_ratio_ramp_steps = 30000
    t._target_ratio_override = None

    t._profile_enabled = False
    return t


# ---------------------------------------------------------------------------
# Component wiring (no forward pass).
# ---------------------------------------------------------------------------


class TestSetup:
    def test_encoder_exists_and_trainable(self, trainer):
        assert isinstance(trainer.encoder, BgKITEncoder)
        assert any(p.requires_grad for p in trainer.encoder.parameters())

    def test_decoder_exists_and_trainable(self, trainer):
        assert isinstance(trainer.decoder, ReconstructionDecoder)
        assert any(p.requires_grad for p in trainer.decoder.parameters())


# ---------------------------------------------------------------------------
# Forward-pass tests — skipped (covered by integration tests + live runs).
# ---------------------------------------------------------------------------


_FORWARD_SKIP = pytest.mark.skip(
    reason="Requires packed-attention forward pass; covered by integration tests.",
)


@_FORWARD_SKIP
class TestForwardBackward:
    def test_returns_loss_and_metrics(self, trainer):
        pass

    def test_loss_is_finite(self, trainer):
        pass

    def test_gradients_flow_to_decoder(self, trainer):
        pass

    def test_multi_file_l0_compression(self, trainer):
        pass


@_FORWARD_SKIP
class TestCheckpoint:
    def test_save_produces_multi_artifact(self, trainer, tmp_path):
        pass

    def test_checkpoint_roundtrip(self, trainer, tmp_path):
        pass

    def test_checkpoint_loadable_by_compression_trainer(self, trainer, tmp_path):
        pass


# ---------------------------------------------------------------------------
# Curriculum ratio math (pure python).
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


class TestPostStep:
    def test_post_step_calls_bidi_warmup(self, trainer):
        called = [False]
        original = trainer.encoder.step_bidi_warmup

        def mock_step():
            called[0] = True

        trainer.encoder.step_bidi_warmup = mock_step
        trainer._post_step(0)
        assert called[0]
        trainer.encoder.step_bidi_warmup = original


class TestLiveConfig:
    def test_target_ratio_override(self, trainer):
        trainer.apply_live_config({"target_ratio": 0.4})
        assert trainer._target_ratio_override == 0.4

    def test_target_ratio_override_none_clears(self, trainer):
        trainer._target_ratio_override = 0.4
        trainer.apply_live_config({"target_ratio": None})
        assert trainer._target_ratio_override is None


class TestResolveStep1:
    def test_auto_resolves_phase1_step1(self, trainer):
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
