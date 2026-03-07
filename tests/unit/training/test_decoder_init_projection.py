"""Tests for projection block training in DecoderInitTrainer."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pathlib import Path

from omegaconf import OmegaConf
from torch import nn

from bgkit.data.collators import collate_chat_repro
from bgkit.models.bgkit_compressor import BgKITCompressor
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.models.projection_block import ProjectionBlock
from bgkit.training.phase1.decoder_init import DecoderInitTrainer

# ---------------------------------------------------------------------------
# Mock backbones (same pattern as test_decoder_init_trainer.py)
# ---------------------------------------------------------------------------


class _Output:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class MockEncoderBackbone(nn.Module):
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64):
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
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        return self.linear(hidden_states)


class MockRotaryEmb(nn.Module):
    def __init__(self, hidden_dim: int = 64):
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
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)


class MockCausalLMBackbone(nn.Module):
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64, num_layers: int = 4):
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

HIDDEN_DIM = 64


def _make_mock_encoder(hidden_dim: int = HIDDEN_DIM) -> BgKITEncoder:
    backbone = MockEncoderBackbone(hidden_dim=hidden_dim)
    compressor_norm = nn.LayerNorm(hidden_dim)
    compressor = BgKITCompressor(backbone, compressor_norm, hidden_dim=hidden_dim)

    proj_layer = MockTransformerLayer(hidden_dim)
    proj_norm = nn.LayerNorm(hidden_dim)
    rotary = MockRotaryEmb(hidden_dim)
    projection_block = ProjectionBlock(proj_layer, proj_norm, rotary, hidden_dim=hidden_dim)

    return BgKITEncoder(compressor, projection_block)


def _make_batch(batch_size: int = 2, seq_len: int = 20, content_len: int = 10,
                prompt_len: int = 5):
    samples = []
    for _ in range(batch_size):
        total_len = seq_len
        content_start = (total_len - content_len) // 2
        content_end = content_start + content_len

        token_ids = torch.randint(0, 1000, (total_len,))
        loss_mask = torch.zeros(total_len, dtype=torch.long)
        loss_mask[content_start:content_end] = 1

        samples.append({
            "token_ids": token_ids,
            "loss_mask": loss_mask,
            "content_token_ids": token_ids[content_start:content_end],
            "compression_prompt_ids": torch.randint(0, 1000, (prompt_len,)),
            "prefix_ids": torch.randint(0, 1000, (content_start,)),
            "language": "python",
        })
    return collate_chat_repro(samples)


def _make_trainer(
    train_projection_block: bool = True,
    projection_only_steps: int = 0,
    projection_lr: float | None = None,
    freeze_top_layer: bool = True,
    lr: float = 1e-3,
    lr_scale_bottom: float = 0.1,
    num_layers: int = 4,
    max_steps: int = 100,
    warmup_steps: int = 10,
) -> DecoderInitTrainer:
    """Create a DecoderInitTrainer with projection block training support."""
    training_cfg = {
        "phase": "phase1_step1",
        "max_steps": max_steps,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "eval_every": 50,
        "save_every": 0,
        "train_projection_block": train_projection_block,
        "projection_only_steps": projection_only_steps,
        "freeze_top_layer": freeze_top_layer,
        "lr_scale_bottom": lr_scale_bottom,
    }
    if projection_lr is not None:
        training_cfg["projection_lr"] = projection_lr

    cfg = OmegaConf.create({
        "training": training_cfg,
        "wandb": {"enabled": False},
    })

    t = DecoderInitTrainer(cfg)
    t.device = torch.device("cpu")

    # Encoder (frozen, then projection block selectively unfrozen)
    t.encoder = _make_mock_encoder(hidden_dim=HIDDEN_DIM)
    t.encoder.requires_grad_(False)
    t.encoder.eval()

    # Store config flags (normally done in setup())
    t._train_projection = train_projection_block
    t._projection_only_steps = projection_only_steps

    # Decoder
    decoder_backbone = MockCausalLMBackbone(hidden_dim=HIDDEN_DIM, num_layers=num_layers)
    t.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=HIDDEN_DIM)
    t.model = t.decoder

    # Use real freeze/optimizer setup
    t._configure_trainable_state()
    t._eval_count = 0

    return t


# ---------------------------------------------------------------------------
# Tests: projection block freeze/unfreeze
# ---------------------------------------------------------------------------


class TestProjectionBlockTrainable:
    def test_projection_block_unfrozen(self):
        """After setup, projection block params have requires_grad=True."""
        t = _make_trainer(train_projection_block=True)
        for name, p in t.encoder.projection_block.named_parameters():
            assert p.requires_grad, f"projection_block.{name} should be trainable"

    def test_compressor_remains_frozen(self):
        """Compressor params should remain frozen even when projection is trainable."""
        t = _make_trainer(train_projection_block=True)
        for name, p in t.encoder.compressor.named_parameters():
            assert not p.requires_grad, f"compressor.{name} should be frozen"

    def test_projection_block_frozen_when_disabled(self):
        """When train_projection_block=False, projection block stays frozen."""
        t = _make_trainer(train_projection_block=False)
        for name, p in t.encoder.projection_block.named_parameters():
            assert not p.requires_grad, f"projection_block.{name} should be frozen"


# ---------------------------------------------------------------------------
# Tests: decoder frozen during warmup
# ---------------------------------------------------------------------------


class TestProjectionOnlyWarmup:
    def test_decoder_frozen_during_warmup(self):
        """With projection_only_steps > 0, decoder params are frozen at step 0."""
        t = _make_trainer(train_projection_block=True, projection_only_steps=100)
        assert t._decoder_frozen is True
        for name, p in t.decoder.named_parameters():
            assert not p.requires_grad, f"decoder.{name} should be frozen during warmup"

    def test_projection_trainable_during_warmup(self):
        """Projection block should be trainable even during decoder warmup."""
        t = _make_trainer(train_projection_block=True, projection_only_steps=100)
        for name, p in t.encoder.projection_block.named_parameters():
            assert p.requires_grad, f"projection_block.{name} should be trainable"

    def test_optimizer_only_has_projection_during_warmup(self):
        """During projection-only phase, optimizer has only projection params."""
        t = _make_trainer(train_projection_block=True, projection_only_steps=100)
        all_opt_params = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        proj_params = {id(p) for p in t.encoder.projection_block.parameters()}

        # All optimizer params should be projection params
        assert all_opt_params == proj_params, "Optimizer should only have projection params"


# ---------------------------------------------------------------------------
# Tests: phase transition
# ---------------------------------------------------------------------------


class TestPhaseTransition:
    def test_transition_unfreezes_decoder(self):
        """After _maybe_end_projection_only, decoder should be unfrozen."""
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=10,
            freeze_top_layer=False,
        )
        assert t._decoder_frozen is True

        # Simulate reaching transition step
        t.global_step = 10
        t._maybe_end_projection_only()

        assert t._decoder_frozen is False
        has_trainable = any(p.requires_grad for p in t.decoder.parameters())
        assert has_trainable, "Decoder should have trainable params after transition"

    def test_transition_preserves_top_freeze(self):
        """After transition, top layer/lm_head/embed_tokens remain frozen."""
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=10,
            freeze_top_layer=True,
        )
        t.global_step = 10
        t._maybe_end_projection_only()

        # Top layer, lm_head, embed_tokens should stay frozen
        top_layer = t.decoder.backbone.model.layers[-1]
        for p in top_layer.parameters():
            assert not p.requires_grad, "Top layer should remain frozen"
        for p in t.decoder.backbone.lm_head.parameters():
            assert not p.requires_grad, "lm_head should remain frozen"
        for p in t.decoder.backbone.model.embed_tokens.parameters():
            assert not p.requires_grad, "embed_tokens should remain frozen"

    def test_transition_preserves_projection_momentum(self):
        """After transition, optimizer state for projection params is preserved."""
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=10,
            freeze_top_layer=False,
        )

        # Do a few train steps to accumulate optimizer state
        batch = _make_batch()
        for step in range(5):
            t.global_step = step
            t.train_step(batch)

        # Capture projection param optimizer state before transition
        proj_params = [
            p for p in t.encoder.projection_block.parameters() if p.requires_grad
        ]
        pre_state = {}
        for p in proj_params:
            state = t.optimizer.state.get(p)
            if state and "exp_avg" in state:
                pre_state[id(p)] = {
                    "param": p,
                    "exp_avg": state["exp_avg"].clone(),
                    "exp_avg_sq": state["exp_avg_sq"].clone(),
                }

        # Trigger transition
        t.global_step = 10
        t._maybe_end_projection_only()

        # Verify optimizer state is preserved (not reset)
        for _pid, old_state in pre_state.items():
            new_state = t.optimizer.state[old_state["param"]]
            assert torch.equal(old_state["exp_avg"], new_state["exp_avg"]), \
                "exp_avg should be preserved across transition"
            assert torch.equal(old_state["exp_avg_sq"], new_state["exp_avg_sq"]), \
                "exp_avg_sq should be preserved across transition"

    def test_transition_applies_lr_schedule(self):
        """After transition, all optimizer param groups have correct LR."""
        from bgkit.training.scheduling import cosine_with_warmup

        max_steps = 100
        warmup_steps = 10
        lr = 1e-3
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=10,
            freeze_top_layer=False,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            lr=lr,
        )

        t.global_step = 10
        t._maybe_end_projection_only()

        expected_lr = cosine_with_warmup(10, max_steps, warmup_steps, lr)
        for pg in t.optimizer.param_groups:
            assert abs(pg["lr"] - expected_lr) < 1e-10, \
                f"LR should be {expected_lr}, got {pg['lr']}"

    def test_no_op_if_not_frozen(self):
        """_maybe_end_projection_only should be a no-op if decoder isn't frozen."""
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=0,
        )
        n_groups_before = len(t.optimizer.param_groups)
        t.global_step = 100
        t._maybe_end_projection_only()
        assert len(t.optimizer.param_groups) == n_groups_before

    def test_no_op_before_threshold(self):
        """_maybe_end_projection_only should not fire before the threshold step."""
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=100,
        )
        assert t._decoder_frozen is True
        t.global_step = 50
        t._maybe_end_projection_only()
        assert t._decoder_frozen is True


# ---------------------------------------------------------------------------
# Tests: optimizer param groups
# ---------------------------------------------------------------------------


class TestOptimizerGroups:
    def test_optimizer_groups_projection_only(self):
        """During projection-only phase, optimizer covers only projection params."""
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=100,
        )
        # Muon may split into multiple sub-groups, but all params should be projection
        all_opt_params = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        proj_params = {id(p) for p in t.encoder.projection_block.parameters() if p.requires_grad}
        assert all_opt_params == proj_params

    def test_optimizer_groups_after_transition(self):
        """After transition with freeze_top_layer, optimizer has projection + decoder groups."""
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=10,
            freeze_top_layer=True,
            num_layers=4,
        )
        # Before: only projection params
        pre_params = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        proj_params = {id(p) for p in t.encoder.projection_block.parameters() if p.requires_grad}
        assert pre_params == proj_params

        t.global_step = 10
        t._maybe_end_projection_only()

        # After: projection + decoder trainable params
        post_params = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        dec_params = {id(p) for p in t.decoder.parameters() if p.requires_grad}
        assert proj_params.issubset(post_params)
        assert dec_params.issubset(post_params)
        assert len(post_params) > len(pre_params)

    def test_optimizer_groups_no_projection(self):
        """Without projection training, optimizer has only decoder groups."""
        t = _make_trainer(
            train_projection_block=False,
            freeze_top_layer=True,
            num_layers=4,
        )
        # All optimizer params should be decoder params only
        all_opt_params = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        dec_params = {id(p) for p in t.decoder.parameters() if p.requires_grad}
        proj_params = {id(p) for p in t.encoder.projection_block.parameters()}
        assert all_opt_params == dec_params
        assert not proj_params.intersection(all_opt_params)

    def test_projection_lr_custom(self):
        """Custom projection_lr should be used for projection param group."""
        t = _make_trainer(
            train_projection_block=True,
            projection_lr=5e-4,
            lr=1e-3,
        )
        # First group should be projection with custom LR
        proj_group = t.optimizer.param_groups[0]
        assert abs(proj_group["lr"] - 5e-4) < 1e-10
        assert abs(proj_group["base_lr"] - 5e-4) < 1e-10

    def test_base_lr_stored_in_all_groups(self):
        """Every param group should have base_lr for per-group scheduling."""
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=0,
            freeze_top_layer=True,
        )
        for g in t.optimizer.param_groups:
            assert "base_lr" in g, "Param group should have base_lr"


# ---------------------------------------------------------------------------
# Tests: checkpoint round-trip with projection training
# ---------------------------------------------------------------------------


class TestProjectionCheckpoint:
    def test_resume_from_projection_only(self, tmp_path):
        """Save during projection-only phase, load and verify state."""
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=100,
        )
        # Do a train step
        t.train_step(_make_batch())
        t.global_step = 5

        # Save
        ckpt_path = t.save_checkpoint(tmp_path)
        assert t._training_state["decoder_frozen"] is True

        # Mutate
        t.global_step = 999

        # Load
        t.load_checkpoint(Path(ckpt_path))
        assert t.global_step == 5
        assert t._decoder_frozen is True

        # Optimizer should have only projection params
        all_opt_params = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        proj_params = {id(p) for p in t.encoder.projection_block.parameters() if p.requires_grad}
        assert all_opt_params == proj_params

    def test_resume_from_post_transition(self, tmp_path):
        """Save after transition, load and verify decoder+projection optimizer."""
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=10,
            freeze_top_layer=False,
        )
        # Simulate being past transition
        t.global_step = 10
        t._maybe_end_projection_only()
        t.train_step(_make_batch())
        t.global_step = 50

        # Save
        ckpt_path = t.save_checkpoint(tmp_path)
        assert t._training_state["decoder_frozen"] is False

        # Mutate
        t.global_step = 0

        # Load
        t.load_checkpoint(Path(ckpt_path))
        assert t.global_step == 50
        assert t._decoder_frozen is False

        # Should have both projection and decoder params
        all_opt_params = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        proj_params = {id(p) for p in t.encoder.projection_block.parameters() if p.requires_grad}
        dec_params = {id(p) for p in t.decoder.parameters() if p.requires_grad}
        assert proj_params.issubset(all_opt_params)
        assert dec_params.issubset(all_opt_params)

    def test_resume_derives_state_from_step(self, tmp_path):
        """Resume ignores saved decoder_frozen and derives from global_step.

        save_checkpoint() overwrites decoder_frozen with the real value, so we
        write the checkpoint directly with contradictory metadata to verify
        that load_checkpoint derives state from step position.
        """
        from bgkit.training.checkpointing import CheckpointMetadata, save_checkpoint

        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=100,
        )
        t.global_step = 50  # Still in warmup

        # Write checkpoint directly with contradictory decoder_frozen=False
        metadata = CheckpointMetadata(
            phase="phase1_step1",
            step=50,
            epoch=0,
            parent_checkpoint=None,
            training_state={"decoder_frozen": False},  # Contradicts step 50 < 100
        )
        ckpt_path = save_checkpoint(
            tmp_path,
            metadata,
            encoder=t.encoder.state_dict(),
            decoder=t.decoder.state_dict(),
            optimizer=t.optimizer.state_dict(),
        )

        # Load - should derive frozen=True from step 50 < 100, not saved metadata
        t.load_checkpoint(Path(ckpt_path))
        assert t._decoder_frozen is True, "Should derive state from step, not saved metadata"

    def test_decoder_frozen_saved_in_checkpoint(self, tmp_path):
        """save_checkpoint should include decoder_frozen in training_state."""
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=100,
        )
        t.global_step = 5
        t.save_checkpoint(tmp_path)
        assert "decoder_frozen" in t._training_state
        assert t._training_state["decoder_frozen"] is True


# ---------------------------------------------------------------------------
# Tests: config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_invalid_config_raises(self):
        """projection_only_steps > 0 with train_projection_block=False raises ValueError."""
        cfg = OmegaConf.create({
            "training": {
                "phase": "phase1_step1",
                "max_steps": 100,
                "lr": 1e-3,
                "warmup_steps": 10,
                "eval_every": 50,
                "save_every": 0,
                "train_projection_block": False,
                "projection_only_steps": 200,
            },
            "model": {
                "bgkit": {
                    "backbone_name": "fake",
                    "hidden_dim": 64,
                },
                "decoder": {
                    "backbone_name": "fake",
                },
            },
            "data": {
                "tokens": {"input_dir": "/fake"},
            },
            "compute": {},
            "wandb": {"enabled": False},
        })

        t = DecoderInitTrainer(cfg)
        with pytest.raises(ValueError, match="projection_only_steps > 0"):
            t.setup()


# ---------------------------------------------------------------------------
# Tests: train_step integration
# ---------------------------------------------------------------------------


class TestTrainStepWithProjection:
    def test_train_step_returns_metrics(self):
        """train_step with projection training returns expected metrics."""
        t = _make_trainer(train_projection_block=True)
        metrics = t.train_step(_make_batch())
        assert "loss" in metrics
        assert "grad_norm" in metrics

    def test_projection_params_in_optimizer(self):
        """Projection block params should be in optimizer param groups."""
        t = _make_trainer(train_projection_block=True)
        opt_params = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        proj_params = {
            id(p) for p in t.encoder.projection_block.parameters() if p.requires_grad
        }
        assert proj_params.issubset(opt_params), "Projection params should be in optimizer"

    def test_projection_receives_gradients(self):
        """Projection block params receive gradients from a direct loss on survivors.

        NOTE: The mock decoder backbone has no cross-position attention, so the
        reconstruction loss doesn't backprop through the survivor prefix. This test
        verifies gradient flow through the projection block via a direct loss on the
        survivor embeddings, which is the intended real-model gradient path.
        """
        t = _make_trainer(train_projection_block=True)
        batch = _make_batch()
        survivors = t._compute_survivors(batch)
        loss = survivors.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in t.encoder.projection_block.parameters()
            if p.requires_grad
        )
        assert has_grad, "Projection block should have gradients from survivors"

    def test_compressor_no_gradients_after_step(self):
        """Compressor params should have no gradients even when projection is trained."""
        t = _make_trainer(train_projection_block=True)
        t.train_step(_make_batch())

        for name, p in t.encoder.compressor.named_parameters():
            assert p.grad is None or (p.grad == 0).all(), \
                f"compressor.{name} should have no gradient"

    def test_decoder_no_gradients_during_warmup(self):
        """During projection-only warmup, decoder should have no gradients."""
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=100,
        )
        t.global_step = 5
        t.train_step(_make_batch())

        for name, p in t.decoder.named_parameters():
            assert p.grad is None, f"decoder.{name} should have no gradient during warmup"

    def test_projection_eval_mode_during_evaluate(self):
        """Projection block should be in eval mode during evaluate()."""
        from torch.utils.data import DataLoader

        t = _make_trainer(train_projection_block=True)

        # Verify starts in train mode
        assert t.encoder.projection_block.training, "Should start in train mode"

        # Set up a minimal eval dataloader
        eval_data = [
            {
                "token_ids": torch.randint(0, 1000, (15,)),
                "loss_mask": torch.cat([torch.zeros(5, dtype=torch.long),
                                        torch.ones(5, dtype=torch.long),
                                        torch.zeros(5, dtype=torch.long)]),
                "content_token_ids": torch.randint(0, 1000, (5,)),
                "compression_prompt_ids": torch.randint(0, 1000, (3,)),
                "prefix_ids": torch.randint(0, 1000, (5,)),
                "language": "python",
            },
        ]
        t.eval_dataloader = DataLoader(
            eval_data, batch_size=1, collate_fn=collate_chat_repro,
        )
        t.evaluate()

        # After evaluate, projection block should be in eval mode
        assert not t.encoder.projection_block.training, \
            "Projection block should be in eval mode after evaluate()"

    def test_projection_train_mode_restored_after_eval(self):
        """train_step should restore projection block to train mode after eval."""
        from torch.utils.data import DataLoader

        t = _make_trainer(train_projection_block=True)

        # Run evaluate to put projection block in eval mode
        eval_data = [
            {
                "token_ids": torch.randint(0, 1000, (15,)),
                "loss_mask": torch.cat([torch.zeros(5, dtype=torch.long),
                                        torch.ones(5, dtype=torch.long),
                                        torch.zeros(5, dtype=torch.long)]),
                "content_token_ids": torch.randint(0, 1000, (5,)),
                "compression_prompt_ids": torch.randint(0, 1000, (3,)),
                "prefix_ids": torch.randint(0, 1000, (5,)),
                "language": "python",
            },
        ]
        t.eval_dataloader = DataLoader(
            eval_data, batch_size=1, collate_fn=collate_chat_repro,
        )
        t.evaluate()
        assert not t.encoder.projection_block.training

        # train_step should restore train mode
        t.train_step(_make_batch())
        assert t.encoder.projection_block.training, \
            "Projection block should be in train mode during train_step"
