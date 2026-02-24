"""Tests for the Muon+AdamW hybrid optimizer factory."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from omegaconf import OmegaConf

from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpointing import CheckpointMetadata, save_checkpoint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyTrainer(BaseTrainer):
    """Minimal trainer for testing optimizer factory."""

    def setup(self):
        pass

    def _forward_backward(self, batch):
        return {}

    def evaluate(self):
        return {}


def _make_trainer(optimizer: str = "adamw") -> _DummyTrainer:
    cfg = OmegaConf.create({"training": {"phase": "test", "optimizer": optimizer}})
    t = _DummyTrainer(cfg)
    t.device = torch.device("cpu")
    return t


def _make_params():
    """Create a mix of 2D and 1D parameters."""
    linear = torch.nn.Linear(8, 4)  # weight is 2D, bias is 1D
    norm = torch.nn.LayerNorm(4)  # weight and bias are 1D
    return linear, norm


# ---------------------------------------------------------------------------
# Core factory tests
# ---------------------------------------------------------------------------


class TestCreateOptimizer:
    def test_adamw_default(self):
        t = _make_trainer("adamw")
        linear, _ = _make_params()
        groups = [{"params": list(linear.parameters()), "lr": 1e-3, "base_lr": 1e-3}]
        opt = t._create_optimizer(groups, 1e-3)
        assert isinstance(opt, torch.optim.AdamW)
        assert t._optimizer_type == "adamw"

    def test_adamw8bit_without_bnb_raises(self):
        t = _make_trainer("adamw8bit")
        linear, _ = _make_params()
        groups = [{"params": list(linear.parameters()), "lr": 1e-3}]
        with (
            patch.dict("sys.modules", {"bitsandbytes": None}),
            pytest.raises((ImportError, RuntimeError)),
        ):
            t._create_optimizer(groups, 1e-3)

    def test_muon_creates_correct_type(self):
        from bgkit.training.muon import Muon

        t = _make_trainer("muon")
        linear, norm = _make_params()
        all_params = list(linear.parameters()) + list(norm.parameters())
        groups = [{"params": all_params, "lr": 1e-3, "base_lr": 1e-3}]
        opt = t._create_optimizer(groups, 1e-3)
        assert isinstance(opt, Muon)
        assert t._optimizer_type == "muon"

    def test_invalid_optimizer_raises(self):
        t = _make_trainer("invalid")
        linear, _ = _make_params()
        groups = [{"params": list(linear.parameters()), "lr": 1e-3}]
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            t._create_optimizer(groups, 1e-3)


# ---------------------------------------------------------------------------
# Muon param splitting
# ---------------------------------------------------------------------------


class TestMuonSplit:
    def test_split_by_ndim(self):
        t = _make_trainer("muon")
        linear, norm = _make_params()
        all_params = list(linear.parameters()) + list(norm.parameters())
        groups = [{"params": all_params, "lr": 1e-3}]
        split = t._split_for_muon(groups)

        muon_groups = [g for g in split if g["use_muon"]]
        adam_groups = [g for g in split if not g["use_muon"]]

        # linear.weight (2D) -> muon
        assert len(muon_groups) == 1
        muon_params = muon_groups[0]["params"]
        assert len(muon_params) == 1
        assert muon_params[0].ndim == 2

        # linear.bias (1D) + norm.weight (1D) + norm.bias (1D) -> adam
        assert len(adam_groups) == 1
        adam_params = adam_groups[0]["params"]
        assert len(adam_params) == 3
        assert all(p.ndim < 2 for p in adam_params)

    def test_split_preserves_metadata(self):
        t = _make_trainer("muon")
        linear, _ = _make_params()
        groups = [
            {
                "params": list(linear.parameters()),
                "lr": 5e-4,
                "base_lr": 5e-4,
                "weight_decay": 0.01,
            }
        ]
        split = t._split_for_muon(groups)
        for g in split:
            assert g["lr"] == 5e-4
            assert g["base_lr"] == 5e-4
            assert g["weight_decay"] == 0.01

    def test_muon_excludes_embedding_params(self):
        t = _make_trainer("muon")
        linear, _ = _make_params()
        emb = torch.nn.Embedding(10, 8)  # 2D but should be excluded
        all_params = list(linear.parameters()) + list(emb.parameters())
        exclude = frozenset(id(p) for p in emb.parameters())
        groups = [{"params": all_params, "lr": 1e-3}]
        split = t._split_for_muon(groups, exclude_ids=exclude)

        muon_groups = [g for g in split if g["use_muon"]]
        adam_groups = [g for g in split if not g["use_muon"]]

        # linear.weight (2D, not excluded) -> muon
        muon_params = [p for g in muon_groups for p in g["params"]]
        assert len(muon_params) == 1
        assert muon_params[0] is linear.weight

        # linear.bias (1D) + emb.weight (2D, excluded) -> adam
        adam_params = [p for g in adam_groups for p in g["params"]]
        adam_ids = {id(p) for p in adam_params}
        assert id(emb.weight) in adam_ids
        assert id(linear.bias) in adam_ids


# ---------------------------------------------------------------------------
# _add_param_group_to_optimizer
# ---------------------------------------------------------------------------


class TestAddParamGroup:
    def test_add_param_group_muon_splits(self):
        from bgkit.training.muon import Muon

        t = _make_trainer("muon")
        linear1, norm1 = _make_params()
        initial_params = list(linear1.parameters()) + list(norm1.parameters())
        t.optimizer = t._create_optimizer(
            [{"params": initial_params, "lr": 1e-3, "base_lr": 1e-3}], 1e-3
        )
        initial_groups = len(t.optimizer.param_groups)

        # Add a new group with mixed params
        linear2 = torch.nn.Linear(4, 2)
        t._add_param_group_to_optimizer(
            {"params": list(linear2.parameters()), "lr": 2e-3, "base_lr": 2e-3}
        )
        # Should add 2 groups (one muon, one adam)
        assert len(t.optimizer.param_groups) == initial_groups + 2

    def test_add_param_group_muon_honors_exclude(self):
        t = _make_trainer("muon")
        emb = torch.nn.Embedding(10, 8)
        linear, _ = _make_params()

        # Set exclude set before creating optimizer
        exclude = frozenset(id(p) for p in emb.parameters())
        t.optimizer = t._create_optimizer(
            [{"params": list(linear.parameters()), "lr": 1e-3, "base_lr": 1e-3}],
            1e-3,
            exclude_from_muon=exclude,
        )

        # Add embedding params — should go to AdamW despite being 2D
        t._add_param_group_to_optimizer(
            {"params": list(emb.parameters()), "lr": 1e-3, "base_lr": 1e-3}
        )
        last_group = t.optimizer.param_groups[-1]
        assert last_group["use_muon"] is False

    def test_muon_group_ordering_stable_across_rebuild(self):
        """Group ordering must match between incremental add and full rebuild.

        If ordering differs, checkpoint resume maps optimizer states to wrong
        param groups (wrong LRs, momentum buffers for wrong shapes).
        """
        t = _make_trainer("muon")

        # Phase 1: build with projection only
        proj = torch.nn.Linear(8, 4)
        t.optimizer = t._create_optimizer(
            [{"params": list(proj.parameters()), "lr": 0.02, "base_lr": 0.02}], 0.02
        )

        # Phase 2: add decoder groups incrementally
        dec = torch.nn.Linear(16, 8)
        t._add_param_group_to_optimizer(
            {"params": list(dec.parameters()), "lr": 0.01, "base_lr": 0.01}
        )
        incremental_order = [
            (g["use_muon"], g["lr"]) for g in t.optimizer.param_groups
        ]

        # Rebuild from scratch with all groups (simulates checkpoint resume)
        t2 = _make_trainer("muon")
        t2.optimizer = t2._create_optimizer(
            [
                {"params": list(proj.parameters()), "lr": 0.02, "base_lr": 0.02},
                {"params": list(dec.parameters()), "lr": 0.01, "base_lr": 0.01},
            ],
            0.02,
        )
        rebuild_order = [
            (g["use_muon"], g["lr"]) for g in t2.optimizer.param_groups
        ]

        assert incremental_order == rebuild_order

    def test_add_param_group_adamw_passthrough(self):
        t = _make_trainer("adamw")
        linear, _ = _make_params()
        t.optimizer = t._create_optimizer(
            [{"params": list(linear.parameters()), "lr": 1e-3, "base_lr": 1e-3}], 1e-3
        )
        initial_groups = len(t.optimizer.param_groups)

        linear2 = torch.nn.Linear(4, 2)
        t._add_param_group_to_optimizer(
            {"params": list(linear2.parameters()), "lr": 2e-3}
        )
        assert len(t.optimizer.param_groups) == initial_groups + 1


# ---------------------------------------------------------------------------
# Checkpoint restore robustness
# ---------------------------------------------------------------------------


class TestCheckpointRestore:
    def test_optimizer_type_mismatch_raises(self, tmp_path):
        t = _make_trainer("adamw")
        model = torch.nn.Linear(4, 2)
        t.model = model
        t.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Save with optimizer_type="muon"
        metadata = CheckpointMetadata(
            phase="test",
            step=10,
            epoch=0,
            parent_checkpoint=None,
            optimizer_type="muon",
        )
        ckpt = save_checkpoint(tmp_path, metadata, model=model.state_dict())

        with pytest.raises(RuntimeError, match="Optimizer type mismatch"):
            t.load_checkpoint(Path(ckpt))

    def test_old_checkpoint_no_optimizer_type_loads(self, tmp_path):
        t = _make_trainer("adamw")
        model = torch.nn.Linear(4, 2)
        t.model = model
        t.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Save without optimizer_type (simulates old checkpoint)
        metadata = CheckpointMetadata(
            phase="test", step=10, epoch=0, parent_checkpoint=None
        )
        ckpt = save_checkpoint(
            tmp_path, metadata, model=model.state_dict(),
            optimizer=t.optimizer.state_dict(),
        )

        # Should not raise
        t.load_checkpoint(Path(ckpt))
        assert t.global_step == 10

    def test_old_checkpoint_different_optimizer_warns(self, tmp_path):
        """Old checkpoint (optimizer_type=None) resumed with different optimizer
        should warn and skip optimizer state, not crash."""
        # Save with AdamW (no optimizer_type in metadata — simulates old checkpoint)
        model = torch.nn.Linear(4, 2)
        adamw_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metadata = CheckpointMetadata(
            phase="test", step=10, epoch=0, parent_checkpoint=None
        )
        ckpt = save_checkpoint(
            tmp_path, metadata, model=model.state_dict(),
            optimizer=adamw_opt.state_dict(),
        )

        # Resume with Muon — different group count, should warn not crash
        t = _make_trainer("muon")
        t.model = model
        t.optimizer = t._create_optimizer(
            [{"params": list(model.parameters()), "lr": 1e-3, "base_lr": 1e-3}],
            1e-3,
        )
        t.load_checkpoint(Path(ckpt))
        assert t.global_step == 10  # state restored despite optimizer skip

    def test_matching_optimizer_type_loads(self, tmp_path):
        t = _make_trainer("adamw")
        model = torch.nn.Linear(4, 2)
        t.model = model
        t.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        metadata = CheckpointMetadata(
            phase="test", step=10, epoch=0, parent_checkpoint=None,
            optimizer_type="adamw",
        )
        ckpt = save_checkpoint(
            tmp_path, metadata, model=model.state_dict(),
            optimizer=t.optimizer.state_dict(),
        )

        t.load_checkpoint(Path(ckpt))
        assert t.global_step == 10

    def test_save_checkpoint_writes_optimizer_type(self, tmp_path):
        t = _make_trainer("muon")
        model = torch.nn.Linear(4, 2)
        t.model = model
        t.optimizer = t._create_optimizer(
            [{"params": list(model.parameters()), "lr": 1e-3, "base_lr": 1e-3}],
            1e-3,
        )

        ckpt_path = t.save_checkpoint(tmp_path)
        meta = json.loads((ckpt_path / "metadata.json").read_text())
        assert meta["optimizer_type"] == "muon"


# ---------------------------------------------------------------------------
# Compression lm_head exclusion
# ---------------------------------------------------------------------------


class TestCompressionMuonExclusion:
    def test_lm_head_excluded_non_peft(self):
        """_muon_excluded_param_ids must find lm_head on the CausalLM level."""
        from bgkit.training.phase1.compression import CompressionTrainer

        # Build a minimal mock matching Qwen CausalLM structure:
        # decoder.backbone.model.embed_tokens (2D), decoder.backbone.lm_head (2D)
        embed = torch.nn.Embedding(100, 64)
        lm_head = torch.nn.Linear(64, 100, bias=False)

        inner = torch.nn.Module()
        inner.embed_tokens = embed

        causal_lm = torch.nn.Module()
        causal_lm.model = inner
        causal_lm.lm_head = lm_head

        decoder = torch.nn.Module()
        decoder.backbone = causal_lm

        # Patch a trainer enough to call _muon_excluded_param_ids
        cfg = OmegaConf.create({"training": {"phase": "test", "optimizer": "muon"}})
        trainer = object.__new__(CompressionTrainer)
        trainer.cfg = cfg
        trainer.decoder = decoder

        exclude = trainer._muon_excluded_param_ids()

        # Both embed and lm_head params should be excluded
        embed_ids = {id(p) for p in embed.parameters()}
        lm_head_ids = {id(p) for p in lm_head.parameters()}
        assert embed_ids.issubset(exclude), "embed_tokens params not excluded"
        assert lm_head_ids.issubset(exclude), "lm_head params not excluded"

    def test_lm_head_excluded_after_peft_wrap(self):
        """After PeftModel wrapping, frozen params should NOT be in the exclusion set."""
        from bgkit.training.phase1.compression import CompressionTrainer

        embed = torch.nn.Embedding(100, 64)
        lm_head = torch.nn.Linear(64, 100, bias=False)
        # Freeze them (simulates LoRA freezing non-adapter params)
        embed.requires_grad_(False)
        lm_head.requires_grad_(False)

        inner = torch.nn.Module()
        inner.embed_tokens = embed

        causal_lm = torch.nn.Module()
        causal_lm.model = inner
        causal_lm.lm_head = lm_head

        # Simulate PeftModel wrapping: backbone.base_model.model = CausalLM
        base_model = torch.nn.Module()
        base_model.model = causal_lm

        peft_backbone = torch.nn.Module()
        peft_backbone.base_model = base_model

        decoder = torch.nn.Module()
        decoder.backbone = peft_backbone

        cfg = OmegaConf.create({"training": {"phase": "test", "optimizer": "muon"}})
        trainer = object.__new__(CompressionTrainer)
        trainer.cfg = cfg
        trainer.decoder = decoder

        exclude = trainer._muon_excluded_param_ids()
        # All frozen → empty set
        assert len(exclude) == 0


# ---------------------------------------------------------------------------
# Muon optimizer basic functionality
# ---------------------------------------------------------------------------


class TestMuonOptimizer:
    def test_step_updates_params(self):
        from bgkit.training.muon import Muon

        linear = torch.nn.Linear(8, 4)
        norm = torch.nn.LayerNorm(4)

        groups = [
            {"params": [linear.weight], "use_muon": True, "lr": 0.02},
            {"params": [linear.bias, norm.weight, norm.bias], "use_muon": False, "lr": 1e-3},
        ]
        opt = Muon(groups)

        # Fake gradients
        for p in [linear.weight, linear.bias, norm.weight, norm.bias]:
            p.grad = torch.randn_like(p)

        w_before = linear.weight.clone()
        b_before = linear.bias.clone()

        opt.step()

        assert not torch.equal(linear.weight, w_before)
        assert not torch.equal(linear.bias, b_before)

    def test_add_param_group_injects_defaults(self):
        """add_param_group must inject momentum/betas defaults for new groups."""
        from bgkit.training.muon import Muon

        linear1 = torch.nn.Linear(8, 4)
        opt = Muon([
            {"params": [linear1.weight], "use_muon": True, "lr": 0.02},
            {"params": [linear1.bias], "use_muon": False, "lr": 1e-3},
        ])

        # Add groups with only lr/base_lr (like decoder_init's projection transition)
        linear2 = torch.nn.Linear(4, 2)
        opt.add_param_group(
            {"params": [linear2.weight], "use_muon": True, "lr": 0.01, "base_lr": 0.01}
        )
        opt.add_param_group(
            {"params": [linear2.bias], "use_muon": False, "lr": 1e-3, "base_lr": 1e-3}
        )

        # Step must not KeyError on 'momentum' or 'betas'
        for p in [linear1.weight, linear1.bias, linear2.weight, linear2.bias]:
            p.grad = torch.randn_like(p)
        opt.step()  # would KeyError without add_param_group override

    def test_missing_use_muon_raises(self):
        from bgkit.training.muon import Muon

        linear = torch.nn.Linear(4, 2)
        with pytest.raises(ValueError, match="use_muon"):
            Muon([{"params": list(linear.parameters()), "lr": 1e-3}])
