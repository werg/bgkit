from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")
from torch import nn

from bgkit.training.phase1.projection_seed_falcon import FalconProjectionSeedTrainer


def _trainer(phase: str) -> FalconProjectionSeedTrainer:
    cfg = OmegaConf.create({
        "training": {"phase": phase, "optimizer": "adamw"},
        "checkpoint_dir": "/tmp/ckpts",
    })
    return FalconProjectionSeedTrainer(cfg)


def test_dense_seed_auto_uses_phase1_step5_checkpoint():
    trainer = _trainer("phase1_falcon_dense_seed")
    source = Path("/tmp/ckpts/phase1_step5_step6000_20260501")

    with patch(
        "bgkit.training.phase1.projection_seed_falcon.resolve_checkpoint",
        return_value=source,
    ) as mock_resolve:
        result = trainer._resolve_source_checkpoint()

    mock_resolve.assert_called_once_with(
        Path("/tmp/ckpts"),
        phase="phase1_step5",
        metric="eval/loss",
        label="bgkit_checkpoint",
    )
    assert result == str(source)


def test_forced_adapt_auto_prefers_dense_seed_checkpoint():
    trainer = _trainer("phase1_falcon_forced_adapt")
    source = Path("/tmp/ckpts/phase1_falcon_dense_seed_step4000_20260501")

    with patch(
        "bgkit.training.phase1.projection_seed_falcon.resolve_checkpoint",
        return_value=source,
    ) as mock_resolve:
        result = trainer._resolve_source_checkpoint()

    mock_resolve.assert_called_once_with(
        Path("/tmp/ckpts"),
        phase="phase1_falcon_dense_seed",
        metric="eval/loss",
        label="bgkit_checkpoint",
    )
    assert result == str(source)


def test_forced_adapt_auto_falls_back_to_step5_when_dense_seed_missing():
    trainer = _trainer("phase1_falcon_forced_adapt")
    source = Path("/tmp/ckpts/phase1_step5_step6000_20260501")

    with patch(
        "bgkit.training.phase1.projection_seed_falcon.resolve_checkpoint",
        side_effect=[ValueError("no dense seed"), source],
    ) as mock_resolve:
        result = trainer._resolve_source_checkpoint()

    assert result == str(source)
    assert [call.kwargs["phase"] for call in mock_resolve.call_args_list] == [
        "phase1_falcon_dense_seed",
        "phase1_step5",
    ]


def test_forced_adapt_auto_reports_candidate_chain_when_missing():
    trainer = _trainer("phase1_falcon_forced_adapt")

    with patch(
        "bgkit.training.phase1.projection_seed_falcon.resolve_checkpoint",
        side_effect=ValueError("missing"),
    ), pytest.raises(ValueError, match=r"phase1_falcon_dense_seed.*phase1_step2p5"):
        trainer._resolve_source_checkpoint()


def test_training_mode_helpers_keep_falcon_projection_trainable():
    trainer = FalconProjectionSeedTrainer.__new__(FalconProjectionSeedTrainer)

    class _Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.l0 = nn.Dropout()
            self.projection_blocks = nn.ModuleDict({
                "qwen35": nn.Dropout(),
                "falcon_h1": nn.Dropout(),
            })

    trainer.encoder = _Encoder()
    trainer.decoder = nn.Dropout()
    trainer._adapt_encoder = False

    trainer._set_training_modes()
    assert trainer.encoder.training is False
    assert trainer.encoder.l0.training is False
    assert trainer.encoder.projection_blocks["qwen35"].training is False
    assert trainer.encoder.projection_blocks["falcon_h1"].training is True

    trainer._set_evaluation_modes()
    assert trainer.encoder.projection_blocks["falcon_h1"].training is False

    trainer._set_training_modes()
    assert trainer.encoder.projection_blocks["falcon_h1"].training is True
