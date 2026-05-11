from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpoint_registry import RegistryEntry
from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer


def _trainer(decoder_family: str) -> KRKBTrainer:
    cfg = OmegaConf.create({
        "training": {
            "phase": "phase2_kb",
            "optimizer": "adamw",
            "phase1_checkpoint": "auto",
        },
        "checkpoint_dir": "/tmp/ckpts",
        "model": {"decoder": {"family": decoder_family}},
    })
    trainer = KRKBTrainer(cfg)
    trainer.step_cfg = cfg.training
    trainer._decoder_family = decoder_family
    return trainer


def test_phase2_falcon_phase1_auto_prefers_falcon_l1_checkpoint():
    trainer = _trainer("falcon_h1")
    source = Path("/tmp/ckpts/phase1_falcon_l1_step20000_20260501")

    with patch(
        "bgkit.training.checkpoint_registry.resolve_checkpoint",
        return_value=source,
    ) as mock_resolve:
        result = trainer._resolve_phase1_checkpoint()

    mock_resolve.assert_called_once_with(
        Path("/tmp/ckpts"),
        phase="phase1_falcon_l1",
        metric="eval/loss",
        lower_is_better=True,
    )
    assert result == str(source)


def test_phase2_falcon_phase1_auto_falls_back_to_falcon_l0_checkpoint():
    trainer = _trainer("falcon_h1")
    source = Path("/tmp/ckpts/phase1_falcon_l0_step12000_20260501")

    with patch(
        "bgkit.training.checkpoint_registry.resolve_checkpoint",
        side_effect=[ValueError("missing l1"), source],
    ) as mock_resolve:
        result = trainer._resolve_phase1_checkpoint()

    assert result == str(source)
    assert [call.kwargs["phase"] for call in mock_resolve.call_args_list] == [
        "phase1_falcon_l1",
        "phase1_falcon_l0",
    ]


def test_phase2_qwen_phase1_auto_still_uses_step6_checkpoint():
    trainer = _trainer("qwen35")
    source = Path("/tmp/ckpts/phase1_step6_step60000_20260501")

    with patch(
        "bgkit.training.checkpoint_registry.resolve_checkpoint",
        return_value=source,
    ) as mock_resolve:
        result = trainer._resolve_phase1_checkpoint()

    mock_resolve.assert_called_once_with(
        Path("/tmp/ckpts"),
        phase="phase1_step6",
        metric="eval/loss",
        lower_is_better=True,
    )
    assert result == str(source)


def test_base_trainer_eval_metric_normalization_is_idempotent():
    assert BaseTrainer._normalize_eval_metrics({
        "loss": 1.0,
        "eval/token_accuracy": 0.5,
    }) == {
        "eval/loss": 1.0,
        "eval/token_accuracy": 0.5,
    }


def test_stage_a_auto_accepts_single_or_legacy_double_eval_prefix():
    trainer = _trainer("falcon_h1")
    trainer.step_cfg.stage_a_checkpoint = "auto"

    entries = [
        RegistryEntry(
            name="stage_a_legacy",
            phase="phase2_kb",
            step=100,
            epoch=0,
            timestamp="",
            metrics={"eval/eval/loss": 1.2},
            config_snapshot={
                "stage": "A",
                "model": {"decoder": {"family": "falcon_h1"}},
            },
        ),
        RegistryEntry(
            name="stage_a_current",
            phase="phase2_kb",
            step=200,
            epoch=0,
            timestamp="",
            metrics={"eval/loss": 0.9},
            config_snapshot={
                "stage": "A",
                "model": {"decoder": {"family": "falcon_h1"}},
            },
        ),
    ]

    class _Registry:
        def __init__(self, _checkpoint_dir):
            pass

        def backfill(self, _checkpoint_dir):
            pass

        def list_entries(self, phase=None, status=None):
            assert phase == "phase2_kb"
            assert status == "completed"
            return entries

    with patch("bgkit.training.checkpoint_registry.CheckpointRegistry", _Registry):
        result = trainer._resolve_stage_a_checkpoint()

    assert result == "/tmp/ckpts/stage_a_current"
