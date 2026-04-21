"""Config gating for gradient checkpointing.

``maybe_enable_gradient_checkpointing`` should enable checkpointing by
default, honor ``compute.gradient_checkpointing: false`` to skip it, and
let ``training.gradient_checkpointing`` override ``compute`` when explicit.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from omegaconf import OmegaConf

from bgkit.training.gradient_utils import (
    gradient_checkpointing_requested,
    maybe_enable_gradient_checkpointing,
)


def _cfg(**sections):
    return OmegaConf.create(sections)


class TestGradientCheckpointingRequested:
    def test_default_is_true(self):
        cfg = _cfg(compute={}, training={})
        assert gradient_checkpointing_requested(cfg) is True

    def test_compute_false_disables(self):
        cfg = _cfg(compute={"gradient_checkpointing": False}, training={})
        assert gradient_checkpointing_requested(cfg) is False

    def test_compute_true_enables(self):
        cfg = _cfg(compute={"gradient_checkpointing": True}, training={})
        assert gradient_checkpointing_requested(cfg) is True

    def test_training_overrides_compute_true(self):
        cfg = _cfg(
            compute={"gradient_checkpointing": False},
            training={"gradient_checkpointing": True},
        )
        assert gradient_checkpointing_requested(cfg) is True

    def test_training_overrides_compute_false(self):
        cfg = _cfg(
            compute={"gradient_checkpointing": True},
            training={"gradient_checkpointing": False},
        )
        assert gradient_checkpointing_requested(cfg) is False


class TestMaybeEnableGradientCheckpointing:
    def test_enables_when_requested(self):
        model = MagicMock()
        cfg = _cfg(compute={"gradient_checkpointing": True}, training={})
        assert maybe_enable_gradient_checkpointing(model, cfg) is True
        model.gradient_checkpointing_enable.assert_called_once()

    def test_skips_when_disabled(self):
        model = MagicMock()
        cfg = _cfg(compute={"gradient_checkpointing": False}, training={})
        assert maybe_enable_gradient_checkpointing(model, cfg) is False
        model.gradient_checkpointing_enable.assert_not_called()

    def test_default_enables(self):
        # No explicit key set anywhere — defaults to True (backward-compat).
        model = MagicMock()
        cfg = _cfg(compute={}, training={})
        assert maybe_enable_gradient_checkpointing(model, cfg) is True
        model.gradient_checkpointing_enable.assert_called_once()
