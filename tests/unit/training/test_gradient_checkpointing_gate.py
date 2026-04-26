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


class TestSelectiveGradientCheckpointing:
    """The ``"selective"`` mode unsets ``gradient_checkpointing`` on each
    DeltaNet decoder layer (HF v5+ ``GradientCheckpointingLayer`` reads the
    per-layer flag in its ``__call__``)."""

    @staticmethod
    def _build_layer_model():
        """Build a mock model with a mix of DeltaNet and FullAttn layers."""
        import torch.nn as nn

        class _Layer(nn.Module):
            def __init__(self, kind: str):
                super().__init__()
                # Mimic HF: each layer carries its own gradient_checkpointing flag.
                self.gradient_checkpointing = False
                if kind == "deltanet":
                    self.linear_attn = nn.Module()
                else:
                    self.self_attn = nn.Module()

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                # 3 DeltaNet + 1 FullAttention, repeated twice → 8 layers.
                kinds = (["deltanet"] * 3 + ["full"]) * 2
                self.layers = nn.ModuleList([_Layer(k) for k in kinds])

            def gradient_checkpointing_enable(self, **kwargs):
                # HF's enable propagates the flag to every layer.
                for layer in self.layers:
                    layer.gradient_checkpointing = True

        return _Model()

    def test_selective_mode_disables_only_deltanet_layers(self):
        model = self._build_layer_model()
        cfg = _cfg(compute={}, training={"gradient_checkpointing": "selective"})
        assert maybe_enable_gradient_checkpointing(model, cfg) is True

        # 6 DeltaNet layers (3 per repeat × 2) had ckpt disabled.
        deltanet_layers = [layer for layer in model.layers if hasattr(layer, "linear_attn")]
        full_layers = [layer for layer in model.layers if hasattr(layer, "self_attn")]
        assert len(deltanet_layers) == 6
        assert len(full_layers) == 2
        for layer in deltanet_layers:
            assert layer.gradient_checkpointing is False, "DeltaNet layer should be skipped"
        for layer in full_layers:
            assert layer.gradient_checkpointing is True, "FullAttn layer should keep ckpt"

    def test_selective_returns_disabled_count(self):
        from bgkit.training.gradient_utils import _install_selective_checkpoint_func
        model = self._build_layer_model()
        # Manually enable first, like maybe_enable_gradient_checkpointing does.
        model.gradient_checkpointing_enable()
        disabled = _install_selective_checkpoint_func(model)
        assert disabled == 6

    def test_selective_no_op_when_ckpt_not_enabled(self):
        """Don't quietly mask misconfiguration — only flip layers that
        actually had ckpt enabled."""
        from bgkit.training.gradient_utils import _install_selective_checkpoint_func
        model = self._build_layer_model()
        # Skip enable.
        disabled = _install_selective_checkpoint_func(model)
        assert disabled == 0

    def test_string_aliases_resolve(self):
        """``selective``/``deltanet_off``/``skip_deltanet`` all map to selective mode."""
        from bgkit.training.gradient_utils import _coerce_gradient_checkpointing_value
        for alias in ("selective", "deltanet_off", "skip_deltanet", "SELECTIVE"):
            assert _coerce_gradient_checkpointing_value(alias) == "selective"
        assert _coerce_gradient_checkpointing_value("true") is True
        assert _coerce_gradient_checkpointing_value("false") is False
