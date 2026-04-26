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
    """The ``"selective"`` mode skips checkpointing for DeltaNet decoder
    layers (those with a ``linear_attn`` submodule) and applies it to all
    other layers via ``torch.utils.checkpoint``."""

    @staticmethod
    def _build_fake_model_and_layers():
        """Mock model + a DeltaNet-flavored and a FullAttn-flavored layer."""
        import torch.nn as nn

        deltanet_layer = nn.Module()
        deltanet_layer.linear_attn = nn.Module()  # marker

        full_attn_layer = nn.Module()
        full_attn_layer.self_attn = nn.Module()  # marker

        model = MagicMock()
        # Selective installer walks model -> model.model -> .language_model.
        # Make .model None so it stops at the top-level mock.
        model.model = None
        return model, deltanet_layer, full_attn_layer

    def test_selective_mode_installs_func(self):
        model, *_ = self._build_fake_model_and_layers()
        cfg = _cfg(compute={}, training={"gradient_checkpointing": "selective"})
        assert maybe_enable_gradient_checkpointing(model, cfg) is True
        model.gradient_checkpointing_enable.assert_called_once()
        assert callable(model._gradient_checkpointing_func)

    def test_selective_skips_deltanet_layer(self):
        """For a DeltaNet layer, the selective func calls fn directly (no checkpoint)."""
        model, deltanet_layer, _ = self._build_fake_model_and_layers()
        cfg = _cfg(compute={}, training={"gradient_checkpointing": "selective"})
        maybe_enable_gradient_checkpointing(model, cfg)

        # Mimic HF's pattern: layer.__call__ is a bound method whose __self__
        # is the layer. We synthesize that by attaching a method to the layer.
        called_with = {}

        def _layer_call(self_arg, x, y):
            called_with["args"] = (x, y)
            return x + y

        deltanet_layer.__class__ = type(
            "DeltaNetMockClass",
            (type(deltanet_layer),),
            {"__call__": _layer_call},
        )

        result = model._gradient_checkpointing_func(deltanet_layer.__call__, 1, 2)
        assert result == 3
        assert called_with["args"] == (1, 2)

    def test_selective_checkpoints_full_attn_layer(self):
        """For a non-DeltaNet layer, the selective func wraps fn in torch.utils.checkpoint."""
        model, _, full_attn_layer = self._build_fake_model_and_layers()
        cfg = _cfg(compute={}, training={"gradient_checkpointing": "selective"})
        maybe_enable_gradient_checkpointing(model, cfg)

        import torch

        def _layer_call(self_arg, t):
            return t * 2

        full_attn_layer.__class__ = type(
            "FullAttnMockClass",
            (type(full_attn_layer),),
            {"__call__": _layer_call},
        )

        x = torch.randn(2, 2, requires_grad=True)
        result = model._gradient_checkpointing_func(full_attn_layer.__call__, x)
        assert torch.allclose(result, x * 2)
        result.sum().backward()
        assert x.grad is not None

    def test_string_aliases_resolve(self):
        """``selective``/``deltanet_off``/``skip_deltanet`` all map to selective mode."""
        from bgkit.training.gradient_utils import _coerce_gradient_checkpointing_value
        for alias in ("selective", "deltanet_off", "skip_deltanet", "SELECTIVE"):
            assert _coerce_gradient_checkpointing_value(alias) == "selective"
        assert _coerce_gradient_checkpointing_value("true") is True
        assert _coerce_gradient_checkpointing_value("false") is False
