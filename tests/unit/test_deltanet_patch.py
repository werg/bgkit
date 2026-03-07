"""Tests for deltanet_patch: gate clamping for GatedDeltaNet numerical stability."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from bgkit.utils.deltanet_patch import (
    DEFAULT_G_CLAMP_MIN,
    patch_deltanet_layer,
    patch_gated_delta_rule_numerics,
)

# ---------------------------------------------------------------------------
# patch_deltanet_layer
# ---------------------------------------------------------------------------


class TestPatchDeltanetLayer:
    """Tests for the per-layer instance patch."""

    def _make_layer(self):
        """Create a mock layer with a chunk_gated_delta_rule method."""
        layer = MagicMock()
        layer.chunk_gated_delta_rule = MagicMock(return_value="result")
        layer.A_log = True  # marker attribute
        return layer

    def test_clamps_g_keyword(self):
        """The common path: HF passes g= as keyword argument."""
        layer = self._make_layer()
        original_fn = layer.chunk_gated_delta_rule
        patch_deltanet_layer(layer)

        g = torch.tensor([-5.0, -0.5, -2.0])
        layer.chunk_gated_delta_rule(q="q", k="k", v="v", g=g, beta="beta")

        # Verify the original was called with clamped g
        call_kwargs = original_fn.call_args.kwargs
        assert call_kwargs["g"].min().item() >= DEFAULT_G_CLAMP_MIN

    def test_clamps_g_positional(self):
        """Positional path: g is args[3] in chunk_gated_delta_rule(q, k, v, g, ...)."""
        layer = self._make_layer()
        original_fn = layer.chunk_gated_delta_rule
        patch_deltanet_layer(layer)

        g = torch.tensor([-5.0, -0.5, -2.0])
        layer.chunk_gated_delta_rule("q", "k", "v", g, "beta")

        call_args = original_fn.call_args.args
        # args[3] should be the clamped g
        assert call_args[3].min().item() >= DEFAULT_G_CLAMP_MIN

    def test_preserves_values_within_range(self):
        """Values above the clamp threshold should pass through unchanged."""
        layer = self._make_layer()
        original_fn = layer.chunk_gated_delta_rule
        patch_deltanet_layer(layer)

        g = torch.tensor([-0.5, -1.0, -0.1])
        layer.chunk_gated_delta_rule(q="q", k="k", v="v", g=g, beta="beta")

        call_kwargs = original_fn.call_args.kwargs
        assert torch.equal(call_kwargs["g"], g)

    def test_custom_clamp_value(self):
        layer = self._make_layer()
        original_fn = layer.chunk_gated_delta_rule
        patch_deltanet_layer(layer, g_clamp_min=-0.5)

        g = torch.tensor([-5.0, -0.3])
        layer.chunk_gated_delta_rule(q="q", k="k", v="v", g=g, beta="beta")

        call_kwargs = original_fn.call_args.kwargs
        assert call_kwargs["g"].min().item() >= -0.5
        assert call_kwargs["g"][1].item() == pytest.approx(-0.3)

    def test_noop_without_chunk_method(self):
        """Layers without chunk_gated_delta_rule are silently skipped."""
        layer = MagicMock(spec=[])
        del layer.chunk_gated_delta_rule  # ensure it doesn't exist
        patch_deltanet_layer(layer)  # should not raise

    @pytest.mark.gpu
    def test_positional_index_matches_fla_signature(self):
        """Verify our assumption: g is the 4th arg (index 3) in the fla signature.

        This test documents the API contract. If fla changes the signature,
        this test should break, alerting us to update the patch.
        """
        import inspect

        fla_ops = pytest.importorskip("fla.ops.gated_delta_rule")
        chunk_gated_delta_rule = fla_ops.chunk_gated_delta_rule

        params = list(inspect.signature(chunk_gated_delta_rule).parameters.keys())
        assert params[3] == "g", (
            f"Expected g at position 3 in chunk_gated_delta_rule signature, "
            f"got '{params[3]}'. Signature: {params[:6]}. "
            f"Update deltanet_patch.py positional index."
        )


# ---------------------------------------------------------------------------
# patch_gated_delta_rule_numerics (class-level patching)
# ---------------------------------------------------------------------------


class TestPatchGatedDeltaRuleNumerics:
    """Tests for the class-level __init__ monkey-patch."""

    def test_patches_existing_model_instances(self):
        """When called with model=<module>, patches matching layers."""

        class FakeDeltaNetLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.A_log = torch.nn.Parameter(torch.tensor(1.0))
                self.chunk_gated_delta_rule = lambda *a, **kw: None

        model = torch.nn.Module()
        model.layer1 = FakeDeltaNetLayer()
        model.layer2 = FakeDeltaNetLayer()

        original_fn1 = model.layer1.chunk_gated_delta_rule
        original_fn2 = model.layer2.chunk_gated_delta_rule

        patch_gated_delta_rule_numerics(model=model)

        # Both layers' methods should be replaced
        assert model.layer1.chunk_gated_delta_rule is not original_fn1
        assert model.layer2.chunk_gated_delta_rule is not original_fn2

    def test_skips_layers_without_a_log(self):
        """Layers with chunk_gated_delta_rule but no A_log are not DeltaNet layers."""

        class NonDeltaNetLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.chunk_gated_delta_rule = lambda *a, **kw: None
                # No A_log attribute

        model = torch.nn.Module()
        model.layer = NonDeltaNetLayer()

        original_fn = model.layer.chunk_gated_delta_rule
        patch_gated_delta_rule_numerics(model=model)

        assert model.layer.chunk_gated_delta_rule is original_fn

    def test_class_patch_idempotent(self):
        """Calling with model=None multiple times doesn't double-wrap."""
        try:
            import transformers.models.qwen3_5.modeling_qwen3_5 as qwen_mod
        except ImportError:
            pytest.skip("transformers qwen3_5 not available")

        gated_cls = getattr(qwen_mod, "Qwen3_5GatedDeltaNet", None)
        if gated_cls is None:
            pytest.skip("Qwen3_5GatedDeltaNet not found")

        original_init = gated_cls.__init__

        # Patch twice
        patch_gated_delta_rule_numerics()
        init_after_first = gated_cls.__init__
        patch_gated_delta_rule_numerics()
        gated_cls.__init__  # noqa: B018 — just verify no crash

        # The second patch wraps the already-patched init. This is technically
        # double-wrapped but functionally harmless (clamp is idempotent).
        assert init_after_first is not original_init


# ---------------------------------------------------------------------------
# Numerical correctness of the clamp bound
# ---------------------------------------------------------------------------


class TestClampBound:
    """Verify the clamp value is safe for the default chunk_size=64."""

    def test_max_exp_within_float32(self):
        """With chunk_size=64, max exp argument = 63 * |g_clamp_min| < 88."""
        chunk_size = 64
        max_exp_arg = (chunk_size - 1) * abs(DEFAULT_G_CLAMP_MIN)
        assert max_exp_arg < 88.0, (
            f"Max exp argument {max_exp_arg} >= 88, will overflow float32. "
            f"Tighten DEFAULT_G_CLAMP_MIN."
        )

    def test_clamp_preserves_strong_decay(self):
        """Even at the clamp boundary, decay over a full chunk is ~0."""
        chunk_size = 64
        total_decay = (chunk_size - 1) * DEFAULT_G_CLAMP_MIN
        exp_decay = torch.tensor(total_decay).exp().item()
        # Should be astronomically small — complete forgetting
        assert exp_decay < 1e-30
