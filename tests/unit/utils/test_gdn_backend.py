"""Tests for the GDN backend resolver.

Verifies BGKIT_GDN_BACKEND-driven selection between fla and FlashQLA
without requiring either to be importable on the host. Uses
sys.modules injection to fake the imports.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

# Resolver itself doesn't need torch.
from bgkit.utils import gdn_backend


def _fake_fla_module() -> types.ModuleType:
    """Build a minimal fla.ops.gated_delta_rule package with a sentinel callable."""
    fla = types.ModuleType("fla")
    fla_ops = types.ModuleType("fla.ops")
    fla_ops_gdr = types.ModuleType("fla.ops.gated_delta_rule")

    def _fake_chunk(*args, **kwargs):
        return "fla-result", None

    _fake_chunk.__name__ = "fake_fla_chunk_gated_delta_rule"
    fla_ops_gdr.chunk_gated_delta_rule = _fake_chunk
    fla.ops = fla_ops
    fla_ops.gated_delta_rule = fla_ops_gdr
    return fla, fla_ops, fla_ops_gdr


def _fake_flashqla_module() -> types.ModuleType:
    fq = types.ModuleType("flash_qla")

    def _fake_chunk(*args, **kwargs):
        return "flashqla-result", None

    _fake_chunk.__name__ = "fake_flashqla_chunk_gated_delta_rule"
    fq.chunk_gated_delta_rule = _fake_chunk
    return fq


@pytest.fixture(autouse=True)
def _reset_resolver_and_modules(monkeypatch):
    """Each test starts with a clean resolver cache and clean sys.modules."""
    gdn_backend._reset_for_test()
    # Snapshot + restore relevant module entries.
    saved = {
        name: sys.modules.get(name)
        for name in ("fla", "fla.ops", "fla.ops.gated_delta_rule", "flash_qla")
    }
    monkeypatch.delenv("BGKIT_GDN_BACKEND", raising=False)
    yield
    gdn_backend._reset_for_test()
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


class TestBackendChoice:
    def test_default_is_fla(self):
        assert gdn_backend._backend_choice() == "fla"

    def test_explicit_fla(self, monkeypatch):
        monkeypatch.setenv("BGKIT_GDN_BACKEND", "fla")
        assert gdn_backend._backend_choice() == "fla"

    def test_flashqla(self, monkeypatch):
        monkeypatch.setenv("BGKIT_GDN_BACKEND", "flashqla")
        assert gdn_backend._backend_choice() == "flashqla"

    def test_auto(self, monkeypatch):
        monkeypatch.setenv("BGKIT_GDN_BACKEND", "auto")
        assert gdn_backend._backend_choice() == "auto"

    def test_unknown_is_rejected(self, monkeypatch):
        monkeypatch.setenv("BGKIT_GDN_BACKEND", "tinkerbell")
        with pytest.raises(ValueError, match="BGKIT_GDN_BACKEND"):
            gdn_backend._backend_choice()

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("BGKIT_GDN_BACKEND", "FlashQLA")
        assert gdn_backend._backend_choice() == "flashqla"


class TestResolution:
    def _install_fla(self):
        fla, fla_ops, fla_ops_gdr = _fake_fla_module()
        sys.modules["fla"] = fla
        sys.modules["fla.ops"] = fla_ops
        sys.modules["fla.ops.gated_delta_rule"] = fla_ops_gdr
        return fla_ops_gdr.chunk_gated_delta_rule

    def _install_flashqla(self):
        fq = _fake_flashqla_module()
        sys.modules["flash_qla"] = fq
        return fq.chunk_gated_delta_rule

    def test_default_resolves_to_fla(self):
        sentinel = self._install_fla()
        fn = gdn_backend.get_chunk_gated_delta_rule()
        assert fn is sentinel
        assert gdn_backend.resolved_backend_name() == "fla"

    def test_explicit_fla_resolves_to_fla(self, monkeypatch):
        monkeypatch.setenv("BGKIT_GDN_BACKEND", "fla")
        sentinel = self._install_fla()
        fn = gdn_backend.get_chunk_gated_delta_rule()
        assert fn is sentinel
        assert gdn_backend.resolved_backend_name() == "fla"

    def test_explicit_flashqla(self, monkeypatch):
        monkeypatch.setenv("BGKIT_GDN_BACKEND", "flashqla")
        sentinel = self._install_flashqla()
        fn = gdn_backend.get_chunk_gated_delta_rule()
        assert fn is sentinel
        assert gdn_backend.resolved_backend_name() == "flashqla"

    def test_explicit_flashqla_missing_raises(self, monkeypatch):
        monkeypatch.setenv("BGKIT_GDN_BACKEND", "flashqla")
        monkeypatch.setattr(
            gdn_backend,
            "_import_flashqla",
            lambda: (_ for _ in ()).throw(ImportError("missing")),
        )
        with pytest.raises(RuntimeError, match="FlashQLA could not be imported"):
            gdn_backend.get_chunk_gated_delta_rule()

    def test_auto_prefers_flashqla(self, monkeypatch):
        monkeypatch.setenv("BGKIT_GDN_BACKEND", "auto")
        self._install_fla()
        fq_sentinel = self._install_flashqla()
        fn = gdn_backend.get_chunk_gated_delta_rule()
        assert fn is fq_sentinel
        assert gdn_backend.resolved_backend_name() == "flashqla"

    def test_auto_falls_back_to_fla(self, monkeypatch):
        monkeypatch.setenv("BGKIT_GDN_BACKEND", "auto")
        monkeypatch.setattr(
            gdn_backend,
            "_import_flashqla",
            lambda: (_ for _ in ()).throw(ImportError("missing")),
        )
        fla_sentinel = self._install_fla()
        fn = gdn_backend.get_chunk_gated_delta_rule()
        assert fn is fla_sentinel
        assert gdn_backend.resolved_backend_name() == "fla"

    def test_resolution_is_cached(self):
        sentinel = self._install_fla()
        fn1 = gdn_backend.get_chunk_gated_delta_rule()
        # Replace the underlying module so a re-resolution would pick a different fn.
        new_sentinel = lambda *a, **kw: "different"  # noqa: E731
        sys.modules["fla.ops.gated_delta_rule"].chunk_gated_delta_rule = new_sentinel
        fn2 = gdn_backend.get_chunk_gated_delta_rule()
        assert fn2 is fn1 is sentinel  # cached, did not pick up the swap

    def test_resolved_backend_name_unresolved(self):
        assert gdn_backend.resolved_backend_name() is None

    def test_flashqla_import_error_wrapped(self, monkeypatch):
        """A non-ImportError raised during flash_qla import (e.g. FlashQLA's
        own sm90-only ValueError) is converted to ImportError so callers
        handle it uniformly."""
        monkeypatch.setenv("BGKIT_GDN_BACKEND", "flashqla")

        # Build a module that raises ValueError on attribute access of
        # chunk_gated_delta_rule, mimicking an architecture gate.
        class _RaisingModule(types.ModuleType):
            def __getattr__(self, name):
                if name == "chunk_gated_delta_rule":
                    raise ValueError("FlashQLA now support sm90 only.")
                raise AttributeError(name)

        sys.modules["flash_qla"] = _RaisingModule("flash_qla")
        with pytest.raises(RuntimeError, match="FlashQLA"):
            gdn_backend.get_chunk_gated_delta_rule()

    def test_describe_backend_environment_does_not_import_flashqla(self, monkeypatch):
        monkeypatch.setenv("BGKIT_GDN_BACKEND", "flashqla")
        sys.modules.pop("flash_qla", None)

        info = gdn_backend.describe_backend_environment()

        assert info["requested_backend"] == "flashqla"
        assert info["resolved_backend"] is None
        assert "flash_qla" not in sys.modules
        assert "modules" in info
        assert "cuda" in info
        assert "tilelang" in info


class TestDeltanetPatchIntegration:
    """Verify deltanet_patch uses FLA by default and FlashQLA when explicit.

    Uses a real object (not MagicMock) for the layer so the existing
    class-level pre-existing bug (MagicMock auto-attribbing _unpatch_chunk_gdr
    to a truthy mock and shadowing the sentinel) doesn't confuse this test.
    """

    def _make_layer(self, gdr_fn):
        torch = pytest.importorskip("torch")  # noqa: F841

        class _Layer:
            pass

        layer = _Layer()
        layer.chunk_gated_delta_rule = gdr_fn
        layer.A_log = True
        # Real forward, returns whatever HF would return; we don't exercise it here.
        layer.forward = lambda *a, **kw: ("orig-forward", None)
        return layer

    def test_patch_default_preserves_fla(self, monkeypatch):
        """With BGKIT_GDN_BACKEND unset, the HF-wired FLA callable is kept."""
        from bgkit.utils import deltanet_patch

        monkeypatch.delenv("BGKIT_GDN_BACKEND", raising=False)

        hf_calls = []

        def hf_default(*args, **kwargs):
            hf_calls.append((args, kwargs))
            return ("hf-default", None)

        layer = self._make_layer(hf_default)
        deltanet_patch.patch_deltanet_layer(layer)

        layer.chunk_gated_delta_rule("q", "k", "v", g=__import__("torch").zeros(3), beta="beta")
        assert len(hf_calls) == 1

    def test_patch_explicit_fla_preserves_layer_fn(self, monkeypatch):
        """BGKIT_GDN_BACKEND=fla keeps the HF-wired FLA callable as escape hatch."""
        from bgkit.utils import deltanet_patch

        monkeypatch.setenv("BGKIT_GDN_BACKEND", "fla")

        calls = []

        def hf_default(*args, **kwargs):
            calls.append(("hf-default", args, kwargs))
            return ("hf-default", None)

        layer = self._make_layer(hf_default)
        deltanet_patch.patch_deltanet_layer(layer)

        layer.chunk_gated_delta_rule("q", "k", "v", g=__import__("torch").zeros(3), beta="beta")
        assert len(calls) == 1
        assert calls[0][0] == "hf-default"

    def test_patch_with_flashqla_swaps_backend(self, monkeypatch):
        """BGKIT_GDN_BACKEND=flashqla swaps in the resolver's pick."""
        from bgkit.utils import deltanet_patch

        # Install a fake flash_qla module with a tracked sentinel.
        fq = _fake_flashqla_module()
        flashqla_calls = []

        def flashqla_sentinel(*args, **kwargs):
            flashqla_calls.append((args, kwargs))
            return ("flashqla-result", None)

        fq.chunk_gated_delta_rule = flashqla_sentinel
        sys.modules["flash_qla"] = fq

        monkeypatch.setenv("BGKIT_GDN_BACKEND", "flashqla")

        hf_calls = []

        def hf_default(*args, **kwargs):
            hf_calls.append((args, kwargs))
            return ("hf-default", None)

        layer = self._make_layer(hf_default)
        deltanet_patch.patch_deltanet_layer(layer)
        layer.chunk_gated_delta_rule("q", "k", "v", g=__import__("torch").zeros(3), beta="beta")

        assert len(flashqla_calls) == 1
        assert len(hf_calls) == 0  # HF default bypassed
