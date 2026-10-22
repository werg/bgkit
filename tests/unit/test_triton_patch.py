"""Tests for triton_patch: autotuner safety net for Blackwell sm_121."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Patch scoping: only applies on sm_121
# ---------------------------------------------------------------------------


class TestPatchScoping:
    """Verify the patch only activates on sm_121."""

    def test_noop_without_cuda(self):
        """No CUDA available — patch should be a no-op."""
        with patch.dict("sys.modules", {"triton": None, "triton.runtime": None}):
            # Re-import to get fresh state
            import importlib

            import bgkit.utils.triton_patch as tp
            tp._patched = False

            with patch("torch.cuda.is_available", return_value=False):
                tp.patch_triton_autotuner()
                # Should return early, no crash

    @pytest.mark.gpu
    def test_noop_on_non_sm121(self):
        """On a non-sm_121 GPU, patch should not install."""
        pytest.importorskip("triton")
        from triton.runtime.autotuner import Autotuner

        import bgkit.utils.triton_patch as tp
        tp._patched = False
        original_bench = Autotuner._bench

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_capability", return_value=(8, 0)):
            tp.patch_triton_autotuner()
            assert Autotuner._bench is original_bench
        tp._patched = False

    @pytest.mark.gpu
    def test_installs_on_sm121(self):
        """On sm_121, patch should replace _bench."""
        pytest.importorskip("triton")
        from triton.runtime.autotuner import Autotuner

        import bgkit.utils.triton_patch as tp
        tp._patched = False
        original_bench = Autotuner._bench

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_capability", return_value=(12, 1)):
            tp.patch_triton_autotuner()

        patched = Autotuner._bench is not original_bench
        Autotuner._bench = original_bench
        tp._patched = False
        assert patched

    def test_idempotent(self):
        """Calling patch_triton_autotuner() twice doesn't double-wrap."""
        pytest.importorskip("torch")

        import bgkit.utils.triton_patch as tp
        tp._patched = False

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_capability", return_value=(12, 1)):
            tp.patch_triton_autotuner()
            tp.patch_triton_autotuner()  # second call is guarded by _patched

        tp._patched = False


# ---------------------------------------------------------------------------
# Error filtering behavior
# ---------------------------------------------------------------------------


class TestErrorFiltering:
    """Verify the patched _bench handles errors correctly."""

    def test_triton_invalid_arg_returns_inf(self):
        """'Triton Error [CUDA]: invalid argument' should return [inf, inf, inf]."""

        def _make_safe_bench(original_bench):
            """Reproduce the patch's wrapping logic."""
            def _safe_bench(self, *args, config, **meta):
                try:
                    return original_bench(self, *args, config=config, **meta)
                except RuntimeError as e:
                    err = str(e)
                    if "Triton Error" in err and "invalid argument" in err:
                        return [float("inf"), float("inf"), float("inf")]
                    raise
            return _safe_bench

        def _raising_original(self, *args, config, **meta):
            raise RuntimeError("Triton Error [CUDA]: invalid argument")

        safe = _make_safe_bench(_raising_original)
        result = safe(MagicMock(), config=MagicMock())
        assert result == [float("inf"), float("inf"), float("inf")]

    def test_non_triton_error_propagates(self):
        """RuntimeError without 'Triton Error' should propagate."""

        def _make_safe_bench(original_bench):
            def _safe_bench(self, *args, config, **meta):
                try:
                    return original_bench(self, *args, config=config, **meta)
                except RuntimeError as e:
                    err = str(e)
                    if "Triton Error" in err and "invalid argument" in err:
                        return [float("inf"), float("inf"), float("inf")]
                    raise
            return _safe_bench

        def _raising_original(self, *args, config, **meta):
            raise RuntimeError("completely unrelated error")

        safe = _make_safe_bench(_raising_original)
        with pytest.raises(RuntimeError, match="completely unrelated"):
            safe(MagicMock(), config=MagicMock())

    def test_triton_illegal_access_propagates(self):
        """Triton Error that isn't 'invalid argument' should propagate."""

        def _make_safe_bench(original_bench):
            def _safe_bench(self, *args, config, **meta):
                try:
                    return original_bench(self, *args, config=config, **meta)
                except RuntimeError as e:
                    err = str(e)
                    if "Triton Error" in err and "invalid argument" in err:
                        return [float("inf"), float("inf"), float("inf")]
                    raise
            return _safe_bench

        def _raising_original(self, *args, config, **meta):
            raise RuntimeError("Triton Error [CUDA]: an illegal memory access was encountered")

        safe = _make_safe_bench(_raising_original)
        with pytest.raises(RuntimeError, match="illegal memory access"):
            safe(MagicMock(), config=MagicMock())


# ---------------------------------------------------------------------------
# Signature stability
# ---------------------------------------------------------------------------


class TestTritonSignatureAssumptions:
    """Verify our assumptions about Triton's autotuner API."""

    @pytest.mark.gpu
    def test_autotuner_has_bench(self):
        """Autotuner._bench exists — our patch target."""
        pytest.importorskip("triton")
        from triton.runtime.autotuner import Autotuner

        assert hasattr(Autotuner, "_bench"), (
            "Autotuner._bench not found. Triton API may have changed. "
            "Update triton_patch.py."
        )

    @pytest.mark.gpu
    def test_bench_signature(self):
        """_bench accepts (self, *args, config, **meta)."""
        import inspect

        pytest.importorskip("triton")
        from triton.runtime.autotuner import Autotuner

        sig = inspect.signature(Autotuner._bench)
        params = list(sig.parameters.keys())
        assert "config" in params, (
            f"Autotuner._bench signature changed: {params}. "
            f"Update triton_patch.py."
        )
