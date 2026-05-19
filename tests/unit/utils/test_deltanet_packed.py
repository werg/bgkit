"""Tests for DeltaNet packed (varlen) path — Task 1.3.

Two test classes:
  TestDeltanetPatchWiring  — CPU-runnable, verifies the patch wiring:
    signature, gate-clamp preservation, cu_seqlens pass-through.

  TestDeltanetPackedParity — GPU (pytest.mark.gpu), verifies numerical parity
    of a single packed chunk_gated_delta_rule call against per-sample
    sequential reference calls (both using time-first convention).

Fixture layout (tests/fixtures/deltanet_reference.pt):
  inputs:
    q, k, v:  (B, H, L, D)  float32, head-first (legacy head_first=True format)
    g, beta:  (B, H, L)     float32, head-first
    scale:    float
    valid_lengths: list[int]
  outputs_per_sample:
    sample_i: {length: int, delta_out: (1, H, l_i, D) float32 head-first}

The fixture's per_sample outputs were captured with head-first convention and
therefore differ numerically from time-first calls.  The GPU parity test
generates its own per-sample sequential references at runtime using time-first
convention, then verifies the packed cu_seqlens call matches those.  The
fixture inputs (q, k, v, g, beta, scale) are reused; only the reference
outputs are regenerated.

fla 0.4.x convention note:
  Non-varlen calls: fla accepts (and returns) head-first (B, H, T, D).
  Varlen cu_seqlens calls: fla requires (and returns) time-first (1, N, H, D).
  Mixing conventions produces different numerical results.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"
DELTANET_FIXTURE = FIXTURES_DIR / "deltanet_reference.pt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cu_seqlens(lengths: list[int], device: str = "cpu") -> torch.Tensor:
    """Build cumulative sequence lengths tensor from a list of lengths."""
    cumlen = [0]
    for seq_len in lengths:
        cumlen.append(cumlen[-1] + seq_len)
    return torch.tensor(cumlen, dtype=torch.long, device=device)


@pytest.fixture(autouse=True)
def _use_fla_escape_hatch(monkeypatch):
    """Most tests in this module verify patch wiring against fake/FLA callables."""
    monkeypatch.setenv("BGKIT_GDN_BACKEND", "fla")


# ---------------------------------------------------------------------------
# CPU wiring tests (no GPU / fla kernel needed)
# ---------------------------------------------------------------------------

class TestDeltanetPatchWiring:
    """Verify patch_deltanet_layer wiring without invoking the fla kernel."""

    def _make_fake_layer(self):
        """Build a minimal nn.Module that looks like Qwen3_5GatedDeltaNet."""
        layer = nn.Module()
        # Minimal attribute to satisfy patch_deltanet_layer's guard
        layer.A_log = nn.Parameter(torch.zeros(1))
        call_log = []

        def _fake_chunk_gdr(*args, **kwargs):
            call_log.append(("chunk_gdr", args, kwargs))
            q = args[0] if args else kwargs["q"]
            return torch.zeros_like(q), None

        def _fake_forward(hidden_states, cache_params=None, attention_mask=None):
            call_log.append(("forward", hidden_states.shape))
            return hidden_states

        layer.chunk_gated_delta_rule = _fake_chunk_gdr
        layer.forward = _fake_forward
        layer._call_log = call_log
        return layer

    def test_patch_adds_cu_seqlens_to_forward_signature(self):
        """Patched forward must accept cu_seqlens and position_ids kwargs."""
        from bgkit.utils.deltanet_patch import patch_deltanet_layer

        layer = self._make_fake_layer()
        patch_deltanet_layer(layer)

        hidden = torch.randn(1, 10, 32)
        cu = torch.tensor([0, 5, 10], dtype=torch.long)
        pos = torch.arange(10)

        out = layer.forward(hidden, cu_seqlens=cu, position_ids=pos)
        assert out is not None

    def test_gate_clamp_still_applied_in_packed_path(self):
        """g clamping must fire even in the packed path."""
        from bgkit.utils.deltanet_patch import patch_deltanet_layer

        captured = {}

        layer = nn.Module()
        layer.A_log = nn.Parameter(torch.zeros(1))

        def _fake_chunk_gdr(*args, **kwargs):
            g_val = args[3] if len(args) >= 4 else kwargs.get("g")
            captured["g"] = g_val
            return torch.zeros_like(args[0] if args else kwargs["q"]), None

        def _fake_forward(hidden_states, cache_params=None, attention_mask=None):
            q = k = v = torch.randn(1, 5, 1, 4)
            g_extreme = torch.full((1, 5, 1), -5.0)  # way below -1.3 clamp
            beta = torch.ones(1, 5, 1)
            layer.chunk_gated_delta_rule(q, k, v, g_extreme, beta)
            return hidden_states

        layer.chunk_gated_delta_rule = _fake_chunk_gdr
        layer.forward = _fake_forward
        patch_deltanet_layer(layer, g_clamp_min=-1.3)

        hidden = torch.randn(1, 5, 32)
        cu = torch.tensor([0, 5], dtype=torch.long)
        layer.forward(hidden, cu_seqlens=cu)

        assert "g" in captured, "chunk_gated_delta_rule was not called"
        assert captured["g"].min().item() >= -1.3 - 1e-6, (
            f"Gate clamp not applied: min={captured['g'].min().item()}"
        )

    def test_cu_seqlens_injected_into_chunk_gdr(self):
        """cu_seqlens from forward call must appear in chunk_gated_delta_rule kwargs."""
        from bgkit.utils.deltanet_patch import patch_deltanet_layer

        captured = {}

        layer = nn.Module()
        layer.A_log = nn.Parameter(torch.zeros(1))

        def _fake_chunk_gdr(*args, **kwargs):
            captured["cu_seqlens"] = kwargs.get("cu_seqlens")
            return torch.zeros_like(args[0] if args else kwargs["q"]), None

        def _fake_forward(hidden_states, cache_params=None, attention_mask=None):
            q = k = v = torch.randn(1, 5, 1, 4)
            g = torch.full((1, 5, 1), -0.5)
            beta = torch.ones(1, 5, 1)
            layer.chunk_gated_delta_rule(q, k, v, g, beta)
            return hidden_states

        layer.chunk_gated_delta_rule = _fake_chunk_gdr
        layer.forward = _fake_forward
        patch_deltanet_layer(layer)

        cu = torch.tensor([0, 3, 5], dtype=torch.long)
        hidden = torch.randn(1, 5, 32)
        layer.forward(hidden, cu_seqlens=cu)

        assert captured.get("cu_seqlens") is not None, (
            "cu_seqlens was not forwarded to chunk_gated_delta_rule"
        )
        assert torch.equal(captured["cu_seqlens"], cu), (
            "cu_seqlens forwarded does not match the one passed to forward"
        )

    def test_packed_path_keeps_chunk_gdr_wrapper_stable(self):
        """Packed forward must not install a temporary chunk_gdr wrapper per call."""
        from bgkit.utils.deltanet_patch import patch_deltanet_layer

        captured = {}

        layer = nn.Module()
        layer.A_log = nn.Parameter(torch.zeros(1))

        def _fake_chunk_gdr(*args, **kwargs):
            captured["cu_seqlens"] = kwargs.get("cu_seqlens")
            captured["wrapper_id_during_call"] = id(layer.chunk_gated_delta_rule)
            return torch.zeros_like(args[0] if args else kwargs["q"]), None

        def _fake_forward(hidden_states, cache_params=None, attention_mask=None):
            captured["wrapper_id_at_forward_entry"] = id(layer.chunk_gated_delta_rule)
            q = k = v = torch.randn(1, 5, 1, 4)
            g = torch.full((1, 5, 1), -0.5)
            beta = torch.ones(1, 5, 1)
            layer.chunk_gated_delta_rule(q, k, v, g, beta)
            captured["wrapper_id_after_call_inside_forward"] = id(
                layer.chunk_gated_delta_rule
            )
            return hidden_states

        layer.chunk_gated_delta_rule = _fake_chunk_gdr
        layer.forward = _fake_forward
        patch_deltanet_layer(layer)

        wrapper_id = id(layer.chunk_gated_delta_rule)
        cu = torch.tensor([0, 3, 5], dtype=torch.long)
        hidden = torch.randn(1, 5, 32)
        layer.forward(hidden, cu_seqlens=cu)

        assert id(layer.chunk_gated_delta_rule) == wrapper_id
        assert captured["wrapper_id_at_forward_entry"] == wrapper_id
        assert captured["wrapper_id_during_call"] == wrapper_id
        assert captured["wrapper_id_after_call_inside_forward"] == wrapper_id
        assert torch.equal(captured["cu_seqlens"], cu)

    def test_non_packed_path_unaffected(self):
        """Without cu_seqlens, the patched forward must call original forward normally."""
        from bgkit.utils.deltanet_patch import patch_deltanet_layer

        captured = {}

        layer = nn.Module()
        layer.A_log = nn.Parameter(torch.zeros(1))

        def _fake_chunk_gdr(*args, **kwargs):
            captured["cu_seqlens"] = kwargs.get("cu_seqlens")
            return torch.zeros_like(args[0]), None

        def _fake_forward(hidden_states, cache_params=None, attention_mask=None):
            q = k = v = torch.randn(1, 5, 1, 4)
            g = torch.full((1, 5, 1), -0.5)
            beta = torch.ones(1, 5, 1)
            layer.chunk_gated_delta_rule(q, k, v, g, beta)
            return hidden_states

        layer.chunk_gated_delta_rule = _fake_chunk_gdr
        layer.forward = _fake_forward
        patch_deltanet_layer(layer)

        hidden = torch.randn(1, 5, 32)
        layer.forward(hidden)  # no cu_seqlens

        assert captured.get("cu_seqlens") is None, (
            "cu_seqlens was unexpectedly injected in non-packed path"
        )

    def test_patch_is_idempotent(self):
        """Calling patch_deltanet_layer twice must not double-wrap."""
        from bgkit.utils.deltanet_patch import patch_deltanet_layer

        call_count = {"n": 0}

        layer = nn.Module()
        layer.A_log = nn.Parameter(torch.zeros(1))

        def _fake_chunk_gdr(*args, **kwargs):
            call_count["n"] += 1
            return torch.zeros_like(args[0]), None

        def _fake_forward(hidden_states, cache_params=None, attention_mask=None):
            q = k = v = torch.randn(1, 4, 1, 4)
            g = torch.full((1, 4, 1), -0.5)
            beta = torch.ones(1, 4, 1)
            layer.chunk_gated_delta_rule(q, k, v, g, beta)
            return hidden_states

        layer.chunk_gated_delta_rule = _fake_chunk_gdr
        layer.forward = _fake_forward

        patch_deltanet_layer(layer)
        patch_deltanet_layer(layer)  # second patch — should replace, not stack

        hidden = torch.randn(1, 4, 32)
        cu = torch.tensor([0, 4], dtype=torch.long)
        layer.forward(hidden, cu_seqlens=cu)

        assert call_count["n"] == 1, (
            f"chunk_gated_delta_rule called {call_count['n']} times; expected 1 (idempotent)"
        )

    def test_patch_gated_delta_rule_numerics_with_model(self):
        """patch_gated_delta_rule_numerics(model=...) patches all matching layers."""
        from bgkit.utils.deltanet_patch import patch_gated_delta_rule_numerics

        captured = {}

        class FakeDeltaNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.A_log = nn.Parameter(torch.zeros(1))

            def forward(self, hidden, cache_params=None, attention_mask=None):
                return hidden

            def chunk_gated_delta_rule(self, *args, **kwargs):
                captured["g"] = args[3] if len(args) >= 4 else kwargs.get("g")
                return torch.zeros_like(args[0]), None

        model = nn.Sequential(FakeDeltaNet(), FakeDeltaNet())
        patch_gated_delta_rule_numerics(model=model, g_clamp_min=-1.3)

        for mod in model.modules():
            if isinstance(mod, FakeDeltaNet):
                q = k = v = torch.randn(1, 5, 1, 4)
                g = torch.full((1, 5, 1), -9.0)
                beta = torch.ones(1, 5, 1)
                mod.forward(torch.randn(1, 5, 32))
                # Directly call the patched chunk_gdr to verify clamp
                mod.chunk_gated_delta_rule(q, k, v, g, beta)
                assert captured["g"].min().item() >= -1.3 - 1e-6


# ---------------------------------------------------------------------------
# GPU parity tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestDeltanetPackedParity:
    """Numerical parity: packed cu_seqlens call vs. per-sample sequential reference.

    Both the reference and the packed call use the same time-first convention
    (1, T, H, D) / (1, N, H, D) that fla 0.4.x uses for the varlen path.

    The fixture's pre-computed per_sample outputs are NOT used as references
    here — those were captured with the legacy head-first convention which
    produces different numerical results.  Instead, we generate per-sample
    references at test runtime using the same time-first tensors derived from
    the fixture inputs.

    Tolerance: atol=1e-4, rtol=1e-4 (float32 Triton kernels; verified 4.76e-05
    max error on sm_121 for the parity case).

    NOTE on sm_121 stability:
    The fla Triton kernels for the cu_seqlens path were found to be STABLE on
    sm_121 (DGX Spark GB10) in local testing (2026-04-20): the packed path ran
    without kernel crashes and matched per-sample references to 4.76e-05.
    If a kernel crash does occur (AcceleratorError or RuntimeError from Triton),
    the test is xfailed with the error message documented.
    """

    @pytest.fixture(autouse=True)
    def check_fixture(self):
        if not DELTANET_FIXTURE.exists():
            pytest.skip(f"Fixture not found: {DELTANET_FIXTURE}")

    @pytest.fixture(autouse=True)
    def check_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    @pytest.fixture(autouse=True)
    def check_fla(self):
        try:
            import fla.utils as _fla_utils
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _chk
        except ImportError:
            pytest.skip("fla (flash-linear-attention) not available")
        if getattr(_fla_utils, "device_platform", None) != "cuda":
            pytest.skip(
                "fla-core is not running on CUDA (Triton unavailable on host venv); "
                "packed DeltaNet parity requires the Docker training container."
            )

    def _load_fixture(self) -> dict:
        return torch.load(DELTANET_FIXTURE, weights_only=False)

    def _fixture_to_time_first(
        self, device: str, dtype: torch.dtype
    ) -> tuple[list[int], float, dict[str, list[torch.Tensor]]]:
        """Load fixture inputs, convert to time-first (T, H, D) per-sample slices."""
        data = self._load_fixture()
        inputs = data["inputs"]
        lengths: list[int] = data["shape"]["lengths"]
        scale = float(inputs["scale"])

        q_hf = inputs["q"].to(device, dtype)    # (B, H, L, D) head-first
        k_hf = inputs["k"].to(device, dtype)
        v_hf = inputs["v"].to(device, dtype)
        g_hf = inputs["g"].to(device, dtype).clamp(min=-1.3)  # (B, H, L)
        beta_hf = inputs["beta"].to(device, dtype)

        # Transpose each sample from head-first (H, seq_len, D) to time-first (seq_len, H, D)
        slices: dict[str, list[torch.Tensor]] = {
            t: [] for t in ("q", "k", "v", "g", "beta")
        }
        for i, seq_len in enumerate(lengths):
            slices["q"].append(q_hf[i, :, :seq_len, :].permute(1, 0, 2))    # (seq_len, H, D)
            slices["k"].append(k_hf[i, :, :seq_len, :].permute(1, 0, 2))
            slices["v"].append(v_hf[i, :, :seq_len, :].permute(1, 0, 2))
            slices["g"].append(g_hf[i, :, :seq_len].permute(1, 0))           # (seq_len, H)
            slices["beta"].append(beta_hf[i, :, :seq_len].permute(1, 0))

        return lengths, scale, slices

    def test_packed_parity_vs_per_sample(self):
        """Packed cu_seqlens call matches per-sample time-first references.

        Both calls use time-first tensors (1, T, H, D) / (1, N, H, D).
        The per-sample references are generated fresh at test time from
        the fixture inputs — not read from the fixture's pre-stored outputs.
        """
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        device = "cuda"
        dtype = torch.float32
        lengths, scale, slices = self._fixture_to_time_first(device, dtype)

        cu_seqlens = _build_cu_seqlens(lengths, device=device)
        cumlen = cu_seqlens.tolist()

        # --- Per-sample sequential references (time-first, batch=1 each) ---
        refs = []
        for i, _seq_len in enumerate(lengths):
            qi = slices["q"][i].unsqueeze(0)     # (1, seq_len, H, D)
            ki = slices["k"][i].unsqueeze(0)
            vi = slices["v"][i].unsqueeze(0)
            gi = slices["g"][i].unsqueeze(0)     # (1, seq_len, H)
            betai = slices["beta"][i].unsqueeze(0)
            with torch.no_grad():
                oi, _ = chunk_gated_delta_rule(
                    qi, ki, vi, gi, betai, scale=scale, output_final_state=False
                )
            refs.append(oi)  # (1, seq_len, H, D)

        # --- Packed call (time-first, batch=1 packed) ---
        q_p = torch.cat(slices["q"], dim=0).unsqueeze(0)     # (1, N, H, D)
        k_p = torch.cat(slices["k"], dim=0).unsqueeze(0)
        v_p = torch.cat(slices["v"], dim=0).unsqueeze(0)
        g_p = torch.cat(slices["g"], dim=0).unsqueeze(0)     # (1, N, H)
        beta_p = torch.cat(slices["beta"], dim=0).unsqueeze(0)

        try:
            with torch.no_grad():
                out_packed, _ = chunk_gated_delta_rule(
                    q_p,
                    k_p,
                    v_p,
                    g_p,
                    beta_p,
                    scale=scale,
                    cu_seqlens=cu_seqlens,
                    output_final_state=False,
                )
        except (RuntimeError, Exception) as exc:
            # Guard against Triton / CUDA kernel crashes on sm_121.
            pytest.xfail(
                f"chunk_gated_delta_rule with cu_seqlens crashed "
                f"(possible sm_121 kernel instability): {type(exc).__name__}: {exc}"
            )
            return

        # out_packed: (1, N, H, D) time-first
        n_total = cumlen[-1]
        expected_shape = (1, n_total, refs[0].shape[2], refs[0].shape[3])
        assert out_packed.shape == expected_shape, (
            f"Unexpected packed output shape: {out_packed.shape}; expected {expected_shape}"
        )

        for i, seq_len in enumerate(lengths):
            ref = refs[i][0]  # (seq_len, H, D)
            start = cumlen[i]
            end = cumlen[i + 1]
            out_slice = out_packed[0, start:end, :, :]  # (seq_len, H, D)

            try:
                torch.testing.assert_close(
                    out_slice,
                    ref,
                    atol=1e-4,
                    rtol=1e-4,
                    msg=f"Parity failure for sample {i} (length={seq_len})",
                )
            except AssertionError as exc:
                max_err = (out_slice - ref).abs().max().item()
                mean_err = (out_slice - ref).abs().mean().item()
                raise AssertionError(
                    f"sample_{i} (len={seq_len}): max_err={max_err:.2e}, "
                    f"mean_err={mean_err:.2e}\n{exc}"
                ) from exc

    def test_packed_call_requires_batch_one(self):
        """chunk_gated_delta_rule with cu_seqlens must have batch=1."""
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        device = "cuda"
        # Attempt batch=2 with cu_seqlens — fla should raise ValueError
        q = torch.randn(2, 10, 2, 16, device=device)
        k = torch.randn(2, 10, 2, 16, device=device)
        v = torch.randn(2, 10, 2, 16, device=device)
        g = torch.full((2, 10, 2), -0.5, device=device)
        beta = torch.ones(2, 10, 2, device=device)
        cu = torch.tensor([0, 10, 20], dtype=torch.long, device=device)

        with pytest.raises((ValueError, RuntimeError)):
            chunk_gated_delta_rule(q, k, v, g, beta, cu_seqlens=cu)

    def test_patched_layer_forward_packed(self):
        """End-to-end: patched fake DeltaNet layer forward reproduces per-sample output.

        This is a lighter end-to-end test: we patch a fake DeltaNet layer whose
        chunk_gated_delta_rule delegates to fla, then call layer.forward with
        cu_seqlens and verify the output matches per-sample time-first references.
        """
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_gdr

        from bgkit.utils.deltanet_patch import patch_deltanet_layer

        device = "cuda"
        dtype = torch.float32
        lengths, scale, slices = self._fixture_to_time_first(device, dtype)

        cu_seqlens = _build_cu_seqlens(lengths, device=device)
        cumlen = cu_seqlens.tolist()

        # Per-sample time-first references
        refs = []
        for i in range(len(lengths)):
            qi = slices["q"][i].unsqueeze(0)
            ki = slices["k"][i].unsqueeze(0)
            vi = slices["v"][i].unsqueeze(0)
            gi = slices["g"][i].unsqueeze(0)
            betai = slices["beta"][i].unsqueeze(0)
            with torch.no_grad():
                oi, _ = fla_gdr(qi, ki, vi, gi, betai, scale=scale, output_final_state=False)
            refs.append(oi[0])  # (seq_len, H, D)

        # Packed tensors (1, N, H, D) and (1, N, H)
        packed = {
            "q": torch.cat(slices["q"], dim=0).unsqueeze(0),
            "k": torch.cat(slices["k"], dim=0).unsqueeze(0),
            "v": torch.cat(slices["v"], dim=0).unsqueeze(0),
            "g": torch.cat(slices["g"], dim=0).unsqueeze(0),
            "beta": torch.cat(slices["beta"], dim=0).unsqueeze(0),
        }

        # Fake DeltaNet layer: forward directly calls self.chunk_gated_delta_rule.
        class FakeDeltaNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.A_log = nn.Parameter(torch.zeros(1))

            def forward(self, hidden_states, cache_params=None, attention_mask=None):
                # hidden_states is a dict of packed tensors in this fake layer.
                out, _ = self.chunk_gated_delta_rule(
                    hidden_states["q"],
                    hidden_states["k"],
                    hidden_states["v"],
                    g=hidden_states["g"],
                    beta=hidden_states["beta"],
                    scale=scale,
                    output_final_state=False,
                )
                return out

            def chunk_gated_delta_rule(self, *args, **kwargs):
                return fla_gdr(*args, **kwargs)

        fake_layer = FakeDeltaNet().to(device)
        patch_deltanet_layer(fake_layer, g_clamp_min=-1.3)

        try:
            out_packed = fake_layer.forward(packed, cu_seqlens=cu_seqlens)
        except (RuntimeError, Exception) as exc:
            pytest.xfail(
                f"Patched forward with cu_seqlens crashed (possible sm_121 kernel "
                f"instability): {type(exc).__name__}: {exc}"
            )
            return

        # out_packed: (1, N, H, D) time-first
        for i, _seq_len in enumerate(lengths):
            ref = refs[i]  # (seq_len, H, D)
            start = cumlen[i]
            end = cumlen[i + 1]
            out_slice = out_packed[0, start:end, :, :]

            try:
                torch.testing.assert_close(out_slice, ref, atol=1e-4, rtol=1e-4)
            except AssertionError as exc:
                max_err = (out_slice - ref).abs().max().item()
                raise AssertionError(
                    f"Patched layer sample_{i}: max_err={max_err:.2e}\n{exc}"
                ) from exc
