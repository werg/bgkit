"""Lock FA4 causal semantics against `module.is_causal` fallback regressions.

Regression guard for commit 94bec17. Prior to that fix,
``bgkit_flash_attention_4_forward`` had ``is_causal: bool = False`` as the
signature default. HF's attention dispatch never passes ``is_causal`` as a
kwarg; it relies on the module attribute (``module.is_causal``). Because the
default was ``False`` (not ``None``), the line

    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", False)

was dead code, and every decoder full-attention layer silently ran
bidirectional for a week of Step 3 training.

These tests call ``bgkit_flash_attention_4_forward`` the same way HF dispatch
does --- with NO ``is_causal`` kwarg --- and assert the module attribute
actually governs whether FA4 applies causal masking. The assertion is on the
observable output vs an SDPA reference, not on the flag.

How to run
----------
Marked ``@pytest.mark.gpu`` because FA4 only runs on CUDA + an owned SM12x
backend. On hosts with those available (e.g. DGX Spark) it runs under both
``pytest tests/unit/utils/test_fa4_causal_semantics.py`` and
``make test-gpu``. On hosts without, it skips cleanly at collection.
"""

from __future__ import annotations

import types

import pytest
import torch

from bgkit.utils.attention_backend import bgkit_flash_attention_4_forward

# Head dim 256 is chosen because the SM12x FA4 build on DGX Spark disables
# head_dim <= 192; tests must pick a supported size or they will trip a
# kernel error unrelated to the semantics under test.
_N = 72  # sum of the packed sequence lengths below
_LENGTHS = (16, 32, 24)
_H = 4
_D = 256


def _skip_if_fa4_varlen_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("FA4 causal-semantics test requires CUDA")
    pytest.importorskip("flash_attn.cute")
    try:
        from flash_attn.cute.native_sm12x import native_sm12x_owned_backend_available

        if not native_sm12x_owned_backend_available():
            pytest.skip("SM12x owned FA4 backend not available")
    except ImportError:
        pytest.skip("flash_attn.cute.native_sm12x not importable")


def _make_packed_qkv(
    device: str = "cuda", dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    torch.manual_seed(42)
    # Scale down so bf16 softmax stays well-conditioned; the causal / non-causal
    # outputs still diverge by O(1) in L_inf, which is orders of magnitude above
    # the 1e-2 parity tolerance.
    q = torch.randn(_N, _H, _D, dtype=dtype, device=device) * 0.5
    k = torch.randn(_N, _H, _D, dtype=dtype, device=device) * 0.5
    v = torch.randn(_N, _H, _D, dtype=dtype, device=device) * 0.5
    cu = torch.zeros(len(_LENGTHS) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(_LENGTHS, dtype=torch.int32, device=device).cumsum(0)
    return q, k, v, cu, max(_LENGTHS)


def _sdpa_reference_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    """Per-segment SDPA reference. Matches FA4 varlen semantics segment-for-segment."""
    outs = []
    for i in range(cu.numel() - 1):
        s, e = int(cu[i].item()), int(cu[i + 1].item())
        length = e - s
        qb = q[s:e].float().transpose(0, 1)  # (H, L, D)
        kb = k[s:e].float().transpose(0, 1)
        vb = v[s:e].float().transpose(0, 1)
        attn_mask: torch.Tensor | None = None
        if is_causal:
            idx = torch.arange(length, device=q.device)
            attn_mask = idx.unsqueeze(0) <= idx.unsqueeze(1)  # (L, L)
        out = torch.nn.functional.scaled_dot_product_attention(
            qb.unsqueeze(0),
            kb.unsqueeze(0),
            vb.unsqueeze(0),
            attn_mask=attn_mask,
            is_causal=False,
        )
        outs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0).to(q.dtype)


@pytest.mark.gpu
def test_fa4_honors_module_is_causal_true_without_kwarg() -> None:
    """module.is_causal=True must produce causal output when no kwarg is passed.

    This is the exact HF-dispatch call shape. Before commit 94bec17 the FA4
    backend silently ran non-causal here.
    """
    _skip_if_fa4_varlen_unavailable()

    q, k, v, cu, max_sl = _make_packed_qkv()
    module = types.SimpleNamespace(is_causal=True)

    # NOTE: no is_causal kwarg --- mirrors HF's dispatch convention.
    out, attn_weights = bgkit_flash_attention_4_forward(module, q, k, v, cu, max_sl)

    assert attn_weights is None
    ref_causal = _sdpa_reference_packed(q, k, v, cu, is_causal=True)
    ref_noncausal = _sdpa_reference_packed(q, k, v, cu, is_causal=False)

    # Tight tolerance against the expected (causal) reference --- bf16 parity.
    torch.testing.assert_close(out, ref_causal, atol=1e-2, rtol=1e-2)

    # Sanity: causal and non-causal references are materially different, so a
    # regression that silently drops causal masking cannot accidentally pass
    # the tight tolerance above.
    gap = (ref_causal - ref_noncausal).abs().max().item()
    assert gap > 0.5, (
        f"Causal vs non-causal reference gap ({gap:.3e}) is unexpectedly small; "
        "test inputs cannot distinguish the two modes."
    )


@pytest.mark.gpu
def test_fa4_honors_module_is_causal_false_without_kwarg() -> None:
    """module.is_causal=False must produce non-causal output when no kwarg is passed."""
    _skip_if_fa4_varlen_unavailable()

    q, k, v, cu, max_sl = _make_packed_qkv()
    module = types.SimpleNamespace(is_causal=False)

    out, _ = bgkit_flash_attention_4_forward(module, q, k, v, cu, max_sl)
    ref_noncausal = _sdpa_reference_packed(q, k, v, cu, is_causal=False)

    torch.testing.assert_close(out, ref_noncausal, atol=1e-2, rtol=1e-2)


@pytest.mark.gpu
def test_fa4_fallback_when_module_lacks_is_causal_attr() -> None:
    """Module with no is_causal attribute and no kwarg defaults to non-causal.

    Locks the ``getattr(module, "is_causal", False)`` fallback so any future
    rewrite of the resolution logic keeps this edge-case behavior.
    """
    _skip_if_fa4_varlen_unavailable()

    q, k, v, cu, max_sl = _make_packed_qkv()
    module = types.SimpleNamespace()  # no is_causal attribute at all

    out, _ = bgkit_flash_attention_4_forward(module, q, k, v, cu, max_sl)
    ref_noncausal = _sdpa_reference_packed(q, k, v, cu, is_causal=False)

    torch.testing.assert_close(out, ref_noncausal, atol=1e-2, rtol=1e-2)
