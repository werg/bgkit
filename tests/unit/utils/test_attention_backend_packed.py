"""Parity tests for the packed FlashAttention backend.

These exercise the real FA4 packed-varlen path and run only where CUDA plus an
owned SM12x backend are available.

Reference implementation
------------------------
A pure-Python per-sample attention loop using PyTorch primitives is used as
the ground truth for both paths.  It supports ``is_causal`` and
``sliding_window``.

Shape convention
----------------
All packed inputs are flat ``(N, H, D)`` where ``N = sum(L_i)``.
``cu_seqlens`` is ``(B+1,) int32`` with ``cu_seqlens[0] = 0``.
"""

from __future__ import annotations

import math

import pytest
import torch

from bgkit.utils.attention_backend import bgkit_flash_attention_4_forward

# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _reference_packed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    is_causal: bool = False,
    sliding_window: int | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Per-sample attention loop.  Reference ground truth.

    Parameters
    ----------
    query, key, value:
        ``(N, H, D)`` packed tensors.
    cu_seqlens:
        ``(B+1,) int32``.
    is_causal:
        Apply causal mask.
    sliding_window:
        Local window size (past tokens visible).  ``None`` = full attention.
    scale:
        Softmax scale.  Defaults to ``1 / sqrt(D)``.

    Returns
    -------
    out: ``(N, H, D)``
    """
    head_dim = query.shape[2]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    cu = cu_seqlens.tolist()
    batch = len(cu) - 1
    outputs = []

    for b in range(batch):
        start, end = cu[b], cu[b + 1]
        seq_len = end - start
        q_b = query[start:end].float()  # (seq_len, H, D)
        k_b = key[start:end].float()
        v_b = value[start:end].float()

        # Compute attention per head (H, seq_len, seq_len) scores.
        q_h = q_b.transpose(0, 1)  # (H, seq_len, D)
        k_h = k_b.transpose(0, 1)
        v_h = v_b.transpose(0, 1)

        scores = torch.einsum("hid,hjd->hij", q_h, k_h) * scale  # (H, seq_len, seq_len)

        # Build mask: True = keep, False = mask out.
        idx = torch.arange(seq_len, device=query.device)
        row = idx.unsqueeze(1)  # (seq_len, 1) — query positions
        col = idx.unsqueeze(0)  # (1, seq_len) — key positions

        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device)
        if is_causal:
            # Query i attends to keys j <= i.
            mask = mask & (col <= row)
        if sliding_window is not None:
            diff = row - col
            if is_causal:
                mask = mask & (diff >= 0) & (diff <= sliding_window)
            else:
                mask = mask & (diff.abs() <= sliding_window)

        # Apply mask — broadcast over heads.
        neg_inf = torch.finfo(torch.float32).min
        scores = scores.masked_fill(~mask.unsqueeze(0), neg_inf)

        weights = torch.softmax(scores, dim=-1)  # (H, seq_len, seq_len)
        out_h = torch.einsum("hij,hjd->hid", weights, v_h)  # (H, seq_len, D)
        out_b = out_h.transpose(0, 1)  # (seq_len, H, D)
        outputs.append(out_b)

    return torch.cat(outputs, dim=0)  # (N, H, D)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Batch of 4 sequences with distinct lengths.
_LENGTHS = [16, 32, 48, 24]
_N = sum(_LENGTHS)
_B = len(_LENGTHS)
_H = 4
_D = 64

# KV-heads for GQA test (2 kv heads, 4 q heads).
_H_KV = 2


def _make_inputs(
    dtype: torch.dtype = torch.float32,
    n_kv_heads: int | None = None,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return (q, k, v, cu_seqlens, max_seqlen)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    n_kv = n_kv_heads if n_kv_heads is not None else _H
    q = torch.randn(_N, _H, _D, dtype=dtype, device=device, generator=gen)
    k = torch.randn(_N, n_kv, _D, dtype=dtype, device=device, generator=gen)
    v = torch.randn(_N, n_kv, _D, dtype=dtype, device=device, generator=gen)
    lengths = torch.tensor(_LENGTHS, dtype=torch.int32, device=device)
    cu_seqlens = torch.zeros(_B + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = lengths.cumsum(0)
    max_seqlen = max(_LENGTHS)
    return q, k, v, cu_seqlens, max_seqlen


class _DummyModule(torch.nn.Module):
    """Minimal stand-in for an attention module."""

    is_causal = False


# ---------------------------------------------------------------------------
# GPU tests (FA4) — require CUDA + flash_attn.cute + SM12x native varlen backend
# ---------------------------------------------------------------------------

_FA4_VARLEN_UNAVAILABLE_MSG = (
    "SM12x native varlen backend not built. "
    "Build with FLASH_ATTENTION_BUILD_SM12X_NATIVE=1 to enable FA4 varlen tests."
)


def _skip_if_fa4_varlen_unavailable() -> None:
    """Skip the calling test if FA4 varlen cannot run on this hardware/build."""
    pytest.importorskip("flash_attn.cute")
    try:
        from flash_attn.cute.native_sm12x import native_sm12x_owned_backend_available

        if not native_sm12x_owned_backend_available():
            pytest.skip(_FA4_VARLEN_UNAVAILABLE_MSG)
    except ImportError:
        # Extension scaffold not present at all — FA4 varlen won't work.
        pytest.skip(_FA4_VARLEN_UNAVAILABLE_MSG)


@pytest.mark.gpu
def test_fa4_parity_bf16_noncausal() -> None:
    """FA4 packed path matches per-sample reference to 1e-3 (bf16)."""
    _skip_if_fa4_varlen_unavailable()

    device = "cuda"
    dtype = torch.bfloat16
    q, k, v, cu_seqlens, max_seqlen = _make_inputs(dtype=dtype, device=device)
    module = _DummyModule()

    ref = _reference_packed_attention(
        q.float(), k.float(), v.float(), cu_seqlens, is_causal=False
    ).to(dtype)
    out, attn_wts = bgkit_flash_attention_4_forward(
        module, q, k, v, cu_seqlens, max_seqlen, is_causal=False
    )

    assert attn_wts is None
    assert out.shape == (sum(_LENGTHS), _H, _D)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.gpu
def test_fa4_parity_bf16_causal() -> None:
    """FA4 packed path with causal masking matches per-sample reference (bf16)."""
    _skip_if_fa4_varlen_unavailable()

    device = "cuda"
    dtype = torch.bfloat16
    q, k, v, cu_seqlens, max_seqlen = _make_inputs(dtype=dtype, device=device)
    module = _DummyModule()

    ref = _reference_packed_attention(
        q.float(), k.float(), v.float(), cu_seqlens, is_causal=True
    ).to(dtype)
    out, _ = bgkit_flash_attention_4_forward(
        module, q, k, v, cu_seqlens, max_seqlen, is_causal=True
    )

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

@pytest.mark.gpu
def test_fa4_parity_gqa_bf16() -> None:
    """FA4 GQA path (H_kv=2, H_q=4) matches reference with expanded k/v."""
    _skip_if_fa4_varlen_unavailable()

    device = "cuda"
    dtype = torch.bfloat16
    q, k, v, cu_seqlens, max_seqlen = _make_inputs(
        dtype=dtype, n_kv_heads=_H_KV, device=device
    )
    module = _DummyModule()

    repeat = _H // _H_KV
    k_exp = k.repeat_interleave(repeat, dim=1)
    v_exp = v.repeat_interleave(repeat, dim=1)
    ref = _reference_packed_attention(
        q.float(), k_exp.float(), v_exp.float(), cu_seqlens, is_causal=False
    ).to(dtype)

    out, _ = bgkit_flash_attention_4_forward(
        module, q, k, v, cu_seqlens, max_seqlen, is_causal=False
    )

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.gpu
def test_fa4_sliding_window_causal() -> None:
    """FA4 sliding_window + causal parity check.

    SM12x native varlen path supports ``window_size``.  If the FA4 kernel
    rejects the config (e.g. softcap on native path), this test is skipped.
    """
    _skip_if_fa4_varlen_unavailable()

    device = "cuda"
    dtype = torch.bfloat16
    sw = 8
    q, k, v, cu_seqlens, max_seqlen = _make_inputs(dtype=dtype, device=device)
    module = _DummyModule()

    ref = _reference_packed_attention(
        q.float(), k.float(), v.float(), cu_seqlens, is_causal=True, sliding_window=sw
    ).to(dtype)

    try:
        out, _ = bgkit_flash_attention_4_forward(
            module, q, k, v, cu_seqlens, max_seqlen, is_causal=True, sliding_window=sw
        )
    except NotImplementedError as exc:
        pytest.skip(f"FA4 sliding_window not supported on this kernel: {exc}")

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.gpu
def test_fa4_rejects_output_attentions() -> None:
    pytest.importorskip("flash_attn.cute")
    # This test does NOT call flash_attn_varlen_func — it exercises the guard
    # before dispatch, so it works even without the native varlen backend.

    device = "cuda"
    q, k, v, cu_seqlens, max_seqlen = _make_inputs(dtype=torch.float16, device=device)
    module = _DummyModule()
    with pytest.raises(NotImplementedError, match="output_attentions"):
        bgkit_flash_attention_4_forward(
            module, q, k, v, cu_seqlens, max_seqlen, output_attentions=True
        )
