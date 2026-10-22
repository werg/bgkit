"""Parity + memory tests for the Falcon-H1 Mamba scan-checkpoint variant.

``BGKIT_FALCON_H1_MAMBA_SCAN_CHECKPOINT=1`` makes the specialized Falcon-H1
fused Mamba op keep only the chunk-boundary states (``O(nchunks)``) in the
forward and recompute the large ``O(L)`` per-position scan intermediates
(``out_x`` / ``CB`` / ``dt_scan`` / ``dA_cumsum`` / the pre-out-projection
activation) in the backward. Because backward replays the *same* forward
kernels on the *same* inputs, the rebuilt intermediates are numerically
equal to the saved ones, so forward output and all input gradients must
match the default save-everything path within tight tolerance.

These tests require CUDA + Triton + mamba-ssm and therefore only run inside
the training Docker image (``-m gpu``). The module also exposes a
``measure_peak()`` harness and a ``__main__`` entry point so the forward
peak at long sequence length can be measured GPU-free of the live trainer:

    python tests/integration/test_falcon_h1_mamba_scan_checkpoint.py --seqlen 8192
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
import torch

# Falcon-H1-Tiny-90M Mamba mixer shape (see config.json):
#   mamba_n_heads=24, mamba_d_head=32, mamba_d_ssm=768, mamba_d_state=64,
#   mamba_n_groups=1, mamba_chunk_size=128, mamba_d_conv=4.
_FALCON_TINY = dict(nheads=24, headdim=32, dstate=64, ngroups=1, chunk_size=128, d_conv=4)


@contextmanager
def _scan_checkpoint_env(enabled: bool):
    """Pin the scan-checkpoint flag (and a stable default save policy) so the
    forward reads a known configuration regardless of ambient env."""
    keys = {
        "BGKIT_FALCON_H1_MAMBA_SCAN_CHECKPOINT": "1" if enabled else "0",
        # Default save-everything policy for the baseline arm.
        "BGKIT_FALCON_H1_MAMBA_SAVE_SCAN": "1",
        "BGKIT_FALCON_H1_MAMBA_SAVE_CONV": "1",
        "BGKIT_FALCON_H1_MAMBA_SAVE_OUT": "1",
    }
    saved = {k: os.environ.get(k) for k in keys}
    try:
        os.environ.update(keys)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _make_inputs(
    *,
    batch: int,
    seqlen: int,
    in_dim: int,
    use_inproj: bool,
    device: str,
    dtype: torch.dtype,
    seed: int = 0,
    shape: dict | None = None,
):
    """Build a fresh set of leaf tensors for one kernel call.

    Returns (kwargs_for_call, leaves) where ``leaves`` is the ordered dict of
    differentiable inputs whose ``.grad`` we compare across modes.
    """
    cfg = shape or _FALCON_TINY
    nheads = cfg["nheads"]
    headdim = cfg["headdim"]
    dstate = cfg["dstate"]
    d_conv = cfg["d_conv"]
    dim = nheads * headdim
    conv_dim = dim + 2 * dstate
    proj_dim = 2 * dim + 2 * dstate + nheads
    out_features = dim  # arbitrary; matches a square out-proj for the test

    g = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*sz):
        return torch.randn(*sz, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)

    if use_inproj:
        first = randn(batch, seqlen, in_dim).requires_grad_(True)
        inproj_weight = randn(proj_dim, in_dim).requires_grad_(True)
        inproj_bias = randn(proj_dim).requires_grad_(True)
    else:
        first = randn(batch, seqlen, proj_dim).requires_grad_(True)
        inproj_weight = None
        inproj_bias = None

    conv1d_weight = randn(conv_dim, d_conv).requires_grad_(True)
    conv1d_bias = randn(conv_dim).requires_grad_(True)
    dt_bias = randn(nheads).requires_grad_(True)
    # A = -exp(A_log) in real use, i.e. strictly negative.
    a = (-torch.rand(nheads, generator=g, dtype=torch.float32) - 0.5).to(
        device=device, dtype=torch.float32
    ).requires_grad_(True)
    d = randn(nheads).requires_grad_(True)
    outproj_weight = randn(out_features, dim).requires_grad_(True)
    outproj_bias = randn(out_features).requires_grad_(True)

    leaves = {
        "first": first,
        "conv1d_weight": conv1d_weight,
        "conv1d_bias": conv1d_bias,
        "dt_bias": dt_bias,
        "a": a,
        "d": d,
        "outproj_weight": outproj_weight,
        "outproj_bias": outproj_bias,
    }
    if use_inproj:
        leaves["inproj_weight"] = inproj_weight
        leaves["inproj_bias"] = inproj_bias

    kwargs = dict(
        zxbcdt=first,
        conv1d_weight=conv1d_weight,
        conv1d_bias=conv1d_bias,
        dt_bias=dt_bias,
        a=a,
        d=d,
        chunk_size=cfg["chunk_size"],
        seq_idx=None,
        activation="silu",
        outproj_weight=outproj_weight,
        outproj_bias=outproj_bias,
        headdim=headdim,
        inproj_weight=inproj_weight,
        inproj_bias=inproj_bias,
    )
    return kwargs, leaves


def _run_once(*, checkpoint: bool, grad_out: torch.Tensor, **make_kwargs):
    """Run one forward+backward of the specialized op and return (out, grads)."""
    from bgkit.kernels.falcon_h1_mamba import falcon_h1_mamba_split_conv1d_scan_combined

    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
    except Exception as exc:  # pragma: no cover - import guard
        pytest.skip(f"requires mamba-ssm SSD kernels: {exc}")

    kwargs, leaves = _make_inputs(**make_kwargs)
    with _scan_checkpoint_env(checkpoint):
        out = falcon_h1_mamba_split_conv1d_scan_combined(
            mamba_split_conv1d_scan_combined, **kwargs
        )
        out.backward(grad_out)
    grads = {k: (v.grad.detach().float().clone() if v.grad is not None else None)
             for k, v in leaves.items()}
    return out.detach().float().clone(), grads


@pytest.mark.gpu
@pytest.mark.parametrize("use_inproj", [False, True])
def test_scan_checkpoint_matches_save_everything(use_inproj: bool):
    """Forward output AND every input gradient match the default path."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    common = dict(
        batch=2,
        seqlen=80,  # spans multiple chunk_size=... boundaries when chunk small
        in_dim=128,
        use_inproj=use_inproj,
        device="cuda",
        dtype=torch.bfloat16,
        seed=7,
        # Smaller mixer so the test is cheap but still multi-chunk.
        shape=dict(nheads=4, headdim=16, dstate=16, ngroups=1, chunk_size=16, d_conv=4),
    )

    # Identical upstream gradient for both arms.
    torch.manual_seed(99)
    grad_out = torch.randn(
        common["batch"], common["seqlen"], common["shape"]["nheads"] * common["shape"]["headdim"],
        device="cuda", dtype=torch.bfloat16,
    )

    out_base, grads_base = _run_once(checkpoint=False, grad_out=grad_out, **common)
    out_ckpt, grads_ckpt = _run_once(checkpoint=True, grad_out=grad_out, **common)

    # Forward path is byte-identical up to what is *saved*, so the output must
    # match to bf16 round-off (recompute happens only in backward).
    torch.testing.assert_close(out_ckpt, out_base, rtol=2e-2, atol=2e-2)

    for name in grads_base:
        gb, gc = grads_base[name], grads_ckpt[name]
        assert (gb is None) == (gc is None), name
        if gb is None:
            continue
        diff = (gc - gb).abs().max().item()
        scale = gb.abs().max().item() + 1e-6
        assert diff / scale < 5e-2, f"grad[{name}] rel diff {diff / scale:.3e} (abs {diff:.3e})"


def measure_peak(
    *, seqlen: int, batch: int = 1, dtype=torch.bfloat16
) -> dict[str, dict[str, float]]:
    """Measure CUDA memory (GB) for both modes at ``seqlen``.

    For each mode returns {"saved_after_fwd": gb, "fwd_bwd_peak": gb} where
    ``saved_after_fwd`` is ``memory_allocated`` right after the forward returns
    (the retained saved-activation footprint this fix targets) and
    ``fwd_bwd_peak`` is the ``max_memory_allocated`` across forward+backward.
    Intended to be run GPU-free of the live trainer (one short call per mode).
    """
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

    from bgkit.kernels.falcon_h1_mamba import falcon_h1_mamba_split_conv1d_scan_combined

    results: dict[str, dict[str, float]] = {}
    for label, ckpt in (("save_everything", False), ("scan_checkpoint", True)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        kwargs, leaves = _make_inputs(
            batch=batch, seqlen=seqlen, in_dim=512, use_inproj=True,
            device="cuda", dtype=dtype, seed=0,
        )
        grad_out = torch.randn(
            batch, seqlen, _FALCON_TINY["nheads"] * _FALCON_TINY["headdim"],
            device="cuda", dtype=dtype,
        )
        base = torch.cuda.memory_allocated()
        with _scan_checkpoint_env(ckpt):
            out = falcon_h1_mamba_split_conv1d_scan_combined(
                mamba_split_conv1d_scan_combined, **kwargs
            )
            torch.cuda.synchronize()
            saved_after_fwd = (torch.cuda.memory_allocated() - base) / 1e9
            out.backward(grad_out)
        torch.cuda.synchronize()
        results[label] = {
            "saved_after_fwd": saved_after_fwd,
            "fwd_bwd_peak": torch.cuda.max_memory_allocated() / 1e9,
        }
        del out, kwargs, leaves, grad_out
    return results


@pytest.mark.gpu
def test_scan_checkpoint_reduces_peak():
    """Checkpoint mode lowers the forward+backward peak at long seqlen."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    try:
        import mamba_ssm
    except Exception:
        pytest.skip("requires mamba-ssm")

    peaks = measure_peak(seqlen=2048, batch=1)
    # The retained saved-activation footprint must drop, and the combined
    # forward+backward peak must not regress.
    assert peaks["scan_checkpoint"]["saved_after_fwd"] < peaks["save_everything"][
        "saved_after_fwd"
    ], peaks
    assert peaks["scan_checkpoint"]["fwd_bwd_peak"] <= peaks["save_everything"][
        "fwd_bwd_peak"
    ] + 1e-3, peaks


if __name__ == "__main__":  # pragma: no cover - manual GPU harness
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    peaks = measure_peak(seqlen=args.seqlen, batch=args.batch)
    print(f"seqlen={args.seqlen} batch={args.batch}")
    for metric in ("saved_after_fwd", "fwd_bwd_peak"):
        save = peaks["save_everything"][metric]
        ckpt = peaks["scan_checkpoint"][metric]
        pct = 100 * (save - ckpt) / save if save else 0.0
        print(f"  {metric:16s}  save={save:8.3f} GB  ckpt={ckpt:8.3f} GB  "
              f"reduction={save - ckpt:8.3f} GB ({pct:5.1f}%)")
