"""Falcon-H1 Tiny specializations for the Mamba2 fused training op.

The HF Falcon-H1 mixer calls Mamba's generic
``mamba_split_conv1d_scan_combined`` autograd function. For
Falcon-H1-Tiny-90M training that generic function always lands in the same
subcase:

* ``d_nonssm == 0``
* ``rmsnorm_weight is None``
* ``outproj_weight is not None``
* ``ngroups == 1``
* training full-sequence path, no final recurrent states

This module owns that subcase explicitly while still reusing the upstream
causal-conv1d and SSD Triton kernels loaded by Transformers' hub-kernel
integration. It is intentionally narrow: callers should check the model shape
before routing here.

By default the forward saves the post-convolution ``x/B/C`` activation used by
the SSD scan. Falcon decoder training is backward-dominated; retaining that
activation avoids replaying causal-conv1d inside backward. Set
``BGKIT_FALCON_H1_MAMBA_SAVE_CONV=0`` to recover the lower-memory recompute
variant.

The forward also saves the Falcon-shaped SSD scan intermediates by default so
backward does not rebuild chunk cumsums, chunk states, and ``CB``. Set
``BGKIT_FALCON_H1_MAMBA_SAVE_SCAN=0`` to fall back to the upstream backward
wrapper's recompute path.

The same autograd boundary can also own Falcon's Mamba ``in_proj``. Passing
``inproj_weight`` (and optional ``inproj_bias``) makes the first tensor argument
the pre-projection hidden state and returns full trainable input-projection
gradients from the custom backward.

On GB10, Falcon-H1's trainable backward is faster when the simple
``D * dout`` contribution to ``dx`` is applied outside the generic SSD
``dx`` kernel. This is enabled by default; set
``BGKIT_FALCON_H1_MAMBA_SKIP_D_IN_DX_KERNEL=0`` to route ``D`` through the
upstream kernel for comparison.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import torch
import torch.nn.functional as F


def _env_truthy(name: str, default: str = "1") -> bool:
    raw = os.environ.get(name, default).strip().lower()
    return raw in {"1", "true", "yes", "on"}


_PROFILE_EVENTS: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}


def _profile_enabled() -> bool:
    return _env_truthy("BGKIT_FALCON_H1_PROFILE_INTERNALS", "0")


def _record_start(tensor: torch.Tensor):
    if not (_profile_enabled() and tensor.is_cuda):
        return None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return start, end


def _record_end(name: str, pair) -> None:
    if pair is None:
        return
    start, end = pair
    end.record()
    _PROFILE_EVENTS.setdefault(name, []).append((start, end))


def reset_falcon_h1_mamba_profile() -> None:
    _PROFILE_EVENTS.clear()


def summarize_falcon_h1_mamba_profile() -> list[dict[str, float | int | str]]:
    if _PROFILE_EVENTS:
        torch.cuda.synchronize()
    items: list[dict[str, float | int | str]] = []
    for name, events in sorted(_PROFILE_EVENTS.items()):
        total = sum(start.elapsed_time(end) for start, end in events)
        calls = len(events)
        items.append({
            "name": name,
            "calls": calls,
            "cuda_ms": total,
            "avg_cuda_ms": total / max(calls, 1),
        })
    items.sort(key=lambda item: float(item["cuda_ms"]), reverse=True)
    return items


def _ssd_module(stock_fn: Any):
    module = sys.modules.get(getattr(stock_fn, "__module__", ""))
    if module is not None and hasattr(module, "_mamba_chunk_scan_combined_fwd"):
        return module
    for cell in getattr(stock_fn, "__closure__", ()) or ():
        try:
            candidate = cell.cell_contents
        except ValueError:
            continue
        if not callable(candidate):
            continue
        candidate_module = sys.modules.get(getattr(candidate, "__module__", ""))
        if candidate_module is not None and hasattr(
            candidate_module, "_mamba_chunk_scan_combined_fwd"
        ):
            return candidate_module
    if module is None:
        raise RuntimeError(f"Mamba SSD module {stock_fn.__module__!r} is not loaded")
    if not hasattr(module, "_mamba_chunk_scan_combined_fwd"):
        raise RuntimeError(
            f"Mamba SSD module {getattr(stock_fn, '__module__', None)!r} lacks SSD internals"
        )
    return module


def _bd_to_bds(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(0, 2, 1)


def _bds_to_bsd(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(0, 2, 1)


def _ensure_stride(ssd, tensor: torch.Tensor, *, channel_dim: int) -> torch.Tensor:
    if hasattr(ssd, "ensure_stride"):
        return ssd.ensure_stride(tensor)
    if hasattr(ssd, "rearrange_and_update_stride"):
        return tensor.contiguous() if tensor.stride(channel_dim) % 8 != 0 else tensor
    return tensor.contiguous() if tensor.stride(channel_dim) % 8 != 0 else tensor


def _ssd_rearrange(ssd, tensor: torch.Tensor, pattern: str, **axes_lengths) -> torch.Tensor:
    if hasattr(ssd, "rearrange"):
        return ssd.rearrange(tensor, pattern, **axes_lengths)
    from einops import rearrange

    return rearrange(tensor, pattern, **axes_lengths)


def _scan_fwd_with_saved_intermediates(
    ssd,
    x: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    chunk_size: int,
    d: torch.Tensor,
    z: torch.Tensor,
    dt_bias: torch.Tensor,
    seq_idx: torch.Tensor | None,
    dt_limit: tuple[float, float],
):
    _, _, _, dstate = b.shape
    timer = _record_start(dt)
    da_cumsum, dt_scan = ssd._chunk_cumsum_fwd(
        dt,
        a,
        chunk_size,
        dt_bias=dt_bias,
        dt_softplus=True,
        dt_limit=dt_limit,
    )
    _record_end("mamba_fwd_scan_chunk_cumsum", timer)
    timer = _record_start(x)
    states = ssd._chunk_state_fwd(
        b,
        x,
        dt_scan,
        da_cumsum,
        seq_idx=seq_idx,
        states_in_fp32=True,
    )
    _record_end("mamba_fwd_scan_chunk_state", timer)
    timer = _record_start(states)
    states, final_states = ssd._state_passing_fwd(
        _ssd_rearrange(ssd, states, "... p n -> ... (p n)"),
        da_cumsum[:, :, :, -1],
        initial_states=None,
        seq_idx=seq_idx,
        chunk_size=chunk_size,
        out_dtype=c.dtype,
    )
    _record_end("mamba_fwd_scan_state_passing", timer)
    states = _ssd_rearrange(ssd, states, "... (p n) -> ... p n", n=dstate)
    timer = _record_start(c)
    cb = ssd._bmm_chunk_fwd(c, b, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    _record_end("mamba_fwd_scan_bmm_chunk", timer)
    timer = _record_start(x)
    out, out_x = ssd._chunk_scan_fwd(
        cb,
        x,
        dt_scan,
        da_cumsum,
        c,
        states,
        D=d,
        z=z,
        seq_idx=seq_idx,
    )
    _record_end("mamba_fwd_scan_chunk_scan", timer)
    return out, out_x, dt_scan, da_cumsum, states, cb, final_states


def _scan_bwd_from_saved_intermediates(
    ssd,
    dout: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    out: torch.Tensor,
    *,
    chunk_size: int,
    d: torch.Tensor,
    z: torch.Tensor,
    dt_bias: torch.Tensor,
    seq_idx: torch.Tensor | None,
    dt_limit: tuple[float, float],
    dt_scan: torch.Tensor,
    da_cumsum: torch.Tensor,
    states: torch.Tensor,
    cb: torch.Tensor,
    dx: torch.Tensor,
    ddt: torch.Tensor,
    db: torch.Tensor,
    dc: torch.Tensor,
    dz: torch.Tensor,
    recompute_output: bool,
):
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    _, _, ngroups, dstate = b.shape
    timer = _record_start(dt)
    dt_in = dt.clone()
    _record_end("mamba_bwd_scan_dt_clone", timer)

    timer = _record_start(dout)
    dz, dout, dd, *rest = ssd._chunk_scan_bwd_dz(
        x,
        z,
        out,
        dout,
        chunk_size=chunk_size,
        has_ddAcs=False,
        D=d,
        dz=dz,
        recompute_output=recompute_output,
    )
    _record_end("mamba_bwd_scan_dz", timer)
    outz = rest[0] if recompute_output else out

    timer = _record_start(dout)
    dstates = ssd._chunk_scan_bwd_dstates(
        c,
        da_cumsum,
        dout,
        seq_idx=seq_idx,
        dtype=states.dtype,
    )
    _record_end("mamba_bwd_scan_dstates", timer)
    timer = _record_start(dstates)
    dstates, dda_chunk_cumsum, _dinitial_states, states_for_bwd = (
        ssd._state_passing_bwd(
            _ssd_rearrange(ssd, states, "... p n -> ... (p n)"),
            da_cumsum[:, :, :, -1],
            _ssd_rearrange(ssd, dstates, "... p n -> ... (p n)"),
            dfinal_states=None,
            seq_idx=seq_idx,
            has_initial_states=False,
            dstates_dtype=x.dtype,
            states_dtype=x.dtype,
            chunk_size=chunk_size,
        )
    )
    _record_end("mamba_bwd_scan_state_passing", timer)
    states_for_bwd = _ssd_rearrange(ssd, states_for_bwd, "... (p n) -> ... p n", n=dstate)
    dstates = _ssd_rearrange(ssd, dstates, "... (p n) -> ... p n", n=dstate)

    skip_d_in_dx_kernel = _env_truthy("BGKIT_FALCON_H1_MAMBA_SKIP_D_IN_DX_KERNEL", "1")
    timer = _record_start(dout)
    dx, ddt_from_x, dd_from_x = ssd._chunk_scan_chunk_state_bwd_dx(
        x,
        dt_scan,
        da_cumsum,
        b,
        cb,
        dout,
        dstates,
        D=None if skip_d_in_dx_kernel else d,
        seq_idx=seq_idx,
        dx=dx,
    )
    _record_end("mamba_bwd_scan_dx", timer)
    if skip_d_in_dx_kernel:
        timer = _record_start(dx)
        d_for_dx = d.view(1, 1, d.shape[0], d.shape[1] if d.dim() == 2 else 1)
        dx.addcmul_(dout, d_for_dx)
        _record_end("mamba_bwd_scan_dx_d_residual", timer)
    timer = _record_start(x)
    db_state, dda_next = ssd._chunk_state_bwd_db(
        x,
        dt_scan,
        da_cumsum,
        dstates,
        seq_idx=seq_idx,
        B=b,
        ngroups=ngroups,
    )
    _record_end("mamba_bwd_scan_db_state", timer)
    timer = _record_start(dout)
    dc_scan, dda_cumsum_prev = ssd._chunk_scan_bwd_dC(
        states_for_bwd.to(x.dtype),
        da_cumsum,
        dout,
        seq_idx=seq_idx,
        C=c,
        ngroups=ngroups,
    )
    _record_end("mamba_bwd_scan_dc", timer)
    timer = _record_start(dout)
    dcb = ssd._chunk_scan_bwd_dcb(
        x,
        dt_scan,
        da_cumsum,
        dout,
        seq_idx=seq_idx,
        ngroups=ngroups,
    )
    _record_end("mamba_bwd_scan_dcb", timer)
    dcb = dcb.to(cb.dtype)
    timer = _record_start(dcb)
    ssd._bmm_chunk_bwd(c, dcb, residual=db_state, out=db)
    _record_end("mamba_bwd_scan_bmm_db", timer)
    timer = _record_start(dcb)
    ssd._bmm_chunk_bwd(
        b,
        _ssd_rearrange(ssd, dcb, "... l s -> ... s l"),
        residual=dc_scan,
        out=dc,
    )
    _record_end("mamba_bwd_scan_bmm_dc", timer)

    dda_cumsum_prev[..., -1] += dda_chunk_cumsum
    dda_prev = dda_cumsum_prev.flip([-1]).cumsum(dim=-1).flip([-1])
    timer = _record_start(dout)
    dda = ssd._chunk_scan_bwd_ddAcs_stable(x, dt_scan, da_cumsum, dout, cb)
    _record_end("mamba_bwd_scan_dda", timer)
    dda += dda_next + dda_prev

    timer = _record_start(dda)
    ddt, da, ddt_bias = ssd._chunk_cumsum_bwd(
        dda,
        ddt_from_x,
        dt_in,
        a,
        dt_bias=dt_bias,
        dt_softplus=True,
        dt_limit=dt_limit,
        ddt=ddt,
    )
    _record_end("mamba_bwd_scan_chunk_cumsum", timer)
    if z is None:
        dd = dd_from_x

    return_vals = (dx, ddt, da, db, dc, dd, dz, ddt_bias, None)
    return return_vals if not recompute_output else (*return_vals, outz)


def _conv1d_fwd(
    ssd,
    x_bds: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    seq_idx: torch.Tensor | None,
    activation: bool,
) -> torch.Tensor:
    if hasattr(ssd, "causal_conv1d_fwd_function") and ssd.causal_conv1d_fwd_function is not None:
        return ssd.causal_conv1d_fwd_function(
            x_bds,
            weight,
            bias,
            seq_idx,
            None,
            None,
            activation,
        )
    if hasattr(ssd, "causal_conv1d_cuda") and ssd.causal_conv1d_cuda is not None:
        return ssd.causal_conv1d_cuda.causal_conv1d_fwd(
            x_bds,
            weight,
            bias,
            seq_idx,
            None,
            None,
            activation,
        )
    try:
        from causal_conv1d.causal_conv1d_interface import causal_conv1d_cuda

        return causal_conv1d_cuda.causal_conv1d_fwd(
            x_bds,
            weight,
            bias,
            seq_idx,
            None,
            None,
            activation,
        )
    except Exception:
        pass
    raise RuntimeError("causal_conv1d forward kernel is not available")


def _conv1d_bwd(
    ssd,
    x_bds: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dx_bds: torch.Tensor,
    seq_idx: torch.Tensor | None,
    dx_given_bds: torch.Tensor,
    activation: bool,
):
    if hasattr(ssd, "causal_conv1d_bwd_function") and ssd.causal_conv1d_bwd_function is not None:
        return ssd.causal_conv1d_bwd_function(
            x_bds,
            weight,
            bias,
            dx_bds,
            seq_idx,
            None,
            None,
            dx_given_bds,
            False,
            activation,
        )
    if hasattr(ssd, "causal_conv1d_cuda") and ssd.causal_conv1d_cuda is not None:
        return ssd.causal_conv1d_cuda.causal_conv1d_bwd(
            x_bds,
            weight,
            bias,
            dx_bds,
            seq_idx,
            None,
            None,
            dx_given_bds,
            False,
            activation,
        )
    try:
        from causal_conv1d.causal_conv1d_interface import causal_conv1d_cuda

        return causal_conv1d_cuda.causal_conv1d_bwd(
            x_bds,
            weight,
            bias,
            dx_bds,
            seq_idx,
            None,
            None,
            dx_given_bds,
            False,
            activation,
        )
    except Exception:
        pass
    raise RuntimeError("causal_conv1d backward kernel is not available")


class _FalconH1MambaSplitConv1dScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        stock_fn,
        zxbcdt: torch.Tensor,
        conv1d_weight: torch.Tensor,
        conv1d_bias: torch.Tensor | None,
        dt_bias: torch.Tensor,
        a: torch.Tensor,
        d: torch.Tensor,
        chunk_size: int,
        seq_idx: torch.Tensor | None,
        activation: str,
        outproj_weight: torch.Tensor,
        outproj_bias: torch.Tensor | None,
        headdim: int,
        dt_limit: tuple[float, float],
        inproj_weight: torch.Tensor | None,
        inproj_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if activation not in {"silu", "swish"}:
            raise RuntimeError(
                f"Falcon-H1 specialized Mamba requires silu/swish, got {activation!r}"
            )
        if d.dim() != 1:
            raise RuntimeError("Falcon-H1 specialized Mamba expects 1D D")
        if outproj_weight is None:
            raise RuntimeError("Falcon-H1 specialized Mamba requires outproj_weight")

        has_inproj = inproj_weight is not None
        input_states = None
        if has_inproj:
            input_states = zxbcdt
            input_shape = tuple(input_states.shape)
            input_flat = input_states.reshape(-1, input_shape[-1])
            timer = _record_start(input_flat)
            zxbcdt = F.linear(input_flat, inproj_weight, inproj_bias).reshape(
                *input_shape[:-1],
                -1,
            )
            _record_end("mamba_fwd_inproj", timer)
        else:
            input_shape = None

        ssd = _ssd_module(stock_fn)
        batch, seqlen, last_dim = zxbcdt.shape
        nheads = d.shape[0]
        dim = nheads * headdim
        dstate = (conv1d_weight.shape[0] - dim) // 2
        expected = 2 * dim + 2 * dstate + nheads
        if last_dim != expected:
            raise RuntimeError(
                "Falcon-H1 specialized Mamba shape mismatch: "
                f"got last_dim={last_dim}, expected={expected}"
            )

        z, xbc, dt = torch.split(zxbcdt, [dim, dim + 2 * dstate, nheads], dim=-1)
        seq_idx = seq_idx.contiguous() if seq_idx is not None else None

        xbc_for_conv = _ensure_stride(ssd, xbc, channel_dim=1)
        timer = _record_start(xbc_for_conv)
        xbc_conv = _bds_to_bsd(
            _conv1d_fwd(
                ssd,
                _bd_to_bds(xbc_for_conv),
                conv1d_weight,
                conv1d_bias,
                seq_idx,
                True,
            )
        )
        _record_end("mamba_fwd_conv", timer)
        save_conv_for_bwd = _env_truthy("BGKIT_FALCON_H1_MAMBA_SAVE_CONV", "1")
        saved_xbc_conv = xbc_conv if save_conv_for_bwd else None
        x, b, c = torch.split(xbc_conv, [dim, dstate, dstate], dim=-1)
        x = x.view(batch, seqlen, nheads, headdim)
        b = b.view(batch, seqlen, 1, dstate)
        c = c.view(batch, seqlen, 1, dstate)
        z = z.view(batch, seqlen, nheads, headdim)

        save_scan_for_bwd = _env_truthy("BGKIT_FALCON_H1_MAMBA_SAVE_SCAN", "1")
        timer = _record_start(x)
        if save_scan_for_bwd:
            out, out_x, dt_scan, da_cumsum, states, cb, _final_states = (
                _scan_fwd_with_saved_intermediates(
                    ssd,
                    x,
                    dt,
                    a,
                    b,
                    c,
                    chunk_size=chunk_size,
                    d=d,
                    z=z,
                    dt_bias=dt_bias,
                    seq_idx=seq_idx,
                    dt_limit=dt_limit,
                )
            )
        else:
            out, out_x, dt_scan, da_cumsum, states, _final_states = (
                ssd._mamba_chunk_scan_combined_fwd(
                    x,
                    dt,
                    a,
                    b,
                    c,
                    chunk_size=chunk_size,
                    D=d,
                    z=z,
                    dt_bias=dt_bias,
                    initial_states=None,
                    seq_idx=seq_idx,
                    dt_softplus=True,
                    dt_limit=dt_limit,
                )
            )
            cb = None
            dt_scan = None
            da_cumsum = None
            states = None
        _record_end("mamba_fwd_scan", timer)
        out_for_linear = out.reshape(batch, seqlen, dim)

        ctx.stock_fn = stock_fn
        ctx.chunk_size = chunk_size
        ctx.headdim = headdim
        ctx.dt_limit = dt_limit
        ctx.has_inproj = has_inproj
        ctx.has_inproj_bias = inproj_bias is not None
        ctx.input_shape = input_shape
        save_out_for_bwd = _env_truthy("BGKIT_FALCON_H1_MAMBA_SAVE_OUT", "1")
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            out_for_linear = out_for_linear.to(dtype)
            outproj_weight = outproj_weight.to(dtype)
            outproj_bias = outproj_bias.to(dtype) if outproj_bias is not None else None
        saved_out_for_linear = out_for_linear if save_out_for_bwd else None
        ctx.save_for_backward(
            zxbcdt,
            conv1d_weight,
            conv1d_bias,
            out_x,
            a,
            d,
            dt_bias,
            seq_idx,
            outproj_weight,
            outproj_bias,
            saved_out_for_linear,
            saved_xbc_conv,
            dt_scan,
            da_cumsum,
            states,
            cb,
            input_states,
            inproj_weight,
        )

        timer = _record_start(out_for_linear)
        out_linear = F.linear(out_for_linear, outproj_weight, outproj_bias)
        _record_end("mamba_fwd_outproj", timer)
        return out_linear

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        (
            zxbcdt,
            conv1d_weight,
            conv1d_bias,
            out_x,
            a,
            d,
            dt_bias,
            seq_idx,
            outproj_weight,
            outproj_bias,
            saved_out_for_linear,
            saved_xbc_conv,
            dt_scan,
            da_cumsum,
            states,
            cb,
            input_states,
            inproj_weight,
        ) = ctx.saved_tensors
        ssd = _ssd_module(ctx.stock_fn)

        batch, seqlen, _ = zxbcdt.shape
        nheads = d.shape[0]
        headdim = ctx.headdim
        dim = nheads * headdim
        dstate = (conv1d_weight.shape[0] - dim) // 2

        z, xbc, dt = torch.split(zxbcdt, [dim, dim + 2 * dstate, nheads], dim=-1)
        xbc_for_conv = _ensure_stride(ssd, xbc, channel_dim=1)
        if saved_xbc_conv is None:
            timer = _record_start(xbc_for_conv)
            xbc_conv = _bds_to_bsd(
                _conv1d_fwd(
                    ssd,
                    _bd_to_bds(xbc_for_conv),
                    conv1d_weight,
                    conv1d_bias,
                    seq_idx,
                    True,
                )
            )
            _record_end("mamba_bwd_recompute_conv", timer)
        else:
            xbc_conv = saved_xbc_conv
        x, b, c = torch.split(xbc_conv, [dim, dstate, dstate], dim=-1)
        x = x.view(batch, seqlen, nheads, headdim)
        b = b.view(batch, seqlen, 1, dstate)
        c = c.view(batch, seqlen, 1, dstate)
        z = z.view(batch, seqlen, nheads, headdim)

        dzxbcdt = torch.empty_like(zxbcdt)
        dz, dxbc_given, ddt_given = torch.split(
            dzxbcdt, [dim, dim + 2 * dstate, nheads], dim=-1
        )
        dxbc = torch.empty_like(xbc)
        dx, db, dc = torch.split(dxbc, [dim, dstate, dstate], dim=-1)
        dz = dz.view(batch, seqlen, nheads, headdim)
        dx = dx.view(batch, seqlen, nheads, headdim)
        db = db.view(batch, seqlen, 1, dstate)
        dc = dc.view(batch, seqlen, 1, dstate)

        dout_og = dout
        timer = _record_start(dout)
        dout = F.linear(dout, outproj_weight.t())
        _record_end("mamba_bwd_outproj_input", timer)
        dout = dout.view(batch, seqlen, nheads, headdim)

        recompute_output = saved_out_for_linear is None
        timer = _record_start(dout)
        if cb is not None:
            dx, _ddt, da, db, dc, dd, dz, ddt_bias, _dinitial_states, *rest = (
                _scan_bwd_from_saved_intermediates(
                    ssd,
                    dout,
                    x,
                    dt,
                    a,
                    b,
                    c,
                    out_x,
                    chunk_size=ctx.chunk_size,
                    d=d,
                    z=z,
                    dt_bias=dt_bias,
                    seq_idx=seq_idx,
                    dt_limit=ctx.dt_limit,
                    dt_scan=dt_scan,
                    da_cumsum=da_cumsum,
                    states=states,
                    cb=cb,
                    dx=dx,
                    ddt=ddt_given,
                    db=db,
                    dc=dc,
                    dz=dz,
                    recompute_output=recompute_output,
                )
            )
        else:
            dx, _ddt, da, db, dc, dd, dz, ddt_bias, _dinitial_states, *rest = (
                ssd._mamba_chunk_scan_combined_bwd(
                    dout,
                    x,
                    dt,
                    a,
                    b,
                    c,
                    out_x,
                    ctx.chunk_size,
                    D=d,
                    z=z,
                    dt_bias=dt_bias,
                    initial_states=None,
                    dfinal_states=None,
                    seq_idx=seq_idx,
                    dt_softplus=True,
                    dt_limit=ctx.dt_limit,
                    dx=dx,
                    ddt=ddt_given,
                    dB=db,
                    dC=dc,
                    dz=dz,
                    recompute_output=recompute_output,
                )
            )
        _record_end("mamba_bwd_scan", timer)
        out_for_linear = (
            rest[0].reshape(batch, seqlen, dim)
            if recompute_output
            else saved_out_for_linear.reshape(batch, seqlen, dim)
        )

        timer = _record_start(dxbc)
        dxbc_update, dweight, dbias, *_ = _conv1d_bwd(
            ssd,
            _bd_to_bds(xbc_for_conv),
            conv1d_weight,
            conv1d_bias,
            _bd_to_bds(_ensure_stride(ssd, dxbc, channel_dim=1)),
            seq_idx,
            _bd_to_bds(_ensure_stride(ssd, dxbc_given, channel_dim=1)),
            True,
        )
        _record_end("mamba_bwd_conv", timer)
        dxbc_update = _bds_to_bsd(dxbc_update)
        if dxbc_given.stride() != dxbc_update.stride():
            dxbc_given.copy_(dxbc_update)
        else:
            dxbc_given = dxbc_update

        dout2d = dout_og.reshape(-1, dout_og.shape[-1])
        out2d = out_for_linear.reshape(-1, out_for_linear.shape[-1])
        timer = _record_start(dout2d)
        doutproj_weight = torch.mm(dout2d.t(), out2d)
        _record_end("mamba_bwd_outproj_weight", timer)
        doutproj_bias = dout_og.sum(dim=(0, 1)) if outproj_bias is not None else None

        dinproj_weight = None
        dinproj_bias = None
        if ctx.has_inproj:
            if input_states is None or inproj_weight is None:
                raise RuntimeError("Falcon-H1 Mamba in-proj backward missing saved tensors")
            dzxbcdt_2d = dzxbcdt.reshape(-1, dzxbcdt.shape[-1])
            input_2d = input_states.reshape(-1, input_states.shape[-1])
            timer = _record_start(dzxbcdt_2d)
            dinput_2d = dzxbcdt_2d.matmul(inproj_weight)
            dzxbcdt_or_input = dinput_2d.reshape(ctx.input_shape)
            dinproj_weight = dzxbcdt_2d.transpose(0, 1).matmul(input_2d)
            if ctx.has_inproj_bias:
                dinproj_bias = dzxbcdt_2d.sum(dim=0)
            _record_end("mamba_bwd_inproj", timer)
        else:
            dzxbcdt_or_input = dzxbcdt

        return (
            None,  # stock_fn
            dzxbcdt_or_input,
            dweight,
            dbias,
            ddt_bias,
            da,
            dd,
            None,  # chunk_size
            None,  # seq_idx
            None,  # activation
            doutproj_weight,
            doutproj_bias,
            None,  # headdim
            None,  # dt_limit
            dinproj_weight,
            dinproj_bias,
        )


def falcon_h1_mamba_split_conv1d_scan_combined(
    stock_fn,
    zxbcdt: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor | None,
    dt_bias: torch.Tensor,
    a: torch.Tensor,
    d: torch.Tensor,
    chunk_size: int,
    *,
    seq_idx: torch.Tensor | None,
    activation: str,
    outproj_weight: torch.Tensor,
    outproj_bias: torch.Tensor | None,
    headdim: int,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    inproj_weight: torch.Tensor | None = None,
    inproj_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the Falcon-H1 Tiny-specialized fused Mamba training op."""

    return _FalconH1MambaSplitConv1dScanFn.apply(
        stock_fn,
        zxbcdt,
        conv1d_weight,
        conv1d_bias,
        dt_bias,
        a,
        d,
        chunk_size,
        seq_idx,
        activation,
        outproj_weight,
        outproj_bias,
        headdim,
        dt_limit,
        inproj_weight,
        inproj_bias,
    )
