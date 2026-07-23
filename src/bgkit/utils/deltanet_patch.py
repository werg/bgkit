"""Patch Qwen3.5 GatedDeltaNet for numerical stability and packed (varlen) operation.

Two concerns are addressed here:

1. Gate-clamping (numerical stability)
   ====================================
   The chunk_gated_delta_rule backward pass computes exp(g_cum[i] - g_cum[j]) for
   position pairs within each chunk (size 64). For non-causal pairs (j > i), these
   differences are positive and can overflow float32 (max ~3.4e38 = exp(88)) when
   per-step gate magnitudes are large, producing inf * 0 = NaN.

   Pretrained Qwen3.5-0.8B-Base has heads with extreme A_log/dt_bias values that
   produce per-step g around -4.75, yielding cumulative g of -300+ over a chunk.
   This causes backward NaN at 83/128 sequence lengths in a single DeltaNet layer.

   Fix: clamp per-step g to -(88 / (chunk_size - 1)) ≈ -1.4 so the max exp
   argument in the backward stays within float32 range. This only affects heads
   with extreme decay rates; semantically, exp(-1.4 * 64) ≈ 1.5e-39 is still
   effectively zero (complete state forgetting within a chunk).

   Known issue upstream:
     - fla-org/flash-linear-attention#389 (closed without fix)
     - fla-org/flash-linear-attention#104
     - unslothai/unsloth#3155 (open, unresolved)

2. Packed (varlen) path
   =====================
   Wave 1 of the FA4 packed-attention migration removes all padded attention paths.
   DeltaNet layers now receive packed inputs: (1, N, H, D) tensors with cu_seqlens
   marking segment boundaries. The gate reset at sample boundaries is handled by
   fla-core when cu_seqlens is passed to chunk_gated_delta_rule.

   The patched layer forward accepts two extra kwargs:
     cu_seqlens : torch.LongTensor | None  — (B+1,) cumulative lengths; None = non-packed
     position_ids : torch.Tensor | None    — (N,) per-sample position restart; consumed
                                             by BidirectionalQwen35, not forwarded to fla

   When cu_seqlens is not None, it is forwarded to chunk_gated_delta_rule so fla
   handles the gate reset at sequence boundaries automatically.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from bgkit.utils.gdn_backend import (
    get_chunk_gated_delta_rule,
    requested_backend_name,
    resolved_backend_name,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# chunk_size=64 is the fla default for chunk_gated_delta_rule.
# Safe per-step clamp: -88 / (chunk_size - 1) ≈ -1.397.
# Empirically verified: -1.5 fails at T=63 (max exp = 63*1.5 = 94.5 > 88).
# -1.3 gives max exp = 63*1.3 = 81.9, safely within float32 range.
DEFAULT_G_CLAMP_MIN = -1.3
_PACKED_CONTEXT: contextvars.ContextVar[
    tuple[torch.Tensor | None, torch.Tensor | None]
] = contextvars.ContextVar("bgkit_deltanet_packed_context", default=(None, None))


@contextlib.contextmanager
def deltanet_packed_context(
    cu_seqlens: torch.Tensor | None,
    position_ids: torch.Tensor | None,
) -> Iterator[None]:
    """Expose packed sequence metadata to DeltaNet layers HF does not call with it."""

    token = _PACKED_CONTEXT.set((cu_seqlens, position_ids))
    try:
        yield
    finally:
        _PACKED_CONTEXT.reset(token)


def current_deltanet_packed_context() -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return packed metadata for the current decoder forward, if any."""

    return _PACKED_CONTEXT.get()


def _raw_gate_in_kernel_enabled() -> bool:
    return os.environ.get("BGKIT_DELTANET_RAW_GATE_IN_KERNEL", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# Diagnostic: dump the exact chunk_gated_delta_rule inputs for large multi-segment
# batches so the sm_121 deadlock can be replayed in isolation. Enabled by setting
# BGKIT_DUMP_DELTANET=<dir>; only dumps calls with > BGKIT_DUMP_DELTANET_MIN_NSEG
# segments (default 8). One file per call: dn_dump_<NNN>.pt.
_DN_DUMP_COUNTER = [0]


def _maybe_dump_deltanet_call(args: tuple, kwargs: dict) -> None:
    dump_dir = os.environ.get("BGKIT_DUMP_DELTANET")
    if not dump_dir:
        return
    cu = kwargs.get("cu_seqlens")
    if not isinstance(cu, torch.Tensor):
        return
    min_nseg = int(os.environ.get("BGKIT_DUMP_DELTANET_MIN_NSEG", "8"))
    if (cu.numel() - 1) <= min_nseg:
        return
    import os.path as _osp

    def _cpu(x):
        return x.detach().cpu() if isinstance(x, torch.Tensor) else x

    payload = {
        "args": [_cpu(a) for a in args],
        "kwargs": {k: _cpu(v) for k, v in kwargs.items()},
    }
    n = _DN_DUMP_COUNTER[0]
    _DN_DUMP_COUNTER[0] += 1
    torch.save(payload, _osp.join(dump_dir, f"dn_dump_{n:03d}.pt"))


def _ensure_raw_gate_stash(layer: nn.Module) -> None:
    """Stash raw a-projection outputs so packed GDR can compute gates in-kernel."""

    in_proj_a = getattr(layer, "in_proj_a", None)
    if in_proj_a is None or hasattr(layer, "_bgkit_original_in_proj_a_forward"):
        return
    original_a_forward = in_proj_a.forward

    def _stashing_in_proj_a_forward(*args, **kwargs):
        out = original_a_forward(*args, **kwargs)
        if _raw_gate_in_kernel_enabled():
            layer._bgkit_last_raw_gate_a = out
        return out

    layer._bgkit_original_in_proj_a_forward = original_a_forward
    in_proj_a.forward = _stashing_in_proj_a_forward


def patch_deltanet_layer(layer: nn.Module, g_clamp_min: float = DEFAULT_G_CLAMP_MIN) -> None:
    """Patch a single Qwen3_5GatedDeltaNet layer for stability and packed inputs.

    Two changes applied:
    1. Wraps self.chunk_gated_delta_rule with g clamping (existing behavior).
    2. Wraps self.forward to accept and forward cu_seqlens to chunk_gated_delta_rule.

    Must be called after the layer is constructed (i.e., after model loading).
    Idempotent: a second call replaces the previous patch (no double-wrapping).
    """
    if not hasattr(layer, "chunk_gated_delta_rule"):
        return

    # Unwrap any previous patch to stay idempotent. Read from the instance dict
    # so MagicMock-style dynamic __getattr__ does not manufacture truthy
    # "_unpatch_*" attributes that were never installed by this patch.
    try:
        layer_vars = vars(layer)
    except TypeError:
        layer_vars = {}
    original_fn = layer_vars.get("_unpatch_chunk_gdr") or layer.chunk_gated_delta_rule
    original_forward = layer_vars.get("_unpatch_forward") or layer.forward

    # ---- 0. Backend swap: replace the HF-wired callable with the resolver's
    #         pick unless the operator requested the FLA path. The HF default
    #         assigned in Qwen3_5GatedDeltaNet.__init__ is already an FLA
    #         callable; BgKIT's default remains FLA on sm_121. FlashQLA and
    #         auto modes resolve here before gate-clamp and cu_seqlens wrapping.
    #
    # BGKIT_GDN_BACKEND=fla preserves whatever HF wired up. That keeps the
    # optimized FLA path available without forcing a second resolver import.
    _gdn_choice = requested_backend_name()
    if _gdn_choice != "fla":
        try:
            backend_fn = get_chunk_gated_delta_rule()
        except Exception as exc:
            # Re-raise: the configured backend could not load. We must not
            # silently fall back to the HF default and pretend nothing happened.
            raise RuntimeError(
                f"deltanet_patch: gdn backend resolution failed: {exc}"
            ) from exc
        if backend_fn is not original_fn:
            original_fn = backend_fn

    # ---- 1. Patch chunk_gated_delta_rule: clamp g + forward cu_seqlens ----

    def _clamped(*args, **kwargs):
        # chunk_gated_delta_rule signature: (q, k, v, g, beta, ...)
        # HF calls with g= keyword; handle positional args too for robustness.
        # NOTE: transformers >= 5.12 passes ``cu_seqlens=kwargs.get("cu_seq_lens_q")``
        # EXPLICITLY (None when the layer forward received no kwargs — which is
        # exactly what happens under this patch's layer-forward wrapper), so the
        # guard must inject whenever the incoming value is None, not merely when
        # the key is absent. The old ``"cu_seqlens" not in kwargs`` guard silently
        # disabled the packed-context reset on 5.12+ → cross-sample DeltaNet
        # state leak on every multi-segment packed forward.
        # Only multi-segment contexts are injected: a single-segment [0, S]
        # context is semantically identical to cu_seqlens=None, and leaving
        # the proven G==1 path on its existing dense-batch kernel dispatch
        # keeps its numerics bit-identical.
        context_cu, _context_position_ids = current_deltanet_packed_context()
        if (
            context_cu is not None
            and context_cu.numel() > 2
            and kwargs.get("cu_seqlens") is None
        ):
            kwargs["cu_seqlens"] = context_cu
        raw_a = getattr(layer, "_bgkit_last_raw_gate_a", None)
        if _raw_gate_in_kernel_enabled() and isinstance(raw_a, torch.Tensor):
            if len(args) >= 4:
                args = list(args)
                args[3] = raw_a
                args = tuple(args)
            else:
                kwargs["g"] = raw_a
            kwargs["use_gate_in_kernel"] = True
            kwargs["A_log"] = layer.A_log
            kwargs["dt_bias"] = layer.dt_bias
            kwargs["gate_clamp_min"] = float(g_clamp_min)
            return original_fn(*args, **kwargs)
        if len(args) >= 4:
            args = list(args)
            args[3] = args[3].clamp(min=g_clamp_min)
            args = tuple(args)
        elif "g" in kwargs:
            kwargs["g"] = kwargs["g"].clamp(min=g_clamp_min)
        # cu_seqlens injected by _packed_forward below; pass straight through.
        _maybe_dump_deltanet_call(args, kwargs)
        return original_fn(*args, **kwargs)

    layer.chunk_gated_delta_rule = _clamped
    layer._unpatch_chunk_gdr = original_fn  # keep reference for idempotency
    layer._bgkit_g_clamp_min = float(g_clamp_min)

    # ---- 1b. Patch causal_conv1d_fn: reset the short conv at packed sample
    #          boundaries. The HF DeltaNet forward runs a depthwise causal
    #          conv (kernel 4) over the packed (1, C, N) stream BEFORE the
    #          delta rule; without a ``seq_idx`` each sample's first
    #          ``kernel_size - 1`` positions read the PREVIOUS sample's tail
    #          — a cross-sample leak the delta-rule ``cu_seqlens`` reset does
    #          not cover. Derive the per-token segment ids from the packed
    #          context and inject them whenever the caller supplied none.
    #          Single-segment contexts skip injection so the proven G==1 path
    #          stays bit-identical.
    #          Install rules: wrap only a live, not-yet-wrapped callable.
    #          - A second patch call leaves the existing wrapper in place
    #            (its closure already holds the original — no re-stacking).
    #          - When a caller NULLS ``causal_conv1d_fn`` after patching
    #            (``bidirectional_qwen35._make_conv_bidirectional`` does this
    #            to force the bidirectional nn.Conv1d fallback), a re-patch
    #            must NOT resurrect the causal CUDA conv from the stash.
    current_conv_fn = getattr(layer, "causal_conv1d_fn", None)
    already_wrapped = layer_vars.get("_unpatch_causal_conv1d_fn") is not None
    original_conv_fn = current_conv_fn if not already_wrapped else None
    if callable(original_conv_fn):

        def _conv_with_packed_seq_idx(*args, **kwargs):
            if kwargs.get("seq_idx") is None:
                context_cu, context_pos = current_deltanet_packed_context()
                x = kwargs.get("x", args[0] if args else None)
                if (
                    isinstance(context_cu, torch.Tensor)
                    and context_cu.numel() > 2
                    and isinstance(context_pos, torch.Tensor)
                    and isinstance(x, torch.Tensor)
                    and x.dim() == 3
                    and x.shape[0] == 1
                    # Guard: only when the conv input covers the full packed
                    # stream (position_ids length is host-side metadata — no
                    # device sync). Prefix-cache / other partial-length calls
                    # fall through untouched.
                    and context_pos.numel() == x.shape[-1]
                ):
                    n_seg = context_cu.numel() - 1
                    seg_ids = torch.arange(
                        n_seg, device=x.device, dtype=torch.int32,
                    )
                    kwargs["seq_idx"] = torch.repeat_interleave(
                        seg_ids,
                        (context_cu[1:] - context_cu[:-1]).to(torch.long),
                        output_size=int(x.shape[-1]),
                    ).unsqueeze(0)
            return original_conv_fn(*args, **kwargs)

        layer.causal_conv1d_fn = _conv_with_packed_seq_idx
        layer._unpatch_causal_conv1d_fn = original_conv_fn

    # ---- 2. Patch layer.forward to accept cu_seqlens / position_ids ----
    # Wave 1.1 will call:
    #   layer(hidden_states, pos_emb, cu_seqlens=cu_seqlens, position_ids=position_ids)
    # The HF forward has signature:
    #   forward(self, hidden_states, cache_params=None, attention_mask=None)
    # We need to intercept cu_seqlens and inject it into the chunk_gated_delta_rule call.
    # position_ids is consumed by the BidirectionalQwen35 rotary path; DeltaNet ignores it.

    def _packed_forward(
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask=None,
        *,
        cu_seqlens: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **_unused_kwargs,
    ):
        """Forward that injects cu_seqlens into chunk_gated_delta_rule.

        cu_seqlens : (B+1,) int32/int64 cumulative sequence lengths for packed input.
                     When provided, the input is (1, N, H, D) packed; fla's internal
                     gate-reset-at-boundaries logic activates.
        position_ids : consumed upstream by the RoPE / BidirectionalQwen35 wrapper;
                       DeltaNet itself does not use positional embeddings here.
        **_unused_kwargs: absorbs TransformersKwargs (cu_seq_lens_q/k, max_length_q/k,
                         etc.) that HF threads through decoder_layer.forward via
                         `**kwargs`. DeltaNet does not consume them, but they must be
                         accepted gracefully so the stock HF decoder loop can pass
                         them unconditionally.
        """
        # Defensive: if hidden_states arrives with 4D shape (a known shape mismatch
        # observed during packed decode), squeeze the singleton batch dimension so
        # the stock HF `batch_size, seq_len, _ = hidden_states.shape` unpack still
        # works. Tracks whether we squeezed so we can restore the shape after.
        _restore_4d = False
        if hidden_states.dim() == 4 and hidden_states.shape[0] == 1:
            hidden_states = hidden_states.squeeze(0)
            _restore_4d = True

        if cu_seqlens is None:
            for _alias in ("cu_seq_lens_q", "cu_seq_lens_k", "cu_seqlens"):
                _value = _unused_kwargs.get(_alias)
                if isinstance(_value, torch.Tensor):
                    cu_seqlens = _value
                    break
        if cu_seqlens is None or position_ids is None:
            # Consult the context for BOTH fields independently: when HF
            # threads cu via the TransformersKwargs aliases the loop above
            # resolves it, but position_ids never reaches the DeltaNet layer
            # that way — and the causal-conv1d seq_idx injection below keys
            # its full-length guard on the context position_ids. Re-entering
            # the context with position_ids=None would silently disable the
            # conv boundary reset on packed multi-sample forwards.
            context_cu, context_position_ids = current_deltanet_packed_context()
            if cu_seqlens is None:
                cu_seqlens = context_cu
            if position_ids is None:
                position_ids = context_position_ids

        if os.environ.get("BGKIT_DEBUG_DELTANET"):
            if isinstance(cu_seqlens, torch.Tensor):
                _segs = (cu_seqlens[1:] - cu_seqlens[:-1])
                _cl = (
                    f"nseg={cu_seqlens.numel() - 1} last={int(cu_seqlens[-1])} "
                    f"min_seg={int(_segs.min())} max_seg={int(_segs.max())} "
                    f"dtype={cu_seqlens.dtype}"
                )
            else:
                _cl = "None"
            print(
                f"[dndbg] hs={tuple(hidden_states.shape)} cu_seqlens[{_cl}]",
                flush=True,
            )

        if _raw_gate_in_kernel_enabled():
            _ensure_raw_gate_stash(layer)

        try:
            if cu_seqlens is None:
                # Non-packed call — legacy path used during transition or
                # single-sample generation.
                out = original_forward(hidden_states, cache_params, attention_mask)
            else:
                # The persistent wrapper on ``chunk_gated_delta_rule`` reads
                # this context and injects ``cu_seqlens``. This avoids creating
                # and assigning a temporary per-call wrapper on every packed
                # DeltaNet layer forward.
                with deltanet_packed_context(cu_seqlens, position_ids):
                    out = original_forward(hidden_states, cache_params, None)
        finally:
            layer._bgkit_last_raw_gate_a = None

        if _restore_4d:
            out = out.unsqueeze(0)
        return out

    layer.forward = _packed_forward
    layer._unpatch_forward = original_forward  # keep reference for idempotency


def patch_gated_delta_rule_numerics(
    model: nn.Module | None = None,
    g_clamp_min: float = DEFAULT_G_CLAMP_MIN,
) -> None:
    """Patch GatedDeltaNet layers for stability and packed (varlen) operation.

    Two modes:
    1. model=None: Patches the class __init__ so all future instances get patched.
    2. model=<nn.Module>: Patches existing DeltaNet layer instances in the model.

    For training, call with model=None early (before model loading) AND then
    call again with the loaded model to patch existing instances.

    The patch is idempotent: repeated calls on the same model do not stack.

    Args:
        model: If provided, patch all DeltaNet layers in this model.
        g_clamp_min: Minimum per-step gate value (negative). Default -1.3.
    """
    if model is not None:
        # Patch existing instances
        count = 0
        for module in model.modules():
            if hasattr(module, "chunk_gated_delta_rule") and hasattr(module, "A_log"):
                patch_deltanet_layer(module, g_clamp_min)
                count += 1
        if count > 0:
            logger.info(
                "GatedDeltaNet patched: %d layers, per-step g clamped to >= %.2f, "
                "cu_seqlens varlen path enabled, gdn_backend=%s",
                count,
                g_clamp_min,
                resolved_backend_name() or requested_backend_name(),
            )
        return

    # Patch the class so future instances are automatically patched
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as qwen_mod
    except ImportError:
        logger.warning("transformers.models.qwen3_5 not found, skipping deltanet patch")
        return

    gated_cls = getattr(qwen_mod, "Qwen3_5GatedDeltaNet", None)
    if gated_cls is None:
        return

    original_init = gated_cls.__init__

    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        patch_deltanet_layer(self, g_clamp_min)

    gated_cls.__init__ = _patched_init

    logger.info(
        "GatedDeltaNet class patched: per-step g will be clamped to >= %.2f "
        "(prevents backward NaN); cu_seqlens varlen path enabled; "
        "gdn_backend=%s (resolves on first layer init)",
        g_clamp_min,
        requested_backend_name(),
    )


def patch_fused_rms_norm_gated_for_sm121() -> None:
    """Replace fla.modules.FusedRMSNormGated.forward with a pure-PyTorch fallback.

    Why: fla's ``layer_norm_gated_bwd`` Triton kernel deadlocks on sm_121
    on certain shapes — observed as a hang at random training steps after
    ~1000-2000 steps (py-spy stack: ``backward → layer_norm_gated_bwd``,
    GPU at 96 percent in an unkillable spin). Replacing the autograd
    Function-wrapped fused kernel with naive PyTorch RMSNorm + gate ops
    sidesteps the problem at the cost of higher activation memory and
    slower per-step compute. Idempotent.
    """
    try:
        from fla.modules import fused_norm_gate as _fng
    except ImportError:
        logger.warning(
            "fla.modules.fused_norm_gate not found; skipping FusedRMSNormGated patch"
        )
        return

    cls = getattr(_fng, "FusedRMSNormGated", None)
    if cls is None:
        return
    if getattr(cls, "_bgkit_sm121_patched", False):
        return

    import torch.nn.functional as F

    def _fallback_forward(
        self,
        x,
        g,
        residual=None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ):
        compute_dtype = torch.float32
        x_in = x.to(compute_dtype) if residual_in_fp32 else x
        if residual is not None:
            res = residual.to(compute_dtype) if residual_in_fp32 else residual
            x_in = x_in + res
        residual_out = x_in
        var = x_in.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_in * torch.rsqrt(var + self.eps)
        if self.elementwise_affine and self.weight is not None:
            x_normed = x_normed * self.weight.to(x_normed.dtype)
        if self.activation in ("swish", "silu"):
            gate = F.silu(g.to(x_normed.dtype))
        elif self.activation == "sigmoid":
            gate = torch.sigmoid(g.to(x_normed.dtype))
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        out = x_normed * gate
        out = out.to(x.dtype)
        if prenorm:
            return out, residual_out.to(x.dtype)
        return out

    cls.forward = _fallback_forward
    cls._bgkit_sm121_patched = True
    logger.info(
        "FusedRMSNormGated patched: pure-PyTorch fallback (sidesteps "
        "layer_norm_gated_bwd Triton hang on sm_121)."
    )
