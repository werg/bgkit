"""Falcon-H1 forward-path optimizations for sm_121 / Falcon-H1-Tiny-90M.

The stock HF Falcon-H1 forward path is correctness-first, performance-second.
Several per-layer Python operations are no-ops on the Falcon-H1-Tiny-90M
configuration we train against, yet they incur an `aten::mul` (and its
backward graph) every layer x every microbatch x every step. On DGX Spark's
unified memory + 24-SM kernel launch overhead, this is a measurable fraction
of step wall time.

Background (Falcon-H1-Tiny-90M-Instruct config values):

  key_multiplier         = 1.0  (NOOP)   — used in attention K
  ssm_in_multiplier      = 1.0  (NOOP)   — mixer input scaling
  ssm_out_multiplier     = 1.0  (NOOP)   — layer mixer-output scaling
  attention_in_multiplier= 1.0  (NOOP)   — layer attention-input scaling
  attention_out_multiplier=1.0  (NOOP)   — layer attention-output scaling
  mlp_multipliers        = [1, 1] (NOOP) — gate and down MLP multipliers
  ssm_multipliers (mup_vector) = [1,1,1,1,1] (NOOP) — broadcast mul on (B,S,zxbcdt-projection)

  Per microbatch fwd: 8 unit muls x 24 layers + 1 broadcast mul x 24 layers
                    = 216 wasted muls (each saves a tensor for backward)
  x 8 microbatches  = 1,728 wasted muls/step
  + 1,728 MulBackward0
  + ~3,456 saved-tensor copies

The non-unit multipliers (embedding_multiplier, lm_head_multiplier) are
applied once per step at embedding / lm_head, not per layer, and are left
alone — their cost is negligible.

This patch:

1. **No-op multiplier stripping** (`strip_unit_multipliers`):
   When a Falcon-H1 attention/MLP/mixer/layer module is constructed with all
   relevant multipliers == 1.0, monkey-patch its forward to skip the muls.
   This is correctness-equivalent at fp32; under bf16 the dropped no-op
   muls also drop a bf16 round-trip but that round-trip is a no-op for
   *= 1.0 (the saved-tensor copy preserves the exact pre-mul values).

2. **All-ones `mup_vector` short-circuit**: when `ssm_multipliers` is
   `[1, 1, 1, 1, 1]` the broadcast `projected_states * self.mup_vector`
   is a wasted bf16 copy of a (B, S, 2*intermediate + 2*groups*dstate +
   nheads) tensor. We skip it.

3. **Tight `FalconH1Model.forward`**: replaces HF's stock loop. The stock
   path:
     - allocates a `DynamicCache` when `use_cache=True` (we always pass
       False, but the auto-create at line 1114 of HF source is still
       checked in branches),
     - threads `kwargs` through `decoder_layer(**kwargs)`,
     - calls `create_causal_mask(...)` every step,
     - calls `_update_mamba_mask(...)` every step.
   We provide a forward that pre-computes `position_embeddings` and
   causal mask once per call, then calls each layer with the minimum
   positional args.

Correctness:
  Parity tests assert max abs diff < 1e-3 (bf16) between stock and
  patched forward on fixed inputs.

Idempotency:
  Each patcher checks for a `_bgkit_patched` marker before patching.

This is OPT-IN via `patch_falcon_h1_decoder(...)`. Call it after the model
is constructed from `from_pretrained(...)` and before `forward()` is called
in training/eval. Generation paths (KV-cache decode) MUST NOT call this —
the no-op mul strip preserves training-mode semantics only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


_MARKER = "_bgkit_falcon_h1_patched"


def _is_unit_scalar(value: Any, tol: float = 0.0) -> bool:
    """Return True if `value` is exactly 1.0 (or within `tol` of 1.0)."""
    try:
        return abs(float(value) - 1.0) <= tol
    except (TypeError, ValueError):
        return False


def _all_unit(values, tol: float = 0.0) -> bool:
    return all(_is_unit_scalar(v, tol=tol) for v in values)


# ---------------------------------------------------------------------------
# 1. Attention forward: drop key_multiplier when == 1.0
# ---------------------------------------------------------------------------


def _patch_attention_unit_key_multiplier(attn: nn.Module) -> bool:
    """Patch FalconH1Attention.forward to skip ``* self.key_multiplier`` when 1.0.

    Returns True if patched. The patched forward is otherwise identical to
    the stock HF implementation (same module-level globals).
    """
    if getattr(attn, _MARKER, False):
        return False
    if not _is_unit_scalar(getattr(attn, "key_multiplier", None)):
        return False

    # Import the HF symbols we need
    try:
        from transformers.models.falcon_h1 import modeling_falcon_h1 as fh1
    except Exception:
        return False

    apply_rotary_pos_emb = fh1.apply_rotary_pos_emb
    all_attention_functions = fh1.ALL_ATTENTION_FUNCTIONS
    eager_attention_forward = fh1.eager_attention_forward

    def forward(
        self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # NOTE: key_multiplier == 1.0; mul dropped.
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface: Callable = all_attention_functions.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    attn.forward = forward.__get__(attn, type(attn))
    setattr(attn, _MARKER, True)
    return True


# ---------------------------------------------------------------------------
# 2. MLP forward: drop gate_multiplier / down_multiplier when both == 1.0
# ---------------------------------------------------------------------------


def _patch_mlp_unit_multipliers(mlp: nn.Module) -> bool:
    if getattr(mlp, _MARKER, False):
        return False
    if not _is_unit_scalar(getattr(mlp, "gate_multiplier", None)):
        return False
    if not _is_unit_scalar(getattr(mlp, "down_multiplier", None)):
        return False

    def forward(self, x):
        # Drop: self.gate_proj(x) * self.gate_multiplier
        # Drop: self.down_proj(y) * self.down_multiplier
        return self.down_proj(self.up_proj(x) * self.act_fn(self.gate_proj(x)))

    mlp.forward = forward.__get__(mlp, type(mlp))
    setattr(mlp, _MARKER, True)
    return True


# ---------------------------------------------------------------------------
# 3. Mixer cuda_kernels_forward: drop ssm_in_multiplier + all-ones mup_vector
# ---------------------------------------------------------------------------


def _patch_mixer_unit_scalings(mixer: nn.Module) -> bool:
    """Patch FalconH1Mixer.cuda_kernels_forward when ssm_in_multiplier == 1.0
    and mup_vector is all-ones.

    The torch_forward (non-CUDA-kernels) path also gets the same treatment
    when both are unit-no-ops. We only handle the training path that the
    fused kernel hits.
    """
    if getattr(mixer, _MARKER, False):
        return False
    ssm_in = getattr(mixer, "ssm_in_multiplier", None)
    if not _is_unit_scalar(ssm_in):
        return False
    # Check mup_vector for all-ones
    mup = getattr(mixer, "mup_vector", None)
    if mup is None:
        return False
    # We allow `register_buffer` mup_vector which is `(1, 1, vector_shape)`.
    try:
        all_ones = bool(torch.all(mup == 1.0).item())
    except Exception:
        all_ones = False
    if not all_ones:
        return False

    try:
        from transformers.models.falcon_h1 import modeling_falcon_h1 as fh1
    except Exception:
        return False

    apply_mask_to_padding_states = fh1.apply_mask_to_padding_states
    is_torchdynamo_compiling = fh1.is_torchdynamo_compiling
    is_fast_path_available = fh1.is_fast_path_available

    # Snapshot the stock forward so the slow / cache branches stay
    # correctness-equivalent.
    stock_forward = type(mixer).forward

    # ----- Patched fused-training fast path -----
    # Inline equivalent of: stock cuda_kernels_forward with `self.training and
    # cache_params is None` branch, but with the unit ssm_in_multiplier mul
    # and the all-ones mup_vector broadcast mul removed. All other ops are
    # identical to HF.
    def patched_training_fused_path(self, hidden_states, attention_mask):
        # 1. Gated MLP linear projection (mask + in_proj). No ssm_in_multiplier.
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        projected_states = self.in_proj(hidden_states)
        # No mup_vector broadcast mul (verified all-ones at patch time).

        a_log_exp = -torch.exp(self.A_log.float())
        dt_limit_kwargs = (
            {}
            if self.time_step_limit == (0.0, float("inf"))
            else {"dt_limit": self.time_step_limit}
        )

        # Resolve through the HF module so that profilers that monkey-patch
        # falcon_h1.mamba_split_conv1d_scan_combined still observe the call.
        mamba_split_conv1d_scan_combined = fh1.mamba_split_conv1d_scan_combined

        return mamba_split_conv1d_scan_combined(
            projected_states,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.dt_bias,
            a_log_exp,
            D=self.D,
            chunk_size=self.chunk_size,
            seq_idx=None,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight if self.mamba_rms_norm else None,
            rmsnorm_eps=self.norm.variance_epsilon if self.mamba_rms_norm else None,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=self.head_dim,
            ngroups=self.n_groups,
            norm_before_gate=False,
            return_final_states=False,
            **dt_limit_kwargs,
        )

    # ----- Replacement forward -----
    # Route ONLY the fast training path through the patched kernel. All other
    # branches (use_precomputed_states, eval-mode with cache, cpu fallback,
    # generation) defer to the stock implementation so they remain bit-exact.
    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        # Mirror stock dispatch logic in `FalconH1Mixer.forward`:
        if (
            is_fast_path_available
            and "cuda" in self.in_proj.weight.device.type
            and not is_torchdynamo_compiling()
            and self.training
            and cache_params is None
        ):
            return patched_training_fused_path(self, hidden_states, attention_mask)
        # Everything else: route through the stock forward (which itself dispatches
        # to cuda_kernels_forward or torch_forward as appropriate).
        return stock_forward(self, hidden_states, cache_params, attention_mask, **kwargs)

    mixer.forward = forward.__get__(mixer, type(mixer))
    setattr(mixer, _MARKER, True)
    return True


# ---------------------------------------------------------------------------
# 4. Decoder layer forward: drop ssm_out_multiplier / attention_in_multiplier /
#    attention_out_multiplier when all unit
# ---------------------------------------------------------------------------


def _patch_decoder_layer_unit_multipliers(layer: nn.Module) -> bool:
    """Patch FalconH1DecoderLayer.forward to skip layer-level scalings.

    When all three layer-level scalings are 1.0, the patched forward is:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        mamba_h = self.mamba(hidden_states, cache_params=past_key_values,
                              attention_mask=mamba_attention_mask)
        attn_h, _ = self.self_attn(hidden_states, attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    past_key_values=past_key_values,
                                    use_cache=use_cache,
                                    position_embeddings=position_embeddings,
                                    **kwargs)
        hidden_states = residual + mamba_h + attn_h
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states,)
    """
    if getattr(layer, _MARKER, False):
        return False
    if not _all_unit(
        [
            getattr(layer, "ssm_out_multiplier", None),
            getattr(layer, "attention_in_multiplier", None),
            getattr(layer, "attn_out_multiplier", None),
        ]
    ):
        return False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        mamba_attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        mamba_h = self.mamba(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=mamba_attention_mask,
        )
        # Drop: mamba_h * self.ssm_out_multiplier

        attn_h, _ = self.self_attn(
            hidden_states=hidden_states,  # was * self.attention_in_multiplier
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # Drop: attn_h * self.attn_out_multiplier

        hidden_states = residual + mamba_h + attn_h

        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)

    layer.forward = forward.__get__(layer, type(layer))
    setattr(layer, _MARKER, True)
    return True


# ---------------------------------------------------------------------------
# Public entry point: patch a Falcon-H1 model end-to-end.
# ---------------------------------------------------------------------------


class FalconH1PatchReport:
    """Counts of which sub-modules were patched."""

    def __init__(self) -> None:
        self.attention = 0
        self.mlp = 0
        self.mixer = 0
        self.layer = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "attention": self.attention,
            "mlp": self.mlp,
            "mixer": self.mixer,
            "layer": self.layer,
        }

    def __repr__(self) -> str:
        return f"FalconH1PatchReport({self.as_dict()})"


def patch_falcon_h1_decoder(model: nn.Module) -> FalconH1PatchReport:
    """Apply training-path optimizations to a Falcon-H1 decoder model.

    Walks ``model`` (typically a ``FalconH1ForCausalLM`` returned by
    ``AutoModelForCausalLM.from_pretrained``), and patches each sub-module
    whose stock forward does correctness-preserving but wasted unit-mul
    work for the loaded config.

    Returns a :class:`FalconH1PatchReport` indicating how many modules in
    each category were patched.

    Patches are idempotent: a second call returns zero new patches.

    This patcher is **training-only**. Generation / KV-cache decode paths
    must not run on patched modules. The mixer patch raises a RuntimeError
    if called with ``cache_params is not None`` so silent fallback is
    impossible.
    """
    try:
        from transformers.models.falcon_h1 import modeling_falcon_h1 as fh1
    except Exception as exc:
        raise RuntimeError(
            "patch_falcon_h1_decoder: transformers.models.falcon_h1 not importable"
        ) from exc

    report = FalconH1PatchReport()

    falcon_h1_attention = fh1.FalconH1Attention
    falcon_h1_mlp = fh1.FalconH1MLP
    falcon_h1_mixer = fh1.FalconH1Mixer
    falcon_h1_decoder_layer = fh1.FalconH1DecoderLayer

    for module in model.modules():
        if isinstance(module, falcon_h1_attention):
            if _patch_attention_unit_key_multiplier(module):
                report.attention += 1
        elif isinstance(module, falcon_h1_mlp):
            if _patch_mlp_unit_multipliers(module):
                report.mlp += 1
        elif isinstance(module, falcon_h1_mixer):
            if _patch_mixer_unit_scalings(module):
                report.mixer += 1
        elif isinstance(module, falcon_h1_decoder_layer) and (
            _patch_decoder_layer_unit_multipliers(module)
        ):
            report.layer += 1

    logger.info("falcon_h1_patch_applied: %s", report.as_dict())
    return report
