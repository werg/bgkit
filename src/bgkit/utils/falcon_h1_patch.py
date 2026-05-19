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

2. **Permanent packed projection parameters**:
   Falcon Q/K/V and MLP gate/up projections are converted to real packed
   trainable parameters (`qkv_proj`, `gate_up_proj`) instead of concatenating
   weights inside every forward. Legacy checkpoint keys and name-keyed
   optimizer moments are spliced into the packed layout on load/restore.

3. **Experimental direct Falcon-H1 FA4 varlen dispatch**:
   In packed CUDA training with `cu_seq_lens_q/k` metadata, Falcon attention
   can call BgKIT's owned FA4 varlen op directly instead of routing through
   HF's generic `AttentionInterface` / SDPA bridge. This remains opt-in
   (`BGKIT_FALCON_H1_DIRECT_FA4_ATTN=1`) because the native SM12x FA4 path
   currently segfaults on Falcon-H1's packed causal GQA training shape. The
   older lower-level FA2 binding is also opt-in
   (`BGKIT_FALCON_H1_DIRECT_FLASH_ATTN=1`) for the same reason.

   Two additional attention bypasses are available for measurement only:
   `BGKIT_FALCON_H1_DIRECT_HF_FLASH_ATTN=1` calls Transformers'
   FlashAttention wrapper directly, while `BGKIT_FALCON_H1_DIRECT_SDPA=1`
   calls PyTorch SDPA directly only when a real attention mask is present.
   They remain opt-in because the current HF dispatch path is faster on the
   fixed Falcon training profile.

4. **Fused SwiGLU for Falcon-H1 MLPs**:
   When `liger-kernel` is installed, replace the stock
   `up_proj(x) * silu(gate_proj(x))` pair with Liger's fused Triton
   `LigerSiLUMulFunction`. If Liger is unavailable, fall back to the plain
   torch expression after stripping the unit multipliers.

5. **Trainable packed MLP autograd boundary**:
   With packed gate/up projections, use a BgKIT-owned autograd Function by
   default (`BGKIT_FALCON_H1_TRAINABLE_MLP_AUTOGRAD=1`). This keeps gate/up
   and down projections fully trainable while writing the SwiGLU derivative
   directly into the packed gate/up gradient activation.

6. **All-ones `mup_vector` short-circuit**: when `ssm_multipliers` is
   `[1, 1, 1, 1, 1]` the broadcast `projected_states * self.mup_vector`
   is a wasted bf16 copy of a (B, S, 2*intermediate + 2*groups*dstate +
   nheads) tensor. We skip it.

7. **Falcon-H1 Tiny Mamba specialization**:
   The fused Mamba training op always runs the no-nonssm / no-Mamba-RMSNorm /
   fused-out-projection subcase for Falcon-H1-Tiny-90M. We route that exact
   shape through a BgKIT-owned autograd wrapper that reuses the upstream
   causal-conv1d and SSD Triton kernels but strips the generic Mamba2 branches.
   By default that wrapper saves the pre-out-projection Mamba activation for
   backward (`BGKIT_FALCON_H1_MAMBA_SAVE_OUT=1`), trading memory for less
   recomputation in the trainable-decoder path. It also saves the causal-conv
   output (`BGKIT_FALCON_H1_MAMBA_SAVE_CONV=1`) so backward does not replay
   conv1d, saves SSD scan intermediates
   (`BGKIT_FALCON_H1_MAMBA_SAVE_SCAN=1`) so backward does not rebuild chunk
   cumsums / chunk states / `CB`, and owns Mamba `in_proj` gradients
   (`BGKIT_FALCON_H1_MAMBA_INPROJ_AUTOGRAD=1`) by default. The backward path
   also applies the simple `D * dout` residual outside the generic SSD `dx`
   kernel by default (`BGKIT_FALCON_H1_MAMBA_SKIP_D_IN_DX_KERNEL=1`), which is
   faster on the fixed Falcon-H1 Tiny training profile.

8. **Tight `FalconH1Model.forward`**: replaces HF's stock loop. The stock
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

9. **Experimental fused Falcon layer input projection**:
   With `BGKIT_FALCON_H1_FUSED_INPUT_PROJ=1`, each decoder layer computes
   attention QKV and Mamba `in_proj` from the shared normalized hidden state
   with one wide `F.linear` and splits the result before calling the attention
   and Mamba submodules. This keeps the existing trainable parameter names
   (`self_attn.qkv_proj`, `mamba.in_proj`) through `torch.cat` autograd, so
   checkpoints and name-keyed optimizer state remain compatible while testing
   whether a permanent packed layer input projection is worth the migration.

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
import os
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def _try_import_liger_silu_mul():
    try:
        from liger_kernel.ops import LigerSiLUMulFunction  # type: ignore

        return LigerSiLUMulFunction
    except Exception:
        return None


def _try_import_liger_rmsnorm_fn():
    try:
        from liger_kernel.ops import LigerRMSNormFunction  # type: ignore

        return LigerRMSNormFunction
    except Exception:
        return None


def _try_import_liger_fused_add_rmsnorm_fn():
    try:
        from liger_kernel.ops import LigerFusedAddRMSNormFunction  # type: ignore

        return LigerFusedAddRMSNormFunction
    except Exception:
        return None


def _try_import_falcon_h1_mamba_specialized():
    try:
        from bgkit.kernels.falcon_h1_mamba import (
            falcon_h1_mamba_split_conv1d_scan_combined,
        )

        return falcon_h1_mamba_split_conv1d_scan_combined
    except Exception:
        return None


def _try_import_falcon_h1_trainable_mlp():
    try:
        from bgkit.kernels.falcon_h1_mlp import falcon_h1_packed_mlp_trainable

        return falcon_h1_packed_mlp_trainable
    except Exception:
        return None


def _try_import_flash_attn_varlen_func():
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func

        return flash_attn_varlen_func
    except Exception:
        return None


def _try_import_bgkit_fa4_attention_forward():
    try:
        from bgkit.utils.attention_backend import bgkit_flash_attention_4_forward

        return bgkit_flash_attention_4_forward
    except Exception:
        return None


def _try_import_hf_flash_attention_forward():
    try:
        from transformers.integrations.flash_attention import flash_attention_forward

        return flash_attention_forward
    except Exception:
        return None


def _direct_sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    dropout: float,
    scaling: float | None,
) -> tuple[torch.Tensor, None]:
    """Falcon-H1 direct SDPA equivalent of HF's sdpa_attention_forward."""

    sdpa_kwargs: dict[str, bool] = {}
    n_kv_groups = int(getattr(module, "num_key_value_groups", 1))
    if n_kv_groups != 1:
        if attention_mask is None:
            sdpa_kwargs["enable_gqa"] = True
        else:
            key = key.repeat_interleave(n_kv_groups, dim=1)
            value = value.repeat_interleave(n_kv_groups, dim=1)

    is_causal = bool(
        query.shape[2] > 1
        and attention_mask is None
        and getattr(module, "is_causal", True)
    )
    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )
    return attn_output.transpose(1, 2).contiguous(), None


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


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
    use_cat_qkv = os.environ.get(
        "BGKIT_FALCON_H1_ATTENTION_CAT_QKV", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    use_packed_qkv = os.environ.get(
        "BGKIT_FALCON_H1_PACKED_QKV", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    direct_flash_attn_fn = (
        _try_import_flash_attn_varlen_func()
        if _env_truthy("BGKIT_FALCON_H1_DIRECT_FLASH_ATTN", "0")
        else None
    )
    direct_fa4_attn_fn = (
        _try_import_bgkit_fa4_attention_forward()
        if _env_truthy("BGKIT_FALCON_H1_DIRECT_FA4_ATTN", "0")
        else None
    )
    direct_hf_flash_attn_fn = (
        _try_import_hf_flash_attention_forward()
        if _env_truthy("BGKIT_FALCON_H1_DIRECT_HF_FLASH_ATTN", "0")
        else None
    )
    use_direct_sdpa = _env_truthy("BGKIT_FALCON_H1_DIRECT_SDPA", "0")

    if (
        use_packed_qkv
        and isinstance(getattr(attn, "q_proj", None), nn.Linear)
        and isinstance(getattr(attn, "k_proj", None), nn.Linear)
        and isinstance(getattr(attn, "v_proj", None), nn.Linear)
    ):
        q_proj = attn.q_proj
        k_proj = attn.k_proj
        v_proj = attn.v_proj
        if (
            q_proj.in_features == k_proj.in_features == v_proj.in_features
            and ((q_proj.bias is None) == (k_proj.bias is None) == (v_proj.bias is None))
        ):
            qkv_proj = nn.Linear(
                q_proj.in_features,
                q_proj.out_features + k_proj.out_features + v_proj.out_features,
                bias=q_proj.bias is not None,
                device=q_proj.weight.device,
                dtype=q_proj.weight.dtype,
            )
            with torch.no_grad():
                qkv_proj.weight.copy_(
                    torch.cat((q_proj.weight, k_proj.weight, v_proj.weight), dim=0)
                )
                if qkv_proj.bias is not None:
                    qkv_proj.bias.copy_(
                        torch.cat((q_proj.bias, k_proj.bias, v_proj.bias), dim=0)
                    )
            attn._bgkit_qkv_split_sizes = (  # type: ignore[attr-defined]
                q_proj.out_features,
                k_proj.out_features,
                v_proj.out_features,
            )
            del attn.q_proj
            del attn.k_proj
            del attn.v_proj
            attn.qkv_proj = qkv_proj  # type: ignore[attr-defined]
            attn._bgkit_packed_qkv = True  # type: ignore[attr-defined]

            def load_packed_qkv(
                module,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
                packed_w = f"{prefix}qkv_proj.weight"
                q_w = f"{prefix}q_proj.weight"
                k_w = f"{prefix}k_proj.weight"
                v_w = f"{prefix}v_proj.weight"
                if (
                    packed_w not in state_dict
                    and q_w in state_dict
                    and k_w in state_dict
                    and v_w in state_dict
                ):
                    state_dict[packed_w] = torch.cat(
                        (
                            state_dict.pop(q_w),
                            state_dict.pop(k_w),
                            state_dict.pop(v_w),
                        ),
                        dim=0,
                    )
                else:
                    state_dict.pop(q_w, None)
                    state_dict.pop(k_w, None)
                    state_dict.pop(v_w, None)

                packed_b = f"{prefix}qkv_proj.bias"
                q_b = f"{prefix}q_proj.bias"
                k_b = f"{prefix}k_proj.bias"
                v_b = f"{prefix}v_proj.bias"
                if (
                    packed_b not in state_dict
                    and q_b in state_dict
                    and k_b in state_dict
                    and v_b in state_dict
                ):
                    state_dict[packed_b] = torch.cat(
                        (
                            state_dict.pop(q_b),
                            state_dict.pop(k_b),
                            state_dict.pop(v_b),
                        ),
                        dim=0,
                    )
                else:
                    state_dict.pop(q_b, None)
                    state_dict.pop(k_b, None)
                    state_dict.pop(v_b, None)

            attn.register_load_state_dict_pre_hook(load_packed_qkv)

    def forward(
        self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        precomputed_qkv = kwargs.pop("bgkit_precomputed_qkv", None)

        if precomputed_qkv is not None:
            query_states, key_states, value_states = precomputed_qkv.split(
                self._bgkit_qkv_split_sizes,
                dim=-1,
            )
        elif getattr(self, "_bgkit_packed_qkv", False):
            qkv = self.qkv_proj(hidden_states)
            query_states, key_states, value_states = qkv.split(
                self._bgkit_qkv_split_sizes,
                dim=-1,
            )
        elif use_cat_qkv and hidden_states.is_cuda:
            qkv_weight = torch.cat(
                (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight),
                dim=0,
            )
            qkv_bias = None
            if (
                self.q_proj.bias is not None
                or self.k_proj.bias is not None
                or self.v_proj.bias is not None
            ):
                if (
                    self.q_proj.bias is None
                    or self.k_proj.bias is None
                    or self.v_proj.bias is None
                ):
                    query_states = self.q_proj(hidden_states)
                    key_states = self.k_proj(hidden_states)
                    value_states = self.v_proj(hidden_states)
                else:
                    qkv_bias = torch.cat(
                        (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias),
                        dim=0,
                    )
                    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
                    query_states, key_states, value_states = qkv.split(
                        (
                            self.q_proj.out_features,
                            self.k_proj.out_features,
                            self.v_proj.out_features,
                        ),
                        dim=-1,
                    )
            else:
                qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
                query_states, key_states, value_states = qkv.split(
                    (
                        self.q_proj.out_features,
                        self.k_proj.out_features,
                        self.v_proj.out_features,
                    ),
                    dim=-1,
                )
            self._bgkit_cat_qkv = True  # type: ignore[attr-defined]
        else:
            query_states = self.q_proj(hidden_states)
            # NOTE: key_multiplier == 1.0; mul dropped.
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attn_output = None
        attn_weights = None
        cu_q = kwargs.get("cu_seq_lens_q")
        cu_k = kwargs.get("cu_seq_lens_k")
        max_q = kwargs.get("max_length_q")
        max_k = kwargs.get("max_length_k")
        dropout_p = 0.0 if not self.training else float(self.attention_dropout)
        if (
            direct_fa4_attn_fn is not None
            and query_states.is_cuda
            and query_states.dim() == 4
            and query_states.shape[0] == 1
            and cu_q is not None
            and cu_k is not None
            and max_q is not None
            and max_k is not None
            and past_key_values is None
        ):
            attn_output, attn_weights = direct_fa4_attn_fn(
                module=self,
                query=query_states,
                key=key_states,
                value=value_states,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=max_q,
                max_seqlen_k=max_k,
                is_causal=True,
                scale=self.scaling,
                pack_gqa=False,
            )
            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]
            self._bgkit_direct_fa4_attn_used = True  # type: ignore[attr-defined]
        elif (
            direct_flash_attn_fn is not None
            and query_states.is_cuda
            and query_states.dim() == 4
            and query_states.shape[0] == 1
            and cu_q is not None
            and cu_k is not None
            and max_q is not None
            and max_k is not None
            and past_key_values is None
        ):
            query_3d = query_states.squeeze(0).transpose(0, 1).contiguous()
            key_3d = key_states.squeeze(0).transpose(0, 1).contiguous()
            value_3d = value_states.squeeze(0).transpose(0, 1).contiguous()
            attn_output = direct_flash_attn_fn(
                query_3d,
                key_3d,
                value_3d,
                cu_q,
                cu_k,
                int(max_q),
                int(max_k),
                dropout_p=dropout_p,
                softmax_scale=self.scaling,
                causal=True,
            )
            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]
            self._bgkit_direct_flash_attn_used = True  # type: ignore[attr-defined]

        if (
            attn_output is None
            and direct_hf_flash_attn_fn is not None
            and str(getattr(self.config, "_attn_implementation", "")).startswith(
                "flash_attention"
            )
        ):
            attn_output, attn_weights = direct_hf_flash_attn_fn(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=dropout_p,
                scaling=self.scaling,
                **kwargs,
            )
            self._bgkit_direct_hf_flash_attn_used = True  # type: ignore[attr-defined]

        if attn_output is None and use_direct_sdpa and (attention_mask is not None or cu_q is None):
            attn_output, attn_weights = _direct_sdpa_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=dropout_p,
                scaling=self.scaling,
            )
            self._bgkit_direct_sdpa_attn_used = True  # type: ignore[attr-defined]

        if attn_output is None:
            attention_interface: Callable = all_attention_functions.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=dropout_p,
                scaling=self.scaling,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    attn.forward = forward.__get__(attn, type(attn))
    if getattr(attn, "_bgkit_packed_qkv", False):
        attn._bgkit_packed_qkv_enabled = True  # type: ignore[attr-defined]
    if use_cat_qkv:
        attn._bgkit_cat_qkv_enabled = True  # type: ignore[attr-defined]
    if direct_fa4_attn_fn is not None:
        attn._bgkit_direct_fa4_attn_enabled = True  # type: ignore[attr-defined]
    if direct_flash_attn_fn is not None:
        attn._bgkit_direct_flash_attn_enabled = True  # type: ignore[attr-defined]
    if direct_hf_flash_attn_fn is not None:
        attn._bgkit_direct_hf_flash_attn_enabled = True  # type: ignore[attr-defined]
    if use_direct_sdpa:
        attn._bgkit_direct_sdpa_attn_enabled = True  # type: ignore[attr-defined]
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

    use_packed_gate_up = _env_truthy("BGKIT_FALCON_H1_PACKED_MLP", "1")
    use_cat_gate_up = _env_truthy("BGKIT_FALCON_H1_MLP_CAT_GATE_UP", "0")
    use_trainable_mlp_autograd = _env_truthy(
        "BGKIT_FALCON_H1_TRAINABLE_MLP_AUTOGRAD",
        "1",
    )
    silu_mul_fn = _try_import_liger_silu_mul()
    trainable_packed_mlp_fn = (
        _try_import_falcon_h1_trainable_mlp() if use_trainable_mlp_autograd else None
    )

    if use_packed_gate_up and hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
        gate_proj = mlp.gate_proj
        up_proj = mlp.up_proj
        if (
            isinstance(gate_proj, nn.Linear)
            and isinstance(up_proj, nn.Linear)
            and gate_proj.in_features == up_proj.in_features
            and gate_proj.out_features == up_proj.out_features
            and ((gate_proj.bias is None) == (up_proj.bias is None))
        ):
            gate_up_proj = nn.Linear(
                gate_proj.in_features,
                gate_proj.out_features + up_proj.out_features,
                bias=gate_proj.bias is not None,
                device=gate_proj.weight.device,
                dtype=gate_proj.weight.dtype,
            )
            with torch.no_grad():
                gate_up_proj.weight.copy_(
                    torch.cat((gate_proj.weight, up_proj.weight), dim=0)
                )
                if gate_up_proj.bias is not None:
                    gate_up_proj.bias.copy_(
                        torch.cat((gate_proj.bias, up_proj.bias), dim=0)
                    )
            del mlp.gate_proj
            del mlp.up_proj
            mlp.gate_up_proj = gate_up_proj  # type: ignore[attr-defined]
            mlp._bgkit_packed_gate_up = True  # type: ignore[attr-defined]

            def load_packed_gate_up(
                module,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
                packed_w = f"{prefix}gate_up_proj.weight"
                gate_w = f"{prefix}gate_proj.weight"
                up_w = f"{prefix}up_proj.weight"
                if packed_w not in state_dict and gate_w in state_dict and up_w in state_dict:
                    state_dict[packed_w] = torch.cat(
                        (state_dict.pop(gate_w), state_dict.pop(up_w)),
                        dim=0,
                    )
                else:
                    state_dict.pop(gate_w, None)
                    state_dict.pop(up_w, None)

                packed_b = f"{prefix}gate_up_proj.bias"
                gate_b = f"{prefix}gate_proj.bias"
                up_b = f"{prefix}up_proj.bias"
                if packed_b not in state_dict and gate_b in state_dict and up_b in state_dict:
                    state_dict[packed_b] = torch.cat(
                        (state_dict.pop(gate_b), state_dict.pop(up_b)),
                        dim=0,
                    )
                else:
                    state_dict.pop(gate_b, None)
                    state_dict.pop(up_b, None)

            mlp.register_load_state_dict_pre_hook(load_packed_gate_up)

    if silu_mul_fn is not None:

        def forward(self, x):
            # Drop unit gate/down multipliers and fuse silu(gate) * up.
            if getattr(self, "_bgkit_packed_gate_up", False):
                if trainable_packed_mlp_fn is not None:
                    return trainable_packed_mlp_fn(
                        x,
                        self.gate_up_proj.weight,
                        self.gate_up_proj.bias,
                        self.down_proj.weight,
                        self.down_proj.bias,
                    )
                gate_up = self.gate_up_proj(x)
                gate, up = gate_up.split(self.intermediate_size, dim=-1)
            elif use_cat_gate_up and x.is_cuda:
                gate_up_weight = torch.cat(
                    (self.gate_proj.weight, self.up_proj.weight),
                    dim=0,
                )
                gate_up_bias = None
                if self.gate_proj.bias is not None or self.up_proj.bias is not None:
                    if self.gate_proj.bias is None or self.up_proj.bias is None:
                        return self.down_proj(
                            silu_mul_fn.apply(self.gate_proj(x), self.up_proj(x))
                        )
                    gate_up_bias = torch.cat(
                        (self.gate_proj.bias, self.up_proj.bias),
                        dim=0,
                    )
                gate_up = F.linear(x, gate_up_weight, gate_up_bias)
                gate, up = gate_up.split(self.gate_proj.out_features, dim=-1)
                self._bgkit_cat_gate_up = True  # type: ignore[attr-defined]
            else:
                gate = self.gate_proj(x)
                up = self.up_proj(x)
            if x.is_cuda:
                return self.down_proj(silu_mul_fn.apply(gate, up))
            return self.down_proj(up * self.act_fn(gate))

        mlp._bgkit_liger_silu_mul = True  # type: ignore[attr-defined]
    else:

        def forward(self, x):
            # Drop: self.gate_proj(x) * self.gate_multiplier
            # Drop: self.down_proj(y) * self.down_multiplier
            if getattr(self, "_bgkit_packed_gate_up", False):
                if trainable_packed_mlp_fn is not None:
                    return trainable_packed_mlp_fn(
                        x,
                        self.gate_up_proj.weight,
                        self.gate_up_proj.bias,
                        self.down_proj.weight,
                        self.down_proj.bias,
                    )
                gate_up = self.gate_up_proj(x)
                gate, up = gate_up.split(self.intermediate_size, dim=-1)
                return self.down_proj(up * self.act_fn(gate))
            if use_cat_gate_up and x.is_cuda:
                gate_up_weight = torch.cat(
                    (self.gate_proj.weight, self.up_proj.weight),
                    dim=0,
                )
                gate_up_bias = None
                if self.gate_proj.bias is not None or self.up_proj.bias is not None:
                    if self.gate_proj.bias is None or self.up_proj.bias is None:
                        return self.down_proj(
                            self.up_proj(x) * self.act_fn(self.gate_proj(x))
                        )
                    gate_up_bias = torch.cat(
                        (self.gate_proj.bias, self.up_proj.bias),
                        dim=0,
                    )
                gate_up = F.linear(x, gate_up_weight, gate_up_bias)
                gate, up = gate_up.split(self.gate_proj.out_features, dim=-1)
                self._bgkit_cat_gate_up = True  # type: ignore[attr-defined]
                return self.down_proj(up * self.act_fn(gate))
            return self.down_proj(self.up_proj(x) * self.act_fn(self.gate_proj(x)))

    if use_cat_gate_up:
        mlp._bgkit_cat_gate_up_enabled = True  # type: ignore[attr-defined]
    if trainable_packed_mlp_fn is not None and getattr(mlp, "_bgkit_packed_gate_up", False):
        mlp._bgkit_trainable_mlp_autograd = True  # type: ignore[attr-defined]

    mlp.forward = forward.__get__(mlp, type(mlp))
    setattr(mlp, _MARKER, True)
    return True


# ---------------------------------------------------------------------------
# 3. Plain RMSNorm forward: fuse Falcon-H1 layer norms on CUDA
# ---------------------------------------------------------------------------


def _patch_rmsnorm_liger(rmsnorm: nn.Module) -> bool:
    if getattr(rmsnorm, _MARKER, False):
        return False
    rmsnorm_fn = _try_import_liger_rmsnorm_fn()
    if rmsnorm_fn is None:
        return False
    if not hasattr(rmsnorm, "weight") or not hasattr(rmsnorm, "variance_epsilon"):
        return False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.is_cuda:
            return rmsnorm_fn.apply(
                hidden_states,
                self.weight,
                self.variance_epsilon,
                0.0,
                "llama",
                False,
                None,
            )
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    rmsnorm.forward = forward.__get__(rmsnorm, type(rmsnorm))
    rmsnorm._bgkit_liger_rmsnorm = True  # type: ignore[attr-defined]
    setattr(rmsnorm, _MARKER, True)
    return True


# ---------------------------------------------------------------------------
# 4. Mixer cuda_kernels_forward: drop ssm_in_multiplier + all-ones mup_vector
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
    specialized_mamba_fn = _try_import_falcon_h1_mamba_specialized()
    use_specialized_mamba = (
        specialized_mamba_fn is not None
        and os.environ.get("BGKIT_FALCON_H1_SPECIALIZED_MAMBA", "1").strip().lower()
        not in {"0", "false", "no", "off"}
        and int(getattr(mixer, "n_groups", -1)) == 1
        and int(getattr(mixer, "intermediate_size", -1))
        == int(getattr(mixer, "num_heads", 0)) * int(getattr(mixer, "head_dim", 0))
        and not bool(getattr(mixer, "mamba_rms_norm", False))
        and str(getattr(mixer, "activation", "")) in {"silu", "swish"}
    )
    use_mamba_inproj_autograd = use_specialized_mamba and _env_truthy(
        "BGKIT_FALCON_H1_MAMBA_INPROJ_AUTOGRAD",
        "1",
    )
    use_mamba_save_scan = use_specialized_mamba and _env_truthy(
        "BGKIT_FALCON_H1_MAMBA_SAVE_SCAN",
        "1",
    )

    # Snapshot the stock forward so the slow / cache branches stay
    # correctness-equivalent.
    stock_forward = type(mixer).forward

    # ----- Patched fused-training fast path -----
    # Inline equivalent of: stock cuda_kernels_forward with `self.training and
    # cache_params is None` branch, but with the unit ssm_in_multiplier mul
    # and the all-ones mup_vector broadcast mul removed. All other ops are
    # identical to HF.
    def patched_training_fused_path(
        self,
        hidden_states,
        attention_mask,
        seq_idx=None,
        precomputed_zxbcdt=None,
    ):
        # 1. Gated MLP linear projection (mask + in_proj). No ssm_in_multiplier.
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        if precomputed_zxbcdt is not None:
            projected_states = precomputed_zxbcdt
            local_use_mamba_inproj_autograd = False
        else:
            local_use_mamba_inproj_autograd = use_mamba_inproj_autograd
            projected_states = (
                None
                if local_use_mamba_inproj_autograd
                else self.in_proj(hidden_states)
            )
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

        if use_specialized_mamba:
            self._bgkit_specialized_mamba_used = True  # type: ignore[attr-defined]
            return specialized_mamba_fn(
                mamba_split_conv1d_scan_combined,
                hidden_states if local_use_mamba_inproj_autograd else projected_states,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.dt_bias,
                a_log_exp,
                self.D,
                self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.head_dim,
                inproj_weight=self.in_proj.weight if local_use_mamba_inproj_autograd else None,
                inproj_bias=self.in_proj.bias if local_use_mamba_inproj_autograd else None,
                **dt_limit_kwargs,
            )
        assert projected_states is not None
        return mamba_split_conv1d_scan_combined(
            projected_states,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.dt_bias,
            a_log_exp,
            D=self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
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
        seq_idx = kwargs.pop("seq_idx", None)
        precomputed_zxbcdt = kwargs.pop("bgkit_precomputed_zxbcdt", None)
        # Mirror stock dispatch logic in `FalconH1Mixer.forward`:
        if (
            is_fast_path_available
            and "cuda" in self.in_proj.weight.device.type
            and not is_torchdynamo_compiling()
            and self.training
            and cache_params is None
        ):
            return patched_training_fused_path(
                self,
                hidden_states,
                attention_mask,
                seq_idx,
                precomputed_zxbcdt,
            )
        # Everything else: route through the stock forward (which itself dispatches
        # to cuda_kernels_forward or torch_forward as appropriate).
        return stock_forward(self, hidden_states, cache_params, attention_mask, **kwargs)

    mixer.forward = forward.__get__(mixer, type(mixer))
    if use_specialized_mamba:
        mixer._bgkit_specialized_mamba_enabled = True  # type: ignore[attr-defined]
    if use_mamba_save_scan:
        mixer._bgkit_mamba_save_scan = True  # type: ignore[attr-defined]
    if use_mamba_inproj_autograd:
        mixer._bgkit_mamba_inproj_autograd = True  # type: ignore[attr-defined]
    setattr(mixer, _MARKER, True)
    return True


# ---------------------------------------------------------------------------
# 5. Decoder layer forward: drop ssm_out_multiplier / attention_in_multiplier /
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

    use_fused_input_proj = _env_truthy("BGKIT_FALCON_H1_FUSED_INPUT_PROJ", "0")

    def _fused_input_projection(self, hidden_states):
        if not use_fused_input_proj or not hidden_states.is_cuda:
            return None, None
        attn = self.self_attn
        mamba = self.mamba
        if not (
            getattr(attn, "_bgkit_packed_qkv", False)
            and hasattr(attn, "qkv_proj")
            and hasattr(mamba, "in_proj")
        ):
            return None, None
        qkv_proj = attn.qkv_proj
        mamba_proj = mamba.in_proj
        if not (
            isinstance(qkv_proj, nn.Linear)
            and isinstance(mamba_proj, nn.Linear)
            and qkv_proj.in_features == mamba_proj.in_features
            and ((qkv_proj.bias is None) == (mamba_proj.bias is None))
        ):
            return None, None

        fused_weight = torch.cat((qkv_proj.weight, mamba_proj.weight), dim=0)
        fused_bias = (
            None
            if qkv_proj.bias is None
            else torch.cat((qkv_proj.bias, mamba_proj.bias), dim=0)
        )
        fused = F.linear(hidden_states, fused_weight, fused_bias)
        qkv_dim = int(qkv_proj.out_features)
        self._bgkit_fused_input_proj_used = True  # type: ignore[attr-defined]
        return fused[..., :qkv_dim], fused[..., qkv_dim:]

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
        mamba_seq_idx = kwargs.pop("mamba_seq_idx", None)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        precomputed_qkv, precomputed_zxbcdt = _fused_input_projection(self, hidden_states)

        mamba_h = self.mamba(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=mamba_attention_mask,
            seq_idx=mamba_seq_idx,
            bgkit_precomputed_zxbcdt=precomputed_zxbcdt,
        )
        # Drop: mamba_h * self.ssm_out_multiplier

        attn_h, _ = self.self_attn(
            hidden_states=hidden_states,  # was * self.attention_in_multiplier
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            bgkit_precomputed_qkv=precomputed_qkv,
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
    if use_fused_input_proj:
        layer._bgkit_fused_input_proj_enabled = True  # type: ignore[attr-defined]
    setattr(layer, _MARKER, True)
    return True


# ---------------------------------------------------------------------------
# 6. FalconH1Model forward: thread packed Mamba seq_idx through the layer loop.
# ---------------------------------------------------------------------------


def _patch_model_packed_seqidx_loop(model: nn.Module) -> bool:
    """Patch FalconH1Model.forward to pass ``mamba_seq_idx`` into each layer.

    BgKIT's packed decoder path supplies one flat ``(1, N, D)`` sequence plus
    per-sample boundary metadata. Mamba's combined scan already supports a
    ``seq_idx`` tensor to reset recurrent state at sample boundaries; HF's
    public Falcon-H1 model forward simply does not thread that argument down
    to the mixer. This loop mirrors the stock model forward and adds that one
    missing argument.
    """
    if getattr(model, _MARKER, False):
        return False

    try:
        from transformers.modeling_outputs import BaseModelOutputWithPast
        from transformers.models.falcon_h1 import modeling_falcon_h1 as fh1
    except Exception:
        return False

    stock_forward = type(model).forward
    create_causal_mask = fh1.create_causal_mask
    dynamic_cache = fh1.DynamicCache

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ):
        mamba_seq_idx = kwargs.pop("mamba_seq_idx", None)
        if mamba_seq_idx is None or use_cache or past_key_values is not None:
            return stock_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embedding_multiplier
        hidden_states = inputs_embeds

        if use_cache and past_key_values is None:
            past_key_values = dynamic_cache(config=self.config)

        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = (
                torch.arange(hidden_states.shape[1], device=hidden_states.device)
                + past_seen_tokens
            )
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        mamba_mask = self._update_mamba_mask(attention_mask, past_key_values)
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                mamba_attention_mask=mamba_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                mamba_seq_idx=mamba_seq_idx,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.final_layernorm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    model.forward = forward.__get__(model, type(model))
    model._bgkit_falcon_h1_packed_seqidx_loop = True  # type: ignore[attr-defined]
    setattr(model, _MARKER, True)
    return True


# ---------------------------------------------------------------------------
# 7. FalconH1Model forward: own the layer loop and fuse residual-add RMSNorms.
# ---------------------------------------------------------------------------


def _patch_model_fused_training_loop(model: nn.Module) -> bool:
    """Patch FalconH1Model.forward for full-sequence training.

    The stock layer boundary materializes two hot patterns every layer:

      1. ``raw_mid = residual + (mamba_h + attn_h); pre_ff_norm(raw_mid)``
      2. ``raw_next = raw_mid + mlp_h; next_input_norm(raw_next)``

    Liger's fused-add RMSNorm autograd kernel exactly matches these boundaries.
    We therefore bypass ``FalconH1DecoderLayer.forward`` in training mode and
    run the layer loop directly. KV-cache/generation paths fall back to the
    installed HF forward.
    """
    if getattr(model, _MARKER, False):
        return False
    if os.environ.get("BGKIT_FALCON_H1_FUSED_LAYER_LOOP", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False

    fused_add_rmsnorm_fn = _try_import_liger_fused_add_rmsnorm_fn()
    if fused_add_rmsnorm_fn is None:
        return False

    try:
        from transformers.modeling_outputs import BaseModelOutputWithPast
        from transformers.models.falcon_h1 import modeling_falcon_h1 as fh1
    except Exception:
        return False

    stock_forward = type(model).forward
    create_causal_mask = fh1.create_causal_mask

    def _rmsnorm_eps(norm_module: nn.Module) -> float:
        return float(norm_module.variance_epsilon)

    def _fused_add_norm(x: torch.Tensor, residual: torch.Tensor, norm_module: nn.Module):
        return fused_add_rmsnorm_fn.apply(
            x,
            residual,
            norm_module.weight,
            _rmsnorm_eps(norm_module),
            0.0,
            "llama",
            False,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ):
        mamba_seq_idx = kwargs.pop("mamba_seq_idx", None)
        if use_cache or past_key_values is not None:
            return stock_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )
        if not self.training:
            return stock_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embedding_multiplier
        if not inputs_embeds.is_cuda:
            return stock_forward(
                self,
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                **kwargs,
            )

        raw_hidden = inputs_embeds
        if position_ids is None:
            position_ids = torch.arange(raw_hidden.shape[1], device=raw_hidden.device)
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        mamba_mask = self._update_mamba_mask(attention_mask, None)
        position_embeddings = self.rotary_emb(raw_hidden, position_ids=position_ids)

        layers = self.layers
        norm_hidden = layers[0].input_layernorm(raw_hidden)
        for layer_idx, decoder_layer in enumerate(layers):
            mamba_hidden = decoder_layer.mamba(
                hidden_states=norm_hidden,
                cache_params=None,
                attention_mask=mamba_mask,
                seq_idx=mamba_seq_idx,
            )
            attention_hidden, _ = decoder_layer.self_attn(
                hidden_states=norm_hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            branch_hidden = mamba_hidden + attention_hidden
            ff_hidden, raw_mid = _fused_add_norm(
                branch_hidden,
                raw_hidden,
                decoder_layer.pre_ff_layernorm,
            )
            mlp_hidden = decoder_layer.feed_forward(ff_hidden)

            next_idx = layer_idx + 1
            if next_idx < len(layers):
                norm_hidden, raw_hidden = _fused_add_norm(
                    mlp_hidden,
                    raw_mid,
                    layers[next_idx].input_layernorm,
                )
            else:
                hidden_states, raw_hidden = _fused_add_norm(
                    mlp_hidden,
                    raw_mid,
                    self.final_layernorm,
                )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
        )

    model.forward = forward.__get__(model, type(model))
    model._bgkit_falcon_h1_fused_training_loop = True  # type: ignore[attr-defined]
    setattr(model, _MARKER, True)
    return True


# ---------------------------------------------------------------------------
# Public entry point: patch a Falcon-H1 model end-to-end.
# ---------------------------------------------------------------------------


class FalconH1PatchReport:
    """Counts of which sub-modules were patched."""

    def __init__(self) -> None:
        self.attention = 0
        self.attention_packed_qkv = 0
        self.attention_cat_qkv = 0
        self.attention_direct_fa4 = 0
        self.attention_direct_flash = 0
        self.attention_direct_hf_flash = 0
        self.attention_direct_sdpa = 0
        self.mlp = 0
        self.mlp_packed_gate_up = 0
        self.mlp_cat_gate_up = 0
        self.mlp_liger = 0
        self.mlp_trainable_autograd = 0
        self.rmsnorm_liger = 0
        self.packed_seqidx_loop = 0
        self.fused_layer_loop = 0
        self.layer_fused_input_proj = 0
        self.mixer = 0
        self.mixer_specialized_mamba = 0
        self.mixer_save_scan = 0
        self.mixer_inproj_autograd = 0
        self.mixer_chunk_size = 0
        self.layer = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "attention": self.attention,
            "attention_packed_qkv": self.attention_packed_qkv,
            "attention_cat_qkv": self.attention_cat_qkv,
            "attention_direct_fa4": self.attention_direct_fa4,
            "attention_direct_flash": self.attention_direct_flash,
            "attention_direct_hf_flash": self.attention_direct_hf_flash,
            "attention_direct_sdpa": self.attention_direct_sdpa,
            "mlp": self.mlp,
            "mlp_packed_gate_up": self.mlp_packed_gate_up,
            "mlp_cat_gate_up": self.mlp_cat_gate_up,
            "mlp_liger": self.mlp_liger,
            "mlp_trainable_autograd": self.mlp_trainable_autograd,
            "rmsnorm_liger": self.rmsnorm_liger,
            "packed_seqidx_loop": self.packed_seqidx_loop,
            "fused_layer_loop": self.fused_layer_loop,
            "layer_fused_input_proj": self.layer_fused_input_proj,
            "mixer": self.mixer,
            "mixer_specialized_mamba": self.mixer_specialized_mamba,
            "mixer_save_scan": self.mixer_save_scan,
            "mixer_inproj_autograd": self.mixer_inproj_autograd,
            "mixer_chunk_size": self.mixer_chunk_size,
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
    falcon_h1_rmsnorm = fh1.FalconH1RMSNorm
    falcon_h1_mixer = fh1.FalconH1Mixer
    falcon_h1_decoder_layer = fh1.FalconH1DecoderLayer
    falcon_h1_model = fh1.FalconH1Model
    chunk_size_override: int | None = None
    raw_chunk_size = os.environ.get("BGKIT_FALCON_H1_MAMBA_CHUNK_SIZE")
    if raw_chunk_size:
        try:
            chunk_size_override = int(raw_chunk_size)
        except ValueError:
            logger.warning(
                "falcon_h1_invalid_mamba_chunk_size_override",
                value=raw_chunk_size,
            )

    for module in model.modules():
        if isinstance(module, falcon_h1_model):
            if _patch_model_fused_training_loop(module):
                report.fused_layer_loop += 1
            elif _patch_model_packed_seqidx_loop(module):
                report.packed_seqidx_loop += 1
        elif isinstance(module, falcon_h1_attention):
            if _patch_attention_unit_key_multiplier(module):
                report.attention += 1
                if getattr(module, "_bgkit_packed_qkv_enabled", False):
                    report.attention_packed_qkv += 1
                if getattr(module, "_bgkit_cat_qkv_enabled", False):
                    report.attention_cat_qkv += 1
                if getattr(module, "_bgkit_direct_fa4_attn_enabled", False):
                    report.attention_direct_fa4 += 1
                if getattr(module, "_bgkit_direct_flash_attn_enabled", False):
                    report.attention_direct_flash += 1
                if getattr(module, "_bgkit_direct_hf_flash_attn_enabled", False):
                    report.attention_direct_hf_flash += 1
                if getattr(module, "_bgkit_direct_sdpa_attn_enabled", False):
                    report.attention_direct_sdpa += 1
        elif isinstance(module, falcon_h1_mlp):
            if _patch_mlp_unit_multipliers(module):
                report.mlp += 1
                if getattr(module, "_bgkit_packed_gate_up", False):
                    report.mlp_packed_gate_up += 1
                if getattr(module, "_bgkit_cat_gate_up_enabled", False):
                    report.mlp_cat_gate_up += 1
                if getattr(module, "_bgkit_liger_silu_mul", False):
                    report.mlp_liger += 1
                if getattr(module, "_bgkit_trainable_mlp_autograd", False):
                    report.mlp_trainable_autograd += 1
        elif isinstance(module, falcon_h1_rmsnorm):
            if _patch_rmsnorm_liger(module):
                report.rmsnorm_liger += 1
        elif isinstance(module, falcon_h1_mixer):
            if chunk_size_override is not None and int(module.chunk_size) != chunk_size_override:
                module.chunk_size = chunk_size_override
                report.mixer_chunk_size += 1
            if _patch_mixer_unit_scalings(module):
                report.mixer += 1
                if getattr(module, "_bgkit_specialized_mamba_enabled", False):
                    report.mixer_specialized_mamba += 1
                if getattr(module, "_bgkit_mamba_save_scan", False):
                    report.mixer_save_scan += 1
                if getattr(module, "_bgkit_mamba_inproj_autograd", False):
                    report.mixer_inproj_autograd += 1
        elif isinstance(module, falcon_h1_decoder_layer) and (
            _patch_decoder_layer_unit_multipliers(module)
        ):
            report.layer += 1
            if getattr(module, "_bgkit_fused_input_proj_enabled", False):
                report.layer_fused_input_proj += 1

    logger.info("falcon_h1_patch_applied: %s", report.as_dict())
    return report
