"""Gradient checkpointing across BgKIT levels, gradient clipping."""

from __future__ import annotations

import os
from typing import Any

import structlog
import torch
import torch.nn as nn
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

logger = structlog.get_logger()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_decoder_layerwise_split(decoder: nn.Module, cfg: Any) -> None:
    """Apply optional decoder split-schedule config to compatible decoders."""

    configure = getattr(decoder, "set_qwen35_layerwise_split", None)
    if configure is None:
        return
    try:
        tcfg = cfg.training
    except AttributeError:
        tcfg = cfg.get("training", {})
    split_cfg = tcfg.get("decoder_layerwise_split", None)
    if split_cfg is None:
        return
    if isinstance(split_cfg, (str, bool)):
        configure(mode=split_cfg)
        logger.info("decoder_layerwise_split_configured", mode=str(split_cfg))
        return
    mode = split_cfg.get("mode", None)
    min_ratio = split_cfg.get("min_ratio", None)
    min_prefix = split_cfg.get("min_prefix", None)
    packed_deltanet = split_cfg.get("packed_deltanet", None)
    configure(
        mode=mode,
        min_ratio=min_ratio,
        min_prefix=min_prefix,
        packed_deltanet=packed_deltanet,
    )
    logger.info(
        "decoder_layerwise_split_configured",
        mode=mode,
        min_ratio=min_ratio,
        min_prefix=min_prefix,
        packed_deltanet=packed_deltanet,
    )


# Megatron-style selective recomputation policy: SAVE matmul + attention outputs,
# RECOMPUTE everything else (layernorms, elementwise, residuals, dropout).
# Reference: Korthikanti et al. 2022 "Reducing Activation Recomputation in
# Large Transformer Models" — recomputing softmax + dropout is ~5% of FLOPs but
# saves the O(N²) attention matrix; storing matmul outputs avoids re-running
# the FLOP-dense projections.
#
# We extend the paper's idea by also dropping LoRA-input saves: LoRA-wrapped
# linears must save `x` for `dA = dy.T @ x`, but if the upstream op is cheap
# (layernorm, residual add), that `x` is recomputed for free along with the
# RECOMPUTE chain. The matmul output we MUST_SAVE is what the next layer needs.
_MEGATRON_SAVE_OPS: set = {
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
    # SDPA backends — saving the attention output avoids re-running attention
    # on backward. FA's own custom autograd Function controls Q/K/V/softmax-stat
    # save/recompute internally; this policy operates one level up.
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
}


def _megatron_policy_fn(ctx, op, *args, **kwargs):
    if op in _MEGATRON_SAVE_OPS:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _megatron_checkpoint_func(forward, *args, **kwargs):
    """Drop-in replacement for ``functools.partial(checkpoint, use_reentrant=False)``
    that selectively saves matmul + SDPA outputs and recomputes everything else.

    HF v5's ``GradientCheckpointingLayer.__call__`` invokes this as
    ``self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)``,
    so the signature must match: first arg is the forward callable, remaining
    positional args are the layer's positional inputs.
    """
    context_fn = lambda: create_selective_checkpoint_contexts(_megatron_policy_fn)  # noqa: E731
    return checkpoint(
        forward, *args, use_reentrant=False, context_fn=context_fn, **kwargs,
    )


def enable_gradient_checkpointing(model: nn.Module) -> None:
    """Enable gradient checkpointing on a model if supported.

    Works with HuggingFace models that support gradient_checkpointing_enable().
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def maybe_enable_frozen_decoder_kernels(decoder: nn.Module, cfg: Any) -> dict[str, int]:
    """Install opt-in kernels that rely on the decoder being frozen."""

    try:
        tcfg = cfg.training
    except AttributeError:
        tcfg = cfg.get("training", {})
    freeze_cfg = tcfg.get("freeze", {}) or {}
    decoder_frozen = bool(freeze_cfg.get("decoder", False))
    kernel_cfg = tcfg.get("frozen_decoder_kernels", {}) or {}
    enable_deltanet_core_bwd = bool(
        kernel_cfg.get(
            "deltanet_core_bwd",
            _env_bool("BGKIT_FROZEN_DELTANET_CORE_BWD", False),
        )
    )
    enable_deltanet_residual_bwd = bool(
        kernel_cfg.get(
            "deltanet_residual_bwd",
            _env_bool("BGKIT_FROZEN_DELTANET_RESIDUAL_BWD", False),
        )
    )
    enable_deltanet_residual_mlp_bwd = bool(
        kernel_cfg.get(
            "deltanet_residual_mlp_bwd",
            _env_bool("BGKIT_FROZEN_DELTANET_RESIDUAL_MLP_BWD", False)
            if enable_deltanet_residual_bwd
            else False,
        )
    )
    enable_deltanet_input_rmsnorm_dx = bool(
        kernel_cfg.get(
            "deltanet_input_rmsnorm_dx",
            _env_bool("BGKIT_FROZEN_DELTANET_INPUT_RMSNORM_DX", False)
            if enable_deltanet_core_bwd or enable_deltanet_residual_bwd
            else False,
        )
    )
    enable_channel_last_conv = bool(
        kernel_cfg.get(
            "deltanet_channel_last_conv",
            _env_bool(
                "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV",
                False,
            )
            if not enable_deltanet_core_bwd and not enable_deltanet_residual_bwd
            else False,
        )
    )
    enable_deltanet_stock_fused_qkv_conv_l2norm = bool(
        kernel_cfg.get(
            "deltanet_stock_fused_qkv_conv_l2norm",
            _env_bool("BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_L2NORM", False)
            if enable_channel_last_conv
            else False,
        )
    )
    enable_deltanet_stock_fused_qkv_conv_l2norm_dx = bool(
        kernel_cfg.get(
            "deltanet_stock_fused_qkv_conv_l2norm_dx",
            _env_bool("BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_L2NORM_DX", False)
            if enable_deltanet_stock_fused_qkv_conv_l2norm
            else False,
        )
    )
    enable_deltanet_stock_fused_qkv_conv_split_dx = bool(
        kernel_cfg.get(
            "deltanet_stock_fused_qkv_conv_split_dx",
            _env_bool("BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_SPLIT_DX", False)
            if enable_deltanet_stock_fused_qkv_conv_l2norm
            else False,
        )
    )
    enable_deltanet_fuse_qk_l2norm_bwd = bool(
        kernel_cfg.get(
            "deltanet_fuse_qk_l2norm_bwd",
            _env_bool("FLA_GDR_FUSE_QK_L2NORM_BWD", False)
            if enable_deltanet_stock_fused_qkv_conv_l2norm
            else False,
        )
    )
    enable_deltanet_raw_gate = bool(
        kernel_cfg.get(
            "deltanet_raw_gate_in_kernel",
            _env_bool("BGKIT_DELTANET_RAW_GATE_IN_KERNEL", False),
        )
    )
    enable_deltanet_pair_qk_l2norm_fwd = bool(
        kernel_cfg.get(
            "deltanet_pair_qk_l2norm_fwd",
            _env_bool("FLA_GDR_PAIR_QK_L2NORM_FWD", False),
        )
    )
    enable_attention_qkv = bool(
        kernel_cfg.get(
            "attention_qkv",
            _env_bool("BGKIT_FROZEN_ATTENTION_QKV_FUSION", False),
        )
    )
    enable_deltanet_core_channel_last_conv = bool(
        kernel_cfg.get(
            "deltanet_core_channel_last_conv",
            _env_bool(
                "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV",
                False,
            )
            if enable_deltanet_core_bwd
            else False,
        )
    )
    enable_deltanet_core_channel_last_conv_dx = bool(
        kernel_cfg.get(
            "deltanet_core_channel_last_conv_dx",
            _env_bool(
                "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX",
                False,
            )
            if enable_deltanet_core_channel_last_conv
            else False,
        )
    )
    enable_deltanet_core_fused_qkv_conv_l2norm = bool(
        kernel_cfg.get(
            "deltanet_core_fused_qkv_conv_l2norm",
            _env_bool(
                "BGKIT_FROZEN_DELTANET_FUSED_QKV_CONV_L2NORM",
                False,
            )
            if enable_deltanet_core_channel_last_conv
            else False,
        )
    )
    deltanet_core_min_seq_len = kernel_cfg.get("deltanet_core_min_seq_len", None)
    deltanet_core_max_seq_len = kernel_cfg.get("deltanet_core_max_seq_len", None)
    enable_mlp_swiglu = bool(
        kernel_cfg.get(
            "mlp_swiglu",
            _env_bool("BGKIT_DECODER_MLP_SWIGLU_FUSION", False),
        )
    )
    enable_mlp_swiglu_triton_forward = bool(
        kernel_cfg.get(
            "mlp_swiglu_triton_forward",
            _env_bool("BGKIT_DECODER_MLP_SWIGLU_TRITON_FWD", False),
        )
    )
    enable_mlp_base = bool(
        kernel_cfg.get(
            "mlp_base",
            _env_bool("BGKIT_DECODER_MLP_BASE_FUSION", False),
        )
    )
    mlp_base_dx = kernel_cfg.get("mlp_base_dx", None)
    mlp_base_direct_dx_max_rows = kernel_cfg.get(
        "mlp_base_direct_dx_max_rows",
        None,
    )
    enable_mlp_residual = bool(
        kernel_cfg.get(
            "mlp_residual",
            _env_bool("BGKIT_DECODER_MLP_RESIDUAL_FUSION", False),
        )
    )
    enable_mlp_quack = bool(
        kernel_cfg.get(
            "mlp_quack",
            _env_bool("BGKIT_DECODER_MLP_QUACK_FUSION", False),
        )
    )
    if enable_channel_last_conv and not decoder_frozen:
        raise ValueError(
            "frozen_decoder_kernels.deltanet_channel_last_conv requires "
            "training.freeze.decoder=true."
        )
    if enable_deltanet_stock_fused_qkv_conv_l2norm and not enable_channel_last_conv:
        raise ValueError(
            "frozen_decoder_kernels.deltanet_stock_fused_qkv_conv_l2norm "
            "requires frozen_decoder_kernels.deltanet_channel_last_conv=true."
        )
    if (
        enable_deltanet_stock_fused_qkv_conv_l2norm_dx
        and not enable_deltanet_stock_fused_qkv_conv_l2norm
    ):
        raise ValueError(
            "frozen_decoder_kernels.deltanet_stock_fused_qkv_conv_l2norm_dx "
            "requires frozen_decoder_kernels.deltanet_stock_fused_qkv_conv_l2norm=true."
        )
    if (
        enable_deltanet_stock_fused_qkv_conv_split_dx
        and not enable_deltanet_stock_fused_qkv_conv_l2norm
    ):
        raise ValueError(
            "frozen_decoder_kernels.deltanet_stock_fused_qkv_conv_split_dx "
            "requires frozen_decoder_kernels.deltanet_stock_fused_qkv_conv_l2norm=true."
        )
    if enable_deltanet_fuse_qk_l2norm_bwd and not enable_deltanet_stock_fused_qkv_conv_l2norm:
        raise ValueError(
            "frozen_decoder_kernels.deltanet_fuse_qk_l2norm_bwd requires "
            "frozen_decoder_kernels.deltanet_stock_fused_qkv_conv_l2norm=true."
        )
    if enable_deltanet_core_bwd and not decoder_frozen:
        raise ValueError(
            "frozen_decoder_kernels.deltanet_core_bwd requires "
            "training.freeze.decoder=true."
        )
    if enable_deltanet_residual_bwd and not decoder_frozen:
        raise ValueError(
            "frozen_decoder_kernels.deltanet_residual_bwd requires "
            "training.freeze.decoder=true."
        )
    if enable_deltanet_residual_mlp_bwd and not enable_deltanet_residual_bwd:
        raise ValueError(
            "frozen_decoder_kernels.deltanet_residual_mlp_bwd requires "
            "frozen_decoder_kernels.deltanet_residual_bwd=true."
        )
    if enable_deltanet_input_rmsnorm_dx and not (
        enable_deltanet_core_bwd or enable_deltanet_residual_bwd
    ):
        raise ValueError(
            "frozen_decoder_kernels.deltanet_input_rmsnorm_dx requires "
            "deltanet_core_bwd or deltanet_residual_bwd."
        )
    if enable_deltanet_raw_gate and not decoder_frozen:
        raise ValueError(
            "frozen_decoder_kernels.deltanet_raw_gate_in_kernel requires "
            "training.freeze.decoder=true."
        )
    if enable_deltanet_pair_qk_l2norm_fwd and not decoder_frozen:
        raise ValueError(
            "frozen_decoder_kernels.deltanet_pair_qk_l2norm_fwd requires "
            "training.freeze.decoder=true."
        )
    if enable_attention_qkv and not decoder_frozen:
        raise ValueError(
            "frozen_decoder_kernels.attention_qkv requires "
            "training.freeze.decoder=true."
        )
    if enable_channel_last_conv and enable_deltanet_core_bwd:
        raise ValueError(
            "frozen_decoder_kernels.deltanet_channel_last_conv and "
            "frozen_decoder_kernels.deltanet_core_bwd are mutually exclusive."
        )
    if enable_deltanet_residual_bwd and (
        enable_channel_last_conv or enable_deltanet_core_bwd
    ):
        raise ValueError(
            "frozen_decoder_kernels.deltanet_residual_bwd is mutually exclusive "
            "with deltanet_channel_last_conv and deltanet_core_bwd."
        )
    if enable_deltanet_raw_gate and enable_deltanet_core_bwd:
        raise ValueError(
            "frozen_decoder_kernels.deltanet_raw_gate_in_kernel is only supported "
            "on the stock or channel-last DeltaNet forward; disable deltanet_core_bwd."
        )
    if enable_deltanet_raw_gate and enable_deltanet_residual_bwd:
        raise ValueError(
            "frozen_decoder_kernels.deltanet_raw_gate_in_kernel is only supported "
            "on the stock or channel-last DeltaNet forward; disable "
            "deltanet_residual_bwd."
        )
    if enable_deltanet_core_channel_last_conv and not enable_deltanet_core_bwd:
        raise ValueError(
            "frozen_decoder_kernels.deltanet_core_channel_last_conv requires "
            "frozen_decoder_kernels.deltanet_core_bwd=true."
        )
    if enable_deltanet_core_channel_last_conv_dx and not enable_deltanet_core_channel_last_conv:
        raise ValueError(
            "frozen_decoder_kernels.deltanet_core_channel_last_conv_dx requires "
            "frozen_decoder_kernels.deltanet_core_channel_last_conv=true."
        )
    if enable_deltanet_core_fused_qkv_conv_l2norm and not enable_deltanet_core_channel_last_conv:
        raise ValueError(
            "frozen_decoder_kernels.deltanet_core_fused_qkv_conv_l2norm requires "
            "frozen_decoder_kernels.deltanet_core_channel_last_conv=true."
        )
    if (
        enable_deltanet_core_bwd
        and deltanet_core_min_seq_len is not None
        and int(deltanet_core_min_seq_len) < 0
    ):
        raise ValueError(
            "frozen_decoder_kernels.deltanet_core_min_seq_len must be non-negative."
        )
    if (
        enable_deltanet_core_bwd
        and deltanet_core_max_seq_len is not None
        and int(deltanet_core_max_seq_len) < 0
    ):
        raise ValueError(
            "frozen_decoder_kernels.deltanet_core_max_seq_len must be non-negative."
        )
    if (
        enable_deltanet_core_bwd
        and deltanet_core_min_seq_len is not None
        and deltanet_core_max_seq_len is not None
        and int(deltanet_core_max_seq_len) > 0
        and int(deltanet_core_min_seq_len) > int(deltanet_core_max_seq_len)
    ):
        raise ValueError(
            "frozen_decoder_kernels.deltanet_core_min_seq_len must be <= "
            "deltanet_core_max_seq_len when the max guard is positive."
        )
    if enable_mlp_swiglu and not decoder_frozen:
        raise ValueError(
            "frozen_decoder_kernels.mlp_swiglu requires "
            "training.freeze.decoder=true."
        )
    if enable_mlp_base and not decoder_frozen:
        raise ValueError(
            "frozen_decoder_kernels.mlp_base requires "
            "training.freeze.decoder=true."
        )
    if enable_mlp_base and mlp_base_dx is not None:
        mode = str(mlp_base_dx).strip().lower()
        if mode not in {"cat", "two", "triton", "direct", "adaptive", "down_cat"}:
            raise ValueError(
                "frozen_decoder_kernels.mlp_base_dx must be one of "
                "cat, two, triton, direct, adaptive, or down_cat."
            )
    if (
        enable_mlp_base
        and mlp_base_direct_dx_max_rows is not None
        and int(mlp_base_direct_dx_max_rows) < 1
    ):
        raise ValueError(
            "frozen_decoder_kernels.mlp_base_direct_dx_max_rows must be "
            "positive."
        )
    if sum(
        bool(flag)
        for flag in (
            enable_mlp_swiglu,
            enable_mlp_base,
            enable_mlp_residual,
            enable_mlp_quack,
            enable_deltanet_residual_mlp_bwd,
        )
    ) > 1:
        raise ValueError(
            "frozen decoder MLP kernels are mutually exclusive; enable only one "
            "of mlp_swiglu, mlp_base, mlp_residual, mlp_quack, or "
            "deltanet_residual_mlp_bwd."
        )
    if enable_mlp_residual and not decoder_frozen:
        raise ValueError(
            "frozen_decoder_kernels.mlp_residual requires "
            "training.freeze.decoder=true."
        )
    if enable_mlp_quack and not decoder_frozen:
        raise ValueError(
            "frozen_decoder_kernels.mlp_quack requires "
            "training.freeze.decoder=true."
        )

    counts: dict[str, int] = {}
    if enable_channel_last_conv:
        if enable_deltanet_stock_fused_qkv_conv_l2norm:
            os.environ["BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_L2NORM"] = "1"
            counts["deltanet_stock_fused_qkv_conv_l2norm"] = 1
        if enable_deltanet_stock_fused_qkv_conv_l2norm_dx:
            os.environ["BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_L2NORM_DX"] = "1"
            counts["deltanet_stock_fused_qkv_conv_l2norm_dx"] = 1
        if enable_deltanet_stock_fused_qkv_conv_split_dx:
            os.environ["BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_SPLIT_DX"] = "1"
            counts["deltanet_stock_fused_qkv_conv_split_dx"] = 1
        if enable_deltanet_fuse_qk_l2norm_bwd:
            os.environ["FLA_GDR_FUSE_QK_L2NORM_BWD"] = "1"
            counts["deltanet_fuse_qk_l2norm_bwd"] = 1
        enable = getattr(decoder, "enable_frozen_deltanet_channel_last_conv", None)
        if enable is None:
            logger.warning(
                "frozen_decoder_kernel_unavailable",
                kernel="deltanet_channel_last_conv",
            )
            counts["deltanet_channel_last_conv"] = 0
        else:
            counts["deltanet_channel_last_conv"] = int(enable())
    if enable_deltanet_core_bwd:
        if enable_deltanet_core_channel_last_conv:
            os.environ.setdefault("BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV", "1")
        if enable_deltanet_core_channel_last_conv_dx:
            os.environ.setdefault("BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX", "1")
        if enable_deltanet_core_fused_qkv_conv_l2norm:
            os.environ.setdefault("BGKIT_FROZEN_DELTANET_FUSED_QKV_CONV_L2NORM", "1")
        if deltanet_core_min_seq_len is not None:
            os.environ["BGKIT_FROZEN_DELTANET_CORE_BWD_MIN_SEQ_LEN"] = str(
                int(deltanet_core_min_seq_len)
            )
        if deltanet_core_max_seq_len is not None:
            os.environ["BGKIT_FROZEN_DELTANET_CORE_BWD_MAX_SEQ_LEN"] = str(
                int(deltanet_core_max_seq_len)
            )
        enable = getattr(decoder, "enable_frozen_deltanet_core_bwd", None)
        if enable is None:
            logger.warning(
                "frozen_decoder_kernel_unavailable",
                kernel="deltanet_core_bwd",
            )
            counts["deltanet_core_bwd"] = 0
        else:
            counts["deltanet_core_bwd"] = int(enable())
        if enable_deltanet_core_channel_last_conv:
            counts["deltanet_core_channel_last_conv"] = 1
        if enable_deltanet_core_channel_last_conv_dx:
            counts["deltanet_core_channel_last_conv_dx"] = 1
        if enable_deltanet_core_fused_qkv_conv_l2norm:
            counts["deltanet_core_fused_qkv_conv_l2norm"] = 1
    if enable_deltanet_residual_bwd:
        if enable_deltanet_residual_mlp_bwd:
            os.environ["BGKIT_FROZEN_DELTANET_RESIDUAL_MLP_BWD"] = "1"
            counts["deltanet_residual_mlp_bwd"] = 1
        if enable_deltanet_input_rmsnorm_dx:
            os.environ["BGKIT_FROZEN_DELTANET_INPUT_RMSNORM_DX"] = "1"
            counts["deltanet_input_rmsnorm_dx"] = 1
        enable = getattr(decoder, "enable_frozen_deltanet_residual_bwd", None)
        if enable is None:
            logger.warning(
                "frozen_decoder_kernel_unavailable",
                kernel="deltanet_residual_bwd",
            )
            counts["deltanet_residual_bwd"] = 0
        else:
            counts["deltanet_residual_bwd"] = int(enable())
    elif enable_deltanet_input_rmsnorm_dx:
        os.environ["BGKIT_FROZEN_DELTANET_INPUT_RMSNORM_DX"] = "1"
        counts["deltanet_input_rmsnorm_dx"] = 1
    if enable_deltanet_raw_gate:
        os.environ["BGKIT_DELTANET_RAW_GATE_IN_KERNEL"] = "1"
        counts["deltanet_raw_gate_in_kernel"] = 1
    if enable_deltanet_pair_qk_l2norm_fwd:
        os.environ["FLA_GDR_PAIR_QK_L2NORM_FWD"] = "1"
        counts["deltanet_pair_qk_l2norm_fwd"] = 1
    if enable_attention_qkv:
        enable = getattr(decoder, "enable_fused_attention_qkv", None)
        if enable is None:
            logger.warning(
                "frozen_decoder_kernel_unavailable",
                kernel="attention_qkv",
            )
            counts["attention_qkv"] = 0
        else:
            counts["attention_qkv"] = int(enable())
    if enable_mlp_swiglu:
        enable = getattr(decoder, "enable_frozen_mlp_swiglu_fusion", None)
        if enable is None:
            logger.warning(
                "frozen_decoder_kernel_unavailable",
                kernel="mlp_swiglu",
            )
            counts["mlp_swiglu"] = 0
        else:
            counts["mlp_swiglu"] = int(
                enable(use_triton_forward=enable_mlp_swiglu_triton_forward)
            )
    if enable_mlp_base:
        if mlp_base_dx is not None:
            os.environ["BGKIT_DECODER_MLP_BASE_DX"] = str(mlp_base_dx).strip().lower()
        if mlp_base_direct_dx_max_rows is not None:
            os.environ["BGKIT_DECODER_MLP_DIRECT_DX_MAX_ROWS"] = str(
                int(mlp_base_direct_dx_max_rows)
            )
        enable = getattr(decoder, "enable_frozen_mlp_fusion", None)
        if enable is None:
            logger.warning(
                "frozen_decoder_kernel_unavailable",
                kernel="mlp_base",
            )
            counts["mlp_base"] = 0
        else:
            counts["mlp_base"] = int(enable())
    if enable_mlp_residual:
        enable = getattr(decoder, "enable_frozen_mlp_residual_fusion", None)
        if enable is None:
            logger.warning(
                "frozen_decoder_kernel_unavailable",
                kernel="mlp_residual",
            )
            counts["mlp_residual"] = 0
        else:
            counts["mlp_residual"] = int(enable())
    if enable_mlp_quack:
        os.environ.setdefault("BGKIT_DECODER_MLP_QUACK", "1")
        enable = getattr(decoder, "enable_frozen_mlp_fusion", None)
        if enable is None:
            logger.warning(
                "frozen_decoder_kernel_unavailable",
                kernel="mlp_quack",
            )
            counts["mlp_quack"] = 0
        else:
            counts["mlp_quack"] = int(enable())
    if counts:
        logger.info("frozen_decoder_kernels_enabled", **counts)
    return counts


def gradient_checkpointing_requested(cfg: Any) -> bool:
    """Return True when the config asks for gradient checkpointing.

    Canonical knob is ``cfg.compute.gradient_checkpointing`` (hardware-scoped,
    since grad-ckpt is a memory/compute tradeoff tied to VRAM budget). Legacy
    ``cfg.training.gradient_checkpointing`` is honored for backward compat if
    explicitly set. Default is False because checkpointing trades speed for
    memory and should be an explicit phase-level opt-in.
    """
    tcfg = getattr(cfg, "training", None)
    compute = getattr(cfg, "compute", None)

    # Training-level override wins when explicitly set.
    if tcfg is not None and hasattr(tcfg, "get"):
        training_val = tcfg.get("gradient_checkpointing", None)
        if training_val is not None:
            return bool(_coerce_gradient_checkpointing_value(training_val))

    if compute is not None and hasattr(compute, "get"):
        return bool(
            _coerce_gradient_checkpointing_value(
                compute.get("gradient_checkpointing", False),
            ),
        )

    return False


def maybe_enable_gradient_checkpointing(model: nn.Module, cfg: Any) -> bool:
    """Gate :func:`enable_gradient_checkpointing` on the config.

    Returns True if checkpointing was enabled. Use this at every call site
    that currently hardcodes ``enable_gradient_checkpointing(model)`` so the
    ``compute.gradient_checkpointing: false`` knob actually takes effect.

    When ``cfg.training.gradient_checkpointing`` (or ``cfg.compute....``) is
    set to the string ``"megatron"``, this enables checkpointing then swaps
    each layer's ``_gradient_checkpointing_func`` for a per-op selective
    variant that MUST_SAVE matmul + SDPA outputs and PREFER_RECOMPUTE
    everything else (layernorms, silu, mul, residual adds). Reference:
    Korthikanti et al. 2022 "Reducing Activation Recomputation in Large
    Transformer Models".
    """
    requested = _resolve_gradient_checkpointing_mode(cfg)
    return _enable_gradient_checkpointing_mode(model, requested)


def _enable_gradient_checkpointing_mode(
    model: nn.Module,
    requested: bool | str | None,
) -> bool:
    if requested is False or requested is None:
        logger.info(
            "gradient_checkpointing_disabled",
            model=model.__class__.__name__,
            mode=requested,
        )
        return False
    enable_gradient_checkpointing(model)
    megatron_layers = 0
    if requested == "megatron":
        megatron_layers = _install_megatron_checkpoint_func(model)
    logger.info(
        "gradient_checkpointing_enabled",
        model=model.__class__.__name__,
        mode=requested,
        megatron_layers=megatron_layers,
    )
    return True


def maybe_enable_decoder_gradient_checkpointing(model: nn.Module, cfg: Any) -> bool:
    """Enable decoder checkpointing only when explicitly requested.

    A frozen decoder still needs activation gradients with respect to survivor
    embeddings, but it does not need decoder weight gradients. For the permanent
    no-LoRA frozen-decoder contract, checkpointing recomputes the whole decoder
    on backward and has been a poor speed tradeoff. Set
    ``training.checkpoint_frozen_decoder: true`` to opt back in for memory
    pressure experiments.

    Trainable Qwen3.5 packed decoder backward currently fails PyTorch
    checkpoint recompute checks, so decoder checkpointing is not inherited from
    the global encoder-side checkpointing knob. Set
    ``training.decoder_gradient_checkpointing`` to true or "megatron" for a
    named A/B only.
    """

    tcfg = getattr(cfg, "training", None)
    freeze_decoder = False
    checkpoint_frozen_decoder = False
    if tcfg is not None and hasattr(tcfg, "get"):
        decoder_requested = tcfg.get("decoder_gradient_checkpointing", None)
        if decoder_requested is not None:
            requested = _coerce_gradient_checkpointing_value(decoder_requested)
            if requested is False or requested is None:
                logger.info(
                    "decoder_gradient_checkpointing_disabled_by_config",
                    model=model.__class__.__name__,
                    mode=requested,
                )
                return False
            return _enable_gradient_checkpointing_mode(model, requested)

        freeze_cfg = tcfg.get("freeze", {}) or {}
        if hasattr(freeze_cfg, "get"):
            freeze_decoder = bool(freeze_cfg.get("decoder", False))
        checkpoint_frozen_decoder = bool(tcfg.get("checkpoint_frozen_decoder", False))

    if freeze_decoder and not checkpoint_frozen_decoder:
        logger.info(
            "decoder_gradient_checkpointing_disabled_for_frozen_decoder",
            model=model.__class__.__name__,
        )
        return False
    if freeze_decoder and checkpoint_frozen_decoder:
        return maybe_enable_gradient_checkpointing(model, cfg)
    logger.info(
        "decoder_gradient_checkpointing_disabled_by_default",
        model=model.__class__.__name__,
    )
    return False


def validate_decoder_lora_freeze_contract(cfg: Any) -> None:
    """Reject decoder LoRA when the decoder is configured permanently frozen.

    The frozen-decoder training contract backpropagates only into survivor
    embeddings and upstream BgKIT modules. Installing LoRA adapters before
    applying ``training.freeze.decoder: true`` creates dead trainable modules
    that are immediately frozen, which is both misleading and outside the
    no-LoRA kernel target.
    """

    tcfg = getattr(cfg, "training", None)
    if tcfg is None or not hasattr(tcfg, "get"):
        return

    freeze_cfg = tcfg.get("freeze", {}) or {}
    lora_cfg = tcfg.get("decoder_lora", {}) or {}
    freeze_decoder = (
        bool(freeze_cfg.get("decoder", False)) if hasattr(freeze_cfg, "get") else False
    )
    lora_enabled = (
        bool(lora_cfg.get("enabled", False)) if hasattr(lora_cfg, "get") else False
    )
    if freeze_decoder and lora_enabled:
        raise ValueError(
            "training.freeze.decoder=true is the no-LoRA frozen-decoder contract; "
            "set training.decoder_lora.enabled=false or unfreeze the decoder."
        )


def _resolve_gradient_checkpointing_mode(cfg: Any) -> bool | str | None:
    """Return True / False / "selective" / None for the gradient-checkpointing knob."""
    tcfg = getattr(cfg, "training", None)
    compute = getattr(cfg, "compute", None)

    if tcfg is not None and hasattr(tcfg, "get"):
        training_val = tcfg.get("gradient_checkpointing", None)
        if training_val is not None:
            return _coerce_gradient_checkpointing_value(training_val)

    if compute is not None and hasattr(compute, "get"):
        compute_val = compute.get("gradient_checkpointing", False)
        return _coerce_gradient_checkpointing_value(compute_val)

    return False


def _coerce_gradient_checkpointing_value(val: Any) -> bool | str:
    if isinstance(val, str):
        normalized = val.strip().lower()
        if normalized in {"megatron", "selective", "selective_ops", "selective_v2"}:
            return "megatron"
        if normalized in {"true", "1", "on", "yes"}:
            return True
        if normalized in {"false", "0", "off", "no"}:
            return False
    return bool(val)


def set_gradient_checkpointing_mode(model: nn.Module, mode: str) -> dict:
    """Flip a model's gradient-checkpointing mode at runtime.

    Modes:
        "off"       — no checkpointing; all activations saved (max memory, max speed)
        "full"      — HF default ``partial(checkpoint, use_reentrant=False)``
                      on every ``GradientCheckpointingLayer``
        "megatron"  — selective per-op SAVE/RECOMPUTE policy (matmul + SDPA saved)

    Idempotent. Used by the compression-aware ckpt scheduler in the trainer:
    as ``actual_ratio`` drops, the decoder's working set shrinks and we can
    afford to recompute less. Walks both pure HF v5+ ``GradientCheckpointingLayer``
    instances and our custom ``PrunedBidirectionalQwen35`` (which uses a
    sibling ``_gradient_checkpointing`` attribute).

    Returns a metrics dict for logging:
        {"mode": str, "layers_with_ckpt": int, "megatron_layers": int,
         "uses_legacy_ckpt_attr": bool}
    """
    if mode not in {"off", "full", "megatron"}:
        raise ValueError(f"mode must be off|full|megatron, got {mode!r}")

    # First, disable everywhere — covers both HF v5+ standard API and our
    # custom PrunedBidirectionalQwen35.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    # Custom encoder backbone (PrunedBidirectionalQwen35) sets a sibling
    # ``_gradient_checkpointing`` attr that its forward consults — used for
    # diagnostics, doesn't change behavior here.
    uses_legacy = hasattr(model, "_gradient_checkpointing")

    if mode == "off":
        return {
            "mode": mode,
            "layers_with_ckpt": 0,
            "megatron_layers": 0,
            "uses_legacy_ckpt_attr": uses_legacy,
        }

    enable_gradient_checkpointing(model)
    layers_with_ckpt = sum(
        1
        for m in model.modules()
        if getattr(m, "gradient_checkpointing", False)
    )
    megatron_layers = 0
    if mode == "megatron":
        megatron_layers = _install_megatron_checkpoint_func(model)
    return {
        "mode": mode,
        "layers_with_ckpt": layers_with_ckpt,
        "megatron_layers": megatron_layers,
        "uses_legacy_ckpt_attr": uses_legacy,
    }


def _install_megatron_checkpoint_func(model: nn.Module) -> int:
    """Replace each layer's ``_gradient_checkpointing_func`` with one that uses
    ``torch.utils.checkpoint.create_selective_checkpoint_contexts`` to apply a
    per-op SAVE/RECOMPUTE policy (Megatron-style selective recomputation).

    Memory savings vs. full ckpt: small (already saving heavy matmul outputs
    means we keep almost as much as no-ckpt for the LoRA-input chain).
    Speed gains vs. full ckpt: large for layer types where most ops are cheap
    (layernorm, silu, mul, add) and only a few are FLOP-dense (matmul, attn).
    The expected per-step throughput recovers ~50-90% of no-ckpt while peak
    memory stays close to no-ckpt's matmul-saved baseline.

    Walks every ``GradientCheckpointingLayer`` in the model that has
    ``gradient_checkpointing=True`` and overwrites its
    ``_gradient_checkpointing_func`` with our selective context wrapper.

    Returns the number of layers that had the func swapped.
    """
    swapped = 0
    for module in model.modules():
        if not hasattr(module, "_gradient_checkpointing_func"):
            continue
        if not getattr(module, "gradient_checkpointing", False):
            continue
        module._gradient_checkpointing_func = _megatron_checkpoint_func
        swapped += 1
    return swapped


def clip_grad_norm(
    parameters,
    max_norm: float = 1.0,
) -> float:
    """Clip gradient norms and return the total norm.

    Args:
        parameters: Model parameters.
        max_norm: Maximum gradient norm.

    Returns:
        Total gradient norm before clipping.
    """
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm).item()
