"""Native NVFP4 reference packing for frozen decoder base weights.

This module intentionally does not use TransformerEngine's runtime FP4
conversion path. DGX Spark currently needs an Atlas-style route: pack weights
with software E4M3/E2M1 conversion, then consume those packed weights from a
custom W4A16 kernel. The module below defines the packed format and a reference
frozen Linear implementation. It is correctness scaffolding for the CUDA kernel,
not the final fast path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

NVFP4_GROUP_SIZE = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0


def _native_nvfp4_kernel_enabled() -> bool:
    return os.environ.get("BGKIT_NATIVE_NVFP4_KERNEL", "1").lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class NVFP4PackedTensor:
    """Packed NVFP4 matrix plus per-group scale metadata.

    ``packed`` stores two E2M1 values per byte. For each byte, the lower nibble
    is the earlier K position and the upper nibble is the later K position.
    ``scale_e4m3`` has one E4M3 scale byte per output row and K-group of 16.
    ``scale2`` is the second-level scalar scale used by Atlas-style NVFP4.
    """

    packed: torch.Tensor
    scale_e4m3: torch.Tensor
    scale2: torch.Tensor
    shape: tuple[int, int]
    group_size: int = NVFP4_GROUP_SIZE


def _require_float8_e4m3() -> torch.dtype:
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        raise RuntimeError("native NVFP4 packing requires torch.float8_e4m3fn support")
    return dtype


def _float_to_e4m3_bytes(x: torch.Tensor) -> torch.Tensor:
    dtype = _require_float8_e4m3()
    return x.to(dtype).view(torch.uint8).contiguous()


def _e4m3_bytes_to_float(x: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    e4m3 = _require_float8_e4m3()
    return x.contiguous().view(e4m3).to(dtype)


def _e2m1_codebook(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=device,
        dtype=dtype,
    )


def _e2m1_decode(codes: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    magnitudes = _e2m1_codebook(codes.device, dtype=dtype)
    mag = magnitudes[(codes & 0x7).to(torch.long)]
    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0).to(dtype=dtype)
    return mag * sign


def _e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    thresholds = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=x.device,
        dtype=x.dtype,
    )
    x = x.clamp(min=-E2M1_MAX, max=E2M1_MAX)
    mag_code = torch.bucketize(x.abs().contiguous(), thresholds)
    sign_bit = (x < 0).to(torch.uint8) << 3
    return (mag_code.to(torch.uint8) | sign_bit).contiguous()


def _pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    lo = codes[:, 0::2] & 0xF
    hi = (codes[:, 1::2] & 0xF) << 4
    return (lo | hi).contiguous()


def _unpack_nibbles(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    codes = torch.empty(
        packed.shape[0],
        in_features,
        device=packed.device,
        dtype=torch.uint8,
    )
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = (packed >> 4) & 0xF
    return codes


def pack_bf16_to_nvfp4(
    weight: torch.Tensor,
    *,
    group_size: int = NVFP4_GROUP_SIZE,
) -> NVFP4PackedTensor:
    """Pack a 2D BF16/FP32 weight matrix to Atlas-style NVFP4 metadata."""

    if weight.ndim != 2:
        raise ValueError(f"NVFP4 packing expects a 2D weight matrix, got {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    if group_size != NVFP4_GROUP_SIZE:
        raise ValueError(f"native NVFP4 currently requires group_size={NVFP4_GROUP_SIZE}")
    if in_features % group_size != 0:
        raise ValueError(
            f"NVFP4 K dimension must be divisible by {group_size}, got {in_features}"
        )
    if in_features % 2 != 0:
        raise ValueError("NVFP4 packing requires an even K dimension")

    w = weight.detach().to(dtype=torch.float32)
    grouped = w.view(out_features, in_features // group_size, group_size)
    group_absmax = grouped.abs().amax(dim=-1)
    global_absmax = group_absmax.amax()

    if float(global_absmax.item()) == 0.0:
        packed = torch.zeros(out_features, in_features // 2, device=w.device, dtype=torch.uint8)
        scales = torch.zeros(
            out_features,
            in_features // group_size,
            device=w.device,
            dtype=torch.uint8,
        )
        return NVFP4PackedTensor(
            packed=packed,
            scale_e4m3=scales,
            scale2=torch.ones((), device=w.device, dtype=torch.float32),
            shape=(out_features, in_features),
            group_size=group_size,
        )

    scale2 = (global_absmax / (E2M1_MAX * E4M3_MAX)).to(dtype=torch.float32)
    e4m3_tiny = torch.finfo(_require_float8_e4m3()).tiny
    raw_scale = group_absmax / (E2M1_MAX * scale2)
    raw_scale = torch.where(
        group_absmax > 0,
        raw_scale.clamp(min=e4m3_tiny, max=E4M3_MAX),
        torch.zeros_like(raw_scale),
    )
    scale_e4m3 = _float_to_e4m3_bytes(raw_scale)
    scale = _e4m3_bytes_to_float(scale_e4m3).repeat_interleave(group_size, dim=1)
    denom = scale * scale2
    normalized = torch.where(denom > 0, w / denom, torch.zeros_like(w))
    codes = _e2m1_encode(normalized)

    return NVFP4PackedTensor(
        packed=_pack_nibbles(codes),
        scale_e4m3=scale_e4m3,
        scale2=scale2,
        shape=(out_features, in_features),
        group_size=group_size,
    )


def dequantize_nvfp4(
    packed: NVFP4PackedTensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize an :class:`NVFP4PackedTensor` to a dense matrix."""

    _out_features, in_features = packed.shape
    codes = _unpack_nibbles(packed.packed, in_features)
    values = _e2m1_decode(codes, dtype=torch.float32)
    scales = _e4m3_bytes_to_float(packed.scale_e4m3).repeat_interleave(
        packed.group_size,
        dim=1,
    )
    dense = values * scales * packed.scale2.to(dtype=torch.float32)
    return dense.to(dtype=dtype)


class FrozenNVFP4Linear(nn.Module):
    """Reference frozen Linear backed by packed native NVFP4 weights.

    The base weight is stored as packed buffers, so it never receives gradients.
    CUDA BF16 inputs use a direct-from-packed Triton W4A16 kernel when
    available; other inputs fall back to reference dequantization.
    """

    in_features: int
    out_features: int

    def __init__(
        self,
        packed: NVFP4PackedTensor,
        *,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        out_features, in_features = packed.shape
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = packed.group_size
        self.register_buffer("weight_packed", packed.packed.contiguous())
        self.register_buffer("weight_scale_e4m3", packed.scale_e4m3.contiguous())
        self.register_buffer("weight_scale2", packed.scale2.detach().to(dtype=torch.float32))
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> FrozenNVFP4Linear:
        packed = pack_bf16_to_nvfp4(linear.weight)
        bias = linear.bias.detach() if linear.bias is not None else None
        return cls(packed, bias=bias).to(device=linear.weight.device)

    def dequantized_weight(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        packed = NVFP4PackedTensor(
            packed=self.weight_packed,
            scale_e4m3=self.weight_scale_e4m3,
            scale2=self.weight_scale2,
            shape=(self.out_features, self.in_features),
            group_size=self.group_size,
        )
        return dequantize_nvfp4(packed, dtype=dtype)

    @property
    def weight(self) -> torch.Tensor:
        """Compatibility view for code that inspects ``base_layer.weight``."""

        return self.dequantized_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _native_nvfp4_kernel_enabled():
            try:
                from bgkit.quant.nvfp4_triton import (
                    can_use_triton_nvfp4_linear,
                    triton_frozen_nvfp4_linear,
                )

                if can_use_triton_nvfp4_linear(
                    x,
                    self.weight_packed,
                    self.weight_scale_e4m3,
                    self.weight_scale2,
                ):
                    return triton_frozen_nvfp4_linear(
                        x,
                        self.weight_packed,
                        self.weight_scale_e4m3,
                        self.weight_scale2,
                        self.bias,
                        out_features=self.out_features,
                        in_features=self.in_features,
                    )
            except Exception:
                if os.environ.get("BGKIT_NATIVE_NVFP4_KERNEL_STRICT", "0") == "1":
                    raise
        weight = self.dequantized_weight(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        bias = self.bias is not None
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={bias}, group_size={self.group_size}"
        )
