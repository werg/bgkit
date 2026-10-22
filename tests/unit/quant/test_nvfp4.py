"""Tests for the native NVFP4 packed reference path."""

# ruff: noqa: I001

from __future__ import annotations

import sysconfig
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from bgkit.quant.nvfp4 import (
    FrozenNVFP4Linear,
    dequantize_nvfp4,
    pack_bf16_to_nvfp4,
)


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="native NVFP4 reference requires torch.float8_e4m3fn",
)


def test_pack_dequantize_nvfp4_shapes_and_error_bound():
    torch.manual_seed(0)
    weight = torch.randn(5, 32, dtype=torch.bfloat16) * 0.2

    packed = pack_bf16_to_nvfp4(weight)
    dense = dequantize_nvfp4(packed)

    assert packed.packed.shape == (5, 16)
    assert packed.scale_e4m3.shape == (5, 2)
    assert packed.packed.dtype == torch.uint8
    assert packed.scale_e4m3.dtype == torch.uint8
    assert dense.shape == weight.shape
    assert torch.isfinite(dense).all()

    cosine = torch.nn.functional.cosine_similarity(
        dense.flatten(),
        weight.float().flatten(),
        dim=0,
    )
    assert float(cosine) > 0.97


def test_pack_nvfp4_rejects_non_group_aligned_k():
    with pytest.raises(ValueError, match="divisible by 16"):
        pack_bf16_to_nvfp4(torch.randn(4, 24))


def test_frozen_nvfp4_linear_backward_only_returns_activation_grad():
    torch.manual_seed(1)
    linear = nn.Linear(32, 7, bias=True, dtype=torch.bfloat16)
    frozen = FrozenNVFP4Linear.from_linear(linear)
    x = torch.randn(3, 32, dtype=torch.bfloat16, requires_grad=True)

    y = frozen(x)
    loss = y.float().square().mean()
    loss.backward()

    assert x.grad is not None
    assert frozen.weight_packed.grad is None
    assert frozen.weight_scale_e4m3.grad is None
    assert frozen.weight_scale2.grad is None
    assert frozen.bias is not None
    assert frozen.bias.grad is None


def test_decoder_native_nvfp4_converts_lora_base_layers():
    from bgkit.models.decoder import DecoderLoRALinear, ReconstructionDecoder

    class TinyBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.q_proj = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
            self.lm_head = nn.Linear(32, 16, bias=False, dtype=torch.bfloat16)

    decoder = ReconstructionDecoder(TinyBackbone(), hidden_dim=32)
    decoder.apply_lora(
        {
            "implementation": "native",
            "target_modules": ["q_proj"],
            "r": 4,
            "alpha": 8,
            "adapter_dtype": "bf16",
        }
    )
    decoder.enable_native_frozen_nvfp4()

    wrapper = decoder.backbone.model.q_proj
    assert isinstance(wrapper, DecoderLoRALinear)
    assert isinstance(wrapper.base_layer, FrozenNVFP4Linear)


def test_triton_nvfp4_linear_matches_reference_cuda(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip(reason="native NVFP4 Triton parity requires CUDA")
    include_dir = sysconfig.get_paths().get("include")
    if include_dir is None or not (Path(include_dir) / "Python.h").exists():
        pytest.skip(reason="local Triton driver build requires Python.h")

    torch.manual_seed(2)
    linear = nn.Linear(64, 48, bias=True, dtype=torch.bfloat16, device="cuda")
    frozen = FrozenNVFP4Linear.from_linear(linear)
    x = torch.randn(5, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    dy = torch.randn(5, 48, dtype=torch.bfloat16, device="cuda")

    monkeypatch.setenv("BGKIT_NATIVE_NVFP4_KERNEL", "0")
    y_ref = frozen(x)
    y_ref.backward(dy)
    dx_ref = x.grad.detach().clone()

    x.grad = None
    monkeypatch.setenv("BGKIT_NATIVE_NVFP4_KERNEL", "1")
    monkeypatch.setenv("BGKIT_NATIVE_NVFP4_KERNEL_STRICT", "1")
    y_tri = frozen(x)
    y_tri.backward(dy)

    torch.testing.assert_close(y_tri.float(), y_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(x.grad.float(), dx_ref.float(), rtol=2e-2, atol=2e-2)
