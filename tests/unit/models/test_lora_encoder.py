"""Tests for per-level LoRA adapters on the encoder."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.lora_encoder import (
    DEFAULT_LORA_TARGETS,
    LoRALinearWrapper,
    LoRARouter,
)


class _MockAttn(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.o_proj(self.v_proj(x) + self.k_proj(x) + self.q_proj(x))


def test_install_wraps_target_linears():
    m = _MockAttn()
    router = LoRARouter.install(
        m,
        target_names=("q_proj", "k_proj", "v_proj", "o_proj"),
        levels={"l0": 4, "l1": 4},
    )
    assert len(router._wrappers) == 4
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert isinstance(getattr(m, name), LoRALinearWrapper)


def test_active_level_adds_delta():
    m = _MockAttn()
    router = LoRARouter.install(m, target_names=("q_proj",), levels={"l0": 4, "l1": 4})
    LoRARouter.bind(router)
    # Make L1 non-zero so we see a difference
    wrapper = m.q_proj
    with torch.no_grad():
        wrapper.adapters["l1"].lora_B.copy_(torch.randn_like(wrapper.adapters["l1"].lora_B))

    x = torch.randn(2, 3, 8)
    base_out = wrapper.base_layer(x)
    with router.active(None):
        neutral = wrapper(x)
    with router.active("l0"):
        l0_out = wrapper(x)
    with router.active("l1"):
        l1_out = wrapper(x)

    assert torch.allclose(neutral, base_out)
    # L0 starts with B = 0 so it's an identity delta
    assert torch.allclose(l0_out, base_out)
    # L1 should differ (we wrote non-zero B)
    assert not torch.allclose(l1_out, base_out)
    LoRARouter.bind(None)


def test_freeze_level_preserves_other_level():
    m = _MockAttn()
    router = LoRARouter.install(m, target_names=("q_proj",), levels={"l0": 4, "l1": 4})
    router.set_level_trainable("l0", False)
    router.set_level_trainable("l1", True)
    for p in router.adapter_parameters("l0"):
        assert not p.requires_grad
    for p in router.adapter_parameters("l1"):
        assert p.requires_grad


def test_remap_base_keys_to_lora_rewrites_target_weights():
    from bgkit.models.lora_encoder import remap_base_keys_to_lora

    base = {
        "backbone.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
        "backbone.layers.0.self_attn.q_proj.bias": torch.randn(4),
        "backbone.layers.0.self_attn.k_proj.weight": torch.randn(4, 4),
        "backbone.layers.0.self_attn.out_proj.weight": torch.randn(4, 4),  # not a target
        "backbone.norm.weight": torch.randn(4),
    }
    remapped = remap_base_keys_to_lora(base, target_names=("q_proj", "k_proj"))
    assert "backbone.layers.0.self_attn.q_proj.base_layer.weight" in remapped
    assert "backbone.layers.0.self_attn.q_proj.base_layer.bias" in remapped
    assert "backbone.layers.0.self_attn.k_proj.base_layer.weight" in remapped
    # Non-targets untouched
    assert "backbone.layers.0.self_attn.out_proj.weight" in remapped
    assert "backbone.norm.weight" in remapped


def test_remap_lora_keys_to_base_strips_adapters():
    from bgkit.models.lora_encoder import remap_lora_keys_to_base

    wrapped = {
        "q_proj.base_layer.weight": torch.randn(4, 4),
        "q_proj.adapters.l0.lora_A": torch.randn(2, 4),
        "q_proj.adapters.l0.lora_B": torch.randn(4, 2),
        "q_proj.adapters.l1.lora_A": torch.randn(2, 4),
        "norm.weight": torch.randn(4),
    }
    unwrapped = remap_lora_keys_to_base(wrapped, target_names=("q_proj",))
    assert "q_proj.weight" in unwrapped
    assert "q_proj.adapters.l0.lora_A" not in unwrapped
    assert "q_proj.adapters.l0.lora_B" not in unwrapped
    assert "norm.weight" in unwrapped


def test_lora_install_save_load_roundtrip():
    """Install LoRA, train adapters to non-zero, save, reload into a fresh
    encoder, verify both base and adapter weights survive."""
    from bgkit.models.lora_encoder import remap_base_keys_to_lora

    # Simulate a "fresh base + LoRA install" workflow
    m1 = _MockAttn()
    router1 = LoRARouter.install(
        m1,
        target_names=("q_proj", "k_proj", "v_proj", "o_proj"),
        levels={"l0": 4, "l1": 4},
    )
    LoRARouter.bind(router1)

    # Manually set adapter weights so we can check they roundtrip
    with torch.no_grad():
        for wrapper in router1._wrappers:
            for level in ("l0", "l1"):
                wrapper.adapters[level].lora_B.copy_(
                    torch.randn_like(wrapper.adapters[level].lora_B)
                )

    saved_state = m1.state_dict()
    # State dict should have base_layer keys and adapter keys
    assert any("base_layer.weight" in k for k in saved_state)
    assert any("adapters.l0.lora_A" in k for k in saved_state)
    assert any("adapters.l1.lora_B" in k for k in saved_state)

    # Fresh encoder → install LoRA → load saved state → verify
    m2 = _MockAttn()
    router2 = LoRARouter.install(
        m2,
        target_names=("q_proj", "k_proj", "v_proj", "o_proj"),
        levels={"l0": 4, "l1": 4},
    )
    missing, unexpected = m2.load_state_dict(saved_state, strict=True)
    assert missing == [] and unexpected == []

    # Base weights match
    for w1, w2 in zip(router1._wrappers, router2._wrappers, strict=True):
        assert torch.allclose(w1.base_layer.weight, w2.base_layer.weight)
        for level in ("l0", "l1"):
            assert torch.allclose(
                w1.adapters[level].lora_A,
                w2.adapters[level].lora_A,
            )
            assert torch.allclose(
                w1.adapters[level].lora_B,
                w2.adapters[level].lora_B,
            )

    # Now bootstrap case: pretend we have a pre-LoRA checkpoint and want to
    # load it into a LoRA-wrapped encoder. The remap helper rewrites the
    # keys, and the load succeeds strict=False (adapter params are missing
    # from the pre-LoRA dict but that's fine — they stay at zero-init).
    m3 = _MockAttn()
    pre_lora_state = m3.state_dict()  # pre-LoRA key shape
    LoRARouter.install(
        m3,
        target_names=("q_proj", "k_proj", "v_proj", "o_proj"),
        levels={"l0": 4, "l1": 4},
    )
    remapped = remap_base_keys_to_lora(
        pre_lora_state,
        target_names=("q_proj", "k_proj", "v_proj", "o_proj"),
    )
    # Strict=False: missing keys are the LoRA adapter params which stay at zero-init.
    missing, unexpected = m3.load_state_dict(remapped, strict=False)
    assert unexpected == []
    # Every missing key should be a LoRA adapter path.
    for k in missing:
        assert ".adapters." in k and ("lora_A" in k or "lora_B" in k)

    LoRARouter.bind(None)
