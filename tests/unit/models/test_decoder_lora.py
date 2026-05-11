"""Tests for ReconstructionDecoder.apply_lora() and merge_lora().

Verifies:
1. apply_lora wraps decoder target modules with PEFT by default
2. merge_lora produces a clean state dict loadable into a plain decoder
3. Merged weights incorporate the LoRA delta (W' = W + scaling * B @ A)
4. Roundtrip: apply_lora → train → merge_lora → load into plain decoder
"""

from __future__ import annotations

import copy
import os

import pytest

torch = pytest.importorskip("torch")

from torch import nn
from torch.nn import functional as F

from bgkit.models.decoder import DecoderLoRALinear, ReconstructionDecoder
from bgkit.models.lora_triton import (
    can_use_triton_gate_up_base_dx,
    can_use_triton_lora_dx_add,
    can_use_triton_lora_pair_dx_add,
    can_use_triton_swiglu_backward,
    can_use_triton_swiglu_forward,
    triton_gate_up_base_dx,
    triton_lora_pair_dx_add_,
    triton_swiglu_backward,
    triton_swiglu_forward,
)

# ---------------------------------------------------------------------------
# Mock backbone with HF CausalLM-style .model / .lm_head nesting
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
HIDDEN_DIM = 32


class _ModelOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _CausalLMOutput:
    def __init__(self, logits):
        self.logits = logits


class _MockMLP(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.down_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _MockInnerModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        # Use named linear layers that match LoRA target module names
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = _MockMLP(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        x = self.q_proj(inputs_embeds)
        x = self.v_proj(x)
        x = self.mlp(x)
        x = self.norm(x)
        return _ModelOutput(last_hidden_state=x)


class MockCausalLMBackbone(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.model = _MockInnerModel(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # HF-compat: peft accesses config via both .attr and .get()
        class _Config(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(name) from None

        self.config = _Config(model_type="mock", tie_word_embeddings=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {}

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = self.lm_head(out.last_hidden_state)
        return _CausalLMOutput(logits=logits)


LORA_CONFIG = {
    "r": 4,
    "alpha": 8,
    "dropout": 0.0,
    "target_modules": ["q_proj", "v_proj"],
}
NATIVE_LORA_CONFIG = {**LORA_CONFIG, "implementation": "native"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestApplyLora:
    def test_falcon_h1_lora_is_disabled_even_with_explicit_targets(self):
        backbone = MockCausalLMBackbone()
        decoder = ReconstructionDecoder(
            backbone,
            hidden_dim=HIDDEN_DIM,
            decoder_family="falcon_h1",
        )

        with pytest.raises(ValueError, match="Falcon-H1 decoder LoRA is disabled"):
            decoder.apply_lora(
                {
                    **LORA_CONFIG,
                    "family": "qwen35",
                    "target_modules": ["q_proj"],
                    "implementation": "native",
                }
            )

        assert not decoder._has_lora

    def test_default_wraps_with_peft(self):
        peft = pytest.importorskip("peft")
        backbone = MockCausalLMBackbone()
        decoder = ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)

        decoder.apply_lora(LORA_CONFIG)

        assert decoder._has_lora
        assert isinstance(decoder.backbone, peft.PeftModel)

    def test_native_wraps_target_linears(self):
        backbone = MockCausalLMBackbone()
        decoder = ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)
        decoder.apply_lora(NATIVE_LORA_CONFIG)

        assert decoder._has_lora
        assert isinstance(decoder.backbone.model.q_proj, DecoderLoRALinear)
        assert isinstance(decoder.backbone.model.v_proj, DecoderLoRALinear)

    def test_reduces_trainable_params(self):
        backbone = MockCausalLMBackbone()
        decoder = ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)

        trainable_before = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

        decoder.apply_lora(LORA_CONFIG)

        trainable_after = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

        assert trainable_after < trainable_before
        assert trainable_after > 0

    def test_adapters_follow_base_dtype_by_default(self):
        backbone = MockCausalLMBackbone().to(dtype=torch.bfloat16)
        decoder = ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)

        decoder.apply_lora(LORA_CONFIG)

        lora_dtypes = {
            p.dtype
            for name, p in decoder.named_parameters()
            if "lora_" in name
        }
        assert lora_dtypes == {torch.bfloat16}

    def test_legacy_dtype_opt_out_keeps_fp32_adapters(self):
        backbone = MockCausalLMBackbone().to(dtype=torch.bfloat16)
        decoder = ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)

        decoder.apply_lora({**LORA_CONFIG, "adapter_dtype": "peft"})

        lora_dtypes = {
            p.dtype
            for name, p in decoder.named_parameters()
            if "lora_" in name
        }
        assert lora_dtypes == {torch.float32}

    def test_peft_implementation_still_available(self):
        peft = pytest.importorskip("peft")
        backbone = MockCausalLMBackbone()
        decoder = ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)

        decoder.apply_lora({**LORA_CONFIG, "implementation": "peft"})

        assert isinstance(decoder.backbone, peft.PeftModel)

    def test_peft_fused_backward_preserves_peft_layout(self):
        pytest.importorskip("peft")
        backbone = MockCausalLMBackbone()
        decoder = ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)

        decoder.apply_lora(
            {
                **LORA_CONFIG,
                "implementation": "peft",
                "peft_fused_backward": True,
            }
        )

        assert decoder._lora_peft_fused_count == 2
        assert decoder._lora_peft_gate_up_fused_count == 0
        assert hasattr(decoder.backbone.base_model.model.model.q_proj, "_bgkit_original_forward")
        assert any(".lora_A.default.weight" in key for key in decoder.state_dict())
        assert not any(key.endswith(".lora_A") for key in decoder.state_dict())

    def test_peft_fused_backward_matches_peft_reference(self):
        pytest.importorskip("peft")
        torch.manual_seed(123)
        backbone = MockCausalLMBackbone()
        ref = ReconstructionDecoder(copy.deepcopy(backbone), hidden_dim=HIDDEN_DIM)
        fused = ReconstructionDecoder(copy.deepcopy(backbone), hidden_dim=HIDDEN_DIM)
        ref.apply_lora({**LORA_CONFIG, "implementation": "peft"})
        fused.apply_lora(
            {
                **LORA_CONFIG,
                "implementation": "peft",
                "peft_fused_backward": True,
            }
        )
        fused.load_state_dict(ref.state_dict())

        ref_module = ref.backbone.base_model.model.model.q_proj
        fused_module = fused.backbone.base_model.model.model.q_proj
        x = torch.randn(2, 4, HIDDEN_DIM, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        grad = torch.randn(2, 4, HIDDEN_DIM)

        y_ref = ref_module(x_ref)
        y_fused = fused_module(x)
        torch.testing.assert_close(y_fused, y_ref)

        y_ref.backward(grad)
        y_fused.backward(grad)

        torch.testing.assert_close(x.grad, x_ref.grad)
        torch.testing.assert_close(
            fused_module.lora_A.default.weight.grad,
            ref_module.lora_A.default.weight.grad,
        )
        torch.testing.assert_close(
            fused_module.lora_B.default.weight.grad,
            ref_module.lora_B.default.weight.grad,
        )

    def test_peft_gate_up_mlp_fusion_matches_peft_reference(self):
        pytest.importorskip("peft")
        torch.manual_seed(321)
        lora_cfg = {
            "r": 4,
            "alpha": 8,
            "dropout": 0.0,
            "implementation": "peft",
            "target_modules": ["gate_proj", "up_proj", "down_proj"],
        }
        backbone = MockCausalLMBackbone()
        ref = ReconstructionDecoder(copy.deepcopy(backbone), hidden_dim=HIDDEN_DIM)
        fused = ReconstructionDecoder(copy.deepcopy(backbone), hidden_dim=HIDDEN_DIM)
        ref.apply_lora({**lora_cfg, "peft_fused_backward": False})
        fused.apply_lora(
            {
                **lora_cfg,
                "peft_fused_backward": True,
                "peft_fuse_gate_up": True,
            }
        )
        fused.load_state_dict(ref.state_dict())

        ref_mlp = ref.backbone.base_model.model.model.mlp
        fused_mlp = fused.backbone.base_model.model.model.mlp
        assert fused._lora_peft_fused_count == 3
        assert fused._lora_peft_gate_up_fused_count == 1
        assert hasattr(fused_mlp, "_bgkit_original_forward")

        x = torch.randn(2, 4, HIDDEN_DIM, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        y_ref = ref_mlp(x_ref)
        y_fused = fused_mlp(x)
        grad = torch.randn_like(y_ref)

        torch.testing.assert_close(y_fused, y_ref)
        y_ref.backward(grad)
        y_fused.backward(grad)

        torch.testing.assert_close(x.grad, x_ref.grad)
        for name in ("gate_proj", "up_proj", "down_proj"):
            ref_module = getattr(ref_mlp, name)
            fused_module = getattr(fused_mlp, name)
            torch.testing.assert_close(
                fused_module.lora_A.default.weight.grad,
                ref_module.lora_A.default.weight.grad,
            )
            torch.testing.assert_close(
                fused_module.lora_B.default.weight.grad,
                ref_module.lora_B.default.weight.grad,
            )

    def test_native_frozen_base_lora_backward_matches_reference(self):
        torch.manual_seed(123)
        base = nn.Linear(7, 5, bias=True).to(dtype=torch.float64)
        wrapper = DecoderLoRALinear(
            base,
            rank=3,
            alpha=6.0,
            dropout=0.0,
            adapter_dtype=torch.float64,
        )
        wrapper.lora_A.data.normal_(0, 0.1)
        wrapper.lora_B.data.normal_(0, 0.1)

        ref_weight = wrapper.base_layer.weight.detach().clone()
        ref_bias = wrapper.base_layer.bias.detach().clone()
        ref_a = wrapper.lora_A.detach().clone().requires_grad_(True)
        ref_b = wrapper.lora_B.detach().clone().requires_grad_(True)

        x = torch.randn(2, 4, 7, dtype=torch.float64, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_(True)
        grad = torch.randn(2, 4, 5, dtype=torch.float64)

        y = wrapper(x)
        ref_y = F.linear(ref_x, ref_weight, ref_bias)
        ref_y = ref_y + F.linear(F.linear(ref_x, ref_a), ref_b) * wrapper.scaling

        torch.testing.assert_close(y, ref_y)
        y.backward(grad)
        ref_y.backward(grad)

        torch.testing.assert_close(x.grad, ref_x.grad)
        torch.testing.assert_close(wrapper.lora_A.grad, ref_a.grad)
        torch.testing.assert_close(wrapper.lora_B.grad, ref_b.grad)

    def test_native_gate_up_fusion_matches_unfused_native_lora(self):
        torch.manual_seed(123)
        lora_cfg = {
            "r": 4,
            "alpha": 8,
            "dropout": 0.0,
            "implementation": "native",
            "target_modules": ["gate_proj", "up_proj", "down_proj"],
        }
        backbone = MockCausalLMBackbone()
        ref = ReconstructionDecoder(copy.deepcopy(backbone), hidden_dim=HIDDEN_DIM)
        fused = ReconstructionDecoder(copy.deepcopy(backbone), hidden_dim=HIDDEN_DIM)
        ref.apply_lora({**lora_cfg, "fuse_gate_up": False})
        fused.apply_lora({**lora_cfg, "fuse_gate_up": True})
        fused.load_state_dict(ref.state_dict())

        ref_mlp = ref.backbone.model.mlp
        fused_mlp = fused.backbone.model.mlp
        assert hasattr(fused_mlp, "_bgkit_original_forward")

        x = torch.randn(2, 4, HIDDEN_DIM, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        y_ref = ref_mlp(x_ref)
        y_fused = fused_mlp(x)
        grad = torch.randn_like(y_ref)

        torch.testing.assert_close(y_fused, y_ref)
        y_ref.backward(grad)
        y_fused.backward(grad)

        torch.testing.assert_close(x.grad, x_ref.grad)
        for name in ("gate_proj", "up_proj", "down_proj"):
            ref_module = getattr(ref_mlp, name)
            fused_module = getattr(fused_mlp, name)
            torch.testing.assert_close(fused_module.lora_A.grad, ref_module.lora_A.grad)
            torch.testing.assert_close(fused_module.lora_B.grad, ref_module.lora_B.grad)

    def test_triton_lora_dx_add_rejects_cpu_tensors(self):
        dx = torch.zeros(4, 8)
        gh = torch.zeros(4, 2)
        a = torch.zeros(2, 8)

        assert not can_use_triton_lora_dx_add(dx, gh, a)

    def test_triton_swiglu_backward_rejects_cpu_tensors(self):
        grad_hidden = torch.zeros(4, 8)
        gate = torch.zeros(4, 8)
        up = torch.zeros(4, 8)

        assert not can_use_triton_swiglu_backward(grad_hidden, gate, up)

    def test_triton_swiglu_forward_rejects_cpu_tensors(self):
        gate = torch.zeros(4, 8)
        up = torch.zeros(4, 8)

        assert not can_use_triton_swiglu_forward(gate, up)

    def test_triton_lora_pair_dx_add_rejects_cpu_tensors(self):
        dx = torch.zeros(4, 8)
        gh = torch.zeros(4, 2)
        a = torch.zeros(2, 8)

        assert not can_use_triton_lora_pair_dx_add(dx, gh, a, gh, a)

    def test_triton_gate_up_base_dx_rejects_cpu_tensors(self):
        grad_gate = torch.zeros(4, 8)
        grad_up = torch.zeros(4, 8)
        gate_weight = torch.zeros(8, 4)
        up_weight = torch.zeros(8, 4)

        assert not can_use_triton_gate_up_base_dx(
            grad_gate,
            gate_weight,
            grad_up,
            up_weight,
        )

    @pytest.mark.skipif(
        not (torch.cuda.is_available() and os.environ.get("BGKIT_RUN_GPU_TESTS")),
        reason="CUDA Triton test; set BGKIT_RUN_GPU_TESTS=1 in the training container",
    )
    def test_triton_swiglu_forward_matches_torch_cuda(self):
        torch.manual_seed(0)
        gate = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)
        up = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)

        actual = triton_swiglu_forward(gate, up)
        expected = F.silu(gate.float()).to(torch.bfloat16) * up

        torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.03)

    @pytest.mark.skipif(
        not (torch.cuda.is_available() and os.environ.get("BGKIT_RUN_GPU_TESTS")),
        reason="CUDA Triton test; set BGKIT_RUN_GPU_TESTS=1 in the training container",
    )
    def test_triton_swiglu_backward_matches_torch_cuda(self):
        torch.manual_seed(0)
        gate = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)
        up = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)
        grad_hidden = torch.randn_like(gate)

        grad_gate, grad_up = triton_swiglu_backward(grad_hidden, gate, up)
        gate_ref = gate.float().requires_grad_(True)
        up_ref = up.float().requires_grad_(True)
        expected = F.silu(gate_ref) * up_ref
        expected.backward(grad_hidden.float())

        torch.testing.assert_close(grad_gate.float(), gate_ref.grad, atol=0.04, rtol=0.04)
        torch.testing.assert_close(grad_up.float(), up_ref.grad, atol=0.04, rtol=0.04)

    @pytest.mark.skipif(
        not (torch.cuda.is_available() and os.environ.get("BGKIT_RUN_GPU_TESTS")),
        reason="CUDA Triton test; set BGKIT_RUN_GPU_TESTS=1 in the training container",
    )
    def test_triton_lora_pair_dx_add_matches_torch_cuda(self):
        torch.manual_seed(0)
        dx = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)
        gh0 = torch.randn(7, 16, device="cuda", dtype=torch.bfloat16)
        gh1 = torch.randn(7, 16, device="cuda", dtype=torch.bfloat16)
        a0 = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
        a1 = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)

        expected = dx + gh0 @ a0 + gh1 @ a1
        actual = triton_lora_pair_dx_add_(dx.clone(), gh0, a0, gh1, a1)

        torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.04)

    @pytest.mark.skipif(
        not (torch.cuda.is_available() and os.environ.get("BGKIT_RUN_GPU_TESTS")),
        reason="CUDA Triton test; set BGKIT_RUN_GPU_TESTS=1 in the training container",
    )
    def test_triton_gate_up_base_dx_matches_torch_cuda(self):
        torch.manual_seed(0)
        grad_gate = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)
        grad_up = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)
        gate_weight = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
        up_weight = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)

        actual = triton_gate_up_base_dx(grad_gate, gate_weight, grad_up, up_weight)
        expected = grad_gate @ gate_weight + grad_up @ up_weight

        torch.testing.assert_close(actual, expected, atol=0.2, rtol=0.04)


class TestLoraStateDictCompatibility:
    def test_native_loads_peft_lora_state_dict(self):
        pytest.importorskip("peft")
        torch.manual_seed(0)

        source = ReconstructionDecoder(MockCausalLMBackbone(), hidden_dim=HIDDEN_DIM)
        source.apply_lora({**LORA_CONFIG, "implementation": "peft"})
        for name, param in source.named_parameters():
            if "lora_" in name:
                param.data.normal_(0, 0.01)
        peft_state = source.state_dict()

        target = ReconstructionDecoder(MockCausalLMBackbone(), hidden_dim=HIDDEN_DIM)
        target.apply_lora(NATIVE_LORA_CONFIG)
        result = target.load_state_dict(peft_state)

        assert result.missing_keys == []
        assert result.unexpected_keys == []
        native_state = target.state_dict()
        assert torch.equal(
            native_state["backbone.model.q_proj.lora_A"],
            peft_state[
                "backbone.base_model.model.model.q_proj.lora_A.default.weight"
            ],
        )
        assert torch.equal(
            native_state["backbone.model.q_proj.lora_B"],
            peft_state[
                "backbone.base_model.model.model.q_proj.lora_B.default.weight"
            ],
        )

    def test_peft_loads_native_lora_state_dict(self):
        peft = pytest.importorskip("peft")
        torch.manual_seed(0)

        source = ReconstructionDecoder(MockCausalLMBackbone(), hidden_dim=HIDDEN_DIM)
        source.apply_lora(NATIVE_LORA_CONFIG)
        for name, param in source.named_parameters():
            if "lora_" in name:
                param.data.normal_(0, 0.01)
        native_state = source.state_dict()

        target = ReconstructionDecoder(MockCausalLMBackbone(), hidden_dim=HIDDEN_DIM)
        target.apply_lora({**LORA_CONFIG, "implementation": "peft"})
        result = target.load_state_dict(native_state)

        assert isinstance(target.backbone, peft.PeftModel)
        assert result.missing_keys == []
        assert result.unexpected_keys == []
        peft_state = target.state_dict()
        assert torch.equal(
            peft_state[
                "backbone.base_model.model.model.q_proj.lora_A.default.weight"
            ],
            native_state["backbone.model.q_proj.lora_A"],
        )
        assert torch.equal(
            peft_state[
                "backbone.base_model.model.model.q_proj.lora_B.default.weight"
            ],
            native_state["backbone.model.q_proj.lora_B"],
        )


class TestMergeLora:
    def test_produces_clean_keys(self):
        """Merged state dict should have standard decoder keys (no peft prefixes)."""
        backbone = MockCausalLMBackbone()
        decoder = ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)
        decoder.apply_lora(LORA_CONFIG)

        merged = decoder.merge_lora()

        for key in merged:
            assert "lora_A" not in key, f"LoRA key leaked: {key}"
            assert "lora_B" not in key, f"LoRA key leaked: {key}"
            assert "base_model.model" not in key, f"PeftModel prefix leaked: {key}"
            assert ".base_layer." not in key, f"base_layer prefix leaked: {key}"

    def test_loadable_into_plain_decoder(self):
        """Merged state dict should load into a fresh (non-LoRA) decoder."""
        backbone = MockCausalLMBackbone()
        decoder = ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)
        decoder.apply_lora(LORA_CONFIG)

        merged = decoder.merge_lora()

        # Create a fresh plain decoder and load merged weights
        fresh_backbone = MockCausalLMBackbone()
        fresh_decoder = ReconstructionDecoder(fresh_backbone, hidden_dim=HIDDEN_DIM)
        fresh_decoder.load_state_dict(merged)

    def test_merged_weights_differ_from_base(self):
        """Merged weights should incorporate the LoRA delta (not identical to base)."""
        backbone = MockCausalLMBackbone()
        decoder = ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)

        # Capture base weights before LoRA
        base_sd = {k: v.clone() for k, v in decoder.state_dict().items()}

        decoder.apply_lora(LORA_CONFIG)

        # Simulate a tiny training step to make LoRA weights non-zero
        for name, param in decoder.named_parameters():
            if "lora_" in name and param.requires_grad:
                param.data.normal_(0, 0.1)

        merged = decoder.merge_lora()

        # At least one target module weight should differ from base
        changed = False
        for key in merged:
            if (
                key in base_sd
                and ("q_proj" in key or "v_proj" in key)
                and not torch.equal(merged[key], base_sd[key])
            ):
                changed = True
                break

        assert changed, "Merged weights should differ from base after LoRA training"

    def test_roundtrip_single_splice_loss_equivalence(self):
        """Single-splice packed loss matches before and after LoRA merge."""
        torch.manual_seed(42)

        backbone = MockCausalLMBackbone()
        decoder = ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)
        decoder.apply_lora(LORA_CONFIG)

        # Simulate training: set LoRA weights to something non-trivial
        for name, param in decoder.named_parameters():
            if "lora_" in name and param.requires_grad:
                param.data.normal_(0, 0.01)

        # Packed inputs: 3 survivors (flat), no prefix, 8 suffix tokens
        surv_flat = torch.randn(3, HIDDEN_DIM)
        cu = torch.tensor([0, 3], dtype=torch.int32)
        target_ids = torch.randint(0, VOCAB_SIZE, (1, 8))

        decoder.eval()
        with torch.no_grad():
            lora_loss = decoder.forward_with_single_splice(
                survivor_embeddings=surv_flat,
                survivor_cu_seqlens=cu,
                prefix_ids=[torch.zeros(0, dtype=torch.long)],
                suffix_ids=[target_ids[0]],
            )

        # Merge and load into plain decoder
        merged = decoder.merge_lora()
        fresh_backbone = MockCausalLMBackbone()
        fresh_decoder = ReconstructionDecoder(fresh_backbone, hidden_dim=HIDDEN_DIM)
        fresh_decoder.load_state_dict(merged)
        fresh_decoder.eval()

        with torch.no_grad():
            plain_loss = fresh_decoder.forward_with_single_splice(
                survivor_embeddings=surv_flat,
                survivor_cu_seqlens=cu,
                prefix_ids=[torch.zeros(0, dtype=torch.long)],
                suffix_ids=[target_ids[0]],
            )

        torch.testing.assert_close(lora_loss, plain_loss, atol=1e-4, rtol=1e-4)

    def test_no_lora_returns_state_dict(self):
        """merge_lora() on a non-LoRA decoder just returns state_dict()."""
        backbone = MockCausalLMBackbone()
        decoder = ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)

        merged = decoder.merge_lora()
        plain = decoder.state_dict()

        assert set(merged.keys()) == set(plain.keys())
        for key in merged:
            assert torch.equal(merged[key], plain[key])
