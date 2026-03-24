"""Tests for ReconstructionDecoder.apply_lora() and merge_lora().

Verifies:
1. apply_lora wraps the backbone with peft
2. merge_lora produces a clean state dict loadable into a plain decoder
3. Merged weights incorporate the LoRA delta (W' = W + scaling * B @ A)
4. Roundtrip: apply_lora → train → merge_lora → load into plain decoder
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from torch import nn

from bgkit.models.decoder import ReconstructionDecoder

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


class _MockInnerModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        # Use named linear layers that match LoRA target module names
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        x = self.q_proj(inputs_embeds)
        x = self.v_proj(x)
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestApplyLora:
    def test_wraps_with_peft(self):
        backbone = MockCausalLMBackbone()
        decoder = ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)
        decoder.apply_lora(LORA_CONFIG)

        assert decoder._has_lora
        assert isinstance(decoder.backbone, peft.PeftModel)

    def test_reduces_trainable_params(self):
        backbone = MockCausalLMBackbone()
        decoder = ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)

        trainable_before = sum(
            p.numel() for p in decoder.parameters() if p.requires_grad
        )

        decoder.apply_lora(LORA_CONFIG)

        trainable_after = sum(
            p.numel() for p in decoder.parameters() if p.requires_grad
        )

        assert trainable_after < trainable_before
        assert trainable_after > 0


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

    def test_roundtrip_forward_equivalence(self):
        """forward() with LoRA decoder ≈ forward() with merged plain decoder."""
        torch.manual_seed(42)

        backbone = MockCausalLMBackbone()
        decoder = ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)
        decoder.apply_lora(LORA_CONFIG)

        # Simulate training: set LoRA weights to something non-trivial
        for name, param in decoder.named_parameters():
            if "lora_" in name and param.requires_grad:
                param.data.normal_(0, 0.01)

        # Forward through LoRA decoder
        survivors = torch.randn(1, 3, HIDDEN_DIM)
        target_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
        target_mask = torch.ones(1, 8, dtype=torch.bool)
        survivor_mask = torch.ones(1, 3, dtype=torch.bool)

        decoder.eval()
        with torch.no_grad():
            lora_logits = decoder(survivors, target_ids, target_mask, survivor_mask)

        # Merge and load into plain decoder
        merged = decoder.merge_lora()
        fresh_backbone = MockCausalLMBackbone()
        fresh_decoder = ReconstructionDecoder(fresh_backbone, hidden_dim=HIDDEN_DIM)
        fresh_decoder.load_state_dict(merged)
        fresh_decoder.eval()

        with torch.no_grad():
            plain_logits = fresh_decoder(
                survivors, target_ids, target_mask, survivor_mask,
            )

        torch.testing.assert_close(lora_logits, plain_logits, atol=1e-4, rtol=1e-4)

    def test_no_lora_returns_state_dict(self):
        """merge_lora() on a non-LoRA decoder just returns state_dict()."""
        backbone = MockCausalLMBackbone()
        decoder = ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)

        merged = decoder.merge_lora()
        plain = decoder.state_dict()

        assert set(merged.keys()) == set(plain.keys())
        for key in merged:
            assert torch.equal(merged[key], plain[key])
