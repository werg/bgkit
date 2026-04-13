"""Tests for ReconstructionDecoder.generate_with_single_splice()."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.decoder import GenerationOutput, ReconstructionDecoder

# ---------------------------------------------------------------------------
# Mock backbone with generate() support
# ---------------------------------------------------------------------------


class _CausalLMOutput:
    def __init__(self, logits):
        self.logits = logits


class MockGenerativeLM(nn.Module):
    """Tiny causal LM that supports both forward() and generate().

    generate() produces a fixed sequence: input_len tokens (echo) followed by
    ``fixed_output_ids`` as the generated content.
    """

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32, fixed_output_ids=None):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.config = type("Config", (), {"eos_token_id": 2})()
        # Default output: tokens [10, 20, 30, 5, 6, 2]
        self._fixed_output_ids = fixed_output_ids or [10, 20, 30, 5, 6, 2]

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        x = self.linear(inputs_embeds)
        logits = self.lm_head(x)
        return _CausalLMOutput(logits=logits)

    def generate(
        self, inputs_embeds=None, attention_mask=None, max_new_tokens=100, **kwargs
    ):
        batch_size = inputs_embeds.size(0)
        input_len = inputs_embeds.size(1)
        device = inputs_embeds.device

        # Produce "input echo" (zeros) + fixed output ids
        input_echo = torch.zeros(batch_size, input_len, dtype=torch.long, device=device)
        new_tokens = torch.tensor(
            self._fixed_output_ids[:max_new_tokens],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0).expand(batch_size, -1)
        return torch.cat([input_echo, new_tokens], dim=1)


class MockTokenizer:
    """Minimal tokenizer mock."""

    eos_token_id = 2
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=False):
        # Filter out special tokens if requested
        if skip_special_tokens:
            ids = [i for i in ids.tolist() if i != self.eos_token_id]
        else:
            ids = ids.tolist()
        return " ".join(str(i) for i in ids)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def decoder():
    backbone = MockGenerativeLM(vocab_size=100, hidden_dim=32)
    return ReconstructionDecoder(backbone, hidden_dim=32)


@pytest.fixture()
def tokenizer():
    return MockTokenizer()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerationOutput:
    def test_generate_returns_generation_output(self, decoder, tokenizer):
        batch_size, num_survivors, hidden_dim = 2, 4, 32
        prefix_len = 3

        result = decoder.generate_with_single_splice(
            survivor_embeddings=torch.randn(batch_size, num_survivors, hidden_dim),
            survivor_attention_mask=torch.ones(batch_size, num_survivors, dtype=torch.bool),
            prefix_ids=torch.randint(0, 100, (batch_size, prefix_len)),
            prefix_attention_mask=torch.ones(batch_size, prefix_len, dtype=torch.bool),
            splice_starts=torch.zeros(batch_size, dtype=torch.long),
            splice_lengths=torch.zeros(batch_size, dtype=torch.long),
            suffix_ids=torch.tensor([5, 6], dtype=torch.long),
            tokenizer=tokenizer,
            max_new_tokens=10,
        )

        assert isinstance(result, GenerationOutput)
        assert len(result.content_ids) == batch_size
        assert len(result.content_text) == batch_size
        assert len(result.full_ids) == batch_size

    def test_content_ids_excludes_suffix(self, decoder, tokenizer):
        """When the model produces tokens ending with suffix, content should exclude them."""
        batch_size = 1

        # Model outputs [10, 20, 30, 5, 6, 2]. Suffix is [5, 6]. EOS is 2.
        # After stripping EOS: [10, 20, 30, 5, 6]
        # After stripping suffix [5, 6]: [10, 20, 30]
        result = decoder.generate_with_single_splice(
            survivor_embeddings=torch.randn(batch_size, 3, 32),
            survivor_attention_mask=torch.ones(batch_size, 3, dtype=torch.bool),
            prefix_ids=torch.randint(0, 100, (batch_size, 2)),
            prefix_attention_mask=torch.ones(batch_size, 2, dtype=torch.bool),
            splice_starts=torch.zeros(batch_size, dtype=torch.long),
            splice_lengths=torch.zeros(batch_size, dtype=torch.long),
            suffix_ids=torch.tensor([5, 6], dtype=torch.long),
            tokenizer=tokenizer,
            max_new_tokens=10,
        )

        assert result.content_ids[0].tolist() == [10, 20, 30]

    def test_greedy_is_deterministic(self, decoder, tokenizer):
        """Temperature=0 should produce identical output on two calls."""
        kwargs = {
            "survivor_embeddings": torch.randn(1, 3, 32),
            "survivor_attention_mask": torch.ones(1, 3, dtype=torch.bool),
            "prefix_ids": torch.randint(0, 100, (1, 2)),
            "prefix_attention_mask": torch.ones(1, 2, dtype=torch.bool),
            "splice_starts": torch.zeros(1, dtype=torch.long),
            "splice_lengths": torch.zeros(1, dtype=torch.long),
            "suffix_ids": torch.tensor([5, 6], dtype=torch.long),
            "tokenizer": tokenizer,
            "max_new_tokens": 10,
            "temperature": 0.0,
        }

        r1 = decoder.generate_with_single_splice(**kwargs)
        r2 = decoder.generate_with_single_splice(**kwargs)
        assert r1.content_ids[0].tolist() == r2.content_ids[0].tolist()

    def test_fallback_when_no_suffix_match(self, tokenizer):
        """When model doesn't produce the expected suffix, content includes all tokens."""
        # Fixed output [10, 20, 30, 5, 6, 2] with suffix [99, 98] (no match)
        backbone = MockGenerativeLM(
            vocab_size=100, hidden_dim=32, fixed_output_ids=[10, 20, 30, 40, 50, 2],
        )
        dec = ReconstructionDecoder(backbone, hidden_dim=32)

        result = dec.generate_with_single_splice(
            survivor_embeddings=torch.randn(1, 3, 32),
            survivor_attention_mask=torch.ones(1, 3, dtype=torch.bool),
            prefix_ids=torch.randint(0, 100, (1, 2)),
            prefix_attention_mask=torch.ones(1, 2, dtype=torch.bool),
            splice_starts=torch.zeros(1, dtype=torch.long),
            splice_lengths=torch.zeros(1, dtype=torch.long),
            suffix_ids=torch.tensor([99, 98], dtype=torch.long),
            tokenizer=tokenizer,
            max_new_tokens=10,
        )

        # No suffix stripped — all non-EOS tokens kept
        assert result.content_ids[0].tolist() == [10, 20, 30, 40, 50]

    def test_content_text_is_nonempty(self, decoder, tokenizer):
        result = decoder.generate_with_single_splice(
            survivor_embeddings=torch.randn(1, 3, 32),
            survivor_attention_mask=torch.ones(1, 3, dtype=torch.bool),
            prefix_ids=torch.randint(0, 100, (1, 2)),
            prefix_attention_mask=torch.ones(1, 2, dtype=torch.bool),
            splice_starts=torch.zeros(1, dtype=torch.long),
            splice_lengths=torch.zeros(1, dtype=torch.long),
            suffix_ids=torch.tensor([5, 6], dtype=torch.long),
            tokenizer=tokenizer,
            max_new_tokens=10,
        )

        assert len(result.content_text[0]) > 0

    def test_generate_with_single_splice_strips_suffix(self, decoder, tokenizer):
        result = decoder.generate_with_single_splice(
            survivor_embeddings=torch.randn(1, 3, 32),
            survivor_attention_mask=torch.ones(1, 3, dtype=torch.bool),
            prefix_ids=torch.randint(0, 100, (1, 5)),
            prefix_attention_mask=torch.ones(1, 5, dtype=torch.bool),
            splice_starts=torch.tensor([2], dtype=torch.long),
            splice_lengths=torch.tensor([0], dtype=torch.long),
            suffix_ids=torch.tensor([5, 6], dtype=torch.long),
            tokenizer=tokenizer,
            max_new_tokens=10,
        )
        assert result.content_ids[0].tolist() == [10, 20, 30]
