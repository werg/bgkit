"""Tests for ReconstructionDecoder.generate_with_single_splice() — packed B=1 loop.

The new generate loop calls backbone.model.forward() (the inner model) per step,
with use_cache=True. The mock backbone is designed to:
  - Return deterministic logits from lm_head applied to a scripted token sequence
  - Support use_cache + past_key_values (tracked as a call counter)
  - Support both inputs_embeds (prefill) and input_ids (decode steps)
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.decoder import GenerationOutput, ReconstructionDecoder

# ---------------------------------------------------------------------------
# Mock inner model that drives deterministic generation
# ---------------------------------------------------------------------------


class _ModelOut:
    def __init__(self, last_hidden_state, past_key_values=None):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values


class _MockInner(nn.Module):
    """Inner model (backbone.model) that returns scripted hidden states.

    For decode steps (use_cache=True), the last position's hidden state is set
    so that the lm_head argmax always picks ``output_seq[step]``.
    """

    def __init__(self, vocab_size: int, hidden_dim: int, output_seq: list[int]):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim
        self._output_seq = output_seq
        self._step = 0
        # Scripted hidden directions: each entry is a unit vector that will be
        # matched by the lm_head to give high logit for that token.
        # We set up _scripted_hiddens lazily after the test wires up lm_head.
        self._lm_head: nn.Linear | None = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> _ModelOut:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        x = self.linear(inputs_embeds)  # (1, L, D)

        # Override last position hidden state to steer greedy decode.
        if use_cache and self._lm_head is not None and self._step < len(self._output_seq):
            tok = self._output_seq[self._step]
            self._step += 1
            # Build a hidden vector h such that lm_head(h)[tok] is the largest.
            # Easiest: set h = lm_head.weight[tok] direction (unnormalised).
            with torch.no_grad():
                w = self._lm_head.weight  # (V, D)
                h_override = w[tok].clone().unsqueeze(0).unsqueeze(0)  # (1, 1, D)
                h_override = h_override * 100.0  # amplify to dominate all other rows
            x = x.clone()
            x[:, -1:, :] = h_override.to(x.dtype)

        # Simulate KV-cache as a simple integer counter.
        pkv = (past_key_values[0] + 1 if past_key_values else 1,) if use_cache else None
        return _ModelOut(last_hidden_state=x, past_key_values=pkv)


class MockGenerativeLM(nn.Module):
    """Tiny causal LM for testing the custom decode loop.

    backbone.model is a _MockInner that overrides hidden states to steer greedy
    generation toward a scripted output sequence.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        hidden_dim: int = 32,
        fixed_output_ids: list[int] | None = None,
    ):
        super().__init__()
        output_seq = list(fixed_output_ids or [10, 20, 30, 5, 6, 2])
        self.model = _MockInner(vocab_size, hidden_dim, output_seq)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.config = type("Config", (), {"eos_token_id": 2})()
        # Wire lm_head into the inner model for hidden-state overrides.
        self.model._lm_head = self.lm_head

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        out = self.model(inputs_embeds=inputs_embeds)

        class _LMOut:
            def __init__(self, lh, pkv):
                self.last_hidden_state = lh
                self.past_key_values = pkv

        return _LMOut(out.last_hidden_state, None)


class MockTokenizer:
    """Minimal tokenizer mock."""

    eos_token_id = 2
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=False):
        if skip_special_tokens:
            ids = [i for i in ids.tolist() if i != self.eos_token_id]
        else:
            ids = ids.tolist()
        return " ".join(str(i) for i in ids)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_single_sample_inputs(
    vocab_size: int = 100,
    hidden_dim: int = 32,
    num_survivors: int = 3,
    prefix_len: int = 2,
):
    """Return (survivor_embeddings, survivor_cu_seqlens, prefix_ids)."""
    surv = torch.randn(num_survivors, hidden_dim)
    cu = torch.tensor([0, num_survivors], dtype=torch.int32)
    pre = torch.randint(0, vocab_size, (prefix_len,))
    return surv, cu, pre


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tokenizer():
    return MockTokenizer()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerationOutput:
    def test_generate_returns_generation_output(self, tokenizer):
        """generate_with_single_splice returns a GenerationOutput."""
        backbone = MockGenerativeLM(vocab_size=100, hidden_dim=32)
        dec = ReconstructionDecoder(backbone, hidden_dim=32)
        surv, cu, pre = _make_single_sample_inputs()
        result = dec.generate_with_single_splice(
            survivor_embeddings=surv,
            survivor_cu_seqlens=cu,
            prefix_ids=pre,
            suffix_ids=torch.tensor([5, 6], dtype=torch.long),
            tokenizer=tokenizer,
            max_new_tokens=10,
        )
        assert isinstance(result, GenerationOutput)
        # B=1 → one element per list
        assert len(result.content_ids) == 1
        assert len(result.content_text) == 1
        assert len(result.full_ids) == 1

    def test_content_ids_excludes_suffix(self, tokenizer):
        """Tokens ending with suffix are stripped from content_ids."""
        # Scripted output: [10, 20, 30, 5, 6, 2]
        # After stripping EOS=2: [10, 20, 30, 5, 6]
        # After stripping suffix=[5,6]: [10, 20, 30]
        backbone = MockGenerativeLM(
            vocab_size=100,
            hidden_dim=32,
            fixed_output_ids=[10, 20, 30, 5, 6, 2],
        )
        dec = ReconstructionDecoder(backbone, hidden_dim=32)
        surv, cu, pre = _make_single_sample_inputs()
        result = dec.generate_with_single_splice(
            survivor_embeddings=surv,
            survivor_cu_seqlens=cu,
            prefix_ids=pre,
            suffix_ids=torch.tensor([5, 6], dtype=torch.long),
            tokenizer=tokenizer,
            max_new_tokens=10,
        )
        assert result.content_ids[0].tolist() == [10, 20, 30]

    def test_fallback_when_no_suffix_match(self, tokenizer):
        """When model doesn't produce the expected suffix, all non-EOS tokens kept."""
        backbone = MockGenerativeLM(
            vocab_size=100,
            hidden_dim=32,
            fixed_output_ids=[10, 20, 30, 40, 50, 2],
        )
        dec = ReconstructionDecoder(backbone, hidden_dim=32)
        surv, cu, pre = _make_single_sample_inputs()
        result = dec.generate_with_single_splice(
            survivor_embeddings=surv,
            survivor_cu_seqlens=cu,
            prefix_ids=pre,
            suffix_ids=torch.tensor([99, 98], dtype=torch.long),
            tokenizer=tokenizer,
            max_new_tokens=10,
        )
        # No suffix stripped — all non-EOS tokens kept
        assert result.content_ids[0].tolist() == [10, 20, 30, 40, 50]

    def test_greedy_is_deterministic(self, tokenizer):
        """Temperature=0 should produce identical output on two calls from the same model."""
        backbone1 = MockGenerativeLM(
            vocab_size=100,
            hidden_dim=32,
            fixed_output_ids=[10, 20, 30, 5, 6, 2],
        )
        backbone2 = MockGenerativeLM(
            vocab_size=100,
            hidden_dim=32,
            fixed_output_ids=[10, 20, 30, 5, 6, 2],
        )
        dec1 = ReconstructionDecoder(backbone1, hidden_dim=32)
        dec2 = ReconstructionDecoder(backbone2, hidden_dim=32)
        # Copy weights so numerically identical.
        dec2.load_state_dict(dec1.state_dict())
        # Also reset lm_head reference in inner model of dec2.
        dec2.backbone.model._lm_head = dec2.backbone.lm_head

        surv, cu, pre = _make_single_sample_inputs()
        suffix = torch.tensor([5, 6], dtype=torch.long)

        r1 = dec1.generate_with_single_splice(
            survivor_embeddings=surv,
            survivor_cu_seqlens=cu,
            prefix_ids=pre,
            suffix_ids=suffix,
            tokenizer=tokenizer,
            max_new_tokens=10,
            temperature=0.0,
        )
        r2 = dec2.generate_with_single_splice(
            survivor_embeddings=surv.clone(),
            survivor_cu_seqlens=cu,
            prefix_ids=pre.clone(),
            suffix_ids=suffix.clone(),
            tokenizer=tokenizer,
            max_new_tokens=10,
            temperature=0.0,
        )
        assert r1.content_ids[0].tolist() == r2.content_ids[0].tolist()

    def test_content_text_is_nonempty(self, tokenizer):
        backbone = MockGenerativeLM(vocab_size=100, hidden_dim=32)
        dec = ReconstructionDecoder(backbone, hidden_dim=32)
        surv, cu, pre = _make_single_sample_inputs()
        result = dec.generate_with_single_splice(
            survivor_embeddings=surv,
            survivor_cu_seqlens=cu,
            prefix_ids=pre,
            suffix_ids=torch.tensor([5, 6], dtype=torch.long),
            tokenizer=tokenizer,
            max_new_tokens=10,
        )
        assert len(result.content_text[0]) > 0

    def test_generate_with_single_splice_strips_suffix(self, tokenizer):
        backbone = MockGenerativeLM(
            vocab_size=100,
            hidden_dim=32,
            fixed_output_ids=[10, 20, 30, 5, 6, 2],
        )
        dec = ReconstructionDecoder(backbone, hidden_dim=32)
        surv, cu, pre = _make_single_sample_inputs()
        result = dec.generate_with_single_splice(
            survivor_embeddings=surv,
            survivor_cu_seqlens=cu,
            prefix_ids=pre,
            suffix_ids=torch.tensor([5, 6], dtype=torch.long),
            tokenizer=tokenizer,
            max_new_tokens=10,
        )
        assert result.content_ids[0].tolist() == [10, 20, 30]

    def test_eos_stops_generation(self, tokenizer):
        """Generation stops immediately when EOS is produced."""
        backbone = MockGenerativeLM(
            vocab_size=100,
            hidden_dim=32,
            fixed_output_ids=[2],  # EOS right away
        )
        dec = ReconstructionDecoder(backbone, hidden_dim=32)
        surv, cu, pre = _make_single_sample_inputs()
        result = dec.generate_with_single_splice(
            survivor_embeddings=surv,
            survivor_cu_seqlens=cu,
            prefix_ids=pre,
            suffix_ids=torch.zeros(0, dtype=torch.long),
            tokenizer=tokenizer,
            max_new_tokens=20,
        )
        # EOS is generated then stripped; content should be empty
        assert result.content_ids[0].tolist() == []
