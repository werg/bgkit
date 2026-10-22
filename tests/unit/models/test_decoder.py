"""Tests for ReconstructionDecoder single-splice hidden-state path (packed API)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.decoder import ReconstructionDecoder

# ---------------------------------------------------------------------------
# Mock causal LM backbone (with lm_head)
# ---------------------------------------------------------------------------


class _CausalLMOutput:
    """Mimics HF CausalLMOutput with .logits (not .last_hidden_state)."""

    def __init__(self, logits):
        self.logits = logits


class _InnerModel(nn.Module):
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        return type("_InnerOut", (), {"last_hidden_state": self.linear(inputs_embeds)})()


class MockCausalLMBackbone(nn.Module):
    """Tiny causal LM with HF-compatible ``.model`` / ``.lm_head`` nesting."""

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64):
        super().__init__()
        self.model = _InnerModel(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = self.lm_head(out.last_hidden_state)
        return _CausalLMOutput(logits=logits)


# ---------------------------------------------------------------------------
# Helpers — convert old (B, K, D) padded survivors to packed form
# ---------------------------------------------------------------------------


def _pack_survivors(
    survivor_embeddings: torch.Tensor,
    survivor_attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert padded (B, K_max, D) + mask to flat (K_total, D) + cu_seqlens."""
    batch_size = survivor_embeddings.shape[0]
    flat_list = []
    lengths = []
    for b in range(batch_size):
        mask_b = survivor_attention_mask[b]  # (K_max,)
        k_i = int(mask_b.sum().item())
        flat_list.append(survivor_embeddings[b, :k_i])
        lengths.append(k_i)
    if flat_list:
        flat = torch.cat(flat_list, dim=0)
    else:
        flat = survivor_embeddings.new_zeros(0, survivor_embeddings.shape[-1])
    cu = torch.zeros(batch_size + 1, dtype=torch.int32)
    for i, length in enumerate(lengths):
        cu[i + 1] = cu[i] + length
    return flat, cu


# ---------------------------------------------------------------------------
# Decoder single-splice tests
# ---------------------------------------------------------------------------


class TestReconstructionDecoderSingleSplice:
    @pytest.fixture()
    def decoder(self):
        hidden_dim = 64
        backbone = MockCausalLMBackbone(vocab_size=1000, hidden_dim=hidden_dim)
        return ReconstructionDecoder(backbone, hidden_dim=hidden_dim)

    def test_hidden_output_shape(self, decoder):
        """Hidden states should cover survivors and target tokens."""
        batch_size, num_survivors, target_len = 2, 5, 10
        survivors = torch.randn(batch_size, num_survivors, 64)
        target_ids = torch.randint(0, 1000, (batch_size, target_len))
        survivor_mask = torch.ones(batch_size, num_survivors, dtype=torch.bool)
        flat_surv, cu = _pack_survivors(survivors, survivor_mask)

        # No prefix; all target tokens as suffix
        output = decoder.forward_with_single_splice(
            survivor_embeddings=flat_surv,
            survivor_cu_seqlens=cu,
            prefix_ids=[torch.zeros(0, dtype=torch.long)] * batch_size,
            suffix_ids=[target_ids[b] for b in range(batch_size)],
            return_hidden_states=True,
        )

        # Each sample has num_survivors + target_len tokens, packed as (1, N_total, D)
        n_total = batch_size * (num_survivors + target_len)
        assert output.hidden_states.shape == (1, n_total, 64)

    def test_padded_survivors(self, decoder):
        """Should work with variable survivor counts across batch."""
        batch_size = 2
        survivors = torch.randn(batch_size, 6, 64)
        target_ids = torch.randint(0, 1000, (batch_size, 8))
        survivor_mask = torch.tensor(
            [
                [True, True, True, True, True, True],
                [True, True, True, False, False, False],
            ]
        )
        flat_surv, cu = _pack_survivors(survivors, survivor_mask)

        output = decoder.forward_with_single_splice(
            survivor_embeddings=flat_surv,
            survivor_cu_seqlens=cu,
            prefix_ids=[torch.zeros(0, dtype=torch.long)] * batch_size,
            suffix_ids=[target_ids[b] for b in range(batch_size)],
            return_hidden_states=True,
        )
        # Sample 0: 6 survivors + 8 suffix = 14; sample 1: 3 survivors + 8 suffix = 11; total = 25
        assert output.hidden_states.shape[0] == 1
        assert output.hidden_states.shape[2] == 64

    def test_gradient_flows_through_decoder(self, decoder):
        """Gradients should flow through decoder parameters."""
        survivors = torch.randn(1, 3, 64)
        target_ids = torch.randint(0, 1000, (1, 5))
        survivor_mask = torch.ones(1, 3, dtype=torch.bool)
        flat_surv, cu = _pack_survivors(survivors, survivor_mask)

        loss = decoder.forward_with_single_splice(
            survivor_embeddings=flat_surv,
            survivor_cu_seqlens=cu,
            prefix_ids=[torch.zeros(0, dtype=torch.long)],
            suffix_ids=[target_ids[0]],
        )
        loss.backward()

        # Decoder backbone params should have gradients
        assert decoder.backbone.model.linear.weight.grad is not None
        assert decoder.backbone.lm_head.weight.grad is not None

    def test_no_gradient_to_frozen_survivors(self, decoder):
        """Frozen survivor inputs should have no gradient."""
        survivors = torch.randn(1, 3, 64)  # no requires_grad
        target_ids = torch.randint(0, 1000, (1, 5))
        survivor_mask = torch.ones(1, 3, dtype=torch.bool)
        flat_surv, cu = _pack_survivors(survivors, survivor_mask)

        loss = decoder.forward_with_single_splice(
            survivor_embeddings=flat_surv,
            survivor_cu_seqlens=cu,
            prefix_ids=[torch.zeros(0, dtype=torch.long)],
            suffix_ids=[target_ids[0]],
        )
        loss.backward()

        assert survivors.grad is None

    def test_single_sample(self, decoder):
        """Should work with batch_size=1."""
        survivors = torch.randn(1, 4, 64)
        target_ids = torch.randint(0, 1000, (1, 6))
        survivor_mask = torch.ones(1, 4, dtype=torch.bool)
        flat_surv, cu = _pack_survivors(survivors, survivor_mask)

        output = decoder.forward_with_single_splice(
            survivor_embeddings=flat_surv,
            survivor_cu_seqlens=cu,
            prefix_ids=[torch.zeros(0, dtype=torch.long)],
            suffix_ids=[target_ids[0]],
            return_hidden_states=True,
        )
        # 4 survivors + 6 suffix = 10, packed as (1, 10, 64)
        assert output.hidden_states.shape == (1, 10, 64)

    def test_loss_finite(self, decoder):
        """The single-splice loss should be finite."""
        survivors = torch.randn(2, 3, 64)
        target_ids = torch.randint(0, 1000, (2, 5))
        survivor_mask = torch.ones(2, 3, dtype=torch.bool)
        flat_surv, cu = _pack_survivors(survivors, survivor_mask)

        loss = decoder.forward_with_single_splice(
            survivor_embeddings=flat_surv,
            survivor_cu_seqlens=cu,
            prefix_ids=[torch.zeros(0, dtype=torch.long)] * 2,
            suffix_ids=[target_ids[b] for b in range(2)],
        )
        assert torch.isfinite(loss)
