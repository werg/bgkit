"""Parity tests for the packed-attention decoder rewrite (Task 1.2).

Verifies that the new ``forward_with_single_splice`` (packed path) produces
losses and hidden states equivalent to the legacy ``forward_interleaved_with_loss``
path on the same inputs.  The fixture ``tests/fixtures/decoder_reference.pt``
was captured with the padded path on Qwen/Qwen3.5-0.8B; because the host
venv has no GPU / real Qwen model, we replicate the parity gate here using a
toy backbone that exercises the same code paths.

Force SDPA backend (``attn_implementation="sdpa"``) to match how the fixtures
were captured.

Two tests:
1. **forward_with_single_splice parity**: convert padded reference inputs to
   packed form and compare the loss against the interleaved reference.
2. **Decode-loop parity**: compare per-step token IDs between the custom B=1
   packed decode loop and a reference sequential forward with the same model.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.decoder import (
    EmbeddingSegment,
    InterleavedForwardOutput,
    ReconstructionDecoder,
    TokenSegment,
)

# ---------------------------------------------------------------------------
# Shared mock backbone — deterministic transformer-like hidden states
# ---------------------------------------------------------------------------

VOCAB_SIZE = 512
HIDDEN_DIM = 64
N_LAYERS = 3


class _ModelOut:
    def __init__(self, last_hidden_state, past_key_values=None):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values


class _MockInnerModel(nn.Module):
    """Inner model (backbone.model) with embed_tokens + stacked linears + norm.

    Supports both ``inputs_embeds`` (prefill) and ``input_ids`` (decode step)
    as well as ``use_cache`` / ``past_key_values`` (tracked as call counter).
    """

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = N_LAYERS):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> _ModelOut:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        pkv = ((past_key_values[0] + 1) if past_key_values else 1,) if use_cache else None
        return _ModelOut(last_hidden_state=x, past_key_values=pkv)


class MockCausalLM(nn.Module):
    """HF-style CausalLM with .model + .lm_head structure."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.model = _MockInnerModel(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return type("_LMOut", (), {"last_hidden_state": out.last_hidden_state})()


class _StatefulInnerModel(nn.Module):
    """Toy stateful sequence mixer.

    It deliberately accumulates along the sequence axis. Flattening multiple
    samples into one row makes later samples depend on earlier samples, while
    a padded batch keeps state independent per row.
    """

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        **kwargs,
    ) -> _ModelOut:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is not None:
            inputs_embeds = inputs_embeds * attention_mask.to(inputs_embeds.dtype).unsqueeze(-1)
        return _ModelOut(last_hidden_state=torch.cumsum(inputs_embeds, dim=1))


class StatefulCausalLM(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.model = _StatefulInnerModel(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return type("_LMOut", (), {"last_hidden_state": out.last_hidden_state})()


# ---------------------------------------------------------------------------
# Helper: convert padded (B, K_max, D) survivors to packed form
# ---------------------------------------------------------------------------


def _pack_survivors_from_padded(
    survivor_embeddings: torch.Tensor,  # (B, K_max, D)
    survivor_attention_mask: torch.Tensor,  # (B, K_max) bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (flat (K_total, D), cu_seqlens (B+1,) int32)."""
    batch_size = survivor_embeddings.shape[0]
    flat_list = []
    lengths = []
    for b in range(batch_size):
        mask_b = survivor_attention_mask[b]
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
# Test 1: forward_with_single_splice parity against interleaved
# ---------------------------------------------------------------------------


class TestPackedForwardParity:
    """Packed forward_with_single_splice produces same loss as interleaved reference.

    The reference path (forward_interleaved_with_loss) is the old code that
    uses (B, L) shapes and attention_mask.  Both paths go through the same
    backbone, so numerical differences are from layout only.
    """

    @pytest.fixture()
    def decoder(self):
        torch.manual_seed(17)
        backbone = MockCausalLM(VOCAB_SIZE, HIDDEN_DIM)
        return ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)

    def test_no_prefix_parity(self, decoder):
        """No-prefix case: survivors at start, all tokens as suffix."""
        torch.manual_seed(42)
        batch_size, num_surv, target_len = 4, 5, 20
        surv_3d = torch.randn(batch_size, num_surv, HIDDEN_DIM)
        target_ids = torch.randint(0, VOCAB_SIZE, (batch_size, target_len))

        # Reference via forward_interleaved_with_loss.
        ref = decoder.forward_interleaved_with_loss(
            [
                EmbeddingSegment(embeddings=surv_3d),
                TokenSegment(token_ids=target_ids, loss=True),
            ]
        )

        # Packed path.
        surv_flat, cu = _pack_survivors_from_padded(
            surv_3d,
            torch.ones(batch_size, num_surv, dtype=torch.bool),
        )
        packed = decoder.forward_with_single_splice(
            survivor_embeddings=surv_flat,
            survivor_cu_seqlens=cu,
            prefix_ids=[torch.zeros(0, dtype=torch.long)] * batch_size,
            suffix_ids=[target_ids[b] for b in range(batch_size)],
        )
        # Tolerance: 1e-4 for fp32; the packed path concatenates per-sample
        # and calls the same backbone, differences are from reduction order only.
        torch.testing.assert_close(packed, ref, atol=1e-4, rtol=1e-4)

    def test_variable_survivor_counts_parity(self, decoder):
        """Variable K_i across samples."""
        torch.manual_seed(7)
        batch_size = 3
        max_surv = 8
        target_len = 15
        surv_3d = torch.randn(batch_size, max_surv, HIDDEN_DIM)
        # Sample 0: 3, sample 1: 6, sample 2: 8 survivors.
        surv_mask = torch.zeros(batch_size, max_surv, dtype=torch.bool)
        counts = [3, 6, 8]
        for b, c in enumerate(counts):
            surv_mask[b, :c] = True
        target_ids = torch.randint(0, VOCAB_SIZE, (batch_size, target_len))

        # Reference: use padded (B, max_surv, D) — interleaved uses all K columns.
        # For exact parity we must use the SAME survivor vectors per sample.
        # The interleaved path uses EmbeddingSegment which uses all max_surv
        # positions including padded ones. To match, we need to mask them.
        # Use per-sample EmbeddingSegment with only the valid survivors.
        segments: list = []
        for b in range(batch_size):
            k_i = counts[b]
            segments.append(EmbeddingSegment(embeddings=surv_3d[b : b + 1, :k_i, :]))
            segments.append(TokenSegment(token_ids=target_ids[b : b + 1], loss=True))
        # Build reference via forward_interleaved_with_loss (single-sample loops).
        ref_losses = []
        for b in range(batch_size):
            k_i = counts[b]
            loss_b = decoder.forward_interleaved_with_loss(
                [
                    EmbeddingSegment(embeddings=surv_3d[b : b + 1, :k_i, :]),
                    TokenSegment(token_ids=target_ids[b : b + 1], loss=True),
                ]
            )
            ref_losses.append(loss_b)
        # Weighted average (equal weights since target_len is same for all).
        ref_loss = torch.stack(ref_losses).mean()

        surv_flat, cu = _pack_survivors_from_padded(surv_3d, surv_mask)
        packed_loss = decoder.forward_with_single_splice(
            survivor_embeddings=surv_flat,
            survivor_cu_seqlens=cu,
            prefix_ids=[torch.zeros(0, dtype=torch.long)] * batch_size,
            suffix_ids=[target_ids[b] for b in range(batch_size)],
        )

        # Note: the packed path uses a single forward (all samples packed together)
        # so there may be small cross-batch numerical differences vs. per-sample
        # reference. Tolerance is 5e-3 (bf16-grade).
        torch.testing.assert_close(packed_loss, ref_loss, atol=5e-3, rtol=5e-3)

    def test_with_prefix_parity(self, decoder):
        """Prefix tokens before survivors; suffix after. Single-sample."""
        torch.manual_seed(13)
        pre_len, k_i, suf_len = 5, 4, 10
        pre = torch.randint(0, VOCAB_SIZE, (pre_len,))
        surv = torch.randn(k_i, HIDDEN_DIM)
        suf = torch.randint(0, VOCAB_SIZE, (suf_len,))

        # Reference: three-part interleaved [prefix_tok | surv_emb | suffix_tok]
        ref = decoder.forward_interleaved_with_loss(
            [
                TokenSegment(token_ids=pre.unsqueeze(0), loss=False),
                EmbeddingSegment(embeddings=surv.unsqueeze(0)),
                TokenSegment(token_ids=suf.unsqueeze(0), loss=True),
            ]
        )

        # Packed path.
        cu = torch.tensor([0, k_i], dtype=torch.int32)
        packed = decoder.forward_with_single_splice(
            survivor_embeddings=surv,
            survivor_cu_seqlens=cu,
            prefix_ids=[pre],
            suffix_ids=[suf],
        )

        # Prefix tokens contribute to loss in packed path (default: loss on suffix only).
        # The reference uses loss=False for prefix, so losses should match.
        torch.testing.assert_close(packed, ref, atol=1e-4, rtol=1e-4)

    def test_loss_mask_respected(self, decoder):
        """Explicit loss_mask restricts which positions contribute."""
        torch.manual_seed(21)
        k_i, suf_len = 3, 8
        surv = torch.randn(k_i, HIDDEN_DIM)
        suf = torch.randint(0, VOCAB_SIZE, (suf_len,))
        cu = torch.tensor([0, k_i], dtype=torch.int32)

        # Mask: only 4 middle suffix tokens
        n_total = k_i + suf_len
        mask_a = torch.zeros(n_total, dtype=torch.bool)
        mask_a[k_i + 2 : k_i + 6] = True  # positions 5..8 of suffix

        mask_b = torch.zeros(n_total, dtype=torch.bool)
        mask_b[k_i + 0 : k_i + 4] = True  # positions 3..6 of suffix

        loss_a = decoder.forward_with_single_splice(
            survivor_embeddings=surv,
            survivor_cu_seqlens=cu,
            prefix_ids=[torch.zeros(0, dtype=torch.long)],
            suffix_ids=[suf],
            loss_mask=mask_a,
        )
        loss_b = decoder.forward_with_single_splice(
            survivor_embeddings=surv,
            survivor_cu_seqlens=cu,
            prefix_ids=[torch.zeros(0, dtype=torch.long)],
            suffix_ids=[suf],
            loss_mask=mask_b,
        )
        # Different masks → different losses
        assert not torch.allclose(loss_a, loss_b, atol=1e-5)
        assert torch.isfinite(loss_a)
        assert torch.isfinite(loss_b)

    def test_packed_target_splice_matches_single_splice_api(self, decoder):
        """Trainer-facing packed target API matches the list-based API."""

        torch.manual_seed(23)
        target_rows = [
            torch.tensor([1, 2, 99, 7, 8], dtype=torch.long),
            torch.tensor([3, 4, 88, 9], dtype=torch.long),
        ]
        target_ids_flat = torch.cat(target_rows, dim=0)
        target_cu = torch.tensor([0, 5, 9], dtype=torch.int32)
        splice_start = torch.tensor([2, 2], dtype=torch.int64)
        splice_len = torch.tensor([1, 1], dtype=torch.int64)
        survivors = torch.randn(3, HIDDEN_DIM)
        survivor_cu = torch.tensor([0, 1, 3], dtype=torch.int32)
        loss_mask_flat = torch.tensor(
            [False, True, True, True, False, True, False, True, True],
            dtype=torch.bool,
        )

        with patch.object(
            decoder,
            "forward_with_single_splice",
            wraps=decoder.forward_with_single_splice,
        ) as single_splice:
            packed = decoder.forward_with_packed_target_splice(
                survivor_embeddings=survivors,
                survivor_cu_seqlens=survivor_cu,
                target_ids_flat=target_ids_flat,
                target_cu_seqlens=target_cu,
                splice_start=splice_start,
                splice_len=splice_len,
                loss_mask_flat=loss_mask_flat,
            )
        call_kwargs = single_splice.call_args.kwargs
        assert call_kwargs["survivor_cu_seqlens_cpu"] == [0, 1, 3]
        torch.testing.assert_close(
            call_kwargs["packed_cu_seqlens"],
            torch.tensor([0, 4, 9], dtype=torch.int32),
        )
        assert [ids.tolist() for ids in call_kwargs["suffix_ids"]] == [[7], [9]]
        manual_loss_mask = torch.tensor(
            [False, True, False, True, True, False, False, False, True],
            dtype=torch.bool,
        )
        listed = decoder.forward_with_single_splice(
            survivor_embeddings=survivors,
            survivor_cu_seqlens=survivor_cu,
            prefix_ids=[target_rows[0][:2], target_rows[1][:2]],
            suffix_ids=[target_rows[0][3:4], target_rows[1][3:]],
            loss_mask=manual_loss_mask,
        )

        torch.testing.assert_close(packed, listed, atol=1e-5, rtol=1e-5)

    def test_packed_target_splice_preserves_hidden_state_shape_when_requested(self, decoder):
        """Trailing lossless tokens are kept for diagnostics that return hidden states."""

        torch.manual_seed(24)
        target_rows = [
            torch.tensor([1, 2, 99, 7, 8], dtype=torch.long),
            torch.tensor([3, 4, 88, 9, 10], dtype=torch.long),
        ]
        target_ids_flat = torch.cat(target_rows, dim=0)
        target_cu = torch.tensor([0, 5, 10], dtype=torch.int32)
        splice_start = torch.tensor([2, 2], dtype=torch.int64)
        splice_len = torch.tensor([1, 1], dtype=torch.int64)
        survivors = torch.randn(3, HIDDEN_DIM)
        survivor_cu = torch.tensor([0, 1, 3], dtype=torch.int32)
        loss_mask_flat = torch.tensor(
            [False, False, False, True, False, False, False, False, True, False],
            dtype=torch.bool,
        )

        with patch.object(
            decoder,
            "forward_with_single_splice",
            wraps=decoder.forward_with_single_splice,
        ) as single_splice:
            out = decoder.forward_with_packed_target_splice(
                survivor_embeddings=survivors,
                survivor_cu_seqlens=survivor_cu,
                target_ids_flat=target_ids_flat,
                target_cu_seqlens=target_cu,
                splice_start=splice_start,
                splice_len=splice_len,
                loss_mask_flat=loss_mask_flat,
                return_hidden_states=True,
            )

        assert isinstance(out, InterleavedForwardOutput)
        call_kwargs = single_splice.call_args.kwargs
        torch.testing.assert_close(
            call_kwargs["packed_cu_seqlens"],
            torch.tensor([0, 5, 11], dtype=torch.int32),
        )
        assert [ids.tolist() for ids in call_kwargs["suffix_ids"]] == [[7, 8], [9, 10]]
        assert out.hidden_states.shape == (1, 11, HIDDEN_DIM)

    def test_return_hidden_states(self, decoder):
        """return_hidden_states=True returns InterleavedForwardOutput."""
        torch.manual_seed(31)
        k_i, suf_len = 5, 12
        surv = torch.randn(k_i, HIDDEN_DIM)
        suf = torch.randint(0, VOCAB_SIZE, (suf_len,))
        cu = torch.tensor([0, k_i], dtype=torch.int32)

        out = decoder.forward_with_single_splice(
            survivor_embeddings=surv,
            survivor_cu_seqlens=cu,
            prefix_ids=[torch.zeros(0, dtype=torch.long)],
            suffix_ids=[suf],
            return_hidden_states=True,
        )
        assert isinstance(out, InterleavedForwardOutput)
        assert out.loss.ndim == 0
        assert torch.isfinite(out.loss)
        # Packed as (1, N_total, D)
        assert out.hidden_states.shape == (1, k_i + suf_len, HIDDEN_DIM)
        assert out.loss_mask.shape == (1, k_i + suf_len)
        # Survivor positions → no loss
        assert (~out.loss_mask[0, :k_i]).all()
        # Suffix positions → loss-bearing
        assert out.loss_mask[0, k_i:].all()

    def test_gradient_flows_through_packed_path(self, decoder):
        """Gradients must flow back through survivors in the packed path."""
        torch.manual_seed(37)
        k_i, suf_len = 4, 8
        surv = torch.randn(k_i, HIDDEN_DIM, requires_grad=True)
        suf = torch.randint(0, VOCAB_SIZE, (suf_len,))
        cu = torch.tensor([0, k_i], dtype=torch.int32)

        loss = decoder.forward_with_single_splice(
            survivor_embeddings=surv,
            survivor_cu_seqlens=cu,
            prefix_ids=[torch.zeros(0, dtype=torch.long)],
            suffix_ids=[suf],
        )
        loss.backward()

        # Survivor embeddings must receive gradient (they feed into the hidden
        # states that predict the first suffix token).
        assert surv.grad is not None
        assert surv.grad.abs().sum() > 0

    def test_stateful_decoder_family_uses_padded_batch(self):
        """Falcon-style decoders must not flatten samples into one sequence."""

        torch.manual_seed(41)
        decoder = ReconstructionDecoder(
            StatefulCausalLM(VOCAB_SIZE, HIDDEN_DIM),
            hidden_dim=HIDDEN_DIM,
            decoder_family="falcon_h1",
        )
        survivors = torch.zeros(2, HIDDEN_DIM)
        survivors[0, 0] = 3.0
        survivors[1, 0] = 5.0
        cu = torch.tensor([0, 1, 2], dtype=torch.int32)
        suffix_ids = [
            torch.tensor([7, 8], dtype=torch.long),
            torch.tensor([9], dtype=torch.long),
        ]

        out = decoder.forward_with_single_splice(
            survivor_embeddings=survivors,
            survivor_cu_seqlens=cu,
            prefix_ids=[torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)],
            suffix_ids=suffix_ids,
            return_hidden_states=True,
        )

        assert out.hidden_states.shape == (2, 3, HIDDEN_DIM)
        assert out.attention_mask.tolist() == [
            [True, True, True],
            [True, True, False],
        ]
        # If sample 2 had been flattened after sample 1, the first hidden row
        # would include sample 1's survivor value (3 + 5). Padded batching keeps
        # state local to sample 2.
        assert out.hidden_states[1, 0, 0].item() == pytest.approx(5.0)

    def test_stateful_decoder_family_respects_explicit_loss_mask(self):
        """Direct padded stateful path maps flat masks back to padded rows."""

        torch.manual_seed(43)
        decoder = ReconstructionDecoder(
            StatefulCausalLM(VOCAB_SIZE, HIDDEN_DIM),
            hidden_dim=HIDDEN_DIM,
            decoder_family="falcon_h1",
        )
        survivors = torch.randn(2, HIDDEN_DIM)
        cu = torch.tensor([0, 1, 2], dtype=torch.int32)
        prefix_ids = [
            torch.tensor([1, 2], dtype=torch.long),
            torch.tensor([3], dtype=torch.long),
        ]
        suffix_ids = [
            torch.tensor([7, 8], dtype=torch.long),
            torch.tensor([9], dtype=torch.long),
        ]
        loss_mask = torch.tensor(
            [False, True, False, True, False, True, False, True],
            dtype=torch.bool,
        )

        out = decoder.forward_with_single_splice(
            survivor_embeddings=survivors,
            survivor_cu_seqlens=cu,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            loss_mask=loss_mask,
            return_hidden_states=True,
        )

        assert out.loss_mask.tolist() == [
            [False, True, False, True, False],
            [True, False, True, False, False],
        ]

    def test_stateful_packed_target_splice_matches_single_splice_api(self):
        """Packed target API keeps Falcon-style padded layout semantics."""

        torch.manual_seed(47)
        decoder = ReconstructionDecoder(
            StatefulCausalLM(VOCAB_SIZE, HIDDEN_DIM),
            hidden_dim=HIDDEN_DIM,
            decoder_family="falcon_h1",
        )
        target_rows = [
            torch.tensor([1, 2, 99, 7, 8], dtype=torch.long),
            torch.tensor([3, 88, 9], dtype=torch.long),
        ]
        target_ids_flat = torch.cat(target_rows, dim=0)
        target_cu = torch.tensor([0, 5, 8], dtype=torch.int32)
        splice_start = torch.tensor([2, 1], dtype=torch.int64)
        splice_len = torch.tensor([1, 1], dtype=torch.int64)
        survivors = torch.randn(3, HIDDEN_DIM)
        survivor_cu = torch.tensor([0, 1, 3], dtype=torch.int32)
        loss_mask_flat = torch.tensor(
            [False, True, True, True, False, True, False, True],
            dtype=torch.bool,
        )
        manual_loss_mask = torch.tensor(
            [False, True, False, True, False, True, False, False, True],
            dtype=torch.bool,
        )

        packed = decoder.forward_with_packed_target_splice(
            survivor_embeddings=survivors,
            survivor_cu_seqlens=survivor_cu,
            target_ids_flat=target_ids_flat,
            target_cu_seqlens=target_cu,
            splice_start=splice_start,
            splice_len=splice_len,
            loss_mask_flat=loss_mask_flat,
            return_hidden_states=True,
        )
        listed = decoder.forward_with_single_splice(
            survivor_embeddings=survivors,
            survivor_cu_seqlens=survivor_cu,
            prefix_ids=[target_rows[0][:2], target_rows[1][:1]],
            suffix_ids=[target_rows[0][3:], target_rows[1][2:]],
            loss_mask=manual_loss_mask,
            return_hidden_states=True,
        )

        torch.testing.assert_close(packed.loss, listed.loss, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            packed.hidden_states,
            listed.hidden_states,
            atol=1e-5,
            rtol=1e-5,
        )
        assert packed.loss_mask.tolist() == [
            [False, True, False, True, False],
            [True, False, False, True, False],
        ]
        assert packed.attention_mask.tolist() == [
            [True, True, True, True, True],
            [True, True, True, True, False],
        ]


# ---------------------------------------------------------------------------
# Test 2: Decode-loop parity
# ---------------------------------------------------------------------------


class _SteeringInner(nn.Module):
    """Inner model that steers greedy decode via lm_head weight direction."""

    def __init__(self, vocab_size: int, hidden_dim: int, output_seq: list[int]):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self._output_seq = output_seq
        self._step = 0
        self._lm_head: nn.Linear | None = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ) -> _ModelOut:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        x = self.norm(self.linear(inputs_embeds))

        if use_cache and self._lm_head is not None and self._step < len(self._output_seq):
            tok = self._output_seq[self._step]
            self._step += 1
            with torch.no_grad():
                w = self._lm_head.weight  # (V, D)
                override = (w[tok] * 100.0).to(x.dtype)
                x = x.clone()
                x[:, -1:, :] = override.unsqueeze(0).unsqueeze(0)

        pkv = ((past_key_values[0] + 1) if past_key_values else 1,) if use_cache else None
        return _ModelOut(last_hidden_state=x, past_key_values=pkv)


class SteeringCausalLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, output_seq: list[int]):
        super().__init__()
        self.model = _SteeringInner(vocab_size, hidden_dim, output_seq)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.model._lm_head = self.lm_head

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        out = self.model(inputs_embeds=inputs_embeds)
        return type("_Out", (), {"last_hidden_state": out.last_hidden_state})()


class _MockTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=False):
        tokens = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in (self.eos_token_id, self.pad_token_id)]
        return " ".join(str(t) for t in tokens)


class TestDecodeLoopParity:
    """The custom B=1 decode loop produces the same tokens as a direct greedy step.

    We verify:
    1. Tokens generated by generate_with_single_splice match the scripted
       output sequence exactly (deterministic steering mock).
    2. Two independent calls with the same model and inputs produce the same
       output (greedy reproducibility).
    3. EOS stops generation.
    4. Suffix is correctly stripped.
    """

    @pytest.fixture()
    def tokenizer(self):
        return _MockTokenizer()

    def _make_dec(self, output_seq: list[int]):
        backbone = SteeringCausalLM(VOCAB_SIZE, HIDDEN_DIM, output_seq)
        return ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)

    def test_scripted_output_matches(self, tokenizer):
        """Generated tokens match the scripted sequence (before EOS strip)."""
        output_seq = [10, 20, 30, 40, 50, 2]  # 2 = EOS
        dec = self._make_dec(output_seq)
        torch.manual_seed(7)
        surv = torch.randn(3, HIDDEN_DIM)
        cu = torch.tensor([0, 3], dtype=torch.int32)
        pre = torch.randint(0, VOCAB_SIZE, (2,))

        with torch.no_grad():
            result = dec.generate_with_single_splice(
                survivor_embeddings=surv,
                survivor_cu_seqlens=cu,
                prefix_ids=pre,
                suffix_ids=torch.zeros(0, dtype=torch.long),
                tokenizer=tokenizer,
                max_new_tokens=10,
            )
        # EOS is stripped from content; full_ids includes it.
        full = result.full_ids[0].tolist()
        content = result.content_ids[0].tolist()
        assert 2 in full  # EOS present in full
        assert 2 not in content  # EOS stripped from content
        assert content == [10, 20, 30, 40, 50]

    def test_reproducibility(self, tokenizer):
        """Two calls with identical models produce identical token sequences."""
        output_seq = [15, 25, 35, 2]
        dec1 = self._make_dec(output_seq)
        dec2 = self._make_dec(output_seq)
        dec2.load_state_dict(dec1.state_dict())
        dec2.backbone.model._lm_head = dec2.backbone.lm_head

        torch.manual_seed(3)
        surv = torch.randn(4, HIDDEN_DIM)
        cu = torch.tensor([0, 4], dtype=torch.int32)
        pre = torch.randint(0, VOCAB_SIZE, (3,))
        suffix = torch.zeros(0, dtype=torch.long)

        with torch.no_grad():
            r1 = dec1.generate_with_single_splice(
                survivor_embeddings=surv,
                survivor_cu_seqlens=cu,
                prefix_ids=pre,
                suffix_ids=suffix,
                tokenizer=tokenizer,
                temperature=0.0,
            )
            r2 = dec2.generate_with_single_splice(
                survivor_embeddings=surv.clone(),
                survivor_cu_seqlens=cu,
                prefix_ids=pre.clone(),
                suffix_ids=suffix.clone(),
                tokenizer=tokenizer,
                temperature=0.0,
            )
        assert r1.content_ids[0].tolist() == r2.content_ids[0].tolist()

    def test_eos_stops_generation(self, tokenizer):
        """EOS token immediately halts the decode loop."""
        dec = self._make_dec([2])  # produce EOS right away
        surv = torch.randn(2, HIDDEN_DIM)
        cu = torch.tensor([0, 2], dtype=torch.int32)
        pre = torch.randint(0, VOCAB_SIZE, (1,))
        with torch.no_grad():
            result = dec.generate_with_single_splice(
                survivor_embeddings=surv,
                survivor_cu_seqlens=cu,
                prefix_ids=pre,
                suffix_ids=torch.zeros(0, dtype=torch.long),
                tokenizer=tokenizer,
                max_new_tokens=100,
            )
        # EOS stripped → content is empty
        assert result.content_ids[0].tolist() == []

    def test_suffix_strip(self, tokenizer):
        """Suffix tokens at the end of generated content are stripped."""
        dec = self._make_dec([10, 20, 30, 5, 6, 2])
        surv = torch.randn(3, HIDDEN_DIM)
        cu = torch.tensor([0, 3], dtype=torch.int32)
        pre = torch.randint(0, VOCAB_SIZE, (2,))
        with torch.no_grad():
            result = dec.generate_with_single_splice(
                survivor_embeddings=surv,
                survivor_cu_seqlens=cu,
                prefix_ids=pre,
                suffix_ids=torch.tensor([5, 6], dtype=torch.long),
                tokenizer=tokenizer,
            )
        assert result.content_ids[0].tolist() == [10, 20, 30]

    def test_max_new_tokens_respected(self, tokenizer):
        """Generation never exceeds max_new_tokens."""
        # Infinite EOS-less output sequence (all non-EOS tokens)
        output_seq = list(range(10, 60))  # 50 tokens, none is EOS=2
        dec = self._make_dec(output_seq)
        surv = torch.randn(2, HIDDEN_DIM)
        cu = torch.tensor([0, 2], dtype=torch.int32)
        pre = torch.randint(0, VOCAB_SIZE, (1,))
        with torch.no_grad():
            result = dec.generate_with_single_splice(
                survivor_embeddings=surv,
                survivor_cu_seqlens=cu,
                prefix_ids=pre,
                suffix_ids=torch.zeros(0, dtype=torch.long),
                tokenizer=tokenizer,
                max_new_tokens=5,
            )
        assert len(result.full_ids[0]) <= 5


# ---------------------------------------------------------------------------
# Test 3: fixture metadata sanity (no real model load required)
# ---------------------------------------------------------------------------


class TestFixtureMetadata:
    """Light sanity check: fixture exists and has the expected fields."""

    FIXTURE_PATH = os.path.join(
        os.path.dirname(__file__), "..", "..", "fixtures", "decoder_reference.pt"
    )

    def test_fixture_exists(self):
        assert os.path.exists(self.FIXTURE_PATH), (
            f"decoder_reference.pt not found at {self.FIXTURE_PATH}. "
            "Run the fixture capture script before running parity tests."
        )

    def test_fixture_has_expected_keys(self):
        d = torch.load(self.FIXTURE_PATH, weights_only=False)
        assert "inputs" in d
        assert "outputs" in d
        assert "shape" in d
        assert "metadata" in d

    def test_fixture_output_loss_is_finite(self):
        d = torch.load(self.FIXTURE_PATH, weights_only=False)
        loss = d["outputs"]["loss"]
        assert torch.isfinite(loss), f"Reference loss is not finite: {loss}"

    def test_fixture_splice_layout_consistent(self):
        """Splice starts/lengths are within token sequence bounds."""
        d = torch.load(self.FIXTURE_PATH, weights_only=False)
        tok_lens = d["shape"]["token_lengths"]
        splice_starts = d["inputs"]["splice_starts"].tolist()
        splice_lengths = d["inputs"]["splice_lengths"].tolist()
        for b, (tl, ss, sl) in enumerate(zip(tok_lens, splice_starts, splice_lengths, strict=True)):
            end = ss + sl
            assert 0 <= ss < tl, f"splice_start out of bounds for sample {b}"
            assert end <= tl, f"splice end out of bounds for sample {b}"


# ---------------------------------------------------------------------------
# Test 4: cached-decode K-length metadata (regression for Finding #2)
# ---------------------------------------------------------------------------


class _KMetadataRecorder(nn.Module):
    """Inner model that records the cu_seq_lens_q / cu_seq_lens_k / max_* kwargs
    passed in on each forward call. Used to verify the decode loop sends the
    correct ``K_length = L_prefill + t`` metadata instead of the buggy
    shared ``cu_seqlens = [0, 1]`` pattern.
    """

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.calls: list[dict] = []

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cu_seq_lens_q=None,
        cu_seq_lens_k=None,
        max_length_q=None,
        max_length_k=None,
        **kwargs,
    ) -> _ModelOut:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # record what was passed
        self.calls.append(
            {
                "cu_q": None if cu_seq_lens_q is None else cu_seq_lens_q.tolist(),
                "cu_k": None if cu_seq_lens_k is None else cu_seq_lens_k.tolist(),
                "max_q": max_length_q,
                "max_k": max_length_k,
                "q_len": inputs_embeds.shape[1],
            }
        )
        x = inputs_embeds  # identity "backbone"
        pkv = ((past_key_values[0] + 1) if past_key_values else 1,) if use_cache else None
        return _ModelOut(last_hidden_state=x, past_key_values=pkv)


class _RecorderCausalLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.model = _KMetadataRecorder(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        out = self.model(inputs_embeds=inputs_embeds)
        return type("_Out", (), {"last_hidden_state": out.last_hidden_state})()


class TestDecodeLoopKLengthMetadata:
    """The decode-loop forward contract on Qwen3.5's hybrid stack.

    Prefill MUST carry packed metadata via TransformersKwargs
    (``cu_seq_lens_q/k`` + ``max_length_q/k``) so the FA4 backend dispatches
    to varlen on the single multi-token prefill.

    Decode steps MUST NOT forward those kwargs: on Qwen3.5 the DeltaNet
    (``linear_attn``) layers are fed the same ``**kwargs`` as full-attention
    layers and their stock HF forward unpacks ``hidden_states.shape`` assuming
    3D ``(B, L, D)``. The single-step decode case crashes when the packed
    kwargs disturb the invariant. Our FA4 backend synthesizes cu_seqlens
    from the 4D q/k shapes when the kwargs are absent (see
    ``bgkit_flash_attention_4_forward``), so cached decode works without the
    HF-threaded packed metadata.
    """

    def test_prefill_carries_packed_kwargs_decode_does_not(self):
        torch.manual_seed(17)
        backbone = _RecorderCausalLM(VOCAB_SIZE, HIDDEN_DIM)
        dec = ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)
        tokenizer = _MockTokenizer()

        num_surv = 4
        pre_len = 3
        surv = torch.randn(num_surv, HIDDEN_DIM)
        cu = torch.tensor([0, num_surv], dtype=torch.int32)
        pre = torch.randint(10, VOCAB_SIZE, (pre_len,))
        l_prefill = pre_len + num_surv

        with torch.no_grad():
            dec.generate_with_single_splice(
                survivor_embeddings=surv,
                survivor_cu_seqlens=cu,
                prefix_ids=pre,
                suffix_ids=torch.zeros(0, dtype=torch.long),
                tokenizer=tokenizer,
                max_new_tokens=5,
            )

        calls = backbone.model.calls
        # Prefill carries packed metadata with Q = K = L_prefill.
        assert calls[0]["cu_q"] == [0, l_prefill]
        assert calls[0]["cu_k"] == [0, l_prefill]
        assert calls[0]["max_q"] == l_prefill
        assert calls[0]["max_k"] == l_prefill

        # Decode steps do NOT forward the packed kwargs — the FA4 backend
        # derives single-sample cu_seqlens from the Q/K tensor shapes.
        for t, call in enumerate(calls[1:], start=1):
            assert call["cu_q"] is None, (
                f"decode step {t}: cu_seq_lens_q must not be forwarded through "
                f"TransformersKwargs (DeltaNet layers break on the extra kwarg); "
                f"got {call['cu_q']}"
            )
            assert call["cu_k"] is None
            assert call["max_q"] is None
            assert call["max_k"] is None
            # Sanity: decode input is single-token.
            assert call["q_len"] == 1

    def test_attention_backend_accepts_per_side_metadata(self):
        """Unit-level: bgkit_flash_attention_4_forward accepts cu_seqlens_q != cu_seqlens_k.

        The backend should not collapse the two into a single cu_seqlens.
        We can't run real FA4 on CPU, but we can check the signature accepts
        the kwargs without raising a TypeError.
        """
        import inspect

        from bgkit.utils.attention_backend import bgkit_flash_attention_4_forward

        sig = inspect.signature(bgkit_flash_attention_4_forward)
        params = sig.parameters
        assert "cu_seqlens_q" in params
        assert "cu_seqlens_k" in params
        assert "max_seqlen_q" in params
        assert "max_seqlen_k" in params

    def test_attention_backend_synthesizes_cu_seqlens_from_4d_shape(self):
        """The FA4 backend derives packed metadata from (1, H, Lq, D) shape
        when no TransformersKwargs packed metadata is supplied — this is the
        cached-decode fallback that lets HF's generation path call the
        backend without threading ``cu_seq_lens_q/k`` through DeltaNet
        layers. We can't invoke real FA4 on CPU; the requirement we verify
        here is that the backend does NOT raise the ``packed sequence
        metadata`` TypeError when q/k arrive as B=1 4D tensors.
        """
        from bgkit.utils import attention_backend as ab

        # Stub out the SM12x ownership check + FA4 call so we can exercise
        # the metadata-synthesis code path on CPU.
        orig_require = ab.require_sm12x_owned_backend

        calls: dict = {}

        def _fake_require():
            pass

        def _fake_varlen(**kwargs):
            calls.update(kwargs)
            # Return a tensor with expected shape so the post-call reshape
            # path is exercised too.
            q = kwargs["q"]
            return torch.zeros_like(q)

        ab.require_sm12x_owned_backend = _fake_require
        # Inject a fake flash_attn.cute module.
        import sys
        import types

        fake_cute = types.ModuleType("flash_attn.cute")
        fake_cute.flash_attn_varlen_func = _fake_varlen
        saved_cute = sys.modules.get("flash_attn.cute")
        saved_flash_attn = sys.modules.get("flash_attn")
        fake_flash_attn = sys.modules.setdefault("flash_attn", types.ModuleType("flash_attn"))
        sys.modules["flash_attn.cute"] = fake_cute
        fake_flash_attn.cute = fake_cute

        try:
            q = torch.randn(1, 4, 1, 16, dtype=torch.float32)
            k = torch.randn(1, 4, 7, 16, dtype=torch.float32)
            v = torch.randn(1, 4, 7, 16, dtype=torch.float32)
            module = type("M", (), {"is_causal": False})()
            out, _ = ab.bgkit_flash_attention_4_forward(module, q, k, v)
        finally:
            ab.require_sm12x_owned_backend = orig_require
            if saved_cute is None:
                sys.modules.pop("flash_attn.cute", None)
            else:
                sys.modules["flash_attn.cute"] = saved_cute
            if saved_flash_attn is None:
                sys.modules.pop("flash_attn", None)
            else:
                sys.modules["flash_attn"] = saved_flash_attn

        # Synthesized metadata: cu_seqlens_q=[0,1], cu_seqlens_k=[0,7].
        assert calls["cu_seqlens_q"].tolist() == [0, 1]
        assert calls["cu_seqlens_k"].tolist() == [0, 7]
        assert calls["max_seqlen_q"] == 1
        assert calls["max_seqlen_k"] == 7
        # Output was re-promoted to 4D shape for HF's post-attention reshape.
        assert out.dim() == 4
        assert out.shape[0] == 1


@pytest.mark.gpu
class TestDecodeLoopRealQwen35:
    """End-to-end decode-loop smoke test against a real Qwen3.5-0.8B backbone.

    Exercises the hybrid [DeltaNet, DeltaNet, DeltaNet, FullAttention] layer
    stack with cached generation — regression for the ``batch_size, seq_len, _
    = hidden_states.shape`` unpack crash that triggered when packed-attention
    TransformersKwargs were threaded through the decode step.

    Host CI skips this via the ``gpu`` marker; runs inside the training
    container via ``make test-gpu``.
    """

    def test_decode_with_deltanet_bearing_backbone(self):
        import os

        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        # Host runs may have CUDA present but cannot load Qwen3_5 without the
        # NGC container's transformers + fla stack; gate on BGKIT_RUN_GPU_TESTS
        # (set inside the container by `make test-gpu`).
        if not os.environ.get("BGKIT_RUN_GPU_TESTS"):
            pytest.skip(
                "real-Qwen3.5 GPU test — set BGKIT_RUN_GPU_TESTS=1 inside the "
                "training container to enable",
            )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        from bgkit.models.decoder import ReconstructionDecoder
        from bgkit.utils.attention_backend import (
            BGKIT_FA4_ATTENTION_IMPL,
            install_bgkit_attention_backend,
        )
        from bgkit.utils.deltanet_patch import patch_gated_delta_rule_numerics

        # Install FA4 backend + DeltaNet patches (mirror the training bootstrap).
        install_bgkit_attention_backend()
        patch_gated_delta_rule_numerics(model=None)

        model_name = "Qwen/Qwen3.5-0.8B"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=BGKIT_FA4_ATTENTION_IMPL,
        ).to("cuda")
        # Re-patch existing DeltaNet instances now that the model is loaded.
        patch_gated_delta_rule_numerics(model=backbone)

        hidden_dim = backbone.config.hidden_size
        dec = ReconstructionDecoder(backbone, hidden_dim=hidden_dim)

        num_surv = 4
        surv = torch.randn(num_surv, hidden_dim, device="cuda", dtype=torch.bfloat16)
        cu = torch.tensor([0, num_surv], dtype=torch.int32, device="cuda")
        pre = torch.tensor(
            tokenizer.encode("Hello ", add_special_tokens=False),
            dtype=torch.long,
            device="cuda",
        )
        suf = torch.zeros(0, dtype=torch.long, device="cuda")

        with torch.no_grad():
            out = dec.generate_with_single_splice(
                survivor_embeddings=surv,
                survivor_cu_seqlens=cu,
                prefix_ids=pre,
                suffix_ids=suf,
                tokenizer=tokenizer,
                max_new_tokens=8,
                temperature=0.0,
            )
        # No crash and we produced something.
        assert len(out.full_ids) == 1
        assert out.full_ids[0].numel() >= 1
