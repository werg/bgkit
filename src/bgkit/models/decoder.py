"""Reconstruction decoder: Qwen3.5-0.8B (instruct) wrapper.

Co-trained with BgKIT to reconstruct original content from compressed
survivor representations via prefix-conditioning. Survivor embeddings are
prepended to the decoder input and attended to via causal self-attention.

The Qwen3.5 decoder uses the same hybrid architecture (18 DeltaNet + 6 full
attention) as the encoder, but in standard causal mode. DeltaNet layers
provide O(L) inference, reducing KV cache memory for long survivor sequences.

Provides the primary training signal for compression quality across four
objectives:
1. Data reconstruction (primary)
2. Description generation
3. Structural/relational reconstruction
4. Commit reproduction
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GenerationOutput:
    """Structured output from decoder generation."""

    content_ids: list[torch.Tensor]  # per-sample variable-length content token IDs
    content_text: list[str]  # decoded content (convenience)
    full_ids: list[torch.Tensor]  # per-sample complete generation for debugging


class ReconstructionDecoder(nn.Module):
    """Causal LM decoder for reconstructing content from BgKIT survivors.

    Wraps Qwen3.5-0.8B with prefix-conditioning: survivor embeddings are
    prepended to the target sequence and attended to via standard causal
    self-attention. No architectural changes to the underlying model.
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim

    def forward(
        self,
        survivor_embeddings: torch.Tensor,
        target_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        survivor_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Generate logits conditioned on survivor embeddings via prefix-conditioning.

        Args:
            survivor_embeddings: (batch, num_survivors, hidden_dim) from BgKIT.
            target_ids: (batch, target_len) token ids for teacher-forced generation.
            target_attention_mask: (batch, target_len) attention mask for targets.
            survivor_attention_mask: (batch, num_survivors) mask for real survivors.

        Returns:
            (batch, target_len, vocab_size) logits.
        """
        # Get target token embeddings
        target_embeddings = self.backbone.get_input_embeddings()(target_ids)

        # Concatenate [survivor_embeddings | target_embeddings] along seq dim
        combined = torch.cat([survivor_embeddings, target_embeddings], dim=1)

        # Build combined attention mask
        combined_mask = torch.cat([survivor_attention_mask, target_attention_mask], dim=1)

        # Forward through backbone (HF builds causal triangular mask internally)
        # AutoModelForCausalLM returns CausalLMOutput with .logits, not .last_hidden_state
        outputs = self.backbone(inputs_embeds=combined, attention_mask=combined_mask)

        # Slice out target portion logits (skip survivor prefix positions)
        num_survivors = survivor_embeddings.size(1)
        logits = outputs.logits[:, num_survivors:, :]

        return logits

    @torch.no_grad()
    def generate(
        self,
        survivor_embeddings: torch.Tensor,
        survivor_attention_mask: torch.Tensor,
        prefix_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        suffix_ids: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> GenerationOutput:
        """Autoregressive generation conditioned on survivor embeddings.

        Leverages HF's ``model.generate()`` with ``inputs_embeds``. Content is
        extracted by token-level slicing (not regex) using the known suffix.

        Args:
            survivor_embeddings: (batch, num_survivors, hidden_dim) from BgKIT.
            survivor_attention_mask: (batch, num_survivors) mask for real survivors.
            prefix_ids: (batch, prefix_len) chat template prefix token IDs.
            prefix_attention_mask: (batch, prefix_len) mask for prefix tokens.
            suffix_ids: (suffix_len,) 1D constant suffix token IDs (unbatched).
            tokenizer: Tokenizer for decoding output.
            max_new_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature. 0.0 = greedy.

        Returns:
            GenerationOutput with content extracted via token-level boundaries.
        """
        # Embed prefix tokens
        prefix_embeddings = self.backbone.get_input_embeddings()(prefix_ids)

        # Concatenate [survivor_embeddings, prefix_embeddings] -> inputs_embeds
        inputs_embeds = torch.cat([survivor_embeddings, prefix_embeddings], dim=1)

        # Build combined attention mask
        combined_mask = torch.cat([survivor_attention_mask, prefix_attention_mask], dim=1)

        # Generate
        do_sample = temperature > 0
        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": combined_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-8)

        output_ids = self.backbone.generate(**gen_kwargs)

        # HF generate returns IDs including input positions. Slice to new tokens only.
        input_len = inputs_embeds.size(1)
        new_ids = output_ids[:, input_len:]

        # Token-level content extraction: strip suffix from each sample
        suffix_ids = suffix_ids.to(new_ids.device)
        suf_len = suffix_ids.size(0)
        batch_size = new_ids.size(0)

        eos_id = getattr(tokenizer, "eos_token_id", None)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        content_ids_list = []
        for b in range(batch_size):
            seq = new_ids[b]
            # Strip trailing EOS tokens
            if eos_id is not None:
                while seq.size(0) > 0 and seq[-1].item() == eos_id:
                    seq = seq[:-1]
            # Strip trailing pad tokens (using tokenizer's pad_token_id)
            if pad_id is not None:
                while seq.size(0) > 0 and seq[-1].item() == pad_id:
                    seq = seq[:-1]

            if seq.size(0) >= suf_len and seq[-suf_len:].equal(suffix_ids):
                content_ids_list.append(seq[:-suf_len])
            else:
                # Model didn't produce expected suffix (truncated or diverged)
                content_ids_list.append(seq)

        content_text = [
            tokenizer.decode(ids, skip_special_tokens=True) for ids in content_ids_list
        ]
        full_ids = [new_ids[b] for b in range(batch_size)]

        return GenerationOutput(
            content_ids=content_ids_list,
            content_text=content_text,
            full_ids=full_ids,
        )
