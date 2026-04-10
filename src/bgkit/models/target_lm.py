"""Target LLM with LoRA and tool-call injection points.

Wraps Qwen3.5-35B (35B total / ~3B active MoE, hidden dim 2560,
64-layer hybrid: 16x(3x Gated DeltaNet-MoE + 1x gated attention-MoE),
262K context). Same architecture family as our 0.8B encoder/decoder.

Loaded in 4-bit quantization (~18GB) for QLoRA training.
LoRA adapters train in BF16.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TargetLMWithInjection(nn.Module):
    """Target LLM wrapper supporting BgKIT vector injection via tool-call frames.

    Projected BgKIT survivors are injected as tool response embeddings within
    Qwen3.5's native tool-call format:

        [assistant] <tool_call>
        {"name": "bgkit_retrieve_context", "arguments": {"source": "compressed_knowledge"}}
        </tool_call>
        [tool] [projected survivor vectors replace placeholder tokens here]

    This reuses the model's existing tool-call understanding and makes
    knowledge sources individually addressable.

    Position detection uses sentinel-based boundary detection (same pattern as
    ChatReproDataset): build messages with CONTENT_SENTINEL as tool response
    content, render via apply_chat_template(tokenize=False), split on sentinel
    to find prefix/suffix boundaries, tokenize piecewise. See
    ``_build_injection_frame()`` in ``kr_step5_trainer.py``.

    The tool_response_start/end_token_id constructor args are retained for
    reference (e.g., for detecting tool response boundaries in generation)
    but position finding is done externally via the sentinel pattern.
    """

    def __init__(
        self,
        model: nn.Module,
        tool_response_start_token_id: int,
        tool_response_end_token_id: int,
    ):
        super().__init__()
        self.model = model
        self.tool_response_start_token_id = tool_response_start_token_id
        self.tool_response_end_token_id = tool_response_end_token_id

    def inject_survivors(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        projected_survivors: torch.Tensor,
        injection_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Replace token embeddings at injection positions with projected survivors.

        Args:
            input_ids: (batch, seq_len) token ids with placeholder tokens at injection sites.
            input_embeds: (batch, seq_len, hidden_dim) token embeddings from the LLM.
            projected_survivors: (batch, num_survivors, hidden_dim) from ProjectionBlock.
            injection_positions: (batch, num_survivors) indices into seq_len.

        Returns:
            (batch, seq_len, hidden_dim) embeddings with survivors injected.
        """
        # Scatter projected survivors into the embedding sequence
        embeds = input_embeds.clone()
        for b in range(input_ids.size(0)):
            positions = injection_positions[b]
            embeds[b, positions] = projected_survivors[b, : len(positions)]
        return embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        projected_survivors: torch.Tensor | None = None,
        injection_positions: torch.Tensor | None = None,
        **kwargs,
    ):
        """Forward pass with optional BgKIT injection.

        Args:
            input_ids: (batch, seq_len) input token ids.
            projected_survivors: Optional projected BgKIT survivors to inject.
            injection_positions: Positions in input_ids to replace with survivors.
            **kwargs: Passed to the underlying model.

        Returns:
            Model output (CausalLMOutput with logits, loss, etc.).
        """
        if projected_survivors is not None and injection_positions is not None:
            # Get base embeddings from the model's embedding layer
            embed_layer = self.model.get_input_embeddings()
            input_embeds = embed_layer(input_ids)
            # Inject BgKIT survivors at specified positions
            input_embeds = self.inject_survivors(
                input_ids, input_embeds, projected_survivors, injection_positions,
            )
            # Forward with embeddings instead of token ids
            kwargs.pop("input_ids", None)
            return self.model(inputs_embeds=input_embeds, **kwargs)

        return self.model(input_ids=input_ids, **kwargs)
