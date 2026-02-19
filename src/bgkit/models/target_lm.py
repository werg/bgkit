"""Target LLM with LoRA and tool-call injection points.

Wraps Qwen3-Coder-Next (80B total / 3B active MoE, hidden dim 2048,
48-layer hybrid: 12x(3x gated DeltaNet-MoE + 1x gated attention-MoE),
512 experts, 10 active + 1 shared per token, 256K context).

Loaded in 4-bit quantization (~40GB) for QLoRA training.
LoRA adapters train in BF16.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TargetLMWithInjection(nn.Module):
    """Target LLM wrapper supporting BgKIT vector injection via tool-call frames.

    Projected BgKIT survivors are injected as tool-call response embeddings:
        <tool_call>bgkit_repo_contents</tool_call>
        <tool_response>[projected survivor vectors]</tool_response>

    This reuses the model's existing tool-call understanding and makes
    knowledge sources individually addressable.

    Implementation note — chat template integration:
        The injection_positions must correspond to the exact token span of the
        <tool_response> content within a tokenizer.apply_chat_template() output.
        Use the same sentinel-based boundary detection pattern as ChatReproDataset
        (see src/bgkit/data/datasets/chat_repro_dataset.py) to locate the tool
        response region:
        1. Build messages with a sentinel string as the tool response content
        2. Call tokenizer.apply_chat_template(messages, tokenize=False)
        3. Split on sentinel to find prefix/suffix boundaries
        4. Tokenize piecewise to get exact token positions
        The tool_response_start/end_token_id constructor args are insufficient
        for this — they identify the *markers* but not the content span between
        them. Consider replacing with a template-aware position finder.
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
    ) -> torch.Tensor:
        """Forward pass with optional BgKIT injection.

        Args:
            input_ids: (batch, seq_len) input token ids.
            projected_survivors: Optional projected BgKIT survivors to inject.
            injection_positions: Positions in input_ids to replace with survivors.
            **kwargs: Passed to the underlying model.

        Returns:
            Model output (logits).
        """
        # TODO: Get embeddings from model, inject survivors, run forward pass
        raise NotImplementedError
