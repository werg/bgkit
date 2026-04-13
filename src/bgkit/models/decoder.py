"""Reconstruction decoder: Qwen3.5-0.8B (instruct) wrapper.

Co-trained with BgKIT to reconstruct original content from compressed
survivor representations. The preferred path splices survivor embeddings
into the decoder sequence at explicit tool-response positions.

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

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

logger = structlog.get_logger()


@dataclass
class TokenSegment:
    """A run of token IDs in an interleaved decoder sequence.

    Every position in a token segment is a candidate loss site for
    next-token prediction. The exact mask depends on:

    - ``loss``: segment-wide flag (default True).
    - ``loss_mask``: optional per-token override. When provided, it wins
      element-wise and ``loss`` is ignored.

    ``token_ids`` and ``loss_mask`` are accepted as either ``(L,)`` or
    ``(B, L)``; unbatched inputs get a leading batch dim added downstream.
    """

    token_ids: torch.Tensor
    loss: bool = True
    loss_mask: torch.Tensor | None = None


@dataclass
class EmbeddingSegment:
    """A run of pre-computed hidden-dim vectors in an interleaved sequence.

    Used to splice BgKIT L1 survivor outputs into the decoder context.
    Embedding-segment positions are never loss-bearing targets (the
    decoder can't predict a vector), but their outputs are legitimate
    *sources* that predict the next token in an adjacent token segment.
    """

    embeddings: torch.Tensor


Segment = TokenSegment | EmbeddingSegment


@dataclass
class InterleavedForwardOutput:
    """Rich output from :meth:`ReconstructionDecoder.forward_interleaved_with_loss`.

    Returned when ``return_hidden_states=True``. Gives the caller
    everything needed to compute metrics (argmax predictions, per-position
    loss contributions, span extraction) without reimplementing any
    decoder internals.
    """

    loss: torch.Tensor  # scalar
    hidden_states: torch.Tensor  # (B, S_total, D)
    token_ids: torch.Tensor  # (B, S_total) long — zeros at embedding-segment positions
    loss_mask: torch.Tensor  # (B, S_total) bool — False at embedding-segment positions
    attention_mask: torch.Tensor  # (B, S_total) bool
    lm_head: nn.Module  # handy for ad-hoc logit computation

    def argmax_predictions(self) -> torch.Tensor:
        """Return ``(B, S_total - 1)`` argmax predictions under next-token shift.

        ``predictions[i]`` is the model's argmax for position ``i+1`` given
        hidden state at position ``i``. Only positions where
        ``loss_mask[i+1] == True`` are valid prediction sites; the caller
        should mask accordingly.
        """
        shift_h = self.hidden_states[:, :-1, :]
        logits = self.lm_head(shift_h)
        return logits.argmax(dim=-1)


def _chunk_ce_fn(
    lm_head_weight: torch.Tensor,
    hidden_chunk: torch.Tensor,
    target_chunk: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute per-token CE for one chunk. Used inside torch.utils.checkpoint."""
    logits = F.linear(hidden_chunk, lm_head_weight, lm_head_bias)
    b, s, v = logits.shape
    return F.cross_entropy(
        logits.view(b * s, v),
        target_chunk.reshape(b * s),
        ignore_index=ignore_index,
        reduction="none",
    ).view(b, s)


def _chunked_lm_ce(
    lm_head: nn.Module,
    hidden_states: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor | None,
    chunk_size: int,
) -> torch.Tensor:
    """CE loss without materializing full (B, S, V) logits tensor.

    Chunks along the sequence dimension and uses activation checkpointing
    per chunk so backward recomputes rather than stores chunk logits.
    """
    # Shift for next-token prediction
    shift_hidden = hidden_states[:, :-1, :]
    shift_targets = target_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].float()
    if loss_mask is not None:
        shift_mask = shift_mask * loss_mask[:, 1:].float()

    _b, seq_len, _h = shift_hidden.shape
    weighted_sum = shift_hidden.new_zeros(())

    lm_head_weight = lm_head.weight
    lm_head_bias = getattr(lm_head, "bias", None)

    # Skip checkpoint overhead when sequence fits in a single chunk
    use_checkpoint = seq_len > chunk_size

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        h_chunk = shift_hidden[:, start:end].contiguous()
        t_chunk = shift_targets[:, start:end].contiguous()

        if use_checkpoint:
            chunk_loss = torch_checkpoint(
                _chunk_ce_fn, lm_head_weight, h_chunk, t_chunk,
                lm_head_bias, use_reentrant=False,
            )
        else:
            chunk_loss = _chunk_ce_fn(lm_head_weight, h_chunk, t_chunk, lm_head_bias)

        weighted_sum = weighted_sum + (chunk_loss * shift_mask[:, start:end]).sum()

    return weighted_sum / shift_mask.sum().clamp(min=1)


@dataclass
class GenerationOutput:
    """Structured output from decoder generation."""

    content_ids: list[torch.Tensor]  # per-sample variable-length content token IDs
    content_text: list[str]  # decoded content (convenience)
    full_ids: list[torch.Tensor]  # per-sample complete generation for debugging


class ReconstructionDecoder(nn.Module):
    """Causal LM decoder for reconstructing content from BgKIT survivors.

    Wraps Qwen3.5-0.8B and exposes interleaved survivor-injection helpers.

    Supports optional NVFP4 quantization via TransformerEngine and LoRA
    via peft. Setup order: construct → load checkpoint → enable_nvfp4() → apply_lora().
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 1024,
        nvfp4: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self._use_te = False
        self._te_recipe = None
        self._has_lora = False
        # Opt-in flag for Liger-fused linear+CE. Trainers toggle this via
        # ``enable_liger_ce(True)`` when the Liger Kernel package is available;
        # the default is off so CPU host tests and un-installed environments
        # keep hitting the existing chunked-CE path.
        self._use_liger_ce = False

        if nvfp4:
            self.enable_nvfp4()

    def enable_liger_ce(self, enabled: bool = True) -> None:
        """Toggle the fused linear+CE path used inside ``forward_interleaved_with_loss``.

        Safe to call regardless of whether Liger is actually installed:
        :func:`bgkit.utils.liger_integration.liger_chunked_ce_loss` falls
        back to :func:`_chunked_lm_ce` when the import fails, so enabling
        without Liger is a silent no-op.
        """
        self._use_liger_ce = bool(enabled)

    def enable_nvfp4(self) -> None:
        """Convert decoder Linear modules to TE Linear with NVFP4 support.

        Call AFTER loading checkpoint weights (te.Linear adds _extra_state keys
        that won't be present in bf16 checkpoints).

        Works both before and after apply_lora():
        - Before LoRA: converts nn.Linear → te.Linear directly
        - After LoRA: swaps base_layer inside each LoRA wrapper from nn.Linear
          to te.Linear, freezes base weights (QLoRA pattern — avoids wgrad
          kernel that requires sm_121a compilation)
        """
        if self._use_te:
            return  # already converted

        from transformer_engine.common.recipe import NVFP4BlockScaling

        if self._has_lora:
            self._convert_lora_base_layers_to_te()
        else:
            from bgkit.utils.te_convert import convert_linear_to_te

            convert_linear_to_te(self.backbone, skip_names=("embed_tokens", "lm_head"))

        self._use_te = True
        self._te_recipe = NVFP4BlockScaling(disable_rht=True)
        logger.info("decoder_nvfp4_enabled", mode="qlora" if self._has_lora else "direct")

    def _convert_lora_base_layers_to_te(self) -> None:
        """Swap nn.Linear base_layer inside LoRA wrappers with te.Linear.

        Freezes base weights so the NVFP4 wgrad kernel (which requires sm_121a)
        is never invoked. LoRA adapters stay bf16 and trainable.
        """
        import transformer_engine.pytorch as te

        try:
            from peft.tuners.lora import Linear as LoraLinear
        except ImportError:
            return

        count = 0
        for _name, module in self.backbone.named_modules():
            if not isinstance(module, LoraLinear):
                continue
            base = module.base_layer
            if isinstance(base, te.Linear):
                continue  # already converted
            if not isinstance(base, nn.Linear):
                continue

            te_linear = te.Linear(
                base.in_features, base.out_features, bias=base.bias is not None,
            )
            te_linear.to(device=base.weight.device, dtype=base.weight.dtype)
            te_linear.weight.data.copy_(base.weight.data)
            if base.bias is not None:
                te_linear.bias.data.copy_(base.bias.data)
            te_linear.weight.requires_grad_(False)
            if te_linear.bias is not None:
                te_linear.bias.requires_grad_(False)

            module.base_layer = te_linear
            count += 1

        logger.info("lora_base_layers_converted_to_te", count=count)

    def _get_inner_model_and_head(self) -> tuple[nn.Module, nn.Module]:
        """Return (inner_model, lm_head) handling plain, PeftModel, and TE cases.

        Also handles torch.compile's OptimizedModule wrapper, which breaks
        isinstance() checks. Unwrap before testing.
        """
        backbone = self.backbone
        # Unwrap torch.compile's OptimizedModule if present
        if hasattr(backbone, "_orig_mod"):
            backbone = backbone._orig_mod
        try:
            from peft import PeftModel

            if isinstance(backbone, PeftModel):
                causal_lm = backbone.base_model.model
                return causal_lm.model, causal_lm.lm_head
        except ImportError:
            pass
        return backbone.model, backbone.lm_head

    def _build_single_splice_segments(
        self,
        *,
        survivor_embeddings: torch.Tensor,
        survivor_attention_mask: torch.Tensor,
        token_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        splice_starts: torch.Tensor,
        splice_lengths: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> tuple[list[Segment], torch.Tensor]:
        """Build batched ``[pre_tokens | embeddings | post_tokens]`` segments.

        Each sample contributes exactly one splice site. The pre-token region
        is left-padded so the splice boundary stays aligned across the batch;
        the post-token region is right-padded so tokens begin immediately after
        the embedding block.
        """
        if token_ids.dim() != 2 or token_attention_mask.dim() != 2:
            raise ValueError("token_ids and token_attention_mask must be 2D")
        if survivor_embeddings.dim() != 3 or survivor_attention_mask.dim() != 2:
            raise ValueError(
                "survivor_embeddings must be 3D and survivor_attention_mask 2D",
            )

        batch_size = token_ids.size(0)
        if survivor_embeddings.size(0) != batch_size:
            raise ValueError("survivor batch size must match token batch size")

        splice_starts = splice_starts.to(device=token_ids.device, dtype=torch.long)
        splice_lengths = splice_lengths.to(device=token_ids.device, dtype=torch.long)
        if splice_starts.shape != (batch_size,) or splice_lengths.shape != (batch_size,):
            raise ValueError("splice_starts and splice_lengths must be shape (B,)")

        pre_ids: list[torch.Tensor] = []
        post_ids: list[torch.Tensor] = []
        pre_loss: list[torch.Tensor] = []
        post_loss: list[torch.Tensor] = []

        max_pre = 0
        max_post = 0
        for b in range(batch_size):
            real_len = int(token_attention_mask[b].sum().item())
            start = int(splice_starts[b].item())
            span_len = int(splice_lengths[b].item())
            end = start + span_len
            if start < 0 or end > real_len:
                raise ValueError(
                    f"invalid splice [{start}, {end}) for sample length {real_len}",
                )

            pre = token_ids[b, :start]
            post = token_ids[b, end:real_len]
            pre_ids.append(pre)
            post_ids.append(post)
            max_pre = max(max_pre, int(pre.size(0)))
            max_post = max(max_post, int(post.size(0)))

            if loss_mask is not None:
                pre_loss.append(loss_mask[b, :start].to(dtype=torch.bool))
                post_loss.append(loss_mask[b, end:real_len].to(dtype=torch.bool))

        segments: list[Segment] = []
        attention_parts: list[torch.Tensor] = []

        if max_pre > 0:
            pre_tokens = torch.zeros(
                batch_size, max_pre, dtype=token_ids.dtype, device=token_ids.device,
            )
            pre_attn = torch.zeros(
                batch_size, max_pre, dtype=torch.bool, device=token_ids.device,
            )
            pre_loss_mask = torch.zeros_like(pre_attn)
            for b, pre in enumerate(pre_ids):
                plen = int(pre.size(0))
                if plen == 0:
                    continue
                pre_tokens[b, max_pre - plen:] = pre
                pre_attn[b, max_pre - plen:] = True
                if loss_mask is not None:
                    pre_loss_mask[b, max_pre - plen:] = pre_loss[b]
            if loss_mask is None:
                segments.append(TokenSegment(token_ids=pre_tokens, loss=True))
            else:
                segments.append(TokenSegment(token_ids=pre_tokens, loss_mask=pre_loss_mask))
            attention_parts.append(pre_attn)

        segments.append(EmbeddingSegment(embeddings=survivor_embeddings))
        attention_parts.append(survivor_attention_mask.to(dtype=torch.bool))

        if max_post > 0:
            post_tokens = torch.zeros(
                batch_size, max_post, dtype=token_ids.dtype, device=token_ids.device,
            )
            post_attn = torch.zeros(
                batch_size, max_post, dtype=torch.bool, device=token_ids.device,
            )
            post_loss_mask = torch.zeros_like(post_attn)
            for b, post in enumerate(post_ids):
                plen = int(post.size(0))
                if plen == 0:
                    continue
                post_tokens[b, :plen] = post
                post_attn[b, :plen] = True
                if loss_mask is not None:
                    post_loss_mask[b, :plen] = post_loss[b]
            if loss_mask is None:
                segments.append(TokenSegment(token_ids=post_tokens, loss=True))
            else:
                segments.append(TokenSegment(token_ids=post_tokens, loss_mask=post_loss_mask))
            attention_parts.append(post_attn)

        return segments, torch.cat(attention_parts, dim=1)

    def forward_with_single_splice(
        self,
        *,
        survivor_embeddings: torch.Tensor,
        survivor_attention_mask: torch.Tensor,
        token_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        splice_starts: torch.Tensor,
        splice_lengths: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        chunk_size: int = 256,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | InterleavedForwardOutput:
        """Forward + loss with one embedding splice per sample."""
        segments, attention_mask = self._build_single_splice_segments(
            survivor_embeddings=survivor_embeddings,
            survivor_attention_mask=survivor_attention_mask,
            token_ids=token_ids,
            token_attention_mask=token_attention_mask,
            splice_starts=splice_starts,
            splice_lengths=splice_lengths,
            loss_mask=loss_mask,
        )
        return self.forward_interleaved_with_loss(
            segments,
            attention_mask=attention_mask,
            chunk_size=chunk_size,
            return_hidden_states=return_hidden_states,
        )

    # ------------------------------------------------------------------
    # Interleaved-segment forward + loss
    # ------------------------------------------------------------------

    def _concat_segments(
        self,
        segments: list[Segment],
        embed_fn: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Walk segments and return ``(inputs_embeds, token_ids_full, loss_mask_full)``.

        - Token segments are embedded via ``embed_fn`` and their token IDs
          copied into ``token_ids_full``. Their per-token loss mask is
          derived from ``seg.loss_mask`` (if set) or ``seg.loss`` expanded.
        - Embedding segments are used as-is and contribute zero tokens and
          ``False`` to the loss mask at those positions — embedding positions
          are never loss-bearing targets.

        All tensors come out with shape ``(B, S_total, ...)``. Unbatched
        inputs (``L`` or ``(K, D)``) get a leading batch dim added.
        """
        if not segments:
            raise ValueError("segments must be non-empty")

        # Cast every embedding segment to the input embedding table's dtype so
        # the concatenated ``inputs_embeds`` matches the rest of the inner
        # model's layers.
        try:
            target_dtype = embed_fn.weight.dtype
        except AttributeError:
            target_dtype = None

        embeds_list: list[torch.Tensor] = []
        tokens_list: list[torch.Tensor] = []
        loss_list: list[torch.Tensor] = []
        batch_size: int | None = None

        for seg in segments:
            if isinstance(seg, TokenSegment):
                ids = seg.token_ids
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                if ids.dim() != 2:
                    raise ValueError(
                        f"TokenSegment.token_ids must be (B, L) or (L,), "
                        f"got shape {tuple(ids.shape)}"
                    )
                if batch_size is None:
                    batch_size = ids.size(0)
                elif ids.size(0) != batch_size:
                    raise ValueError(
                        f"segment batch size mismatch: {ids.size(0)} vs {batch_size}"
                    )
                ids = ids.to(dtype=torch.long)
                emb = embed_fn(ids)
                if target_dtype is None:
                    target_dtype = emb.dtype
                embeds_list.append(emb)
                tokens_list.append(ids)

                if seg.loss_mask is not None:
                    lm = seg.loss_mask
                    if lm.dim() == 1:
                        lm = lm.unsqueeze(0)
                    if lm.shape != ids.shape:
                        raise ValueError(
                            f"TokenSegment.loss_mask shape {tuple(lm.shape)} "
                            f"does not match token_ids {tuple(ids.shape)}"
                        )
                    lm = lm.to(dtype=torch.bool, device=ids.device)
                else:
                    lm = torch.full(
                        ids.shape, bool(seg.loss),
                        dtype=torch.bool, device=ids.device,
                    )
                loss_list.append(lm)
            elif isinstance(seg, EmbeddingSegment):
                emb = seg.embeddings
                if emb.dim() == 2:
                    emb = emb.unsqueeze(0)
                if emb.dim() != 3:
                    raise ValueError(
                        f"EmbeddingSegment.embeddings must be (B, K, D) or "
                        f"(K, D), got shape {tuple(emb.shape)}"
                    )
                if batch_size is None:
                    batch_size = emb.size(0)
                elif emb.size(0) != batch_size:
                    raise ValueError(
                        f"segment batch size mismatch: {emb.size(0)} vs {batch_size}"
                    )
                if target_dtype is None:
                    target_dtype = emb.dtype
                embeds_list.append(emb)
                k = emb.size(1)
                tokens_list.append(
                    torch.zeros(batch_size, k, dtype=torch.long, device=emb.device)
                )
                loss_list.append(
                    torch.zeros(batch_size, k, dtype=torch.bool, device=emb.device)
                )
            else:
                raise TypeError(f"unknown segment type: {type(seg).__name__}")

        inputs_embeds = torch.cat(
            [e.to(dtype=target_dtype) for e in embeds_list], dim=1,
        )
        token_ids_full = torch.cat(tokens_list, dim=1)
        loss_mask_full = torch.cat(loss_list, dim=1)
        return inputs_embeds, token_ids_full, loss_mask_full

    def _compute_lm_ce(
        self,
        *,
        lm_head: nn.Module,
        hidden_states: torch.Tensor,
        token_ids_full: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask_full: torch.Tensor | None,
        chunk_size: int,
    ) -> torch.Tensor:
        """Dispatch CE computation: Liger fused path if available, else chunked.

        Controlled by the module-level flag ``self._use_liger_ce`` (default
        True if set on the decoder, matching the trainer gate). When Liger
        is installed *and* the flag is on, we call ``liger_chunked_ce_loss``
        which never materialises the full ``(B, S, V)`` logits tensor.
        Otherwise we stay on the existing ``_chunked_lm_ce`` path for exact
        backward compatibility.
        """
        use_liger = getattr(self, "_use_liger_ce", False)
        if use_liger:
            from bgkit.utils.liger_integration import (
                is_liger_available,
                liger_chunked_ce_loss,
            )

            if is_liger_available():
                # Build a (B, S) float mask that combines attention_mask with
                # loss_mask_full. liger_chunked_ce_loss applies its own shift.
                combined = attention_mask.to(dtype=torch.bool)
                if loss_mask_full is not None:
                    combined = combined & loss_mask_full.to(dtype=torch.bool)
                return liger_chunked_ce_loss(
                    hidden_states=hidden_states,
                    lm_head_weight=lm_head.weight,
                    lm_head_bias=getattr(lm_head, "bias", None),
                    labels=token_ids_full,
                    mask=combined,
                    chunk_size=chunk_size,
                )

        return _chunked_lm_ce(
            lm_head,
            hidden_states,
            token_ids_full,
            attention_mask,
            loss_mask_full,
            chunk_size,
        )

    def _inner_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Run ``inner_model`` with TE/NVFP4 seq alignment applied.

        Returns ``(hidden_states, seq_pad)`` where ``seq_pad`` is the number
        of alignment-padding positions appended to the end; the caller is
        responsible for stripping them from downstream tensors before
        computing loss.
        """
        inner_model, _lm_head = self._get_inner_model_and_head()
        seq_pad = 0
        padded_embeds = inputs_embeds
        padded_mask = attention_mask

        if self._use_te:
            import transformer_engine.pytorch as te

            b, s, _h = inputs_embeds.shape
            align = 16  # NVFP4_BLOCK_SIZE
            if (b * s) % align != 0:
                for seq_pad in range(1, align + 1):
                    if (b * (s + seq_pad)) % align == 0:
                        break
                pad_emb = inputs_embeds.new_zeros(b, seq_pad, inputs_embeds.size(-1))
                padded_embeds = torch.cat([inputs_embeds, pad_emb], dim=1)
                pad_mask = attention_mask.new_zeros(b, seq_pad)
                padded_mask = torch.cat([attention_mask, pad_mask], dim=1)
            with te.fp8_autocast(enabled=True, fp8_recipe=self._te_recipe):
                hidden = inner_model(
                    inputs_embeds=padded_embeds, attention_mask=padded_mask,
                ).last_hidden_state
        else:
            hidden = inner_model(
                inputs_embeds=padded_embeds, attention_mask=padded_mask,
            ).last_hidden_state
        return hidden, seq_pad

    def forward_interleaved_with_loss(
        self,
        segments: list[Segment],
        *,
        attention_mask: torch.Tensor | None = None,
        chunk_size: int = 256,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | InterleavedForwardOutput:
        """Forward + loss over a heterogeneous segment sequence.

        Concatenates ``segments`` in order along the sequence dimension,
        forwards through the inner model with TE/NVFP4 alignment, and
        computes a chunked next-token CE over positions marked loss-bearing
        by the segment metadata.

        Shift semantics: at each shifted position ``i``, the loss is
        ``CE(logits[i], token_ids_full[i+1]) * loss_mask_full[i+1]``.
        That means embedding-segment positions contribute zero loss *as
        targets* (never predictable), but their outputs still drive
        predictions of the following token segment's first token — exactly
        the mechanism that lets BgKIT survivors condition next-token
        generation.

        Args:
            segments: ordered list of :class:`TokenSegment` /
                :class:`EmbeddingSegment` entries. All must share batch size.
            attention_mask: optional ``(B, S_total)`` mask. Defaults to all-ones,
                which is correct for the typical single-sample call pattern.
            chunk_size: sequence chunk size for the chunked CE path.
            return_hidden_states: if True, return a
                :class:`InterleavedForwardOutput` with the scalar loss plus
                the hidden states, concatenated token IDs, and loss mask
                (all shape-compatible for downstream argmax / metric
                computation). If False (default), return only the scalar
                loss.

        Returns:
            Scalar loss tensor, or :class:`InterleavedForwardOutput` when
            ``return_hidden_states=True``.
        """
        inner_model, lm_head = self._get_inner_model_and_head()
        embed_fn = inner_model.get_input_embeddings()
        inputs_embeds, token_ids_full, loss_mask_full = self._concat_segments(
            segments, embed_fn,
        )

        b, s, _ = inputs_embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                b, s, dtype=torch.bool, device=inputs_embeds.device,
            )
        else:
            attention_mask = attention_mask.to(device=inputs_embeds.device)
            if tuple(attention_mask.shape) != (b, s):
                raise ValueError(
                    f"attention_mask shape {tuple(attention_mask.shape)} "
                    f"does not match concatenated segment shape ({b}, {s})"
                )

        hidden, seq_pad = self._inner_forward(inputs_embeds, attention_mask)
        if seq_pad > 0:
            hidden = hidden[:, :-seq_pad, :]

        # Prefer Liger's fused linear+CE (no (B, S, V) logits materialised)
        # when available; otherwise fall back to the chunked path that
        # materialises chunks of logits under activation checkpointing.
        # The fallback is transparent — same scalar output either way.
        loss = self._compute_lm_ce(
            lm_head=lm_head,
            hidden_states=hidden,
            token_ids_full=token_ids_full,
            attention_mask=attention_mask,
            loss_mask_full=loss_mask_full,
            chunk_size=chunk_size,
        )

        if return_hidden_states:
            return InterleavedForwardOutput(
                loss=loss,
                hidden_states=hidden,
                token_ids=token_ids_full,
                loss_mask=loss_mask_full,
                attention_mask=attention_mask,
                lm_head=lm_head,
            )
        return loss

    def apply_lora(self, lora_config: dict) -> None:
        """Wrap backbone with LoRA adapters.

        Args:
            lora_config: Dict with keys: r, alpha, dropout (optional, default 0.05),
                target_modules (list of module name strings).
        """
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("alpha", 32),
            lora_dropout=lora_config.get("dropout", 0.05),
            target_modules=list(lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ])),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.backbone = get_peft_model(self.backbone, config)
        self._has_lora = True

        trainable = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self.backbone.parameters())
        logger.info(
            "decoder_lora_applied",
            trainable_params=trainable,
            total_params=total,
            ratio=f"{trainable / total:.4f}",
            r=config.r,
            alpha=config.lora_alpha,
        )

    def merge_lora(self) -> dict:
        """Merge LoRA adapters into base weights and return a clean state dict.

        Produces a state dict with standard ReconstructionDecoder keys (no
        PeftModel prefix, no lora_A/B keys) so downstream trainers can load
        it as a plain decoder.
        """
        if not self._has_lora:
            return self.state_dict()

        peft_sd = self.state_dict()
        merged = {}
        peft_prefix = "backbone.base_model.model."
        decoder_prefix = "backbone."

        # Collect LoRA A/B pairs keyed by their target module path
        lora_pairs: dict[str, dict[str, torch.Tensor]] = {}
        for key, val in peft_sd.items():
            if ".lora_A." in key:
                base_key = key.split(".lora_A.")[0]
                lora_pairs.setdefault(base_key, {})["A"] = val
            elif ".lora_B." in key:
                base_key = key.split(".lora_B.")[0]
                lora_pairs.setdefault(base_key, {})["B"] = val

        # Get scaling factor from LoRA config
        try:
            lora_cfg = self.backbone.peft_config["default"]
            scaling = lora_cfg.lora_alpha / lora_cfg.r
        except (AttributeError, KeyError):
            scaling = 2.0  # default alpha=32 / r=16

        for key, val in peft_sd.items():
            # Skip LoRA adapter keys
            if ".lora_A." in key or ".lora_B." in key:
                continue

            # Strip PeftModel prefix: backbone.base_model.model.X -> backbone.X
            if key.startswith(peft_prefix):
                clean_key = decoder_prefix + key[len(peft_prefix):]
            else:
                clean_key = key

            # Strip .base_layer. inserted by peft for LoRA target modules
            clean_key = clean_key.replace(".base_layer.", ".")

            # If this is a LoRA target weight, merge A/B into it
            peft_key = (
                key.split(".base_layer.weight")[0]
                if ".base_layer.weight" in key
                else None
            )
            if peft_key is None:
                peft_key = key.split(".weight")[0] if key.endswith(".weight") else None
            if peft_key and peft_key in lora_pairs:
                pair = lora_pairs[peft_key]
                if "A" in pair and "B" in pair:
                    val = val + scaling * (pair["B"] @ pair["A"])

            merged[clean_key] = val

        return merged

    @torch.no_grad()
    def generate_with_single_splice(
        self,
        *,
        survivor_embeddings: torch.Tensor,
        survivor_attention_mask: torch.Tensor,
        prefix_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        splice_starts: torch.Tensor,
        splice_lengths: torch.Tensor,
        suffix_ids: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> GenerationOutput:
        """Autoregressive generation with one in-sequence embedding splice."""
        segments, attention_mask = self._build_single_splice_segments(
            survivor_embeddings=survivor_embeddings,
            survivor_attention_mask=survivor_attention_mask,
            token_ids=prefix_ids,
            token_attention_mask=prefix_attention_mask,
            splice_starts=splice_starts,
            splice_lengths=splice_lengths,
            loss_mask=None,
        )
        embed_fn = self.backbone.get_input_embeddings()
        inputs_embeds, _token_ids_full, _loss_mask_full = self._concat_segments(
            segments, embed_fn,
        )

        do_sample = temperature > 0
        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-8)

        output_ids = self.backbone.generate(**gen_kwargs)
        input_len = inputs_embeds.size(1)
        new_ids = output_ids[:, input_len:]

        suffix_ids = suffix_ids.to(new_ids.device)
        suf_len = suffix_ids.size(0)
        batch_size = new_ids.size(0)

        eos_id = getattr(tokenizer, "eos_token_id", None)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        content_ids_list = []
        for b in range(batch_size):
            seq = new_ids[b]
            if eos_id is not None:
                while seq.size(0) > 0 and seq[-1].item() == eos_id:
                    seq = seq[:-1]
            if pad_id is not None:
                while seq.size(0) > 0 and seq[-1].item() == pad_id:
                    seq = seq[:-1]
            if seq.size(0) >= suf_len and seq[-suf_len:].equal(suffix_ids):
                content_ids_list.append(seq[:-suf_len])
            else:
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
