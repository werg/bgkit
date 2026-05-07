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

Attention regime: packed (varlen) only. No padded / masked path.
``cu_seqlens: (B+1,) int32``, ``position_ids: (N,) int64`` per-sample restart.
The HF backbone is called with packed inputs: flat ``(1, N, D)`` embeddings,
per-sample-restart ``position_ids``, and ``cu_seq_lens_q/k`` kwargs that flow
through to the registered attention backend. No ``attention_mask`` is
constructed or passed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from bgkit.utils.packing import position_ids_from_cu

logger = structlog.get_logger()

DEFAULT_LM_CE_CHUNK_SIZE = int(os.environ.get("BGKIT_DECODER_CE_CHUNK_SIZE", "2048"))
DEFAULT_LM_CE_IMPL = os.environ.get("BGKIT_DECODER_CE_IMPL", "cce").strip().lower()
LM_CE_IMPLS = frozenset(
    {
        "auto",
        "chunked",
        "liger",
        "cce",
        "cce_exact",
        "cce_kahan_full",
        "cce_kahan_full_c",
        "cce_kahan_full_e",
        "cce_kahan_full_c_full_e",
        "torch_compile",
    }
)


def _resolve_ce_chunk_size(chunk_size: int | None) -> int:
    return DEFAULT_LM_CE_CHUNK_SIZE if chunk_size is None else int(chunk_size)


def _resolve_lm_ce_impl(impl: str | None) -> str:
    resolved = DEFAULT_LM_CE_IMPL if impl is None else impl.strip().lower()
    if resolved not in LM_CE_IMPLS:
        raise ValueError(
            f"Unsupported decoder CE implementation {impl!r}; "
            f"expected one of {sorted(LM_CE_IMPLS)}"
        )
    return resolved


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
    chunk_size: int | None,
) -> torch.Tensor:
    """CE loss without materializing full (B, S, V) logits tensor.

    Chunks along the sequence dimension and uses activation checkpointing
    per chunk so backward recomputes rather than stores chunk logits.
    Works with both ``(B, L, D)`` and ``(1, N, D)`` packed-as-single-sample
    shapes since they are numerically equivalent.
    """
    # Shift for next-token prediction
    shift_hidden = hidden_states[:, :-1, :]
    shift_targets = target_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].float()
    if loss_mask is not None:
        shift_mask = shift_mask * loss_mask[:, 1:].float()

    chunk_size = _resolve_ce_chunk_size(chunk_size)
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
                _chunk_ce_fn,
                lm_head_weight,
                h_chunk,
                t_chunk,
                lm_head_bias,
                use_reentrant=False,
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


class DecoderLoRALinear(nn.Module):
    """Small always-on LoRA wrapper for frozen decoder ``nn.Linear`` modules."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        adapter_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, base_layer.in_features, device=base_layer.weight.device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base_layer.out_features,
                self.rank,
                device=base_layer.weight.device,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.lora_A.data = self.lora_A.data.to(dtype=adapter_dtype)
        self.lora_B.data = self.lora_B.data.to(dtype=adapter_dtype)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.base_layer.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base_layer(x)
        h = self.dropout(x)
        if h.dtype != self.lora_A.dtype:
            h = h.to(dtype=self.lora_A.dtype)
        h = F.linear(h, self.lora_A)
        h = F.linear(h, self.lora_B)
        return y + (h * self.scaling).to(dtype=y.dtype)

    def merged_weight(self) -> torch.Tensor:
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return self.base_layer.weight + delta.to(dtype=self.base_layer.weight.dtype)


class ReconstructionDecoder(nn.Module):
    """Causal LM decoder for reconstructing content from BgKIT survivors.

    Wraps Qwen3.5-0.8B and exposes interleaved survivor-injection helpers.

    Supports optional NVFP4 quantization via TransformerEngine and decoder
    LoRA. Setup order: construct → load checkpoint → enable_nvfp4() →
    apply_lora().

    Attention regime: packed (varlen) only.  No padded / masked path.
    ``forward_with_single_splice`` accepts flat packed inputs;
    ``generate_with_single_splice`` uses a custom B=1 autoregressive loop
    driven by ``backbone.forward`` — not ``backbone.generate``.
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
        self._use_native_nvfp4 = False
        self._te_recipe = None
        self._has_lora = False
        self._lora_impl: str | None = None
        self._lora_target_modules: tuple[str, ...] = ()
        self._lora_scaling: float | None = None
        # Opt-in flag for Liger-fused linear+CE. Trainers toggle this via
        # ``enable_liger_ce(True)`` when the Liger Kernel package is available;
        # the default is off so CPU host tests and un-installed environments
        # keep hitting the existing chunked-CE path.
        self._use_liger_ce = False
        self._lm_ce_impl = DEFAULT_LM_CE_IMPL

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

    def set_lm_ce_impl(self, impl: str | None) -> None:
        """Select decoder LM CE implementation.

        The default is ``cce`` because it is faster and much lower memory on
        GB10 when the optional ``cut_cross_entropy`` package is installed. If
        that package is missing and strict mode is not set, the CCE integration
        falls back to BgKIT's chunked CE. ``auto`` preserves the old behaviour:
        use the Liger CE adapter when ``enable_liger_ce(True)`` has been called,
        and otherwise use chunked CE.
        """

        self._lm_ce_impl = _resolve_lm_ce_impl(impl)

    @property
    def lm_ce_impl(self) -> str:
        """Return the effective decoder LM CE implementation."""

        return self._lm_ce_impl

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
        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (12, 1)
            and os.environ.get("BGKIT_ALLOW_NVFP4_SM121", "0") != "1"
        ):
            raise RuntimeError(
                "decoder NVFP4 is disabled for the current sm_121 container. "
                "TransformerEngine FP4 conversion requires an architecture-specific "
                "sm_121a build; the current image emits device-side PTX errors. "
                "The Atlas/Spark route avoids this by packing weights with software "
                "E4M3/E2M1 conversion and using custom W4A16 kernels instead of TE. "
                "Set BGKIT_ALLOW_NVFP4_SM121=1 only for explicit TE experiments."
            )

        from transformer_engine.common.recipe import NVFP4BlockScaling

        if self._has_lora:
            self._convert_lora_base_layers_to_te()
        else:
            from bgkit.utils.te_convert import convert_linear_to_te

            convert_linear_to_te(self.backbone, skip_names=("embed_tokens", "lm_head"))

        self._use_te = True
        self._te_recipe = NVFP4BlockScaling(disable_rht=True)
        logger.info("decoder_nvfp4_enabled", mode="qlora" if self._has_lora else "direct")

    def enable_native_frozen_nvfp4(
        self,
        *,
        target_modules: tuple[str, ...] | None = None,
    ) -> None:
        """Pack frozen decoder base Linear weights into BgKIT-native NVFP4.

        This is an explicit experimental path for LoRA-style training where the
        base decoder weights are frozen and only activation gradients are needed.
        The current module uses a reference dequantizing forward; it establishes
        the packed format and autograd contract for the forthcoming W4A16 CUDA
        kernel, and is deliberately not a default training path.
        """

        if self._use_native_nvfp4:
            return
        targets = target_modules or self._lora_target_modules or (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
        if self._has_lora:
            count = self._convert_lora_base_layers_to_native_nvfp4(tuple(targets))
        else:
            count = self._convert_linear_layers_to_native_nvfp4(tuple(targets))
        if count == 0:
            raise ValueError(
                f"native NVFP4 found no decoder Linear targets in {sorted(set(targets))!r}"
            )
        self._use_native_nvfp4 = True
        logger.info(
            "decoder_native_frozen_nvfp4_enabled",
            mode="lora_base" if self._has_lora else "direct",
            count=count,
        )

    def _convert_lora_base_layers_to_te(self) -> None:
        """Swap nn.Linear base_layer inside LoRA wrappers with te.Linear.

        Freezes base weights so the NVFP4 wgrad kernel (which requires sm_121a)
        is never invoked. LoRA adapters stay bf16 and trainable.
        """
        import transformer_engine.pytorch as te

        wrapper_types: tuple[type[nn.Module], ...] = (DecoderLoRALinear,)
        try:
            from peft.tuners.lora import Linear as LoraLinear
            wrapper_types = (DecoderLoRALinear, LoraLinear)
        except ImportError:
            pass

        count = 0
        for _name, module in self.backbone.named_modules():
            if not isinstance(module, wrapper_types):
                continue
            base = module.base_layer
            if isinstance(base, te.Linear):
                continue  # already converted
            if not isinstance(base, nn.Linear):
                continue

            te_linear = te.Linear(
                base.in_features,
                base.out_features,
                bias=base.bias is not None,
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

    def _convert_lora_base_layers_to_native_nvfp4(
        self,
        target_modules: tuple[str, ...],
    ) -> int:
        from bgkit.quant.nvfp4 import FrozenNVFP4Linear

        targets = set(target_modules)
        count = 0
        for name, module in self.backbone.named_modules():
            base = getattr(module, "base_layer", None)
            if isinstance(base, FrozenNVFP4Linear):
                continue
            if not isinstance(base, nn.Linear):
                continue
            local_name = name.rsplit(".", 1)[-1]
            if local_name not in targets:
                continue
            module.base_layer = FrozenNVFP4Linear.from_linear(base)
            count += 1
        return count

    def _convert_linear_layers_to_native_nvfp4(
        self,
        target_modules: tuple[str, ...],
    ) -> int:
        from bgkit.quant.nvfp4 import FrozenNVFP4Linear

        targets = set(target_modules)
        count = 0
        for parent in list(self.backbone.modules()):
            for child_name, child in list(parent.named_children()):
                if child_name not in targets or not isinstance(child, nn.Linear):
                    continue
                setattr(parent, child_name, FrozenNVFP4Linear.from_linear(child))
                count += 1
        return count

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

    def _packed_forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Run ``inner_model`` in packed (varlen) mode.

        Parameters
        ----------
        inputs_embeds:
            Shape ``(1, N, D)`` — all samples concatenated as a single
            batch item. No padding tokens; every position is a real token.
        cu_seqlens:
            Shape ``(B+1,)`` int32. Cumulative per-sample lengths.
            ``cu_seqlens[0]==0``, ``cu_seqlens[-1]==N``.
        max_seqlen:
            Maximum per-sample length ``max(L_i)``.
        position_ids:
            Shape ``(N,)`` int64 with per-sample restart (0 at each
            sample boundary). Passed to the backbone as ``(1, N)`` so RoPE
            computes correct per-sample positions.

        Returns
        -------
        (hidden_states, seq_pad)
            ``hidden_states`` has shape ``(1, N + seq_pad, D)``; the caller
            strips ``seq_pad`` alignment positions appended by NVFP4.
        """
        inner_model, _lm_head = self._get_inner_model_and_head()

        # Promote position_ids to (1, N) for the HF model signature.
        pos_ids_2d = position_ids.unsqueeze(0)  # (1, N)

        # cu_seqlens_q/k flow through TransformersKwargs → FlashAttentionKwargs
        # to the registered attention backend, preventing cross-sample attention.
        # Names follow HF's TransformersKwargs convention.
        packed_attn_kwargs: dict = {
            "cu_seq_lens_q": cu_seqlens,
            "cu_seq_lens_k": cu_seqlens,
            "max_length_q": max_seqlen,
            "max_length_k": max_seqlen,
        }

        seq_pad = 0
        padded_embeds = inputs_embeds

        if self._use_te:
            import transformer_engine.pytorch as te

            _b, s, _h = inputs_embeds.shape
            align = 16  # NVFP4_BLOCK_SIZE
            if (_b * s) % align != 0:
                for seq_pad in range(1, align + 1):
                    if (_b * (s + seq_pad)) % align == 0:
                        break
                pad_emb = inputs_embeds.new_zeros(_b, seq_pad, inputs_embeds.size(-1))
                padded_embeds = torch.cat([inputs_embeds, pad_emb], dim=1)
                # Extend position_ids with zeros for the padding positions.
                pad_pos = position_ids.new_zeros(seq_pad)
                pos_ids_2d = torch.cat([position_ids, pad_pos], dim=0).unsqueeze(0)
            with te.fp8_autocast(enabled=True, fp8_recipe=self._te_recipe):
                hidden = inner_model(
                    inputs_embeds=padded_embeds,
                    position_ids=pos_ids_2d,
                    **packed_attn_kwargs,
                ).last_hidden_state
        else:
            hidden = inner_model(
                inputs_embeds=padded_embeds,
                position_ids=pos_ids_2d,
                **packed_attn_kwargs,
            ).last_hidden_state

        return hidden, seq_pad

    def forward_with_single_splice(
        self,
        *,
        survivor_embeddings: torch.Tensor,
        survivor_cu_seqlens: torch.Tensor,
        prefix_ids: list[torch.Tensor],
        suffix_ids: list[torch.Tensor],
        loss_mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | InterleavedForwardOutput:
        """Forward + loss with one embedding splice per sample — packed path.

        Builds ``[prefix_i | survivors_i | suffix_i]`` per sample, concatenates
        into a single flat sequence of length ``N_total``, runs the backbone once
        (as a ``(1, N_total, D)`` sequence with per-sample ``position_ids`` and
        ``cu_seqlens``), then computes the shifted CE loss masked to suffix
        positions.

        Parameters
        ----------
        survivor_embeddings:
            Flat ``(K_total, D)`` survivor embeddings from the encoder.
            ``K_total = sum(K_i)`` where ``K_i`` is the survivor count for
            sample ``i``.
        survivor_cu_seqlens:
            Shape ``(B+1,)`` int32. Cumulative survivor counts per sample.
        prefix_ids:
            Length-``B`` list of 1-D ``(L_pre_i,)`` int64 token ID tensors.
            Each tensor is the portion of the token sequence that appears
            *before* the survivor splice.
        suffix_ids:
            Length-``B`` list of 1-D ``(L_suf_i,)`` int64 token ID tensors.
            Appears *after* the survivors. These are the loss-bearing targets
            unless ``loss_mask`` overrides.
        loss_mask:
            Optional flat ``(N_total,)`` bool mask marking which positions
            in the concatenated sequence contribute to the CE loss.  ``None``
            means "all suffix positions contribute."
        chunk_size:
            Sequence chunk size for the chunked CE path.
        return_hidden_states:
            If True, return an :class:`InterleavedForwardOutput` with the
            scalar loss plus the hidden states, token IDs, and loss mask.

        Returns
        -------
        Scalar loss tensor, or :class:`InterleavedForwardOutput` when
        ``return_hidden_states=True``.
        """
        batch_size = len(prefix_ids)
        if len(suffix_ids) != batch_size:
            raise ValueError(
                f"prefix_ids and suffix_ids must have the same length; "
                f"got {batch_size} and {len(suffix_ids)}"
            )
        if survivor_cu_seqlens.shape[0] != batch_size + 1:
            raise ValueError(
                f"survivor_cu_seqlens must have shape (B+1,) = ({batch_size + 1},); "
                f"got {tuple(survivor_cu_seqlens.shape)}"
            )

        inner_model, lm_head = self._get_inner_model_and_head()
        embed_fn = inner_model.get_input_embeddings()

        # Determine target dtype from the embedding table.
        try:
            target_dtype = embed_fn.weight.dtype
        except AttributeError:
            target_dtype = survivor_embeddings.dtype

        device = survivor_embeddings.device

        # ----------------------------------------------------------------
        # Build per-sample [prefix | survivors | suffix] embeddings and
        # collect the flat token IDs + loss mask. Vectorized to one
        # embed_fn call for all prefixes + one for all suffixes (was
        # 2*B small kernel launches that drove a ~17 s/step regression
        # via host-side launch overhead on unified memory).
        # ----------------------------------------------------------------
        surv_cu_list = survivor_cu_seqlens.tolist()  # one sync, used downstream
        # Move prefix/suffix tensors to device once (host-side; no kernels).
        prefix_on_device = [p.to(device=device, dtype=torch.long) for p in prefix_ids]
        suffix_on_device = [s.to(device=device, dtype=torch.long) for s in suffix_ids]

        prefix_lens = [p.shape[0] for p in prefix_on_device]
        suffix_lens = [s.shape[0] for s in suffix_on_device]
        surv_lens = [surv_cu_list[b + 1] - surv_cu_list[b] for b in range(batch_size)]
        seg_lengths = [prefix_lens[b] + surv_lens[b] + suffix_lens[b] for b in range(batch_size)]

        # Single concat + single embed for all prefixes (and again for suffixes).
        if any(prefix_lens):
            all_prefix_ids = torch.cat(prefix_on_device, dim=0)
            emb_prefix_all = embed_fn(all_prefix_ids).to(dtype=target_dtype)
        else:
            emb_prefix_all = torch.empty(
                0,
                embed_fn.weight.shape[1],
                dtype=target_dtype,
                device=device,
            )
        if any(suffix_lens):
            all_suffix_ids = torch.cat(suffix_on_device, dim=0)
            emb_suffix_all = embed_fn(all_suffix_ids).to(dtype=target_dtype)
        else:
            emb_suffix_all = torch.empty(
                0,
                embed_fn.weight.shape[1],
                dtype=target_dtype,
                device=device,
            )

        # Assemble per-sample [emb_prefix | survivors | emb_suffix] via slice
        # concatenations into Python lists; one final torch.cat builds the
        # flat tensors. No per-element kernel launches inside the loop.
        sample_embeds: list[torch.Tensor] = []
        sample_token_ids: list[torch.Tensor] = []
        sample_loss: list[torch.Tensor] = []
        p_off = 0
        s_off = 0
        for b in range(batch_size):
            l_pre = prefix_lens[b]
            l_suf = suffix_lens[b]
            k_i = surv_lens[b]

            emb_pre = emb_prefix_all[p_off : p_off + l_pre]
            emb_suf = emb_suffix_all[s_off : s_off + l_suf]
            surv = survivor_embeddings[surv_cu_list[b] : surv_cu_list[b + 1]].to(dtype=target_dtype)
            sample_embeds.append(torch.cat([emb_pre, surv, emb_suf], dim=0))

            # Token IDs: prefix + zeros for survivor splice + suffix
            pre = prefix_on_device[b]
            suf = suffix_on_device[b]
            mid_zeros = pre.new_zeros(k_i)
            sample_token_ids.append(torch.cat([pre, mid_zeros, suf], dim=0))

            # Default loss mask: True only on suffix positions
            lm_pre = pre.new_zeros(l_pre, dtype=torch.bool)
            lm_mid = pre.new_zeros(k_i, dtype=torch.bool)
            lm_suf = pre.new_ones(l_suf, dtype=torch.bool)
            sample_loss.append(torch.cat([lm_pre, lm_mid, lm_suf], dim=0))

            p_off += l_pre
            s_off += l_suf

        # ----------------------------------------------------------------
        # Pack into flat (1, N_total, D) and build cu_seqlens / position_ids.
        # ----------------------------------------------------------------
        inputs_embeds = torch.cat(sample_embeds, dim=0).unsqueeze(0)  # (1, N_total, D)
        token_ids_flat = torch.cat(sample_token_ids, dim=0)  # (N_total,)
        default_loss_mask = torch.cat(sample_loss, dim=0)  # (N_total,)

        n_total = int(inputs_embeds.shape[1])
        # Build cu via cumulative sum on a Python list (cheap; no GPU sync).
        cu_list = [0]
        running = 0
        for sl in seg_lengths:
            running += sl
            cu_list.append(running)
        cu = torch.tensor(cu_list, dtype=torch.int32, device=device)
        max_seqlen = max(seg_lengths) if seg_lengths else 0
        pos_ids = position_ids_from_cu(cu, n_total)  # (N_total,)

        # Apply caller-supplied loss_mask if provided; otherwise use default.
        if loss_mask is not None:
            if loss_mask.shape != (n_total,):
                raise ValueError(
                    f"loss_mask shape {tuple(loss_mask.shape)} does not match N_total={n_total}"
                )
            final_loss_mask = loss_mask.to(device=device, dtype=torch.bool)
        else:
            final_loss_mask = default_loss_mask

        # Zero out cross-sample boundaries: the first token of each sample
        # (cu[i] for i>0) appears at a shifted-target position; its source
        # is the last token of the previous sample, which is semantically
        # invalid. Vectorized: one indexed write instead of (B-1) .item()
        # syncs.
        if batch_size > 1:
            bnds = cu[1:batch_size].long()
            final_loss_mask = final_loss_mask.index_fill(0, bnds, False)

        # ----------------------------------------------------------------
        # Packed backbone forward.
        # ----------------------------------------------------------------
        hidden, seq_pad = self._packed_forward(
            inputs_embeds=inputs_embeds,
            cu_seqlens=cu,
            max_seqlen=max_seqlen,
            position_ids=pos_ids,
        )
        if seq_pad > 0:
            hidden = hidden[:, :-seq_pad, :]

        # loss_mask and token_ids as (1, N_total) for the CE path.
        loss_mask_2d = final_loss_mask.unsqueeze(0)  # (1, N_total)
        token_ids_2d = token_ids_flat.unsqueeze(0)  # (1, N_total)
        attn_mask_2d = torch.ones(1, n_total, dtype=torch.bool, device=device)

        loss = self._compute_lm_ce(
            lm_head=lm_head,
            hidden_states=hidden,
            token_ids_full=token_ids_2d,
            attention_mask=attn_mask_2d,
            loss_mask_full=loss_mask_2d,
            chunk_size=chunk_size,
        )

        if return_hidden_states:
            return InterleavedForwardOutput(
                loss=loss,
                hidden_states=hidden,
                token_ids=token_ids_2d,
                loss_mask=loss_mask_2d,
                attention_mask=attn_mask_2d,
                lm_head=lm_head,
            )
        return loss

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
                    raise ValueError(f"segment batch size mismatch: {ids.size(0)} vs {batch_size}")
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
                        ids.shape,
                        bool(seg.loss),
                        dtype=torch.bool,
                        device=ids.device,
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
                    raise ValueError(f"segment batch size mismatch: {emb.size(0)} vs {batch_size}")
                if target_dtype is None:
                    target_dtype = emb.dtype
                embeds_list.append(emb)
                k = emb.size(1)
                tokens_list.append(torch.zeros(batch_size, k, dtype=torch.long, device=emb.device))
                loss_list.append(torch.zeros(batch_size, k, dtype=torch.bool, device=emb.device))
            else:
                raise TypeError(f"unknown segment type: {type(seg).__name__}")

        inputs_embeds = torch.cat(
            [e.to(dtype=target_dtype) for e in embeds_list],
            dim=1,
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
        chunk_size: int | None,
    ) -> torch.Tensor:
        """Dispatch CE computation among chunked, Liger, and optional CCE paths.

        Controlled by the module-level flag ``self._use_liger_ce`` (default
        True if set on the decoder, matching the trainer gate). When Liger
        is installed *and* the flag is on, we call ``liger_chunked_ce_loss``
        which never materialises the full ``(B, S, V)`` logits tensor.
        Explicit ``BGKIT_DECODER_CE_IMPL=cce`` / ``cce_exact`` /
        ``torch_compile`` requests route through Apple CCE when available.
        """
        chunk_size = _resolve_ce_chunk_size(chunk_size)
        ce_impl = _resolve_lm_ce_impl(getattr(self, "_lm_ce_impl", None))

        if ce_impl not in {"auto", "chunked", "liger"}:
            from bgkit.utils.cce_integration import cut_cross_entropy_lm_ce

            return cut_cross_entropy_lm_ce(
                hidden_states=hidden_states,
                lm_head_weight=lm_head.weight,
                lm_head_bias=getattr(lm_head, "bias", None),
                labels=token_ids_full,
                attention_mask=attention_mask,
                loss_mask=loss_mask_full,
                impl=ce_impl,
                chunk_size=chunk_size,
                strict=os.environ.get("BGKIT_DECODER_CE_STRICT", "0") == "1",
            )

        use_liger = getattr(self, "_use_liger_ce", False)
        if ce_impl == "liger" or (ce_impl == "auto" and use_liger):
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

        Used by :meth:`forward_interleaved_with_loss` for the segment-based
        API (used by KRKBTrainer). Accepts ``(B, S, D)`` + ``(B, S)`` mask
        and delegates to the backbone in ``(B, S, D)`` form without
        ``cu_seqlens``. This path is retained for the interleaved-segment
        callers until Wave 3 trainer rewrites land.

        Returns ``(hidden_states, seq_pad)`` where ``seq_pad`` is the number
        of alignment-padding positions appended to the end; the caller strips
        them before computing loss.
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
                    inputs_embeds=padded_embeds,
                    attention_mask=padded_mask,
                ).last_hidden_state
        else:
            hidden = inner_model(
                inputs_embeds=padded_embeds,
                attention_mask=padded_mask,
            ).last_hidden_state
        return hidden, seq_pad

    def forward_interleaved_with_loss(
        self,
        segments: list[Segment],
        *,
        attention_mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
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
            segments,
            embed_fn,
        )

        b, s, _ = inputs_embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                b,
                s,
                dtype=torch.bool,
                device=inputs_embeds.device,
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
            lora_config: Dict with keys: r, alpha, dropout (optional, default 0.0),
                target_modules (list of module name strings), and optional
                dtype/adapter_dtype. Decoder adapters default to the base
                model dtype. ``implementation`` defaults to ``"peft"``;
                ``"native"`` is retained as a compatibility/debug path.
        """
        rank = int(lora_config.get("r", 16))
        alpha = float(lora_config.get("alpha", 32))
        dropout = float(lora_config.get("dropout", 0.0))
        target_modules = tuple(
            lora_config.get(
                "target_modules",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
        )
        adapter_dtype = self._resolve_lora_adapter_dtype(
            lora_config.get("adapter_dtype", lora_config.get("dtype"))
        )
        implementation = str(
            lora_config.get(
                "implementation",
                os.environ.get("BGKIT_DECODER_LORA_IMPL", "peft"),
            )
        ).strip().lower()
        if implementation in {"native", "bgkit", "lightweight"}:
            wrapped = self._apply_native_lora(
                target_modules=target_modules,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                adapter_dtype=adapter_dtype or torch.float32,
            )
            cast_lora_params = wrapped
            self._lora_impl = "native"
        elif implementation == "peft":
            cast_lora_params = self._apply_peft_lora(
                target_modules=target_modules,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                adapter_dtype=adapter_dtype,
                autocast_adapter_dtype=bool(
                    lora_config.get("autocast_adapter_dtype", adapter_dtype is None)
                ),
            )
            self._lora_impl = "peft"
        else:
            raise ValueError(
                "decoder LoRA implementation must be native or peft; "
                f"got {implementation!r}"
            )
        self._has_lora = True
        self._lora_target_modules = target_modules
        self._lora_scaling = alpha / max(rank, 1)

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        logger.info(
            "decoder_lora_applied",
            implementation=self._lora_impl,
            trainable_params=trainable,
            total_params=total,
            ratio=f"{trainable / total:.4f}",
            r=rank,
            alpha=alpha,
            dropout=dropout,
            adapter_dtype=str(adapter_dtype) if adapter_dtype is not None else "peft",
            cast_lora_params=cast_lora_params,
        )

    def _apply_peft_lora(
        self,
        *,
        target_modules: tuple[str, ...],
        rank: int,
        alpha: float,
        dropout: float,
        adapter_dtype: torch.dtype | None,
        autocast_adapter_dtype: bool,
    ) -> int:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=list(target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        try:
            self.backbone = get_peft_model(
                self.backbone,
                config,
                autocast_adapter_dtype=bool(autocast_adapter_dtype),
            )
        except TypeError:
            self.backbone = get_peft_model(self.backbone, config)
        return self._cast_lora_adapters(adapter_dtype)

    def _apply_native_lora(
        self,
        *,
        target_modules: tuple[str, ...],
        rank: int,
        alpha: float,
        dropout: float,
        adapter_dtype: torch.dtype,
    ) -> int:
        self.backbone.requires_grad_(False)
        targets = set(target_modules)
        wrapped_params = 0
        for parent in list(self.backbone.modules()):
            for child_name, child in list(parent.named_children()):
                if child_name not in targets or not isinstance(child, nn.Linear):
                    continue
                wrapper = DecoderLoRALinear(
                    child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    adapter_dtype=adapter_dtype,
                )
                setattr(parent, child_name, wrapper)
                wrapped_params += wrapper.lora_A.numel() + wrapper.lora_B.numel()
        if wrapped_params == 0:
            raise ValueError(
                f"decoder LoRA found no nn.Linear target modules in {sorted(targets)!r}"
            )
        return wrapped_params

    def _base_lora_dtype(self) -> torch.dtype:
        """Return the decoder's compute dtype for LoRA adapter matmuls."""
        try:
            inner_model, _lm_head = self._get_inner_model_and_head()
            return inner_model.get_input_embeddings().weight.dtype
        except Exception:
            return next(self.backbone.parameters()).dtype

    def _resolve_lora_adapter_dtype(self, requested: object | None) -> torch.dtype | None:
        """Resolve LoRA adapter dtype.

        ``None`` means use ``BGKIT_DECODER_LORA_DTYPE`` or ``base``. Returning
        ``None`` means keep PEFT's dtype, useful for legacy fp32-adapter runs.
        """
        value = requested
        if value is None:
            value = os.environ.get("BGKIT_DECODER_LORA_DTYPE", "base")
        if isinstance(value, torch.dtype):
            return value
        key = str(value).strip().lower()
        if key in {"", "base", "model", "auto"}:
            return self._base_lora_dtype()
        if key in {"peft", "keep", "legacy"}:
            return None
        if key in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if key in {"fp16", "float16", "half"}:
            return torch.float16
        if key in {"fp32", "float32", "full"}:
            return torch.float32
        raise ValueError(
            "decoder LoRA dtype must be one of base, bf16, fp16, fp32, "
            f"or peft; got {value!r}"
        )

    def _cast_lora_adapters(self, dtype: torch.dtype | None) -> int:
        """Cast PEFT LoRA A/B modules to ``dtype`` and return parameter count."""
        if dtype is None:
            return 0
        count = 0
        for module in self.backbone.modules():
            for attr in ("lora_A", "lora_B"):
                adapters = getattr(module, attr, None)
                if not isinstance(adapters, nn.ModuleDict):
                    continue
                for adapter in adapters.values():
                    adapter.to(dtype=dtype)
                    count += sum(p.numel() for p in adapter.parameters())
        return count

    def merge_lora(self) -> dict:
        """Merge LoRA adapters into base weights and return a clean state dict.

        Produces a state dict with standard ReconstructionDecoder keys (no
        PeftModel prefix, no lora_A/B keys) so downstream trainers can load
        it as a plain decoder.
        """
        if not self._has_lora:
            return self.state_dict()
        if self._lora_impl == "native":
            return self._merge_native_lora()
        return self._merge_peft_lora()

    def _merge_peft_lora(self) -> dict:
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
                clean_key = decoder_prefix + key[len(peft_prefix) :]
            else:
                clean_key = key

            # Strip .base_layer. inserted by peft for LoRA target modules
            clean_key = clean_key.replace(".base_layer.", ".")

            # If this is a LoRA target weight, merge A/B into it
            peft_key = key.split(".base_layer.weight")[0] if ".base_layer.weight" in key else None
            if peft_key is None:
                peft_key = key.split(".weight")[0] if key.endswith(".weight") else None
            if peft_key and peft_key in lora_pairs:
                pair = lora_pairs[peft_key]
                if "A" in pair and "B" in pair:
                    val = val + scaling * (pair["B"] @ pair["A"])

            merged[clean_key] = val

        return merged

    def _merge_native_lora(self) -> dict:
        sd = self.state_dict()
        modules = dict(self.named_modules())
        merged = {}
        for key, val in sd.items():
            if key.endswith(".lora_A") or key.endswith(".lora_B"):
                continue
            clean_key = key.replace(".base_layer.", ".")
            if key.endswith(".base_layer.weight"):
                module_key = key[: -len(".base_layer.weight")]
                module = modules.get(module_key)
                if isinstance(module, DecoderLoRALinear):
                    val = module.merged_weight()
            merged[clean_key] = val
        return merged

    @torch.no_grad()
    def generate_with_single_splice(
        self,
        *,
        survivor_embeddings: torch.Tensor,
        survivor_cu_seqlens: torch.Tensor,
        prefix_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> GenerationOutput:
        """Autoregressive generation with one in-sequence embedding splice.

        Runs a custom B=1 per-sample decode loop driven by
        ``backbone.forward`` — not ``backbone.generate``. This avoids the
        dense-mask path that HF's ``generate`` would construct.

        Prefill: one ``backbone.forward`` per sample with
        ``inputs_embeds=(1, L, D)``, ``position_ids=(1, L)``,
        ``use_cache=True``. Captures ``past_key_values``.

        Decode step: for ``t`` in ``[0, max_new_tokens)``, calls
        ``backbone.forward(input_ids=(1, 1), past_key_values=past,
        use_cache=True)`` with ``position_ids=(1, 1)`` set to ``L + t``; the
        new token is sampled / argmax'd, appended to the output, and the loop
        stops on EOS or suffix-match.

        Parameters
        ----------
        survivor_embeddings:
            Flat ``(K_total, D)`` survivors (one pack for all samples, B
            may be 1 or more — each sample is processed independently in
            sequence).
        survivor_cu_seqlens:
            Shape ``(B+1,)`` int32. Boundary indices into
            ``survivor_embeddings`` for each sample.
        prefix_ids:
            ``(L_pre,)`` int64 — token IDs appearing before the survivors.
            Single-sample scalar (no batch dimension).
        suffix_ids:
            ``(L_suf,)`` int64 — token IDs to match against at the end of
            generation. Used for suffix stripping only; not fed to the model.
        tokenizer:
            Tokenizer used for text decoding and EOS detection.
        max_new_tokens:
            Maximum number of tokens to generate per sample.
        temperature:
            Sampling temperature. ``0.0`` means greedy argmax.

        Returns
        -------
        :class:`GenerationOutput` with per-sample content IDs and decoded text.
        """
        batch_size = int(survivor_cu_seqlens.shape[0]) - 1
        device = survivor_embeddings.device

        inner_model, lm_head = self._get_inner_model_and_head()
        embed_fn = inner_model.get_input_embeddings()
        try:
            target_dtype = embed_fn.weight.dtype
        except AttributeError:
            target_dtype = survivor_embeddings.dtype

        eos_id = getattr(tokenizer, "eos_token_id", None)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        suffix_ids = suffix_ids.to(device=device, dtype=torch.long)
        suf_len = int(suffix_ids.shape[0])
        surv_cu = survivor_cu_seqlens.tolist()

        do_sample = temperature > 0.0
        prefix_ids_dev = prefix_ids.to(device=device, dtype=torch.long)

        content_ids_list: list[torch.Tensor] = []
        full_ids_list: list[torch.Tensor] = []

        for b in range(batch_size):
            k_start, k_end = surv_cu[b], surv_cu[b + 1]
            surv = survivor_embeddings[k_start:k_end].to(dtype=target_dtype)  # (K_i, D)

            # Build prefill embeddings: [prefix | survivors].
            emb_pre = embed_fn(prefix_ids_dev).to(dtype=target_dtype)  # (L_pre, D)
            prefill_emb = torch.cat([emb_pre, surv], dim=0)  # (L_prefill, D)
            l_prefill = int(prefill_emb.shape[0])

            # Prefill: (1, L_prefill, D)
            prefill_emb_b = prefill_emb.unsqueeze(0)
            pos_ids_prefill = torch.arange(l_prefill, device=device).unsqueeze(0)  # (1, L)

            # Pass cu_seq_lens_q/k so the attention backend can handle
            # the single-segment prefill as a packed (B=1) sequence.
            # Prefill: Q = K = [0, L_prefill]; max_q = max_k = L_prefill.
            cu_prefill = torch.tensor([0, l_prefill], dtype=torch.int32, device=device)

            prefill_out = inner_model(
                inputs_embeds=prefill_emb_b,
                position_ids=pos_ids_prefill,
                use_cache=True,
                cu_seq_lens_q=cu_prefill,
                cu_seq_lens_k=cu_prefill,
                max_length_q=l_prefill,
                max_length_k=l_prefill,
            )
            past_kv = prefill_out.past_key_values

            # Get the first token from prefill logits.
            first_logits = lm_head(prefill_out.last_hidden_state[:, -1:, :])  # (1, 1, V)
            if do_sample:
                first_token = _sample_token(first_logits.squeeze(0), temperature)
            else:
                first_token = first_logits[0, 0].argmax().unsqueeze(0)  # (1,)

            generated: list[torch.Tensor] = [first_token]
            stopped = eos_id is not None and first_token.item() == eos_id

            # Decode loop: one step at a time.
            #
            # NOTE: We deliberately do NOT forward ``cu_seq_lens_q/k`` or
            # ``max_length_q/k`` through HF's TransformersKwargs path during
            # cached decode. Those kwargs are consumed by *every* decoder
            # layer via ``**kwargs``, and on Qwen3.5's hybrid stack the
            # DeltaNet (``linear_attn``) layers propagate their input shape
            # through the stock HF forward, which unpacks
            # ``batch_size, seq_len, _ = hidden_states.shape`` — passing the
            # packed kwargs disturbs that invariant on the B=1 single-step
            # case. Instead, we let HF's cached-generation path drive each
            # layer normally: DeltaNet uses its recurrent state from
            # ``cache_params``, and full-attention layers receive
            # ``(1, H, Lq, D)`` / ``(1, H, Lk, D)`` q/k with the extended
            # K coming from ``past_key_values``. Our FA4 backend
            # (``bgkit_flash_attention_4_forward``) synthesizes single-sample
            # cu_seqlens from the 4D q/k shapes when no packed kwargs are
            # supplied — see the fallback in ``attention_backend.py``.
            for t in range(1, max_new_tokens):
                if stopped:
                    break

                # generated[-1] is shape (1,) (from either argmax().unsqueeze(0)
                # in the greedy path or multinomial(...).squeeze(0) in the sample
                # path). The backbone expects (B=1, L=1) int64 ids, so we need a
                # single .unsqueeze(0) to add the batch dim. A prior version
                # called .unsqueeze(0) twice and produced a (1,1,1) tensor, which
                # then embedded to (1,1,1,D) — Qwen3.5 full-attn layers went on
                # to write 5D key/value states into the KV cache, and the second
                # decode step crashed in cache_utils.update when concatenating a
                # (now consistent) 5D stored tensor with incoming 5D. The
                # DeltaNet layers also crash earlier on 4D hidden_states via
                # ``batch_size, seq_len, _ = hidden_states.shape``. Either way,
                # the second unsqueeze was the root cause.
                cur_id_2d = generated[-1].unsqueeze(0)  # (1,) → (1, 1)
                pos = torch.tensor([[l_prefill + t - 1]], device=device, dtype=torch.long)

                step_out = inner_model(
                    input_ids=cur_id_2d,
                    position_ids=pos,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                past_kv = step_out.past_key_values

                step_logits = lm_head(step_out.last_hidden_state[:, -1:, :])  # (1, 1, V)
                if do_sample:
                    new_token = _sample_token(step_logits.squeeze(0), temperature)
                else:
                    new_token = step_logits[0, 0].argmax().unsqueeze(0)  # (1,)

                generated.append(new_token)
                if eos_id is not None and new_token.item() == eos_id:
                    stopped = True

            # Build output tensor.
            gen_ids = torch.cat(generated, dim=0)  # (T,)
            full_ids_list.append(gen_ids)

            # Strip EOS and padding from the end.
            seq = gen_ids
            if eos_id is not None:
                while seq.shape[0] > 0 and seq[-1].item() == eos_id:
                    seq = seq[:-1]
            if pad_id is not None:
                while seq.shape[0] > 0 and seq[-1].item() == pad_id:
                    seq = seq[:-1]

            # Strip suffix if present.
            if suf_len > 0 and seq.shape[0] >= suf_len and seq[-suf_len:].equal(suffix_ids):
                seq = seq[:-suf_len]

            content_ids_list.append(seq)

        content_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in content_ids_list]
        return GenerationOutput(
            content_ids=content_ids_list,
            content_text=content_text,
            full_ids=full_ids_list,
        )


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------


def _sample_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sample a single token from ``logits`` at the given temperature.

    Parameters
    ----------
    logits:
        Shape ``(1, V)`` — raw logits for a single position.
    temperature:
        Must be > 0.

    Returns
    -------
    Tensor
        Shape ``(1,)`` int64 — the sampled token index.
    """
    probs = torch.softmax(logits.float() / max(temperature, 1e-8), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(0)
