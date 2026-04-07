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

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

logger = structlog.get_logger()


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

    Wraps Qwen3.5-0.8B with prefix-conditioning: survivor embeddings are
    prepended to the target sequence and attended to via standard causal
    self-attention. No architectural changes to the underlying model.

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

        if nvfp4:
            self.enable_nvfp4()

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
        for name, module in self.backbone.named_modules():
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

    def forward_with_loss(
        self,
        survivor_embeddings: torch.Tensor,
        target_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        survivor_attention_mask: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        chunk_size: int = 256,
        loss_type: str = "ce",
        teacher_logprobs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fused forward + loss without materializing full (B, S, V) logits.

        Runs the inner model (skipping lm_head), then computes loss in chunks
        along the sequence dimension. Each chunk's logits are computed and
        discarded, keeping peak memory proportional to chunk_size * vocab_size.

        Args:
            survivor_embeddings: (batch, num_survivors, hidden_dim) from BgKIT.
            target_ids: (batch, target_len) token ids for teacher-forced generation.
            target_attention_mask: (batch, target_len) attention mask for targets.
            survivor_attention_mask: (batch, num_survivors) mask for real survivors.
            loss_mask: (batch, target_len) optional mask for content-only tokens.
            chunk_size: Sequence chunk size for chunked CE computation.
            loss_type: "ce" for cross-entropy (Phase 1), "kl" for KL divergence
                (Phase 2a model ladder distillation).
            teacher_logprobs: (batch, target_len, vocab_size) teacher log-probs
                for KL loss. Required when loss_type="kl".

        Returns:
            Scalar loss tensor.
        """
        inner_model, lm_head = self._get_inner_model_and_head()

        # Embed target tokens and concatenate with survivors
        target_emb = inner_model.get_input_embeddings()(target_ids)
        combined = torch.cat([survivor_embeddings, target_emb], dim=1)
        combined_mask = torch.cat([survivor_attention_mask, target_attention_mask], dim=1)

        # Forward through inner model (no lm_head)
        # NVFP4/FP8 requires batch*seq divisible by 8 and hidden_dim by 16.
        # Pad sequence length to satisfy alignment, mask padding via attention_mask.
        seq_pad = 0
        if self._use_te:
            import transformer_engine.pytorch as te

            b, s, _h = combined.shape
            # NVFP4 needs batch*seq divisible by 16 (NVFP4_BLOCK_SIZE)
            align = 16
            if (b * s) % align != 0:
                for seq_pad in range(1, align + 1):
                    if (b * (s + seq_pad)) % align == 0:
                        break
                pad_emb = combined.new_zeros(b, seq_pad, combined.size(-1))
                combined = torch.cat([combined, pad_emb], dim=1)
                pad_mask = combined_mask.new_zeros(b, seq_pad)
                combined_mask = torch.cat([combined_mask, pad_mask], dim=1)

        if self._use_te:
            with te.fp8_autocast(enabled=True, fp8_recipe=self._te_recipe):
                hidden = inner_model(
                    inputs_embeds=combined, attention_mask=combined_mask,
                ).last_hidden_state
        else:
            hidden = inner_model(
                inputs_embeds=combined, attention_mask=combined_mask,
            ).last_hidden_state

        # Strip seq padding and slice out target portion hidden states
        if seq_pad > 0:
            hidden = hidden[:, :-seq_pad, :]
        target_hidden = hidden[:, survivor_embeddings.size(1):, :]

        if loss_type == "ce":
            return _chunked_lm_ce(
                lm_head, target_hidden, target_ids,
                target_attention_mask, loss_mask, chunk_size,
            )
        elif loss_type == "kl":
            raise NotImplementedError("KL loss type for Phase 2a not yet implemented")
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

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
