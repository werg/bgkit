"""Phase 2 Step 5 trainer: target LLM injection via tool-call framing.

Extends KRTrainer with:
- Target LLM loading: Qwen3.5-35B in 4-bit (QLoRA via peft)
- Projection block extension: 1024 -> 2560 (block-diagonal initialization)
- Tool-call framed injection: BgKIT survivors are injected as <tool_response>
  content within Qwen3.5's chat template, replacing placeholder token embeddings
- Multi-track data mixing: 40% A / 15% B / 15% C / 5% reconstruction / 25% no-injection
- Full backprop from target LLM loss through projection block (optionally through compressor)
- Decoder regularizer on a subset of examples

Injection approach:
  1. Build a chat-template sequence with a <tool_response> region containing
     K placeholder tokens (one per BgKIT survivor)
  2. Tokenize via chat template, using sentinel-based boundary detection
     (same pattern as ChatReproDataset) to find the exact token positions
     of the placeholder region
  3. Get input embeddings from the target LM for the full sequence
  4. Replace embeddings at placeholder positions with projected BgKIT survivors
  5. Forward through the target LM with labels for next-token prediction
"""

from __future__ import annotations

import json
import random

import structlog
import torch
import torch.nn as nn

from bgkit.data.chat_template import CONTENT_SENTINEL
from bgkit.models.target_lm import TargetLMWithInjection
from bgkit.training.phase2.kr_trainer import KRTrainer, _Phase2Model

logger = structlog.get_logger()

# Tool call for requesting BgKIT context, matching Qwen3.5's native JSON format.
_BGKIT_TOOL_CALL = {
    "name": "bgkit_retrieve_context",
    "arguments": {"source": "compressed_knowledge"},
}


class _ProjectionExtension(nn.Module):
    """Extends projection from encoder hidden_dim (1024) to target LLM hidden_dim (2560).

    Uses block-diagonal initialization: the original 1024-dim projection is
    preserved, and the extension dimensions are initialized near-zero.
    """

    def __init__(self, source_dim: int = 1024, target_dim: int = 2560):
        super().__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.extension = nn.Linear(source_dim, target_dim, bias=False)
        with torch.no_grad():
            nn.init.zeros_(self.extension.weight)
            # Block-diagonal: first source_dim output dims map 1:1 from input
            self.extension.weight[:source_dim, :source_dim] = (
                torch.eye(source_dim) * 0.1
            )
            if target_dim > source_dim:
                nn.init.normal_(
                    self.extension.weight[source_dim:, :],
                    mean=0.0,
                    std=0.01,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extension(x)


def _build_injection_frame(
    tokenizer,
    question_text: str,
    num_survivors: int,
) -> tuple[torch.Tensor, int, int]:
    """Build chat-template token IDs with placeholder positions for injection.

    Uses the sentinel-based boundary detection pattern from ChatReproDataset:
    1. Build messages with a sentinel string as tool response content
    2. Render via apply_chat_template (tokenize=False)
    3. Split on sentinel to find prefix/suffix boundaries
    4. Tokenize piecewise to get exact token positions

    The sentinel occupies the exact span where BgKIT survivors will be injected.
    We use num_survivors copies of a single-char placeholder so the tokenizer
    produces a predictable number of tokens.

    Args:
        tokenizer: Target LLM tokenizer (Qwen3.5-35B).
        question_text: User question / issue text.
        num_survivors: Number of BgKIT survivors (K) to inject.

    Returns:
        (token_ids, inject_start, inject_end) where:
        - token_ids: (seq_len,) full tokenized chat sequence with placeholder tokens
        - inject_start: index where placeholder tokens begin
        - inject_end: index where placeholder tokens end (inject_end - inject_start = K)
    """
    tool_call_json = json.dumps(_BGKIT_TOOL_CALL, ensure_ascii=False)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable assistant with access to retrieved context "
                "from BgKIT compressed knowledge sources."
            ),
        },
        {"role": "user", "content": question_text},
        {
            "role": "assistant",
            "content": f"<tool_call>\n{tool_call_json}\n</tool_call>",
        },
        {
            "role": "tool",
            "content": CONTENT_SENTINEL,
        },
    ]

    # Render template without tokenizing to get a string we can split
    try:
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        # Fallback for tokenizers without apply_chat_template support
        full_text = (
            "<|im_start|>system\n"
            "You are a knowledgeable assistant with access to retrieved context "
            "from BgKIT compressed knowledge sources.<|im_end|>\n"
            f"<|im_start|>user\n{question_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"<tool_call>\n{tool_call_json}\n</tool_call><|im_end|>\n"
            f"<|im_start|>tool\n{CONTENT_SENTINEL}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    # Validate sentinel uniqueness
    sentinel_count = full_text.count(CONTENT_SENTINEL)
    if sentinel_count != 1:
        raise ValueError(
            f"Expected exactly 1 sentinel in template, found {sentinel_count}."
        )

    # Split on sentinel to get prefix and suffix strings
    prefix_str, suffix_str = full_text.split(CONTENT_SENTINEL, 1)

    # Tokenize prefix and suffix separately
    prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_str, add_special_tokens=False)

    # The injection span starts right after prefix
    inject_start = len(prefix_ids)
    inject_end = inject_start + num_survivors

    # Build full token IDs: prefix + placeholder tokens + suffix
    # Use pad_token_id as placeholder (will be replaced with projected embeddings)
    pad_id = tokenizer.pad_token_id or 0
    placeholder_ids = [pad_id] * num_survivors
    full_ids = prefix_ids + placeholder_ids + suffix_ids

    return torch.tensor(full_ids, dtype=torch.long), inject_start, inject_end


class KRStep5Trainer(KRTrainer):
    """Step 5: target LLM injection with QLoRA via tool-call embedding replacement.

    Loads Qwen3.5-35B in 4-bit quantization, applies LoRA adapters,
    and trains with BgKIT context injection via tool-call embedding replacement.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._target_lm = None
        self._target_lm_wrapper: TargetLMWithInjection | None = None
        self._projection_ext = None
        self._target_tokenizer = None
        self._no_injection_fraction = float(
            self.step_cfg.get("data_mixing", {}).get("no_injection", 0.25),
        )
        self._decoder_reg_weight = float(
            self.step_cfg.get("decoder_regularizer", {}).get("weight", 0.1),
        )
        self._decoder_reg_fraction = float(
            self.step_cfg.get("decoder_regularizer", {}).get("fraction", 0.2),
        )

    def setup(self) -> None:
        from transformers import AutoTokenizer

        # Call parent setup for encoder, decoder, datasets, etc.
        super().setup()

        # Load target LLM
        self._load_target_lm()

        # Target tokenizer for chat template construction
        target_cfg = self.step_cfg.get("target_lm", {})
        target_name = target_cfg.get("backbone_name", "Qwen/Qwen3.5-35B")
        self._target_tokenizer = AutoTokenizer.from_pretrained(
            target_name, trust_remote_code=True,
        )
        if self._target_tokenizer.pad_token_id is None:
            self._target_tokenizer.pad_token_id = self._target_tokenizer.eos_token_id

        # Build projection extension
        proj_cfg = self.step_cfg.get("projection_extension", {})
        source_dim = int(proj_cfg.get("source_dim", 1024))
        target_dim = int(proj_cfg.get("target_dim", 2560))
        self._projection_ext = _ProjectionExtension(source_dim, target_dim)
        self._projection_ext.to(self.device)

        # Rebuild model container with target LM
        self.model = _Phase2Model(
            decoder=self.decoder,
            encoder=self.encoder,
            ice=self.ice,
            topic_embeddings=self.topic_embeddings,
            cache_projection=self._cache_projection,
            projection_block=(
                self.encoder.projection_block
                if self.encoder and hasattr(self.encoder, "projection_block")
                else None
            ),
        )
        # Register projection extension as submodule
        self.model.projection_ext = self._projection_ext
        self.model.to(self.device)

        # Rebuild optimizer with target LM LoRA params + projection extension
        param_groups = self._build_optimizer_groups()
        if not param_groups:
            raise ValueError("Step 5 trainer has no trainable parameters")
        self.optimizer = self._create_optimizer(
            param_groups,
            default_lr=float(self.step_cfg.get("lr", 1.0e-4)),
        )

    def _load_target_lm(self) -> None:
        """Load target LLM with 4-bit quantization and LoRA, wrapped for injection."""
        target_cfg = self.step_cfg.get("target_lm", {})
        model_name = target_cfg.get("backbone_name", "Qwen/Qwen3.5-35B")
        quant = target_cfg.get("quantization", "4bit")

        logger.info("step5_loading_target_lm", model=model_name, quantization=quant)

        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "Step 5 requires peft and bitsandbytes. "
                "Install with: pip install peft bitsandbytes",
            ) from exc

        # 4-bit quantization config
        if quant == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        target_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
        )

        # Prepare for k-bit training
        target_model = prepare_model_for_kbit_training(target_model)

        # Apply LoRA
        lora_rank = int(target_cfg.get("lora_rank", 64))
        lora_alpha = int(target_cfg.get("lora_alpha", 128))
        lora_targets = list(target_cfg.get("lora_target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]))

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_targets,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        target_model = get_peft_model(target_model, lora_config)
        target_model.print_trainable_parameters()

        self._target_lm = target_model

        # Resolve tool_response token IDs for the TargetLMWithInjection wrapper
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tr_start_ids = tokenizer.encode("<tool_response>", add_special_tokens=False)
        tr_end_ids = tokenizer.encode("</tool_response>", add_special_tokens=False)
        tr_start_id = tr_start_ids[0] if tr_start_ids else 0
        tr_end_id = tr_end_ids[0] if tr_end_ids else 0

        self._target_lm_wrapper = TargetLMWithInjection(
            model=target_model,
            tool_response_start_token_id=tr_start_id,
            tool_response_end_token_id=tr_end_id,
        )

        trainable = sum(
            p.numel() for p in target_model.parameters() if p.requires_grad
        )
        logger.info("step5_target_lm_loaded", trainable_params=trainable)

    def _build_optimizer_groups(self) -> list[dict]:
        """Build optimizer groups including target LM LoRA and projection extension."""
        groups = super()._build_optimizer_groups()

        # Target LM LoRA params
        if self._target_lm is not None:
            lora_params = [p for p in self._target_lm.parameters() if p.requires_grad]
            if lora_params:
                groups.append({
                    "params": lora_params,
                    "lr": float(self.step_cfg.get("decoder_lr", 2.0e-4)),
                })

        # Projection extension params
        if self._projection_ext is not None:
            ext_params = [p for p in self._projection_ext.parameters() if p.requires_grad]
            if ext_params:
                groups.append({
                    "params": ext_params,
                    "lr": float(self.step_cfg.get("lr", 1.0e-4)),
                })

        return groups

    def _forward_backward(self, batch) -> dict[str, float]:
        """Forward pass through target LLM with tool-call framed BgKIT injection.

        When injecting:
        1. Get compressed survivors from BgKIT pipeline
        2. Project to target LLM dimension (1024 -> 2560)
        3. Build chat-template sequence with placeholder tokens in <tool_response>
        4. Replace placeholder embeddings with projected survivors
        5. Append answer tokens and forward through target LLM
        """
        # Decide whether to inject BgKIT context (vs no-injection baseline)
        inject = random.random() > self._no_injection_fraction

        target_ids = batch["target_token_ids"].to(self.device)
        target_attention_mask = batch["target_attention_mask"].to(self.device)
        target_loss_mask = batch["target_loss_mask"].to(self.device)

        if inject and self._target_lm is not None:
            # --- Step 1: Get compressed survivors ---
            prompt, prompt_mask = self._compose_prompt(batch)

            # --- Step 2: Project to target LLM dimension ---
            projected = self._projection_ext(prompt)  # (B, K, 2560)
            num_survivors = projected.size(1)
            batch_size = target_ids.size(0)

            # --- Step 3: Build tool-call framed sequences ---
            tokenizer = self._target_tokenizer
            question_ids = batch["question_token_ids"]
            question_mask = batch["question_attention_mask"]

            all_frame_ids = []
            all_inject_starts = []
            all_inject_ends = []

            for i in range(batch_size):
                q_len = int(question_mask[i].sum().item())
                q_text = tokenizer.decode(
                    question_ids[i, :q_len], skip_special_tokens=True,
                )
                frame_ids, inj_start, inj_end = _build_injection_frame(
                    tokenizer, q_text, num_survivors,
                )
                all_frame_ids.append(frame_ids)
                all_inject_starts.append(inj_start)
                all_inject_ends.append(inj_end)

            # Pad framed sequences to same length
            max_frame_len = max(ids.size(0) for ids in all_frame_ids)
            pad_id = tokenizer.pad_token_id or 0
            padded_frame_ids = torch.full(
                (batch_size, max_frame_len), pad_id,
                dtype=torch.long, device=self.device,
            )
            frame_mask = torch.zeros(
                batch_size, max_frame_len,
                dtype=torch.long, device=self.device,
            )
            for i, ids in enumerate(all_frame_ids):
                padded_frame_ids[i, :ids.size(0)] = ids.to(self.device)
                frame_mask[i, :ids.size(0)] = 1

            # --- Step 4: Get embeddings and inject survivors ---
            embed_layer = self._target_lm.get_input_embeddings()
            frame_embeds = embed_layer(padded_frame_ids)

            # Build injection_positions tensor for TargetLMWithInjection
            injection_positions = torch.zeros(
                batch_size, num_survivors, dtype=torch.long, device=self.device,
            )
            for i in range(batch_size):
                s = all_inject_starts[i]
                injection_positions[i] = torch.arange(
                    s, s + num_survivors, dtype=torch.long, device=self.device,
                )

            # Use TargetLMWithInjection.inject_survivors for the replacement
            frame_embeds = self._target_lm_wrapper.inject_survivors(
                padded_frame_ids, frame_embeds, projected, injection_positions,
            )

            # --- Step 5: Append answer tokens and forward ---
            answer_embeds = embed_layer(target_ids)
            combined_embeds = torch.cat([frame_embeds, answer_embeds], dim=1)
            combined_mask = torch.cat(
                [frame_mask, target_attention_mask.long()], dim=1,
            )

            # Labels: -100 for framed prefix, actual labels for answer region
            frame_labels = torch.full(
                (batch_size, max_frame_len), -100,
                dtype=torch.long, device=self.device,
            )
            answer_labels = target_ids.clone()
            answer_labels[~target_loss_mask.bool()] = -100
            labels = torch.cat([frame_labels, answer_labels], dim=1)

            outputs = self._target_lm(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                labels=labels,
            )
            target_lm_loss = outputs.loss
            prompt_tok_val = float(prompt_mask.sum().item() / prompt_mask.size(0))

        elif self._target_lm is not None:
            # No-injection: straight target LM forward
            target_labels = target_ids.clone()
            target_labels[~target_loss_mask.bool()] = -100
            outputs = self._target_lm(
                input_ids=target_ids,
                attention_mask=target_attention_mask,
                labels=target_labels,
            )
            target_lm_loss = outputs.loss
            prompt_tok_val = 0.0
        else:
            target_lm_loss = torch.tensor(0.0, device=self.device)
            prompt_tok_val = 0.0

        total_loss = target_lm_loss

        # Decoder regularizer on a subset
        decoder_reg_loss = 0.0
        if (
            self._decoder_reg_weight > 0
            and random.random() < self._decoder_reg_fraction
            and inject
        ):
            prompt_for_dec, mask_for_dec = self._compose_prompt(batch)
            reg_loss = self.decoder.forward_with_loss(
                prompt_for_dec,
                target_ids,
                target_attention_mask,
                mask_for_dec,
                loss_mask=target_loss_mask,
            )
            decoder_reg_loss = reg_loss.item()
            total_loss = total_loss + self._decoder_reg_weight * reg_loss

        (total_loss / self._accum_steps).backward()

        tlm_loss_val = (
            target_lm_loss.detach().item()
            if torch.is_tensor(target_lm_loss)
            else 0.0
        )
        return {
            "loss": total_loss.detach(),
            "target_lm_loss": tlm_loss_val,
            "decoder_reg_loss": decoder_reg_loss,
            "injected": float(inject),
            "prompt_tokens": prompt_tok_val,
            "target_ratio": self._target_ratio(),
        }
