"""Phase 1, Step 5: Compression training (4 objectives, curriculum).

Introduces the drop-flag mechanism with four core objectives:
1. Data reconstruction (primary, ~40%)
2. Description generation (~20%)
3. Structural/relational reconstruction (~15%)
4. Commit reproduction (~25%)

Curriculum: L0 objectives first, then L1 once L0 stabilizes.
Survivor selection: deterministic top-k by live ICE scoring with threshold.

The BgKIT encoder and decoder are both trainable (key change from Step 1
where BgKIT was frozen). ICE model remains frozen.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from pathlib import Path
from typing import ClassVar

import numpy as np
import structlog
import torch
from torch.utils.data import DataLoader, random_split

from bgkit.data.collators import collate_compression
from bgkit.data.datasets.compression_dataset import CompressionDataset
from bgkit.data.samplers import PackedTokenBudgetSampler
from bgkit.models.decoder import ReconstructionDecoder, normalize_decoder_family
from bgkit.models.encoder import BgKITEncoder, EncoderOutput
from bgkit.models.projection_block import effective_projection_counts, effective_projection_cu
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpoint_registry import resolve_checkpoint
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.training.gradient_utils import (
    configure_decoder_layerwise_split,
    maybe_enable_decoder_gradient_checkpointing,
    maybe_enable_frozen_decoder_kernels,
    maybe_enable_gradient_checkpointing,
    validate_decoder_lora_freeze_contract,
)
from bgkit.training.objectives.commit_reproduction import commit_reproduction_loss
from bgkit.training.objectives.data_reconstruction import data_reconstruction_loss
from bgkit.training.objectives.description_generation import description_generation_loss
from bgkit.training.objectives.structural_relational import structural_relational_loss
from bgkit.training.ratio_sampling import (
    build_ratio_sampler_config,
    resolve_anchor_grid,
    sample_ratio,
)
from bgkit.utils.attention_backend import (
    resolve_attention_implementation,
    resolve_decoder_attention_implementation,
)

logger = structlog.get_logger()

# Maps objective name to its loss function
_LOSS_FNS = {
    "data_reconstruction": data_reconstruction_loss,
    "description_generation": description_generation_loss,
    "structural_relational": structural_relational_loss,
    "commit_reproduction": commit_reproduction_loss,
}


class _DecoderOnlyL0PrefixCache:
    """Bounded LRU for frozen L0 prefix activations."""

    def __init__(self, max_bytes: int, *, device: str | torch.device = "cpu"):
        self.max_bytes = max(0, int(max_bytes))
        self.device = torch.device(device)
        self.current_bytes = 0
        self._entries: OrderedDict[str, tuple[dict, int]] = OrderedDict()

    @staticmethod
    def _entry_bytes(entry: dict) -> int:
        total = 0
        for key in ("hidden", "base_raw"):
            tensor = entry[key]
            total += int(tensor.numel() * tensor.element_size())
        return total

    def get(self, key: str) -> dict | None:
        found = self._entries.pop(key, None)
        if found is None:
            return None
        entry, size = found
        self._entries[key] = (entry, size)
        return entry

    def put(self, key: str, entry: dict) -> None:
        if self.max_bytes <= 0:
            return
        size = self._entry_bytes(entry)
        if size > self.max_bytes:
            return
        old = self._entries.pop(key, None)
        if old is not None:
            self.current_bytes -= old[1]
        self._entries[key] = (entry, size)
        self.current_bytes += size
        while self.current_bytes > self.max_bytes and self._entries:
            _, (_, evicted_size) = self._entries.popitem(last=False)
            self.current_bytes -= evicted_size

    def __len__(self) -> int:
        return len(self._entries)


class _DecoderOnlyL0PrefixCacheMissError(Exception):
    def __init__(self, hits: int, misses: int):
        super().__init__("decoder-only L0 prefix cache miss")
        self.hits = int(hits)
        self.misses = int(misses)


class CompressionTrainer(BaseTrainer):
    """Step 2: Compression training with multi-objective curriculum.

    Packed-attention pipeline (FA4 varlen). File batches run a single
    packed L0 forward. Repo batches run a single packed L0 forward across
    every file in every repo (segmented by ``cu_file_seqlens``), then
    regroup survivors per repo using ``cu_repo_seqlens`` and run a second
    single packed L1 forward across the whole microbatch. The previous
    per-sample L0 loop + per-repo L1 loop is gone — packing removes the
    quadratic-padding cost that motivated the serial path.

    Overrides train() with a custom loop that adds a pre-fetch hook for L1
    dataloader rebuild. When L1 is introduced (at curriculum step threshold),
    the dataset needs to be rebuilt between completing one train_step and
    fetching the next batch.
    """

    LIVE_CONFIG_FIELDS: ClassVar[dict[str, str]] = {
        "target_ratio_ramp_steps": "_target_ratio_ramp_steps",
        "target_ratio_start": "_target_ratio_start",
        "target_ratio_end": "_target_ratio_end",
        "l1_introduction_step": "_l1_introduction_step",
        "head_warmup_steps": "_head_warmup_steps",
    }

    LIVE_CONFIG_HANDLERS: ClassVar[dict[str, str]] = {
        "target_ratio": "_handle_target_ratio",
        "target_ratio_sampling_window_above": "_handle_ratio_sampling_window_above",
        "sample_target_ratio_during_training": "_handle_ratio_sampling_enabled",
        "target_ratio_anchor_sampling_prob": "_handle_ratio_sampling_anchor_prob",
        # Phase C/D head-supervision and head-aux weight knobs. Targets the
        # ``self._surv_l0`` / ``self._surv_l1`` LevelLossCfg dataclasses
        # populated at setup; updates take effect on the next batch.
        "forced_survivor_bce_weight_l0": "_handle_surv_l0_forced_bce_weight",
        "forced_survivor_bce_weight_l1": "_handle_surv_l1_forced_bce_weight",
        "forced_survivor_bce_anchor_ratio_l0": "_handle_surv_l0_forced_bce_anchor",
        "forced_survivor_bce_anchor_ratio_l1": "_handle_surv_l1_forced_bce_anchor",
        "utility_grad_loss_weight_l0": "_handle_surv_l0_utility_grad_weight",
        "utility_grad_loss_weight_l1": "_handle_surv_l1_utility_grad_weight",
        "decisiveness_loss_weight_l0": "_handle_surv_l0_decisiveness_weight",
        "decisiveness_loss_weight_l1": "_handle_surv_l1_decisiveness_weight",
        "moment_match_weight_l0": "_handle_surv_l0_moment_match_weight",
        "moment_match_weight_l1": "_handle_surv_l1_moment_match_weight",
        "min_survivors_loss_weight_l0": "_handle_surv_l0_min_survivors_weight",
        "min_survivors_loss_weight_l1": "_handle_surv_l1_min_survivors_weight",
        # Switch between Phase C (forced-mask override) and Phase D
        # (head selects). Read each step via cfg.training.get, so we
        # update the OmegaConf node directly.
        "use_forced_survivor_mask_l0": "_handle_use_forced_survivor_mask_l0",
    }

    def setup(self) -> None:
        """Load trainable encoder/decoder, create dataset and optimizer."""
        import gc

        tcfg = self.cfg.training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        freeze_cfg = tcfg.get("freeze", {})
        self._bgkit_frozen = bool(freeze_cfg.get("bgkit", False))
        self._decoder_only_fastpath = bool(
            self._bgkit_frozen and tcfg.get("decoder_only_fastpath", True)
        )
        attention_impl = resolve_attention_implementation(
            self.cfg.compute.get("attention_implementation", "auto")
        )

        # --- BgKIT encoder (trainable in Step 2) ---
        bgkit_cfg = self.cfg.model.bgkit
        backbone_name = bgkit_cfg.backbone_name
        backbone_revision = bgkit_cfg.get("backbone_revision", None)
        hidden_dim = bgkit_cfg.get("hidden_dim", 1024)
        projection_num_layers = int(bgkit_cfg.get("projection_num_layers", 1))
        bidi_warmup = self.cfg.training.get("bidi_warmup_steps", 1000)
        model_cfg = self.cfg.model
        ctrl_src = model_cfg.get("threshold_controller", {})
        target_ratio_start = float(tcfg.get("target_ratio_start", 0.30))
        threshold_controller_cfg = {
            "init_theta": float(
                ctrl_src.get("init_theta", 1.0 - 2.0 * target_ratio_start),
            ),
            "lr": float(ctrl_src.get("lr", 0.02)),
            "momentum": float(ctrl_src.get("momentum", 0.0)),
            "clamp": float(ctrl_src.get("clamp", 0.99)),
            "anchor_ratios": list(ctrl_src.get("anchor_ratios", [])) or None,
            "ratio_space": str(ctrl_src.get("ratio_space", "log")),
            "init_target_ratio": target_ratio_start,
            "default_query_ratio": target_ratio_start,
        }

        step1_checkpoint = self._resolve_step1_checkpoint()
        step1_state_dicts: dict | None = None
        if step1_checkpoint is not None:
            logger.info("loading_step1_checkpoint", path=step1_checkpoint)
            _, step1_state_dicts = load_checkpoint(Path(step1_checkpoint))
            step1_state_dicts.pop("optimizer", None)
            if "encoder" not in step1_state_dicts:
                raise ValueError(
                    f"Step 3 checkpoint missing 'encoder' key: {step1_checkpoint}. "
                    f"Found keys: {list(step1_state_dicts.keys())}"
                )

        # Auto-detect pruned architecture from state dict keys
        if step1_state_dicts:
            from bgkit.models.encoder import is_pruned_encoder_state_dict
            pruned = is_pruned_encoder_state_dict(step1_state_dicts["encoder"])
            logger.info(
                "loading_bgkit_encoder",
                model=backbone_name, revision=backbone_revision, pruned=pruned,
            )
            self.encoder = BgKITEncoder.from_pretrained_with_state_dict(
                backbone_name,
                step1_state_dicts.pop("encoder"),
                hidden_dim=hidden_dim,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                revision=backbone_revision,
                attn_implementation=attention_impl,
                bidi_warmup_steps=bidi_warmup,
                threshold_controller_cfg=threshold_controller_cfg,
                projection_num_layers=projection_num_layers,
            )
        else:
            logger.info(
                "loading_bgkit_encoder",
                model=backbone_name, revision=backbone_revision, pruned=False,
            )
            self.encoder = BgKITEncoder.from_pretrained(
                backbone_name,
                hidden_dim=hidden_dim,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                revision=backbone_revision,
                attn_implementation=attention_impl,
                bidi_warmup_steps=bidi_warmup,
                threshold_controller_cfg=threshold_controller_cfg,
                projection_num_layers=projection_num_layers,
            )
        self.encoder.to(device)
        gc.collect()

        # Encoder is trainable in Step 4
        self.encoder.requires_grad_(True)
        self.encoder.train()
        maybe_enable_gradient_checkpointing(
            self.encoder.l0.backbone, self.cfg,
        )

        # --- Decoder (trainable, with drift monitoring) ---
        decoder_cfg = tcfg.get("model", {}).get("decoder", self.cfg.model.decoder)
        decoder_family = normalize_decoder_family(decoder_cfg.get("family", "qwen35"))
        self.encoder.set_active_decoder_family(decoder_family)
        if self._decoder_only_fastpath and decoder_family == "falcon_h1":
            # The Falcon decoder run keeps BgKIT frozen, so Qwen encoder full
            # attention is a fixed feature extractor. On sm_121 the FA4 packed
            # varlen kernel can hard-segfault before Python can fall back; use
            # segmented SDPA unless the operator explicitly forces another mode.
            os.environ.setdefault("BGKIT_QWEN35_PACKED_ATTENTION", "sdpa")
        decoder_attention_impl = resolve_decoder_attention_implementation(
            self.cfg.compute.get(
                "decoder_attention_implementation",
                self.cfg.compute.get("attention_implementation", "auto"),
            ),
            decoder_family=decoder_family,
        )
        decoder_name = decoder_cfg.backbone_name
        decoder_revision = decoder_cfg.get("backbone_revision", None)
        logger.info("loading_decoder", model=decoder_name, revision=decoder_revision)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Build decoder WITHOUT NVFP4 first so checkpoint loads cleanly
        decoder_backbone = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=decoder_revision,
            attn_implementation=decoder_attention_impl,
            device_map=device,
        )
        decoder_hidden = int(decoder_backbone.get_input_embeddings().weight.shape[1])
        self.decoder = ReconstructionDecoder(
            decoder_backbone,
            hidden_dim=decoder_hidden,
            decoder_family=decoder_family,
        )
        self.decoder.set_lm_ce_impl(
            tcfg.get("decoder_ce_impl", self.cfg.compute.get("decoder_ce_impl", None))
        )
        self.decoder.set_lm_ce_strict(
            tcfg.get("decoder_ce_strict", self.cfg.compute.get("decoder_ce_strict", None))
        )
        logger.info(
            "decoder_ce_impl_selected",
            impl=self.decoder.lm_ce_impl,
            strict=self.decoder.lm_ce_strict,
        )
        configure_decoder_layerwise_split(self.decoder, self.cfg)

        # Load decoder from Step 1 checkpoint
        if step1_state_dicts is not None:
            decoder_sd = step1_state_dicts.pop(
                "decoder_merged", step1_state_dicts.pop("decoder", None)
            )
            if decoder_sd is not None:
                self.decoder.load_state_dict(decoder_sd)
        del step1_state_dicts
        gc.collect()

        # LoRA wrapping (after checkpoint load, before gradient checkpointing)
        self._decoder_lora = False
        lora_cfg = tcfg.get("decoder_lora", {})
        validate_decoder_lora_freeze_contract(self.cfg)
        if lora_cfg.get("enabled", False):
            self.decoder.apply_lora(lora_cfg)
            self._decoder_lora = True

        # NVFP4 disabled: TE's fp8_autocast saves non-deterministic scaling
        # tensors that break PyTorch gradient checkpointing's save/restore
        # invariants (122 vs 87 tensors between forward and recompute, causing
        # AssertionError in unpack_hook). This is a fundamental incompatibility
        # between TE's stateful fp8 context and PyTorch's checkpoint replay.
        # Bypassing the validation check isn't enough — the internal handle
        # mapping also breaks. Needs upstream TE fix.

        maybe_enable_decoder_gradient_checkpointing(self.decoder.backbone, self.cfg)

        # torch.compile disabled: sm_121 shared memory (101 KB) is too small for
        # inductor-generated Triton kernels (need ~148 KB). "reduce-overhead" and
        # "cudagraphs" backends also use inductor or suffer from dynamic shape
        # re-recording overhead. PeftModel isinstance fix landed in decoder.py
        # for when this becomes viable.

        # Optional Liger Kernel fused kernels. The module patcher is
        # Qwen-specific; Falcon-H1 uses stateful Mamba blocks and should rely
        # on the HF Mamba/causal-conv fast path plus the generic LM CE kernel.
        if tcfg.get("use_liger", True) and not self._decoder_only_fastpath:
            from bgkit.utils.liger_integration import (
                apply_liger_to_decoder,
                apply_liger_to_qwen35,
            )

            patch_rmsnorm = bool(tcfg.get("use_liger_rmsnorm", False))
            patch_swiglu = bool(tcfg.get("use_liger_swiglu", True))
            patch_rope = bool(tcfg.get("use_liger_rope", True))
            decoder_family = getattr(self.decoder, "decoder_family", "qwen35")
            decoder_qwen_liger = decoder_family == "qwen35"
            use_liger_ce = (
                bool(tcfg.get("use_liger_ce", True))
                and decoder_qwen_liger
                and self.decoder.lm_ce_impl in {"auto", "liger"}
            )
            enc_patched = apply_liger_to_qwen35(
                self.encoder,
                patch_rmsnorm=patch_rmsnorm,
                patch_swiglu=patch_swiglu,
                patch_rope=patch_rope,
            )
            # apply_liger_to_decoder no-ops on Falcon, matching the
            # decoder_qwen_liger gate.
            dec_patched = apply_liger_to_decoder(
                self.decoder,
                patch_rmsnorm=patch_rmsnorm,
                patch_swiglu=patch_swiglu,
                patch_rope=patch_rope,
            )
            if use_liger_ce:
                self.decoder.enable_liger_ce(True)
            logger.info(
                "liger_kernel_applied",
                encoder_modules=enc_patched,
                decoder_modules=dec_patched,
                decoder_qwen_liger=decoder_qwen_liger,
                patch_rmsnorm=patch_rmsnorm,
                patch_swiglu=patch_swiglu,
                patch_rope=patch_rope,
                use_liger_ce=use_liger_ce,
                decoder_ce_impl=self.decoder.lm_ce_impl,
            )
        elif self._decoder_only_fastpath:
            logger.info(
                "liger_kernel_skipped",
                reason="decoder_only_fastpath_freezes_bgkit",
            )

        # BaseTrainer uses self.model for logging
        self.model = self.decoder

        # --- Tokenizer ---
        tokenizer_name = decoder_cfg.backbone_name
        tokenizer_revision = decoder_cfg.get("backbone_revision", None)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            revision=tokenizer_revision,
        )
        encoder_tokenizer_name = bgkit_cfg.get("backbone_name", backbone_name)
        encoder_tokenizer_revision = bgkit_cfg.get("backbone_revision", backbone_revision)
        if encoder_tokenizer_name == tokenizer_name and (
            encoder_tokenizer_revision == tokenizer_revision
        ):
            self.encoder_tokenizer = self.tokenizer
        else:
            self.encoder_tokenizer = AutoTokenizer.from_pretrained(
                encoder_tokenizer_name,
                trust_remote_code=True,
                revision=encoder_tokenizer_revision,
            )
        logger.info(
            "tokenizers_loaded",
            decoder_tokenizer=tokenizer_name,
            encoder_tokenizer=encoder_tokenizer_name,
        )

        # Falcon-H1-Tiny-Instruct's chat template has an off-by-one
        # indexing bug at the tool-message branch (closing <|im_end|>
        # after </tool_response> is skipped when followed by an
        # assistant turn — malformed ChatML, decoder confused). Patch
        # the loaded tokenizer in-place; no-op for tokenizers that
        # don't carry the buggy pattern.
        from bgkit.data.chat_template import patch_falcon_h1_chat_template

        if patch_falcon_h1_chat_template(self.tokenizer):
            logger.info(
                "falcon_chat_template_patched",
                decoder_tokenizer=tokenizer_name,
            )

        # B3 diagnostic: render the decoder's chat template once at setup
        # so we can verify that Falcon-H1 (or any non-Qwen decoder)
        # honors the Qwen-style tool_calls structure cleanly. A mangled
        # render makes the decoder predict suffix tokens conditioned on
        # garbage prefix tokens — a candidate contributor to the
        # 179-nat starting loss in the Falcon stages. We don't fail
        # training on this; it's logged for inspection.
        try:
            from bgkit.data.chat_template import (
                BGKIT_TOOL_RESPONSE_SENTINEL,
                TOOL_CONFIGS,
                build_messages,
                build_tools,
            )

            probe_cfg = TOOL_CONFIGS["file_read_repro"]
            probe_variant = {
                "system_prompt": (
                    "You are an AI coding assistant with access to the "
                    "bgkit_read_file tool for reading file contents."
                ),
                "user_prompt": "Read the file `{file_path}`",
                "compression_prompt": "Return the file contents verbatim",
                "response_prefix": "Here are the contents of `{file_path}`:",
            }
            probe_msgs = build_messages(
                probe_variant,
                probe_cfg,
                file_path="src/example/placeholder.py",
                language="python",
                content_placeholder="X",
                tool_response_content=BGKIT_TOOL_RESPONSE_SENTINEL,
            )
            probe_tools = build_tools(probe_cfg)
            template_str = self.tokenizer.apply_chat_template(
                probe_msgs,
                tokenize=False,
                add_generation_prompt=False,
                tools=probe_tools,
            )
            template_tokens = self.tokenizer.encode(
                template_str, add_special_tokens=False,
            )
            logger.info(
                "decoder_chat_template_probe",
                decoder_family=decoder_family,
                template_token_count=len(template_tokens),
                template_chars=len(template_str),
                template_preview=template_str[:500],
                template_tail=template_str[-300:] if len(template_str) > 500 else "",
                has_bgkit_sentinel=(BGKIT_TOOL_RESPONSE_SENTINEL in template_str),
                sentinel_count=template_str.count(BGKIT_TOOL_RESPONSE_SENTINEL),
            )
        except Exception as probe_exc:
            logger.warning(
                "decoder_chat_template_probe_failed",
                error=str(probe_exc),
            )

        # --- Survivorship head config (per-level) ---
        from bgkit.training.survivorship_helpers import (
            init_state,
            load_reference_moments,
            resolve_level_ice_cfg,
            resolve_level_loss_cfg,
        )

        surv_cfg = tcfg.get("survivorship", {})
        self._surv_l0 = resolve_level_loss_cfg(surv_cfg.get("l0", {}))
        self._surv_l1 = resolve_level_loss_cfg(surv_cfg.get("l1", {}))

        ice_cfg = tcfg.get("ice_distillation", {})
        self._ice_l0 = resolve_level_ice_cfg(ice_cfg.get("l0", {}))
        self._ice_l1 = resolve_level_ice_cfg(ice_cfg.get("l1", {}))
        self._max_warmup_step = max(
            self._ice_l0.bce_warmup_steps if self._ice_l0.enabled else 0,
            self._ice_l1.bce_warmup_steps if self._ice_l1.enabled else 0,
        )
        self._ice_teacher = None
        if (
            (self._ice_l0.enabled and self._ice_l0.bce_warmup_weight > 0)
            or (self._ice_l1.enabled and self._ice_l1.bce_warmup_weight > 0)
        ):
            from bgkit.models.ice_teacher import ICETeacher
            ice_path = ice_cfg["checkpoint_path"]
            embed_tokens = self.encoder.l0.backbone.embed_tokens
            self._ice_teacher = ICETeacher(
                ice_path, embed_tokens,
                input_dim=int(ice_cfg.get("input_dim", 1024)),
                hidden_dim=int(ice_cfg.get("hidden_dim", 192)),
                num_layers=int(ice_cfg.get("num_layers", 3)),
                kernel_size=int(ice_cfg.get("kernel_size", 5)),
            ).to(device)

        mm_ref = tcfg.get("moment_match_reference", {})
        self._ref_moments_l0 = None
        self._ref_moments_l1 = None
        # OmegaConf's DictConfig is not a dict subclass; use duck-typed access.
        _l0_block = mm_ref.get("l0", None) if hasattr(mm_ref, "get") else None
        l0_path = (
            _l0_block.get("path", None)
            if _l0_block is not None and hasattr(_l0_block, "get")
            else None
        )
        if self._surv_l0.moment_match_weight > 0 and l0_path:
            self._ref_moments_l0 = load_reference_moments(l0_path)
        _l1_block = mm_ref.get("l1", None) if hasattr(mm_ref, "get") else None
        l1_path = (
            _l1_block.get("path", None)
            if _l1_block is not None and hasattr(_l1_block, "get")
            else None
        )
        if self._surv_l1.moment_match_weight > 0 and l1_path:
            self._ref_moments_l1 = load_reference_moments(l1_path)

        self._surv_state_l0 = init_state()
        self._surv_state_l1 = init_state()
        self._last_post_step_metrics: dict[str, float] = {}

        # --- Dataset ---
        # Training YAML's `data:` section lives under cfg.training.data
        # (Hydra composes training configs under the `training` group).
        data_cfg = tcfg.data
        objective_weights = tcfg.get("objectives", None)
        seed = self.cfg.get("seed", 42)
        self.compression_dataset = CompressionDataset.from_config(
            data_cfg, self.tokenizer, seed=seed,
            objective_weights=objective_weights,
            encoder_tokenizer=self.encoder_tokenizer,
        )

        # Train/eval split
        max_eval_samples = tcfg.get("max_eval_samples", 10000)
        total = len(self.compression_dataset)
        eval_size = min(max(1, int(total * 0.1)), max_eval_samples)
        train_size = total - eval_size
        if train_size < 1:
            raise ValueError(
                f"Dataset too small for train/eval split (got {total} samples, "
                "need at least 2)"
            )
        split_generator = torch.Generator().manual_seed(int(seed))
        self.train_dataset, self.eval_dataset = random_split(
            self.compression_dataset, [train_size, eval_size], generator=split_generator,
        )

        # --- Sampler + DataLoader ---
        # IMPORTANT: Samplers must emit indices scoped to the *Subset*, not the
        # full dataset.  Subset[i] maps to compression_dataset[subset.indices[i]],
        # so lengths must be gathered for the Subset's own index space.
        max_batch_tokens = tcfg.get("max_batch_tokens", 65536)
        # Eval defaults to 2x train budget (no backward -> lower peak at
        # same budget). Overridable via training.max_batch_tokens_eval.
        max_batch_tokens_eval = self._resolve_eval_batch_budget(tcfg, max_batch_tokens)
        sampler_cost_multiplier, sampler_eval_cost_multiplier = (
            self._resolve_sampler_cost_multipliers(tcfg)
        )
        num_workers = self.cfg.compute.get("num_workers", 4)
        pin_memory = self.cfg.compute.get("pin_memory", False)

        import numpy as np

        sampler_length_source = self._resolve_sampler_length_source(tcfg, decoder_family)
        sampler_budget_mode = self._resolve_sampler_budget_mode(tcfg, decoder_family)
        train_lengths = self._sampler_lengths(
            self.compression_dataset,
            self.train_dataset.indices,
            source=sampler_length_source,
        )
        eval_lengths = self._sampler_lengths(
            self.compression_dataset,
            self.eval_dataset.indices,
            source=sampler_length_source,
        )
        # Content-only lengths drive min_sample_length (encoder input);
        # `lengths` includes chat-template overhead used by sampler budget.
        train_content_lengths = np.array([
            self.compression_dataset.content_token_length(i)
            for i in self.train_dataset.indices
        ], dtype=np.int64)

        # Stash for live-tunable budget rebuild (see BaseTrainer._handle_max_batch_tokens)
        self._train_lengths = train_lengths
        self._train_content_lengths = train_content_lengths
        self._eval_lengths = eval_lengths
        self._train_collate_fn = collate_compression
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._max_batch_tokens = max_batch_tokens
        self._max_batch_tokens_eval = max_batch_tokens_eval
        self._sampler_cost_multiplier = sampler_cost_multiplier
        self._sampler_eval_cost_multiplier = sampler_eval_cost_multiplier
        self._sampler_budget_mode = sampler_budget_mode
        self._sampler_eval_budget_mode = sampler_budget_mode
        self._sampler_length_source = sampler_length_source

        self.train_sampler = PackedTokenBudgetSampler(
            self.train_dataset,
            lengths=train_lengths,
            max_batch_tokens=max_batch_tokens,
            shuffle=True,
            seed=seed,
            cost_multiplier=sampler_cost_multiplier,
            budget_mode=sampler_budget_mode,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=collate_compression,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        eval_sampler = PackedTokenBudgetSampler(
            self.eval_dataset,
            lengths=eval_lengths,
            max_batch_tokens=max_batch_tokens_eval,
            shuffle=False,
            cost_multiplier=sampler_eval_cost_multiplier,
            budget_mode=sampler_budget_mode,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_sampler=eval_sampler,
            collate_fn=collate_compression,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        if self._bgkit_frozen:
            self.encoder.requires_grad_(False)
            self.encoder.eval()
            logger.info(
                "bgkit_encoder_frozen_by_config",
                decoder_only_fastpath=self._decoder_only_fastpath,
            )
        if bool(freeze_cfg.get("decoder", False)):
            self.decoder.requires_grad_(False)
            maybe_enable_frozen_decoder_kernels(self.decoder, self.cfg)
            logger.info("decoder_frozen_by_config")
        self._decoder_only_selection_mode = str(
            tcfg.get("decoder_only_selection_mode", "exact_topk")
        )
        cache_gb = float(
            os.environ.get(
                "BGKIT_DECODER_ONLY_L0_PREFIX_CACHE_GB",
                tcfg.get("decoder_only_l0_prefix_cache_max_gb", 8.0),
            )
        )
        self._decoder_only_l0_prefix_cache_enabled = bool(
            self._decoder_only_fastpath
            and tcfg.get("decoder_only_l0_prefix_cache", True)
            and cache_gb > 0.0
        )
        cache_device = str(
            os.environ.get(
                "BGKIT_DECODER_ONLY_L0_PREFIX_CACHE_DEVICE",
                tcfg.get(
                    "decoder_only_l0_prefix_cache_device",
                    "cuda" if device.type == "cuda" else "cpu",
                ),
            )
        ).lower()
        if cache_device.startswith("cuda") and device.type != "cuda":
            cache_device = "cpu"
        if cache_device not in {"cpu", "cuda"} and not cache_device.startswith("cuda:"):
            logger.warning(
                "decoder_only_l0_prefix_cache_device_invalid",
                requested=cache_device,
                fallback="cpu",
            )
            cache_device = "cpu"
        self._decoder_only_l0_prefix_cache = (
            _DecoderOnlyL0PrefixCache(
                int(cache_gb * (1024**3)),
                device=cache_device,
            )
            if self._decoder_only_l0_prefix_cache_enabled
            else None
        )
        self._decoder_only_l0_prefix_cache_last = {
            "hits": 0,
            "misses": 0,
            "entries": 0,
            "bytes": 0,
        }
        if self._decoder_only_l0_prefix_cache_enabled:
            logger.info(
                "decoder_only_l0_prefix_cache_enabled",
                max_gb=cache_gb,
                device=cache_device,
            )

        # --- Optimizer ---
        self._setup_optimizer()

        # --- Curriculum state ---
        self._l1_introduction_step = tcfg.get("l1_introduction_step", 20000)
        # Head warmup: when > 0, the first N steps freeze both backbones
        # and train only heads + auto_repro_head + projection + decoder LoRA
        # at target_ratio=1.0 (no compression). Default 0 for Step 6, which
        # picks up a Step 5 checkpoint with already-warmed heads.
        self._head_warmup_steps = int(tcfg.get("head_warmup_steps", 0))
        self._head_warmup_active = False

        # Ratio targeting — top-level config keys
        self._target_ratio_start = tcfg.get("target_ratio_start", 0.30)
        self._target_ratio_end = tcfg.get("target_ratio_end", 0.15)
        self._target_ratio_ramp_steps = tcfg.get("target_ratio_ramp_steps", 100000)
        self._target_ratio_override: float | None = None
        anchor_grid = resolve_anchor_grid(
            self.cfg.model,
            float(self._target_ratio_start),
            getattr(self.encoder.l0.threshold, "anchor_ratios", None),
        )
        self._target_ratio_sampler_cfg = build_ratio_sampler_config(
            {
                "enabled": tcfg.get("sample_target_ratio_during_training", False),
                "mode": tcfg.get("target_ratio_sampling_mode", "window"),
                "window_above": tcfg.get("target_ratio_sampling_window_above", 0.10),
                "anchor_sampling_prob": tcfg.get(
                    "target_ratio_anchor_sampling_prob", 0.30,
                ),
                "jitter_abs": tcfg.get("target_ratio_jitter_abs", 0.0),
                "jitter_rel": tcfg.get("target_ratio_jitter_rel", 0.0),
            },
            anchor_grid=anchor_grid,
            default_ratio=float(self._target_ratio_start),
            enabled_default=False,
            mode_default="window",
        )
        import random

        self._target_ratio_rng = random.Random(int(self.cfg.get("seed", 42)))
        self._last_sampled_target_ratio: float | None = None

        # Live curriculum values
        self._l1_enabled = False
        self._l1_transitioned = False
        self._l1_rebuild_pending = False

        # Eval isolation flag
        self._is_evaluating = False

        # --- Metric tracking ---
        self._eval_count = 0

        # --- Profiling ---
        self._profile_enabled = os.environ.get("BGKIT_PROFILE", "") == "1"

        # --- Diagnostic metrics cadence ---
        self._diagnostic_metrics_every_n_steps = int(
            tcfg.get("diagnostic_metrics_every_n_steps", 10),
        )

        # Head-tanh temperature calibration for both levels. See
        # CommitEncodingTrainer._calibrate_head_tanh_temperatures for the
        # rationale; the probe is cheap and confirms/corrects the loaded
        # checkpoint's T values against the current head outputs.
        self._calibrate_head_tanh_temperatures()

        logger.info(
            "compression_trainer_setup",
            train_samples=train_size,
            eval_samples=eval_size,
            device=str(device),
            l1_introduction_step=self._l1_introduction_step,
            profile=self._profile_enabled,
        )

    def _calibrate_head_tanh_temperatures(self, n_probe_batches: int = 4) -> None:
        """Probe L0 + L1 head output std at startup and set per-level
        ``head_tanh_temperature_l{0,1}`` buffers."""
        if getattr(self, "_decoder_only_fastpath", False):
            logger.info("head_tanh_temperature_calibration_skipped", reason="bgkit_frozen")
            return
        from bgkit.training.survivorship_helpers import (
            calibrate_head_tanh_temperature,
        )
        for level in ("l0", "l1"):
            calibrated_t = calibrate_head_tanh_temperature(
                self.encoder,
                self.train_dataloader,
                self.device,
                level=level,
                n_probe_batches=n_probe_batches,
            )
            if calibrated_t is not None:
                logger.info(
                    "head_tanh_temperature_calibrated",
                    level=level,
                    T=calibrated_t,
                )

    def _resolve_step1_checkpoint(self) -> str | None:
        """Resolve step1_checkpoint.

        The candidate phase chain depends on the *decoder family* under
        training, not on the literal phase name: Qwen runs walk the
        Qwen-family Phase 1 ladder (step5 best) while Falcon runs walk
        their Falcon-family ladder so the Falcon projection block is not
        accidentally reinitialized from Qwen-only Phase 1 state. The
        per-stage entry point is still keyed on ``training.phase`` so a
        stage that runs *during* (e.g. phase1_falcon_l1) doesn't try to
        load itself.
        """
        step1_checkpoint = self.cfg.get("step1_checkpoint", None)

        # Initialize lineage tracking
        self._input_sources = {}

        if step1_checkpoint == "auto":
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            # Read decoder family from training override first, then top-level
            # model.decoder, then default to qwen35. Tolerant of test fixtures
            # that don't construct the full config tree.
            decoder_cfg = (
                self.cfg.training.get("model", {}).get("decoder", None)
                or (self.cfg.get("model", {}) or {}).get("decoder", None)
                or {}
            )
            decoder_family = normalize_decoder_family(
                decoder_cfg.get("family", "qwen35"),
            )
            training_phase = self.cfg.get("training", {}).get("phase", "")
            # Phase-name fallback: legacy configs (and unit tests) set the
            # phase to a Falcon stage without setting model.decoder.family.
            # Treat that as an implicit "falcon_h1" family.
            if (
                decoder_family == "qwen35"
                and isinstance(training_phase, str)
                and training_phase.startswith("phase1_falcon_")
            ):
                decoder_family = "falcon_h1"
            if decoder_family == "falcon_h1":
                # Source-checkpoint preference (latest stage first that's
                # not the current one). 2026-05-14 update: re-prefer
                # forced_adapt over dense_seed for l0_align. After the
                # B1/B3 fixes (asymmetric BCE, chat-template patch, eval-
                # leak fix) and the reweighted-loss rerun (cos_sim 0.52→
                # 0.58 trained), forced_adapt's projection+encoder state
                # is a strictly better warm-start than dense_seed for
                # Phase C end-to-end. dense_seed remains a last-resort
                # fallback if no forced_adapt checkpoint is registered.
                if training_phase == "phase1_falcon_l0_align":
                    candidate_phases = (
                        "phase1_falcon_forced_adapt",
                        "phase1_falcon_dense_seed",
                    )
                elif training_phase == "phase1_falcon_l0":
                    candidate_phases = (
                        "phase1_falcon_l0_align",
                        "phase1_falcon_forced_adapt",
                        "phase1_falcon_dense_seed",
                    )
                else:  # phase1_falcon_l1 and any future Falcon stage
                    candidate_phases = (
                        "phase1_falcon_l0",
                        "phase1_falcon_l0_align",
                        "phase1_falcon_forced_adapt",
                        "phase1_falcon_dense_seed",
                    )
            else:
                candidate_phases = ("phase1_step5",)

            errors: list[str] = []
            # For Falcon family, use eval/cos_sim (higher-is-better) to avoid
            # the sign-flip trap: forced_adapt's eval/loss switched magnitude
            # under the cos-weight reweighting (0.25 pre, 0.84 post), so
            # ranking by eval/loss picks the worse pre-reweighting checkpoint.
            # eval/cos_sim has the same direction-of-better in both regimes.
            falcon_metric = decoder_family == "falcon_h1"
            metric_key = "eval/cos_sim" if falcon_metric else "eval/loss"
            lower_better = not falcon_metric
            for phase in candidate_phases:
                try:
                    resolved = resolve_checkpoint(
                        checkpoint_dir,
                        phase=phase,
                        metric=metric_key,
                        lower_is_better=lower_better,
                        label="step1_checkpoint",
                    )
                    step1_checkpoint = str(resolved)
                    break
                except ValueError as exc:
                    errors.append(str(exc))
            else:
                raise ValueError(
                    f"step1_checkpoint=auto could not resolve any candidate "
                    f"phase for decoder_family={decoder_family!r} "
                    f"(phase={training_phase!r}): {', '.join(candidate_phases)}. "
                    + " | ".join(errors)
                )

        if step1_checkpoint is not None:
            self._input_sources["step1"] = Path(step1_checkpoint).name

        return step1_checkpoint

    # ------------------------------------------------------------------
    # Live-config handlers for survivorship knobs
    # ------------------------------------------------------------------
    # Update the mutable LevelLossCfg dataclasses populated at setup.
    # ``_surv_l0`` / ``_surv_l1`` are read fresh by
    # ``_compute_survivorship_losses`` each batch via ``getattr(self, ...)``,
    # so a field-level mutation is picked up on the next step without any
    # rewiring.

    def _update_surv_field(self, level: str, field_name: str, val) -> None:
        """Shared body for survivorship-weight live handlers.

        ``level`` is "l0" or "l1"; mutates the corresponding
        ``self._surv_<level>`` LevelLossCfg field. Coerces ``val`` to
        float with non-negative validation; rejects with a warning
        otherwise.
        """
        from bgkit.training.survivorship_helpers import LevelLossCfg

        attr_name = f"_surv_{level}"
        cfg_obj = getattr(self, attr_name, None)
        if cfg_obj is None:
            cfg_obj = LevelLossCfg()
            setattr(self, attr_name, cfg_obj)
        if not isinstance(val, (int, float)) or float(val) < 0:
            logger.warning(
                "live_surv_field_invalid",
                level=level,
                field=field_name,
                value=val,
                expected="non-negative float",
            )
            return
        old = getattr(cfg_obj, field_name)
        new_val = float(val)
        setattr(cfg_obj, field_name, new_val)
        logger.info(
            "live_surv_field_update",
            level=level,
            field=field_name,
            old=old,
            new=new_val,
        )

    def _handle_surv_l0_forced_bce_weight(self, val) -> None:
        self._update_surv_field("l0", "forced_survivor_bce_weight", val)

    def _handle_surv_l1_forced_bce_weight(self, val) -> None:
        self._update_surv_field("l1", "forced_survivor_bce_weight", val)

    def _handle_surv_l0_forced_bce_anchor(self, val) -> None:
        self._update_surv_field("l0", "forced_survivor_bce_anchor_ratio", val)

    def _handle_surv_l1_forced_bce_anchor(self, val) -> None:
        self._update_surv_field("l1", "forced_survivor_bce_anchor_ratio", val)

    def _handle_surv_l0_utility_grad_weight(self, val) -> None:
        self._update_surv_field("l0", "utility_grad_loss_weight", val)

    def _handle_surv_l1_utility_grad_weight(self, val) -> None:
        self._update_surv_field("l1", "utility_grad_loss_weight", val)

    def _handle_surv_l0_decisiveness_weight(self, val) -> None:
        self._update_surv_field("l0", "decisiveness_loss_weight", val)

    def _handle_surv_l1_decisiveness_weight(self, val) -> None:
        self._update_surv_field("l1", "decisiveness_loss_weight", val)

    def _handle_surv_l0_moment_match_weight(self, val) -> None:
        self._update_surv_field("l0", "moment_match_weight", val)

    def _handle_surv_l1_moment_match_weight(self, val) -> None:
        self._update_surv_field("l1", "moment_match_weight", val)

    def _handle_surv_l0_min_survivors_weight(self, val) -> None:
        self._update_surv_field("l0", "min_survivors_loss_weight", val)

    def _handle_surv_l1_min_survivors_weight(self, val) -> None:
        self._update_surv_field("l1", "min_survivors_loss_weight", val)

    def _handle_use_forced_survivor_mask_l0(self, val) -> None:
        """Toggle the forced-mask override mid-run.

        Switches between Phase C (encoder uses companion forced_mask as
        the survivor selector) and Phase D (head selects). Mutates the
        OmegaConf node in ``self.cfg.training`` because
        ``_compress_file_batch`` reads the value fresh each step. The
        DualThresholdController will rediscover its theta under the new
        regime via dual-ascent over the next several steps.
        """
        bool_val = bool(val)
        try:
            old = bool(self.cfg.training.get("use_forced_survivor_mask_l0", False))
            self.cfg.training.use_forced_survivor_mask_l0 = bool_val
        except Exception as exc:
            logger.warning(
                "live_use_forced_survivor_mask_l0_failed",
                value=val,
                error=repr(exc),
            )
            return
        logger.info(
            "live_use_forced_survivor_mask_l0_update",
            old=old,
            new=bool_val,
        )

    def _muon_excluded_param_ids(self) -> frozenset[int]:
        """Return param IDs of decoder embedding/lm_head — 2D but should not use Muon.

        After LoRA wrapping, embed_tokens and lm_head are frozen (only LoRA
        adapters are trainable), so the exclusion set only matters for the
        non-LoRA case.

        Qwen CausalLM structure: backbone.model.embed_tokens, backbone.lm_head.
        PeftModel wraps backbone: backbone.base_model.model → original CausalLM.
        """
        exclude = set()
        # Unwrap PeftModel if present to reach the original CausalLM
        backbone = self.decoder.backbone
        try:
            from peft import PeftModel
            is_peft = isinstance(backbone, PeftModel)
        except ImportError:
            is_peft = False
        causal_lm = backbone.base_model.model if is_peft else backbone
        # embed_tokens lives on causal_lm.model (the inner Qwen model)
        inner = getattr(causal_lm, "model", None)
        if inner is not None and hasattr(inner, "embed_tokens"):
            for p in inner.embed_tokens.parameters():
                if p.requires_grad:
                    exclude.add(id(p))
        # lm_head lives on causal_lm (the CausalLM wrapper)
        lm_head = getattr(causal_lm, "lm_head", None)
        if lm_head is not None:
            for p in lm_head.parameters():
                if p.requires_grad:
                    exclude.add(id(p))
        # LoRA factors must use AdamW (not Muon) — see
        # ``BaseTrainer._lora_param_ids`` docstring.
        exclude |= set(self._lora_param_ids(self.decoder))
        return frozenset(exclude)

    def _setup_optimizer(self) -> None:
        """Create optimizer with encoder + decoder param groups."""
        tcfg = self.cfg.training

        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        decoder_params = [p for p in self.decoder.parameters() if p.requires_grad]

        encoder_lr = tcfg.get("encoder_lr", tcfg.lr)
        decoder_lr = tcfg.get("decoder_lr", tcfg.lr)

        param_groups = []
        if encoder_params:
            param_groups.append({
                "params": encoder_params,
                "lr": encoder_lr,
                "base_lr": encoder_lr,
            })
        if decoder_params:
            param_groups.append({
                "params": decoder_params,
                "lr": decoder_lr,
                "base_lr": decoder_lr,
            })

        self.optimizer = self._create_optimizer(
            param_groups, tcfg.lr, exclude_from_muon=self._muon_excluded_param_ids()
        )

    def _trainable_params(self) -> list[torch.nn.Parameter]:
        """Collect all trainable parameters for gradient clipping."""
        params = [p for p in self.encoder.parameters() if p.requires_grad]
        params += [p for p in self.decoder.parameters() if p.requires_grad]
        return params

    # ------------------------------------------------------------------
    def _get_bidi_alpha(self) -> float:
        """Get the current bidirectional warmup alpha from the encoder."""
        from bgkit.models.bidirectional_qwen35 import BidirectionalQwen35
        from bgkit.models.pruned_qwen35 import PrunedBidirectionalQwen35

        for module in self.encoder.modules():
            if isinstance(module, (BidirectionalQwen35, PrunedBidirectionalQwen35)):
                return module.bidi_alpha
        return 1.0

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _check_l1_introduction(self) -> None:
        """Check if L1 should be introduced based on curriculum step."""
        if self.global_step >= self._l1_introduction_step and not self._l1_enabled:
            self._l1_enabled = True
            self._l1_rebuild_pending = True
            logger.info("l1_curriculum_triggered", step=self.global_step)

    def _maybe_apply_head_warmup_freeze(self) -> None:
        """Freeze backbones for head warmup; unfreeze when warmup ends."""
        if self._head_warmup_steps <= 0:
            return
        in_warmup = self.global_step < self._head_warmup_steps
        if in_warmup and not self._head_warmup_active:
            self.encoder.l0.backbone.requires_grad_(False)
            self.encoder.l1.backbone.requires_grad_(False)
            for p in self.encoder.l0.head.parameters():
                p.requires_grad_(True)
            for p in self.encoder.l1.head.parameters():
                p.requires_grad_(True)
            if self.encoder.l0.auto_repro_head is not None:
                for p in self.encoder.l0.auto_repro_head.parameters():
                    p.requires_grad_(True)
            for p in self.encoder.projection_block.parameters():
                p.requires_grad_(True)
            self._head_warmup_active = True
            logger.info("head_warmup_started", until_step=self._head_warmup_steps)
        elif not in_warmup and self._head_warmup_active:
            self.encoder.l0.backbone.requires_grad_(True)
            self.encoder.l1.backbone.requires_grad_(True)
            self._head_warmup_active = False
            logger.info("head_warmup_ended", step=self.global_step)

    def _current_target_ratio(self) -> float:
        """Compute the current target compression ratio from the ramp or override."""
        if (
            self._head_warmup_steps > 0
            and self.global_step < self._head_warmup_steps
        ):
            return 1.0
        if self._target_ratio_override is not None:
            return self._target_ratio_override
        step = self.global_step
        if step >= self._target_ratio_ramp_steps:
            return self._target_ratio_end
        t = step / max(self._target_ratio_ramp_steps, 1)
        return self._target_ratio_start + t * (self._target_ratio_end - self._target_ratio_start)

    def _sample_target_ratio(self) -> float:
        """Sample a requested ratio for the current microbatch."""
        if not hasattr(self, "_target_ratio_sampler_cfg"):
            tcfg = self.cfg.training
            model_cfg = getattr(self.cfg, "model", {})
            anchor_grid = resolve_anchor_grid(
                model_cfg,
                float(self._target_ratio_start),
                getattr(self.encoder.l0.threshold, "anchor_ratios", None),
            )
            self._target_ratio_sampler_cfg = build_ratio_sampler_config(
                {
                    "enabled": tcfg.get("sample_target_ratio_during_training", False),
                    "mode": tcfg.get("target_ratio_sampling_mode", "window"),
                    "window_above": tcfg.get("target_ratio_sampling_window_above", 0.10),
                    "anchor_sampling_prob": tcfg.get(
                        "target_ratio_anchor_sampling_prob", 0.30,
                    ),
                    "jitter_abs": tcfg.get("target_ratio_jitter_abs", 0.0),
                    "jitter_rel": tcfg.get("target_ratio_jitter_rel", 0.0),
                },
                anchor_grid=anchor_grid,
                default_ratio=float(self._target_ratio_start),
                enabled_default=False,
                mode_default="window",
            )
        if not hasattr(self, "_target_ratio_rng"):
            import random

            self._target_ratio_rng = random.Random(int(self.cfg.get("seed", 42)))
        if not hasattr(self, "_last_sampled_target_ratio"):
            self._last_sampled_target_ratio = None
        return sample_ratio(
            rng=self._target_ratio_rng,
            config=self._target_ratio_sampler_cfg,
            base_ratio=self._current_target_ratio(),
            is_evaluating=self._is_evaluating,
            override_active=self._target_ratio_override is not None,
        )

    @staticmethod
    def _resolve_sampler_cost_multipliers(tcfg) -> tuple[float, float]:
        sampler_cfg = tcfg.get("sampler", {}) or {}
        train_multiplier = sampler_cfg.get(
            "cost_multiplier",
            tcfg.get("sampler_cost_multiplier", 1.0),
        )
        eval_multiplier = sampler_cfg.get(
            "eval_cost_multiplier",
            tcfg.get("sampler_eval_cost_multiplier", train_multiplier),
        )
        return float(train_multiplier), float(eval_multiplier)

    @staticmethod
    def _resolve_sampler_budget_mode(tcfg, decoder_family: str) -> str:
        sampler_cfg = tcfg.get("sampler", {}) or {}
        explicit = sampler_cfg.get("budget_mode", tcfg.get("sampler_budget_mode", None))
        mode = str(explicit) if explicit is not None else "packed_quadratic"
        if mode not in ("packed_quadratic", "padded_quadratic"):
            raise ValueError(
                "training.sampler.budget_mode must be 'packed_quadratic' or "
                f"'padded_quadratic', got {mode!r}"
            )
        return mode

    @staticmethod
    def _resolve_sampler_length_source(tcfg, decoder_family: str) -> str:
        sampler_cfg = tcfg.get("sampler", {}) or {}
        explicit = sampler_cfg.get("length_source", tcfg.get("sampler_length_source", None))
        source = str(explicit) if explicit is not None else "token"
        if source not in ("token", "decoder"):
            raise ValueError(
                "training.sampler.length_source must be 'token' or 'decoder', "
                f"got {source!r}"
            )
        return source

    @staticmethod
    def _sampler_lengths(dataset, indices, *, source: str) -> np.ndarray:
        if source == "decoder" and hasattr(dataset, "decoder_token_length"):
            return np.array([
                dataset.decoder_token_length(i)
                for i in indices
            ], dtype=np.int64)
        return np.array([
            dataset.token_length(i)
            for i in indices
        ], dtype=np.int64)

    def _perform_l1_rebuild(self) -> None:
        """Rebuild dataset and dataloader for L1 phase."""
        import numpy as np

        logger.info("performing_l1_rebuild", step=self.global_step)

        # Enable L1 on the dataset
        self.compression_dataset.rebuild_for_l1()

        sampler_length_source = getattr(self, "_sampler_length_source", "token")
        sampler_budget_mode = getattr(self, "_sampler_budget_mode", "packed_quadratic")
        # Recompute subset-scoped lengths and rebuild sampler
        train_lengths = self._sampler_lengths(
            self.compression_dataset,
            self.train_dataset.indices,
            source=sampler_length_source,
        )
        # Refresh content lengths too so the min_sample_length filter
        # (lazily snapshotted in _rebuild_train_dataloader_with_budget)
        # picks up L1's new sample shapes on the next rebuild.
        train_content_lengths = np.array([
            self.compression_dataset.content_token_length(i)
            for i in self.train_dataset.indices
        ], dtype=np.int64)
        max_batch_tokens = self.cfg.training.get("max_batch_tokens", 65536)
        max_batch_tokens_eval = self._resolve_eval_batch_budget(
            self.cfg.training, max_batch_tokens,
        )
        sampler_cost_multiplier, sampler_eval_cost_multiplier = (
            self._resolve_sampler_cost_multipliers(self.cfg.training)
        )
        seed = self.cfg.get("seed", 42)
        # Update stashes so any subsequent live rebuild sees fresh L1 lengths.
        # Drop the cached "_full" snapshots so the next live filter rebuild
        # re-snapshots from the L1-updated lengths/dataset.
        self._train_lengths = train_lengths
        self._train_content_lengths = train_content_lengths
        self._max_batch_tokens = max_batch_tokens
        self._max_batch_tokens_eval = max_batch_tokens_eval
        self._sampler_cost_multiplier = sampler_cost_multiplier
        self._sampler_eval_cost_multiplier = sampler_eval_cost_multiplier
        self._sampler_budget_mode = sampler_budget_mode
        self._sampler_eval_budget_mode = sampler_budget_mode
        for cached in ("_train_dataset_full", "_train_lengths_full", "_train_content_lengths_full"):
            if hasattr(self, cached):
                delattr(self, cached)
        self.train_sampler = PackedTokenBudgetSampler(
            self.train_dataset,
            lengths=train_lengths,
            max_batch_tokens=max_batch_tokens,
            shuffle=True,
            seed=seed,
            cost_multiplier=sampler_cost_multiplier,
            budget_mode=sampler_budget_mode,
        )

        # Rebuild dataloader
        num_workers = self.cfg.compute.get("num_workers", 4)
        pin_memory = self.cfg.compute.get("pin_memory", False)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=collate_compression,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Rebuild eval sampler/dataloader with updated L1 lengths
        eval_lengths = self._sampler_lengths(
            self.compression_dataset,
            self.eval_dataset.indices,
            source=sampler_length_source,
        )
        self._eval_lengths = eval_lengths
        eval_sampler = PackedTokenBudgetSampler(
            self.eval_dataset,
            lengths=eval_lengths,
            max_batch_tokens=max_batch_tokens_eval,
            shuffle=False,
            cost_multiplier=sampler_eval_cost_multiplier,
            budget_mode=sampler_budget_mode,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_sampler=eval_sampler,
            collate_fn=collate_compression,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self._l1_rebuild_pending = False
        self._l1_transitioned = True
        logger.info("l1_rebuild_complete", step=self.global_step)

    # ------------------------------------------------------------------
    # Survivorship losses
    # ------------------------------------------------------------------

    def _compute_survivorship_losses(
        self,
        enc_out,
        target_ratio: float,
        level: str = "l0",
        content_token_ids: torch.Tensor | None = None,
        content_cu_seqlens: torch.Tensor | None = None,
        forced_survivor_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Delegate to the shared helpers. Accumulate microbatch state."""
        from bgkit.training.survivorship_helpers import (
            LevelICECfg,
            LevelLossCfg,
            accumulate,
            compute_survivorship_losses,
            init_state,
            survivorship_diagnostics,
        )

        if level == "l0":
            weights = getattr(self, "_surv_l0", LevelLossCfg())
            ice_cfg = getattr(self, "_ice_l0", LevelICECfg())
            ref_moments = getattr(self, "_ref_moments_l0", None)
            state = getattr(self, "_surv_state_l0", None)
            if state is None:
                state = init_state()
                self._surv_state_l0 = state
        else:
            weights = getattr(self, "_surv_l1", LevelLossCfg())
            ice_cfg = getattr(self, "_ice_l1", LevelICECfg())
            ref_moments = getattr(self, "_ref_moments_l1", None)
            state = getattr(self, "_surv_state_l1", None)
            if state is None:
                state = init_state()
                self._surv_state_l1 = state

        if getattr(self, "_decoder_only_fastpath", False):
            accumulate(state, enc_out, target_ratio=target_ratio)
            device = getattr(
                getattr(enc_out, "survivor_embeddings", None),
                "device",
                self.device,
            )
            zero = torch.tensor(0.0, device=device)
            return zero, survivorship_diagnostics(
                enc_out, level=level, global_step=self.global_step,
                every_n_steps=int(
                    getattr(self, "_diagnostic_metrics_every_n_steps", 1) or 1
                ),
            )

        loss, metrics = compute_survivorship_losses(
            enc_out=enc_out,
            level=level,
            weights=weights,
            ice_cfg=ice_cfg,
            ref_moments=ref_moments,
            ice_teacher=getattr(self, "_ice_teacher", None),
            global_step=self.global_step,
            content_token_ids=content_token_ids,
            content_cu_seqlens=content_cu_seqlens,
            target_ratio=target_ratio,
            forced_survivor_mask=forced_survivor_mask,
        )
        accumulate(state, enc_out, target_ratio=target_ratio)
        out_metrics = {f"{level}_{k}": v for k, v in metrics.items()}
        diag_every_n = int(getattr(self, "_diagnostic_metrics_every_n_steps", 1) or 1)
        out_metrics.update(
            survivorship_diagnostics(
                enc_out, level=level, global_step=self.global_step,
                every_n_steps=diag_every_n,
            )
        )
        return loss, out_metrics

    # ------------------------------------------------------------------
    # Compression pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def _packed_cu_from_lengths(lengths: list[int]) -> torch.Tensor:
        cu = torch.zeros(len(lengths) + 1, dtype=torch.int32)
        if lengths:
            cu[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
        return cu

    @staticmethod
    def _l0_prefix_cache_key(
        content_ids: torch.Tensor,
        prompt_ids: torch.Tensor | None,
    ) -> str:
        h = hashlib.blake2b(digest_size=20)
        content_cpu = content_ids.detach().to("cpu").contiguous()
        h.update(b"content:")
        h.update(str(tuple(content_cpu.shape)).encode("ascii"))
        h.update(content_cpu.numpy().tobytes())
        if prompt_ids is None:
            h.update(b":prompt:none")
        else:
            prompt_cpu = prompt_ids.detach().to("cpu").contiguous()
            h.update(b":prompt:")
            h.update(str(tuple(prompt_cpu.shape)).encode("ascii"))
            h.update(prompt_cpu.numpy().tobytes())
        return h.hexdigest()

    def _compress_file_batch_decoder_only_cached(
        self,
        batch: dict,
        target_ratio: float,
    ) -> EncoderOutput:
        from bgkit.training.survivorship_helpers import LevelLossCfg
        from bgkit.utils.packing import lengths_from_cu, position_ids_from_cu

        cache = self._decoder_only_l0_prefix_cache
        if cache is None:
            raise RuntimeError("decoder-only L0 prefix cache is not enabled")

        device = self.device
        content_ids_cpu = batch["content_token_ids"]
        content_cu_cpu = batch["content_cu_seqlens"]
        prompt_ids_cpu = batch["compression_prompt_ids"]
        prompt_cu_cpu = batch["prompt_cu_seqlens"]
        content_starts = content_cu_cpu.to(torch.int64).tolist()
        prompt_starts = prompt_cu_cpu.to(torch.int64).tolist()
        batch_size = len(content_starts) - 1
        has_prompt = int(prompt_ids_cpu.numel()) > 0

        entries: list[dict | None] = [None] * batch_size
        keys: list[str] = [""] * batch_size
        miss_indices: list[int] = []
        hits = 0
        for i in range(batch_size):
            c0, c1 = content_starts[i], content_starts[i + 1]
            p0, p1 = prompt_starts[i], prompt_starts[i + 1]
            prompt_slice = prompt_ids_cpu[p0:p1] if has_prompt else None
            key = self._l0_prefix_cache_key(content_ids_cpu[c0:c1], prompt_slice)
            keys[i] = key
            entry = cache.get(key)
            if entry is None:
                miss_indices.append(i)
            else:
                hits += 1
                entries[i] = entry

        if miss_indices:
            # The original decoder-only cache was batch-atomic: any miss forced
            # the whole microbatch through the frozen encoder. Mixed hit/miss
            # batches are common with shuffled training, so only encode the
            # missing samples, insert their frozen prefix activations, then
            # assemble the full batch from cache below.
            if hits == 0:
                raise _DecoderOnlyL0PrefixCacheMissError(
                    hits=hits,
                    misses=len(miss_indices),
                )
            miss_batch = self._slice_batch(batch, miss_indices)
            miss_enc_out = self._compress_file_batch_uncached(
                miss_batch,
                target_ratio=target_ratio,
            )
            miss_enc_out.release()
            still_missing = 0
            for i in miss_indices:
                entry = cache.get(keys[i])
                if entry is None:
                    still_missing += 1
                else:
                    entries[i] = entry
            if still_missing:
                raise _DecoderOnlyL0PrefixCacheMissError(
                    hits=hits,
                    misses=still_missing,
                )

        hidden_blocks: list[torch.Tensor] = []
        base_raw_blocks: list[torch.Tensor] = []
        combined_lengths: list[int] = []
        content_mask_blocks: list[torch.Tensor] = []
        pos_blocks: list[torch.Tensor] = []
        for entry in entries:
            if entry is None:
                raise RuntimeError("internal L0 prefix cache assembly miss")
            hidden = entry["hidden"].to(device=device, non_blocking=True)
            base_raw = entry["base_raw"].to(device=device, non_blocking=True)
            prompt_len = int(entry["prompt_len"])
            content_len = int(entry["content_len"])
            entry_has_prompt = bool(entry["has_prompt"])
            total_len = int(hidden.shape[0])
            content_start = prompt_len + 1 if entry_has_prompt else 0
            mask = torch.zeros(total_len, dtype=torch.bool, device=device)
            mask[content_start : content_start + content_len] = True
            hidden_blocks.append(hidden)
            base_raw_blocks.append(base_raw)
            combined_lengths.append(total_len)
            content_mask_blocks.append(mask)
            pos_blocks.append(torch.arange(total_len, dtype=torch.int64, device=device))

        prefix_hidden = torch.cat(hidden_blocks, dim=0)
        base_raw = torch.cat(base_raw_blocks, dim=0)
        combined_cu = self._packed_cu_from_lengths(combined_lengths).to(device)
        combined_pos = torch.cat(pos_blocks, dim=0)
        content_pos_mask = torch.cat(content_mask_blocks, dim=0)
        content_cu = content_cu_cpu.to(device)

        surv_l0 = getattr(self, "_surv_l0", LevelLossCfg())
        l0_out = self.encoder.l0.decoder_only_from_prefix(
            prefix_hidden=prefix_hidden,
            combined_cu_seqlens=combined_cu,
            combined_position_ids=combined_pos,
            content_pos_mask=content_pos_mask,
            content_cu_seqlens=content_cu,
            base_raw=base_raw,
            target_ratio=target_ratio,
            min_per_sample=int(surv_l0.min_survivors_absolute_min),
            selection_mode=self._decoder_only_selection_mode,
        )

        proj_input = l0_out.survivor_embeddings
        proj_cu = l0_out.survivor_cu_seqlens
        proj_lengths = lengths_from_cu(proj_cu).to(torch.int64)
        proj_max = int(proj_lengths.max().item()) if proj_lengths.numel() else 0
        proj_pos = position_ids_from_cu(proj_cu, int(proj_input.shape[0]))
        proj_out = self.encoder.get_active_projection_block()(
            proj_input,
            cu_seqlens=proj_cu,
            max_seqlen=proj_max,
            position_ids=proj_pos,
            survivor_mask=None,
        )
        out_cu = effective_projection_cu(proj_out, proj_cu)
        counts = effective_projection_counts(proj_out, proj_cu)

        misses = len(miss_indices)
        self._decoder_only_l0_prefix_cache_last = {
            "hits": hits,
            "misses": misses,
            "entries": len(cache),
            "bytes": cache.current_bytes,
        }
        return EncoderOutput(
            survivor_embeddings=proj_out.projected_embeddings,
            survivor_cu_seqlens=out_cu,
            survivor_counts=counts,
            l0=l0_out,
            l1=None,
        )

    def _store_decoder_only_l0_prefix_cache(
        self,
        batch: dict,
        l0_out,
    ) -> None:
        cache = self._decoder_only_l0_prefix_cache
        prefix = getattr(l0_out, "_decoder_only_prefix_cache", None)
        if cache is None or not prefix:
            return
        hidden = prefix.get("hidden")
        base_raw = prefix.get("base_raw")
        combined_cu = prefix.get("combined_cu_seqlens")
        if hidden is None or base_raw is None or combined_cu is None:
            return

        content_ids_cpu = batch["content_token_ids"]
        content_cu_cpu = batch["content_cu_seqlens"]
        prompt_ids_cpu = batch["compression_prompt_ids"]
        prompt_cu_cpu = batch["prompt_cu_seqlens"]
        content_starts = content_cu_cpu.to(torch.int64).tolist()
        prompt_starts = prompt_cu_cpu.to(torch.int64).tolist()
        combined_starts = combined_cu.to(torch.int64).tolist()
        has_prompt = int(prompt_ids_cpu.numel()) > 0
        hits = 0
        misses = 0
        for i in range(len(content_starts) - 1):
            c0, c1 = content_starts[i], content_starts[i + 1]
            p0, p1 = prompt_starts[i], prompt_starts[i + 1]
            prompt_slice = prompt_ids_cpu[p0:p1] if has_prompt else None
            key = self._l0_prefix_cache_key(content_ids_cpu[c0:c1], prompt_slice)
            if cache.get(key) is not None:
                hits += 1
                continue
            h0, h1 = combined_starts[i], combined_starts[i + 1]
            entry = {
                "hidden": hidden[h0:h1].detach().to(cache.device).contiguous(),
                "base_raw": base_raw[c0:c1].detach().to(cache.device).contiguous(),
                "prompt_len": p1 - p0 if has_prompt else 0,
                "content_len": c1 - c0,
                "has_prompt": has_prompt,
            }
            cache.put(key, entry)
            misses += 1
        self._decoder_only_l0_prefix_cache_last = {
            "hits": hits,
            "misses": misses,
            "entries": len(cache),
            "bytes": cache.current_bytes,
        }

    def _compress_file_batch_uncached(
        self, batch: dict, target_ratio: float,
    ):
        """Run L0 compression on a packed FileCompressionSample batch.

        The collator gives us flat content embeddings with per-sample
        ``content_cu_seqlens`` and aligned per-sample prompts.

        When ``training.use_forced_survivor_mask_l0`` is True AND the
        batch carries a ``forced_survivor_mask_l0`` (Falcon companion
        loaded), the mask is passed to the encoder so survivors = forced
        positions instead of the head's selection. This is the Phase C
        regime: filter the bogus 2-Falcon-token expansions at positions
        where Qwen→Falcon alignment is heuristic, so the decoder only
        sees the well-aligned subset while everything else trains
        end-to-end. The head still fires for diagnostic + BCE supervision
        because forced_mask triggers ``compression_off=False`` in
        LevelCompressor regardless of target_ratio.
        """
        from bgkit.training.survivorship_helpers import LevelLossCfg
        from bgkit.utils.packing import position_ids_from_cu

        device = self.device
        content_ids = batch["content_token_ids"].to(device)
        content_cu = batch["content_cu_seqlens"].to(device)
        content_position_ids = batch["content_position_ids"].to(device)
        prompt_ids = batch["compression_prompt_ids"].to(device)
        prompt_cu = batch["prompt_cu_seqlens"].to(device)
        prompt_position_ids = position_ids_from_cu(prompt_cu, int(prompt_ids.shape[0]))

        bgkit_embed = self.encoder.l0.backbone.get_input_embeddings()
        content_emb = bgkit_embed(content_ids)
        prompt_emb = bgkit_embed(prompt_ids)

        surv_l0 = getattr(self, "_surv_l0", LevelLossCfg())
        surv_l1 = getattr(self, "_surv_l1", LevelLossCfg())
        util_active = (
            surv_l0.utility_grad_loss_weight > 0.0
            and not getattr(self, "_decoder_only_fastpath", False)
        )

        forced_mask_l0: torch.Tensor | None = None
        if bool(self.cfg.training.get("use_forced_survivor_mask_l0", False)):
            raw_forced = batch.get("forced_survivor_mask_l0")
            if raw_forced is not None:
                forced_mask_l0 = raw_forced.to(device=device, dtype=torch.bool)

        enc_out = self.encoder(
            content_embeddings=content_emb,
            content_cu_seqlens=content_cu,
            content_position_ids=content_position_ids,
            prompt_embeddings=prompt_emb,
            prompt_cu_seqlens=prompt_cu,
            prompt_position_ids=prompt_position_ids,
            target_ratio_l0=target_ratio,
            utility_grad_active_l0=util_active,
            min_per_sample_l0=int(surv_l0.min_survivors_absolute_min),
            min_per_sample_l1=int(surv_l1.min_survivors_absolute_min),
            forced_survivor_mask_l0=forced_mask_l0,
            selection_mode_l0=(
                self._decoder_only_selection_mode
                if getattr(self, "_decoder_only_fastpath", False)
                else "threshold"
            ),
            capture_decoder_only_prefix_l0=(
                getattr(self, "_decoder_only_l0_prefix_cache_enabled", False)
                and getattr(self, "_decoder_only_fastpath", False)
            ),
        )
        if getattr(self, "_decoder_only_l0_prefix_cache_enabled", False):
            self._store_decoder_only_l0_prefix_cache(batch, enc_out.l0)
        return enc_out

    def _compress_file_batch(
        self, batch: dict, target_ratio: float,
    ):
        if (
            getattr(self, "_decoder_only_l0_prefix_cache_enabled", False)
            and not bool(self.cfg.training.get("use_forced_survivor_mask_l0", False))
        ):
            try:
                return self._compress_file_batch_decoder_only_cached(
                    batch,
                    target_ratio=target_ratio,
                )
            except _DecoderOnlyL0PrefixCacheMissError as miss:
                self._decoder_only_l0_prefix_cache_last = {
                    "hits": miss.hits,
                    "misses": miss.misses,
                    "entries": len(self._decoder_only_l0_prefix_cache),
                    "bytes": self._decoder_only_l0_prefix_cache.current_bytes,
                }

        return self._compress_file_batch_uncached(batch, target_ratio=target_ratio)

    def _compress_repo_l0_packed(self, batch: dict, target_ratio: float):
        """Single packed L0 forward across every file in every repo.

        Uses ``cu_file_seqlens`` (one segment per file) for encoder
        attention and the per-file-tiled ``prompt_token_ids`` from the repo
        collator. Survivors emerge flat; ``survivor_cu_seqlens`` is
        aligned 1:1 with ``cu_file_seqlens`` (one survivor group per file).
        """
        from bgkit.training.survivorship_helpers import LevelLossCfg
        from bgkit.utils.packing import position_ids_from_cu

        device = self.device
        file_ids = batch["content_token_ids"].to(device)
        cu_file = batch["cu_file_seqlens"].to(device)
        content_position_ids = batch["content_position_ids"].to(device)
        prompt_ids = batch["prompt_token_ids"].to(device)
        prompt_cu = batch["prompt_cu_seqlens"].to(device)
        prompt_position_ids = position_ids_from_cu(prompt_cu, int(prompt_ids.shape[0]))

        bgkit_embed = self.encoder.l0.backbone.get_input_embeddings()
        content_emb = bgkit_embed(file_ids)
        prompt_emb = bgkit_embed(prompt_ids)

        surv_l0 = getattr(self, "_surv_l0", LevelLossCfg())
        util_active = (
            surv_l0.utility_grad_loss_weight > 0.0
            and not getattr(self, "_decoder_only_fastpath", False)
        )
        return self.encoder.l0(
            content_embeddings=content_emb,
            content_cu_seqlens=cu_file,
            content_position_ids=content_position_ids,
            prompt_embeddings=prompt_emb,
            prompt_cu_seqlens=prompt_cu,
            prompt_position_ids=prompt_position_ids,
            target_ratio=target_ratio,
            utility_grad_active=util_active,
            min_per_sample=int(surv_l0.min_survivors_absolute_min),
            selection_mode=(
                self._decoder_only_selection_mode
                if getattr(self, "_decoder_only_fastpath", False)
                else "threshold"
            ),
        )

    @staticmethod
    def _regroup_survivors_per_repo(
        l0_survivors: torch.Tensor,
        l0_survivor_cu: torch.Tensor,
        cu_repo_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert per-file survivor groups into per-repo survivor groups.

        ``l0_survivor_cu`` has shape ``(total_files + 1,)`` — one
        cumulative boundary per file. ``cu_repo_seqlens`` has shape
        ``(B + 1,)`` and holds indices *into* the file-axis: each entry is
        a file count, so the survivors belonging to repo ``b`` are the
        flat range ``[l0_survivor_cu[cu_repo_seqlens[b]],
        l0_survivor_cu[cu_repo_seqlens[b+1]])``.

        Since the encoder already emits survivors in file-order, no
        reshuffling of ``l0_survivors`` is needed — we only need to build
        a new ``(B+1,)`` cu_seqlens whose boundaries land at each repo's
        file-range end.

        Returns ``(survivors, survivor_cu_repo)`` with ``survivors ===
        l0_survivors`` (same tensor) and ``survivor_cu_repo`` shape
        ``(B+1,)`` int32.
        """
        cu_repo = cu_repo_seqlens.to(torch.int64)
        cu_file = l0_survivor_cu.to(torch.int64)
        # Gather: for each repo boundary r_idx, take the corresponding file
        # boundary from cu_file. That's our new cu_seqlens in flat-survivor space.
        survivor_cu_repo = cu_file[cu_repo]
        return l0_survivors, survivor_cu_repo.to(torch.int32)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _decoder_forward_single_splice(
        self,
        survivors: torch.Tensor,
        survivor_cu_seqlens: torch.Tensor,
        batch: dict,
        sample_indices: list[int] | None = None,
    ) -> torch.Tensor:
        """Run fused decoder forward + CE loss (packed).

        Builds per-sample prefix/suffix from the flat ``target_token_ids``
        using each sample's ``bgkit_splice_start`` / ``bgkit_splice_len``.
        ``sample_indices`` optionally selects a subset of samples from the
        batch (e.g. for mixed file+repo batches).
        """
        device = self.device
        target_ids_flat = batch["target_token_ids"]
        target_cu = batch["target_cu_seqlens"]
        splice_start = batch["bgkit_splice_start"]
        splice_len = batch["bgkit_splice_len"]
        loss_mask_flat = batch.get("target_loss_mask")
        if loss_mask_flat is not None:
            loss_mask_flat = loss_mask_flat.to(device=device, dtype=torch.bool)

        packed_splice_plan = None
        if sample_indices is None and batch.get("_bgkit_cuda_graph_probe_batch", False):
            plan_key = "_bgkit_decoder_packed_splice_plan"
            packed_splice_plan = batch.get(plan_key)
            if packed_splice_plan is None:
                if getattr(self, "_suppress_forward_backward_metrics", False):
                    raise RuntimeError(
                        "static decoder splice plan was not warmed before CUDA graph capture"
                    )
                packed_splice_plan = self.decoder.build_packed_target_splice_plan(
                    survivor_cu_seqlens=survivor_cu_seqlens,
                    target_cu_seqlens=target_cu,
                    splice_start=splice_start,
                    splice_len=splice_len,
                    loss_mask_flat=loss_mask_flat,
                    sample_indices=None,
                )
                batch[plan_key] = packed_splice_plan

        return self.decoder.forward_with_packed_target_splice(
            survivor_embeddings=survivors,
            survivor_cu_seqlens=survivor_cu_seqlens,
            target_ids_flat=target_ids_flat,
            target_cu_seqlens=target_cu,
            splice_start=splice_start,
            splice_len=splice_len,
            loss_mask_flat=loss_mask_flat,
            sample_indices=sample_indices,
            packed_splice_plan=packed_splice_plan,
        )

    @staticmethod
    def _materialize_decoder_input(tensor: torch.Tensor) -> torch.Tensor:
        """Convert inference-mode tensors before feeding trainable decoder weights."""
        is_inference = getattr(tensor, "is_inference", None)
        if callable(is_inference) and is_inference():
            return tensor.clone()
        return tensor

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> list:
        return self._trainable_params()

    def _forward_backward(self, batch: dict) -> dict[str, float]:
        """Forward pass + scaled backward (packed). No optimizer ops.

        Both file and repo batches use a single packed L0 forward. Repo
        batches additionally run a single packed L1 forward across
        per-repo survivor groups.
        """
        if getattr(self, "_decoder_only_fastpath", False):
            self.encoder.eval()
        else:
            self.encoder.train()
        self.decoder.train()

        # Check L1 introduction
        self._check_l1_introduction()

        # Handle mixed batches
        if batch.get("mixed", False):
            return self._forward_backward_mixed(batch)

        sample_type = batch["sample_type"]
        target_ratio = self._sample_target_ratio()
        self._last_sampled_target_ratio = target_ratio

        if sample_type == "file":
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda",
            ):
                if getattr(self, "_decoder_only_fastpath", False):
                    with torch.inference_mode():
                        enc_out = self._compress_file_batch(
                            batch, target_ratio=target_ratio,
                        )
                    enc_out.survivor_embeddings = self._materialize_decoder_input(
                        enc_out.survivor_embeddings,
                    )
                else:
                    enc_out = self._compress_file_batch(batch, target_ratio=target_ratio)
                survivors = enc_out.survivor_embeddings
                survivor_cu = enc_out.survivor_cu_seqlens
                loss = self._decoder_forward_single_splice(survivors, survivor_cu, batch)

            total_loss_t = loss

            # Survivorship auxiliary losses (L0 — file compression path)
            surv_metrics: dict[str, float] = {}
            if enc_out.l0.logits_for_op is not None:
                # Forced-survivor mask from the Falcon companion (when
                # present); flows to ``compute_survivorship_losses`` as
                # head-BCE supervision under ``forced_survivor_bce_weight``.
                # ``_collate_file_samples`` emits this key as ``None`` if
                # any sample in the batch lacks the mask, in which case the
                # forced-BCE branch in the helper stays inert.
                forced_mask_l0 = batch.get("forced_survivor_mask_l0")
                if forced_mask_l0 is not None:
                    forced_mask_l0 = forced_mask_l0.to(self.device)
                surv_loss, surv_metrics = self._compute_survivorship_losses(
                    enc_out.l0, target_ratio,
                    level="l0",
                    content_token_ids=batch["content_token_ids"].to(self.device),
                    content_cu_seqlens=batch["content_cu_seqlens"].to(self.device),
                    forced_survivor_mask=forced_mask_l0,
                )
                total_loss_t = total_loss_t + surv_loss

            (total_loss_t / self._accum_steps).backward()

            # Utility-gradient BCE distillation (post-backward).
            from bgkit.training.survivorship_helpers import LevelLossCfg
            _surv_l0 = getattr(self, "_surv_l0", LevelLossCfg())
            if (
                _surv_l0.utility_grad_loss_weight > 0.0
                and not getattr(self, "_decoder_only_fastpath", False)
                and enc_out.l0.post_head_content_values is not None
            ):
                from bgkit.training.survivorship_helpers import utility_grad_bce_loss

                util_loss, util_metrics = utility_grad_bce_loss(
                    base_raw_for_util=enc_out.l0.base_raw_for_util,
                    content_grad=enc_out.l0.get_content_grad(),
                    content_values=enc_out.l0.post_head_content_values,
                    valid_mask=None,
                    pinned_mask=None,
                    target_ratio=target_ratio,
                    content_cu_seqlens=batch["content_cu_seqlens"].to(self.device),
                )
                if util_loss.requires_grad:
                    w = _surv_l0.utility_grad_loss_weight
                    (util_loss * w / self._accum_steps).backward()
                for k, v in util_metrics.items():
                    surv_metrics[f"l0_{k}"] = v

            if getattr(self, "_suppress_forward_backward_metrics", False):
                enc_out.release()
                return {
                    "loss": 0.0,
                    "sample_type": sample_type,
                    "min_target_ratio": float(target_ratio),
                    "sampled_target_ratio": float(target_ratio),
                    "actual_ratio": 0.0,
                    "l1_enabled": float(self._l1_enabled),
                }

            n_survivors = (
                int(enc_out.l0.survivor_mask.sum().item())
                if enc_out.l0.survivor_mask is not None else 0
            )
            n_valid = int(batch["content_token_ids"].shape[0])
            total_loss = loss.item()

            # Drop tensor refs on the file-path enc_out.
            enc_out.release()
        else:
            # Repo batches: packed L0 across all files, then packed L1
            # per repo. See ``_forward_backward_repo_packed``.
            total_loss, n_survivors, n_valid, repo_surv_metrics = (
                self._forward_backward_repo_packed(batch, target_ratio=target_ratio)
            )
            surv_metrics = repo_surv_metrics

        min_target_ratio = self._current_target_ratio()
        actual_ratio = n_survivors / max(n_valid, 1)
        metrics = {
            "loss": total_loss,
            "sample_type": sample_type,
            "min_target_ratio": min_target_ratio,
            "sampled_target_ratio": target_ratio,
            "actual_ratio": actual_ratio,
            "l1_enabled": float(self._l1_enabled),
        }
        if getattr(self, "_decoder_only_l0_prefix_cache_enabled", False):
            cache_stats = getattr(self, "_decoder_only_l0_prefix_cache_last", {})
            hits = int(cache_stats.get("hits", 0))
            misses = int(cache_stats.get("misses", 0))
            denom = max(hits + misses, 1)
            metrics.update({
                "decoder_only_l0_prefix_cache_hit_rate": hits / denom,
                "decoder_only_l0_prefix_cache_entries": float(
                    cache_stats.get("entries", 0)
                ),
                "decoder_only_l0_prefix_cache_gb": float(
                    cache_stats.get("bytes", 0)
                ) / float(1024**3),
            })
        metrics.update(surv_metrics)
        return metrics

    def _forward_backward_repo_packed(
        self,
        batch: dict,
        target_ratio: float,
        scale_override: float | None = None,
    ) -> tuple[float, int, int, dict[str, float]]:
        """Packed repo-batch forward + backward (no per-sample loop).

        Algorithm:
          1. ONE packed L0 forward across every file in every repo in the
             microbatch. ``cu_file_seqlens`` gives per-file segmentation;
             the encoder's packed attention keeps files from attending
             across their boundaries. Prompt is already per-file-tiled by
             the collator.
          2. Regroup the flat L0 survivors into per-repo groups using
             ``cu_repo_seqlens`` (indices into ``cu_file_seqlens``). This
             is a pure index-shuffle over ``survivor_cu_seqlens`` — the
             survivor tensor itself is unchanged.
          3. ONE packed L1 forward across the whole microbatch (one
             segment per repo). A synthetic prompt CU of all-zero lengths
             keeps the L1 encoder's prompt path inert (L1 reuses the L0
             survivors as its input; the prompt conditioning already
             flowed through L0).
          4. ONE packed decoder call across all repos with their
             per-repo L1 survivors spliced into the packed target
             sequence.

        Returns ``(avg_loss, total_survivors, total_valid_tokens,
        surv_metrics)``.
        """
        from bgkit.training.survivorship_helpers import LevelLossCfg
        from bgkit.utils.packing import position_ids_from_cu

        device = self.device
        cu_repo = batch["cu_repo_seqlens"].to(device)
        batch_size = int(cu_repo.shape[0]) - 1
        scale = (
            scale_override
            if scale_override is not None
            else 1.0 / (batch_size * self._accum_steps)
        )

        surv_metrics: dict[str, float] = {}

        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda",
        ):
            with torch.set_grad_enabled(
                not getattr(self, "_decoder_only_fastpath", False)
            ):
                # --- Step 1: packed L0 across all files in all repos ---
                l0_out = self._compress_repo_l0_packed(batch, target_ratio=target_ratio)
                l0_survivors = l0_out.survivor_embeddings  # (N_surv_l0_total, D)
                l0_survivor_cu = l0_out.survivor_cu_seqlens  # (total_files + 1,)

                # Collect L0 survivorship aux loss (per-file content segmentation).
                l0_surv_loss = None
                if l0_out.logits_for_op is not None:
                    loss_v, l0_metrics = self._compute_survivorship_losses(
                        l0_out, target_ratio,
                        level="l0",
                        content_token_ids=batch["content_token_ids"].to(device),
                        content_cu_seqlens=batch["cu_file_seqlens"].to(device),
                    )
                    l0_surv_loss = loss_v
                    surv_metrics.update(l0_metrics)

                # --- Step 2: regroup flat L0 survivors into per-repo groups ---
                l1_input_flat, l1_input_cu = self._regroup_survivors_per_repo(
                    l0_survivors, l0_survivor_cu, cu_repo,
                )

                # Bridge L0 final hidden states into L1's input-embedding space
                # via L0's auto_repro_head. L1's backbone was deepcopied from
                # L0 and expects input-embedding-distributed inputs.
                l1_input_bridged = self.encoder.l0.auto_reproduce(l1_input_flat)
                n_surv_total = int(l1_input_bridged.shape[0])
                l1_input_positions = position_ids_from_cu(l1_input_cu, n_surv_total)

                surv_l1 = getattr(self, "_surv_l1", LevelLossCfg())
                util_w_l1 = (
                    0.0 if getattr(self, "_decoder_only_fastpath", False)
                    else surv_l1.utility_grad_loss_weight
                )

                # --- Step 3: packed L1 forward across all repos ---
                l1_out = self.encoder.l1(
                    content_embeddings=l1_input_bridged,
                    content_cu_seqlens=l1_input_cu,
                    content_position_ids=l1_input_positions,
                    target_ratio=target_ratio,
                    utility_grad_active=util_w_l1 > 0.0,
                    min_per_sample=int(surv_l1.min_survivors_absolute_min),
                    selection_mode=(
                        self._decoder_only_selection_mode
                        if getattr(self, "_decoder_only_fastpath", False)
                        else "threshold"
                    ),
                )

                l1_surv_loss = None
                if l1_out.logits_for_op is not None:
                    loss_v, l1_metrics = self._compute_survivorship_losses(
                        l1_out, target_ratio,
                        level="l1",
                        content_token_ids=None,
                        content_cu_seqlens=l1_input_cu,
                    )
                    l1_surv_loss = loss_v
                    surv_metrics.update(l1_metrics)

                # --- Step 3.5: project L1 survivors through the shared
                # projection_block before handing off to the decoder.
                from bgkit.utils.packing import lengths_from_cu
                proj_cu = l1_out.survivor_cu_seqlens
                proj_lengths = lengths_from_cu(proj_cu).to(torch.int64)
                proj_max = int(proj_lengths.max().item()) if proj_lengths.numel() else 0
                proj_pos = position_ids_from_cu(
                    proj_cu, int(l1_out.survivor_embeddings.shape[0])
                )
                proj_out = self.encoder.projection_block(
                    l1_out.survivor_embeddings,
                    cu_seqlens=proj_cu,
                    max_seqlen=proj_max,
                    position_ids=proj_pos,
                    survivor_mask=None,
                )
                projected_cu = effective_projection_cu(proj_out, proj_cu)

            # --- Step 4: packed decoder across all repos ---
            loss = self._decoder_forward_single_splice(
                proj_out.projected_embeddings,
                projected_cu,
                batch,
            )

            total_loss_t = loss
            if l0_surv_loss is not None and l0_surv_loss.requires_grad:
                total_loss_t = total_loss_t + l0_surv_loss
            if l1_surv_loss is not None and l1_surv_loss.requires_grad:
                total_loss_t = total_loss_t + l1_surv_loss

        (total_loss_t * batch_size * scale).backward()

        total_loss = loss.item() * batch_size
        total_survivors = int(l1_out.survivor_embeddings.shape[0])
        total_valid = int(batch["content_token_ids"].shape[0])

        # Utility-gradient BCE — L0 first, then L1.
        _surv_l0 = getattr(self, "_surv_l0", LevelLossCfg())
        if (
            _surv_l0.utility_grad_loss_weight > 0.0
            and not getattr(self, "_decoder_only_fastpath", False)
            and l0_out.post_head_content_values is not None
        ):
            from bgkit.training.survivorship_helpers import utility_grad_bce_loss

            util_loss, _ = utility_grad_bce_loss(
                base_raw_for_util=l0_out.base_raw_for_util,
                content_grad=l0_out.get_content_grad(),
                content_values=l0_out.post_head_content_values,
                valid_mask=None,
                pinned_mask=None,
                target_ratio=target_ratio,
                content_cu_seqlens=batch["cu_file_seqlens"].to(device),
            )
            if util_loss.requires_grad:
                (util_loss * _surv_l0.utility_grad_loss_weight
                 * batch_size * scale).backward()

        if (
            util_w_l1 > 0.0
            and not getattr(self, "_decoder_only_fastpath", False)
            and l1_out.post_head_content_values is not None
        ):
            from bgkit.training.survivorship_helpers import utility_grad_bce_loss

            util_loss, _ = utility_grad_bce_loss(
                base_raw_for_util=l1_out.base_raw_for_util,
                content_grad=l1_out.get_content_grad(),
                content_values=l1_out.post_head_content_values,
                valid_mask=None,
                pinned_mask=None,
                target_ratio=target_ratio,
                content_cu_seqlens=l1_input_cu,
            )
            if util_loss.requires_grad:
                (util_loss * util_w_l1 * batch_size * scale).backward()

        l0_out.release()
        l1_out.release()

        return total_loss / batch_size, total_survivors, total_valid, surv_metrics

    def _forward_backward_mixed(self, batch: dict) -> dict[str, float]:
        """Handle mixed file+repo batch with sample-count-weighted scaling.

        Both portions use the packed path. Each portion is scaled by its
        sample count relative to the combined total so file and repo
        samples contribute roughly equally to the accumulated gradient.
        """
        file_batch = batch["file_batch"]
        repo_batch = batch["repo_batch"]

        n_file_samples = int(file_batch["content_cu_seqlens"].shape[0]) - 1
        n_repo_samples = int(repo_batch["cu_repo_seqlens"].shape[0]) - 1
        total_samples = n_file_samples + n_repo_samples
        target_ratio = self._sample_target_ratio()
        self._last_sampled_target_ratio = target_ratio

        # --- File portion (packed forward + backward) ---
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda",
        ):
            with torch.set_grad_enabled(
                not getattr(self, "_decoder_only_fastpath", False)
            ):
                file_enc_out = self._compress_file_batch(
                    file_batch, target_ratio=target_ratio,
                )
            file_loss = self._decoder_forward_single_splice(
                file_enc_out.survivor_embeddings,
                file_enc_out.survivor_cu_seqlens,
                file_batch,
            )
        file_scale = n_file_samples / (total_samples * self._accum_steps)
        (file_loss * file_scale).backward()

        # --- Repo portion (packed) ---
        repo_scale = 1.0 / (total_samples * self._accum_steps)
        avg_repo_loss, repo_survivors, n_valid_repo, _ = (
            self._forward_backward_repo_packed(
                repo_batch,
                target_ratio=target_ratio,
                scale_override=repo_scale,
            )
        )

        combined_loss = (
            file_loss.item() * n_file_samples
            + avg_repo_loss * n_repo_samples
        ) / total_samples
        file_surv_count = (
            int(file_enc_out.l0.survivor_mask.sum().item())
            if file_enc_out.l0.survivor_mask is not None else 0
        )
        total_survivors = file_surv_count + repo_survivors
        n_valid_file = int(file_batch["content_token_ids"].shape[0])
        actual_ratio = total_survivors / max(n_valid_file + n_valid_repo, 1)

        metrics = {
            "loss": combined_loss,
            "sample_type": "mixed",
            "min_target_ratio": self._current_target_ratio(),
            "sampled_target_ratio": target_ratio,
            "actual_ratio": actual_ratio,
            "l1_enabled": float(self._l1_enabled),
        }
        file_enc_out.release()
        return metrics

    # ------------------------------------------------------------------
    # Custom train() with pre-fetch L1 rebuild hook
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # BaseTrainer hooks
    # ------------------------------------------------------------------

    _log_every = 100
    _use_device_prefetcher = False  # save memory on unified-memory GPU

    def _pre_train_loop(self) -> None:
        """Rebuild dataloader if resuming from a checkpoint with L1 already active."""
        if self._l1_transitioned or self._l1_rebuild_pending:
            self._perform_l1_rebuild()
        self._maybe_apply_head_warmup_freeze()

    def _pre_step_hook(self) -> None:
        """Rebuild dataloader when L1 curriculum transition is pending."""
        if self._l1_rebuild_pending:
            self._perform_l1_rebuild()
            self._dataloader_invalidated = True
        self._maybe_apply_head_warmup_freeze()

    def _post_optimizer_step(self, step: int) -> None:
        """Advance bidi warmup; run dual-ascent θ + EMA μ updates per level."""
        self.encoder.step_bidi_warmup()

        from bgkit.training.survivorship_helpers import (
            apply_post_step_updates,
            init_state,
            maybe_unload_ice,
        )

        merged: dict[str, float] = {}
        for level, state_attr in (("l0", "_surv_state_l0"), ("l1", "_surv_state_l1")):
            state = getattr(self, state_attr, None)
            if state is None:
                continue
            update_metrics = apply_post_step_updates(
            self.encoder, state,
                target_ratio=None, level=level,
            )
            merged.update(update_metrics)
            setattr(self, state_attr, init_state())
        self._last_post_step_metrics = merged

        unloaded = maybe_unload_ice(
            getattr(self, "_ice_teacher", None),
            step,
            getattr(self, "_max_warmup_step", 0),
        )
        if unloaded:
            logger.info("ice_teacher_unloaded", step=step)

    def _add_step_metrics(self, metrics: dict[str, float]) -> None:
        """Add compression-specific metrics (without clobbering base keys)."""
        metrics["bidi_alpha"] = self._get_bidi_alpha()
        if self._last_sampled_target_ratio is not None:
            metrics.setdefault("sampled_target_ratio", self._last_sampled_target_ratio)
        post = getattr(self, "_last_post_step_metrics", None)
        if post:
            for k, v in post.items():
                metrics.setdefault(k, v)

    def _build_training_state(
        self,
        es_best: float | None,
        es_evals_without_improvement: int,
        wandb_run,
    ) -> dict:
        """Build training state dict with curriculum fields."""
        state = super()._build_training_state(
            es_best, es_evals_without_improvement, wandb_run,
        )
        state.update({
            "l1_enabled": self._l1_enabled,
            "l1_transitioned": self._l1_transitioned,
            "l1_rebuild_pending": self._l1_rebuild_pending,
            "target_ratio_override": self._target_ratio_override,
        })
        return state

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation on the eval split.

        File batches stay batched. Repo batches use per-sample forward for
        memory efficiency. Token-weighted loss accumulation preserved for
        comparable perplexity across runs.
        """
        encoder_was_training = self.encoder.training
        decoder_was_training = self.decoder.training
        self.encoder.eval()
        decoder_family = getattr(self.decoder, "decoder_family", "qwen35")
        # Falcon-H1 Mamba has two completely different code paths gated
        # by ``self.training``: the fused training kernel
        # ``mamba_split_conv1d_scan_combined`` vs the unfused inference
        # path (softplus + causal_conv1d + selective_scan with
        # ``apply_mask_to_padding_states`` applied to ``projected_states``).
        # The two paths produce numerically different forward outputs
        # (verified via investigation 2026-05-15) — and the inference
        # path exists upstream to support autoregressive single-token
        # generation with a KV-style cache, which we never use. For
        # full-sequence teacher-forced LM CE eval the fused training
        # kernel is the correct and consistent choice. Keep the decoder
        # in training mode during evaluate(); ``@torch.no_grad()``
        # disables gradient computation regardless of model.training, so
        # this is safe.
        if decoder_family == "falcon_h1":
            self.decoder.train()
        else:
            self.decoder.eval()
        self._is_evaluating = True
        self._eval_count += 1

        try:
            total_loss = 0.0
            total_tokens = 0.0
            per_objective_loss_sum: dict[str, float] = {}
            per_objective_tokens: dict[str, float] = {}
            mixed_batches = 0
            # First-class always-on eval accumulators (reset per call).
            # Populated by _evaluate_file_batch via _record_eval_sub_batch_stats
            # and evaluate_ablation_hook.
            self._eval_sample_stats = {
                "n_samples": 0,
                "content_total": 0,
                "target_total": 0,
                "lm_total": 0,
                "survivor_total": 0,
                # list of (loss_value, sub_tokens) tuples, one per sub-batch
                "loss_points": [],
            }
            self._eval_ablation_state: dict[str, dict[str, float]] = {}

            # Periodic unsliced-vs-sliced check. Defaults to every 10th
            # eval; configurable via training.eval_unsliced_every_n_evals.
            unsliced_every = int(
                self.cfg.training.get("eval_unsliced_every_n_evals", 10),
            )
            # _eval_count was incremented above. Fire on the first eval
            # (eval_count == 1) and every Nth thereafter so the signal
            # appears early in training, not only after N evals.
            self._run_unsliced_this_eval = (
                unsliced_every > 0
                and (
                    self._eval_count == 1
                    or self._eval_count % unsliced_every == 0
                )
            )
            self._eval_unsliced_consumed = False
            self._eval_unsliced_loss_sum = 0.0
            self._eval_unsliced_tokens = 0.0

            num_batches = len(self.eval_dataloader)
            for batch_idx, batch in enumerate(self.eval_dataloader):
                if batch_idx % 100 == 0:
                    logger.info("eval_progress", batch=batch_idx, total=num_batches)

                if batch.get("mixed", False):
                    mixed_batches += 1
                    batch_parts = [batch["file_batch"], batch["repo_batch"]]
                else:
                    batch_parts = [batch]

                for part in batch_parts:
                    objectives = part.get("objectives", [])
                    if objectives:
                        grouped_indices: dict[str, list[int]] = {}
                        for idx, obj in enumerate(objectives):
                            grouped_indices.setdefault(obj, []).append(idx)
                    else:
                        grouped_indices = {"unknown": list(range(self._batch_size(part)))}

                    for obj, indices in grouped_indices.items():
                        obj_batch = (
                            part if len(indices) == self._batch_size(part)
                            else self._slice_batch(part, indices)
                        )
                        if obj_batch["sample_type"] == "file":
                            loss_sum, token_count = self._evaluate_file_batch(obj_batch)
                        else:
                            loss_sum, token_count = self._evaluate_repo_batch_persample(obj_batch)

                        total_loss += loss_sum
                        total_tokens += token_count
                        per_objective_loss_sum[obj] = (
                            per_objective_loss_sum.get(obj, 0.0) + loss_sum
                        )
                        per_objective_tokens[obj] = (
                            per_objective_tokens.get(obj, 0.0) + token_count
                        )

            avg_loss = total_loss / max(total_tokens, 1)
            perplexity = torch.exp(torch.tensor(avg_loss)).item()

            metrics: dict[str, float] = {
                "loss": avg_loss,
                "perplexity": perplexity,
                "mixed_batches": float(mixed_batches),
            }

            for obj, loss_sum in per_objective_loss_sum.items():
                obj_tokens = per_objective_tokens.get(obj, 0.0)
                metrics[f"{obj}_loss"] = loss_sum / max(obj_tokens, 1.0)
                metrics[f"{obj}_tokens"] = obj_tokens
                metrics[f"{obj}_token_fraction"] = obj_tokens / max(total_tokens, 1.0)

            # ----- Always-on per-sample distribution stats -----
            # Cheap reductions, emitted every eval into the ``eval/``
            # namespace so wandb plots them alongside ``eval/loss``.
            stats = self._eval_sample_stats
            n_samples = max(1, int(stats["n_samples"]))
            metrics["n_samples"] = float(stats["n_samples"])
            metrics["avg_content_len"] = stats["content_total"] / n_samples
            metrics["avg_target_len"] = stats["target_total"] / n_samples
            metrics["avg_lm_tokens"] = stats["lm_total"] / n_samples
            metrics["avg_survivors"] = stats["survivor_total"] / n_samples
            loss_points = stats["loss_points"]
            if loss_points:
                losses_t = torch.tensor(
                    [lp[0] for lp in loss_points], dtype=torch.float64,
                )
                weights_t = torch.tensor(
                    [lp[1] for lp in loss_points], dtype=torch.float64,
                )
                metrics["per_sample_loss_min"] = float(losses_t.min().item())
                metrics["per_sample_loss_max"] = float(losses_t.max().item())
                # Unweighted mean of per-sub-batch losses (denominator =
                # number of sub-batches, not tokens). For the canonical
                # token-weighted mean see ``eval/loss``.
                metrics["per_sample_loss_mean"] = float(losses_t.mean().item())
                if losses_t.numel() >= 2:
                    metrics["per_sample_loss_std"] = float(
                        losses_t.std(unbiased=False).item(),
                    )
                else:
                    metrics["per_sample_loss_std"] = 0.0
                # Token-weighted std as a sanity check: weight each
                # sub-batch by its token count.
                tw = float(weights_t.sum().item())
                if tw > 0 and losses_t.numel() >= 2:
                    weighted_mean = float(
                        (losses_t * weights_t).sum().item() / tw,
                    )
                    weighted_var = float(
                        ((losses_t - weighted_mean) ** 2 * weights_t).sum().item()
                        / tw,
                    )
                    metrics["per_sample_loss_std_token_weighted"] = (
                        weighted_var ** 0.5
                    )

            # ----- Ablation metrics from accumulator -----
            # Default suffix from CompressionTrainer is
            # ``perfect_projection``; emit existing metric names for
            # wandb continuity.
            for suffix, accum in self._eval_ablation_state.items():
                ab_tokens = float(accum["tokens"])
                if ab_tokens <= 0:
                    continue
                ab_avg = float(accum["loss_sum"]) / ab_tokens
                metrics[f"ablation_{suffix}_loss"] = ab_avg
                metrics[f"ablation_{suffix}_tokens"] = ab_tokens
                metrics[f"{suffix}_quality_gap"] = avg_loss - ab_avg

            # ----- Periodic unsliced-vs-sliced agreement (every Nth eval) -----
            if (
                self._run_unsliced_this_eval
                and self._eval_unsliced_tokens > 0
            ):
                metrics["unsliced_loss"] = (
                    self._eval_unsliced_loss_sum / self._eval_unsliced_tokens
                )
                metrics["unsliced_tokens"] = self._eval_unsliced_tokens

            # ----- Always-on train-subset comparison -----
            # One mini-batch forward (~8 samples) — trivial cost. If
            # ``eval/train_subset_loss`` ever drifts away from
            # ``eval/loss``, train and eval pipelines have diverged
            # again (the 2026-05-14 bug class).
            train_ts = self._compute_train_subset_loss()
            if train_ts is not None:
                ts_loss, ts_tokens = train_ts
                metrics["train_subset_loss"] = ts_loss
                metrics["train_subset_tokens"] = ts_tokens
                metrics["train_eval_gap"] = avg_loss - ts_loss
        finally:
            self._is_evaluating = False
            self.encoder.train(encoder_was_training)
            self.decoder.train(decoder_was_training)

        return metrics

    def _compute_train_subset_loss(
        self, n_samples: int = 8,
    ) -> tuple[float, float] | None:
        """Run ``n_samples`` train chunks through the eval forward path.

        Always-on diagnostic: if this drifts from ``eval/loss``, the
        train and eval pipelines have diverged (different collator
        path, missing field in slicing, different ratio handling, etc.).
        Returns ``(avg_loss, token_count)`` or ``None`` on failure /
        when train_dataset isn't available. Cost: one ~8-sample forward
        per eval — trivial.
        """
        try:
            import random as _random

            from bgkit.data.collators import collate_compression

            train_ds = getattr(self, "train_dataset", None)
            if train_ds is None or len(train_ds) == 0:
                return None
            n_train = len(train_ds)
            # Seed deterministically per-eval so the same chunks
            # aren't sampled every time (would over-fit the signal to
            # a fixed subset) but the eval is reproducible at a step.
            rng = _random.Random(42 + int(self._eval_count))
            pick = rng.sample(range(n_train), min(n_samples, n_train))
            synth_samples = [train_ds[i] for i in pick]
            synth_batch = collate_compression(synth_samples)
            for k, v in synth_batch.items():
                if torch.is_tensor(v):
                    synth_batch[k] = v.to(self.device)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                synth_enc = self._compress_file_batch(
                    synth_batch, target_ratio=self._current_target_ratio(),
                )
                synth_loss = self._decoder_forward_single_splice(
                    synth_enc.survivor_embeddings,
                    synth_enc.survivor_cu_seqlens,
                    synth_batch,
                )
            synth_lm = synth_batch.get("target_loss_mask")
            synth_tokens = (
                int(synth_lm.sum().item()) if synth_lm is not None
                else int(synth_batch["target_token_ids"].shape[0])
            )
            del synth_enc
            return float(synth_loss.item()), float(synth_tokens)
        except Exception as exc:
            logger.warning("eval_train_subset_failed", error=str(exc))
            return None

    def _batch_size(self, batch: dict) -> int:
        """Return the number of samples represented by a collated batch."""
        objectives = batch.get("objectives", [])
        if objectives:
            return len(objectives)
        if batch["sample_type"] == "file":
            return int(batch["content_cu_seqlens"].shape[0]) - 1
        return int(batch["cu_repo_seqlens"].shape[0]) - 1

    def _slice_batch(self, batch: dict, indices: list[int]) -> dict:
        """Slice a packed collated batch down to a subset of sample indices.

        Packed tensors (flat token buffers) must be re-packed from the
        per-sample segments; non-packed per-sample fields (e.g.
        ``bgkit_splice_start``, ``compression_ratios``) are plain indexed.

        Postcondition (asserted): every key present in ``batch`` is also
        present in the returned ``sliced`` dict. This guards against the
        2026-05-15 bug where new fields added to the collator (e.g.
        ``forced_survivor_mask_l0``, ``target_falcon_pair_ids_per_survivor``)
        were silently dropped during eval slicing because they were
        missed in _slice_batch's hand-maintained allow-list. The dropped
        fields turned eval into a different setup than train, hiding
        for weeks. The assert raises so future regressions surface
        immediately.
        """
        from bgkit.data.collators import _make_cu_seqlens
        from bgkit.utils.packing import position_ids_from_cu

        sliced: dict = {}

        if batch.get("sample_type") == "file":
            # Rebuild packed content/target/prefix/prompt from sample ranges.
            def _rebuild(flat_key: str, cu_key: str) -> tuple[torch.Tensor, torch.Tensor]:
                flat = batch[flat_key]
                cu = batch[cu_key].to(torch.int64)
                starts = cu[:-1]
                ends = cu[1:]
                parts = [flat[int(starts[i]):int(ends[i])] for i in indices]
                new_flat = torch.cat(parts, dim=0) if parts else flat.new_zeros(0)
                new_lengths = [int(p.shape[0]) for p in parts]
                device = cu.device if cu.is_cuda else "cpu"
                return new_flat, _make_cu_seqlens(new_lengths).to(device)

            c_flat, c_cu = _rebuild("content_token_ids", "content_cu_seqlens")
            t_flat, t_cu = _rebuild("target_token_ids", "target_cu_seqlens")
            l_flat, _ = _rebuild("target_loss_mask", "target_cu_seqlens")
            p_flat, p_cu = _rebuild("prefix_ids", "prefix_cu_seqlens")
            pp_flat, pp_cu = _rebuild("compression_prompt_ids", "prompt_cu_seqlens")

            scalar_passthrough = {
                k: v for k, v in batch.items()
                if not isinstance(v, (torch.Tensor, list))
            }
            sliced = {
                **scalar_passthrough,
                "sample_type": "file",
                "content_token_ids": c_flat,
                # alias of content_token_ids from the collator; preserve
                # so the slice-completeness assert stays clean
                "encoder_content_token_ids": c_flat,
                "content_cu_seqlens": c_cu,
                "content_position_ids": position_ids_from_cu(c_cu, int(c_flat.shape[0])),
                "content_max_seqlen": max(
                    (int(c_cu[i + 1] - c_cu[i]) for i in range(c_cu.shape[0] - 1)),
                    default=0,
                ),
                "target_token_ids": t_flat,
                "target_cu_seqlens": t_cu,
                "target_loss_mask": l_flat,
                "prefix_ids": p_flat,
                "prefix_cu_seqlens": p_cu,
                "compression_prompt_ids": pp_flat,
                "prompt_cu_seqlens": pp_cu,
            }
            # Per-sample scalar tensors.
            for k in (
                "compression_ratios", "compression_levels",
                "bgkit_splice_start", "bgkit_splice_len",
            ):
                if k in batch and isinstance(batch[k], torch.Tensor):
                    sliced[k] = batch[k][indices]
            if "objectives" in batch and isinstance(batch["objectives"], list):
                sliced["objectives"] = [batch["objectives"][i] for i in indices]

            # Slice forced_survivor_mask_l0 (flat over content axis) along
            # the same per-sample segments as content_token_ids. CRITICAL
            # for eval correctness: without this, encoder's eval path
            # never receives the forced mask, falls back to head's
            # natural selection, and eval measures a different setup than
            # train (looks like overfitting; actually train/eval pipeline
            # divergence). Fix landed 2026-05-14 after misleading eval
            # loss climb.
            forced_full = batch.get("forced_survivor_mask_l0")
            if forced_full is not None:
                content_cu_in = batch["content_cu_seqlens"].to(torch.int64)
                content_starts = content_cu_in[:-1]
                content_ends = content_cu_in[1:]
                forced_parts = [
                    forced_full[int(content_starts[i]) : int(content_ends[i])]
                    for i in indices
                ]
                sliced["forced_survivor_mask_l0"] = (
                    torch.cat(forced_parts, dim=0) if forced_parts
                    else forced_full.new_zeros(0)
                )

            # Slice target_falcon_pair_ids_per_survivor by per-sample
            # forced counts. Each sample's forced count = forced_mask
            # sum within that sample's content segment.
            pair_ids_full = batch.get("target_falcon_pair_ids_per_survivor")
            if pair_ids_full is not None and forced_full is not None:
                # Per-sample forced counts in ORIGINAL batch order.
                content_cu_in2 = batch["content_cu_seqlens"].to(torch.int64)
                per_sample_forced = []
                for i in range(int(content_cu_in2.shape[0]) - 1):
                    s = int(content_cu_in2[i])
                    e = int(content_cu_in2[i + 1])
                    per_sample_forced.append(int(forced_full[s:e].sum().item()))
                # Cumulative offsets into pair_ids_full per original sample.
                cum_forced = [0]
                for c in per_sample_forced:
                    cum_forced.append(cum_forced[-1] + c)
                pair_id_parts = []
                for i in indices:
                    a = cum_forced[i]
                    b = cum_forced[i + 1]
                    pair_id_parts.append(pair_ids_full[a:b])
                sliced["target_falcon_pair_ids_per_survivor"] = (
                    torch.cat(pair_id_parts, dim=0) if pair_id_parts
                    else pair_ids_full.new_zeros((0, 2))
                )

            # Postcondition: every key in the input batch must appear in
            # the sliced output. If a new key was added to the collator
            # without updating _slice_batch, this assert fires loudly.
            missing_keys = set(batch.keys()) - set(sliced.keys())
            if missing_keys:
                raise RuntimeError(
                    f"_slice_batch dropped keys {sorted(missing_keys)!r} "
                    "from the sliced batch. Any new field added to the "
                    "collator must also be sliced here. See compression.py:"
                    "_slice_batch docstring for the 2026-05-15 incident."
                )
            return sliced

        # Repo sample type — slicing requires cu_repo/cu_file surgery.
        # Not used by the current eval path, kept as a plain error to
        # flag misuse.
        raise NotImplementedError(
            "_slice_batch on repo batches is not implemented in the packed path"
        )

    def _evaluate_file_batch(self, batch: dict) -> tuple[float, float]:
        """Token-weighted eval forward for a packed file batch.

        Always-on signals (collected for every batch, aggregated by
        ``evaluate()``):

        - Per-sample survivor / content / target / loss-mask counts,
          fed into ``self._eval_sample_stats`` for distribution summary.
        - Per-sub-batch loss values, fed into the same accumulator so
          ``evaluate()`` can report per-sample loss min/max/std/mean.
        - Ablation hook ``evaluate_ablation_hook(sub_batch, enc_out,
          sub_tokens)`` — runs by default; the no-op base returns zero
          tokens and is skipped. ``CompressionTrainer`` overrides it to
          emit the perfect-projection floor.

        Periodic signal (every ``eval_unsliced_every_n_evals`` evals,
        default 10):

        - Full-batch unsliced forward; logged into ``_eval_unsliced_*``
          accumulators so ``evaluate()`` can emit ``eval/unsliced_loss``
          on those evals. Confirms the sliced/unsliced numerics agree
          (the 2026-05-14 bug class).
        """
        eval_sub = 4
        n_file = int(batch["content_cu_seqlens"].shape[0]) - 1
        file_loss_sum = 0.0
        file_tokens = 0.0

        # Periodic unsliced-vs-sliced comparison (every Nth eval). The
        # full-batch forward is non-trivial so we don't pay it every
        # eval. Cadence configurable via training.eval_unsliced_every_n_evals
        # (default 10). The accumulator is filled lazily — first eval
        # batch where _run_unsliced_this_eval is True does the work.
        run_unsliced = bool(getattr(self, "_run_unsliced_this_eval", False))
        if run_unsliced and n_file >= 2 and not self._eval_unsliced_consumed:
            self._eval_unsliced_consumed = True
            try:
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ):
                    diag_enc = self._compress_file_batch(
                        batch, target_ratio=self._current_target_ratio(),
                    )
                    diag_loss = self._decoder_forward_single_splice(
                        diag_enc.survivor_embeddings,
                        diag_enc.survivor_cu_seqlens,
                        batch,
                    )
                diag_lm = batch.get("target_loss_mask")
                diag_tokens = (
                    int(diag_lm.sum().item()) if diag_lm is not None
                    else int(batch["target_token_ids"].shape[0])
                )
                self._eval_unsliced_loss_sum = (
                    float(diag_loss.item()) * float(diag_tokens)
                )
                self._eval_unsliced_tokens = float(diag_tokens)
                del diag_enc
            except Exception as exc:
                logger.warning("eval_unsliced_check_failed", error=str(exc))

        for fs in range(0, n_file, eval_sub):
            fe = min(fs + eval_sub, n_file)
            sub_batch = self._slice_batch(batch, list(range(fs, fe)))
            with torch.autocast(
                "cuda", dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                enc_out = self._compress_file_batch(
                    sub_batch, target_ratio=self._current_target_ratio(),
                )
                loss = self._decoder_forward_single_splice(
                    enc_out.survivor_embeddings,
                    enc_out.survivor_cu_seqlens,
                    sub_batch,
                )
            eval_loss_mask = sub_batch.get("target_loss_mask")
            if eval_loss_mask is not None:
                sub_tokens = eval_loss_mask.sum().item()
            else:
                sub_tokens = sub_batch["target_token_ids"].shape[0]
            sub_loss_value = float(loss.item())
            sub_tokens_f = float(sub_tokens)
            file_loss_sum += sub_loss_value * sub_tokens_f
            file_tokens += sub_tokens_f

            # Always-on per-sub-batch distribution signal. The
            # sub-batch is the granularity we already have token
            # counts for; aggregating across sub-batches gives
            # min/max/std on the eval-loss distribution + averages
            # on content / target / survivor lengths.
            self._record_eval_sub_batch_stats(
                sub_batch, enc_out, sub_loss_value, sub_tokens_f,
            )

            # Always-on ablation hook. Default base returns
            # (0.0, 0.0, ""); CompressionTrainer overrides to compute
            # the perfect-projection floor when companion data + config
            # flag are present.
            ab_sum, ab_tokens, ab_suffix = self.evaluate_ablation_hook(
                sub_batch, enc_out, sub_tokens_f,
            )
            if ab_suffix and ab_tokens > 0:
                state = self._eval_ablation_state.setdefault(
                    ab_suffix, {"loss_sum": 0.0, "tokens": 0.0},
                )
                state["loss_sum"] += float(ab_sum)
                state["tokens"] += float(ab_tokens)

        return file_loss_sum, file_tokens

    def _record_eval_sub_batch_stats(
        self,
        sub_batch: dict,
        enc_out,
        sub_loss_value: float,
        sub_tokens: float,
    ) -> None:
        """Accumulate always-on per-sub-batch eval stats.

        Cheap reductions on existing tensors; no extra forward passes.
        Aggregated and reported by ``evaluate()`` as ``eval/avg_*`` /
        ``eval/per_sample_loss_*`` metrics.
        """
        try:
            ccu = sub_batch["content_cu_seqlens"].to(torch.int64)
            tcu = sub_batch["target_cu_seqlens"].to(torch.int64)
            scu = enc_out.survivor_cu_seqlens.to(torch.int64)
            n = int(ccu.shape[0]) - 1
            content_total = int((ccu[-1] - ccu[0]).item())
            target_total = int((tcu[-1] - tcu[0]).item())
            survivor_total = int((scu[-1] - scu[0]).item())
            lm = sub_batch.get("target_loss_mask")
            lm_total = int(lm.sum().item()) if lm is not None else target_total

            stats = self._eval_sample_stats
            stats["n_samples"] += n
            stats["content_total"] += content_total
            stats["target_total"] += target_total
            stats["lm_total"] += lm_total
            stats["survivor_total"] += survivor_total
            # Per-sub-batch loss point (weighted by token count when
            # we compute std). Granularity = sub-batch (~4 samples).
            stats["loss_points"].append((sub_loss_value, sub_tokens))
        except Exception as exc:
            # Never let diagnostics break eval.
            logger.warning("eval_sample_stats_failed", error=str(exc))

    def evaluate_ablation_hook(
        self, sub_batch: dict, enc_out, sub_tokens: float,
    ) -> tuple[float, float, str]:
        """Perfect-projection ablation: substitute decoder.embed_tokens(
        target_falcon_pair_ids) for the projection output at survivor
        positions, decoder runs as normal. Isolates projection-quality
        cost from the floor that decoder + compression impose.

        Conditions for the ablation to fire (all required):

        - ``target_falcon_pair_ids_per_survivor`` present on the batch
          (Falcon companion loaded).
        - ``training.use_forced_survivor_mask_l0`` is True (survivor
          count matches pair-id count by construction).
        - The survivor / pair-id arithmetic checks out
          (``n_surv == 2 * N_forced``).

        Returns ``(loss_sum, tokens, "perfect_projection")`` on success,
        ``(0.0, 0.0, "")`` when conditions fail. The caller emits
        ``eval/ablation_perfect_projection_loss`` and
        ``eval/projection_quality_gap`` from the accumulated values.
        """
        pair_ids_full = sub_batch.get("target_falcon_pair_ids_per_survivor")
        forced_mask_full = sub_batch.get("forced_survivor_mask_l0")
        if pair_ids_full is None or forced_mask_full is None:
            return 0.0, 0.0, ""
        if not bool(
            self.cfg.training.get("use_forced_survivor_mask_l0", False),
        ):
            return 0.0, 0.0, ""
        n_surv = int(enc_out.survivor_embeddings.shape[0])
        # When use_forced_survivor_mask_l0=True the encoder selects
        # exactly the forced positions, so 2*N_forced survivor
        # projection slots are produced (output_split=2). pair_ids_full
        # is (N_forced, 2) → embed → (N_forced, 2, D) → flatten →
        # (2*N_forced, D). The flat order matches survivor_embeddings.
        if pair_ids_full.shape[0] * 2 != n_surv:
            return 0.0, 0.0, ""
        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            inner_model, _ = self.decoder._get_inner_model_and_head()
            falcon_embed = inner_model.get_input_embeddings()
            pair_ids_dev = pair_ids_full.to(self.device).long()
            perfect_emb = falcon_embed(pair_ids_dev)
            perfect_flat = perfect_emb.reshape(-1, perfect_emb.shape[-1])
            perfect_flat = perfect_flat.to(
                enc_out.survivor_embeddings.dtype,
            )
            ablation_loss = self._decoder_forward_single_splice(
                perfect_flat,
                enc_out.survivor_cu_seqlens,
                sub_batch,
            )
        return (
            float(ablation_loss.item()) * float(sub_tokens),
            float(sub_tokens),
            "perfect_projection",
        )

    def _evaluate_repo_batch_persample(
        self,
        batch: dict,
    ) -> tuple[float, float]:
        """Packed repo-batch eval forward.

        Uses the same single-packed-L0 → per-repo-L1 algorithm as the
        training step. ``persample`` in the name is legacy — we no longer
        loop over samples.
        """
        from bgkit.training.survivorship_helpers import LevelLossCfg
        from bgkit.utils.packing import position_ids_from_cu

        device = self.device
        cu_repo = batch["cu_repo_seqlens"].to(device)

        loss_mask_flat = batch.get("target_loss_mask")
        if loss_mask_flat is not None:
            loss_mask_flat = loss_mask_flat.to(device).to(torch.bool)

        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda",
        ):
            target_ratio = self._current_target_ratio()
            l0_out = self._compress_repo_l0_packed(batch, target_ratio=target_ratio)
            l1_input_flat, l1_input_cu = self._regroup_survivors_per_repo(
                l0_out.survivor_embeddings,
                l0_out.survivor_cu_seqlens,
                cu_repo,
            )
            l1_input_bridged = self.encoder.l0.auto_reproduce(l1_input_flat)
            n_surv_total = int(l1_input_bridged.shape[0])
            l1_input_positions = position_ids_from_cu(l1_input_cu, n_surv_total)

            surv_l1 = getattr(self, "_surv_l1", LevelLossCfg())
            l1_out = self.encoder.l1(
                content_embeddings=l1_input_bridged,
                content_cu_seqlens=l1_input_cu,
                content_position_ids=l1_input_positions,
                target_ratio=target_ratio,
                min_per_sample=int(surv_l1.min_survivors_absolute_min),
            )
            from bgkit.utils.packing import lengths_from_cu
            proj_cu = l1_out.survivor_cu_seqlens
            proj_lengths = lengths_from_cu(proj_cu).to(torch.int64)
            proj_max = int(proj_lengths.max().item()) if proj_lengths.numel() else 0
            proj_pos = position_ids_from_cu(proj_cu, int(l1_out.survivor_embeddings.shape[0]))
            proj_out = self.encoder.projection_block(
                l1_out.survivor_embeddings,
                cu_seqlens=proj_cu,
                max_seqlen=proj_max,
                position_ids=proj_pos,
                survivor_mask=None,
            )
            projected_cu = effective_projection_cu(proj_out, proj_cu)
            loss = self._decoder_forward_single_splice(
                proj_out.projected_embeddings,
                projected_cu,
                batch,
            )

        # Token count — prefer loss_mask sum; fall back to full target length.
        if loss_mask_flat is not None:
            batch_tokens = float(loss_mask_flat.sum().item())
        else:
            batch_tokens = float(batch["target_token_ids"].shape[0])

        l0_out.release()
        l1_out.release()
        return float(loss.item()) * batch_tokens, batch_tokens

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self, checkpoint_dir: Path, metrics: dict[str, float] | None = None,
    ) -> Path:
        """Save encoder, decoder, optimizer, and curriculum state."""
        if self._training_state is None:
            self._training_state = {}

        # Inject curriculum state
        self._training_state.update({
            "l1_enabled": self._l1_enabled,
            "l1_transitioned": self._l1_transitioned,
            "l1_rebuild_pending": self._l1_rebuild_pending,
            "target_ratio_override": self._target_ratio_override,
        })

        metadata = CheckpointMetadata(
            phase=self.cfg.training.phase,
            step=self.global_step,
            epoch=self.epoch,
            parent_checkpoint=self._last_checkpoint_path,
            metrics=metrics,
            schedule_params=self._schedule_params,
            training_state=self._training_state,
            optimizer_type=self._optimizer_type,
        )
        save_kwargs = dict(
            encoder=self.encoder.state_dict(),
            decoder=self.decoder.state_dict(),
            optimizer_state_by_name=self._build_optimizer_state_by_name(),
        )
        if getattr(self, "_decoder_lora", False):
            save_kwargs["decoder_merged"] = self.decoder.merge_lora()

        ckpt_path = save_checkpoint(checkpoint_dir, metadata, **save_kwargs)
        self._last_checkpoint_path = str(ckpt_path)
        return ckpt_path

    def _named_parameters_for_optimizer(self):
        """Yield (name, param) pairs across encoder + decoder."""
        for name, param in self.encoder.named_parameters():
            yield f"encoder.{name}", param
        for name, param in self.decoder.named_parameters():
            yield f"decoder.{name}", param

    def _restore_model_state(self, state_dicts: dict) -> None:
        if "encoder" in state_dicts:
            enc_state = state_dicts["encoder"]
            result = self.encoder.load_state_dict(enc_state, strict=False)
            if result.missing_keys:
                logger.info(
                    "encoder_missing_keys",
                    keys=result.missing_keys,
                    hint="Expected for new survivorship head components",
                )
        if "decoder" in state_dicts:
            self.decoder.load_state_dict(state_dicts["decoder"])

    def _restore_training_state(self, training_state: dict) -> None:
        self._l1_enabled = training_state.get("l1_enabled", False)
        self._l1_transitioned = training_state.get("l1_transitioned", False)
        self._l1_rebuild_pending = training_state.get("l1_rebuild_pending", False)
        self._target_ratio_override = training_state.get("target_ratio_override")
