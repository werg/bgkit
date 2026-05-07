"""Commit encoding (Phase 1 Step 5): two-level L0/L1 compression on commits.

Two-stage curriculum (see ``plans/step5-l0-freeze-curriculum.md``):

* Stage 0 (``step < stage0_end_step``): L0 entirely frozen via
  ``torch.no_grad()`` + eval mode (NOT just ``requires_grad_(False)``).
  L0 ratio ramps ``stage0_l0_ratio_start`` -> ``stage0_l0_ratio_end``;
  L1 ratio fixed at ``stage0_target_ratio_l1`` (= 1.0). Trains L1, the
  bridge (``encoder.l0.auto_repro_head``), the projection block, and
  decoder LoRA. The L0/L1 heads are pretrained from Step 4 (head fires
  at the block 1 hook) so no head warmup is needed.
* Stage 1 (``step >= stage0_end_step``): two microbatch types
  alternated by ``stage1_l0_unfrozen_period``. Most steps run L0 in
  ``no_grad`` + eval with a large sample-length / batch-token budget;
  every Nth step pulls a small-commit microbatch with L0 backward
  enabled. L1 ratio ramps ``stage1_target_ratio_l1_start`` ->
  ``stage1_target_ratio_l1_end`` over
  ``[stage1_l1_ramp_start_step, stage1_l1_ramp_end_step]``; L0 ratio
  holds at ``stage1_target_ratio_l0``.

All curriculum keys are live-tunable via ``control.json``. User-pinned
``target_ratio_l0`` / ``target_ratio_l1`` overrides the curriculum.
``max_sample_length`` / ``max_batch_tokens`` overrides override the
per-stage routing values.

Two-level packing: each batch carries ``cu_file_seqlens`` (one segment
per file across all commits) plus ``cu_repo_seqlens`` (indices into
``cu_file_seqlens`` marking commit boundaries). Pipeline per microbatch:

1. Pack L0 across every per-file diff with ``cu_file_seqlens``.
2. Regroup the file-level survivor cu_seqlens into per-commit boundaries
   via ``cu_repo_seqlens`` (``survivor_cu[cu_repo]``).
3. Bridge L0 survivors through ``encoder.l0.auto_reproduce`` into L1's
   input-embedding distribution.
4. Pack L1 across per-commit groups.
5. Project L1 survivors and splice into the per-commit decoder targets.
"""

from __future__ import annotations

import dataclasses as _dc
import json
import os
from pathlib import Path
from typing import ClassVar

import numpy as np
import structlog
import torch
from torch.utils.data import DataLoader, random_split

from bgkit.data.collators import collate_compression
from bgkit.data.datasets.commit_encoding_dataset import CommitEncodingDataset
from bgkit.data.samplers import PackedTokenBudgetSampler
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpoint_registry import resolve_checkpoint
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.training.gradient_utils import maybe_enable_gradient_checkpointing
from bgkit.training.ratio_sampling import (
    build_ratio_sampler_config,
    resolve_anchor_grid,
)
from bgkit.utils.attention_backend import resolve_attention_implementation
from bgkit.utils.packing import position_ids_from_cu

logger = structlog.get_logger()


class CommitEncodingTrainer(BaseTrainer):
    """Step 5: commit encoding with split-L0/L1 + L0-freeze curriculum."""

    LIVE_CONFIG_FIELDS: ClassVar[dict[str, str]] = {
        # Stage transitions
        "auto_repro_warmup_steps": "_auto_repro_warmup_steps",
        "head_warmup_steps": "_head_warmup_steps",
        "stage0_end_step": "_stage0_end_step",
        "stage1_l1_ramp_start_step": "_stage1_l1_ramp_start_step",
        "stage1_l1_ramp_end_step": "_stage1_l1_ramp_end_step",
        # L0 ratio curriculum
        "stage0_l0_ratio_start": "_stage0_l0_ratio_start",
        "stage0_l0_ratio_end": "_stage0_l0_ratio_end",
        "stage1_target_ratio_l0": "_stage1_target_ratio_l0",
        # L1 ratio curriculum
        "stage0_target_ratio_l1": "_stage0_target_ratio_l1",
        "stage1_target_ratio_l1_start": "_stage1_target_ratio_l1_start",
        "stage1_target_ratio_l1_end": "_stage1_target_ratio_l1_end",
        # Stage-1 routing
        "stage1_l0_unfrozen_period": "_stage1_l0_unfrozen_period",
        # Diagnostics cadence
        "diagnostic_metrics_every_n_steps": "_diagnostic_metrics_every_n_steps",
    }

    LIVE_CONFIG_HANDLERS: ClassVar[dict[str, str]] = {
        # User overrides (pin / clear via null) — composes ON TOP of curriculum.
        "target_ratio_l0": "_handle_target_ratio_l0",
        "target_ratio_l1": "_handle_target_ratio_l1",
        # Stage 0 sample-length cap (rebuilds primary dataloader).
        "stage0_max_sample_length": "_handle_stage0_max_sample_length",
        # Stage 1 routing budgets (rebuild small/large dataloaders).
        "stage1_l0_frozen_max_sample_length": "_handle_stage1_l0_frozen_max_sample_length",
        "stage1_l0_unfrozen_max_sample_length": (
            "_handle_stage1_l0_unfrozen_max_sample_length"
        ),
        "stage1_l0_frozen_max_batch_tokens": "_handle_stage1_l0_frozen_max_batch_tokens",
        "stage1_l0_unfrozen_max_batch_tokens": "_handle_stage1_l0_unfrozen_max_batch_tokens",
        # Survivorship loss weights (per-level utility-grad gating).
        "utility_grad_loss_weight_l0": "_handle_utility_grad_loss_weight_l0",
        "utility_grad_loss_weight_l1": "_handle_utility_grad_loss_weight_l1",
        # Ratio-sampler legacy keys still honored (sample within window above floor).
        "target_ratio_sampling_window_above": "_handle_ratio_sampling_window_above",
        "sample_target_ratio_during_training": "_handle_ratio_sampling_enabled",
        "target_ratio_anchor_sampling_prob": "_handle_ratio_sampling_anchor_prob",
    }

    def setup(self) -> None:
        tcfg = self.cfg.training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        attention_impl = resolve_attention_implementation(
            self.cfg.compute.get("attention_implementation", "auto")
        )

        # ---- Curriculum config (all live-tunable) ----
        self._init_curriculum_state(tcfg)

        # ---- Encoder ----
        bgkit_cfg = self.cfg.model.bgkit
        backbone_name = bgkit_cfg.backbone_name
        backbone_revision = bgkit_cfg.get("backbone_revision", None)
        hidden_dim = bgkit_cfg.get("hidden_dim", 1024)
        bidi_warmup = tcfg.get("bidi_warmup_steps", 0)
        model_cfg = self.cfg.model
        ctrl_src = model_cfg.get("threshold_controller", {})
        ratio_init = float(self._stage0_l0_ratio_start)
        threshold_controller_cfg = {
            "init_theta": float(ctrl_src.get("init_theta", 1.0 - 2.0 * ratio_init)),
            "lr": float(ctrl_src.get("lr", 0.02)),
            "momentum": float(ctrl_src.get("momentum", 0.0)),
            "clamp": float(ctrl_src.get("clamp", 0.99)),
            "anchor_ratios": list(ctrl_src.get("anchor_ratios", [])) or None,
            "ratio_space": str(ctrl_src.get("ratio_space", "log")),
            "init_target_ratio": ratio_init,
            "default_query_ratio": ratio_init,
        }

        step1_checkpoint = self._resolve_step1_checkpoint()
        step1_state_dicts: dict | None = None
        if step1_checkpoint is not None:
            logger.info("loading_step1_checkpoint", path=step1_checkpoint)
            _, step1_state_dicts = load_checkpoint(Path(step1_checkpoint))
            if "encoder" not in step1_state_dicts:
                raise ValueError(
                    f"step1 checkpoint missing 'encoder' key: {step1_checkpoint}. "
                    f"Found keys: {list(step1_state_dicts.keys())}"
                )

        if step1_state_dicts:
            self.encoder = BgKITEncoder.from_pretrained_with_state_dict(
                backbone_name,
                step1_state_dicts["encoder"],
                hidden_dim=hidden_dim,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                revision=backbone_revision,
                attn_implementation=attention_impl,
                bidi_warmup_steps=bidi_warmup,
                threshold_controller_cfg=threshold_controller_cfg,
            )
        else:
            self.encoder = BgKITEncoder.from_pretrained(
                backbone_name,
                hidden_dim=hidden_dim,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                revision=backbone_revision,
                attn_implementation=attention_impl,
                bidi_warmup_steps=bidi_warmup,
                threshold_controller_cfg=threshold_controller_cfg,
            )
        self.encoder.to(device)

        # ---- Decoder ----
        decoder_cfg = self.cfg.model.decoder
        decoder_name = decoder_cfg.backbone_name
        decoder_revision = decoder_cfg.get("backbone_revision", None)
        logger.info("loading_decoder", model=decoder_name, revision=decoder_revision)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        decoder_backbone = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=decoder_revision,
            attn_implementation=attention_impl,
            device_map=device,
        )
        self.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=hidden_dim)
        self.decoder.set_lm_ce_impl(
            tcfg.get("decoder_ce_impl", self.cfg.compute.get("decoder_ce_impl", None))
        )
        logger.info("decoder_ce_impl_selected", impl=self.decoder.lm_ce_impl)

        if step1_state_dicts is not None:
            decoder_sd = step1_state_dicts.get(
                "decoder_merged", step1_state_dicts.get("decoder")
            )
            if decoder_sd is not None:
                self.decoder.load_state_dict(decoder_sd)

        self._decoder_lora = False
        lora_cfg = tcfg.get("decoder_lora", {})
        if lora_cfg.get("enabled", False):
            self.decoder.apply_lora(lora_cfg)
            self._decoder_lora = True

        maybe_enable_gradient_checkpointing(self.encoder.l0.backbone, self.cfg)
        maybe_enable_gradient_checkpointing(self.encoder.l1.backbone, self.cfg)
        maybe_enable_gradient_checkpointing(self.decoder.backbone, self.cfg)

        if tcfg.get("use_liger", True):
            from bgkit.utils.liger_integration import apply_liger_to_qwen35

            patch_rmsnorm = bool(tcfg.get("use_liger_rmsnorm", False))
            patch_swiglu = bool(tcfg.get("use_liger_swiglu", True))
            patch_rope = bool(tcfg.get("use_liger_rope", True))
            use_liger_ce = (
                bool(tcfg.get("use_liger_ce", True))
                and self.decoder.lm_ce_impl in {"auto", "liger"}
            )
            enc_patched = apply_liger_to_qwen35(
                self.encoder,
                patch_rmsnorm=patch_rmsnorm,
                patch_swiglu=patch_swiglu,
                patch_rope=patch_rope,
            )
            dec_patched = apply_liger_to_qwen35(
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
                patch_rmsnorm=patch_rmsnorm,
                patch_swiglu=patch_swiglu,
                patch_rope=patch_rope,
                use_liger_ce=use_liger_ce,
                decoder_ce_impl=self.decoder.lm_ce_impl,
            )

        self.model = self.decoder

        # ---- Tokenizer ----
        tokenizer_name = decoder_cfg.backbone_name
        tokenizer_revision = decoder_cfg.get("backbone_revision", None)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            revision=tokenizer_revision,
        )

        # ---- Dataset ----
        data_cfg = tcfg.data
        seed = self.cfg.get("seed", 42)
        variant_bank = self._load_variant_bank(data_cfg)

        from bgkit.data.chat_template import TOOL_CONFIGS

        config = TOOL_CONFIGS["commit_encoding"]

        self.commit_dataset = CommitEncodingDataset(
            data_dir=data_cfg.commit_encoding_dir,
            tokenizer=self.tokenizer,
            variant_bank=variant_bank,
            config=config,
            max_diff_tokens_per_file=data_cfg.get("max_diff_tokens_per_file", 4096),
            max_files_per_commit=data_cfg.get("max_files_per_commit", 16),
            max_message_tokens=data_cfg.get("max_message_tokens", 256),
            seed=seed,
        )

        max_eval_samples = tcfg.get("max_eval_samples", 5000)
        total = len(self.commit_dataset)
        eval_size = min(max(1, int(total * 0.1)), max_eval_samples)
        train_size = total - eval_size
        if train_size < 1:
            raise ValueError(
                f"Dataset too small for train/eval split (got {total} samples)"
            )
        self.train_dataset, self.eval_dataset = random_split(
            self.commit_dataset, [train_size, eval_size],
        )

        num_workers = self.cfg.compute.get("num_workers", 4)
        pin_memory = self.cfg.compute.get("pin_memory", False)

        train_lengths = np.array([
            self.commit_dataset.token_length(i)
            for i in self.train_dataset.indices
        ], dtype=np.int64)
        eval_lengths = np.array([
            self.commit_dataset.token_length(i)
            for i in self.eval_dataset.indices
        ], dtype=np.int64)

        # Stash for live-tunable rebuild paths in BaseTrainer.
        self._train_lengths = train_lengths
        self._train_content_lengths = train_lengths
        self._eval_lengths = eval_lengths
        self._train_collate_fn = collate_compression
        self._num_workers = num_workers
        self._pin_memory = pin_memory

        # Initial budget = stage 0 defaults (L0-frozen ramp).
        self._max_batch_tokens = int(self._stage0_max_batch_tokens)
        self._max_batch_tokens_eval = self._resolve_eval_batch_budget(
            tcfg, self._max_batch_tokens,
        )
        self._max_sample_length = int(self._stage0_max_sample_length)
        self._min_sample_length = int(tcfg.get("min_sample_length", 0) or 0)

        self.train_sampler = PackedTokenBudgetSampler(
            self.train_dataset,
            lengths=train_lengths,
            max_batch_tokens=self._max_batch_tokens,
            shuffle=True,
            seed=seed,
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
            max_batch_tokens=self._max_batch_tokens_eval,
            shuffle=False,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_sampler=eval_sampler,
            collate_fn=collate_compression,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Stage-1 secondary "small / L0-unfrozen" dataloader. Built lazily on
        # first stage-1 microbatch — until then it's None and the trainer
        # only ever pulls from `train_dataloader`.
        self._stage1_small_dataloader = None
        self._stage1_small_iter = None

        # ---- Survivorship loss / ICE config (per-level) ----
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

        mm_ref = tcfg.get("moment_match_reference", {}) or {}
        self._ref_moments_l0 = None
        self._ref_moments_l1 = None
        l0_block = mm_ref.get("l0", None) if hasattr(mm_ref, "get") else None
        l0_path = (
            l0_block.get("path", None)
            if l0_block is not None and hasattr(l0_block, "get") else None
        )
        if self._surv_l0.moment_match_weight > 0 and l0_path:
            self._ref_moments_l0 = load_reference_moments(l0_path)
        l1_block = mm_ref.get("l1", None) if hasattr(mm_ref, "get") else None
        l1_path = (
            l1_block.get("path", None)
            if l1_block is not None and hasattr(l1_block, "get") else None
        )
        if self._surv_l1.moment_match_weight > 0 and l1_path:
            self._ref_moments_l1 = load_reference_moments(l1_path)

        self._surv_state_l0 = init_state()
        self._surv_state_l1 = init_state()
        self._last_post_step_metrics: dict[str, float] = {}

        self._diagnostic_metrics_every_n_steps = int(
            tcfg.get("diagnostic_metrics_every_n_steps", 10),
        )

        # Configure freeze flags + optimizer.
        self._configure_trainable_state()

        # Calibrate per-level head_tanh_temperature.
        self._calibrate_head_tanh_temperatures()

        self._profile_enabled = os.environ.get("BGKIT_PROFILE", "") == "1"
        self._last_sampled_target_ratio: float | None = None

        logger.info(
            "commit_encoding_trainer_setup",
            train_samples=train_size,
            eval_samples=eval_size,
            device=str(device),
            stage0_end_step=self._stage0_end_step,
        )

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _init_curriculum_state(self, tcfg) -> None:
        # Two-phase warmup before stage 0:
        #
        # Phase A (auto_repro_warmup): backbones FROZEN + ratios=1.0 (no
        # compression). Only auto_repro_head + projection_block + decoder
        # LoRA train (heads don't fire when ratio>=0.999). This re-anchors
        # the bridge and decoder-projection on their TRAINED distribution
        # — auto_repro_head was trained in Joint Block on full-content L0
        # outputs (no compression, no survive_embedding); projection_block
        # in Step 2.5 was anchored against decoder.embed_tokens(content_ids)
        # at target_ratio=None. Phase A lets both fine-tune to the current
        # backbone state without the additional compression/survive_embedding
        # distribution shift confound.
        #
        # Phase B (head_warmup): backbones STILL frozen, ratios start moving
        # (L0 ramps from 0.9, L1 at target 0.33). Heads fire and start
        # training. auto_repro_head + projection_block re-tune as the
        # input distribution shifts to compressed.
        #
        # Then stage 0: L1 backbone unfreezes, L0 stays no_grad-frozen,
        # ratio ramps continue.
        self._auto_repro_warmup_steps = int(tcfg.get("auto_repro_warmup_steps", 500))
        self._head_warmup_steps = int(tcfg.get("head_warmup_steps", 500))
        self._stage0_end_step = int(tcfg.get("stage0_end_step", 3000))
        self._stage1_l1_ramp_start_step = int(
            tcfg.get("stage1_l1_ramp_start_step", 3000),
        )
        self._stage1_l1_ramp_end_step = int(
            tcfg.get("stage1_l1_ramp_end_step", 5000),
        )

        self._stage0_l0_ratio_start = float(tcfg.get("stage0_l0_ratio_start", 0.9))
        self._stage0_l0_ratio_end = float(tcfg.get("stage0_l0_ratio_end", 0.15))
        self._stage1_target_ratio_l0 = float(tcfg.get("stage1_target_ratio_l0", 0.30))

        self._stage0_target_ratio_l1 = float(tcfg.get("stage0_target_ratio_l1", 1.0))
        self._stage1_target_ratio_l1_start = float(
            tcfg.get("stage1_target_ratio_l1_start", 1.0),
        )
        self._stage1_target_ratio_l1_end = float(
            tcfg.get("stage1_target_ratio_l1_end", 0.33),
        )

        self._stage0_max_sample_length = int(tcfg.get("stage0_max_sample_length", 2000))
        self._stage1_l0_frozen_max_sample_length = int(
            tcfg.get("stage1_l0_frozen_max_sample_length", 6000),
        )
        self._stage1_l0_unfrozen_max_sample_length = int(
            tcfg.get("stage1_l0_unfrozen_max_sample_length", 1500),
        )
        self._stage0_max_batch_tokens = int(
            tcfg.get("stage0_max_batch_tokens", tcfg.get("max_batch_tokens", 8192)),
        )
        self._stage1_l0_frozen_max_batch_tokens = int(
            tcfg.get("stage1_l0_frozen_max_batch_tokens", 16384),
        )
        self._stage1_l0_unfrozen_max_batch_tokens = int(
            tcfg.get("stage1_l0_unfrozen_max_batch_tokens", 4096),
        )
        self._stage1_l0_unfrozen_period = int(
            tcfg.get("stage1_l0_unfrozen_period", 8),
        )

        # User overrides — None means "follow curriculum".
        self._target_ratio_l0_override: float | None = None
        self._target_ratio_l1_override: float | None = None
        self._target_ratio_override: float | None = None  # unified BaseTrainer hook

        # Ratio sampling (window above floor) — kept for compat.
        anchor_grid = resolve_anchor_grid(
            self.cfg.get("model", {}),
            self._stage0_l0_ratio_start,
            getattr(
                self.encoder.l0.threshold,
                "anchor_ratios",
                torch.tensor([self._stage0_l0_ratio_start], dtype=torch.float32),
            ).tolist() if hasattr(self, "encoder") else None,
        ) if hasattr(self, "encoder") else []
        self._target_ratio_sampler_cfg = build_ratio_sampler_config(
            {
                "enabled": tcfg.get("sample_target_ratio_during_training", False),
                "mode": tcfg.get("target_ratio_sampling_mode", "window"),
                "window_above": tcfg.get("target_ratio_sampling_window_above", 0.0),
                "anchor_sampling_prob": tcfg.get(
                    "target_ratio_anchor_sampling_prob", 0.0,
                ),
                "jitter_abs": tcfg.get("target_ratio_jitter_abs", 0.0),
                "jitter_rel": tcfg.get("target_ratio_jitter_rel", 0.0),
            },
            anchor_grid=anchor_grid,
            default_ratio=self._stage0_l0_ratio_start,
            enabled_default=False,
            mode_default="window",
        )
        import random
        self._target_ratio_rng = random.Random(int(self.cfg.get("seed", 42)))

    def _curriculum_state(
        self, step: int,
    ) -> tuple[float, float, bool, int, int]:
        """Compute (ratio_l0, ratio_l1, l0_trainable, max_sample_length, max_batch_tokens).

        Stage 1 has two microbatch types: L0-frozen (default) and L0-unfrozen
        (every Nth step). The boolean ``l0_trainable`` is True only on the
        L0-unfrozen microbatches. ``max_sample_length`` / ``max_batch_tokens``
        track the same routing — large/16K for frozen, small/4K for
        unfrozen.

        User overrides ``target_ratio_l0`` / ``target_ratio_l1`` win over
        the curriculum schedule.
        """
        if step < self._auto_repro_warmup_steps:
            # Phase A: ratios FORCED to 1.0 — no compression at either
            # level. Heads dormant; auto_repro_head + projection_block
            # + decoder LoRA train on their familiar distribution.
            ratio_l0 = 1.0
            ratio_l1 = 1.0
            l0_trainable = False
            max_sample_length = self._stage0_max_sample_length
            max_batch_tokens = self._stage0_max_batch_tokens
        elif step < self._stage0_end_step:
            # Stage 0 (incl. head_warmup as a freeze sub-phase). L0 ramp
            # starts after auto_repro_warmup, completes at stage0_end_step.
            ramp_offset = step - self._auto_repro_warmup_steps
            ramp_span = max(self._stage0_end_step - self._auto_repro_warmup_steps, 1)
            t = ramp_offset / ramp_span
            t = max(0.0, min(1.0, t))
            ratio_l0 = (
                self._stage0_l0_ratio_start
                + t * (self._stage0_l0_ratio_end - self._stage0_l0_ratio_start)
            )
            ratio_l1 = self._stage0_target_ratio_l1
            l0_trainable = False
            max_sample_length = self._stage0_max_sample_length
            max_batch_tokens = self._stage0_max_batch_tokens
        else:
            ratio_l0 = self._stage1_target_ratio_l0
            ramp_start = self._stage1_l1_ramp_start_step
            ramp_end = self._stage1_l1_ramp_end_step
            if step < ramp_start:
                ratio_l1 = self._stage1_target_ratio_l1_start
            elif step >= ramp_end:
                ratio_l1 = self._stage1_target_ratio_l1_end
            else:
                t1 = (step - ramp_start) / max(ramp_end - ramp_start, 1)
                ratio_l1 = (
                    self._stage1_target_ratio_l1_start
                    + t1 * (
                        self._stage1_target_ratio_l1_end
                        - self._stage1_target_ratio_l1_start
                    )
                )
            period = max(int(self._stage1_l0_unfrozen_period), 1)
            l0_trainable = (step % period) == 0
            if l0_trainable:
                max_sample_length = self._stage1_l0_unfrozen_max_sample_length
                max_batch_tokens = self._stage1_l0_unfrozen_max_batch_tokens
            else:
                max_sample_length = self._stage1_l0_frozen_max_sample_length
                max_batch_tokens = self._stage1_l0_frozen_max_batch_tokens

        # Apply user overrides last — pin / clear via control.json.
        if self._target_ratio_l0_override is not None:
            ratio_l0 = self._target_ratio_l0_override
        if self._target_ratio_l1_override is not None:
            ratio_l1 = self._target_ratio_l1_override
        # Legacy single-knob override: applies to L0 (the dominant level).
        if self._target_ratio_override is not None:
            ratio_l0 = self._target_ratio_override

        return float(ratio_l0), float(ratio_l1), bool(l0_trainable), int(max_sample_length), int(max_batch_tokens)

    def _current_target_ratio(self) -> float:
        """Compatibility shim — returns current L0 ratio."""
        ratio_l0, _, _, _, _ = self._curriculum_state(self.global_step)
        return ratio_l0

    def _curriculum_ratio_l1(self) -> float:
        _, ratio_l1, _, _, _ = self._curriculum_state(self.global_step)
        return ratio_l1

    # ------------------------------------------------------------------
    # Trainable state
    # ------------------------------------------------------------------

    def _configure_trainable_state(self) -> None:
        """Set requires_grad flags + optimizer based on current curriculum stage.

        Two coarse modes:
        * ``stage0`` or ``stage1_frozen`` (L0 frozen, L1 trainable): heads +
          bridge + projection_block + decoder LoRA + L1 train. L0 backbone
          params don't receive grad regardless because we wrap L0 forward in
          ``torch.no_grad()``.
        * ``stage1_unfrozen`` (every Nth stage-1 step): everything trainable.
        """
        step = int(self.global_step)
        stage = self._coarse_stage(step)

        # Always-trainable surfaces (in EVERY stage).
        always_trainable = [
            self.encoder.l0.head,
            self.encoder.l1.head,
            self.encoder.l0.auto_repro_head,
            self.encoder.projection_block,
        ]

        # Reset all encoder params to frozen first.
        self.encoder.requires_grad_(False)
        for module in always_trainable:
            module.requires_grad_(True)

        if stage == "auto_repro_warmup":
            # Phase A: backbones frozen, ratios=1.0 (so heads don't fire,
            # no head gradient). auto_repro_head + projection_block +
            # decoder LoRA carry the gradient signal.
            self._l0_trainable_static = False
            self._l1_trainable_static = False
        elif stage == "head_warmup":
            # Phase B: backbones still frozen, ratios start moving. Heads
            # fire and train.
            self._l0_trainable_static = False
            self._l1_trainable_static = False
        elif stage == "stage0" or stage == "stage1_frozen":
            self._l0_trainable_static = False
            self._l1_trainable_static = True
            self.encoder.l1.requires_grad_(True)
        else:  # stage1_unfrozen
            self._l0_trainable_static = True
            self._l1_trainable_static = True
            self.encoder.l0.requires_grad_(True)
            self.encoder.l1.requires_grad_(True)

        # Ensure projection block + bridge stay trainable after the broad reset.
        for module in always_trainable:
            module.requires_grad_(True)

        # Decoder: respect LoRA, otherwise full train.
        self._unfreeze_decoder_respecting_lora()

        self._setup_optimizer()

        proj_params = sum(
            p.numel() for p in self.encoder.projection_block.parameters() if p.requires_grad
        )
        l0_params = sum(p.numel() for p in self.encoder.l0.parameters() if p.requires_grad)
        l1_params = sum(p.numel() for p in self.encoder.l1.parameters() if p.requires_grad)
        dec_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        logger.info(
            "trainable_state_configured",
            step=step,
            stage=stage,
            l0_params=l0_params,
            l1_params=l1_params,
            projection_params=proj_params,
            decoder_params=dec_params,
        )

    def _coarse_stage(self, step: int) -> str:
        if step < self._auto_repro_warmup_steps:
            return "auto_repro_warmup"
        if step < self._auto_repro_warmup_steps + self._head_warmup_steps:
            return "head_warmup"
        if step < self._stage0_end_step:
            return "stage0"
        period = max(int(self._stage1_l0_unfrozen_period), 1)
        return "stage1_unfrozen" if (step % period) == 0 else "stage1_frozen"

    def _unfreeze_decoder_respecting_lora(self) -> None:
        if getattr(self.decoder, "_has_lora", False):
            for name, p in self.decoder.named_parameters():
                p.requires_grad_("lora_" in name)
        else:
            self.decoder.requires_grad_(True)

    def _muon_excluded_param_ids(self) -> frozenset[int]:
        exclude = set()
        backbone = self.decoder.backbone
        try:
            from peft import PeftModel
            causal_lm = backbone.base_model.model if isinstance(backbone, PeftModel) else backbone
        except ImportError:
            causal_lm = backbone
        inner = getattr(causal_lm, "model", None)
        if inner is not None and hasattr(inner, "embed_tokens"):
            for p in inner.embed_tokens.parameters():
                if p.requires_grad:
                    exclude.add(id(p))
        lm_head = getattr(causal_lm, "lm_head", None)
        if lm_head is not None:
            for p in lm_head.parameters():
                if p.requires_grad:
                    exclude.add(id(p))
        return frozenset(exclude)

    def _setup_optimizer(self) -> None:
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
            param_groups, tcfg.lr, exclude_from_muon=self._muon_excluded_param_ids(),
        )

    def trainable_parameters(self) -> list:
        params = [p for p in self.encoder.parameters() if p.requires_grad]
        params += [p for p in self.decoder.parameters() if p.requires_grad]
        return params

    # ------------------------------------------------------------------
    # Live-config handlers
    # ------------------------------------------------------------------

    def _handle_target_ratio_l0(self, val) -> None:
        if val is None:
            self._target_ratio_l0_override = None
            logger.info("live_target_ratio_l0_cleared", resuming="curriculum")
            return
        if isinstance(val, (int, float)) and 0 < float(val) <= 1.0:
            self._target_ratio_l0_override = float(val)
            logger.info("live_target_ratio_l0_update", target_ratio=float(val))
            return
        logger.warning(
            "live_target_ratio_l0_invalid", value=val, expected="None or float in (0, 1]",
        )

    def _handle_target_ratio_l1(self, val) -> None:
        if val is None:
            self._target_ratio_l1_override = None
            logger.info("live_target_ratio_l1_cleared", resuming="curriculum")
            return
        if isinstance(val, (int, float)) and 0 < float(val) <= 1.0:
            self._target_ratio_l1_override = float(val)
            logger.info("live_target_ratio_l1_update", target_ratio=float(val))
            return
        logger.warning(
            "live_target_ratio_l1_invalid", value=val, expected="None or float in (0, 1]",
        )

    def _handle_stage0_max_sample_length(self, val) -> None:
        self._stage0_max_sample_length = int(val)
        if self._coarse_stage(int(self.global_step)) == "stage0":
            self._refresh_active_dataloader_for_stage()

    def _handle_stage1_l0_frozen_max_sample_length(self, val) -> None:
        self._stage1_l0_frozen_max_sample_length = int(val)
        # Refresh primary (large) loader if we're already in stage1.
        if self._coarse_stage(int(self.global_step)).startswith("stage1"):
            self._refresh_active_dataloader_for_stage()

    def _handle_stage1_l0_unfrozen_max_sample_length(self, val) -> None:
        self._stage1_l0_unfrozen_max_sample_length = int(val)
        # Force lazy rebuild of small-loader on next stage-1 unfrozen tick.
        self._stage1_small_dataloader = None
        self._stage1_small_iter = None

    def _handle_stage1_l0_frozen_max_batch_tokens(self, val) -> None:
        self._stage1_l0_frozen_max_batch_tokens = int(val)
        if self._coarse_stage(int(self.global_step)).startswith("stage1"):
            self._refresh_active_dataloader_for_stage()

    def _handle_stage1_l0_unfrozen_max_batch_tokens(self, val) -> None:
        self._stage1_l0_unfrozen_max_batch_tokens = int(val)
        self._stage1_small_dataloader = None
        self._stage1_small_iter = None

    def _handle_utility_grad_loss_weight_l0(self, val) -> None:
        if not isinstance(val, (int, float)) or float(val) < 0:
            return
        self._surv_l0 = _dc.replace(self._surv_l0, utility_grad_loss_weight=float(val))

    def _handle_utility_grad_loss_weight_l1(self, val) -> None:
        if not isinstance(val, (int, float)) or float(val) < 0:
            return
        self._surv_l1 = _dc.replace(self._surv_l1, utility_grad_loss_weight=float(val))

    def _refresh_active_dataloader_for_stage(self) -> None:
        """Rebuild the primary train dataloader so its budget + filters
        match the current curriculum stage. Keeps cursor across rebuild.
        """
        _, _, _, max_sample_length, max_batch_tokens = self._curriculum_state(
            int(self.global_step),
        )
        self._max_sample_length = int(max_sample_length)
        self._max_batch_tokens = int(max_batch_tokens)
        self._rebuild_train_dataloader_with_budget(self._max_batch_tokens)

    # ------------------------------------------------------------------
    # Stage-1 small (L0-unfrozen) dataloader
    # ------------------------------------------------------------------

    def _build_stage1_small_dataloader(self):
        """Create a separate small-commit / tight-budget dataloader for the
        L0-unfrozen microbatches. Filtered by ``stage1_l0_unfrozen_max_sample_length``,
        budgeted at ``stage1_l0_unfrozen_max_batch_tokens``.
        """
        max_len = int(self._stage1_l0_unfrozen_max_sample_length)
        budget = int(self._stage1_l0_unfrozen_max_batch_tokens)
        from torch.utils.data import Subset

        full_lengths = (
            self._train_lengths_full
            if hasattr(self, "_train_lengths_full")
            else self._train_lengths
        )
        full_dataset = (
            self._train_dataset_full
            if hasattr(self, "_train_dataset_full")
            else self.train_dataset
        )
        valid = np.where(full_lengths <= max_len)[0]
        if len(valid) == 0:
            logger.warning(
                "stage1_small_dataloader_filter_dropped_all",
                max_len=max_len,
                full_max=int(full_lengths.max()) if len(full_lengths) else 0,
            )
            valid = np.arange(len(full_lengths))
        ds = Subset(full_dataset, valid.tolist())
        sampler = PackedTokenBudgetSampler(
            ds,
            lengths=full_lengths[valid],
            max_batch_tokens=budget,
            shuffle=True,
            seed=int(self.cfg.get("seed", 42)) + 7919,
        )
        loader = DataLoader(
            ds,
            batch_sampler=sampler,
            collate_fn=collate_compression,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )
        return loader

    def _next_stage1_small_batch(self):
        if self._stage1_small_dataloader is None:
            self._stage1_small_dataloader = self._build_stage1_small_dataloader()
            self._stage1_small_iter = iter(self._stage1_small_dataloader)
        try:
            return next(self._stage1_small_iter)
        except StopIteration:
            self._stage1_small_iter = iter(self._stage1_small_dataloader)
            return next(self._stage1_small_iter)

    # ------------------------------------------------------------------
    # Variant bank + checkpoint resolution
    # ------------------------------------------------------------------

    def _load_variant_bank(self, data_cfg) -> list[dict[str, str]]:
        variant_dir = getattr(data_cfg, "prompt_variants_dir", None)
        if variant_dir:
            bank_path = Path(variant_dir) / "commit_encoding.json"
            if bank_path.exists():
                with open(bank_path) as f:
                    bank = json.load(f)
                if bank:
                    return bank
        return [{
            "system_prompt": "You are an AI assistant with access to the "
                             "bgkit_reproduce_commit tool.",
            "user_prompt": "Reproduce the commit from repository {file_path}",
            "compression_prompt": "Reproduce the complete commit from "
                                  "compressed context",
            "response_prefix": "Here is the reconstructed commit:",
        }]

    def _resolve_step1_checkpoint(self) -> str | None:
        step1_checkpoint = self.cfg.get("step1_checkpoint", None)
        self._input_sources = {}
        if step1_checkpoint == "auto":
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            # Prefer phase1_step4p7_v2 (bridge re-distilled with extended
            # ratio range and decoder LoRA preserved from a Step 5 run)
            # -> phase1_step4p7 (original bridge distillation)
            # -> phase1_step4_split (raw split conversion)
            # -> legacy phase1_step4 (Step 5 cannot consume directly).
            resolved = None
            for phase in (
                "phase1_step4p7_v3",
                "phase1_step4p7_v2",
                "phase1_step4p7",
                "phase1_step4_split",
                "phase1_step4",
            ):
                try:
                    resolved = resolve_checkpoint(
                        checkpoint_dir,
                        phase=phase,
                        metric="eval/loss",
                        label="step1_checkpoint",
                    )
                    break
                except (FileNotFoundError, RuntimeError, ValueError):
                    continue
            if resolved is None:
                raise RuntimeError(
                    "step1_checkpoint=auto but no phase1_step4p7_v2, "
                    "phase1_step4p7, phase1_step4_split, or phase1_step4 "
                    "checkpoint found. Run scripts/convert_step4_to_split_l0l1.py "
                    "first, then optionally phase1_step4p7 bridge distillation.",
                )
            step1_checkpoint = str(resolved)
        if step1_checkpoint is not None:
            self._input_sources["step1"] = Path(step1_checkpoint).name
        return step1_checkpoint

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _calibrate_head_tanh_temperatures(self, n_probe_batches: int = 4) -> None:
        from bgkit.training.survivorship_helpers import calibrate_head_tanh_temperature
        for level in ("l0", "l1"):
            try:
                T = calibrate_head_tanh_temperature(
                    self.encoder,
                    self.train_dataloader,
                    self.device,
                    level=level,
                    n_probe_batches=n_probe_batches,
                    batch_to_content=_commit_batch_to_content,
                )
                if T is not None:
                    logger.info("head_tanh_temperature_calibrated", level=level, T=T)
            except Exception as exc:
                logger.warning(
                    "head_tanh_temperature_calibration_failed",
                    level=level,
                    error=str(exc),
                )

    # ------------------------------------------------------------------
    # Two-level encoder forward
    # ------------------------------------------------------------------

    def _encode_two_level(
        self,
        batch: dict,
        target_ratio_l0: float,
        target_ratio_l1: float,
        l0_trainable: bool,
    ):
        """Run the L0 → bridge → L1 → projection chain on a packed repo batch.

        Returns ``(projected_survivors, projected_survivor_cu, l0_out, l1_out,
        actual_l0_ratio_info)``. ``l0_out`` / ``l1_out`` are ``LevelOutput``s.

        When ``l0_trainable=False`` we wrap the L0 forward in
        ``torch.no_grad()`` and put L0 in eval mode. Activations are NOT
        stored, so memory usage drops sharply (the load-bearing
        "maximum advantage of frozen L0" property).
        """
        device = self.device
        file_ids = batch["content_token_ids"].to(device)
        cu_file = batch["cu_file_seqlens"].to(device)
        content_position_ids = batch["content_position_ids"].to(device)
        prompt_ids = batch["prompt_token_ids"].to(device)
        prompt_cu = batch["prompt_cu_seqlens"].to(device)
        prompt_position_ids = position_ids_from_cu(prompt_cu, int(prompt_ids.shape[0]))
        cu_repo = batch["cu_repo_seqlens"].to(device)

        embed = self.encoder.l0.backbone.get_input_embeddings()
        content_emb = embed(file_ids)
        prompt_emb = embed(prompt_ids)

        util_active_l0 = (
            l0_trainable and self._surv_l0.utility_grad_loss_weight > 0.0
        )
        util_active_l1 = self._surv_l1.utility_grad_loss_weight > 0.0

        # ---- L0 forward (per-file packing) ----
        if l0_trainable:
            self.encoder.l0.train()
            l0_out = self.encoder.l0(
                content_embeddings=content_emb,
                content_cu_seqlens=cu_file,
                content_position_ids=content_position_ids,
                prompt_embeddings=prompt_emb,
                prompt_cu_seqlens=prompt_cu,
                prompt_position_ids=prompt_position_ids,
                target_ratio=target_ratio_l0,
                utility_grad_active=util_active_l0,
            )
        else:
            self.encoder.l0.eval()
            with torch.no_grad():
                l0_out = self.encoder.l0(
                    content_embeddings=content_emb,
                    content_cu_seqlens=cu_file,
                    content_position_ids=content_position_ids,
                    prompt_embeddings=prompt_emb,
                    prompt_cu_seqlens=prompt_cu,
                    prompt_position_ids=prompt_position_ids,
                    target_ratio=target_ratio_l0,
                    utility_grad_active=False,
                )

        # ---- Regroup per-commit + bridge ----
        l0_survivors = l0_out.survivor_embeddings
        l0_surv_cu = l0_out.survivor_cu_seqlens
        # cu_repo holds indices INTO cu_file (file-count boundaries).
        cu_repo_64 = cu_repo.to(torch.int64)
        cu_file_64 = l0_surv_cu.to(torch.int64)
        l1_input_cu = cu_file_64[cu_repo_64].to(torch.int32)

        bridged = self.encoder.l0.auto_reproduce(l0_survivors)
        n_l1 = int(bridged.shape[0])
        l1_pos = position_ids_from_cu(l1_input_cu, n_l1)

        # ---- L1 forward (per-commit packing) ----
        self.encoder.l1.train(self._l1_trainable_static)
        l1_out = self.encoder.l1(
            content_embeddings=bridged,
            content_cu_seqlens=l1_input_cu,
            content_position_ids=l1_pos,
            target_ratio=target_ratio_l1,
            utility_grad_active=util_active_l1,
        )

        # ---- Projection ----
        proj_input = l1_out.survivor_embeddings
        proj_cu = l1_out.survivor_cu_seqlens
        proj_pos = position_ids_from_cu(proj_cu, int(proj_input.shape[0]))
        from bgkit.utils.packing import lengths_from_cu
        proj_max = int(lengths_from_cu(proj_cu).max().item()) if proj_cu.numel() else 0
        proj_out = self.encoder.projection_block(
            proj_input,
            cu_seqlens=proj_cu,
            max_seqlen=proj_max,
            position_ids=proj_pos,
            survivor_mask=None,
        )

        return (
            proj_out.projected_embeddings,
            proj_cu,
            l0_out,
            l1_out,
            cu_file,
        )

    # ------------------------------------------------------------------
    # Decoder forward / loss
    # ------------------------------------------------------------------

    def _decoder_forward_single_splice(
        self,
        survivors: torch.Tensor,
        survivor_cu_seqlens: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        device = self.device
        target_ids_flat = batch["target_token_ids"].to(device)
        target_cu = batch["target_cu_seqlens"].to(device)
        splice_start = batch["bgkit_splice_start"].to(device)
        splice_len = batch["bgkit_splice_len"].to(device)
        loss_mask_flat = batch.get("target_loss_mask")
        if loss_mask_flat is not None:
            loss_mask_flat = loss_mask_flat.to(device).to(torch.bool)

        batch_size = int(target_cu.shape[0]) - 1
        tok_cu_list = target_cu.to(torch.int64).tolist()
        surv_cu_list = survivor_cu_seqlens.to(torch.int64).tolist()

        prefix_ids: list[torch.Tensor] = []
        suffix_ids: list[torch.Tensor] = []
        per_segment_loss_masks: list[torch.Tensor] = []
        for b in range(batch_size):
            sample_start = int(tok_cu_list[b])
            sample_end = int(tok_cu_list[b + 1])
            sample_tokens = target_ids_flat[sample_start:sample_end]
            splice_b_start = int(splice_start[b].item())
            splice_b_len = int(splice_len[b].item())
            if splice_b_start < 0:
                splice_b_start = sample_tokens.shape[0]
                splice_b_len = 0
            pre = sample_tokens[:splice_b_start]
            suf = sample_tokens[splice_b_start + splice_b_len :]
            prefix_ids.append(pre)
            suffix_ids.append(suf)

            k_i = int(surv_cu_list[b + 1]) - int(surv_cu_list[b])
            surv_mask = torch.zeros(k_i, dtype=torch.bool, device=device)
            if loss_mask_flat is not None:
                sample_loss = loss_mask_flat[sample_start:sample_end]
                pre_mask = sample_loss[:splice_b_start]
                suf_mask = sample_loss[splice_b_start + splice_b_len :]
            else:
                pre_mask = torch.zeros(pre.shape[0], dtype=torch.bool, device=device)
                suf_mask = torch.ones(suf.shape[0], dtype=torch.bool, device=device)
            per_segment_loss_masks.append(torch.cat([pre_mask, surv_mask, suf_mask], dim=0))

        flat_loss_mask = (
            torch.cat(per_segment_loss_masks, dim=0)
            if per_segment_loss_masks else None
        )
        return self.decoder.forward_with_single_splice(
            survivor_embeddings=survivors,
            survivor_cu_seqlens=survivor_cu_seqlens,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            loss_mask=flat_loss_mask,
        )

    # ------------------------------------------------------------------
    # Survivorship aux losses
    # ------------------------------------------------------------------

    def _apply_surv_losses(
        self,
        l_out,
        level: str,
        target_ratio: float,
        content_token_ids: torch.Tensor | None,
        content_cu_seqlens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        from bgkit.training.survivorship_helpers import (
            LevelICECfg,
            LevelLossCfg,
            accumulate,
            compute_survivorship_losses,
            init_state,
            survivorship_diagnostics,
        )

        if level == "l0":
            weights = self._surv_l0
            ice_cfg = self._ice_l0
            ref_moments = self._ref_moments_l0
            state = self._surv_state_l0 or init_state()
            self._surv_state_l0 = state
        else:
            weights = self._surv_l1
            ice_cfg = self._ice_l1
            ref_moments = self._ref_moments_l1
            state = self._surv_state_l1 or init_state()
            self._surv_state_l1 = state

        loss, metrics = compute_survivorship_losses(
            enc_out=l_out,
            level=level,
            weights=weights,
            ice_cfg=ice_cfg,
            ref_moments=ref_moments,
            ice_teacher=self._ice_teacher,
            global_step=self.global_step,
            content_token_ids=content_token_ids,
            content_cu_seqlens=content_cu_seqlens,
            target_ratio=target_ratio,
        )
        accumulate(state, l_out, target_ratio=target_ratio)
        out_metrics = {f"{level}_{k}": v for k, v in metrics.items()}
        diag_every_n = int(getattr(self, "_diagnostic_metrics_every_n_steps", 1) or 1)
        out_metrics.update(
            survivorship_diagnostics(
                l_out, level=level, global_step=self.global_step,
                every_n_steps=diag_every_n,
            )
        )
        return loss, out_metrics

    def _accumulate_l0_state_only(self, l0_out, target_ratio: float) -> None:
        """Update dual-ascent θ_L0 state without computing surv losses.

        Used when L0 backbone is frozen but the curriculum's
        target_ratio_l0 is moving — keeps θ_L0 in sync with the target
        without flowing gradient through L0. apply_post_step_updates picks
        up _surv_state_l0 unconditionally and updates θ.
        """
        from bgkit.training.survivorship_helpers import accumulate, init_state

        state = self._surv_state_l0 or init_state()
        self._surv_state_l0 = state
        accumulate(state, l0_out, target_ratio=target_ratio)

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _pre_step_hook(self) -> None:
        """Reconfigure trainable state if we crossed a stage boundary."""
        step = int(self.global_step)
        new_stage = self._coarse_stage(step)
        prev_stage = getattr(self, "_last_coarse_stage", None)
        if prev_stage != new_stage:
            self._configure_trainable_state()
            # When entering stage 1 / stage 2, switch the primary loader to the
            # stage-appropriate budget. (Stage 0 uses the original setup defaults.)
            if prev_stage is not None:
                self._refresh_active_dataloader_for_stage()
            self._last_coarse_stage = new_stage

    def _route_microbatch(
        self, default_batch: dict, l0_trainable: bool,
    ) -> tuple[dict, bool]:
        """Decide whether this microbatch comes from the primary (L0-frozen)
        dataloader or the stage-1 small (L0-unfrozen) dataloader.

        ``default_batch`` was already pulled from the primary loader by the
        BaseTrainer loop. If we're entering an L0-unfrozen step in stage 1,
        we discard ``default_batch`` and pull from the small loader instead.
        """
        if not l0_trainable:
            return default_batch, False
        # L0-unfrozen step in stage 1 — try to pull a small-commit microbatch.
        try:
            small_batch = self._next_stage1_small_batch()
        except StopIteration:
            # Empty small loader — fall back to the primary batch but train L0.
            return default_batch, True
        return small_batch, True

    def _forward_backward(self, batch: dict) -> dict[str, float]:
        from bgkit.training.survivorship_helpers import LevelLossCfg

        target_ratio_l0, target_ratio_l1, want_l0_train, _, _ = self._curriculum_state(
            int(self.global_step),
        )

        batch, l0_trainable = self._route_microbatch(batch, want_l0_train)
        self._last_sampled_target_ratio = target_ratio_l0

        device = self.device
        cu_repo = batch["cu_repo_seqlens"].to(device)
        batch_size = int(cu_repo.shape[0]) - 1
        scale = 1.0 / (batch_size * self._accum_steps)
        aux_metrics: dict[str, float] = {}

        self.decoder.train()

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            (
                projected_survivors,
                projected_survivor_cu,
                l0_out,
                l1_out,
                cu_file,
            ) = self._encode_two_level(
                batch,
                target_ratio_l0=target_ratio_l0,
                target_ratio_l1=target_ratio_l1,
                l0_trainable=l0_trainable,
            )

            l0_surv_loss = None
            if l0_trainable and l0_out.logits_for_op is not None:
                l0_surv_loss, l0_metrics = self._apply_surv_losses(
                    l0_out,
                    level="l0",
                    target_ratio=target_ratio_l0,
                    content_token_ids=batch["content_token_ids"].to(device),
                    content_cu_seqlens=cu_file,
                )
                aux_metrics.update(l0_metrics)
            elif l0_out.logits_for_op is not None:
                # L0 backbone is frozen this microbatch, so we skip the loss
                # backward through L0. But dual-ascent θ_L0 still needs to
                # track the curriculum's L0 ratio target — otherwise θ_L0
                # stays frozen at its loaded value and L0 actual_ratio
                # diverges from target_ratio_l0 (observed on phase1_step5
                # stage 0: θ_L0 calibrated for ratio 0.08 stuck the L0
                # selection at ~0.2 while curriculum asked for 0.5+).
                self._accumulate_l0_state_only(l0_out, target_ratio_l0)

            l1_surv_loss = None
            if l1_out.logits_for_op is not None:
                l1_surv_loss, l1_metrics = self._apply_surv_losses(
                    l1_out,
                    level="l1",
                    target_ratio=target_ratio_l1,
                    content_token_ids=None,
                    content_cu_seqlens=l1_out.content_cu_seqlens,
                )
                aux_metrics.update(l1_metrics)

            loss = self._decoder_forward_single_splice(
                projected_survivors, projected_survivor_cu, batch,
            )

            total_loss_t = loss
            if l0_surv_loss is not None and l0_surv_loss.requires_grad:
                total_loss_t = total_loss_t + l0_surv_loss
            if l1_surv_loss is not None and l1_surv_loss.requires_grad:
                total_loss_t = total_loss_t + l1_surv_loss

        (total_loss_t * batch_size * scale).backward()

        # Utility-gradient BCE (post-backward) on L0 + L1.
        util_w_l0 = self._surv_l0.utility_grad_loss_weight if l0_trainable else 0.0
        util_w_l1 = self._surv_l1.utility_grad_loss_weight
        if util_w_l0 > 0.0 and l0_out.post_head_content_values is not None:
            from bgkit.training.survivorship_helpers import utility_grad_bce_loss
            util_loss, util_metrics = utility_grad_bce_loss(
                base_raw_for_util=l0_out.base_raw_for_util,
                content_grad=l0_out.get_content_grad(),
                content_values=l0_out.post_head_content_values,
                valid_mask=None,
                pinned_mask=None,
                target_ratio=target_ratio_l0,
                content_cu_seqlens=cu_file,
            )
            if util_loss.requires_grad:
                (util_loss * util_w_l0 * batch_size * scale).backward()
            for k, v in util_metrics.items():
                aux_metrics[f"l0_{k}"] = v

        if util_w_l1 > 0.0 and l1_out.post_head_content_values is not None:
            from bgkit.training.survivorship_helpers import utility_grad_bce_loss
            util_loss, util_metrics = utility_grad_bce_loss(
                base_raw_for_util=l1_out.base_raw_for_util,
                content_grad=l1_out.get_content_grad(),
                content_values=l1_out.post_head_content_values,
                valid_mask=None,
                pinned_mask=None,
                target_ratio=target_ratio_l1,
                content_cu_seqlens=l1_out.content_cu_seqlens,
            )
            if util_loss.requires_grad:
                (util_loss * util_w_l1 * batch_size * scale).backward()
            for k, v in util_metrics.items():
                aux_metrics[f"l1_{k}"] = v

        total_survivors = int(projected_survivors.shape[0])
        total_valid = int(batch["content_token_ids"].shape[0])
        actual_ratio = total_survivors / max(total_valid, 1)

        result = {
            "loss": float(loss.item()),
            "target_ratio_l0": target_ratio_l0,
            "target_ratio_l1": target_ratio_l1,
            "actual_ratio": actual_ratio,
            "l0_trainable": float(l0_trainable),
        }
        result.update(aux_metrics)

        l0_out.release()
        l1_out.release()
        # Suppress unused-var warnings (LevelLossCfg only used to keep import).
        _ = LevelLossCfg
        return result

    def _post_step(self, step: int) -> None:
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
                self.encoder,
                state,
                target_ratio=None,
                level=level,
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
        if self._last_sampled_target_ratio is not None:
            metrics.setdefault("sampled_target_ratio", self._last_sampled_target_ratio)
        post = getattr(self, "_last_post_step_metrics", None)
        if post:
            for k, v in post.items():
                metrics.setdefault(k, v)
        # Always log curriculum state for observability.
        ratio_l0, ratio_l1, _, _, _ = self._curriculum_state(int(self.global_step))
        metrics.setdefault("curriculum_target_ratio_l0", ratio_l0)
        metrics.setdefault("curriculum_target_ratio_l1", ratio_l1)
        metrics.setdefault("curriculum_stage", _stage_id(self._coarse_stage(int(self.global_step))))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Eval = present-survivor reconstruction loss + ablation gaps.

        Three passes:
          - ``present``: real encoder survivors (full eval set)
          - ``zeroed``: survivors zeroed before splice (subset; ablation)
          - ``noise``: survivors replaced by ``randn_like`` (subset; ablation)

        Reports ``eval/loss`` (present) and ``eval/ablation/{loss,gap}_{zeroed,noise}``.
        Positive ``gap_*`` means the encoder is contributing — the decoder's
        reconstruction loss is lower with real survivors than with the ablated
        substitute. ``gap`` near zero means the decoder is silently ignoring
        the encoder.

        Ablation passes run on a SUBSET (``eval.ablation_max_batches``,
        default 100) and only every Nth eval (``eval.ablation_every_n_evals``,
        default 1) so they don't dominate eval cost.
        """
        self.encoder.eval()
        self.decoder.eval()

        target_ratio_l0, target_ratio_l1, _, _, _ = self._curriculum_state(
            int(self.global_step),
        )
        eval_cfg = self.cfg.get("eval", {})
        ablation_max_batches = int(eval_cfg.get("ablation_max_batches", 100))
        ablation_every_n_evals = int(eval_cfg.get("ablation_every_n_evals", 1))
        self._eval_count = getattr(self, "_eval_count", 0) + 1
        do_ablation = (
            ablation_max_batches > 0
            and ablation_every_n_evals > 0
            and (self._eval_count % ablation_every_n_evals == 0)
        )

        def _run_pass(survivor_transform, max_batches=None, label="present"):
            """Loop over eval set, optionally transform survivors before splice."""
            total_loss = 0.0
            total_tokens = 0.0
            num_batches = len(self.eval_dataloader)
            for batch_idx, batch in enumerate(self.eval_dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                if batch_idx % 100 == 0:
                    logger.info(
                        "eval_progress", batch=batch_idx, total=num_batches,
                        pass_label=label,
                    )
                loss_mask_flat = batch.get("target_loss_mask")
                if loss_mask_flat is not None:
                    loss_mask_flat = loss_mask_flat.to(self.device).to(torch.bool)
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ):
                    (
                        projected_survivors,
                        projected_survivor_cu,
                        l0_out,
                        l1_out,
                        _,
                    ) = self._encode_two_level(
                        batch,
                        target_ratio_l0=target_ratio_l0,
                        target_ratio_l1=target_ratio_l1,
                        l0_trainable=False,
                    )
                    if survivor_transform is not None:
                        projected_survivors = survivor_transform(projected_survivors)
                    loss = self._decoder_forward_single_splice(
                        projected_survivors, projected_survivor_cu, batch,
                    )
                if loss_mask_flat is not None:
                    batch_tokens = float(loss_mask_flat.sum().item())
                else:
                    batch_tokens = float(batch["target_token_ids"].shape[0])
                total_loss += float(loss.item()) * batch_tokens
                total_tokens += batch_tokens
                l0_out.release()
                l1_out.release()
            return total_loss / max(total_tokens, 1)

        try:
            present_loss = _run_pass(survivor_transform=None, label="present")
            results = {
                "loss": present_loss,
                "perplexity": torch.exp(torch.tensor(present_loss)).item(),
            }
            if do_ablation:
                # Subset present-loss for fair gap arithmetic against subset ablations.
                subset_present_loss = _run_pass(
                    survivor_transform=None,
                    max_batches=ablation_max_batches,
                    label="present_subset",
                )
                zeroed_loss = _run_pass(
                    survivor_transform=lambda s: torch.zeros_like(s),
                    max_batches=ablation_max_batches,
                    label="zeroed",
                )
                noise_loss = _run_pass(
                    survivor_transform=lambda s: torch.randn_like(s) * s.std(),
                    max_batches=ablation_max_batches,
                    label="noise",
                )
                results["ablation/loss_present_subset"] = subset_present_loss
                results["ablation/loss_zeroed"] = zeroed_loss
                results["ablation/loss_noise"] = noise_loss
                results["ablation/gap_zeroed"] = zeroed_loss - subset_present_loss
                results["ablation/gap_noise"] = noise_loss - subset_present_loss
                results["ablation/n_batches"] = float(ablation_max_batches)
            return results
        finally:
            self.encoder.train()
            self.decoder.train()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self, checkpoint_dir: Path, metrics: dict[str, float] | None = None,
    ) -> Path:
        if self._training_state is None:
            self._training_state = {}
        self._training_state.update({
            "target_ratio_l0_override": self._target_ratio_l0_override,
            "target_ratio_l1_override": self._target_ratio_l1_override,
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
        for name, param in self.encoder.named_parameters():
            yield f"encoder.{name}", param
        for name, param in self.decoder.named_parameters():
            yield f"decoder.{name}", param

    def _restore_model_state(self, state_dicts: dict) -> None:
        if "encoder" in state_dicts:
            self.encoder.load_state_dict(state_dicts["encoder"])
        if "decoder" in state_dicts:
            self.decoder.load_state_dict(state_dicts["decoder"])

    def _restore_training_state(self, training_state: dict) -> None:
        self._target_ratio_l0_override = training_state.get("target_ratio_l0_override")
        self._target_ratio_l1_override = training_state.get("target_ratio_l1_override")


def _stage_id(stage_name: str) -> float:
    """Numeric stage id for wandb logging — float so the dict stays uniform."""
    return {
        "auto_repro_warmup": -1.0,
        "head_warmup": -0.5,
        "stage0": 0.0,
        "stage1_frozen": 1.0,
        "stage1_unfrozen": 1.5,
    }.get(stage_name, -1.0)


def _commit_batch_to_content(batch):
    """Extract (content_token_ids, content_cu_seqlens) from a commit-encoding
    batch — calibrate_head_tanh_temperature default looks for
    ``content_token_ids`` + ``content_cu_seqlens``, but our collator emits
    file-level segmentation under ``cu_file_seqlens``.
    """
    if not isinstance(batch, dict):
        return None
    ids = batch.get("content_token_ids")
    cu = batch.get("cu_file_seqlens")
    if ids is None or cu is None:
        return None
    return ids, cu
