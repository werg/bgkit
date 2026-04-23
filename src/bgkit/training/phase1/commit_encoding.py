"""Commit encoding: two-level hierarchical encoder training on git commits.

Single-objective trainer with L0 (per-file) + L1 (cross-file) compression
from step 0. Asymmetric encoder/decoder prompting: encoder gets commit
message as conditioning, decoder must reconstruct the full commit purely
from compressed survivors.

Pipeline position: Step 4 — between Step 3 (pruned reconstruction) and Step 5
(multi-objective compression). Takes Step 1 or Step 2 checkpoint, unfreezes
both encoder and decoder. Produces checkpoints in the same multi-artifact
format as CompressionTrainer (encoder.pt + decoder.pt) so Step 4 can consume them.
"""

from __future__ import annotations

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
    sample_ratio,
)
from bgkit.utils.attention_backend import resolve_attention_implementation

logger = structlog.get_logger()


class CommitEncodingTrainer(BaseTrainer):
    """Single-objective trainer for two-level commit encoding (packed).

    Packed-attention pipeline (FA4 varlen). Each training step runs:
      1. ONE packed L0 forward across every per-file diff in every
         commit in the microbatch, segmented by ``cu_file_seqlens``.
      2. Regroup the flat L0 survivors into per-commit groups using
         ``cu_repo_seqlens`` (pure index shuffle on survivor_cu_seqlens).
      3. ONE packed L1 forward across all commits in the microbatch.
      4. ONE packed decoder call splicing each commit's L1 survivors
         into its target sequence.

    The previous per-sample L0 loop + per-group L1 loop is gone — packing
    removes the quadratic-padding cost that motivated the serial path.

    Uses BaseTrainer.train() loop — no L1 introduction hook needed since
    both L0 and L1 are active from step 0. Overrides save_checkpoint()
    and load_checkpoint() to use the multi-artifact format that
    CompressionTrainer expects.
    """

    LIVE_CONFIG_FIELDS: ClassVar[dict[str, str]] = {
        "target_ratio_ramp_steps": "_target_ratio_ramp_steps",
        "target_ratio_start": "_target_ratio_start",
        "target_ratio_end": "_target_ratio_end",
    }

    LIVE_CONFIG_HANDLERS: ClassVar[dict[str, str]] = {
        "target_ratio": "_handle_target_ratio",
    }

    def setup(self) -> None:
        """Load trainable encoder/decoder, create dataset and optimizer."""
        tcfg = self.cfg.training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        attention_impl = resolve_attention_implementation(
            self.cfg.compute.get("attention_implementation", "auto")
        )

        # --- BgKIT encoder (trainable) ---
        bgkit_cfg = self.cfg.model.bgkit
        backbone_name = bgkit_cfg.backbone_name
        backbone_revision = bgkit_cfg.get("backbone_revision", None)
        hidden_dim = bgkit_cfg.get("hidden_dim", 1024)
        bidi_warmup = tcfg.get("bidi_warmup_steps", 500)
        model_cfg = self.cfg.model
        ctrl_src = model_cfg.get("threshold_controller", {})
        curriculum = tcfg.get("curriculum", {})
        target_ratio_start = float(
            curriculum.get("target_ratio_start", tcfg.get("target_ratio_start", 0.50)),
        )
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
            if "encoder" not in step1_state_dicts:
                raise ValueError(
                    f"Step 2 checkpoint missing 'encoder' key: {step1_checkpoint}. "
                    f"Found keys: {list(step1_state_dicts.keys())}"
                )

        # Auto-detect pruned architecture from state dict keys
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

        self.encoder.requires_grad_(True)
        self.encoder.train()
        maybe_enable_gradient_checkpointing(
            self.encoder.compressor.backbone, self.cfg,
        )

        # --- Decoder (trainable) ---
        decoder_cfg = self.cfg.model.decoder
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
            attn_implementation=attention_impl,
            device_map=device,  # load directly to CUDA, avoid CPU staging copy
        )
        self.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=hidden_dim)

        if step1_state_dicts is not None:
            # Prefer merged decoder (LoRA weights folded in) over raw PeftModel state
            decoder_sd = step1_state_dicts.get(
                "decoder_merged", step1_state_dicts.get("decoder")
            )
            if decoder_sd is not None:
                self.decoder.load_state_dict(decoder_sd)

        # LoRA wrapping (after checkpoint load, before NVFP4/gradient checkpointing)
        self._decoder_lora = False
        lora_cfg = tcfg.get("decoder_lora", {})
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

        maybe_enable_gradient_checkpointing(self.decoder.backbone, self.cfg)

        # torch.compile disabled: sm_121 shared memory (101 KB) is too small for
        # inductor-generated Triton kernels (need ~148 KB). "reduce-overhead" and
        # "cudagraphs" backends also use inductor or suffer from dynamic shape
        # re-recording overhead. PeftModel isinstance fix landed in decoder.py
        # for when this becomes viable.

        # Optional Liger Kernel fused kernels (RMSNorm / SwiGLU / RoPE +
        # fused linear+CE). Gated on ``training.use_liger`` (default True);
        # no-op when liger-kernel is not installed.
        if tcfg.get("use_liger", True):
            from bgkit.utils.liger_integration import apply_liger_to_qwen35

            enc_patched = apply_liger_to_qwen35(self.encoder)
            dec_patched = apply_liger_to_qwen35(self.decoder)
            self.decoder.enable_liger_ce(True)
            logger.info(
                "liger_kernel_applied",
                encoder_modules=enc_patched,
                decoder_modules=dec_patched,
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

        # --- Dataset ---
        data_cfg = tcfg.data
        seed = self.cfg.get("seed", 42)

        # Load variant bank
        variant_bank = self._load_variant_bank(data_cfg)

        from bgkit.data.chat_template import TOOL_CONFIGS

        config = TOOL_CONFIGS["commit_encoding"]

        commit_encoding_dir = data_cfg.commit_encoding_dir
        self.commit_dataset = CommitEncodingDataset(
            data_dir=commit_encoding_dir,
            tokenizer=self.tokenizer,
            variant_bank=variant_bank,
            config=config,
            max_diff_tokens_per_file=data_cfg.get("max_diff_tokens_per_file", 4096),
            max_files_per_commit=data_cfg.get("max_files_per_commit", 16),
            max_message_tokens=data_cfg.get("max_message_tokens", 256),
            seed=seed,
        )

        # Train/eval split
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

        # --- Sampler + DataLoader ---
        max_batch_tokens = tcfg.get("max_batch_tokens", 65536)
        # Eval defaults to 2× train budget (no backward → lower peak at
        # same budget). Overridable via training.max_batch_tokens_eval.
        max_batch_tokens_eval = self._resolve_eval_batch_budget(tcfg, max_batch_tokens)
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

        # Stash for live-tunable budget rebuild (see BaseTrainer._handle_max_batch_tokens)
        self._train_lengths = train_lengths
        self._eval_lengths = eval_lengths
        self._train_collate_fn = collate_compression
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._max_batch_tokens = max_batch_tokens
        self._max_batch_tokens_eval = max_batch_tokens_eval

        self.train_sampler = PackedTokenBudgetSampler(
            self.train_dataset,
            lengths=train_lengths,
            max_batch_tokens=max_batch_tokens,
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
            max_batch_tokens=max_batch_tokens_eval,
            shuffle=False,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_sampler=eval_sampler,
            collate_fn=collate_compression,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # --- Optimizer ---
        self._setup_optimizer()

        # --- Curriculum state ---
        curriculum = tcfg.get("curriculum", {})
        self._target_ratio_start = curriculum.get(
            "target_ratio_start", tcfg.get("target_ratio_start", 0.50),
        )
        self._target_ratio_end = curriculum.get(
            "target_ratio_end", tcfg.get("target_ratio_end", 0.20),
        )
        self._target_ratio_ramp_steps = curriculum.get(
            "target_ratio_ramp_steps", tcfg.get("target_ratio_ramp_steps", 30000),
        )
        self._target_ratio_override: float | None = None
        anchor_grid = resolve_anchor_grid(
            self.cfg.model,
            float(self._target_ratio_start),
            getattr(self.encoder.compressor.threshold_l0, "anchor_ratios", None),
        )
        self._target_ratio_sampler_cfg = build_ratio_sampler_config(
            {
                "enabled": tcfg.get("sample_target_ratio_during_training", False),
                "mode": tcfg.get("target_ratio_sampling_mode", "window"),
                "sampling_max": tcfg.get(
                    "target_ratio_sampling_max",
                    max(float(self._target_ratio_start), max(anchor_grid)),
                ),
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

        # --- Survivorship head config (per-level) ---
        # Step 4 is where L1 first activates — L0 head is trained from Step 3
        # and just continues; L1 cold-starts via decisiveness + utility-grad BCE.
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
            embed_tokens = self.encoder.compressor.backbone.embed_tokens
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
        l0_path = _l0_block.get("path", None) if _l0_block is not None and hasattr(_l0_block, "get") else None
        if self._surv_l0.moment_match_weight > 0 and l0_path:
            self._ref_moments_l0 = load_reference_moments(l0_path)
        _l1_block = mm_ref.get("l1", None) if hasattr(mm_ref, "get") else None
        l1_path = _l1_block.get("path", None) if _l1_block is not None and hasattr(_l1_block, "get") else None
        if self._surv_l1.moment_match_weight > 0 and l1_path:
            self._ref_moments_l1 = load_reference_moments(l1_path)

        self._surv_state_l0 = init_state()
        self._surv_state_l1 = init_state()
        self._last_post_step_metrics: dict[str, float] = {}

        # --- Diagnostic metrics cadence ---
        # Gates the GPU→CPU syncs for head-health metrics (organic rate
        # std, undecided fraction, floor trigger rate, θ). Mirrors Step 3.
        self._diagnostic_metrics_every_n_steps = int(
            tcfg.get("diagnostic_metrics_every_n_steps", 10),
        )

        # --- Head-tanh temperature calibration (L0 + L1) ---
        # L0 inherits from Step 3's sidecar-calibrated T via the loaded
        # checkpoint; L1 is brand-new at Step 4 and needs its own T set
        # against its head's init-scale output std. Running the probe
        # unconditionally at both levels is cheap (4 batches, no backward)
        # and robust to checkpoint-vs-fresh-head distinctions: if L0 was
        # already calibrated, the probe confirms it; if the user resumes
        # from a checkpoint with drifted T, the probe corrects it.
        self._calibrate_head_tanh_temperatures()

        # --- Profiling ---
        self._profile_enabled = os.environ.get("BGKIT_PROFILE", "") == "1"

        logger.info(
            "commit_encoding_trainer_setup",
            train_samples=train_size,
            eval_samples=eval_size,
            device=str(device),
            profile=self._profile_enabled,
        )

    def _calibrate_head_tanh_temperatures(self, n_probe_batches: int = 4) -> None:
        """Probe L0 + L1 head output std at startup and set their per-level
        ``head_tanh_temperature_l{0,1}`` buffers. See
        :func:`bgkit.training.survivorship_helpers.calibrate_head_tanh_temperature`.
        """
        from bgkit.training.survivorship_helpers import (
            calibrate_head_tanh_temperature,
        )
        for level in ("l0", "l1"):
            calibrated_T = calibrate_head_tanh_temperature(
                self.encoder.compressor,
                self.train_dataloader,
                self.device,
                level=level,
                n_probe_batches=n_probe_batches,
            )
            if calibrated_T is not None:
                logger.info(
                    "head_tanh_temperature_calibrated",
                    level=level,
                    T=calibrated_T,
                )

    def _load_variant_bank(self, data_cfg) -> list[dict[str, str]]:
        """Load commit_encoding variant bank from prompt_variants_dir."""
        variant_dir = getattr(data_cfg, "prompt_variants_dir", None)
        if variant_dir:
            bank_path = Path(variant_dir) / "commit_encoding.json"
            if bank_path.exists():
                with open(bank_path) as f:
                    bank = json.load(f)
                if bank:
                    return bank

        # Fallback
        return [{
            "system_prompt": "You are an AI assistant with access to the "
                             "bgkit_reproduce_commit tool.",
            "user_prompt": "Reproduce the commit from repository {file_path}",
            "compression_prompt": "Reproduce the complete commit from "
                                  "compressed context",
            "response_prefix": "Here is the reconstructed commit:",
        }]

    def _resolve_step1_checkpoint(self) -> str | None:
        """Resolve step1_checkpoint: auto -> best phase1_step4 checkpoint."""
        step1_checkpoint = self.cfg.get("step1_checkpoint", None)
        self._input_sources = {}

        if step1_checkpoint == "auto":
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            resolved = resolve_checkpoint(
                checkpoint_dir,
                phase="phase1_step4",
                metric="eval/loss",
                label="step1_checkpoint",
            )
            step1_checkpoint = str(resolved)

        if step1_checkpoint is not None:
            self._input_sources["step1"] = Path(step1_checkpoint).name

        return step1_checkpoint

    def _muon_excluded_param_ids(self) -> frozenset[int]:
        """Return param IDs of decoder embedding/lm_head — should not use Muon."""
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
        params = [p for p in self.encoder.parameters() if p.requires_grad]
        params += [p for p in self.decoder.parameters() if p.requires_grad]
        return params

    def trainable_parameters(self) -> list:
        return self._trainable_params()

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _current_target_ratio(self) -> float:
        if self._target_ratio_override is not None:
            return self._target_ratio_override
        step = self.global_step
        if step >= self._target_ratio_ramp_steps:
            return self._target_ratio_end
        t = step / max(self._target_ratio_ramp_steps, 1)
        return self._target_ratio_start + t * (
            self._target_ratio_end - self._target_ratio_start
        )

    def _sample_target_ratio(self) -> float:
        """Sample a requested ratio for the current packed batch."""
        if not hasattr(self, "_target_ratio_sampler_cfg"):
            tcfg = self.cfg.training
            curriculum = tcfg.get("curriculum", {})
            model_cfg = getattr(self.cfg, "model", {})
            anchor_grid = resolve_anchor_grid(
                model_cfg,
                float(self._target_ratio_start),
                getattr(self.encoder.compressor.threshold_l0, "anchor_ratios", None),
            )
            self._target_ratio_sampler_cfg = build_ratio_sampler_config(
                {
                    "enabled": tcfg.get("sample_target_ratio_during_training", False),
                    "mode": tcfg.get("target_ratio_sampling_mode", "window"),
                    "sampling_max": tcfg.get(
                        "target_ratio_sampling_max",
                        max(float(self._target_ratio_start), max(anchor_grid)),
                    ),
                    "anchor_sampling_prob": tcfg.get(
                        "target_ratio_anchor_sampling_prob", 0.30,
                    ),
                    "jitter_abs": tcfg.get("target_ratio_jitter_abs", 0.0),
                    "jitter_rel": tcfg.get("target_ratio_jitter_rel", 0.0),
                },
                anchor_grid=anchor_grid,
                default_ratio=float(
                    curriculum.get(
                        "target_ratio_start",
                        tcfg.get("target_ratio_start", self._target_ratio_start),
                    ),
                ),
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
            is_evaluating=not self.encoder.training,
            override_active=self._target_ratio_override is not None,
        )

    # ------------------------------------------------------------------
    # Compression pipeline: packed L0 across files -> packed L1 per commit
    # ------------------------------------------------------------------

    def _compress_repo_l0_packed(self, batch: dict, target_ratio: float):
        """Single packed L0 forward across every file in every commit.

        Uses ``cu_file_seqlens`` (one segment per file) and per-file-tiled
        ``prompt_token_ids`` from the repo collator.
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

        bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()
        content_emb = bgkit_embed(file_ids)
        prompt_emb = bgkit_embed(prompt_ids)

        util_active = getattr(
            self, "_surv_l0", LevelLossCfg(),
        ).utility_grad_loss_weight > 0.0
        return self.encoder(
            content_embeddings=content_emb,
            content_cu_seqlens=cu_file,
            content_position_ids=content_position_ids,
            prompt_embeddings=prompt_emb,
            prompt_cu_seqlens=prompt_cu,
            prompt_position_ids=prompt_position_ids,
            target_ratio=target_ratio,
            level="l0",
            utility_grad_active=util_active,
        )

    @staticmethod
    def _regroup_survivors_per_repo(
        l0_survivors: torch.Tensor,
        l0_survivor_cu: torch.Tensor,
        cu_repo_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert per-file survivor groups into per-commit survivor groups.

        ``l0_survivor_cu`` has shape ``(total_files + 1,)``;
        ``cu_repo_seqlens`` has shape ``(B + 1,)`` and holds indices
        *into* the file axis. Since survivors are already emitted in
        file-order, building the commit-level cu_seqlens is just a
        gather over file boundaries — no tensor shuffle.
        """
        cu_repo = cu_repo_seqlens.to(torch.int64)
        cu_file = l0_survivor_cu.to(torch.int64)
        survivor_cu_repo = cu_file[cu_repo]
        return l0_survivors, survivor_cu_repo.to(torch.int32)

    def _decoder_forward_single_splice(
        self,
        survivors: torch.Tensor,
        survivor_cu_seqlens: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        """Packed decoder forward + CE loss.

        Splits flat ``target_token_ids`` at each sample's splice
        boundaries to produce per-sample prefix/suffix lists, carries
        the per-segment loss mask over the ``[prefix | survivors |
        suffix]`` layout that ``forward_with_single_splice`` expects.
        """
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

    def _apply_surv_losses(
        self,
        enc_out,
        level: str,
        target_ratio: float,
        content_token_ids: torch.Tensor | None = None,
        content_cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute per-level survivorship loss + accumulate microbatch state."""
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
    # Forward/backward
    # ------------------------------------------------------------------

    def _forward_backward(self, batch: dict) -> dict[str, float]:
        """Packed forward + backward — single L0, single L1, single decoder.

        See class docstring for the full flow.
        """
        from bgkit.training.survivorship_helpers import LevelLossCfg
        from bgkit.utils.packing import position_ids_from_cu

        self.encoder.train()
        self.decoder.train()

        device = self.device
        target_ratio = self._sample_target_ratio()
        self._last_sampled_target_ratio = target_ratio
        cu_repo = batch["cu_repo_seqlens"].to(device)
        batch_size = int(cu_repo.shape[0]) - 1
        scale = 1.0 / (batch_size * self._accum_steps)
        aux_metrics: dict[str, float] = {}

        util_w_l0 = getattr(
            self, "_surv_l0", LevelLossCfg(),
        ).utility_grad_loss_weight
        util_w_l1 = getattr(
            self, "_surv_l1", LevelLossCfg(),
        ).utility_grad_loss_weight

        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda",
        ):
            # Step 1: packed L0 across all files in all commits.
            l0_out = self._compress_repo_l0_packed(batch, target_ratio=target_ratio)
            l0_surv_loss = None
            if l0_out.logits_for_op is not None:
                l0_surv_loss, l0_metrics = self._apply_surv_losses(
                    l0_out, level="l0", target_ratio=target_ratio,
                    content_token_ids=batch["content_token_ids"].to(device),
                    content_cu_seqlens=batch["cu_file_seqlens"].to(device),
                )
                aux_metrics.update(l0_metrics)

            # Step 2: regroup per-commit.
            l1_input_flat, l1_input_cu = self._regroup_survivors_per_repo(
                l0_out.survivor_embeddings,
                l0_out.survivor_cu_seqlens,
                cu_repo,
            )
            n_surv_total = int(l1_input_flat.shape[0])
            l1_input_positions = position_ids_from_cu(l1_input_cu, n_surv_total)
            empty_prompt_cu = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=device,
            )
            empty_prompt_emb = l1_input_flat.new_zeros(0, l1_input_flat.shape[-1])
            empty_prompt_pos = torch.zeros(0, dtype=torch.int64, device=device)

            # Step 3: packed L1 across all commits.
            l1_out = self.encoder(
                content_embeddings=l1_input_flat,
                content_cu_seqlens=l1_input_cu,
                content_position_ids=l1_input_positions,
                prompt_embeddings=empty_prompt_emb,
                prompt_cu_seqlens=empty_prompt_cu,
                prompt_position_ids=empty_prompt_pos,
                target_ratio=target_ratio,
                level="l1",
                utility_grad_active=util_w_l1 > 0.0,
            )
            l1_surv_loss = None
            if l1_out.logits_for_op is not None:
                l1_surv_loss, l1_metrics = self._apply_surv_losses(
                    l1_out, level="l1", target_ratio=target_ratio,
                    content_token_ids=None,
                    content_cu_seqlens=l1_input_cu,
                )
                aux_metrics.update(l1_metrics)

            # Step 4: packed decoder.
            loss = self._decoder_forward_single_splice(
                l1_out.survivor_embeddings,
                l1_out.survivor_cu_seqlens,
                batch,
            )

            total_loss_t = loss
            if l0_surv_loss is not None and l0_surv_loss.requires_grad:
                total_loss_t = total_loss_t + l0_surv_loss
            if l1_surv_loss is not None and l1_surv_loss.requires_grad:
                total_loss_t = total_loss_t + l1_surv_loss

        (total_loss_t * batch_size * scale).backward()

        # Utility-gradient BCE (post-backward) for both levels.
        if util_w_l0 > 0.0 and l0_out.post_head_content_values is not None:
            from bgkit.training.survivorship_helpers import utility_grad_bce_loss

            util_loss, util_metrics = utility_grad_bce_loss(
                base_raw_for_util=l0_out.base_raw_for_util,
                content_grad=l0_out.get_content_grad(),
                content_values=l0_out.post_head_content_values,
                valid_mask=None,
                pinned_mask=None,
                target_ratio=target_ratio,
                content_cu_seqlens=batch["cu_file_seqlens"].to(device),
            )
            if util_loss.requires_grad:
                (util_loss * util_w_l0 * batch_size * scale).backward()
            for k, v in util_metrics.items():
                aux_metrics[f"l0_{k}"] = v
            if l0_out.get_content_grad() is not None:
                aux_metrics["l0_content_grad_norm"] = float(
                    l0_out.get_content_grad().norm().item(),
                )

        if util_w_l1 > 0.0 and l1_out.post_head_content_values is not None:
            from bgkit.training.survivorship_helpers import utility_grad_bce_loss

            util_loss, util_metrics = utility_grad_bce_loss(
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
            for k, v in util_metrics.items():
                aux_metrics[f"l1_{k}"] = v
            if l1_out.get_content_grad() is not None:
                aux_metrics["l1_content_grad_norm"] = float(
                    l1_out.get_content_grad().norm().item(),
                )

        total_survivors = int(l1_out.survivor_embeddings.shape[0])
        total_valid = int(batch["content_token_ids"].shape[0])
        actual_ratio = total_survivors / max(total_valid, 1)

        result = {
            "loss": float(loss.item()),
            "target_ratio": self._current_target_ratio(),
            "sampled_target_ratio": target_ratio,
            "actual_ratio": actual_ratio,
        }
        result.update(aux_metrics)

        l0_out.release()
        l1_out.release()

        return result

    def _post_step(self, step: int) -> None:
        """Advance bidirectional warmup and run dual-ascent θ + EMA μ updates."""
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
                self.encoder.compressor,
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

    def _get_bidi_alpha(self) -> float:
        from bgkit.models.bidirectional_qwen35 import BidirectionalQwen35
        from bgkit.models.pruned_qwen35 import PrunedBidirectionalQwen35

        for module in self.encoder.modules():
            if isinstance(module, (BidirectionalQwen35, PrunedBidirectionalQwen35)):
                return module.bidi_alpha
        return 1.0

    def _add_step_metrics(self, metrics: dict[str, float]) -> None:
        """Attach post-step θ/μ updates (without clobbering base metrics)."""
        if self._last_sampled_target_ratio is not None:
            metrics.setdefault("sampled_target_ratio", self._last_sampled_target_ratio)
        post = getattr(self, "_last_post_step_metrics", None)
        if post:
            for k, v in post.items():
                metrics.setdefault(k, v)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Packed eval — same L0 → L1 → decoder algorithm, no backward."""
        from bgkit.utils.packing import position_ids_from_cu

        self.encoder.eval()
        self.decoder.eval()

        try:
            total_loss = 0.0
            total_tokens = 0.0
            target_ratio = self._current_target_ratio()

            device = self.device
            num_batches = len(self.eval_dataloader)
            for batch_idx, batch in enumerate(self.eval_dataloader):
                if batch_idx % 100 == 0:
                    logger.info("eval_progress", batch=batch_idx, total=num_batches)

                cu_repo = batch["cu_repo_seqlens"].to(device)
                batch_size = int(cu_repo.shape[0]) - 1

                loss_mask_flat = batch.get("target_loss_mask")
                if loss_mask_flat is not None:
                    loss_mask_flat = loss_mask_flat.to(device).to(torch.bool)

                with torch.autocast(
                    "cuda", dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ):
                    l0_out = self._compress_repo_l0_packed(
                        batch, target_ratio=target_ratio,
                    )
                    l1_input_flat, l1_input_cu = self._regroup_survivors_per_repo(
                        l0_out.survivor_embeddings,
                        l0_out.survivor_cu_seqlens,
                        cu_repo,
                    )
                    n_surv_total = int(l1_input_flat.shape[0])
                    l1_input_positions = position_ids_from_cu(l1_input_cu, n_surv_total)
                    empty_prompt_cu = torch.zeros(
                        batch_size + 1, dtype=torch.int32, device=device,
                    )
                    empty_prompt_emb = l1_input_flat.new_zeros(
                        0, l1_input_flat.shape[-1],
                    )
                    empty_prompt_pos = torch.zeros(0, dtype=torch.int64, device=device)
                    l1_out = self.encoder(
                        content_embeddings=l1_input_flat,
                        content_cu_seqlens=l1_input_cu,
                        content_position_ids=l1_input_positions,
                        prompt_embeddings=empty_prompt_emb,
                        prompt_cu_seqlens=empty_prompt_cu,
                        prompt_position_ids=empty_prompt_pos,
                        target_ratio=target_ratio,
                        level="l1",
                    )
                    loss = self._decoder_forward_single_splice(
                        l1_out.survivor_embeddings,
                        l1_out.survivor_cu_seqlens,
                        batch,
                    )

                if loss_mask_flat is not None:
                    batch_tokens = float(loss_mask_flat.sum().item())
                else:
                    batch_tokens = float(batch["target_token_ids"].shape[0])
                total_loss += float(loss.item()) * batch_tokens
                total_tokens += batch_tokens

                l0_out.release()
                l1_out.release()

            avg_loss = total_loss / max(total_tokens, 1)
            perplexity = torch.exp(torch.tensor(avg_loss)).item()

            return {
                "loss": avg_loss,
                "perplexity": perplexity,
            }
        finally:
            self.encoder.train()
            self.decoder.train()

    # ------------------------------------------------------------------
    # Checkpointing — multi-artifact format (matches CompressionTrainer)
    # ------------------------------------------------------------------

    def save_checkpoint(
        self, checkpoint_dir: Path, metrics: dict[str, float] | None = None,
    ) -> Path:
        """Save encoder, decoder, optimizer, and calibrator state."""
        if self._training_state is None:
            self._training_state = {}

        self._training_state.update({
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
            from bgkit.models.encoder import migrate_legacy_threshold_controller_state_dict

            enc_state = migrate_legacy_threshold_controller_state_dict(
                state_dicts["encoder"],
                self.encoder,
            )
            self.encoder.load_state_dict(enc_state)
        if "decoder" in state_dicts:
            self.decoder.load_state_dict(state_dicts["decoder"])

    def _restore_training_state(self, training_state: dict) -> None:
        self._target_ratio_override = training_state.get("target_ratio_override")
