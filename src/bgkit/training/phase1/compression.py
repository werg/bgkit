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

import os
from pathlib import Path
from typing import ClassVar

import structlog
import torch
from torch.utils.data import DataLoader, random_split

from bgkit.data.collators import collate_compression
from bgkit.data.datasets.compression_dataset import CompressionDataset
from bgkit.data.samplers import PackedTokenBudgetSampler
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpoint_registry import resolve_checkpoint
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.training.gradient_utils import enable_gradient_checkpointing
from bgkit.training.objectives.commit_reproduction import commit_reproduction_loss
from bgkit.training.objectives.data_reconstruction import data_reconstruction_loss
from bgkit.training.objectives.description_generation import description_generation_loss
from bgkit.training.objectives.structural_relational import structural_relational_loss
from bgkit.utils.attention_backend import resolve_attention_implementation

logger = structlog.get_logger()

# Maps objective name to its loss function
_LOSS_FNS = {
    "data_reconstruction": data_reconstruction_loss,
    "description_generation": description_generation_loss,
    "structural_relational": structural_relational_loss,
    "commit_reproduction": commit_reproduction_loss,
}


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
    }

    LIVE_CONFIG_HANDLERS: ClassVar[dict[str, str]] = {
        "target_ratio": "_handle_target_ratio",
    }

    def setup(self) -> None:
        """Load trainable encoder/decoder, create dataset and optimizer."""
        import gc

        tcfg = self.cfg.training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        attention_impl = resolve_attention_implementation(
            self.cfg.compute.get("attention_implementation", "auto")
        )

        # --- BgKIT encoder (trainable in Step 2) ---
        bgkit_cfg = self.cfg.model.bgkit
        backbone_name = bgkit_cfg.backbone_name
        backbone_revision = bgkit_cfg.get("backbone_revision", None)
        hidden_dim = bgkit_cfg.get("hidden_dim", 1024)
        bidi_warmup = self.cfg.training.get("bidi_warmup_steps", 1000)

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
            )
        self.encoder.to(device)
        gc.collect()

        # Encoder is trainable in Step 4
        self.encoder.requires_grad_(True)
        self.encoder.train()
        enable_gradient_checkpointing(self.encoder.compressor.backbone)

        # --- Decoder (trainable, with drift monitoring) ---
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
            device_map=device,
        )
        self.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=hidden_dim)

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

        enable_gradient_checkpointing(self.decoder.backbone)

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

        # --- Dataset ---
        # Training YAML's `data:` section lives under cfg.training.data
        # (Hydra composes training configs under the `training` group).
        data_cfg = tcfg.data
        objective_weights = tcfg.get("objectives", None)
        seed = self.cfg.get("seed", 42)
        self.compression_dataset = CompressionDataset.from_config(
            data_cfg, self.tokenizer, seed=seed,
            objective_weights=objective_weights,
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
        self.train_dataset, self.eval_dataset = random_split(
            self.compression_dataset, [train_size, eval_size],
        )

        # --- Sampler + DataLoader ---
        # IMPORTANT: Samplers must emit indices scoped to the *Subset*, not the
        # full dataset.  Subset[i] maps to compression_dataset[subset.indices[i]],
        # so lengths must be gathered for the Subset's own index space.
        max_batch_tokens = tcfg.get("max_batch_tokens", 65536)
        # Eval has no backward — its packed budget can be larger than
        # training's. Falls back to ``max_batch_tokens`` when unset.
        max_batch_tokens_eval = tcfg.get("max_batch_tokens_eval", max_batch_tokens)
        num_workers = self.cfg.compute.get("num_workers", 4)
        pin_memory = self.cfg.compute.get("pin_memory", False)

        import numpy as np

        train_lengths = np.array([
            self.compression_dataset.token_length(i)
            for i in self.train_dataset.indices
        ], dtype=np.int64)
        eval_lengths = np.array([
            self.compression_dataset.token_length(i)
            for i in self.eval_dataset.indices
        ], dtype=np.int64)

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
        self._l1_introduction_step = tcfg.get("l1_introduction_step", 20000)

        # Ratio targeting — top-level config keys
        self._target_ratio_start = tcfg.get("target_ratio_start", 0.30)
        self._target_ratio_end = tcfg.get("target_ratio_end", 0.15)
        self._target_ratio_ramp_steps = tcfg.get("target_ratio_ramp_steps", 100000)
        self._target_ratio_override: float | None = None

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

    def _resolve_step1_checkpoint(self) -> str | None:
        """Resolve step1_checkpoint: auto -> best phase1_step5 checkpoint.

        Step 4 expects a checkpoint from step 3 (commit encoding), which has
        both a trained encoder and decoder.
        """
        step1_checkpoint = self.cfg.get("step1_checkpoint", None)

        # Initialize lineage tracking
        self._input_sources = {}

        if step1_checkpoint == "auto":
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            resolved = resolve_checkpoint(
                checkpoint_dir,
                phase="phase1_step5",
                metric="eval/loss",
                label="step1_checkpoint",
            )
            step1_checkpoint = str(resolved)

        if step1_checkpoint is not None:
            self._input_sources["step1"] = Path(step1_checkpoint).name

        return step1_checkpoint

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

    def _current_target_ratio(self) -> float:
        """Compute the current target compression ratio from the ramp or override."""
        if self._target_ratio_override is not None:
            return self._target_ratio_override
        step = self.global_step
        if step >= self._target_ratio_ramp_steps:
            return self._target_ratio_end
        t = step / max(self._target_ratio_ramp_steps, 1)
        return self._target_ratio_start + t * (self._target_ratio_end - self._target_ratio_start)

    def _perform_l1_rebuild(self) -> None:
        """Rebuild dataset and dataloader for L1 phase."""
        import numpy as np

        logger.info("performing_l1_rebuild", step=self.global_step)

        # Enable L1 on the dataset
        self.compression_dataset.rebuild_for_l1()

        # Recompute subset-scoped lengths and rebuild sampler
        train_lengths = np.array([
            self.compression_dataset.token_length(i)
            for i in self.train_dataset.indices
        ], dtype=np.int64)
        max_batch_tokens = self.cfg.training.get("max_batch_tokens", 65536)
        max_batch_tokens_eval = self.cfg.training.get(
            "max_batch_tokens_eval", max_batch_tokens,
        )
        seed = self.cfg.get("seed", 42)
        self.train_sampler = PackedTokenBudgetSampler(
            self.train_dataset,
            lengths=train_lengths,
            max_batch_tokens=max_batch_tokens,
            shuffle=True,
            seed=seed,
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
        eval_lengths = np.array([
            self.compression_dataset.token_length(i)
            for i in self.eval_dataset.indices
        ], dtype=np.int64)
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
        accumulate(state, enc_out)
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

    def _compress_file_batch(
        self, batch: dict,
    ):
        """Run L0 compression on a packed FileCompressionSample batch.

        The collator gives us flat content embeddings with per-sample
        ``content_cu_seqlens`` and aligned per-sample prompts.
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

        bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()
        content_emb = bgkit_embed(content_ids)
        prompt_emb = bgkit_embed(prompt_ids)

        target_ratio = self._current_target_ratio()
        util_active = getattr(
            self, "_surv_l0", LevelLossCfg(),
        ).utility_grad_loss_weight > 0.0
        return self.encoder(
            content_embeddings=content_emb,
            content_cu_seqlens=content_cu,
            content_position_ids=content_position_ids,
            prompt_embeddings=prompt_emb,
            prompt_cu_seqlens=prompt_cu,
            prompt_position_ids=prompt_position_ids,
            target_ratio=target_ratio,
            level="l0",
            utility_grad_active=util_active,
        )

    def _compress_repo_l0_packed(self, batch: dict):
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

        bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()
        content_emb = bgkit_embed(file_ids)
        prompt_emb = bgkit_embed(prompt_ids)

        target_ratio = self._current_target_ratio()
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
        target_ids_flat = batch["target_token_ids"].to(device)
        target_cu = batch["target_cu_seqlens"].to(device)
        splice_start = batch["bgkit_splice_start"].to(device)
        splice_len = batch["bgkit_splice_len"].to(device)
        loss_mask_flat = batch.get("target_loss_mask")
        if loss_mask_flat is not None:
            loss_mask_flat = loss_mask_flat.to(device).to(torch.bool)

        batch_size = int(target_cu.shape[0]) - 1
        if sample_indices is None:
            sample_indices = list(range(batch_size))
        tok_cu_list = target_cu.to(torch.int64).tolist()
        surv_cu_list = survivor_cu_seqlens.to(torch.int64).tolist()

        prefix_ids: list[torch.Tensor] = []
        suffix_ids: list[torch.Tensor] = []
        per_segment_loss_masks: list[torch.Tensor] = []
        for out_idx, b in enumerate(sample_indices):
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

            k_i = int(surv_cu_list[out_idx + 1]) - int(surv_cu_list[out_idx])
            surv_mask = torch.zeros(k_i, dtype=torch.bool, device=device)
            if loss_mask_flat is not None:
                sample_loss = loss_mask_flat[sample_start:sample_end]
                pre_mask = sample_loss[:splice_b_start]
                suf_mask = sample_loss[splice_b_start + splice_b_len :]
            else:
                # Default: suffix positions bear loss (matches decoder default).
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
        self.encoder.train()
        self.decoder.train()

        # Check L1 introduction
        self._check_l1_introduction()

        # Handle mixed batches
        if batch.get("mixed", False):
            return self._forward_backward_mixed(batch)

        sample_type = batch["sample_type"]

        if sample_type == "file":
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda",
            ):
                enc_out = self._compress_file_batch(batch)
                survivors = enc_out.survivor_embeddings
                survivor_cu = enc_out.survivor_cu_seqlens
                loss = self._decoder_forward_single_splice(survivors, survivor_cu, batch)

            total_loss_t = loss

            # Survivorship auxiliary losses (L0 — file compression path)
            surv_metrics: dict[str, float] = {}
            if enc_out.logits_for_op is not None:
                surv_loss, surv_metrics = self._compute_survivorship_losses(
                    enc_out, self._current_target_ratio(),
                    level="l0",
                    content_token_ids=batch["content_token_ids"].to(self.device),
                    content_cu_seqlens=batch["content_cu_seqlens"].to(self.device),
                )
                total_loss_t = total_loss_t + surv_loss

            (total_loss_t / self._accum_steps).backward()

            # Utility-gradient BCE distillation (post-backward).
            from bgkit.training.survivorship_helpers import LevelLossCfg
            _surv_l0 = getattr(self, "_surv_l0", LevelLossCfg())
            if (
                _surv_l0.utility_grad_loss_weight > 0.0
                and enc_out.post_head_content_values is not None
            ):
                from bgkit.training.survivorship_helpers import utility_grad_bce_loss

                util_loss, util_metrics = utility_grad_bce_loss(
                    base_raw_for_util=enc_out.base_raw_for_util,
                    content_grad=enc_out.get_content_grad(),
                    content_values=enc_out.post_head_content_values,
                    valid_mask=None,
                    pinned_mask=None,
                    target_ratio=self._current_target_ratio(),
                    content_cu_seqlens=batch["content_cu_seqlens"].to(self.device),
                )
                if util_loss.requires_grad:
                    w = _surv_l0.utility_grad_loss_weight
                    (util_loss * w / self._accum_steps).backward()
                for k, v in util_metrics.items():
                    surv_metrics[f"l0_{k}"] = v

            n_survivors = (
                int(enc_out.survivor_mask.sum().item())
                if enc_out.survivor_mask is not None else 0
            )
            n_valid = int(batch["content_token_ids"].shape[0])
            total_loss = loss.item()

            # Drop tensor refs on the file-path enc_out — see
            # ``CompressorOutput.release()``.
            enc_out.release()
        else:
            # Repo batches: packed L0 across all files, then packed L1
            # per repo. See ``_forward_backward_repo_packed``.
            total_loss, n_survivors, n_valid, repo_surv_metrics = (
                self._forward_backward_repo_packed(batch)
            )
            surv_metrics = repo_surv_metrics

        target_ratio = self._current_target_ratio()
        actual_ratio = n_survivors / max(n_valid, 1)
        metrics = {
            "loss": total_loss,
            "sample_type": sample_type,
            "min_target_ratio": target_ratio,
            "actual_ratio": actual_ratio,
            "l1_enabled": float(self._l1_enabled),
        }
        metrics.update(surv_metrics)
        return metrics

    def _forward_backward_repo_packed(
        self,
        batch: dict,
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
            # --- Step 1: packed L0 across all files in all repos ---
            l0_out = self._compress_repo_l0_packed(batch)
            l0_survivors = l0_out.survivor_embeddings  # (N_surv_l0_total, D)
            l0_survivor_cu = l0_out.survivor_cu_seqlens  # (total_files + 1,)

            # Collect L0 survivorship aux loss (per-file content segmentation).
            l0_surv_loss = None
            if l0_out.logits_for_op is not None:
                loss_v, l0_metrics = self._compute_survivorship_losses(
                    l0_out, self._current_target_ratio(),
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

            n_surv_total = int(l1_input_flat.shape[0])
            l1_input_positions = position_ids_from_cu(l1_input_cu, n_surv_total)

            # Build a synthetic "empty prompt" segmentation so the L1
            # encoder runs without a conditioning prompt — the L0 stage
            # already consumed the repo's compression prompt.
            empty_prompt_cu = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=device,
            )
            empty_prompt_emb = l1_input_flat.new_zeros(0, l1_input_flat.shape[-1])
            empty_prompt_pos = torch.zeros(0, dtype=torch.int64, device=device)

            target_ratio = self._current_target_ratio()
            util_w_l1 = getattr(
                self, "_surv_l1", LevelLossCfg(),
            ).utility_grad_loss_weight

            # --- Step 3: packed L1 forward across all repos ---
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
                loss_v, l1_metrics = self._compute_survivorship_losses(
                    l1_out, target_ratio,
                    level="l1",
                    content_token_ids=None,
                    content_cu_seqlens=l1_input_cu,
                )
                l1_surv_loss = loss_v
                surv_metrics.update(l1_metrics)

            # --- Step 4: packed decoder across all repos ---
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

        total_loss = loss.item() * batch_size
        total_survivors = int(l1_out.survivor_embeddings.shape[0])
        total_valid = int(batch["content_token_ids"].shape[0])

        # Utility-gradient BCE — L0 first, then L1.
        _surv_l0 = getattr(self, "_surv_l0", LevelLossCfg())
        if (
            _surv_l0.utility_grad_loss_weight > 0.0
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

        # --- File portion (packed forward + backward) ---
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda",
        ):
            file_enc_out = self._compress_file_batch(file_batch)
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
                scale_override=repo_scale,
            )
        )

        combined_loss = (
            file_loss.item() * n_file_samples
            + avg_repo_loss * n_repo_samples
        ) / total_samples
        file_surv_count = (
            int(file_enc_out.survivor_mask.sum().item())
            if file_enc_out.survivor_mask is not None else 0
        )
        total_survivors = file_surv_count + repo_survivors
        n_valid_file = int(file_batch["content_token_ids"].shape[0])
        actual_ratio = total_survivors / max(n_valid_file + n_valid_repo, 1)

        target_ratio = self._current_target_ratio()
        metrics = {
            "loss": combined_loss,
            "sample_type": "mixed",
            "min_target_ratio": target_ratio,
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

    def _pre_step_hook(self) -> None:
        """Rebuild dataloader when L1 curriculum transition is pending."""
        if self._l1_rebuild_pending:
            self._perform_l1_rebuild()
            self._dataloader_invalidated = True

    def _post_optimizer_step(self, step: int) -> None:
        """Advance bidi warmup; run dual-ascent θ + EMA μ updates per level."""
        self.encoder.step_bidi_warmup()

        from bgkit.training.survivorship_helpers import (
            apply_post_step_updates,
            init_state,
            maybe_unload_ice,
        )

        target_ratio = self._current_target_ratio()
        merged: dict[str, float] = {}
        for level, state_attr in (("l0", "_surv_state_l0"), ("l1", "_surv_state_l1")):
            state = getattr(self, state_attr, None)
            if state is None:
                continue
            update_metrics = apply_post_step_updates(
                self.encoder.compressor, state,
                target_ratio=target_ratio, level=level,
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
        self.encoder.eval()
        self.decoder.eval()
        self._is_evaluating = True
        self._eval_count += 1

        try:
            total_loss = 0.0
            total_tokens = 0.0
            per_objective_loss_sum: dict[str, float] = {}
            per_objective_tokens: dict[str, float] = {}
            mixed_batches = 0

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
        finally:
            self._is_evaluating = False
            self.encoder.train()
            self.decoder.train()

        return metrics

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
            return sliced

        # Repo sample type — slicing requires cu_repo/cu_file surgery.
        # Not used by the current eval path, kept as a plain error to
        # flag misuse.
        raise NotImplementedError(
            "_slice_batch on repo batches is not implemented in the packed path"
        )

    def _evaluate_file_batch(self, batch: dict) -> tuple[float, float]:
        """Token-weighted eval forward for a packed file batch."""
        eval_sub = 4
        n_file = int(batch["content_cu_seqlens"].shape[0]) - 1
        file_loss_sum = 0.0
        file_tokens = 0.0
        for fs in range(0, n_file, eval_sub):
            fe = min(fs + eval_sub, n_file)
            sub_batch = self._slice_batch(batch, list(range(fs, fe)))
            with torch.autocast(
                "cuda", dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                enc_out = self._compress_file_batch(sub_batch)
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
            file_loss_sum += loss.item() * sub_tokens
            file_tokens += sub_tokens
        return file_loss_sum, file_tokens

    def _evaluate_repo_batch_persample(
        self,
        batch: dict,
    ) -> tuple[float, float]:
        """Packed repo-batch eval forward.

        Uses the same single-packed-L0 → per-repo-L1 algorithm as the
        training step. ``persample`` in the name is legacy — we no longer
        loop over samples.
        """
        from bgkit.utils.packing import position_ids_from_cu

        device = self.device
        cu_repo = batch["cu_repo_seqlens"].to(device)
        batch_size = int(cu_repo.shape[0]) - 1

        loss_mask_flat = batch.get("target_loss_mask")
        if loss_mask_flat is not None:
            loss_mask_flat = loss_mask_flat.to(device).to(torch.bool)

        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda",
        ):
            l0_out = self._compress_repo_l0_packed(batch)
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

            l1_out = self.encoder(
                content_embeddings=l1_input_flat,
                content_cu_seqlens=l1_input_cu,
                content_position_ids=l1_input_positions,
                prompt_embeddings=empty_prompt_emb,
                prompt_cu_seqlens=empty_prompt_cu,
                prompt_position_ids=empty_prompt_pos,
                target_ratio=self._current_target_ratio(),
                level="l1",
            )
            loss = self._decoder_forward_single_splice(
                l1_out.survivor_embeddings,
                l1_out.survivor_cu_seqlens,
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
            result = self.encoder.load_state_dict(state_dicts["encoder"], strict=False)
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

