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
from bgkit.data.samplers import LengthSortedBatchSampler
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

        self.train_sampler = LengthSortedBatchSampler(
            self.train_dataset,
            max_batch_tokens,
            shuffle=True,
            seed=seed,
            lengths=train_lengths,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=collate_compression,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        eval_sampler = LengthSortedBatchSampler(
            self.eval_dataset,
            max_batch_tokens,
            shuffle=False,
            lengths=eval_lengths,
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
        self.train_sampler.rebuild(lengths=train_lengths)

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
        max_batch_tokens = self.cfg.training.get("max_batch_tokens", 65536)
        eval_sampler = LengthSortedBatchSampler(
            self.eval_dataset,
            max_batch_tokens,
            shuffle=False,
            lengths=eval_lengths,
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
        content_attn_mask: torch.Tensor | None = None,
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
            content_attn_mask=content_attn_mask,
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
        """Run L0 compression on a batch of FileCompressionSamples.

        Returns:
            CompressionOutput from the encoder.
        """
        content_ids = batch["content_token_ids"].to(self.device)
        content_mask = batch["content_attention_mask"].to(self.device)
        prompt_ids = batch["compression_prompt_ids"].to(self.device)
        prompt_mask = batch["compression_prompt_mask"].to(self.device)

        bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()
        content_emb = bgkit_embed(content_ids)
        prompt_emb = bgkit_embed(prompt_ids)

        target_ratio = self._current_target_ratio()
        from bgkit.training.survivorship_helpers import LevelLossCfg
        util_active = getattr(
            self, "_surv_l0", LevelLossCfg(),
        ).utility_grad_loss_weight > 0.0
        enc_out = self.encoder(
            input_embeddings=content_emb,
            attention_mask=content_mask,
            prompt_embeddings=prompt_emb,
            prompt_attention_mask=prompt_mask,
            target_ratio=target_ratio,
            level="l0",
            utility_grad_active=util_active,
        )

        return enc_out

    @staticmethod
    def _compress_files_batched_pure(
        compressor: torch.nn.Module,
        all_f_emb: torch.Tensor,      # (n_files, max_seq, D)
        all_f_mask: torch.Tensor,      # (n_files, max_seq) bool
        prompt_emb: torch.Tensor,      # (n_files, prompt_len, D)
        prompt_mask: torch.Tensor,     # (n_files, prompt_len) bool
        target_ratio: float,
    ) -> torch.Tensor:
        """Batched L0 compression for checkpointing.

        Returns concatenated survivors ``(total_surv, D)``.
        """
        comp_out = compressor(
            all_f_emb,
            attention_mask=all_f_mask,
            prompt_embeddings=prompt_emb,
            prompt_attention_mask=prompt_mask,
            target_ratio=target_ratio,
            level="l0",
        )
        content_normed = comp_out.normed_embeddings[:, comp_out.content_slice, :]
        survivor_mask = comp_out.survivor_mask
        all_survivors = []
        if survivor_mask is not None:
            for f in range(all_f_emb.shape[0]):
                surv = content_normed[f][survivor_mask[f]]
                all_survivors.append(compressor.auto_reproduce(surv.unsqueeze(0))[0])
        if all_survivors:
            return torch.cat(all_survivors, dim=0)
        return torch.zeros(
            1, compressor.hidden_dim, device=all_f_emb.device, dtype=all_f_emb.dtype,
        )

    def _compress_l0_batched(
        self,
        file_ids: torch.Tensor,
        file_masks: torch.Tensor,
        n_files: int,
        prompt_emb: torch.Tensor,
        prompt_mask: torch.Tensor,
        bgkit_embed: torch.nn.Module,
    ) -> torch.Tensor:
        """Batched L0 per-file compression for one sample -> (total_surv, D).

        Embeds all files at once, then runs a single batched compressor
        forward with the survivorship head producing masks internally.
        Uses selective activation checkpointing based on total token count.
        """
        from torch.utils.checkpoint import checkpoint as torch_checkpoint

        compressor = self.encoder.compressor
        target_ratio = self._current_target_ratio()

        # 1. Embed all files at once
        all_f_emb = bgkit_embed(file_ids[:n_files])          # (n_files, max_seq, D)
        all_f_mask = file_masks[:n_files]                     # (n_files, max_seq)

        # 2. Sort files by token count (descending) to reduce padding waste
        token_counts = all_f_mask.sum(dim=1)
        sorted_indices = token_counts.argsort(descending=True)
        all_f_emb = all_f_emb[sorted_indices]
        all_f_mask = all_f_mask[sorted_indices]

        # 3. Sub-batch files to bound peak memory
        max_files_per_sub = 8
        checkpoint_threshold = 4096
        all_survivors = []

        for start in range(0, n_files, max_files_per_sub):
            end = min(start + max_files_per_sub, n_files)
            sub_emb = all_f_emb[start:end]
            sub_f_mask = all_f_mask[start:end]
            sub_size = end - start
            sub_prompt_emb = prompt_emb.expand(sub_size, -1, -1)
            sub_prompt_mask = prompt_mask.expand(sub_size, -1)

            sub_tokens = sub_f_mask.sum().item()
            if sub_tokens > checkpoint_threshold:
                sub_surv = torch_checkpoint(
                    self._compress_files_batched_pure,
                    compressor, sub_emb, sub_f_mask,
                    sub_prompt_emb, sub_prompt_mask, target_ratio,
                    use_reentrant=False,
                )
            else:
                sub_surv = self._compress_files_batched_pure(
                    compressor, sub_emb, sub_f_mask,
                    sub_prompt_emb, sub_prompt_mask, target_ratio,
                )
            all_survivors.append(sub_surv)

        if all_survivors:
            all_surv = torch.cat(all_survivors, dim=0)
        else:
            return torch.zeros(
                1, compressor.hidden_dim,
                device=self.device, dtype=torch.bfloat16,
            )

        if all_surv.shape[0] == 0:
            return torch.zeros(
                1, compressor.hidden_dim,
                device=self.device, dtype=torch.bfloat16,
            )
        return all_surv

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _decoder_forward_single_splice(
        self,
        survivors: torch.Tensor,
        survivor_mask: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        """Run fused decoder forward + CE loss without materializing full logits.

        All objectives use the same CE loss (they were always identical);
        loss_mask ensures loss is computed only on content tokens.
        """
        target_ids = batch["target_token_ids"].to(self.device)
        target_mask = batch["target_attention_mask"].to(self.device)
        splice_start = batch["bgkit_splice_start"].to(self.device)
        splice_len = batch["bgkit_splice_len"].to(self.device)
        loss_mask = batch.get("target_loss_mask")
        if loss_mask is not None:
            loss_mask = loss_mask.to(self.device)

        return self.decoder.forward_with_single_splice(
            survivor_embeddings=survivors,
            survivor_attention_mask=survivor_mask,
            token_ids=target_ids,
            token_attention_mask=target_mask,
            splice_starts=splice_start,
            splice_lengths=splice_len,
            loss_mask=loss_mask,
        )

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> list:
        return self._trainable_params()

    def _forward_backward(self, batch: dict) -> dict[str, float]:
        """Forward pass + scaled backward. No optimizer ops.

        File batches stay batched (single-file samples are small).
        Repo batches use per-sample L0→L1→decoder→backward for memory efficiency.
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
            # File batches: stay batched (single-file, small)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda",
            ):
                enc_out = self._compress_file_batch(batch)
                survivors = enc_out.survivor_embeddings
                survivor_mask = enc_out.survivor_attention_mask
                loss = self._decoder_forward_single_splice(survivors, survivor_mask, batch)

            total_loss_t = loss

            # Survivorship auxiliary losses (L0 — file compression path)
            surv_metrics: dict[str, float] = {}
            if enc_out.logits_for_op is not None:
                surv_loss, surv_metrics = self._compute_survivorship_losses(
                    enc_out, self._current_target_ratio(),
                    level="l0",
                    content_token_ids=batch["content_token_ids"].to(self.device),
                    content_attn_mask=batch["content_attention_mask"].to(self.device),
                )
                total_loss_t = total_loss_t + surv_loss

            (total_loss_t / self._accum_steps).backward()

            # Utility-gradient BCE distillation (post-backward). Replaces
            # soft-attn: the main backward already routed the decoder-CE
            # gradient back to ``content_hidden`` where the compressor's
            # backward hook captured it.
            from bgkit.training.survivorship_helpers import LevelLossCfg
            _surv_l0 = getattr(self, "_surv_l0", LevelLossCfg())
            if (
                _surv_l0.utility_grad_loss_weight > 0.0
                and enc_out.post_head_content_values is not None
            ):
                from bgkit.training.survivorship_helpers import utility_grad_bce_loss

                content_mask = batch["content_attention_mask"].to(self.device)
                util_loss, util_metrics = utility_grad_bce_loss(
                    base_raw_for_util=enc_out.base_raw_for_util,
                    content_grad=enc_out.get_content_grad(),
                    content_values=enc_out.post_head_content_values,
                    valid_mask=content_mask,
                    pinned_mask=None,
                    target_ratio=self._current_target_ratio(),
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
            n_valid = int(batch["content_attention_mask"].sum().item())
            total_loss = loss.item()

            # Drop tensor refs on the file-path enc_out — see
            # ``CompressorOutput.release()``.
            enc_out.release()
        else:
            # Repo batches: per-sample accumulation. Cleanup of
            # per-sample / per-bundle enc_outs lives inside
            # ``_forward_backward_repo_persample`` (enc_outs are bound
            # to scopes that close naturally before return).
            total_loss, n_survivors, n_valid = self._forward_backward_repo_persample(batch)
            surv_metrics = {}

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

    def _forward_backward_repo_persample(
        self,
        batch: dict,
        scale_override: float | None = None,
        flush_calibrator: bool = True,
    ) -> tuple[float, int, int]:
        """Grouped micro-batch forward + backward for repo batches.

        Two-phase pipeline:
        1. Per-sample L0 compression (batched per-file). Collect all survivors.
        2. Group samples by survivor count, run per-sample L1 scoring +
           full encoder, then batch decoder + backward per group.

        Grouping amortizes backward through the 24-layer decoder across
        multiple samples, replacing batch=1 matmuls with batch=group_size.

        Args:
            batch: Collated repo batch.
            scale_override: If set, use this as the per-sample gradient scale
                instead of ``1 / (batch_size * accum_steps)``. Used by
                ``_forward_backward_mixed`` to weight repo samples relative
                to the combined file+repo sample count.
            flush_calibrator: Whether to flush calibrator scores at the end.
                Set to False when the caller will flush after additional work.

        Returns:
            ``(avg_loss, total_survivors, total_valid_tokens)``
        """
        file_ids = batch["file_token_ids"].to(self.device)
        file_masks = batch["file_attention_masks"].to(self.device)
        file_count = batch["file_count"]
        prompt_ids = batch["compression_prompt_ids"].to(self.device)
        prompt_mask = batch["compression_prompt_mask"].to(self.device)
        target_ids = batch["target_token_ids"].to(self.device)
        target_mask = batch["target_attention_mask"].to(self.device)
        splice_start_batch = batch["bgkit_splice_start"].to(self.device)
        splice_len_batch = batch["bgkit_splice_len"].to(self.device)
        loss_mask_batch = batch.get("target_loss_mask")
        if loss_mask_batch is not None:
            loss_mask_batch = loss_mask_batch.to(self.device)

        batch_size = file_ids.size(0)
        bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()
        hidden_dim = self.encoder.compressor.hidden_dim
        group_size = 4  # micro-batch size for decoder
        scale = (
            scale_override
            if scale_override is not None
            else 1.0 / (batch_size * self._accum_steps)
        )

        total_loss = 0.0
        total_survivors = 0
        total_valid = 0

        # Profiling accumulators
        if self._profile_enabled:
            prof_l0 = 0.0
            prof_l1 = 0.0
            prof_dec = 0.0  # decoder + backward combined

        # --- Phase 1: L0 per-file compression per sample ---
        # Collect survivors; L1 + decoder deferred to Phase 2.
        sample_data = []
        for b in range(batch_size):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                n_files = int(file_count[b].item())
                prompt_emb_b = bgkit_embed(prompt_ids[b:b + 1])

                if self._profile_enabled:
                    ev_l0_start = torch.cuda.Event(enable_timing=True)
                    ev_l0_end = torch.cuda.Event(enable_timing=True)
                    ev_l0_start.record()

                l0_surv = self._compress_l0_batched(
                    file_ids[b], file_masks[b], n_files,
                    prompt_emb_b, prompt_mask[b:b + 1],
                    bgkit_embed,
                )  # (total_surv, D)

                if self._profile_enabled:
                    ev_l0_end.record()
                    torch.cuda.synchronize()
                    prof_l0 += ev_l0_start.elapsed_time(ev_l0_end)

                total_valid += int(file_masks[b].sum().item())
                sample_data.append({
                    'survivors': l0_surv,
                    'idx': b,
                    'prompt_emb': prompt_emb_b,
                })

        # --- Phase 2: Group L1 + decoder + backward ---
        # Sort by survivor count for efficient padding within groups
        sample_data.sort(key=lambda x: x['survivors'].size(0))

        for g_start in range(0, batch_size, group_size):
            g_end = min(g_start + group_size, batch_size)
            group = sample_data[g_start:g_end]
            g_size = len(group)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                if self._profile_enabled:
                    ev_l1_start = torch.cuda.Event(enable_timing=True)
                    ev_l1_end = torch.cuda.Event(enable_timing=True)
                    ev_l1_start.record()

                # Per-sample L1 scoring via survivorship head + full encoder pass.
                group_survivors = []
                group_surv_masks = []
                l1_util_bundles: list[dict] = []
                from bgkit.training.survivorship_helpers import LevelLossCfg
                util_w_l1 = getattr(
                    self, "_surv_l1", LevelLossCfg(),
                ).utility_grad_loss_weight
                target_ratio = self._current_target_ratio()
                for d in group:
                    surv_input = d['survivors']
                    l1_input = surv_input.unsqueeze(0)
                    l1_mask = torch.ones(
                        1, surv_input.size(0), dtype=torch.bool,
                        device=self.device,
                    )
                    l1_out = self.encoder(
                        input_embeddings=l1_input,
                        attention_mask=l1_mask,
                        prompt_embeddings=d['prompt_emb'],
                        prompt_attention_mask=prompt_mask[d['idx']:d['idx'] + 1],
                        target_ratio=target_ratio,
                        level="l1",
                        utility_grad_active=util_w_l1 > 0.0,
                    )
                    group_survivors.append(l1_out.survivor_embeddings)
                    group_surv_masks.append(l1_out.survivor_attention_mask)
                    if (
                        util_w_l1 > 0.0
                        and l1_out.post_head_content_values is not None
                    ):
                        l1_util_bundles.append({
                            "enc_out": l1_out,
                            "content_values": l1_out.post_head_content_values,
                            "content_mask": l1_mask,
                        })

                if self._profile_enabled:
                    ev_l1_end.record()
                    ev_dec_start = torch.cuda.Event(enable_timing=True)
                    ev_dec_end = torch.cuda.Event(enable_timing=True)
                    ev_dec_start.record()

                # Decide: batch decoder or per-sample fallback.
                # Padding short survivors to match long ones wastes O(n^2)
                # attention compute. Fall back to per-sample when padding
                # ratio is too high or group has only 1 sample.
                surv_counts = [s.size(1) for s in group_survivors]
                min_sc = min(surv_counts)
                max_sc = max(surv_counts)
                batch_decoder = g_size > 1 and (min_sc == 0 or max_sc <= 2 * min_sc)

                if batch_decoder:
                    # Pad survivors to max in group for batched decoder
                    dec_surv = torch.zeros(
                        g_size, max_sc, hidden_dim,
                        device=self.device, dtype=torch.bfloat16,
                    )
                    dec_mask = torch.zeros(
                        g_size, max_sc, dtype=torch.bool, device=self.device,
                    )
                    for k, (s, m) in enumerate(
                        zip(group_survivors, group_surv_masks, strict=True),
                    ):
                        n = s.size(1)
                        dec_surv[k, :n] = s[0]
                        dec_mask[k, :n] = m[0]

                    group_indices = [d['idx'] for d in group]
                    group_target_ids = target_ids[group_indices]
                    group_target_mask = target_mask[group_indices]
                    group_loss_mask = None
                    if loss_mask_batch is not None:
                        group_loss_mask = loss_mask_batch[group_indices]

                    loss = self.decoder.forward_with_single_splice(
                        survivor_embeddings=dec_surv,
                        survivor_attention_mask=dec_mask,
                        token_ids=group_target_ids,
                        token_attention_mask=group_target_mask,
                        splice_starts=splice_start_batch[group_indices],
                        splice_lengths=splice_len_batch[group_indices],
                        loss_mask=group_loss_mask,
                    )

                    (loss * g_size * scale).backward()

                    total_loss += loss.item() * g_size
                    total_survivors += int(dec_mask.sum().item())
                else:
                    # Per-sample fallback: high padding ratio
                    for k, d in enumerate(group):
                        s_loss_mask = (
                            loss_mask_batch[d['idx']:d['idx'] + 1]
                            if loss_mask_batch is not None else None
                        )
                        loss = self.decoder.forward_with_single_splice(
                            survivor_embeddings=group_survivors[k],
                            survivor_attention_mask=group_surv_masks[k],
                            token_ids=target_ids[d['idx']:d['idx'] + 1],
                            token_attention_mask=target_mask[d['idx']:d['idx'] + 1],
                            splice_starts=splice_start_batch[d['idx']:d['idx'] + 1],
                            splice_lengths=splice_len_batch[d['idx']:d['idx'] + 1],
                            loss_mask=s_loss_mask,
                        )
                        (loss * scale).backward()
                        total_loss += loss.item()
                        total_survivors += int(group_surv_masks[k].sum().item())

                if self._profile_enabled:
                    ev_dec_end.record()
                    torch.cuda.synchronize()
                    prof_l1 += ev_l1_start.elapsed_time(ev_l1_end)
                    # decoder+backward combined (interleaved in grouped path)
                    prof_dec += ev_dec_start.elapsed_time(ev_dec_end)

            # --- L1 utility-gradient BCE distillation (post-group-backward) ---
            if util_w_l1 > 0.0 and l1_util_bundles:
                from bgkit.training.survivorship_helpers import utility_grad_bce_loss

                for bundle in l1_util_bundles:
                    enc_out = bundle["enc_out"]
                    content_grad = enc_out.get_content_grad()
                    if content_grad is None:
                        continue
                    util_loss, _ = utility_grad_bce_loss(
                        base_raw_for_util=enc_out.base_raw_for_util,
                        content_grad=content_grad,
                        content_values=bundle["content_values"],
                        valid_mask=bundle["content_mask"],
                        pinned_mask=None,
                        target_ratio=target_ratio,
                    )
                    if util_loss.requires_grad:
                        (util_loss * util_w_l1 * scale).backward()

            # Drop every cached enc_out's tensor refs now that utility-
            # grad has consumed them (see ``CompressorOutput.release()``).
            for bundle in l1_util_bundles:
                enc_out = bundle.get("enc_out")
                if enc_out is not None and hasattr(enc_out, "release"):
                    enc_out.release()

        # (flush_calibrator parameter retained for API compat but no-op)

        if self._profile_enabled:
            prof_total = prof_l0 + prof_l1 + prof_dec
            logger.info(
                "[PROFILE]",
                step=self.global_step,
                l0=f"{prof_l0:.0f}ms",
                l1=f"{prof_l1:.0f}ms",
                dec_bwd=f"{prof_dec:.0f}ms",
                total=f"{prof_total:.0f}ms",
                batch_size=batch_size,
            )

        return total_loss / batch_size, total_survivors, total_valid

    def _forward_backward_mixed(self, batch: dict) -> dict[str, float]:
        """Handle mixed batch with sample-count-weighted scaling.

        File portion stays batched (token-weighted within the file sub-batch).
        Repo portion uses per-sample loop (sample-weighted). Each portion is
        scaled by its sample count relative to the combined total so that
        file and repo samples contribute roughly equally. The file portion is
        only approximately sample-weighted — exact equivalence requires all
        file targets to have the same token count (typical for single-file
        samples packed by max_batch_tokens).
        """
        file_batch = batch["file_batch"]
        repo_batch = batch["repo_batch"]

        n_file_samples = file_batch["target_token_ids"].size(0)
        n_repo_samples = repo_batch["file_token_ids"].size(0)
        total_samples = n_file_samples + n_repo_samples

        # --- File portion: batched forward, one backward ---
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda",
        ):
            file_enc_out = self._compress_file_batch(file_batch)
            file_survivors = file_enc_out.survivor_embeddings
            file_mask = file_enc_out.survivor_attention_mask
            file_loss = self._decoder_forward_single_splice(
                file_survivors, file_mask, file_batch,
            )
        file_scale = n_file_samples / (total_samples * self._accum_steps)
        (file_loss * file_scale).backward()

        # --- Repo portion: per-sample loop (reuse shared helper) ---
        repo_scale = 1.0 / (total_samples * self._accum_steps)
        avg_repo_loss, repo_survivors, n_valid_repo = (
            self._forward_backward_repo_persample(
                repo_batch,
                scale_override=repo_scale,
                flush_calibrator=False,
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
        n_valid_file = int(file_batch["content_attention_mask"].sum().item())
        actual_ratio = total_survivors / max(n_valid_file + n_valid_repo, 1)

        target_ratio = self._current_target_ratio()
        metrics = {
            "loss": combined_loss,
            "sample_type": "mixed",
            "min_target_ratio": target_ratio,
            "actual_ratio": actual_ratio,
            "l1_enabled": float(self._l1_enabled),
        }
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
            return int(batch["content_token_ids"].size(0))
        return int(batch["file_token_ids"].size(0))

    def _slice_batch(self, batch: dict, indices: list[int]) -> dict:
        """Slice a collated batch down to a subset of sample indices."""
        sliced: dict = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                sliced[key] = value[indices]
            elif isinstance(value, list):
                sliced[key] = [value[i] for i in indices]
            else:
                sliced[key] = value
        return sliced

    def _evaluate_file_batch(self, batch: dict) -> tuple[float, float]:
        """Token-weighted eval forward for a file batch."""
        eval_sub = 4
        n_file = batch["content_token_ids"].size(0)
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
                    enc_out.survivor_embeddings, enc_out.survivor_attention_mask,
                    sub_batch,
                )
            eval_loss_mask = sub_batch.get("target_loss_mask")
            if eval_loss_mask is not None:
                sub_tokens = eval_loss_mask.sum().item()
            else:
                sub_tokens = sub_batch["target_attention_mask"].sum().item()
            file_loss_sum += loss.item() * sub_tokens
            file_tokens += sub_tokens
        return file_loss_sum, file_tokens

    def _evaluate_repo_batch_persample(
        self,
        batch: dict,
    ) -> tuple[float, float]:
        """Per-sample eval forward for a repo batch.

        Processes one sample at a time to minimize peak memory — no grouping,
        no Phase 1 collection. Eval has no backward so per-sample processing
        is cheap and avoids holding multiple samples' activations.

        Returns (weighted_loss_sum, token_count).
        """
        file_ids = batch["file_token_ids"].to(self.device)
        file_masks = batch["file_attention_masks"].to(self.device)
        file_count = batch["file_count"]
        prompt_ids = batch["compression_prompt_ids"].to(self.device)
        prompt_mask = batch["compression_prompt_mask"].to(self.device)
        target_ids = batch["target_token_ids"].to(self.device)
        target_mask = batch["target_attention_mask"].to(self.device)
        splice_start_batch = batch["bgkit_splice_start"].to(self.device)
        splice_len_batch = batch["bgkit_splice_len"].to(self.device)
        loss_mask_batch = batch.get("target_loss_mask")
        if loss_mask_batch is not None:
            loss_mask_batch = loss_mask_batch.to(self.device)

        batch_size = file_ids.size(0)
        bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()

        batch_loss_sum = 0.0
        batch_token_count = 0.0

        for b in range(batch_size):
          with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda",
          ):
            n_files = int(file_count[b].item())
            prompt_emb_b = bgkit_embed(prompt_ids[b:b + 1])

            # L0 per-file compression
            surv = self._compress_l0_batched(
                file_ids[b], file_masks[b], n_files,
                prompt_emb_b, prompt_mask[b:b + 1],
                bgkit_embed,
            )

            # L1 compression via survivorship head + full encoder
            l1_input = surv.unsqueeze(0)
            l1_mask = torch.ones(
                1, surv.size(0), dtype=torch.bool, device=self.device,
            )
            target_ratio = self._current_target_ratio()
            l1_out = self.encoder(
                input_embeddings=l1_input,
                attention_mask=l1_mask,
                prompt_embeddings=prompt_emb_b,
                prompt_attention_mask=prompt_mask[b:b + 1],
                target_ratio=target_ratio,
                level="l1",
            )
            sample_survivors = l1_out.survivor_embeddings
            sample_surv_mask = l1_out.survivor_attention_mask

            sample_loss_mask = (
                loss_mask_batch[b:b + 1] if loss_mask_batch is not None else None
            )

            loss = self.decoder.forward_with_single_splice(
                survivor_embeddings=sample_survivors,
                survivor_attention_mask=sample_surv_mask,
                token_ids=target_ids[b:b + 1],
                token_attention_mask=target_mask[b:b + 1],
                splice_starts=splice_start_batch[b:b + 1],
                splice_lengths=splice_len_batch[b:b + 1],
                loss_mask=sample_loss_mask,
            )

            eval_loss_mask = batch.get("target_loss_mask")
            if eval_loss_mask is not None:
                sample_tokens = eval_loss_mask[b].sum().item()
            else:
                sample_tokens = batch["target_attention_mask"][b].sum().item()
            batch_loss_sum += loss.item() * sample_tokens
            batch_token_count += sample_tokens

        return batch_loss_sum, batch_token_count

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

    # ------------------------------------------------------------------
    # Live config
    # ------------------------------------------------------------------

    def apply_live_config(self, changes: dict) -> None:
        """Support live tuning of target compression ratio + base fields."""
        if "target_ratio" in changes:
            val = changes["target_ratio"]
            if val is None:
                self._target_ratio_override = None
                logger.info("live_target_ratio_cleared", resuming="ramp")
            elif isinstance(val, (int, float)) and 0 < val < 1:
                self._target_ratio_override = float(val)
                logger.info("live_target_ratio_update", target_ratio=self._target_ratio_override)
        super().apply_live_config(changes)
