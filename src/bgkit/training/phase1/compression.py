"""Phase 1, Step 2: Compression training (4 objectives, curriculum).

Introduces the drop-flag mechanism with four core objectives:
1. Data reconstruction (primary, ~40%)
2. Description generation (~20%)
3. Structural/relational reconstruction (~15%)
4. Commit reproduction (~25%)

Curriculum: L0 objectives first, then L1 once L0 stabilizes.
Survivor selection: deterministic top-k by live ICE scoring with threshold.

The BgKIT encoder and decoder are both trainable (key change from Step 1
where BgKIT was frozen). ICE model remains frozen. Decoder drift monitoring
triggers LoRA fallback if embeddings deviate too far from the token manifold.
"""

from __future__ import annotations

from pathlib import Path

import structlog
import torch
from torch.utils.data import DataLoader, random_split

from bgkit.data.collators import collate_compression
from bgkit.data.datasets.compression_dataset import CompressionDataset
from bgkit.data.samplers import LengthSortedBatchSampler
from bgkit.data.survivor_selection import fill_survivor_gaps, select_survivors_by_threshold
from bgkit.data.threshold_calibrator import ThresholdCalibrator
from bgkit.eval.metrics.embedding_health import embedding_drift_metrics
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.training.base_trainer import BaseTrainer, _average_metrics
from bgkit.training.checkpoint_manager import CheckpointManager
from bgkit.training.checkpoint_registry import (
    CheckpointRegistry,
    resolve_checkpoint,
    resolve_latest_checkpoint,
)
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.training.gradient_utils import clip_grad_norm, enable_gradient_checkpointing
from bgkit.training.interruption import GracefulInterruptor
from bgkit.training.live_config import LiveConfig
from bgkit.training.objectives.commit_reproduction import commit_reproduction_loss
from bgkit.training.objectives.data_reconstruction import data_reconstruction_loss
from bgkit.training.objectives.description_generation import description_generation_loss
from bgkit.training.objectives.structural_relational import structural_relational_loss
from bgkit.training.scheduling import cosine_with_warmup

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

    LIVE_CONFIG_FIELDS = {
        "max_survivor_gap": "_max_gap",
        "drift_threshold": "_drift_threshold",
    }

    def setup(self) -> None:
        """Load trainable encoder/decoder, frozen ICE, create dataset and optimizer."""
        tcfg = self.cfg.training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # --- BgKIT encoder (trainable in Step 2) ---
        bgkit_cfg = self.cfg.model.bgkit
        backbone_name = bgkit_cfg.backbone_name
        backbone_revision = bgkit_cfg.get("backbone_revision", None)
        hidden_dim = bgkit_cfg.get("hidden_dim", 1024)
        logger.info("loading_bgkit_encoder", model=backbone_name, revision=backbone_revision)
        bidi_warmup = self.cfg.training.get("bidi_warmup_steps", 1000)
        self.encoder = BgKITEncoder.from_pretrained(
            backbone_name,
            hidden_dim=hidden_dim,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=backbone_revision,
            attn_implementation="sdpa",
            bidi_warmup_steps=bidi_warmup,
        )
        self.encoder.to(device)

        # Load encoder+decoder from Step 1 checkpoint
        step1_checkpoint = self._resolve_step1_checkpoint()
        step1_state_dicts: dict | None = None
        if step1_checkpoint is not None:
            logger.info("loading_step1_checkpoint", path=step1_checkpoint)
            _, step1_state_dicts = load_checkpoint(Path(step1_checkpoint))
            if "encoder" in step1_state_dicts:
                self.encoder.load_state_dict(step1_state_dicts["encoder"])

        # Encoder is trainable in Step 2
        self.encoder.requires_grad_(True)
        self.encoder.train()
        enable_gradient_checkpointing(self.encoder.compressor.backbone)

        # --- Decoder (trainable, with drift monitoring) ---
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
            attn_implementation="sdpa",
        )
        self.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=hidden_dim)
        self.decoder.to(device)

        # Load decoder from Step 1 checkpoint
        if step1_state_dicts is not None and "decoder" in step1_state_dicts:
            self.decoder.load_state_dict(step1_state_dicts["decoder"])

        enable_gradient_checkpointing(self.decoder.backbone)

        # torch.compile for speedup
        try:
            self.decoder.backbone = torch.compile(self.decoder.backbone)
            logger.info("torch_compile_decoder_enabled")
        except Exception:
            logger.warning("torch_compile_decoder_failed", exc_info=True)

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

        # --- Frozen ICE model ---
        from bgkit.models.ice import ICE

        ice_cfg = tcfg.ice
        if not ice_cfg.get("checkpoint_path"):
            raise ValueError(
                "training.ice.checkpoint_path must be set to a trained ICE checkpoint "
                "or 'auto' for automatic resolution"
            )

        # Auto-resolution: find best ICE checkpoint from registry
        if ice_cfg.checkpoint_path == "auto":
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            ice_ckpt_path = resolve_checkpoint(
                checkpoint_dir,
                phase="ice",
                metric="eval/mse",
                label="training.ice.checkpoint_path",
            )
            self._input_sources["ice"] = ice_ckpt_path.name
            logger.info("ice_auto_resolved", checkpoint=ice_ckpt_path.name)
        else:
            ice_ckpt_path = Path(ice_cfg.checkpoint_path)
            self._input_sources["ice"] = ice_ckpt_path.name

        if not ice_ckpt_path.exists():
            raise FileNotFoundError(
                f"ICE checkpoint not found: {ice_ckpt_path}\n"
                "Set training.ice.checkpoint_path to a valid ICE checkpoint."
            )
        self.ice_model = ICE(
            input_dim=ice_cfg.get("input_dim", 1024),
            hidden_dim=ice_cfg.get("hidden_dim", 128),
            num_layers=ice_cfg.get("num_layers", 2),
            kernel_size=ice_cfg.get("kernel_size", 5),
        )
        _, ice_state_dicts = load_checkpoint(ice_ckpt_path)
        self.ice_model.load_state_dict(ice_state_dicts["model"])
        self.ice_model.to(device)
        self.ice_model.eval()
        self.ice_model.requires_grad_(False)
        logger.info("loaded_ice_model", path=str(ice_ckpt_path))

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
        curriculum = tcfg.get("curriculum", {})
        self._l1_introduction_step = curriculum.get("l1_introduction_step", 20000)

        # Ratio targeting curriculum (replaces threshold ramp + min/max clamps)
        self._target_ratio_start = curriculum.get("target_ratio_start", 0.30)
        self._target_ratio_end = curriculum.get("target_ratio_end", 0.15)
        self._target_ratio_ramp_steps = curriculum.get("target_ratio_ramp_steps", 100000)
        self._target_ratio_override: float | None = None
        self._max_gap = curriculum.get("max_survivor_gap", 64)

        # Calibrators
        fallback = curriculum.get("fallback_threshold", 3.0)
        self._l0_calibrator = ThresholdCalibrator(
            ema_decay=curriculum.get("calibrator_ema_decay", 0.99),
            warmup_batches=curriculum.get("calibrator_warmup_batches", 50),
            fallback_threshold=fallback,
        )
        self._l1_calibrator = ThresholdCalibrator(
            ema_decay=0.95,  # starts fast
            warmup_batches=curriculum.get("calibrator_warmup_batches", 50),
            fallback_threshold=fallback,
        )
        self._l1_calibrator_fast_batches = curriculum.get("l1_calibrator_fast_batches", 200)
        self._l1_calibrator_slow_decay = curriculum.get("calibrator_ema_decay", 0.99)
        self._l1_batches_seen = 0

        # Score buffers for deferred calibrator update (once per train_step)
        self._pending_l0_scores: list[torch.Tensor] = []
        self._pending_l1_scores: list[torch.Tensor] = []

        # Live curriculum values
        self._l1_enabled = False
        self._l1_transitioned = False
        self._l1_rebuild_pending = False

        # Eval isolation flag
        self._is_evaluating = False

        # --- Decoder drift monitoring ---
        decoder_cfg_section = tcfg.get("decoder", {})
        self._drift_threshold = decoder_cfg_section.get("drift_threshold", 0.5)
        self._drift_check_every = decoder_cfg_section.get("drift_check_every", 5000)
        self._lora_active = False

        # --- Metric tracking ---
        self._eval_count = 0

        logger.info(
            "compression_trainer_setup",
            train_samples=train_size,
            eval_samples=eval_size,
            device=str(device),
            l1_introduction_step=self._l1_introduction_step,
        )

    def _resolve_step1_checkpoint(self) -> str | None:
        """Resolve step1_checkpoint config: 'auto' -> best phase1_step1, explicit -> pass.

        Also populates self._input_sources["step1"] for lineage tracking.
        """
        step1_checkpoint = self.cfg.get("step1_checkpoint", None)
        if step1_checkpoint == "auto":
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            resolved = resolve_checkpoint(
                checkpoint_dir,
                phase="phase1_step1",
                metric="eval/loss",
                label="step1_checkpoint",
            )
            step1_checkpoint = str(resolved)

        # Track lineage for BOTH auto-resolved and explicit paths
        self._input_sources = {}
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
        causal_lm = (
            backbone.base_model.model if hasattr(backbone, "base_model") else backbone
        )
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

        for module in self.encoder.modules():
            if isinstance(module, BidirectionalQwen35):
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

    def _flush_calibrator_scores(self) -> None:
        """Flush buffered ICE scores into calibrators. Called once per train_step."""
        if self._pending_l0_scores:
            non_empty = [s for s in self._pending_l0_scores if s.numel() > 0]
            if non_empty:
                combined = torch.cat(non_empty)
                self._l0_calibrator.update_from_flat(combined)
            self._pending_l0_scores.clear()
        if self._pending_l1_scores:
            non_empty = [s for s in self._pending_l1_scores if s.numel() > 0]
            if non_empty:
                combined = torch.cat(non_empty)
                self._l1_calibrator.update_from_flat(combined)
                # Adaptive L1 decay — only count steps with actual score data
                self._l1_batches_seen += 1
                if self._l1_batches_seen >= self._l1_calibrator_fast_batches:
                    self._l1_calibrator.set_decay(self._l1_calibrator_slow_decay)
            self._pending_l1_scores.clear()

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
    # ICE scoring
    # ------------------------------------------------------------------

    def _score_and_select(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        level: str = "l0",
    ) -> torch.Tensor:
        """Score embeddings with ICE and select survivors via calibrated threshold.

        Uses the appropriate calibrator (L0 or L1) to convert the current
        target compression ratio into a threshold. Applies gap-filling to
        prevent long stretches without survivors.

        Args:
            embeddings: (B, L, D) embeddings to score.
            attention_mask: (B, L) bool mask for valid positions.
            level: ``"l0"`` or ``"l1"`` — selects which calibrator to use.

        Returns:
            (B, L) bool survivor mask.
        """
        batch_size, seq_len, _ = embeddings.shape

        with torch.no_grad():
            ice_scores = self.ice_model(embeddings.float())  # (B, L)
        ice_scores = ice_scores.masked_fill(~attention_mask, float("-inf"))
        if seq_len > 0:
            ice_scores[:, 0] = float("-inf")  # position 0 unsupervised during ICE training

        calibrator = self._l0_calibrator if level == "l0" else self._l1_calibrator

        # Buffer scores for deferred calibrator update (training only)
        if not self._is_evaluating:
            stats_mask = attention_mask.clone()
            if seq_len > 0:
                stats_mask[:, 0] = False
            valid_scores_flat = ice_scores[stats_mask].detach()
            if level == "l0":
                self._pending_l0_scores.append(valid_scores_flat)
            else:
                self._pending_l1_scores.append(valid_scores_flat)

        target_ratio = self._current_target_ratio()
        threshold = calibrator.get_threshold(target_ratio)

        survivor_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=embeddings.device)
        for b in range(batch_size):
            valid_len = int(attention_mask[b].sum().item())
            if valid_len == 0:
                continue
            valid_scores = ice_scores[b, :valid_len]
            indices = select_survivors_by_threshold(valid_scores, threshold)
            indices = fill_survivor_gaps(indices, valid_scores, self._max_gap, valid_len)
            survivor_mask[b, indices] = True

        # ICE distribution monitoring (exclude position 0 which is forced to -inf)
        eval_every = self.cfg.training.get("eval_every", 0)
        if eval_every > 0 and self.global_step % eval_every == 0:
            stats_mask = attention_mask.clone()
            if seq_len > 0:
                stats_mask[:, 0] = False
            valid_scores = ice_scores[stats_mask]
            selected_scores = ice_scores[survivor_mask]
            if valid_scores.numel() > 1 and selected_scores.numel() > 1:
                logger.info(
                    "ice_survivor_stats",
                    step=self.global_step,
                    level=level,
                    pre_mean=valid_scores.mean().item(),
                    pre_std=valid_scores.std().item(),
                    post_mean=selected_scores.mean().item(),
                    post_std=selected_scores.std().item(),
                    post_min=selected_scores.min().item(),
                    post_max=selected_scores.max().item(),
                )

        return survivor_mask

    # ------------------------------------------------------------------
    # Compression pipeline
    # ------------------------------------------------------------------

    def _compress_file_batch(
        self, batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run L0 compression on a batch of FileCompressionSamples.

        Returns:
            (survivor_embeddings, survivor_attention_mask)
        """
        content_ids = batch["content_token_ids"].to(self.device)
        content_mask = batch["content_attention_mask"].to(self.device)
        prompt_ids = batch["compression_prompt_ids"].to(self.device)
        prompt_mask = batch["compression_prompt_mask"].to(self.device)

        # Get input embeddings
        bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()
        content_emb = bgkit_embed(content_ids)
        prompt_emb = bgkit_embed(prompt_ids)

        # Live ICE scoring for survivor selection
        survivor_mask = self._score_and_select(content_emb, content_mask)

        # Run encoder forward with compression
        enc_out = self.encoder(
            input_embeddings=content_emb,
            survivor_mask=survivor_mask,
            attention_mask=content_mask,
            prompt_embeddings=prompt_emb,
            prompt_attention_mask=prompt_mask,
        )

        return enc_out.survivor_embeddings, enc_out.survivor_attention_mask

    def _compress_single_direct_l1(
        self,
        f_ids: torch.Tensor,
        f_mask: torch.Tensor,
        prompt_emb: torch.Tensor,
        prompt_mask: torch.Tensor,
        bgkit_embed: torch.nn.Module,
    ) -> torch.Tensor:
        """Run direct L1 on a single sample (1, L) -> (n_surv, D).

        Skips L0; embeds file tokens, scores with ICE, runs one encoder pass.
        """
        f_emb = bgkit_embed(f_ids)
        survivor_mask = self._score_and_select(f_emb, f_mask)
        enc_out = self.encoder(
            input_embeddings=f_emb,
            survivor_mask=survivor_mask,
            attention_mask=f_mask,
            prompt_embeddings=prompt_emb,
            prompt_attention_mask=prompt_mask,
        )
        return enc_out.survivor_embeddings[0]  # (n_surv, D)

    def _compress_single_l0_l1(
        self,
        file_ids: torch.Tensor,
        file_masks: torch.Tensor,
        n_files: int,
        prompt_emb: torch.Tensor,
        prompt_mask: torch.Tensor,
        bgkit_embed: torch.nn.Module,
    ) -> torch.Tensor:
        """Run L0 per-file + collect survivors for one sample -> (total_surv, D).

        Returns L0 survivors mapped back to input embedding space via auto_repro.
        """
        sample_survivors = []
        for f in range(n_files):
            f_ids = file_ids[f].unsqueeze(0)
            f_mask = file_masks[f].unsqueeze(0)
            f_emb = bgkit_embed(f_ids)

            s_mask = self._score_and_select(f_emb, f_mask)

            enc_out = self.encoder(
                input_embeddings=f_emb,
                survivor_mask=s_mask,
                attention_mask=f_mask,
                prompt_embeddings=prompt_emb,
                prompt_attention_mask=prompt_mask,
            )

            # Map L0 output back to input embedding space for L1
            all_normed = enc_out.all_embeddings[0]
            surv_mask_out = enc_out.survivor_mask[0]
            survivor_normed = all_normed[surv_mask_out]
            survivor_input_space = self.encoder.auto_reproduce(
                survivor_normed.unsqueeze(0),
            )[0]
            sample_survivors.append(survivor_input_space)

        if sample_survivors:
            return torch.cat(sample_survivors, dim=0)
        return torch.zeros(
            1, self.encoder.compressor.hidden_dim,
            device=self.device, dtype=torch.bfloat16,
        )

    def _compress_repo_batch(
        self, batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run compression on a batch of RepoCompressionSamples.

        Handles mixed batches with two distinct paths:
        - direct_l1 samples: single encoder pass (skip L0), producing final
          decoder-space survivors directly.
        - Regular samples: L0 per-file → auto_repro to input space → L1
          (ICE + encoder), producing decoder-space survivors.

        Final outputs from both paths are padded and combined.

        Returns:
            (survivor_embeddings, survivor_attention_mask)
        """
        file_ids = batch["file_token_ids"].to(self.device)
        file_masks = batch["file_attention_masks"].to(self.device)
        file_count = batch["file_count"]
        prompt_ids = batch["compression_prompt_ids"].to(self.device)
        prompt_mask = batch["compression_prompt_mask"].to(self.device)
        direct_l1_flags = batch["direct_l1"]  # (B,) bool tensor

        batch_size = file_ids.size(0)
        bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()
        prompt_emb = bgkit_embed(prompt_ids)

        # Collect final decoder-space survivors per sample.
        # direct_l1: one encoder pass → done.
        # regular: L0 per-file → auto_repro intermediates → batched L1.
        final_survivors: list[torch.Tensor | None] = [None] * batch_size
        l0_intermediates: dict[int, torch.Tensor] = {}  # b -> input-space surv

        for b in range(batch_size):
            n_files = int(file_count[b].item())

            if direct_l1_flags[b]:
                if n_files != 1:
                    raise ValueError(
                        f"direct_l1 sample {b} has file_count={n_files}, expected 1"
                    )
                final_survivors[b] = self._compress_single_direct_l1(
                    file_ids[b, 0].unsqueeze(0),
                    file_masks[b, 0].unsqueeze(0),
                    prompt_emb[b:b + 1],
                    prompt_mask[b:b + 1],
                    bgkit_embed,
                )
            else:
                l0_intermediates[b] = self._compress_single_l0_l1(
                    file_ids[b], file_masks[b], n_files,
                    prompt_emb[b:b + 1], prompt_mask[b:b + 1],
                    bgkit_embed,
                )

        # Run batched L1 on non-direct_l1 samples only
        if l0_intermediates:
            l1_indices = sorted(l0_intermediates.keys())
            l1_survs = [l0_intermediates[b] for b in l1_indices]
            l1_prompts = torch.stack([prompt_emb[b] for b in l1_indices])
            l1_pmasks = torch.stack([prompt_mask[b] for b in l1_indices])

            max_surv = max(s.size(0) for s in l1_survs)
            hidden_dim = l1_survs[0].size(-1)
            n_l1 = len(l1_indices)
            padded = torch.zeros(
                n_l1, max_surv, hidden_dim,
                device=self.device, dtype=l1_survs[0].dtype,
            )
            pad_mask = torch.zeros(
                n_l1, max_surv, dtype=torch.bool, device=self.device,
            )
            for i, surv in enumerate(l1_survs):
                padded[i, :surv.size(0)] = surv
                pad_mask[i, :surv.size(0)] = True

            l1_survivor_mask = self._score_and_select(padded, pad_mask, level="l1")

            l1_out = self.encoder(
                input_embeddings=padded,
                survivor_mask=l1_survivor_mask,
                attention_mask=pad_mask,
                prompt_embeddings=l1_prompts,
                prompt_attention_mask=l1_pmasks,
            )

            for i, b in enumerate(l1_indices):
                # Extract per-sample survivors (remove batch padding)
                smask = l1_out.survivor_attention_mask[i]
                n_valid = int(smask.sum().item())
                final_survivors[b] = l1_out.survivor_embeddings[i, :n_valid]

        # Pad all final survivors across the full batch
        all_final = [s for s in final_survivors if s is not None]
        assert len(all_final) == batch_size
        max_final = max(s.size(0) for s in all_final)
        hidden_dim = all_final[0].size(-1)
        out_emb = torch.zeros(
            batch_size, max_final, hidden_dim,
            device=self.device, dtype=all_final[0].dtype,
        )
        out_mask = torch.zeros(
            batch_size, max_final, dtype=torch.bool, device=self.device,
        )
        for b, surv in enumerate(all_final):
            out_emb[b, :surv.size(0)] = surv
            out_mask[b, :surv.size(0)] = True

        return out_emb, out_mask

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        logits: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        """Compute loss using the appropriate objective loss function.

        Objective weighting is baked into dataset sizes (uniform average here).
        """
        target_ids = batch["target_token_ids"].to(self.device)
        target_mask = batch["target_attention_mask"].to(self.device)
        loss_mask = batch.get("target_loss_mask")
        if loss_mask is not None:
            loss_mask = loss_mask.to(self.device)

        # Determine the dominant objective in this batch (batches typically homogeneous)
        objectives = batch.get("objectives", [])
        objective = objectives[0] if objectives else "data_reconstruction"

        loss_fn = _LOSS_FNS.get(objective, data_reconstruction_loss)

        # data_reconstruction_loss takes loss_mask; others only take attention_mask
        if objective == "data_reconstruction" and loss_mask is not None:
            loss = loss_fn(logits, target_ids, target_mask, loss_mask=loss_mask)
        else:
            if loss_mask is not None:
                # Combine attention mask with loss mask for non-DR objectives
                combined_mask = target_mask * loss_mask
                loss = loss_fn(logits, target_ids, combined_mask)
            else:
                loss = loss_fn(logits, target_ids, target_mask)

        return loss

    # ------------------------------------------------------------------
    # Decoder drift monitoring
    # ------------------------------------------------------------------

    def _check_decoder_drift(self, survivors: torch.Tensor) -> None:
        """Check for decoder embedding drift and apply LoRA fallback if needed."""
        if self.global_step % self._drift_check_every != 0 or self.global_step == 0:
            return

        token_emb = (
            self.encoder.compressor.backbone.get_input_embeddings().weight.detach()
        )
        metrics = embedding_drift_metrics(survivors.detach(), token_emb)

        logger.info(
            "decoder_drift_check",
            step=self.global_step,
            mean_sim=metrics["mean_max_cosine_sim"],
            threshold=self._drift_threshold,
        )

        if metrics["mean_max_cosine_sim"] < self._drift_threshold and not self._lora_active:
            logger.warning(
                "decoder_drift_detected",
                mean_sim=metrics["mean_max_cosine_sim"],
                threshold=self._drift_threshold,
            )
            self._apply_lora_fallback()

    def _apply_lora_fallback(self, rebuild_optimizer: bool = True) -> None:
        """Switch decoder from full fine-tuning to high-rank LoRA.

        Args:
            rebuild_optimizer: Whether to rebuild the optimizer after applying
                LoRA. Set False during checkpoint restore (optimizer is loaded
                from checkpoint separately).
        """
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=64,
                lora_alpha=128,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            self.decoder.backbone = get_peft_model(self.decoder.backbone, lora_config)
            self._lora_active = True

            # Freeze non-LoRA decoder params
            for name, param in self.decoder.backbone.named_parameters():
                if "lora_" not in name:
                    param.requires_grad_(False)

            # Rebuild optimizer with new params (skip during checkpoint restore)
            if rebuild_optimizer:
                self._setup_optimizer()

            logger.info(
                "lora_fallback_applied",
                step=self.global_step,
                lora_params=sum(
                    p.numel()
                    for p in self.decoder.backbone.parameters()
                    if p.requires_grad
                ),
            )
        except ImportError:
            logger.warning(
                "lora_fallback_skipped",
                reason="peft not installed",
            )

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> list:
        return self._trainable_params()

    def train_step(self, batch: dict) -> dict[str, float]:
        """Override base train_step to add drift check after optimizer step."""
        self._last_survivors = None
        metrics = super().train_step(batch)
        if self._last_survivors is not None:
            self._check_decoder_drift(self._last_survivors)
        return metrics

    def _forward_backward(self, batch: dict) -> dict[str, float]:
        """Forward pass + scaled backward. No optimizer ops.

        Stashes survivors on self._last_survivors for post-step drift check.
        """
        self.encoder.train()
        self.decoder.train()

        # Check L1 introduction + flush calibrator scores
        self._check_l1_introduction()

        # Handle mixed batches
        if batch.get("mixed", False):
            return self._forward_backward_mixed(batch)

        sample_type = batch["sample_type"]

        # Run compression pipeline (returns actual sampled ratio)
        if sample_type == "file":
            survivors, survivor_mask = self._compress_file_batch(batch)
        else:
            survivors, survivor_mask = self._compress_repo_batch(batch)

        # Decoder forward
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            logits = self.decoder(
                survivor_embeddings=survivors,
                target_ids=batch["target_token_ids"].to(self.device),
                target_attention_mask=batch["target_attention_mask"].to(self.device),
                survivor_attention_mask=survivor_mask,
            )

        # Compute loss
        loss = self._compute_loss(logits, batch)

        # Flush calibrator scores (once per micro-batch)
        self._flush_calibrator_scores()

        # Scaled backward (for gradient accumulation)
        (loss / self._accum_steps).backward()

        # Stash survivors for post-step drift check
        self._last_survivors = survivors.detach()

        target_ratio = self._current_target_ratio()
        n_survivors = int(survivor_mask.sum().item())
        if sample_type == "file":
            n_valid = int(batch["content_attention_mask"].sum().item())
        else:
            n_valid = int(batch["file_attention_masks"].sum().item())
        actual_ratio = n_survivors / max(n_valid, 1)
        metrics = {
            "loss": loss.item(),
            "sample_type": sample_type,
            "target_ratio": target_ratio,
            "actual_ratio": actual_ratio,
            "calibrated_threshold_l0": self._l0_calibrator.get_threshold(target_ratio),
            "l1_enabled": float(self._l1_enabled),
        }
        if self._l1_enabled:
            metrics["calibrated_threshold_l1"] = self._l1_calibrator.get_threshold(target_ratio)
        return metrics

    def _forward_backward_mixed(self, batch: dict) -> dict[str, float]:
        """Handle mixed batch: forward + scaled backward, no optimizer ops."""
        file_batch = batch["file_batch"]
        repo_batch = batch["repo_batch"]

        # Process file portion
        file_survivors, file_mask = self._compress_file_batch(file_batch)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            file_logits = self.decoder(
                survivor_embeddings=file_survivors,
                target_ids=file_batch["target_token_ids"].to(self.device),
                target_attention_mask=file_batch["target_attention_mask"].to(self.device),
                survivor_attention_mask=file_mask,
            )
        file_loss = self._compute_loss(file_logits, file_batch)

        # Process repo portion
        repo_survivors, repo_mask = self._compress_repo_batch(repo_batch)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            repo_logits = self.decoder(
                survivor_embeddings=repo_survivors,
                target_ids=repo_batch["target_token_ids"].to(self.device),
                target_attention_mask=repo_batch["target_attention_mask"].to(self.device),
                survivor_attention_mask=repo_mask,
            )
        repo_loss = self._compute_loss(repo_logits, repo_batch)

        # Average losses
        loss = (file_loss + repo_loss) / 2.0

        # Flush calibrator scores (once per micro-batch)
        self._flush_calibrator_scores()

        # Scaled backward (for gradient accumulation)
        (loss / self._accum_steps).backward()

        # Stash survivors for post-step drift check (use repo portion)
        self._last_survivors = repo_survivors.detach()

        target_ratio = self._current_target_ratio()
        total_survivors = int(file_mask.sum().item()) + int(repo_mask.sum().item())
        n_valid_file = int(file_batch["content_attention_mask"].sum().item())
        n_valid_repo = int(repo_batch["file_attention_masks"].sum().item())
        actual_ratio = total_survivors / max(n_valid_file + n_valid_repo, 1)
        metrics = {
            "loss": loss.item(),
            "sample_type": "mixed",
            "target_ratio": target_ratio,
            "actual_ratio": actual_ratio,
            "calibrated_threshold_l0": self._l0_calibrator.get_threshold(target_ratio),
            "l1_enabled": float(self._l1_enabled),
        }
        if self._l1_enabled:
            metrics["calibrated_threshold_l1"] = self._l1_calibrator.get_threshold(target_ratio)
        return metrics

    # ------------------------------------------------------------------
    # Custom train() with pre-fetch L1 rebuild hook
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Custom training loop with pre-fetch hook for L1 dataloader rebuild.

        This overrides BaseTrainer.train() to add a check between fetching
        batches that triggers dataloader rebuild when L1 is introduced.
        """
        self.setup()

        tcfg = self.cfg.training
        if (
            not hasattr(tcfg, "max_steps")
            or not hasattr(tcfg, "lr")
            or not isinstance(tcfg.lr, (int, float))
        ):
            phase = getattr(tcfg, "phase", "<unknown>")
            raise TypeError(
                f"CompressionTrainer.train() requires scalar training.max_steps and "
                f"training.lr, but phase '{phase}' uses a different schema."
            )
        eval_every = tcfg.eval_every
        save_every = tcfg.save_every
        checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))

        # Checkpoint registry
        registry = CheckpointRegistry(checkpoint_dir)

        # Early stopping config
        es_cfg = tcfg.get("early_stopping", {})
        if isinstance(es_cfg, bool):
            es_enabled = es_cfg
            es_cfg = {}
        else:
            es_enabled = es_cfg.get("enabled", False) if es_cfg else False
        es_patience = es_cfg.get("patience", 5) if es_cfg else 5
        es_min_delta = es_cfg.get("min_delta", 0.001) if es_cfg else 0.001
        es_metric = es_cfg.get("metric", "eval/loss") if es_cfg else "eval/loss"
        es_best: float | None = None
        es_evals_without_improvement = 0

        # Resume from checkpoint: explicit path, or auto-resolve latest for this phase
        resume_path = self.cfg.get("resume_checkpoint", None)
        if resume_path is None:
            phase = getattr(tcfg, "phase", None)
            if phase:
                auto_resolved = resolve_latest_checkpoint(checkpoint_dir, phase)
                if auto_resolved is not None:
                    resume_path = str(auto_resolved)
                    logger.info("auto_resume_resolved", checkpoint=resume_path, phase=phase)
        is_resuming = False
        if resume_path is not None:
            self.load_checkpoint(Path(resume_path))
            self.global_step += 1
            is_resuming = True
            if self._training_state is not None:
                es_best = self._training_state.get("es_best")
                es_evals_without_improvement = self._training_state.get(
                    "es_evals_without_improvement", 0,
                )
            logger.info(
                "resuming_training",
                from_step=self.global_step,
                es_best=es_best,
                es_evals_without_improvement=es_evals_without_improvement,
            )

        # LR schedule params
        reset_schedule = tcfg.get("reset_schedule", False)
        if self._schedule_params is not None and not reset_schedule:
            max_steps = int(self._schedule_params["max_steps"])
            base_lr = self._schedule_params["base_lr"]
            warmup_steps = int(self._schedule_params["warmup_steps"])
            if tcfg.max_steps > max_steps:
                max_steps = tcfg.max_steps
        else:
            max_steps = tcfg.max_steps
            base_lr = tcfg.lr
            warmup_steps = tcfg.warmup_steps

        self._schedule_params = {
            "max_steps": max_steps,
            "base_lr": base_lr,
            "warmup_steps": warmup_steps,
        }

        # Wandb
        wandb_run = None
        wandb_run_id = (
            self._training_state.get("wandb_run_id") if self._training_state else None
        )
        if self.cfg.get("wandb", {}).get("enabled", False):
            try:
                import wandb
                from omegaconf import OmegaConf

                wandb_kwargs = dict(
                    project=self.cfg.wandb.get("project", "bgkit"),
                    entity=self.cfg.wandb.get("entity", None),
                    name=self.cfg.get("run_name", None),
                    tags=list(self.cfg.wandb.get("tags", [])),
                    config=OmegaConf.to_container(self.cfg, resolve=True),
                )
                if is_resuming and wandb_run_id is not None:
                    wandb_kwargs["id"] = wandb_run_id
                    wandb_kwargs["resume"] = "must"
                wandb_run = wandb.init(**wandb_kwargs)
            except ImportError:
                logger.warning("wandb_not_installed")

        # Sync epoch
        self._sync_epoch(self.epoch)

        # If resuming and L1 was already transitioned, rebuild dataloader
        if self._l1_transitioned or self._l1_rebuild_pending:
            self._perform_l1_rebuild()

        dataloader_iter = iter(self.train_dataloader)

        accum_steps = self._validate_accum_steps(
            tcfg.get("gradient_accumulation_steps", 1)
        )
        self._accum_steps = accum_steps
        self._last_survivors = None

        logger.info(
            "training_start",
            max_steps=max_steps,
            lr=base_lr,
            start_step=self.global_step,
            gradient_accumulation_steps=accum_steps,
        )

        # Live config
        control_file = self.cfg.get("control_file", None)
        if control_file is None:
            control_file = checkpoint_dir / "control.json"
        live_config = LiveConfig(Path(control_file))

        # Checkpoint pruning
        prune_cfg = tcfg.get("checkpoint_pruning", {})
        prune_enabled = prune_cfg.get("enabled", False) if prune_cfg else False
        if prune_enabled:
            ckpt_manager = CheckpointManager(
                keep_best=prune_cfg.get("keep_best", 3),
                keep_latest=prune_cfg.get("keep_latest", 2),
                metric=prune_cfg.get("metric", es_metric),
                lower_is_better=prune_cfg.get("lower_is_better", True),
                phase=tcfg.phase,
                registry=registry,
            )
            ckpt_manager.load_existing(checkpoint_dir)
        else:
            ckpt_manager = None

        last_eval_metrics: dict[str, float] | None = None
        last_eval_step = -1

        stopped_early = False
        try:
            with GracefulInterruptor() as interruptor:
                step = self.global_step
                while step < max_steps:
                    self.global_step = step

                    # PRE-FETCH HOOK: rebuild dataloader if L1 transition pending
                    if self._l1_rebuild_pending:
                        self._perform_l1_rebuild()
                        dataloader_iter = iter(self.train_dataloader)

                    # LR schedule
                    lr = cosine_with_warmup(step, max_steps, warmup_steps, base_lr)
                    for pg in self.optimizer.param_groups:
                        group_base = pg.get("base_lr", base_lr)
                        pg["lr"] = cosine_with_warmup(
                            step, max_steps, warmup_steps, group_base,
                        )

                    # Accumulation loop
                    self.optimizer.zero_grad()
                    accum_metrics = []
                    for _micro in range(accum_steps):
                        try:
                            batch = next(dataloader_iter)
                        except StopIteration:
                            self.epoch += 1
                            self._sync_epoch(self.epoch)
                            dataloader_iter = iter(self.train_dataloader)
                            batch = next(dataloader_iter)
                        micro_metrics = self._forward_backward(batch)
                        accum_metrics.append(micro_metrics)

                    grad_norm = clip_grad_norm(self.trainable_parameters())
                    self.optimizer.step()
                    self.encoder.step_bidi_warmup()

                    # Drift check AFTER optimizer step (can rebuild optimizer)
                    if self._last_survivors is not None:
                        self._check_decoder_drift(self._last_survivors)

                    metrics = _average_metrics(accum_metrics)
                    metrics["grad_norm"] = grad_norm
                    metrics["lr"] = lr
                    metrics["bidi_alpha"] = self._get_bidi_alpha()
                    if len(self.optimizer.param_groups) > 1:
                        metrics["lr_min"] = min(
                            pg["lr"] for pg in self.optimizer.param_groups
                        )

                    # Log
                    if step % 100 == 0:
                        logger.info("train_step", step=step, **metrics)
                    if wandb_run is not None:
                        wandb_run.log(metrics, step=step)

                    # Eval
                    if eval_every > 0 and step > 0 and step % eval_every == 0:
                        eval_metrics = self.evaluate()
                        eval_metrics = {
                            f"eval/{k}": v for k, v in eval_metrics.items()
                        }
                        logger.info("eval", step=step, **eval_metrics)
                        if wandb_run is not None:
                            wandb_run.log(eval_metrics, step=step)

                        last_eval_metrics = eval_metrics
                        last_eval_step = step

                        # Early stopping
                        if es_enabled:
                            if es_metric not in eval_metrics:
                                raise KeyError(
                                    f"Early stopping metric '{es_metric}' not "
                                    f"found in eval results."
                                )
                            current_val = eval_metrics[es_metric]
                            if es_best is None or current_val < es_best - es_min_delta:
                                es_best = current_val
                                es_evals_without_improvement = 0
                            else:
                                es_evals_without_improvement += 1
                                if es_evals_without_improvement >= es_patience:
                                    logger.info(
                                        "early_stopping",
                                        step=step,
                                        metric=es_metric,
                                        best=es_best,
                                    )
                                    stopped_early = True
                                    break

                    # Live config
                    changes = live_config.poll()
                    if changes:
                        if "lr" in changes:
                            new_lr = changes["lr"]
                            old_base_lr = base_lr
                            if isinstance(new_lr, (int, float)) and new_lr > 0:
                                ratio = new_lr / old_base_lr
                                base_lr = new_lr
                                self._schedule_params["base_lr"] = base_lr
                                for pg in self.optimizer.param_groups:
                                    pg["base_lr"] = pg.get("base_lr", old_base_lr) * ratio
                        if "early_stopping_patience" in changes:
                            val = changes["early_stopping_patience"]
                            if isinstance(val, int) and val > 0:
                                es_patience = val
                                logger.info("live_es_patience_update", patience=val)
                        if "eval_every" in changes:
                            val = changes["eval_every"]
                            if isinstance(val, int) and val > 0:
                                eval_every = val
                                logger.info("live_eval_every_update", eval_every=val)
                        if "save_every" in changes:
                            val = changes["save_every"]
                            if isinstance(val, int) and val > 0:
                                save_every = val
                                logger.info("live_save_every_update", save_every=val)
                        if "max_steps" in changes:
                            val = changes["max_steps"]
                            if isinstance(val, int) and val > step:
                                max_steps = val
                                self._schedule_params["max_steps"] = val
                                logger.info("live_max_steps_update", max_steps=val)
                        self.apply_live_config(changes)

                    # Checkpoint
                    saved_this_step = False
                    if save_every > 0 and step > 0 and step % save_every == 0:
                        self._training_state = self._build_training_state(
                            es_best, es_evals_without_improvement, wandb_run,
                        )
                        step_metrics = (
                            last_eval_metrics if last_eval_step == step else None
                        )
                        parent = self._registry_parent()
                        ckpt_path = self.save_checkpoint(
                            checkpoint_dir, metrics=step_metrics,
                        )
                        registry.register(self._build_registry_entry(
                            ckpt_path, step_metrics, wandb_run,
                            parent_checkpoint=parent,
                        ))
                        if ckpt_manager is not None:
                            ckpt_manager.record(ckpt_path, step, step_metrics)
                            ckpt_manager.prune()
                        (checkpoint_dir / ".last_checkpoint").write_text(str(ckpt_path))
                        saved_this_step = True

                    # Graceful shutdown
                    if interruptor.should_stop:
                        if not saved_this_step:
                            self._training_state = self._build_training_state(
                                es_best, es_evals_without_improvement, wandb_run,
                            )
                            parent = self._registry_parent()
                            ckpt_path = self.save_checkpoint(checkpoint_dir)
                            registry.register(self._build_registry_entry(
                                ckpt_path, None, wandb_run,
                                status="interrupted",
                                parent_checkpoint=parent,
                            ))
                            if ckpt_manager is not None:
                                ckpt_manager.record(ckpt_path, step, None)
                                ckpt_manager.prune()
                            (checkpoint_dir / ".last_checkpoint").write_text(
                                str(ckpt_path)
                            )
                        logger.info(
                            "graceful_shutdown_complete",
                            step=step,
                            signal=interruptor.received_signal.name
                            if interruptor.received_signal
                            else None,
                        )
                        return

                    step += 1

                # Final eval + checkpoint
                eval_metrics = self.evaluate()
                eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                logger.info("final_eval", **eval_metrics)
                self._training_state = self._build_training_state(
                    es_best, es_evals_without_improvement, wandb_run,
                )
                parent = self._registry_parent()
                ckpt_path = self.save_checkpoint(checkpoint_dir, metrics=eval_metrics)
                registry.register(self._build_registry_entry(
                    ckpt_path, eval_metrics, wandb_run,
                    parent_checkpoint=parent,
                ))
                if ckpt_manager is not None:
                    ckpt_manager.record(ckpt_path, self.global_step, eval_metrics)
                    ckpt_manager.prune()
                (checkpoint_dir / ".last_checkpoint").write_text(str(ckpt_path))
        finally:
            if wandb_run is not None:
                wandb_run.finish()

        if stopped_early:
            logger.info("training_complete_early_stop", total_steps=self.global_step)
        else:
            logger.info("training_complete", total_steps=max_steps)

    def _build_training_state(
        self,
        es_best: float | None,
        es_evals_without_improvement: int,
        wandb_run,
    ) -> dict:
        """Build training state dict for checkpointing."""
        return {
            "es_best": es_best,
            "es_evals_without_improvement": es_evals_without_improvement,
            "wandb_run_id": wandb_run.id if wandb_run is not None else None,
            "l1_enabled": self._l1_enabled,
            "l1_transitioned": self._l1_transitioned,
            "l1_rebuild_pending": self._l1_rebuild_pending,
            "lora_active": self._lora_active,
            "l1_batches_seen": self._l1_batches_seen,
            "target_ratio_override": self._target_ratio_override,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation on the eval split."""
        self.encoder.eval()
        self.decoder.eval()
        self._is_evaluating = True
        self._eval_count += 1

        try:
            total_loss = 0.0
            total_tokens = 0.0
            per_objective_loss: dict[str, float] = {}
            per_objective_count: dict[str, int] = {}

            all_survivors: list[torch.Tensor] = []

            num_batches = len(self.eval_dataloader)
            for batch_idx, batch in enumerate(self.eval_dataloader):
                if batch_idx % 100 == 0:
                    logger.info("eval_progress", batch=batch_idx, total=num_batches)

                if batch.get("mixed", False):
                    # Skip mixed batches in eval for simplicity
                    continue

                sample_type = batch["sample_type"]
                if sample_type == "file":
                    survivors, survivor_mask = self._compress_file_batch(batch)
                else:
                    survivors, survivor_mask = self._compress_repo_batch(batch)

                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda",
                ):
                    logits = self.decoder(
                        survivor_embeddings=survivors,
                        target_ids=batch["target_token_ids"].to(self.device),
                        target_attention_mask=batch["target_attention_mask"].to(
                            self.device,
                        ),
                        survivor_attention_mask=survivor_mask,
                    )
                    loss = self._compute_loss(logits, batch)

                target_mask = batch["target_attention_mask"]
                batch_tokens = target_mask[:, 1:].sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

                # Per-objective tracking
                objectives = batch.get("objectives", [])
                if objectives:
                    obj = objectives[0]
                    per_objective_loss[obj] = per_objective_loss.get(obj, 0.0) + loss.item()
                    per_objective_count[obj] = per_objective_count.get(obj, 0) + 1

                # Collect survivors for drift metrics
                total_vecs = sum(s.size(0) for s in all_survivors)
                if total_vecs < 512:
                    flat = survivors[survivor_mask].detach() if survivor_mask is not None else (
                        survivors.reshape(-1, survivors.size(-1)).detach()
                    )
                    remaining = 512 - total_vecs
                    all_survivors.append(flat[:remaining])

            avg_loss = total_loss / max(total_tokens, 1)
            perplexity = torch.exp(torch.tensor(avg_loss)).item()

            metrics: dict[str, float] = {
                "loss": avg_loss,
                "perplexity": perplexity,
            }

            # Per-objective losses
            for obj, total in per_objective_loss.items():
                count = per_objective_count.get(obj, 1)
                metrics[f"{obj}_loss"] = total / count

            # Embedding drift metrics
            if all_survivors:
                combined = torch.cat(all_survivors, dim=0)
                token_emb = (
                    self.encoder.compressor.backbone.get_input_embeddings().weight.detach()
                )
                health = embedding_drift_metrics(combined, token_emb)
                metrics["mean_max_cosine_sim"] = health["mean_max_cosine_sim"]
                metrics["std_max_cosine_sim"] = health["std_max_cosine_sim"]
        finally:
            self._is_evaluating = False
            self.encoder.train()
            self.decoder.train()

        return metrics

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
            "lora_active": self._lora_active,
            "l1_batches_seen": self._l1_batches_seen,
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
        ckpt_path = save_checkpoint(
            checkpoint_dir,
            metadata,
            encoder=self.encoder.state_dict(),
            decoder=self.decoder.state_dict(),
            optimizer=self.optimizer.state_dict(),
            l0_calibrator=self._l0_calibrator.state_dict(),
            l1_calibrator=self._l1_calibrator.state_dict(),
        )
        self._last_checkpoint_path = str(ckpt_path)
        return ckpt_path

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load checkpoint and restore all training state."""
        metadata, state_dicts = load_checkpoint(checkpoint_path)
        self._check_optimizer_type_compat(metadata)

        # Restore step position and curriculum state BEFORE loading weights,
        # so LoRA can be applied to the decoder first if needed.
        self.global_step = metadata.step
        self.epoch = metadata.epoch
        self._last_checkpoint_path = str(checkpoint_path)
        if metadata.schedule_params is not None:
            self._schedule_params = metadata.schedule_params
        if metadata.training_state is not None:
            self._training_state = metadata.training_state

            # Restore curriculum state
            ts = metadata.training_state
            self._l1_enabled = ts.get("l1_enabled", False)
            self._l1_transitioned = ts.get("l1_transitioned", False)
            self._l1_rebuild_pending = ts.get("l1_rebuild_pending", False)
            self._lora_active = ts.get("lora_active", False)
            self._l1_batches_seen = ts.get("l1_batches_seen", 0)
            if self._l1_batches_seen >= self._l1_calibrator_fast_batches:
                self._l1_calibrator.set_decay(self._l1_calibrator_slow_decay)
            self._target_ratio_override = ts.get("target_ratio_override")

        # If LoRA was active when checkpoint was saved, apply LoRA wrapper
        # to the decoder BEFORE loading state dict (which contains LoRA keys).
        if self._lora_active:
            self._apply_lora_fallback(rebuild_optimizer=False)

        # Restore model weights
        if "encoder" in state_dicts:
            self.encoder.load_state_dict(state_dicts["encoder"])
        if "decoder" in state_dicts:
            self.decoder.load_state_dict(state_dicts["decoder"])

        # Restore optimizer
        if "optimizer" in state_dicts:
            try:
                self.optimizer.load_state_dict(state_dicts["optimizer"])
            except (ValueError, KeyError, RuntimeError) as e:
                logger.warning(
                    "optimizer_state_load_failed",
                    error=str(e),
                    hint="config topology may have changed; fresh optimizer moments",
                )

        # Restore calibrators (saved as .pt sidecar files)
        if "l0_calibrator" in state_dicts:
            self._l0_calibrator.load_state_dict(state_dicts["l0_calibrator"])
        if "l1_calibrator" in state_dicts:
            self._l1_calibrator.load_state_dict(state_dicts["l1_calibrator"])

        logger.info("restored_from_checkpoint", step=self.global_step)

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
