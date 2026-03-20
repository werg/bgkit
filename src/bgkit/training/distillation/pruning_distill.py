"""Phase 1, Step 2: DeltaNet pruning distillation.

Distills a full BgKIT encoder (teacher) into a pruned version (student)
with DeltaNet layers removed and lightweight conv1d added. Uses staged
unfreezing with boundary MSE, auto-repro, projection, and cosine losses.

Pipeline position: phase1_step1 -> phase1_step2 -> phase1_step3 -> phase1_step4
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from bgkit.data.collators import collate_chat_repro
from bgkit.data.datasets.chat_repro_dataset import ChatReproDataset
from bgkit.data.datasets.mmap_token_dataset import MmapTokenDataset
from bgkit.data.samplers import TokenBudgetBatchSampler
from bgkit.data.survivor_selection import fill_survivor_gaps, select_survivors_by_threshold
from bgkit.data.threshold_calibrator import ThresholdCalibrator
from bgkit.models.encoder import BgKITEncoder, _expand_survivor_mask
from bgkit.models.pruned_qwen35 import PrunedBidirectionalQwen35
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpoint_registry import resolve_checkpoint
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.training.gradient_utils import enable_gradient_checkpointing

logger = structlog.get_logger()


class PruningDistillTrainer(BaseTrainer):
    """Step 1a: DeltaNet pruning distillation with staged unfreezing."""

    LIVE_CONFIG_FIELDS: dict[str, str] = {
        # Compression
        "target_ratio_min": "_target_ratio_min",
        "target_ratio_max": "_target_ratio_max",
        "eval_ratio": "_eval_ratio",
        "max_survivor_gap": "_max_gap",
        # Loss weights
        "w_boundary": "_w_boundary",
        "w_repro": "_w_repro",
        "w_proj": "_w_proj",
        "w_cosine": "_w_cosine",
        # Warmup
        "local_warmup_steps": "_local_warmup_steps",
    }

    _log_every: int = 50
    _use_device_prefetcher: bool = True

    def setup(self) -> None:
        """Load teacher/student encoders, ICE model, configure staged curriculum."""
        tcfg = self.cfg.training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        bgkit_cfg = self.cfg.model.bgkit
        backbone_name = bgkit_cfg.backbone_name
        hidden_dim = bgkit_cfg.get("hidden_dim", 1024)

        # --- Resolve step1 checkpoint ---
        step1_checkpoint = self._resolve_step1_checkpoint()
        if step1_checkpoint is None:
            raise ValueError(
                "phase1_step2 requires a step1 checkpoint. "
                "Set step1_checkpoint to a path or 'auto'."
            )

        # --- Teacher: separate checkpoint load, fully frozen ---
        logger.info("loading_teacher_encoder", checkpoint=step1_checkpoint)
        self.teacher_encoder = BgKITEncoder.from_pretrained(
            backbone_name,
            hidden_dim=hidden_dim,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=bgkit_cfg.get("backbone_revision", None),
            attn_implementation="sdpa",
            bidi_warmup_steps=0,  # teacher is already bidirectional
        )
        _, teacher_state = load_checkpoint(Path(step1_checkpoint))
        self.teacher_encoder.load_state_dict(teacher_state["encoder"])
        self.teacher_encoder.to(device)
        self.teacher_encoder.requires_grad_(False)
        self.teacher_encoder.eval()

        # --- Student: separate load + pruning surgery ---
        logger.info("loading_student_encoder", checkpoint=step1_checkpoint)
        student_encoder_full = BgKITEncoder.from_pretrained(
            backbone_name,
            hidden_dim=hidden_dim,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=bgkit_cfg.get("backbone_revision", None),
            attn_implementation="sdpa",
            bidi_warmup_steps=tcfg.get("bidi_warmup_steps", 0),
        )
        _, student_state = load_checkpoint(Path(step1_checkpoint))
        student_encoder_full.load_state_dict(student_state["encoder"])

        # Pruning surgery: consume the BidirectionalQwen35 backbone
        conv_kernel_size = tcfg.get("conv_kernel_size", 16)
        pruned_backbone = PrunedBidirectionalQwen35.from_unpruned(
            student_encoder_full.compressor.backbone,
            hidden_dim=hidden_dim,
            conv_kernel_size=conv_kernel_size,
            bidi_warmup_steps=tcfg.get("bidi_warmup_steps", 0),
        )
        # Replace the backbone in the compressor.
        # Norm correctness: from_pretrained() already called _set_norm_to_identity()
        # on the BidirectionalQwen35 backbone, so from_unpruned() received
        # nn.Identity() as the norm. The pruned backbone's self.norm is Identity
        # (no-op in forward), and the compressor's self.norm is the real LayerNorm.
        # Single normalization path: backbone(Identity) -> compressor.norm(real).
        student_encoder_full.compressor.backbone = pruned_backbone

        self.student_encoder = student_encoder_full
        self.student_encoder.to(device)

        # Enable gradient checkpointing on FullAttn layers in pruned backbone
        enable_gradient_checkpointing(self.student_encoder.compressor.backbone)

        # Also save the decoder state for pass-through in checkpoints
        self._decoder_state_dict = teacher_state.get("decoder", None)

        # BaseTrainer uses self.model for parameter counting
        self.model = self.student_encoder

        # --- ICE model (frozen) ---
        self._load_ice_model(tcfg)

        # --- Compression config ---
        compression_cfg = tcfg.get("compression", {})
        self._target_ratio_min = compression_cfg.get("target_ratio_min", 0.10)
        self._target_ratio_max = compression_cfg.get("target_ratio_max", 1.0)
        self._eval_ratio = compression_cfg.get("eval_ratio", 0.10)
        self._max_gap = compression_cfg.get("max_survivor_gap", 64)
        self._calibrator = ThresholdCalibrator(
            ema_decay=compression_cfg.get("calibrator_ema_decay", 0.99),
            warmup_batches=compression_cfg.get("calibrator_warmup_batches", 50),
            fallback_threshold=compression_cfg.get("fallback_threshold", 3.0),
        )
        self._pending_scores: list[torch.Tensor] = []
        self._is_evaluating = False

        # --- Stage thresholds ---
        stages = tcfg.get("stages", {})
        self._stage_thresholds = [
            stages.get("stage_0_steps", 0),
            stages.get("stage_1_steps", 500),
            stages.get("stage_2_steps", 2000),
            stages.get("stage_3_steps", 5000),
        ]
        self._current_stage = -1
        self._local_warmup_steps = tcfg.get("local_warmup_steps", 500)

        # --- Loss weights ---
        loss_weights = tcfg.get("loss_weights", {})
        self._w_boundary = loss_weights.get("boundary", 1.0)
        self._w_repro = loss_weights.get("repro", 2.0)
        self._w_proj = loss_weights.get("proj", 1.0)
        self._w_cosine = loss_weights.get("cosine", 0.2)

        # --- Dataset (same MmapTokenDataset as step 1) ---
        data_dir = self.cfg.data.tokens.input_dir
        max_seq_len = self.cfg.data.tokens.get("max_seq_len", 8192)
        variant_bank_path = self.cfg.data.tokens.variant_bank_path

        tokenizer_name = self.cfg.model.decoder.backbone_name
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True,
            revision=self.cfg.model.decoder.get("backbone_revision", None),
        )

        inner_dataset = MmapTokenDataset(
            data_dir, max_seq_len=max_seq_len, include_metadata=True,
        )
        full_dataset = ChatReproDataset(
            inner_dataset, tokenizer=self.tokenizer,
            variant_bank_path=variant_bank_path,
            seed=self.cfg.get("seed", 42),
        )

        max_eval_samples = tcfg.get("max_eval_samples", 2000)
        eval_size = min(max(1, int(len(full_dataset) * 0.1)), max_eval_samples)
        train_size = len(full_dataset) - eval_size
        self.train_dataset, self.eval_dataset = random_split(
            full_dataset, [train_size, eval_size],
        )

        max_batch_tokens = tcfg.get("max_batch_tokens", 16384)
        num_workers = self.cfg.compute.get("num_workers", 4)
        pin_memory = self.cfg.compute.get("pin_memory", False)
        seed = self.cfg.get("seed", 42)

        train_lengths = full_dataset.lengths[np.array(self.train_dataset.indices)]
        eval_lengths = full_dataset.lengths[np.array(self.eval_dataset.indices)]

        self.train_sampler = TokenBudgetBatchSampler(
            train_lengths, max_batch_tokens, shuffle=True, seed=seed,
        )
        eval_sampler = TokenBudgetBatchSampler(
            eval_lengths, max_batch_tokens, shuffle=False,
        )

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_sampler=self.train_sampler,
            collate_fn=collate_chat_repro, num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset, batch_sampler=eval_sampler,
            collate_fn=collate_chat_repro, num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # --- Initial stage + optimizer ---
        self._advance_to_current_stage()
        self._setup_optimizer()

        logger.info(
            "pruning_distill_setup",
            train_samples=train_size,
            eval_samples=eval_size,
            device=str(device),
            stage_thresholds=self._stage_thresholds,
            conv_kernel_size=conv_kernel_size,
        )

    # ------------------------------------------------------------------
    # Checkpoint resolution
    # ------------------------------------------------------------------

    def _resolve_step1_checkpoint(self) -> str | None:
        """Resolve step1_checkpoint with auto-resolution."""
        step1_checkpoint = self.cfg.get("step1_checkpoint", None)
        self._input_sources = {}

        if step1_checkpoint == "auto":
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            from bgkit.training.checkpoint_registry import CheckpointRegistry

            registry = CheckpointRegistry(checkpoint_dir)
            registry.backfill(checkpoint_dir)

            # Use latest phase1_step1 (not best-by-metric; same reasoning as
            # CompressionTrainer — early checkpoints have misleading metrics)
            best = registry.latest(phase="phase1_step1")
            if best is not None:
                step1_checkpoint = str(checkpoint_dir / best.name)
                self._input_sources["step1"] = best.name
                logger.info("step1_auto_resolved", checkpoint=best.name)
                return step1_checkpoint

            raise ValueError(
                "step1_checkpoint=auto but no phase1_step1 checkpoint found. "
                "Run phase1_step1 first."
            )

        if step1_checkpoint is not None:
            self._input_sources["step1"] = Path(step1_checkpoint).name
        return step1_checkpoint

    # ------------------------------------------------------------------
    # ICE model
    # ------------------------------------------------------------------

    def _load_ice_model(self, tcfg) -> None:
        """Load frozen ICE model for compression scoring."""
        from bgkit.models.ice import ICE

        ice_cfg = tcfg.get("ice", {})
        if not ice_cfg.get("checkpoint_path"):
            raise ValueError(
                "training.ice.checkpoint_path is required for pruning distillation."
            )

        if ice_cfg.checkpoint_path == "auto":
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            ice_ckpt_path = resolve_checkpoint(
                checkpoint_dir, phase="ice", metric="eval/mse",
                label="training.ice.checkpoint_path",
            )
            self._input_sources["ice"] = ice_ckpt_path.name
        else:
            ice_ckpt_path = Path(ice_cfg.checkpoint_path)
            self._input_sources["ice"] = ice_ckpt_path.name

        self.ice_model = ICE(
            input_dim=ice_cfg.get("input_dim", 1024),
            hidden_dim=ice_cfg.get("hidden_dim", 128),
            num_layers=ice_cfg.get("num_layers", 2),
            kernel_size=ice_cfg.get("kernel_size", 5),
        )
        _, ice_state = load_checkpoint(ice_ckpt_path)
        self.ice_model.load_state_dict(ice_state["model"])
        self.ice_model.to(self.device)
        self.ice_model.eval()
        self.ice_model.requires_grad_(False)

    # ------------------------------------------------------------------
    # ICE scoring
    # ------------------------------------------------------------------

    def _score_and_select(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score with ICE, select survivors. Random ratio per sample in training."""
        batch_size, seq_len, _ = embeddings.shape

        with torch.no_grad():
            ice_scores = self.ice_model(embeddings.float())
        ice_scores = ice_scores.masked_fill(~attention_mask, float("-inf"))
        if seq_len > 0:
            ice_scores[:, 0] = float("-inf")

        if not self._is_evaluating:
            stats_mask = attention_mask.clone()
            if seq_len > 0:
                stats_mask[:, 0] = False
            valid = ice_scores[stats_mask].detach()
            self._pending_scores.append(valid)

        survivor_mask = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=embeddings.device,
        )
        for b in range(batch_size):
            valid_len = int(attention_mask[b].sum().item())
            if valid_len == 0:
                continue

            if self._is_evaluating:
                ratio = self._eval_ratio
            else:
                ratio = random.uniform(self._target_ratio_min, self._target_ratio_max)

            threshold = self._calibrator.get_threshold(ratio)
            valid_scores = ice_scores[b, :valid_len]
            indices = select_survivors_by_threshold(valid_scores, threshold)
            indices = fill_survivor_gaps(indices, valid_scores, self._max_gap, valid_len)
            survivor_mask[b, indices] = True

        return survivor_mask

    def _flush_calibrator_scores(self) -> None:
        if self._pending_scores:
            non_empty = [s for s in self._pending_scores if s.numel() > 0]
            if non_empty:
                self._calibrator.update_from_flat(torch.cat(non_empty))
            self._pending_scores.clear()

    # ------------------------------------------------------------------
    # Staged unfreezing
    # ------------------------------------------------------------------

    def _advance_to_current_stage(self) -> None:
        """Set stage based on current global_step (for setup and resume)."""
        new_stage = -1
        for i, threshold in enumerate(self._stage_thresholds):
            if self.global_step >= threshold:
                new_stage = i
        if new_stage >= 0:
            self._current_stage = new_stage
            self.student_encoder.compressor.backbone.freeze_stage(new_stage)
            logger.info("stage_set", stage=new_stage, step=self.global_step)

    def _maybe_advance_stage(self) -> None:
        """Check if we should advance to next stage, add param groups if so."""
        for i, threshold in enumerate(self._stage_thresholds):
            if i > self._current_stage and self.global_step >= threshold:
                self._current_stage = i
                self.student_encoder.compressor.backbone.freeze_stage(i)

                # Add newly trainable params to optimizer
                new_params = [
                    p for p in self.student_encoder.parameters()
                    if p.requires_grad and not any(
                        p is existing
                        for pg in self.optimizer.param_groups
                        for existing in pg["params"]
                    )
                ]
                if new_params:
                    lr = self.cfg.training.lr
                    self._add_param_group_to_optimizer({
                        "params": new_params,
                        "lr": lr,
                        "base_lr": lr,
                        "warmup_start_step": self.global_step,
                        "local_warmup_steps": self._local_warmup_steps,
                    })

                logger.info(
                    "stage_advanced",
                    stage=i,
                    step=self.global_step,
                    new_params=sum(p.numel() for p in new_params),
                )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def _muon_excluded_param_ids(self) -> frozenset[int]:
        """Return param IDs of encoder embed_tokens — 2D but should not use Muon."""
        exclude = set()
        embed = self.student_encoder.compressor.backbone.get_input_embeddings()
        for p in embed.parameters():
            exclude.add(id(p))
        return frozenset(exclude)

    def _setup_optimizer(self) -> None:
        """Build optimizer from currently trainable params."""
        tcfg = self.cfg.training
        lr = tcfg.lr
        trainable = [p for p in self.student_encoder.parameters() if p.requires_grad]
        if not trainable:
            # Stage -1 or no params — create with dummy
            trainable = [nn.Parameter(torch.zeros(1, device=self.device))]

        param_groups = [{
            "params": trainable,
            "lr": lr,
            "base_lr": lr,
            "warmup_start_step": 0,
            "local_warmup_steps": 0,  # initial group uses base trainer's global warmup
        }]
        self.optimizer = self._create_optimizer(
            param_groups, lr, exclude_from_muon=self._muon_excluded_param_ids(),
        )

    def trainable_parameters(self) -> list:
        return [p for p in self.student_encoder.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _compute_distillation_loss(
        self,
        teacher_intermediates: list[torch.Tensor],
        student_intermediates: list[torch.Tensor],
        teacher_repro: torch.Tensor,
        student_repro: torch.Tensor,
        teacher_proj: torch.Tensor,
        student_proj: torch.Tensor,
        teacher_final: torch.Tensor,
        student_final: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute weighted distillation loss."""
        # Boundary MSE: average across block boundaries
        n_boundaries = min(len(teacher_intermediates), len(student_intermediates))
        boundary_loss = torch.tensor(0.0, device=teacher_final.device)
        if n_boundaries > 0:
            for t_h, s_h in zip(teacher_intermediates[:n_boundaries],
                                student_intermediates[:n_boundaries], strict=True):
                boundary_loss = boundary_loss + F.mse_loss(s_h, t_h.detach())
            boundary_loss = boundary_loss / n_boundaries

        # Auto-repro MSE
        repro_loss = F.mse_loss(student_repro, teacher_repro.detach())

        # Projection MSE
        proj_loss = F.mse_loss(student_proj, teacher_proj.detach())

        # Cosine similarity loss on final compressor output
        cos_sim = F.cosine_similarity(
            student_final.flatten(0, 1),
            teacher_final.detach().flatten(0, 1),
            dim=-1,
        )
        cosine_loss = (1.0 - cos_sim).mean()

        total = (
            self._w_boundary * boundary_loss
            + self._w_repro * repro_loss
            + self._w_proj * proj_loss
            + self._w_cosine * cosine_loss
        )

        metrics = {
            "loss": total,
            "loss/boundary": boundary_loss,
            "loss/repro": repro_loss,
            "loss/proj": proj_loss,
            "loss/cosine": cosine_loss,
        }
        return total, metrics

    def _forward_backward(self, batch) -> dict[str, float]:
        self.student_encoder.train()

        content_token_ids = batch["content_token_ids"].to(self.device)
        content_attention_mask = batch["content_attention_mask"].to(self.device)
        compression_prompt_ids = batch["compression_prompt_ids"].to(self.device)
        compression_prompt_mask = batch["compression_prompt_mask"].to(self.device)

        # --- Teacher forward (fully frozen) ---
        with torch.no_grad():
            teacher_embed = self.teacher_encoder.compressor.backbone.get_input_embeddings()
            content_emb = teacher_embed(content_token_ids)
            prompt_emb = teacher_embed(compression_prompt_ids)

            # ICE scoring on teacher embeddings (deterministic targets)
            survivor_mask = self._score_and_select(content_emb, content_attention_mask)

            # Teacher compressor with intermediates
            teacher_comp = self.teacher_encoder.compressor(
                content_emb,
                survivor_mask=survivor_mask,
                attention_mask=content_attention_mask,
                prompt_embeddings=prompt_emb,
                prompt_attention_mask=compression_prompt_mask,
                return_intermediates=True,
            )

            # Teacher auto-repro
            content_slice = teacher_comp.content_slice
            teacher_normed_content = teacher_comp.normed_embeddings[:, content_slice, :]
            teacher_repro = self.teacher_encoder.compressor.auto_reproduce(
                teacher_normed_content,
            )

            # Teacher projection
            full_raw = teacher_comp.raw_embeddings
            full_mask = teacher_comp.attention_mask
            full_survivor_mask = _expand_survivor_mask(
                survivor_mask, content_slice, full_raw.size(1),
            )
            teacher_proj_out = self.teacher_encoder.projection_block(
                full_raw, full_mask, survivor_mask=full_survivor_mask,
            )

        # --- Student forward ---
        student_embed = self.student_encoder.compressor.backbone.get_input_embeddings()
        student_content_emb = student_embed(content_token_ids)
        student_prompt_emb = student_embed(compression_prompt_ids)

        # Student compressor with intermediates (uses same survivor mask from teacher)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            student_comp = self.student_encoder.compressor(
                student_content_emb,
                survivor_mask=survivor_mask,
                attention_mask=content_attention_mask,
                prompt_embeddings=student_prompt_emb,
                prompt_attention_mask=compression_prompt_mask,
                return_intermediates=True,
            )

            # Student auto-repro
            student_normed_content = student_comp.normed_embeddings[:, content_slice, :]
            student_repro = self.student_encoder.compressor.auto_reproduce(
                student_normed_content,
            )

            # Student projection
            student_full_raw = student_comp.raw_embeddings
            student_full_mask = student_comp.attention_mask
            student_proj_out = self.student_encoder.projection_block(
                student_full_raw, student_full_mask,
                survivor_mask=full_survivor_mask,
            )

            # Compute distillation loss
            loss, metrics = self._compute_distillation_loss(
                teacher_intermediates=teacher_comp.intermediates or [],
                student_intermediates=student_comp.intermediates or [],
                teacher_repro=teacher_repro,
                student_repro=student_repro,
                teacher_proj=teacher_proj_out.projected_embeddings,
                student_proj=student_proj_out.projected_embeddings,
                teacher_final=teacher_comp.raw_embeddings[:, content_slice, :],
                student_final=student_comp.raw_embeddings[:, content_slice, :],
            )

        (loss / self._accum_steps).backward()

        self._flush_calibrator_scores()

        # Track compression stats
        n_survivors = int(survivor_mask.sum().item())
        n_valid = int(content_attention_mask.sum().item())
        metrics["actual_ratio"] = n_survivors / max(n_valid, 1)
        metrics["stage"] = self._current_stage

        return {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _pre_step_hook(self) -> None:
        self._maybe_advance_stage()

    def _post_optimizer_step(self, step: int) -> None:
        self.student_encoder.compressor.backbone.step_bidi_warmup()

    def _add_step_metrics(self, metrics: dict[str, float]) -> None:
        metrics["bidi_alpha"] = self.student_encoder.compressor.backbone.bidi_alpha
        metrics["stage"] = self._current_stage
        metrics["calibrated_threshold"] = self._calibrator.get_threshold(self._eval_ratio)
        metrics["trainable_params"] = sum(
            p.numel() for p in self.student_encoder.parameters() if p.requires_grad
        )

    def _build_training_state(self, es_best, es_evals_without_improvement, wandb_run) -> dict:
        state = super()._build_training_state(es_best, es_evals_without_improvement, wandb_run)
        state["stage"] = self._current_stage
        state["stage_thresholds"] = self._stage_thresholds
        # Persist live-tunable fields for resume
        state["target_ratio_min"] = self._target_ratio_min
        state["target_ratio_max"] = self._target_ratio_max
        state["eval_ratio"] = self._eval_ratio
        state["w_boundary"] = self._w_boundary
        state["w_repro"] = self._w_repro
        state["w_proj"] = self._w_proj
        state["w_cosine"] = self._w_cosine
        return state

    def apply_live_config(self, changes: dict) -> None:
        """Support live tuning of stage thresholds via control.json."""
        for i in range(4):
            key = f"stage_{i}_steps"
            if key in changes:
                val = changes.pop(key)
                if isinstance(val, (int, float)):
                    self._stage_thresholds[i] = int(val)
                    logger.info("live_stage_threshold_update", stage=i, steps=int(val))
        super().apply_live_config(changes)

    def _post_lr_schedule(self, step: int) -> None:
        """Apply per-group local warmup ramp after the base trainer sets LR."""
        for pg in self.optimizer.param_groups:
            local_start = pg.get("warmup_start_step", 0)
            local_warmup = pg.get("local_warmup_steps", self._local_warmup_steps)
            if local_warmup > 0 and step < local_start + local_warmup:
                ramp = max((step - local_start + 1) / local_warmup, 0.0)
                pg["lr"] *= ramp

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.student_encoder.eval()
        self._is_evaluating = True

        try:
            total_metrics: dict[str, float] = {}
            count = 0
            for batch in self.eval_dataloader:
                content_token_ids = batch["content_token_ids"].to(self.device)
                content_attention_mask = batch["content_attention_mask"].to(self.device)
                compression_prompt_ids = batch["compression_prompt_ids"].to(self.device)
                compression_prompt_mask = batch["compression_prompt_mask"].to(self.device)

                # Teacher forward
                teacher_embed = self.teacher_encoder.compressor.backbone.get_input_embeddings()
                content_emb = teacher_embed(content_token_ids)
                prompt_emb = teacher_embed(compression_prompt_ids)
                survivor_mask = self._score_and_select(content_emb, content_attention_mask)

                teacher_comp = self.teacher_encoder.compressor(
                    content_emb, survivor_mask=survivor_mask,
                    attention_mask=content_attention_mask,
                    prompt_embeddings=prompt_emb,
                    prompt_attention_mask=compression_prompt_mask,
                    return_intermediates=True,
                )
                content_slice = teacher_comp.content_slice
                teacher_normed = teacher_comp.normed_embeddings[:, content_slice, :]
                teacher_repro = self.teacher_encoder.compressor.auto_reproduce(teacher_normed)
                full_survivor_mask = _expand_survivor_mask(
                    survivor_mask, content_slice, teacher_comp.raw_embeddings.size(1),
                )
                teacher_proj = self.teacher_encoder.projection_block(
                    teacher_comp.raw_embeddings, teacher_comp.attention_mask,
                    survivor_mask=full_survivor_mask,
                )

                # Student forward
                student_embed = self.student_encoder.compressor.backbone.get_input_embeddings()
                s_content = student_embed(content_token_ids)
                s_prompt = student_embed(compression_prompt_ids)
                student_comp = self.student_encoder.compressor(
                    s_content, survivor_mask=survivor_mask,
                    attention_mask=content_attention_mask,
                    prompt_embeddings=s_prompt,
                    prompt_attention_mask=compression_prompt_mask,
                    return_intermediates=True,
                )
                student_normed = student_comp.normed_embeddings[:, content_slice, :]
                student_repro = self.student_encoder.compressor.auto_reproduce(student_normed)
                student_proj = self.student_encoder.projection_block(
                    student_comp.raw_embeddings, student_comp.attention_mask,
                    survivor_mask=full_survivor_mask,
                )

                _, batch_metrics = self._compute_distillation_loss(
                    teacher_comp.intermediates or [],
                    student_comp.intermediates or [],
                    teacher_repro, student_repro,
                    teacher_proj.projected_embeddings, student_proj.projected_embeddings,
                    teacher_comp.raw_embeddings[:, content_slice, :],
                    student_comp.raw_embeddings[:, content_slice, :],
                )

                for k, v in batch_metrics.items():
                    val = v.item() if hasattr(v, "item") else v
                    total_metrics[k] = total_metrics.get(k, 0.0) + val
                count += 1

            if count > 0:
                total_metrics = {k: v / count for k, v in total_metrics.items()}

            return total_metrics
        finally:
            self._is_evaluating = False

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self, checkpoint_dir: Path, metrics: dict[str, float] | None = None,
    ) -> Path:
        if self._training_state is None:
            self._training_state = {}
        self._training_state["stage"] = self._current_stage
        self._training_state["stage_thresholds"] = self._stage_thresholds

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
            encoder=self.student_encoder.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )
        # Pass through decoder from original step1 checkpoint (unchanged)
        if self._decoder_state_dict is not None:
            save_kwargs["decoder"] = self._decoder_state_dict
        # Save calibrator
        save_kwargs["l0_calibrator"] = self._calibrator.state_dict()

        ckpt_path = save_checkpoint(checkpoint_dir, metadata, **save_kwargs)
        self._last_checkpoint_path = str(ckpt_path)
        return ckpt_path

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load step1a checkpoint: restore student encoder + stage state."""
        metadata, state_dicts = load_checkpoint(checkpoint_path)
        self._check_optimizer_type_compat(metadata)

        # Restore student encoder
        if "encoder" in state_dicts:
            self.student_encoder.load_state_dict(state_dicts["encoder"])

        # Restore step position
        self.global_step = metadata.step
        self.epoch = metadata.epoch
        self._last_checkpoint_path = str(checkpoint_path)
        if metadata.schedule_params is not None:
            self._schedule_params = metadata.schedule_params
        if metadata.training_state is not None:
            self._training_state = metadata.training_state
            self._current_stage = metadata.training_state.get("stage", -1)
            saved_thresholds = metadata.training_state.get("stage_thresholds")
            if saved_thresholds:
                self._stage_thresholds = saved_thresholds
            # Restore live-tuned fields
            ts = metadata.training_state
            for attr, key in [
                ("_target_ratio_min", "target_ratio_min"),
                ("_target_ratio_max", "target_ratio_max"),
                ("_eval_ratio", "eval_ratio"),
                ("_w_boundary", "w_boundary"),
                ("_w_repro", "w_repro"),
                ("_w_proj", "w_proj"),
                ("_w_cosine", "w_cosine"),
            ]:
                if key in ts:
                    setattr(self, attr, ts[key])

        # Restore calibrator
        if "l0_calibrator" in state_dicts:
            self._calibrator.load_state_dict(state_dicts["l0_calibrator"])

        # Reapply freeze state from restored stage
        self._advance_to_current_stage()
        self._setup_optimizer()

        # Load optimizer state
        if "optimizer" in state_dicts:
            try:
                self.optimizer.load_state_dict(state_dicts["optimizer"])
            except (ValueError, KeyError, RuntimeError) as e:
                logger.warning(
                    "optimizer_state_load_failed", error=str(e),
                    hint="stage topology may have changed; fresh optimizer moments",
                )

        logger.info("restored_from_checkpoint", step=self.global_step, stage=self._current_stage)
