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
from pathlib import Path
from typing import ClassVar

import numpy as np
import structlog
import torch
from torch.utils.data import DataLoader, random_split

from bgkit.data.collators import collate_compression
from bgkit.data.datasets.commit_encoding_dataset import CommitEncodingDataset
from bgkit.data.samplers import LengthSortedBatchSampler
from bgkit.data.survivor_selection import fill_survivor_gaps, select_survivors_by_threshold
from bgkit.data.threshold_calibrator import ThresholdCalibrator
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpoint_registry import resolve_checkpoint
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.training.gradient_utils import enable_gradient_checkpointing

logger = structlog.get_logger()


class CommitEncodingTrainer(BaseTrainer):
    """Single-objective trainer for two-level commit encoding.

    Uses BaseTrainer.train() loop — no L1 introduction hook needed since
    both L0 and L1 are active from step 0. Overrides save_checkpoint()
    and load_checkpoint() to use the multi-artifact format that
    CompressionTrainer expects.
    """

    LIVE_CONFIG_FIELDS: ClassVar[dict[str, str]] = {
        "max_survivor_gap": "_max_gap",
        "target_ratio_ramp_steps": "_target_ratio_ramp_steps",
        "target_ratio_start": "_target_ratio_start",
        "target_ratio_end": "_target_ratio_end",
    }

    def setup(self) -> None:
        """Load trainable encoder/decoder, frozen ICE, create dataset and optimizer."""
        tcfg = self.cfg.training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # --- BgKIT encoder (trainable) ---
        bgkit_cfg = self.cfg.model.bgkit
        backbone_name = bgkit_cfg.backbone_name
        backbone_revision = bgkit_cfg.get("backbone_revision", None)
        hidden_dim = bgkit_cfg.get("hidden_dim", 1024)
        bidi_warmup = tcfg.get("bidi_warmup_steps", 500)

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
                attn_implementation="sdpa",
                bidi_warmup_steps=bidi_warmup,
            )
        else:
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

        self.encoder.requires_grad_(True)
        self.encoder.train()
        enable_gradient_checkpointing(self.encoder.compressor.backbone)

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
            attn_implementation="sdpa",
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

        # NVFP4 disabled: TE on sm_121 lacks sm_121a compilation (wgrad kernel
        # crash) and conflicts with gradient checkpointing (state divergence).
        # Revisit when TE is rebuilt with NVTE_CUDA_ARCHS=121a.

        enable_gradient_checkpointing(self.decoder.backbone)

        # torch.compile disabled: dynamic batch shapes cause continuous recompilation
        # on sm_121. See decoder_init.py for details.

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

        if ice_cfg.checkpoint_path == "auto":
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            ice_ckpt_path = resolve_checkpoint(
                checkpoint_dir,
                phase="ice",
                metric="eval/mse",
                label="training.ice.checkpoint_path",
            )
            self._input_sources["ice"] = ice_ckpt_path.name
        else:
            ice_ckpt_path = Path(ice_cfg.checkpoint_path)
            self._input_sources["ice"] = ice_ckpt_path.name

        if not ice_ckpt_path.exists():
            raise FileNotFoundError(f"ICE checkpoint not found: {ice_ckpt_path}")
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
        self._target_ratio_start = curriculum.get("target_ratio_start", 0.50)
        self._target_ratio_end = curriculum.get("target_ratio_end", 0.20)
        self._target_ratio_ramp_steps = curriculum.get("target_ratio_ramp_steps", 30000)
        self._target_ratio_override: float | None = None
        self._max_gap = curriculum.get("max_survivor_gap", 64)

        # Calibrators (both active from step 0)
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

        # Score buffers for deferred calibrator update
        self._pending_l0_scores: list[torch.Tensor] = []
        self._pending_l1_scores: list[torch.Tensor] = []

        # Eval isolation flag
        self._is_evaluating = False

        logger.info(
            "commit_encoding_trainer_setup",
            train_samples=train_size,
            eval_samples=eval_size,
            device=str(device),
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
        """Resolve step1_checkpoint: auto -> best phase1_step3 checkpoint."""
        step1_checkpoint = self.cfg.get("step1_checkpoint", None)
        self._input_sources = {}

        if step1_checkpoint == "auto":
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            resolved = resolve_checkpoint(
                checkpoint_dir,
                phase="phase1_step3",
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

    def _flush_calibrator_scores(self) -> None:
        """Flush buffered ICE scores into calibrators."""
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
                self._l1_batches_seen += 1
                if self._l1_batches_seen >= self._l1_calibrator_fast_batches:
                    self._l1_calibrator.set_decay(self._l1_calibrator_slow_decay)
            self._pending_l1_scores.clear()

    # ------------------------------------------------------------------
    # ICE scoring
    # ------------------------------------------------------------------

    def _score_and_select(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        level: str = "l0",
    ) -> torch.Tensor:
        """Score embeddings with ICE and select survivors via calibrated threshold."""
        batch_size, seq_len, _ = embeddings.shape

        with torch.no_grad():
            ice_scores = self.ice_model(embeddings.float())
        ice_scores = ice_scores.masked_fill(~attention_mask, float("-inf"))
        if seq_len > 0:
            ice_scores[:, 0] = float("-inf")

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

        survivor_mask = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=embeddings.device,
        )
        for b in range(batch_size):
            valid_len = int(attention_mask[b].sum().item())
            if valid_len == 0:
                continue
            valid_scores = ice_scores[b, :valid_len]
            indices = select_survivors_by_threshold(valid_scores, threshold)
            indices = fill_survivor_gaps(indices, valid_scores, self._max_gap, valid_len)
            survivor_mask[b, indices] = True

        return survivor_mask

    # ------------------------------------------------------------------
    # Compression pipeline: L0 per-file → L1 cross-file
    # ------------------------------------------------------------------

    @staticmethod
    def _compress_single_file_pure(
        compressor: torch.nn.Module,
        f_emb: torch.Tensor,
        s_mask: torch.Tensor,
        f_mask: torch.Tensor,
        prompt_emb: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pure (side-effect-free) single-file L0 compression for checkpointing."""
        comp_out = compressor(
            f_emb,
            survivor_mask=s_mask,
            attention_mask=f_mask,
            prompt_embeddings=prompt_emb,
            prompt_attention_mask=prompt_mask,
        )
        content_normed = comp_out.normed_embeddings[:, comp_out.content_slice, :]
        survivor_normed = content_normed[0][s_mask[0]]
        return compressor.auto_reproduce(survivor_normed.unsqueeze(0))[0]

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

        Uses activation checkpointing per file to limit encoder memory to one
        file at a time. Side effects (calibrator score buffering) happen outside
        the checkpointed region.
        """
        from torch.utils.checkpoint import checkpoint as torch_checkpoint

        compressor = self.encoder.compressor
        sample_survivors = []
        for f in range(n_files):
            f_ids = file_ids[f].unsqueeze(0)
            f_mask = file_masks[f].unsqueeze(0)
            f_emb = bgkit_embed(f_ids)

            # INVARIANT: _score_and_select MUST stay outside the checkpoint
            # boundary — it appends to _pending_l0_scores (side effect that
            # would be doubled by checkpoint recomputation on backward).
            s_mask = self._score_and_select(f_emb, f_mask)

            # Pure encoder forward INSIDE checkpoint (recomputed on backward).
            # All inputs are deterministic (s_mask pre-computed above).
            survivor_input_space = torch_checkpoint(
                self._compress_single_file_pure,
                compressor, f_emb, s_mask, f_mask,
                prompt_emb, prompt_mask,
                use_reentrant=False,
            )
            sample_survivors.append(survivor_input_space)

        if sample_survivors:
            return torch.cat(sample_survivors, dim=0)
        return torch.zeros(
            1, compressor.hidden_dim,
            device=self.device, dtype=torch.bfloat16,
        )

    # ------------------------------------------------------------------
    # Forward/backward
    # ------------------------------------------------------------------

    def _forward_backward(self, batch: dict) -> dict[str, float]:
        """Per-sample forward + backward for memory-efficient gradient accumulation.

        Processes each sample's L0→L1→decoder pipeline independently, calling
        .backward() after each to free the autograd graph immediately. Peak
        memory is proportional to the single largest sample, not the batch sum.
        """
        self.encoder.train()
        self.decoder.train()

        file_ids = batch["file_token_ids"].to(self.device)
        file_masks = batch["file_attention_masks"].to(self.device)
        file_count = batch["file_count"]
        prompt_ids = batch["compression_prompt_ids"].to(self.device)
        prompt_mask = batch["compression_prompt_mask"].to(self.device)
        target_ids = batch["target_token_ids"].to(self.device)
        target_mask = batch["target_attention_mask"].to(self.device)
        loss_mask_batch = batch.get("target_loss_mask")
        if loss_mask_batch is not None:
            loss_mask_batch = loss_mask_batch.to(self.device)

        batch_size = file_ids.size(0)
        bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()
        scale = 1.0 / (batch_size * self._accum_steps)

        total_loss = 0.0
        total_survivors = 0
        total_valid = 0

        for b in range(batch_size):
            n_files = int(file_count[b].item())

            # Recompute prompt_emb per sample — .backward() frees the autograd
            # graph, so a shared prompt_emb would crash on the second sample.
            prompt_emb_b = bgkit_embed(prompt_ids[b:b + 1])

            # L0 per-file (checkpointed)
            l0_surv = self._compress_single_l0_l1(
                file_ids[b], file_masks[b], n_files,
                prompt_emb_b, prompt_mask[b:b + 1],
                bgkit_embed,
            )

            # L1 for this sample only
            l1_input = l0_surv.unsqueeze(0)
            l1_mask = torch.ones(
                1, l0_surv.size(0), dtype=torch.bool, device=self.device,
            )
            l1_survivor_mask = self._score_and_select(l1_input, l1_mask, level="l1")

            l1_out = self.encoder(
                input_embeddings=l1_input,
                survivor_mask=l1_survivor_mask,
                attention_mask=l1_mask,
                prompt_embeddings=prompt_emb_b,
                prompt_attention_mask=prompt_mask[b:b + 1],
            )

            sample_survivors = l1_out.survivor_embeddings
            sample_surv_mask = l1_out.survivor_attention_mask

            sample_loss_mask = loss_mask_batch[b:b + 1] if loss_mask_batch is not None else None

            # Decoder for this sample only
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda",
            ):
                loss = self.decoder.forward_with_loss(
                    survivor_embeddings=sample_survivors,
                    target_ids=target_ids[b:b + 1],
                    target_attention_mask=target_mask[b:b + 1],
                    survivor_attention_mask=sample_surv_mask,
                    loss_mask=sample_loss_mask,
                )

            # Backward immediately — frees this sample's autograd graph
            (loss * scale).backward()

            total_loss += loss.item()
            total_survivors += int(sample_surv_mask.sum().item())
            total_valid += int(file_masks[b].sum().item())

        # Flush calibrator scores (once per micro-batch)
        self._flush_calibrator_scores()

        target_ratio = self._current_target_ratio()
        actual_ratio = total_survivors / max(total_valid, 1)

        return {
            "loss": total_loss / batch_size,
            "target_ratio": target_ratio,
            "actual_ratio": actual_ratio,
            "calibrated_threshold_l0": self._l0_calibrator.get_threshold(target_ratio),
            "calibrated_threshold_l1": self._l1_calibrator.get_threshold(target_ratio),
        }

    def _post_step(self, step: int) -> None:
        """Advance bidirectional warmup after each optimizer step."""
        self.encoder.step_bidi_warmup()

    def _get_bidi_alpha(self) -> float:
        from bgkit.models.bidirectional_qwen35 import BidirectionalQwen35
        from bgkit.models.pruned_qwen35 import PrunedBidirectionalQwen35

        for module in self.encoder.modules():
            if isinstance(module, (BidirectionalQwen35, PrunedBidirectionalQwen35)):
                return module.bidi_alpha
        return 1.0

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.encoder.eval()
        self.decoder.eval()
        self._is_evaluating = True

        try:
            total_loss = 0.0
            total_tokens = 0.0

            num_batches = len(self.eval_dataloader)
            for batch_idx, batch in enumerate(self.eval_dataloader):
                if batch_idx % 100 == 0:
                    logger.info("eval_progress", batch=batch_idx, total=num_batches)

                file_ids = batch["file_token_ids"].to(self.device)
                file_masks = batch["file_attention_masks"].to(self.device)
                file_count = batch["file_count"]
                prompt_ids = batch["compression_prompt_ids"].to(self.device)
                prompt_mask = batch["compression_prompt_mask"].to(self.device)
                target_ids = batch["target_token_ids"].to(self.device)
                target_mask = batch["target_attention_mask"].to(self.device)
                loss_mask_batch = batch.get("target_loss_mask")
                if loss_mask_batch is not None:
                    loss_mask_batch = loss_mask_batch.to(self.device)

                batch_size = file_ids.size(0)
                bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()

                for b in range(batch_size):
                    n_files = int(file_count[b].item())
                    prompt_emb_b = bgkit_embed(prompt_ids[b:b + 1])

                    l0_surv = self._compress_single_l0_l1(
                        file_ids[b], file_masks[b], n_files,
                        prompt_emb_b, prompt_mask[b:b + 1],
                        bgkit_embed,
                    )

                    l1_input = l0_surv.unsqueeze(0)
                    l1_mask = torch.ones(
                        1, l0_surv.size(0), dtype=torch.bool, device=self.device,
                    )
                    l1_survivor_mask = self._score_and_select(
                        l1_input, l1_mask, level="l1",
                    )

                    l1_out = self.encoder(
                        input_embeddings=l1_input,
                        survivor_mask=l1_survivor_mask,
                        attention_mask=l1_mask,
                        prompt_embeddings=prompt_emb_b,
                        prompt_attention_mask=prompt_mask[b:b + 1],
                    )

                    sample_loss_mask = (
                        loss_mask_batch[b:b + 1] if loss_mask_batch is not None else None
                    )

                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16,
                        enabled=self.device.type == "cuda",
                    ):
                        loss = self.decoder.forward_with_loss(
                            survivor_embeddings=l1_out.survivor_embeddings,
                            target_ids=target_ids[b:b + 1],
                            target_attention_mask=target_mask[b:b + 1],
                            survivor_attention_mask=l1_out.survivor_attention_mask,
                            loss_mask=sample_loss_mask,
                        )

                    # Token-weighted accumulation for comparable perplexity
                    eval_loss_mask = batch.get("target_loss_mask")
                    if eval_loss_mask is not None:
                        sample_tokens = eval_loss_mask[b, 1:].sum().item()
                    else:
                        sample_tokens = batch["target_attention_mask"][b, 1:].sum().item()
                    total_loss += loss.item() * sample_tokens
                    total_tokens += sample_tokens

            avg_loss = total_loss / max(total_tokens, 1)
            perplexity = torch.exp(torch.tensor(avg_loss)).item()

            return {
                "loss": avg_loss,
                "perplexity": perplexity,
            }
        finally:
            self._is_evaluating = False
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
        save_kwargs = dict(
            encoder=self.encoder.state_dict(),
            decoder=self.decoder.state_dict(),
            optimizer=self.optimizer.state_dict(),
            l0_calibrator=self._l0_calibrator.state_dict(),
            l1_calibrator=self._l1_calibrator.state_dict(),
        )
        if getattr(self, "_decoder_lora", False):
            save_kwargs["decoder_merged"] = self.decoder.merge_lora()

        ckpt_path = save_checkpoint(checkpoint_dir, metadata, **save_kwargs)
        self._last_checkpoint_path = str(ckpt_path)
        return ckpt_path

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load checkpoint and restore all training state."""
        metadata, state_dicts = load_checkpoint(checkpoint_path)
        self._check_optimizer_type_compat(metadata)

        # Restore step/epoch/schedule BEFORE loading weights
        self.global_step = metadata.step
        self.epoch = metadata.epoch
        self._last_checkpoint_path = str(checkpoint_path)
        if metadata.schedule_params is not None:
            self._schedule_params = metadata.schedule_params
        if metadata.training_state is not None:
            self._training_state = metadata.training_state
            ts = metadata.training_state
            self._l1_batches_seen = ts.get("l1_batches_seen", 0)
            self._target_ratio_override = ts.get("target_ratio_override")
            if self._l1_batches_seen >= self._l1_calibrator_fast_batches:
                self._l1_calibrator.set_decay(self._l1_calibrator_slow_decay)

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
                logger.warning("optimizer_state_load_failed", error=str(e))

        # Restore calibrators
        if "l0_calibrator" in state_dicts:
            self._l0_calibrator.load_state_dict(state_dicts["l0_calibrator"])
        if "l1_calibrator" in state_dicts:
            self._l1_calibrator.load_state_dict(state_dicts["l1_calibrator"])

        logger.info("restored_from_checkpoint", step=self.global_step)

    # ------------------------------------------------------------------
    # Live config
    # ------------------------------------------------------------------

    def apply_live_config(self, changes: dict) -> None:
        if "target_ratio" in changes:
            val = changes["target_ratio"]
            if val is None:
                self._target_ratio_override = None
            elif isinstance(val, (int, float)) and 0 < val < 1:
                self._target_ratio_override = float(val)
        super().apply_live_config(changes)
