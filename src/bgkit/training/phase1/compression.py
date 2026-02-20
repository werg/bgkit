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
from bgkit.data.survivor_selection import select_survivors_by_threshold
from bgkit.eval.metrics.embedding_health import embedding_drift_metrics
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpoint_manager import CheckpointManager
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
        self.encoder = BgKITEncoder.from_pretrained(
            backbone_name,
            hidden_dim=hidden_dim,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=backbone_revision,
            attn_implementation="sdpa",
        )
        self.encoder.to(device)

        # Load encoder from Step 1 checkpoint
        step1_checkpoint = self.cfg.get("step1_checkpoint", None)
        if step1_checkpoint is not None:
            logger.info("loading_step1_checkpoint", path=step1_checkpoint)
            _, state_dicts = load_checkpoint(Path(step1_checkpoint))
            if "encoder" in state_dicts:
                self.encoder.load_state_dict(state_dicts["encoder"])

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
        if step1_checkpoint is not None and "decoder" in state_dicts:
            self.decoder.load_state_dict(state_dicts["decoder"])

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
        ice_ckpt_path = Path(ice_cfg.checkpoint_path)
        self.ice_model = ICE(
            input_dim=ice_cfg.get("input_dim", 1024),
            hidden_dim=ice_cfg.get("hidden_dim", 256),
            num_layers=ice_cfg.get("num_layers", 4),
            kernel_size=ice_cfg.get("kernel_size", 7),
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
        self._ice_threshold_start = curriculum.get("ice_threshold_start", 2.0)
        self._ice_threshold_end = curriculum.get("ice_threshold_end", 4.0)
        self._ice_threshold_ramp_steps = curriculum.get("ice_threshold_ramp_steps", 100000)
        self._min_compression_ratio = curriculum.get("min_compression_ratio", 0.05)
        self._max_compression_ratio = curriculum.get("max_compression_ratio", 0.30)

        # Live curriculum values
        self._ice_threshold = self._ice_threshold_start
        self._l1_enabled = False
        self._l1_transitioned = False
        self._l1_rebuild_pending = False

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

    def _setup_optimizer(self) -> None:
        """Create AdamW optimizer with encoder + decoder param groups."""
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

        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(param_groups, lr=tcfg.lr)
            logger.info("using_adamw8bit")
        except ImportError:
            self.optimizer = torch.optim.AdamW(param_groups, lr=tcfg.lr)
            logger.info("using_adamw_fp32", reason="bitsandbytes not available")

    def _trainable_params(self) -> list[torch.nn.Parameter]:
        """Collect all trainable parameters for gradient clipping."""
        params = [p for p in self.encoder.parameters() if p.requires_grad]
        params += [p for p in self.decoder.parameters() if p.requires_grad]
        return params

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _update_curriculum(self) -> None:
        """Update curriculum state based on current global_step."""
        step = self.global_step

        # ICE threshold ramp: linear from start to end over ramp_steps
        if step >= self._ice_threshold_ramp_steps:
            self._ice_threshold = self._ice_threshold_end
        else:
            t = step / max(self._ice_threshold_ramp_steps, 1)
            self._ice_threshold = (
                self._ice_threshold_start
                + t * (self._ice_threshold_end - self._ice_threshold_start)
            )

        # L1 introduction
        if step >= self._l1_introduction_step and not self._l1_enabled:
            self._l1_enabled = True
            self._l1_rebuild_pending = True
            logger.info("l1_curriculum_triggered", step=step)

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
    ) -> torch.Tensor:
        """Score embeddings with ICE and select survivors by threshold.

        Uses self._ice_threshold (from curriculum) and clamps survivor
        count per sample to [min_survivors, max_survivors] derived from
        the static compression ratio bounds.

        Args:
            embeddings: (B, L, D) embeddings to score.
            attention_mask: (B, L) bool mask for valid positions.

        Returns:
            (B, L) bool survivor mask.
        """
        batch_size, seq_len, _ = embeddings.shape

        with torch.no_grad():
            ice_scores = self.ice_model(embeddings.float())  # (B, L)
        ice_scores = ice_scores.masked_fill(~attention_mask, float("-inf"))

        survivor_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=embeddings.device)
        for b in range(batch_size):
            valid_len = int(attention_mask[b].sum().item())
            min_surv = max(1, int(valid_len * self._min_compression_ratio))
            max_surv = max(1, int(valid_len * self._max_compression_ratio))
            valid_scores = ice_scores[b, :valid_len]
            indices = select_survivors_by_threshold(
                valid_scores, self._ice_threshold,
                min_survivors=min_surv, max_survivors=max_surv,
            )
            survivor_mask[b, indices] = True

        # ICE distribution monitoring
        eval_every = self.cfg.training.get("eval_every", 0)
        if eval_every > 0 and self.global_step % eval_every == 0:
            valid_scores = ice_scores[attention_mask]
            selected_scores = ice_scores[survivor_mask]
            if selected_scores.numel() > 1:
                logger.info(
                    "ice_survivor_stats",
                    step=self.global_step,
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

    def _compress_direct_l1(
        self, batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run direct L1 compression (skip L0) for single-file repo samples.

        Used for commit reproduction where tokens are already cross-file
        representations and don't need L0 per-file compression.

        Returns:
            (survivor_embeddings, survivor_attention_mask)
        """
        file_ids = batch["file_token_ids"].to(self.device)  # (B, 1, L)
        file_masks = batch["file_attention_masks"].to(self.device)
        file_count = batch["file_count"]
        prompt_ids = batch["compression_prompt_ids"].to(self.device)
        prompt_mask = batch["compression_prompt_mask"].to(self.device)

        # Validate: direct_l1 requires exactly one file per sample
        if file_ids.size(1) != 1:
            raise ValueError(
                f"direct_l1 batch has {file_ids.size(1)} file slots, expected 1"
            )
        if not (file_count == 1).all():
            raise ValueError(
                f"direct_l1 batch has file_count {file_count.tolist()}, expected all 1"
            )

        # Squeeze out the single-file dimension: (B, 1, L) -> (B, L)
        token_ids = file_ids[:, 0]
        attention_mask = file_masks[:, 0]

        bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()
        embeddings = bgkit_embed(token_ids)
        prompt_emb = bgkit_embed(prompt_ids)

        # Live ICE scoring for survivor selection
        survivor_mask = self._score_and_select(embeddings, attention_mask)

        # Single encoder call at L1 level
        enc_out = self.encoder(
            input_embeddings=embeddings,
            survivor_mask=survivor_mask,
            attention_mask=attention_mask,
            prompt_embeddings=prompt_emb,
            prompt_attention_mask=prompt_mask,
        )

        return enc_out.survivor_embeddings, enc_out.survivor_attention_mask

    def _compress_repo_batch(
        self, batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run L0+L1 compression on a batch of RepoCompressionSamples.

        For each sample, runs L0 compression on each file independently,
        concatenates L0 survivors, then runs L1 compression on the
        concatenated result.

        Returns:
            (l1_survivor_embeddings, l1_survivor_attention_mask)
        """
        # Direct L1 dispatch: skip L0 for commit reproduction and similar
        if batch.get("direct_l1", False):
            return self._compress_direct_l1(batch)

        file_ids = batch["file_token_ids"].to(self.device)  # (B, max_files, max_file_len)
        file_masks = batch["file_attention_masks"].to(self.device)
        file_count = batch["file_count"]  # (B,)
        prompt_ids = batch["compression_prompt_ids"].to(self.device)
        prompt_mask = batch["compression_prompt_mask"].to(self.device)

        batch_size = file_ids.size(0)
        bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()
        prompt_emb = bgkit_embed(prompt_ids)

        # Process each sample: L0 per-file, then concat
        all_l0_survivors = []

        for b in range(batch_size):
            n_files = int(file_count[b].item())
            sample_survivors = []

            for f in range(n_files):
                f_ids = file_ids[b, f].unsqueeze(0)  # (1, file_len)
                f_mask = file_masks[b, f].unsqueeze(0)  # (1, file_len)
                f_emb = bgkit_embed(f_ids)

                # Live ICE scoring for survivor selection
                s_mask = self._score_and_select(f_emb, f_mask)

                # L0 compression for this file
                enc_out = self.encoder(
                    input_embeddings=f_emb,
                    survivor_mask=s_mask,
                    attention_mask=f_mask,
                    prompt_embeddings=prompt_emb[b:b + 1],
                    prompt_attention_mask=prompt_mask[b:b + 1],
                )

                # Map L0 output back to input embedding space for L1
                # all_embeddings = (1, L_content, D), normed pre-drop embeddings
                # survivor_mask = (1, L_content) bool from encoder output
                all_normed = enc_out.all_embeddings[0]  # (L_content, D)
                surv_mask_out = enc_out.survivor_mask[0]  # (L_content,) bool
                survivor_normed = all_normed[surv_mask_out]  # (n_surv, D)
                survivor_input_space = self.encoder.auto_reproduce(
                    survivor_normed.unsqueeze(0),
                )[0]  # (n_surv, D)
                sample_survivors.append(survivor_input_space)

            # Concatenate all L0 survivors for this sample
            if sample_survivors:
                concat_survivors = torch.cat(sample_survivors, dim=0)  # (total_surv, D)
            else:
                concat_survivors = torch.zeros(
                    1, self.encoder.compressor.hidden_dim,
                    device=self.device, dtype=torch.bfloat16,
                )

            all_l0_survivors.append(concat_survivors)

        # Pad L0 survivors across batch
        max_surv = max(s.size(0) for s in all_l0_survivors)
        hidden_dim = all_l0_survivors[0].size(-1)
        padded_l0 = torch.zeros(
            batch_size, max_surv, hidden_dim,
            device=self.device, dtype=all_l0_survivors[0].dtype,
        )
        l0_mask = torch.zeros(
            batch_size, max_surv, dtype=torch.bool, device=self.device,
        )
        for b, surv in enumerate(all_l0_survivors):
            padded_l0[b, :surv.size(0)] = surv
            l0_mask[b, :surv.size(0)] = True

        # L1 compression: score input-space L0 survivors with ICE
        l1_survivor_mask = self._score_and_select(padded_l0, l0_mask)

        # Run encoder for L1 (shared weights)
        l1_out = self.encoder(
            input_embeddings=padded_l0,
            survivor_mask=l1_survivor_mask,
            attention_mask=l0_mask,
            prompt_embeddings=prompt_emb,
            prompt_attention_mask=prompt_mask,
        )

        return l1_out.survivor_embeddings, l1_out.survivor_attention_mask

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

    def train_step(self, batch: dict) -> dict[str, float]:
        """Execute a single training step."""
        self.encoder.train()
        self.decoder.train()

        # Update curriculum
        self._update_curriculum()

        # Handle mixed batches
        if batch.get("mixed", False):
            return self._train_step_mixed(batch)

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

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm(self._trainable_params())
        self.optimizer.step()

        # Drift check
        self._check_decoder_drift(survivors)

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "sample_type": sample_type,
            "ice_threshold": self._ice_threshold,
            "l1_enabled": float(self._l1_enabled),
        }

    def _train_step_mixed(self, batch: dict) -> dict[str, float]:
        """Handle mixed batch containing both file and repo samples."""
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

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm(self._trainable_params())
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "sample_type": "mixed",
            "ice_threshold": self._ice_threshold,
            "l1_enabled": float(self._l1_enabled),
        }

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

        # Resume from checkpoint
        resume_path = self.cfg.get("resume_checkpoint", None)
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

        logger.info(
            "training_start",
            max_steps=max_steps,
            lr=base_lr,
            start_step=self.global_step,
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
            )
            ckpt_manager.load_existing(checkpoint_dir)
        else:
            ckpt_manager = None

        last_eval_metrics: dict[str, float] | None = None
        last_eval_step = -1

        stopped_early = False
        try:
            with GracefulInterruptor() as interruptor:
                for step in range(self.global_step, max_steps):
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

                    # Get batch
                    try:
                        batch = next(dataloader_iter)
                    except StopIteration:
                        self.epoch += 1
                        self._sync_epoch(self.epoch)
                        dataloader_iter = iter(self.train_dataloader)
                        batch = next(dataloader_iter)

                    # Train step
                    metrics = self.train_step(batch)
                    metrics["lr"] = lr
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
                        ckpt_path = self.save_checkpoint(
                            checkpoint_dir, metrics=step_metrics,
                        )
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
                            ckpt_path = self.save_checkpoint(checkpoint_dir)
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

                # Final eval + checkpoint
                eval_metrics = self.evaluate()
                eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                logger.info("final_eval", **eval_metrics)
                self._training_state = self._build_training_state(
                    es_best, es_evals_without_improvement, wandb_run,
                )
                ckpt_path = self.save_checkpoint(checkpoint_dir, metrics=eval_metrics)
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
            "ice_threshold": self._ice_threshold,
            "lora_active": self._lora_active,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation on the eval split."""
        self.encoder.eval()
        self.decoder.eval()
        self._eval_count += 1

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
                    target_attention_mask=batch["target_attention_mask"].to(self.device),
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
            "ice_threshold": self._ice_threshold,
            "lora_active": self._lora_active,
        })

        metadata = CheckpointMetadata(
            phase=self.cfg.training.phase,
            step=self.global_step,
            epoch=self.epoch,
            parent_checkpoint=self._last_checkpoint_path,
            metrics=metrics,
            schedule_params=self._schedule_params,
            training_state=self._training_state,
        )
        ckpt_path = save_checkpoint(
            checkpoint_dir,
            metadata,
            encoder=self.encoder.state_dict(),
            decoder=self.decoder.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )
        self._last_checkpoint_path = str(ckpt_path)
        return ckpt_path

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load checkpoint and restore all training state."""
        metadata, state_dicts = load_checkpoint(checkpoint_path)

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
            self._ice_threshold = ts.get("ice_threshold", self._ice_threshold_start)
            self._lora_active = ts.get("lora_active", False)

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

        logger.info("restored_from_checkpoint", step=self.global_step)

    # ------------------------------------------------------------------
    # Live config
    # ------------------------------------------------------------------

    def apply_live_config(self, changes: dict) -> None:
        """Support live tuning of ICE threshold and compression ratio bounds."""
        if "ice_threshold" in changes:
            val = changes["ice_threshold"]
            if isinstance(val, (int, float)) and val >= 0:
                self._ice_threshold = float(val)
                logger.info("live_ice_threshold_update", ice_threshold=self._ice_threshold)

        if "compression_ratio_min" in changes:
            val = changes["compression_ratio_min"]
            if isinstance(val, (int, float)) and 0 < val < 1:
                self._min_compression_ratio = float(val)
                logger.info(
                    "live_compression_ratio_update",
                    min=self._min_compression_ratio,
                    max=self._max_compression_ratio,
                )

        if "compression_ratio_max" in changes:
            val = changes["compression_ratio_max"]
            if isinstance(val, (int, float)) and 0 < val < 1:
                self._max_compression_ratio = float(val)
                logger.info(
                    "live_compression_ratio_update",
                    min=self._min_compression_ratio,
                    max=self._max_compression_ratio,
                )
