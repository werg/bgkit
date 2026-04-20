"""Joint block pretraining trainer.

Jointly pretrains the penultimate compressor layer (for auto-reproduction)
and the projection block (for decoder alignment). Two-objective training
loop where gradients flow freely between both objectives.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoModel

from bgkit.data.chat_template import (
    build_encoder_prefix_ids,
    build_encoder_user_only_prefix_ids,
    load_all_variant_banks,
    select_variant,
)
from bgkit.data.collators import collate_token_ids
from bgkit.data.datasets.mmap_token_dataset import MmapTokenDataset
from bgkit.data.samplers import TokenBudgetBatchSampler
from bgkit.models.components.auto_reproduction import auto_reproduction_loss
from bgkit.models.encoder import BgKITEncoder, _resolve_layers
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.training.gradient_utils import enable_gradient_checkpointing
from bgkit.utils.attention_backend import resolve_attention_implementation
from bgkit.utils.model_utils import count_parameters, slerp_merge

logger = structlog.get_logger()


def joint_block_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length token ID samples into a batch."""
    return collate_token_ids(batch)


@dataclass
class _ForwardResult:
    """Intermediate forward pass results for both objectives."""

    comp_out: object  # CompressorOutput
    auto_repro_pred: torch.Tensor  # content-only slice
    proj_out: object  # ProjectionOutput
    proj_content: torch.Tensor  # content-only slice of projected embeddings
    loss_repro: torch.Tensor
    loss_proj: torch.Tensor
    loss: torch.Tensor


class JointBlockTrainer(BaseTrainer):
    """Trainer for joint block pretraining: auto-repro + decoder alignment."""

    LIVE_CONFIG_FIELDS = {
        "w_repro": "w_repro",
        "w_proj": "w_proj",
    }

    def setup(self) -> None:
        """Construct encoder, load decoder embeddings, freeze layers, create dataset."""
        tcfg = self.cfg.training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        attention_impl = resolve_attention_implementation(
            self.cfg.compute.get("attention_implementation", "auto")
        )

        # Load backbone
        backbone_name = self.cfg.model.bgkit.backbone_name
        backbone_revision = self.cfg.model.bgkit.get("backbone_revision", None)
        hidden_dim = self.cfg.model.bgkit.get("hidden_dim", 1024)
        logger.info("loading_backbone", model=backbone_name, revision=backbone_revision)

        backbone = AutoModel.from_pretrained(
            backbone_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=backbone_revision,
            attn_implementation=attention_impl,
        )

        # Optional SLERP merge with decoder backbone
        slerp_t = tcfg.get("slerp_t", None)
        if slerp_t is not None:
            decoder_name = self.cfg.model.decoder.backbone_name
            decoder_revision = self.cfg.model.decoder.get("backbone_revision", None)
            logger.info("loading_decoder_for_slerp", model=decoder_name, t=slerp_t)
            decoder_for_slerp = AutoModel.from_pretrained(
                decoder_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                revision=decoder_revision,
                attn_implementation=attention_impl,
            )

            sd_a = backbone.state_dict()
            sd_b = decoder_for_slerp.state_dict()

            common_keys = set(sd_a.keys()) & set(sd_b.keys())
            overlap_ratio = len(common_keys) / max(len(sd_a), 1)
            logger.info("slerp_key_overlap", ratio=overlap_ratio, common=len(common_keys))
            if overlap_ratio < 0.5:
                raise ValueError(
                    f"SLERP merge key overlap too low ({overlap_ratio:.1%}). "
                    "Backbone and decoder likely have incompatible architectures."
                )

            merged_sd = slerp_merge(sd_a, sd_b, t=slerp_t)
            backbone.load_state_dict(merged_sd, strict=False)
            del decoder_for_slerp, sd_a, sd_b, merged_sd

        # Construct encoder (splits backbone into compressor + projection block).
        # Stay fully causal (-1): frozen backbone layers can't adapt to bidi masks,
        # and the task is too trivial to benefit. Step 2 handles the gradual warmup.
        self.encoder = BgKITEncoder.from_pretrained(
            backbone, hidden_dim=hidden_dim, bidi_warmup_steps=-1,
        )
        self.encoder.to(device)

        # Enable gradient checkpointing on the backbone to reduce memory
        # (essential when using torch DeltaNet fallback without fla Triton kernels)
        enable_gradient_checkpointing(self.encoder.compressor.backbone)

        # Load decoder's embedding matrix as frozen reference target
        decoder_name = self.cfg.model.decoder.backbone_name
        decoder_revision = self.cfg.model.decoder.get("backbone_revision", None)
        logger.info("loading_decoder_embeddings", model=decoder_name)
        decoder_model = AutoModel.from_pretrained(
            decoder_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=decoder_revision,
            attn_implementation=attention_impl,
        )
        self.decoder_embed = decoder_model.get_input_embeddings()
        self.decoder_embed.requires_grad_(False)
        self.decoder_embed.to(device)
        del decoder_model

        # Freeze everything, then unfreeze target components
        self.encoder.requires_grad_(False)

        heads_only = tcfg.get("heads_only", False)
        if heads_only:
            # Only train the linear projection heads
            self.encoder.compressor.auto_repro_head.requires_grad_(True)
            self.encoder.projection_block.projection_head.requires_grad_(True)
        else:
            # Train heads + transformer blocks + norms
            compressor_layers = _resolve_layers(self.encoder.compressor.backbone)
            compressor_layers[-1].requires_grad_(True)
            self.encoder.compressor.norm.requires_grad_(True)
            self.encoder.compressor.auto_repro_head.requires_grad_(True)
            self.encoder.projection_block.requires_grad_(True)

        trainable = count_parameters(self.encoder, trainable_only=True)
        total = count_parameters(self.encoder, trainable_only=False)
        logger.info("param_counts", trainable=trainable, total=total)

        # BaseTrainer uses self.model for default checkpoint save/load
        self.model = self.encoder

        # Dataset + split
        data_dir = self.cfg.data.tokens.input_dir
        max_seq_len = self.cfg.data.tokens.get("max_seq_len", 8192)
        full_dataset = MmapTokenDataset(
            data_dir, max_seq_len=max_seq_len, include_metadata=False
        )
        max_eval_samples = tcfg.get("max_eval_samples", 10000)
        eval_size = min(max(1, int(len(full_dataset) * 0.1)), max_eval_samples)
        train_size = len(full_dataset) - eval_size
        if train_size < 1:
            raise ValueError(
                f"Dataset too small for train/eval split (got {len(full_dataset)} samples, "
                "need at least 2)"
            )
        self.train_dataset, self.eval_dataset = random_split(
            full_dataset, [train_size, eval_size]
        )

        max_batch_tokens = tcfg.get("max_batch_tokens", 65536)
        num_workers = self.cfg.compute.get("num_workers", 4)
        pin_memory = self.cfg.compute.get("pin_memory", False)

        train_lengths = full_dataset.lengths[np.array(self.train_dataset.indices)]
        eval_lengths = full_dataset.lengths[np.array(self.eval_dataset.indices)]

        self.train_sampler = TokenBudgetBatchSampler(
            train_lengths, max_batch_tokens, shuffle=True, seed=self.cfg.get("seed", 42),
        )
        eval_sampler = TokenBudgetBatchSampler(
            eval_lengths, max_batch_tokens, shuffle=False,
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=joint_block_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_sampler=eval_sampler,
            collate_fn=joint_block_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Loss weights
        self.w_repro = tcfg.get("w_repro", 1.0)
        self.w_proj = tcfg.get("w_proj", 1.0)

        # Optimizer -- trainable params only
        trainable_params = [p for p in self.encoder.parameters() if p.requires_grad]
        self.optimizer = self._create_optimizer(
            [{"params": trainable_params, "lr": tcfg.lr, "base_lr": tcfg.lr}],
            tcfg.lr,
        )

        # ChatML prefix for encoder conditioning (with prompt rotation)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            backbone_name, revision=backbone_revision, trust_remote_code=True,
        )
        self._tokenizer = tokenizer
        self._seed = self.cfg.get("seed", 42)

        # Load variant banks for per-epoch prompt rotation
        variant_dir = getattr(tcfg, "prompt_variants_dir", None)
        if variant_dir and Path(variant_dir).is_dir():
            self._variant_bank = load_all_variant_banks(variant_dir)
            logger.info(
                "variant_bank_loaded",
                variant_dir=str(variant_dir),
                num_variants=len(self._variant_bank),
            )
        else:
            self._variant_bank = []

        if self._variant_bank:
            # Start with first variant
            variant = select_variant(self._variant_bank, 0, self._seed)
            prefix_ids = build_encoder_prefix_ids(
                tokenizer, variant["compression_prompt"]
            ).to(device)
            logger.info(
                "encoder_chatml_prefix",
                prefix_len=prefix_ids.size(0),
                prompt=variant["compression_prompt"][:60],
            )
        else:
            prefix_ids = build_encoder_user_only_prefix_ids(tokenizer).to(device)
            logger.info("encoder_chatml_prefix", prefix_len=prefix_ids.size(0))

        # Pre-compute and freeze the prompt embeddings (1, prefix_len, hidden_dim)
        with torch.no_grad():
            self._prompt_embeddings = self._get_input_embeddings(prefix_ids.unsqueeze(0))

        logger.info(
            "joint_block_trainer_setup",
            train_samples=train_size,
            eval_samples=eval_size,
            device=str(device),
            w_repro=self.w_repro,
            w_proj=self.w_proj,
        )

    def _get_input_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get input embeddings from the compressor's backbone embedding layer."""
        return self.encoder.compressor.backbone.get_input_embeddings()(token_ids)

    def _sync_epoch(self, epoch: int) -> None:
        """Propagate epoch + rotate encoder prompt if variant bank is loaded."""
        super()._sync_epoch(epoch)

        if not self._variant_bank:
            return

        variant = select_variant(self._variant_bank, epoch, self._seed + epoch)
        prefix_ids = build_encoder_prefix_ids(
            self._tokenizer, variant["compression_prompt"]
        ).to(self.device)
        with torch.no_grad():
            self._prompt_embeddings = self._get_input_embeddings(prefix_ids.unsqueeze(0))
        logger.info(
            "epoch_prompt_rotation",
            epoch=epoch,
            prompt=variant["compression_prompt"][:60],
        )

    def _amp_context(self):
        """Return autocast context for CUDA, or nullcontext for CPU."""
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _get_prompt_embeddings(self) -> torch.Tensor | None:
        """Return cached ChatML prefix embeddings (1, prefix_len, hidden_dim)."""
        return getattr(self, "_prompt_embeddings", None)

    def _forward_both(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        target_repro: torch.Tensor,
        target_proj: torch.Tensor,
    ) -> _ForwardResult:
        """Run both forward passes and compute losses."""
        prompt_emb = self._get_prompt_embeddings()
        if prompt_emb is not None:
            # Expand to batch size
            prompt_emb = prompt_emb.expand(input_embeddings.size(0), -1, -1)

        comp_out = self.encoder.compressor(
            input_embeddings, attention_mask=attention_mask,
            prompt_embeddings=prompt_emb,
        )

        auto_repro_pred_full = self.encoder.compressor.auto_reproduce(
            comp_out.normed_embeddings,
        )
        # Slice to content-only positions (targets don't include prefix)
        cs = comp_out.content_slice
        auto_repro_pred = auto_repro_pred_full[:, cs, :]
        loss_repro = auto_reproduction_loss(
            auto_repro_pred, target_repro, mask=attention_mask.float(),
        )

        proj_out = self.encoder.projection_block(
            comp_out.raw_embeddings,
            attention_mask=comp_out.attention_mask,
            survivor_mask=None,
        )
        proj_content = proj_out.projected_embeddings[:, cs, :]
        loss_proj = auto_reproduction_loss(
            proj_content, target_proj, mask=attention_mask.float(),
        )

        loss = self.w_repro * loss_repro + self.w_proj * loss_proj

        return _ForwardResult(
            comp_out=comp_out,
            auto_repro_pred=auto_repro_pred,
            proj_out=proj_out,
            proj_content=proj_content,
            loss_repro=loss_repro,
            loss_proj=loss_proj,
            loss=loss,
        )

    def trainable_parameters(self) -> list:
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def _forward_backward(self, batch) -> dict[str, float]:
        self.encoder.train()

        token_ids = batch["token_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Input embeddings -- targets are detached copies
        input_embeddings = self._get_input_embeddings(token_ids)
        target_repro = input_embeddings.detach()
        target_proj = self.decoder_embed(token_ids).detach()

        with self._amp_context():
            fwd = self._forward_both(input_embeddings, attention_mask, target_repro, target_proj)

        # Scaled backward (for gradient accumulation)
        (fwd.loss / self._accum_steps).backward()

        # Cosine similarity metrics (content-only, matching targets)
        # Return detached tensors — .item() is deferred to _average_metrics
        # to avoid GPU sync between micro-batches in the accumulation loop.
        with torch.no_grad():
            mask_f = attention_mask.float()
            cos_repro = F.cosine_similarity(fwd.auto_repro_pred, target_repro, dim=-1)
            cos_repro_avg = (cos_repro * mask_f).sum() / mask_f.sum().clamp(min=1)

            cos_proj = F.cosine_similarity(fwd.proj_content, target_proj, dim=-1)
            cos_proj_avg = (cos_proj * mask_f).sum() / mask_f.sum().clamp(min=1)

        metrics = {
            "loss": fwd.loss.detach(),
            "loss_repro": fwd.loss_repro.detach(),
            "loss_proj": fwd.loss_proj.detach(),
            "cosine_sim_repro": cos_repro_avg.detach(),
            "cosine_sim_proj": cos_proj_avg.detach(),
        }

        # Drop CompressorOutput tensor refs per release() contract.
        fwd.comp_out.release()

        return metrics

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.encoder.eval()

        total_mse_repro = 0.0
        total_mse_proj = 0.0
        total_cosine_repro = 0.0
        total_cosine_proj = 0.0
        n = 0.0

        num_batches = len(self.eval_dataloader)
        for batch_idx, batch in enumerate(self.eval_dataloader):
            if batch_idx % 100 == 0:
                logger.info("eval_progress", batch=batch_idx, total=num_batches)
            token_ids = batch["token_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            input_embeddings = self._get_input_embeddings(token_ids)
            target_repro = input_embeddings.detach()
            target_proj = self.decoder_embed(token_ids).detach()

            with self._amp_context():
                fwd = self._forward_both(
                    input_embeddings, attention_mask, target_repro, target_proj,
                )

            mask_f = attention_mask.float()
            batch_n = mask_f.sum().item()
            n += batch_n

            # MSE
            mse_repro = F.mse_loss(
                fwd.auto_repro_pred, target_repro, reduction="none",
            ).mean(dim=-1)
            total_mse_repro += (mse_repro * mask_f).sum().item()

            mse_proj = F.mse_loss(
                fwd.proj_content, target_proj, reduction="none",
            ).mean(dim=-1)
            total_mse_proj += (mse_proj * mask_f).sum().item()

            # Cosine similarity
            cos_repro = F.cosine_similarity(fwd.auto_repro_pred, target_repro, dim=-1)
            total_cosine_repro += (cos_repro * mask_f).sum().item()

            cos_proj = F.cosine_similarity(fwd.proj_content, target_proj, dim=-1)
            total_cosine_proj += (cos_proj * mask_f).sum().item()

        return {
            "mse_repro": total_mse_repro / max(n, 1),
            "mse_proj": total_mse_proj / max(n, 1),
            "cosine_sim_repro": total_cosine_repro / max(n, 1),
            "cosine_sim_proj": total_cosine_proj / max(n, 1),
        }

    def save_checkpoint(
        self, checkpoint_dir: Path, metrics: dict[str, float] | None = None
    ) -> Path:
        """Save encoder state dict with phase metadata."""
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
            optimizer_state_by_name=self._build_optimizer_state_by_name(),
        )
        self._last_checkpoint_path = str(ckpt_path)
        return ckpt_path

    def _named_parameters_for_optimizer(self):
        """Yield (name, param) pairs across the encoder only."""
        for name, param in self.encoder.named_parameters():
            yield f"encoder.{name}", param

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load encoder state dict and restore training state.

        Raises on optimizer type mismatch (e.g. checkpoint saved with adamw,
        config now says muon). Topology mismatches within the same optimizer
        type (e.g. switching heads_only mode) now preserve per-param
        moments where names match — the name-keyed optimizer state
        refactor replaces the old all-or-nothing native load.
        """
        metadata, state_dicts = load_checkpoint(checkpoint_path)
        self._check_optimizer_type_compat(metadata)
        self.encoder.load_state_dict(state_dicts["encoder"])
        if "optimizer_state_by_name" in state_dicts:
            self._restore_optimizer_state_by_name(
                state_dicts["optimizer_state_by_name"],
            )
        elif not self._legacy_optimizer_fallback(state_dicts):
            logger.warning(
                "optimizer_state_missing_using_fresh_moments",
                hint="checkpoint predates the name-keyed optimizer state refactor",
            )
        self.global_step = metadata.step
        self.epoch = metadata.epoch
        self._last_checkpoint_path = str(checkpoint_path)
        if metadata.schedule_params is not None:
            self._schedule_params = metadata.schedule_params
        if metadata.training_state is not None:
            self._training_state = metadata.training_state
            self._microbatches_in_epoch = int(
                metadata.training_state.get("microbatches_in_epoch", 0),
            )
        logger.info("restored_from_checkpoint", step=self.global_step)
