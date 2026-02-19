"""Joint block pretraining trainer.

Jointly pretrains the penultimate compressor layer (for auto-reproduction)
and the projection block (for decoder alignment). Two-objective training
loop where gradients flow freely between both objectives.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import structlog
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoModel

from bgkit.data.collators import collate_token_ids
from bgkit.data.datasets.token_chunk_dataset import TokenChunkDataset
from bgkit.data.samplers import TokenBudgetBatchSampler
from bgkit.models.components.auto_reproduction import auto_reproduction_loss
from bgkit.models.encoder import BgKITEncoder, _resolve_layers
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.training.gradient_utils import clip_grad_norm
from bgkit.utils.model_utils import count_parameters, slerp_merge

logger = structlog.get_logger()


def joint_block_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length token ID samples into a batch."""
    return collate_token_ids(batch)


@dataclass
class _ForwardResult:
    """Intermediate forward pass results for both objectives."""

    comp_out: object  # CompressorOutput
    auto_repro_pred: torch.Tensor
    proj_out: object  # ProjectionOutput
    loss_repro: torch.Tensor
    loss_proj: torch.Tensor
    loss: torch.Tensor


class JointBlockTrainer(BaseTrainer):
    """Trainer for joint block pretraining: auto-repro + decoder alignment."""

    def setup(self) -> None:
        """Construct encoder, load decoder embeddings, freeze layers, create dataset."""
        tcfg = self.cfg.training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

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

        # Construct encoder (splits backbone into compressor + projection block)
        self.encoder = BgKITEncoder.from_pretrained(backbone, hidden_dim=hidden_dim)
        self.encoder.to(device)

        # Load decoder's embedding matrix as frozen reference target
        decoder_name = self.cfg.model.decoder.backbone_name
        decoder_revision = self.cfg.model.decoder.get("backbone_revision", None)
        logger.info("loading_decoder_embeddings", model=decoder_name)
        decoder_model = AutoModel.from_pretrained(
            decoder_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=decoder_revision,
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
        full_dataset = TokenChunkDataset(data_dir, max_seq_len=max_seq_len)
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

        train_lengths = [full_dataset.lengths[i] for i in self.train_dataset.indices]
        eval_lengths = [full_dataset.lengths[i] for i in self.eval_dataset.indices]

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
        self.optimizer = torch.optim.AdamW(trainable_params, lr=tcfg.lr)

        logger.info(
            "joint_block_trainer_setup",
            train_samples=train_size,
            eval_samples=eval_size,
            device=str(device),
            w_repro=self.w_repro,
            w_proj=self.w_proj,
        )

    def apply_live_config(self, changes: dict) -> None:
        if "w_repro" in changes:
            self.w_repro = changes["w_repro"]
        if "w_proj" in changes:
            self.w_proj = changes["w_proj"]

    def _get_input_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get input embeddings from the compressor's backbone embedding layer."""
        return self.encoder.compressor.backbone.get_input_embeddings()(token_ids)

    def _amp_context(self):
        """Return autocast context for CUDA, or nullcontext for CPU."""
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _forward_both(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        target_repro: torch.Tensor,
        target_proj: torch.Tensor,
    ) -> _ForwardResult:
        """Run both forward passes and compute losses."""
        comp_out = self.encoder.compressor(
            input_embeddings, survivor_mask=None, attention_mask=attention_mask,
        )

        auto_repro_pred = self.encoder.compressor.auto_reproduce(comp_out.normed_embeddings)
        loss_repro = auto_reproduction_loss(
            auto_repro_pred, target_repro, mask=attention_mask.float(),
        )

        proj_out = self.encoder.projection_block(
            comp_out.raw_embeddings,
            attention_mask=comp_out.attention_mask,
            survivor_mask=None,
        )
        loss_proj = auto_reproduction_loss(
            proj_out.projected_embeddings, target_proj, mask=attention_mask.float(),
        )

        loss = self.w_repro * loss_repro + self.w_proj * loss_proj

        return _ForwardResult(
            comp_out=comp_out,
            auto_repro_pred=auto_repro_pred,
            proj_out=proj_out,
            loss_repro=loss_repro,
            loss_proj=loss_proj,
            loss=loss,
        )

    def train_step(self, batch) -> dict[str, float]:
        self.encoder.train()

        token_ids = batch["token_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Input embeddings -- targets are detached copies
        input_embeddings = self._get_input_embeddings(token_ids)
        target_repro = input_embeddings.detach()
        target_proj = self.decoder_embed(token_ids).detach()

        with self._amp_context():
            fwd = self._forward_both(input_embeddings, attention_mask, target_repro, target_proj)

        # Backward
        self.optimizer.zero_grad()
        fwd.loss.backward()
        grad_norm = clip_grad_norm(
            [p for p in self.encoder.parameters() if p.requires_grad]
        )
        self.optimizer.step()

        # Cosine similarity metrics
        with torch.no_grad():
            mask_f = attention_mask.float()
            cos_repro = F.cosine_similarity(fwd.auto_repro_pred, target_repro, dim=-1)
            cos_repro_avg = (cos_repro * mask_f).sum() / mask_f.sum().clamp(min=1)

            cos_proj = F.cosine_similarity(
                fwd.proj_out.projected_embeddings, target_proj, dim=-1,
            )
            cos_proj_avg = (cos_proj * mask_f).sum() / mask_f.sum().clamp(min=1)

        return {
            "loss": fwd.loss.item(),
            "loss_repro": fwd.loss_repro.item(),
            "loss_proj": fwd.loss_proj.item(),
            "cosine_sim_repro": cos_repro_avg.item(),
            "cosine_sim_proj": cos_proj_avg.item(),
            "grad_norm": grad_norm,
        }

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
                fwd.proj_out.projected_embeddings, target_proj, reduction="none",
            ).mean(dim=-1)
            total_mse_proj += (mse_proj * mask_f).sum().item()

            # Cosine similarity
            cos_repro = F.cosine_similarity(fwd.auto_repro_pred, target_repro, dim=-1)
            total_cosine_repro += (cos_repro * mask_f).sum().item()

            cos_proj = F.cosine_similarity(
                fwd.proj_out.projected_embeddings, target_proj, dim=-1,
            )
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
        )
        ckpt_path = save_checkpoint(
            checkpoint_dir,
            metadata,
            encoder=self.encoder.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )
        self._last_checkpoint_path = str(ckpt_path)
        return ckpt_path

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load encoder state dict and restore training state."""
        metadata, state_dicts = load_checkpoint(checkpoint_path)
        self.encoder.load_state_dict(state_dicts["encoder"])
        if "optimizer" in state_dicts:
            self.optimizer.load_state_dict(state_dicts["optimizer"])
        self.global_step = metadata.step
        self.epoch = metadata.epoch
        self._last_checkpoint_path = str(checkpoint_path)
        if metadata.schedule_params is not None:
            self._schedule_params = metadata.schedule_params
        if metadata.training_state is not None:
            self._training_state = metadata.training_state
        logger.info("restored_from_checkpoint", step=self.global_step)
