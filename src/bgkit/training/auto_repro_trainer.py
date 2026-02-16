"""Auto-reproduction trainer.

Retrains BgKIT's last transformer block (all other layers frozen)
to reproduce per-position input embeddings. Used for source model
selection (embedding model vs SLERP merge).
"""

from __future__ import annotations

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoModel

from bgkit.data.datasets.auto_repro_dataset import AutoReproDataset
from bgkit.models.bgkit_compressor import BgKITCompressor
from bgkit.models.components.auto_reproduction import auto_reproduction_loss
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.gradient_utils import clip_grad_norm
from bgkit.utils.model_utils import count_parameters, slerp_merge

logger = structlog.get_logger()


def auto_repro_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length token ID samples into a batch.

    Args:
        batch: List of dicts with "token_ids" (L,).

    Returns:
        Dict with padded "token_ids" (B, max_L) and "attention_mask" (B, max_L).
    """
    token_ids_list = [s["token_ids"] for s in batch]
    max_len = max(t.size(0) for t in token_ids_list)

    padded_token_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, tids in enumerate(token_ids_list):
        padded_token_ids[i, : tids.size(0)] = tids
        attention_mask[i, : tids.size(0)] = 1

    return {"token_ids": padded_token_ids, "attention_mask": attention_mask}


def _resolve_last_block(backbone: nn.Module) -> nn.Module:
    """Find the last transformer block across HF model variants."""
    for attr in ("layers", "encoder.layers", "transformer.h"):
        obj = backbone
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return obj[-1]
    raise ValueError(
        f"Cannot find transformer layers in {type(backbone).__name__}. "
        f"Known attributes: {[n for n, _ in backbone.named_children()]}"
    )


def _resolve_final_norm(backbone: nn.Module) -> nn.Module:
    """Find the final layer norm across HF model variants."""
    for attr in ("norm", "ln_f", "final_layer_norm"):
        norm = getattr(backbone, attr, None)
        if norm is not None:
            return norm
    raise ValueError(
        f"Cannot find final norm in {type(backbone).__name__}. "
        f"Known attributes: {[n for n, _ in backbone.named_children()]}"
    )


class AutoReproTrainer(BaseTrainer):
    """Trainer for auto-reproduction source model selection."""

    def setup(self) -> None:
        """Load backbone, optionally SLERP merge, freeze layers, create dataset."""
        tcfg = self.cfg.training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load backbone
        backbone_name = self.cfg.model.bgkit.backbone_name
        backbone_revision = self.cfg.model.bgkit.get("backbone_revision", None)
        logger.info("loading_backbone", model=backbone_name, revision=backbone_revision)
        backbone = AutoModel.from_pretrained(
            backbone_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=backbone_revision,
        )

        # Optional SLERP merge with decoder
        slerp_t = tcfg.get("slerp_t", None)
        if slerp_t is not None:
            decoder_name = self.cfg.model.decoder.backbone_name
            decoder_revision = self.cfg.model.decoder.get("backbone_revision", None)
            logger.info("loading_decoder_for_slerp", model=decoder_name, t=slerp_t)
            decoder = AutoModel.from_pretrained(
                decoder_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                revision=decoder_revision,
            )

            sd_a = backbone.state_dict()
            sd_b = decoder.state_dict()

            # Check key overlap
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
            del decoder, sd_a, sd_b, merged_sd

        # Wrap in BgKITCompressor
        hidden_dim = self.cfg.model.bgkit.get("hidden_dim", 1024)
        self.model = BgKITCompressor(backbone, hidden_dim=hidden_dim)
        self.model.to(device)

        # Freeze everything, then unfreeze target components
        self.model.requires_grad_(False)

        # Unfreeze last transformer block
        last_block = _resolve_last_block(self.model.backbone)
        last_block.requires_grad_(True)

        # Unfreeze final layer norm
        final_norm = _resolve_final_norm(self.model.backbone)
        final_norm.requires_grad_(True)

        # Unfreeze auto-repro head
        self.model.auto_repro_head.requires_grad_(True)

        trainable = count_parameters(self.model, trainable_only=True)
        total = count_parameters(self.model, trainable_only=False)
        logger.info("param_counts", trainable=trainable, total=total)

        # Dataset + split
        data_dir = self.cfg.data.tokens.input_dir
        max_seq_len = self.cfg.data.tokens.get("max_seq_len", 8192)
        full_dataset = AutoReproDataset(data_dir, max_seq_len=max_seq_len)
        eval_size = max(1, int(len(full_dataset) * 0.1))
        train_size = len(full_dataset) - eval_size
        if train_size < 1:
            raise ValueError(
                f"Dataset too small for train/eval split (got {len(full_dataset)} samples, "
                "need at least 2)"
            )
        self.train_dataset, self.eval_dataset = random_split(
            full_dataset, [train_size, eval_size]
        )

        batch_size = self.cfg.get("batch_size", 32)
        num_workers = self.cfg.compute.get("num_workers", 4)
        pin_memory = self.cfg.compute.get("pin_memory", False)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=auto_repro_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=auto_repro_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Optimizer — trainable params only
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=tcfg.lr)

        logger.info(
            "auto_repro_trainer_setup",
            train_samples=train_size,
            eval_samples=eval_size,
            device=str(device),
        )

    def _get_input_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get input embeddings from the backbone's embedding layer.

        Args:
            token_ids: (B, L) token IDs.

        Returns:
            (B, L, hidden_dim) input embeddings.
        """
        return self.model.backbone.get_input_embeddings()(token_ids)

    def train_step(self, batch) -> dict[str, float]:
        self.model.train()

        token_ids = batch["token_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Input embeddings — target is detached copy
        input_embeddings = self._get_input_embeddings(token_ids)
        target = input_embeddings.detach()

        # Forward through backbone
        if self.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model.backbone(
                    inputs_embeds=input_embeddings, attention_mask=attention_mask
                )
                predicted = self.model.auto_reproduce(outputs.last_hidden_state)
        else:
            outputs = self.model.backbone(
                inputs_embeds=input_embeddings, attention_mask=attention_mask
            )
            predicted = self.model.auto_reproduce(outputs.last_hidden_state)

        # Loss
        loss = auto_reproduction_loss(predicted, target, mask=attention_mask.float())

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm(
            [p for p in self.model.parameters() if p.requires_grad]
        )
        self.optimizer.step()

        # Cosine similarity metric
        with torch.no_grad():
            cos_sim = F.cosine_similarity(predicted, target, dim=-1)
            mask_f = attention_mask.float()
            masked_cos_sim = (cos_sim * mask_f).sum() / mask_f.sum().clamp(min=1)

        return {
            "loss": loss.item(),
            "cosine_sim": masked_cos_sim.item(),
            "grad_norm": grad_norm,
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()

        total_mse = 0.0
        total_cosine = 0.0
        n = 0.0

        for batch in self.eval_dataloader:
            token_ids = batch["token_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            input_embeddings = self._get_input_embeddings(token_ids)
            target = input_embeddings.detach()

            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model.backbone(
                        inputs_embeds=input_embeddings, attention_mask=attention_mask
                    )
                    predicted = self.model.auto_reproduce(outputs.last_hidden_state)
            else:
                outputs = self.model.backbone(
                    inputs_embeds=input_embeddings, attention_mask=attention_mask
                )
                predicted = self.model.auto_reproduce(outputs.last_hidden_state)

            mask_f = attention_mask.float()
            batch_n = mask_f.sum().item()
            n += batch_n

            # MSE: average over hidden_dim first, then masked mean over positions
            mse = F.mse_loss(predicted, target, reduction="none").mean(dim=-1)
            total_mse += (mse * mask_f).sum().item()

            # Cosine similarity
            cos_sim = F.cosine_similarity(predicted, target, dim=-1)
            total_cosine += (cos_sim * mask_f).sum().item()

        return {
            "mse": total_mse / max(n, 1),
            "cosine_sim": total_cosine / max(n, 1),
        }
