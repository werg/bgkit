"""ICE trainer: trains the information content estimator.

Trained offline before all other work. Regresses per-token cross-entropy
values from token embedding sequences with a uniformity regularizer.
"""

from __future__ import annotations

import numpy as np
import structlog
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoModel

from bgkit.data.datasets.mmap_ice_dataset import MmapICEDataset
from bgkit.data.samplers import TokenBudgetBatchSampler
from bgkit.models.ice import ICE
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.gradient_utils import clip_grad_norm

logger = structlog.get_logger()


def ice_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length ICE samples into a batch.

    Args:
        batch: List of dicts with "token_ids" (L,) and "ce_values" (L-1,).

    Returns:
        Dict with padded "token_ids" (B, max_L), "ce_values" (B, max_L-1),
        and "attention_mask" (B, max_L).
    """
    token_ids_list = [s["token_ids"] for s in batch]
    ce_values_list = [s["ce_values"] for s in batch]

    max_token_len = max(t.size(0) for t in token_ids_list)
    max_ce_len = max(c.size(0) for c in ce_values_list)

    padded_token_ids = torch.zeros(len(batch), max_token_len, dtype=torch.long)
    padded_ce_values = torch.zeros(len(batch), max_ce_len, dtype=torch.float32)
    attention_mask = torch.zeros(len(batch), max_token_len, dtype=torch.long)

    for i, (tids, cev) in enumerate(zip(token_ids_list, ce_values_list, strict=True)):
        padded_token_ids[i, : tids.size(0)] = tids
        padded_ce_values[i, : cev.size(0)] = cev
        attention_mask[i, : tids.size(0)] = 1

    return {
        "token_ids": padded_token_ids,
        "ce_values": padded_ce_values,
        "attention_mask": attention_mask,
    }


class ICETrainer(BaseTrainer):
    """Trainer for the ICE convolutional predictor."""

    def setup(self) -> None:
        """Load frozen embedding model, create ICE, dataset, optimizer."""
        tcfg = self.cfg.training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Frozen embedding layer (table lookup only — matches inference in CompressionTrainer)
        emb_model_name = self.cfg.model.ice.get(
            "embedding_model", "Qwen/Qwen3-Embedding-0.6B"
        )
        emb_revision = self.cfg.model.ice.get("embedding_model_revision", None)
        logger.info("loading_embedding_layer", model=emb_model_name, revision=emb_revision)
        emb_model = AutoModel.from_pretrained(
            emb_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=emb_revision,
        )
        self.embedding_layer = emb_model.get_input_embeddings()
        self.embedding_layer.to(device)
        self.embedding_layer.requires_grad_(False)
        del emb_model  # free the transformer layers

        # ICE model
        ice_cfg = self.cfg.model.ice
        self.model = ICE(
            input_dim=ice_cfg.get("input_dim", 1024),
            hidden_dim=ice_cfg.get("hidden_dim", 128),
            num_layers=ice_cfg.get("num_layers", 2),
            kernel_size=ice_cfg.get("kernel_size", 5),
            dropout=ice_cfg.get("dropout", 0.1),
        )
        self.model.to(device)

        # Dataset + split
        data_dir = self.cfg.data.ice_labels.output_dir
        full_dataset = MmapICEDataset(data_dir)
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

        # Token-budget batching: pack samples so batch_size * max_seq_len <= budget
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
            collate_fn=ice_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_sampler=eval_sampler,
            collate_fn=ice_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Optimizer — ICE params only
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=tcfg.lr)

        self.uniformity_weight = tcfg.get("uniformity_reg_weight", 0.1)
        self.uniform_reg_weight = tcfg.get("uniform_output_reg_weight", 0.0)

        logger.info(
            "ice_trainer_setup",
            train_samples=train_size,
            eval_samples=eval_size,
            device=str(device),
        )

    def apply_live_config(self, changes: dict) -> None:
        if "uniformity_reg_weight" in changes:
            self.uniformity_weight = changes["uniformity_reg_weight"]
        if "uniform_output_reg_weight" in changes:
            self.uniform_reg_weight = changes["uniform_output_reg_weight"]

    def _embed_tokens(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings from frozen embedding table.

        Args:
            token_ids: (B, L) token IDs.
            attention_mask: (B, L) attention mask (unused, kept for API compat).

        Returns:
            (B, L, embedding_dim) token embeddings.
        """
        with torch.no_grad():
            return self.embedding_layer(token_ids).float()

    def _align_and_mask(
        self, ce_pred: torch.Tensor, ce_values: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Align ICE predictions with CE targets and build validity mask.

        ICE(pos_i) predicts CE(token_i). Position 0 has no CE target (no prior
        context), so we drop it: ce_pred[:, 1:] is paired with ce_values[:, :].

        Args:
            ce_pred: (B, L) raw ICE predictions.
            ce_values: (B, max_ce_len) padded CE targets (ce_values[k] = CE of token k+1).
            attention_mask: (B, L) token-level attention mask.

        Returns:
            (ce_pred_aligned, ce_target, ce_mask) all (B, min_len).
        """
        ce_pred_aligned = ce_pred[:, 1:].float()
        ce_mask = attention_mask[:, 1:].float()
        min_len = min(ce_pred_aligned.size(1), ce_values.size(1))
        return (
            ce_pred_aligned[:, :min_len],
            ce_values[:, :min_len],
            ce_mask[:, :min_len],
        )

    def train_step(self, batch) -> dict[str, float]:
        self.model.train()

        token_ids = batch["token_ids"].to(self.device)
        ce_values = batch["ce_values"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Get embeddings from frozen model
        embeddings = self._embed_tokens(token_ids, attention_mask)

        # Forward ICE
        if self.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ce_pred = self.model(embeddings)  # (B, L)
        else:
            ce_pred = self.model(embeddings)

        # Align predictions with targets
        ce_pred_aligned, ce_target, ce_mask = self._align_and_mask(
            ce_pred, ce_values, attention_mask
        )

        # MSE loss (masked)
        sq_err = (ce_pred_aligned - ce_target) ** 2
        mse_loss = (sq_err * ce_mask).sum() / ce_mask.sum().clamp(min=1)

        # Uniformity regularizer: penalize low variance in predictions
        # Only compute over valid positions
        masked_pred = ce_pred_aligned * ce_mask
        pred_mean = masked_pred.sum() / ce_mask.sum().clamp(min=1)
        pred_var = (
            ((masked_pred - pred_mean * ce_mask) ** 2 * ce_mask).sum()
            / ce_mask.sum().clamp(min=1)
        )
        uniformity_loss = -pred_var

        loss = mse_loss + self.uniformity_weight * uniformity_loss

        # Uniform-output regularization on random embeddings
        uniform_output_loss_val = 0.0
        rand_pred_var_val = 0.0
        if self.uniform_reg_weight > 0:
            rand_embeddings = torch.randn_like(embeddings)
            with torch.no_grad():
                valid_emb = embeddings[attention_mask.bool()]
                emb_norm = (
                    valid_emb.norm(dim=-1).mean()
                    if valid_emb.numel() > 0
                    else embeddings.new_tensor(1.0)
                )
                rand_embeddings = rand_embeddings * emb_norm / (
                    rand_embeddings.norm(dim=-1, keepdim=True) + 1e-8
                )
            rand_pred = self.model(rand_embeddings)
            rand_valid = rand_pred[attention_mask.bool()]
            uniform_output_loss = (
                rand_valid.var(correction=0)
                if rand_valid.numel() > 0
                else rand_pred.new_tensor(0.0)
            )
            loss = loss + self.uniform_reg_weight * uniform_output_loss
            uniform_output_loss_val = uniform_output_loss.item()
            rand_pred_var_val = uniform_output_loss_val

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm(self.model.parameters())
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mse_loss": mse_loss.item(),
            "uniformity_loss": uniformity_loss.item(),
            "pred_variance": pred_var.item(),
            "grad_norm": grad_norm,
            "uniform_output_loss": uniform_output_loss_val,
            "rand_pred_variance": rand_pred_var_val,
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()

        # Vectorized accumulators for MSE and Pearson correlation
        total_mse = 0.0
        n = 0.0
        sum_p = 0.0
        sum_t = 0.0
        sum_p2 = 0.0
        sum_t2 = 0.0
        sum_pt = 0.0

        num_batches = len(self.eval_dataloader)
        for batch_idx, batch in enumerate(self.eval_dataloader):
            if batch_idx % 100 == 0:
                logger.info("eval_progress", batch=batch_idx, total=num_batches)
            token_ids = batch["token_ids"].to(self.device)
            ce_values = batch["ce_values"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            embeddings = self._embed_tokens(token_ids, attention_mask)

            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ce_pred = self.model(embeddings)
            else:
                ce_pred = self.model(embeddings)

            ce_pred_aligned, ce_target, ce_mask = self._align_and_mask(
                ce_pred, ce_values, attention_mask
            )

            sq_err = (ce_pred_aligned - ce_target) ** 2
            total_mse += (sq_err * ce_mask).sum().item()

            # Batch-level accumulation for Pearson (masked)
            batch_n = ce_mask.sum().item()
            n += batch_n
            sum_p += (ce_pred_aligned * ce_mask).sum().item()
            sum_t += (ce_target * ce_mask).sum().item()
            sum_p2 += (ce_pred_aligned**2 * ce_mask).sum().item()
            sum_t2 += (ce_target**2 * ce_mask).sum().item()
            sum_pt += (ce_pred_aligned * ce_target * ce_mask).sum().item()

        mse = total_mse / max(n, 1)

        if n > 1:
            # Pearson from sufficient statistics
            mean_p = sum_p / n
            mean_t = sum_t / n
            var_p = sum_p2 / n - mean_p**2
            var_t = sum_t2 / n - mean_t**2
            cov_pt = sum_pt / n - mean_p * mean_t
            pearson = cov_pt / (max(var_p, 0) ** 0.5 * max(var_t, 0) ** 0.5 + 1e-8)
        else:
            var_p = 0.0
            pearson = 0.0

        return {
            "mse": mse,
            "pearson": pearson,
            "pred_variance": var_p,
        }
