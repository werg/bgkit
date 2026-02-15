"""ICE trainer: trains the information content estimator.

Trained offline before all other work. Regresses per-token cross-entropy
values from token embedding sequences with a uniformity regularizer.
"""

from __future__ import annotations

import structlog
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoModel

from bgkit.data.datasets.ice_dataset import ICEDataset
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

        # Frozen embedding model
        emb_model_name = self.cfg.model.ice.get(
            "embedding_model", "Qwen/Qwen3-Embedding-0.6B"
        )
        logger.info("loading_embedding_model", model=emb_model_name)
        self.embedding_model = AutoModel.from_pretrained(
            emb_model_name, torch_dtype=torch.bfloat16
        )
        self.embedding_model.to(device)
        self.embedding_model.eval()
        for p in self.embedding_model.parameters():
            p.requires_grad = False

        # ICE model
        ice_cfg = self.cfg.model.ice
        self.model = ICE(
            input_dim=ice_cfg.get("input_dim", 1024),
            hidden_dim=ice_cfg.get("hidden_dim", 256),
            num_layers=ice_cfg.get("num_layers", 4),
            kernel_size=ice_cfg.get("kernel_size", 7),
            dropout=ice_cfg.get("dropout", 0.1),
        )
        self.model.to(device)

        # Dataset + split
        data_dir = self.cfg.data.ice_labels.output_dir
        full_dataset = ICEDataset(data_dir)
        eval_size = max(1, int(len(full_dataset) * 0.1))
        train_size = len(full_dataset) - eval_size
        self.train_dataset, self.eval_dataset = random_split(
            full_dataset, [train_size, eval_size]
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.get("batch_size", 32),
            shuffle=True,
            collate_fn=ice_collate_fn,
            num_workers=self.cfg.get("num_workers", 4),
            pin_memory=(device.type == "cuda"),
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.cfg.get("batch_size", 32),
            shuffle=False,
            collate_fn=ice_collate_fn,
            num_workers=self.cfg.get("num_workers", 4),
            pin_memory=(device.type == "cuda"),
        )

        # Optimizer — ICE params only
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=tcfg.lr)

        self.uniformity_weight = tcfg.get("uniformity_reg_weight", 0.1)

        logger.info(
            "ice_trainer_setup",
            train_samples=train_size,
            eval_samples=eval_size,
            device=str(device),
        )

    def _embed_tokens(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run frozen embedding model to get token embeddings.

        Args:
            token_ids: (B, L) token IDs.
            attention_mask: (B, L) attention mask.

        Returns:
            (B, L, hidden_dim) token embeddings.
        """
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.embedding_model(
                        input_ids=token_ids, attention_mask=attention_mask
                    )
            else:
                outputs = self.embedding_model(
                    input_ids=token_ids, attention_mask=attention_mask
                )
            return outputs.last_hidden_state.float()

    def _align_and_mask(
        self, ce_pred: torch.Tensor, ce_values: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Align ICE predictions with CE targets and build validity mask.

        Args:
            ce_pred: (B, L) raw ICE predictions.
            ce_values: (B, max_ce_len) padded CE targets.
            attention_mask: (B, L) token-level attention mask.

        Returns:
            (ce_pred_aligned, ce_target, ce_mask) all (B, min_len).
        """
        ce_pred_aligned = ce_pred[:, :-1].float()
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

        for batch in self.eval_dataloader:
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
