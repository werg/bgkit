"""Falcon dense-seed trainer running on pre-computed survivor embeddings.

Companion to ``scripts/build_dense_seed_cache.py``. The encoder is frozen and
the forced survivor mask is fixed, so the pre-projection survivor embeddings
are a deterministic function of the source bgkit checkpoint. We pay that
encoder forward ONCE in the cache builder, then this trainer runs the
projection block as a pure feedforward problem against the cached embeddings.

Trainable parameters: ``encoder.projection_blocks.falcon_h1`` (~19M). No
encoder forward, no decoder layers; the Falcon decoder is loaded only to
expose ``embed_tokens`` as the target embedding lookup.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForCausalLM

from bgkit.data.datasets.cached_survivor_dataset import (
    CachedSurvivorDataset,
    collate_cached_dense_seed,
)
from bgkit.data.samplers import PackedTokenBudgetSampler
from bgkit.models.decoder import normalize_decoder_family
from bgkit.models.encoder import BgKITEncoder
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpoint_registry import resolve_checkpoint
from bgkit.training.checkpointing import (
    CheckpointMetadata,
    load_checkpoint,
    save_checkpoint,
)
from bgkit.utils.attention_backend import (
    resolve_attention_implementation,
    resolve_decoder_attention_implementation,
)
from bgkit.utils.packing import position_ids_from_cu

logger = structlog.get_logger()


class FalconProjectionCachedTrainer(BaseTrainer):
    """Train Falcon projection block on cached survivor embeddings."""

    def setup(self) -> None:
        tcfg = self.cfg.training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -------- cache dir --------
        training_data = tcfg.get("data", None) or {}
        cache_dir = (
            training_data.get("cache_dir", None)
            or self.cfg.data.get("falcon_dense_seed_cache_dir", None)
        )
        if not cache_dir:
            raise ValueError(
                "FalconProjectionCachedTrainer requires "
                "training.data.cache_dir (or data.falcon_dense_seed_cache_dir) "
                "to point at the output of scripts/build_dense_seed_cache.py"
            )

        # -------- encoder (for the projection block module + its starting weights) --------
        attention_impl = resolve_attention_implementation(
            self.cfg.compute.get("attention_implementation", "auto")
        )
        bgkit_cfg = self.cfg.model.bgkit
        hidden_dim = int(bgkit_cfg.get("hidden_dim", 1024))
        src_ckpt = self._resolve_source_checkpoint()
        _meta, state_dicts = load_checkpoint(Path(src_ckpt))
        if "encoder" not in state_dicts:
            raise ValueError(f"checkpoint {src_ckpt} missing 'encoder' key")

        projection_num_layers = int(bgkit_cfg.get("projection_num_layers", 1))
        self.encoder = BgKITEncoder.from_pretrained_with_state_dict(
            bgkit_cfg.backbone_name,
            state_dicts["encoder"],
            hidden_dim=hidden_dim,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            revision=bgkit_cfg.get("backbone_revision", None),
            attn_implementation=attention_impl,
            bidi_warmup_steps=0,
            active_decoder_family="falcon_h1",
            projection_num_layers=projection_num_layers,
        ).to(self.device)

        # We only need the projection block from the encoder; freeze everything
        # else, then move only the projection block to GPU under bf16.
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.encoder.projection_blocks["qwen35"].requires_grad_(False)
        proj_block = self.encoder.projection_blocks["falcon_h1"]
        proj_block.requires_grad_(True)
        proj_block.train()

        # -------- Falcon decoder (only its embed_tokens for target lookup) --------
        decoder_cfg = tcfg.get("model", {}).get("decoder", self.cfg.model.decoder)
        decoder_family = normalize_decoder_family(decoder_cfg.get("family", "falcon_h1"))
        if decoder_family != "falcon_h1":
            raise ValueError(
                "FalconProjectionCachedTrainer is Falcon-only "
                f"(got decoder_family={decoder_family!r})"
        )
        decoder_attention_impl = resolve_decoder_attention_implementation(
            self.cfg.compute.get(
                "decoder_attention_implementation",
                self.cfg.compute.get("attention_implementation", "auto"),
            ),
            decoder_family=decoder_family,
        )
        decoder_name = decoder_cfg.get(
            "backbone_name", "tiiuae/Falcon-H1-Tiny-90M-Instruct"
        )
        decoder_backbone = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            revision=decoder_cfg.get("backbone_revision", None),
            attn_implementation=decoder_attention_impl,
        )
        # Keep only the embedding (and move only it to GPU) — the decoder layers
        # cost RAM we won't use.
        self.falcon_embed = decoder_backbone.get_input_embeddings().to(self.device)
        self.falcon_embed.requires_grad_(False)
        self.falcon_embed.eval()
        del decoder_backbone  # free Mamba state etc.

        self.model = proj_block  # what trainer save / restore acts on

        # -------- loss weights --------
        self._anchor_weight = float(tcfg.get("anchor_weight", 1.0))
        self._cos_weight = float(tcfg.get("cos_weight", 0.5))
        self._norm_weight = float(tcfg.get("norm_weight", 0.2))

        # -------- dataset --------
        dataset = CachedSurvivorDataset(cache_dir)
        manifest = dataset.manifest
        # Sanity: cache must have been built against the same source checkpoint
        # we're loading from (otherwise the cached embeddings drift from what
        # the projection_block was last trained against — silently bad).
        if manifest.get("source_checkpoint") and Path(
            manifest["source_checkpoint"]
        ).name != Path(str(src_ckpt)).name:
            logger.warning(
                "cache_source_checkpoint_mismatch",
                cache_built_against=manifest["source_checkpoint"],
                trainer_loading=str(src_ckpt),
                action="continuing — verify this is intentional",
            )

        max_eval_samples = int(tcfg.get("max_eval_samples", 1000))
        eval_size = min(max(1, len(dataset) // 10), max_eval_samples)
        train_size = len(dataset) - eval_size
        generator = torch.Generator().manual_seed(int(self.cfg.get("seed", 42)))
        self.train_dataset, self.eval_dataset = random_split(
            dataset, [train_size, eval_size], generator=generator
        )

        train_lengths = dataset.lengths[np.asarray(self.train_dataset.indices)]
        eval_lengths = dataset.lengths[np.asarray(self.eval_dataset.indices)]
        max_batch_tokens = int(tcfg.get("max_batch_tokens", 2048))
        eval_budget = int(tcfg.get("max_batch_tokens_eval", max_batch_tokens * 2))

        self.train_sampler = PackedTokenBudgetSampler(
            self.train_dataset,
            lengths=train_lengths,
            max_batch_tokens=max_batch_tokens,
            shuffle=True,
            seed=int(self.cfg.get("seed", 42)),
        )
        eval_sampler = PackedTokenBudgetSampler(
            self.eval_dataset,
            lengths=eval_lengths,
            max_batch_tokens=eval_budget,
            shuffle=False,
        )

        num_workers = int(self.cfg.compute.get("num_workers", 2))
        pin_memory = bool(self.cfg.compute.get("pin_memory", False))
        dl_kwargs = {
            "collate_fn": collate_cached_dense_seed,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            dl_kwargs["persistent_workers"] = True
            dl_kwargs["prefetch_factor"] = 2
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            **dl_kwargs,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_sampler=eval_sampler,
            **dl_kwargs,
        )

        # -------- optimizer --------
        proj_params = [p for p in proj_block.parameters() if p.requires_grad]
        lr = float(tcfg.get("projection_lr", tcfg.lr))
        param_groups = [{"params": proj_params, "lr": lr, "base_lr": lr}]
        self.optimizer = self._create_optimizer(
            param_groups,
            lr,
            exclude_from_muon=frozenset(),
        )

        logger.info(
            "falcon_projection_cached_setup",
            source_checkpoint=str(src_ckpt),
            cache_dir=str(cache_dir),
            cache_chunks=manifest.get("n_chunks"),
            cache_survivors=manifest.get("total_survivors"),
            cache_hidden_dim=manifest.get("hidden_dim"),
            cache_source=manifest.get("source_checkpoint"),
            train_chunks=train_size,
            eval_chunks=eval_size,
            projection_params=sum(p.numel() for p in proj_params),
            max_batch_tokens=max_batch_tokens,
        )

    def _resolve_source_checkpoint(self) -> str:
        src = self.cfg.get("bgkit_checkpoint", None)
        if src and src != "auto":
            self._input_sources = {"bgkit": Path(str(src)).name}
            return str(src)
        checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
        candidate_phases = (
            "phase1_step5",
            "phase1_step6",
            "phase1_step2p5",
            "phase1_step2",
        )
        errors: list[str] = []
        for phase in candidate_phases:
            try:
                resolved = resolve_checkpoint(
                    checkpoint_dir,
                    phase=phase,
                    metric="eval/loss",
                    label="bgkit_checkpoint",
                )
                self._input_sources = {"bgkit": resolved.name}
                return str(resolved)
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError(
            "bgkit_checkpoint=auto could not resolve any candidate phase: "
            f"{', '.join(candidate_phases)}. " + " | ".join(errors)
        )

    def trainable_parameters(self) -> list:
        return [
            p
            for p in self.encoder.projection_blocks["falcon_h1"].parameters()
            if p.requires_grad
        ]

    def _named_parameters_for_optimizer(self):
        for name, param in self.encoder.projection_blocks["falcon_h1"].named_parameters():
            yield f"projection_block.{name}", param

    # ------------------------------------------------------------------ forward
    def _losses(
        self,
        projected: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid_proj = projected[mask]
        valid_target = target[mask]
        n = max(int(valid_proj.shape[0]), 1)
        mse = (valid_proj - valid_target).pow(2).sum() / (n * projected.shape[-1])
        cos = F.cosine_similarity(valid_proj.float(), valid_target.float(), dim=-1)
        cos_loss = 1.0 - cos.mean()
        proj_norm = valid_proj.float().norm(dim=-1).clamp(min=1e-6)
        target_norm = valid_target.float().norm(dim=-1).clamp(min=1e-6)
        norm_loss = (torch.log(proj_norm) - torch.log(target_norm)).pow(2).mean()
        total = (
            self._anchor_weight * mse
            + self._cos_weight * cos_loss
            + self._norm_weight * norm_loss
        )
        return total, {
            "anchor_mse": mse.detach(),
            "anchor_cos": cos_loss.detach(),
            "anchor_norm_log": norm_loss.detach(),
            "cos_sim": cos.mean().detach(),
            "norm_ratio": (proj_norm / target_norm).mean().detach(),
        }

    def _forward_one(self, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        proj_block = self.encoder.projection_blocks["falcon_h1"]
        emb = batch["survivor_embeddings"].to(self.device, non_blocking=True)
        cu = batch["cu_seqlens"].to(self.device, non_blocking=True)
        n_total = int(emb.shape[0])
        pos = position_ids_from_cu(cu, n_total)
        # `survivor_mask=None` because all rows are already survivors.
        max_seqlen = int(batch["survivor_counts"].max().item())
        proj_out = proj_block(
            emb,
            cu_seqlens=cu,
            max_seqlen=max_seqlen,
            position_ids=pos,
            survivor_mask=None,
        )
        projected = proj_out.projected_embeddings  # (2*N_total, falcon_dim) for split=2
        target_ids = batch["target_pair_ids"].to(self.device, non_blocking=True)
        target_mask = batch["pair_loss_mask"].to(self.device, non_blocking=True).to(torch.bool)
        if projected.shape[0] != target_ids.shape[0]:
            raise RuntimeError(
                "Projection output / pair target shape mismatch: "
                f"{projected.shape[0]} vs {target_ids.shape[0]}"
            )
        with torch.no_grad():
            target_emb = self.falcon_embed(target_ids)
        total, stats = self._losses(projected, target_emb, target_mask)
        return total, stats

    def _forward_backward(self, batch) -> dict[str, float]:
        proj_block = self.encoder.projection_blocks["falcon_h1"]
        proj_block.train()
        total, stats = self._forward_one(batch)
        metrics: dict[str, torch.Tensor] = {
            "loss": total,
            "loss/anchor_mse": stats["anchor_mse"],
            "loss/anchor_cos": stats["anchor_cos"],
            "loss/anchor_norm_log": stats["anchor_norm_log"],
            "diag/cos_sim": stats["cos_sim"],
            "diag/norm_ratio": stats["norm_ratio"],
            "diag/survivors_per_batch": torch.tensor(
                float(batch["survivor_embeddings"].shape[0])
            ),
            "diag/chunks_per_batch": torch.tensor(
                float(batch["survivor_counts"].shape[0])
            ),
            "diag/alignment_score": batch["alignment_scores"].float().mean(),
        }
        (total / self._accum_steps).backward()
        return {k: float(v.item()) for k, v in metrics.items()}

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        proj_block = self.encoder.projection_blocks["falcon_h1"]
        proj_block.eval()
        totals: dict[str, float] = {}
        count = 0
        try:
            for batch in self.eval_dataloader:
                _total, stats = self._forward_one(batch)
                totals["loss"] = totals.get("loss", 0.0) + float(_total.item())
                for name, value in stats.items():
                    totals[name] = totals.get(name, 0.0) + float(value.item())
                count += 1
            if count:
                totals = {k: v / count for k, v in totals.items()}
            return totals
        finally:
            proj_block.train()

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(
        self,
        checkpoint_dir: Path,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        # Save only the projection_blocks.falcon_h1 sub-state — small (~38 MB),
        # and the encoder.l0/l1 are untouched so we don't need to save them.
        # But for compatibility with downstream phases that resolve from a
        # phase1_falcon_dense_seed checkpoint and expect a full encoder, we
        # save the FULL encoder state dict (the rest is unchanged from src).
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

    def _restore_model_state(self, state_dicts: dict) -> None:
        if "encoder" in state_dicts:
            self.encoder.load_state_dict(state_dicts["encoder"], strict=False)
