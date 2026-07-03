"""Joint block pretraining trainer — packed FA4 form.

Jointly pretrains L0's auto_reproduction head (the L0→L1 bridge) and the
projection block (for decoder alignment). Two-objective training loop
where gradients flow freely between both objectives.

All inputs are packed: flat ``(N,)`` token IDs, ``cu_seqlens`` for
segmentation, ``position_ids`` for per-sample RoPE. No ``attention_mask``.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

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
from bgkit.data.samplers import PackedTokenBudgetSampler
from bgkit.models.decoder import normalize_decoder_family
from bgkit.models.encoder import BgKITEncoder, _resolve_layers
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpointing import CheckpointMetadata, save_checkpoint
from bgkit.training.gradient_utils import maybe_enable_gradient_checkpointing
from bgkit.utils.attention_backend import (
    resolve_attention_implementation,
    resolve_decoder_attention_implementation,
)
from bgkit.utils.model_utils import count_parameters, slerp_merge
from bgkit.utils.packing import position_ids_from_cu

logger = structlog.get_logger()


def joint_block_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate variable-length token ID samples into a packed batch.

    Returns a dict with packed flat tensors:
    - ``"input_ids"``: ``(N,)`` int64 flat concatenation.
    - ``"position_ids"``: ``(N,)`` int64 per-sample restart.
    - ``"cu_seqlens"``: ``(B+1,)`` int32.
    - ``"max_seqlen"``: int.
    """
    return collate_token_ids(batch)


@dataclass
class _ForwardResult:
    """Intermediate forward pass results for both objectives."""

    comp_out: object  # bgkit.models.encoder.EncoderOutput
    auto_repro_pred: torch.Tensor  # (N_content, D) flat
    proj_content: torch.Tensor  # (N_content, D) flat projected embeddings
    loss_repro: torch.Tensor
    loss_proj: torch.Tensor
    loss: torch.Tensor


class JointBlockTrainer(BaseTrainer):
    """Trainer for joint block pretraining: auto-repro + decoder alignment."""

    LIVE_CONFIG_FIELDS: ClassVar[dict] = {
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
        decoder_family = normalize_decoder_family(
            self.cfg.model.decoder.get("family", "qwen35")
        )
        decoder_attention_impl = resolve_decoder_attention_implementation(
            self.cfg.compute.get(
                "decoder_attention_implementation",
                self.cfg.compute.get("attention_implementation", "auto"),
            ),
            decoder_family=decoder_family,
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
                attn_implementation=decoder_attention_impl,
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

        # Construct encoder (splits backbone into l0/l1 LevelCompressors + projection block).
        # Stay fully causal (-1): frozen backbone layers can't adapt to bidi masks,
        # and the task is too trivial to benefit. Step 2 handles the gradual warmup.
        self.encoder = BgKITEncoder.from_pretrained(
            backbone, hidden_dim=hidden_dim, bidi_warmup_steps=-1,
        )
        self.encoder.to(device)

        # Config-gated checkpointing on both backbones. Defaults come from
        # compute config; phases can override at training scope.
        maybe_enable_gradient_checkpointing(self.encoder.l0.backbone, self.cfg)
        maybe_enable_gradient_checkpointing(self.encoder.l1.backbone, self.cfg)

        # Load decoder's embedding matrix as frozen reference target
        decoder_name = self.cfg.model.decoder.backbone_name
        decoder_revision = self.cfg.model.decoder.get("backbone_revision", None)
        logger.info("loading_decoder_embeddings", model=decoder_name)
        decoder_model = AutoModel.from_pretrained(
            decoder_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=decoder_revision,
            attn_implementation=decoder_attention_impl,
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
            self.encoder.l0.auto_repro_head.requires_grad_(True)
            self.encoder.projection_block.projection_head.requires_grad_(True)
        else:
            # Train heads + transformer blocks + norms (L0 only — joint block
            # pretraining does not exercise L1; L1 inherits L0's weights at the
            # subsequent split-encoder construction).
            l0_layers = _resolve_layers(self.encoder.l0.backbone)
            l0_layers[-1].requires_grad_(True)
            self.encoder.l0.backbone.norm.requires_grad_(True)
            self.encoder.l0.auto_repro_head.requires_grad_(True)
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
        split_generator = torch.Generator().manual_seed(int(self.cfg.get("seed", 42)))
        self.train_dataset, self.eval_dataset = random_split(
            full_dataset, [train_size, eval_size], generator=split_generator
        )

        max_batch_tokens = tcfg.get("max_batch_tokens", 65536)
        # Eval defaults to 2x train budget (no backward -> lower peak at
        # same budget). Overridable via training.max_batch_tokens_eval.
        max_batch_tokens_eval = self._resolve_eval_batch_budget(tcfg, max_batch_tokens)
        num_workers = self.cfg.compute.get("num_workers", 4)
        pin_memory = self.cfg.compute.get("pin_memory", False)

        train_lengths = full_dataset.lengths[np.array(self.train_dataset.indices)]
        eval_lengths = full_dataset.lengths[np.array(self.eval_dataset.indices)]

        # Stash for live-tunable budget rebuild (see BaseTrainer._handle_max_batch_tokens)
        self._train_lengths = train_lengths
        self._eval_lengths = eval_lengths
        self._train_collate_fn = joint_block_collate_fn
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._max_batch_tokens = max_batch_tokens
        self._max_batch_tokens_eval = max_batch_tokens_eval

        self.train_sampler = PackedTokenBudgetSampler(
            None, train_lengths, max_batch_tokens, shuffle=True, seed=self.cfg.get("seed", 42),
        )
        eval_sampler = PackedTokenBudgetSampler(
            None, eval_lengths, max_batch_tokens_eval, shuffle=False,
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

        # Pre-compute and freeze the prompt embeddings (prefix_len, hidden_dim)
        with torch.no_grad():
            self._prompt_embeddings = self._get_input_embeddings(prefix_ids)
            self._prompt_len = int(prefix_ids.size(0))

        logger.info(
            "joint_block_trainer_setup",
            train_samples=train_size,
            eval_samples=eval_size,
            device=str(device),
            w_repro=self.w_repro,
            w_proj=self.w_proj,
        )

    def _get_input_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get input embeddings from L0's backbone embedding layer."""
        return self.encoder.l0.backbone.get_input_embeddings()(token_ids)

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
            self._prompt_embeddings = self._get_input_embeddings(prefix_ids)
            self._prompt_len = int(prefix_ids.size(0))
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
        """Return cached ChatML prefix embeddings (prefix_len, D), or None."""
        return getattr(self, "_prompt_embeddings", None)

    def _build_prompt_pack(
        self, num_samples: int, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Build packed prompt tensors tiled for B samples.

        Returns:
            Tuple of (prompt_embeddings, prompt_cu_seqlens, prompt_position_ids)
            each ready for BgKITEncoder.forward, or (None, None, None) if no prompt.
        """
        prompt_emb = self._get_prompt_embeddings()
        if prompt_emb is None:
            return None, None, None

        # prompt_emb: (prefix_len, D) — tile num_samples times into flat buffer
        prefix_len = prompt_emb.size(0)
        prompt_flat = (
            prompt_emb.unsqueeze(0)
            .expand(num_samples, -1, -1)
            .reshape(num_samples * prefix_len, -1)
        )

        # cu_seqlens: [0, p, 2p, ..., num_samples*p]
        lengths = torch.full((num_samples,), prefix_len, dtype=torch.int32, device=device)
        prompt_cu = torch.zeros(num_samples + 1, dtype=torch.int32, device=device)
        torch.cumsum(lengths, dim=0, out=prompt_cu[1:])

        # position_ids: per-sample restart [0..p-1, 0..p-1, ...]
        prompt_pos = position_ids_from_cu(prompt_cu, num_samples * prefix_len)

        return prompt_flat.to(device), prompt_cu, prompt_pos

    def _forward_both(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_ids: torch.Tensor,
        target_repro: torch.Tensor,
        target_proj: torch.Tensor,
    ) -> _ForwardResult:
        """Run both forward passes and compute losses (packed).

        Args:
            input_ids: ``(N,)`` int64 flat token IDs.
            cu_seqlens: ``(B+1,)`` int32.
            max_seqlen: int.
            position_ids: ``(N,)`` int64 per-sample positions.
            target_repro: ``(N, D)`` target for auto-repro objective.
            target_proj: ``(N, D)`` target for projection objective.

        Returns:
            :class:`_ForwardResult` with losses and metric tensors.
        """
        num_samples = int(cu_seqlens.shape[0]) - 1
        content_embeddings = self._get_input_embeddings(input_ids)

        prompt_emb, prompt_cu, prompt_pos = self._build_prompt_pack(
            num_samples, input_ids.device
        )

        comp_out = self.encoder(
            content_embeddings=content_embeddings,
            content_cu_seqlens=cu_seqlens,
            content_position_ids=position_ids,
            prompt_embeddings=prompt_emb,
            prompt_cu_seqlens=prompt_cu,
            prompt_position_ids=prompt_pos,
            target_ratio_l0=None,  # no compression in joint-block pretraining
            target_ratio_l1=None,
        )

        # auto_repro_pred: (N_content, D) — flat over all content tokens.
        # L0's content_embeddings carries the post-norm last-block hidden states.
        auto_repro_pred = self.encoder.l0.auto_reproduce(comp_out.l0.content_embeddings)

        # proj_content: (N_content, D) — survivor_embeddings when no compression
        # equals all content positions (survivor_cu_seqlens == content_cu_seqlens).
        proj_content = comp_out.survivor_embeddings

        # MSE losses over flat positions — no mask needed (no padding)
        loss_repro = F.mse_loss(auto_repro_pred, target_repro)
        loss_proj = F.mse_loss(proj_content, target_proj)

        loss = self.w_repro * loss_repro + self.w_proj * loss_proj

        return _ForwardResult(
            comp_out=comp_out,
            auto_repro_pred=auto_repro_pred,
            proj_content=proj_content,
            loss_repro=loss_repro,
            loss_proj=loss_proj,
            loss=loss,
        )

    def trainable_parameters(self) -> list:
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def _forward_backward(self, batch) -> dict[str, float]:
        self.encoder.train()

        input_ids = batch["input_ids"].to(self.device)
        cu_seqlens = batch["cu_seqlens"].to(self.device)
        max_seqlen = int(batch["max_seqlen"])
        position_ids = batch["position_ids"].to(self.device)

        # Input embeddings -- targets are detached copies
        content_embeddings = self._get_input_embeddings(input_ids)
        target_repro = content_embeddings.detach()
        target_proj = self.decoder_embed(input_ids).detach()

        with self._amp_context():
            fwd = self._forward_both(
                input_ids, cu_seqlens, max_seqlen, position_ids, target_repro, target_proj,
            )

        # Scaled backward (for gradient accumulation)
        (fwd.loss / self._accum_steps).backward()

        # Cosine similarity metrics (flat, no padding to mask)
        with torch.no_grad():
            cos_repro = F.cosine_similarity(fwd.auto_repro_pred, target_repro, dim=-1).mean()
            cos_proj = F.cosine_similarity(fwd.proj_content, target_proj, dim=-1).mean()

        metrics = {
            "loss": fwd.loss.detach(),
            "loss_repro": fwd.loss_repro.detach(),
            "loss_proj": fwd.loss_proj.detach(),
            "cosine_sim_repro": cos_repro.detach(),
            "cosine_sim_proj": cos_proj.detach(),
        }

        # Drop EncoderOutput tensor refs per release() contract.
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

            input_ids = batch["input_ids"].to(self.device)
            cu_seqlens = batch["cu_seqlens"].to(self.device)
            max_seqlen = int(batch["max_seqlen"])
            position_ids = batch["position_ids"].to(self.device)

            content_embeddings = self._get_input_embeddings(input_ids)
            target_repro = content_embeddings.detach()
            target_proj = self.decoder_embed(input_ids).detach()

            with self._amp_context():
                fwd = self._forward_both(
                    input_ids, cu_seqlens, max_seqlen, position_ids, target_repro, target_proj,
                )

            # Accumulate over flat tokens (num_tokens per batch)
            n += float(input_ids.size(0))

            mse_repro = F.mse_loss(fwd.auto_repro_pred, target_repro, reduction="none").mean(
                dim=-1
            )
            total_mse_repro += mse_repro.sum().item()

            mse_proj = F.mse_loss(fwd.proj_content, target_proj, reduction="none").mean(dim=-1)
            total_mse_proj += mse_proj.sum().item()

            cos_repro = F.cosine_similarity(fwd.auto_repro_pred, target_repro, dim=-1)
            total_cosine_repro += cos_repro.sum().item()

            cos_proj = F.cosine_similarity(fwd.proj_content, target_proj, dim=-1)
            total_cosine_proj += cos_proj.sum().item()

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
            run_name=self.cfg.get("run_name", None),
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

    def _restore_model_state(self, state_dicts: dict) -> None:
        """Load encoder state dict.

        Topology mismatches within the same optimizer type (e.g.
        switching heads_only mode) preserve per-param moments where
        names match via the name-keyed optimizer state path — this
        load uses strict matching for the encoder weights themselves.
        """
        self.encoder.load_state_dict(state_dicts["encoder"])
