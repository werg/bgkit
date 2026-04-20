"""Phase 1 Step 2.5: projection embed-anchor repair.

Targeted follow-up to Step 2. Re-trains ONLY the encoder's projection
block so that its output lands on the decoder's token-embedding
manifold, fixing the large norm / orthogonal direction drift discovered
by ``scripts/analyze_embedding_deviation.py`` (projected output has
‖q‖ ≈ 112× the decoder embedding norm and cosine ≈ 0 to
``decoder.embed_tokens``).

Rationale: Step 1/2's only supervision for the projection is end-to-end
CE through the decoder (Step 1) plus student↔teacher proj MSE (Step 2).
Neither anchors the projection to ``decoder.embed_tokens``. The decoder
adapted by rescaling off-manifold vectors in its lower layers. The
diagnostic's identity upper bound (feed ``decoder.embed_tokens(ids)``
as survivors → CE ≈ 0.01) confirms the decoder still processes
on-manifold embeddings well, so re-training only the projection is
sufficient — we do not need to re-train the decoder.

Design:

- Load encoder + decoder from a Step 2 (or later) checkpoint.
- Freeze compressor + decoder entirely. Only ``projection_block``
  parameters train (~19 M).
- Primary loss: ``MSE(proj_output, decoder.embed_tokens(content_ids))``
  at every valid content position. This is the embed-anchor.
- Secondary loss (optional, weight ``ce_weight`` in config): end-to-end
  reconstruction CE through the frozen decoder, identical to Step 1's
  objective. Keeps the projection's output useful for reconstruction
  even if the per-token identity target is slightly suboptimal (e.g.
  loss of aggregation-into-survivors).
- Runs at ``target_ratio=None`` (no compression). The anchor target is
  1:1 with content positions; compression is a second-order concern to
  revisit once the manifold is reached.

Produces a drop-in encoder checkpoint; the decoder state_dict is
passed through unchanged so downstream Step 3/4/5/6 trainers can
resolve it like any other Step-2-family checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import structlog
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer

from bgkit.data.collators import collate_chat_repro
from bgkit.data.datasets.chat_repro_dataset import ChatReproDataset
from bgkit.data.datasets.mmap_token_dataset import MmapTokenDataset
from bgkit.data.samplers import PackedTokenBudgetSampler
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpoint_registry import resolve_checkpoint
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.training.gradient_utils import enable_gradient_checkpointing
from bgkit.utils.attention_backend import resolve_attention_implementation

logger = structlog.get_logger()


class ProjectionRepairTrainer(BaseTrainer):
    """Phase 1 Step 2.5: projection-only retrain with decoder-embed anchor."""

    LIVE_CONFIG_FIELDS: ClassVar[dict[str, str]] = {
        "anchor_weight": "_anchor_weight",
        "cos_weight": "_cos_weight",
        "norm_weight": "_norm_weight",
        "ce_weight": "_ce_weight",
    }

    def setup(self) -> None:
        tcfg = self.cfg.training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        attention_impl = resolve_attention_implementation(
            self.cfg.compute.get("attention_implementation", "auto")
        )

        # --- Resolve source checkpoint (Step 2 family) ---
        src_ckpt = self._resolve_source_checkpoint()
        if src_ckpt is None:
            raise ValueError(
                "phase1_step2p5 requires bgkit_checkpoint set (path or 'auto')."
            )

        # --- Load encoder from checkpoint (auto-detects pruned arch) ---
        bgkit_cfg = self.cfg.model.bgkit
        backbone_name = bgkit_cfg.backbone_name
        hidden_dim = bgkit_cfg.get("hidden_dim", 1024)

        logger.info("loading_source_checkpoint", path=src_ckpt)
        metadata, state_dicts = load_checkpoint(Path(src_ckpt))
        if "encoder" not in state_dicts:
            raise ValueError(f"checkpoint {src_ckpt} missing 'encoder' key")

        self.encoder = BgKITEncoder.from_pretrained_with_state_dict(
            backbone_name,
            state_dicts["encoder"],
            hidden_dim=hidden_dim,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=bgkit_cfg.get("backbone_revision", None),
            attn_implementation=attention_impl,
            bidi_warmup_steps=0,
        )
        self.encoder.to(device)
        # Freeze everything; unfreeze projection_block (and optionally
        # compressor final blocks) below.
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        # Projection block trains — keep it in train() mode for dropout if any.
        self.encoder.projection_block.requires_grad_(True)
        self.encoder.projection_block.train()

        # Optionally unfreeze the last N blocks of the compressor
        # backbone. Projection-only repair plateaus around cos_sim ≈ 0.3
        # (see 2026-04-17 run #1/#2) — the compressor's frozen features
        # don't contain enough rotation-ready structure for a single
        # transformer layer to align them with decoder.embed_tokens.
        # Unfreezing the last block(s) gives the repair more capacity.
        n_unfreeze = int(tcfg.get("unfreeze_compressor_final_blocks", 0))
        self._compressor_trainable_params: list = []
        if n_unfreeze > 0:
            backbone = self.encoder.compressor.backbone
            if not hasattr(backbone, "blocks"):
                raise ValueError(
                    "unfreeze_compressor_final_blocks requires a pruned "
                    "backbone with .blocks attribute"
                )
            blocks = backbone.blocks
            unfrozen_blocks = list(blocks)[-n_unfreeze:]
            for b in unfrozen_blocks:
                b.requires_grad_(True)
                b.train()
                self._compressor_trainable_params.extend(
                    p for p in b.parameters() if p.requires_grad
                )
            # Also enable gradient checkpointing on the compressor backbone
            # so the unfrozen blocks' activations are recomputed rather
            # than stored — the frozen blocks don't need activations for
            # backward, but PyTorch still allocates them without ckpt.
            enable_gradient_checkpointing(backbone)
            logger.info(
                "compressor_final_blocks_unfrozen",
                n_unfrozen=n_unfreeze,
                n_params=sum(p.numel() for p in self._compressor_trainable_params),
            )

        # Stash decoder state for checkpoint pass-through.
        self._decoder_state_dict = state_dicts.get("decoder", None)

        # --- Decoder: load fresh weights then overlay from checkpoint ---
        decoder_cfg = self.cfg.model.decoder
        decoder_name = decoder_cfg.backbone_name
        logger.info("loading_decoder", model=decoder_name)
        decoder_backbone = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=decoder_cfg.get("backbone_revision", None),
            attn_implementation=attention_impl,
            device_map=device,
        )
        self.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=hidden_dim)
        if self._decoder_state_dict is not None:
            self.decoder.load_state_dict(self._decoder_state_dict)
            logger.info("loaded_decoder_from_checkpoint")
        self.decoder.requires_grad_(False)
        self.decoder.eval()

        # Optional Liger kernels (matches other Phase 1 steps).
        if tcfg.get("use_liger", True):
            from bgkit.utils.liger_integration import apply_liger_to_qwen35

            patch_rmsnorm = bool(tcfg.get("use_liger_rmsnorm", False))
            patch_swiglu = bool(tcfg.get("use_liger_swiglu", True))
            patch_rope = bool(tcfg.get("use_liger_rope", True))
            apply_liger_to_qwen35(
                self.encoder,
                patch_rmsnorm=patch_rmsnorm,
                patch_swiglu=patch_swiglu,
                patch_rope=patch_rope,
            )
            apply_liger_to_qwen35(
                self.decoder,
                patch_rmsnorm=patch_rmsnorm,
                patch_swiglu=patch_swiglu,
                patch_rope=patch_rope,
            )
            if bool(tcfg.get("use_liger_ce", True)):
                self.decoder.enable_liger_ce(True)

        # Required by BaseTrainer for logging.
        self.model = self.encoder

        # --- Tokenizer + dataset ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            decoder_name,
            trust_remote_code=True,
            revision=decoder_cfg.get("backbone_revision", None),
        )

        data_dir = self.cfg.data.tokens.input_dir
        max_seq_len = self.cfg.data.tokens.get("max_seq_len", 8192)
        variant_bank_path = self.cfg.data.tokens.variant_bank_path

        inner_dataset = MmapTokenDataset(
            data_dir, max_seq_len=max_seq_len, include_metadata=True,
        )
        full_dataset = ChatReproDataset(
            inner_dataset, tokenizer=self.tokenizer,
            variant_bank_path=variant_bank_path,
            seed=self.cfg.get("seed", 42),
        )

        max_eval_samples = tcfg.get("max_eval_samples", 1000)
        eval_size = min(max(1, int(len(full_dataset) * 0.1)), max_eval_samples)
        train_size = len(full_dataset) - eval_size
        self.train_dataset, self.eval_dataset = random_split(
            full_dataset, [train_size, eval_size],
        )

        max_batch_tokens = tcfg.get("max_batch_tokens", 32768)
        # Eval has no backward — its packed budget can be larger than
        # training's. Falls back to ``max_batch_tokens`` when unset.
        max_batch_tokens_eval = tcfg.get("max_batch_tokens_eval", max_batch_tokens)
        num_workers = self.cfg.compute.get("num_workers", 4)
        pin_memory = self.cfg.compute.get("pin_memory", False)
        seed = self.cfg.get("seed", 42)

        train_lengths = full_dataset.lengths[np.array(self.train_dataset.indices)]
        eval_lengths = full_dataset.lengths[np.array(self.eval_dataset.indices)]
        self.train_sampler = PackedTokenBudgetSampler(
            self.train_dataset,
            lengths=train_lengths,
            max_batch_tokens=max_batch_tokens,
            shuffle=True,
            seed=seed,
        )
        eval_sampler = PackedTokenBudgetSampler(
            self.eval_dataset,
            lengths=eval_lengths,
            max_batch_tokens=max_batch_tokens_eval,
            shuffle=False,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=collate_chat_repro,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_sampler=eval_sampler,
            collate_fn=collate_chat_repro,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # --- Loss weights (live-tunable) ---
        self._anchor_weight = float(tcfg.get("anchor_weight", 1.0))
        self._cos_weight = float(tcfg.get("cos_weight", 0.0))
        self._norm_weight = float(tcfg.get("norm_weight", 0.0))
        self._ce_weight = float(tcfg.get("ce_weight", 0.0))

        # --- Optimizer ---
        proj_params = [
            p for p in self.encoder.projection_block.parameters() if p.requires_grad
        ]
        proj_lr = float(tcfg.get("projection_lr", tcfg.lr))
        param_groups = [{"params": proj_params, "lr": proj_lr, "base_lr": proj_lr}]
        # Separate param group for the unfrozen compressor blocks at a
        # lower LR (they carry pretrained structure — don't want to
        # destabilize them).
        if self._compressor_trainable_params:
            compressor_lr = float(tcfg.get("compressor_lr", proj_lr * 0.1))
            param_groups.append({
                "params": self._compressor_trainable_params,
                "lr": compressor_lr,
                "base_lr": compressor_lr,
            })
            logger.info(
                "compressor_param_group_added",
                compressor_lr=compressor_lr,
                proj_lr=proj_lr,
            )
        self.optimizer = self._create_optimizer(
            param_groups, proj_lr, exclude_from_muon=frozenset(),
        )

        # Cache decoder embedding module (stable; no LoRA yet at this step).
        inner, _lm_head = self.decoder._get_inner_model_and_head()
        self._decoder_embed = inner.get_input_embeddings()

        logger.info(
            "projection_repair_setup",
            src_checkpoint=src_ckpt,
            projection_params=sum(p.numel() for p in proj_params),
            train_samples=train_size,
            eval_samples=eval_size,
            anchor_weight=self._anchor_weight,
            cos_weight=self._cos_weight,
            norm_weight=self._norm_weight,
            ce_weight=self._ce_weight,
        )

    # ------------------------------------------------------------------
    # Checkpoint resolution
    # ------------------------------------------------------------------

    def _resolve_source_checkpoint(self) -> str | None:
        src = self.cfg.get("bgkit_checkpoint", None)
        if src == "auto":
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            resolved = resolve_checkpoint(
                checkpoint_dir,
                phase="phase1_step2",
                metric="eval/loss",
                label="bgkit_checkpoint",
            )
            src = str(resolved)
        self._input_sources = {"bgkit": Path(src).name} if src else {}
        return src

    def trainable_parameters(self) -> list:
        params = [
            p for p in self.encoder.projection_block.parameters() if p.requires_grad
        ]
        params.extend(self._compressor_trainable_params)
        return params

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _encoder_forward_nocomp(self, batch: dict):
        """Run the (frozen) compressor + (trainable) projection at ratio=None.

        Packed inputs: flat ``(N_content, D)`` content embeddings +
        ``content_cu_seqlens``. Returns the encoder output, flat content
        token IDs, and the content segmentation for downstream reductions.
        """
        device = self.device
        content_token_ids = batch["content_token_ids"].to(device)
        content_cu = batch["content_cu_seqlens"].to(device)
        content_position_ids = batch["content_position_ids"].to(device)
        prompt_ids = batch["compression_prompt_ids"].to(device)
        prompt_cu = batch["compression_prompt_cu_seqlens"].to(device)
        from bgkit.utils.packing import position_ids_from_cu
        prompt_position_ids = position_ids_from_cu(prompt_cu, int(prompt_ids.shape[0]))
        bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()
        enc_out = self.encoder(
            content_embeddings=bgkit_embed(content_token_ids),
            content_cu_seqlens=content_cu,
            content_position_ids=content_position_ids,
            prompt_embeddings=bgkit_embed(prompt_ids),
            prompt_cu_seqlens=prompt_cu,
            prompt_position_ids=prompt_position_ids,
            target_ratio=None,
            level="l0",
            min_per_sample=0,
        )
        return enc_out, content_token_ids, content_cu

    def _anchor_losses(
        self,
        proj: torch.Tensor,      # (N_content, D)
        target: torch.Tensor,    # (N_content, D)
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Compute MSE + cosine + norm-match losses on flat (N, D) packed tensors.

        All content positions in a packed batch are valid (no padding), so
        we simply mean over the flat axis.
        """
        n_pos = max(proj.shape[0], 1)

        # MSE over all valid flat positions and D.
        diff = proj - target
        mse = diff.pow(2).sum() / (n_pos * proj.size(-1))

        # Cosine direction loss: 1 - cos.
        cos = F.cosine_similarity(proj, target, dim=-1)  # (N,)
        cos_loss = (1.0 - cos).sum() / n_pos

        # Norm-match (log-ratio squared, robust to scale).
        proj_norm = proj.norm(dim=-1).clamp(min=1e-6)
        tgt_norm = target.norm(dim=-1).clamp(min=1e-6)
        norm_loss = (
            (torch.log(proj_norm) - torch.log(tgt_norm)).pow(2)
        ).sum() / n_pos

        return mse, {
            "anchor_mse": mse.detach(),
            "anchor_cos": cos_loss.detach(),
            "anchor_norm_log": norm_loss.detach(),
            "cos_sim_mean": cos.sum().detach() / n_pos,
            "norm_ratio_mean": (proj_norm / tgt_norm).sum().detach() / n_pos,
        }, cos_loss, norm_loss

    def _decoder_ce(
        self,
        batch: dict,
        proj: torch.Tensor,
        survivor_cu: torch.Tensor,
    ) -> torch.Tensor:
        """End-to-end reconstruction CE via the frozen decoder (packed).

        Carves per-sample prefix/suffix token lists from the packed
        ``token_ids`` using each sample's ``bgkit_splice_start`` (the
        position where the survivor embeddings are spliced into the
        token sequence). ``bgkit_splice_len`` is the span of dummy
        placeholder tokens the collator left open for the splice
        — typically 0 in repro batches.
        """
        device = self.device
        token_ids_flat = batch["token_ids"].to(device)
        tok_cu = batch["cu_seqlens"].to(device)
        splice_start = batch["bgkit_splice_start"].to(device)
        splice_len = batch["bgkit_splice_len"].to(device)
        loss_mask_flat = batch["loss_mask"].to(device)

        batch_size = int(tok_cu.shape[0]) - 1
        tok_cu_list = tok_cu.to(torch.int64).tolist()
        surv_cu_list = survivor_cu.to(torch.int64).tolist()

        # Per-sample prefix/suffix + the assembled per-segment loss mask.
        prefix_ids: list[torch.Tensor] = []
        suffix_ids: list[torch.Tensor] = []
        per_segment_loss_masks: list[torch.Tensor] = []
        for b in range(batch_size):
            sample_start = int(tok_cu_list[b])
            sample_end = int(tok_cu_list[b + 1])
            sample_tokens = token_ids_flat[sample_start:sample_end]
            sample_loss = loss_mask_flat[sample_start:sample_end].to(torch.bool)
            splice_b_start = int(splice_start[b].item())
            splice_b_len = int(splice_len[b].item())
            if splice_b_start < 0:
                # No splice — treat whole sequence as prefix (nothing after).
                splice_b_start = sample_tokens.shape[0]
                splice_b_len = 0
            pre = sample_tokens[:splice_b_start]
            suf = sample_tokens[splice_b_start + splice_b_len :]
            prefix_ids.append(pre)
            suffix_ids.append(suf)

            pre_mask = sample_loss[:splice_b_start]
            suf_mask = sample_loss[splice_b_start + splice_b_len :]
            k_i = int(surv_cu_list[b + 1]) - int(surv_cu_list[b])
            surv_mask = torch.zeros(k_i, dtype=torch.bool, device=device)
            per_segment_loss_masks.append(torch.cat([pre_mask, surv_mask, suf_mask], dim=0))

        flat_loss_mask = torch.cat(per_segment_loss_masks, dim=0)
        return self.decoder.forward_with_single_splice(
            survivor_embeddings=proj,
            survivor_cu_seqlens=survivor_cu,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            loss_mask=flat_loss_mask,
        )

    def _forward_backward(self, batch) -> dict[str, float]:
        # Projection block only trainable; compressor + decoder frozen but
        # we still need grads to flow through the projection, so a local
        # autocast+no-grad-on-frozen setup is not strictly necessary. The
        # frozen params have requires_grad=False.
        enc_out, content_ids, _content_cu = self._encoder_forward_nocomp(batch)
        proj = enc_out.survivor_embeddings  # (N_content, D) at ratio=None
        survivor_cu = enc_out.survivor_cu_seqlens  # same as content_cu when uncompressed

        with torch.no_grad():
            target_emb = self._decoder_embed(content_ids).detach()

        mse, stats, cos_loss, norm_loss = self._anchor_losses(proj, target_emb)

        total = (
            self._anchor_weight * mse
            + self._cos_weight * cos_loss
            + self._norm_weight * norm_loss
        )

        metrics: dict[str, torch.Tensor | float] = {
            "loss": total,
            "loss/anchor_mse": stats["anchor_mse"],
            "loss/anchor_cos": stats["anchor_cos"],
            "loss/anchor_norm_log": stats["anchor_norm_log"],
            "diag/cos_sim": stats["cos_sim_mean"],
            "diag/norm_ratio": stats["norm_ratio_mean"],
        }

        if self._ce_weight > 0.0:
            ce = self._decoder_ce(batch, proj, survivor_cu)
            total = total + self._ce_weight * ce
            metrics["loss/ce"] = ce.detach()
            metrics["loss"] = total

        (total / self._accum_steps).backward()

        # Drop CompressorOutput tensor refs — see
        # ``CompressionOutput.release()`` docstring for the motivating
        # leak. Projection repair runs at ``target_ratio=None`` so
        # ``_utility_grad_state`` is empty, but the call is safe and
        # standardises the per-step cleanup pattern across trainers.
        enc_out.release()

        return {k: (v.item() if hasattr(v, "item") else v) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run eval. Returns unprefixed keys — BaseTrainer adds 'eval/' prefix."""
        self.encoder.projection_block.eval()
        totals: dict[str, float] = {}
        count = 0
        for batch in self.eval_dataloader:
            enc_out, content_ids, _ = self._encoder_forward_nocomp(batch)
            proj = enc_out.survivor_embeddings
            survivor_cu = enc_out.survivor_cu_seqlens
            target_emb = self._decoder_embed(content_ids)
            mse, stats, cos_loss, norm_loss = self._anchor_losses(proj, target_emb)
            ce_val = None
            if self._ce_weight > 0.0:
                ce_val = self._decoder_ce(batch, proj, survivor_cu)

            totals["anchor_mse"] = totals.get("anchor_mse", 0.0) + float(mse.item())
            totals["anchor_cos"] = totals.get("anchor_cos", 0.0) + float(cos_loss.item())
            totals["anchor_norm_log"] = totals.get("anchor_norm_log", 0.0) + float(norm_loss.item())
            totals["cos_sim"] = totals.get("cos_sim", 0.0) + float(stats["cos_sim_mean"].item())
            totals["norm_ratio"] = totals.get("norm_ratio", 0.0) + float(stats["norm_ratio_mean"].item())
            if ce_val is not None:
                totals["ce"] = totals.get("ce", 0.0) + float(ce_val.item())
            count += 1

        if count > 0:
            totals = {k: v / count for k, v in totals.items()}
        # Primary metric for checkpoint registry and downstream auto-pickup
        # (`bgkit-ckpt best --metric eval/loss`). Mirror anchor_mse.
        totals["loss"] = totals.get("anchor_mse", 0.0)
        self.encoder.projection_block.train()
        return totals

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self, checkpoint_dir: Path, metrics: dict[str, float] | None = None,
    ) -> Path:
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
            optimizer_state_by_name=self._build_optimizer_state_by_name(),
        )
        if self._decoder_state_dict is not None:
            save_kwargs["decoder"] = self._decoder_state_dict
        ckpt_path = save_checkpoint(checkpoint_dir, metadata, **save_kwargs)
        self._last_checkpoint_path = str(ckpt_path)
        return ckpt_path

    def _named_parameters_for_optimizer(self):
        """Yield (name, param) pairs for the encoder only.

        Step 2.5 freezes the decoder and only trains ``projection_block``
        inside the encoder (plus optionally the final compressor blocks).
        Decoder state is passed through unchanged from the Step 2 source
        checkpoint and carries no optimizer state.
        """
        for name, param in self.encoder.named_parameters():
            yield f"encoder.{name}", param

    def _restore_model_state(self, state_dicts: dict) -> None:
        if "encoder" in state_dicts:
            self.encoder.load_state_dict(state_dicts["encoder"], strict=False)
        if "decoder" in state_dicts:
            self._decoder_state_dict = state_dicts["decoder"]
