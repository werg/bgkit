"""Phase 1 Qwen projection warmup.

Distills a pre-Falcon Qwen-aligned encoder (teacher) into the current
encoder's ``projection_blocks.qwen35`` (student). All student params
are frozen except the Qwen projection block. The Falcon-only training
stretch drifted the encoder backbone and the L0/L1 heads, while the
Qwen projection was untouched (Falcon was the active family during
that period). This stage gives the Qwen projection a tight,
constrained task: map the *current* encoder's survivor embeddings into
the projected space the Qwen decoder was last trained against.

Design:

- Teacher: a pre-Falcon Qwen-decoder checkpoint
  (default: ``phase1_step6_step1500_20260510_064202`` — the last
  completed Step 6 run before Falcon stages began). Loaded entirely
  frozen, eval mode. Active family = ``qwen35``.
- Student: the current encoder (typically the latest Falcon-l0
  checkpoint). All params frozen except
  ``encoder.projection_blocks.qwen35``. Active family = ``qwen35``.
- For each microbatch:
    1. Teacher runs forward at its operating ratio (default 0.10) and
       produces ``(survivor_mask, survivor_embeddings)``.
    2. Student runs forward with ``forced_survivor_mask_l0`` set to the
       teacher's mask, so survivor positions align exactly.
    3. Loss = MSE + cosine + log-norm-match on the
       position-aligned survivor embeddings, mirroring the structure
       of Step 2.5 (``ProjectionRepairTrainer``).
- No decoder is loaded — the loss is purely on projection outputs.
- No curriculum, no head losses, no decisiveness / moment-match /
  utility-grad signal: this stage exists only to make the Qwen
  projection consistent with the teacher's projection target on the
  *new* encoder's outputs.

The output checkpoint carries the current encoder state (backbone +
heads from the student source) with the Qwen projection block replaced
by the warmed-up weights. Pass it to ``train-qwen-realign`` (or any
Qwen-decoder phase) via ``bgkit_checkpoint:`` to resume joint training
with the projection now in shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import structlog
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from bgkit.data.collators import collate_chat_repro
from bgkit.data.datasets.chat_repro_dataset import ChatReproDataset
from bgkit.data.datasets.mmap_token_dataset import MmapTokenDataset
from bgkit.data.samplers import PackedTokenBudgetSampler
from bgkit.models.decoder import normalize_decoder_family
from bgkit.models.encoder import BgKITEncoder
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpoint_registry import resolve_checkpoint
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.utils.attention_backend import resolve_attention_implementation
from bgkit.utils.packing import position_ids_from_cu

logger = structlog.get_logger()


class QwenProjectionWarmupTrainer(BaseTrainer):
    """Distill teacher encoder's qwen35 projection into student's projection."""

    LIVE_CONFIG_FIELDS: ClassVar[dict[str, str]] = {
        "mse_weight": "_mse_weight",
        "cos_weight": "_cos_weight",
        "norm_weight": "_norm_weight",
        "target_ratio_l0": "_target_ratio_l0",
    }

    def setup(self) -> None:
        tcfg = self.cfg.training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        attention_impl = resolve_attention_implementation(
            self.cfg.compute.get("attention_implementation", "auto")
        )

        bgkit_cfg = self.cfg.model.bgkit
        backbone_name = bgkit_cfg.backbone_name
        hidden_dim = bgkit_cfg.get("hidden_dim", 1024)

        # Threshold controller anchor count must match each encoder's
        # state_dict (teacher and student were trained with different
        # anchor sets — Step 6 had 7 anchors up to 0.95, Falcon-l0 has 6
        # capped at 0.6). Pull the anchor list from each checkpoint
        # rather than from training YAML.
        model_cfg = self.cfg.model
        ctrl_src = tcfg.get("model", {}).get(
            "threshold_controller", model_cfg.get("threshold_controller", {}),
        )

        def _build_threshold_cfg(state_dict: dict) -> dict:
            anchor_t = state_dict.get("l0.threshold.anchor_ratios")
            anchors = anchor_t.tolist() if anchor_t is not None else (
                list(ctrl_src.get("anchor_ratios", [])) or None
            )
            return {
                "init_theta": float(ctrl_src.get("init_theta", 0.0)),
                "lr": float(ctrl_src.get("lr", 0.02)),
                "momentum": float(ctrl_src.get("momentum", 0.0)),
                "clamp": float(ctrl_src.get("clamp", 0.99)),
                "anchor_ratios": anchors,
                "ratio_space": str(ctrl_src.get("ratio_space", "log")),
                "init_target_ratio": float(tcfg.get("target_ratio_l0", 0.10)),
                "default_query_ratio": float(tcfg.get("target_ratio_l0", 0.10)),
                "kernel_bandwidth": (
                    float(ctrl_src["kernel_bandwidth"])
                    if ctrl_src.get("kernel_bandwidth") is not None
                    else None
                ),
            }

        decoder_cfg = tcfg.get("model", {}).get("decoder", self.cfg.model.decoder)
        decoder_family = normalize_decoder_family(decoder_cfg.get("family", "qwen35"))
        if decoder_family != "qwen35":
            raise ValueError(
                f"qwen_projection_warmup is qwen35-only (got {decoder_family!r}). "
                "Switch decoder.family to qwen35 in the experiment config."
            )

        # --- Resolve checkpoints ---
        teacher_ckpt = self._resolve_teacher_checkpoint()
        student_ckpt = self._resolve_student_checkpoint()
        if teacher_ckpt is None:
            raise ValueError("teacher_checkpoint must be set (path or 'auto').")
        if student_ckpt is None:
            raise ValueError("bgkit_checkpoint must be set (path or 'auto').")
        logger.info(
            "warmup_checkpoints_resolved",
            teacher=teacher_ckpt,
            student=student_ckpt,
        )

        # --- Load teacher encoder (frozen) ---
        _t_meta, teacher_state = load_checkpoint(Path(teacher_ckpt))
        if "encoder" not in teacher_state:
            raise ValueError(f"teacher checkpoint {teacher_ckpt} missing 'encoder' key")
        self.teacher_encoder = BgKITEncoder.from_pretrained_with_state_dict(
            backbone_name,
            teacher_state["encoder"],
            hidden_dim=hidden_dim,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=bgkit_cfg.get("backbone_revision", None),
            attn_implementation=attention_impl,
            bidi_warmup_steps=0,
            threshold_controller_cfg=_build_threshold_cfg(teacher_state["encoder"]),
        )
        self.teacher_encoder.to(device)
        self.teacher_encoder.set_active_decoder_family("qwen35")
        self.teacher_encoder.requires_grad_(False)
        self.teacher_encoder.eval()

        # --- Load student encoder (only qwen35 projection trains) ---
        _s_meta, student_state = load_checkpoint(Path(student_ckpt))
        if "encoder" not in student_state:
            raise ValueError(f"student checkpoint {student_ckpt} missing 'encoder' key")
        self.encoder = BgKITEncoder.from_pretrained_with_state_dict(
            backbone_name,
            student_state["encoder"],
            hidden_dim=hidden_dim,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=bgkit_cfg.get("backbone_revision", None),
            attn_implementation=attention_impl,
            bidi_warmup_steps=0,
            threshold_controller_cfg=_build_threshold_cfg(student_state["encoder"]),
        )
        self.encoder.to(device)
        self.encoder.set_active_decoder_family("qwen35")
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        # Unfreeze ONLY the Qwen projection. eval() above keeps dropout
        # off on the frozen modules; we set the projection back to
        # train() so any in-block dropout / norm-running-stats update
        # respects training semantics.
        qwen_proj = self.encoder.projection_blocks["qwen35"]
        qwen_proj.requires_grad_(True)
        qwen_proj.train()
        # Stash decoder state for checkpoint pass-through (student's
        # decoder is whatever the student source carried — in the
        # Falcon-l0 case that's a Falcon decoder; we hand it back
        # unchanged so resume code doesn't re-init from HF).
        self._decoder_state_dict = student_state.get("decoder", None)

        # Required by BaseTrainer for logging.
        self.model = self.encoder

        # --- Tokenizer + dataset (Qwen base, matches realign run) ---
        encoder_tokenizer_name = bgkit_cfg.get(
            "tokenizer_name", "Qwen/Qwen3.5-0.8B-Base",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder_tokenizer_name, trust_remote_code=True,
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

        max_eval_samples = tcfg.get("max_eval_samples", 500)
        eval_size = min(max(1, int(len(full_dataset) * 0.1)), max_eval_samples)
        train_size = len(full_dataset) - eval_size
        split_generator = torch.Generator().manual_seed(int(self.cfg.get("seed", 42)))
        self.train_dataset, self.eval_dataset = random_split(
            full_dataset, [train_size, eval_size], generator=split_generator,
        )

        max_batch_tokens = tcfg.get("max_batch_tokens", 4096)
        max_batch_tokens_eval = self._resolve_eval_batch_budget(tcfg, max_batch_tokens)
        num_workers = self.cfg.compute.get("num_workers", 4)
        pin_memory = self.cfg.compute.get("pin_memory", False)
        seed = self.cfg.get("seed", 42)

        train_lengths = full_dataset.lengths[np.array(self.train_dataset.indices)]
        eval_lengths = full_dataset.lengths[np.array(self.eval_dataset.indices)]
        train_content_lengths = full_dataset.content_lengths[
            np.array(self.train_dataset.indices)
        ]

        self._train_lengths = train_lengths
        self._train_content_lengths = train_content_lengths
        self._eval_lengths = eval_lengths
        self._train_collate_fn = collate_chat_repro
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._max_batch_tokens = max_batch_tokens
        self._max_batch_tokens_eval = max_batch_tokens_eval

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
            self.train_dataset, batch_sampler=self.train_sampler,
            collate_fn=collate_chat_repro,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset, batch_sampler=eval_sampler,
            collate_fn=collate_chat_repro,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        # --- Loss weights + operating ratio (live-tunable) ---
        self._mse_weight = float(tcfg.get("mse_weight", 1.0))
        self._cos_weight = float(tcfg.get("cos_weight", 0.5))
        self._norm_weight = float(tcfg.get("norm_weight", 0.1))
        # Teacher's natural operating ratio (Step 6 ran at ~0.10). The
        # warmup target is to match teacher's projection where it spent
        # most of its training mass, so we sit at a single fixed ratio.
        # Operating ratio. Two modes:
        #   - fixed: pass ``target_ratio_l0`` (scalar) — every batch
        #     trains at this ratio.
        #   - range: pass ``target_ratio_l0_range: [lo, hi]`` to sample
        #     each batch's ratio log-uniformly from [lo, hi]. Use this
        #     when the downstream realign run walks a curriculum across
        #     a range; training the projection at only the floor ratio
        #     (0.10) leaves it unvalidated for the higher-ratio /
        #     larger-survivor regimes the realign curriculum starts in
        #     (observed 2026-06-03: realign at curriculum 0.55→0.10 hit
        #     eval/loss plateau because most realign survivors fell
        #     outside the warmup's 0.10-only training distribution).
        ratio_range = tcfg.get("target_ratio_l0_range", None)
        if ratio_range is not None:
            lo, hi = float(ratio_range[0]), float(ratio_range[1])
            if not (0.0 < lo <= hi <= 1.0):
                raise ValueError(
                    f"target_ratio_l0_range must satisfy 0 < lo <= hi <= 1; "
                    f"got [{lo}, {hi}]",
                )
            self._target_ratio_l0_range: tuple[float, float] | None = (lo, hi)
            # Geometric-mean fallback for eval/probe paths that need a scalar.
            self._target_ratio_l0 = float(np.sqrt(lo * hi))
        else:
            self._target_ratio_l0_range = None
            self._target_ratio_l0 = float(tcfg.get("target_ratio_l0", 0.10))
        self._ratio_rng = np.random.default_rng(int(self.cfg.get("seed", 42)) + 7)

        # --- Optimizer (qwen35 projection params only) ---
        proj_params = [p for p in qwen_proj.parameters() if p.requires_grad]
        proj_lr = float(tcfg.get("projection_lr", tcfg.lr))
        param_groups = [{"params": proj_params, "lr": proj_lr, "base_lr": proj_lr}]
        self.optimizer = self._create_optimizer(
            param_groups, proj_lr, exclude_from_muon=frozenset(),
        )

        logger.info(
            "qwen_projection_warmup_setup",
            teacher_checkpoint=teacher_ckpt,
            student_checkpoint=student_ckpt,
            projection_params=sum(p.numel() for p in proj_params),
            train_samples=train_size,
            eval_samples=eval_size,
            mse_weight=self._mse_weight,
            cos_weight=self._cos_weight,
            norm_weight=self._norm_weight,
            target_ratio_l0=self._target_ratio_l0,
        )

    # ------------------------------------------------------------------
    # Checkpoint resolution
    # ------------------------------------------------------------------

    def _resolve_teacher_checkpoint(self) -> str | None:
        src = self.cfg.get("teacher_checkpoint", None)
        if src == "auto":
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            resolved = resolve_checkpoint(
                checkpoint_dir, phase="phase1_step6",
                metric="eval/loss", label="teacher_checkpoint",
            )
            src = str(resolved)
        return src

    def _resolve_student_checkpoint(self) -> str | None:
        src = self.cfg.get("bgkit_checkpoint", None)
        if src == "auto":
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            resolved = resolve_checkpoint(
                checkpoint_dir, phase="phase1_falcon_l0",
                metric="eval/loss", label="bgkit_checkpoint",
            )
            src = str(resolved)
        self._input_sources = {"bgkit": Path(src).name} if src else {}
        return src

    def trainable_parameters(self) -> list:
        return [
            p for p in self.encoder.projection_blocks["qwen35"].parameters()
            if p.requires_grad
        ]

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _sample_target_ratio_l0(self) -> float:
        """Return per-batch target ratio. Log-uniform when a range is set,
        constant otherwise. Stash the chosen ratio on ``self._last_ratio_l0``
        so ``_student_forward_forced`` and metrics agree on what teacher used.
        """
        if self._target_ratio_l0_range is not None:
            lo, hi = self._target_ratio_l0_range
            r = float(np.exp(self._ratio_rng.uniform(np.log(lo), np.log(hi))))
            r = float(np.clip(r, lo, hi))
        else:
            r = self._target_ratio_l0
        self._last_ratio_l0 = r
        return r

    def _encode_batch(self, encoder: BgKITEncoder, batch: dict, *, with_grad: bool):
        device = self.device
        content_token_ids = batch["content_token_ids"].to(device)
        content_cu = batch["content_cu_seqlens"].to(device)
        content_position_ids = batch["content_position_ids"].to(device)
        prompt_ids = batch["compression_prompt_ids"].to(device)
        prompt_cu = batch["compression_prompt_cu_seqlens"].to(device)
        prompt_position_ids = position_ids_from_cu(prompt_cu, int(prompt_ids.shape[0]))
        embed = encoder.l0.backbone.get_input_embeddings()
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        # Sample a fresh per-batch ratio for the teacher. Both training
        # _forward_backward and evaluate() call _encode_batch with
        # with_grad=False (teacher path), so this samples once per batch
        # in either case; evaluate averaging over batches gives a clean
        # range-wide eval signal.
        ratio = self._sample_target_ratio_l0()
        with ctx:
            # exact_topk for the teacher: pick exactly target_ratio×N
            # tokens by head score regardless of threshold calibration.
            # Threshold mode requires the teacher's theta/T to remain
            # well-calibrated for the current input distribution; that
            # held during Step 6 training but drifted relative to the
            # FineWeb-Edu input distribution we're warming against now
            # (saw 0/N survivors at ratio 0.10 with threshold mode).
            # The student's projection only ever cares about which
            # positions to project, not how they were chosen.
            return encoder(
                content_embeddings=embed(content_token_ids),
                content_cu_seqlens=content_cu,
                content_position_ids=content_position_ids,
                prompt_embeddings=embed(prompt_ids),
                prompt_cu_seqlens=prompt_cu,
                prompt_position_ids=prompt_position_ids,
                target_ratio_l0=ratio,
                target_ratio_l1=None,
                selection_mode_l0="exact_topk",
            ), content_cu

    def _student_forward_forced(
        self, batch: dict, forced_mask: torch.Tensor,
    ):
        device = self.device
        content_token_ids = batch["content_token_ids"].to(device)
        content_cu = batch["content_cu_seqlens"].to(device)
        content_position_ids = batch["content_position_ids"].to(device)
        prompt_ids = batch["compression_prompt_ids"].to(device)
        prompt_cu = batch["compression_prompt_cu_seqlens"].to(device)
        prompt_position_ids = position_ids_from_cu(prompt_cu, int(prompt_ids.shape[0]))
        embed = self.encoder.l0.backbone.get_input_embeddings()
        # target_ratio_l0=None + forced mask: head/threshold are bypassed,
        # selection is set verbatim from the teacher's mask. This keeps
        # survivor positions aligned 1:1 with the teacher output so the
        # MSE / cosine targets compare apples to apples.
        return self.encoder(
            content_embeddings=embed(content_token_ids),
            content_cu_seqlens=content_cu,
            content_position_ids=content_position_ids,
            prompt_embeddings=embed(prompt_ids),
            prompt_cu_seqlens=prompt_cu,
            prompt_position_ids=prompt_position_ids,
            target_ratio_l0=None,
            target_ratio_l1=None,
            forced_survivor_mask_l0=forced_mask,
        )

    @staticmethod
    def _projection_losses(
        student: torch.Tensor, teacher: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        n_pos = max(student.shape[0], 1)
        diff = student - teacher
        mse = diff.pow(2).sum() / (n_pos * student.size(-1))
        cos = F.cosine_similarity(student, teacher, dim=-1)  # (N,)
        cos_loss = (1.0 - cos).sum() / n_pos
        s_norm = student.norm(dim=-1).clamp(min=1e-6)
        t_norm = teacher.norm(dim=-1).clamp(min=1e-6)
        norm_loss = (torch.log(s_norm) - torch.log(t_norm)).pow(2).sum() / n_pos
        return mse, {
            "mse": mse.detach(),
            "cos_loss": cos_loss.detach(),
            "norm_log_loss": norm_loss.detach(),
            "cos_sim_mean": cos.sum().detach() / n_pos,
            "norm_ratio_mean": (s_norm / t_norm).sum().detach() / n_pos,
        }, cos_loss, norm_loss

    def _forward_backward(self, batch) -> dict[str, float]:
        teacher_out, _ = self._encode_batch(
            self.teacher_encoder, batch, with_grad=False,
        )
        teacher_mask = teacher_out.l0.survivor_mask.detach()
        teacher_proj = teacher_out.survivor_embeddings.detach()

        if self.global_step < 3:
            logger.info(
                "warmup_debug_shapes",
                step=self.global_step,
                content_n=int(batch["content_token_ids"].shape[0]),
                teacher_mask_total=int(teacher_mask.numel()),
                teacher_mask_true=int(teacher_mask.sum().item()),
                teacher_proj_shape=tuple(teacher_proj.shape),
                teacher_survivor_counts=teacher_out.survivor_counts.tolist()
                if teacher_out.survivor_counts is not None else None,
            )

        student_out = self._student_forward_forced(batch, teacher_mask)
        student_proj = student_out.survivor_embeddings

        if student_proj.shape != teacher_proj.shape:
            raise RuntimeError(
                f"projection shape mismatch student={tuple(student_proj.shape)} "
                f"teacher={tuple(teacher_proj.shape)} — forced mask alignment "
                "broken",
            )

        mse, stats, cos_loss, norm_loss = self._projection_losses(
            student_proj, teacher_proj,
        )
        total = (
            self._mse_weight * mse
            + self._cos_weight * cos_loss
            + self._norm_weight * norm_loss
        )

        (total / self._accum_steps).backward()

        teacher_out.release()
        student_out.release()

        metrics = {
            "loss": total,
            "loss/mse": stats["mse"],
            "loss/cos": stats["cos_loss"],
            "loss/norm_log": stats["norm_log_loss"],
            "diag/cos_sim": stats["cos_sim_mean"],
            "diag/norm_ratio": stats["norm_ratio_mean"],
            "diag/target_ratio_l0": getattr(self, "_last_ratio_l0", self._target_ratio_l0),
        }
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        qwen_proj = self.encoder.projection_blocks["qwen35"]
        qwen_proj.eval()
        totals: dict[str, float] = {}
        count = 0
        for batch in self.eval_dataloader:
            teacher_out, _ = self._encode_batch(
                self.teacher_encoder, batch, with_grad=False,
            )
            teacher_mask = teacher_out.l0.survivor_mask
            teacher_proj = teacher_out.survivor_embeddings

            student_out = self._student_forward_forced(batch, teacher_mask)
            student_proj = student_out.survivor_embeddings

            mse, stats, cos_loss, norm_loss = self._projection_losses(
                student_proj, teacher_proj,
            )
            totals["mse"] = totals.get("mse", 0.0) + float(mse.item())
            totals["cos"] = totals.get("cos", 0.0) + float(cos_loss.item())
            totals["norm_log"] = totals.get("norm_log", 0.0) + float(norm_loss.item())
            totals["cos_sim"] = totals.get("cos_sim", 0.0) + float(
                stats["cos_sim_mean"].item(),
            )
            totals["norm_ratio"] = totals.get("norm_ratio", 0.0) + float(
                stats["norm_ratio_mean"].item(),
            )
            count += 1

            teacher_out.release()
            student_out.release()

        if count > 0:
            totals = {k: v / count for k, v in totals.items()}
        # Primary checkpoint-registry metric.
        totals["loss"] = totals.get("mse", 0.0)
        qwen_proj.train()
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
            run_name=self.cfg.get("run_name", None),
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
        # Only the qwen35 projection trains; optimizer state is scoped
        # to those params only. Name-prefix matches the encoder
        # state_dict key path so name-keyed save/restore lines up.
        for name, param in self.encoder.projection_blocks["qwen35"].named_parameters():
            yield f"encoder.projection_blocks.qwen35.{name}", param

    def _restore_model_state(self, state_dicts: dict) -> None:
        if "encoder" in state_dicts:
            self.encoder.load_state_dict(state_dicts["encoder"], strict=False)
        if "decoder" in state_dicts:
            self._decoder_state_dict = state_dicts["decoder"]
