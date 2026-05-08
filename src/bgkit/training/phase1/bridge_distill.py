"""Phase 1 Step 4.7: Bridge distillation.

Repairs the L0->L1 ``auto_repro_head`` bridge and adapts the last L0 backbone
block + first few L1 backbone blocks + projection_block by distilling against
the frozen Step 4 (L0-only) encoder. See ``plans/bridge-distill-step.md`` for
the full motivation and math; this file implements the trainer end-to-end.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import ClassVar

import numpy as np
import structlog
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from bgkit.data.collators import collate_compression
from bgkit.data.datasets.commit_encoding_dataset import CommitEncodingDataset
from bgkit.data.samplers import PackedTokenBudgetSampler
from bgkit.models.encoder import BgKITEncoder
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpoint_registry import resolve_checkpoint
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.training.gradient_utils import maybe_enable_gradient_checkpointing
from bgkit.utils.attention_backend import resolve_attention_implementation
from bgkit.utils.packing import position_ids_from_cu

logger = structlog.get_logger()


def _build_l0_and_l1_forced_masks(
    teacher_mask: torch.Tensor,
    cu_file: torch.Tensor,
    frac_extras: float,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct forced L0 mask (over content positions) + forced L1 mask
    (over L0's reduced output) for Path B.

    For each file, doomed positions ``D = ~teacher_mask`` are partitioned:
    a fraction ``frac_extras`` survives at L0 (and dies at L1), the rest die
    at L0. Returns ``(l0_mask, l1_mask)`` both packed/flat.
    """
    device = teacher_mask.device
    cu = cu_file.to(torch.int64).tolist()
    b = len(cu) - 1

    l0_mask = teacher_mask.clone()
    for i in range(b):
        s, e = int(cu[i]), int(cu[i + 1])
        if e <= s:
            continue
        sample_t = teacher_mask[s:e]
        doomed_idx = (~sample_t).nonzero(as_tuple=True)[0]
        n_doomed = int(doomed_idx.numel())
        if n_doomed == 0:
            continue
        n_extras = math.ceil(frac_extras * n_doomed)
        n_extras = min(max(n_extras, 0), n_doomed)
        if n_extras == 0:
            continue
        perm = torch.randperm(n_doomed, generator=rng, device=device)[:n_extras]
        extras_idx = doomed_idx[perm]
        l0_mask[s + extras_idx] = True

    l1_mask = teacher_mask[l0_mask]
    return l0_mask, l1_mask


class BridgeDistillTrainer(BaseTrainer):
    """Phase 1 Step 4.7: bridge + last-L0-block + first-L1-blocks + projection."""

    LIVE_CONFIG_FIELDS: ClassVar[dict[str, str]] = {
        "cos_weight": "_cos_weight",
        "mse_weight": "_mse_weight",
        "curriculum_steps": "_curriculum_steps",
        "frac_extras_start": "_frac_extras_start",
        "frac_extras_end": "_frac_extras_end",
        "path_a_prob": "_path_a_prob",
    }

    def setup(self) -> None:
        tcfg = self.cfg.training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        attention_impl = resolve_attention_implementation(
            self.cfg.compute.get("attention_implementation", "auto")
        )

        # ---- Distillation knobs ----
        self._teacher_ratio = float(tcfg.get("teacher_ratio", 0.20))
        self._mse_weight = float(tcfg.get("mse_weight", 1.0))
        self._cos_weight = float(tcfg.get("cos_weight", 1.0))
        # Floor on per-sample (per-file) survivor count. Without this,
        # tight teacher_ratio (e.g. 0.10-0.15) on the per-file packed
        # commit_encoding dataset produces 1-2-token "samples" for short
        # files, which crashes flash_attn varlen with shape errors
        # (e.g. ``[B, 2, H, D]`` reshape on a 2048-element tensor).
        # Default 4 keeps FA's varlen kernel in its valid range.
        self._min_per_sample = int(tcfg.get("min_per_sample", 4))
        self._curriculum_steps = int(tcfg.get("curriculum_steps", 2500))
        self._frac_extras_start = float(tcfg.get("frac_extras_start", 1.0))
        self._frac_extras_end = float(tcfg.get("frac_extras_end", 0.31))
        self._path_a_prob = float(tcfg.get("path_a_prob", 0.5))

        unfreeze_cfg = tcfg.get("unfreeze", {}) or {}
        self._unfreeze_l0_last_blocks = int(unfreeze_cfg.get("l0_last_blocks", 1))
        self._unfreeze_l1_first_blocks = int(unfreeze_cfg.get("l1_first_blocks", 2))
        self._unfreeze_bridge = bool(unfreeze_cfg.get("bridge", True))
        self._unfreeze_projection_block = bool(unfreeze_cfg.get("projection_block", True))
        self._unfreeze_l0_norm = bool(unfreeze_cfg.get("l0_norm", True))
        self._unfreeze_l1_norm = bool(unfreeze_cfg.get("l1_norm", True))

        # Dedicated CUDA RNG so curriculum sampling is independent of the
        # global generator (which the dataloader / model also touch).
        seed = int(self.cfg.get("seed", 42))
        self._sampling_rng = torch.Generator(device=device)
        self._sampling_rng.manual_seed(seed)

        # ---- Resolve source checkpoint ----
        src_ckpt = self._resolve_source_checkpoint()
        if src_ckpt is None:
            raise ValueError(
                "phase1_step4p7 requires bgkit_checkpoint set (path or 'auto')."
            )

        bgkit_cfg = self.cfg.model.bgkit
        backbone_name = bgkit_cfg.backbone_name
        hidden_dim = bgkit_cfg.get("hidden_dim", 1024)

        logger.info("loading_source_checkpoint", path=src_ckpt)
        _, state_dicts = load_checkpoint(Path(src_ckpt))
        if "encoder" not in state_dicts:
            raise ValueError(f"checkpoint {src_ckpt} missing 'encoder' key")

        # Stash decoder state for pass-through at save time. Decoder LoRA
        # is preserved from whatever checkpoint we loaded (typically the
        # student's), so resuming Step 5 from this run keeps decoder
        # adaptation intact (modulo re-aligning to the new projection).
        # Both ``decoder`` (PEFT-wrapped state) and ``decoder_merged``
        # (full state with LoRA merged into base) are passed through;
        # Step 5's resume loader prefers ``decoder_merged`` and fails to
        # load the bare-key state dict it expects from PEFT-wrapped
        # ``decoder.pt`` alone.
        self._decoder_state_dict = state_dicts.get("decoder")
        self._decoder_merged_state_dict = state_dicts.get("decoder_merged")

        encoder_state = state_dicts["encoder"]

        # Optional separate teacher: distill student into the projection
        # quality of a different checkpoint (e.g. teacher = original
        # phase1_step4p7, student = current phase1_step5). When unset,
        # teacher = student (original behavior).
        teacher_ckpt = self._resolve_teacher_checkpoint()
        if teacher_ckpt is not None:
            logger.info("loading_teacher_checkpoint", path=teacher_ckpt)
            _, teacher_state_dicts = load_checkpoint(Path(teacher_ckpt))
            if "encoder" not in teacher_state_dicts:
                raise ValueError(
                    f"teacher_checkpoint {teacher_ckpt} missing 'encoder' key"
                )
            teacher_encoder_state = teacher_state_dicts["encoder"]
        else:
            teacher_encoder_state = encoder_state

        # ---- Build teacher (frozen) and student (selectively trainable) ----
        common_kwargs = dict(
            hidden_dim=hidden_dim,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=bgkit_cfg.get("backbone_revision", None),
            attn_implementation=attention_impl,
            bidi_warmup_steps=0,
        )

        self.encoder_teacher = BgKITEncoder.from_pretrained_with_state_dict(
            backbone_name, teacher_encoder_state, **common_kwargs,
        )
        self.encoder_teacher.to(device)
        self.encoder_teacher.requires_grad_(False)
        self.encoder_teacher.eval()

        self.encoder_student = BgKITEncoder.from_pretrained_with_state_dict(
            backbone_name, encoder_state, **common_kwargs,
        )
        self.encoder_student.to(device)
        self._freeze_for_bridge_distill(self.encoder_student)

        if tcfg.get("use_liger", True):
            from bgkit.utils.liger_integration import apply_liger_to_qwen35

            patch_rmsnorm = bool(tcfg.get("use_liger_rmsnorm", False))
            patch_swiglu = bool(tcfg.get("use_liger_swiglu", True))
            patch_rope = bool(tcfg.get("use_liger_rope", True))
            apply_liger_to_qwen35(
                self.encoder_teacher,
                patch_rmsnorm=patch_rmsnorm,
                patch_swiglu=patch_swiglu,
                patch_rope=patch_rope,
            )
            apply_liger_to_qwen35(
                self.encoder_student,
                patch_rmsnorm=patch_rmsnorm,
                patch_swiglu=patch_swiglu,
                patch_rope=patch_rope,
            )

        # Gradient checkpointing on student backbones — frozen blocks pay no
        # activation cost, but unfrozen blocks would otherwise hold full acts.
        maybe_enable_gradient_checkpointing(self.encoder_student.l0.backbone, self.cfg)
        maybe_enable_gradient_checkpointing(self.encoder_student.l1.backbone, self.cfg)

        self.model = self.encoder_student

        # ---- Tokenizer (for dataset) ----
        from transformers import AutoTokenizer

        decoder_cfg = self.cfg.model.decoder
        decoder_name = decoder_cfg.backbone_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            decoder_name,
            trust_remote_code=True,
            revision=decoder_cfg.get("backbone_revision", None),
        )

        # ---- Dataset (commit encoding; matches Step 5 distribution) ----
        data_cfg = tcfg.data
        variant_bank = self._load_variant_bank(data_cfg)

        from bgkit.data.chat_template import TOOL_CONFIGS

        config = TOOL_CONFIGS["commit_encoding"]
        self.commit_dataset = CommitEncodingDataset(
            data_dir=data_cfg.commit_encoding_dir,
            tokenizer=self.tokenizer,
            variant_bank=variant_bank,
            config=config,
            max_diff_tokens_per_file=data_cfg.get("max_diff_tokens_per_file", 4096),
            max_files_per_commit=data_cfg.get("max_files_per_commit", 16),
            max_message_tokens=data_cfg.get("max_message_tokens", 256),
            seed=seed,
        )

        max_eval_samples = tcfg.get("max_eval_samples", 1000)
        total = len(self.commit_dataset)
        eval_size = min(max(1, int(total * 0.1)), max_eval_samples)
        train_size = total - eval_size
        if train_size < 1:
            raise ValueError(
                f"Dataset too small for train/eval split (got {total} samples)"
            )
        split_generator = torch.Generator().manual_seed(int(seed))
        self.train_dataset, self.eval_dataset = random_split(
            self.commit_dataset, [train_size, eval_size], generator=split_generator,
        )

        num_workers = self.cfg.compute.get("num_workers", 4)
        pin_memory = self.cfg.compute.get("pin_memory", False)

        train_lengths = np.array([
            self.commit_dataset.token_length(i) for i in self.train_dataset.indices
        ], dtype=np.int64)
        eval_lengths = np.array([
            self.commit_dataset.token_length(i) for i in self.eval_dataset.indices
        ], dtype=np.int64)

        self._train_lengths = train_lengths
        self._train_content_lengths = train_lengths
        self._eval_lengths = eval_lengths
        self._train_collate_fn = collate_compression
        self._num_workers = num_workers
        self._pin_memory = pin_memory

        max_batch_tokens = int(tcfg.get("max_batch_tokens", 8192))
        self._max_batch_tokens = max_batch_tokens
        self._max_batch_tokens_eval = self._resolve_eval_batch_budget(
            tcfg, max_batch_tokens,
        )
        self._min_sample_length = int(tcfg.get("min_sample_length", 0) or 0)
        self._max_sample_length = int(tcfg.get("max_sample_length", 0) or 0)

        self.train_sampler = PackedTokenBudgetSampler(
            self.train_dataset,
            lengths=train_lengths,
            max_batch_tokens=max_batch_tokens,
            shuffle=True,
            seed=seed,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=collate_compression,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        eval_sampler = PackedTokenBudgetSampler(
            self.eval_dataset,
            lengths=eval_lengths,
            max_batch_tokens=self._max_batch_tokens_eval,
            shuffle=False,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_sampler=eval_sampler,
            collate_fn=collate_compression,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # ---- Optimizer ----
        trainable = list(self.trainable_parameters())
        if not trainable:
            raise RuntimeError(
                "BridgeDistillTrainer found no trainable parameters; check "
                "the freeze plan and unfreeze config."
            )
        param_groups = [{"params": trainable, "lr": tcfg.lr, "base_lr": tcfg.lr}]
        self.optimizer = self._create_optimizer(
            param_groups, float(tcfg.lr), exclude_from_muon=frozenset(),
        )

        component_counts = self._trainable_component_counts()
        logger.info(
            "bridge_distill_setup",
            src_checkpoint=src_ckpt,
            train_samples=train_size,
            eval_samples=eval_size,
            teacher_ratio=self._teacher_ratio,
            curriculum_steps=self._curriculum_steps,
            trainable_params_total=sum(p.numel() for p in trainable),
            **component_counts,
        )

    # ------------------------------------------------------------------
    # Freeze plan
    # ------------------------------------------------------------------

    def _freeze_for_bridge_distill(self, encoder: BgKITEncoder) -> None:
        encoder.requires_grad_(False)
        encoder.eval()

        if self._unfreeze_bridge:
            encoder.l0.auto_repro_head.requires_grad_(True)
            encoder.l0.auto_repro_head.train()
        if self._unfreeze_l0_norm:
            encoder.l0.norm.requires_grad_(True)
            encoder.l0.norm.train()
        if self._unfreeze_l1_norm:
            encoder.l1.norm.requires_grad_(True)
            encoder.l1.norm.train()
        if self._unfreeze_projection_block:
            encoder.projection_block.requires_grad_(True)
            encoder.projection_block.train()

        l0_blocks = self._resolve_blocks(encoder.l0.backbone)
        if self._unfreeze_l0_last_blocks > 0:
            for b in l0_blocks[-self._unfreeze_l0_last_blocks:]:
                b.requires_grad_(True)
                b.train()

        l1_blocks = self._resolve_blocks(encoder.l1.backbone)
        if self._unfreeze_l1_first_blocks > 0:
            for b in l1_blocks[: self._unfreeze_l1_first_blocks]:
                b.requires_grad_(True)
                b.train()

    @staticmethod
    def _resolve_blocks(backbone) -> list:
        for attr in ("blocks", "layers"):
            mod = getattr(backbone, attr, None)
            if mod is not None and len(mod) > 0:
                return list(mod)
        raise ValueError(
            f"Cannot find blocks/layers in {type(backbone).__name__}"
        )

    def _trainable_component_counts(self) -> dict[str, int]:
        enc = self.encoder_student
        counts = {
            "trainable_bridge": sum(
                p.numel() for p in enc.l0.auto_repro_head.parameters() if p.requires_grad
            ),
            "trainable_l0_norm": sum(
                p.numel() for p in enc.l0.norm.parameters() if p.requires_grad
            ),
            "trainable_l1_norm": sum(
                p.numel() for p in enc.l1.norm.parameters() if p.requires_grad
            ),
            "trainable_projection": sum(
                p.numel() for p in enc.projection_block.parameters() if p.requires_grad
            ),
        }
        l0_blocks = self._resolve_blocks(enc.l0.backbone)
        l1_blocks = self._resolve_blocks(enc.l1.backbone)
        counts["trainable_l0_blocks"] = sum(
            p.numel()
            for b in l0_blocks[-self._unfreeze_l0_last_blocks:]
            for p in b.parameters() if p.requires_grad
        ) if self._unfreeze_l0_last_blocks > 0 else 0
        counts["trainable_l1_blocks"] = sum(
            p.numel()
            for b in l1_blocks[: self._unfreeze_l1_first_blocks]
            for p in b.parameters() if p.requires_grad
        ) if self._unfreeze_l1_first_blocks > 0 else 0
        return counts

    def trainable_parameters(self) -> list:
        return [p for p in self.encoder_student.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _current_frac_extras(self) -> float:
        if self._curriculum_steps <= 0:
            return self._frac_extras_end
        progress = min(1.0, self.global_step / self._curriculum_steps)
        return (
            self._frac_extras_start
            + (self._frac_extras_end - self._frac_extras_start) * progress
        )

    def _current_ratios(self, frac_extras: float) -> tuple[float, float]:
        # Used for diagnostics / theta lookup only — selection is forced.
        r_l0 = self._teacher_ratio + frac_extras * (1.0 - self._teacher_ratio)
        r_l0 = min(max(r_l0, 1e-3), 1.0)
        r_l1 = self._teacher_ratio / r_l0
        r_l1 = min(max(r_l1, 1e-3), 1.0)
        return r_l0, r_l1

    # ------------------------------------------------------------------
    # Checkpoint / dataset helpers
    # ------------------------------------------------------------------

    def _resolve_source_checkpoint(self) -> str | None:
        src = self.cfg.get("bgkit_checkpoint", None)
        if src == "auto":
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            resolved = None
            # Default chain (used when student=teacher): step4 family. The
            # phase1_step5 case (re-running 4.7 from a Step 5 checkpoint) is
            # opted into by setting ``bgkit_checkpoint`` explicitly to
            # ``auto_step5`` — see _resolve_phase_chain.
            for phase in ("phase1_step4_split", "phase1_step4"):
                try:
                    resolved = resolve_checkpoint(
                        checkpoint_dir,
                        phase=phase,
                        metric="eval/loss",
                        label="bgkit_checkpoint",
                    )
                    break
                except (FileNotFoundError, RuntimeError, ValueError):
                    continue
            if resolved is None:
                raise RuntimeError(
                    "bgkit_checkpoint=auto but no phase1_step4_split or "
                    "phase1_step4 checkpoint found in registry."
                )
            src = str(resolved)
        elif isinstance(src, str) and src.startswith("auto_"):
            # auto_<phase> — resolve latest checkpoint of the named phase.
            phase = src[len("auto_"):]
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            try:
                resolved = resolve_checkpoint(
                    checkpoint_dir,
                    phase=phase,
                    metric="eval/loss",
                    label="bgkit_checkpoint",
                )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    f"bgkit_checkpoint={src!r} but no {phase!r} checkpoint "
                    f"found in registry: {exc}"
                ) from exc
            src = str(resolved)
        self._input_sources = {"bgkit": Path(src).name} if src else {}
        return src

    def _resolve_teacher_checkpoint(self) -> str | None:
        """Resolve the optional ``teacher_checkpoint`` config.

        When set, the teacher encoder is loaded from this path (instead of
        the same checkpoint as the student). Used to distill a student
        from a known-good teacher (e.g. teacher = phase1_step4p7,
        student = current phase1_step5). Returns None if not configured;
        callers should fall back to using the student's checkpoint as
        teacher (original behavior).

        Supports the same ``auto_<phase>`` shorthand as
        ``bgkit_checkpoint``.
        """
        tch = self.cfg.get("teacher_checkpoint", None)
        if tch is None:
            return None
        if tch == "auto":
            tch = "auto_phase1_step4p7"
        if isinstance(tch, str) and tch.startswith("auto_"):
            phase = tch[len("auto_"):]
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            try:
                resolved = resolve_checkpoint(
                    checkpoint_dir,
                    phase=phase,
                    metric="eval/loss",
                    label="teacher_checkpoint",
                )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    f"teacher_checkpoint={tch!r} but no {phase!r} checkpoint "
                    f"found in registry: {exc}"
                ) from exc
            tch = str(resolved)
        if hasattr(self, "_input_sources") and isinstance(tch, str):
            self._input_sources["teacher"] = Path(tch).name
        return tch

    def _load_variant_bank(self, data_cfg) -> list[dict[str, str]]:
        variant_dir = getattr(data_cfg, "prompt_variants_dir", None)
        if variant_dir:
            bank_path = Path(variant_dir) / "commit_encoding.json"
            if bank_path.exists():
                with open(bank_path) as f:
                    bank = json.load(f)
                if bank:
                    return bank
        return [{
            "system_prompt": "You are an AI assistant with access to the "
                             "bgkit_reproduce_commit tool.",
            "user_prompt": "Reproduce the commit from repository {file_path}",
            "compression_prompt": "Reproduce the complete commit from "
                                  "compressed context",
            "response_prefix": "Here is the reconstructed commit:",
        }]

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _move_batch(self, batch: dict) -> dict:
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _content_inputs(self, batch: dict):
        device = self.device
        content_ids = batch["content_token_ids"]
        cu_file = batch["cu_file_seqlens"]
        n = int(content_ids.shape[0])
        content_pos = position_ids_from_cu(cu_file, n)
        prompt_ids = batch["prompt_token_ids"]
        prompt_cu = batch["prompt_cu_seqlens"]
        prompt_pos = position_ids_from_cu(prompt_cu, int(prompt_ids.shape[0]))
        embed = self.encoder_student.l0.backbone.get_input_embeddings()
        teacher_embed = self.encoder_teacher.l0.backbone.get_input_embeddings()
        content_emb_student = embed(content_ids)
        with torch.no_grad():
            content_emb_teacher = teacher_embed(content_ids)
            prompt_emb_teacher = teacher_embed(prompt_ids)
        prompt_emb_student = embed(prompt_ids)
        return (
            content_emb_student,
            content_emb_teacher,
            prompt_emb_student,
            prompt_emb_teacher,
            cu_file.to(device),
            content_pos.to(device),
            prompt_cu.to(device),
            prompt_pos.to(device),
        )

    def _teacher_forward(self, ctx) -> tuple[torch.Tensor, torch.Tensor]:
        (
            _,
            content_emb_t,
            _,
            prompt_emb_t,
            cu_file,
            content_pos,
            prompt_cu,
            prompt_pos,
        ) = ctx
        with torch.no_grad():
            out = self.encoder_teacher(
                content_embeddings=content_emb_t,
                content_cu_seqlens=cu_file,
                content_position_ids=content_pos,
                prompt_embeddings=prompt_emb_t,
                prompt_cu_seqlens=prompt_cu,
                prompt_position_ids=prompt_pos,
                target_ratio_l0=self._teacher_ratio,
                target_ratio_l1=None,
                min_per_sample_l0=self._min_per_sample,
            )
            mask_t = out.l0.survivor_mask.detach()
            proj_t = out.survivor_embeddings.detach()
        out.release()
        return mask_t, proj_t

    def _student_forward_path_a(self, ctx, mask_t):
        (
            content_emb_s,
            _,
            prompt_emb_s,
            _,
            cu_file,
            content_pos,
            prompt_cu,
            prompt_pos,
        ) = ctx
        out = self.encoder_student(
            content_embeddings=content_emb_s,
            content_cu_seqlens=cu_file,
            content_position_ids=content_pos,
            prompt_embeddings=prompt_emb_s,
            prompt_cu_seqlens=prompt_cu,
            prompt_position_ids=prompt_pos,
            target_ratio_l0=self._teacher_ratio,
            target_ratio_l1=None,
            forced_survivor_mask_l0=mask_t,
        )
        return out

    def _student_forward_path_b(self, ctx, mask_t, frac_extras):
        (
            content_emb_s,
            _,
            prompt_emb_s,
            _,
            cu_file,
            content_pos,
            prompt_cu,
            prompt_pos,
        ) = ctx
        l0_mask, l1_mask = _build_l0_and_l1_forced_masks(
            mask_t, cu_file, frac_extras, self._sampling_rng,
        )
        r_l0, r_l1 = self._current_ratios(frac_extras)
        out = self.encoder_student(
            content_embeddings=content_emb_s,
            content_cu_seqlens=cu_file,
            content_position_ids=content_pos,
            prompt_embeddings=prompt_emb_s,
            prompt_cu_seqlens=prompt_cu,
            prompt_position_ids=prompt_pos,
            target_ratio_l0=r_l0,
            target_ratio_l1=r_l1,
            forced_survivor_mask_l0=l0_mask,
            forced_survivor_mask_l1=l1_mask,
        )
        return out

    def _distill_loss(
        self, proj_s: torch.Tensor, proj_t: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if proj_s.shape != proj_t.shape:
            raise RuntimeError(
                f"student / teacher projection shape mismatch: "
                f"{tuple(proj_s.shape)} vs {tuple(proj_t.shape)}"
            )
        proj_s32 = proj_s.float()
        proj_t32 = proj_t.float()
        mse = F.mse_loss(proj_s32, proj_t32)
        cos = F.cosine_similarity(proj_s32, proj_t32, dim=-1)
        cos_mean = cos.mean()
        cos_loss = 1.0 - cos_mean
        total = self._mse_weight * mse + self._cos_weight * cos_loss
        stats = {
            "loss/mse": mse.detach(),
            "loss/cos": cos_loss.detach(),
            "diag/cos_sim": cos_mean.detach(),
        }
        return total, stats

    def _pick_path(self) -> str:
        u = torch.rand(1, generator=self._sampling_rng, device=self.device).item()
        return "A" if u < self._path_a_prob else "B"

    def _forward_backward(self, batch: dict) -> dict[str, float]:
        batch = self._move_batch(batch)
        ctx = self._content_inputs(batch)
        mask_t, proj_t = self._teacher_forward(ctx)
        if mask_t.numel() == 0 or int(mask_t.sum().item()) == 0:
            return {
                "loss": 0.0, "loss/mse": 0.0, "loss/cos": 0.0,
                "diag/cos_sim": 0.0, "skipped_empty_teacher": 1.0,
            }

        path = self._pick_path()
        frac_extras = self._current_frac_extras()
        if path == "A":
            student_out = self._student_forward_path_a(ctx, mask_t)
        else:
            student_out = self._student_forward_path_b(ctx, mask_t, frac_extras)
        proj_s = student_out.survivor_embeddings

        total, stats = self._distill_loss(proj_s, proj_t)
        (total / self._accum_steps).backward()

        metrics = {"loss": total.detach(), **stats}
        metrics["diag/path_a_frac"] = torch.tensor(1.0 if path == "A" else 0.0)
        metrics["diag/frac_extras"] = torch.tensor(frac_extras)
        student_out.release()
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.encoder_student.eval()
        totals_a: dict[str, float] = {"loss": 0.0, "cos_sim": 0.0, "mse": 0.0, "n": 0.0}
        totals_b: dict[str, float] = {"loss": 0.0, "cos_sim": 0.0, "mse": 0.0, "n": 0.0}
        frac_extras = self._current_frac_extras()
        for batch in self.eval_dataloader:
            batch = self._move_batch(batch)
            ctx = self._content_inputs(batch)
            mask_t, proj_t = self._teacher_forward(ctx)
            if mask_t.numel() == 0 or int(mask_t.sum().item()) == 0:
                continue

            for path, totals in (("A", totals_a), ("B", totals_b)):
                if path == "A":
                    student_out = self._student_forward_path_a(ctx, mask_t)
                else:
                    student_out = self._student_forward_path_b(ctx, mask_t, frac_extras)
                proj_s = student_out.survivor_embeddings
                if proj_s.shape != proj_t.shape:
                    student_out.release()
                    continue
                total, stats = self._distill_loss(proj_s, proj_t)
                totals["loss"] += float(total.item())
                totals["cos_sim"] += float(stats["diag/cos_sim"].item())
                totals["mse"] += float(stats["loss/mse"].item())
                totals["n"] += 1.0
                student_out.release()

        # Re-enable train() mode on the trainable components for the next step.
        self._restore_train_mode()

        out: dict[str, float] = {}
        for label, totals in (("path_A", totals_a), ("path_B", totals_b)):
            n = max(totals["n"], 1.0)
            out[f"loss_{label}"] = totals["loss"] / n
            out[f"cosine_{label}"] = totals["cos_sim"] / n
            out[f"mse_{label}"] = totals["mse"] / n
        # Primary metric for registry (lower-is-better).
        out["loss"] = 0.5 * (out["loss_path_A"] + out["loss_path_B"])
        return out

    def _restore_train_mode(self) -> None:
        if self._unfreeze_bridge:
            self.encoder_student.l0.auto_repro_head.train()
        if self._unfreeze_l0_norm:
            self.encoder_student.l0.norm.train()
        if self._unfreeze_l1_norm:
            self.encoder_student.l1.norm.train()
        if self._unfreeze_projection_block:
            self.encoder_student.projection_block.train()
        l0_blocks = self._resolve_blocks(self.encoder_student.l0.backbone)
        if self._unfreeze_l0_last_blocks > 0:
            for b in l0_blocks[-self._unfreeze_l0_last_blocks:]:
                b.train()
        l1_blocks = self._resolve_blocks(self.encoder_student.l1.backbone)
        if self._unfreeze_l1_first_blocks > 0:
            for b in l1_blocks[: self._unfreeze_l1_first_blocks]:
                b.train()

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
            encoder=self.encoder_student.state_dict(),
            optimizer_state_by_name=self._build_optimizer_state_by_name(),
        )
        if self._decoder_state_dict is not None:
            save_kwargs["decoder"] = self._decoder_state_dict
        if self._decoder_merged_state_dict is not None:
            save_kwargs["decoder_merged"] = self._decoder_merged_state_dict
        ckpt_path = save_checkpoint(checkpoint_dir, metadata, **save_kwargs)
        self._last_checkpoint_path = str(ckpt_path)
        return ckpt_path

    def _named_parameters_for_optimizer(self):
        for name, param in self.encoder_student.named_parameters():
            yield f"encoder_student.{name}", param

    def _restore_model_state(self, state_dicts: dict) -> None:
        # Teacher stays pinned to the Step 4 source checkpoint loaded in setup;
        # only the student moves on resume.
        if "encoder" in state_dicts:
            self.encoder_student.load_state_dict(state_dicts["encoder"], strict=False)
        if "decoder" in state_dicts:
            self._decoder_state_dict = state_dicts["decoder"]


__all__ = ["BridgeDistillTrainer", "_build_l0_and_l1_forced_masks"]
