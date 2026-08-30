"""Phase 2 (recursive L1): L1->L1 bridge distillation.

Trains ONLY the encoder-internal ``encoder.l1l1_bridge`` (the L1->L1
recursive bridge — the analog of ``l0.auto_repro_head``, the L0->L1 bridge),
optionally also the L1 final norm / last L1 backbone block. Everything else is
frozen. The bridge is DECODER-AGNOSTIC: no decoder is loaded and
``projection_blocks`` are never consulted.

Two distillation objectives are computed jointly per microbatch and SUMMED;
both backprop only into the L1->L1 bridge (+ optional L1 norm / last block):

* **Objective A (identity).** teacher = ``L0(r_l0) -> auto_repro_head_l0``;
  student feeds that same representation into ``L1(r=1, all-survive) ->
  l1l1_bridge``. At an ideal bridge the student reproduces the teacher
  (identity in L1-input space). Sweeps ``r_l0``.

* **Objective B (compression-matching).** teacher = ``L0(r_target) ->
  auto_repro_head_l0`` (L0 COMPRESSES at ``r_target``); student =
  ``L0(r=1, all-survive) -> auto_repro_head_l0 -> L1(r_target,
  forced-survivor-mask = L0(r_target)'s mask) -> l1l1_bridge`` (L0 full, L1
  compresses at ``r_target``). Forcing L1's survivor mask to equal
  ``L0(r_target)``'s mask keeps the two on the SAME content positions in the
  SAME order, so the per-position MSE+cosine+log-norm loss is aligned. This
  teaches L1's compression (in L1-input space) to REPLICATE L0's compression —
  the real basis for recursive L1, not just identity. Sweeps ``r_target``.

Loss surface: ``ProjectionRepairTrainer._anchor_losses`` (MSE + cosine +
log-norm), reused directly.

Start checkpoint: ``phase2_kb_step1194`` (loaded via
``BgKITEncoder.from_pretrained_with_state_dict``; the bridge is clone-init'd
from ``l0.auto_repro_head`` because the checkpoint predates ``l1l1_bridge.*``).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import structlog
import torch
from torch.utils.data import DataLoader, random_split

from bgkit.data.collators import collate_token_ids
from bgkit.data.datasets.mmap_token_dataset import MmapTokenDataset
from bgkit.data.samplers import PackedTokenBudgetSampler
from bgkit.models.encoder import BgKITEncoder
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpoint_registry import resolve_checkpoint
from bgkit.training.checkpointing import load_checkpoint
from bgkit.training.gradient_utils import maybe_enable_gradient_checkpointing
from bgkit.training.phase1.projection_repair import ProjectionRepairTrainer
from bgkit.utils.attention_backend import resolve_attention_implementation
from bgkit.utils.packing import position_ids_from_cu

logger = structlog.get_logger()


def _ratio_or_none(r: float) -> float | None:
    """Map a sampled ratio to the LevelCompressor convention: ``>= 0.999``
    means "no compression / all survive" (compression_off), expressed as None."""
    return None if float(r) >= 0.999 else float(r)


class L1L1RecursiveDistillTrainer(BaseTrainer):
    """Phase 2 recursive-L1: distill the L1->L1 bridge (``encoder.l1l1_bridge``)."""

    LIVE_CONFIG_FIELDS: ClassVar[dict[str, str]] = {
        "anchor_weight": "_anchor_weight",
        "cos_weight": "_cos_weight",
        "norm_weight": "_norm_weight",
        "objective_a_weight": "_objective_a_weight",
        "objective_b_weight": "_objective_b_weight",
    }

    def setup(self) -> None:
        tcfg = self.cfg.training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        attention_impl = resolve_attention_implementation(
            self.cfg.compute.get("attention_implementation", "auto")
        )

        # ---- Loss weights (live-tunable) ----
        self._anchor_weight = float(tcfg.get("anchor_weight", 1.0))
        self._cos_weight = float(tcfg.get("cos_weight", 1.0))
        self._norm_weight = float(tcfg.get("norm_weight", 0.1))
        self._objective_a_weight = float(tcfg.get("objective_a_weight", 1.0))
        self._objective_b_weight = float(tcfg.get("objective_b_weight", 1.0))

        # ---- Ratio sweeps ----
        self._a_ratios = [
            float(r) for r in tcfg.get("objective_a_l0_ratios", [1.0, 0.5, 0.1, 0.05, 0.02])
        ]
        self._b_ratios = [
            float(r) for r in tcfg.get("objective_b_ratios", [0.5, 0.1, 0.05, 0.02])
        ]
        # "sweep" (default): every ratio per microbatch (matches the original
        # task's "Sweep ... per microbatch"). "sample": one ratio per objective
        # per microbatch (cheaper; covers the lists stochastically across
        # microbatches). Orchestrator can flip via config for memory.
        self._ratio_selection = str(tcfg.get("ratio_selection", "sweep")).lower()
        if self._ratio_selection not in {"sweep", "sample"}:
            raise ValueError(
                f"ratio_selection must be 'sweep' or 'sample'; got {self._ratio_selection!r}"
            )

        # Floor on per-sample survivor count (keeps FA varlen kernels in range
        # on short samples at tight ratios — same rationale as bridge_distill).
        self._min_per_sample = int(tcfg.get("min_per_sample", 4))

        unfreeze_cfg = tcfg.get("unfreeze", {}) or {}
        self._unfreeze_l1_norm = bool(unfreeze_cfg.get("l1_norm", False))
        self._unfreeze_l1_last_blocks = int(unfreeze_cfg.get("l1_last_blocks", 0))

        seed = int(self.cfg.get("seed", 42))
        self._sampling_rng = torch.Generator(device=device)
        self._sampling_rng.manual_seed(seed)

        # ---- Resolve + load source checkpoint ----
        src_ckpt = self._resolve_source_checkpoint()
        if src_ckpt is None:
            raise ValueError(
                "phase2_kb_l1l1 requires bgkit_checkpoint set (path or 'auto')."
            )
        bgkit_cfg = self.cfg.model.bgkit
        backbone_name = bgkit_cfg.backbone_name
        hidden_dim = bgkit_cfg.get("hidden_dim", 1024)

        logger.info("loading_source_checkpoint", path=src_ckpt)
        _, state_dicts = load_checkpoint(Path(src_ckpt))
        # KRKBTrainer (Stage A) saves the wrapped model under "model" with
        # encoder params prefixed "encoder." (decoder under "decoder."); older
        # phases may save a bare "encoder" dict. Mirror KRKBTrainer's loader.
        encoder_state = state_dicts.get("encoder")
        if encoder_state is None:
            model_state = state_dicts.get("model", {})
            encoder_state = {
                k.replace("encoder.", "", 1): v
                for k, v in model_state.items()
                if k.startswith("encoder.")
            }
        if not encoder_state:
            raise ValueError(f"checkpoint {src_ckpt} contains no encoder state")
        # Decoder state (if any) is irrelevant here — passed through unchanged
        # so a downstream resume can still find it.
        self._decoder_state_dict = state_dicts.get("decoder")
        self._decoder_merged_state_dict = state_dicts.get("decoder_merged")

        # Threshold-controller anchor grid: the per-level DualThresholdController
        # buffers (anchor_ratios / anchor_thetas / _anchor_velocity) are sized by
        # the anchor grid used at the checkpoint's train time. The Stage A
        # (phase2_kb_step1194) encoder inherits the Falcon/summarization 6-anchor
        # grid, NOT the Qwen 7-anchor model default — so derive the grid from the
        # SAVED state to size the reconstructed buffers identically, else
        # load_state_dict raises a size mismatch on l{0,1}.threshold.anchor_*
        # (6 vs 7). Mirrors KRKBTrainer's setup. The threshold controller is
        # frozen in this distillation; only the anchor COUNT (buffer shape) has
        # to match for the state load.
        threshold_cfg = dict(self.cfg.model.get("threshold_controller", {}) or {})
        saved_anchors = encoder_state.get("l0.threshold.anchor_ratios")
        if saved_anchors is not None:
            threshold_cfg["anchor_ratios"] = saved_anchors.tolist()

        self.encoder = BgKITEncoder.from_pretrained_with_state_dict(
            backbone_name,
            encoder_state,
            hidden_dim=hidden_dim,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=bgkit_cfg.get("backbone_revision", None),
            attn_implementation=attention_impl,
            bidi_warmup_steps=0,
            threshold_controller_cfg=threshold_cfg or None,
        )
        self.encoder.to(device)
        self._freeze_for_l1l1_distill(self.encoder)
        self.model = self.encoder

        if tcfg.get("use_liger", True):
            from bgkit.utils.liger_integration import apply_liger_to_qwen35

            apply_liger_to_qwen35(
                self.encoder,
                patch_rmsnorm=bool(tcfg.get("use_liger_rmsnorm", False)),
                patch_swiglu=bool(tcfg.get("use_liger_swiglu", True)),
                patch_rope=bool(tcfg.get("use_liger_rope", True)),
            )

        # L1 backbone GC only matters when its last block is unfrozen.
        if self._unfreeze_l1_last_blocks > 0:
            maybe_enable_gradient_checkpointing(self.encoder.l1.backbone, self.cfg)

        # ---- Dataset: NL token corpus (L0-encodable content; no QA labels) ----
        data_dir = self.cfg.data.tokens.input_dir
        max_seq_len = self.cfg.data.tokens.get("max_seq_len", 4096)
        full_dataset = MmapTokenDataset(
            data_dir, max_seq_len=max_seq_len, include_metadata=False,
        )

        max_eval_samples = tcfg.get("max_eval_samples", 1000)
        total = len(full_dataset)
        eval_size = min(max(1, int(total * 0.1)), max_eval_samples)
        train_size = total - eval_size
        if train_size < 1:
            raise ValueError(f"Dataset too small for split (got {total} samples)")
        split_generator = torch.Generator().manual_seed(seed)
        self.train_dataset, self.eval_dataset = random_split(
            full_dataset, [train_size, eval_size], generator=split_generator,
        )

        num_workers = self.cfg.compute.get("num_workers", 4)
        pin_memory = self.cfg.compute.get("pin_memory", False)
        train_lengths = full_dataset.lengths[np.array(self.train_dataset.indices)]
        eval_lengths = full_dataset.lengths[np.array(self.eval_dataset.indices)]

        self._train_lengths = train_lengths
        self._train_content_lengths = train_lengths
        self._eval_lengths = eval_lengths
        self._train_collate_fn = collate_token_ids
        self._num_workers = num_workers
        self._pin_memory = pin_memory

        max_batch_tokens = int(tcfg.get("max_batch_tokens", 8192))
        self._max_batch_tokens = max_batch_tokens
        self._max_batch_tokens_eval = self._resolve_eval_batch_budget(tcfg, max_batch_tokens)
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
            collate_fn=collate_token_ids,
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
            collate_fn=collate_token_ids,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # ---- Optimizer (bridge + optional L1 norm/last block) ----
        trainable = self.trainable_parameters()
        if not trainable:
            raise RuntimeError(
                "L1L1RecursiveDistillTrainer found no trainable parameters."
            )
        param_groups = [{"params": trainable, "lr": tcfg.lr, "base_lr": tcfg.lr}]
        self.optimizer = self._create_optimizer(
            param_groups, float(tcfg.lr), exclude_from_muon=frozenset(),
        )

        logger.info(
            "l1l1_distill_setup",
            src_checkpoint=src_ckpt,
            train_samples=train_size,
            eval_samples=eval_size,
            a_ratios=self._a_ratios,
            b_ratios=self._b_ratios,
            ratio_selection=self._ratio_selection,
            unfreeze_l1_norm=self._unfreeze_l1_norm,
            unfreeze_l1_last_blocks=self._unfreeze_l1_last_blocks,
            trainable_params=sum(p.numel() for p in trainable),
        )

    # ------------------------------------------------------------------
    # Freeze plan
    # ------------------------------------------------------------------

    def _freeze_for_l1l1_distill(self, encoder: BgKITEncoder) -> None:
        encoder.requires_grad_(False)
        encoder.eval()

        encoder.l1l1_bridge.requires_grad_(True)
        encoder.l1l1_bridge.train()

        if self._unfreeze_l1_norm:
            encoder.l1.norm.requires_grad_(True)
            encoder.l1.norm.train()
        if self._unfreeze_l1_last_blocks > 0:
            for b in self._resolve_blocks(encoder.l1.backbone)[-self._unfreeze_l1_last_blocks:]:
                b.requires_grad_(True)
                b.train()

    @staticmethod
    def _resolve_blocks(backbone) -> list:
        for attr in ("blocks", "layers"):
            mod = getattr(backbone, attr, None)
            if mod is not None and len(mod) > 0:
                return list(mod)
        raise ValueError(f"Cannot find blocks/layers in {type(backbone).__name__}")

    def trainable_parameters(self) -> list:
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def _restore_train_mode(self) -> None:
        self.encoder.l1l1_bridge.train()
        if self._unfreeze_l1_norm:
            self.encoder.l1.norm.train()
        if self._unfreeze_l1_last_blocks > 0:
            l1_blocks = self._resolve_blocks(self.encoder.l1.backbone)
            for b in l1_blocks[-self._unfreeze_l1_last_blocks:]:
                b.train()

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _content_inputs(self, batch: dict):
        device = self.device
        input_ids = batch["input_ids"].to(device)
        cu = batch["cu_seqlens"].to(device)
        pos = batch["position_ids"].to(device)
        embed = self.encoder.l0.backbone.get_input_embeddings()
        content_emb = embed(input_ids)
        return content_emb, cu, pos

    # ------------------------------------------------------------------
    # Objective forwards
    #
    # The whole L0 side (teacher + student's L0 stage) is frozen, so it runs
    # under no_grad to save memory. Gradient is needed only from the L1
    # backbone forward (when the last L1 block is unfrozen) and the
    # l1l1_bridge / l1.norm, which run in grad mode. The L1 input is a detached
    # tensor, so the graph stops there and only the bridge (+ optional L1
    # bits) accumulate gradient.
    # ------------------------------------------------------------------

    def _objective_a(self, content_emb, cu, pos, r_l0):
        """Identity: teacher = L0(r_l0)->bridge_l0; student feeds it to
        L1(r=1, all-survive)->l1l1_bridge."""
        with torch.no_grad():
            l0_out = self.encoder.l0(
                content_embeddings=content_emb,
                content_cu_seqlens=cu,
                content_position_ids=pos,
                target_ratio=_ratio_or_none(r_l0),
                min_per_sample=self._min_per_sample,
            )
            target = self.encoder.l0.auto_reproduce(l0_out.survivor_embeddings).detach()
            surv_cu = l0_out.survivor_cu_seqlens.clone()
        l0_out.release()
        if target.shape[0] == 0:
            return None
        l1_pos = position_ids_from_cu(surv_cu, target.shape[0])
        l1_out = self.encoder.l1(
            content_embeddings=target,
            content_cu_seqlens=surv_cu,
            content_position_ids=l1_pos,
            target_ratio=None,  # r=1, all-survive
        )
        student = self.encoder.l1_auto_reproduce(l1_out.survivor_embeddings)
        l1_out.release()
        return student, target

    def _objective_b(self, content_emb, cu, pos, r_target):
        """Compression-matching: teacher = L0(r_target)->bridge_l0; student =
        L0(r=1)->bridge_l0 -> L1(r_target, forced mask = L0(r_target) mask) ->
        l1l1_bridge. Forced mask keeps L1 on the exact L0 survivor positions so
        the per-position loss is aligned."""
        with torch.no_grad():
            l0_t = self.encoder.l0(
                content_embeddings=content_emb,
                content_cu_seqlens=cu,
                content_position_ids=pos,
                target_ratio=_ratio_or_none(r_target),
                min_per_sample=self._min_per_sample,
            )
            teacher = self.encoder.l0.auto_reproduce(l0_t.survivor_embeddings).detach()
            teacher_mask = l0_t.survivor_mask.detach().clone()  # (N_content,)
            l0_t.release()
            # Student's L0 stage: L0 full (r=1) -> all content survives.
            l0_full = self.encoder.l0(
                content_embeddings=content_emb,
                content_cu_seqlens=cu,
                content_position_ids=pos,
                target_ratio=None,
            )
            l1_input = self.encoder.l0.auto_reproduce(l0_full.survivor_embeddings).detach()
            l1_input_cu = l0_full.survivor_cu_seqlens.clone()
            l0_full.release()
        if teacher.shape[0] == 0:
            return None
        # l1_input spans all N_content positions; teacher_mask is over the same
        # N_content axis, so it forces L1 to keep exactly L0(r_target)'s
        # survivors, in content order -> aligned with `teacher`.
        if teacher_mask.shape[0] != l1_input.shape[0]:
            return None
        l1_pos = position_ids_from_cu(l1_input_cu, l1_input.shape[0])
        l1_out = self.encoder.l1(
            content_embeddings=l1_input,
            content_cu_seqlens=l1_input_cu,
            content_position_ids=l1_pos,
            target_ratio=float(r_target),
            forced_survivor_mask=teacher_mask,
            min_per_sample=self._min_per_sample,
        )
        student = self.encoder.l1_auto_reproduce(l1_out.survivor_embeddings)
        l1_out.release()
        return student, teacher

    def _anchor_total(self, student, target):
        # Reuse ProjectionRepairTrainer._anchor_losses verbatim (it reads no
        # state off `self`; passing this trainer as `self` is harmless).
        mse, stats, cos_loss, norm_loss = ProjectionRepairTrainer._anchor_losses(
            self, student, target,
        )
        total = (
            self._anchor_weight * mse
            + self._cos_weight * cos_loss
            + self._norm_weight * norm_loss
        )
        return total, mse, cos_loss, norm_loss, stats

    def _pick_ratios(self, ratios: list[float]) -> list[float]:
        if self._ratio_selection == "sweep":
            return ratios
        idx = int(
            torch.randint(
                0, len(ratios), (1,), generator=self._sampling_rng, device=self.device,
            ).item()
        )
        return [ratios[idx]]

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _accumulate_objective(self, fn, content_emb, cu, pos, ratios):
        """Run an objective over its ratio set; return (mean_loss, mean_mse,
        mean_cos_sim, n) — loss summed then averaged over valid ratios."""
        total = torch.zeros((), device=self.device)
        mse_sum = torch.zeros((), device=self.device)
        cos_sum = torch.zeros((), device=self.device)
        n = 0
        for r in self._pick_ratios(ratios):
            res = fn(content_emb, cu, pos, r)
            if res is None:
                continue
            student, target = res
            if student.shape != target.shape:
                continue
            loss_r, mse, _cos_loss, _norm_loss, stats = self._anchor_total(student, target)
            total = total + loss_r
            mse_sum = mse_sum + mse.detach()
            cos_sum = cos_sum + stats["cos_sim_mean"]
            n += 1
        if n == 0:
            return None
        return total / n, mse_sum / n, cos_sum / n, n

    def _forward_backward(self, batch: dict) -> dict[str, float]:
        content_emb, cu, pos = self._content_inputs(batch)

        res_a = self._accumulate_objective(self._objective_a, content_emb, cu, pos, self._a_ratios)
        res_b = self._accumulate_objective(self._objective_b, content_emb, cu, pos, self._b_ratios)

        if res_a is None and res_b is None:
            return {"loss": 0.0, "skipped_empty": 1.0}

        total = torch.zeros((), device=self.device)
        metrics: dict[str, float | torch.Tensor] = {}
        if res_a is not None:
            la, mse_a, cos_a, _ = res_a
            total = total + self._objective_a_weight * la
            metrics["loss/objective_a"] = la.detach()
            metrics["diag/mse_a"] = mse_a
            metrics["diag/cos_sim_a"] = cos_a
        if res_b is not None:
            lb, mse_b, cos_b, _ = res_b
            total = total + self._objective_b_weight * lb
            metrics["loss/objective_b"] = lb.detach()
            metrics["diag/mse_b"] = mse_b
            metrics["diag/cos_sim_b"] = cos_b

        metrics["loss"] = total.detach()
        (total / self._accum_steps).backward()
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.encoder.eval()
        totals: dict[str, float] = {}
        count = 0
        for batch in self.eval_dataloader:
            content_emb, cu, pos = self._content_inputs(batch)
            res_a = self._accumulate_objective(
                self._objective_a, content_emb, cu, pos, self._a_ratios,
            )
            res_b = self._accumulate_objective(
                self._objective_b, content_emb, cu, pos, self._b_ratios,
            )
            if res_a is None and res_b is None:
                continue
            la = float(res_a[0].item()) if res_a is not None else 0.0
            lb = float(res_b[0].item()) if res_b is not None else 0.0
            totals["loss_objective_a"] = totals.get("loss_objective_a", 0.0) + la
            totals["loss_objective_b"] = totals.get("loss_objective_b", 0.0) + lb
            totals["loss"] = totals.get("loss", 0.0) + (
                self._objective_a_weight * la + self._objective_b_weight * lb
            )
            if res_a is not None:
                totals["cos_sim_a"] = totals.get("cos_sim_a", 0.0) + float(res_a[2].item())
            if res_b is not None:
                totals["cos_sim_b"] = totals.get("cos_sim_b", 0.0) + float(res_b[2].item())
            count += 1
        if count > 0:
            totals = {k: v / count for k, v in totals.items()}
        self._restore_train_mode()
        return totals

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def checkpoint_models(self) -> dict[str, torch.nn.Module]:
        """Modules this trainer owns.

        Was a hand-rolled save_checkpoint override calling module-level
        save_checkpoint() directly, bypassing _write_checkpoint — writing to
        the spinning HDD and skipping async archival (the 2026-06-10 NVMe
        routing bug, fixed once in summarization_round_robin and never
        propagated). Eleven trainers carried it.
        """
        return {"encoder": self.encoder}

    def checkpoint_extra_state(self) -> dict[str, dict]:
        """Frozen decoder + its LoRA-merged form, when supplied."""
        out: dict[str, dict] = {}
        if getattr(self, "_decoder_state_dict", None) is not None:
            out["decoder"] = self._decoder_state_dict
        if getattr(self, "_decoder_merged_state_dict", None) is not None:
            out["decoder_merged"] = self._decoder_merged_state_dict
        return out


    def _named_parameters_for_optimizer(self):
        for name, param in self.encoder.named_parameters():
            yield f"encoder.{name}", param

    def _restore_extra_state(self, state_dicts: dict) -> None:
        """Stash the frozen decoder for a later phase; not a live module."""
        if "decoder" in state_dicts:
            self._decoder_state_dict = state_dicts["decoder"]

    # ------------------------------------------------------------------
    # Checkpoint resolution
    # ------------------------------------------------------------------

    def _resolve_source_checkpoint(self) -> str | None:
        src = self.cfg.get("bgkit_checkpoint", None)
        if isinstance(src, str) and src.startswith("auto"):
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            phase = src[len("auto_"):] if src.startswith("auto_") else "phase2_kb"
            resolved = resolve_checkpoint(
                checkpoint_dir,
                phase=phase,
                metric="eval/loss",
                label="bgkit_checkpoint",
            )
            src = str(resolved)
        self._input_sources = {"bgkit": Path(src).name} if src else {}
        return src


__all__ = ["L1L1RecursiveDistillTrainer"]
