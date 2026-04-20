"""Commit encoding: two-level hierarchical encoder training on git commits.

Single-objective trainer with L0 (per-file) + L1 (cross-file) compression
from step 0. Asymmetric encoder/decoder prompting: encoder gets commit
message as conditioning, decoder must reconstruct the full commit purely
from compressed survivors.

Pipeline position: Step 4 — between Step 3 (pruned reconstruction) and Step 5
(multi-objective compression). Takes Step 1 or Step 2 checkpoint, unfreezes
both encoder and decoder. Produces checkpoints in the same multi-artifact
format as CompressionTrainer (encoder.pt + decoder.pt) so Step 4 can consume them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import ClassVar

import numpy as np
import structlog
import torch
from torch.utils.data import DataLoader, random_split

from bgkit.data.collators import collate_compression
from bgkit.data.datasets.commit_encoding_dataset import CommitEncodingDataset
from bgkit.data.samplers import LengthSortedBatchSampler
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpoint_registry import resolve_checkpoint
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.training.gradient_utils import enable_gradient_checkpointing
from bgkit.utils.attention_backend import resolve_attention_implementation

logger = structlog.get_logger()


class CommitEncodingTrainer(BaseTrainer):
    """Single-objective trainer for two-level commit encoding.

    Uses BaseTrainer.train() loop — no L1 introduction hook needed since
    both L0 and L1 are active from step 0. Overrides save_checkpoint()
    and load_checkpoint() to use the multi-artifact format that
    CompressionTrainer expects.
    """

    LIVE_CONFIG_FIELDS: ClassVar[dict[str, str]] = {
        "target_ratio_ramp_steps": "_target_ratio_ramp_steps",
        "target_ratio_start": "_target_ratio_start",
        "target_ratio_end": "_target_ratio_end",
    }

    def setup(self) -> None:
        """Load trainable encoder/decoder, create dataset and optimizer."""
        tcfg = self.cfg.training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        attention_impl = resolve_attention_implementation(
            self.cfg.compute.get("attention_implementation", "auto")
        )

        # --- BgKIT encoder (trainable) ---
        bgkit_cfg = self.cfg.model.bgkit
        backbone_name = bgkit_cfg.backbone_name
        backbone_revision = bgkit_cfg.get("backbone_revision", None)
        hidden_dim = bgkit_cfg.get("hidden_dim", 1024)
        bidi_warmup = tcfg.get("bidi_warmup_steps", 500)

        step1_checkpoint = self._resolve_step1_checkpoint()
        step1_state_dicts: dict | None = None
        if step1_checkpoint is not None:
            logger.info("loading_step1_checkpoint", path=step1_checkpoint)
            _, step1_state_dicts = load_checkpoint(Path(step1_checkpoint))
            if "encoder" not in step1_state_dicts:
                raise ValueError(
                    f"Step 2 checkpoint missing 'encoder' key: {step1_checkpoint}. "
                    f"Found keys: {list(step1_state_dicts.keys())}"
                )

        # Auto-detect pruned architecture from state dict keys
        if step1_state_dicts:
            self.encoder = BgKITEncoder.from_pretrained_with_state_dict(
                backbone_name,
                step1_state_dicts["encoder"],
                hidden_dim=hidden_dim,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                revision=backbone_revision,
                attn_implementation=attention_impl,
                bidi_warmup_steps=bidi_warmup,
            )
        else:
            self.encoder = BgKITEncoder.from_pretrained(
                backbone_name,
                hidden_dim=hidden_dim,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                revision=backbone_revision,
                attn_implementation=attention_impl,
                bidi_warmup_steps=bidi_warmup,
            )
        self.encoder.to(device)

        self.encoder.requires_grad_(True)
        self.encoder.train()
        enable_gradient_checkpointing(self.encoder.compressor.backbone)

        # --- Decoder (trainable) ---
        decoder_cfg = self.cfg.model.decoder
        decoder_name = decoder_cfg.backbone_name
        decoder_revision = decoder_cfg.get("backbone_revision", None)
        logger.info("loading_decoder", model=decoder_name, revision=decoder_revision)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Build decoder WITHOUT NVFP4 first so checkpoint loads cleanly
        decoder_backbone = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=decoder_revision,
            attn_implementation=attention_impl,
            device_map=device,  # load directly to CUDA, avoid CPU staging copy
        )
        self.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=hidden_dim)

        if step1_state_dicts is not None:
            # Prefer merged decoder (LoRA weights folded in) over raw PeftModel state
            decoder_sd = step1_state_dicts.get(
                "decoder_merged", step1_state_dicts.get("decoder")
            )
            if decoder_sd is not None:
                self.decoder.load_state_dict(decoder_sd)

        # LoRA wrapping (after checkpoint load, before NVFP4/gradient checkpointing)
        self._decoder_lora = False
        lora_cfg = tcfg.get("decoder_lora", {})
        if lora_cfg.get("enabled", False):
            self.decoder.apply_lora(lora_cfg)
            self._decoder_lora = True

        # NVFP4 disabled: TE's fp8_autocast saves non-deterministic scaling
        # tensors that break PyTorch gradient checkpointing's save/restore
        # invariants (122 vs 87 tensors between forward and recompute, causing
        # AssertionError in unpack_hook). This is a fundamental incompatibility
        # between TE's stateful fp8 context and PyTorch's checkpoint replay.
        # Bypassing the validation check isn't enough — the internal handle
        # mapping also breaks. Needs upstream TE fix.

        enable_gradient_checkpointing(self.decoder.backbone)

        # torch.compile disabled: sm_121 shared memory (101 KB) is too small for
        # inductor-generated Triton kernels (need ~148 KB). "reduce-overhead" and
        # "cudagraphs" backends also use inductor or suffer from dynamic shape
        # re-recording overhead. PeftModel isinstance fix landed in decoder.py
        # for when this becomes viable.

        # Optional Liger Kernel fused kernels (RMSNorm / SwiGLU / RoPE +
        # fused linear+CE). Gated on ``training.use_liger`` (default True);
        # no-op when liger-kernel is not installed.
        if tcfg.get("use_liger", True):
            from bgkit.utils.liger_integration import apply_liger_to_qwen35

            enc_patched = apply_liger_to_qwen35(self.encoder)
            dec_patched = apply_liger_to_qwen35(self.decoder)
            self.decoder.enable_liger_ce(True)
            logger.info(
                "liger_kernel_applied",
                encoder_modules=enc_patched,
                decoder_modules=dec_patched,
            )

        # BaseTrainer uses self.model for logging
        self.model = self.decoder

        # --- Tokenizer ---
        tokenizer_name = decoder_cfg.backbone_name
        tokenizer_revision = decoder_cfg.get("backbone_revision", None)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            revision=tokenizer_revision,
        )

        # --- Dataset ---
        data_cfg = tcfg.data
        seed = self.cfg.get("seed", 42)

        # Load variant bank
        variant_bank = self._load_variant_bank(data_cfg)

        from bgkit.data.chat_template import TOOL_CONFIGS

        config = TOOL_CONFIGS["commit_encoding"]

        commit_encoding_dir = data_cfg.commit_encoding_dir
        self.commit_dataset = CommitEncodingDataset(
            data_dir=commit_encoding_dir,
            tokenizer=self.tokenizer,
            variant_bank=variant_bank,
            config=config,
            max_diff_tokens_per_file=data_cfg.get("max_diff_tokens_per_file", 4096),
            max_files_per_commit=data_cfg.get("max_files_per_commit", 16),
            max_message_tokens=data_cfg.get("max_message_tokens", 256),
            seed=seed,
        )

        # Train/eval split
        max_eval_samples = tcfg.get("max_eval_samples", 5000)
        total = len(self.commit_dataset)
        eval_size = min(max(1, int(total * 0.1)), max_eval_samples)
        train_size = total - eval_size
        if train_size < 1:
            raise ValueError(
                f"Dataset too small for train/eval split (got {total} samples)"
            )
        self.train_dataset, self.eval_dataset = random_split(
            self.commit_dataset, [train_size, eval_size],
        )

        # --- Sampler + DataLoader ---
        max_batch_tokens = tcfg.get("max_batch_tokens", 65536)
        num_workers = self.cfg.compute.get("num_workers", 4)
        pin_memory = self.cfg.compute.get("pin_memory", False)

        train_lengths = np.array([
            self.commit_dataset.token_length(i)
            for i in self.train_dataset.indices
        ], dtype=np.int64)
        eval_lengths = np.array([
            self.commit_dataset.token_length(i)
            for i in self.eval_dataset.indices
        ], dtype=np.int64)

        self.train_sampler = LengthSortedBatchSampler(
            self.train_dataset,
            max_batch_tokens,
            shuffle=True,
            seed=seed,
            lengths=train_lengths,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=collate_compression,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        eval_sampler = LengthSortedBatchSampler(
            self.eval_dataset,
            max_batch_tokens,
            shuffle=False,
            lengths=eval_lengths,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_sampler=eval_sampler,
            collate_fn=collate_compression,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # --- Optimizer ---
        self._setup_optimizer()

        # --- Curriculum state ---
        curriculum = tcfg.get("curriculum", {})
        self._target_ratio_start = curriculum.get(
            "target_ratio_start", tcfg.get("target_ratio_start", 0.50),
        )
        self._target_ratio_end = curriculum.get(
            "target_ratio_end", tcfg.get("target_ratio_end", 0.20),
        )
        self._target_ratio_ramp_steps = curriculum.get(
            "target_ratio_ramp_steps", tcfg.get("target_ratio_ramp_steps", 30000),
        )
        self._target_ratio_override: float | None = None

        # --- Survivorship head config (per-level) ---
        # Step 4 is where L1 first activates — L0 head is trained from Step 3
        # and just continues; L1 cold-starts via decisiveness + utility-grad BCE.
        from bgkit.training.survivorship_helpers import (
            init_state,
            load_reference_moments,
            resolve_level_ice_cfg,
            resolve_level_loss_cfg,
        )

        surv_cfg = tcfg.get("survivorship", {})
        self._surv_l0 = resolve_level_loss_cfg(surv_cfg.get("l0", {}))
        self._surv_l1 = resolve_level_loss_cfg(surv_cfg.get("l1", {}))

        ice_cfg = tcfg.get("ice_distillation", {})
        self._ice_l0 = resolve_level_ice_cfg(ice_cfg.get("l0", {}))
        self._ice_l1 = resolve_level_ice_cfg(ice_cfg.get("l1", {}))
        self._max_warmup_step = max(
            self._ice_l0.bce_warmup_steps if self._ice_l0.enabled else 0,
            self._ice_l1.bce_warmup_steps if self._ice_l1.enabled else 0,
        )
        self._ice_teacher = None
        if (
            (self._ice_l0.enabled and self._ice_l0.bce_warmup_weight > 0)
            or (self._ice_l1.enabled and self._ice_l1.bce_warmup_weight > 0)
        ):
            from bgkit.models.ice_teacher import ICETeacher
            ice_path = ice_cfg["checkpoint_path"]
            embed_tokens = self.encoder.compressor.backbone.embed_tokens
            self._ice_teacher = ICETeacher(
                ice_path, embed_tokens,
                input_dim=int(ice_cfg.get("input_dim", 1024)),
                hidden_dim=int(ice_cfg.get("hidden_dim", 192)),
                num_layers=int(ice_cfg.get("num_layers", 3)),
                kernel_size=int(ice_cfg.get("kernel_size", 5)),
            ).to(device)

        mm_ref = tcfg.get("moment_match_reference", {})
        self._ref_moments_l0 = None
        self._ref_moments_l1 = None
        # OmegaConf's DictConfig is not a dict subclass; use duck-typed access.
        _l0_block = mm_ref.get("l0", None) if hasattr(mm_ref, "get") else None
        l0_path = _l0_block.get("path", None) if _l0_block is not None and hasattr(_l0_block, "get") else None
        if self._surv_l0.moment_match_weight > 0 and l0_path:
            self._ref_moments_l0 = load_reference_moments(l0_path)
        _l1_block = mm_ref.get("l1", None) if hasattr(mm_ref, "get") else None
        l1_path = _l1_block.get("path", None) if _l1_block is not None and hasattr(_l1_block, "get") else None
        if self._surv_l1.moment_match_weight > 0 and l1_path:
            self._ref_moments_l1 = load_reference_moments(l1_path)

        self._surv_state_l0 = init_state()
        self._surv_state_l1 = init_state()
        self._last_post_step_metrics: dict[str, float] = {}

        # --- Diagnostic metrics cadence ---
        # Gates the GPU→CPU syncs for head-health metrics (organic rate
        # std, undecided fraction, floor trigger rate, θ). Mirrors Step 3.
        self._diagnostic_metrics_every_n_steps = int(
            tcfg.get("diagnostic_metrics_every_n_steps", 10),
        )

        # --- Head-tanh temperature calibration (L0 + L1) ---
        # L0 inherits from Step 3's sidecar-calibrated T via the loaded
        # checkpoint; L1 is brand-new at Step 4 and needs its own T set
        # against its head's init-scale output std. Running the probe
        # unconditionally at both levels is cheap (4 batches, no backward)
        # and robust to checkpoint-vs-fresh-head distinctions: if L0 was
        # already calibrated, the probe confirms it; if the user resumes
        # from a checkpoint with drifted T, the probe corrects it.
        self._calibrate_head_tanh_temperatures()

        # --- Profiling ---
        self._profile_enabled = os.environ.get("BGKIT_PROFILE", "") == "1"

        logger.info(
            "commit_encoding_trainer_setup",
            train_samples=train_size,
            eval_samples=eval_size,
            device=str(device),
            profile=self._profile_enabled,
        )

    def _calibrate_head_tanh_temperatures(self, n_probe_batches: int = 4) -> None:
        """Probe L0 + L1 head output std at startup and set their per-level
        ``head_tanh_temperature_l{0,1}`` buffers. See
        :func:`bgkit.training.survivorship_helpers.calibrate_head_tanh_temperature`.
        """
        from bgkit.training.survivorship_helpers import (
            calibrate_head_tanh_temperature,
        )
        for level in ("l0", "l1"):
            calibrated_T = calibrate_head_tanh_temperature(
                self.encoder.compressor,
                self.train_dataloader,
                self.device,
                level=level,
                n_probe_batches=n_probe_batches,
            )
            if calibrated_T is not None:
                logger.info(
                    "head_tanh_temperature_calibrated",
                    level=level,
                    T=calibrated_T,
                )

    def _load_variant_bank(self, data_cfg) -> list[dict[str, str]]:
        """Load commit_encoding variant bank from prompt_variants_dir."""
        variant_dir = getattr(data_cfg, "prompt_variants_dir", None)
        if variant_dir:
            bank_path = Path(variant_dir) / "commit_encoding.json"
            if bank_path.exists():
                with open(bank_path) as f:
                    bank = json.load(f)
                if bank:
                    return bank

        # Fallback
        return [{
            "system_prompt": "You are an AI assistant with access to the "
                             "bgkit_reproduce_commit tool.",
            "user_prompt": "Reproduce the commit from repository {file_path}",
            "compression_prompt": "Reproduce the complete commit from "
                                  "compressed context",
            "response_prefix": "Here is the reconstructed commit:",
        }]

    def _resolve_step1_checkpoint(self) -> str | None:
        """Resolve step1_checkpoint: auto -> best phase1_step4 checkpoint."""
        step1_checkpoint = self.cfg.get("step1_checkpoint", None)
        self._input_sources = {}

        if step1_checkpoint == "auto":
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            resolved = resolve_checkpoint(
                checkpoint_dir,
                phase="phase1_step4",
                metric="eval/loss",
                label="step1_checkpoint",
            )
            step1_checkpoint = str(resolved)

        if step1_checkpoint is not None:
            self._input_sources["step1"] = Path(step1_checkpoint).name

        return step1_checkpoint

    def _muon_excluded_param_ids(self) -> frozenset[int]:
        """Return param IDs of decoder embedding/lm_head — should not use Muon."""
        exclude = set()
        backbone = self.decoder.backbone
        try:
            from peft import PeftModel
            causal_lm = backbone.base_model.model if isinstance(backbone, PeftModel) else backbone
        except ImportError:
            causal_lm = backbone
        inner = getattr(causal_lm, "model", None)
        if inner is not None and hasattr(inner, "embed_tokens"):
            for p in inner.embed_tokens.parameters():
                if p.requires_grad:
                    exclude.add(id(p))
        lm_head = getattr(causal_lm, "lm_head", None)
        if lm_head is not None:
            for p in lm_head.parameters():
                if p.requires_grad:
                    exclude.add(id(p))
        return frozenset(exclude)

    def _setup_optimizer(self) -> None:
        """Create optimizer with encoder + decoder param groups."""
        tcfg = self.cfg.training

        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        decoder_params = [p for p in self.decoder.parameters() if p.requires_grad]

        encoder_lr = tcfg.get("encoder_lr", tcfg.lr)
        decoder_lr = tcfg.get("decoder_lr", tcfg.lr)

        param_groups = []
        if encoder_params:
            param_groups.append({
                "params": encoder_params,
                "lr": encoder_lr,
                "base_lr": encoder_lr,
            })
        if decoder_params:
            param_groups.append({
                "params": decoder_params,
                "lr": decoder_lr,
                "base_lr": decoder_lr,
            })

        self.optimizer = self._create_optimizer(
            param_groups, tcfg.lr, exclude_from_muon=self._muon_excluded_param_ids()
        )

    def _trainable_params(self) -> list[torch.nn.Parameter]:
        params = [p for p in self.encoder.parameters() if p.requires_grad]
        params += [p for p in self.decoder.parameters() if p.requires_grad]
        return params

    def trainable_parameters(self) -> list:
        return self._trainable_params()

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _current_target_ratio(self) -> float:
        if self._target_ratio_override is not None:
            return self._target_ratio_override
        step = self.global_step
        if step >= self._target_ratio_ramp_steps:
            return self._target_ratio_end
        t = step / max(self._target_ratio_ramp_steps, 1)
        return self._target_ratio_start + t * (
            self._target_ratio_end - self._target_ratio_start
        )

    # ------------------------------------------------------------------
    # Compression pipeline: L0 per-file -> L1 cross-file
    # ------------------------------------------------------------------

    @staticmethod
    def _compress_files_batched_pure(
        compressor: torch.nn.Module,
        all_f_emb: torch.Tensor,
        all_f_mask: torch.Tensor,
        prompt_emb: torch.Tensor,
        prompt_mask: torch.Tensor,
        target_ratio: float,
    ) -> torch.Tensor:
        """Batched L0 compression for checkpointing.

        Returns concatenated survivors as (total_surv, D). Used by the
        activation-checkpointing wrapper, where the CompressorOutput can't
        cross the checkpoint boundary — losses/state accumulation happen
        in the non-checkpointed path only.
        """
        comp_out = compressor(
            all_f_emb,
            attention_mask=all_f_mask,
            prompt_embeddings=prompt_emb,
            prompt_attention_mask=prompt_mask,
            target_ratio=target_ratio,
            level="l0",
        )
        content_normed = comp_out.normed_embeddings[:, comp_out.content_slice, :]
        survivor_mask = comp_out.survivor_mask
        # Extract per-file survivors and auto_reproduce each (variable survivor count)
        all_survivors = []
        if survivor_mask is not None:
            for f in range(all_f_emb.shape[0]):
                surv = content_normed[f][survivor_mask[f]]
                all_survivors.append(compressor.auto_reproduce(surv.unsqueeze(0))[0])
        if all_survivors:
            return torch.cat(all_survivors, dim=0)
        return torch.zeros(
            1, compressor.hidden_dim, device=all_f_emb.device, dtype=all_f_emb.dtype,
        )

    def _l0_loss_args(self) -> dict:
        """Bundle per-level loss config for the checkpointed loss wrapper."""
        return {
            "weights": self._surv_l0,
            "ice_cfg": self._ice_l0,
            "ref_moments": self._ref_moments_l0,
            "ice_teacher": self._ice_teacher,
            "global_step": self.global_step,
            "target_ratio": self._current_target_ratio(),
        }

    @staticmethod
    def _compress_files_batched_with_loss(
        compressor: torch.nn.Module,
        all_f_emb: torch.Tensor,
        all_f_mask: torch.Tensor,
        prompt_emb: torch.Tensor,
        prompt_mask: torch.Tensor,
        target_ratio: float,
        sub_file_ids: torch.Tensor,
        loss_args: dict,
        grad_capture: dict | None = None,
        utility_grad_active: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Checkpointable: L0 compress + L0 survivorship loss in one region.

        Returns ``(survivors, l0_loss, base_raw_for_util, post_head_content_values)``.
        CompressorOutput stays inside the checkpoint region so its
        activations can be recomputed on backward.
        ``base_raw_for_util`` is ``head(content_hidden.detach())`` —
        backward through it only touches head weights (subgraph fully
        disjoint from the main backward). ``post_head_content_values``
        is the detached forward-time stash used to form the
        ``-(grad · value)`` utility teacher after the main backward.

        When ``utility_grad_active=True``, ``grad_capture`` must be a
        mutable dict held by the trainer; the compressor's backward hook
        writes ``post_head_content_grad`` into it during the main
        backward pass so the trainer can build the utility-grad teacher
        after ``total_loss.backward()`` completes.
        """
        from bgkit.training.survivorship_helpers import compute_survivorship_losses

        comp_out = compressor(
            all_f_emb,
            attention_mask=all_f_mask,
            prompt_embeddings=prompt_emb,
            prompt_attention_mask=prompt_mask,
            target_ratio=target_ratio,
            level="l0",
            utility_grad_active=utility_grad_active,
            utility_grad_capture=grad_capture,
        )
        content_normed = comp_out.normed_embeddings[:, comp_out.content_slice, :]
        survivor_mask = comp_out.survivor_mask
        all_survivors = []
        if survivor_mask is not None:
            for f in range(all_f_emb.shape[0]):
                surv = content_normed[f][survivor_mask[f]]
                all_survivors.append(compressor.auto_reproduce(surv.unsqueeze(0))[0])
        if all_survivors:
            survivors = torch.cat(all_survivors, dim=0)
        else:
            survivors = torch.zeros(
                1, compressor.hidden_dim,
                device=all_f_emb.device, dtype=all_f_emb.dtype,
            )

        l0_loss, _ = compute_survivorship_losses(
            enc_out=comp_out,
            level="l0",
            weights=loss_args["weights"],
            ice_cfg=loss_args["ice_cfg"],
            ref_moments=loss_args["ref_moments"],
            ice_teacher=loss_args["ice_teacher"],
            global_step=loss_args["global_step"],
            content_token_ids=sub_file_ids,
            content_attn_mask=all_f_mask,
            target_ratio=loss_args["target_ratio"],
        )
        return (
            survivors,
            l0_loss,
            comp_out.base_raw_for_util,
            comp_out.post_head_content_values,
        )

    @staticmethod
    def _compress_files_batched_full(
        compressor: torch.nn.Module,
        all_f_emb: torch.Tensor,
        all_f_mask: torch.Tensor,
        prompt_emb: torch.Tensor,
        prompt_mask: torch.Tensor,
        target_ratio: float,
        utility_grad_active: bool = False,
    ) -> tuple[torch.Tensor, object]:
        """Same as _compress_files_batched_pure but also returns CompressorOutput."""
        comp_out = compressor(
            all_f_emb,
            attention_mask=all_f_mask,
            prompt_embeddings=prompt_emb,
            prompt_attention_mask=prompt_mask,
            target_ratio=target_ratio,
            level="l0",
            utility_grad_active=utility_grad_active,
        )
        content_normed = comp_out.normed_embeddings[:, comp_out.content_slice, :]
        survivor_mask = comp_out.survivor_mask
        all_survivors = []
        if survivor_mask is not None:
            for f in range(all_f_emb.shape[0]):
                surv = content_normed[f][survivor_mask[f]]
                all_survivors.append(compressor.auto_reproduce(surv.unsqueeze(0))[0])
        if all_survivors:
            return torch.cat(all_survivors, dim=0), comp_out
        return (
            torch.zeros(
                1, compressor.hidden_dim,
                device=all_f_emb.device, dtype=all_f_emb.dtype,
            ),
            comp_out,
        )

    def _compress_l0_batched(
        self,
        file_ids: torch.Tensor,
        file_masks: torch.Tensor,
        n_files: int,
        prompt_emb: torch.Tensor,
        prompt_mask: torch.Tensor,
        bgkit_embed: torch.nn.Module,
        surv_loss_accum: list | None = None,
        file_token_ids_l0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Batched L0 per-file compression for one sample -> (total_surv, D).

        Embeds all files at once, then runs a single batched compressor
        forward with the survivorship head producing masks internally.
        Uses selective activation checkpointing based on total token count.
        """
        from torch.utils.checkpoint import checkpoint as torch_checkpoint

        from bgkit.training.survivorship_helpers import LevelLossCfg
        _default_surv_level = LevelLossCfg()

        compressor = self.encoder.compressor
        target_ratio = self._current_target_ratio()

        # 1. Embed all files at once
        all_f_emb = bgkit_embed(file_ids[:n_files])          # (n_files, max_seq, D)
        all_f_mask = file_masks[:n_files]                     # (n_files, max_seq)

        # 2. Sort files by token count (descending) to reduce padding waste
        token_counts = all_f_mask.sum(dim=1)
        sorted_indices = token_counts.argsort(descending=True)
        all_f_emb = all_f_emb[sorted_indices]
        all_f_mask = all_f_mask[sorted_indices]

        # 3. Sub-batch files to bound peak memory
        max_files_per_sub = 8
        checkpoint_threshold = 4096
        all_survivors = []

        for start in range(0, n_files, max_files_per_sub):
            end = min(start + max_files_per_sub, n_files)
            sub_emb = all_f_emb[start:end]
            sub_f_mask = all_f_mask[start:end]
            sub_size = end - start
            sub_prompt_emb = prompt_emb.expand(sub_size, -1, -1)
            sub_prompt_mask = prompt_mask.expand(sub_size, -1)

            sub_tokens = sub_f_mask.sum().item()
            sub_file_ids = (
                file_token_ids_l0[start:end]
                if file_token_ids_l0 is not None else None
            )
            util_active_l0 = getattr(
                self, "_surv_l0", _default_surv_level,
            ).utility_grad_loss_weight > 0.0
            if sub_tokens > checkpoint_threshold:
                if surv_loss_accum is not None and sub_file_ids is not None:
                    # Checkpointed path with L0 loss: run a wrapper that
                    # computes the loss INSIDE the checkpoint region (so
                    # activations are recomputed on backward). The loss
                    # scalar crosses the checkpoint boundary; the
                    # CompressorOutput does NOT need to. For utility-grad
                    # we also pass a mutable ``grad_capture`` dict: the
                    # compressor's backward hook writes the captured
                    # content-position gradient into it during the single
                    # main backward (use_reentrant=False preserves the
                    # closure binding across recompute + backward).
                    grad_capture: dict = {}
                    (
                        sub_surv, sub_loss,
                        sub_base_raw_for_util, sub_content_values,
                    ) = torch_checkpoint(
                        self._compress_files_batched_with_loss,
                        compressor, sub_emb, sub_f_mask,
                        sub_prompt_emb, sub_prompt_mask, target_ratio,
                        sub_file_ids, self._l0_loss_args(),
                        grad_capture, util_active_l0,
                        use_reentrant=False,
                    )
                    # Also stash a metrics sentinel so the post-step
                    # aggregation sees this sub-batch (microbatch state
                    # accumulation happens in _apply_surv_losses). We
                    # rebuild state from the loss-computing call inside
                    # the checkpoint — so instead of going through
                    # _apply_surv_losses, append the (already-scalar) loss
                    # with a stage-B-style marker.
                    surv_loss_accum.append((
                        "ckpt_loss", sub_loss,
                        sub_base_raw_for_util, sub_content_values,
                        sub_f_mask, grad_capture,
                    ))
                else:
                    sub_surv = torch_checkpoint(
                        self._compress_files_batched_pure,
                        compressor, sub_emb, sub_f_mask,
                        sub_prompt_emb, sub_prompt_mask, target_ratio,
                        use_reentrant=False,
                    )
            else:
                if surv_loss_accum is not None and sub_file_ids is not None:
                    sub_surv, sub_out = self._compress_files_batched_full(
                        compressor, sub_emb, sub_f_mask,
                        sub_prompt_emb, sub_prompt_mask, target_ratio,
                        utility_grad_active=util_active_l0,
                    )
                    surv_loss_accum.append(
                        ("full", sub_out, sub_file_ids, sub_f_mask)
                    )
                else:
                    sub_surv = self._compress_files_batched_pure(
                        compressor, sub_emb, sub_f_mask,
                        sub_prompt_emb, sub_prompt_mask, target_ratio,
                    )
            all_survivors.append(sub_surv)

        if all_survivors:
            all_surv = torch.cat(all_survivors, dim=0)
        else:
            return torch.zeros(
                1, compressor.hidden_dim,
                device=self.device, dtype=torch.bfloat16,
            )

        if all_surv.shape[0] == 0:
            return torch.zeros(
                1, compressor.hidden_dim,
                device=self.device, dtype=torch.bfloat16,
            )
        return all_surv

    # ------------------------------------------------------------------
    # Forward/backward
    # ------------------------------------------------------------------

    def _forward_backward(self, batch: dict) -> dict[str, float]:
        """Grouped micro-batch forward + backward for efficient gradient accumulation.

        Two-phase pipeline:
        1. L0 per-file compression per sample (fast, ~3% of step time)
        2. Group samples by L0 survivor count, batch L1 encoder + decoder + backward

        Grouping amortizes backward through the 24-layer decoder + 14-layer encoder
        across multiple samples, replacing batch=1 matmuls with batch=group_size.
        """
        self.encoder.train()
        self.decoder.train()

        file_ids = batch["file_token_ids"].to(self.device)
        file_masks = batch["file_attention_masks"].to(self.device)
        file_count = batch["file_count"]
        prompt_ids = batch["compression_prompt_ids"].to(self.device)
        prompt_mask = batch["compression_prompt_mask"].to(self.device)
        target_ids = batch["target_token_ids"].to(self.device)
        target_mask = batch["target_attention_mask"].to(self.device)
        splice_start_batch = batch["bgkit_splice_start"].to(self.device)
        splice_len_batch = batch["bgkit_splice_len"].to(self.device)
        loss_mask_batch = batch.get("target_loss_mask")
        if loss_mask_batch is not None:
            loss_mask_batch = loss_mask_batch.to(self.device)

        batch_size = file_ids.size(0)
        bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()
        hidden_dim = self.encoder.compressor.hidden_dim
        group_size = 4  # micro-batch size for L1+decoder

        total_loss = 0.0
        total_survivors = 0
        total_valid = 0
        # Accumulators for per-level survivorship losses across all microbatches
        # in this optimizer step. L0 losses are deferred until after backward
        # through the decoder so the autograd graph is intact.
        l0_surv_accum: list = []
        aux_metrics: dict[str, float] = {}
        # Utility-gradient BCE distillation bundles. L0 bundles are
        # collected during Phase 1 (each sub-batch carries its own
        # grad_capture dict that the main backward in Phase 2 writes to);
        # L1 bundles are collected during Phase 2 for each group's L1
        # forward. Both are drained post-backward to build BCE teachers
        # from the captured ``-(grad · value)`` utility ranking.
        util_l0_bundles: list[dict] = []
        util_l1_bundles: list[dict] = []
        from bgkit.training.survivorship_helpers import LevelLossCfg
        _default_surv = LevelLossCfg()
        util_w_l0 = getattr(
            self, "_surv_l0", _default_surv,
        ).utility_grad_loss_weight
        util_w_l1 = getattr(
            self, "_surv_l1", _default_surv,
        ).utility_grad_loss_weight

        # Profiling accumulators
        if self._profile_enabled:
            prof_l0 = 0.0
            prof_l1 = 0.0
            prof_dec = 0.0  # decoder + backward combined

        # --- Phase 1: L0 compression per sample (fast, ~3% of step time) ---
        l0_data = []  # list of (l0_surv, prompt_emb, sample_idx)
        for b in range(batch_size):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                n_files = int(file_count[b].item())
                prompt_emb_b = bgkit_embed(prompt_ids[b:b + 1])

                if self._profile_enabled:
                    ev_l0_start = torch.cuda.Event(enable_timing=True)
                    ev_l0_end = torch.cuda.Event(enable_timing=True)
                    ev_l0_start.record()

                # L0 per-file (batched with selective checkpointing). Pass
                # per-sample accumulator so non-checkpointed sub-batches
                # stash their CompressorOutput; L0 survivorship loss is
                # backwarded immediately per sub-batch (no retain_graph)
                # to free the encoder graph — each sub-batch has its own
                # independent autograd graph since nothing downstream
                # (decoder, L1) consumes the L0 sub-batch's compressor
                # activations (only the auto-reproduce output flows
                # forward, and its .grad path is already captured above).
                per_sample_l0: list = []
                l0_surv = self._compress_l0_batched(
                    file_ids[b], file_masks[b], n_files,
                    prompt_emb_b, prompt_mask[b:b + 1],
                    bgkit_embed,
                    surv_loss_accum=per_sample_l0,
                    file_token_ids_l0=file_ids[b][:n_files],
                )
                scale = 1.0 / (batch_size * self._accum_steps)
                for entry in per_sample_l0:
                    if entry[0] == "ckpt_loss":
                        # Checkpointed path: loss already computed inside
                        # the checkpoint region. Just scale + backward.
                        (
                            _, sub_loss,
                            sub_base_raw_for_util, sub_content_values,
                            sub_mask, grad_capture,
                        ) = entry
                        if isinstance(sub_loss, torch.Tensor) and sub_loss.requires_grad:
                            (sub_loss * scale).backward()
                        if util_w_l0 > 0.0 and sub_content_values is not None:
                            util_l0_bundles.append({
                                "base_raw_for_util": sub_base_raw_for_util,
                                "content_values": sub_content_values,
                                "content_mask": sub_mask,
                                "grad_capture": grad_capture,
                                "enc_out": None,
                            })
                    else:
                        # Non-checkpointed path: compute loss + accumulate
                        # state from the live CompressorOutput.
                        _, sub_out, sub_ids, sub_mask = entry
                        l0_loss, l0_metrics = self._apply_surv_losses(
                            sub_out, level="l0",
                            content_token_ids=sub_ids,
                            content_attn_mask=sub_mask,
                        )
                        if l0_metrics:
                            aux_metrics.update(l0_metrics)
                        if isinstance(l0_loss, torch.Tensor) and l0_loss.requires_grad:
                            (l0_loss * scale).backward()
                        if (
                            util_w_l0 > 0.0
                            and sub_out.post_head_content_values is not None
                        ):
                            util_l0_bundles.append({
                                "base_raw_for_util": sub_out.base_raw_for_util,
                                "content_values": sub_out.post_head_content_values,
                                "content_mask": sub_mask,
                                "grad_capture": None,
                                "enc_out": sub_out,
                            })
                per_sample_l0.clear()

                if self._profile_enabled:
                    ev_l0_end.record()
                    torch.cuda.synchronize()
                    prof_l0 += ev_l0_start.elapsed_time(ev_l0_end)

            total_valid += int(file_masks[b].sum().item())
            l0_data.append((l0_surv, prompt_emb_b, b))

        # --- Phase 2: Group L1 + decoder + backward ---
        # Sort by survivor count for efficient padding within groups
        l0_data.sort(key=lambda x: x[0].size(0))

        for g_start in range(0, batch_size, group_size):
            g_end = min(g_start + group_size, batch_size)
            group = l0_data[g_start:g_end]
            g_size = len(group)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                if self._profile_enabled:
                    ev_l1_start = torch.cuda.Event(enable_timing=True)
                    ev_l1_end = torch.cuda.Event(enable_timing=True)
                    ev_l1_start.record()

                # Check padding ratio to decide batched vs per-sample
                surv_counts = [s.size(0) for s, _, _ in group]
                min_sc = min(surv_counts)
                max_sc = max(surv_counts)
                batch_group = g_size > 1 and (min_sc == 0 or max_sc <= 2 * min_sc)

                target_ratio = self._current_target_ratio()

                if batch_group:
                    # Pad L0 survivors to max in group
                    l1_input = torch.zeros(
                        g_size, max_sc, hidden_dim,
                        device=self.device, dtype=torch.bfloat16,
                    )
                    l1_mask = torch.zeros(
                        g_size, max_sc, dtype=torch.bool, device=self.device,
                    )
                    for j, (surv, _, _) in enumerate(group):
                        sc = surv.size(0)
                        l1_input[j, :sc] = surv
                        l1_mask[j, :sc] = True

                    # Batched L1 encoder + decoder forward
                    prompt_embs = torch.cat([p for _, p, _ in group], dim=0)
                    prompt_masks = torch.cat(
                        [prompt_mask[b:b + 1] for _, _, b in group], dim=0,
                    )
                    l1_out = self.encoder(
                        input_embeddings=l1_input,
                        attention_mask=l1_mask,
                        prompt_embeddings=prompt_embs,
                        prompt_attention_mask=prompt_masks,
                        target_ratio=target_ratio,
                        level="l1",
                        utility_grad_active=util_w_l1 > 0.0,
                    )

                    # L1 survivorship loss + state accumulation. Content
                    # token_ids aren't meaningful at L1 (inputs are
                    # compressed embeddings, not tokens), so BCE is
                    # effectively disabled — plan sets L1 BCE off.
                    l1_surv_loss, l1_metrics = self._apply_surv_losses(
                        l1_out, level="l1",
                        content_token_ids=None,
                        content_attn_mask=l1_mask,
                    )
                    if l1_metrics:
                        aux_metrics.update(l1_metrics)
                    if (
                        util_w_l1 > 0.0
                        and l1_out.post_head_content_values is not None
                    ):
                        util_l1_bundles.append({
                            "base_raw_for_util": l1_out.base_raw_for_util,
                            "content_values": l1_out.post_head_content_values,
                            "content_mask": l1_mask,
                            "grad_capture": None,
                            "enc_out": l1_out,
                        })

                    if self._profile_enabled:
                        ev_l1_end.record()
                        ev_dec_start = torch.cuda.Event(enable_timing=True)
                        ev_dec_end = torch.cuda.Event(enable_timing=True)
                        ev_dec_start.record()

                    group_indices = [b for _, _, b in group]
                    group_target_ids = target_ids[group_indices]
                    group_target_mask = target_mask[group_indices]
                    group_loss_mask = None
                    if loss_mask_batch is not None:
                        group_loss_mask = loss_mask_batch[group_indices]

                    loss = self.decoder.forward_with_single_splice(
                        survivor_embeddings=l1_out.survivor_embeddings,
                        survivor_attention_mask=l1_out.survivor_attention_mask,
                        token_ids=group_target_ids,
                        token_attention_mask=group_target_mask,
                        splice_starts=splice_start_batch[group_indices],
                        splice_lengths=splice_len_batch[group_indices],
                        loss_mask=group_loss_mask,
                    )
                    group_scale = g_size / (batch_size * self._accum_steps)
                    total = loss * group_scale
                    if isinstance(l1_surv_loss, torch.Tensor) and l1_surv_loss.requires_grad:
                        total = total + l1_surv_loss * group_scale

                    total.backward()

                    total_loss += loss.item() * g_size
                    total_survivors += int(
                        l1_out.survivor_attention_mask.sum().item()
                    )
                else:
                    # Per-sample fallback: high padding ratio
                    if self._profile_enabled:
                        ev_l1_end.record()
                        ev_dec_start = torch.cuda.Event(enable_timing=True)
                        ev_dec_end = torch.cuda.Event(enable_timing=True)
                        ev_dec_start.record()

                    scale = 1.0 / (batch_size * self._accum_steps)
                    for surv, p_emb, b_idx in group:
                        l1_in = surv.unsqueeze(0)
                        l1_m = torch.ones(
                            1, surv.size(0), dtype=torch.bool,
                            device=self.device,
                        )
                        l1_o = self.encoder(
                            input_embeddings=l1_in,
                            attention_mask=l1_m,
                            prompt_embeddings=p_emb,
                            prompt_attention_mask=prompt_mask[b_idx:b_idx + 1],
                            target_ratio=target_ratio,
                            level="l1",
                            utility_grad_active=util_w_l1 > 0.0,
                        )
                        l1_surv_loss, l1_metrics = self._apply_surv_losses(
                            l1_o, level="l1",
                            content_token_ids=None,
                            content_attn_mask=l1_m,
                        )
                        if l1_metrics:
                            aux_metrics.update(l1_metrics)
                        if (
                            util_w_l1 > 0.0
                            and l1_o.post_head_content_values is not None
                        ):
                            util_l1_bundles.append({
                                "base_raw_for_util": l1_o.base_raw_for_util,
                                "content_values": l1_o.post_head_content_values,
                                "content_mask": l1_m,
                                "grad_capture": None,
                                "enc_out": l1_o,
                            })
                        s_lm = (
                            loss_mask_batch[b_idx:b_idx + 1]
                            if loss_mask_batch is not None else None
                        )
                        loss = self.decoder.forward_with_single_splice(
                            survivor_embeddings=l1_o.survivor_embeddings,
                            survivor_attention_mask=l1_o.survivor_attention_mask,
                            token_ids=target_ids[b_idx:b_idx + 1],
                            token_attention_mask=target_mask[b_idx:b_idx + 1],
                            splice_starts=splice_start_batch[b_idx:b_idx + 1],
                            splice_lengths=splice_len_batch[b_idx:b_idx + 1],
                            loss_mask=s_lm,
                        )
                        total = loss * scale
                        if isinstance(l1_surv_loss, torch.Tensor) and l1_surv_loss.requires_grad:
                            total = total + l1_surv_loss * scale
                        total.backward()
                        total_loss += loss.item()
                        total_survivors += int(
                            l1_o.survivor_attention_mask.sum().item()
                        )

                if self._profile_enabled:
                    ev_dec_end.record()
                    torch.cuda.synchronize()
                    prof_l1 += ev_l1_start.elapsed_time(ev_l1_end)
                    prof_dec += ev_dec_start.elapsed_time(ev_dec_end)

        # --- Utility-gradient BCE distillation (post-backward) ---
        # All Phase 1 + Phase 2 backwards have completed. For each level,
        # rebuild the BCE teacher from the captured ``-(grad · value)``
        # utility and run a small head-local backward.
        self._apply_utility_grad_bce(
            bundles=util_l0_bundles, level="l0",
            weight=util_w_l0, batch_size=batch_size,
            metrics=aux_metrics,
        )
        self._apply_utility_grad_bce(
            bundles=util_l1_bundles, level="l1",
            weight=util_w_l1, batch_size=batch_size,
            metrics=aux_metrics,
        )

        if self._profile_enabled:
            prof_total = prof_l0 + prof_l1 + prof_dec
            logger.info(
                "[PROFILE]",
                step=self.global_step,
                l0=f"{prof_l0:.0f}ms",
                l1=f"{prof_l1:.0f}ms",
                dec_bwd=f"{prof_dec:.0f}ms",
                total=f"{prof_total:.0f}ms",
                batch_size=batch_size,
            )

        target_ratio = self._current_target_ratio()
        actual_ratio = total_survivors / max(total_valid, 1)

        result = {
            "loss": total_loss / batch_size,
            "target_ratio": target_ratio,
            "actual_ratio": actual_ratio,
        }
        result.update(aux_metrics)

        # Release every CompressorOutput referenced by the bundles.
        # Utility-grad BCE has already consumed everything it needs
        # from them at this point. See ``CompressorOutput.release()``
        # for the leak this mitigates.
        for bundle in (*util_l0_bundles, *util_l1_bundles):
            enc_out = bundle.get("enc_out")
            if enc_out is not None and hasattr(enc_out, "release"):
                enc_out.release()

        return result

    def _apply_utility_grad_bce(
        self,
        bundles: list[dict],
        level: str,
        weight: float,
        batch_size: int,
        metrics: dict[str, float],
    ) -> None:
        """Run utility-gradient BCE distillation for one level.

        Runs after ALL main backwards (both the per-sub-batch L0
        ``sub_loss.backward()`` and the per-group Phase 2
        ``total.backward()``) have completed. The backward hook on
        ``content_hidden`` fires on each backward that traverses it;
        for a given sub-batch, the Phase 2 decoder backward is the
        later one, so the final captured grad reflects the full
        ``decoder_CE + aux`` composition flowing back through L0
        (this is the signal we want for the utility-teacher ranking).

        Each ``bundle`` carries either a ``grad_capture`` dict populated
        by the compressor's backward hook (checkpointed L0 path) or a
        live ``enc_out`` whose ``_utility_grad_state`` is populated.
        ``base_raw_for_util`` is the detached-input head fork stashed at
        compressor forward time — the BCE backward traverses only
        ``util_loss → base_raw_for_util → head.weights``.
        """
        if weight <= 0.0 or not bundles:
            return
        from bgkit.training.survivorship_helpers import utility_grad_bce_loss

        target_ratio = self._current_target_ratio()
        scale = 1.0 / (batch_size * self._accum_steps)
        total_grad_norm = 0.0
        n_bundles_with_grad = 0
        emitted_metric = False
        for bundle in bundles:
            grad_capture = bundle.get("grad_capture")
            enc_out = bundle.get("enc_out")
            if grad_capture is not None:
                content_grad = grad_capture.get("post_head_content_grad")
            elif enc_out is not None:
                content_grad = enc_out.get_content_grad()
            else:
                content_grad = None
            if content_grad is None:
                continue
            util_loss, util_metrics = utility_grad_bce_loss(
                base_raw_for_util=bundle.get("base_raw_for_util"),
                content_grad=content_grad,
                content_values=bundle["content_values"],
                valid_mask=bundle["content_mask"],
                pinned_mask=None,
                target_ratio=target_ratio,
            )
            if util_loss.requires_grad:
                (util_loss * weight * scale).backward()
            if not emitted_metric and util_metrics:
                for k, v in util_metrics.items():
                    metrics[f"{level}_{k}"] = v
                emitted_metric = True
            total_grad_norm += float(content_grad.norm().item())
            n_bundles_with_grad += 1
        if n_bundles_with_grad > 0:
            metrics[f"{level}_content_grad_norm"] = (
                total_grad_norm / n_bundles_with_grad
            )

    def _apply_surv_losses(
        self,
        enc_out,
        level: str,
        content_token_ids: torch.Tensor | None = None,
        content_attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute per-level survivorship loss + accumulate microbatch state."""
        from bgkit.training.survivorship_helpers import (
            LevelICECfg,
            LevelLossCfg,
            accumulate,
            compute_survivorship_losses,
            init_state,
            survivorship_diagnostics,
        )

        if level == "l0":
            weights = getattr(self, "_surv_l0", LevelLossCfg())
            ice_cfg = getattr(self, "_ice_l0", LevelICECfg())
            ref_moments = getattr(self, "_ref_moments_l0", None)
            state = getattr(self, "_surv_state_l0", None)
            if state is None:
                state = init_state()
                self._surv_state_l0 = state
        else:
            weights = getattr(self, "_surv_l1", LevelLossCfg())
            ice_cfg = getattr(self, "_ice_l1", LevelICECfg())
            ref_moments = getattr(self, "_ref_moments_l1", None)
            state = getattr(self, "_surv_state_l1", None)
            if state is None:
                state = init_state()
                self._surv_state_l1 = state

        target_ratio = self._current_target_ratio()
        loss, metrics = compute_survivorship_losses(
            enc_out=enc_out,
            level=level,
            weights=weights,
            ice_cfg=ice_cfg,
            ref_moments=ref_moments,
            ice_teacher=getattr(self, "_ice_teacher", None),
            global_step=self.global_step,
            content_token_ids=content_token_ids,
            content_attn_mask=content_attn_mask,
            target_ratio=target_ratio,
        )
        accumulate(state, enc_out)
        out_metrics = {f"{level}_{k}": v for k, v in metrics.items()}
        # Head-health diagnostics (organic-rate std, undecided fraction,
        # floor/pinned/θ). Each read is a GPU→CPU sync; gate via the
        # configured diagnostic cadence.
        diag_every_n = int(getattr(self, "_diagnostic_metrics_every_n_steps", 1) or 1)
        out_metrics.update(
            survivorship_diagnostics(
                enc_out, level=level, global_step=self.global_step,
                every_n_steps=diag_every_n,
            )
        )
        return loss, out_metrics

    def _post_step(self, step: int) -> None:
        """Advance bidirectional warmup and run dual-ascent θ + EMA μ updates."""
        self.encoder.step_bidi_warmup()

        from bgkit.training.survivorship_helpers import (
            apply_post_step_updates,
            init_state,
            maybe_unload_ice,
        )

        target_ratio = self._current_target_ratio()
        merged: dict[str, float] = {}
        for level, state_attr in (("l0", "_surv_state_l0"), ("l1", "_surv_state_l1")):
            state = getattr(self, state_attr, None)
            if state is None:
                continue
            update_metrics = apply_post_step_updates(
                self.encoder.compressor,
                state,
                target_ratio=target_ratio,
                level=level,
            )
            merged.update(update_metrics)
            setattr(self, state_attr, init_state())

        self._last_post_step_metrics = merged
        unloaded = maybe_unload_ice(
            getattr(self, "_ice_teacher", None),
            step,
            getattr(self, "_max_warmup_step", 0),
        )
        if unloaded:
            logger.info("ice_teacher_unloaded", step=step)

    def _get_bidi_alpha(self) -> float:
        from bgkit.models.bidirectional_qwen35 import BidirectionalQwen35
        from bgkit.models.pruned_qwen35 import PrunedBidirectionalQwen35

        for module in self.encoder.modules():
            if isinstance(module, (BidirectionalQwen35, PrunedBidirectionalQwen35)):
                return module.bidi_alpha
        return 1.0

    def _add_step_metrics(self, metrics: dict[str, float]) -> None:
        """Attach post-step θ/μ updates (without clobbering base metrics)."""
        post = getattr(self, "_last_post_step_metrics", None)
        if post:
            for k, v in post.items():
                metrics.setdefault(k, v)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.encoder.eval()
        self.decoder.eval()

        try:
            total_loss = 0.0
            total_tokens = 0.0
            target_ratio = self._current_target_ratio()

            num_batches = len(self.eval_dataloader)
            for batch_idx, batch in enumerate(self.eval_dataloader):
                if batch_idx % 100 == 0:
                    logger.info("eval_progress", batch=batch_idx, total=num_batches)

                file_ids = batch["file_token_ids"].to(self.device)
                file_masks = batch["file_attention_masks"].to(self.device)
                file_count = batch["file_count"]
                prompt_ids = batch["compression_prompt_ids"].to(self.device)
                prompt_mask = batch["compression_prompt_mask"].to(self.device)
                target_ids = batch["target_token_ids"].to(self.device)
                target_mask = batch["target_attention_mask"].to(self.device)
                splice_start_batch = batch["bgkit_splice_start"].to(self.device)
                splice_len_batch = batch["bgkit_splice_len"].to(self.device)
                loss_mask_batch = batch.get("target_loss_mask")
                if loss_mask_batch is not None:
                    loss_mask_batch = loss_mask_batch.to(self.device)

                batch_size = file_ids.size(0)
                bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()

                # Per-sample eval, no grouping, minimal peak memory.
                for b in range(batch_size):
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16,
                        enabled=self.device.type == "cuda",
                    ):
                        n_files = int(file_count[b].item())
                        prompt_emb_b = bgkit_embed(prompt_ids[b:b + 1])

                        l0_surv = self._compress_l0_batched(
                            file_ids[b], file_masks[b], n_files,
                            prompt_emb_b, prompt_mask[b:b + 1],
                            bgkit_embed,
                        )

                        l1_input = l0_surv.unsqueeze(0)
                        l1_mask = torch.ones(
                            1, l0_surv.size(0), dtype=torch.bool, device=self.device,
                        )

                        l1_out = self.encoder(
                            input_embeddings=l1_input,
                            attention_mask=l1_mask,
                            prompt_embeddings=prompt_emb_b,
                            prompt_attention_mask=prompt_mask[b:b + 1],
                            target_ratio=target_ratio,
                            level="l1",
                        )

                        sample_loss_mask = (
                            loss_mask_batch[b:b + 1]
                            if loss_mask_batch is not None else None
                        )

                        loss = self.decoder.forward_with_single_splice(
                            survivor_embeddings=l1_out.survivor_embeddings,
                            survivor_attention_mask=l1_out.survivor_attention_mask,
                            token_ids=target_ids[b:b + 1],
                            token_attention_mask=target_mask[b:b + 1],
                            splice_starts=splice_start_batch[b:b + 1],
                            splice_lengths=splice_len_batch[b:b + 1],
                            loss_mask=sample_loss_mask,
                        )

                        eval_loss_mask = batch.get("target_loss_mask")
                        if eval_loss_mask is not None:
                            sample_tokens = eval_loss_mask[b].sum().item()
                        else:
                            sample_tokens = batch["target_attention_mask"][b].sum().item()
                        total_loss += loss.item() * sample_tokens
                        total_tokens += sample_tokens

            avg_loss = total_loss / max(total_tokens, 1)
            perplexity = torch.exp(torch.tensor(avg_loss)).item()

            return {
                "loss": avg_loss,
                "perplexity": perplexity,
            }
        finally:
            self.encoder.train()
            self.decoder.train()

    # ------------------------------------------------------------------
    # Checkpointing — multi-artifact format (matches CompressionTrainer)
    # ------------------------------------------------------------------

    def save_checkpoint(
        self, checkpoint_dir: Path, metrics: dict[str, float] | None = None,
    ) -> Path:
        """Save encoder, decoder, optimizer, and calibrator state."""
        if self._training_state is None:
            self._training_state = {}

        self._training_state.update({
            "target_ratio_override": self._target_ratio_override,
        })

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
            decoder=self.decoder.state_dict(),
            optimizer_state_by_name=self._build_optimizer_state_by_name(),
        )
        if getattr(self, "_decoder_lora", False):
            save_kwargs["decoder_merged"] = self.decoder.merge_lora()

        ckpt_path = save_checkpoint(checkpoint_dir, metadata, **save_kwargs)
        self._last_checkpoint_path = str(ckpt_path)
        return ckpt_path

    def _named_parameters_for_optimizer(self):
        """Yield (name, param) pairs across encoder + decoder."""
        for name, param in self.encoder.named_parameters():
            yield f"encoder.{name}", param
        for name, param in self.decoder.named_parameters():
            yield f"decoder.{name}", param

    def _restore_model_state(self, state_dicts: dict) -> None:
        if "encoder" in state_dicts:
            self.encoder.load_state_dict(state_dicts["encoder"])
        if "decoder" in state_dicts:
            self.decoder.load_state_dict(state_dicts["decoder"])

    def _restore_training_state(self, training_state: dict) -> None:
        self._target_ratio_override = training_state.get("target_ratio_override")

    # ------------------------------------------------------------------
    # Live config
    # ------------------------------------------------------------------

    def apply_live_config(self, changes: dict) -> None:
        if "target_ratio" in changes:
            val = changes["target_ratio"]
            if val is None:
                self._target_ratio_override = None
            elif isinstance(val, (int, float)) and 0 < val < 1:
                self._target_ratio_override = float(val)
        super().apply_live_config(changes)
