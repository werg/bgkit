"""Phase 1, Step 1: Decoder initialization with encoder unfreeze + compression curriculum.

Trains the reconstruction decoder to generate text from BgKIT's output
representations, using Qwen3's native chat template with tool-call format.

Training progresses through four curriculum phases:
1. Projection-only warmup (decoder frozen)
2. Decoder + projection training (encoder frozen, no compression)
3. Encoder unfreeze with bidirectional warmup (no compression yet)
4. Compression introduction with survivorship head + ratio targeting

Loss is computed only on the file content tokens inside the chat template's
code fence (via loss_mask), so the decoder learns pure reconstruction.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import structlog
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer

from bgkit.data.collators import collate_chat_repro
from bgkit.data.datasets.chat_repro_dataset import ChatReproDataset
from bgkit.data.datasets.mmap_token_dataset import MmapTokenDataset
from bgkit.data.samplers import PackedTokenBudgetSampler
from bgkit.eval.metrics.embedding_health import embedding_drift_metrics
from bgkit.eval.metrics.reconstruction import parse_success_rate
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpoint_registry import resolve_checkpoint
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.training.gradient_utils import maybe_enable_gradient_checkpointing
from bgkit.training.ratio_sampling import (
    build_ratio_sampler_config,
    resolve_anchor_grid,
    sample_ratio,
)
from bgkit.training.scheduling import cosine_with_warmup
from bgkit.utils.attention_backend import resolve_attention_implementation
from bgkit.utils.memory_budget import memory_budget_scope

logger = structlog.get_logger()


class _InterleavingDataLoader:
    """Wraps two dataloaders, alternating batches to maintain a target ratio.

    The ratio is tracked by sample count (batch size), not batch count,
    since TokenBudgetBatchSampler produces variable-size batches.

    When the primary loader exhausts, raises StopIteration (signals epoch end
    to the base trainer). The secondary loader cycles independently.
    """

    def __init__(self, primary, secondary, secondary_ratio: float = 0.3):
        self._primary = primary
        self._secondary = secondary
        self._ratio = secondary_ratio

    @property
    def batch_sampler(self):
        return self._primary.batch_sampler

    @property
    def dataset(self):
        return self._primary.dataset

    def __iter__(self):
        return _InterleavingIterator(
            self._primary, self._secondary, self._ratio,
        )

    def __len__(self):
        return len(self._primary)


class _InterleavingIterator:
    def __init__(self, primary, secondary, ratio):
        self._primary_loader = primary
        self._secondary_loader = secondary
        self._primary_iter = iter(primary)
        self._secondary_iter = iter(secondary)
        self._ratio = ratio
        self._primary_samples = 0
        self._secondary_samples = 0

    @staticmethod
    def _batch_sample_count(batch: dict) -> int:
        """Per-sample count from the packed cu_seqlens (shape B+1)."""
        cu = batch.get("cu_seqlens")
        if cu is not None:
            return int(cu.shape[0]) - 1
        # Fallback for legacy batches — shouldn't hit on the packed path.
        return int(batch["token_ids"].size(0))

    def __next__(self):
        total = self._primary_samples + self._secondary_samples + 1
        current_ratio = self._secondary_samples / total

        if current_ratio < self._ratio:
            try:
                batch = next(self._secondary_iter)
                self._secondary_samples += self._batch_sample_count(batch)
                return batch
            except StopIteration:
                # Secondary exhausted — restart it
                self._secondary_iter = iter(self._secondary_loader)
                # Fall through to primary

        # Primary batch (or secondary was exhausted)
        batch = next(self._primary_iter)  # StopIteration propagates = epoch end
        self._primary_samples += self._batch_sample_count(batch)
        return batch


class DecoderInitTrainer(BaseTrainer):
    """Step 1: Decoder init with encoder unfreeze + compression curriculum."""

    LIVE_CONFIG_FIELDS: ClassVar[dict[str, str]] = {
        "target_ratio_ramp_steps": "_target_ratio_ramp_steps",
        "target_ratio_start": "_target_ratio_start",
        "target_ratio_end": "_target_ratio_end",
        "compression_introduction_step": "_compression_introduction_step",
        "encoder_unfreeze_step": "_encoder_unfreeze_step",
        # Flow-control scalars (all cheap to change mid-run):
        "diagnostic_metrics_every_n_steps": "_diagnostic_metrics_every_n_steps",
        # Generation-eval frequency (eval_every / save_every are live-tunable
        # in BaseTrainer directly). gen_eval has a separate cadence since it
        # runs only every Nth eval to amortize the KV-cache cost.
        "generation_eval_every": "_generation_eval_every",
    }

    LIVE_CONFIG_HANDLERS: ClassVar[dict[str, str]] = {
        "target_ratio": "_handle_target_ratio",
        "target_ratio_sampling_window_above": "_handle_ratio_sampling_window_above",
        "sample_target_ratio_during_training": "_handle_ratio_sampling_enabled",
        "target_ratio_anchor_sampling_prob": "_handle_ratio_sampling_anchor_prob",
    }

    def setup(self) -> None:
        """Load BgKIT encoder, decoder, and configure compression curriculum."""
        tcfg = self.cfg.training

        # Config validation
        if tcfg.get("projection_only_steps", 0) > 0 and not tcfg.get(
            "train_projection_block", True
        ):
            raise ValueError("projection_only_steps > 0 requires train_projection_block: true")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        attention_impl = resolve_attention_implementation(
            self.cfg.compute.get("attention_implementation", "auto")
        )

        # --- Curriculum config ---
        self._encoder_unfreeze_step = tcfg.get("encoder_unfreeze_step", None)
        self._compression_introduction_step = tcfg.get("compression_introduction_step", None)
        bidi_warmup = tcfg.get("bidi_warmup_steps", 0)

        # --- BgKIT encoder (starts frozen, optionally unfreezes later) ---
        bgkit_cfg = self.cfg.model.bgkit
        backbone_name = bgkit_cfg.backbone_name
        backbone_revision = bgkit_cfg.get("backbone_revision", None)
        hidden_dim = bgkit_cfg.get("hidden_dim", 1024)

        # Projection block training flags
        self._train_projection = tcfg.get("train_projection_block", True)
        self._projection_only_steps = tcfg.get("projection_only_steps", 0)

        # Threshold-controller config. Derive init_theta from target_ratio_start
        # under the tanh-bounded-logit convention: θ = 1 − 2·target_ratio so a
        # symmetric base distribution produces ~target_ratio initial keep-rate.
        # Config init_theta wins if set explicitly (e.g., to tune the warmup
        # bias); otherwise it's computed.
        model_cfg = self.cfg.model
        ctrl_src = model_cfg.get("threshold_controller", {})
        target_ratio_start = float(tcfg.get("target_ratio_start", 0.10))
        threshold_controller_cfg = {
            "init_theta": float(
                ctrl_src.get("init_theta", 1.0 - 2.0 * target_ratio_start),
            ),
            "lr": float(ctrl_src.get("lr", 0.02)),
            "momentum": float(ctrl_src.get("momentum", 0.0)),
            "clamp": float(ctrl_src.get("clamp", 0.99)),
            "anchor_ratios": list(ctrl_src.get("anchor_ratios", [])) or None,
            "ratio_space": str(ctrl_src.get("ratio_space", "log")),
            "init_target_ratio": target_ratio_start,
            "default_query_ratio": target_ratio_start,
        }
        # Load BgKIT from checkpoint if available (auto-detects pruned architecture)
        bgkit_checkpoint = self._resolve_bgkit_checkpoint()
        if bgkit_checkpoint is not None:
            logger.info("loading_bgkit_checkpoint", path=bgkit_checkpoint)
            _, bgkit_state_dicts = load_checkpoint(Path(bgkit_checkpoint))
            if "encoder" not in bgkit_state_dicts:
                raise ValueError(
                    f"bgkit_checkpoint {bgkit_checkpoint} missing 'encoder' key. "
                    f"Found keys: {list(bgkit_state_dicts.keys())}"
                )
            self.encoder = BgKITEncoder.from_pretrained_with_state_dict(
                backbone_name,
                bgkit_state_dicts["encoder"],
                hidden_dim=hidden_dim,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                revision=backbone_revision,
                attn_implementation=attention_impl,
                bidi_warmup_steps=bidi_warmup,
                threshold_controller_cfg=threshold_controller_cfg,
            )
            # Also load decoder weights if present (step 2+ checkpoints include them).
            # Prefer "decoder_merged" over "decoder" when both are present: the
            # former is the base-Qwen-shaped state_dict produced by peft's
            # merge_and_unload after LoRA training (Step 3 / Step 4), while
            # "decoder" carries the LoRA-wrapped key prefixes (``base_model.model.*``)
            # that don't match a fresh base-Qwen decoder's state_dict. The fresh
            # decoder applied LoRA LATER in setup (see line ~285), so the target
            # at load time is always the base shape. Step 2 checkpoints only
            # carry "decoder" (no LoRA trained yet) so the fallback is correct.
            self._bgkit_decoder_state = (
                bgkit_state_dicts.get("decoder_merged")
                or bgkit_state_dicts.get("decoder")
            )
            # Optional: load a pre-distilled head_base_l0 sidecar as a warm
            # start. Produced by scripts/pretrain_survivorship_head.py.
            sidecar_path = self.cfg.training.get(
                "survivorship_head_sidecar", None,
            )
            if sidecar_path:
                sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=True)
                res = self.encoder.load_state_dict(sidecar, strict=False)
                n_loaded = len(sidecar) - len(res.unexpected_keys)
                self._sidecar_loaded = True
                logger.info(
                    "loaded_survivorship_sidecar",
                    path=sidecar_path,
                    loaded_keys=n_loaded,
                    unexpected=len(res.unexpected_keys),
                )
            else:
                self._sidecar_loaded = False
        else:
            logger.info("loading_bgkit_encoder", model=backbone_name, revision=backbone_revision)
            self.encoder = BgKITEncoder.from_pretrained(
                backbone_name,
                hidden_dim=hidden_dim,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                revision=backbone_revision,
                attn_implementation=attention_impl,
                bidi_warmup_steps=bidi_warmup,
                threshold_controller_cfg=threshold_controller_cfg,
            )
            self._bgkit_decoder_state = None
            self._sidecar_loaded = False
        self.encoder.to(device)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

        # --- Decoder (trainable) ---
        decoder_cfg = self.cfg.model.decoder
        decoder_name = decoder_cfg.backbone_name
        decoder_revision = decoder_cfg.get("backbone_revision", None)
        logger.info("loading_decoder", model=decoder_name, revision=decoder_revision)
        decoder_backbone = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=decoder_revision,
            attn_implementation=attention_impl,
            device_map=device,  # load directly to CUDA, avoid CPU staging copy
        )
        self.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=hidden_dim)

        # Load decoder weights from bgkit checkpoint if available (step 2 includes decoder).
        # ``load_decoder_from_bgkit_checkpoint: false`` skips this and keeps the
        # fresh HF base decoder — used by Step 3 after the Step 2.5 projection
        # repair so the decoder re-adapts against the repaired projection under
        # the Step 3 compression curriculum rather than carrying forward the
        # Step 1 decoder that was trained against the pre-repair (off-manifold)
        # projection.
        load_decoder = bool(tcfg.get("load_decoder_from_bgkit_checkpoint", True))
        if self._bgkit_decoder_state is not None:
            if load_decoder:
                self.decoder.load_state_dict(self._bgkit_decoder_state)
                logger.info("loaded_decoder_from_bgkit_checkpoint")
            else:
                logger.info(
                    "skipped_decoder_load_from_bgkit_checkpoint_per_config",
                    reason="training.load_decoder_from_bgkit_checkpoint=false",
                )
            del self._bgkit_decoder_state

        # Optional LoRA wrapping for decoder (step 3 uses this for parameter-efficient adaptation)
        self._decoder_lora = False
        lora_cfg = tcfg.get("decoder_lora", {})
        if lora_cfg.get("enabled", False):
            self.decoder.apply_lora(lora_cfg)
            self._decoder_lora = True

        maybe_enable_gradient_checkpointing(self.decoder.backbone, self.cfg)

        # Optional Liger Kernel fused kernels (RMSNorm / SwiGLU / RoPE +
        # fused linear+CE). Gated on ``training.use_liger`` (default True);
        # no-op when liger-kernel is not installed.
        if tcfg.get("use_liger", True):
            from bgkit.utils.liger_integration import apply_liger_to_qwen35

            # Per-kernel toggles for bisecting which Liger component
            # regressed. ``use_liger_rmsnorm`` defaults to False
            # (2026-04-16: liger-kernel 0.7.x's LigerRMSNorm silently
            # corrupts Qwen3.5 backward — see commit 313f597). SwiGLU /
            # RoPE / fused linear-CE default to True — they provide the
            # bulk of the throughput win and are unaffected.
            patch_rmsnorm = bool(tcfg.get("use_liger_rmsnorm", False))
            patch_swiglu = bool(tcfg.get("use_liger_swiglu", True))
            patch_rope = bool(tcfg.get("use_liger_rope", True))
            use_liger_ce = bool(tcfg.get("use_liger_ce", True))
            enc_patched = apply_liger_to_qwen35(
                self.encoder,
                patch_rmsnorm=patch_rmsnorm,
                patch_swiglu=patch_swiglu,
                patch_rope=patch_rope,
            )
            dec_patched = apply_liger_to_qwen35(
                self.decoder,
                patch_rmsnorm=patch_rmsnorm,
                patch_swiglu=patch_swiglu,
                patch_rope=patch_rope,
            )
            if use_liger_ce:
                self.decoder.enable_liger_ce(True)
            logger.info(
                "liger_kernel_applied",
                encoder_modules=enc_patched,
                decoder_modules=dec_patched,
                patch_rmsnorm=patch_rmsnorm,
                patch_swiglu=patch_swiglu,
                patch_rope=patch_rope,
                use_liger_ce=use_liger_ce,
            )

        # BaseTrainer logging/device logic uses self.model
        self.model = self.decoder

        # --- Tokenizer (for chat template) ---
        tokenizer_name = decoder_cfg.backbone_name
        tokenizer_revision = decoder_cfg.get("backbone_revision", None)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            revision=tokenizer_revision,
        )

        # --- Compression curriculum state ---
        self._init_curriculum_state(tcfg)

        # --- Survivorship head config (per-level) ---
        from bgkit.training.survivorship_helpers import (
            init_state,
            load_reference_moments,
            resolve_level_ice_cfg,
            resolve_level_loss_cfg,
        )

        surv_cfg = tcfg.get("survivorship", {})
        # Step 3 only consumes l0; l1 may be absent.
        self._surv_l0 = resolve_level_loss_cfg(surv_cfg.get("l0", {}))
        self._surv_l1 = resolve_level_loss_cfg(surv_cfg.get("l1", {}))

        # Diagnostic metrics gating — live-tunable via control.json.
        self._diagnostic_metrics_every_n_steps = int(
            tcfg.get("diagnostic_metrics_every_n_steps", 10),
        )
        # Generation-eval cadence (runs every N evaluate() calls). Promoted to
        # an instance attribute so the live-config path can mutate it at
        # runtime (see LIVE_CONFIG_FIELDS).
        self._generation_eval_every = int(
            tcfg.get("eval", {}).get("generation_eval_every", 4),
        )

        # Floor knobs for the operator (passed to encoder.forward).
        ctrl_cfg = self.cfg.model.get("threshold_controller", {})
        self._floor_warmup = int(ctrl_cfg.get("min_per_sample_during_warmup", 1))
        self._floor_post_warmup = int(ctrl_cfg.get("min_per_sample_post_warmup", 0))

        # --- ICE distillation teacher (per-level) ---
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
                ice_path,
                embed_tokens,
                input_dim=int(ice_cfg.get("input_dim", 1024)),
                hidden_dim=int(ice_cfg.get("hidden_dim", 192)),
                num_layers=int(ice_cfg.get("num_layers", 3)),
                kernel_size=int(ice_cfg.get("kernel_size", 5)),
            ).to(device)
            logger.info(
                "ice_teacher_loaded",
                path=ice_path,
                bce_warmup_steps_l0=self._ice_l0.bce_warmup_steps,
                bce_warmup_steps_l1=self._ice_l1.bce_warmup_steps,
            )

        # --- Moment-match reference moments (per-level) ---
        # OmegaConf's DictConfig is not a dict subclass; use duck-typed access.
        mm_ref = tcfg.get("moment_match_reference", {}) or {}
        self._ref_moments_l0 = None
        self._ref_moments_l1 = None
        _l0_block = mm_ref.get("l0", None) if hasattr(mm_ref, "get") else None
        l0_path = _l0_block.get("path", None) if _l0_block is not None and hasattr(_l0_block, "get") else None
        if self._surv_l0.moment_match_weight > 0 and l0_path:
            self._ref_moments_l0 = load_reference_moments(l0_path)
            logger.info("moment_match_ref_l0_loaded", path=l0_path,
                        skew=self._ref_moments_l0[0], kurt=self._ref_moments_l0[1])
        elif self._surv_l0.moment_match_weight > 0:
            raise ValueError(
                "survivorship.l0.moment_match_weight > 0 requires "
                "moment_match_reference.l0.path. Run "
                "scripts/probe_ice_distribution.py to generate the file."
            )
        _l1_block = mm_ref.get("l1", None) if hasattr(mm_ref, "get") else None
        l1_path = _l1_block.get("path", None) if _l1_block is not None and hasattr(_l1_block, "get") else None
        if self._surv_l1.moment_match_weight > 0 and l1_path:
            self._ref_moments_l1 = load_reference_moments(l1_path)

        # Per-optimizer-step microbatch aggregation state (re-init at each
        # optimizer step boundary).
        self._surv_state_l0 = init_state()
        self._surv_state_l1 = init_state()

        # --- Dataset ---
        data_dir = self.cfg.data.tokens.input_dir
        max_seq_len = self.cfg.data.tokens.get("max_seq_len", 8192)
        variant_bank_path = self.cfg.data.tokens.variant_bank_path

        qa_data_dir = getattr(self.cfg.data, "qa_data_dir", None)
        qa_ratio = tcfg.get("qa_ratio", 0.0)
        inner_dataset = MmapTokenDataset(
            data_dir, max_seq_len=max_seq_len,
            include_metadata=True,
        )
        full_dataset = ChatReproDataset(
            inner_dataset,
            tokenizer=self.tokenizer,
            variant_bank_path=variant_bank_path,
            seed=self.cfg.get("seed", 42),
        )

        max_eval_samples = tcfg.get("max_eval_samples", 10000)
        eval_size = min(max(1, int(len(full_dataset) * 0.1)), max_eval_samples)
        train_size = len(full_dataset) - eval_size
        if train_size < 1:
            raise ValueError(
                f"Dataset too small for train/eval split (got {len(full_dataset)} samples, "
                "need at least 2)"
            )
        self.chat_dataset = full_dataset
        self.train_dataset, self.eval_dataset = random_split(
            full_dataset, [train_size, eval_size]
        )
        self._eval_count = 0

        # Token-budget batching (based on chat-formatted lengths)
        max_batch_tokens = tcfg.get("max_batch_tokens", 65536)
        # Eval defaults to 2× train budget (no backward → lower peak at
        # same budget). Overridable via training.max_batch_tokens_eval.
        max_batch_tokens_eval = self._resolve_eval_batch_budget(tcfg, max_batch_tokens)
        num_workers = self.cfg.compute.get("num_workers", 4)
        pin_memory = self.cfg.compute.get("pin_memory", False)
        seed = self.cfg.get("seed", 42)

        train_lengths = full_dataset.lengths[np.array(self.train_dataset.indices)]
        eval_lengths = full_dataset.lengths[np.array(self.eval_dataset.indices)]

        # Stash for live-tunable budget rebuild (see BaseTrainer._handle_max_batch_tokens)
        self._train_lengths = train_lengths
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

        # --- Optional QA dataset for dual-loader training ---
        self._qa_ratio = qa_ratio
        self._qa_dataset = None
        self._qa_train_dataloader = None
        self._qa_eval_dataloader = None

        if qa_data_dir and Path(qa_data_dir).exists() and self._qa_ratio > 0:
            from bgkit.data.datasets.qa_chat_repro_dataset import QAChatReproDataset
            from bgkit.data.datasets.qa_conditioned_dataset import MmapQAConditionedDataset

            try:
                qa_mmap = MmapQAConditionedDataset(qa_data_dir, max_seq_len=2048)
                qa_full = QAChatReproDataset(
                    qa_mmap, inner_dataset, self.tokenizer, seed=seed,
                )

                if len(qa_full) > 0:
                    qa_eval_size = min(
                        max(1, int(len(qa_full) * 0.1)), max_eval_samples // 3,
                    )
                    qa_train_size = len(qa_full) - qa_eval_size
                    self._qa_dataset = qa_full
                    qa_train_ds, qa_eval_ds = random_split(
                        qa_full, [qa_train_size, qa_eval_size]
                    )

                    qa_train_lengths = qa_full.lengths[np.array(qa_train_ds.indices)]
                    qa_eval_lengths = qa_full.lengths[np.array(qa_eval_ds.indices)]

                    qa_train_sampler = PackedTokenBudgetSampler(
                        qa_train_ds,
                        lengths=qa_train_lengths,
                        max_batch_tokens=max_batch_tokens,
                        shuffle=True,
                        seed=seed,
                    )
                    qa_eval_sampler = PackedTokenBudgetSampler(
                        qa_eval_ds,
                        lengths=qa_eval_lengths,
                        max_batch_tokens=max_batch_tokens_eval,
                        shuffle=False,
                    )

                    self._qa_train_dataloader = DataLoader(
                        qa_train_ds,
                        batch_sampler=qa_train_sampler,
                        collate_fn=collate_chat_repro,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                    )
                    self._qa_eval_dataloader = DataLoader(
                        qa_eval_ds,
                        batch_sampler=qa_eval_sampler,
                        collate_fn=collate_chat_repro,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                    )
                    logger.info(
                        "qa_dataset_loaded",
                        qa_train=qa_train_size,
                        qa_eval=qa_eval_size,
                        qa_ratio=self._qa_ratio,
                    )
            except (FileNotFoundError, ValueError) as e:
                logger.warning("qa_dataset_load_failed", error=str(e))

        # If QA dataset is loaded, wrap train_dataloader with interleaving
        if self._qa_train_dataloader is not None:
            self.train_dataloader = _InterleavingDataLoader(
                primary=self.train_dataloader,
                secondary=self._qa_train_dataloader,
                secondary_ratio=self._qa_ratio,
            )

        # --- Optimizer (with projection-aware freeze/unfreeze) ---
        self._configure_trainable_state()

        # Auto-calibrate the operator-facing tanh temperature from the
        # sidecar-loaded head's output distribution. We want base_raw/T to
        # have std ~1 so tanh operates in its linear region and ranking
        # info survives. BCE-with-logits distillation naturally produces
        # confident (large-magnitude) logits; hardcoding T=5 works for the
        # current sidecar but is brittle to hyperparameter changes in the
        # distillation pipeline. Probing once at startup makes T track
        # whatever the sidecar actually produced.
        self._calibrate_head_tanh_temperature()

        logger.info(
            "decoder_init_trainer_setup",
            train_samples=train_size,
            eval_samples=eval_size,
            device=str(device),
            qa_enabled=self._qa_train_dataloader is not None,
            encoder_unfreeze_step=self._encoder_unfreeze_step,
            compression_introduction_step=self._compression_introduction_step,
            bidi_warmup_steps=bidi_warmup,
        )

    # ------------------------------------------------------------------
    # Tanh temperature calibration
    # ------------------------------------------------------------------

    def _calibrate_head_tanh_temperature(self, n_probe_batches: int = 4) -> None:
        """Auto-calibrate L0's head-tanh temperature from the current head.

        Step 3 only trains L0; L1 calibration is the responsibility of
        the trainers that first introduce L1 (Step 4+, Phase 2).

        Skipped when no sidecar was loaded — the fresh default T=5.0 is
        fine for an all-random head whose output std is near-zero anyway,
        and re-probing at step 0 would lock in a meaningless T before
        any training has happened.
        """
        if not getattr(self, "_sidecar_loaded", False):
            return
        from bgkit.training.survivorship_helpers import (
            calibrate_head_tanh_temperature,
        )
        calibrated_T = calibrate_head_tanh_temperature(
            self.encoder.compressor,
            self.train_dataloader,
            self.device,
            level="l0",
            n_probe_batches=n_probe_batches,
        )
        if calibrated_T is not None:
            logger.info(
                "head_tanh_temperature_calibrated",
                level="l0",
                T=calibrated_T,
            )

    # ------------------------------------------------------------------
    # Curriculum initialization
    # ------------------------------------------------------------------

    def _init_curriculum_state(self, tcfg) -> None:
        """Initialize compression curriculum state variables."""
        # Ratio targeting — read from top-level config keys
        self._target_ratio_start = tcfg.get("target_ratio_start", 0.10)
        self._target_ratio_end = tcfg.get("target_ratio_end", 0.10)
        self._target_ratio_ramp_steps = tcfg.get("target_ratio_ramp_steps", 1)
        self._target_ratio_override: float | None = None
        anchor_grid = resolve_anchor_grid(
            self.cfg.model,
            self._target_ratio_start,
            getattr(
                self.encoder.compressor.threshold_l0,
                "anchor_ratios",
                torch.tensor([self._target_ratio_start], dtype=torch.float32),
            ).tolist(),
        )
        self._target_ratio_sampler_cfg = build_ratio_sampler_config(
            {
                "enabled": tcfg.get("sample_target_ratio_during_training", False),
                "mode": tcfg.get("target_ratio_sampling_mode", "window"),
                "window_above": tcfg.get("target_ratio_sampling_window_above", 0.10),
                "anchor_sampling_prob": tcfg.get(
                    "target_ratio_anchor_sampling_prob", 0.30,
                ),
                "jitter_abs": tcfg.get("target_ratio_jitter_abs", 0.0),
                "jitter_rel": tcfg.get("target_ratio_jitter_rel", 0.0),
            },
            anchor_grid=anchor_grid,
            default_ratio=self._target_ratio_start,
            enabled_default=False,
            mode_default="window",
        )
        import random

        self._target_ratio_rng = random.Random(int(self.cfg.get("seed", 42)))
        self._last_sampled_target_ratio: float | None = None
        self._warned_floor_past_anchor_grid = False

        # Runtime state (set by _configure_trainable_state or transitions)
        self._encoder_frozen = True
        self._compression_active = False
        self._is_evaluating = False

    # ------------------------------------------------------------------
    # Epoch sync
    # ------------------------------------------------------------------

    def _sync_epoch(self, epoch: int) -> None:
        """Propagate epoch to both verbatim and QA datasets."""
        super()._sync_epoch(epoch)
        qa_ds = getattr(self, "_qa_dataset", None)
        if qa_ds is not None and hasattr(qa_ds, "set_epoch"):
            qa_ds.set_epoch(epoch)

    # ------------------------------------------------------------------
    # Checkpoint resolution
    # ------------------------------------------------------------------

    def _resolve_bgkit_checkpoint(self) -> str | None:
        """Resolve bgkit_checkpoint: 'auto' -> best from preceding phase.

        Resolution by phase:
          - phase1_step1: joint_block_pretrain.
          - phase1_step3 (reconstruction with compression curriculum):
            prefer phase1_step2p5 (projection embed-anchor repair),
            fall back to phase1_step2 (pruned encoder).
          - phase1_step4 (QA-conditioned head supervision): phase1_step3
            (the reconstruction-trained encoder + LoRA decoder). Falls
            back to phase1_step2p5 / phase1_step2 only if no Step 3
            checkpoint is registered — logged prominently because QA
            training on the bare pruned encoder is not the intended flow.
        """
        bgkit_checkpoint = self.cfg.get("bgkit_checkpoint", None)
        if bgkit_checkpoint == "auto":
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            phase = self.cfg.training.get("phase", "phase1_step1")
            if phase == "phase1_step3":
                resolved = self._resolve_step2p5_with_step2_fallback(
                    checkpoint_dir, phase
                )
            elif phase == "phase1_step4":
                try:
                    resolved = resolve_checkpoint(
                        checkpoint_dir,
                        phase="phase1_step3",
                        metric="eval/loss",
                        label="bgkit_checkpoint",
                    )
                    logger.info(
                        "resolved_from_phase1_step3",
                        phase=phase,
                        checkpoint=resolved.name,
                    )
                except ValueError:
                    logger.warning(
                        "no_phase1_step3_checkpoint_falling_back_to_2p5_or_2",
                        phase=phase,
                    )
                    resolved = self._resolve_step2p5_with_step2_fallback(
                        checkpoint_dir, phase
                    )
            else:
                resolved = resolve_checkpoint(
                    checkpoint_dir,
                    phase="joint_block_pretrain",
                    metric="eval/mse_repro",
                    label="bgkit_checkpoint",
                )
            bgkit_checkpoint = str(resolved)

        self._input_sources = {}
        if bgkit_checkpoint is not None:
            self._input_sources["bgkit"] = Path(bgkit_checkpoint).name

        return bgkit_checkpoint

    def _resolve_step2p5_with_step2_fallback(
        self, checkpoint_dir: Path, phase: str
    ) -> Path:
        """Prefer phase1_step2p5, fall back to phase1_step2.

        Step 2.5 re-anchors the projection to decoder.embed_tokens;
        skipping it bakes the off-manifold equilibrium into the caller.
        """
        try:
            resolved = resolve_checkpoint(
                checkpoint_dir,
                phase="phase1_step2p5",
                metric="eval/loss",
                label="bgkit_checkpoint",
            )
            logger.info(
                "resolved_from_phase1_step2p5",
                phase=phase,
                checkpoint=resolved.name,
            )
            return resolved
        except ValueError:
            logger.info(
                "no_phase1_step2p5_checkpoint_falling_back_to_step2",
                phase=phase,
            )
            return resolve_checkpoint(
                checkpoint_dir,
                phase="phase1_step2",
                metric="eval/loss",
                label="bgkit_checkpoint",
            )

    # ------------------------------------------------------------------
    # Freeze / unfreeze management
    # ------------------------------------------------------------------

    def _unfreeze_decoder_respecting_lora(self) -> None:
        """Set requires_grad=True on the decoder, honoring LoRA freezing.

        When LoRA is applied, ``peft.get_peft_model`` has already frozen
        the base decoder weights and left only the LoRA adapters
        trainable. A naive ``self.decoder.requires_grad_(True)`` here
        would undo that — we'd train the full 758 M base decoder
        alongside the 6 M LoRA adapters, silently defeating the whole
        point of the LoRA config (bug caught 2026-04-18 after
        ``trainable_state_configured decoder_params=758782784`` was
        logged alongside ``decoder_lora_applied ratio=0.0084``).

        When LoRA is not applied, behave as before: unfreeze everything.
        """
        if getattr(self.decoder, "_has_lora", False):
            n_adapter = 0
            n_total = 0
            for name, p in self.decoder.named_parameters():
                is_adapter = "lora_" in name
                p.requires_grad_(is_adapter)
                n_total += 1
                if is_adapter:
                    n_adapter += 1
            logger.info(
                "decoder_unfreeze_lora_only",
                adapter_params=n_adapter,
                total_params=n_total,
            )
        else:
            self.decoder.requires_grad_(True)

    def _apply_freeze(self) -> None:
        """Freeze top transformer layer, lm_head, and embed_tokens if configured."""
        tcfg = self.cfg.training
        if not tcfg.get("freeze_top_layer", False):
            return

        self.decoder.backbone.model.layers[-1].requires_grad_(False)
        self.decoder.backbone.lm_head.requires_grad_(False)
        self.decoder.backbone.model.embed_tokens.requires_grad_(False)

        num_layers = len(self.decoder.backbone.model.layers)
        logger.info(
            "decoder_layer_freeze",
            total_layers=num_layers,
            trainable_layers=num_layers - 1,
            lr_scale_bottom=tcfg.get("lr_scale_bottom", 0.1),
        )

    def _configure_trainable_state(self) -> None:
        """Set requires_grad flags and build optimizer based on current global_step.

        Called by setup() (global_step=0) and load_checkpoint() (after restoring
        global_step). Makes freeze/optimizer state a pure function of step position.
        """
        # Projection block: unfreeze and set to train mode (dropout etc.)
        if self._train_projection:
            self.encoder.projection_block.requires_grad_(True)
            self.encoder.projection_block.train()

        # Decoder: frozen during projection-only warmup, unfrozen after
        self._decoder_frozen = (
            self._projection_only_steps > 0
            and self.global_step < self._projection_only_steps
        )
        if self._decoder_frozen:
            self.decoder.requires_grad_(False)
        else:
            self._unfreeze_decoder_respecting_lora()
            self._apply_freeze()

        # Encoder: frozen until encoder_unfreeze_step
        self._encoder_frozen = (
            self._encoder_unfreeze_step is None
            or self.global_step < self._encoder_unfreeze_step
        )
        if not self._encoder_frozen:
            self.encoder.compressor.requires_grad_(True)
            self.encoder.compressor.train()
            maybe_enable_gradient_checkpointing(
                self.encoder.compressor.backbone, self.cfg,
            )
        else:
            self.encoder.compressor.requires_grad_(False)
            self.encoder.compressor.eval()

        # Compression: active after compression_introduction_step
        self._compression_active = (
            self._compression_introduction_step is not None
            and self.global_step >= self._compression_introduction_step
        )

        # Build optimizer matching current trainable params
        self._setup_optimizer()

        # Log for verification
        proj_params = sum(
            p.numel() for p in self.encoder.projection_block.parameters() if p.requires_grad
        )
        dec_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        enc_params = sum(
            p.numel() for p in self.encoder.compressor.parameters() if p.requires_grad
        )
        logger.info(
            "trainable_state_configured",
            step=self.global_step,
            decoder_frozen=self._decoder_frozen,
            encoder_frozen=self._encoder_frozen,
            compression_active=self._compression_active,
            projection_params=proj_params,
            decoder_params=dec_params,
            encoder_params=enc_params,
        )

    # ------------------------------------------------------------------
    # Curriculum transitions
    # ------------------------------------------------------------------

    def _maybe_end_projection_only(self) -> None:
        """Transition from projection-only warmup to full training."""
        if not self._decoder_frozen:
            return
        if self.global_step < self._projection_only_steps:
            return

        self._unfreeze_decoder_respecting_lora()
        self._apply_freeze()
        self._decoder_frozen = False

        decoder_groups = self._build_decoder_param_groups()
        for group in decoder_groups:
            self._add_param_group_to_optimizer(group)

        sp = self._schedule_params or {
            "max_steps": self.cfg.training.max_steps,
            "base_lr": self.cfg.training.lr,
            "warmup_steps": self.cfg.training.warmup_steps,
        }
        for pg in self.optimizer.param_groups:
            group_base = pg.get("base_lr", sp["base_lr"])
            pg["lr"] = cosine_with_warmup(
                self.global_step, int(sp["max_steps"]), int(sp["warmup_steps"]), group_base
            )

        logger.info(
            "projection_only_phase_ended",
            step=self.global_step,
            decoder_param_groups=len(decoder_groups),
        )

    def _maybe_unfreeze_encoder(self) -> None:
        """Transition from frozen encoder to trainable encoder."""
        if not self._encoder_frozen:
            return
        if self._encoder_unfreeze_step is None:
            return
        if self.global_step < self._encoder_unfreeze_step:
            return

        # Unfreeze compressor backbone
        self.encoder.compressor.requires_grad_(True)
        self.encoder.compressor.train()
        maybe_enable_gradient_checkpointing(
            self.encoder.compressor.backbone, self.cfg,
        )
        self._encoder_frozen = False

        # Add compressor params to optimizer (projection block already there)
        encoder_groups = self._build_encoder_param_groups()
        for group in encoder_groups:
            self._add_param_group_to_optimizer(group)

        # Sync LR schedule for all param groups
        sp = self._schedule_params or {
            "max_steps": self.cfg.training.max_steps,
            "base_lr": self.cfg.training.lr,
            "warmup_steps": self.cfg.training.warmup_steps,
        }
        for pg in self.optimizer.param_groups:
            group_base = pg.get("base_lr", sp["base_lr"])
            pg["lr"] = cosine_with_warmup(
                self.global_step, int(sp["max_steps"]), int(sp["warmup_steps"]), group_base
            )

        logger.info(
            "encoder_unfrozen",
            step=self.global_step,
            encoder_param_groups=len(encoder_groups),
        )

    def _maybe_introduce_compression(self) -> None:
        """Start using survivorship head compression when the curriculum step is reached."""
        if self._compression_active:
            return
        if self._compression_introduction_step is None:
            return
        if self.global_step < self._compression_introduction_step:
            return

        self._compression_active = True
        logger.info("compression_introduced", step=self.global_step)

    # ------------------------------------------------------------------
    # Optimizer construction
    # ------------------------------------------------------------------

    def _build_decoder_param_groups(self) -> list[dict]:
        """Build decoder param groups (with differential LR if freeze_top_layer)."""
        tcfg = self.cfg.training
        decoder_lr = tcfg.get("decoder_lr", tcfg.lr)
        if tcfg.get("freeze_top_layer", False):
            return self._build_layerwise_param_groups(
                decoder_lr, tcfg.get("lr_scale_bottom", 0.1),
            )
        else:
            params = [p for p in self.decoder.parameters() if p.requires_grad]
            if params:
                return [{"params": params, "lr": decoder_lr, "base_lr": decoder_lr}]
            return []

    def _build_layerwise_param_groups(
        self, base_lr: float, lr_scale_bottom: float,
    ) -> list[dict]:
        """Build optimizer param groups with rising LR from bottom to top.

        Called only when freeze_top_layer is enabled. Assumes the top layer
        and lm_head are already frozen via requires_grad_(False).
        """
        layers = self.decoder.backbone.model.layers
        num_trainable = len(layers) - 1  # last layer is frozen

        param_groups = []
        for i in range(num_trainable):
            t = i / max(num_trainable - 1, 1)
            scale = lr_scale_bottom + t * (1.0 - lr_scale_bottom)
            group_lr = base_lr * scale
            params = [p for p in layers[i].parameters() if p.requires_grad]
            if params:
                param_groups.append({
                    "params": params,
                    "lr": group_lr,
                    "base_lr": group_lr,
                })

        # Final norm
        norm_params = [
            p for p in self.decoder.backbone.model.norm.parameters() if p.requires_grad
        ]
        if norm_params:
            param_groups.append({
                "params": norm_params,
                "lr": base_lr,
                "base_lr": base_lr,
            })

        return param_groups

    def _build_projection_param_groups(self) -> list[dict]:
        """Build projection block param group."""
        tcfg = self.cfg.training
        proj_lr = tcfg.get("projection_lr")
        if proj_lr is None:
            proj_lr = tcfg.lr
        proj_params = [
            p for p in self.encoder.projection_block.parameters() if p.requires_grad
        ]
        if proj_params:
            return [{"params": proj_params, "lr": proj_lr, "base_lr": proj_lr}]
        return []

    def _build_encoder_param_groups(self) -> list[dict]:
        """Build encoder compressor param groups (excludes projection block)."""
        tcfg = self.cfg.training
        encoder_lr = tcfg.get("encoder_lr", tcfg.lr)
        compressor_params = [
            p for p in self.encoder.compressor.parameters() if p.requires_grad
        ]
        if compressor_params:
            return [{"params": compressor_params, "lr": encoder_lr, "base_lr": encoder_lr}]
        return []

    def _muon_excluded_param_ids(self) -> frozenset[int]:
        """Return param IDs of embedding/lm_head — 2D but should not use Muon.

        Handles both bare CausalLM and PeftModel-wrapped decoder.
        """
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
                exclude.add(id(p))
        lm_head = getattr(causal_lm, "lm_head", None)
        if lm_head is not None:
            for p in lm_head.parameters():
                exclude.add(id(p))
        return frozenset(exclude)

    def _setup_optimizer(self) -> None:
        """Create optimizer from current trainable param groups."""
        tcfg = self.cfg.training

        param_groups = []
        if self._train_projection:
            param_groups.extend(self._build_projection_param_groups())
        if not self._decoder_frozen:
            param_groups.extend(self._build_decoder_param_groups())
        if not self._encoder_frozen:
            param_groups.extend(self._build_encoder_param_groups())

        self.optimizer = self._create_optimizer(
            param_groups, tcfg.lr, exclude_from_muon=self._muon_excluded_param_ids()
        )

    def trainable_parameters(self) -> list:
        params = [p for p in self.decoder.parameters() if p.requires_grad]
        if self._train_projection:
            params += [
                p for p in self.encoder.projection_block.parameters() if p.requires_grad
            ]
        if not self._encoder_frozen:
            params += [
                p for p in self.encoder.compressor.parameters() if p.requires_grad
            ]
        return params

    # ------------------------------------------------------------------
    # Survivorship head + compression
    # ------------------------------------------------------------------

    def _current_target_ratio(self) -> float:
        """Compute the current target compression ratio from the ramp or override."""
        if self._target_ratio_override is not None:
            return self._target_ratio_override
        if not self._compression_active or self._compression_introduction_step is None:
            return 1.0  # no compression

        # Ramp relative to compression introduction step
        steps_since_intro = self.global_step - self._compression_introduction_step
        if steps_since_intro >= self._target_ratio_ramp_steps:
            return self._target_ratio_end
        t = max(steps_since_intro, 0) / max(self._target_ratio_ramp_steps, 1)
        return self._target_ratio_start + t * (self._target_ratio_end - self._target_ratio_start)

    def _sample_target_ratio(self) -> float:
        """Sample a requested ratio using the configured window or jitter policy."""
        floor = self._current_target_ratio()
        anchor_max = (
            max(self._target_ratio_sampler_cfg.anchor_grid)
            if self._target_ratio_sampler_cfg.anchor_grid else 0.95
        )
        if (
            self._target_ratio_sampler_cfg.mode == "window"
            and floor > anchor_max + 1e-8
            and not self._warned_floor_past_anchor_grid
        ):
            logger.warning(
                "target_ratio_floor_past_anchor_grid",
                floor=round(floor, 4),
                anchor_max=round(anchor_max, 4),
                hint="Threshold curve has no calibrated anchors for ratios "
                     "this high; θ(r) saturates to the top-anchor value.",
            )
            self._warned_floor_past_anchor_grid = True
        return sample_ratio(
            rng=self._target_ratio_rng,
            config=self._target_ratio_sampler_cfg,
            base_ratio=floor,
            is_evaluating=self._is_evaluating,
            override_active=self._target_ratio_override is not None,
        )

    def _compute_survivorship_losses(
        self,
        enc_out,
        target_ratio: float,
        content_token_ids: torch.Tensor | None = None,
        content_cu_seqlens: torch.Tensor | None = None,
        level: str = "l0",
        answer_position_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Delegate to ``compute_survivorship_losses`` from the shared helpers.

        Packed inputs: ``content_token_ids`` is flat ``(N,)`` and
        ``content_cu_seqlens`` encodes per-sample boundaries.
        """
        from bgkit.training.survivorship_helpers import (
            LevelICECfg,
            LevelLossCfg,
            compute_survivorship_losses,
        )

        if level == "l0":
            weights = getattr(self, "_surv_l0", LevelLossCfg())
            ice_cfg = getattr(self, "_ice_l0", LevelICECfg())
            ref_moments = getattr(self, "_ref_moments_l0", None)
        elif level == "l1":
            weights = getattr(self, "_surv_l1", LevelLossCfg())
            ice_cfg = getattr(self, "_ice_l1", LevelICECfg())
            ref_moments = getattr(self, "_ref_moments_l1", None)
        else:
            raise ValueError(f"Unknown level: {level!r}")

        return compute_survivorship_losses(
            enc_out=enc_out,
            level=level,
            weights=weights,
            ice_cfg=ice_cfg,
            ref_moments=ref_moments,
            ice_teacher=getattr(self, "_ice_teacher", None),
            global_step=self.global_step,
            content_token_ids=content_token_ids,
            content_cu_seqlens=content_cu_seqlens,
            target_ratio=target_ratio,
            answer_position_mask=answer_position_mask,
        )

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _compute_survivors(
        self,
        batch: dict[str, torch.Tensor],
        *,
        target_ratio: float | None = None,
    ):
        """Run content through BgKIT encoder (packed) and return encoder output.

        Inputs come from :func:`bgkit.data.collators.collate_chat_repro`:
        flat ``content_token_ids`` / ``compression_prompt_ids`` with their
        respective ``cu_seqlens``. The encoder is called in fully packed
        form; no attention masks are constructed.
        """
        from bgkit.utils.packing import position_ids_from_cu

        device = self.device
        content_token_ids = batch["content_token_ids"].to(device)
        content_cu = batch["content_cu_seqlens"].to(device)
        content_position_ids = batch["content_position_ids"].to(device)
        prompt_ids = batch["compression_prompt_ids"].to(device)
        prompt_cu = batch["compression_prompt_cu_seqlens"].to(device)
        prompt_position_ids = position_ids_from_cu(prompt_cu, int(prompt_ids.shape[0]))

        if target_ratio is None and self._compression_active:
            target_ratio = self._current_target_ratio()
        util_active = (
            self._compression_active
            and not self._encoder_frozen
            and self._surv_l0.utility_grad_loss_weight > 0.0
        )

        # Per-sample floor: active during BCE warmup (so the head doesn't
        # starve the decoder while it learns), disabled afterward so
        # zero-survivor samples are accepted as legitimate signal at full
        # compression. Pin to the L0 schedule (Step 3 trains L0 only).
        ice_teacher = getattr(self, "_ice_teacher", None)
        ice_l0 = getattr(self, "_ice_l0", None)
        if (
            ice_teacher is not None
            and ice_l0 is not None
            and ice_l0.enabled
            and self.global_step < ice_l0.bce_warmup_steps
        ):
            min_per_sample = getattr(self, "_floor_warmup", 1)
        else:
            min_per_sample = getattr(self, "_floor_post_warmup", 0)

        # When the compressor is frozen AND the projection block is NOT
        # trainable, wrap in no_grad to save memory. Otherwise we must run
        # with autograd enabled so gradients can flow through the trainable
        # projection block (frozen compressor params still won't receive
        # gradients because their requires_grad=False).
        use_no_grad = self._encoder_frozen and not self._train_projection

        bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()
        content_emb = bgkit_embed(content_token_ids)
        prompt_emb = bgkit_embed(prompt_ids)

        if use_no_grad:
            with torch.no_grad():
                enc_out = self.encoder(
                    content_embeddings=content_emb,
                    content_cu_seqlens=content_cu,
                    content_position_ids=content_position_ids,
                    prompt_embeddings=prompt_emb,
                    prompt_cu_seqlens=prompt_cu,
                    prompt_position_ids=prompt_position_ids,
                    target_ratio=target_ratio,
                    level="l0",
                    min_per_sample=min_per_sample,
                )
        else:
            enc_out = self.encoder(
                content_embeddings=content_emb,
                content_cu_seqlens=content_cu,
                content_position_ids=content_position_ids,
                prompt_embeddings=prompt_emb,
                prompt_cu_seqlens=prompt_cu,
                prompt_position_ids=prompt_position_ids,
                target_ratio=target_ratio,
                level="l0",
                min_per_sample=min_per_sample,
                utility_grad_active=util_active,
            )

        return enc_out

    def _decoder_forward_packed(
        self, batch: dict, enc_out,
    ) -> torch.Tensor:
        """Run the packed decoder forward from a chat-repro batch.

        Splits the flat token_ids / loss_mask at each sample's
        ``bgkit_splice_start`` into prefix/suffix lists, and builds the
        per-segment loss mask over the assembled ``[prefix | survivors |
        suffix]`` layout that ``forward_with_single_splice`` expects.
        """
        device = self.device
        token_ids_flat = batch["token_ids"].to(device)
        tok_cu = batch["cu_seqlens"].to(device)
        splice_start = batch["bgkit_splice_start"].to(device)
        splice_len = batch["bgkit_splice_len"].to(device)
        loss_mask_flat = batch["loss_mask"].to(device).to(torch.bool)

        survivors = enc_out.survivor_embeddings
        survivor_cu = enc_out.survivor_cu_seqlens

        batch_size = int(tok_cu.shape[0]) - 1
        # Pull the four small int tensors to CPU once each (4 syncs total)
        # instead of doing per-sample `.item()` calls inside the loop
        # (was 2*B = ~16 syncs/microbatch on ARM unified memory). The
        # trailing assembly stays on-device — only the loop's control
        # variables are CPU ints.
        tok_cu_list = tok_cu.cpu().to(torch.int64).tolist()
        surv_cu_list = survivor_cu.cpu().to(torch.int64).tolist()
        splice_start_list = splice_start.cpu().to(torch.int64).tolist()
        splice_len_list = splice_len.cpu().to(torch.int64).tolist()

        prefix_ids: list[torch.Tensor] = []
        suffix_ids: list[torch.Tensor] = []
        per_segment_loss_masks: list[torch.Tensor] = []
        for b in range(batch_size):
            sample_start = tok_cu_list[b]
            sample_end = tok_cu_list[b + 1]
            sample_tokens = token_ids_flat[sample_start:sample_end]
            sample_loss = loss_mask_flat[sample_start:sample_end]
            splice_b_start = splice_start_list[b]
            splice_b_len = splice_len_list[b]
            if splice_b_start < 0:
                splice_b_start = sample_end - sample_start
                splice_b_len = 0
            pre = sample_tokens[:splice_b_start]
            suf = sample_tokens[splice_b_start + splice_b_len :]
            prefix_ids.append(pre)
            suffix_ids.append(suf)

            pre_mask = sample_loss[:splice_b_start]
            suf_mask = sample_loss[splice_b_start + splice_b_len :]
            k_i = surv_cu_list[b + 1] - surv_cu_list[b]
            surv_mask = pre_mask.new_zeros(k_i)
            per_segment_loss_masks.append(torch.cat([pre_mask, surv_mask, suf_mask], dim=0))

        flat_loss_mask = torch.cat(per_segment_loss_masks, dim=0)
        return self.decoder.forward_with_single_splice(
            survivor_embeddings=survivors,
            survivor_cu_seqlens=survivor_cu,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            loss_mask=flat_loss_mask,
        )

    def _forward_backward(self, batch) -> dict[str, float]:
        self.decoder.train()
        if self._train_projection:
            self.encoder.projection_block.train()
        if not self._encoder_frozen:
            self.encoder.compressor.train()

        target_ratio = self._sample_target_ratio() if self._compression_active else None
        if self._encoder_frozen and not self._train_projection:
            with torch.no_grad():
                enc_out = self._compute_survivors(batch, target_ratio=target_ratio)
        else:
            enc_out = self._compute_survivors(batch, target_ratio=target_ratio)

        # BF16 autocast for decoder forward + backward
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            loss = self._decoder_forward_packed(batch, enc_out)

        total_loss = loss

        metrics: dict[str, float] = {"loss": loss.item()}

        # Survivorship head auxiliary losses (when compression is active).
        # Note: survive_probs_metrics is the canonical "probability that this
        # position survives" exposed for metrics; survive_probs is its alias
        # on CompressorOutput for backwards compatibility.
        if self._compression_active and enc_out.logits_for_op is not None:
            from bgkit.training.survivorship_helpers import accumulate

            self._last_sampled_target_ratio = target_ratio
            answer_position_mask = batch.get("answer_position_mask")
            if answer_position_mask is not None:
                answer_position_mask = answer_position_mask.to(self.device).to(torch.bool)
            content_token_ids = batch["content_token_ids"].to(self.device)
            content_cu = batch["content_cu_seqlens"].to(self.device)
            surv_loss, surv_metrics = self._compute_survivorship_losses(
                enc_out,
                target_ratio,
                content_token_ids=content_token_ids,
                content_cu_seqlens=content_cu,
                level="l0",
                answer_position_mask=answer_position_mask,
            )
            total_loss = total_loss + surv_loss
            metrics.update(surv_metrics)

            # Aggregate per-microbatch (sum, count) for true-mean θ/μ updates
            # at the optimizer-step boundary.
            accumulate(self._surv_state_l0, enc_out, target_ratio=target_ratio)

            # Logging metrics (all detached to avoid retaining autograd graph).
            # Heavy diagnostic metrics require .item() syncs; gate them on
            # a modest interval to keep them off the hot path.
            diag_interval = int(self._diagnostic_metrics_every_n_steps)
            emit_diag = (
                diag_interval <= 1 or self.global_step % diag_interval == 0
            )
            if enc_out.survivor_mask is not None:
                metrics["min_target_ratio"] = target_ratio
                metrics["sampled_target_ratio"] = target_ratio
                # Packed: all content positions are valid; valid count is N.
                n_valid = int(content_token_ids.shape[0])
                if emit_diag:
                    n_survivors = int(enc_out.survivor_mask.sum().item())
                    metrics["actual_ratio"] = n_survivors / max(n_valid, 1)
                # floor_trigger_rate / theta are zero-dim tensors on
                # device; .item() at log time (still gated on emit_diag
                # to keep off hot path).
                if emit_diag:
                    if enc_out.floor_trigger_rate is not None:
                        metrics["floor_trigger_rate"] = float(
                            enc_out.floor_trigger_rate.item(),
                        )
                    if enc_out.theta_tensor is not None:
                        metrics["theta_l0_step"] = float(enc_out.theta_tensor.item())
                    # Head health
                    if enc_out.base_raw is not None:
                        base = enc_out.base_raw.detach()
                        base_norm = float(base.norm().item()) / max(n_valid ** 0.5, 1.0)
                        metrics["base_norm"] = base_norm
                    if enc_out.logits_for_op is not None:
                        logits = enc_out.logits_for_op.detach()
                        metrics["head_logit_std"] = float(logits.std().item())
                        metrics["head_logit_min"] = float(logits.min().item())
                        metrics["head_logit_max"] = float(logits.max().item())
                    # L0 collapse-detection diagnostics. See
                    # ``survivorship_diagnostics`` docstring.
                    if enc_out.organic_rate_std is not None:
                        metrics["l0_organic_rate_std"] = float(
                            enc_out.organic_rate_std.item(),
                        )
                    if enc_out.undecided_fraction is not None:
                        metrics["l0_undecided_fraction"] = float(
                            enc_out.undecided_fraction.item(),
                        )

        # Main scaled backward (for gradient accumulation). This populates
        # ``enc_out.post_head_content_grad`` via the compressor's backward
        # hook, which we consume below for utility-gradient BCE.
        (total_loss / self._accum_steps).backward()

        # Utility-gradient BCE distillation (post-backward). Replaces the
        # old soft-attn replay: the main backward already routed the
        # decoder-CE gradient to ``content_hidden``; we use that captured
        # gradient to build a binary top-k teacher for a second small
        # backward that cleanly terminates at head weights.
        from bgkit.training.survivorship_helpers import utility_grad_bce_loss

        if (
            self._compression_active
            and not self._encoder_frozen
            and self._surv_l0.utility_grad_loss_weight > 0.0
            and enc_out.post_head_content_values is not None
        ):
            content_cu = batch["content_cu_seqlens"].to(self.device)
            content_grad = enc_out.get_content_grad()
            util_loss, util_metrics = utility_grad_bce_loss(
                base_raw_for_util=enc_out.base_raw_for_util,
                content_grad=content_grad,
                content_values=enc_out.post_head_content_values,
                valid_mask=None,
                pinned_mask=None,
                target_ratio=self._current_target_ratio(),
                content_cu_seqlens=content_cu,
            )
            if util_loss.requires_grad:
                w = self._surv_l0.utility_grad_loss_weight
                (util_loss * w / self._accum_steps).backward()
            for k, v in util_metrics.items():
                metrics[f"l0_{k}"] = v
            if content_grad is not None:
                metrics["l0_content_grad_norm"] = float(content_grad.norm().item())

        # Explicit release of CompressorOutput tensor references to
        # avoid the utility-grad hook-state leak (diagnosed 2026-04-20,
        # ~140 MB/step growth to 107 GB by step 2650). See
        # ``CompressorOutput.release()`` docstring for the mechanism.
        enc_out.release()

        return metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.decoder.eval()
        if self._train_projection:
            self.encoder.projection_block.eval()
        if not self._encoder_frozen:
            self.encoder.compressor.eval()
        self._is_evaluating = True
        self._eval_count += 1

        try:
            total_loss = 0.0
            total_content_tokens = 0.0
            total_survivors = 0
            total_content = 0

            num_batches = len(self.eval_dataloader)
            for batch_idx, batch in enumerate(self.eval_dataloader):
                if batch_idx % 100 == 0:
                    logger.info("eval_progress", batch=batch_idx, total=num_batches)
                loss_mask = batch["loss_mask"].to(self.device)

                enc_out = self._compute_survivors(batch)

                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"
                ):
                    loss = self._decoder_forward_packed(batch, enc_out)

                batch_content_tokens = loss_mask.sum().item()
                total_loss += loss.item() * batch_content_tokens
                total_content_tokens += batch_content_tokens

                if self._compression_active and enc_out.survivor_mask is not None:
                    total_survivors += int(enc_out.survivor_mask.sum().item())
                    # Packed: all content positions are valid.
                    total_content += int(batch["content_token_ids"].shape[0])

                # Drop per-iteration encoder output so it doesn't linger
                # in the allocator until the next iteration overwrites
                # the reference; matches training-path discipline.
                enc_out.release()

            avg_loss = total_loss / max(total_content_tokens, 1)
            perplexity = torch.exp(torch.tensor(avg_loss)).item()

            metrics: dict[str, float] = {
                "loss": avg_loss,
                "perplexity": perplexity,
            }

            if self._compression_active and total_content > 0:
                metrics["compression_ratio"] = total_survivors / total_content

            # QA eval (separate dataloader with different suffix_ids)
            qa_eval_loader = getattr(self, "_qa_eval_dataloader", None)
            qa_dataset = getattr(self, "_qa_dataset", None)
            if qa_eval_loader is not None and qa_dataset is not None:
                qa_loss, qa_tokens = 0.0, 0.0
                for batch in qa_eval_loader:
                    loss_mask = batch["loss_mask"].to(self.device)
                    enc_out = self._compute_survivors(batch)
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"
                    ):
                        loss = self._decoder_forward_packed(batch, enc_out)
                    bt = loss_mask.sum().item()
                    qa_loss += loss.item() * bt
                    qa_tokens += bt
                    enc_out.release()

                qa_avg = qa_loss / max(qa_tokens, 1)
                metrics["qa_loss"] = qa_avg
                metrics["qa_perplexity"] = torch.exp(torch.tensor(qa_avg)).item()

                combined_tokens = total_content_tokens + qa_tokens
                if combined_tokens > 0:
                    metrics["combined_loss"] = (
                        total_loss + qa_loss
                    ) / combined_tokens

            # Generation metrics (expensive -- only every Nth eval).
            # Scoped separately from outer ``evaluate`` because the KV
            # cache under ``max_new_tokens`` is a different memory regime
            # than CE-only eval and deserves its own budget contract.
            # Cadence is live-tunable via ``_generation_eval_every`` —
            # control.json can bump it up/down without restart.
            # Fall back to the config value when setup() hasn't populated the
            # instance attr yet (light-weight test fixtures that bypass
            # ``setup`` still construct the trainer directly).
            gen_every_default = self.cfg.training.get("eval", {}).get(
                "generation_eval_every", 4,
            )
            gen_every = max(
                1,
                int(getattr(self, "_generation_eval_every", gen_every_default)),
            )
            if self._eval_count % gen_every == 0:
                # Defense-in-depth: gen_eval is optional observability, not a
                # training-correctness path. On failure (OOM in gen scope,
                # transient FA4 quirk, tokenizer edge case, etc.) skip the
                # metrics and keep training rather than tearing down the run.
                try:
                    with memory_budget_scope(
                        "gen_eval", cap_gb=self._scope_cap("gen_eval"),
                    ):
                        gen_metrics = self._run_generation_eval()
                    metrics.update(gen_metrics)
                except Exception as exc:
                    logger.warning(
                        "gen_eval_failed",
                        error=str(exc),
                        exc_type=type(exc).__name__,
                        step=self.global_step,
                        eval_count=self._eval_count,
                    )

        finally:
            self._is_evaluating = False

        return metrics

    @torch.no_grad()
    def _run_generation_eval(self) -> dict[str, float]:
        """Run generation-based eval metrics on a small subset (packed path).

        ``generate_with_single_splice`` is a custom B=1 per-sample loop on
        the packed decoder, so we drive it one sample at a time using the
        per-sample survivor slices from ``survivor_cu_seqlens``.
        """
        tcfg = self.cfg.training
        eval_cfg = tcfg.get("eval", {})
        max_gen_samples = eval_cfg.get("generation_samples", 50)
        gen_max_new_tokens = int(eval_cfg.get("generation_max_new_tokens", 256))

        gen_metrics: dict[str, float] = {}
        generated_texts: list[str] = []
        generated_languages: list[str] = []
        all_survivors: list[torch.Tensor] = []
        samples_seen = 0

        suffix_ids = self.chat_dataset.suffix_ids.to(self.device)

        for batch in self.eval_dataloader:
            if samples_seen >= max_gen_samples:
                break

            prefix_ids_flat = batch["prefix_ids"].to(self.device)
            prefix_cu = batch["prefix_cu_seqlens"].to(self.device)

            enc_out = self._compute_survivors(batch)
            survivors = enc_out.survivor_embeddings
            survivor_cu = enc_out.survivor_cu_seqlens

            batch_size = int(survivor_cu.shape[0]) - 1
            surv_cu_list = survivor_cu.to(torch.int64).tolist()
            pre_cu_list = prefix_cu.to(torch.int64).tolist()

            # Collect survivors for embedding health (subsample to ~512 vectors)
            total_vecs = sum(s.size(0) for s in all_survivors)
            if total_vecs < 512:
                remaining = 512 - total_vecs
                all_survivors.append(survivors[:remaining].detach())

            # Per-sample generation.
            for b in range(batch_size):
                if samples_seen >= max_gen_samples:
                    break
                k_start, k_end = surv_cu_list[b], surv_cu_list[b + 1]
                pre_start, pre_end = pre_cu_list[b], pre_cu_list[b + 1]
                surv_slice = survivors[k_start:k_end]
                cu_single = torch.tensor(
                    [0, k_end - k_start], dtype=torch.int32, device=self.device,
                )
                gen_output = self.decoder.generate_with_single_splice(
                    survivor_embeddings=surv_slice,
                    survivor_cu_seqlens=cu_single,
                    prefix_ids=prefix_ids_flat[pre_start:pre_end],
                    suffix_ids=suffix_ids,
                    tokenizer=self.tokenizer,
                    max_new_tokens=gen_max_new_tokens,
                    temperature=0.0,
                )
                generated_texts.extend(gen_output.content_text)
                generated_languages.append(batch["languages"][b])
                samples_seen += 1
                del gen_output

            del survivors
            enc_out.release()

        # Parse success rate
        if generated_texts:
            gen_metrics["parse_success_rate"] = parse_success_rate(
                generated_texts, languages=generated_languages,
            )

        # Embedding health
        if all_survivors:
            combined_survivors = torch.cat(all_survivors, dim=0)
            token_emb = (
                self.encoder.compressor.backbone.get_input_embeddings().weight.detach()
            )
            health = embedding_drift_metrics(combined_survivors, token_emb)
            gen_metrics["mean_max_cosine_sim"] = health["mean_max_cosine_sim"]
            gen_metrics["std_max_cosine_sim"] = health["std_max_cosine_sim"]

        return gen_metrics

    # ------------------------------------------------------------------
    # BaseTrainer hooks
    # ------------------------------------------------------------------

    def _pre_step_hook(self) -> None:
        """Check curriculum transitions before each training step."""
        self._maybe_end_projection_only()
        self._maybe_unfreeze_encoder()
        self._maybe_introduce_compression()

    def _post_optimizer_step(self, step: int) -> None:
        """Advance bidi warmup; run dual-ascent θ + EMA μ updates per optimizer step.

        Token-budget batching gives microbatches with variable valid/controllable
        counts, so θ and μ MUST be updated using the true global mean across
        all microbatches in this optimizer step (NOT mean of per-microbatch
        means). State was accumulated by ``accumulate(self._surv_state_l0,
        enc_out)`` in ``_forward_backward``.
        """
        if not self._encoder_frozen:
            self.encoder.step_bidi_warmup()

        if self._compression_active and not self._encoder_frozen:
            from bgkit.training.survivorship_helpers import (
                apply_post_step_updates,
                init_state,
                maybe_unload_ice,
            )

            state_l0 = getattr(self, "_surv_state_l0", None)
            if state_l0 is not None:
                update_metrics = apply_post_step_updates(
                    self.encoder.compressor,
                    state_l0,
                    target_ratio=None,
                    level="l0",
                )
                self._last_post_step_metrics = update_metrics
                # Reset the L0 state for the next optimizer step.
                self._surv_state_l0 = init_state()

            # Unload ICE once warmup ends across all levels (idempotent).
            unloaded = maybe_unload_ice(
                getattr(self, "_ice_teacher", None),
                step,
                getattr(self, "_max_warmup_step", 0),
            )
            if unloaded:
                logger.info("ice_teacher_unloaded", step=step)

    def _add_step_metrics(self, metrics: dict[str, float]) -> None:
        """Add curriculum-specific metrics to step logging."""
        metrics["encoder_frozen"] = float(self._encoder_frozen)
        metrics["compression_active"] = float(self._compression_active)
        if not self._encoder_frozen:
            metrics["bidi_alpha"] = self._get_bidi_alpha()
        if self._compression_active:
            metrics["target_ratio"] = self._current_target_ratio()
        if self._last_sampled_target_ratio is not None:
            metrics["sampled_target_ratio"] = self._last_sampled_target_ratio
        post = getattr(self, "_last_post_step_metrics", None)
        if post is not None:
            # Merge without clobbering base metrics (grad_norm, loss, etc).
            for k, v in post.items():
                metrics.setdefault(k, v)

    def _get_bidi_alpha(self) -> float:
        """Get the current bidirectional warmup alpha from the encoder."""
        from bgkit.models.bidirectional_qwen35 import BidirectionalQwen35

        for module in self.encoder.modules():
            if isinstance(module, BidirectionalQwen35):
                return module.bidi_alpha
        return 1.0

    def _build_training_state(
        self,
        es_best: float | None,
        es_evals_without_improvement: int,
        wandb_run,
    ) -> dict:
        """Build training state dict with curriculum fields."""
        state = super()._build_training_state(
            es_best, es_evals_without_improvement, wandb_run,
        )
        state.update({
            "decoder_frozen": self._decoder_frozen,
            "encoder_frozen": self._encoder_frozen,
            "compression_active": self._compression_active,
            "target_ratio_override": self._target_ratio_override,
            # Round-trip the ratio-sampling RNG so resume continues the
            # same pseudo-random sequence of requested ratios.
            # ``random.Random.getstate()`` returns a tuple that's safely
            # pickled by torch.save.
            "target_ratio_rng_state": self._target_ratio_rng.getstate(),
        })
        return state

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self, checkpoint_dir: Path, metrics: dict[str, float] | None = None
    ) -> Path:
        """Save both BgKIT encoder and decoder models."""
        if self._training_state is None:
            self._training_state = {}
        self._training_state["decoder_frozen"] = getattr(self, "_decoder_frozen", False)
        self._training_state["encoder_frozen"] = getattr(self, "_encoder_frozen", True)
        self._training_state["compression_active"] = getattr(
            self, "_compression_active", False,
        )
        self._training_state["target_ratio_override"] = getattr(
            self, "_target_ratio_override", None,
        )

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
        # For LoRA: also save a merged decoder state dict so downstream steps
        # can load it as a plain decoder without the PeftModel wrapper.
        if getattr(self, "_decoder_lora", False):
            save_kwargs["decoder_merged"] = self.decoder.merge_lora()

        ckpt_path = save_checkpoint(checkpoint_dir, metadata, **save_kwargs)
        self._last_checkpoint_path = str(ckpt_path)
        return ckpt_path

    def _named_parameters_for_optimizer(self):
        """Yield (name, param) pairs across encoder, decoder, projection_block.

        Names are stable module paths within each top-level component; the
        top-level prefix disambiguates the three modules. Matches the set
        of params ``_build_decoder_param_groups`` /
        ``_build_encoder_param_groups`` / ``_build_projection_param_groups``
        feed to the optimizer. Required by BaseTrainer's name-keyed
        optimizer state save/load so Step 3 can survive LoRA toggling or
        freeze-flag changes between save and load.
        """
        for name, param in self.encoder.named_parameters():
            yield f"encoder.{name}", param
        for name, param in self.decoder.named_parameters():
            yield f"decoder.{name}", param
        # projection_block lives under encoder; already covered above.
        # Listed here only for trainers where it becomes a separate module.

    def _restore_model_state(self, state_dicts: dict) -> None:
        """Restore encoder (with legacy-key tolerance) + decoder."""
        if "encoder" not in state_dicts:
            raise ValueError(
                f"Resume checkpoint missing 'encoder' key. Found: {list(state_dicts.keys())}"
            )
        from bgkit.models.encoder import migrate_legacy_threshold_controller_state_dict

        enc_state = migrate_legacy_threshold_controller_state_dict(
            state_dicts["encoder"],
            self.encoder,
        )
        result = self.encoder.load_state_dict(enc_state, strict=False)
        if result.missing_keys:
            logger.info(
                "encoder_missing_keys",
                keys=result.missing_keys,
                hint="Expected for new survivorship head components",
            )
        self.decoder.load_state_dict(state_dicts["decoder"])

    def _restore_training_state(self, training_state: dict) -> None:
        self._target_ratio_override = training_state.get("target_ratio_override")
        rng_state = training_state.get("target_ratio_rng_state")
        if rng_state is not None:
            try:
                self._target_ratio_rng.setstate(tuple(rng_state))
            except (TypeError, ValueError):
                # Incompatible state shape (e.g. Python random internals
                # changed across versions). Warn and keep the freshly
                # seeded RNG rather than crashing the resume.
                logger.warning(
                    "target_ratio_rng_state_restore_failed",
                    hint="Keeping freshly seeded RNG — ratio sampling sequence "
                         "will diverge from the pre-restart run",
                )

    def _post_weight_load_hook(self) -> None:
        # Recompute freeze state + rebuild optimizer from restored global_step.
        self._configure_trainable_state()
