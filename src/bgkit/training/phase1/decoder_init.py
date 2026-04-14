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
from bgkit.data.samplers import TokenBudgetBatchSampler
from bgkit.eval.metrics.embedding_health import embedding_drift_metrics
from bgkit.eval.metrics.reconstruction import parse_success_rate
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpoint_registry import resolve_checkpoint
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.training.gradient_utils import enable_gradient_checkpointing
from bgkit.training.scheduling import cosine_with_warmup

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

    def __next__(self):
        total = self._primary_samples + self._secondary_samples + 1
        current_ratio = self._secondary_samples / total

        if current_ratio < self._ratio:
            try:
                batch = next(self._secondary_iter)
                self._secondary_samples += batch["token_ids"].size(0)
                return batch
            except StopIteration:
                # Secondary exhausted — restart it
                self._secondary_iter = iter(self._secondary_loader)
                # Fall through to primary

        # Primary batch (or secondary was exhausted)
        batch = next(self._primary_iter)  # StopIteration propagates = epoch end
        self._primary_samples += batch["token_ids"].size(0)
        return batch


class DecoderInitTrainer(BaseTrainer):
    """Step 1: Decoder init with encoder unfreeze + compression curriculum."""

    LIVE_CONFIG_FIELDS: ClassVar[dict[str, str]] = {
        "target_ratio_ramp_steps": "_target_ratio_ramp_steps",
        "target_ratio_start": "_target_ratio_start",
        "target_ratio_end": "_target_ratio_end",
        "compression_introduction_step": "_compression_introduction_step",
        "encoder_unfreeze_step": "_encoder_unfreeze_step",
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
                attn_implementation="sdpa",
                bidi_warmup_steps=bidi_warmup,
            )
            # Also load decoder weights if present (step 2 checkpoints include it)
            self._bgkit_decoder_state = bgkit_state_dicts.get("decoder", None)
        else:
            logger.info("loading_bgkit_encoder", model=backbone_name, revision=backbone_revision)
            self.encoder = BgKITEncoder.from_pretrained(
                backbone_name,
                hidden_dim=hidden_dim,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                revision=backbone_revision,
                attn_implementation="sdpa",
                bidi_warmup_steps=bidi_warmup,
            )
            self._bgkit_decoder_state = None
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
            attn_implementation="sdpa",
            device_map=device,  # load directly to CUDA, avoid CPU staging copy
        )
        self.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=hidden_dim)

        # Load decoder weights from bgkit checkpoint if available (step 2 includes decoder)
        if self._bgkit_decoder_state is not None:
            self.decoder.load_state_dict(self._bgkit_decoder_state)
            del self._bgkit_decoder_state
            logger.info("loaded_decoder_from_bgkit_checkpoint")

        # Optional LoRA wrapping for decoder (step 3 uses this for parameter-efficient adaptation)
        self._decoder_lora = False
        lora_cfg = tcfg.get("decoder_lora", {})
        if lora_cfg.get("enabled", False):
            self.decoder.apply_lora(lora_cfg)
            self._decoder_lora = True

        enable_gradient_checkpointing(self.decoder.backbone)

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

        # --- Survivorship head config ---
        surv_cfg = tcfg.get("survivorship", {})
        self._ratio_loss_weight = surv_cfg.get("ratio_loss_weight", 0.1)
        self._decisiveness_loss_weight = surv_cfg.get("decisiveness_loss_weight", 0.05)
        self._soft_attn_loss_weight = surv_cfg.get("soft_attn_loss_weight", 0.05)
        self._soft_attn_every_n_steps = surv_cfg.get("soft_attn_every_n_steps", 4)

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
        num_workers = self.cfg.compute.get("num_workers", 4)
        pin_memory = self.cfg.compute.get("pin_memory", False)
        seed = self.cfg.get("seed", 42)

        train_lengths = full_dataset.lengths[np.array(self.train_dataset.indices)]
        eval_lengths = full_dataset.lengths[np.array(self.eval_dataset.indices)]

        self.train_sampler = TokenBudgetBatchSampler(
            train_lengths, max_batch_tokens, shuffle=True, seed=seed,
        )
        eval_sampler = TokenBudgetBatchSampler(
            eval_lengths, max_batch_tokens, shuffle=False,
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

                    qa_train_sampler = TokenBudgetBatchSampler(
                        qa_train_lengths, max_batch_tokens, shuffle=True, seed=seed,
                    )
                    qa_eval_sampler = TokenBudgetBatchSampler(
                        qa_eval_lengths, max_batch_tokens, shuffle=False,
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
    # Curriculum initialization
    # ------------------------------------------------------------------

    def _init_curriculum_state(self, tcfg) -> None:
        """Initialize compression curriculum state variables."""
        # Ratio targeting — read from top-level config keys
        self._target_ratio_start = tcfg.get("target_ratio_start", 0.10)
        self._target_ratio_end = tcfg.get("target_ratio_end", 0.10)
        self._target_ratio_ramp_steps = tcfg.get("target_ratio_ramp_steps", 1)
        self._target_ratio_override: float | None = None

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

        For phase1_step1: resolves from joint_block_pretrain.
        For phase1_step3: resolves from phase1_step2 (pruned encoder).
        """
        bgkit_checkpoint = self.cfg.get("bgkit_checkpoint", None)
        if bgkit_checkpoint == "auto":
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            phase = self.cfg.training.get("phase", "phase1_step1")
            if phase == "phase1_step3":
                resolved = resolve_checkpoint(
                    checkpoint_dir,
                    phase="phase1_step2",
                    metric="eval/loss",
                    label="bgkit_checkpoint",
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

    # ------------------------------------------------------------------
    # Freeze / unfreeze management
    # ------------------------------------------------------------------

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
            self.decoder.requires_grad_(True)
            self._apply_freeze()

        # Encoder: frozen until encoder_unfreeze_step
        self._encoder_frozen = (
            self._encoder_unfreeze_step is None
            or self.global_step < self._encoder_unfreeze_step
        )
        if not self._encoder_frozen:
            self.encoder.compressor.requires_grad_(True)
            self.encoder.compressor.train()
            enable_gradient_checkpointing(self.encoder.compressor.backbone)
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

        self.decoder.requires_grad_(True)
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
        enable_gradient_checkpointing(self.encoder.compressor.backbone)
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

    def _compute_survivorship_losses(
        self,
        enc_out,
        target_ratio: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute auxiliary survivorship head losses.

        Returns:
            (total_aux_loss, metrics_dict) where total_aux_loss is ready to add
            to the main loss and metrics_dict has per-loss scalars.
        """
        probs = enc_out.survive_probs  # (B, L_content)
        if probs is None:
            return torch.tensor(0.0, device=self.device), {}

        metrics: dict[str, float] = {}

        # Aggregate ratio loss: (mean(probs) - target)^2
        mean_prob = probs.mean()
        ratio_loss = (mean_prob - target_ratio) ** 2
        metrics["ratio_loss"] = ratio_loss.item()
        metrics["mean_survive_prob"] = mean_prob.item()

        # Decisiveness loss: mean(4 * p * (1-p)) — penalizes probs near 0.5
        decisiveness_loss = (4 * probs * (1 - probs)).mean()
        metrics["decisiveness_loss"] = decisiveness_loss.item()

        total = (
            self._ratio_loss_weight * ratio_loss
            + self._decisiveness_loss_weight * decisiveness_loss
        )
        return total, metrics

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _compute_survivors(
        self, batch: dict[str, torch.Tensor],
    ):
        """Run content through BgKIT encoder and return encoder output.

        Returns:
            CompressionOutput from the encoder. Callers use
            ``enc_out.survivor_embeddings`` and ``enc_out.survivor_attention_mask``
            for the decoder, plus ``enc_out.survive_probs`` and
            ``enc_out.layer7_embeddings`` for auxiliary losses.
        """
        content_token_ids = batch["content_token_ids"].to(self.device)
        content_attention_mask = batch["content_attention_mask"].to(self.device)
        compression_prompt_ids = batch["compression_prompt_ids"].to(self.device)
        compression_prompt_mask = batch["compression_prompt_mask"].to(self.device)

        target_ratio = self._current_target_ratio() if self._compression_active else None

        # When the compressor is frozen AND the projection block is NOT
        # trainable, wrap in no_grad to save memory. Otherwise we must run
        # with autograd enabled so gradients can flow through the trainable
        # projection block (frozen compressor params still won't receive
        # gradients because their requires_grad=False).
        use_no_grad = self._encoder_frozen and not self._train_projection

        bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()

        if use_no_grad:
            with torch.no_grad():
                enc_out = self.encoder(
                    input_embeddings=bgkit_embed(content_token_ids),
                    attention_mask=content_attention_mask,
                    prompt_embeddings=bgkit_embed(compression_prompt_ids),
                    prompt_attention_mask=compression_prompt_mask,
                    target_ratio=target_ratio,
                    level="l0",
                )
        else:
            enc_out = self.encoder(
                input_embeddings=bgkit_embed(content_token_ids),
                attention_mask=content_attention_mask,
                prompt_embeddings=bgkit_embed(compression_prompt_ids),
                prompt_attention_mask=compression_prompt_mask,
                target_ratio=target_ratio,
                level="l0",
            )

        return enc_out

    def _forward_backward(self, batch) -> dict[str, float]:
        self.decoder.train()
        if self._train_projection:
            self.encoder.projection_block.train()
        if not self._encoder_frozen:
            self.encoder.compressor.train()

        token_ids = batch["token_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        loss_mask = batch["loss_mask"].to(self.device)
        splice_start = batch["bgkit_splice_start"].to(self.device)
        splice_len = batch["bgkit_splice_len"].to(self.device)

        if self._encoder_frozen and not self._train_projection:
            with torch.no_grad():
                enc_out = self._compute_survivors(batch)
        else:
            enc_out = self._compute_survivors(batch)

        survivors = enc_out.survivor_embeddings
        survivor_mask = enc_out.survivor_attention_mask

        # BF16 autocast for decoder forward + backward
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            loss = self.decoder.forward_with_single_splice(
                survivor_embeddings=survivors,
                survivor_attention_mask=survivor_mask,
                token_ids=token_ids,
                token_attention_mask=attention_mask,
                splice_starts=splice_start,
                splice_lengths=splice_len,
                loss_mask=loss_mask,
            )

        total_loss = loss

        metrics: dict[str, float] = {"loss": loss.item()}

        # Survivorship head auxiliary losses (when compression is active)
        if self._compression_active and enc_out.survive_probs is not None:
            target_ratio = self._current_target_ratio()
            surv_loss, surv_metrics = self._compute_survivorship_losses(enc_out, target_ratio)
            total_loss = total_loss + surv_loss
            metrics.update(surv_metrics)

            # Track actual compression ratio
            if enc_out.survivor_mask is not None:
                content_mask = batch["content_attention_mask"].to(self.device)
                n_survivors = int(enc_out.survivor_mask.sum().item())
                n_valid = int(content_mask.sum().item())
                metrics["min_target_ratio"] = target_ratio
                metrics["actual_ratio"] = n_survivors / max(n_valid, 1)

            # Soft attention branch (every Nth step)
            if (
                self._soft_attn_loss_weight > 0
                and self.global_step % self._soft_attn_every_n_steps == 0
                and enc_out.layer7_embeddings is not None
                and not self._encoder_frozen
            ):
                soft_loss = self._soft_attention_forward(
                    enc_out, batch, token_ids, attention_mask, loss_mask,
                    splice_start, splice_len,
                )
                total_loss = total_loss + self._soft_attn_loss_weight * soft_loss
                metrics["soft_attn_loss"] = soft_loss.item()

        # Scaled backward (for gradient accumulation)
        (total_loss / self._accum_steps).backward()

        return metrics

    def _soft_attention_forward(
        self, enc_out, batch, token_ids, attention_mask, loss_mask,
        splice_start, splice_len,
    ) -> torch.Tensor:
        """Second decoder forward on prob-gated layer-7 embeddings.

        Provides a differentiable gradient path from the decoder loss to
        the survivorship head via survive_probs.
        """
        # Prob-gate the layer-7 embeddings
        gated = enc_out.survive_probs.unsqueeze(-1) * enc_out.layer7_embeddings

        # Use the real content attention mask so padded positions do not
        # contribute noise to the soft attention signal. The projection
        # block treats it as a padding mask; the returned soft_mask then
        # reflects which positions are real content.
        content_attn_mask = batch["content_attention_mask"].to(self.device)

        # Use projection block to get into decoder space
        proj_out = self.encoder.projection_block(
            gated, content_attn_mask, survivor_mask=None,
        )
        soft_survivors = proj_out.projected_embeddings
        soft_mask = content_attn_mask.clone()

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            soft_loss = self.decoder.forward_with_single_splice(
                survivor_embeddings=soft_survivors,
                survivor_attention_mask=soft_mask,
                token_ids=token_ids,
                token_attention_mask=attention_mask,
                splice_starts=splice_start,
                splice_lengths=splice_len,
                loss_mask=loss_mask,
            )
        return soft_loss

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
                token_ids = batch["token_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)
                splice_start = batch["bgkit_splice_start"].to(self.device)
                splice_len = batch["bgkit_splice_len"].to(self.device)

                enc_out = self._compute_survivors(batch)
                survivors = enc_out.survivor_embeddings
                survivor_mask = enc_out.survivor_attention_mask

                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"
                ):
                    loss = self.decoder.forward_with_single_splice(
                        survivor_embeddings=survivors,
                        survivor_attention_mask=survivor_mask,
                        token_ids=token_ids,
                        token_attention_mask=attention_mask,
                        splice_starts=splice_start,
                        splice_lengths=splice_len,
                        loss_mask=loss_mask,
                    )

                batch_content_tokens = loss_mask.sum().item()
                total_loss += loss.item() * batch_content_tokens
                total_content_tokens += batch_content_tokens

                if self._compression_active and enc_out.survivor_mask is not None:
                    total_survivors += int(enc_out.survivor_mask.sum().item())
                    content_mask = batch["content_attention_mask"].to(self.device)
                    total_content += int(content_mask.sum().item())

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
                    token_ids = batch["token_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    loss_mask = batch["loss_mask"].to(self.device)
                    splice_start = batch["bgkit_splice_start"].to(self.device)
                    splice_len = batch["bgkit_splice_len"].to(self.device)

                    enc_out = self._compute_survivors(batch)
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"
                    ):
                        loss = self.decoder.forward_with_single_splice(
                            survivor_embeddings=enc_out.survivor_embeddings,
                            survivor_attention_mask=enc_out.survivor_attention_mask,
                            token_ids=token_ids,
                            token_attention_mask=attention_mask,
                            splice_starts=splice_start,
                            splice_lengths=splice_len,
                            loss_mask=loss_mask,
                        )
                    bt = loss_mask.sum().item()
                    qa_loss += loss.item() * bt
                    qa_tokens += bt

                qa_avg = qa_loss / max(qa_tokens, 1)
                metrics["qa_loss"] = qa_avg
                metrics["qa_perplexity"] = torch.exp(torch.tensor(qa_avg)).item()

                combined_tokens = total_content_tokens + qa_tokens
                if combined_tokens > 0:
                    metrics["combined_loss"] = (
                        total_loss + qa_loss
                    ) / combined_tokens

            # Generation metrics (expensive -- only every Nth eval)
            tcfg = self.cfg.training
            gen_every = tcfg.get("eval", {}).get("generation_eval_every", 4)
            if self._eval_count % gen_every == 0:
                gen_metrics = self._run_generation_eval()
                metrics.update(gen_metrics)

        finally:
            self._is_evaluating = False

        return metrics

    @torch.no_grad()
    def _run_generation_eval(self) -> dict[str, float]:
        """Run generation-based eval metrics on a small subset."""
        tcfg = self.cfg.training
        max_gen_samples = tcfg.get("eval", {}).get("generation_samples", 50)

        gen_metrics: dict[str, float] = {}
        generated_texts: list[str] = []
        generated_languages: list[str] = []
        all_survivors: list[torch.Tensor] = []
        samples_seen = 0

        suffix_ids = self.chat_dataset.suffix_ids.to(self.device)

        for batch in self.eval_dataloader:
            if samples_seen >= max_gen_samples:
                break

            prefix_ids = batch["prefix_ids"].to(self.device)
            prefix_attention_mask = batch["prefix_attention_mask"].to(self.device)

            enc_out = self._compute_survivors(batch)
            survivors = enc_out.survivor_embeddings
            survivor_mask = enc_out.survivor_attention_mask

            # Collect survivors for embedding health (subsample to ~512 vectors)
            total_vecs = sum(s.size(0) for s in all_survivors)
            if total_vecs < 512:
                flat = survivors[survivor_mask].detach()
                remaining = 512 - total_vecs
                all_survivors.append(flat[:remaining])

            # Generate
            gen_output = self.decoder.generate_with_single_splice(
                survivor_embeddings=survivors,
                survivor_attention_mask=survivor_mask,
                prefix_ids=prefix_ids,
                prefix_attention_mask=prefix_attention_mask,
                splice_starts=batch["bgkit_splice_start"].to(self.device),
                splice_lengths=batch["bgkit_splice_len"].to(self.device),
                suffix_ids=suffix_ids,
                tokenizer=self.tokenizer,
                max_new_tokens=2048,
                temperature=0.0,
            )
            generated_texts.extend(gen_output.content_text)
            generated_languages.extend(batch["languages"])
            samples_seen += survivors.size(0)

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
        """Advance bidirectional warmup after each optimizer step (only when unfrozen)."""
        if not self._encoder_frozen:
            self.encoder.step_bidi_warmup()

    def _add_step_metrics(self, metrics: dict[str, float]) -> None:
        """Add curriculum-specific metrics to step logging."""
        metrics["encoder_frozen"] = float(self._encoder_frozen)
        metrics["compression_active"] = float(self._compression_active)
        if not self._encoder_frozen:
            metrics["bidi_alpha"] = self._get_bidi_alpha()
        if self._compression_active:
            metrics["target_ratio"] = self._current_target_ratio()

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
            optimizer=self.optimizer.state_dict(),
        )
        # For LoRA: also save a merged decoder state dict so downstream steps
        # can load it as a plain decoder without the PeftModel wrapper.
        if getattr(self, "_decoder_lora", False):
            save_kwargs["decoder_merged"] = self.decoder.merge_lora()

        ckpt_path = save_checkpoint(checkpoint_dir, metadata, **save_kwargs)
        self._last_checkpoint_path = str(ckpt_path)
        return ckpt_path

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load both BgKIT encoder and decoder models + curriculum state."""
        metadata, state_dicts = load_checkpoint(checkpoint_path)
        self._check_optimizer_type_compat(metadata)

        # Restore model weights
        if "encoder" not in state_dicts:
            raise ValueError(
                f"Resume checkpoint missing 'encoder' key. Found: {list(state_dicts.keys())}"
            )
        result = self.encoder.load_state_dict(state_dicts["encoder"], strict=False)
        if result.missing_keys:
            logger.info(
                "encoder_missing_keys",
                keys=result.missing_keys,
                hint="Expected for new survivorship head components",
            )
        self.decoder.load_state_dict(state_dicts["decoder"])

        # Restore step position
        self.global_step = metadata.step
        self.epoch = metadata.epoch
        self._last_checkpoint_path = str(checkpoint_path)
        if metadata.schedule_params is not None:
            self._schedule_params = metadata.schedule_params
        if metadata.training_state is not None:
            self._training_state = metadata.training_state
            # Restore curriculum overrides
            self._target_ratio_override = metadata.training_state.get(
                "target_ratio_override",
            )

        # Recompute freeze state + rebuild optimizer from restored global_step
        self._configure_trainable_state()

        # Load optimizer state
        if "optimizer" in state_dicts:
            try:
                self.optimizer.load_state_dict(state_dicts["optimizer"])
            except (ValueError, KeyError, RuntimeError) as e:
                logger.warning(
                    "optimizer_state_load_failed",
                    error=str(e),
                    hint="config topology may have changed between runs; "
                    "optimizer state reset, training continues with fresh moments",
                )

        logger.info("restored_from_checkpoint", step=self.global_step)

    # ------------------------------------------------------------------
    # Live config
    # ------------------------------------------------------------------

    def apply_live_config(self, changes: dict) -> None:
        """Support live tuning of target compression ratio + base fields."""
        if "target_ratio" in changes:
            val = changes["target_ratio"]
            if val is None:
                self._target_ratio_override = None
                logger.info("live_target_ratio_cleared", resuming="ramp")
            elif isinstance(val, (int, float)) and 0 < val < 1:
                self._target_ratio_override = float(val)
                logger.info(
                    "live_target_ratio_update", target_ratio=self._target_ratio_override,
                )
        super().apply_live_config(changes)
