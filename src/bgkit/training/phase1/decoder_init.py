"""Phase 1, Step 1: Decoder initialization with encoder unfreeze + compression curriculum.

Trains the reconstruction decoder to generate text from BgKIT's output
representations, using Qwen3's native chat template with tool-call format.

Training progresses through four curriculum phases:
1. Projection-only warmup (decoder frozen)
2. Decoder + projection training (encoder frozen, no compression)
3. Encoder unfreeze with bidirectional warmup (no compression yet)
4. Compression introduction with ICE survivor selection + ratio ramp

Loss is computed only on the file content tokens inside the chat template's
code fence (via loss_mask), so the decoder learns pure reconstruction.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import structlog
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer

from bgkit.data.collators import collate_chat_repro
from bgkit.data.datasets.chat_repro_dataset import ChatReproDataset
from bgkit.data.datasets.mmap_token_dataset import MmapTokenDataset
from bgkit.data.samplers import TokenBudgetBatchSampler
from bgkit.data.survivor_selection import fill_survivor_gaps, select_survivors_by_threshold
from bgkit.data.threshold_calibrator import ThresholdCalibrator
from bgkit.eval.metrics.embedding_health import embedding_drift_metrics
from bgkit.eval.metrics.reconstruction import parse_success_rate
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder, _expand_survivor_mask
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpoint_registry import resolve_checkpoint
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.training.gradient_utils import enable_gradient_checkpointing
from bgkit.training.objectives.data_reconstruction import data_reconstruction_loss
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

    LIVE_CONFIG_FIELDS = {
        "max_survivor_gap": "_max_gap",
        "target_ratio_ramp_steps": "_target_ratio_ramp_steps",
        "target_ratio_start": "_target_ratio_start",
        "target_ratio_end": "_target_ratio_end",
        "compression_introduction_step": "_compression_introduction_step",
        "encoder_unfreeze_step": "_encoder_unfreeze_step",
    }

    def setup(self) -> None:
        """Load BgKIT encoder, decoder, optional ICE model, and configure curriculum."""
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
        self.encoder.to(device)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

        # Projection block training flags
        self._train_projection = tcfg.get("train_projection_block", True)
        self._projection_only_steps = tcfg.get("projection_only_steps", 0)

        # Load BgKIT from checkpoint if available.
        bgkit_checkpoint = self._resolve_bgkit_checkpoint()
        if bgkit_checkpoint is not None:
            logger.info("loading_bgkit_checkpoint", path=bgkit_checkpoint)
            _, state_dicts = load_checkpoint(Path(bgkit_checkpoint))
            if "encoder" not in state_dicts:
                raise ValueError(
                    f"bgkit_checkpoint {bgkit_checkpoint} missing 'encoder' key. "
                    f"Found keys: {list(state_dicts.keys())}"
                )
            self.encoder.load_state_dict(state_dicts["encoder"])

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
        )
        self.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=hidden_dim)
        self.decoder.to(device)

        enable_gradient_checkpointing(self.decoder.backbone)

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

        # --- Optional ICE model (for compression curriculum) ---
        self.ice_model = None
        if self._compression_introduction_step is not None:
            self._load_ice_model(tcfg)

        # --- Compression curriculum state ---
        self._init_curriculum_state(tcfg)

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
    # ICE + curriculum initialization
    # ------------------------------------------------------------------

    def _load_ice_model(self, tcfg) -> None:
        """Load frozen ICE model for compression scoring."""
        from bgkit.models.ice import ICE

        ice_cfg = tcfg.get("ice", {})
        if not ice_cfg.get("checkpoint_path"):
            raise ValueError(
                "compression_introduction_step is set but training.ice.checkpoint_path "
                "is missing. Set it to a trained ICE checkpoint or 'auto'."
            )

        if ice_cfg.checkpoint_path == "auto":
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            ice_ckpt_path = resolve_checkpoint(
                checkpoint_dir,
                phase="ice",
                metric="eval/mse",
                label="training.ice.checkpoint_path",
            )
            self._input_sources["ice"] = ice_ckpt_path.name
            logger.info("ice_auto_resolved", checkpoint=ice_ckpt_path.name)
        else:
            ice_ckpt_path = Path(ice_cfg.checkpoint_path)
            self._input_sources["ice"] = ice_ckpt_path.name

        if not ice_ckpt_path.exists():
            raise FileNotFoundError(
                f"ICE checkpoint not found: {ice_ckpt_path}\n"
                "Set training.ice.checkpoint_path to a valid ICE checkpoint."
            )

        self.ice_model = ICE(
            input_dim=ice_cfg.get("input_dim", 1024),
            hidden_dim=ice_cfg.get("hidden_dim", 128),
            num_layers=ice_cfg.get("num_layers", 2),
            kernel_size=ice_cfg.get("kernel_size", 5),
        )
        _, ice_state_dicts = load_checkpoint(ice_ckpt_path)
        self.ice_model.load_state_dict(ice_state_dicts["model"])
        self.ice_model.to(self.device)
        self.ice_model.eval()
        self.ice_model.requires_grad_(False)
        logger.info("loaded_ice_model", path=str(ice_ckpt_path))

    def _init_curriculum_state(self, tcfg) -> None:
        """Initialize compression curriculum state variables."""
        curriculum = tcfg.get("curriculum", {})

        # Ratio targeting
        self._target_ratio_start = curriculum.get("target_ratio_start", 0.50)
        self._target_ratio_end = curriculum.get("target_ratio_end", 0.25)
        self._target_ratio_ramp_steps = curriculum.get("target_ratio_ramp_steps", 40000)
        self._target_ratio_override: float | None = None
        self._max_gap = curriculum.get("max_survivor_gap", 64)

        # Calibrator
        fallback = curriculum.get("fallback_threshold", 3.0)
        self._calibrator = ThresholdCalibrator(
            ema_decay=curriculum.get("calibrator_ema_decay", 0.99),
            warmup_batches=curriculum.get("calibrator_warmup_batches", 50),
            fallback_threshold=fallback,
        )
        self._pending_scores: list[torch.Tensor] = []

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
        """Resolve bgkit_checkpoint: 'auto' -> best joint_block_pretrain."""
        bgkit_checkpoint = self.cfg.get("bgkit_checkpoint", None)
        if bgkit_checkpoint == "auto":
            checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
            resolved = resolve_checkpoint(
                checkpoint_dir,
                phase="joint_block_pretrain",
                metric="eval/mse_repro",
                label="bgkit_checkpoint",
            )
            bgkit_checkpoint = str(resolved)

        # Track lineage for BOTH auto-resolved and explicit paths
        self._input_sources = {}
        if bgkit_checkpoint is not None:
            self._input_sources["joint_block_pretrain"] = Path(bgkit_checkpoint).name

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
        """Start using ICE survivor selection when the curriculum step is reached."""
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
            return [
                {
                    "params": list(self.decoder.parameters()),
                    "lr": decoder_lr,
                    "base_lr": decoder_lr,
                }
            ]

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
        """Return param IDs of embedding/lm_head — 2D but should not use Muon."""
        exclude = set()
        for p in self.decoder.backbone.model.embed_tokens.parameters():
            exclude.add(id(p))
        for p in self.decoder.backbone.lm_head.parameters():
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
    # ICE scoring + compression
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

    def _score_and_select(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score embeddings with ICE and select survivors via calibrated threshold.

        During training, each sample in the batch gets a random compression
        ratio drawn uniformly between the current min target ratio (from the
        ramp) and 1.0, so the network trains on diverse retention levels.
        During evaluation, the min target ratio is used for all samples.

        Args:
            embeddings: (B, L, D) embeddings to score.
            attention_mask: (B, L) bool mask for valid positions.

        Returns:
            (B, L) bool survivor mask.
        """
        batch_size, seq_len, _ = embeddings.shape

        with torch.no_grad():
            ice_scores = self.ice_model(embeddings.float())  # (B, L)
        ice_scores = ice_scores.masked_fill(~attention_mask, float("-inf"))
        if seq_len > 0:
            ice_scores[:, 0] = float("-inf")  # position 0 unsupervised during ICE training

        # Buffer scores for calibrator update (training only)
        if not self._is_evaluating:
            stats_mask = attention_mask.clone()
            if seq_len > 0:
                stats_mask[:, 0] = False
            valid_scores_flat = ice_scores[stats_mask].detach()
            self._pending_scores.append(valid_scores_flat)

        min_target_ratio = self._current_target_ratio()

        survivor_mask = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=embeddings.device,
        )
        for b in range(batch_size):
            valid_len = int(attention_mask[b].sum().item())
            if valid_len == 0:
                continue

            # Random ratio per sample during training; fixed min during eval
            # Random ratio per sample during training; fixed min during eval
            if self._is_evaluating:
                ratio = min_target_ratio
            else:
                ratio = random.uniform(min_target_ratio, 1.0)

            threshold = self._calibrator.get_threshold(ratio)
            valid_scores = ice_scores[b, :valid_len]
            indices = select_survivors_by_threshold(valid_scores, threshold)
            indices = fill_survivor_gaps(indices, valid_scores, self._max_gap, valid_len)
            survivor_mask[b, indices] = True

        return survivor_mask

    def _flush_calibrator_scores(self) -> None:
        """Flush buffered ICE scores into calibrator. Called once per micro-batch."""
        if self._pending_scores:
            non_empty = [s for s in self._pending_scores if s.numel() > 0]
            if non_empty:
                combined = torch.cat(non_empty)
                self._calibrator.update_from_flat(combined)
            self._pending_scores.clear()

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _compute_survivors(
        self, batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run content through BgKIT encoder and return (embeddings, attention_mask).

        Handles all curriculum phases:
        - Encoder frozen, no compression: projection-only gradients
        - Encoder unfrozen, no compression: full encoder gradients
        - Encoder unfrozen, compression: ICE scoring + survivor selection
        """
        content_token_ids = batch["content_token_ids"].to(self.device)
        content_attention_mask = batch["content_attention_mask"].to(self.device)
        compression_prompt_ids = batch["compression_prompt_ids"].to(self.device)
        compression_prompt_mask = batch["compression_prompt_mask"].to(self.device)

        if self._encoder_frozen:
            # Compressor is frozen; only projection block gets gradients
            with torch.no_grad():
                bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()
                content_emb = bgkit_embed(content_token_ids)
                prompt_emb = bgkit_embed(compression_prompt_ids)

                # ICE scoring on raw embeddings (outside compressor)
                if self._compression_active:
                    survivor_mask = self._score_and_select(
                        content_emb, content_attention_mask,
                    )
                else:
                    survivor_mask = None

                comp_out = self.encoder.compressor(
                    content_emb,
                    survivor_mask=survivor_mask,
                    attention_mask=content_attention_mask,
                    prompt_embeddings=prompt_emb,
                    prompt_attention_mask=compression_prompt_mask,
                )

            if self._train_projection:
                # Detach compressor output so gradients don't flow back
                full_raw = comp_out.raw_embeddings.detach()
                full_mask = comp_out.attention_mask

                # Expand content-only survivor mask to full sequence
                # (prompt+sep positions marked as doomed)
                if survivor_mask is not None:
                    full_survivor_mask = _expand_survivor_mask(
                        survivor_mask, comp_out.content_slice, full_raw.size(1),
                    )
                else:
                    full_survivor_mask = None

                # Projection block outside no_grad (gets gradients)
                proj_out = self.encoder.projection_block(
                    full_raw, full_mask, survivor_mask=full_survivor_mask,
                )

                if self._compression_active:
                    return proj_out.projected_embeddings, proj_out.survivor_attention_mask
                else:
                    # Slice to content-only
                    content_proj = proj_out.projected_embeddings[:, comp_out.content_slice, :]
                    return content_proj, content_attention_mask
            else:
                # Entire encoder under no_grad (rare fallback)
                enc_out = self.encoder(
                    input_embeddings=content_emb,
                    survivor_mask=survivor_mask,
                    attention_mask=content_attention_mask,
                    prompt_embeddings=prompt_emb,
                    prompt_attention_mask=compression_prompt_mask,
                )
                if self._compression_active:
                    return enc_out.survivor_embeddings, enc_out.survivor_attention_mask
                else:
                    return enc_out.survivor_embeddings, content_attention_mask
        else:
            # Encoder is trainable (compressor + projection)
            bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()
            content_emb = bgkit_embed(content_token_ids)
            prompt_emb = bgkit_embed(compression_prompt_ids)

            if self._compression_active:
                survivor_mask = self._score_and_select(content_emb, content_attention_mask)
            else:
                survivor_mask = None

            enc_out = self.encoder(
                input_embeddings=content_emb,
                survivor_mask=survivor_mask,
                attention_mask=content_attention_mask,
                prompt_embeddings=prompt_emb,
                prompt_attention_mask=compression_prompt_mask,
            )

            if self._compression_active:
                return enc_out.survivor_embeddings, enc_out.survivor_attention_mask
            else:
                return enc_out.survivor_embeddings, content_attention_mask

    def _forward_backward(self, batch) -> dict[str, float]:
        self.decoder.train()
        if self._train_projection:
            self.encoder.projection_block.train()
        if not self._encoder_frozen:
            self.encoder.compressor.train()

        token_ids = batch["token_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        loss_mask = batch["loss_mask"].to(self.device)

        if self._encoder_frozen and not self._train_projection:
            with torch.no_grad():
                survivors, survivor_mask = self._compute_survivors(batch)
        else:
            survivors, survivor_mask = self._compute_survivors(batch)

        # BF16 autocast for decoder forward + backward
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            logits = self.decoder(
                survivor_embeddings=survivors,
                target_ids=token_ids,
                target_attention_mask=attention_mask,
                survivor_attention_mask=survivor_mask,
            )
            loss = data_reconstruction_loss(
                logits, token_ids, attention_mask, loss_mask=loss_mask,
            )

        # Scaled backward (for gradient accumulation)
        (loss / self._accum_steps).backward()

        # Flush calibrator scores after backward
        if self._compression_active:
            self._flush_calibrator_scores()

        metrics: dict[str, float] = {"loss": loss.item()}
        if self._compression_active:
            content_mask = batch["content_attention_mask"].to(self.device)
            n_survivors = int(survivor_mask.sum().item())
            n_valid = int(content_mask.sum().item())
            metrics["min_target_ratio"] = self._current_target_ratio()
            metrics["actual_ratio"] = n_survivors / max(n_valid, 1)

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
                token_ids = batch["token_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)

                survivors, survivor_mask = self._compute_survivors(batch)

                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"
                ):
                    logits = self.decoder(
                        survivor_embeddings=survivors,
                        target_ids=token_ids,
                        target_attention_mask=attention_mask,
                        survivor_attention_mask=survivor_mask,
                    )
                    loss = data_reconstruction_loss(
                        logits, token_ids, attention_mask, loss_mask=loss_mask,
                    )

                batch_content_tokens = loss_mask[:, 1:].sum().item()
                total_loss += loss.item() * batch_content_tokens
                total_content_tokens += batch_content_tokens

                if self._compression_active:
                    total_survivors += int(survivor_mask.sum().item())
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

                    survivors, survivor_mask = self._compute_survivors(batch)
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"
                    ):
                        logits = self.decoder(
                            survivor_embeddings=survivors,
                            target_ids=token_ids,
                            target_attention_mask=attention_mask,
                            survivor_attention_mask=survivor_mask,
                        )
                        loss = data_reconstruction_loss(
                            logits, token_ids, attention_mask, loss_mask=loss_mask,
                        )
                    bt = loss_mask[:, 1:].sum().item()
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

            survivors, survivor_mask = self._compute_survivors(batch)

            # Collect survivors for embedding health (subsample to ~512 vectors)
            total_vecs = sum(s.size(0) for s in all_survivors)
            if total_vecs < 512:
                flat = survivors[survivor_mask].detach()
                remaining = 512 - total_vecs
                all_survivors.append(flat[:remaining])

            # Generate
            gen_output = self.decoder.generate(
                survivor_embeddings=survivors,
                survivor_attention_mask=survivor_mask,
                prefix_ids=prefix_ids,
                prefix_attention_mask=prefix_attention_mask,
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
            metrics["calibrated_threshold"] = self._calibrator.get_threshold(
                self._current_target_ratio()
            )

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
        """Save both BgKIT encoder and decoder models, plus calibrator state."""
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
        # Save calibrator state for resume
        if self._compression_active or self._compression_introduction_step is not None:
            save_kwargs["l0_calibrator"] = self._calibrator.state_dict()

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
        self.encoder.load_state_dict(state_dicts["encoder"])
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

        # Restore calibrator state
        if "l0_calibrator" in state_dicts:
            self._calibrator.load_state_dict(state_dicts["l0_calibrator"])

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
