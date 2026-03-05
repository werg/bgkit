"""Phase 1, Step 1: Decoder initialization on uncompressed output.

Train the reconstruction decoder to generate text from BgKIT's full
(uncompressed) output representations, using Qwen3's native chat template
with tool-call format for in-distribution agentic conversation.

The BgKIT encoder runs the full compressor + projection block pipeline
without compression (survivor_mask=None). Prompt conditioning uses the
encoder's forward method which handles prompt prepending and content
slicing internally.

Loss is computed only on the file content tokens inside the chat template's
code fence (via loss_mask), so the decoder learns pure reconstruction.
"""

from __future__ import annotations

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
from bgkit.eval.metrics.embedding_health import embedding_drift_metrics
from bgkit.eval.metrics.reconstruction import parse_success_rate
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.checkpoint_registry import resolve_checkpoint
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.training.gradient_utils import enable_gradient_checkpointing
from bgkit.training.objectives.data_reconstruction import data_reconstruction_loss
from bgkit.training.scheduling import cosine_with_warmup

logger = structlog.get_logger()


class DecoderInitTrainer(BaseTrainer):
    """Step 1: Initialize decoder on uncompressed BgKIT output."""

    def setup(self) -> None:
        """Load frozen BgKIT encoder and trainable decoder."""
        tcfg = self.cfg.training

        # Config validation
        if tcfg.get("projection_only_steps", 0) > 0 and not tcfg.get(
            "train_projection_block", True
        ):
            raise ValueError("projection_only_steps > 0 requires train_projection_block: true")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # --- BgKIT encoder (frozen) ---
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
            bidi_warmup_steps=0,  # encoder is frozen in Step 1
        )
        self.encoder.to(device)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

        # Projection block training flags
        self._train_projection = tcfg.get("train_projection_block", True)
        self._projection_only_steps = tcfg.get("projection_only_steps", 0)

        # Load BgKIT from checkpoint if available.
        # Joint block pretrain checkpoints save under key "encoder";
        # legacy auto-repro checkpoints save under "model".
        bgkit_checkpoint = self._resolve_bgkit_checkpoint()
        if bgkit_checkpoint is not None:
            logger.info("loading_bgkit_checkpoint", path=bgkit_checkpoint)
            _, state_dicts = load_checkpoint(Path(bgkit_checkpoint))
            if "encoder" in state_dicts:
                self.encoder.load_state_dict(state_dicts["encoder"])
            elif "model" in state_dicts:
                # Legacy auto-repro checkpoint -- load into compressor only
                logger.info("loading_legacy_auto_repro_checkpoint")
                self.encoder.compressor.load_state_dict(state_dicts["model"], strict=False)

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

        # torch.compile: ~15-30% speedup on Blackwell. If compile fails on sm_121,
        # remove this line -- training is correct without it.
        try:
            self.decoder.backbone = torch.compile(self.decoder.backbone)
            logger.info("torch_compile_enabled")
        except Exception:
            logger.warning("torch_compile_failed", exc_info=True)

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

        # --- Dataset ---
        data_dir = self.cfg.data.tokens.input_dir
        max_seq_len = self.cfg.data.tokens.get("max_seq_len", 8192)
        variant_bank_path = self.cfg.data.tokens.variant_bank_path

        inner_dataset = MmapTokenDataset(data_dir, max_seq_len=max_seq_len)
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

        train_lengths = full_dataset.lengths[np.array(self.train_dataset.indices)]
        eval_lengths = full_dataset.lengths[np.array(self.eval_dataset.indices)]

        self.train_sampler = TokenBudgetBatchSampler(
            train_lengths, max_batch_tokens, shuffle=True, seed=self.cfg.get("seed", 42),
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

        # --- Optimizer (with projection-aware freeze/unfreeze) ---
        self._configure_trainable_state()

        logger.info(
            "decoder_init_trainer_setup",
            train_samples=train_size,
            eval_samples=eval_size,
            device=str(device),
        )

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
            self._apply_freeze()  # re-freeze top layer, lm_head, embed_tokens

        # Build optimizer matching current trainable params
        self._setup_optimizer()

        # Log for verification
        proj_params = sum(
            p.numel() for p in self.encoder.projection_block.parameters() if p.requires_grad
        )
        dec_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        logger.info(
            "trainable_state_configured",
            step=self.global_step,
            decoder_frozen=self._decoder_frozen,
            projection_params=proj_params,
            decoder_params=dec_params,
        )

    def _build_decoder_param_groups(self) -> list[dict]:
        """Build decoder param groups (with differential LR if freeze_top_layer)."""
        tcfg = self.cfg.training
        if tcfg.get("freeze_top_layer", False):
            return self._build_param_groups(tcfg.lr, tcfg.get("lr_scale_bottom", 0.1))
        else:
            return [
                {"params": list(self.decoder.parameters()), "lr": tcfg.lr, "base_lr": tcfg.lr}
            ]

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

        self.optimizer = self._create_optimizer(
            param_groups, tcfg.lr, exclude_from_muon=self._muon_excluded_param_ids()
        )

    def _build_param_groups(self, base_lr: float, lr_scale_bottom: float) -> list[dict]:
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

    def _maybe_end_projection_only(self) -> None:
        """Transition from projection-only warmup to full training."""
        if not self._decoder_frozen:
            return
        if self.global_step < self._projection_only_steps:
            return

        # Unfreeze decoder, re-apply intentional freezes
        self.decoder.requires_grad_(True)
        self._apply_freeze()  # re-freeze top layer, lm_head, embed_tokens
        self._decoder_frozen = False

        # Add decoder param groups to existing optimizer (preserves projection momentum)
        decoder_groups = self._build_decoder_param_groups()
        for group in decoder_groups:
            self._add_param_group_to_optimizer(group)

        # Sync LR schedule for new groups
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

    def trainable_parameters(self) -> list:
        params = [p for p in self.decoder.parameters() if p.requires_grad]
        if self._train_projection:
            params += [
                p for p in self.encoder.projection_block.parameters() if p.requires_grad
            ]
        return params

    def _forward_backward(self, batch) -> dict[str, float]:
        self.decoder.train()
        if self._train_projection:
            self.encoder.projection_block.train()

        # Phase transition check (before anything else)
        self._maybe_end_projection_only()

        token_ids = batch["token_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        loss_mask = batch["loss_mask"].to(self.device)
        content_attention_mask = batch["content_attention_mask"].to(self.device)

        if self._train_projection:
            survivors = self._compute_survivors(batch)  # no outer no_grad
        else:
            with torch.no_grad():
                survivors = self._compute_survivors(batch)

        # BF16 autocast for decoder forward + backward
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            logits = self.decoder(
                survivor_embeddings=survivors,
                target_ids=token_ids,
                target_attention_mask=attention_mask,
                survivor_attention_mask=content_attention_mask,
            )
            loss = data_reconstruction_loss(
                logits, token_ids, attention_mask, loss_mask=loss_mask,
            )

        # Scaled backward (for gradient accumulation)
        (loss / self._accum_steps).backward()

        return {
            "loss": loss.item(),
        }

    def _compute_survivors(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run content through BgKIT encoder and return output embeddings.

        When projection block is trainable, only the compressor runs under
        no_grad; the projection block gets gradients. When projection is
        frozen, the entire encoder runs under no_grad (caller wraps).
        """
        content_token_ids = batch["content_token_ids"].to(self.device)
        content_attention_mask = batch["content_attention_mask"].to(self.device)
        compression_prompt_ids = batch["compression_prompt_ids"].to(self.device)
        compression_prompt_mask = batch["compression_prompt_mask"].to(self.device)

        if self._train_projection:
            # Split: compressor under no_grad, projection block gets gradients
            with torch.no_grad():
                bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()
                content_emb = bgkit_embed(content_token_ids)
                prompt_emb = bgkit_embed(compression_prompt_ids)

                comp_out = self.encoder.compressor(
                    content_emb,
                    survivor_mask=None,
                    attention_mask=content_attention_mask,
                    prompt_embeddings=prompt_emb,
                    prompt_attention_mask=compression_prompt_mask,
                )

            # Detach compressor output so gradients don't flow back
            full_raw = comp_out.raw_embeddings.detach()
            full_mask = comp_out.attention_mask

            # Projection block outside no_grad (gets gradients)
            proj_out = self.encoder.projection_block(full_raw, full_mask, survivor_mask=None)

            # Slice to content-only (replicate BgKITEncoder.forward no-compression path)
            content_proj = proj_out.projected_embeddings[:, comp_out.content_slice, :]
            return content_proj
        else:
            # Original path: entire encoder under no_grad (caller wraps)
            bgkit_embed = self.encoder.compressor.backbone.get_input_embeddings()
            content_emb = bgkit_embed(content_token_ids)
            prompt_emb = bgkit_embed(compression_prompt_ids)

            enc_out = self.encoder(
                input_embeddings=content_emb,
                survivor_mask=None,
                attention_mask=content_attention_mask,
                prompt_embeddings=prompt_emb,
                prompt_attention_mask=compression_prompt_mask,
            )
            return enc_out.survivor_embeddings

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.decoder.eval()
        if self._train_projection:
            self.encoder.projection_block.eval()
        self._eval_count += 1

        total_loss = 0.0
        total_content_tokens = 0.0

        num_batches = len(self.eval_dataloader)
        for batch_idx, batch in enumerate(self.eval_dataloader):
            if batch_idx % 100 == 0:
                logger.info("eval_progress", batch=batch_idx, total=num_batches)
            token_ids = batch["token_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            loss_mask = batch["loss_mask"].to(self.device)
            content_attention_mask = batch["content_attention_mask"].to(self.device)

            survivors = self._compute_survivors(batch)

            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"
            ):
                logits = self.decoder(
                    survivor_embeddings=survivors,
                    target_ids=token_ids,
                    target_attention_mask=attention_mask,
                    survivor_attention_mask=content_attention_mask,
                )
                loss = data_reconstruction_loss(
                    logits, token_ids, attention_mask, loss_mask=loss_mask,
                )

            # Weight by content tokens only (loss_mask[:, 1:] for shift alignment)
            batch_content_tokens = loss_mask[:, 1:].sum().item()
            total_loss += loss.item() * batch_content_tokens
            total_content_tokens += batch_content_tokens

        avg_loss = total_loss / max(total_content_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        metrics: dict[str, float] = {
            "loss": avg_loss,
            "perplexity": perplexity,
        }

        # Generation metrics (expensive -- only every Nth eval)
        tcfg = self.cfg.training
        gen_every = tcfg.get("eval", {}).get("generation_eval_every", 4)
        if self._eval_count % gen_every == 0:
            gen_metrics = self._run_generation_eval()
            metrics.update(gen_metrics)

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

            content_attention_mask = batch["content_attention_mask"].to(self.device)
            prefix_ids = batch["prefix_ids"].to(self.device)
            prefix_attention_mask = batch["prefix_attention_mask"].to(self.device)

            survivors = self._compute_survivors(batch)

            # Collect survivors for embedding health (subsample to ~512 vectors)
            total_vecs = sum(s.size(0) for s in all_survivors)
            if total_vecs < 512:
                flat = survivors[content_attention_mask].detach()
                remaining = 512 - total_vecs
                all_survivors.append(flat[:remaining])

            # Generate
            gen_output = self.decoder.generate(
                survivor_embeddings=survivors,
                survivor_attention_mask=content_attention_mask,
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

    def save_checkpoint(
        self, checkpoint_dir: Path, metrics: dict[str, float] | None = None
    ) -> Path:
        """Save both BgKIT encoder and decoder models."""
        # Inject phase state into training_state
        if self._training_state is None:
            self._training_state = {}
        self._training_state["decoder_frozen"] = getattr(self, "_decoder_frozen", False)

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
            decoder=self.decoder.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )
        self._last_checkpoint_path = str(ckpt_path)
        return ckpt_path

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load both BgKIT encoder and decoder models."""
        metadata, state_dicts = load_checkpoint(checkpoint_path)
        self._check_optimizer_type_compat(metadata)

        # Restore model weights
        if "encoder" in state_dicts:
            self.encoder.load_state_dict(state_dicts["encoder"])
        elif "bgkit_model" in state_dicts:
            # Legacy checkpoint -- load into compressor only
            self.encoder.compressor.load_state_dict(state_dicts["bgkit_model"], strict=False)
        self.decoder.load_state_dict(state_dicts["decoder"])

        # Restore step position
        self.global_step = metadata.step
        self.epoch = metadata.epoch
        self._last_checkpoint_path = str(checkpoint_path)
        if metadata.schedule_params is not None:
            self._schedule_params = metadata.schedule_params
        if metadata.training_state is not None:
            self._training_state = metadata.training_state

        # Sanity-check: log if saved phase state doesn't match derived state
        if metadata.training_state:
            saved_frozen = metadata.training_state.get("decoder_frozen")
            expected_frozen = (
                self._projection_only_steps > 0
                and self.global_step < self._projection_only_steps
            )
            if saved_frozen is not None and saved_frozen != expected_frozen:
                logger.warning(
                    "phase_state_mismatch",
                    saved=saved_frozen,
                    derived=expected_frozen,
                    step=self.global_step,
                )

        # Recompute freeze state + rebuild optimizer from restored global_step
        self._configure_trainable_state()

        # Load optimizer state. If config topology changed between runs,
        # param-group count won't match — warn and skip optimizer restore.
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
