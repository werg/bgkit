"""Phase 2 knowledge-retrieval trainer.

Replaces the previous token-subsampling stand-in with the real BgKIT encoder +
ICE pipeline.  Supports:
- Live L0 (Step 1): encoder runs forward, ICE scores, survivors selected
- Frozen L0 (Step 2): encoder runs but L0 compressor is frozen per config
- Pre-computed L0 (Steps 3-4): survivors loaded from sharded mmap cache
- Optional L1 cross-document compression
- Optional reconstruction regularizer
- Topic embedding conditioning
- Per-dataset evaluation metrics
"""

from __future__ import annotations

import math
from pathlib import Path

import structlog
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split

from bgkit.data.collators import collate_qa
from bgkit.data.datasets.git_history_dataset import GitHistoryDataset
from bgkit.data.datasets.kilt_dataset import KILTDataset
from bgkit.data.datasets.memory_dataset import MemoryDataset
from bgkit.data.datasets.msmarco_dataset import MSMARCODataset
from bgkit.data.datasets.narrativeqa_dataset import NarrativeQADataset
from bgkit.data.datasets.newsqa_dataset import NewsQADataset
from bgkit.data.datasets.precomputed_l0_cache import PrecomputedL0Cache
from bgkit.data.datasets.pubmedqa_dataset import PubMedQADataset
from bgkit.data.datasets.searchqa_dataset import SearchQADataset
from bgkit.data.samplers import QueryAwareBatchSampler, TokenBudgetBatchSampler
from bgkit.data.taxonomy import TagTaxonomy
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.topic_embeddings import TopicEmbeddingModule
from bgkit.training.base_trainer import BaseTrainer

logger = structlog.get_logger()

_DATASET_REGISTRY = {
    "pubmedqa": PubMedQADataset,
    "newsqa": NewsQADataset,
    "searchqa": SearchQADataset,
    "msmarco_passage": MSMARCODataset,
    "narrativeqa": NarrativeQADataset,
    "kilt": KILTDataset,
    "git_history": GitHistoryDataset,
    "memory": MemoryDataset,
    "msc": MemoryDataset,
    "share": MemoryDataset,
    "chronicles": MemoryDataset,
    "perltqa": MemoryDataset,
    "laps": MemoryDataset,
}


class _Phase2Model(nn.Module):
    """Container module for checkpointing Phase 2 trainable state."""

    def __init__(
        self,
        decoder: ReconstructionDecoder,
        encoder: nn.Module | None = None,
        ice: nn.Module | None = None,
        topic_embeddings: TopicEmbeddingModule | None = None,
        cache_projection: nn.Module | None = None,
        projection_block: nn.Module | None = None,
    ):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder
        self.ice = ice
        self.topic_embeddings = topic_embeddings
        self.cache_projection = cache_projection
        self.projection_block = projection_block


class KRTrainer(BaseTrainer):
    """Phase 2 trainer using BgKIT encoder pipeline over QA datasets."""

    _log_every = 5

    # Ablation modes for eval scripts.  Set via set_ablation_mode() rather
    # than monkey-patching _compose_prompt.
    ABLATION_NONE = None           # Normal operation
    ABLATION_ZEROED = "zeroed"     # Zero all compressed survivors
    ABLATION_NOISE = "noise"       # Replace survivors with random noise
    ABLATION_NO_TOPICS = "no_topics"        # Drop topic embeddings only
    ABLATION_TOPICS_ONLY = "topics_only"    # Drop compressed context, keep topics
    ABLATION_NEITHER = "neither"            # Drop both

    def __init__(self, cfg):
        super().__init__(cfg)
        self.step_cfg = self._resolve_step_cfg()
        self._l0_cache: PrecomputedL0Cache | None = None
        self._cache_projection: nn.Module | None = None
        self.topic_embeddings: TopicEmbeddingModule | None = None
        self.taxonomy: TagTaxonomy | None = None
        self.encoder = None
        self.ice = None
        self._l0_calibrator = None
        self._reconstruction_weight = float(
            self.step_cfg.get("reconstruction_weight", 0.0),
        )
        self._ablation_mode: str | None = None

    def _resolve_step_cfg(self):
        training_cfg = self.cfg.training
        step_name = training_cfg.get("step")
        if step_name:
            self.step_name = str(step_name)
            return training_cfg

        if "step1" in training_cfg and isinstance(training_cfg.step1, dict):
            raise ValueError(
                "Phase 2 requires `training.step` to select a stage. "
                "Use configs/training/phase2_step*.yaml or set training.step manually.",
            )
        self.step_name = "phase2"
        return training_cfg

    def _resolve_decoder_backbone_name(self) -> str:
        if self.step_name == "step5":
            target_lm = self.cfg.model.get("target_lm", {})
            return target_lm.get("backbone_name", self.cfg.model.decoder.backbone_name)
        return self.cfg.model.decoder.backbone_name

    def _resolve_dataset_dir(self, name: str) -> str:
        data_cfg = self.cfg.get("data", {})
        if name in data_cfg:
            entry = data_cfg[name]
            if isinstance(entry, dict):
                for key in ("mmap_dir", "data_dir", "path"):
                    if key in entry:
                        return str(entry[key])
        data_root = Path(self.cfg.get("data_dir", Path.home() / "data"))
        return str(data_root / "mmap" / "phase2" / name)

    def _build_dataset(self, name: str):
        try:
            dataset_cls = _DATASET_REGISTRY[name]
        except KeyError as exc:
            raise KeyError(f"Unknown Phase 2 dataset {name!r}") from exc
        kwargs = {}
        if dataset_cls is MemoryDataset:
            kwargs["dataset_name"] = name
        return dataset_cls(self._resolve_dataset_dir(name), **kwargs)

    def _load_taxonomy(self) -> TagTaxonomy | None:
        topic_cfg = self.step_cfg.get("topic_embeddings", {})
        taxonomy_path = topic_cfg.get("taxonomy_path")
        if not topic_cfg.get("enabled", False) or not taxonomy_path:
            return None
        path = Path(str(taxonomy_path))
        if not path.exists():
            logger.warning("topic_taxonomy_missing", path=str(path))
            return None
        return TagTaxonomy.load(path)

    def _use_precomputed_l0(self) -> bool:
        return bool(self.step_cfg.get("use_precomputed_l0", False))

    def _use_live_l0(self) -> bool:
        return not self._use_precomputed_l0() and self.encoder is not None

    def _load_encoder_and_ice(self) -> None:
        """Load BgKIT encoder and ICE from Phase 1 checkpoint."""
        from bgkit.models.encoder import BgKITEncoder
        from bgkit.models.ice import ICE

        phase1_ckpt_path = self.step_cfg.get("phase1_checkpoint")
        if not phase1_ckpt_path:
            logger.info("phase2_no_encoder", msg="No Phase 1 checkpoint; using prompt compressor")
            return

        phase1_ckpt_path = str(phase1_ckpt_path)
        if phase1_ckpt_path == "auto":
            from bgkit.training.checkpoint_registry import resolve_checkpoint

            checkpoint_dir = Path(str(self.cfg.get("checkpoint_dir", "checkpoints")))
            try:
                resolved = resolve_checkpoint(
                    checkpoint_dir,
                    phase="phase1_step5",
                    metric="eval/loss",
                    lower_is_better=True,
                )
                phase1_ckpt_path = str(resolved)
            except ValueError:
                logger.warning("phase2_auto_resolve_failed")
                return

        logger.info("phase2_loading_encoder", checkpoint=phase1_ckpt_path)

        from bgkit.training.checkpointing import load_checkpoint

        _metadata, state_dicts = load_checkpoint(Path(phase1_ckpt_path))

        # load_checkpoint returns {name: state_dict} for each *.pt file
        model_state = state_dicts.get("model", {})
        encoder_state = {
            k.replace("encoder.", "", 1): v
            for k, v in model_state.items() if k.startswith("encoder.")
        }

        encoder_cfg = self.cfg.model.get("encoder", {})
        backbone_name = encoder_cfg.get("backbone_name", "Qwen/Qwen3.5-0.8B-Base")
        hidden_dim = int(encoder_cfg.get("hidden_dim", 1024))

        if encoder_state:
            self.encoder = BgKITEncoder.from_pretrained_with_state_dict(
                backbone_name, encoder_state, hidden_dim=hidden_dim,
            )
        else:
            self.encoder = BgKITEncoder.from_pretrained(
                backbone_name, hidden_dim=hidden_dim,
            )

        self.encoder.to(self.device)

        # Load ICE
        ice_state = {
            k.replace("ice.", "", 1): v
            for k, v in model_state.items() if k.startswith("ice.")
        }
        ice_cfg = self.cfg.model.get("ice", {})
        self.ice = ICE(
            input_dim=int(ice_cfg.get("input_dim", 1024)),
            hidden_dim=int(ice_cfg.get("hidden_dim", 128)),
            num_layers=int(ice_cfg.get("num_layers", 3)),
        )
        if ice_state:
            self.ice.load_state_dict(ice_state, strict=False)
        self.ice.to(self.device)
        self.ice.eval()
        self.ice.requires_grad_(False)

        # Apply freeze config to encoder
        freeze_cfg = self.step_cfg.get("freeze", {})
        if freeze_cfg.get("l0_compressor", False):
            if hasattr(self.encoder, "compressor"):
                self.encoder.compressor.requires_grad_(False)
                logger.info("phase2_freeze_l0_compressor")
        if freeze_cfg.get("projection_block", False):
            if hasattr(self.encoder, "projection_block"):
                self.encoder.projection_block.requires_grad_(False)
                logger.info("phase2_freeze_projection_block")
        if freeze_cfg.get("encoder", False):
            self.encoder.requires_grad_(False)
            logger.info("phase2_freeze_encoder")

    def setup(self) -> None:
        from transformers import AutoModelForCausalLM

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load decoder
        decoder_name = self._resolve_decoder_backbone_name()
        decoder_backbone = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        )
        decoder_backbone.to(self.device)
        self.decoder = ReconstructionDecoder(
            decoder_backbone,
            hidden_dim=decoder_backbone.get_input_embeddings().weight.shape[1],
        )
        self.decoder.train()
        self._apply_freeze_config()

        # Load encoder + ICE from Phase 1 checkpoint (if available)
        if not self._use_precomputed_l0():
            self._load_encoder_and_ice()

        # Topic embeddings
        self.taxonomy = self._load_taxonomy()
        if self.taxonomy is not None:
            topic_cfg = self.step_cfg.get("topic_embeddings", {})
            self.topic_embeddings = TopicEmbeddingModule(
                self.taxonomy,
                positions_per_tag=int(topic_cfg.get("positions_per_tag", 8)),
                hidden_dim=int(self.decoder.hidden_dim),
            ).to(self.device)

        # Pre-computed L0 cache
        cache_dir = self.step_cfg.get("l0_cache_dir")
        if cache_dir:
            cache_path = Path(str(cache_dir))
            if cache_path.exists():
                self._l0_cache = PrecomputedL0Cache(str(cache_path))
                logger.info("phase2_l0_cache_loaded", entries=len(self._l0_cache))
            else:
                logger.warning("phase2_l0_cache_missing", path=str(cache_path))

        # Build model container
        self.model = _Phase2Model(
            decoder=self.decoder,
            encoder=self.encoder,
            ice=self.ice,
            topic_embeddings=self.topic_embeddings,
            cache_projection=self._cache_projection,
            projection_block=(
                self.encoder.projection_block
                if self.encoder and hasattr(self.encoder, "projection_block")
                else None
            ),
        )
        self.model.to(self.device)

        # Build datasets
        dataset_names = list(self.step_cfg.get("datasets", []))
        if not dataset_names:
            raise ValueError(f"No datasets configured for Phase 2 step {self.step_name}")

        # KILT tasks have no inline documents — require precomputed L0
        _provenance_only = {"kilt", "kilt_nq", "kilt_hotpotqa", "kilt_fever", "kilt_zsre",
                            "kilt_trex", "kilt_wow", "kilt_eli5", "kilt_aidayago2",
                            "kilt_wned", "kilt_cweb", "kilt_triviaqa"}
        if not self._use_precomputed_l0() and _provenance_only & set(dataset_names):
            raise ValueError(
                "KILT task datasets have no inline documents and require "
                "use_precomputed_l0: true with a populated l0_cache_dir. "
                "Set these in your training config."
            )

        datasets = [self._build_dataset(name) for name in dataset_names]
        dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

        total = len(dataset)
        eval_size = min(max(1, int(total * 0.1)), int(self.step_cfg.get("max_eval_samples", 1024)))
        train_size = total - eval_size
        if train_size < 1:
            raise ValueError("Phase 2 dataset too small for train/eval split")
        generator = torch.Generator().manual_seed(int(self.cfg.get("seed", 42)))
        self.train_dataset, self.eval_dataset = random_split(
            dataset,
            [train_size, eval_size],
            generator=generator,
        )

        max_batch_tokens = int(self.step_cfg.get("max_batch_tokens", 16384))
        batch_size = int(self.cfg.get("batch_size", 4))
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self._build_sampler(
                self.train_dataset, max_batch_tokens, batch_size, True,
            ),
            collate_fn=collate_qa,
            num_workers=int(self.cfg.compute.get("num_workers", 0)),
            pin_memory=bool(self.cfg.compute.get("pin_memory", False)),
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_sampler=self._build_sampler(
                self.eval_dataset, max_batch_tokens, batch_size, False,
            ),
            collate_fn=collate_qa,
            num_workers=int(self.cfg.compute.get("num_workers", 0)),
            pin_memory=bool(self.cfg.compute.get("pin_memory", False)),
        )

        # Optimizer
        param_groups = self._build_optimizer_groups()
        if not param_groups:
            raise ValueError("Phase 2 trainer has no trainable parameters after freeze config")
        self.optimizer = self._create_optimizer(
            param_groups,
            default_lr=float(self.step_cfg.get("lr", 1.0e-4)),
        )

    def _build_optimizer_groups(self) -> list[dict]:
        """Build optimizer param groups for all trainable components."""
        groups = []

        # Decoder params
        decoder_params = [p for p in self.decoder.parameters() if p.requires_grad]
        if decoder_params:
            groups.append({
                "params": decoder_params,
                "lr": float(self.step_cfg.get("decoder_lr", self.step_cfg.get("lr", 1.0e-4))),
            })

        # Encoder params (if trainable)
        if self.encoder is not None:
            encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
            if encoder_params:
                groups.append({
                    "params": encoder_params,
                    "lr": float(
                        self.step_cfg.get("encoder_lr", self.step_cfg.get("lr", 1.0e-4)),
                    ),
                })

        # Topic embedding params (per-tag LR scaling)
        if self.topic_embeddings is not None:
            groups.extend(
                self.topic_embeddings.get_optimizer_groups(
                    float(self.step_cfg.get("lr", 1.0e-4)),
                ),
            )

        # Cache projection params
        if self._cache_projection is not None:
            proj_params = [p for p in self._cache_projection.parameters() if p.requires_grad]
            if proj_params:
                groups.append({
                    "params": proj_params,
                    "lr": float(self.step_cfg.get("lr", 1.0e-4)),
                })

        return groups

    def _apply_freeze_config(self) -> None:
        freeze_cfg = self.step_cfg.get("freeze", {})
        if freeze_cfg.get("decoder", False):
            self.decoder.requires_grad_(False)
        if freeze_cfg.get("target_lm_base", False):
            self.decoder.backbone.requires_grad_(False)

    def _build_sampler(self, dataset, max_batch_tokens: int, batch_size: int, shuffle: bool):
        inner = getattr(dataset, "dataset", dataset)
        if hasattr(inner, "metadata") and self.step_name in {"step2", "step3", "step4"}:
            return QueryAwareBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)

        lengths = []
        for idx in dataset.indices if isinstance(dataset, Subset) else range(len(dataset)):
            sample = inner[idx]
            lengths.append(
                int(
                    sample.content_token_ids.size(0)
                    + sample.question_token_ids.size(0)
                    + sample.answer_token_ids.size(0)
                ),
            )
        return TokenBudgetBatchSampler(lengths, max_batch_tokens=max_batch_tokens, shuffle=shuffle)

    def _target_ratio(self) -> float:
        curriculum = self.step_cfg.get("curriculum", {})
        start = float(curriculum.get("target_ratio_start", 0.10))
        end = float(curriculum.get("target_ratio_end", start))
        ramp = max(int(curriculum.get("target_ratio_ramp_steps", 1)), 1)
        progress = min(1.0, self.global_step / ramp)
        return start + (end - start) * progress

    def _encode_live(
        self,
        content_ids: torch.Tensor,
        content_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run live L0 encoding through BgKIT encoder + ICE."""

        # Get embeddings from encoder's embedding layer
        embed_tokens = self.encoder.compressor.backbone.get_input_embeddings()
        input_embeddings = embed_tokens(content_ids)

        # Score with ICE to determine survivors
        with torch.no_grad():
            ice_scores = self.ice(input_embeddings)  # (B, L)

        # Select survivors based on target ratio
        ratio = self._target_ratio()
        batch_size, seq_len = content_ids.shape
        survivor_masks = torch.zeros_like(content_mask, dtype=torch.bool)

        for i in range(batch_size):
            length = int(content_mask[i].sum())
            keep = max(1, math.ceil(length * ratio))
            scores_i = ice_scores[i, :length]
            _, topk_indices = torch.topk(scores_i, min(keep, length))
            survivor_masks[i, topk_indices] = True

        # Run encoder with survivor masks
        output = self.encoder(
            input_embeddings=input_embeddings,
            survivor_mask=survivor_masks,
            attention_mask=content_mask,
        )

        return output.survivor_embeddings, output.survivor_attention_mask

    def _cached_survivors(self, batch) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._l0_cache is None:
            return None
        document_ids = [doc_id for doc_id in batch["document_ids"] if doc_id is not None]
        if len(document_ids) != len(batch["document_ids"]):
            return None
        survivors, mask = self._l0_cache.get_survivors_batch(document_ids, self._target_ratio())
        survivors = survivors.to(self.device)
        mask = mask.to(self.device)
        if survivors.size(-1) != self.decoder.hidden_dim:
            if self._cache_projection is None:
                self._cache_projection = nn.Linear(survivors.size(-1), self.decoder.hidden_dim).to(
                    self.device,
                )
                self.model.cache_projection = self._cache_projection
            survivors = self._cache_projection(survivors)
        return survivors, mask

    def set_ablation_mode(self, mode: str | None) -> None:
        """Set the ablation mode for eval.  Use class constants (ABLATION_*)."""
        self._ablation_mode = mode

    def _compose_prompt(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the decoder prompt from compressed survivors + topic embeddings.

        Respects ``_ablation_mode`` for eval ablation studies — no need to
        monkey-patch this method from external scripts.
        """
        skip_context = self._ablation_mode in (
            self.ABLATION_TOPICS_ONLY, self.ABLATION_NEITHER,
        )
        skip_topics = self._ablation_mode in (
            self.ABLATION_NO_TOPICS, self.ABLATION_NEITHER,
        )

        # --- compressed context ---
        if skip_context:
            # Need a zero-length prompt so shape is valid; we'll skip concat
            prompt = None
            mask = None
        else:
            cached = self._cached_survivors(batch)
            if cached is not None:
                prompt, mask = cached
            elif self._use_live_l0():
                prompt, mask = self._encode_live(
                    batch["content_token_ids"].to(self.device),
                    batch["content_attention_mask"].to(self.device),
                )
            else:
                prompt, mask = self._subsample_embeddings(
                    batch["content_token_ids"].to(self.device),
                    batch["content_attention_mask"].to(self.device),
                )

            # Apply ablation transforms to compressed context
            if self._ablation_mode == self.ABLATION_ZEROED and prompt is not None:
                prompt = torch.zeros_like(prompt)
            elif self._ablation_mode == self.ABLATION_NOISE and prompt is not None:
                prompt = torch.randn_like(prompt) * 0.02

        # --- topic embeddings ---
        if not skip_topics and self.topic_embeddings is not None:
            topic_prompt, topic_mask = self.topic_embeddings(batch["tags"])
            if topic_prompt.size(1) > 0:
                if prompt is not None:
                    prompt = torch.cat([prompt, topic_prompt], dim=1)
                    mask = torch.cat([mask, topic_mask], dim=1)
                else:
                    prompt = topic_prompt
                    mask = topic_mask

        # If both context and topics were skipped, return a minimal empty prompt
        if prompt is None:
            batch_size = batch["content_token_ids"].size(0)
            prompt = torch.zeros(
                batch_size, 1, self.decoder.hidden_dim,
                device=self.device, dtype=torch.bfloat16,
            )
            mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device)

        return prompt, mask

    def _subsample_embeddings(
        self,
        content_token_ids: torch.Tensor,
        content_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fallback: subsample token embeddings when no encoder is available."""
        embed_tokens = self.decoder.backbone.get_input_embeddings()
        embedded = embed_tokens(content_token_ids)
        batch, _, hidden = embedded.shape
        ratio = self._target_ratio()
        lengths = content_attention_mask.sum(dim=1).tolist()
        max_keep = max(1, max(math.ceil(length * ratio) for length in lengths))
        prompts = embedded.new_zeros((batch, max_keep, hidden))
        prompt_mask = torch.zeros(batch, max_keep, dtype=torch.bool, device=embedded.device)

        for row, length in enumerate(lengths):
            length = int(length)
            keep = max(1, math.ceil(length * ratio))
            if length == 1:
                indices = torch.zeros(1, dtype=torch.long, device=embedded.device)
            else:
                indices = torch.linspace(
                    0, length - 1, steps=keep, device=embedded.device,
                ).round().long()
            selected = embedded[row, indices]
            prompts[row, :keep] = selected
            prompt_mask[row, :keep] = True
        return prompts, prompt_mask

    def _forward_backward(self, batch) -> dict[str, float]:
        prompt, prompt_mask = self._compose_prompt(batch)
        target_ids = batch["target_token_ids"].to(self.device)
        target_attention_mask = batch["target_attention_mask"].to(self.device)
        target_loss_mask = batch["target_loss_mask"].to(self.device)

        # QA loss
        loss = self.decoder.forward_with_loss(
            prompt,
            target_ids,
            target_attention_mask,
            prompt_mask,
            loss_mask=target_loss_mask,
        )

        # Optional reconstruction regularizer
        repro_loss = 0.0
        if self._reconstruction_weight > 0 and self._use_live_l0():
            content_ids = batch["content_token_ids"].to(self.device)
            content_mask = batch["content_attention_mask"].to(self.device)
            repro = self.decoder.forward_with_loss(
                prompt, content_ids, content_mask, prompt_mask,
            )
            repro_loss = repro.item()
            loss = loss + self._reconstruction_weight * repro

        (loss / self._accum_steps).backward()
        return {
            "loss": loss.detach(),
            "repro_loss": repro_loss,
            "prompt_tokens": float(prompt_mask.sum().item() / prompt_mask.size(0)),
            "target_ratio": self._target_ratio(),
        }

    def _add_step_metrics(self, metrics: dict[str, float]) -> None:
        """Add per-tag embedding norms for topic embedding monitoring."""
        if self.topic_embeddings is not None:
            norms = {}
            for tag in self.topic_embeddings.taxonomy.tags[:20]:  # Top 20 tags
                key = self.topic_embeddings._key(tag)
                if key in self.topic_embeddings.embeddings:
                    norms[f"topic_norm/{tag}"] = (
                        self.topic_embeddings.embeddings[key].data.norm().item()
                    )
            metrics.update(norms)

    def _get_eval_tokenizer(self):
        """Lazily load and cache the tokenizer for eval decoding."""
        if not hasattr(self, "_eval_tokenizer") or self._eval_tokenizer is None:
            from transformers import AutoTokenizer

            decoder_name = self._resolve_decoder_backbone_name()
            self._eval_tokenizer = AutoTokenizer.from_pretrained(
                decoder_name, trust_remote_code=True,
            )
        return self._eval_tokenizer

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        from bgkit.eval.metrics.qa_metrics import exact_match, token_f1

        self.model.eval()
        tokenizer = self._get_eval_tokenizer()

        total_loss = 0.0
        total_batches = 0
        total_answer_tokens = 0
        total_correct = 0

        # Per-dataset accumulators for token accuracy
        per_dataset_correct: dict[str, int] = {}
        per_dataset_total: dict[str, int] = {}

        # Per-dataset accumulators for decoded-text QA metrics
        per_dataset_em: dict[str, list[float]] = {}
        per_dataset_f1: dict[str, list[float]] = {}
        all_em: list[float] = []
        all_f1: list[float] = []

        for batch in self.eval_dataloader:
            prompt, prompt_mask = self._compose_prompt(batch)
            target_ids = batch["target_token_ids"].to(self.device)
            target_attention_mask = batch["target_attention_mask"].to(self.device)
            target_loss_mask = batch["target_loss_mask"].to(self.device)
            answer_token_ids = batch["answer_token_ids"]  # (B, answer_len), stays on CPU
            answer_attention_mask = batch["answer_attention_mask"]  # (B, answer_len)

            loss = self.decoder.forward_with_loss(
                prompt,
                target_ids,
                target_attention_mask,
                prompt_mask,
                loss_mask=target_loss_mask,
            )
            logits = self.decoder(
                prompt,
                target_ids,
                target_attention_mask,
                prompt_mask,
            )
            shifted_logits = logits[:, :-1]
            shifted_labels = target_ids[:, 1:]
            shifted_mask = target_loss_mask[:, 1:]
            predictions = shifted_logits.argmax(dim=-1)
            correct = ((predictions == shifted_labels) & shifted_mask).sum().item()
            answer_tokens = shifted_mask.sum().item()
            total_correct += correct
            total_answer_tokens += answer_tokens
            total_loss += float(loss.item())
            total_batches += 1

            # Decode gold answer text from answer_token_ids
            batch_size = answer_token_ids.size(0)
            for i in range(batch_size):
                # Gold answer: decode from answer_token_ids (unpadded via mask)
                ans_mask_i = answer_attention_mask[i]
                ans_len = int(ans_mask_i.sum().item())
                gold_ids = answer_token_ids[i, :ans_len]
                gold_text = tokenizer.decode(gold_ids, skip_special_tokens=True)

                # Predicted answer: extract answer-region predictions from shifted logits
                # target_loss_mask marks which positions are answer tokens.
                # shifted_mask[i] aligns with shifted_labels (target_ids[:, 1:]).
                mask_i = shifted_mask[i]  # (shifted_len,)
                pred_i = predictions[i]   # (shifted_len,)
                # Extract predicted token IDs where loss mask is True
                answer_pred_ids = pred_i[mask_i.bool()].cpu()
                pred_text = tokenizer.decode(answer_pred_ids, skip_special_tokens=True)

                # Compute QA metrics (exact_match and token_f1 expect list of references)
                em_score = exact_match(pred_text, [gold_text])
                f1_score = token_f1(pred_text, [gold_text])
                all_em.append(em_score)
                all_f1.append(f1_score)

                # Per-dataset QA metrics
                ds_name = None
                dataset_names = batch.get("dataset_names", [])
                if i < len(dataset_names) and dataset_names[i] is not None:
                    ds_name = str(dataset_names[i])

                if ds_name is not None:
                    per_dataset_em.setdefault(ds_name, []).append(em_score)
                    per_dataset_f1.setdefault(ds_name, []).append(f1_score)

            # Per-dataset token accuracy tracking
            for i, ds_name in enumerate(batch.get("dataset_names", [])):
                if ds_name is None:
                    continue
                ds_name = str(ds_name)
                mask_i = shifted_mask[i]
                pred_i = predictions[i]
                label_i = shifted_labels[i]
                ds_correct = ((pred_i == label_i) & mask_i).sum().item()
                ds_total = mask_i.sum().item()
                per_dataset_correct[ds_name] = per_dataset_correct.get(ds_name, 0) + ds_correct
                per_dataset_total[ds_name] = per_dataset_total.get(ds_name, 0) + ds_total

        self.model.train()
        if total_batches == 0:
            return {
                "eval/loss": 0.0,
                "eval/answer_token_accuracy": 0.0,
                "eval/exact_match": 0.0,
                "eval/token_f1": 0.0,
            }

        metrics = {
            "eval/loss": total_loss / total_batches,
            "eval/answer_token_accuracy": (
                total_correct / total_answer_tokens if total_answer_tokens else 0.0
            ),
            "eval/exact_match": sum(all_em) / len(all_em) if all_em else 0.0,
            "eval/token_f1": sum(all_f1) / len(all_f1) if all_f1 else 0.0,
        }

        # Add per-dataset token accuracy
        for ds_name in per_dataset_correct:
            ds_total = per_dataset_total.get(ds_name, 0)
            if ds_total > 0:
                metrics[f"eval/{ds_name}/token_accuracy"] = (
                    per_dataset_correct[ds_name] / ds_total
                )

        # Add per-dataset QA metrics
        for ds_name, scores in per_dataset_em.items():
            if scores:
                metrics[f"eval/{ds_name}/exact_match"] = sum(scores) / len(scores)
        for ds_name, scores in per_dataset_f1.items():
            if scores:
                metrics[f"eval/{ds_name}/token_f1"] = sum(scores) / len(scores)

        return metrics
