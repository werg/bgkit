"""Phase 2 knowledge-retrieval trainer.

Full BgKIT encoder pipeline with hierarchical L0/L1 compression:
- Live L0 (Step 1): encoder runs forward, ICE scores, survivors selected
- Frozen L0 (Step 2): encoder runs but L0 compressor is frozen per config
- Pre-computed L0 (Steps 3-4): survivors loaded from sharded mmap cache
- Query-conditioned L1: L0 survivors re-compressed through the encoder with
  the question as prompt, enabling cross-document attention guided by the query
- Topic embeddings as a separate tool-call region (distinct from corpus encoding)
- Per-dataset QA evaluation metrics (exact_match, token_f1)
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
    """Phase 2 trainer with hierarchical L0/L1 compression."""

    _log_every = 5

    # Ablation modes — use set_ablation_mode() instead of monkey-patching.
    ABLATION_NONE = None
    ABLATION_ZEROED = "zeroed"
    ABLATION_NOISE = "noise"
    ABLATION_NO_TOPICS = "no_topics"
    ABLATION_TOPICS_ONLY = "topics_only"
    ABLATION_NEITHER = "neither"

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
        self._l1_calibrator = None
        self._l1_enabled = False
        self._l1_batches_seen = 0
        self._reconstruction_weight = float(
            self.step_cfg.get("reconstruction_weight", 0.0),
        )
        self._ablation_mode: str | None = None

    # ------------------------------------------------------------------
    # Config resolution
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Encoder / ICE loading
    # ------------------------------------------------------------------

    def _load_encoder_and_ice(self) -> None:
        """Load BgKIT encoder and ICE from Phase 1 checkpoint."""
        from bgkit.models.encoder import BgKITEncoder
        from bgkit.models.ice import ICE

        phase1_ckpt_path = self.step_cfg.get("phase1_checkpoint")
        if not phase1_ckpt_path:
            logger.info(
                "phase2_no_encoder",
                msg="No Phase 1 checkpoint; using prompt compressor",
            )
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

        # Load ICE (shared between L0 and L1 — same model scores both)
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

        # Apply freeze config
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

    # ------------------------------------------------------------------
    # L1 calibrator setup
    # ------------------------------------------------------------------

    def _setup_l1(self) -> None:
        """Initialize L1 calibrator and state."""
        from bgkit.data.threshold_calibrator import ThresholdCalibrator

        l1_cfg = self.step_cfg.get("l1", {})
        self._l1_enabled = bool(l1_cfg.get("enabled", False))

        if self._l1_enabled and self.encoder is not None:
            fallback = float(l1_cfg.get("fallback_threshold", 3.0))
            self._l1_calibrator = ThresholdCalibrator(
                ema_decay=0.95,  # fast initial adaptation
                warmup_batches=int(l1_cfg.get("calibrator_warmup_batches", 50)),
                fallback_threshold=fallback,
            )
            self._l1_calibrator_fast_batches = int(
                l1_cfg.get("calibrator_fast_batches", 200),
            )
            self._l1_calibrator_slow_decay = float(
                l1_cfg.get("calibrator_ema_decay", 0.99),
            )
            logger.info("phase2_l1_enabled")
        else:
            self._l1_calibrator = None

    # ------------------------------------------------------------------
    # setup()
    # ------------------------------------------------------------------

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

        # L1 calibrator (needs encoder + ICE to be loaded first)
        self._setup_l1()

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
            raise ValueError(
                f"No datasets configured for Phase 2 step {self.step_name}",
            )

        # KILT tasks have no inline documents — require precomputed L0
        _provenance_only = {
            "kilt", "kilt_nq", "kilt_hotpotqa", "kilt_fever", "kilt_zsre",
            "kilt_trex", "kilt_wow", "kilt_eli5", "kilt_aidayago2",
            "kilt_wned", "kilt_cweb", "kilt_triviaqa",
        }
        if not self._use_precomputed_l0() and _provenance_only & set(dataset_names):
            raise ValueError(
                "KILT task datasets have no inline documents and require "
                "use_precomputed_l0: true with a populated l0_cache_dir.",
            )

        datasets = [self._build_dataset(name) for name in dataset_names]
        dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

        total = len(dataset)
        eval_size = min(
            max(1, int(total * 0.1)),
            int(self.step_cfg.get("max_eval_samples", 1024)),
        )
        train_size = total - eval_size
        if train_size < 1:
            raise ValueError("Phase 2 dataset too small for train/eval split")
        generator = torch.Generator().manual_seed(int(self.cfg.get("seed", 42)))
        self.train_dataset, self.eval_dataset = random_split(
            dataset, [train_size, eval_size], generator=generator,
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
            raise ValueError(
                "Phase 2 trainer has no trainable parameters after freeze config",
            )
        self.optimizer = self._create_optimizer(
            param_groups,
            default_lr=float(self.step_cfg.get("lr", 1.0e-4)),
        )

    def _build_optimizer_groups(self) -> list[dict]:
        groups = []
        decoder_params = [p for p in self.decoder.parameters() if p.requires_grad]
        if decoder_params:
            groups.append({
                "params": decoder_params,
                "lr": float(
                    self.step_cfg.get("decoder_lr", self.step_cfg.get("lr", 1e-4)),
                ),
            })
        if self.encoder is not None:
            encoder_params = [
                p for p in self.encoder.parameters() if p.requires_grad
            ]
            if encoder_params:
                groups.append({
                    "params": encoder_params,
                    "lr": float(
                        self.step_cfg.get("encoder_lr", self.step_cfg.get("lr", 1e-4)),
                    ),
                })
        if self.topic_embeddings is not None:
            groups.extend(
                self.topic_embeddings.get_optimizer_groups(
                    float(self.step_cfg.get("lr", 1e-4)),
                ),
            )
        if self._cache_projection is not None:
            proj_params = [
                p for p in self._cache_projection.parameters() if p.requires_grad
            ]
            if proj_params:
                groups.append({
                    "params": proj_params,
                    "lr": float(self.step_cfg.get("lr", 1e-4)),
                })
        return groups

    def _apply_freeze_config(self) -> None:
        freeze_cfg = self.step_cfg.get("freeze", {})
        if freeze_cfg.get("decoder", False):
            self.decoder.requires_grad_(False)
        if freeze_cfg.get("target_lm_base", False):
            self.decoder.backbone.requires_grad_(False)

    def _build_sampler(
        self, dataset, max_batch_tokens: int, batch_size: int, shuffle: bool,
    ):
        inner = getattr(dataset, "dataset", dataset)
        if hasattr(inner, "metadata") and self.step_name in {
            "step2", "step3", "step4",
        }:
            return QueryAwareBatchSampler(
                dataset, batch_size=batch_size, shuffle=shuffle,
            )
        lengths = []
        for idx in (
            dataset.indices if isinstance(dataset, Subset) else range(len(dataset))
        ):
            sample = inner[idx]
            lengths.append(int(
                sample.content_token_ids.size(0)
                + sample.question_token_ids.size(0)
                + sample.answer_token_ids.size(0)
            ))
        return TokenBudgetBatchSampler(
            lengths, max_batch_tokens=max_batch_tokens, shuffle=shuffle,
        )

    # ------------------------------------------------------------------
    # Compression curriculum (separate L0 and L1 ratios)
    # ------------------------------------------------------------------

    def _l0_ratio(self) -> float:
        """L0 within-document compression ratio (ramps over training)."""
        curriculum = self.step_cfg.get("curriculum", {})
        start = float(curriculum.get("l0_ratio_start", curriculum.get("target_ratio_start", 0.10)))
        end = float(curriculum.get("l0_ratio_end", curriculum.get("target_ratio_end", start)))
        ramp_raw = curriculum.get(
            "l0_ratio_ramp_steps", curriculum.get("target_ratio_ramp_steps", 1),
        )
        ramp = max(int(ramp_raw), 1)
        progress = min(1.0, self.global_step / ramp)
        return start + (end - start) * progress

    def _l1_ratio(self) -> float:
        """L1 cross-document compression ratio (more generous than L0).

        L1's job is query-conditioned cross-document fusion, not aggressive
        compression.  A typical L1 ratio of 0.50 keeps half the L0 survivors
        after query-guided re-scoring.
        """
        l1_cfg = self.step_cfg.get("l1", {})
        start = float(l1_cfg.get("ratio_start", 0.50))
        end = float(l1_cfg.get("ratio_end", start))
        ramp = max(int(l1_cfg.get("ratio_ramp_steps", 1)), 1)
        progress = min(1.0, self.global_step / ramp)
        return start + (end - start) * progress

    def _target_ratio(self) -> float:
        """Backward-compat alias: returns the L0 ratio."""
        return self._l0_ratio()

    # ------------------------------------------------------------------
    # L0 encoding (live)
    # ------------------------------------------------------------------

    def _encode_live_l0(
        self,
        content_ids: torch.Tensor,
        content_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run live L0 encoding: embed → ICE score → select → encoder."""
        embed_tokens = self.encoder.compressor.backbone.get_input_embeddings()
        input_embeddings = embed_tokens(content_ids)

        with torch.no_grad():
            ice_scores = self.ice(input_embeddings)

        ratio = self._target_ratio()
        batch_size = content_ids.size(0)
        survivor_masks = torch.zeros_like(content_mask, dtype=torch.bool)

        for i in range(batch_size):
            length = int(content_mask[i].sum())
            keep = max(1, math.ceil(length * ratio))
            scores_i = ice_scores[i, :length]
            _, topk_indices = torch.topk(scores_i, min(keep, length))
            survivor_masks[i, topk_indices] = True

        output = self.encoder(
            input_embeddings=input_embeddings,
            survivor_mask=survivor_masks,
            attention_mask=content_mask,
        )
        return output.survivor_embeddings, output.survivor_attention_mask

    # ------------------------------------------------------------------
    # L0 from cache
    # ------------------------------------------------------------------

    def _cached_l0_survivors(
        self, batch,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._l0_cache is None:
            return None
        document_ids = [
            doc_id for doc_id in batch["document_ids"] if doc_id is not None
        ]
        if len(document_ids) != len(batch["document_ids"]):
            return None
        survivors, mask = self._l0_cache.get_survivors_batch(
            document_ids, self._target_ratio(),
        )
        survivors = survivors.to(self.device)
        mask = mask.to(self.device)
        if survivors.size(-1) != self.decoder.hidden_dim:
            if self._cache_projection is None:
                self._cache_projection = nn.Linear(
                    survivors.size(-1), self.decoder.hidden_dim,
                ).to(self.device)
                self.model.cache_projection = self._cache_projection
            survivors = self._cache_projection(survivors)
        return survivors, mask

    # ------------------------------------------------------------------
    # L1 query-conditioned compression
    # ------------------------------------------------------------------

    def _compress_l1(
        self,
        l0_survivors: torch.Tensor,
        l0_mask: torch.Tensor,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run L1 compression on L0 survivors with query as prompt.

        The encoder processes L0 survivors as "content" with the question
        embedded as a prompt prefix. This enables:
        - Cross-document attention among L0 survivors from different docs
        - Query-guided survivor selection (ICE scores in context of question)
        - Light compression (e.g. 500 L0 survivors at L1 ratio 0.50 → 250)

        Args:
            l0_survivors: (B, K, D) L0 survivor embeddings.
            l0_mask: (B, K) attention mask for L0 survivors.
            question_ids: (B, Q) question token IDs.
            question_mask: (B, Q) question attention mask.

        Returns:
            (l1_survivors, l1_mask) after query-conditioned compression.
        """
        # Embed question tokens into encoder space for use as L1 prompt
        embed_tokens = self.encoder.compressor.backbone.get_input_embeddings()
        question_emb = embed_tokens(question_ids.to(self.device))
        question_mask = question_mask.to(self.device)

        batch_size = l0_survivors.size(0)
        l1_results = []
        l1_masks = []

        # Process per-sample (L1 survivor counts vary per sample)
        for i in range(batch_size):
            n_surv = int(l0_mask[i].sum())
            if n_surv == 0:
                l1_results.append(l0_survivors[i:i + 1, :1])
                l1_masks.append(torch.zeros(1, 1, dtype=torch.bool, device=self.device))
                continue

            # L1 input: this sample's L0 survivors
            l1_input = l0_survivors[i:i + 1, :n_surv]  # (1, n_surv, D)
            l1_attn = torch.ones(
                1, n_surv, dtype=torch.bool, device=self.device,
            )

            # Question prompt for this sample
            q_len = int(question_mask[i].sum())
            q_emb = question_emb[i:i + 1, :q_len]  # (1, Q, D)
            q_mask = question_mask[i:i + 1, :q_len]  # (1, Q)

            # Score L0 survivors with ICE (in context of the full sequence)
            with torch.no_grad():
                ice_scores = self.ice(l1_input)  # (1, n_surv)

            # Select L1 survivors — generous ratio (cross-doc fusion, not
            # aggressive compression)
            ratio = self._l1_ratio()
            keep = max(1, math.ceil(n_surv * ratio))
            scores_flat = ice_scores[0, :n_surv]
            _, topk = torch.topk(scores_flat, min(keep, n_surv))
            l1_survivor_mask = torch.zeros(
                1, n_surv, dtype=torch.bool, device=self.device,
            )
            l1_survivor_mask[0, topk] = True

            # Update L1 calibrator if training
            if self._l1_calibrator is not None and self.model.training:
                self._l1_calibrator.update_from_flat(scores_flat.detach())
                self._l1_batches_seen += 1
                if self._l1_batches_seen >= self._l1_calibrator_fast_batches:
                    self._l1_calibrator.set_decay(self._l1_calibrator_slow_decay)

            # Encoder forward: L0 survivors as content, question as prompt
            l1_out = self.encoder(
                input_embeddings=l1_input,
                survivor_mask=l1_survivor_mask,
                attention_mask=l1_attn,
                prompt_embeddings=q_emb,
                prompt_attention_mask=q_mask,
            )
            l1_results.append(l1_out.survivor_embeddings)
            l1_masks.append(l1_out.survivor_attention_mask)

        # Pad across batch
        max_l1 = max(r.size(1) for r in l1_results)
        hidden = l1_results[0].size(-1)
        padded = l0_survivors.new_zeros((batch_size, max_l1, hidden))
        padded_mask = torch.zeros(
            batch_size, max_l1, dtype=torch.bool, device=self.device,
        )
        for i, (r, m) in enumerate(zip(l1_results, l1_masks, strict=True)):
            n = r.size(1)
            padded[i, :n] = r[0]
            padded_mask[i, :n] = m[0]

        return padded, padded_mask

    # ------------------------------------------------------------------
    # Prompt composition (L0 → optional L1 → separate topic embeddings)
    # ------------------------------------------------------------------

    def set_ablation_mode(self, mode: str | None) -> None:
        """Set ablation mode for eval. Use class constants (ABLATION_*)."""
        self._ablation_mode = mode

    def _compose_prompt(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Build decoder prompt: L0 → L1 compression → topic embeddings.

        Returns two tensors for the 0.8B decoder (Steps 1-4):
          (survivors, mask) where survivors may include a learned separator
          between compressed context and topic embeddings.

        For Step 5 (target LLM), KRStep5Trainer overrides this to produce
        separate tool-call regions.
        """
        skip_context = self._ablation_mode in (
            self.ABLATION_TOPICS_ONLY, self.ABLATION_NEITHER,
        )
        skip_topics = self._ablation_mode in (
            self.ABLATION_NO_TOPICS, self.ABLATION_NEITHER,
        )

        # --- L0 survivors ---
        if skip_context:
            context = None
            context_mask = None
        else:
            cached = self._cached_l0_survivors(batch)
            if cached is not None:
                context, context_mask = cached
            elif self._use_live_l0():
                context, context_mask = self._encode_live_l0(
                    batch["content_token_ids"].to(self.device),
                    batch["content_attention_mask"].to(self.device),
                )
            else:
                context, context_mask = self._subsample_embeddings(
                    batch["content_token_ids"].to(self.device),
                    batch["content_attention_mask"].to(self.device),
                )

            # --- L1 query-conditioned compression ---
            if (
                context is not None
                and self._l1_enabled
                and self.encoder is not None
                and "question_token_ids" in batch
            ):
                context, context_mask = self._compress_l1(
                    context,
                    context_mask,
                    batch["question_token_ids"],
                    batch["question_attention_mask"],
                )

            # Ablation transforms
            if self._ablation_mode == self.ABLATION_ZEROED and context is not None:
                context = torch.zeros_like(context)
            elif self._ablation_mode == self.ABLATION_NOISE and context is not None:
                context = torch.randn_like(context) * 0.02

        # --- Topic embeddings (separate from compressed context) ---
        topic_block = None
        topic_mask = None
        if not skip_topics and self.topic_embeddings is not None:
            topic_block, topic_mask = self.topic_embeddings(batch["tags"])
            if topic_block.size(1) == 0:
                topic_block = None
                topic_mask = None

        # --- Assemble: [context | separator | topic_block] ---
        if context is not None and topic_block is not None:
            # Learned separator between context and topics
            sep = self.encoder.compressor.prompt_separator_embedding if (
                self.encoder is not None
                and hasattr(self.encoder, "compressor")
            ) else None
            if sep is not None:
                b = context.size(0)
                sep_emb = sep.unsqueeze(0).unsqueeze(0).expand(b, 1, -1)
                sep_mask = torch.ones(
                    b, 1, dtype=torch.bool, device=self.device,
                )
                prompt = torch.cat([context, sep_emb, topic_block], dim=1)
                mask = torch.cat([context_mask, sep_mask, topic_mask], dim=1)
            else:
                prompt = torch.cat([context, topic_block], dim=1)
                mask = torch.cat([context_mask, topic_mask], dim=1)
        elif context is not None:
            prompt = context
            mask = context_mask
        elif topic_block is not None:
            prompt = topic_block
            mask = topic_mask
        else:
            # Both skipped — minimal empty prompt
            b = batch["content_token_ids"].size(0)
            prompt = torch.zeros(
                b, 1, self.decoder.hidden_dim,
                device=self.device, dtype=torch.bfloat16,
            )
            mask = torch.zeros(b, 1, dtype=torch.bool, device=self.device)

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
        max_keep = max(1, max(math.ceil(ln * ratio) for ln in lengths))
        prompts = embedded.new_zeros((batch, max_keep, hidden))
        prompt_mask = torch.zeros(
            batch, max_keep, dtype=torch.bool, device=embedded.device,
        )
        for row, ln in enumerate(lengths):
            ln = int(ln)
            keep = max(1, math.ceil(ln * ratio))
            if ln == 1:
                indices = torch.zeros(1, dtype=torch.long, device=embedded.device)
            else:
                indices = torch.linspace(
                    0, ln - 1, steps=keep, device=embedded.device,
                ).round().long()
            prompts[row, :keep] = embedded[row, indices]
            prompt_mask[row, :keep] = True
        return prompts, prompt_mask

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _forward_backward(self, batch) -> dict[str, float]:
        prompt, prompt_mask = self._compose_prompt(batch)
        target_ids = batch["target_token_ids"].to(self.device)
        target_attention_mask = batch["target_attention_mask"].to(self.device)
        target_loss_mask = batch["target_loss_mask"].to(self.device)

        loss = self.decoder.forward_with_loss(
            prompt, target_ids, target_attention_mask, prompt_mask,
            loss_mask=target_loss_mask,
        )

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
            "l1_enabled": float(self._l1_enabled),
            "l0_ratio": self._l0_ratio(),
            "l1_ratio": self._l1_ratio() if self._l1_enabled else 0.0,
        }

    def _add_step_metrics(self, metrics: dict[str, float]) -> None:
        if self.topic_embeddings is not None:
            norms = {}
            for tag in self.topic_embeddings.taxonomy.tags[:20]:
                key = self.topic_embeddings._key(tag)
                if key in self.topic_embeddings.embeddings:
                    norms[f"topic_norm/{tag}"] = (
                        self.topic_embeddings.embeddings[key].data.norm().item()
                    )
            metrics.update(norms)
        if self._l1_calibrator is not None:
            metrics["l1_calibrator_threshold"] = (
                self._l1_calibrator.get_threshold(self._target_ratio())
            )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _get_eval_tokenizer(self):
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

        per_dataset_correct: dict[str, int] = {}
        per_dataset_total: dict[str, int] = {}
        per_dataset_em: dict[str, list[float]] = {}
        per_dataset_f1: dict[str, list[float]] = {}
        all_em: list[float] = []
        all_f1: list[float] = []

        for batch in self.eval_dataloader:
            prompt, prompt_mask = self._compose_prompt(batch)
            target_ids = batch["target_token_ids"].to(self.device)
            target_attention_mask = batch["target_attention_mask"].to(self.device)
            target_loss_mask = batch["target_loss_mask"].to(self.device)
            answer_token_ids = batch["answer_token_ids"]
            answer_attention_mask = batch["answer_attention_mask"]

            loss = self.decoder.forward_with_loss(
                prompt, target_ids, target_attention_mask, prompt_mask,
                loss_mask=target_loss_mask,
            )
            logits = self.decoder(
                prompt, target_ids, target_attention_mask, prompt_mask,
            )
            shifted_logits = logits[:, :-1]
            shifted_labels = target_ids[:, 1:]
            shifted_mask = target_loss_mask[:, 1:]
            predictions = shifted_logits.argmax(dim=-1)
            correct = (
                (predictions == shifted_labels) & shifted_mask
            ).sum().item()
            answer_tokens = shifted_mask.sum().item()
            total_correct += correct
            total_answer_tokens += answer_tokens
            total_loss += float(loss.item())
            total_batches += 1

            # Decode and compute QA metrics
            batch_size = answer_token_ids.size(0)
            for i in range(batch_size):
                ans_len = int(answer_attention_mask[i].sum())
                gold_ids = answer_token_ids[i, :ans_len]
                gold_text = tokenizer.decode(gold_ids, skip_special_tokens=True)

                mask_i = shifted_mask[i]
                pred_i = predictions[i]
                answer_pred_ids = pred_i[mask_i.bool()].cpu()
                pred_text = tokenizer.decode(
                    answer_pred_ids, skip_special_tokens=True,
                )

                em_score = exact_match(pred_text, [gold_text])
                f1_score = token_f1(pred_text, [gold_text])
                all_em.append(em_score)
                all_f1.append(f1_score)

                ds_name = None
                dataset_names = batch.get("dataset_names", [])
                if i < len(dataset_names) and dataset_names[i] is not None:
                    ds_name = str(dataset_names[i])
                if ds_name is not None:
                    per_dataset_em.setdefault(ds_name, []).append(em_score)
                    per_dataset_f1.setdefault(ds_name, []).append(f1_score)

            for i, ds_name in enumerate(batch.get("dataset_names", [])):
                if ds_name is None:
                    continue
                ds_name = str(ds_name)
                mask_i = shifted_mask[i]
                pred_i = predictions[i]
                label_i = shifted_labels[i]
                ds_c = ((pred_i == label_i) & mask_i).sum().item()
                ds_t = mask_i.sum().item()
                per_dataset_correct[ds_name] = (
                    per_dataset_correct.get(ds_name, 0) + ds_c
                )
                per_dataset_total[ds_name] = (
                    per_dataset_total.get(ds_name, 0) + ds_t
                )

        self.model.train()
        if total_batches == 0:
            return {
                "eval/loss": 0.0,
                "eval/answer_token_accuracy": 0.0,
                "eval/exact_match": 0.0,
                "eval/token_f1": 0.0,
            }

        metrics: dict[str, float] = {
            "eval/loss": total_loss / total_batches,
            "eval/answer_token_accuracy": (
                total_correct / total_answer_tokens
                if total_answer_tokens else 0.0
            ),
            "eval/exact_match": (
                sum(all_em) / len(all_em) if all_em else 0.0
            ),
            "eval/token_f1": (
                sum(all_f1) / len(all_f1) if all_f1 else 0.0
            ),
        }
        for ds_name in per_dataset_correct:
            ds_t = per_dataset_total.get(ds_name, 0)
            if ds_t > 0:
                metrics[f"eval/{ds_name}/token_accuracy"] = (
                    per_dataset_correct[ds_name] / ds_t
                )
        for ds_name, scores in per_dataset_em.items():
            if scores:
                metrics[f"eval/{ds_name}/exact_match"] = (
                    sum(scores) / len(scores)
                )
        for ds_name, scores in per_dataset_f1.items():
            if scores:
                metrics[f"eval/{ds_name}/token_f1"] = (
                    sum(scores) / len(scores)
                )
        return metrics
