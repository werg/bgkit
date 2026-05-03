"""Phase 2 KB-scale trainer.

Implements the query-conditioned browse + bgkit training loop described in
``docs/phase2_kb.md`` (a.k.a. the "quirky-drifting-moore" plan):

- Base encoder is frozen (Phase 1 weights). Per-level LoRA adapters shape
  L0 and L1 behavior.
- L0 is live in Stage A (L0 LoRA trainable) and cached in Stages B/C
  (L0 LoRA frozen, survivors loaded from :class:`bgkit.data.l0_cache.L0Cache`).
- L1 is always live and query-conditioned. Every bgkit tool call runs L1
  fresh over the referenced articles' L0 survivors with the question as
  prompt and article-ID tokens pinned into the survivor set.
- Decoder forward interleaves tokenized chat (browse responses, assistant
  tool calls, etc.) with live-computed L1 survivor embeddings at each
  ``bgkit`` sentinel.
- Per-turn loss masking from the trajectory's ``loss`` flags trains the
  decoder only on primary-path tool calls and the final answer. Sibling
  exploration tool calls still run L1 so their encoder gradient flows
  through the decoder's attention, but their decoder tokens get no CE loss.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path

import structlog
import torch
import torch.nn as nn
from omegaconf import open_dict
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split

from bgkit.data.article_token_store import ArticleTokenStore
from bgkit.data.bgkit_tool_template import (
    BGKIT_SENTINEL,
    BGKIT_TOPIC_SENTINEL,
    make_system_prompt,
    tokenize_trajectory,
)
from bgkit.data.browse_tree import BrowseTree
from bgkit.data.datasets.phase2_kb_dataset import KBSample, KBTrajectoryDataset
from bgkit.data.l0_cache import L0Cache
from bgkit.data.taxonomy import TagTaxonomy
from bgkit.models.decoder import (
    EmbeddingSegment,
    ReconstructionDecoder,
    Segment,
    TokenSegment,
)
from bgkit.models.lora_encoder import (
    DEFAULT_LORA_TARGETS,
    LoRALinearWrapper,
    LoRARouter,
    remap_base_keys_to_lora,
)
from bgkit.models.topic_embeddings import TopicEmbeddingModule
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.ratio_sampling import (
    build_ratio_sampler_config,
    resolve_anchor_grid,
    sample_ratio,
)
from bgkit.utils.attention_backend import resolve_attention_implementation
from bgkit.utils.packing import (
    lengths_from_cu,
    position_ids_from_cu,
    segment_ids_from_cu,
    segment_mean,
    segment_sum,
)

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Model container
# ---------------------------------------------------------------------------


class _KBModel(nn.Module):
    """Registers all trainable parameters under a single nn.Module for
    checkpointing."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        lora_router: LoRARouter | None,
        topic_embeddings: TopicEmbeddingModule | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lora_router = lora_router
        self.topic_embeddings = topic_embeddings


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def _collate_kb(batch: list[KBSample]) -> list[KBSample]:
    """KBSample objects cannot be stacked — we flatten and per-sample
    process inside :meth:`_forward_backward`."""
    return list(batch)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class _SpliceSlice:
    """Location of a single bgkit sentinel within a tokenized trajectory."""

    start: int
    length: int  # number of sentinel tokens (the sentinel may tokenize to >1 token)


@dataclass
class _KBDecodeTrace:
    """Concat-coordinate metadata attached to an interleaved decoder segment
    sequence built by :meth:`KRKBTrainer._build_decoder_segments_core`.

    All spans are ``[start, end)`` and live in the same coordinate space
    as ``InterleavedForwardOutput.token_ids`` (post-topic-block,
    post-survivor-splice). Eval harnesses (see
    :mod:`bgkit.eval.kb_trajectory_eval`) use these spans to score
    trajectory step accuracy, tool-call ID accuracy, and answer F1
    without reimplementing any remap logic.
    """

    answer_span: tuple[int, int] | None
    browse_turns: list
    browse_call_spans: list[tuple[int, int]]
    bgkit_turns: list
    bgkit_call_spans: list[tuple[int, int]]


class KRKBTrainer(BaseTrainer):
    """Knowledge-retrieval KB-scale trainer.

    Config schema under ``training``::

        phase: phase2_kb
        stage: "A" | "B" | "C"
        datasets: [kilt_wikipedia, pubmedqa, ...]
        max_steps: 30000
        browse_tree_dir: {DATA_DIR}/browse_trees
        trajectory_dir: {DATA_DIR}/trajectories
        l0_cache_dir: {DATA_DIR}/l0_cache         # None for Stage A (live L0)
        live_l0: false                            # true for Stage A
        l0_retention:
          kilt_wikipedia: 0.05
          pubmedqa: 0.20
          ...
        l1_retention: 0.50
        lora:
          l0_rank: 32
          l1_rank: 32
          alpha: 64
    """

    _log_every = 5
    _use_device_prefetcher = False  # samples carry variable-shaped state

    # Ablation modes — set via set_ablation_mode() during eval.
    ABLATION_NONE = None
    ABLATION_ZEROED = "zeroed"       # survivors → zeros (no context info)
    ABLATION_NOISE = "noise"         # survivors → gaussian noise
    ABLATION_NO_TOPICS = "no_topics"  # drop topic embedding segment
    ABLATION_TOPICS_ONLY = "topics_only"  # drop bgkit survivor segments
    ABLATION_NEITHER = "neither"     # drop both topics and bgkit survivors

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.step_cfg = cfg.training
        self.device: torch.device = torch.device("cpu")
        self.encoder: nn.Module | None = None
        self.decoder: ReconstructionDecoder | None = None
        self.tokenizer = None
        self.lora_router: LoRARouter | None = None
        self._trees: dict[str, BrowseTree] = {}
        # Per-dataset map from browse-tree article id (human-readable
        # title for KILT/PubMedQA) → canonical mmap document_id used by
        # the L0 cache and ArticleTokenStore. Loaded at setup time from
        # the ``{dataset}_title_to_doc_id.json`` sidecar written by
        # scripts/build_browse_tree.py. Datasets without a sidecar get
        # an empty dict and lookups pass through unchanged, which
        # preserves the legacy numeric-id flow for git history, memory
        # datasets, and any corpus whose browse-tree ids ARE mmap keys.
        self._title_to_doc_id: dict[str, dict[str, str]] = {}
        # Reverse of :attr:`_title_to_doc_id`: per-dataset
        # ``document_id → browse-tree article id`` (the title string).
        # Used by :meth:`_prepare_l1_turn` to decide what text to pin
        # for each article's L1 survivors, so the decoder sees the
        # human-readable title embedded alongside the L0 survivors even
        # after the mmap-key translation. Rebuilt together with
        # :attr:`_title_to_doc_id` inside :meth:`_load_browse_trees`.
        self._doc_id_to_title: dict[str, dict[str, str]] = {}
        self._l0_cache: L0Cache | None = None
        self._token_store: ArticleTokenStore | None = None
        self._l0_retention: dict[str, float] = {}
        self._l1_retention: float = 0.15
        self._live_l0: bool = False
        self.taxonomy = None
        self.topic_embeddings = None
        self._ablation_mode: str | None = None
        # Training-time random ablation (capability regression prevention).
        # Rolled once per sample in _build_decoder_segments_core; disabled
        # during eval. Probabilities are cfg-driven and sum independently.
        tt_ablation = dict(self.step_cfg.get("training_time_ablation", {}) or {})
        self._p_skip_bgkit_training = float(tt_ablation.get("p_skip_bgkit", 0.15))
        self._p_skip_topic_training = float(tt_ablation.get("p_skip_topic", 0.15))
        self._p_noise_bgkit_training = float(tt_ablation.get("p_noise_bgkit", 0.0))
        import random as _random

        self._ablation_rng = _random.Random(int(cfg.get("seed", 42)))

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _stage(self) -> str:
        return str(self.step_cfg.get("stage", "A")).upper()

    def _resolve_dir(self, key: str, default: str) -> Path:
        value = self.step_cfg.get(key, None)
        if value:
            return Path(str(value))
        from bgkit.env import DATA_DIR

        return Path(DATA_DIR) / default

    # ------------------------------------------------------------------
    # setup()
    # ------------------------------------------------------------------

    def setup(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._live_l0 = bool(self.step_cfg.get("live_l0", self._stage() == "A"))
        attention_impl = resolve_attention_implementation(
            self.cfg.compute.get("attention_implementation", "auto")
        )

        # --- Decoder ---
        decoder_name = self.cfg.model.decoder.backbone_name
        decoder_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        decoder_backbone = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            trust_remote_code=True,
            torch_dtype=decoder_dtype,
            attn_implementation=attention_impl,
        ).to(self.device)
        hidden = decoder_backbone.get_input_embeddings().weight.shape[1]
        self.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=hidden)
        self.decoder.train()

        # Activation checkpointing on the decoder — significant memory win
        # at a ~30% throughput cost. Default on; disable via
        # ``training.activation_checkpointing.decoder: false``.
        ac_cfg = self.step_cfg.get("activation_checkpointing", {}) or {}
        if ac_cfg.get("decoder", True) and hasattr(
            decoder_backbone, "gradient_checkpointing_enable",
        ):
            try:
                decoder_backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
                logger.info("phase2_kb_decoder_gc_enabled")
            except TypeError:
                # Older HF versions don't accept kwargs.
                decoder_backbone.gradient_checkpointing_enable()
                logger.info("phase2_kb_decoder_gc_enabled_legacy")
        self._checkpoint_encoder = bool(ac_cfg.get("encoder", True))
        # CPU offload for encoder activation checkpointing. When enabled,
        # _checkpointed_encoder routes through ``cpu_offload_checkpoint``
        # which parks the saved input tensors on pinned host memory between
        # forward and backward (~30% extra activation memory saving at
        # ~1-2% throughput cost). Off by default for parity with the
        # existing plain-checkpoint path.
        self._cpu_offload_activations = bool(
            self.step_cfg.get("cpu_offload_activations", False),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            decoder_name, trust_remote_code=True,
        )

        # --- Encoder ---
        self._load_encoder()

        # Encoder-side tokenizer (usually the same vocab as the decoder for
        # Qwen3.5, but we load it from the encoder's backbone name so we're
        # robust to vocab mismatches between encoder and decoder).
        encoder_name = self.cfg.model.get("encoder", {}).get(
            "backbone_name", "Qwen/Qwen3.5-0.8B-Base",
        )
        if encoder_name == decoder_name:
            self.encoder_tokenizer = self.tokenizer
        else:
            self.encoder_tokenizer = AutoTokenizer.from_pretrained(
                encoder_name, trust_remote_code=True,
            )

        # Vocab alignment sanity check: every token ID the encoder
        # tokenizer produces must index into the encoder's embedding
        # table, and likewise for the decoder. A silent vocab mismatch
        # is otherwise a garbage-embedding bug that's impossible to
        # debug from training loss alone.
        self._assert_vocab_alignment()

        # --- LoRA adapters ---
        self._install_lora()

        # --- L0 survivor cache (Stages B/C) ---
        if not self._live_l0:
            cache_dir = self._resolve_dir("l0_cache_dir", "l0_cache")
            self._l0_cache = L0Cache(str(cache_dir))
            # Verify each active dataset's cache manifest matches the
            # configured checkpoints + LoRA shape. Raises
            # L0CacheManifestMismatch if Stage B is about to train on a
            # cache that was built with a different L0 LoRA, which is
            # the failure mode the manifest exists to catch.
            self._verify_l0_cache_manifests(cache_dir)

        # --- Article token store (Stage A live L0 needs raw tokens) ---
        if self._live_l0:
            mmap_root = self._resolve_dir("mmap_dir", "mmap/phase2")
            self._token_store = ArticleTokenStore(mmap_root)

        # --- Browse trees ---
        self._load_browse_trees()

        # --- Retention ratios ---
        # L0 retention supports two formats per dataset:
        #   static:    l0_retention: {pubmedqa: 0.20}
        #   curriculum: l0_retention: {pubmedqa: {start: 0.20, end: 0.01, ramp_steps: 15000}}
        # Static values are stored as-is; curriculum dicts are stored
        # whole and interpolated by global_step in _l0_retention_for().
        self._l0_retention: dict[str, float | dict] = {}
        for k, v in dict(self.step_cfg.get("l0_retention", {})).items():
            if isinstance(v, (int, float)):
                self._l0_retention[str(k)] = float(v)
            else:
                self._l0_retention[str(k)] = dict(v)
        self._l1_retention = float(self.step_cfg.get("l1_retention", 0.15))
        anchor_grid = resolve_anchor_grid(
            self.cfg.model,
            float(self._l1_retention),
            getattr(self.encoder.l0.threshold, "anchor_ratios", None),
        )
        self._l0_ratio_sampler_cfg = build_ratio_sampler_config(
            self.step_cfg.get("l0_retention_jitter", {}) or {},
            anchor_grid=anchor_grid,
            default_ratio=float(self.step_cfg.get("default_l0_retention", 0.10)),
            enabled_default=False,
            mode_default="jitter",
        )
        self._l1_ratio_sampler_cfg = build_ratio_sampler_config(
            self.step_cfg.get("l1_retention_jitter", {}) or {},
            anchor_grid=anchor_grid,
            default_ratio=float(self._l1_retention),
            enabled_default=False,
            mode_default="jitter",
        )
        import random as _random

        self._l0_ratio_rng = _random.Random(int(self.cfg.get("seed", 42)) + 101)
        self._l1_ratio_rng = _random.Random(int(self.cfg.get("seed", 42)) + 202)
        self._step_sampled_l0_ratios: list[float] = []
        self._step_sampled_l1_ratios: list[float] = []

        # --- Survivorship head aux losses ---
        # Phase 2 layers per-level config on top of the legacy trainer-scope
        # ratio/decisiveness/relevance knobs. The per-level blocks drive
        # BCE warmup + moment match + soft-attn weight; the legacy knobs
        # remain for L0/L1 aggregate-ratio + relevance (which has no
        # Phase 1 analogue).
        from bgkit.training.survivorship_helpers import (
            init_state,
            load_reference_moments,
            resolve_level_ice_cfg,
            resolve_level_loss_cfg,
        )

        surv_cfg = self.step_cfg.get("survivorship", {}) or {}
        self._ratio_loss_weight = float(surv_cfg.get("ratio_loss_weight", 0.1))
        self._decisiveness_loss_weight = float(
            surv_cfg.get("decisiveness_loss_weight", 0.05),
        )
        self._relevance_loss_weight = float(surv_cfg.get("relevance_loss_weight", 0.05))

        self._surv_l0 = resolve_level_loss_cfg(surv_cfg.get("l0", {}))
        self._surv_l1 = resolve_level_loss_cfg(surv_cfg.get("l1", {}))

        ice_cfg = self.step_cfg.get("ice_distillation", {}) or {}
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
            embed_tokens = self.encoder.l0.backbone.embed_tokens
            self._ice_teacher = ICETeacher(
                ice_path, embed_tokens,
                input_dim=int(ice_cfg.get("input_dim", 1024)),
                hidden_dim=int(ice_cfg.get("hidden_dim", 192)),
                num_layers=int(ice_cfg.get("num_layers", 3)),
                kernel_size=int(ice_cfg.get("kernel_size", 5)),
            ).to(self.device)

        mm_ref = self.step_cfg.get("moment_match_reference", {}) or {}
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

        # Stage B flag: when True, L0 is cached + L0 LoRA frozen, so we
        # skip L0 θ/μ updates in post-step. Must be set explicitly —
        # defaulting silently would cause wrong θ/μ behavior at Stage B
        # if the key is ever omitted.
        if "live_l0" not in self.step_cfg:
            raise ValueError(
                "Phase 2 KR/KB config missing required key 'live_l0'. "
                "Set `live_l0: true` for Stage A (live-L0 training) or "
                "`live_l0: false` for Stage B (cached L0, L0 LoRA frozen)."
            )
        self._live_l0 = bool(self.step_cfg.get("live_l0"))

        self._surv_state_l0 = init_state()
        self._surv_state_l1 = init_state()
        self._last_post_step_metrics: dict[str, float] = {}
        # Per-position target ratio multipliers applied atop the global
        # target. Gold-article positions get upsampled (retain more);
        # distractor positions get downsampled (retain less). A value of
        # 1.0 means "no bias". These scale the per-group target used by
        # the weighted aggregate-ratio term in the relevance loss.
        self._relevance_gold_boost = float(surv_cfg.get("relevance_gold_boost", 1.5))
        self._relevance_distractor_damp = float(
            surv_cfg.get("relevance_distractor_damp", 0.5),
        )
        self._n_distractors = int(surv_cfg.get("n_distractors", 3))

        # --- Datasets ---
        self._build_dataloaders()

        # --- Topic embeddings (optional) ---
        # Built AFTER datasets so we can count real tag occurrences from the
        # loaded trajectories instead of guessing from browse tree sizes.
        self._load_topic_embeddings()

        # --- Trajectory → L0 coverage validation ---
        # Fail loudly at setup if any trajectory references articles the L0
        # source is missing, so training never silently degrades on drops.
        self._validate_trajectory_article_coverage()

        # --- Optional Liger Kernel fused kernels ---
        # RMSNorm / SwiGLU / RoPE + fused linear+CE over the 248K Qwen vocab.
        # Gated on ``training.use_liger`` (default True); no-op when
        # liger-kernel is not installed in the environment.
        if bool(self.step_cfg.get("use_liger", True)):
            from bgkit.utils.liger_integration import apply_liger_to_qwen35

            enc_patched = apply_liger_to_qwen35(self.encoder)
            dec_patched = apply_liger_to_qwen35(self.decoder)
            self.decoder.enable_liger_ce(True)
            logger.info(
                "phase2_kb_liger_applied",
                encoder_modules=enc_patched,
                decoder_modules=dec_patched,
            )

        # --- Model container + optimizer ---
        self.model = _KBModel(
            encoder=self.encoder,
            decoder=self.decoder,
            lora_router=self.lora_router,
            topic_embeddings=self.topic_embeddings,
        ).to(self.device)
        self.optimizer = self._create_optimizer(
            self._build_optimizer_groups(),
            default_lr=float(self.step_cfg.get("lr", 1e-4)),
        )

    # ------------------------------------------------------------------
    # L0 cache manifest verification
    # ------------------------------------------------------------------

    def _verify_l0_cache_manifests(self, cache_dir: Path) -> None:
        """Cross-check the trainer config against each dataset's cached
        provenance manifest.

        For Stage B/C this catches the most common silent-corruption
        path: someone re-ran ``precompute_l0_subset.py`` without
        ``--stage-a-checkpoint`` and the cache now contains Phase-1-only
        L0 survivors instead of the Stage-A LoRA-shaped ones the trainer
        is about to use. Without the manifest the trainer would happily
        train on the wrong cache and the only symptom would be quietly
        worse downstream metrics.
        """
        from bgkit.data.l0_cache import (
            L0CacheManifestMismatch,
            assert_cache_manifest_matches,
            read_cache_manifest,
        )
        from bgkit.training.checkpoint_registry import resolve_checkpoint

        phase1_ckpt = self.step_cfg.get("phase1_checkpoint")
        if phase1_ckpt and str(phase1_ckpt) == "auto":
            checkpoint_dir = Path(str(self.cfg.get("checkpoint_dir", "checkpoints")))
            try:
                phase1_ckpt = str(resolve_checkpoint(
                    checkpoint_dir,
                    phase="phase1_step6",
                    metric="eval/loss",
                    lower_is_better=True,
                ))
            except Exception:
                phase1_ckpt = None
        phase1_path = Path(str(phase1_ckpt)) if phase1_ckpt else None

        stage_a_ckpt = self.step_cfg.get("stage_a_checkpoint")
        if stage_a_ckpt and str(stage_a_ckpt) == "auto":
            checkpoint_dir = Path(str(self.cfg.get("checkpoint_dir", "checkpoints")))
            try:
                stage_a_ckpt = str(resolve_checkpoint(
                    checkpoint_dir,
                    phase="phase2_kb",
                    metric="eval/loss",
                    lower_is_better=True,
                ))
            except Exception:
                stage_a_ckpt = None
        stage_a_path = Path(str(stage_a_ckpt)) if stage_a_ckpt else None

        lora_cfg = self.step_cfg.get("lora", {}) or {}
        lora_rank = int(lora_cfg.get("l0_rank", 32))
        lora_alpha_cfg = lora_cfg.get("alpha")
        lora_alpha = (
            float(lora_alpha_cfg) if lora_alpha_cfg is not None else None
        )

        for name in list(self.step_cfg.get("datasets", [])):
            retention = float(self._l0_retention.get(name, 0.10)) if self._l0_retention else None
            manifest = read_cache_manifest(cache_dir, name)
            if manifest is None:
                logger.warning(
                    "phase2_kb_l0_cache_no_manifest",
                    dataset=name,
                    cache_dir=str(cache_dir),
                    msg=(
                        "no cache_manifest.json — this cache predates "
                        "manifest tracking. Re-run precompute_l0_subset.py "
                        "with the same checkpoints to add a manifest."
                    ),
                )
                continue
            try:
                assert_cache_manifest_matches(
                    cache_dir,
                    name,
                    phase1_checkpoint=phase1_path,
                    stage_a_checkpoint=stage_a_path,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    retention=retention,
                )
            except L0CacheManifestMismatch:
                logger.error(
                    "phase2_kb_l0_cache_manifest_mismatch",
                    dataset=name,
                    cache_dir=str(cache_dir),
                )
                raise
            logger.info(
                "phase2_kb_l0_cache_manifest_ok",
                dataset=name,
                phase1_sha=manifest.get("phase1_sha"),
                stage_a_sha=manifest.get("stage_a_sha"),
            )

    # ------------------------------------------------------------------
    # Encoder loading
    # ------------------------------------------------------------------

    def _load_encoder(self) -> None:
        from bgkit.models.encoder import BgKITEncoder
        from bgkit.training.checkpointing import load_checkpoint

        phase1_ckpt = self.step_cfg.get("phase1_checkpoint")
        if not phase1_ckpt:
            raise ValueError(
                "phase2_kb requires training.phase1_checkpoint (the Phase 1 encoder)"
            )
        if str(phase1_ckpt) == "auto":
            from bgkit.training.checkpoint_registry import resolve_checkpoint

            checkpoint_dir = Path(str(self.cfg.get("checkpoint_dir", "checkpoints")))
            phase1_ckpt = str(resolve_checkpoint(
                checkpoint_dir,
                phase="phase1_step6",
                metric="eval/loss",
                lower_is_better=True,
            ))

        _meta, state_dicts = load_checkpoint(Path(str(phase1_ckpt)))
        model_state = state_dicts.get("model", {})
        encoder_state = {
            k.replace("encoder.", "", 1): v
            for k, v in model_state.items()
            if k.startswith("encoder.")
        }

        encoder_cfg = self.cfg.model.get("encoder", {})
        self.encoder = BgKITEncoder.from_pretrained_with_state_dict(
            encoder_cfg.get("backbone_name", "Qwen/Qwen3.5-0.8B-Base"),
            encoder_state,
            hidden_dim=int(encoder_cfg.get("hidden_dim", 1024)),
        ).to(self.device)

    def _assert_vocab_alignment(self) -> None:
        """Verify encoder-tokenizer and decoder-tokenizer IDs index into
        the corresponding model's embedding table.

        This catches the class of bug where the encoder and decoder have
        tokenizers with slightly different vocabulary sizes (e.g., one
        uses ``-Base`` and the other doesn't, and the instruct variant
        has special-token additions). Caught early via an explicit
        assertion rather than via silent garbage embeddings.
        """
        enc_embed = self.encoder.l0.backbone.get_input_embeddings()
        enc_vocab = enc_embed.num_embeddings
        dec_embed = self.decoder.backbone.get_input_embeddings()
        dec_vocab = dec_embed.weight.shape[0]

        probes = [
            "The quick brown fox",
            "sample_article_id",
            "<|im_start|>user",
        ]
        for text in probes:
            enc_ids = self.encoder_tokenizer.encode(text, add_special_tokens=False)
            if enc_ids and max(enc_ids) >= enc_vocab:
                raise RuntimeError(
                    f"Encoder tokenizer produced ID {max(enc_ids)} for "
                    f"{text!r} but encoder embedding table has only "
                    f"{enc_vocab} entries. Check encoder/decoder "
                    f"tokenizer compatibility."
                )
            dec_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if dec_ids and max(dec_ids) >= dec_vocab:
                raise RuntimeError(
                    f"Decoder tokenizer produced ID {max(dec_ids)} for "
                    f"{text!r} but decoder embedding table has only "
                    f"{dec_vocab} entries."
                )

        # Cross-tokenizer roundtrip: when encoder and decoder share the
        # same vocab (common for Qwen3.5 base/instruct pairs), an ID
        # encoded by one must decode to the same text via the other.
        # When vocabs differ we skip this check (they'll produce
        # different tokens anyway, but that's OK as long as each is
        # self-consistent with its own embedding table).
        if self.encoder_tokenizer is self.tokenizer or enc_vocab == dec_vocab:
            sample = "Wave_mechanics"
            enc_ids = self.encoder_tokenizer.encode(
                sample, add_special_tokens=False,
            )
            roundtrip = self.encoder_tokenizer.decode(
                enc_ids, skip_special_tokens=True,
            )
            if roundtrip.strip() != sample:
                logger.warning(
                    "phase2_kb_tokenizer_lossy_roundtrip",
                    original=sample,
                    roundtrip=roundtrip,
                    hint="tokenizer may not preserve article IDs exactly",
                )
        logger.info(
            "phase2_kb_vocab_alignment_ok",
            enc_vocab=enc_vocab, dec_vocab=dec_vocab,
        )

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def _install_lora(self) -> None:
        """Install L1-only LoRA. Stage A trains ``encoder.l0`` weights
        directly (no LoRA wrapper); Stage B freezes ``encoder.l0`` entirely
        and trains ``l1`` LoRA + decoder.
        """
        lora_cfg = self.step_cfg.get("lora", {})
        # Only install on L1's submodule — keep L0 free of LoRA wrappers.
        levels = {"l1": int(lora_cfg.get("l1_rank", 32))}
        targets = tuple(lora_cfg.get("target_modules", DEFAULT_LORA_TARGETS))
        alpha = lora_cfg.get("alpha")
        self.lora_router = LoRARouter.install(
            self.encoder.l1,
            target_names=targets,
            levels=levels,
            alpha=float(alpha) if alpha is not None else None,
            dropout=float(lora_cfg.get("dropout", 0.0)),
        )
        LoRARouter.bind(self.lora_router)

        # Freeze encoder base; unfreeze L1 LoRA always.
        self.encoder.requires_grad_(False)
        self.lora_router.set_level_trainable("l1", True)
        # Stage A: train L0 weights directly (head + auto_repro_head + backbone).
        # Stage B: keep L0 frozen.
        if self._live_l0:
            self.encoder.l0.requires_grad_(True)

    # ------------------------------------------------------------------
    # Browse trees
    # ------------------------------------------------------------------

    def _load_browse_trees(self) -> None:
        """Load every configured dataset's browse tree + title sidecar.

        The title → document_id sidecar at
        ``{browse_tree_dir}/{dataset}_title_to_doc_id.json`` is optional.
        When present it translates browse-tree article ids (human-readable
        titles the decoder memorizes and emits) into mmap document_ids
        that key the L0 cache and ArticleTokenStore. When absent, the
        sidecar dict stays empty and every
        :meth:`_article_ids_to_document_ids` lookup is a pass-through.
        Logs the sidecar status per dataset so the operator can tell at a
        glance whether the browse tree is title-keyed (KILT/PubMedQA) or
        id-keyed (git history, memory datasets).
        """
        import json as _json

        tree_dir = self._resolve_dir("browse_tree_dir", "browse_trees")
        for name in list(self.step_cfg.get("datasets", [])):
            path = tree_dir / f"{name}.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"Browse tree missing for dataset {name!r}: {path}. "
                    "Run scripts/build_browse_tree.py first."
                )
            self._trees[name] = BrowseTree.load(path, dataset=name)
            sidecar = tree_dir / f"{name}_title_to_doc_id.json"
            if sidecar.exists():
                with sidecar.open() as f:
                    raw_map = _json.load(f)
                if not isinstance(raw_map, dict):
                    raise ValueError(
                        f"Title sidecar {sidecar} must be a JSON object; "
                        f"got {type(raw_map).__name__}",
                    )
                forward = {
                    str(k): str(v) for k, v in raw_map.items()
                }
                self._title_to_doc_id[name] = forward
                # Build the reverse map so :meth:`_prepare_l1_turn` can
                # pin human-readable titles into L1 content even when
                # the article list it receives is already translated to
                # document ids. Last-writer-wins on collisions; for
                # KILT/PubMedQA the hierarchy builder pre-disambiguates
                # titles so this is effectively 1:1.
                self._doc_id_to_title[name] = {
                    doc_id: title for title, doc_id in forward.items()
                }
                logger.info(
                    "phase2_kb_title_sidecar_loaded",
                    dataset=name,
                    n_entries=len(forward),
                )
            else:
                self._title_to_doc_id[name] = {}
                self._doc_id_to_title[name] = {}
                logger.info(
                    "phase2_kb_title_sidecar_absent",
                    dataset=name,
                    msg="browse-tree ids treated as canonical mmap document_ids",
                )

    def _article_ids_to_document_ids(
        self, dataset: str, article_ids: list[str],
    ) -> list[str]:
        """Translate browse-tree article ids → mmap document_ids.

        ``article_ids`` here is the post-resolve list produced by
        :meth:`_resolve_article_ids` — each entry is a browse-tree node
        id of kind ``article`` (a human-readable title for title-keyed
        datasets, a numeric document_id for legacy datasets). The return
        value is the same length but every entry is the canonical mmap
        key consumed by :class:`ArticleTokenStore` and
        :class:`L0Cache`.

        The translation falls back to the input id unchanged when the
        dataset has no sidecar or when a specific entry is not in the
        sidecar — the latter should not happen in practice (the sidecar
        is built from the same hierarchy JSONL that seeded the browse
        tree), but it keeps the pre-Phase-2 numeric-id flow working
        unchanged and lets synthetic unit/integration tests use
        identity-keyed fake browse trees.
        """
        # Use getattr so test fixtures that bypass __init__ via __new__
        # and only set a subset of trainer attributes don't need to know
        # about the sidecar machinery. A missing attribute is equivalent
        # to "no sidecar loaded" → identity passthrough.
        maps = getattr(self, "_title_to_doc_id", None) or {}
        mapping = maps.get(dataset) if maps else None
        if not mapping:
            return list(article_ids)
        return [mapping.get(aid, aid) for aid in article_ids]

    def _load_topic_embeddings(self) -> None:
        """Load a tag taxonomy and build the topic embedding module.

        Disabled by default; enable via::

            training:
              topic_embeddings:
                enabled: true
                # Option A: explicit pre-built taxonomy JSON
                taxonomy_path: ${DATA_DIR}/taxonomy/phase2_kb.json
                # Option B: auto-build from currently-loaded browse trees
                auto_from_browse_trees: true
                positions_per_tag: 8

        Auto-build unions every loaded browse tree into a single
        taxonomy. Tag IDs from different datasets don't collide because
        they're already namespaced by the dataset's top-level topic
        (e.g. ``Physics/Quantum_mechanics`` vs ``A01/A01.456``).
        """
        cfg = self.step_cfg.get("topic_embeddings", {}) or {}
        if not cfg.get("enabled", False):
            return

        taxonomy: TagTaxonomy | None = None
        tax_path = cfg.get("taxonomy_path")
        if tax_path:
            path = Path(str(tax_path))
            if path.exists():
                taxonomy = TagTaxonomy.load(path)
            else:
                logger.warning("phase2_kb_topic_taxonomy_missing", path=str(path))

        if taxonomy is None and cfg.get("auto_from_browse_trees", True):
            if not self._trees:
                logger.warning(
                    "phase2_kb_topic_embeddings_no_source",
                    msg="topic_embeddings.enabled=true but no taxonomy_path "
                    "and no loaded browse trees — skipping topic embeddings",
                )
                return
            # Union every loaded browse tree into one taxonomy. Frequencies
            # are placeholders at this stage — they get replaced below from
            # actual trajectory tag counts.
            merged_nodes: dict[str, object] = {}
            for tree in self._trees.values():
                tree_tax = TagTaxonomy.from_browse_tree(
                    tree, frequency_from_size=False,
                )
                merged_nodes.update(tree_tax._nodes)
            taxonomy = TagTaxonomy(merged_nodes, separator="/")

        if taxonomy is None:
            return

        # Recount tag frequencies from the loaded trajectories so the
        # optimizer's sqrt-frequency LR scaling uses real occurrence
        # counts, not browse-tree size proxies. A rare tag that shows up
        # on 3 training samples gets a much larger effective LR than a
        # common tag that shows up on 5000, which is what we want for
        # sparse-embedding training.
        tag_counts = self._count_trajectory_tag_frequencies(taxonomy)
        taxonomy = taxonomy.with_frequencies(tag_counts)

        self.taxonomy = taxonomy
        self.topic_embeddings = TopicEmbeddingModule(
            self.taxonomy,
            positions_per_tag=int(cfg.get("positions_per_tag", 8)),
            hidden_dim=int(self.decoder.hidden_dim),
        ).to(self.device)
        logger.info(
            "phase2_kb_topic_embeddings_loaded",
            n_tags=len(self.taxonomy),
            nonzero_freq_tags=sum(1 for v in tag_counts.values() if v > 0),
        )

    def _count_trajectory_tag_frequencies(
        self, taxonomy: TagTaxonomy,
    ) -> dict[str, int]:
        """Walk the loaded training + eval trajectories and count how
        many samples reference each tag (expanded through ancestors).

        Only tags already present in ``taxonomy`` get counted; unknown
        tag IDs from a trajectory (e.g. garbage from a dataset whose
        hierarchy wasn't matched) are silently ignored.
        """
        from collections import Counter

        counts: Counter[str] = Counter()
        seen_total = 0
        for dataset in (self.train_dataset, self.eval_dataset):
            if dataset is None:
                continue
            # Support plain Dataset and torch.utils.data.Subset uniformly.
            for i in range(len(dataset)):
                sample = dataset[i]
                seen_total += 1
                tags = self._sample_tags_for(sample)
                # Expand each tag's ancestor chain so parent tags
                # accumulate frequency from every descendant they cover.
                expanded = taxonomy.expand_tags(tags)
                for tag in expanded:
                    if tag in taxonomy:
                        counts[tag] += 1
        logger.info(
            "phase2_kb_topic_tag_counts",
            n_samples=seen_total,
            n_unique_tags=len(counts),
        )
        return dict(counts)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def _apply_top_tag_whitelist(
        self,
        dataset: KBTrajectoryDataset,
        whitelist: set[str],
    ) -> torch.utils.data.Subset:
        """Filter a dataset to samples whose primary bgkit turn targets
        a leaf under one of the whitelisted top-level topics.

        A sample passes if ANY of its bgkit turns' referenced tag IDs
        has a path in the browse tree whose first non-root ancestor is
        in ``whitelist``. Lookups go through the loaded browse tree for
        the sample's dataset_name.
        """

        kept_indices: list[int] = []
        for i in range(len(dataset)):
            sample = dataset[i]
            tree = self._trees.get(sample.dataset_name)
            if tree is None:
                continue
            tags = self._sample_tags_for(sample)
            matched = False
            for tag in tags:
                if tag not in tree:
                    continue
                try:
                    path = tree.path_to(tag)
                except KeyError:
                    continue
                # path[0] is always "root"; path[1] is the top-level topic.
                if len(path) >= 2 and path[1] in whitelist:
                    matched = True
                    break
            if matched:
                kept_indices.append(i)
        return Subset(dataset, kept_indices)

    def _build_dataloaders(self) -> None:
        """Load per-dataset trajectory parquets, optionally subset them
        according to the stage curriculum, and wrap in DataLoaders.

        Supported subsetting config keys under ``training``::

            max_samples_per_dataset: {kilt_wikipedia: 1500, pubmedqa: 1500, ...}
            max_samples_per_dataset_default: 1500  # fallback
            top_tag_whitelist:
              kilt_wikipedia: [Physics, Biology, Mathematics]

        ``max_samples_per_dataset`` caps the number of trajectories loaded
        from each dataset via :class:`torch.utils.data.Subset`. Stage A
        uses it to stay within the ~6K-article budget even when the
        underlying trajectory parquets cover millions of samples.

        ``top_tag_whitelist`` restricts each dataset to samples whose
        primary bgkit turn targets a leaf under one of the listed
        top-level browse-tree topics. Stage B uses this to pick the 10
        Wikipedia top-tags it actually trains on without rebuilding any
        parquet files.
        """

        traj_dir = self._resolve_dir("trajectory_dir", "trajectories")

        max_per_dataset_cfg: dict = dict(
            self.step_cfg.get("max_samples_per_dataset", {}) or {},
        )
        max_per_dataset_default = self.step_cfg.get(
            "max_samples_per_dataset_default", None,
        )
        top_tag_whitelist_cfg: dict = dict(
            self.step_cfg.get("top_tag_whitelist", {}) or {},
        )

        datasets = []
        for name in list(self.step_cfg.get("datasets", [])):
            path = traj_dir / f"{name}.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"Trajectory parquet missing for dataset {name!r}: {path}. "
                    "Run scripts/build_teacher_trajectories.py first."
                )
            ds = KBTrajectoryDataset(path)
            cap = max_per_dataset_cfg.get(name, max_per_dataset_default)
            whitelist = top_tag_whitelist_cfg.get(name)
            if whitelist is not None:
                ds = self._apply_top_tag_whitelist(ds, set(whitelist))
            if cap is not None and cap < len(ds):
                # Deterministic first-N slice. Trajectories inside the
                # parquet are in insertion order from
                # build_teacher_trajectories.py; taking the first N keeps
                # the subset stable across runs.
                ds = Subset(ds, list(range(int(cap))))
            logger.info(
                "phase2_kb_dataset_loaded",
                dataset=name,
                size=len(ds),
                cap=cap,
                whitelist_size=(
                    len(whitelist) if whitelist is not None else None
                ),
            )
            datasets.append(ds)
        if not datasets:
            raise ValueError("phase2_kb: training.datasets must be non-empty")
        full = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

        total = len(full)
        eval_size = min(
            max(1, int(total * 0.05)),
            int(self.step_cfg.get("max_eval_samples", 256)),
        )
        train_size = total - eval_size
        generator = torch.Generator().manual_seed(int(self.cfg.get("seed", 42)))
        self.train_dataset, self.eval_dataset = random_split(
            full, [train_size, eval_size], generator=generator,
        )
        batch_size = int(self.cfg.get("batch_size", 1))
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_kb,
            num_workers=int(self.cfg.compute.get("num_workers", 0)),
            pin_memory=False,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=_collate_kb,
            num_workers=0,
            pin_memory=False,
        )

        # ``max_steps: null`` means "exactly one epoch over the train
        # split". Used by Stage A to avoid pointless data repetition on
        # the small bootstrap corpus. The training loop reads
        # ``self.step_cfg.max_steps`` later, so resolve it here so the
        # rest of the trainer doesn't need to know about ``null``.
        configured_max_steps = self.step_cfg.get("max_steps", None)
        if configured_max_steps is None:
            one_epoch = math.ceil(len(self.train_dataset) / max(batch_size, 1))
            with open_dict(self.step_cfg):
                self.step_cfg.max_steps = one_epoch
            logger.info(
                "phase2_kb_max_steps_one_epoch",
                stage=self._stage(),
                train_size=len(self.train_dataset),
                batch_size=batch_size,
                max_steps=one_epoch,
            )

        # --- Head-tanh temperature calibration (L0 + L1) ---
        # Inherited from Phase 1 Step 6 checkpoint but re-probed against
        # Phase 2 text corpus (Wikipedia / KILT / PubMed-MeSH / etc.)
        # so T reflects the actual head-output std Stage A / Stage B
        # will see. Skipped when no KB text can be extracted.
        self._calibrate_head_tanh_temperatures()

    def _calibrate_head_tanh_temperatures(self, n_probe_batches: int = 4) -> None:
        """Probe L0 + L1 head output std against Phase 2 KB content.

        Phase 2 batches are lists of ``KBSample`` objects, not dicts of
        tensors, so the shared calibration helper can't read
        ``content_token_ids`` directly. We plug in a KB-aware extractor
        that tokenizes each sample's ``question`` + the first rendered
        trajectory step via ``encoder_tokenizer``. Question + trajectory
        text is representative Phase 2 content; tokenizing a few of them
        gives the backbone realistic input for the probe.
        """
        from bgkit.training.survivorship_helpers import (
            calibrate_head_tanh_temperature,
        )

        tokenizer = getattr(self, "encoder_tokenizer", None)
        if tokenizer is None:
            logger.info("phase2_kb_tanh_calibration_skipped", reason="no_tokenizer")
            return

        def _batch_to_content(batch):
            # batch is a list of KBSample objects post-_collate_kb. Pick
            # the first sample; tokenize its question text as a probe.
            if not batch:
                return None
            sample = batch[0]
            text = getattr(sample, "question", None) or ""
            if not text:
                return None
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                return None
            # Truncate to a sensible probe length to bound backbone cost.
            ids = ids[:512]
            token_ids = torch.tensor([ids], dtype=torch.long)
            mask = torch.ones_like(token_ids, dtype=torch.bool)
            return token_ids, mask

        for level in ("l0", "l1"):
            calibrated_T = calibrate_head_tanh_temperature(
                self.encoder,
                self.train_dataloader,
                self.device,
                level=level,
                n_probe_batches=n_probe_batches,
                batch_to_content=_batch_to_content,
            )
            if calibrated_T is not None:
                logger.info(
                    "head_tanh_temperature_calibrated",
                    level=level,
                    T=calibrated_T,
                    phase="phase2_kb",
                )
            else:
                logger.info(
                    "phase2_kb_tanh_calibration_skipped",
                    level=level,
                    reason="no_extractable_content",
                )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def _build_optimizer_groups(self) -> list[dict]:
        groups: list[dict] = []
        dec_params = [p for p in self.decoder.parameters() if p.requires_grad]
        if dec_params:
            groups.append({
                "params": dec_params,
                "lr": float(self.step_cfg.get(
                    "decoder_lr", self.step_cfg.get("lr", 1e-4)
                )),
            })
        # Stage A: L0 weights train directly (no LoRA wrapper).
        if self._live_l0:
            l0_params = [p for p in self.encoder.l0.parameters() if p.requires_grad]
            if l0_params:
                groups.append({
                    "params": l0_params,
                    "lr": float(
                        self.step_cfg.get("l0_lr", self.step_cfg.get("lr", 1e-4))
                    ),
                })
        for level in sorted(self.lora_router.levels):
            lvl_params = [
                p for p in self.lora_router.adapter_parameters(level) if p.requires_grad
            ]
            if not lvl_params:
                continue
            lr_key = f"{level}_lr"
            groups.append({
                "params": lvl_params,
                "lr": float(self.step_cfg.get(lr_key, self.step_cfg.get("lr", 1e-4))),
            })
        if self.topic_embeddings is not None:
            groups.extend(
                self.topic_embeddings.get_optimizer_groups(
                    float(self.step_cfg.get("topic_lr", self.step_cfg.get("lr", 1e-4))),
                ),
            )
        return groups

    # ------------------------------------------------------------------
    # Activation-checkpointed encoder forward
    # ------------------------------------------------------------------

    # Canonical positional argument order for a single LevelCompressor
    # forward (encoder.l0 or encoder.l1).
    _LEVEL_ARG_ORDER = (
        "content_embeddings",
        "content_cu_seqlens",
        "content_position_ids",
        "prompt_embeddings",
        "prompt_cu_seqlens",
        "prompt_position_ids",
        "pinned_positions",
        "target_ratio",
        "min_per_sample",
        "utility_grad_active",
        "utility_grad_capture",
    )

    def _checkpointed_level(self, level: str, **kwargs):
        """Call ``self.encoder.{level}(**kwargs)`` with optional activation checkpointing.

        ``level`` is ``"l0"`` or ``"l1"``. Falls back to a plain forward
        when checkpointing is disabled, when in eval mode, or when no
        input tensor tracks gradients.
        """
        lc = getattr(self.encoder, level)
        if not getattr(self, "_checkpoint_encoder", False):
            return lc(**kwargs)
        if not self.encoder.training:
            return lc(**kwargs)
        any_requires = any(
            isinstance(v, torch.Tensor) and v.requires_grad
            for v in kwargs.values()
        )
        if not any_requires:
            return lc(**kwargs)

        from torch.utils.checkpoint import checkpoint

        positional = tuple(kwargs.get(name) for name in self._LEVEL_ARG_ORDER)
        arg_names = self._LEVEL_ARG_ORDER

        def _forward(*args):
            return lc(**dict(zip(arg_names, args, strict=True)))

        if getattr(self, "_cpu_offload_activations", False):
            from bgkit.utils.cpu_offload_checkpoint import cpu_offload_checkpoint

            return cpu_offload_checkpoint(_forward, *positional, enabled=True)

        return checkpoint(_forward, *positional, use_reentrant=False)

    # ------------------------------------------------------------------
    # L0 — live or cached
    # ------------------------------------------------------------------

    def _l0_retention_for(self, dataset: str) -> float:
        """Return the current L0 retention ratio for ``dataset``.

        Supports two config shapes::

            # Static (same ratio throughout training):
            l0_retention:
              pubmedqa: 0.20

            # Curriculum (linear ramp from start → end over ramp_steps):
            l0_retention:
              pubmedqa:
                start: 0.20
                end: 0.01
                ramp_steps: 15000

        Curriculum interpolates by ``self.global_step``. After
        ``ramp_steps`` the ratio stays at ``end``.
        """
        entry = self._l0_retention.get(dataset)
        if entry is None:
            return float(self.step_cfg.get("default_l0_retention", 0.10))
        if isinstance(entry, (int, float)):
            return float(entry)
        start = float(entry.get("start", 0.10))
        end = float(entry.get("end", start))
        ramp = max(1, int(entry.get("ramp_steps", 1)))
        t = min(1.0, self.global_step / ramp)
        return start + (end - start) * t

    def _sample_l0_retention_for(self, dataset: str) -> float:
        """Sample an L0 retention ratio around the dataset's current base rate."""
        ratio = sample_ratio(
            rng=self._l0_ratio_rng,
            config=self._l0_ratio_sampler_cfg,
            base_ratio=self._l0_retention_for(dataset),
            is_evaluating=not self.encoder.training,
            override_active=False,
        )
        if self.encoder.training:
            self._step_sampled_l0_ratios.append(float(ratio))
        return float(ratio)

    def _sample_l1_retention(self) -> float:
        """Sample an L1 retention ratio around the configured base rate."""
        ratio = sample_ratio(
            rng=self._l1_ratio_rng,
            config=self._l1_ratio_sampler_cfg,
            base_ratio=self._l1_retention,
            is_evaluating=not self.encoder.training,
            override_active=False,
        )
        if self.encoder.training:
            self._step_sampled_l1_ratios.append(float(ratio))
        return float(ratio)

    def _live_l0_encode(
        self,
        dataset: str,
        article_ids: list[str],
    ):
        """Run the encoder live on each article's tokens to produce L0 survivors.

        Stage A only. Tokens are fetched by ``document_id`` from the canonical
        Phase 2 mmap layout via :class:`ArticleTokenStore` — the same files
        the single-doc Phase 2 datasets read from, so there is no duplicate
        token store to maintain.

        Packed-attention form: every article's token sequence is concatenated
        into a flat ``(N_content,)`` buffer with per-article ``cu_seqlens``.
        No padding tokens are constructed. The survivorship head inside the
        encoder produces the survivor mask internally based on the target
        retention ratio.

        Returns ``(out, content_cu_seqlens, ratio)`` where ``out`` is the
        :class:`bgkit.models.level_compressor.LevelOutput` (carrying flat
        head outputs + ``base_raw_for_util`` / ``post_head_content_values``
        / ``_utility_grad_state``) and ``content_cu_seqlens`` marks the
        per-article boundaries inside the flat buffer, so downstream aux
        losses can do segment-aware reductions without repacking.
        """
        if self._token_store is None:
            raise RuntimeError(
                "_live_l0_encode called but ArticleTokenStore is unset; "
                "this path is only valid when live_l0=True."
            )
        # Pull variable-length token sequences one article at a time and
        # concatenate flat. The ArticleTokenStore exposes ``get()`` for
        # single-article access (and its ``get_batch`` would pad — which
        # we don't want).
        token_seqs = [
            self._token_store.get(dataset, aid).to(self.device)
            for aid in article_ids
        ]
        lengths = [int(seq.size(0)) for seq in token_seqs]
        tokens_flat = torch.cat(token_seqs, dim=0)  # (N_content,)
        cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=self.device)
        cu_seqlens[1:] = torch.tensor(
            lengths, dtype=torch.int32, device=self.device,
        ).cumsum(0)
        position_ids = position_ids_from_cu(cu_seqlens, int(tokens_flat.shape[0]))

        embed_tokens = self.encoder.l0.backbone.get_input_embeddings()
        input_embeddings = embed_tokens(tokens_flat)  # (N_content, D)

        ratio = self._sample_l0_retention_for(dataset)
        from bgkit.training.survivorship_helpers import LevelLossCfg as _LLC
        util_active = getattr(
            self, "_surv_l0", _LLC(),
        ).utility_grad_loss_weight > 0.0
        grad_capture: dict | None = {} if util_active else None

        out = self._checkpointed_level(
            "l0",
            content_embeddings=input_embeddings,
            content_cu_seqlens=cu_seqlens,
            content_position_ids=position_ids,
            target_ratio=ratio,
            utility_grad_active=util_active,
            utility_grad_capture=grad_capture,
        )
        if grad_capture is not None:
            out._l0_grad_capture = grad_capture  # type: ignore[attr-defined]
        return out, cu_seqlens, ratio

    def _l0_for_articles(
        self, dataset: str, article_ids: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return packed L0 survivors for each article.

        Packed form:

        - ``l0_flat``: ``(N_survivors, D)`` flat concatenation of per-article
          L0 survivor rows. ``N_survivors = sum(K_i)``.
        - ``cu_seqlens``: ``(B+1,)`` int32, per-article boundaries in the
          flat buffer. ``cu_seqlens[i+1] - cu_seqlens[i] == K_i``.

        In the cached path the rows come from :class:`L0Cache`
        (variable-length per article, concatenated flat); in the live path
        they come from :meth:`_live_l0_encode` and we keep the encoder's
        own ``survivor_cu_seqlens`` unchanged.

        When the live path runs, the encoder output is appended to
        ``self._pending_l0_outputs`` with the pre-compression
        ``content_cu_seqlens`` so the trainer can apply survivorship
        auxiliary losses (aggregate ratio, decisiveness, min-survivors,
        utility-grad BCE) after the main forward.
        """
        if self._live_l0:
            out, content_cu, ratio = self._live_l0_encode(dataset, article_ids)
            survivors = out.survivor_embeddings  # (N_survivors, D)
            cu_seqlens = out.survivor_cu_seqlens  # (B+1,)
            if self.encoder.training and hasattr(self, "_pending_l0_outputs"):
                self._pending_l0_outputs.append({
                    "dataset": dataset,
                    "enc_out": out,
                    "ratio": ratio,
                    # Packed pre-compression content cu_seqlens so aux
                    # losses can do segment-aware reductions.
                    "cu_seqlens": content_cu,
                })
            return survivors, cu_seqlens
        if self._l0_cache is None:
            raise RuntimeError("L0 cache is None but live_l0 is False")
        # Cached path: pull each article's variable-length rows from the
        # on-disk cache and pack them into a single flat buffer with
        # per-article cu_seqlens.
        rows_list: list[torch.Tensor] = [
            self._l0_cache.get(dataset, aid) for aid in article_ids
        ]
        lengths = [int(r.size(0)) for r in rows_list]
        if sum(lengths) == 0:
            hidden = self.encoder.l0.hidden_dim
            flat = torch.zeros((0, hidden), dtype=torch.bfloat16, device=self.device)
        else:
            flat = torch.cat(rows_list, dim=0).to(self.device, dtype=torch.bfloat16)
        cu_seqlens = torch.zeros(
            len(lengths) + 1, dtype=torch.int32, device=self.device,
        )
        cu_seqlens[1:] = torch.tensor(
            lengths, dtype=torch.int32, device=self.device,
        ).cumsum(0)
        return flat, cu_seqlens

    # ------------------------------------------------------------------
    # L1 — query-conditioned with ID pinning
    # ------------------------------------------------------------------

    def _resolve_article_ids(
        self, dataset: str, tag_or_article_ids: list[str],
    ) -> list[str]:
        """Resolve a bgkit turn's ``ids`` argument to a flat list of articles.

        Expansion happens against the browse tree (which for title-keyed
        datasets such as KILT and PubMedQA contains human-readable
        article ids), and the resulting browse-tree ids are then
        translated into canonical mmap ``document_id`` strings via the
        per-dataset ``title → document_id`` sidecar. Returning mmap keys
        keeps every downstream caller —
        :class:`ArticleTokenStore`, :class:`L0Cache`, and the
        trajectory-article coverage check — working on a single stable
        id space, and shields them from ever seeing the browse-tree's
        human-readable titles.

        The returned list is pre-filtered against whichever L0 source is
        active for this stage (live ``ArticleTokenStore`` at Stage A,
        cached ``L0Cache`` at Stages B/C) so callers never see document
        ids that would crash a downstream lookup.
        """
        tree = self._trees[dataset]
        browse_ids: list[str] = []
        for raw in tag_or_article_ids:
            if raw not in tree:
                parent = tree.leaf_tag_for_article(raw)
                if parent is not None:
                    browse_ids.append(raw)
                continue
            node = tree.get(raw)
            if node.is_article:
                browse_ids.append(raw)
            elif node.is_leaf_tag:
                browse_ids.extend(node.articles)
            else:
                browse_ids.extend(tree.articles(raw))
        document_ids = self._article_ids_to_document_ids(dataset, browse_ids)
        return self._filter_missing_articles(dataset, document_ids)

    def _filter_missing_articles(
        self, dataset: str, article_ids: list[str],
    ) -> list[str]:
        """Verify every article exists in the active L0 source, else raise.

        After :meth:`_validate_trajectory_article_coverage` has run in setup,
        every article any trajectory can reference MUST be present in either
        the live token store (Stage A) or the pre-computed L0 cache (Stages
        B/C). Missing articles at training time indicate a drift between
        the pre-compute set and the trainer's article resolution, which is
        a data-pipeline bug — we fail loudly rather than silently drop and
        degrade the training signal.
        """
        if not article_ids:
            return article_ids
        if self._live_l0 and self._token_store is not None:
            def source_has(aid: str) -> bool:
                return self._token_store.has(dataset, aid)
        elif self._l0_cache is not None:
            def source_has(aid: str) -> bool:
                return self._l0_cache.has(dataset, aid)
        else:
            return article_ids

        missing = [aid for aid in article_ids if not source_has(aid)]
        if missing:
            raise RuntimeError(
                f"phase2_kb: {len(missing)} article IDs in dataset {dataset!r} "
                f"are not in the active L0 source. Sample: {missing[:5]}. "
                "This indicates drift between build_trajectory_set + "
                "precompute_l0_subset and the trainer's article resolution. "
                "Re-run the data prep pipeline (build_browse_tree → "
                "build_teacher_trajectories → build_trajectory_set → "
                "precompute_l0_subset) against the same browse tree the "
                "trainer is using."
            )
        return article_ids

    def _validate_trajectory_article_coverage(self) -> None:
        """Walk every training + eval trajectory at setup time and assert
        that every article they can reference is present in the L0 source.

        This catches data-pipeline drift before training starts rather than
        letting it manifest as a cryptic RuntimeError mid-step. We scan the
        full train/eval split once; for each bgkit turn we resolve the
        referenced tag/article IDs against the browse tree and check the
        L0 source. Any missing article aborts setup with a detailed report.

        Runs only when an L0 source is configured (otherwise nothing to
        check). Logs progress so the one-time cost is visible.
        """
        if self._live_l0 and self._token_store is None:
            return
        if not self._live_l0 and self._l0_cache is None:
            return

        logger.info("phase2_kb_coverage_scan_start")
        missing_by_dataset: dict[str, list[str]] = {}
        checked_articles: dict[str, set[str]] = {}

        def _check_sample(sample: KBSample) -> None:
            for turn in sample.trajectory:
                if turn.kind != "bgkit":
                    continue
                tag_ids = list(turn.args.get("ids", []))
                if not tag_ids:
                    continue
                # Resolve without invoking the fail-loud filter.
                tree = self._trees[sample.dataset_name]
                browse_ids: list[str] = []
                for raw in tag_ids:
                    if raw not in tree:
                        parent = tree.leaf_tag_for_article(raw)
                        if parent is not None:
                            browse_ids.append(raw)
                        continue
                    node = tree.get(raw)
                    if node.is_article:
                        browse_ids.append(raw)
                    elif node.is_leaf_tag:
                        browse_ids.extend(node.articles)
                    else:
                        browse_ids.extend(tree.articles(raw))
                # Translate browse-tree ids (titles, for KILT/PubMedQA)
                # into canonical mmap document_ids before checking the
                # L0 source, which is keyed by document_id.
                article_ids = self._article_ids_to_document_ids(
                    sample.dataset_name, browse_ids,
                )

                seen = checked_articles.setdefault(sample.dataset_name, set())
                if self._live_l0 and self._token_store is not None:
                    has = self._token_store.has
                else:
                    has = self._l0_cache.has  # type: ignore[union-attr]
                for aid in article_ids:
                    if aid in seen:
                        continue
                    seen.add(aid)
                    if not has(sample.dataset_name, aid):
                        missing_by_dataset.setdefault(
                            sample.dataset_name, [],
                        ).append(aid)

        for split_name, split in (
            ("train", self.train_dataset), ("eval", self.eval_dataset),
        ):
            n = len(split)
            for idx in range(n):
                sample = split[idx]
                _check_sample(sample)
            logger.info(
                "phase2_kb_coverage_scan_split",
                split=split_name,
                samples=n,
            )

        if missing_by_dataset:
            details = {
                ds: {"count": len(ids), "sample": ids[:5]}
                for ds, ids in missing_by_dataset.items()
            }
            raise RuntimeError(
                "phase2_kb: trajectory article coverage check failed. "
                "The L0 source is missing articles that trajectories "
                "reference. This is a data-pipeline drift — rebuild the "
                "L0 pre-compute with the same browse tree and trajectory "
                f"set the trainer is using. Missing: {details}"
            )
        total_checked = sum(len(s) for s in checked_articles.values())
        logger.info(
            "phase2_kb_coverage_scan_ok",
            datasets=len(checked_articles),
            unique_articles=total_checked,
        )

    def _sample_distractors(
        self, dataset: str, gold_article_ids: list[str], n: int,
    ) -> list[str]:
        """Sample up to ``n`` distractor article mmap IDs for the given dataset.

        Distractors are articles NOT in ``gold_article_ids``. Prefers siblings
        under the same leaf tag as a gold article; falls back to articles
        from random leaves elsewhere. Returns mmap document_ids.
        """
        if n <= 0:
            return []
        browse_trees = getattr(self, "_trees", None) or {}
        tree: BrowseTree | None = browse_trees.get(dataset)
        if tree is None:
            return []

        # Translate gold mmap IDs back to browse-tree IDs if we have the sidecar
        doc_to_title = getattr(self, "_doc_id_to_title", {}) or {}
        title_to_doc = getattr(self, "_title_to_doc_id", {}) or {}
        rev = doc_to_title.get(dataset, {}) if doc_to_title else {}
        gold_tree_ids = {rev.get(aid, aid) for aid in gold_article_ids}

        # Find sibling articles (same leaf tag as any gold article)
        candidate_pool: list[str] = []
        for gold_tree_id in gold_tree_ids:
            try:
                leaf_tag = tree.leaf_tag_for_article(gold_tree_id)
            except Exception:
                leaf_tag = None
            if leaf_tag is None:
                continue
            try:
                siblings = tree.articles(leaf_tag)
            except Exception:
                siblings = []
            for sib in siblings:
                if sib not in gold_tree_ids:
                    candidate_pool.append(sib)

        # Fallback: pick articles from other leaf tags in the tree
        if len(candidate_pool) < n:
            all_nodes = getattr(tree, "_nodes", {}) or {}
            other_leaves: list[str] = []
            for node_id, node in all_nodes.items():
                try:
                    if node.is_leaf_tag:
                        for art in node.articles:
                            if art not in gold_tree_ids:
                                other_leaves.append(art)
                except Exception:
                    continue
            candidate_pool.extend(other_leaves)

        if not candidate_pool:
            return []

        # Deduplicate, preserving sibling-first ordering
        seen: set[str] = set()
        ordered: list[str] = []
        for c in candidate_pool:
            if c not in seen:
                seen.add(c)
                ordered.append(c)

        import random as _random
        rng = _random.Random(self.global_step)
        sampled_tree_ids = rng.sample(ordered, min(n, len(ordered)))

        # Translate back to mmap document IDs via title sidecar
        name_map = title_to_doc.get(dataset, {}) if title_to_doc else {}
        result: list[str] = []
        for tid in sampled_tree_ids:
            result.append(name_map.get(tid, tid))
        return result

    def _prepare_l1_turn(
        self, dataset: str, tag_or_article_ids: list[str], query: str,
        distractor_ids: list[str] | None = None,
    ) -> dict | None:
        """Build per-turn packed content + query tensors without running the encoder.

        Returns a dict with fields:

        - ``content``: ``(L_content, D)`` flat concatenation of pinned-id
          + L0-survivor blocks across all resolved articles (gold +
          distractors, in that order).
        - ``pinned``: ``(L_content,)`` bool marking positions that must
          survive the L1 head (pinned-ID tokens of gold articles).
        - ``relevance_mask``: ``(L_content,)`` bool — True inside gold
          articles, False inside distractors. Drives the relevance loss.
        - ``query_emb``: ``(L_query, D)`` flat query embeddings.

        Returns ``None`` if the turn has no articles (the caller emits an
        empty L1 output).

        The survivor mask is produced internally by the encoder's
        survivorship head based on ``target_ratio`` and ``pinned_positions``.
        """
        article_ids = self._resolve_article_ids(dataset, tag_or_article_ids)
        if not article_ids:
            return None

        # Distractor sampling (only during training, when configured).
        # Use getattr for resilience when tests bypass __init__.
        n_distractors = getattr(self, "_n_distractors", 0)
        use_distractors = (
            distractor_ids is None
            and self.encoder.training
            and n_distractors > 0
        )
        if distractor_ids is None and use_distractors:
            distractor_ids = self._sample_distractors(
                dataset, article_ids, n_distractors,
            )
        elif distractor_ids is None:
            distractor_ids = []

        # Combine gold + distractors, track which are relevant
        all_ids = list(article_ids) + list(distractor_ids)
        is_relevant_per_article = (
            [True] * len(article_ids) + [False] * len(distractor_ids)
        )

        # Packed L0 survivors for all articles in this turn.
        l0_flat, l0_cu = self._l0_for_articles(dataset, all_ids)
        l0_lengths = lengths_from_cu(l0_cu).tolist()

        rev_maps = getattr(self, "_doc_id_to_title", None) or {}
        doc_to_title = rev_maps.get(dataset, {}) if rev_maps else {}
        pin_texts = [doc_to_title.get(aid, aid) for aid in all_ids]

        id_token_lists: list[list[int]] = []
        for pin_text in pin_texts:
            ids = self.encoder_tokenizer.encode(
                f" {pin_text}", add_special_tokens=False,
            )
            if not ids:
                ids = [0]
            id_token_lists.append(ids)

        embed_tokens = self.encoder.l0.backbone.get_input_embeddings()

        pieces: list[torch.Tensor] = []
        pinned_list: list[bool] = []
        relevance_list: list[bool] = []
        for i, aid_tokens in enumerate(id_token_lists):
            is_relevant = is_relevant_per_article[i]
            # Pin ID tokens for GOLD articles only. Distractor ID tokens are
            # not pinned so the head can learn to drop them.
            pin_these = is_relevant
            id_ids = torch.tensor(aid_tokens, dtype=torch.long, device=self.device)
            id_emb = embed_tokens(id_ids).to(l0_flat.dtype)
            pieces.append(id_emb)
            pinned_list.extend([pin_these] * len(aid_tokens))
            relevance_list.extend([is_relevant] * len(aid_tokens))

            k_i = int(l0_lengths[i]) if i < len(l0_lengths) else 0
            if k_i > 0:
                start = int(l0_cu[i].item())
                pieces.append(l0_flat[start : start + k_i].to(l0_flat.dtype))
                pinned_list.extend([False] * k_i)
                relevance_list.extend([is_relevant] * k_i)

        content = torch.cat(pieces, dim=0)  # (L_content, D)
        pinned = torch.tensor(pinned_list, dtype=torch.bool, device=self.device)
        relevance_mask = torch.tensor(
            relevance_list, dtype=torch.bool, device=self.device,
        )

        # Query prompt
        q_ids = self.encoder_tokenizer.encode(query, add_special_tokens=False)
        if not q_ids:
            q_ids = [0]
        q_tensor = torch.tensor(q_ids, dtype=torch.long, device=self.device)
        q_emb = embed_tokens(q_tensor).to(content.dtype)  # (L_query, D)

        return {
            "content": content,
            "pinned": pinned,
            "relevance_mask": relevance_mask,
            "query_emb": q_emb,
        }

    def _run_l1_batch(
        self,
        prepared: list[dict | None],
        target_ratio: float | None = None,
    ) -> list[torch.Tensor]:
        """Run a packed encoder forward for every non-None turn in ``prepared``.

        All turns from one sample (or one bucket when called from
        :meth:`_forward_backward`) are packed into a single flat
        ``(N_content, D)`` content buffer + ``(N_query, D)`` query buffer.
        Per-turn segmentation is encoded in ``cu_seqlens``; no padding
        tokens appear anywhere in the flat buffers.

        The survivor mask is produced internally by the encoder's
        survivorship head based on ``target_ratio`` and ``pinned_positions``.

        Returns a list matching ``prepared`` by index. Entries for None turns
        fall back to a 1-vector zero tensor so the decoder sentinel splice
        always has something to drop in.
        """
        hidden_dim = self.encoder.l1.hidden_dim
        zero_fallback = torch.zeros(
            (1, hidden_dim), device=self.device, dtype=torch.bfloat16,
        )
        non_null = [t for t in prepared if t is not None]
        if not non_null:
            return [zero_fallback for _ in prepared]

        target_dtype = non_null[0]["content"].dtype
        batch_size = len(non_null)

        # Pack all turns' content + query flat, with per-turn cu_seqlens.
        content_pieces: list[torch.Tensor] = [t["content"] for t in non_null]
        query_pieces: list[torch.Tensor] = [t["query_emb"] for t in non_null]
        pinned_pieces: list[torch.Tensor] = [t["pinned"] for t in non_null]
        relevance_pieces: list[torch.Tensor] = [
            t["relevance_mask"].to(self.device) for t in non_null
        ]

        content_lengths = [int(c.size(0)) for c in content_pieces]
        query_lengths = [int(q.size(0)) for q in query_pieces]

        content_flat = torch.cat(content_pieces, dim=0).to(target_dtype)
        query_flat = torch.cat(query_pieces, dim=0).to(target_dtype)
        pinned_flat = torch.cat(pinned_pieces, dim=0).to(self.device)
        relevance_flat = torch.cat(relevance_pieces, dim=0)

        content_cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        content_cu[1:] = torch.tensor(
            content_lengths, dtype=torch.int32, device=self.device,
        ).cumsum(0)
        query_cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        query_cu[1:] = torch.tensor(
            query_lengths, dtype=torch.int32, device=self.device,
        ).cumsum(0)

        content_pos_ids = position_ids_from_cu(content_cu, int(content_flat.shape[0]))
        query_pos_ids = position_ids_from_cu(query_cu, int(query_flat.shape[0]))

        from bgkit.training.survivorship_helpers import LevelLossCfg as _LLC
        util_active_l1 = getattr(
            self, "_surv_l1", _LLC(),
        ).utility_grad_loss_weight > 0.0
        l1_grad_capture: dict | None = {} if util_active_l1 else None
        if target_ratio is None:
            target_ratio = self._sample_l1_retention()

        # Bridge L0 survivors + pinned ID embeddings through L0's
        # auto_repro_head so L1's independent backbone (a deepcopy of L0
        # at construction) sees its expected input-embedding distribution.
        bridged_content = self.encoder.l0.auto_reproduce(content_flat)
        # Activate L1 LoRA so wrapped Linears in L1's backbone apply the
        # adapter delta during forward.
        with self.lora_router.active("l1"):
            out = self._checkpointed_level(
                "l1",
                content_embeddings=bridged_content,
                content_cu_seqlens=content_cu,
                content_position_ids=content_pos_ids,
                prompt_embeddings=query_flat,
                prompt_cu_seqlens=query_cu,
                prompt_position_ids=query_pos_ids,
                pinned_positions=pinned_flat,
                target_ratio=target_ratio,
                utility_grad_active=util_active_l1,
                utility_grad_capture=l1_grad_capture,
            )
        if l1_grad_capture is not None:
            out._l1_grad_capture = l1_grad_capture  # type: ignore[attr-defined]

        # Stash encoder output for aux loss computation (when training).
        # Everything is flat: relevance_mask and pinned are flat (N_content,)
        # bool, cu_seqlens is the per-turn segmentation of the flat buffer.
        if self.encoder.training and hasattr(self, "_pending_l1_outputs"):
            self._pending_l1_outputs.append({
                "enc_out": out,
                "cu_seqlens": content_cu,
                "pinned": pinned_flat,
                "relevance_mask": relevance_flat,
                "ratio": target_ratio,
            })

        # Project L1 survivors into decoder space via the shared
        # projection block. L1's raw survivor_embeddings are post-norm
        # backbone hidden states; the decoder expects projected vectors.
        surv_emb = out.survivor_embeddings  # (N_surv_total, D)
        surv_cu_t = out.survivor_cu_seqlens
        if int(surv_cu_t[-1].item()) > 0:
            from bgkit.utils.packing import lengths_from_cu as _lfc
            from bgkit.utils.packing import position_ids_from_cu as _pfc

            surv_lengths = _lfc(surv_cu_t).to(torch.int64)
            surv_max = int(surv_lengths.max().item()) if surv_lengths.numel() else 0
            surv_pos = _pfc(surv_cu_t, int(surv_emb.shape[0]))
            proj_out = self.encoder.projection_block(
                surv_emb,
                cu_seqlens=surv_cu_t,
                max_seqlen=surv_max,
                position_ids=surv_pos,
                survivor_mask=None,
            )
            projected = proj_out.projected_embeddings
        else:
            projected = surv_emb

        # Extract per-turn projected survivors via per-turn boundaries.
        surv_cu = surv_cu_t.to(torch.int64).tolist()
        per_turn: list[torch.Tensor] = []
        for i in range(batch_size):
            start = int(surv_cu[i])
            end = int(surv_cu[i + 1])
            if end <= start:
                per_turn.append(zero_fallback)
            else:
                per_turn.append(projected[start:end])

        # Re-interleave with None fallbacks
        results: list[torch.Tensor] = []
        it = iter(per_turn)
        for t in prepared:
            if t is None:
                results.append(zero_fallback)
            else:
                results.append(next(it))
        return results

    # ------------------------------------------------------------------
    # Decoder segment construction
    # ------------------------------------------------------------------

    def _build_decoder_segments_with_trace(
        self, sample: KBSample,
    ) -> tuple[list[Segment], _KBDecodeTrace]:
        """Like :meth:`_build_decoder_segments`, but also returns a trace
        object carrying the concat-coordinate spans of the answer turn
        plus every browse/bgkit tool-call emission.

        The trace is used by the KB-scale eval harness
        (:mod:`bgkit.eval.kb_trajectory_eval`) to score tool-call ID
        accuracy and trajectory step accuracy without reimplementing any
        of the remap logic.
        """
        return self._build_decoder_segments_core(sample)

    def _build_decoder_segments(
        self, sample: KBSample,
    ) -> tuple[list[Segment], tuple[int, int] | None]:
        """Walk a rendered trajectory, produce interleaved decoder segments.

        Each run of non-sentinel tokens becomes a :class:`TokenSegment`
        carrying the trajectory's per-turn loss flags; each ``bgkit``
        sentinel becomes an :class:`EmbeddingSegment` whose contents are
        the live L1 survivors for that call. Gradient flows through the L1
        encoder for every bgkit turn (primary and exploration alike — the
        trajectory's loss flags live in the token segments, not the
        embedding segments).

        All bgkit turns of a single sample run through one batched L1
        encoder call (padded to the max content/query length across turns).
        This amortizes the per-call encoder launch overhead across the
        trajectory's ~3-4 bgkit calls.

        Returns:
            ``(segments, answer_span)`` where ``answer_span`` is the
            ``[start, end)`` range of the final ``answer`` turn in the
            *concatenated segment sequence* (after topic embeddings and
            bgkit survivors have been spliced in), or ``None`` if the
            trajectory has no answer turn. The trainer's eval path uses
            this range to compute EM/F1 over only the answer tokens
            (excluding browse/bgkit tool call emissions that also bear
            loss).
        """
        segments, trace = self._build_decoder_segments_core(sample)
        return segments, trace.answer_span

    def _build_decoder_segments_core(
        self, sample: KBSample,
    ) -> tuple[list[Segment], _KBDecodeTrace]:
        """Shared implementation for :meth:`_build_decoder_segments` and
        :meth:`_build_decoder_segments_with_trace`.

        Produces the segment list and a trace of concat-coordinate spans
        that downstream eval harnesses can use without reimplementing
        the rendered-to-concat remap.

        This is the single-sample path: it prepares every bgkit turn's L1
        input, runs a single batched encoder forward within this sample,
        then assembles segments. The training loop
        (:meth:`_forward_backward`) uses a different path that batches
        L1 turns across the *entire* training batch via
        :meth:`_assemble_sample_segments`.
        """
        with self._training_ablation_override():
            prep = self._prepare_sample_for_decode(sample)
            all_survivors = self._run_l1_batch(prep["prepared_turns"])
            return self._assemble_sample_segments(prep, all_survivors)

    @contextlib.contextmanager
    def _training_ablation_override(self):
        """Randomly ablate context during training for capability regression
        prevention (plan 4.5 + 5.6). Each training sample rolls independently:

        - ``p_skip_bgkit``: drop all bgkit survivors for this sample.
        - ``p_skip_topic``: drop the topic embedding block.
        - ``p_noise_bgkit``: replace bgkit survivors with small Gaussian
          noise (less destructive than zeroing).

        The rolled mode is applied by temporarily overriding
        ``self._ablation_mode`` for the duration of the sample's
        segment build. Rolls are disabled when the trainer is in eval
        mode (``self.model is None`` or ``not self.model.training``).
        Explicit eval-time ablation modes (set via
        :meth:`set_ablation_mode`) take precedence over training rolls.
        """
        # Don't override if the caller explicitly set an ablation mode
        # (e.g., eval-time ablation sweep). Respect their choice.
        if self._ablation_mode is not None:
            yield
            return
        # Only apply training-time rolls when we're actually training.
        model = getattr(self, "model", None)
        is_training = (
            model is not None and getattr(model, "training", False)
        )
        if not is_training:
            yield
            return

        original = self._ablation_mode
        skip_bgkit = (
            self._p_skip_bgkit_training > 0
            and self._ablation_rng.random() < self._p_skip_bgkit_training
        )
        skip_topic = (
            self._p_skip_topic_training > 0
            and self._ablation_rng.random() < self._p_skip_topic_training
        )
        noise_bgkit = (
            not skip_bgkit
            and self._p_noise_bgkit_training > 0
            and self._ablation_rng.random() < self._p_noise_bgkit_training
        )

        if skip_bgkit and skip_topic:
            rolled = self.ABLATION_NEITHER
        elif skip_bgkit:
            rolled = self.ABLATION_TOPICS_ONLY
        elif skip_topic:
            rolled = self.ABLATION_NO_TOPICS
        elif noise_bgkit:
            rolled = self.ABLATION_NOISE
        else:
            rolled = None
        self._ablation_mode = rolled
        try:
            yield
        finally:
            self._ablation_mode = original

    def _prepare_sample_for_decode(self, sample: KBSample) -> dict:
        """Render the sample and prepare all per-turn L1 inputs.

        Produces a dict with everything the downstream assembly step needs:
        rendered trajectory, tokenized ids + loss mask, sentinel lengths,
        the prepared L1 turn dicts (each returned by
        :meth:`_prepare_l1_turn`), the topic embedding block (if any),
        and the corresponding topic sentinel length. Does NOT run the
        encoder — that's deferred so the caller can batch across samples.

        When topic embeddings are enabled and the sample has at least one
        known tag, ``tokenize_trajectory`` is called with
        ``topic_knowledge_tags=[...]`` and the resulting rendered sequence
        contains a ``BGKIT_TOPIC_SENTINEL`` at a known absolute position.
        The trainer splices the pre-computed topic embedding block into
        that sentinel at assembly time, same as bgkit L1 survivors.
        """
        system_prompt = self._system_prompt_for(sample)

        # Build the topic embedding block up-front so we know whether to
        # ask the tokenizer to inject a topic-knowledge tool-call pair.
        topic_block = self._topic_embedding_segment(sample)
        if topic_block is not None:
            topic_tags = self._sample_tags_for(sample)
            if not topic_tags:
                topic_block = None
                topic_tags = []
        else:
            topic_tags = []

        rendered = tokenize_trajectory(
            self.tokenizer,
            system_prompt,
            sample.question,
            sample.trajectory,
            topic_knowledge_tags=topic_tags or None,
        )
        token_ids = rendered.token_ids.to(self.device)
        loss_mask = rendered.loss_mask.to(self.device)

        sentinel_ids = self.tokenizer.encode(BGKIT_SENTINEL, add_special_tokens=False)
        sentinel_len = len(sentinel_ids)
        topic_sentinel_ids = self.tokenizer.encode(
            BGKIT_TOPIC_SENTINEL, add_special_tokens=False,
        )
        topic_sentinel_len = len(topic_sentinel_ids)

        prepared_turns: list[dict | None] = [
            self._prepare_l1_turn(
                sample.dataset_name,
                list(turn.args.get("ids", [])),
                str(turn.args.get("query", "")),
            )
            for turn in rendered.bgkit_turns
        ]

        return {
            "sample": sample,
            "rendered": rendered,
            "token_ids": token_ids,
            "loss_mask": loss_mask,
            "sentinel_len": sentinel_len,
            "topic_sentinel_len": topic_sentinel_len,
            "prepared_turns": prepared_turns,
            "topic_block": topic_block,
        }

    def _assemble_sample_segments(
        self, prep: dict, survivors_per_turn: list[torch.Tensor],
    ) -> tuple[list[Segment], _KBDecodeTrace]:
        """Assemble decoder segments given pre-computed L1 survivors.

        Args:
            prep: output of :meth:`_prepare_sample_for_decode`
            survivors_per_turn: list of (K_i, D) tensors, one per bgkit
                turn in ``prep["rendered"].bgkit_turns``.

        Returns ``(segments, trace)`` — same shape as
        :meth:`_build_decoder_segments_core`.

        The assembly walks two kinds of splice points in rendered-token
        order: the (optional) topic-knowledge sentinel and every bgkit
        sentinel. At each splice it emits the preceding token segment,
        the embedding segment to drop in, and advances the cursor past
        the sentinel. Every rendered-to-concat coordinate shift applied
        to the answer span and tool-call spans is computed from the
        SAME list of splice deltas.
        """
        rendered = prep["rendered"]
        token_ids = prep["token_ids"]
        loss_mask = prep["loss_mask"]
        sentinel_len = prep["sentinel_len"]
        topic_sentinel_len = prep["topic_sentinel_len"]
        topic_block = prep["topic_block"]

        decoder_hidden = self.decoder.hidden_dim

        # Ablation switches
        skip_survivors = self._ablation_mode in (
            self.ABLATION_TOPICS_ONLY, self.ABLATION_NEITHER,
        )
        skip_topic = self._ablation_mode in (
            self.ABLATION_NO_TOPICS, self.ABLATION_NEITHER,
        )

        # Build an ordered splice event list: (start_rendered, kind, payload).
        # kind is "topic" or "bgkit". payload is the embedding tensor to
        # splice in (shape (K, D)) for bgkit, or the topic block tensor
        # (shape (1, P, D) already batched) for topic.
        splice_events: list[tuple[int, str, torch.Tensor]] = []
        if (
            topic_block is not None
            and not skip_topic
            and rendered.topic_sentinel_position is not None
        ):
            splice_events.append(
                (rendered.topic_sentinel_position, "topic", topic_block.embeddings[0]),
            )
        for survivors, start in zip(
            survivors_per_turn, rendered.bgkit_sentinel_positions, strict=True,
        ):
            if survivors.size(-1) != decoder_hidden:
                raise RuntimeError(
                    f"survivor hidden dim {survivors.size(-1)} != decoder hidden "
                    f"dim {decoder_hidden}; add a cache_projection."
                )
            survivors = self._apply_context_ablation(survivors, skip=skip_survivors)
            splice_events.append((start, "bgkit", survivors))
        splice_events.sort(key=lambda e: e[0])

        # Cumulative deltas at each splice point, used for span remapping.
        # Each entry is (rendered_end_position, concat_delta_after_splice).
        cumulative_deltas: list[tuple[int, int]] = []
        running_delta = 0

        segments: list[Segment] = []
        cursor = 0
        for start, kind, payload in splice_events:
            sentinel_tok_len = (
                topic_sentinel_len if kind == "topic" else sentinel_len
            )
            end = start + sentinel_tok_len

            if start > cursor:
                segments.append(TokenSegment(
                    token_ids=token_ids[cursor:start].unsqueeze(0),
                    loss_mask=loss_mask[cursor:start].unsqueeze(0),
                ))
            n_emb = int(payload.size(0))
            segments.append(EmbeddingSegment(embeddings=payload.unsqueeze(0)))
            running_delta += n_emb - sentinel_tok_len
            cumulative_deltas.append((end, running_delta))
            cursor = end

        if cursor < token_ids.size(0):
            segments.append(TokenSegment(
                token_ids=token_ids[cursor:].unsqueeze(0),
                loss_mask=loss_mask[cursor:].unsqueeze(0),
            ))
        elif not segments:
            # No splice events and no prefix — only possible if the
            # trajectory is empty. Fall through with an empty token
            # segment so the decoder call fails loudly rather than
            # returning garbage.
            segments.append(TokenSegment(
                token_ids=token_ids.unsqueeze(0),
                loss_mask=loss_mask.unsqueeze(0),
            ))

        def _remap(span: tuple[int, int]) -> tuple[int, int]:
            """Translate a rendered-trajectory span into concat coords.

            For a rendered position ``p``, apply the cumulative delta from
            every splice whose END is <= p. Splice events are monotonic
            by start position so we can early-out as soon as we find one
            that starts after ``p``.
            """
            start, end = span
            delta = 0
            for splice_end, cum in cumulative_deltas:
                if splice_end <= start:
                    delta = cum
                else:
                    break
            return (start + delta, end + delta)

        answer_span_concat: tuple[int, int] | None = None
        if rendered.answer_span is not None:
            answer_span_concat = _remap(rendered.answer_span)
        browse_spans_concat = [_remap(span) for span in rendered.browse_call_spans]
        bgkit_spans_concat = [_remap(span) for span in rendered.bgkit_call_spans]

        trace = _KBDecodeTrace(
            answer_span=answer_span_concat,
            browse_turns=list(rendered.browse_turns),
            browse_call_spans=browse_spans_concat,
            bgkit_turns=list(rendered.bgkit_turns),
            bgkit_call_spans=bgkit_spans_concat,
        )
        return segments, trace

    def _system_prompt_for(self, sample: KBSample) -> str:
        # Auto-promote samples whose dataset's browse tree is flat to the
        # ``flat`` system prompt — even if the provenance script wrote
        # ``pre_scoped`` or ``topic_list``. The trainer is the source of
        # truth for whether browse navigation is meaningful, since it
        # owns the loaded BrowseTree objects. Flat trees with browse
        # references in the system prompt would mislead the decoder
        # into trying to navigate a non-navigable corpus.
        template = sample.scope_template
        tree = self._trees.get(sample.dataset_name)
        if tree is not None and tree.is_flat() and template != "flat":
            template = "flat"
        scope_desc = sample.scope_description
        if template == "flat" and not scope_desc:
            scope_desc = sample.dataset_name
        return make_system_prompt(
            template,
            topic_list=sample.topic_list or None,
            scope_description=scope_desc or None,
        )

    def _sample_tags_for(self, sample: KBSample) -> list[str]:
        """Derive the tag set for a sample. For KB-scale trajectories we
        extract tags from the bgkit turns' referenced tag IDs — each call's
        ``ids`` is a list of tag or article IDs, and the tag IDs are what
        the taxonomy indexes.
        """
        tags: list[str] = []
        for turn in sample.trajectory:
            if turn.kind == "bgkit":
                for tid in turn.args.get("ids", []):
                    tags.append(str(tid))
        return tags

    def _topic_embedding_segment(self, sample: KBSample) -> EmbeddingSegment | None:
        """Build a single-sample topic embedding EmbeddingSegment or None.

        Returns None when topic embeddings are disabled, ablated out, or
        when no known tags match this sample.
        """
        if self.topic_embeddings is None:
            return None
        if self._ablation_mode in (self.ABLATION_NO_TOPICS, self.ABLATION_NEITHER):
            return None
        tags = self._sample_tags_for(sample)
        if not tags:
            return None
        block, _mask = self.topic_embeddings([tags])
        if block.size(1) == 0:
            return None
        # TopicEmbeddingModule returns (B, P, D) on its own device/dtype;
        # the decoder forward will cast as needed.
        return EmbeddingSegment(embeddings=block.to(self.device))

    # ------------------------------------------------------------------
    # Checkpoint loading (with LoRA key remap fallback)
    # ------------------------------------------------------------------

    def _restore_model_state(self, state_dicts: dict) -> None:
        """Load model state with a pre-LoRA → post-LoRA key remap fallback.

        The strict load fails when a user points ``resume_checkpoint``
        at a pre-LoRA checkpoint (e.g. a raw Phase 1 checkpoint) while
        the current trainer has already installed LoRA wrappers in
        ``setup()`` — on-disk keys are ``q_proj.weight`` but in-memory
        keys are ``q_proj.base_layer.weight``.

        On the first ``RuntimeError`` we retry with
        :func:`remap_base_keys_to_lora` applied to the encoder sub-state
        and ``strict=False``, leaving LoRA adapter params at their
        zero-initialized state.
        """
        model_state = state_dicts["model"]
        try:
            self.model.load_state_dict(model_state)
        except RuntimeError as e:
            if not self._has_lora_installed():
                raise
            logger.warning(
                "phase2_kb_checkpoint_load_remap",
                error=str(e)[:300],
                hint="retrying with pre-LoRA → post-LoRA key remap",
            )
            remapped = self._remap_pre_lora_state_dict(model_state)
            missing, unexpected = self.model.load_state_dict(
                remapped, strict=False,
            )
            bad_missing = [
                k for k in missing
                if ".adapters." not in k or (
                    "lora_A" not in k and "lora_B" not in k
                )
            ]
            if bad_missing or unexpected:
                raise RuntimeError(
                    f"LoRA remap load failed: unexpected={unexpected[:5]}, "
                    f"missing_non_adapter={bad_missing[:5]}"
                ) from e
            logger.info("phase2_kb_checkpoint_loaded_via_remap")

    def _has_lora_installed(self) -> bool:
        """Return True if the encoder has any LoRALinearWrapper children."""
        if self.encoder is None:
            return False
        return any(
            isinstance(module, LoRALinearWrapper)
            for module in self.encoder.modules()
        )

    def _remap_pre_lora_state_dict(
        self, model_state: dict,
    ) -> dict:
        """Remap an encoder.* sub-dict with pre-LoRA keys into post-LoRA keys.

        Non-encoder keys (decoder.*, topic_embeddings.*,
        lora_router.*) pass through untouched.
        """
        remapped: dict = {}
        encoder_sub: dict = {}
        for k, v in model_state.items():
            if k.startswith("encoder."):
                encoder_sub[k[len("encoder."):]] = v
            else:
                remapped[k] = v
        encoder_sub = remap_base_keys_to_lora(
            encoder_sub, target_names=DEFAULT_LORA_TARGETS,
        )
        for sub_k, v in encoder_sub.items():
            remapped[f"encoder.{sub_k}"] = v
        return remapped

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _compute_sample_loss(
        self, sample: KBSample,
    ) -> tuple[torch.Tensor, int]:
        segments, _answer_span = self._build_decoder_segments(sample)
        loss = self.decoder.forward_interleaved_with_loss(segments)
        n_tokens = 0
        for seg in segments:
            if isinstance(seg, TokenSegment) and seg.loss_mask is not None:
                n_tokens += int(seg.loss_mask.sum().item())
        return loss, n_tokens

    # ------------------------------------------------------------------
    # Survivorship auxiliary losses
    # ------------------------------------------------------------------

    def _compute_survivorship_aux_losses(
        self,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute aux survivorship losses over all L0 and L1 encoder outputs
        accumulated during the current train step.

        Packed-input convention (FA4 varlen):
          - L0 entries: ``logits_for_op`` is flat ``(N_l0,)``;
            ``cu_seqlens`` is ``(B_l0+1,) int32`` marking per-article
            boundaries in the flat buffer.  All positions are valid —
            no padding tokens exist in the packed layout.
          - L1 entries: ``logits_for_op`` is flat ``(N_l1,)``;
            ``cu_seqlens`` is ``(B_l1+1,) int32``; ``relevance_mask``
            is flat ``(N_l1,) bool`` identifying gold vs. distractor
            positions.  No ``content_mask`` filtering is needed because
            every flat position is a real token.

        Losses:
          - Aggregate ratio: segment-mean(probs) minus target per article,
            then mean over articles.
          - Decisiveness: segment-mean(4*p*(1-p)).
          - Relevance (L1 only): flat gold-group mean vs. flat
            distractor-group mean, both compared against boost/damp
            targets.
          - Min-survivors: per-article soft-survivor count hinge via
            segment_sum.

        Returns (total_weighted_loss, metrics_dict).
        """
        metrics: dict[str, float] = {}
        total = torch.zeros((), device=self.device, dtype=torch.float32)

        # ----------------------------------------------------------------
        # L0: ratio + decisiveness
        # Each entry is one _l0_for_articles call.  Packed: logits_for_op
        # is flat (N_l0,) with cu_seqlens marking per-article segments.
        # ----------------------------------------------------------------
        l0_ratio_losses: list[torch.Tensor] = []
        l0_decisive_losses: list[torch.Tensor] = []
        for entry in self._pending_l0_outputs:
            enc_out = entry["enc_out"]
            logits_for_op = enc_out.logits_for_op
            if logits_for_op is None:
                continue
            # Flatten (1, L) → (L,) if the encoder still returns a
            # single-sample batch tensor; packed encoders emit (L,) directly.
            logits_for_op = logits_for_op.reshape(-1)
            theta_t = getattr(enc_out, "theta_tensor", None)
            if theta_t is None:
                legacy = getattr(enc_out, "theta_value", 0.0)
                theta_t = torch.tensor(
                    float(legacy), dtype=torch.float32,
                    device=logits_for_op.device,
                )
            cu_seqlens = entry.get("cu_seqlens")
            probs_f = torch.sigmoid(
                logits_for_op.float() - theta_t.to(logits_for_op.device).float()
            )
            target_ratio = entry["ratio"]
            if cu_seqlens is not None:
                # Segment-aware mean: (B_articles,)
                n_articles = int(cu_seqlens.shape[0]) - 1
                seg_ids = segment_ids_from_cu(cu_seqlens, int(probs_f.shape[0]))
                mean_probs = segment_mean(probs_f, seg_ids, n_articles)  # (B,)
                l0_ratio_losses.append(((mean_probs - target_ratio) ** 2).mean())
                decisive = segment_mean(
                    4.0 * probs_f * (1.0 - probs_f), seg_ids, n_articles,
                )
                l0_decisive_losses.append(decisive.mean())
            else:
                # Single segment (legacy / un-batched L0 call).
                mean_prob = probs_f.mean()
                l0_ratio_losses.append((mean_prob - target_ratio) ** 2)
                l0_decisive_losses.append(
                    (4.0 * probs_f * (1.0 - probs_f)).mean()
                )

        if l0_ratio_losses:
            l0_ratio_loss = torch.stack(l0_ratio_losses).mean()
            l0_decisive_loss = torch.stack(l0_decisive_losses).mean()
            total = (
                total
                + self._ratio_loss_weight * l0_ratio_loss
                + self._decisiveness_loss_weight * l0_decisive_loss
            )
            metrics["l0_ratio_loss"] = l0_ratio_loss.item()
            metrics["l0_decisiveness_loss"] = l0_decisive_loss.item()

        # ----------------------------------------------------------------
        # L0 min-survivors loss
        # Packed: segment_sum over flat soft-gates per article.
        # ----------------------------------------------------------------
        if self._surv_l0.min_survivors_loss_weight > 0.0 and self._pending_l0_outputs:
            l0_min_surv_losses: list[torch.Tensor] = []
            for entry in self._pending_l0_outputs:
                enc_out = entry["enc_out"]
                logits_for_op = enc_out.logits_for_op
                if logits_for_op is None:
                    continue
                logits_for_op = logits_for_op.reshape(-1)
                theta_t = getattr(enc_out, "theta_tensor", None)
                if theta_t is None:
                    legacy = getattr(enc_out, "theta_value", 0.0)
                    theta_t = torch.tensor(
                        float(legacy), dtype=torch.float32,
                        device=logits_for_op.device,
                    )
                tau = max(1e-3, self._surv_l0.min_survivors_tau)
                soft_gates = torch.sigmoid(
                    (logits_for_op.float() - theta_t.to(logits_for_op.device).float())
                    / tau,
                )
                cu_seqlens = entry.get("cu_seqlens")
                if cu_seqlens is not None:
                    n_articles = int(cu_seqlens.shape[0]) - 1
                    seg_ids = segment_ids_from_cu(cu_seqlens, int(soft_gates.shape[0]))
                    soft_count = segment_sum(soft_gates, seg_ids, n_articles)  # (B,)
                    content_len = lengths_from_cu(cu_seqlens).to(
                        dtype=soft_count.dtype, device=soft_count.device,
                    )
                else:
                    # Single segment.
                    soft_count = soft_gates.sum().unsqueeze(0)
                    content_len = torch.tensor(
                        [float(logits_for_op.shape[0])],
                        dtype=soft_count.dtype, device=soft_count.device,
                    )
                target_min = torch.clamp(
                    torch.ceil(content_len * self._surv_l0.min_survivors_floor_ratio),
                    min=float(self._surv_l0.min_survivors_absolute_min),
                )
                deficit = (1.0 - soft_count / target_min.clamp(min=1.0)).clamp(min=0.0)
                l0_min_surv_losses.append((deficit ** 2).mean())
            if l0_min_surv_losses:
                l0_min_surv_loss = torch.stack(l0_min_surv_losses).mean()
                total = total + self._surv_l0.min_survivors_loss_weight * l0_min_surv_loss
                metrics["l0_min_survivors_loss"] = l0_min_surv_loss.item()

        # ----------------------------------------------------------------
        # L1: ratio + decisiveness + relevance
        # Packed: logits_for_op is flat (N_l1,); relevance_mask is flat
        # (N_l1,) bool; cu_seqlens marks per-article segment boundaries.
        # No content_mask filtering: every flat position is a real token.
        # ----------------------------------------------------------------
        l1_ratio_losses: list[torch.Tensor] = []
        l1_decisive_losses: list[torch.Tensor] = []
        l1_relevance_losses: list[torch.Tensor] = []
        for entry in self._pending_l1_outputs:
            enc_out = entry["enc_out"]
            logits_for_op = enc_out.logits_for_op
            if logits_for_op is None:
                continue
            # Flatten potential (B, L) padded tensors that may still arrive
            # during the Wave 3.4 transition; once Wave 3.4 lands this
            # reshape is a no-op.
            logits_for_op = logits_for_op.reshape(-1)
            theta_t = getattr(enc_out, "theta_tensor", None)
            if theta_t is None:
                legacy = getattr(enc_out, "theta_value", 0.0)
                theta_t = torch.tensor(
                    float(legacy), dtype=torch.float32,
                    device=logits_for_op.device,
                )
            probs_f = torch.sigmoid(
                logits_for_op.float() - theta_t.to(logits_for_op.device).float()
            )
            target_ratio = entry["ratio"]
            relevance = entry["relevance_mask"]  # (N_l1,) bool, flat

            # Packed: all positions are valid.  Segment-aware per-article
            # reductions for ratio and decisiveness.
            cu_seqlens = entry.get("cu_seqlens")
            if cu_seqlens is not None:
                n_arts = int(cu_seqlens.shape[0]) - 1
                seg_ids = segment_ids_from_cu(cu_seqlens, int(probs_f.shape[0]))
                mean_probs = segment_mean(probs_f, seg_ids, n_arts)  # (B,)
                if mean_probs.numel() == 0:
                    continue
                l1_ratio_losses.append(((mean_probs - target_ratio) ** 2).mean())
                decisive = segment_mean(
                    4.0 * probs_f * (1.0 - probs_f), seg_ids, n_arts,
                )
                l1_decisive_losses.append(decisive.mean())
            else:
                # Single-segment fallback.
                if probs_f.numel() == 0:
                    continue
                l1_ratio_losses.append((probs_f.mean() - target_ratio) ** 2)
                l1_decisive_losses.append(
                    (4.0 * probs_f * (1.0 - probs_f)).mean()
                )

            # Relevance loss: two per-group aggregate-ratio targets.
            # - Gold positions (including pinned ID tokens) should survive
            #   at ~gold_boost * target_ratio (upsample the whole block,
            #   IDs + content).
            # - Distractor positions (content only — their ID tokens are
            #   not pinned) should survive at ~distractor_damp * target_ratio
            #   (downsample but don't suppress to zero — distractor IDs may
            #   still be referenced in later bgkit calls).
            # relevance is flat (N_l1,) bool — unchanged if already flat.
            if relevance is not None:
                relevance_flat = relevance.reshape(-1)
                gold_target = min(1.0, target_ratio * self._relevance_gold_boost)
                distractor_target = max(0.0, target_ratio * self._relevance_distractor_damp)
                per_group_losses: list[torch.Tensor] = []
                if relevance_flat.any():
                    gold_mean = probs_f[relevance_flat].mean()
                    per_group_losses.append((gold_mean - gold_target) ** 2)
                distractor_mask = ~relevance_flat
                if distractor_mask.any():
                    distractor_mean = probs_f[distractor_mask].mean()
                    per_group_losses.append((distractor_mean - distractor_target) ** 2)
                if per_group_losses:
                    l1_relevance_losses.append(torch.stack(per_group_losses).mean())

        if l1_ratio_losses:
            l1_ratio_loss = torch.stack(l1_ratio_losses).mean()
            l1_decisive_loss = torch.stack(l1_decisive_losses).mean()
            total = (
                total
                + self._ratio_loss_weight * l1_ratio_loss
                + self._decisiveness_loss_weight * l1_decisive_loss
            )
            metrics["l1_ratio_loss"] = l1_ratio_loss.item()
            metrics["l1_decisiveness_loss"] = l1_decisive_loss.item()

        # ----------------------------------------------------------------
        # L1 min-survivors loss
        # Packed: segment_sum over flat soft-gates per article.
        # ----------------------------------------------------------------
        if self._surv_l1.min_survivors_loss_weight > 0.0 and self._pending_l1_outputs:
            l1_min_surv_losses: list[torch.Tensor] = []
            for entry in self._pending_l1_outputs:
                enc_out = entry["enc_out"]
                logits_for_op = enc_out.logits_for_op
                if logits_for_op is None:
                    continue
                logits_for_op = logits_for_op.reshape(-1)
                theta_t = getattr(enc_out, "theta_tensor", None)
                if theta_t is None:
                    legacy = getattr(enc_out, "theta_value", 0.0)
                    theta_t = torch.tensor(
                        float(legacy), dtype=torch.float32,
                        device=logits_for_op.device,
                    )
                tau = max(1e-3, self._surv_l1.min_survivors_tau)
                soft_gates = torch.sigmoid(
                    (logits_for_op.float() - theta_t.to(logits_for_op.device).float())
                    / tau,
                )
                cu_seqlens = entry.get("cu_seqlens")
                if cu_seqlens is not None:
                    n_arts = int(cu_seqlens.shape[0]) - 1
                    seg_ids = segment_ids_from_cu(cu_seqlens, int(soft_gates.shape[0]))
                    soft_count = segment_sum(soft_gates, seg_ids, n_arts)  # (B,)
                    content_len = lengths_from_cu(cu_seqlens).to(
                        dtype=soft_count.dtype, device=soft_count.device,
                    )
                    if content_len.sum() == 0:
                        continue
                else:
                    soft_count = soft_gates.sum().unsqueeze(0)
                    content_len = torch.tensor(
                        [float(logits_for_op.shape[0])],
                        dtype=soft_count.dtype, device=soft_count.device,
                    )
                    if content_len[0] == 0:
                        continue
                target_min = torch.clamp(
                    torch.ceil(content_len * self._surv_l1.min_survivors_floor_ratio),
                    min=float(self._surv_l1.min_survivors_absolute_min),
                )
                deficit = (1.0 - soft_count / target_min.clamp(min=1.0)).clamp(min=0.0)
                l1_min_surv_losses.append((deficit ** 2).mean())
            if l1_min_surv_losses:
                l1_min_surv_loss = torch.stack(l1_min_surv_losses).mean()
                total = total + self._surv_l1.min_survivors_loss_weight * l1_min_surv_loss
                metrics["l1_min_survivors_loss"] = l1_min_surv_loss.item()

        if l1_relevance_losses:
            l1_relevance_loss = torch.stack(l1_relevance_losses).mean()
            total = total + self._relevance_loss_weight * l1_relevance_loss
            metrics["l1_relevance_loss"] = l1_relevance_loss.item()

        return total, metrics

    def _apply_utility_grad_bce_phase2(self) -> dict[str, float]:
        """Run post-backward utility-gradient BCE distillation for Phase 2.

        Walks ``_pending_l0_outputs`` and ``_pending_l1_outputs``; for
        each entry whose enc_out captured a non-None content-grad in
        its backward hook, builds a top-k teacher over controllable
        positions (pinned positions excluded for L1 via the entry's
        pinned mask) and runs a small head-local backward through
        ``base_raw_for_util → head.weights``. Because
        ``base_raw_for_util`` was computed inside the encoder's LoRA
        context during the main forward, any active LoRA adapter (e.g.
        L0 LoRA in Stage A) receives the BCE gradient directly — no
        need to re-enter the router here. Clears large per-entry
        stashes before return to bound peak memory.
        """
        from bgkit.training.survivorship_helpers import (
            LevelLossCfg as _LLC,
            utility_grad_bce_loss,
        )

        metrics: dict[str, float] = {}
        w_l0 = getattr(self, "_surv_l0", _LLC()).utility_grad_loss_weight
        w_l1 = getattr(self, "_surv_l1", _LLC()).utility_grad_loss_weight

        if w_l0 > 0.0 and self._pending_l0_outputs:
            grad_norms: list[float] = []
            for entry in self._pending_l0_outputs:
                enc_out = entry.get("enc_out")
                if enc_out is None:
                    continue
                grad_capture = getattr(enc_out, "_l0_grad_capture", None)
                content_grad = (
                    grad_capture.get("post_head_content_grad")
                    if grad_capture is not None
                    else enc_out.get_content_grad()
                )
                content_values = enc_out.post_head_content_values
                if content_grad is None or content_values is None:
                    continue
                util_loss, _ = utility_grad_bce_loss(
                    base_raw_for_util=enc_out.base_raw_for_util,
                    content_grad=content_grad,
                    content_values=content_values,
                    # Packed form: every content position is valid (no
                    # padding). Pass ``None`` so the helper short-circuits
                    # to an all-True mask.
                    valid_mask=None,
                    pinned_mask=None,
                    target_ratio=float(entry.get("ratio", self._l1_retention)),
                    content_cu_seqlens=entry.get("cu_seqlens"),
                )
                if util_loss.requires_grad:
                    (util_loss * w_l0 / self._accum_steps).backward()
                grad_norms.append(float(content_grad.norm().item()))
                # Release references to free memory.
                entry["enc_out"] = None
            if grad_norms:
                metrics["l0_content_grad_norm"] = sum(grad_norms) / len(grad_norms)

        if w_l1 > 0.0 and self._pending_l1_outputs:
            grad_norms = []
            for entry in self._pending_l1_outputs:
                enc_out = entry.get("enc_out")
                if enc_out is None:
                    continue
                grad_capture = getattr(enc_out, "_l1_grad_capture", None)
                content_grad = (
                    grad_capture.get("post_head_content_grad")
                    if grad_capture is not None
                    else enc_out.get_content_grad()
                )
                content_values = enc_out.post_head_content_values
                if content_grad is None or content_values is None:
                    continue
                util_loss, _ = utility_grad_bce_loss(
                    base_raw_for_util=enc_out.base_raw_for_util,
                    content_grad=content_grad,
                    content_values=content_values,
                    # Packed form: no padding.
                    valid_mask=None,
                    pinned_mask=entry.get("pinned"),
                    target_ratio=float(entry.get("ratio", self._l1_retention)),
                    content_cu_seqlens=entry.get("cu_seqlens"),
                )
                if util_loss.requires_grad:
                    (util_loss * w_l1 / self._accum_steps).backward()
                grad_norms.append(float(content_grad.norm().item()))
                entry["enc_out"] = None
            if grad_norms:
                metrics["l1_content_grad_norm"] = sum(grad_norms) / len(grad_norms)

        return metrics

    def _forward_backward(self, batch) -> dict[str, float]:
        """Cross-sample batched forward + backward.

        Every sample in the training batch has ~3-4 bgkit turns. We pool
        ALL turns from ALL samples, bucket them by content length (power
        of 2), and run each bucket as a single padded encoder forward.
        Then each sample is assembled into segments using the pre-computed
        L1 outputs and the decoder forward runs per-sample. This collapses
        (batch_size * turns_per_sample) encoder launches into at most a
        handful of bucket forwards per training step.
        """
        if not batch:
            return {"loss": torch.zeros((), device=self.device), "tokens": 0.0}

        # Reset per-step accumulators for survivorship aux losses
        self._pending_l0_outputs: list[dict] = []
        self._pending_l1_outputs: list[dict] = []
        self._step_sampled_l0_ratios = []
        self._step_sampled_l1_ratios = []

        # Stamp per-batch tag usage so the topic embeddings can divide
        # each tag parameter's gradient by the number of batch members
        # that referenced it (averaging instead of summing). No-op when
        # topic embeddings are disabled. Must be done before backward;
        # the divide itself happens in apply_gradient_averaging() below.
        if self.topic_embeddings is not None:
            self.topic_embeddings.record_batch_usage(
                [self._sample_tags_for(s) for s in batch],
            )

        # Phase 1: per-sample rendering + L1 input preparation (no encoder)
        preps: list[dict] = [self._prepare_sample_for_decode(s) for s in batch]

        # Phase 2: flatten every non-None prepared turn with its source
        # (sample_idx, turn_idx) address so we can scatter results back.
        flat: list[tuple[int, int, dict | None]] = []
        for s_idx, prep in enumerate(preps):
            for t_idx, turn in enumerate(prep["prepared_turns"]):
                flat.append((s_idx, t_idx, turn))

        # Bucket non-None turns by power-of-2 content length. Padding
        # waste inside a bucket is at most 50% (the largest and smallest
        # in a bucket differ by at most 2x).
        buckets: dict[int, list[tuple[int, int, dict]]] = {}
        for s_idx, t_idx, prep in flat:
            if prep is None:
                continue
            n_content = int(prep["content"].size(0))
            # bucket_key = ceil(log2(max(n_content, 1)))
            bucket_key = max(0, (max(n_content, 1) - 1).bit_length())
            buckets.setdefault(bucket_key, []).append((s_idx, t_idx, prep))

        hidden_dim = self.encoder.l1.hidden_dim
        zero_fallback = torch.zeros(
            (1, hidden_dim), device=self.device, dtype=torch.bfloat16,
        )
        survivors_by_address: dict[tuple[int, int], torch.Tensor] = {}

        # Run one encoder forward per bucket. _run_l1_batch takes a list
        # that may include None entries; we pre-filter because all
        # entries in a bucket are non-None by construction.
        for _bucket_key, items in buckets.items():
            bucket_preps = [p for _, _, p in items]
            bucket_out = self._run_l1_batch(bucket_preps)
            for (s_idx, t_idx, _prep), surv in zip(
                items, bucket_out, strict=True,
            ):
                survivors_by_address[(s_idx, t_idx)] = surv

        # Fill in zero-fallbacks for None turns
        for s_idx, t_idx, prep in flat:
            if prep is None:
                survivors_by_address[(s_idx, t_idx)] = zero_fallback

        # Phase 3: per-sample decoder forward + loss accumulation
        total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        total_tokens = 0
        n_samples = 0
        n_turns_total = 0
        for s_idx, (_sample, prep) in enumerate(zip(batch, preps, strict=True)):
            per_turn = [
                survivors_by_address[(s_idx, t_idx)]
                for t_idx in range(len(prep["prepared_turns"]))
            ]
            segments, _trace = self._assemble_sample_segments(prep, per_turn)
            if not segments:
                continue
            sample_loss = self.decoder.forward_interleaved_with_loss(segments)
            sample_tokens = 0
            for seg in segments:
                if isinstance(seg, TokenSegment) and seg.loss_mask is not None:
                    sample_tokens += int(seg.loss_mask.sum().item())
            if sample_tokens == 0:
                continue
            total_loss = total_loss + sample_loss
            total_tokens += sample_tokens
            n_samples += 1
            n_turns_total += len(prep["prepared_turns"])

        if n_samples == 0:
            return {"loss": torch.zeros((), device=self.device), "tokens": 0.0}

        decoder_loss = total_loss / n_samples
        total_weighted = decoder_loss

        # Survivorship auxiliary losses over accumulated L0 + L1 outputs
        aux_loss, aux_metrics = self._compute_survivorship_aux_losses()
        if aux_loss.requires_grad or aux_loss.item() != 0.0:
            total_weighted = total_weighted + aux_loss

        (total_weighted / self._accum_steps).backward()

        # --- Utility-gradient BCE distillation (post-backward) ---
        # The main backward populated each LevelOutput's captured
        # ``post_head_content_grad`` via the level's backward hook.
        # Rebuild the top-k teacher from ``-(grad · value)`` and run a
        # small head-local backward per level.
        util_metrics = self._apply_utility_grad_bce_phase2()

        # Average shared-tag gradients across the batch (see
        # TopicEmbeddingModule.apply_gradient_averaging). Only touches
        # parameters whose tags appeared in record_batch_usage above.
        if self.topic_embeddings is not None:
            self.topic_embeddings.apply_gradient_averaging()

        metrics_out = {
            "loss": decoder_loss.detach(),
            "tokens": float(total_tokens / n_samples),
            "l1_retention": self._l1_retention,
            "live_l0": 1.0 if self._live_l0 else 0.0,
            "l1_turns_per_sample": float(n_turns_total / n_samples),
            "l1_buckets_per_step": float(len(buckets)),
        }
        if self._step_sampled_l0_ratios:
            metrics_out["sampled_l0_ratio_mean"] = float(
                sum(self._step_sampled_l0_ratios) / len(self._step_sampled_l0_ratios),
            )
        if self._step_sampled_l1_ratios:
            metrics_out["sampled_l1_ratio_mean"] = float(
                sum(self._step_sampled_l1_ratios) / len(self._step_sampled_l1_ratios),
            )
        metrics_out.update(aux_metrics)
        metrics_out.update(util_metrics)

        # Head-health diagnostics (organic-rate std, undecided fraction,
        # floor/pinned/θ) per level. Sampled from the last accumulated
        # enc_out for each level — adequate for detecting collapse modes
        # without paying for a reduce across all accumulated outputs.
        from bgkit.training.survivorship_helpers import survivorship_diagnostics
        diag_every_n = int(
            self.step_cfg.get("diagnostic_metrics_every_n_steps", 10) or 1,
        )
        if self._pending_l0_outputs:
            last_l0 = self._pending_l0_outputs[-1].get("enc_out")
            if last_l0 is not None:
                metrics_out.update(
                    survivorship_diagnostics(
                        last_l0, level="l0",
                        global_step=self.global_step,
                        every_n_steps=diag_every_n,
                    )
                )
        if self._pending_l1_outputs:
            last_l1 = self._pending_l1_outputs[-1].get("enc_out")
            if last_l1 is not None:
                metrics_out.update(
                    survivorship_diagnostics(
                        last_l1, level="l1",
                        global_step=self.global_step,
                        every_n_steps=diag_every_n,
                    )
                )

        # Drop LevelOutput tensor refs held by this step's pending
        # outputs. Even though the lists are re-assigned at the top of
        # the next ``_forward_backward`` call, explicit release here
        # ensures the hook-closure retention (see
        # ``LevelOutput.release()``) doesn't keep subgraph activations
        # pinned across optimizer steps.
        for _entry in (*self._pending_l0_outputs, *self._pending_l1_outputs):
            _enc_out = _entry.get("enc_out")
            if _enc_out is not None and hasattr(_enc_out, "release"):
                _enc_out.release()

        return metrics_out

    # ------------------------------------------------------------------
    # Ablation
    # ------------------------------------------------------------------

    def set_ablation_mode(self, mode: str | None) -> None:
        """Set the ablation mode used on the next forward pass.

        Use the class constants ``ABLATION_*``. Set ``ABLATION_NONE``
        (``None``) to disable ablation.
        """
        self._ablation_mode = mode

    def _apply_context_ablation(
        self, survivors: torch.Tensor, *, skip: bool,
    ) -> torch.Tensor:
        """Apply any context-level ablation transform to survivor vectors.

        - ``skip`` True (``TOPICS_ONLY`` / ``NEITHER``): collapse to a single
          zero vector so the sentinel splice still has a slot.
        - ``ABLATION_ZEROED``: zero out the vectors in place but keep count.
        - ``ABLATION_NOISE``: replace with small Gaussian noise.
        """
        if skip:
            return torch.zeros(
                1, survivors.size(-1),
                device=survivors.device, dtype=survivors.dtype,
            )
        if self._ablation_mode == self.ABLATION_ZEROED:
            return torch.zeros_like(survivors)
        if self._ablation_mode == self.ABLATION_NOISE:
            return torch.randn_like(survivors) * 0.02
        return survivors

    # ------------------------------------------------------------------
    # Post-optimizer-step hooks
    # ------------------------------------------------------------------

    def _post_optimizer_step(self, step: int) -> None:
        """Run dual-ascent θ + EMA μ updates per level using true-mean aggregation.

        Stage A: live L0 → update both L0 and L1.
        Stage B: cached L0 with L0 LoRA frozen → skip L0 updates.
        """
        # Accumulate pending L0/L1 outputs into per-level state, then apply.
        from bgkit.training.survivorship_helpers import (
            accumulate,
            apply_post_step_updates,
            init_state,
            maybe_unload_ice,
        )

        # L0 target: per-microbatch ratios differ by dataset. Weight each
        # microbatch by its controllable_count so θ targets the true
        # batch-weighted mean ratio — matches the aggregation semantics of
        # the sum/count accumulator.
        l0_target_num = 0.0
        l0_target_den = 0.0
        l1_target_num = 0.0
        l1_target_den = 0.0
        for entry in getattr(self, "_pending_l0_outputs", []):
            enc_out = entry.get("enc_out")
            if enc_out is None:
                continue
            ratio = float(entry.get("ratio", self._l1_retention))
            accumulate(self._surv_state_l0, enc_out, target_ratio=ratio)
            if enc_out.controllable_count is not None:
                cc = int(enc_out.controllable_count.item())
                l0_target_num += ratio * cc
                l0_target_den += cc
        for entry in getattr(self, "_pending_l1_outputs", []):
            enc_out = entry.get("enc_out")
            if enc_out is None:
                continue
            ratio = float(entry.get("ratio", self._l1_retention))
            accumulate(self._surv_state_l1, enc_out, target_ratio=ratio)
            if enc_out.controllable_count is not None:
                cc = int(enc_out.controllable_count.item())
                l1_target_num += ratio * cc
                l1_target_den += cc

        if l0_target_den > 0:
            target_l0 = l0_target_num / l0_target_den
        elif isinstance(self._l0_retention, (int, float)):
            target_l0 = float(self._l0_retention)
        else:
            # No per-dataset weighting available; fall back to the mean
            # across configured datasets.
            vals = [float(v) for v in self._l0_retention.values()] if self._l0_retention else []
            target_l0 = sum(vals) / len(vals) if vals else 0.10
        if l1_target_den > 0:
            target_l1 = l1_target_num / l1_target_den
        else:
            target_l1 = float(self._l1_retention)

        merged: dict[str, float] = {}
        # L0 update: skip if Stage B (cached L0, L0 frozen).
        l0_metrics = apply_post_step_updates(
            self.encoder,
            self._surv_state_l0,
            target_ratio=target_l0,
            level="l0",
            skip_threshold_step=not self._live_l0,
        )
        merged.update(l0_metrics)
        self._surv_state_l0 = init_state()

        # L1 always updates.
        l1_metrics = apply_post_step_updates(
            self.encoder,
            self._surv_state_l1,
            target_ratio=target_l1,
            level="l1",
        )
        merged.update(l1_metrics)
        self._surv_state_l1 = init_state()

        self._last_post_step_metrics = merged

        unloaded = maybe_unload_ice(
            getattr(self, "_ice_teacher", None),
            step,
            getattr(self, "_max_warmup_step", 0),
        )
        if unloaded:
            logger.info("ice_teacher_unloaded", step=step)

    def _add_step_metrics(self, metrics: dict[str, float]) -> None:
        """Attach post-step θ/μ updates (without clobbering base keys)."""
        post = getattr(self, "_last_post_step_metrics", None)
        if post:
            for k, v in post.items():
                metrics.setdefault(k, v)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Full KB-scale eval: loss, next-token accuracy on loss-bearing
        positions, plus exact_match / token_f1 against the gold answer by
        decoding the argmax over loss-masked tokens. Per-dataset breakdowns
        are emitted under ``eval/{dataset_name}/...`` keys.

        Also emits KB trajectory metrics from
        :mod:`bgkit.eval.kb_trajectory_eval` under the ``eval/kb/...``
        prefix — trajectory step accuracy (micro over tokens), tool-call
        ID accuracy per browse/bgkit/overall (micro over calls), and
        answer token F1 (macro over samples). The existing ``eval/loss``,
        ``eval/n_samples``, and ``eval/tokens_per_sample`` keys are
        preserved for backward compatibility.
        """

        self.model.eval()

        total_loss_weighted = 0.0
        total_tokens = 0
        n_samples = 0
        total_correct = 0

        per_dataset_correct: dict[str, int] = {}
        per_dataset_total: dict[str, int] = {}
        per_dataset_em: dict[str, list[float]] = {}
        per_dataset_f1: dict[str, list[float]] = {}
        all_em: list[float] = []
        all_f1: list[float] = []

        # KB-harness trajectory metric accumulators. Tool-call ID
        # accuracy is micro-averaged across (browse, bgkit, overall) by
        # summing the per-call scores and dividing by total call counts;
        # answer F1 is macro-averaged across samples with an answer.
        kb_browse_sum = 0.0
        kb_browse_n = 0
        kb_bgkit_sum = 0.0
        kb_bgkit_n = 0
        kb_f1_sum = 0.0
        kb_f1_n = 0

        for batch in self.eval_dataloader:
            for sample in batch:
                result = self._eval_one_sample(sample)
                if result is None:
                    continue
                sample_loss = result["loss"]
                sample_tokens = result["tokens"]
                total_loss_weighted += float(sample_loss) * sample_tokens
                total_tokens += sample_tokens
                total_correct += result["correct"]
                n_samples += 1
                all_em.append(result["em"])
                all_f1.append(result["f1"])
                ds = sample.dataset_name
                per_dataset_em.setdefault(ds, []).append(result["em"])
                per_dataset_f1.setdefault(ds, []).append(result["f1"])
                per_dataset_correct[ds] = per_dataset_correct.get(ds, 0) + result["correct"]
                per_dataset_total[ds] = per_dataset_total.get(ds, 0) + sample_tokens

                # KB harness metric accumulation (tool-call ID
                # accuracy + answer F1). Already computed as part of
                # _eval_one_sample so no extra forward pass.
                tool_call = result.get("tool_call_id", None)
                if tool_call is not None:
                    n_br = int(tool_call.get("n_browse", 0))
                    n_bg = int(tool_call.get("n_bgkit", 0))
                    if n_br:
                        kb_browse_sum += float(tool_call.get("browse", 0.0)) * n_br
                        kb_browse_n += n_br
                    if n_bg:
                        kb_bgkit_sum += float(tool_call.get("bgkit", 0.0)) * n_bg
                        kb_bgkit_n += n_bg
                if sample.gold_answer:
                    kb_f1_sum += float(result["f1"])
                    kb_f1_n += 1

        self.model.train()
        if total_tokens == 0:
            return {
                "eval/loss": 0.0,
                "eval/answer_token_accuracy": 0.0,
                "eval/exact_match": 0.0,
                "eval/token_f1": 0.0,
                "eval/n_samples": 0.0,
                "eval/kb/trajectory_step_accuracy": 0.0,
                "eval/kb/tool_call_id_accuracy/browse": 0.0,
                "eval/kb/tool_call_id_accuracy/bgkit": 0.0,
                "eval/kb/tool_call_id_accuracy/overall": 0.0,
                "eval/kb/answer_token_f1": 0.0,
            }

        kb_browse_acc = kb_browse_sum / kb_browse_n if kb_browse_n else 0.0
        kb_bgkit_acc = kb_bgkit_sum / kb_bgkit_n if kb_bgkit_n else 0.0
        kb_tool_n = kb_browse_n + kb_bgkit_n
        kb_overall_acc = (
            (kb_browse_sum + kb_bgkit_sum) / kb_tool_n if kb_tool_n else 0.0
        )
        kb_f1 = kb_f1_sum / kb_f1_n if kb_f1_n else 0.0

        metrics: dict[str, float] = {
            "eval/loss": total_loss_weighted / total_tokens,
            "eval/answer_token_accuracy": total_correct / total_tokens,
            "eval/exact_match": sum(all_em) / len(all_em),
            "eval/token_f1": sum(all_f1) / len(all_f1),
            "eval/n_samples": float(n_samples),
            "eval/tokens_per_sample": float(total_tokens / max(n_samples, 1)),
            # KB trajectory metrics. Micro over tokens / calls; macro
            # over samples (for F1).
            "eval/kb/trajectory_step_accuracy": total_correct / total_tokens,
            "eval/kb/tool_call_id_accuracy/browse": kb_browse_acc,
            "eval/kb/tool_call_id_accuracy/bgkit": kb_bgkit_acc,
            "eval/kb/tool_call_id_accuracy/overall": kb_overall_acc,
            "eval/kb/answer_token_f1": kb_f1,
            "eval/kb/n_browse_calls": float(kb_browse_n),
            "eval/kb/n_bgkit_calls": float(kb_bgkit_n),
        }
        for ds, c in per_dataset_correct.items():
            t = per_dataset_total.get(ds, 0)
            if t:
                metrics[f"eval/{ds}/token_accuracy"] = c / t
        for ds, scores in per_dataset_em.items():
            if scores:
                metrics[f"eval/{ds}/exact_match"] = sum(scores) / len(scores)
        for ds, scores in per_dataset_f1.items():
            if scores:
                metrics[f"eval/{ds}/token_f1"] = sum(scores) / len(scores)
        return metrics

    @torch.no_grad()
    def _eval_one_sample(self, sample: KBSample) -> dict | None:
        """Run one eval sample: loss, next-token argmax accuracy, EM/F1.

        ``loss`` and ``correct`` are computed over *all* loss-bearing
        positions in the sample (including browse and bgkit tool-call
        turns). ``em`` and ``f1`` are computed ONLY over the final
        ``answer`` turn's token range, so tool-call imitation accuracy
        doesn't pollute the answer quality metric.

        Also computes the KB-scale tool-call ID accuracy (per browse /
        per bgkit / overall) using the concat-coordinate spans from the
        decode trace — returned under the ``tool_call_id`` key for
        :meth:`evaluate` to roll up.

        Returns ``None`` if the sample has no loss-bearing tokens.
        """
        from bgkit.eval.kb_trajectory_eval import (
            _score_call_span,
            _tool_call_gold_ids,
        )
        from bgkit.eval.metrics.qa_metrics import exact_match, token_f1

        segments, trace = self._build_decoder_segments_with_trace(sample)
        answer_span = trace.answer_span
        output = self.decoder.forward_interleaved_with_loss(
            segments, return_hidden_states=True,
        )
        token_ids_full = output.token_ids
        loss_mask_full = output.loss_mask

        # Shifted CE positions: hidden[i] predicts token[i+1].
        shift_t = token_ids_full[:, 1:]
        shift_m = loss_mask_full[:, 1:]
        if int(shift_m.sum().item()) == 0:
            return None

        preds = output.argmax_predictions()  # (B, S-1)
        correct = int(((preds == shift_t) & shift_m).sum().item())
        tokens = int(shift_m.sum().item())

        # Compute EM/F1 against the gold answer ONLY. The answer span is
        # in concat coordinates; we translate to shifted coordinates via
        # ``shifted_i = concat_i - 1`` and clip to valid shifted range.
        if answer_span is None:
            em = 0.0
            f1 = 0.0
            pred_text = ""
            gold_text = str(sample.gold_answer)
        else:
            a_start, a_end = answer_span
            shift_a_start = max(0, a_start - 1)
            shift_a_end = max(0, a_end - 1)
            if shift_a_end > shift_a_start:
                pred_ids = preds[0, shift_a_start:shift_a_end].cpu()
                gold_ids = shift_t[0, shift_a_start:shift_a_end].cpu()
                # Filter by the loss mask in the same range so padding-like
                # positions don't pollute the decode.
                answer_mask = shift_m[0, shift_a_start:shift_a_end].cpu()
                if answer_mask.any():
                    pred_ids = pred_ids[answer_mask]
                    gold_ids = gold_ids[answer_mask]
                pred_text = self.tokenizer.decode(
                    pred_ids, skip_special_tokens=True,
                )
                gold_text = self.tokenizer.decode(
                    gold_ids, skip_special_tokens=True,
                )
            else:
                pred_text = ""
                gold_text = str(sample.gold_answer)
            em = exact_match(pred_text, [gold_text])
            f1 = token_f1(pred_text, [gold_text])

        # Per-call tool-ID accuracy, scored via the KB trajectory eval
        # helper functions so the scoring logic lives in one place.
        browse_scores: list[float] = []
        for turn, span in zip(
            trace.browse_turns, trace.browse_call_spans, strict=True,
        ):
            browse_scores.append(
                _score_call_span(
                    preds, shift_m, span,
                    _tool_call_gold_ids(turn), self.tokenizer,
                )
            )
        bgkit_scores: list[float] = []
        for turn, span in zip(
            trace.bgkit_turns, trace.bgkit_call_spans, strict=True,
        ):
            bgkit_scores.append(
                _score_call_span(
                    preds, shift_m, span,
                    _tool_call_gold_ids(turn), self.tokenizer,
                )
            )
        n_browse = len(browse_scores)
        n_bgkit = len(bgkit_scores)
        tool_call_id = {
            "browse": sum(browse_scores) / n_browse if n_browse else 0.0,
            "bgkit": sum(bgkit_scores) / n_bgkit if n_bgkit else 0.0,
            "overall": (
                (sum(browse_scores) + sum(bgkit_scores)) / (n_browse + n_bgkit)
                if (n_browse + n_bgkit) else 0.0
            ),
            "n_browse": n_browse,
            "n_bgkit": n_bgkit,
        }

        return {
            "loss": float(output.loss.item()),
            "tokens": tokens,
            "correct": correct,
            "em": em,
            "f1": f1,
            "pred_text": pred_text,
            "gold_text": gold_text,
            "tool_call_id": tool_call_id,
        }
