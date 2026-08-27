"""Phase 2 KB-scale trainer.

Implements the query-conditioned browse + bgkit training loop described in
``docs/phase2_kb.md`` (a.k.a. the "quirky-drifting-moore" plan):

- Base encoder starts from Phase 1 weights. The default Qwen path uses
  encoder-side LoRA for L1 query fusion; Falcon-family configs disable encoder
  LoRA and train the selected encoder levels directly.
- L0 is live in Stage A and cached in Stages B/C (survivors loaded from
  :class:`bgkit.data.l0_cache.L0Cache`).
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
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

import structlog
import torch
import torch.nn as nn
from omegaconf import open_dict
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split

from bgkit.data.article_token_store import ArticleTokenStore
from bgkit.data.bgkit_tool_template import (
    BGKIT_SENTINEL,
    BGKIT_TOOL,
    BGKIT_TOPIC_KNOWLEDGE_TOOL,
    BGKIT_TOPIC_SENTINEL,
    assistant_generation_prompt_ids,
    assistant_turn_end_glue,
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
    normalize_decoder_family,
)
from bgkit.models.lora_encoder import (
    DEFAULT_LORA_TARGETS,
    LoRALinearWrapper,
    LoRARouter,
    remap_base_keys_to_lora,
)
from bgkit.models.projection_block import effective_projection_cu
from bgkit.models.recursive_l1 import encode_tree_node
from bgkit.models.topic_embeddings import TopicEmbeddingModule
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.compression_curriculum import CompressionCurriculumMixin
from bgkit.training.ratio_sampling import (
    build_ratio_sampler_config,
    resolve_anchor_grid,
    sample_ratio,
)
from bgkit.utils.attention_backend import resolve_decoder_attention_implementation
from bgkit.utils.packing import (
    lengths_from_cu,
    position_ids_from_cu,
    segment_ids_from_cu,
    segment_mean,
    segment_sum,
)

logger = structlog.get_logger()

# Default general (query-agnostic) compression prompt fed to the shared repo
# tree (L0 + L1) in per-repo full-backprop mode.  Overridable via
# ``training.recursive_l1_tree.general_prompt``.
DEFAULT_RECURSIVE_GENERAL_PROMPT = "compress this repository's commit history"

# Per-phase CUDA-memory breakdown of the per-repo full-backprop step. Gated by
# the ``BGKIT_MEM_BREAKDOWN`` env flag (default off → zero cost: one bool check
# per phase boundary, a handful per repo). Read once at import.
_MEM_BREAKDOWN = os.environ.get("BGKIT_MEM_BREAKDOWN", "").strip().lower() not in (
    "", "0", "false", "no", "off",
)


# ---------------------------------------------------------------------------
# Model container
# ---------------------------------------------------------------------------


class _KBModel(nn.Module):
    """Registers all trainable parameters under a single nn.Module for
    checkpointing."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module | None = None,
        lora_router: LoRARouter | None = None,
        topic_embeddings: TopicEmbeddingModule | None = None,
        decoders: dict[str, nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        # Round-robin training holds one decoder per family in a ModuleDict so a
        # single ``state_dict()`` round-trips both decoders (keys
        # ``decoders.qwen35.*`` / ``decoders.falcon_h1.*``); single-decoder
        # stages keep the plain ``decoder`` attribute. Either way the trainer's
        # ``self.decoder`` pointer is aimed at the active family's module before
        # every forward. No code reads ``model.decoder`` directly (the trainer
        # uses ``self.decoder``), so ``decoder=None`` in dict mode is safe.
        if decoders is not None:
            self.decoders = nn.ModuleDict(decoders)
            self.decoder = None
        else:
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
    bgkit_turns: list
    bgkit_call_spans: list[tuple[int, int]]


class KRKBTrainer(CompressionCurriculumMixin, BaseTrainer):
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

    # Live-config handlers (merged across the MRO with BaseTrainer's, so the
    # inherited lr / max_batch_tokens / ... handlers are preserved). The
    # per-repo file-sample cap is nullable (null = unlimited), so it needs a
    # handler rather than the declarative numeric LIVE_CONFIG_FIELDS path.
    LIVE_CONFIG_HANDLERS: ClassVar[dict[str, str]] = {
        "max_file_samples_per_repo": "_handle_max_file_samples_per_repo",
        "per_repo_sample_group_size": "_handle_per_repo_sample_group_size",
        "per_repo_inner_loop": "_handle_per_repo_inner_loop",
        "per_repo_option_a": "_handle_per_repo_option_a",
        "option_a_max_subsets": "_handle_option_a_max_subsets",
        "drill_checkpoint_min_seqlen": "_handle_drill_checkpoint_min_seqlen",
        "per_repo_inner_subset_size": "_handle_per_repo_inner_subset_size",
        "per_repo_max_inner_steps": "_handle_per_repo_max_inner_steps",
        "inner_loop_warmup_steps": "_handle_inner_loop_warmup_steps",
        "max_l0_encode_tokens": "_handle_max_l0_encode_tokens",
        "max_decode_tokens": "_handle_max_decode_tokens",
        "checkpoint_tree_encode": "_handle_checkpoint_tree_encode",
        "tree_checkpoint_min_nodes": "_handle_tree_checkpoint_min_nodes",
        "tree_checkpoint_min_tokens": "_handle_tree_checkpoint_min_tokens",
        "decode_gc_min_seqlen": "_handle_decode_gc_min_seqlen",
        # --- Round-robin decoder routing ---
        "qwen_decoder_prob": "_handle_qwen_decoder_prob",
        # --- Survivorship aux losses (master switch + weights) ---
        "survivorship_aux": "_handle_survivorship_aux",
        # Projection-output norm-band regularizer weight (collapse fix).
        "projection_norm_reg_weight": "_handle_projection_norm_reg_weight",
        "ratio_loss_weight": "_handle_ratio_loss_weight",
        "span_relevance_weight": "_handle_span_relevance_weight",
        "decisiveness_loss_weight": "_handle_decisiveness_loss_weight",
        "relevance_loss_weight": "_handle_relevance_loss_weight",
        "relevance_gold_boost": "_handle_relevance_gold_boost",
        "relevance_distractor_damp": "_handle_relevance_distractor_damp",
        "n_distractors": "_handle_n_distractors",
        # --- Training-time random ablation probabilities ---
        "p_skip_bgkit": "_handle_p_skip_bgkit",
        "p_skip_topic": "_handle_p_skip_topic",
        "p_noise_bgkit": "_handle_p_noise_bgkit",
        # --- Retention ratios (read fresh each step) ---
        "l1_retention": "_handle_l1_retention",
        "recursive_l1_retention": "_handle_recursive_l1_retention",
        "recursive_l0_retention": "_handle_recursive_l0_retention",
        "recursive_general_prompt": "_handle_recursive_general_prompt",
        # --- Query-conditioned drill nodes (per-sample task-query re-encode) ---
        "query_conditioned_drill_nodes": "_handle_query_conditioned_drill_nodes",
        "drill_node_retention": "_handle_drill_node_retention",
        "drill_leaf_l0_retention": "_handle_drill_leaf_l0_retention",
        "drill_leaf_l1_retention": "_handle_drill_leaf_l1_retention",
        # --- Memory / speed toggles ---
        "checkpoint_encoder": "_handle_checkpoint_encoder",
        "profile_timing": "_handle_profile_timing",
        # --- Diagnostics ---
        "ablation_probe_steps": "_handle_ablation_probe_steps",
        # --- Per-repo SIZE FILTERS (full effect on next dataloader rebuild) ---
        "max_repo_leaf_tokens": "_handle_max_repo_leaf_tokens",
        "max_repo_file_samples": "_handle_max_repo_file_samples",
        "max_repo_tree_nodes": "_handle_max_repo_tree_nodes",
    }

    # Ablation modes — set via set_ablation_mode() during eval.
    ABLATION_NONE = None
    ABLATION_ZEROED = "zeroed"       # survivors → zeros (no context info)
    ABLATION_NOISE = "noise"         # survivors → gaussian noise
    ABLATION_NO_TOPICS = "no_topics"  # drop topic embedding segment
    ABLATION_TOPICS_ONLY = "topics_only"  # drop bgkit survivor segments
    ABLATION_NEITHER = "neither"     # drop both topics and bgkit survivors
    # Oracle-span diagnostic (2026-08-24): gold answer-span tokens are FORCED
    # to win the exact_topk selection at both levels (same rep budget — they
    # displace the lowest-ranked organic picks). Splits the verbatim-
    # extraction failure between selection (oracle EM >> headline) and rep
    # read-out capacity (oracle EM ≈ headline). Eval-only; requires live L0.
    ABLATION_ORACLE_SPAN = "oracle_span"
    # Modes under which the bgkit splice DELIBERATELY carries degenerate reps
    # (zeros / noise / a single zero placeholder). The decoder's spliced-rep
    # norm guard raises on sustained out-of-band reps (2026-08-22 escalation);
    # while one of these modes is active it must stand down — the degenerate
    # reps are the experiment, not a collapse.
    _DEGENERATE_REP_ABLATIONS = frozenset({"zeroed", "noise", "topics_only", "neither"})

    @property
    def _ablation_mode(self) -> str | None:
        return self.__dict__.get("_ablation_mode_value")

    @_ablation_mode.setter
    def _ablation_mode(self, mode: str | None) -> None:
        """Every assignment site (eval sweep, in-step gap probe, training-time
        random ablation) flows through here, so the decoder-side guard flag
        can never drift out of sync with the trainer's ablation state."""
        self.__dict__["_ablation_mode_value"] = mode
        expect_degenerate = mode in self._DEGENERATE_REP_ABLATIONS
        for dec in self._guarded_decoders():
            dec._rep_norm_guard_expect_degenerate = expect_degenerate

    def _guarded_decoders(self) -> list:
        """Decoders whose splice guard tracks the ablation state (both
        families under round-robin). getattr-guarded: runs on ``__new__``
        test doubles and before the decoders are built in ``__init__``."""
        by_family = getattr(self, "_decoders_by_family", None)
        if by_family:
            return list(by_family.values())
        dec = getattr(self, "decoder", None)
        return [dec] if dec is not None else []

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
        self._ablation_mode = None
        # READ-ONLY ablation-gap diagnostic probe. When > 0, the first N
        # per-repo (Option A) steps decode ONE representative group TWICE under
        # ``torch.no_grad()`` — once normally, once with the compressed
        # survivors (bgkit drill + browse node-reps) forced to zero — and log
        # ``ablation_loss_gap``. A gap ≈ 0 confirms the git-repro decoder is
        # ignoring the survivors. Pure diagnostic: no gradient, no optimizer
        # step, no mutation of real training state. Live-tunable via
        # control.json (``ablation_probe_steps``).
        self._ablation_probe_steps: int = int(
            self.step_cfg.get("ablation_probe_steps", 0) or 0,
        )
        self._encoder_lora_enabled = True
        self._direct_l1_trainable = False
        # Recursive-L1 (Phase 3) path-selective browse encode. Default OFF —
        # the non-recursive browse-text path is fully preserved. Set in
        # setup() from ``training.recursive_l1_tree.enabled``.
        self._recursive_l1: bool = False
        self._l1_tree_cache = None
        self._recursive_l1_retention: float = 0.15
        # BUG-1 FIX (frozen-policy L1 selection): the L1 survivorship head is
        # grad-starved on the recursive-tree distribution (aux off, hard mask
        # detached, no STE), so its ``tanh(base_raw/T)`` logits are frozen AND
        # tanh-saturated — the dual-ascent θ controller can never reach the
        # target keep-rate (θ pins at the tanh ceiling, keep-rate stuck ~0.78
        # vs target ~0.25). ``exact_topk`` switches EVERY L1 selection in the
        # recursive-tree forward to DETERMINISTIC per-sample top-k, realizing
        # the curriculum's target ratio directly from the (fixed) head logits
        # and bypassing θ. Default ``threshold`` preserves the θ-controlled
        # behaviour for every other dataset/run. Set in setup() from
        # ``recursive_l1_tree.selection_mode_l1``.
        self._selection_mode_l1: str = "threshold"
        self._selection_mode_l0: str = "threshold"
        # Fixed held-out text slice for the plain-LM health metric; loaded
        # once on first eval (see _lm_health_metrics).
        self._lm_health_chunks: list | None = None
        # QUERY-CONDITIONED drill nodes (2026-07-31 redesign). When True, EVERY
        # trajectory drill turn — head, on-path interior nodes, wrong-sibling
        # distractor node drills, and the retrieve leaf — is encoded LIVE per
        # sample, conditioned on the sample's TASK QUERY, instead of splicing
        # the static general-prompt shared-tree rep. Interior/head node turns
        # route through the generalized :meth:`_shared_tree_head_survivor`
        # (one query-conditioned L1 node forward over the node's shared-tree
        # child survivors, detach-reaccumulate) at ``drill_node_retention``;
        # the leaf re-encodes the raw diff at ``drill_leaf_retention.{l0,l1}``.
        # The shared tree itself (child-survivor source + the single deferred
        # tree backward) is UNCHANGED at the recursive ramps. Default OFF —
        # flag-off preserves exact legacy behavior. Set in setup() from
        # ``recursive_l1_tree.query_conditioned_drill_nodes``.
        self._query_conditioned_drill_nodes: bool = False
        # Retention for the per-sample query-conditioned node forwards (scalar
        # or {start,end,ramp_steps}); None → the recursive L1 ramp (legacy).
        self._drill_node_retention_cfg: float | dict | None = None
        # Retrieve-leaf drill retention override (net = l0 * l1); None per
        # level → legacy source (per-dataset ``l0_retention`` map for L0, the
        # recursive L1 ramp / sampled L1 for L1).
        self._drill_leaf_l0_retention_cfg: float | dict | None = None
        self._drill_leaf_l1_retention_cfg: float | dict | None = None
        # Full-backprop recursive-L1 (git-repro full-tree curriculum). When
        # True, the WHOLE window subtree is re-encoded live every step and
        # gradient flows to ALL nodes (on-path AND off-path) — no cached/
        # detached off-path reads. SMALL trees only (git-repro windows).
        self._recursive_l1_full_backprop: bool = False
        # PER-REPO full-backprop (git_commit_repro file-state reconstruction).
        # When True the whole window-0 subtree is encoded ONCE per repo
        # (shared, gradient-accumulated across the repo's file-samples) with a
        # GENERAL query-agnostic prompt; each file-sample's browse turns splice
        # the shared node reps while only its drill-down bgkit turn re-encodes
        # the drilled file-diff with the SPECIFIC commit/filename query. Set in
        # setup() from ``training.recursive_l1_tree.full_backprop_per_repo``.
        self._per_repo_full_backprop: bool = False
        # Per-repo cost knob (None = unlimited; int K subsamples K file-samples
        # per repo-batch, re-seeded per epoch). Set in setup() from
        # ``recursive_l1_tree.max_file_samples_per_repo``.
        self._max_file_samples_per_repo: int | None = None
        # Per-repo SIZE FILTERS — DROP over-threshold repos at grouping (None =
        # off). Set in setup() from ``recursive_l1_tree.max_repo_leaf_tokens`` /
        # ``max_repo_file_samples``.
        self._max_repo_leaf_tokens: int | None = None
        self._max_repo_file_samples: int | None = None
        # Per-repo subtree NODE-COUNT cap (None = off). Drops the heavy-tree
        # tail whose retained shared-tree graph is the OOM driver (esp. for the
        # inner-loop, which holds the tree across K steps). Set in setup() from
        # ``recursive_l1_tree.max_repo_tree_nodes``.
        self._max_repo_tree_nodes: int | None = None
        # Per-repo group-batch size. G=1: files are processed ONE-AT-A-TIME.
        # Group-batching (G>1) was a confirmed memory regression (concurrent
        # per-group graphs, ≥3 OOMs in the real no-sync run; the profile only
        # "fit" because cuda.synchronize masked the concurrency) with no speed
        # gain, so the default is 1. ``_encode_decode_group`` still works at
        # G=1 (one sample/group); the inner-loop subsets also run G=1.
        self._per_repo_sample_group_size: int = 1
        # INNER-LOOP per-repo mode (default OFF — the one-step detach-
        # reaccumulate path is the default + the warmup). When on (Model B,
        # sampler-subset + tree-cache): the tree is encoded ONCE per repo and
        # reused (progressively stale) across K optimizer steps, each on a
        # file-subset, backproping through the LIVE retained tree
        # (delayed-gradient — a deliberate choice). Knobs are live-tunable.
        self._per_repo_inner_loop: bool = False
        self._per_repo_inner_subset_size: int = 16
        self._per_repo_max_inner_steps: int = 12
        self._inner_loop_warmup_steps: int = 300
        # OPTION A (default OFF) — the crash-free amortized per-repo mode.
        # The ORIGINAL inner-loop (_per_repo_inner_loop) does K backwards
        # through the LIVE retained tree with optimizer.step() between them;
        # the step mutates the encoder weights R's retained graph depends on →
        # "variable needed for gradient modified by an inplace operation".
        # Option A separates the cadences: the DECODER reads a DETACHED tree
        # snapshot R_d and is stepped per file-subset (K updates, optimal batch
        # S); the ENCODER (every param R depends on — L0/L1/bridge/projection/
        # LoRA, shared by the live drill AND the tree) is NOT stepped until the
        # end, so R's graph stays valid for ONE final tree-backward, then the
        # encoder is stepped ONCE/repo. Whole-repo batches (NOT the subset
        # sampler); partitioned internally. Owns its optimizer cadence via
        # selective param-group stepping on the single self.optimizer.
        self._per_repo_option_a: bool = False
        # Option A subset cap (0 = UNLIMITED — process ALL contributing files
        # across K=ceil(n_contrib/S) subsets; the user's core motivation is NO
        # file cap). The legacy per_repo_max_inner_steps cap was for the
        # original inner-loop's STALENESS; Option A steps the encoder ONCE per
        # repo (no staleness) so it is unnecessary. Live-tunable.
        self._option_a_max_subsets: int = 0
        # Drill (per-file live L1 encode) activation-checkpoint threshold (0 =
        # off). When >0 AND the drill's flat content seqlen exceeds it, the
        # per-group run_l1_and_project forward in _run_l1_batch is wrapped in
        # torch.utils.checkpoint(use_reentrant=False) so its activations free
        # after the forward and recompute (bit-exact) during backward — bounding
        # Option A's deferred-encoder-grad peak to ~one file's drill instead of
        # O(files). Parallel to decode_gc_min_seqlen. θ is accumulated ONCE
        # outside the checkpoint (recompute must not double-count).
        self._drill_checkpoint_min_seqlen: int = 0
        # Inner-loop runtime state (Model B driver):
        #   _inner_loop_active        — whether the subset-emitting sampler +
        #                               inner-loop step path are live (set after
        #                               warmup; flipped in _pre_step_hook).
        #   _inner_loop_repo_key      — window-node root id of the repo whose
        #                               LIVE tree is currently cached/retained.
        #   _inner_loop_repo_steps    — inner-step index within the current repo.
        #   _repo_group_keys / _repo_dropped_keys / _repo_num_workers — stashed
        #                               sampler-build inputs so _pre_step_hook
        #                               can rebuild the loader at the switch.
        self._inner_loop_active: bool = False
        self._inner_loop_repo_key: str | None = None
        self._inner_loop_repo_steps: int = 0
        self._repo_group_keys: list[str] | None = None
        self._repo_dropped_keys: set[str] | None = None
        self._repo_num_workers: int = 0
        # SINGLE-FORWARD ACTIVATION-PEAK bounds (0 = off). These cap the two
        # unbounded per-forward seqlens that produced the ~100GB single-step
        # peak (steady cuda_allocated ~8-12GB):
        #   _max_l0_encode_tokens — truncate the per-leaf L0 encode buffer. The
        #     window-0 OLDEST commit is often the initial-import (whole initial
        #     codebase in one commit, up to the ~75k window token budget) → a
        #     single huge L0 varlen forward retained in the shared tree. The
        #     node-count filter does NOT bound this (one leaf, many tokens).
        #   _max_decode_tokens — skip samples whose rendered DECODE sequence
        #     (prefix + survivors + whole-file gold blob) exceeds the cap. The
        #     gold is a whole file → a pathologically long file ⇒ a single
        #     decoder (HF Qwen3.5) forward spike (the observed `_norm` OOM site).
        self._max_l0_encode_tokens: int = 0
        self._max_decode_tokens: int = 0
        # FIX 2: checkpoint each tree node's L1 forward (recompute on the final
        # backward) so the encode forward peak is bounded to ~one node instead
        # of O(nodes). Default off; on for the per-repo git_repro run.
        self._checkpoint_tree_encode: bool = False
        # SPEED: size thresholds above which checkpoint/GC engage. Small steps
        # stay below them and run un-checkpointed (fast); only the rare large
        # steps pay recompute. Live-tunable.
        self._tree_checkpoint_min_nodes: int = 64
        # Token-based tree-checkpoint gate (0 = off). Load-bearing for the FULL
        # (uncapped-L0) tree: forces the per-node checkpoint when total L0 leaf
        # tokens exceed this, so a low-node repo with a huge initial-import leaf
        # still bounds the encode forward + retained R to ~one node.
        self._tree_checkpoint_min_tokens: int = 0
        self._decode_gc_min_seqlen: int = 4096
        # Per-tree decision (set in _compute_shared_repo_tree from the cheap
        # browse-tree node count). True ⇒ tree large enough to checkpoint the
        # encode; default True so single-node / non-per-repo recursive paths
        # (where no tree-wide count is computed) preserve FIX 2/2b behaviour.
        self._tree_encode_ckpt_active: bool = True
        # --- Projection-output NORM-BAND regularizer (2026-07-31 collapse fix).
        # Off by default (only the git-repro config enables it). Keeps every
        # projected/spliced survivor-rep's L2 norm inside the decoder's readable
        # band (target[fam] = target_ratio[fam] * mean embed_tokens row-norm) via
        # a permissive hinge penalty, folded into the per-group reconstruction
        # loss so its gradient flows through projection_blocks into the backbone.
        # See _projection_norm_reg_term / setup() parse.
        self._proj_norm_reg_enabled: bool = False
        self._proj_norm_reg_weight: float = 0.0
        # Scalar tolerance = the fallback band width; per-family overrides live
        # in _proj_norm_reg_tolerances (mirrors _proj_norm_reg_target_ratios).
        self._proj_norm_reg_tolerance: float = 2.0
        self._proj_norm_reg_tolerances: dict[str, float] = {}
        self._proj_norm_reg_target_ratios: dict[str, float] = {}
        # Cached per-family mean embed_tokens row-L2-norm (computed once, lazily).
        self._proj_norm_reg_embed_ref_cache: dict[str, float] = {}
        # Per-step logging accumulator: {family: [rep_norm/embed_ref ratio ...]}.
        self._proj_norm_ratio_accum: dict[str, list[float]] = {}
        # Recursive retention ramps (float OR {start,end,ramp_steps}); resolved
        # by global_step in _recursive_l{0,1}_retention_now().  ``None`` keeps
        # the legacy scalar fallback (``_recursive_l1_retention``).
        self._recursive_l0_retention_cfg: float | dict | None = None
        self._recursive_l1_retention_cfg: float | dict | None = None
        # Transient L0 retention override active only while the shared repo
        # tree is being encoded (so the tree's L0 leaves use the recursive L0
        # ramp, not the per-dataset ``l0_retention``).  None = inactive.
        self._recursive_l0_override: float | None = None
        # Shared per-repo tree node reps (node_id -> (proj, l1out)) + a flag
        # telling _recursive_browse_node_reps to LOOK UP rather than re-encode.
        self._shared_tree_memo: dict | None = None
        self._per_repo_shared_tree_active: bool = False
        # Detach-and-reaccumulate (per-repo): the per-sample browse splices read
        # DETACHED leaf copies of the shared-tree node reps (``_shared_tree_
        # splice_reps``), so per-sample backward accumulates into those leaves'
        # ``.grad`` instead of re-running the shared tree Nx. ``_shared_tree_
        # used_nodes`` records which leaves a sample actually spliced (so the
        # single final reaccumulate backward only feeds used nodes).
        # ``_shared_tree_forward_count`` is a tripwire asserting the shared-tree
        # forward runs exactly once per optimizer microbatch.
        self._shared_tree_splice_reps: dict | None = None
        self._shared_tree_used_nodes: set | None = None
        self._shared_tree_forward_count: int = 0
        # Drill-down HEAD reaccumulate (git_commit_repro pure drill-down): the
        # per-sample task-query head re-encodes the window node's children LIVE,
        # but consumes DETACHED requires_grad copies of the shared-tree children
        # L1-outputs (``memo[child][1]``) so the per-group/per-subset backward
        # accumulates into ``child_l1_reps[child].grad`` instead of freeing the
        # shared-tree graph. ``_shared_tree_child_l1_used`` records the children
        # actually consumed so the single final tree backward feeds each child's
        # accumulated gradient back into ``memo[child][1]``.
        self._shared_tree_child_l1_reps: dict | None = None
        self._shared_tree_child_l1_used: set | None = None
        # EVAL-only: (root, decoder_family) of the shared repo tree currently
        # installed for eval decoding (see _ensure_eval_shared_tree). The family
        # is part of the key because round-robin alternates decoders per sample
        # and each family's projection emits a different survivor dim. None
        # outside eval.
        self._eval_shared_tree_key: tuple[str, str | None] | None = None
        # EVAL-only: per-evaluate() cache {(root, family): (memo, splice)} reused
        # across ablation-mode passes so each (repo, family) tree is encoded once,
        # not per mode.
        self._eval_tree_cache: dict | None = None
        # General (query-agnostic) compression prompt fed to BOTH L0 and L1 of
        # the shared repo tree.  Per-repo mode only; set in setup().
        self._recursive_general_prompt: str = DEFAULT_RECURSIVE_GENERAL_PROMPT
        # Training-time random ablation (capability regression prevention).
        # Rolled once per sample in _build_decoder_segments_core; disabled
        # during eval. Probabilities are cfg-driven and sum independently.
        tt_ablation = dict(self.step_cfg.get("training_time_ablation", {}) or {})
        self._p_skip_bgkit_training = float(tt_ablation.get("p_skip_bgkit", 0.15))
        self._p_skip_topic_training = float(tt_ablation.get("p_skip_topic", 0.15))
        self._p_noise_bgkit_training = float(tt_ablation.get("p_noise_bgkit", 0.0))
        import random as _random

        self._ablation_rng = _random.Random(int(cfg.get("seed", 42)))

        # Per-repo per-component timing (default OFF). When on,
        # _forward_backward_per_repo emits a ``per_repo_timing`` log per repo
        # with synced perf_counter splits — the synchronize() is taken ONLY
        # under this flag so the real run keeps full CPU/GPU overlap.
        self._profile_timing = bool(self.step_cfg.get("profile_timing", False))

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _stage(self) -> str:
        return str(self.step_cfg.get("stage", "A")).upper()

    def _resolve_dir(self, key: str, default: str) -> Path:
        value = self.step_cfg.get(key, None)
        if value:
            return Path(str(value))
        from bgkit.env import get_data_dir

        return get_data_dir() / default

    def _interleaved_decode_gc_mode(self) -> str | None:
        """Resolve the interleaved-decode force-GC mode (FIX 1b). Returns
        ``"reentrant"`` when ``training.decoder_gradient_checkpointing`` is
        truthy (so GC is forced on the inner model the interleaved decode runs),
        else ``None`` (no-op — non-opted phases unchanged)."""
        req = self.step_cfg.get(
            "decoder_gradient_checkpointing",
            self.cfg.compute.get("decoder_gradient_checkpointing", None)
            if hasattr(self.cfg, "compute") else None,
        )
        if req is None or req is False or (isinstance(req, str) and req.lower() in (
            "false", "off", "0", "none", "",
        )):
            return None
        return "reentrant"

    def _propagate_decode_gc_min_seqlen(self) -> None:
        """Push ``self._decode_gc_min_seqlen`` onto every decoder instance.
        Called after the recursive-l1 config is parsed (decoders are built
        earlier in setup) and from the live-config handler."""
        for dec in getattr(self, "_decoders_by_family", {}).values():
            if dec is not None:
                dec._decode_gc_min_seqlen = self._decode_gc_min_seqlen

    def _build_decoder_for_family(self, decoder_cfg):
        """Build one ``ReconstructionDecoder`` + tokenizer from a decoder config
        block, with identical CE-impl / gradient-checkpointing / tokenizer /
        Falcon-template wiring. Factored out of :meth:`setup` so round-robin can
        build both families (qwen35 + falcon_h1). Returns ``(decoder, tokenizer)``.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from bgkit.data.chat_template import patch_falcon_h1_chat_template

        decoder_name = decoder_cfg["backbone_name"]
        family = normalize_decoder_family(decoder_cfg.get("family", "qwen35"))
        attn_impl = resolve_decoder_attention_implementation(
            self.cfg.compute.get(
                "decoder_attention_implementation",
                self.cfg.compute.get("attention_implementation", "auto"),
            ),
            decoder_family=family,
        )
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        backbone = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
        ).to(self.device)
        hidden = backbone.get_input_embeddings().weight.shape[1]
        decoder = ReconstructionDecoder(
            backbone, hidden_dim=hidden, decoder_family=family,
        )
        decoder.set_lm_ce_impl(
            self.step_cfg.get(
                "decoder_ce_impl", self.cfg.compute.get("decoder_ce_impl", None),
            ),
        )
        decoder.set_lm_ce_strict(
            self.step_cfg.get(
                "decoder_ce_strict", self.cfg.compute.get("decoder_ce_strict", None),
            ),
        )
        decoder.train()
        # Decoder GC via the reentrant mechanism the summarization trainer uses
        # (works with FLA DeltaNet/Mamba in both families; see the primary decoder
        # build above).
        from bgkit.training.gradient_utils import (
            maybe_enable_decoder_gradient_checkpointing,
        )

        maybe_enable_decoder_gradient_checkpointing(backbone, self.cfg)
        # FIX 1b: force per-layer reentrant GC inside the interleaved decode
        # (the OOM path) — the wrapper-level enable above did not engage there.
        # SPEED: GC only engages when seqlen > _decode_gc_min_seqlen.
        decoder._interleaved_gc_mode = self._interleaved_decode_gc_mode()
        decoder._decode_gc_min_seqlen = self._decode_gc_min_seqlen
        tokenizer = AutoTokenizer.from_pretrained(
            decoder_name, trust_remote_code=True,
        )
        if patch_falcon_h1_chat_template(tokenizer):
            logger.info("falcon_chat_template_patched", decoder_tokenizer=decoder_name)
        logger.info(
            "phase2_kb_decoder_built",
            family=family, model=decoder_name,
            ce_impl=decoder.lm_ce_impl, ce_strict=decoder.lm_ce_strict,
        )
        return decoder, tokenizer

    # ------------------------------------------------------------------
    # setup()
    # ------------------------------------------------------------------

    def setup(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._live_l0 = bool(self.step_cfg.get("live_l0", self._stage() == "A"))

        # --- Decoder ---
        decoder_cfg = self.step_cfg.get("model", {}).get("decoder", self.cfg.model.decoder)
        decoder_name = decoder_cfg.backbone_name
        decoder_family = normalize_decoder_family(decoder_cfg.get("family", "qwen35"))
        self._decoder_family = decoder_family
        decoder_attention_impl = resolve_decoder_attention_implementation(
            self.cfg.compute.get(
                "decoder_attention_implementation",
                self.cfg.compute.get("attention_implementation", "auto"),
            ),
            decoder_family=decoder_family,
        )
        decoder_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        decoder_backbone = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            trust_remote_code=True,
            torch_dtype=decoder_dtype,
            attn_implementation=decoder_attention_impl,
        ).to(self.device)
        hidden = decoder_backbone.get_input_embeddings().weight.shape[1]
        self.decoder = ReconstructionDecoder(
            decoder_backbone,
            hidden_dim=hidden,
            decoder_family=self._decoder_family,
        )
        self.decoder.set_lm_ce_impl(
            self.step_cfg.get(
                "decoder_ce_impl",
                self.cfg.compute.get("decoder_ce_impl", None),
            )
        )
        self.decoder.set_lm_ce_strict(
            self.step_cfg.get(
                "decoder_ce_strict",
                self.cfg.compute.get("decoder_ce_strict", None),
            )
        )
        logger.info(
            "phase2_kb_decoder_ce_impl_selected",
            impl=self.decoder.lm_ce_impl,
            strict=self.decoder.lm_ce_strict,
        )
        self.decoder.train()

        # Activation checkpointing on the decoder is a memory/speed tradeoff.
        # Default off; enable explicitly via
        # ``training.activation_checkpointing.decoder: true`` when a phase
        # needs the extra activation headroom.
        ac_cfg = self.step_cfg.get("activation_checkpointing", {}) or {}
        # Decoder GC via the SAME mechanism the summarization round-robin trainer
        # uses (training.decoder_gradient_checkpointing: "reentrant"). Reentrant
        # mode skips the use_reentrant=False tensor-match check, so it works with
        # the FLA DeltaNet/Mamba kernels in Qwen3.5 + Falcon-H1 (use_reentrant=False
        # raised CheckpointError: 666 vs 382 saved tensors).
        from bgkit.training.gradient_utils import (
            maybe_enable_decoder_gradient_checkpointing,
        )

        maybe_enable_decoder_gradient_checkpointing(decoder_backbone, self.cfg)
        # FIX 1b: force per-layer reentrant GC inside the interleaved decode
        # (the OOM path) for the primary decoder too.
        self.decoder._interleaved_gc_mode = self._interleaved_decode_gc_mode()
        self.decoder._decode_gc_min_seqlen = self._decode_gc_min_seqlen
        self._checkpoint_encoder = bool(ac_cfg.get("encoder", False))
        logger.info(
            "phase2_kb_encoder_activation_checkpointing_resolved",
            enabled=self._checkpoint_encoder,
        )
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

        # Falcon-H1's chat template has an off-by-one bug at the tool
        # branch (closing <|im_end|> after </tool_response> skipped when
        # followed by assistant). Without the patch, our trajectory
        # builder's prefix-extension assertion in
        # bgkit.data.bgkit_tool_template fires on the first browse turn.
        from bgkit.data.chat_template import patch_falcon_h1_chat_template

        if patch_falcon_h1_chat_template(self.tokenizer):
            logger.info(
                "falcon_chat_template_patched",
                decoder_tokenizer=decoder_name,
            )

        # --- Round-robin second decoder (dual-family training) ---
        # By default Phase 2 trains the single decoder built above. When
        # ``round_robin: true`` (Stage A handoff from the summarization run) we
        # ALSO build the other family's decoder + tokenizer; the encoder's
        # survivors then train against both, with one family picked per batch
        # (``_pick_decoder_family`` / ``_set_active_decoder``). The primary
        # decoder above is the qwen35 family (model.decoder default == the
        # decoder_qwen block); the secondary is Falcon.
        self._round_robin = bool(self.step_cfg.get("round_robin", False))
        self._qwen_decoder_prob = float(self.step_cfg.get("qwen_decoder_prob", 0.3))
        self._microbatch_counter = 0
        self._decoders_by_family = {self._decoder_family: self.decoder}
        self._tokenizer_by_family = {self._decoder_family: self.tokenizer}
        if self._round_robin:
            other_family = "falcon_h1" if self._decoder_family == "qwen35" else "qwen35"
            other_key = "decoder_falcon" if other_family == "falcon_h1" else "decoder_qwen"
            other_cfg = self.step_cfg.get(other_key)
            if not other_cfg:
                raise ValueError(
                    f"round_robin=true requires a training.{other_key} block "
                    f"(building the {other_family} decoder)"
                )
            other_dec, other_tok = self._build_decoder_for_family(other_cfg)
            self._decoders_by_family[other_family] = other_dec
            self._tokenizer_by_family[other_family] = other_tok
            logger.info(
                "phase2_kb_round_robin_decoders",
                families=sorted(self._decoders_by_family),
                primary=self._decoder_family,
                qwen_decoder_prob=self._qwen_decoder_prob,
            )

        # --- Encoder ---
        self._load_encoder()
        self.encoder.set_active_decoder_family(self._decoder_family)

        # Per-task learnable L0 prompts (gated; default OFF). When
        # l0_prompt_tokens > 0, attach a learnable (P, D) prompt per dataset to
        # the encoder so it is saved/loaded with the encoder state dict and L0's
        # within-document compression becomes task-conditioned. With
        # l0_freeze_backbone the L0 backbone is frozen (prompt-tuning). Off =>
        # unchanged live trainable-L0 path.
        self._l0_prompt_tokens = int(self.step_cfg.get("l0_prompt_tokens", 0) or 0)
        self._l0_freeze_backbone = bool(
            self.step_cfg.get("l0_freeze_backbone", False)
        )
        if self._l0_prompt_tokens > 0:
            import torch.nn as _nn

            prompt_count = self._l0_prompt_tokens
            prompt_dim = int(self.encoder.l0.survive_embedding.shape[-1])
            _ref = next(self.encoder.l0.parameters())
            self.encoder.l0_task_prompts = _nn.ParameterDict({
                str(_ds): _nn.Parameter(torch.zeros(prompt_count, prompt_dim))
                for _ds in list(self.step_cfg.get("datasets", []))
            }).to(device=self.device, dtype=_ref.dtype)
            logger.info(
                "phase2_kb_l0_task_prompts_built",
                n_datasets=len(self.encoder.l0_task_prompts),
                prompt_tokens=prompt_count, hidden_dim=prompt_dim,
                freeze_backbone=self._l0_freeze_backbone,
            )

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

        # Stage B (and the prompt-fit bridge stage) must continue from the
        # complete Phase-2 state, not merely use Stage A to build the detached
        # L0 cache.  Apply the deferred overlay only after task prompts and
        # optional LoRA wrappers exist, otherwise their checkpoint keys have no
        # destination and are silently dropped by ``strict=False``.
        self._apply_phase2_handoff_state()

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

        # --- Recursive-L1 tree (Phase 3, optional) ---
        # Two modes, both flag-gated (default OFF — the non-recursive trainer
        # never touches either):
        #
        #  * CACHED PATH-SELECTIVE (default recursive mode): off-path
        #    browse-tree nodes read their query-agnostic subtree summaries
        #    (built offline by scripts/precompute_l1_tree.py) from the L1-tree
        #    cache, DETACHED; only the live search path carries gradient. For
        #    BIG trees (pubmedqa/KILT) where the whole subtree cannot fit live.
        #
        #  * FULL-BACKPROP (``full_backprop: true``): the WHOLE window subtree
        #    is re-encoded LIVE every step and gradient flows to ALL nodes
        #    (on-path + off-path). No L1-tree cache is needed (only the L0
        #    cache of the file-diff leaves). SMALL trees ONLY — git-repro
        #    windows (~16-64 commits). Must NOT be enabled on big trees.
        rec_cfg = self.step_cfg.get("recursive_l1_tree", {}) or {}
        self._recursive_l1 = bool(rec_cfg.get("enabled", False))
        self._recursive_l1_full_backprop = bool(rec_cfg.get("full_backprop", False))
        # Per-repo batching is a refinement of full-backprop: the whole
        # window-0 subtree is encoded ONCE per repo (shared, grad-accumulated)
        # instead of per file-sample.  Implies full_backprop.
        self._per_repo_full_backprop = bool(
            rec_cfg.get("full_backprop_per_repo", False),
        )
        if self._per_repo_full_backprop:
            self._recursive_l1_full_backprop = True
        # Per-level SELECTION MODE — first-class, parsed UNCONDITIONALLY.
        # ``training.selection_mode: {l0: threshold|exact_topk, l1: ...}``;
        # the legacy ``recursive_l1_tree.selection_mode_l1`` is honored as the
        # L1 fallback. (2026-08-22: the legacy key was parsed only under
        # ``recursive_l1_tree.enabled`` — every widenet config's
        # ``selection_mode_l1: exact_topk`` was silently ignored and BOTH
        # levels ran threshold mode: L0 on the untrained base keeps 0.0-0.3%
        # ("confidently silent"), then the aux losses flip the saturated head
        # to ~100% within ~40 steps → 50-78K-token decodes.) exact_topk
        # realizes the configured ratio deterministically (ceil ≥ 1) and skips
        # that level's dual-ascent θ step (θ is irrelevant under top-k).
        sel_cfg = dict(self.step_cfg.get("selection_mode", {}) or {})
        legacy_l1 = rec_cfg.get("selection_mode_l1", None)
        self._selection_mode_l0 = str(sel_cfg.get("l0", "threshold"))
        self._selection_mode_l1 = str(
            sel_cfg.get("l1", legacy_l1 if legacy_l1 is not None else "threshold"),
        )
        for _lvl, _mode in (("l0", self._selection_mode_l0), ("l1", self._selection_mode_l1)):
            if _mode not in {"threshold", "exact_topk"}:
                raise ValueError(
                    f"selection_mode.{_lvl} must be 'threshold' or 'exact_topk'; got {_mode!r}"
                )
        logger.info(
            "phase2_kb_selection_modes",
            l0=self._selection_mode_l0,
            l1=self._selection_mode_l1,
        )
        if self._recursive_l1:
            # Retention may be a scalar (legacy) OR a step-interpolated ramp
            # ``{start, end, ramp_steps}`` (per-repo curriculum). Store the raw
            # cfg for ramp resolution and keep a resolved scalar for the legacy
            # path-selective code + back-compat.
            self._recursive_l1_retention_cfg = rec_cfg.get(
                "l1_retention", self.step_cfg.get("l1_retention", 0.15),
            )
            self._recursive_l0_retention_cfg = rec_cfg.get("l0_retention", None)
            self._recursive_l1_retention = self._interp_recursive_ratio(
                self._recursive_l1_retention_cfg, default=0.15,
            )
            self._recursive_general_prompt = str(
                rec_cfg.get("general_prompt", DEFAULT_RECURSIVE_GENERAL_PROMPT),
            )
            # BUG-1 (L1 selection mode for the recursive tree + drill) is now
            # parsed unconditionally above (``training.selection_mode`` with the
            # legacy ``recursive_l1_tree.selection_mode_l1`` fallback).
            # QUERY-CONDITIONED drill nodes (2026-07-31): per-sample live
            # task-query re-encode of EVERY trajectory drill node (head +
            # on-path interiors + distractor node drills) over the shared
            # tree's child survivors, plus the leaf retention override.
            # Flag-off = exact legacy behavior (static node splices, tree-ramp
            # drill ratios). All four knobs live-tunable via control.json.
            self._query_conditioned_drill_nodes = bool(
                rec_cfg.get("query_conditioned_drill_nodes", False),
            )
            self._drill_node_retention_cfg = rec_cfg.get(
                "drill_node_retention", None,
            )
            _dlr = rec_cfg.get("drill_leaf_retention", {}) or {}
            self._drill_leaf_l0_retention_cfg = _dlr.get("l0", None)
            self._drill_leaf_l1_retention_cfg = _dlr.get("l1", None)
            if self._query_conditioned_drill_nodes:
                logger.info(
                    "phase2_kb_query_conditioned_drill_nodes_enabled",
                    drill_node_retention=self._drill_node_retention_cfg,
                    drill_leaf_l0=self._drill_leaf_l0_retention_cfg,
                    drill_leaf_l1=self._drill_leaf_l1_retention_cfg,
                    note=(
                        "every trajectory drill node (head + on-path + "
                        "distractors) re-encoded LIVE per sample under the "
                        "task query; leaf at drill_leaf_retention; shared "
                        "tree unchanged at the recursive ramps"
                    ),
                )
            # Per-repo cost knob: each file-sample's backward recomputes the
            # GC'd shared-tree forward once (Nx per repo), and the staged drill
            # graphs scale with the repo's file-sample count. ``null``
            # (default) = use every file-sample; an int K subsamples up to K
            # file-samples per repo-batch (re-seeded per epoch so coverage
            # varies across epochs). Lets the GPU profile bound per-step
            # cost + memory without changing the data.
            _mfs = rec_cfg.get("max_file_samples_per_repo", None)
            self._max_file_samples_per_repo = (
                int(_mfs) if _mfs is not None else None
            )
            # Per-repo SIZE FILTERS (DROP over-threshold repos at grouping, vs
            # max_file_samples_per_repo which only SUBSAMPLES a kept repo).
            # null = off. See _build_dataloaders for semantics.
            _mlt = rec_cfg.get("max_repo_leaf_tokens", None)
            self._max_repo_leaf_tokens = int(_mlt) if _mlt is not None else None
            _mrfs = rec_cfg.get("max_repo_file_samples", None)
            self._max_repo_file_samples = int(_mrfs) if _mrfs is not None else None
            _mrtn = rec_cfg.get("max_repo_tree_nodes", None)
            self._max_repo_tree_nodes = int(_mrtn) if _mrtn is not None else None
            _grp = rec_cfg.get(
                "per_repo_sample_group_size", self._per_repo_sample_group_size,
            )
            self._per_repo_sample_group_size = max(1, int(_grp or 1))
            # Inner-loop knobs (default OFF). Live-tunable. Model B driver wired
            # below (sampler-subset + tree-cache; no base_trainer.train()
            # change). For the first ``inner_loop_warmup_steps`` global steps
            # the exact one-step detach-reaccumulate path runs (clean gradient);
            # the switch to inner-loop is a sampler rebuild in _pre_step_hook.
            self._per_repo_inner_loop = bool(rec_cfg.get("per_repo_inner_loop", False))
            self._per_repo_inner_subset_size = max(
                1, int(rec_cfg.get("per_repo_inner_subset_size", 16) or 1),
            )
            self._per_repo_max_inner_steps = max(
                1, int(rec_cfg.get("per_repo_max_inner_steps", 12) or 1),
            )
            self._inner_loop_warmup_steps = max(
                0, int(rec_cfg.get("inner_loop_warmup_steps", 300) or 0),
            )
            # OPTION A (default OFF): crash-free amortized per-repo mode. Reuses
            # the subset-size / max-steps knobs above (S, K cap) but uses
            # WHOLE-repo batches (not the subset sampler) + decoder-per-subset /
            # encoder-per-repo split-cadence stepping. Mutually exclusive with
            # the legacy live-tree inner loop.
            self._per_repo_option_a = bool(rec_cfg.get("per_repo_option_a", False))
            if self._per_repo_option_a:
                raise ValueError(
                    "recursive_l1_tree.per_repo_option_a was removed because it "
                    "performed multiple adaptive decoder updates against one "
                    "stale tree, then one encoder update from gradients produced "
                    "by several decoder states. Use the coherent one-step "
                    "per-repo path (per_repo_option_a=false)."
                )
            # Option A: uncapped subsets (process all files) + per-file drill
            # activation-checkpoint (bounds the deferred-encoder-grad peak).
            self._option_a_max_subsets = max(
                0, int(rec_cfg.get("option_a_max_subsets", 0) or 0),
            )
            self._drill_checkpoint_min_seqlen = max(
                0, int(rec_cfg.get("drill_checkpoint_min_seqlen", 0) or 0),
            )
            # Single-forward activation-peak bounds (0 = off).
            self._max_l0_encode_tokens = max(
                0, int(rec_cfg.get("max_l0_encode_tokens", 0) or 0),
            )
            self._max_decode_tokens = max(
                0, int(rec_cfg.get("max_decode_tokens", 0) or 0),
            )
            self._checkpoint_tree_encode = bool(
                rec_cfg.get("checkpoint_tree_encode", False),
            )
            # SPEED: make encode-checkpoint + decode-GC CONDITIONAL on size so
            # the typical small step skips recompute. The tree encode is only
            # checkpointed when n_nodes > tree_checkpoint_min_nodes; the decode
            # is only GC'd when seqlen > decode_gc_min_seqlen.
            self._tree_checkpoint_min_nodes = max(
                0, int(rec_cfg.get("tree_checkpoint_min_nodes", 64) or 0),
            )
            self._tree_checkpoint_min_tokens = max(
                0, int(rec_cfg.get("tree_checkpoint_min_tokens", 0) or 0),
            )
            self._decode_gc_min_seqlen = max(
                0, int(rec_cfg.get("decode_gc_min_seqlen", 4096) or 0),
            )
            # rec_cfg is parsed AFTER the decoders are built, so push the
            # resolved threshold onto every decoder instance now.
            self._propagate_decode_gc_min_seqlen()
            if self._per_repo_full_backprop:
                self._l1_tree_cache = None
                logger.warning(
                    "phase2_kb_recursive_l1_per_repo_full_backprop_enabled",
                    l0_retention=self._recursive_l0_retention_cfg,
                    l1_retention=self._recursive_l1_retention_cfg,
                    general_prompt=self._recursive_general_prompt,
                    note=(
                        "window-0 subtree encoded ONCE per repo (shared, "
                        "grad-accumulated via retain_graph across the repo's "
                        "file-samples); browse turns splice shared node reps, "
                        "drill-down bgkit re-encodes the file-diff with the "
                        "specific commit/filename query; SMALL trees ONLY"
                    ),
                )
            elif self._recursive_l1_full_backprop:
                # No L1-tree cache precompute — the tree is encoded live each
                # step from the L0 leaf cache.
                self._l1_tree_cache = None
                logger.warning(
                    "phase2_kb_recursive_l1_full_backprop_enabled",
                    l1_retention=self._recursive_l1_retention,
                    note=(
                        "whole window subtree re-encoded LIVE each step; "
                        "gradient flows to ALL nodes; SMALL trees ONLY "
                        "(git-repro windows) — do NOT enable on big trees "
                        "(pubmedqa/KILT)"
                    ),
                )
            else:
                from bgkit.data.l0_cache import SurvivorBlockCache

                tree_cache_dir = self._resolve_dir(
                    "recursive_l1_tree_cache_dir", "l1_tree_cache_kb",
                )
                if rec_cfg.get("cache_dir"):
                    tree_cache_dir = Path(str(rec_cfg.get("cache_dir")))
                self._l1_tree_cache = SurvivorBlockCache(str(tree_cache_dir))
                logger.info(
                    "phase2_kb_recursive_l1_enabled",
                    cache_dir=str(tree_cache_dir),
                    l1_retention=self._recursive_l1_retention,
                )

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
        self._l0_ratio_sampler_cfg = self._anchor_free_if_topk("l0", self._l0_ratio_sampler_cfg)
        self._l1_ratio_sampler_cfg = self._anchor_free_if_topk("l1", self._l1_ratio_sampler_cfg)
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
        # Master switch for survivorship-head supervision and the pending
        # LevelOutputs that feed it.  Keep the default backward-compatible for
        # generic Phase-2 jobs; retrieval jobs are expected to enable this so a
        # hard top-k policy is not permanently tied to an unrelated checkpoint's
        # ranking.  Theta control remains available when this is disabled.
        self._survivorship_aux = bool(surv_cfg.get("survivorship_aux", False))
        self._ratio_loss_weight = float(surv_cfg.get("ratio_loss_weight", 0.1))
        self._decisiveness_loss_weight = float(
            surv_cfg.get("decisiveness_loss_weight", 0.05),
        )
        self._relevance_loss_weight = float(surv_cfg.get("relevance_loss_weight", 0.05))
        # v5: span-level relevance — push the gold answer span's positions to
        # survive at L0 AND L1 (target prob 1.0). 0.0 = off (default).
        self._span_relevance_weight = float(surv_cfg.get("span_relevance_weight", 0.0))
        # Per-term gradient attribution at the L1 selector head, every N steps
        # (0 = off). Answers "which aux loss owns this head's gradient?", which
        # no existing metric could: grad_norm/l1_head reports the SUM.
        self._aux_grad_attribution_every = int(
            surv_cfg.get("grad_attribution_every", 0) or 0
        )

        # Which per-level aux weights were EXPLICITLY configured (see _aux_weight).
        self._level_explicit_aux = {
            lvl: set(dict(surv_cfg.get(lvl, {}) or {}).keys()) for lvl in ("l0", "l1")
        }
        self._surv_l0 = resolve_level_loss_cfg(surv_cfg.get("l0", {}))
        self._surv_l1 = resolve_level_loss_cfg(surv_cfg.get("l1", {}))

        # --- Projection-output norm-band regularizer (collapse fix). Config
        # block ``survivorship.projection_norm_reg``; absent → disabled (so only
        # the git-repro run turns it on, all other phase2_kb runs are untouched).
        pnr_cfg = surv_cfg.get("projection_norm_reg", {}) or {}
        self._proj_norm_reg_enabled = bool(pnr_cfg.get("enabled", False))
        self._proj_norm_reg_weight = float(pnr_cfg.get("weight", 0.1))
        # ``tolerance`` accepts a scalar (back-compat, same band width for every
        # family) OR a per-family dict {family: tol} mirroring ``target_ratio``
        # — needed so one family's band can be tightened without penalizing
        # another that is already in-band (qwen35 drift vs stable falcon_h1,
        # 2026-08-01). The scalar stays as the fallback for families absent
        # from the dict.
        _tol = pnr_cfg.get("tolerance", 2.0)
        if hasattr(_tol, "items"):  # dict / OmegaConf DictConfig → per-family
            self._proj_norm_reg_tolerances = {
                str(k): float(v) for k, v in _tol.items()
            }
            self._proj_norm_reg_tolerance = 2.0
        else:
            self._proj_norm_reg_tolerances = {}
            self._proj_norm_reg_tolerance = float(_tol)
        _tr = pnr_cfg.get("target_ratio", {}) or {}
        # OmegaConf DictConfig → plain dict[str,float].
        self._proj_norm_reg_target_ratios = {
            str(k): float(v)
            for k, v in (_tr.items() if hasattr(_tr, "items") else {})
        }
        self._proj_norm_reg_embed_ref_cache = {}
        if self._proj_norm_reg_enabled:
            logger.info(
                "phase2_kb_projection_norm_reg_enabled",
                weight=self._proj_norm_reg_weight,
                tolerance=self._proj_norm_reg_tolerance,
                tolerance_by_family=self._proj_norm_reg_tolerances,
                target_ratio=self._proj_norm_reg_target_ratios,
            )

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
        l0_path = (
            _l0_block.get("path", None)
            if _l0_block is not None and hasattr(_l0_block, "get")
            else None
        )
        if self._surv_l0.moment_match_weight > 0 and l0_path:
            self._ref_moments_l0 = load_reference_moments(l0_path)
        _l1_block = mm_ref.get("l1", None) if hasattr(mm_ref, "get") else None
        l1_path = (
            _l1_block.get("path", None)
            if _l1_block is not None and hasattr(_l1_block, "get")
            else None
        )
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
        # The module patcher is Qwen-specific. Keep using it for the Qwen
        # encoder, but do not apply it to Falcon-H1 decoders; Falcon-H1's hot
        # path is its Mamba/causal-conv kernels plus the generic LM CE kernel.
        if bool(self.step_cfg.get("use_liger", True)):
            from bgkit.utils.liger_integration import (
                apply_liger_to_decoder,
                apply_liger_to_qwen35,
            )

            decoder_family = getattr(self.decoder, "decoder_family", "qwen35")
            decoder_qwen_liger = decoder_family == "qwen35"
            use_liger_ce = (
                bool(self.step_cfg.get("use_liger_ce", True))
                and decoder_qwen_liger
                and self.decoder.lm_ce_impl in {"auto", "liger"}
            )
            enc_patched = apply_liger_to_qwen35(self.encoder)
            # apply_liger_to_decoder no-ops on Falcon, matching
            # decoder_qwen_liger.
            dec_patched = apply_liger_to_decoder(self.decoder)
            if use_liger_ce:
                self.decoder.enable_liger_ce(True)
            logger.info(
                "phase2_kb_liger_applied",
                encoder_modules=enc_patched,
                decoder_modules=dec_patched,
                decoder_qwen_liger=decoder_qwen_liger,
                use_liger_ce=use_liger_ce,
                decoder_ce_impl=self.decoder.lm_ce_impl,
            )

        # --- Model container + optimizer ---
        self.model = _KBModel(
            encoder=self.encoder,
            decoder=None if self._round_robin else self.decoder,
            decoders=self._decoders_by_family if self._round_robin else None,
            lora_router=self.lora_router,
            topic_embeddings=self.topic_embeddings,
        ).to(self.device)
        if self._round_robin:
            # self.decoder is a live pointer into the container's ModuleDict;
            # keep it aimed at the primary family until the first per-batch swap.
            self._decoders_by_family = dict(self.model.decoders)
            self.decoder = self._decoders_by_family[self._decoder_family]
        self._freeze_decoder_embeddings()
        self.optimizer = self._create_optimizer(
            self._build_optimizer_groups(),
            default_lr=float(self.step_cfg.get("lr", 1e-4)),
        )
        # Option A: snapshot the DECODER param ids so we can classify each
        # optimizer param-group as decoder vs encoder for split-cadence
        # stepping (decoder-per-subset, encoder-per-repo). Captured by IDENTITY
        # so it survives Muon's 2D/1D group splitting (which preserves params,
        # only regroups them). Everything NOT a decoder param is "encoder"
        # (L0/L1/bridge/projection/LoRA/topic — all params R's tree graph
        # depends on, so none may be stepped until the final tree-backward).
        self._option_a_decoder_param_ids = frozenset(
            id(p)
            for dec in self._all_decoders()
            for p in dec.parameters()
            if p.requires_grad
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

        phase1_ckpt = self._resolve_phase1_checkpoint(required=False)
        phase1_path = Path(str(phase1_ckpt)) if phase1_ckpt else None

        stage_a_ckpt = (
            None
            if getattr(self, "_phase2_handoff_is_phase1", False)
            else getattr(self, "_phase2_handoff_checkpoint", None)
        )
        if stage_a_ckpt is None:
            stage_a_ckpt = self._resolve_stage_a_checkpoint(
                required=bool(self.step_cfg.get("stage_a_checkpoint")),
            )
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
                if getattr(self, "_phase2_handoff_checkpoint", None) is not None:
                    raise L0CacheManifestMismatch(
                        f"Dataset {name!r} has no cache_manifest.json, but "
                        f"Stage B loads Phase-2 handoff {stage_a_path}. Rebuild "
                        "the cache with precompute_l0_subset.py using the exact "
                        "same --phase1-checkpoint and --stage-a-checkpoint."
                    )
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

    def _phase1_auto_candidates(self) -> tuple[str, ...]:
        # ``_decoder_family`` is already canonicalized via
        # ``normalize_decoder_family`` at setup time, so accept only the
        # canonical names here. Adding a second tolerant set behind the
        # canonical one would let an un-normalized value silently bypass
        # the chain.
        family = str(getattr(self, "_decoder_family", "qwen35") or "qwen35")
        if family == "falcon_h1":
            return ("phase1_falcon_l1", "phase1_falcon_l0")
        return ("phase1_step6",)

    def _resolve_phase1_checkpoint(self, *, required: bool = True) -> str | None:
        phase1_ckpt = self.step_cfg.get("phase1_checkpoint")
        if not phase1_ckpt:
            if required:
                raise ValueError(
                    "phase2_kb requires training.phase1_checkpoint "
                    "(the Phase 1 encoder)"
                )
            return None
        if str(phase1_ckpt) != "auto":
            return str(phase1_ckpt)

        from bgkit.training.checkpoint_registry import resolve_checkpoint

        checkpoint_dir = Path(str(self.cfg.get("checkpoint_dir", "checkpoints")))
        candidate_phases = self._phase1_auto_candidates()
        errors: list[str] = []
        for phase in candidate_phases:
            try:
                return str(resolve_checkpoint(
                    checkpoint_dir,
                    phase=phase,
                    metric="eval/loss",
                    lower_is_better=True,
                ))
            except ValueError as exc:
                errors.append(str(exc))
        if not required:
            return None
        raise ValueError(
            "phase1_checkpoint=auto could not resolve any candidate phase "
            f"for decoder family {getattr(self, '_decoder_family', 'qwen35')!r}: "
            f"{', '.join(candidate_phases)}. "
            + " | ".join(errors)
        )

    def _resolve_stage_a_checkpoint(self, *, required: bool = True) -> str | None:
        stage_a_ckpt = self.step_cfg.get("stage_a_checkpoint")
        if not stage_a_ckpt:
            if required:
                raise ValueError("stage_a_checkpoint is required")
            return None
        if str(stage_a_ckpt) != "auto":
            return str(stage_a_ckpt)

        from bgkit.training.checkpoint_registry import CheckpointRegistry

        checkpoint_dir = Path(str(self.cfg.get("checkpoint_dir", "checkpoints")))
        registry = CheckpointRegistry(checkpoint_dir)
        registry.backfill(checkpoint_dir)
        family = str(getattr(self, "_decoder_family", "qwen35") or "qwen35")

        # Prompt-fit is the latest intentional Stage-A-family handoff for the
        # normal Stage B pipeline.  The prompt-fit stage itself must start from
        # plain Stage A.  Operators can override the accepted order for custom
        # pipelines via ``stage_a_source_stages``.
        configured_stages = self.step_cfg.get("stage_a_source_stages", None)
        if configured_stages:
            preferred_stages = [str(s).upper() for s in configured_stages]
        elif self._stage() == "PROMPT_FIT":
            preferred_stages = ["A"]
        else:
            preferred_stages = ["PROMPT_FIT", "A"]
        stage_rank = {stage: idx for idx, stage in enumerate(preferred_stages)}

        candidates = []
        for entry in registry.list_entries(phase="phase2_kb", status="completed"):
            metrics = entry.metrics or {}
            eval_loss = metrics.get("eval/loss", metrics.get("eval/eval/loss"))
            if not entry.on_disk or eval_loss is None:
                continue
            snapshot = entry.config_snapshot or {}
            entry_stage = str(snapshot.get("stage", "")).upper()
            if entry_stage not in stage_rank:
                continue
            decoder_cfg = (snapshot.get("model") or {}).get("decoder") or {}
            entry_family = normalize_decoder_family(decoder_cfg.get("family", "qwen35"))
            if entry_family != family:
                continue
            candidates.append((entry, stage_rank[entry_stage], float(eval_loss)))

        if candidates:
            best, _rank, _loss = min(
                candidates, key=lambda item: (item[1], item[2]),
            )
            return str(checkpoint_dir / best.name)
        if not required:
            return None
        raise ValueError(
            "stage_a_checkpoint=auto could not resolve a completed Stage A "
            f"phase2_kb checkpoint for decoder family {family!r}; accepted "
            f"stages={preferred_stages}"
        )

    def _load_encoder(self) -> None:
        from bgkit.models.encoder import BgKITEncoder
        from bgkit.training.checkpointing import load_checkpoint

        phase1_ckpt = self._resolve_phase1_checkpoint(required=True)

        _meta, state_dicts = load_checkpoint(Path(str(phase1_ckpt)))
        encoder_state = state_dicts.get("encoder")
        if encoder_state is None:
            model_state = state_dicts.get("model", {})
            encoder_state = {
                k.replace("encoder.", "", 1): v
                for k, v in model_state.items()
                if k.startswith("encoder.")
            }
        if not encoder_state:
            raise ValueError(
                f"Checkpoint {phase1_ckpt} does not contain an encoder state"
            )

        # Resolve the Phase-2 handoff independently of the Phase-1 base.  A
        # legacy config may point ``phase1_checkpoint`` directly at a Phase-2
        # checkpoint; retain that supported form, but new configs use the
        # explicit ``stage_a_checkpoint`` key so cache and trainer provenance
        # name the same two sources.
        stage_a_ckpt = self._resolve_stage_a_checkpoint(required=False)
        self._phase2_handoff_checkpoint: str | None = None
        self._phase2_handoff_state_dicts: dict | None = None
        self._phase2_handoff_is_phase1 = False
        if stage_a_ckpt:
            _stage_meta, stage_state_dicts = load_checkpoint(Path(stage_a_ckpt))
            self._phase2_handoff_checkpoint = str(stage_a_ckpt)
            self._phase2_handoff_state_dicts = stage_state_dicts
        elif str(getattr(_meta, "phase", "")) == "phase2_kb":
            self._phase2_handoff_checkpoint = str(phase1_ckpt)
            self._phase2_handoff_state_dicts = state_dicts
            self._phase2_handoff_is_phase1 = True
        elif self._stage() in {"B", "PROMPT_FIT"}:
            raise ValueError(
                f"Phase 2 stage {self._stage()} requires a complete Phase-2 "
                "handoff. Configure training.stage_a_checkpoint (normally "
                "'auto'), or point phase1_checkpoint at a Phase-2 checkpoint."
            )

        encoder_cfg = self.cfg.model.get("encoder", {})
        # Per-training-config override takes precedence so Falcon-family
        # phases can cap anchors at 0.60 without forking model defaults.
        step_model_cfg = self.step_cfg.get("model", {}) or {}
        threshold_cfg = dict(step_model_cfg.get(
            "threshold_controller",
            self.cfg.model.get("threshold_controller", {}),
        ) or {})
        # The encoder's per-level DualThresholdController buffers
        # (anchor_ratios / anchor_thetas / _anchor_velocity) are sized by the
        # anchor grid used at Phase-1 train time. The summarization encoder was
        # trained with the Falcon 6-anchor grid, not the Qwen 7-anchor model
        # default, so derive the grid from the SAVED state to size the
        # reconstructed buffers identically — otherwise load_state_dict raises a
        # size mismatch on l{0,1}.threshold.anchor_* (6 vs 7). Other threshold
        # params (lr / init_theta / kernel_bandwidth / clamp) still come from
        # config so Phase 2 controls its own dual-ascent dynamics.
        saved_anchors = encoder_state.get("l0.threshold.anchor_ratios")
        if saved_anchors is not None:
            threshold_cfg["anchor_ratios"] = saved_anchors.tolist()
        self.encoder = BgKITEncoder.from_pretrained_with_state_dict(
            encoder_cfg.get("backbone_name", "Qwen/Qwen3.5-0.8B-Base"),
            encoder_state,
            hidden_dim=int(encoder_cfg.get("hidden_dim", 1024)),
            active_decoder_family=getattr(self, "_decoder_family", "qwen35"),
            threshold_controller_cfg=threshold_cfg or None,
        ).to(self.device)

        # Load the co-trained decoder(s) from the SAME Phase-1 checkpoint (the
        # summarization handoff saved decoder_qwen / decoder_falcon). Without
        # this the decoders stay at fresh-HF init and the summarization decoder
        # training is discarded. Applies to single-decoder stages too.
        _ckpt_decoder_key = {"qwen35": "decoder_qwen", "falcon_h1": "decoder_falcon"}
        for family, dec in getattr(self, "_decoders_by_family", {}).items():
            key = _ckpt_decoder_key.get(family)
            if key and key in state_dicts:
                missing, unexpected = dec.load_state_dict(
                    state_dicts[key], strict=False,
                )
                logger.info(
                    "phase2_kb_decoder_loaded_from_phase1",
                    family=family, ckpt_key=key,
                    missing=len(missing), unexpected=len(unexpected),
                )
            else:
                logger.info(
                    "phase2_kb_decoder_fresh_hf_init",
                    family=family, reason=f"{key!r} not in checkpoint",
                )

        # Encoder activation checkpointing via the SAME per-backbone-layer
        # mechanism the summarization trainer uses (gated by
        # training.gradient_checkpointing). Each backbone transformer layer is a
        # pure attn+MLP sub-forward, so it is checkpoint-safe — unlike the coarse
        # _checkpointed_level path (disabled via activation_checkpointing.encoder:
        # false), which wraps the whole non-pure LevelCompressor forward (incl.
        # the side-effecting survivorship logic) and fails the use_reentrant=False
        # tensor-match check (666 vs 382 saved tensors).
        from bgkit.training.gradient_utils import maybe_enable_gradient_checkpointing

        # Checkpoint BOTH L0 and L1 backbones (per-layer GC — the summarization
        # trainer's proven mechanism). With LoRA dropped (lora.enabled: false /
        # train_l1_direct), L1's backbone is plain weights again, so its recompute
        # is deterministic — the L1 LoRA was the ONLY thing that broke checkpoint
        # recompute (LoRARouter thread-local adapter state). L0 is the live
        # per-bgkit-call memory hog; L1 the bucketed cross-doc fusion — both now
        # capped, which is what fits round-robin Stage A in budget.
        maybe_enable_gradient_checkpointing(self.encoder.l0.backbone, self.cfg)
        if getattr(self.encoder, "l1", None) is not None:
            maybe_enable_gradient_checkpointing(self.encoder.l1.backbone, self.cfg)

    @staticmethod
    def _prefixed_component_state(
        state_dicts: dict,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        """Extract ``prefix.*`` from a standard ``model`` checkpoint payload."""
        model_state = state_dicts.get("model", {}) or {}
        dotted = f"{prefix}."
        return {
            key[len(dotted):]: value
            for key, value in model_state.items()
            if key.startswith(dotted)
        }

    def _apply_phase2_handoff_state(self) -> None:
        """Overlay all load-bearing Stage-A/prompt-fit model components.

        This runs after model structure is complete and restores the encoder
        (L0, L0→L1 bridge, L1, L1→L1 bridge, projection blocks, thresholds and
        task prompts) plus whichever decoder families are active.  Loading only
        the cached L0 survivors is insufficient: Stage B consumes those
        survivors through the co-trained bridge and L1 distribution.
        """
        state_dicts = getattr(self, "_phase2_handoff_state_dicts", None)
        checkpoint = getattr(self, "_phase2_handoff_checkpoint", None)
        if not state_dicts or not checkpoint:
            return

        encoder_state = state_dicts.get("encoder") or self._prefixed_component_state(
            state_dicts, "encoder",
        )
        if not encoder_state:
            raise ValueError(
                f"Phase-2 handoff checkpoint {checkpoint} has no encoder state"
            )
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        # Missing task prompts are allowed when continuing from plain Stage A;
        # everything else denotes an architecture/config mismatch that would
        # make the claimed handoff incomplete.
        material_missing = [
            key for key in missing if not key.startswith("l0_task_prompts.")
        ]
        material_unexpected = [
            key for key in unexpected if not key.startswith("l0_task_prompts.")
        ]
        if material_missing or material_unexpected:
            raise RuntimeError(
                f"Incomplete Phase-2 encoder handoff from {checkpoint}: "
                f"missing={material_missing[:20]}, "
                f"unexpected={material_unexpected[:20]}"
            )

        loaded_families: list[str] = []
        for family, decoder in getattr(self, "_decoders_by_family", {}).items():
            decoder_state = self._prefixed_component_state(
                state_dicts, f"decoders.{family}",
            )
            if not decoder_state:
                decoder_state = self._prefixed_component_state(state_dicts, "decoder")
            if not decoder_state:
                legacy_key = {
                    "qwen35": "decoder_qwen",
                    "falcon_h1": "decoder_falcon",
                }.get(family)
                decoder_state = state_dicts.get(legacy_key, {}) if legacy_key else {}
            if not decoder_state:
                raise ValueError(
                    f"Phase-2 handoff checkpoint {checkpoint} has no decoder "
                    f"state for family {family!r}"
                )
            dec_missing, dec_unexpected = decoder.load_state_dict(
                decoder_state, strict=False,
            )
            if dec_missing or dec_unexpected:
                raise RuntimeError(
                    f"Incomplete Phase-2 decoder handoff for {family!r} from "
                    f"{checkpoint}: missing={dec_missing[:20]}, "
                    f"unexpected={dec_unexpected[:20]}"
                )
            loaded_families.append(family)

        self.register_checkpoint_source("phase2_handoff", checkpoint)
        logger.info(
            "phase2_kb_complete_handoff_loaded",
            checkpoint=checkpoint,
            encoder_keys=len(encoder_state),
            decoder_families=loaded_families,
        )

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

    def _set_l0_trainability(self) -> None:
        """Live-L0 trainability. Default: train L0 fully (direct). With
        ``l0_freeze_backbone`` (prompt-tuning) freeze the L0 backbone and train
        only the L0 head + threshold. The per-task L0 prompts (if present) are
        always trainable."""
        if getattr(self, "_l0_freeze_backbone", False):
            self.encoder.l0.requires_grad_(False)
            self.encoder.l0.head.requires_grad_(True)
            self.encoder.l0.threshold.requires_grad_(True)
            # Keep the L0->L1 bridge trainable even with a frozen backbone: the
            # task prompt shifts L0's survivor distribution, so the bridge should
            # adapt to it (mirrors "the bridge stays trainable as L0 evolves").
            # It's a single ~1M-param head, so still parameter-efficient. Freeze
            # it too if you want strict prompt-only tuning.
            if getattr(self.encoder.l0, "auto_repro_head", None) is not None:
                self.encoder.l0.auto_repro_head.requires_grad_(True)
            logger.info("phase2_kb_l0_backbone_frozen")
        else:
            self.encoder.l0.requires_grad_(True)
        _tp = getattr(self.encoder, "l0_task_prompts", None)
        if _tp is not None:
            _tp.requires_grad_(True)

    def _install_lora(self) -> None:
        """Configure encoder trainability for Phase 2.

        The legacy/default path installs L1-only LoRA: Stage A trains
        ``encoder.l0`` directly while Stage B consumes cached L0 survivors,
        and both stages train L1 through adapters. Falcon configs set
        ``training.lora.enabled: false`` to avoid encoder adapters entirely;
        in that mode L1 trains directly unless ``train_l1_direct: false`` is
        set explicitly.
        """
        lora_cfg = self.step_cfg.get("lora", {})
        self._encoder_lora_enabled = bool(lora_cfg.get("enabled", True))
        self._direct_l1_trainable = False
        if not self._encoder_lora_enabled:
            self.lora_router = None
            LoRARouter.bind(None)
            self.encoder.requires_grad_(False)
            if getattr(self, "_round_robin", False):
                self.encoder.projection_blocks.requires_grad_(True)
            if self._live_l0:
                self._set_l0_trainability()
            self._direct_l1_trainable = bool(lora_cfg.get("train_l1_direct", True))
            if self._direct_l1_trainable:
                self.encoder.l1.requires_grad_(True)
                # The L1↔L1 bridge (encoder.l1l1_bridge) is a top-level sibling
                # of encoder.l1 — NOT inside it — so encoder.l1.requires_grad_
                # above does NOT unfreeze it. It MUST train: it adapts L0's
                # evolving output into L1's input space and is exercised on
                # every interior tree node via l1_auto_reproduce. Left frozen it
                # would stay pinned at the distilled init and misalign as L0/L1
                # drift. (Trains with L1; tracked in _build_optimizer_groups.)
                if getattr(self.encoder, "l1l1_bridge", None) is not None:
                    self.encoder.l1l1_bridge.requires_grad_(True)
            logger.info(
                "phase2_kb_encoder_lora_disabled",
                live_l0=self._live_l0,
                train_l1_direct=self._direct_l1_trainable,
            )
            return

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
        if getattr(self, "_round_robin", False):
            # Co-train both families' projection blocks (they track the decoders).
            self.encoder.projection_blocks.requires_grad_(True)
        self.lora_router.set_level_trainable("l1", True)
        # Stage A: train L0 weights directly (head + auto_repro_head + backbone).
        # Stage B: keep L0 frozen.
        if self._live_l0:
            self._set_l0_trainability()
        logger.info(
            "phase2_kb_encoder_lora_enabled",
            levels=sorted(self.lora_router.levels),
            live_l0=self._live_l0,
        )

    def _l1_adapter_context(self):
        if self.lora_router is None:
            return contextlib.nullcontext()
        return self.lora_router.active("l1")

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

    @staticmethod
    def _resolve_one_epoch_max_steps(
        n_batches: int, accum_steps: int, epochs: int | None,
    ) -> int:
        """Resolve ``max_steps`` for the ``max_steps: null`` config path.

        The train loop (``base_trainer.train``) runs ``while step < max_steps``,
        consuming ``accum_steps`` microbatches per optimizer step. So ONE real
        pass over the data is ``ceil(n_batches / accum_steps)`` optimizer steps.

        - ``epochs`` set -> ``epochs * ceil(n_batches / accum_steps)`` — TRUE
          epoch count.
        - ``epochs`` is None -> legacy ``n_batches`` (preserved EXACTLY so no
          existing/live config changes silently). NB: with ``accum_steps > 1``
          that legacy value actually runs ``accum_steps`` real epochs, not one —
          set ``training.epochs`` to fix. (With packing, ``n_batches`` is the
          microbatch count, much smaller than the sample count, so if it drops
          below ``warmup_steps`` the cosine LR never leaves warmup.)
        """
        if epochs is not None:
            return int(epochs) * math.ceil(n_batches / max(accum_steps, 1))
        return n_batches

    def _validate_git_artifact_generation(
        self,
        dataset: KBTrajectoryDataset,
    ) -> None:
        """Require tree, mmap, and trajectory artifacts from one build.

        Every component is independently loadable, so checking only article
        coverage can still accept a mixed generation whose IDs happen to
        overlap. Provenance hashes make that state an explicit setup error.
        """
        import json as _json

        from bgkit.data.commit_repro import (
            GIT_REPRO_SCHEMA_VERSION,
            ID_SCHEME_VERSION,
            file_sha256,
        )

        trajectory_manifest = dataset.artifact_manifest
        tree_dir = self._resolve_dir("browse_tree_dir", "browse_trees")
        tree_path = tree_dir / "git_commit_repro.parquet"
        tree_manifest_path = tree_dir / "git_commit_repro.manifest.json"
        mmap_dir = self._resolve_dir("mmap_dir", "mmap/phase2")
        mmap_manifest_path = mmap_dir / "git_commit_repro" / "manifest.json"
        for path in (tree_manifest_path, mmap_manifest_path):
            if not path.exists():
                raise ValueError(
                    f"git_commit_repro artifact manifest missing: {path}. "
                    "Rebuild raw, tree, mmap, trajectories, and Arrow IPC together."
                )
        tree_manifest = _json.loads(tree_manifest_path.read_text())
        mmap_manifest = _json.loads(mmap_manifest_path.read_text())
        if int(tree_manifest.get("schema_version", 0)) != GIT_REPRO_SCHEMA_VERSION:
            raise ValueError("git_commit_repro browse-tree schema mismatch")
        if int(mmap_manifest.get("dataset_schema_version", 0)) != (
            GIT_REPRO_SCHEMA_VERSION
        ):
            raise ValueError("git_commit_repro mmap schema mismatch")
        actual_tree_sha = file_sha256(tree_path)
        manifests = {
            "trajectory": trajectory_manifest,
            "tree": tree_manifest,
            "mmap": mmap_manifest,
        }
        expected = {
            "source_sha256": str(trajectory_manifest.get("source_sha256", "")),
            "tree_sha256": actual_tree_sha,
            "id_salt": str(trajectory_manifest.get("id_salt", "")),
            "id_scheme_version": ID_SCHEME_VERSION,
        }
        for artifact, manifest in manifests.items():
            for key, value in expected.items():
                observed = manifest.get(key)
                if str(observed) != str(value):
                    raise ValueError(
                        "mixed git_commit_repro artifact generations: "
                        f"{artifact}.{key}={observed!r}, expected {value!r}. "
                        "Rebuild and deploy the complete artifact set together."
                    )
        observed_modes = {
            str(mode)
            for mode, count in dict(
                trajectory_manifest.get("drill_mode_counts", {})
            ).items()
            if int(count) > 0
        }
        allowed_raw = self.step_cfg.get(
            "git_repro_allowed_drill_modes", ["full"],
        )
        if isinstance(allowed_raw, str):
            allowed_modes = {allowed_raw}
        else:
            allowed_modes = {str(mode) for mode in allowed_raw}
        if not observed_modes or not observed_modes.issubset(allowed_modes):
            raise ValueError(
                "git_commit_repro trajectory modes are incompatible with this "
                f"run: observed={sorted(observed_modes)}, "
                f"allowed={sorted(allowed_modes)}"
            )
        build_config = trajectory_manifest.get("build_config")
        if not isinstance(build_config, dict):
            raise ValueError(
                "git_commit_repro trajectory manifest has no build_config; rebuild it"
            )
        if allowed_modes == {"full"} and int(build_config.get("max_touching", -1)) != 0:
            raise ValueError(
                "production git_commit_repro requires the complete anchor-to-target "
                "history (trajectory build_config.max_touching must be 0)"
            )
        if self._live_l0 and self._token_store is not None:
            tree_articles = set(self._trees["git_commit_repro"].articles("root"))
            mmap_articles = set(
                self._token_store.document_ids("git_commit_repro")
            )
            if tree_articles != mmap_articles:
                missing = sorted(tree_articles - mmap_articles)[:5]
                extra = sorted(mmap_articles - tree_articles)[:5]
                raise ValueError(
                    "git_commit_repro tree/mmap article sets differ: "
                    f"missing={missing}, extra={extra}"
                )
            validated = getattr(self, "_artifact_coverage_validated", set())
            validated.add("git_commit_repro")
            self._artifact_coverage_validated = validated

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
            if name == "git_commit_repro":
                self._validate_git_artifact_generation(ds)
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
        seed = int(self.cfg.get("seed", 42))
        split_labels = self._dataset_split_labels(full)
        self._artifact_group_sizes = None
        if split_labels:
            if len(split_labels) != total:
                raise ValueError("explicit trajectory split column has wrong length")
            train_indices = [i for i, label in enumerate(split_labels) if label == "train"]
            eval_indices = [i for i, label in enumerate(split_labels) if label == "eval"]
            unknown = sorted(set(split_labels) - {"train", "eval"})
            if unknown or not train_indices or not eval_indices:
                raise ValueError(
                    "explicit trajectory splits must contain non-empty train/eval "
                    f"partitions only; unknown={unknown}"
                )
            self.train_dataset = Subset(full, train_indices)
            self.eval_dataset = Subset(full, eval_indices)
            all_group_keys = self._dataset_group_keys(full)
            if all_group_keys is not None:
                from collections import Counter
                self._artifact_group_sizes = Counter(all_group_keys)
            train_repos = set(self._dataset_repo_ids(self.train_dataset) or [])
            eval_repos = set(self._dataset_repo_ids(self.eval_dataset) or [])
            overlap = train_repos & eval_repos
            if overlap:
                raise ValueError(
                    "repository leakage in explicit trajectory split: "
                    f"{len(overlap)} overlapping repos; sample={sorted(overlap)[:5]}"
                )
            logger.info(
                "phase2_kb_explicit_repo_split",
                train_size=len(train_indices), eval_size=len(eval_indices),
                train_repos=len(train_repos), eval_repos=len(eval_repos),
            )
        else:
            eval_size = min(
                max(1, int(total * 0.05)),
                int(self.step_cfg.get("max_eval_samples", 256)),
            )
            train_size = total - eval_size
            generator = torch.Generator().manual_seed(seed)
            self.train_dataset, self.eval_dataset = random_split(
                full, [train_size, eval_size], generator=generator,
            )
        # Trim outlier samples by live-L0 token budget BEFORE building the
        # dataloader + computing max_steps below, so both reflect the filtered
        # set. Drops the genuine narrativeqa/pubmedqa token-outlier tails
        # (profiled 2026-06-28); memory's large-but-typical samples pass.
        self._filter_train_by_token_budget()
        # Per-phase batch_size override takes precedence over the global
        # default in configs/config.yaml (which is sized for Phase 1).
        batch_size = int(
            self.step_cfg.get("batch_size", self.cfg.get("batch_size", 1))
        )
        num_workers = int(self.cfg.compute.get("num_workers", 0))
        # Token-budget microbatch packing. Live L0 is costed in input tokens;
        # cached L0 is costed in survivor rows, which is the actual Stage-B L1
        # and decoder work.  Keeping separate config keys prevents accidental
        # reuse of a raw-token budget for a fundamentally smaller unit.
        mb_budget_key = (
            "max_microbatch_l0_tokens"
            if self._live_l0
            else "max_microbatch_l0_survivors"
        )
        mb_budget = int(self.step_cfg.get(mb_budget_key, 0) or 0)
        self._train_batch_sampler = None
        if getattr(self, "_per_repo_full_backprop", False):
            # PER-REPO: one batch = all of a repo's window-0 file-samples
            # (grouped by the shared-subtree root node id). The shared tree is
            # then encoded once per batch; internal micro-batching over the
            # repo's file-samples bounds the per-step decoder peak.
            from collections import Counter

            # Bulk turns-only read (reads just trajectory_json, never pages the
            # gold blob) — unwraps the random_split Subset / ConcatDataset to
            # reach KBTrajectoryDataset.group_keys(). The per-sample fallback is
            # a >1h hang at 1.87M git-repro samples.
            group_keys = self._dataset_group_keys(self.train_dataset)
            if group_keys is None:
                group_keys = [
                    self._repo_group_key(self.train_dataset[i])
                    for i in range(len(self.train_dataset))
                ]
            # --- Per-repo SIZE FILTER (drop monster / low-signal repos) -------
            # Two thresholds (either may be null = off), evaluated per UNIQUE
            # repo-window group key:
            #   max_repo_leaf_tokens  — total L0-encoded leaf-diff tokens in the
            #       window-0 subtree (the shared-tree-encode memory driver).
            #   max_repo_file_samples — number of (file, target) reconstruction
            #       samples in the repo-window (per-step work + a low-signal /
            #       generated-repo proxy: e.g. doc/icon repos have huge tails).
            # An over-threshold group is dropped ENTIRELY (its samples never
            # enter a batch), so its shared tree is never encoded.
            n_samples_before = len(group_keys)
            group_sizes = Counter(group_keys)
            n_repos_before = len(group_sizes)
            eval_group_keys = self._dataset_group_keys(self.eval_dataset)
            if eval_group_keys is None:
                eval_group_keys = [
                    self._repo_group_key(self.eval_dataset[i])
                    for i in range(len(self.eval_dataset))
                ]
            eval_group_sizes = Counter(eval_group_keys)
            max_leaf_tokens = getattr(self, "_max_repo_leaf_tokens", None)
            max_file_samples = getattr(self, "_max_repo_file_samples", None)
            max_tree_nodes = getattr(self, "_max_repo_tree_nodes", None)
            _ds_list = list(self.step_cfg.get("datasets", []))
            _ds = _ds_list[0] if _ds_list else None
            artifact_sizes = getattr(self, "_artifact_group_sizes", None)

            def _ineligible(keys: set[str]) -> set[str]:
                dropped: set[str] = set()
                if max_file_samples is not None:
                    dropped |= {
                        key for key in keys
                        if int(
                            artifact_sizes.get(key, 0)
                            if artifact_sizes is not None
                            else group_sizes.get(key, eval_group_sizes.get(key, 0))
                        ) > int(max_file_samples)
                    }
                if max_tree_nodes is not None and _ds is not None:
                    for key in keys - dropped:
                        if self._repo_tree_node_count(_ds, key) > int(max_tree_nodes):
                            dropped.add(key)
                if max_leaf_tokens is not None and _ds is not None:
                    for key in keys - dropped:
                        if self._repo_leaf_token_count(_ds, key) > int(max_leaf_tokens):
                            dropped.add(key)
                return dropped

            dropped_keys = _ineligible(set(group_sizes))
            eval_dropped_keys = _ineligible(set(eval_group_sizes))
            dropped_samples = sum(
                group_sizes[k] for k in dropped_keys
            )
            eval_before = len(self.eval_dataset)
            if eval_dropped_keys:
                keep_eval = [
                    index for index, key in enumerate(eval_group_keys)
                    if key not in eval_dropped_keys
                ]
                self.eval_dataset = Subset(self.eval_dataset, keep_eval)
            # Stash sampler-build inputs so the warmup→inner-loop switch in
            # _pre_step_hook can rebuild the loader (subset-emitting mode)
            # without recomputing the filter.
            self._repo_group_keys = group_keys
            self._repo_dropped_keys = dropped_keys
            self._repo_num_workers = num_workers
            # Initial mode: inner-loop only if enabled AND already past warmup
            # (resume); otherwise the one-step (warmup) grouping. _inner_loop_active
            # tracks the live mode so _pre_step_hook can flip it at the boundary.
            self._inner_loop_active = self._should_use_inner_loop()
            n_train_batches = self._build_repo_grouped_dataloader(
                inner_loop=self._inner_loop_active,
            )
            logger.info(
                "phase2_kb_per_repo_size_filter",
                max_repo_leaf_tokens=max_leaf_tokens,
                max_repo_file_samples=max_file_samples,
                max_repo_tree_nodes=max_tree_nodes,
                n_repos_before=n_repos_before,
                n_repos_after=len(group_sizes) - len(dropped_keys),
                n_repos_dropped=len(dropped_keys),
                n_samples_before=n_samples_before,
                n_samples_after=n_samples_before - dropped_samples,
                n_samples_dropped=dropped_samples,
                eval_repos_before=len(eval_group_sizes),
                eval_repos_after=len(eval_group_sizes) - len(eval_dropped_keys),
                eval_samples_before=eval_before,
                eval_samples_after=len(self.eval_dataset),
            )
            logger.info(
                "phase2_kb_per_repo_grouping",
                n_samples=n_samples_before - dropped_samples,
                n_repos=len(group_sizes) - len(dropped_keys),
                n_train_batches=n_train_batches,
                inner_loop_active=self._inner_loop_active,
                inner_subset_size=self._per_repo_inner_subset_size,
                max_inner_steps=self._per_repo_max_inner_steps,
            )
        elif mb_budget > 0:
            loads = (
                getattr(self, "_train_token_loads", None)
                if self._live_l0 else None
            )
            if loads is None or len(loads) != len(self.train_dataset):
                loads = [
                    (
                        self._sample_l0_token_load(self.train_dataset[i])
                        if self._live_l0
                        else self._sample_l0_survivor_load(self.train_dataset[i])
                    )
                    for i in range(len(self.train_dataset))
                ]
            from bgkit.data.samplers import KBTokenBudgetBatchSampler
            # Cap samples/microbatch to bound the decoder backward — a microbatch
            # of many tiny samples runs that many sequential decoder backwards,
            # spiking the peak (2026-06-28: 90GB packed vs 73GB single). 0 = token
            # budget only.
            mb_max_samples = int(self.step_cfg.get("max_microbatch_samples", 0) or 0)
            self._train_batch_sampler = KBTokenBudgetBatchSampler(
                loads, mb_budget, max_samples=(mb_max_samples or None), shuffle=True,
                seed=int(self.cfg.get("seed", 42)),
            )
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_sampler=self._train_batch_sampler,
                collate_fn=_collate_kb,
                num_workers=num_workers,
                pin_memory=False,
            )
            n_train_batches = len(self._train_batch_sampler)
            logger.info(
                "phase2_kb_token_budget_packing",
                unit=("input_tokens" if self._live_l0 else "cached_survivors"),
                budget=mb_budget,
                n_samples=len(self.train_dataset),
                n_microbatches=n_train_batches,
                avg_samples_per_microbatch=round(
                    len(self.train_dataset) / max(n_train_batches, 1), 2,
                ),
            )
        else:
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=_collate_kb,
                num_workers=num_workers,
                pin_memory=False,
            )
            n_train_batches = math.ceil(
                len(self.train_dataset) / max(batch_size, 1)
            )
        # Explicit repository splits are filtered for eligibility before this
        # reporting cap.  This avoids the old mixture where training dropped an
        # oversized repo only after one of its rows had entered evaluation.
        max_eval_samples = int(self.step_cfg.get("max_eval_samples", 256))
        if max_eval_samples > 0 and len(self.eval_dataset) > max_eval_samples:
            generator = torch.Generator().manual_seed(
                int(self.cfg.get("seed", 42)) + 1,
            )
            keep = torch.randperm(
                len(self.eval_dataset), generator=generator,
            )[:max_eval_samples].sort().values.tolist()
            self.eval_dataset = Subset(self.eval_dataset, keep)
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
            accum_steps = int(
                self.step_cfg.get("gradient_accumulation_steps", 1) or 1
            )
            epochs = self.step_cfg.get("epochs", None)
            one_epoch = self._resolve_one_epoch_max_steps(
                n_train_batches, accum_steps, epochs,
            )
            with open_dict(self.step_cfg):
                self.step_cfg.max_steps = one_epoch
            logger.info(
                "phase2_kb_max_steps_one_epoch",
                stage=self._stage(),
                train_size=len(self.train_dataset),
                batch_size=(None if self._train_batch_sampler else batch_size),
                packing=bool(self._train_batch_sampler),
                n_batches=n_train_batches,
                accum_steps=accum_steps,
                epochs_cfg=(None if epochs is None else int(epochs)),
                real_epochs=round(
                    one_epoch * accum_steps / max(n_train_batches, 1), 2,
                ),
                max_steps=one_epoch,
            )

        # --- Head-tanh temperature calibration (L0 + L1) ---
        # Inherited from Phase 1 Step 6 checkpoint but re-probed against
        # Phase 2 text corpus (Wikipedia / KILT / PubMed-MeSH / etc.)
        # so T reflects the actual head-output std Stage A / Stage B
        # will see. Skipped when no KB text can be extracted.
        self._calibrate_head_tanh_temperatures()

    # ------------------------------------------------------------------
    # Inner-loop (Model B) sampler/loader wiring
    # ------------------------------------------------------------------

    def _should_use_inner_loop(self) -> bool:
        """Inner-loop is live when enabled AND the global step is past the
        warmup window. (At setup/resume global_step reflects the resume point,
        so a resume past warmup builds the subset sampler directly.)"""
        return bool(
            getattr(self, "_per_repo_full_backprop", False)
            and getattr(self, "_per_repo_inner_loop", False)
            and int(getattr(self, "global_step", 0))
            >= int(getattr(self, "_inner_loop_warmup_steps", 0))
        )

    def _build_repo_grouped_dataloader(self, *, inner_loop: bool) -> int:
        """(Re)build ``train_dataloader`` with a RepoGroupedBatchSampler in the
        chosen mode from the stashed group-keys + dropped-keys. ``inner_loop``
        emits each repo's files as K consecutive subset-batches (S, K-cap);
        else one batch per repo (the one-step / warmup path). Returns the batch
        count. Used at setup AND at the warmup→inner-loop switch."""
        from bgkit.data.samplers import RepoGroupedBatchSampler

        self._train_batch_sampler = RepoGroupedBatchSampler(
            self._repo_group_keys or [],
            shuffle=True,
            seed=int(self.cfg.get("seed", 42)),
            drop_keys=self._repo_dropped_keys or set(),
            inner_loop=inner_loop,
            inner_subset_size=int(self._per_repo_inner_subset_size),
            max_inner_steps=int(self._per_repo_max_inner_steps),
        )
        # Preserve the sampler's epoch so a mid-run rebuild keeps the shuffle
        # schedule aligned (the base loop drives set_epoch on rollover).
        self._train_batch_sampler.set_epoch(int(getattr(self, "epoch", 0)))
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self._train_batch_sampler,
            collate_fn=_collate_kb,
            num_workers=int(getattr(self, "_repo_num_workers", 0)),
            pin_memory=False,
        )
        return len(self._train_batch_sampler)

    def _pre_step_hook(self) -> None:
        """Warmup→inner-loop switch (Model B). For the first
        ``inner_loop_warmup_steps`` global steps the one-step detach-
        reaccumulate path runs (one optimizer step per repo, clean gradient).
        At the boundary, rebuild the sampler into subset-emitting mode and flip
        ``_inner_loop_active`` — the base loop re-creates the dataloader iter
        when ``_dataloader_invalidated`` is set. The switch lands on a repo
        boundary (warmup = one batch per repo), so there is no half-repo state.
        """
        if (
            getattr(self, "_per_repo_full_backprop", False)
            and getattr(self, "_per_repo_inner_loop", False)
            and not getattr(self, "_inner_loop_active", False)
            and self._repo_group_keys is not None
            and int(getattr(self, "global_step", 0))
            >= int(getattr(self, "_inner_loop_warmup_steps", 0))
        ):
            # Free any retained one-step state defensively, then rebuild.
            self._free_inner_loop_tree()
            self._inner_loop_repo_key = None
            self._inner_loop_active = True
            n_batches = self._build_repo_grouped_dataloader(inner_loop=True)
            self._dataloader_invalidated = True
            logger.info(
                "phase2_kb_inner_loop_switch",
                global_step=int(self.global_step),
                warmup_steps=int(self._inner_loop_warmup_steps),
                n_subset_batches=n_batches,
                inner_subset_size=int(self._per_repo_inner_subset_size),
                max_inner_steps=int(self._per_repo_max_inner_steps),
            )

    def _calibrate_head_tanh_temperatures(
        self, n_probe_batches: int = 4, levels: tuple[str, ...] = ("l0", "l1"),
    ) -> None:
        """Probe L0 + L1 head output std against Phase 2 KB content.

        Phase 2 batches are lists of ``KBSample`` objects, not dicts of
        tensors, so the shared calibration helper can't read
        ``content_token_ids`` directly. We plug in a KB-aware extractor:
        the first retrieval turn's ARTICLE tokens from the live token store
        (a 2K-token prefix — what the L0 head actually scores in the flat
        wide-net path), falling back to the sample's ``question`` text when
        no article resolves (cached-L0 stages, no retrieval turn).
        """
        from bgkit.training.survivorship_helpers import (
            calibrate_head_tanh_temperature,
        )

        tokenizer = getattr(self, "encoder_tokenizer", None)
        if tokenizer is None:
            logger.info("phase2_kb_tanh_calibration_skipped", reason="no_tokenizer")
            return

        store = getattr(self, "_token_store", None)

        def _article_probe(sample):
            if store is None:
                return None
            for turn in getattr(sample, "trajectory", None) or []:
                if getattr(turn, "kind", None) != "bgkit":
                    continue
                ids = list(turn.args.get("ids", []))
                try:
                    arts = self._resolve_article_ids(sample.dataset_name, ids)
                except Exception:
                    arts = []
                if arts:
                    toks = store.get(sample.dataset_name, arts[0])
                    if toks is not None and int(toks.numel()) > 0:
                        return toks[:2048].to(torch.long)
            return None

        def _batch_to_content(batch):
            # batch is a list of KBSample objects post-_collate_kb. Probe the
            # first sample's article content; fall back to its question text.
            if not batch:
                return None
            sample = batch[0]
            token_ids = _article_probe(sample)
            if token_ids is None:
                text = getattr(sample, "question", None) or ""
                if not text:
                    return None
                ids = tokenizer.encode(text, add_special_tokens=False)
                if not ids:
                    return None
                token_ids = torch.tensor(ids[:512], dtype=torch.long)
            # Packed varlen format: flat (N,) tokens + cu_seqlens (B+1,) int32.
            # Single-sample probe → cu_seqlens = [0, L].
            cu_seqlens = torch.tensor([0, int(token_ids.numel())], dtype=torch.int32)
            return token_ids, cu_seqlens

        for level in levels:
            calibrated_t = calibrate_head_tanh_temperature(
                self.encoder,
                self.train_dataloader,
                self.device,
                level=level,
                n_probe_batches=n_probe_batches,
                batch_to_content=_batch_to_content,
            )
            if calibrated_t is not None:
                logger.info(
                    "head_tanh_temperature_calibrated",
                    enc_level=level,  # ``level`` is structlog's reserved key
                    T=calibrated_t,
                    phase="phase2_kb",
                )
            else:
                logger.info(
                    "phase2_kb_tanh_calibration_skipped",
                    enc_level=level,
                    reason="no_extractable_content",
                )

    def _pre_train_loop(self) -> None:
        """After setup AND checkpoint restore: under ``exact_topk`` the head
        temperature ``T`` is not an operator parameter but the score
        normalizer every loss is conditioned on (``sigmoid(base_raw/T - theta)``), so
        it must track the CURRENT head. The setup-time probe runs on the
        pre-restore weights and the restored checkpoint then overwrites T
        with its saved value (v5b: T stayed 2.24 while the head's raw scores
        drifted far below the tanh floor → every loss frozen). Re-estimate
        it from the loaded head for each exact_topk level on resume.
        """
        super()._pre_train_loop()
        topk_levels = tuple(
            lvl for lvl in ("l0", "l1")
            if getattr(self, f"_selection_mode_{lvl}", "threshold") == "exact_topk"
        )
        if topk_levels and int(getattr(self, "global_step", 0) or 0) > 0:
            logger.info(
                "phase2_kb_tanh_temperature_recalibrated_on_resume",
                enc_levels=list(topk_levels),
                step=int(self.global_step),
            )
            self._calibrate_head_tanh_temperatures(levels=topk_levels)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def _build_optimizer_groups(self) -> list[dict]:
        groups: list[dict] = []
        # Round-robin optimizes BOTH decoders + BOTH families' projection
        # blocks; single-decoder stages optimize just self.decoder.
        dec_params = [
            p
            for dec in self._all_decoders()
            for p in dec.parameters()
            if p.requires_grad
        ]
        if dec_params:
            groups.append({
                "params": dec_params,
                "lr": float(self.step_cfg.get(
                    "decoder_lr", self.step_cfg.get("lr", 1e-4)
                )),
            })
        if getattr(self, "_round_robin", False):
            proj_params = [
                p
                for p in self.encoder.projection_blocks.parameters()
                if p.requires_grad
            ]
            if proj_params:
                groups.append({
                    "params": proj_params,
                    "lr": float(self.step_cfg.get(
                        "projection_lr", self.step_cfg.get("lr", 1e-4)
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
            # Per-task learnable L0 prompts (gated). They live on
            # encoder.l0_task_prompts (a sibling of encoder.l0), so they are NOT
            # in l0.parameters() above — add them explicitly at the L0 LR.
            _tp = getattr(self.encoder, "l0_task_prompts", None)
            if _tp is not None:
                _tp_params = [p for p in _tp.parameters() if p.requires_grad]
                if _tp_params:
                    groups.append({
                        "params": _tp_params,
                        "lr": float(
                            self.step_cfg.get("l0_lr", self.step_cfg.get("lr", 1e-4))
                        ),
                    })
        if self.lora_router is not None:
            for level in sorted(self.lora_router.levels):
                lvl_params = [
                    p
                    for p in self.lora_router.adapter_parameters(level)
                    if p.requires_grad
                ]
                if not lvl_params:
                    continue
                lr_key = f"{level}_lr"
                groups.append({
                    "params": lvl_params,
                    "lr": float(
                        self.step_cfg.get(lr_key, self.step_cfg.get("lr", 1e-4))
                    ),
                })
        else:
            l1_params = [p for p in self.encoder.l1.parameters() if p.requires_grad]
            if l1_params:
                lr_key = "l1_lr"
                groups.append({
                    "params": l1_params,
                    "lr": float(
                        self.step_cfg.get(lr_key, self.step_cfg.get("lr", 1e-4))
                    ),
                })
        # L1↔L1 bridge (encoder.l1l1_bridge): a top-level sibling of encoder.l1,
        # so it is NOT in encoder.l1.parameters() and NOT covered by the L1 LoRA
        # adapter set above. When trainable (direct-L1 / recursive runs) it must
        # be optimized — same LR as the L1 backbone group. Guarded by
        # requires_grad so configs that keep it frozen add no empty group.
        bridge = getattr(self.encoder, "l1l1_bridge", None)
        if bridge is not None:
            bridge_params = [p for p in bridge.parameters() if p.requires_grad]
            if bridge_params:
                groups.append({
                    "params": bridge_params,
                    "lr": float(
                        self.step_cfg.get("l1_lr", self.step_cfg.get("lr", 1e-4))
                    ),
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

    # Canonical positional argument order + defaults for a single
    # LevelCompressor forward (encoder.l0 or encoder.l1). When activation
    # checkpointing is on we convert kwargs → positional and a bare
    # ``kwargs.get(name)`` returns None for unset args, which overrides
    # the function's intrinsic default (e.g. ``min_per_sample: int = 0``
    # becomes None and crashes inside the head hook). Keep this in sync
    # with LevelCompressor.forward().
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
        "selection_mode",
        "must_keep_mask",
        "utility_grad_active",
        "utility_grad_capture",
    )
    _LEVEL_ARG_DEFAULTS: ClassVar[dict[str, object]] = {
        "prompt_embeddings": None,
        "prompt_cu_seqlens": None,
        "prompt_position_ids": None,
        "pinned_positions": None,
        "target_ratio": None,
        "min_per_sample": 0,
        # Captured so the L0-forward checkpoint replays it. The DECISION is made
        # by the caller (_l0_leaf_forward hardcodes "exact_topk"), NOT by reading
        # transient _recursive_l0_override inside the encode — so the L1-node
        # checkpoint's recompute (which re-runs _l0_leaf_forward) re-derives the
        # SAME mode. Default "threshold" preserves flat _l0_for_articles.
        "selection_mode": "threshold",
        "must_keep_mask": None,
        "utility_grad_active": False,
        "utility_grad_capture": None,
    }

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

        positional = tuple(
            kwargs.get(name, self._LEVEL_ARG_DEFAULTS.get(name))
            for name in self._LEVEL_ARG_ORDER
        )
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
        # While the shared per-repo tree is being encoded, the tree's L0
        # leaves use the recursive L0 ramp, not the per-dataset rate.
        override = getattr(self, "_recursive_l0_override", None)
        if override is not None:
            return float(override)
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

    @staticmethod
    def _interp_ratio_ramp(cfg, step: int, default: float = 0.15) -> float:
        """Interpolate a retention ratio from a scalar OR ``{start, end,
        ramp_steps}`` ramp, by ``step``.  Mirrors :meth:`_l0_retention_for`'s
        curriculum math but is stateless so it can drive the recursive L0/L1
        ramps. After ``ramp_steps`` the ratio stays at ``end``.
        """
        if cfg is None:
            return float(default)
        if isinstance(cfg, (int, float)):
            return float(cfg)
        start = float(cfg.get("start", default))
        end = float(cfg.get("end", start))
        ramp = max(1, int(cfg.get("ramp_steps", 1)))
        t = min(1.0, max(0, int(step)) / ramp)
        return start + (end - start) * t

    def _interp_recursive_ratio(self, cfg, default: float = 0.15) -> float:
        return self._interp_ratio_ramp(cfg, int(getattr(self, "global_step", 0)), default)

    def _recursive_l0_retention_now(self) -> float:
        """Current recursive-L0 retention (shared-tree leaves). Falls back to
        the per-dataset default when no recursive L0 ramp is configured."""
        cfg = getattr(self, "_recursive_l0_retention_cfg", None)
        if cfg is None:
            return float(self.step_cfg.get("default_l0_retention", 0.10))
        return self._interp_recursive_ratio(cfg, default=0.10)

    def _recursive_l1_retention_now(self) -> float:
        """Current recursive-L1 retention, ramp-aware.  Prefers the configured
        ramp; falls back to the legacy scalar attribute (set directly by unit
        tests) and finally to the sampled L1 ratio."""
        cfg = getattr(self, "_recursive_l1_retention_cfg", None)
        if cfg is not None:
            return self._interp_recursive_ratio(cfg, default=0.15)
        scalar = getattr(self, "_recursive_l1_retention", None)
        if scalar is not None:
            return float(scalar)
        return self._sample_l1_retention()

    def _drill_node_retention_now(self) -> float:
        """Retention for the per-sample QUERY-CONDITIONED drill-node forwards
        (head + interior + distractor node turns). Falls back to the recursive
        L1 ramp when ``recursive_l1_tree.drill_node_retention`` is unset, so
        flag-off / knob-off behavior is exactly legacy."""
        cfg = getattr(self, "_drill_node_retention_cfg", None)
        if cfg is None:
            return self._recursive_l1_retention_now()
        return self._interp_recursive_ratio(cfg, default=0.15)

    def _drill_leaf_l0_retention_now(self) -> float | None:
        """Retrieve-leaf drill L0 retention override
        (``recursive_l1_tree.drill_leaf_retention.l0``). ``None`` = knob unset
        → the per-dataset ``l0_retention`` map applies (legacy)."""
        cfg = getattr(self, "_drill_leaf_l0_retention_cfg", None)
        if cfg is None:
            return None
        return self._interp_recursive_ratio(cfg, default=0.10)

    def _drill_leaf_l1_retention_override(self) -> float | None:
        """Retrieve-leaf drill L1 retention override
        (``recursive_l1_tree.drill_leaf_retention.l1``); ``None`` when unset."""
        cfg = getattr(self, "_drill_leaf_l1_retention_cfg", None)
        if cfg is None:
            return None
        return self._interp_recursive_ratio(cfg, default=0.15)

    def _drill_leaf_l1_retention_now(self) -> float:
        """Effective retrieve-leaf drill L1 retention for the per-repo decode
        drivers: the ``drill_leaf_retention.l1`` knob when set, else the
        recursive L1 ramp (the legacy drill↔tree coupling)."""
        override = self._drill_leaf_l1_retention_override()
        if override is not None:
            return override
        return self._recursive_l1_retention_now()

    def _anchor_free_if_topk(self, level: str, cfg):
        """Anchor sampling (``anchor_sampling_prob``: uniform picks from the
        FULL θ-anchor grid, up to 0.95) exists to calibrate the threshold
        curve θ(r) far from the operating point. Under ``exact_topk`` θ is
        never consulted, so anchor samples buy nothing and cost a lot: a 0.95
        draw on a 40K-token window splices ~37K reps (memory/time tail seen
        in v5b). Couple the two at the source: an exact_topk level keeps the
        window/jitter band (ratio robustness near the operating point) and
        has anchor sampling disabled. Returns ``cfg`` unchanged otherwise."""
        mode = getattr(self, f"_selection_mode_{level}", "threshold")
        prob = float(getattr(cfg, "anchor_sampling_prob", 0.0) or 0.0)
        if mode != "exact_topk" or prob <= 0.0:
            return cfg
        import dataclasses

        logger.info(
            "phase2_kb_ratio_sampler_anchor_disabled",
            enc_level=level,  # ``level`` is structlog's reserved log-level key
            selection_mode=mode,
            anchor_sampling_prob_was=prob,
        )
        return dataclasses.replace(cfg, anchor_sampling_prob=0.0)

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
        query_emb: torch.Tensor | None = None,
        ratio: float | None = None,
        selection_mode: str = "threshold",
        gold_spans: dict[str, tuple[int, int]] | None = None,
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
        # SINGLE-FORWARD BOUND: cap the per-encode token buffer so one huge
        # leaf (the window-0 initial-import commit, up to the ~75k window
        # budget) can't blow a single L0 varlen forward / the retained tree.
        # Keep whole files up to the cap; the file straddling the cap is
        # included truncated (a bounded prefix — better than OOM). No-op (0).
        cap = int(getattr(self, "_max_l0_encode_tokens", 0) or 0)
        if cap > 0 and int(tokens_flat.shape[0]) > cap:
            full = int(tokens_flat.shape[0])
            tokens_flat = tokens_flat[:cap]
            cu = cu_seqlens[cu_seqlens <= cap]
            if int(cu[-1].item()) != cap:
                cu = torch.cat([
                    cu, torch.tensor([cap], dtype=torch.int32, device=self.device),
                ])
            cu_seqlens = cu
            logger.warning(
                "phase2_kb_l0_encode_truncated",
                full_tokens=full, capped_to=cap, n_segments=int(cu.shape[0]) - 1,
            )
        position_ids = position_ids_from_cu(cu_seqlens, int(tokens_flat.shape[0]))

        # Oracle-span ablation (eval diagnostic): force the gold answer span's
        # token positions to win the L0 exact_topk selection at the same
        # budget. Built HERE (not in _l0_for_articles) so the offsets are
        # against the FINAL (possibly cap-truncated) flat buffer.
        must_keep = None
        if gold_spans and self._ablation_mode == self.ABLATION_ORACLE_SPAN:
            must_keep = torch.zeros(
                int(tokens_flat.shape[0]), dtype=torch.bool, device=self.device,
            )
            n_seg = int(cu_seqlens.shape[0]) - 1
            for i, aid in enumerate(article_ids[:n_seg]):
                sp = gold_spans.get(aid)
                if sp is None:
                    continue
                a0 = int(cu_seqlens[i].item())
                a1 = int(cu_seqlens[i + 1].item())
                s = min(max(a0 + int(sp[0]), a0), a1)
                e = min(max(a0 + int(sp[1]), a0), a1)
                if e > s:
                    must_keep[s:e] = True
            if not bool(must_keep.any()):
                must_keep = None
        if self._ablation_mode == self.ABLATION_ORACLE_SPAN:
            # Liveness line (2026-08-24): the first oracle run came back
            # IDENTICAL to the headline — never interpret that without
            # positive evidence the mask was actually built and non-empty.
            logger.info(
                "oracle_span_liveness",
                dataset=dataset,
                had_spans=bool(gold_spans),
                n_forced=int(must_keep.sum().item()) if must_keep is not None else 0,
                n_content=int(tokens_flat.shape[0]),
            )

        embed_tokens = self.encoder.l0.backbone.get_input_embeddings()
        input_embeddings = embed_tokens(tokens_flat)  # (N_content, D)

        # ``ratio`` may be supplied (FIX 2b: sampled ONCE outside a checkpoint
        # so the recompute is deterministic — re-sampling here would diverge).
        # None → sample as before (the drill / non-checkpointed paths).
        ratio = float(ratio) if ratio is not None else self._sample_l0_retention_for(dataset)
        from bgkit.training.survivorship_helpers import LevelLossCfg
        util_active = getattr(
            self, "_surv_l0", LevelLossCfg(),
        ).utility_grad_loss_weight > 0.0
        grad_capture: dict | None = {} if util_active else None

        # L0 compression prompt — replicated once per article (mirrors the
        # summarization trainer's per-section prompt). Two mutually-exclusive
        # regimes:
        #   - Per-task learnable prompt (l0_prompt_tokens > 0): the FROZEN-L0
        #     conditioning, for when L0 is cached and a per-query prompt can't be
        #     baked into the cache (Stage B / frozen backbone).
        #   - Actual per-sample query (default for LIVE L0): the real question is
        #     fed to L0 so its within-document compression is query-aware. Always
        #     applied in the live path so it is never silently dropped.
        l0_prompt_emb = None
        l0_prompt_cu = None
        l0_prompt_pos = None
        _tp = getattr(self.encoder, "l0_task_prompts", None)
        if (
            getattr(self, "_l0_prompt_tokens", 0) > 0
            and _tp is not None
            and dataset in _tp
        ):
            _prompt_src = _tp[dataset]  # (P, D) learnable per-task prompt
        elif query_emb is not None and int(query_emb.shape[0]) > 0:
            _prompt_src = query_emb  # (L_query, D) the actual question
        else:
            _prompt_src = None
        if _prompt_src is not None:
            prompt_length = int(_prompt_src.shape[0])
            _n_art = len(lengths)
            l0_prompt_emb = (
                _prompt_src.unsqueeze(0).expand(_n_art, -1, -1)
                .reshape(_n_art * prompt_length, -1).to(input_embeddings.dtype)
            )
            l0_prompt_cu = torch.arange(
                0, (_n_art + 1) * prompt_length, prompt_length,
                dtype=torch.int32, device=self.device,
            )
            l0_prompt_pos = position_ids_from_cu(
                l0_prompt_cu, int(l0_prompt_emb.shape[0]),
            )

        # selection_mode is a CALLER decision threaded in (NOT read from transient
        # _recursive_l0_override here) so the L1-node checkpoint's recompute —
        # which re-runs _l0_leaf_forward → this method — re-derives the SAME mode.
        # _l0_leaf_forward passes "exact_topk" (tree leaf: ceil>=1, never 0 → fixes
        # the collapse); flat _l0_for_articles keeps "threshold".
        out = self._checkpointed_level(
            "l0",
            content_embeddings=input_embeddings,
            content_cu_seqlens=cu_seqlens,
            content_position_ids=position_ids,
            prompt_embeddings=l0_prompt_emb,
            prompt_cu_seqlens=l0_prompt_cu,
            prompt_position_ids=l0_prompt_pos,
            target_ratio=ratio,
            selection_mode=selection_mode,
            must_keep_mask=must_keep,
            utility_grad_active=util_active,
            utility_grad_capture=grad_capture,
        )
        if grad_capture is not None:
            out._l0_grad_capture = grad_capture  # type: ignore[attr-defined]
        return out, cu_seqlens, ratio

    def _l0_for_articles(
        self, dataset: str, article_ids: list[str],
        query_emb: torch.Tensor | None = None,
        selection_mode: str = "threshold",
        ratio: float | None = None,
        gold_spans: dict[str, tuple[int, int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return packed L0 survivors for each article.

        ``ratio`` pins the LIVE encode's L0 retention (the retrieve-leaf
        drill's ``drill_leaf_retention.l0`` override); ``None`` = the
        per-dataset ``l0_retention`` map (legacy). The cached path ignores it
        (cache rows are pre-baked at precompute retention).

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
            out, content_cu, ratio = self._live_l0_encode(
                dataset, article_ids, query_emb=query_emb,
                ratio=ratio, selection_mode=selection_mode,
                gold_spans=gold_spans,
            )
            survivors = out.survivor_embeddings  # (N_survivors, D)
            cu_seqlens = out.survivor_cu_seqlens  # (B+1,)
            # v5 span-level relevance: flat (N_content,) bool marking the gold
            # answer span's token positions (per article, from gold_spans).
            span_mask = None
            if gold_spans:
                span_mask = torch.zeros(
                    int(content_cu[-1].item()), dtype=torch.bool, device=content_cu.device,
                )
                for i, aid in enumerate(article_ids):
                    sp = gold_spans.get(aid)
                    if sp is None:
                        continue
                    a0 = int(content_cu[i].item())
                    a1 = int(content_cu[i + 1].item())
                    s = min(max(a0 + int(sp[0]), a0), a1)
                    e = min(max(a0 + int(sp[1]), a0), a1)
                    if e > s:
                        span_mask[s:e] = True
            # Stash for _prepare_l1_turn (span -> L1 position mapping).
            self._last_l0_survivor_mask = getattr(out, "survivor_mask", None)
            self._last_l0_content_cu = content_cu
            self._last_l0_span_mask = span_mask
            if self.encoder.training:
                if getattr(self, "_survivorship_aux", True):
                    if hasattr(self, "_pending_l0_outputs"):
                        self._pending_l0_outputs.append({
                            "dataset": dataset,
                            "enc_out": out,
                            "ratio": ratio,
                            # Packed pre-compression content cu_seqlens so aux
                            # losses can do segment-aware reductions.
                            "cu_seqlens": content_cu,
                            "span_mask": span_mask,
                        })
                else:
                    # Aux OFF: accumulate ONLY the θ control scalars (no graph
                    # retention) so the head freezes but θ still tracks the ramp.
                    self._accumulate_theta_state("l0", out, ratio)
            return survivors, cu_seqlens
        if self._ablation_mode == self.ABLATION_ORACLE_SPAN:
            raise RuntimeError(
                "ablation 'oracle_span' requires live_l0=True — the cached L0 "
                "path cannot force span survival (rows are pre-baked)."
            )
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

    def _sample_l0_token_load(self, sample) -> int:
        """Total live-L0 token load for one sample = sum over its bgkit turns
        of the resolved articles' token lengths. Used by the token-budget
        outlier filter. Cheap: offsets-CSR length lookups, no token loads."""
        total = 0
        for turn in sample.trajectory:
            if turn.kind != "bgkit":
                continue
            ids = list(turn.args.get("ids", []))
            if not ids:
                continue
            try:
                doc_ids = self._resolve_article_ids(sample.dataset_name, ids)
            except Exception:
                continue
            for d in doc_ids:
                try:
                    total += self._token_store.length(sample.dataset_name, d)
                except KeyError:
                    continue
        return total

    def _sample_l0_survivor_load(self, sample) -> int:
        """Cached-L0 cost for one sample, measured in survivor rows.

        This reads only shard offsets; survivor embeddings stay memory-mapped
        and are not materialized during sampler construction.
        """
        if self._l0_cache is None:
            return 0
        total = 0
        for turn in sample.trajectory:
            if turn.kind != "bgkit":
                continue
            ids = list(turn.args.get("ids", []))
            if not ids:
                continue
            try:
                article_ids = self._resolve_article_ids(sample.dataset_name, ids)
                doc_ids = self._article_ids_to_document_ids(
                    sample.dataset_name, article_ids,
                )
            except Exception:
                continue
            for doc_id in doc_ids:
                try:
                    total += self._l0_cache.length(sample.dataset_name, doc_id)
                except KeyError:
                    continue
        return total

    def _filter_train_by_token_budget(self) -> None:
        """Apply the same live-L0 eligibility rule to train and evaluation.

        Drop samples whose live-L0 token load exceeds
        ``max_sample_l0_tokens`` — the genuine outliers (narrativeqa / pubmedqa
        tails) that would OOM the live-L0 forward. Memory's large-but-typical
        samples (median ~42K tokens) pass. Applying a different eligibility
        rule to evaluation creates a distribution that cannot be interpreted.
        No-op when the budget is 0/unset or L0 is cached (Stage B)."""
        budget = int(self.step_cfg.get("max_sample_l0_tokens", 0) or 0)
        if budget <= 0 or not self._live_l0 or self._token_store is None:
            return
        from torch.utils.data import Subset
        def _filter(dataset):
            valid: list[int] = []
            loads: list[int] = []
            dropped: dict[str, int] = {}
            for idx in range(len(dataset)):
                sample = dataset[idx]
                load = self._sample_l0_token_load(sample)
                if load <= budget:
                    valid.append(idx)
                    loads.append(load)
                else:
                    key = getattr(sample, "dataset_name", "?")
                    dropped[key] = dropped.get(key, 0) + 1
            return Subset(dataset, valid), loads, dropped

        train_before = len(self.train_dataset)
        eval_before = len(self.eval_dataset)
        self.train_dataset, valid_loads, train_dropped = _filter(self.train_dataset)
        self.eval_dataset, _eval_loads, eval_dropped = _filter(self.eval_dataset)
        # Cache per-sample loads (aligned with the post-filter Subset) so the
        # token-budget microbatch sampler can reuse them without a second walk.
        self._train_token_loads = valid_loads
        logger.info(
            "phase2_kb_token_budget_filter", budget=budget,
            train_kept=len(self.train_dataset),
            train_dropped=train_before - len(self.train_dataset),
            eval_kept=len(self.eval_dataset),
            eval_dropped=eval_before - len(self.eval_dataset),
            train_dropped_by_dataset=train_dropped,
            eval_dropped_by_dataset=eval_dropped,
        )

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
        # Retrieval ids that resolve to NO article via the browse tree (not a
        # node, no leaf tag). `_resolve_article_ids` drops them silently → the
        # turn becomes None → a ZERO splice — the sample trains on nothing.
        # 2026-08-22: the flat tree builder's `--leaf-cap` silently truncated
        # swerecall to 3,200/8,088 spans; 60% of its trajectories were zero
        # splices while this scan reported `coverage_scan_ok` (it only checked
        # the token store). Unresolvable ids are now fatal here.
        unresolved_by_dataset: dict[str, list[str]] = {}
        checked_articles: dict[str, set[str]] = {}

        def _check_sample(sample: KBSample) -> None:
            if sample.dataset_name in getattr(
                self, "_artifact_coverage_validated", set(),
            ):
                return
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
                        else:
                            unresolved_by_dataset.setdefault(
                                sample.dataset_name, [],
                            ).append(raw)
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

        # The full O(n) walk parses every trajectory (incl. gold blobs) from
        # parquet single-threaded — pathological at 1.87M+ samples
        # (git_commit_repro / Stage B), where this check is a no-op when ids
        # already align. BGKIT_COVERAGE_SCAN_MAX_SAMPLES caps it to a bounded
        # random sample per split, which still catches SYSTEMATIC id drift in
        # seconds. Unset (or <=0) = full scan (default, unchanged behavior).
        import os
        import random as _random
        _cap_env = os.environ.get("BGKIT_COVERAGE_SCAN_MAX_SAMPLES", "").strip()
        _cap = int(_cap_env) if _cap_env.isdigit() and int(_cap_env) > 0 else None
        _rng = _random.Random(17)
        for split_name, split in (
            ("train", self.train_dataset), ("eval", self.eval_dataset),
        ):
            n = len(split)
            if _cap is not None and n > _cap:
                # Sorted = sequential parquet-row access; random order is
                # cache-hostile (each split[idx] re-reads a row group).
                indices = sorted(_rng.sample(range(n), _cap))
                sampled = True
            else:
                indices = list(range(n))
                sampled = False
            for idx in indices:
                sample = split[idx]
                _check_sample(sample)
            logger.info(
                "phase2_kb_coverage_scan_split",
                split=split_name,
                samples=len(indices),
                sampled=sampled,
                total=n,
            )

        if unresolved_by_dataset:
            details = {
                ds: {"count": len(ids), "sample": ids[:5]}
                for ds, ids in unresolved_by_dataset.items()
            }
            raise RuntimeError(
                "phase2_kb: trajectory retrieval ids that resolve to NO browse-tree "
                "article (not a tree node, no leaf tag) — every such bgkit turn "
                "would splice a ZERO survivor and the sample would train on "
                "nothing. Rebuild the browse tree so it contains every article "
                "the trajectories reference (flat trees: raise "
                f"`build_browse_tree.py --leaf-cap`). Unresolved: {details}"
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
            for _node_id, node in all_nodes.items():
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
        l0_selection_mode: str = "threshold",
        l0_ratio: float | None = None,
        gold_span: tuple[int, int] | None = None,
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

        # L0 input embeddings + the per-sample query, built up front so the live
        # L0 forward can attend to the query (query-conditioned within-document
        # compression). The same q_emb feeds L1's compression prompt below.
        embed_tokens = self.encoder.l0.backbone.get_input_embeddings()
        q_ids = self.encoder_tokenizer.encode(query, add_special_tokens=False) or [0]
        q_tensor = torch.tensor(q_ids, dtype=torch.long, device=self.device)
        q_emb = embed_tokens(q_tensor)  # (L_query, D), L0 input space

        # Packed L0 survivors for all articles in this turn — QUERY-CONDITIONED
        # in the live path (the question is fed to L0 as its compression prompt).
        # ``l0_selection_mode`` lets the recursive full-backprop leaf drill force
        # ``exact_topk`` (deterministic ceil>=1 retention) instead of the frozen-
        # policy threshold θ, which stalls near the tanh ceiling and starves the
        # retrieve leaf's L0 to ~0.9% (2026-07-30 recon fix). Default threshold
        # keeps every other dataset/run unchanged. ``l0_ratio`` pins the leaf's
        # L0 retention (drill_leaf_retention.l0); it is threaded CONDITIONALLY
        # so monkeypatched test doubles with the legacy _l0_for_articles
        # signature keep working when the knob is unset.
        l0_kwargs: dict = {"query_emb": q_emb, "selection_mode": l0_selection_mode}
        if l0_ratio is not None:
            l0_kwargs["ratio"] = float(l0_ratio)
        # v5: the gold answer span lives in the FIRST gold article (flat
        # single-gold datasets). Threaded conditionally (legacy test doubles).
        self._last_l0_survivor_mask = None
        self._last_l0_content_cu = None
        if gold_span is not None and article_ids:
            l0_kwargs["gold_spans"] = {article_ids[0]: gold_span}
        l0_flat, l0_cu = self._l0_for_articles(dataset, all_ids, **l0_kwargs)
        l0_lengths = lengths_from_cu(l0_cu).tolist()
        # Per-article L1 span flags: for article 0 (gold w/ span), which of its
        # L0 SURVIVORS originate inside the span (survivor order == content
        # order, so the k-th survivor row maps to the k-th True in the mask).
        l1_span_flags: dict[int, list[bool]] = {}
        _sm = getattr(self, "_last_l0_survivor_mask", None)
        _ccu = getattr(self, "_last_l0_content_cu", None)
        if gold_span is not None and _sm is not None and _ccu is not None:
            a0, a1 = int(_ccu[0].item()), int(_ccu[1].item())
            art_mask = _sm[a0:a1].to("cpu")
            surv_pos = art_mask.nonzero().flatten().tolist()  # article-local
            s, e = int(gold_span[0]), int(gold_span[1])
            l1_span_flags[0] = [s <= pos < e for pos in surv_pos]

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

        pieces: list[torch.Tensor] = []
        pinned_list: list[bool] = []
        relevance_list: list[bool] = []
        # True at L0-survivor positions, False at pinned-ID-token positions — so
        # _run_l1_batch bridges ONLY the survivors (the IDs are already
        # input-space and must not pass through auto_reproduce).
        survivor_list: list[bool] = []
        span_list: list[bool] = []
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
            survivor_list.extend([False] * len(aid_tokens))
            span_list.extend([False] * len(aid_tokens))

            k_i = int(l0_lengths[i]) if i < len(l0_lengths) else 0
            if k_i > 0:
                start = int(l0_cu[i].item())
                pieces.append(l0_flat[start : start + k_i].to(l0_flat.dtype))
                pinned_list.extend([False] * k_i)
                relevance_list.extend([is_relevant] * k_i)
                survivor_list.extend([True] * k_i)
                flags = l1_span_flags.get(i)
                if flags is not None and len(flags) == k_i:
                    span_list.extend(flags)
                else:
                    span_list.extend([False] * k_i)

        content = torch.cat(pieces, dim=0)  # (L_content, D)
        pinned = torch.tensor(pinned_list, dtype=torch.bool, device=self.device)
        relevance_mask = torch.tensor(
            relevance_list, dtype=torch.bool, device=self.device,
        )
        survivor_mask = torch.tensor(
            survivor_list, dtype=torch.bool, device=self.device,
        )
        span_mask_l1 = torch.tensor(span_list, dtype=torch.bool, device=self.device)

        return {
            "content": content,
            "pinned": pinned,
            "relevance_mask": relevance_mask,
            "span_mask": span_mask_l1,
            "survivor_mask": survivor_mask,
            "query_emb": q_emb.to(content.dtype),
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
        hidden_dim = self.encoder.active_projection_output_dim
        zero_fallback = torch.zeros(
            (1, hidden_dim), device=self.device, dtype=torch.bfloat16,
        )
        # Mode-tagged drill-down turns (head / node) carry no packed "content"
        # buffer — they are resolved via :meth:`_resolve_special_survivor`, not
        # the packed L1 forward. Only plain leaf-drill dicts join ``non_null``.
        def _is_special(t) -> bool:
            return isinstance(t, dict) and "mode" in t

        non_null = [t for t in prepared if t is not None and not _is_special(t)]
        if not non_null:
            return [
                self._resolve_special_survivor(t) if _is_special(t) else zero_fallback
                for t in prepared
            ]

        target_dtype = non_null[0]["content"].dtype
        batch_size = len(non_null)

        # Pack all turns' content + query flat, with per-turn cu_seqlens.
        content_pieces: list[torch.Tensor] = [t["content"] for t in non_null]
        query_pieces: list[torch.Tensor] = [t["query_emb"] for t in non_null]
        pinned_pieces: list[torch.Tensor] = [t["pinned"] for t in non_null]
        relevance_pieces: list[torch.Tensor] = [
            t["relevance_mask"].to(self.device) for t in non_null
        ]
        span_pieces: list[torch.Tensor] = [
            (
                t["span_mask"].to(self.device)
                if t.get("span_mask") is not None
                else torch.zeros(int(t["content"].size(0)), dtype=torch.bool, device=self.device)
            )
            for t in non_null
        ]

        content_lengths = [int(c.size(0)) for c in content_pieces]
        query_lengths = [int(q.size(0)) for q in query_pieces]

        content_flat = torch.cat(content_pieces, dim=0).to(target_dtype)
        query_flat = torch.cat(query_pieces, dim=0).to(target_dtype)
        pinned_flat = torch.cat(pinned_pieces, dim=0).to(self.device)
        relevance_flat = torch.cat(relevance_pieces, dim=0)
        span_flat = torch.cat(span_pieces, dim=0)

        content_cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        content_cu[1:] = torch.tensor(
            content_lengths, dtype=torch.int32, device=self.device,
        ).cumsum(0)
        query_cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        query_cu[1:] = torch.tensor(
            query_lengths, dtype=torch.int32, device=self.device,
        ).cumsum(0)

        query_pos_ids = position_ids_from_cu(query_cu, int(query_flat.shape[0]))

        from bgkit.training.survivorship_helpers import LevelLossCfg
        util_active_l1 = getattr(
            self, "_surv_l1", LevelLossCfg(),
        ).utility_grad_loss_weight > 0.0
        l1_grad_capture: dict | None = {} if util_active_l1 else None
        if target_ratio is None:
            # Retrieve-leaf drill L1 override (drill_leaf_retention.l1): applies
            # when the caller didn't pin a ratio — notably the SINGLE-SAMPLE /
            # EVAL path (_build_decoder_segments_core) — so eval measures the
            # SAME leaf regime as training instead of the sampled base
            # l1_retention (the pre-2026-07-31 eval/train divergence). Unset →
            # sampled L1 retention, exactly legacy.
            target_ratio = self._drill_leaf_l1_retention_override()
        if target_ratio is None:
            target_ratio = self._sample_l1_retention()

        # Oracle-span ablation: the L1 span mask (True at L0-survivor rows
        # originating inside the gold answer span) becomes a must-keep so the
        # span survives L1 at the same budget too. Eval-only (the training
        # drill-checkpoint branch never sees an explicit ablation mode).
        must_keep_l1 = None
        if self._ablation_mode == self.ABLATION_ORACLE_SPAN and bool(span_flat.any()):
            must_keep_l1 = span_flat

        # Bridge ONLY the L0-survivor positions (hidden→input space) through
        # auto_repro_head. The interleaved pinned article-ID embeddings are
        # ALREADY input-space (embed_tokens) and must NOT be bridged —
        # auto_reproduce norms+projects, which mangles them. This matches
        # encoder.forward (bridges L0 survivors only). ``survivor_mask`` comes
        # from _prepare_l1_turn: True at L0-survivor positions, False at ID
        # tokens.
        survivor_flat = torch.cat(
            [t["survivor_mask"] for t in non_null], dim=0,
        ).to(self.device)
        bridged_content = content_flat.clone()
        if bool(survivor_flat.any()):
            bridged_content[survivor_flat] = self.encoder.l0.auto_reproduce(
                content_flat[survivor_flat],
            ).to(content_flat.dtype)

        # Run the SHARED L1 stage (cross-section merge → L1 → projection) — the
        # very encoder.run_l1_and_project that encoder.forward uses, so the
        # Phase-2 and summarization L0→L1 paths cannot diverge. content_cu is
        # already per-turn (sections joined in _prepare_l1_turn), so no
        # content_group_cu_seqlens re-segmentation is needed here. Activate the
        # L1 LoRA when installed (a no-op under direct training).
        #
        # DRILL CHECKPOINT (Option A retained-drill OOM fix): when this drill
        # forward's activations would otherwise be RETAINED for the deferred
        # encoder backward (Option A — aux OFF, no util capture, large seqlen),
        # wrap run_l1_and_project in torch.utils.checkpoint so its activations
        # free after the forward and recompute (bit-exact) during backward.
        # θ is accumulated ONCE outside the checkpoint (recompute would
        # otherwise double-count). Gated OFF when aux ON / util capture active
        # (those retain ``out`` for the aux backward anyway) or below threshold.
        drill_ckpt = (
            self.encoder.training
            and not getattr(self, "_survivorship_aux", True)
            and not util_active_l1
            and int(getattr(self, "_drill_checkpoint_min_seqlen", 0)) > 0
            and bridged_content.requires_grad
            and int(bridged_content.shape[0])
            > int(self._drill_checkpoint_min_seqlen)
        )
        if drill_ckpt:
            from torch.utils.checkpoint import checkpoint

            projected, projected_cu_t, l1_counts = checkpoint(
                self._l1_project_pure,
                bridged_content, content_cu, float(target_ratio),
                query_flat, query_cu, query_pos_ids, pinned_flat,
                use_reentrant=False,
            )
            # θ ONCE, outside the checkpoint (detached counts from the forward).
            self._accumulate_theta_from_counts("l1", l1_counts, float(target_ratio))
        else:
            with self._l1_adapter_context():
                out, proj_out, proj_cu = self.encoder.run_l1_and_project(
                    l1_input_embeddings=bridged_content,
                    l1_input_cu_seqlens=content_cu,
                    target_ratio_l1=target_ratio,
                    content_group_cu_seqlens=None,
                    prompt_embeddings_l1=query_flat,
                    prompt_cu_seqlens_l1=query_cu,
                    prompt_position_ids_l1=query_pos_ids,
                    pinned_positions_l1=pinned_flat,
                    utility_grad_active_l1=util_active_l1,
                    utility_grad_capture_l1=l1_grad_capture,
                    selection_mode_l1=getattr(self, "_selection_mode_l1", "threshold"),
                    must_keep_mask_l1=must_keep_l1,
                )
            if l1_grad_capture is not None:
                out._l1_grad_capture = l1_grad_capture  # type: ignore[attr-defined]

            # L1 selection stash, mirroring the L0 one in _l0_for_articles and
            # set UNCONDITIONALLY (the aux-loss stash below is training-only).
            # Without it an eval-mode probe cannot see which L1 rows survived
            # and has to ASSUME the span survives at the uniform L1 keep rate —
            # which is what diag_span_survival did, making its end-to-end
            # number a restatement of the L0 number rather than a measurement
            # (2026-08-27).
            self._last_l1_survivor_mask = getattr(out, "survivor_mask", None)
            self._last_l1_span_mask = span_flat
            self._last_l1_content_cu = content_cu

            # Stash encoder output for aux loss computation (when training).
            # Everything is flat: relevance_mask and pinned are flat (N_content,)
            # bool, cu_seqlens is the per-turn segmentation of the flat buffer.
            if self.encoder.training:
                if getattr(self, "_survivorship_aux", True):
                    if hasattr(self, "_pending_l1_outputs"):
                        self._pending_l1_outputs.append({
                            "enc_out": out,
                            "cu_seqlens": content_cu,
                            "pinned": pinned_flat,
                            "relevance_mask": relevance_flat,
                            "span_mask": span_flat,
                            "ratio": target_ratio,
                        })
                else:
                    # Aux OFF: θ-only accumulation, no graph retention.
                    self._accumulate_theta_state("l1", out, target_ratio)

            projected = proj_out.projected_embeddings
            projected_cu_t = effective_projection_cu(proj_out, proj_cu)

        # Extract per-turn projected survivors via per-turn boundaries.
        surv_cu = projected_cu_t.to(torch.int64).tolist()
        per_turn: list[torch.Tensor] = []
        for i in range(batch_size):
            start = int(surv_cu[i])
            end = int(surv_cu[i + 1])
            if end <= start:
                per_turn.append(zero_fallback)
            else:
                per_turn.append(projected[start:end])

        # Re-interleave with None fallbacks + mode-tagged drill-down survivors.
        results: list[torch.Tensor] = []
        it = iter(per_turn)
        for t in prepared:
            if t is None:
                results.append(zero_fallback)
            elif _is_special(t):
                results.append(self._resolve_special_survivor(t))
            else:
                results.append(next(it))
        return results

    def _l1_project_pure(
        self,
        bridged_content: torch.Tensor,
        content_cu: torch.Tensor,
        target_ratio: float,
        query_flat: torch.Tensor,
        query_cu: torch.Tensor,
        query_pos_ids: torch.Tensor,
        pinned_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PURE drill L1 forward (no θ / _pending / util side-effects) —
        checkpoint-safe. Runs ``run_l1_and_project`` (inside the L1 LoRA
        context so the recompute uses the same adapter) and returns ONLY
        tensors: ``(projected_embeddings, effective_projection_cu, counts)``
        where ``counts`` is a detached ``(3,)`` ``[organic, controllable,
        valid]`` for θ accumulation OUTSIDE the checkpoint. Used only when
        survivorship aux is OFF and utility-grad capture is inactive (Option A),
        so dropping the ``out`` object loses nothing the caller needs."""
        with self._l1_adapter_context():
            out, proj_out, proj_cu = self.encoder.run_l1_and_project(
                l1_input_embeddings=bridged_content,
                l1_input_cu_seqlens=content_cu,
                target_ratio_l1=target_ratio,
                content_group_cu_seqlens=None,
                prompt_embeddings_l1=query_flat,
                prompt_cu_seqlens_l1=query_cu,
                prompt_position_ids_l1=query_pos_ids,
                pinned_positions_l1=pinned_flat,
                utility_grad_active_l1=False,
                utility_grad_capture_l1=None,
                selection_mode_l1=getattr(self, "_selection_mode_l1", "threshold"),
            )

        def _scalar(x) -> float:
            return float(x.item()) if torch.is_tensor(x) else float(x or 0)

        counts = torch.tensor(
            [
                _scalar(getattr(out, "organic_count", 0)),
                _scalar(getattr(out, "controllable_count", 0)),
                _scalar(getattr(out, "valid_count", 0)),
            ],
            dtype=torch.float32, device=self.device,
        )
        return (
            proj_out.projected_embeddings,
            effective_projection_cu(proj_out, proj_cu),
            counts,
        )

    # ------------------------------------------------------------------
    # Recursive-L1 path-selective browse encode (Phase 3)
    # ------------------------------------------------------------------

    def _encode_subtree(
        self,
        dataset: str,
        tree,
        node_id: str,
        q_emb: torch.Tensor,
        memo: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]],
        stats: dict[str, int],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Recursively encode a node's full subtree LIVE (full-backprop mode).

        - leaf node → its articles' L0 survivors (LIVE-encoded when
          ``live_l0: true`` — e.g. the per-repo run; cached/frozen otherwise) →
          ``l0.auto_reproduce`` (L0-output → L1-input) →
          :meth:`_encode_tree_node_live` (shared ID-injecting primitive);
        - interior node → :meth:`_encode_tree_node_live` over
          ``[l1_auto_reproduce(encode_subtree(child).l1out) for ALL children]``
          — every child is encoded live + recursively, NO detach anywhere, so
          gradient reaches every node in the subtree.

        Returns ``(projected_decoder_embeddings, l1_output_survivors)``;
        ``(None, None)`` when the node resolves to nothing. Memoized by
        ``node_id`` so each node is encoded exactly once per sample.
        """
        cached = memo.get(node_id)
        if cached is not None:
            return cached
        # In-progress sentinel to break cycles: a browse tree SHOULD be acyclic,
        # but an id collision can introduce one, and this recursion would then
        # hang forever. A cyclic re-entry now returns (None, None) (skipped).
        memo[node_id] = (None, None)
        node = tree.get(node_id)
        if not node.children:
            result = self._encode_leaf_subtree(dataset, node, q_emb)
        else:
            children_ids: list[str] = []
            children_survivors_l1in: list[torch.Tensor] = []
            for cid in node.children:
                if cid not in tree:
                    continue
                _cproj, c_l1out = self._encode_subtree(
                    dataset, tree, cid, q_emb, memo, stats,
                )
                if c_l1out is None or c_l1out.numel() == 0:
                    continue
                # Bridge L1-output → L1-input caller-side; the shared primitive
                # injects the child id (cid) into the node rep.
                children_ids.append(cid)
                children_survivors_l1in.append(
                    self.encoder.l1_auto_reproduce(c_l1out),
                )
            if not children_survivors_l1in:
                result = (None, None)
            else:
                result = self._encode_tree_node_live(
                    children_ids, children_survivors_l1in, q_emb,
                )
        memo[node_id] = result
        if result[0] is not None:
            stats["nodes"] += 1
        return result

    def _encode_leaf_subtree(
        self, dataset: str, node, q_emb: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Live L0→L1 encode of a leaf node's articles.

        Pulls the node's articles' L0 survivors (live in Stage A, cached
        otherwise — the L0 cache is the frozen precomputed input), bridges
        L0-output → L1-input via ``encoder.l0.auto_reproduce``, and runs the
        recursive L1 stage. Returns ``(projected_decoder_embeddings,
        l1_output_survivors)``, or ``(None, None)`` if the node resolves to no
        cached articles. Shared by the path-selective base case and the
        full-backprop leaf case.
        """
        article_ids = self._resolve_article_ids(dataset, [node.id])
        if not article_ids:
            return None, None

        # FIX 2b: checkpoint the L0 LEAF ENCODE (the per-node L0 forward over the
        # leaf's tokens). FIX 2 only checkpointed the L1 recursion; the residual
        # encode peak was the L0 leaf forward retained across all nodes. When
        # enabled (git_repro), the L0 leaf forward's activations are recomputed
        # (one node at a time) on the final backward instead of all retained.
        # ``ratio`` is sampled ONCE here (outside the checkpoint) so the
        # recompute is deterministic; θ is accumulated ONCE outside (the same
        # double-count hazard as FIX 2). Gradient is exact (checkpoint recompute).
        use_ckpt = (
            getattr(self, "_checkpoint_tree_encode", False)
            and getattr(self, "_tree_encode_ckpt_active", True)
            and self.encoder.training
        )
        if use_ckpt:
            from torch.utils.checkpoint import checkpoint

            ratio_l0 = self._sample_l0_retention_for(dataset)
            l1in, surv_cu, l0_counts = checkpoint(
                self._l0_leaf_forward,
                dataset, article_ids, q_emb, ratio_l0,
                use_reentrant=False,
            )
            if l1in is None or l1in.numel() == 0:
                return None, None
            if not getattr(self, "_survivorship_aux", True):
                self._accumulate_theta_from_counts("l0", l0_counts, ratio_l0)
        else:
            # Default (non-checkpointed) path. Also a TREE LEAF, so it must use
            # exact_topk (>=1 survivor) — the θ path here was the ~16% residual
            # 0-node collapse on small trees after the checkpointed path was
            # fixed. Non-checkpointed → no recompute risk; selection_mode is
            # captured anyway so the inner L0 encode is stable regardless.
            l0_flat, surv_cu = self._l0_for_articles(
                dataset, article_ids, query_emb=q_emb,
                selection_mode="exact_topk",
            )
            if l0_flat.numel() == 0:
                return None, None
            l1in = self.encoder.l0.auto_reproduce(l0_flat)

        # Split the bridged L0 survivors per article (surv_cu carries the
        # per-article boundaries) and inject each article's id (file name /
        # title) via the SHARED encode primitive — a leaf node's rep then
        # carries which files it holds, by id.
        children_ids, children_survivors_l1in = self._leaf_children_from_survivors(
            dataset, article_ids, l1in, surv_cu,
        )
        if not children_survivors_l1in:
            return None, None
        return self._encode_tree_node_live(
            children_ids, children_survivors_l1in, q_emb,
        )

    def _leaf_children_from_survivors(
        self,
        dataset: str,
        article_ids: list[str],
        l1in: torch.Tensor,
        surv_cu: torch.Tensor,
    ) -> tuple[list[str], list[torch.Tensor]]:
        """Split flat bridged L0 survivors ``l1in`` into per-article children
        (by ``surv_cu`` boundaries) paired with each article's display id.

        Ids follow the same rule as ``_prepare_l1_turn``: the ``_doc_id_to_title``
        map when present (title-keyed datasets), else the raw article id (for
        git-repro the data layer sets leaf article ids = file names, so this is
        the file name). Zero-survivor articles are dropped."""
        doc_to_title = (getattr(self, "_doc_id_to_title", None) or {}).get(
            dataset, {},
        )
        cu = surv_cu.to(torch.int64).tolist()
        children_ids: list[str] = []
        survivors: list[torch.Tensor] = []
        for i in range(len(cu) - 1):
            start, end = int(cu[i]), int(cu[i + 1])
            if end <= start:
                continue
            aid = article_ids[i] if i < len(article_ids) else article_ids[-1]
            children_ids.append(doc_to_title.get(aid, aid))
            survivors.append(l1in[start:end])
        return children_ids, survivors

    @contextlib.contextmanager
    def _suspend_inner_backbone_ckpt(self, *backbones):
        """Temporarily disable per-block gradient checkpointing on the given
        encoder ``backbone``(s) for the duration of an OUTER tree/leaf
        checkpoint segment.

        The Option-A tree-encode already wraps each leaf / interior node in a
        ``torch.utils.checkpoint(use_reentrant=False)`` segment, so the whole
        node forward is recomputed on backward. Leaving the backbone's own
        block-level checkpointing on means the backbone blocks get recomputed a
        SECOND time inside that recompute — pure redundant compute (~1.5x the
        node cost) with no memory benefit, since the outer segment already
        discards the node's activations. Disabling it here trades a modestly
        higher transient per-node activation peak (bounded to a single node's
        backbone, well within headroom) for the removed double-recompute.

        Toggled INSIDE the checkpoint target fn with try/finally so the flag is
        identical on the forward and recompute passes (bit-exact, keeps
        ``use_reentrant=False`` tensor-metadata matching) and restored for every
        other path (decoder, flat ``_l0_for_articles``, eval)."""
        saved = [
            (b, getattr(b, "_gradient_checkpointing", False))
            for b in backbones
            if b is not None and hasattr(b, "_gradient_checkpointing")
        ]
        for b, _ in saved:
            b._gradient_checkpointing = False
        try:
            yield
        finally:
            for b, prev in saved:
                b._gradient_checkpointing = prev

    def _l0_leaf_forward(
        self, dataset: str, article_ids: list[str], q_emb: torch.Tensor,
        ratio: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """PURE leaf L0 forward (no θ / _pending side-effects) — checkpoint-safe.

        Runs the live L0 encode at the SUPPLIED ``ratio`` (deterministic across
        checkpoint recompute), bridges L0-output → L1-input via
        ``l0.auto_reproduce``, and returns ``(l1in, survivor_cu, counts)`` where
        ``survivor_cu`` carries the per-article survivor boundaries (so the
        caller can split ``l1in`` into per-article children for id injection)
        and ``counts`` is a detached ``[organic, controllable, valid]`` tensor
        for θ accumulation OUTSIDE the checkpoint. Returns
        ``(None, empty, zeros)`` when the leaf resolves to no survivors."""
        # exact_topk (ceil(count*ratio) >= 1, never 0) fixes the retention-ramp
        # 0-node collapse. Hardcoded here (a tree leaf is ALWAYS exact_topk) so the
        # L1-node checkpoint recompute re-derives it identically — no transient
        # _recursive_l0_override read, which is what crashed the earlier attempts.
        with self._suspend_inner_backbone_ckpt(
            getattr(getattr(self.encoder, "l0", None), "backbone", None),
        ):
            out, _content_cu, _ratio = self._live_l0_encode(
                dataset, article_ids, query_emb=q_emb, ratio=ratio,
                selection_mode="exact_topk",
            )
        survivors = out.survivor_embeddings
        if survivors is None or survivors.numel() == 0:
            return (
                None,
                torch.zeros(1, dtype=torch.int32, device=self.device),
                torch.zeros(3, dtype=torch.float32, device=self.device),
            )
        l1in = self.encoder.l0.auto_reproduce(survivors)
        counts = torch.tensor(
            self._l1_counts(out), dtype=torch.float32, device=self.device,
        )
        return l1in, out.survivor_cu_seqlens, counts

    @staticmethod
    def _l1_counts(l1_out) -> list[float]:
        """Detached ``[organic, controllable, valid]`` keep-rate counts from an
        L1 ``LevelOutput`` (for θ dual-ascent accumulation OUTSIDE checkpoints)."""

        def _scalar(x) -> float:
            return float(x.item()) if torch.is_tensor(x) else float(x or 0)

        return [
            _scalar(getattr(l1_out, "organic_count", 0)),
            _scalar(getattr(l1_out, "controllable_count", 0)),
            _scalar(getattr(l1_out, "valid_count", 0)),
        ]

    def _encode_tree_node_forward(
        self,
        children_ids: list[str],
        children_survivors_l1in: list[torch.Tensor],
        q_emb: torch.Tensor,
        ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PURE per-node encode (no side-effects) — checkpoint-safe.

        Funnels through the SHARED :func:`encode_tree_node` primitive so the
        child-ID injection is identical to every other tree path. Returns
        ``(projected_embeddings, l1_survivors, counts)`` where ``counts`` is a
        detached ``(3,)`` ``[organic, controllable, valid]`` used by the caller
        for θ accumulation OUTSIDE any checkpoint (so recompute does not
        double-count). No θ side-effect here."""
        proj, l1_out = encode_tree_node(
            self.encoder,
            self.encoder_tokenizer,
            list(children_ids),
            children_survivors_l1in,
            q_emb,
            float(ratio),
            selection_mode=getattr(self, "_selection_mode_l1", "threshold"),
            project=True,
            adapter_context=self._l1_adapter_context(),
        )
        counts = torch.tensor(
            self._l1_counts(l1_out), dtype=torch.float32, device=self.device,
        )
        return proj, l1_out.survivor_embeddings, counts

    def _encode_tree_node_forward_cat(
        self,
        surv_cat: torch.Tensor,
        sizes: tuple[int, ...],
        children_ids: tuple[str, ...],
        q_emb: torch.Tensor,
        ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Checkpoint entry point: a SINGLE gradient-carrying tensor
        (``surv_cat``, all children's bridged survivors concatenated) crosses
        the checkpoint boundary — the proven full-backprop mechanism. Split
        back into the per-child list (``sizes``) and defer to
        :meth:`_encode_tree_node_forward`; the child-ID embeddings are rebuilt
        INSIDE (recomputed on backward, so ``embed_tokens`` gradient flows)."""
        survivors = list(torch.split(surv_cat, list(sizes), dim=0))
        with self._suspend_inner_backbone_ckpt(
            getattr(getattr(self.encoder, "l1", None), "backbone", None),
        ):
            return self._encode_tree_node_forward(
                list(children_ids), survivors, q_emb, ratio,
            )

    def _encode_tree_node_live(
        self,
        children_ids: list[str],
        children_survivors_l1in: list[torch.Tensor],
        q_emb: torch.Tensor,
        ratio: float | None = None,
        force_checkpoint: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one node via the shared primitive with the trainer's ratio
        sampling, θ dual-ascent, and (optional) encode-chunking checkpoint.
        Returns ``(projected_decoder_embeddings, l1_output_survivors)``.

        ``ratio``: L1 retention for this node forward; ``None`` (all legacy
        callers) → the recursive L1 ramp. The query-conditioned drill path
        passes ``drill_node_retention``.
        ``force_checkpoint``: wrap the node forward in ``torch.utils.checkpoint``
        REGARDLESS of the per-tree size gates. Required for the per-sample
        query-conditioned drill re-encodes, which run O(path-length x samples)
        per repo — forcing the checkpoint frees each node's activations after
        its forward so peak memory does NOT scale with path length (still
        skipped outside training / when no input carries grad, where there is
        nothing to retain).

        FIX 2 (encode-chunking): when ``checkpoint_tree_encode`` is on AND we're
        building the live tree under grad, the per-node forward is wrapped in
        ``torch.utils.checkpoint`` (use_reentrant=False) so the node's internal
        activations are FREED after the forward (recomputed once during the
        final backward) — bounding the encode forward peak to ~one node instead
        of all-nodes-at-once. To keep the single-tensor checkpoint boundary that
        the full-backprop path relies on, the children survivors are
        concatenated before the checkpoint and re-split inside. Gradient is
        exact (checkpoint recompute) and still reaches every node + every child
        ID embedding.
        """
        ratio = (
            self._recursive_l1_retention_now() if ratio is None else float(ratio)
        )
        requires_grad = any(
            torch.is_tensor(s) and s.requires_grad
            for s in children_survivors_l1in
        )
        use_ckpt = (
            (
                (
                    getattr(self, "_checkpoint_tree_encode", False)
                    and getattr(self, "_tree_encode_ckpt_active", True)
                )
                or force_checkpoint
            )
            and self.encoder.training
            and requires_grad
        )
        if use_ckpt:
            from torch.utils.checkpoint import checkpoint

            sizes = tuple(int(s.shape[0]) for s in children_survivors_l1in)
            surv_cat = torch.cat(
                [s.to(children_survivors_l1in[0].dtype)
                 for s in children_survivors_l1in],
                dim=0,
            )
            proj_emb, l1_surv, counts = checkpoint(
                self._encode_tree_node_forward_cat,
                surv_cat, sizes, tuple(children_ids), q_emb, ratio,
                use_reentrant=False,
            )
        else:
            proj_emb, l1_surv, counts = self._encode_tree_node_forward(
                children_ids, children_survivors_l1in, q_emb, ratio,
            )
        # θ accumulation OUTSIDE any checkpoint (runs exactly once; counts are
        # detached scalars). When survivorship aux is OFF the recursive tree's
        # L1 nodes are the dominant L1 compression path, so feed their per-node
        # keep-rate to the L1 dual-ascent state at the recursive ramp ratio so
        # θ tracks the recursive ramp (0.30→0.05). Aux-ON keeps drill-driven θ.
        if self.encoder.training and not getattr(self, "_survivorship_aux", True):
            self._accumulate_theta_from_counts("l1", counts, ratio)
        return proj_emb, l1_surv

    # ------------------------------------------------------------------
    # Pure drill-down survivor dispatch (git_commit_repro)
    # ------------------------------------------------------------------

    def _drilldown_zero_survivor(self, reason: str = "unknown") -> torch.Tensor:
        """1-vector zero survivor (decoder/projection space) so a sentinel
        splice always has something to drop in when a drill resolves to
        nothing.

        INSTRUMENTED: a zero survivor means the decoder sees NOTHING at that
        splice — and if it fires broadly (tree collapse, or a trajectory
        node-id ↔ tree/memo-key mismatch), any ablation gap reads ~0 and the
        reps look "decorative" when they are actually ABSENT (this masked the
        shared-tree collapse for a long time — 2026-07-06). Counted per reason;
        logged on first few + every 200th so a systematic miss is visible."""
        c = getattr(self, "_zero_survivor_counts", None)
        if c is None:
            c = self._zero_survivor_counts = {}
        c[reason] = c.get(reason, 0) + 1
        if c[reason] <= 3 or c[reason] % 200 == 0:
            logger.warning(
                "phase2_kb_drilldown_zero_survivor",
                reason=reason,
                count=c[reason],
                n_spliced=len(getattr(self, "_shared_tree_splice_reps", {}) or {}),
            )
        return torch.zeros(
            (1, self.encoder.active_projection_output_dim),
            device=self.device, dtype=torch.bfloat16,
        )

    def _resolve_special_survivor(self, entry: dict) -> torch.Tensor:
        """Resolve a mode-tagged prepared turn (``node`` / ``head``) to its
        projected survivor tensor for the bgkit sentinel splice.

        - ``node`` — legacy: present the window subtree node's GENERIC
          shared-tree rep (memo / detached-splice lookup, NO live encode).
          With ``query_conditioned_drill_nodes`` ON and a task query tagged on
          the turn, the node (on-path AND distractor alike) is instead
          re-encoded LIVE under the task query via the generalized
          :meth:`_shared_tree_head_survivor` at ``drill_node_retention`` with
          a FORCED per-node activation checkpoint.
        - ``head`` — LIVE task-query recursive-L1 over the window node's
          children (detach-and-reaccumulate; see :meth:`_shared_tree_head_survivor`).
          Under query-conditioned-drill-nodes it additionally takes the drill
          ratio + forced checkpoint (uniform with the other node turns).
        """
        mode = entry.get("mode")
        qc_drill = bool(getattr(self, "_query_conditioned_drill_nodes", False))
        if mode == "node":
            query = str(entry.get("query", ""))
            if qc_drill and query:
                return self._shared_tree_head_survivor(
                    str(entry.get("node_id", "")),
                    query,
                    str(entry.get("dataset", "")),
                    ratio=self._drill_node_retention_now(),
                    force_checkpoint=True,
                )
            return self._shared_tree_node_survivor(
                str(entry.get("node_id", "")),
                str(entry.get("dataset", "")),
            )
        if mode == "head":
            if qc_drill:
                return self._shared_tree_head_survivor(
                    str(entry.get("node_id", "")),
                    str(entry.get("query", "")),
                    str(entry.get("dataset", "")),
                    ratio=self._drill_node_retention_now(),
                    force_checkpoint=True,
                )
            return self._shared_tree_head_survivor(
                str(entry.get("node_id", "")),
                str(entry.get("query", "")),
                str(entry.get("dataset", "")),
            )
        return self._drilldown_zero_survivor()

    def _shared_tree_node_survivor(
        self, node_id: str, dataset: str = "",
    ) -> torch.Tensor:
        """NAVIGATION drill: the generic shared-tree rep of ``node_id``.

        Reads the DETACHED requires_grad leaf ``_shared_tree_splice_reps[node_id]``
        (so per-group backward accumulates into it; the single final tree backward
        reaccumulates) and records the node as used — identical bookkeeping to
        :meth:`_recursive_browse_node_reps` (lines that resolve a browse node).
        Falls back to the live ``memo`` proj when no splice dict is installed
        (e.g. the eval / non-per-repo path). When neither the live splice nor
        the live memo carries the node AND the offline L1-tree cache is loaded
        (``_l1_tree_cache is not None`` — the non-per-repo path; STATICALLY
        UNREACHABLE for git_commit_repro which keeps it ``None``), resolve the
        node's generic survivor from the cached 2nd-level survivors. Else a
        zero survivor."""
        splice = getattr(self, "_shared_tree_splice_reps", None)
        used = getattr(self, "_shared_tree_used_nodes", None)
        if splice is not None:
            rep = splice.get(node_id)
            if rep is not None:
                if used is not None:
                    used.add(node_id)
                return rep
        memo = getattr(self, "_shared_tree_memo", None) or {}
        m = memo.get(node_id)
        if m is not None and m[0] is not None:
            return m[0]
        if getattr(self, "_l1_tree_cache", None) is not None and dataset:
            return self._cached_tree_node_survivor(dataset, node_id)
        return self._drilldown_zero_survivor()

    def _cached_tree_node_survivor(
        self, dataset: str, node_id: str,
    ) -> torch.Tensor:
        """QA NODE drill resolved from the offline ``SurvivorBlockCache``.

        Query-AGNOSTIC (uses the general compression prompt). Reads the node's
        DIRECT children's cached L1-outputs (or, for a leaf, the node's own
        cached L1-output), bridges each via the live ``l1_auto_reproduce``, and
        runs one query-agnostic recursive-L1 pass so gradient still flows
        through the live L1 + projection. The cached reads are ``.detach()``-ed
        (they're frozen mmap tensors — backprop INTO them errors). Reads DIRECT
        children only (one cache level; the memory-safe design).

        STATICALLY UNREACHABLE for git_commit_repro: only invoked from
        :meth:`_shared_tree_node_survivor` behind ``_l1_tree_cache is not None``,
        which git-repro's ``_per_repo_full_backprop`` branch keeps ``None``."""
        tree = getattr(self, "_trees", None) or {}
        tree = tree.get(dataset)
        if tree is None or node_id not in tree:
            return self._drilldown_zero_survivor()
        cache = self._l1_tree_cache
        node = tree.get(node_id)
        q_emb = self._recursive_general_prompt_emb()
        children_ids: list[str] = []
        children_survivors_l1in: list[torch.Tensor] = []
        if node.children:
            for cid in node.children:
                if not cache.has(dataset, cid):
                    continue
                c = cache.get(dataset, cid)
                if c.numel() == 0:
                    continue
                c = c.to(self.device, torch.bfloat16).detach()
                children_ids.append(cid)
                children_survivors_l1in.append(self.encoder.l1_auto_reproduce(c))
        elif cache.has(dataset, node_id):
            c = cache.get(dataset, node_id)
            if c.numel() != 0:
                c = c.to(self.device, torch.bfloat16).detach()
                # Leaf-self: the only child is the node's own cached rep.
                children_ids.append(node_id)
                children_survivors_l1in.append(self.encoder.l1_auto_reproduce(c))
        if not children_survivors_l1in:
            return self._drilldown_zero_survivor()
        proj, _l1out = self._encode_tree_node_live(
            children_ids, children_survivors_l1in, q_emb,
        )
        return proj

    def _head_query_emb(self, query: str) -> torch.Tensor:
        """Task-query embedding in L0/L1-input (embed_tokens) space — built
        exactly like :meth:`_prepare_l1_turn`'s ``q_emb`` so the head is
        conditioned on the same signal a leaf drill would use."""
        embed_tokens = self.encoder.l0.backbone.get_input_embeddings()
        q_ids = self.encoder_tokenizer.encode(query, add_special_tokens=False) or [0]
        q_tensor = torch.tensor(q_ids, dtype=torch.long, device=self.device)
        return embed_tokens(q_tensor)

    def _shared_tree_head_survivor(
        self, node_id: str, query: str, dataset: str,
        ratio: float | None = None,
        force_checkpoint: bool = False,
    ) -> torch.Tensor:
        """QUERY-CONDITIONED node drill: LIVE task-query recursive-L1 over a
        tree node's children. Originally the HEAD (window node) drill; the
        query-conditioned-drill-nodes mode routes EVERY trajectory node turn
        (on-path interior + distractor node drills alike) through this same
        proven machinery — ``node_id`` is any tree node, ``ratio`` /
        ``force_checkpoint`` are threaded to :meth:`_encode_tree_node_live`
        (drill ratio + per-sample activation checkpoint).

        The children's L1-outputs live in the shared-tree ``memo`` (built once
        per repo by :meth:`_compute_shared_repo_tree`; ``memo[c] = (proj, l1out)``).
        To keep the detach-and-reaccumulate contract (the shared tree is
        backwarded ONCE per repo, not per group), each child L1-output is
        DETACHED into a requires_grad leaf cached in ``_shared_tree_child_l1_reps``;
        the head reads those leaves so the per-group/per-subset backward
        accumulates into ``rep.grad`` instead of freeing the shared-tree graph.
        Used children are recorded so the final per-repo tree backward feeds each
        child's accumulated gradient back into ``memo[c][1]``. Task-conditioned —
        distinct from the node's general-prompt ``memo[node_id][0]``.

        Falls back to the LIVE ``memo`` l1outs when no reaccumulate dict is
        installed (e.g. the eval / non-per-repo path). When the live ``memo``
        is EMPTY and the offline L1-tree cache is loaded
        (``_l1_tree_cache is not None`` — the non-per-repo path; STATICALLY
        UNREACHABLE for git_commit_repro which keeps it ``None``), the children
        L1-outputs are read from the cache instead — a per-sample task-query
        head over the cached 2nd-level survivors."""
        memo = getattr(self, "_shared_tree_memo", None) or {}
        trees = getattr(self, "_trees", None) or {}
        tree = trees.get(dataset)
        if tree is None or node_id not in tree:
            return self._drilldown_zero_survivor()
        children = tree.get(node_id).children

        children_ids: list[str] = []
        children_survivors_l1in: list[torch.Tensor] = []
        if not memo and getattr(self, "_l1_tree_cache", None) is not None:
            # Cached fallback: build the children reps from the offline
            # L1-tree cache. Frozen mmap tensors → ``.detach()`` (backprop INTO
            # them errors); grad still flows through the live l1_auto_reproduce
            # + recursive-L1.
            cache = self._l1_tree_cache
            for c in children:
                if not cache.has(dataset, c):
                    continue
                cr = cache.get(dataset, c)
                if cr.numel() == 0:
                    continue
                cr = cr.to(self.device, torch.bfloat16).detach()
                children_ids.append(c)
                children_survivors_l1in.append(self.encoder.l1_auto_reproduce(cr))
        else:
            child_reps = getattr(self, "_shared_tree_child_l1_reps", None)
            child_used = getattr(self, "_shared_tree_child_l1_used", None)
            for c in children:
                m = memo.get(c)
                if m is None or m[1] is None or m[1].numel() == 0:
                    continue
                if child_reps is not None:
                    rep = child_reps.get(c)
                    if rep is None:
                        rep = m[1].detach().requires_grad_(True)
                        child_reps[c] = rep
                    if child_used is not None:
                        child_used.add(c)
                    l1out = rep
                else:
                    # No reaccumulate infra (eval path) — consume the live l1out.
                    l1out = m[1]
                children_ids.append(c)
                children_survivors_l1in.append(self.encoder.l1_auto_reproduce(l1out))
        if not children_survivors_l1in:
            return self._drilldown_zero_survivor()
        q_emb = self._head_query_emb(query)
        # Thread the drill kwargs only when non-default so the legacy (flag-
        # off) call is byte-identical to the pre-2026-07-31 path — including
        # for monkeypatched legacy-signature test doubles.
        if ratio is not None or force_checkpoint:
            proj, _l1out = self._encode_tree_node_live(
                children_ids, children_survivors_l1in, q_emb,
                ratio=ratio, force_checkpoint=force_checkpoint,
            )
        else:
            proj, _l1out = self._encode_tree_node_live(
                children_ids, children_survivors_l1in, q_emb,
            )
        return proj

    # ------------------------------------------------------------------
    # Per-repo shared tree (git_commit_repro full-backprop)
    # ------------------------------------------------------------------

    def _dataset_group_keys(self, ds) -> list[str] | None:
        """Bulk per-repo group keys for ``ds``, unwrapping the random_split
        ``Subset`` and ``ConcatDataset`` wrappers to reach
        :meth:`KBTrajectoryDataset.group_keys` (a turns-only bulk read, ~20s vs
        >1h for the per-sample parse). Returns ``None`` if no wrapped dataset
        exposes ``group_keys`` (caller falls back to the per-sample path)."""
        from torch.utils.data import ConcatDataset, Subset

        if hasattr(ds, "group_keys"):
            return list(ds.group_keys())
        if isinstance(ds, Subset):
            parent = self._dataset_group_keys(ds.dataset)
            return None if parent is None else [parent[i] for i in ds.indices]
        if isinstance(ds, ConcatDataset):
            parts = [self._dataset_group_keys(d) for d in ds.datasets]
            if any(p is None for p in parts):
                return None
            out: list[str] = []
            for p in parts:
                out.extend(p)
            return out
        return None

    def _dataset_optional_column(self, ds, method_name: str) -> list[str] | None:
        """Bulk-read an optional trajectory column through dataset wrappers."""
        from torch.utils.data import ConcatDataset, Subset

        method = getattr(ds, method_name, None)
        if callable(method):
            values = list(method())
            return values or None
        if isinstance(ds, Subset):
            parent = self._dataset_optional_column(ds.dataset, method_name)
            return None if parent is None else [parent[i] for i in ds.indices]
        if isinstance(ds, ConcatDataset):
            parts = [
                self._dataset_optional_column(part, method_name)
                for part in ds.datasets
            ]
            if any(part is None for part in parts):
                return None
            values: list[str] = []
            for part in parts:
                values.extend(part or [])
            return values
        return None

    def _dataset_split_labels(self, ds) -> list[str] | None:
        return self._dataset_optional_column(ds, "split_labels")

    def _dataset_repo_ids(self, ds) -> list[str] | None:
        return self._dataset_optional_column(ds, "repo_ids")

    def _repo_group_key(self, sample: KBSample) -> str:
        """Return the shared-subtree root node id for a git_commit_repro
        file-sample — the ``is_head`` bgkit-turn's window node id, which for the
        ``root → repo → repo/wK → c16 → c4 → commit`` layout is the window node
        ``repo/wK`` (the drill-down head).  Every file-sample of one
        ``(repo, window)`` shares this key, so grouping by it batches a repo's
        file-samples together and identifies the root of that window subtree to
        encode once.  Returns ``""`` when the sample has no ``is_head`` drill (no
        shared tree — falls back to per-sample handling)."""
        explicit = str(getattr(sample, "group_id", "") or "")
        if explicit:
            return explicit
        for turn in sample.trajectory:
            if turn.kind == "bgkit" and bool(turn.args.get("is_head")):
                ids = turn.args.get("ids", [])
                return str(ids[0]) if ids else ""
        return ""

    def _subsample_repo_batch(self, batch: list, root_node_id: str) -> list:
        """Optionally cap a repo-batch to ``max_file_samples_per_repo`` (M1/M2
        cost knob). ``None`` → unchanged. Re-seeded per epoch (and per repo) so
        the K-subset rotates across epochs rather than always training the same
        K files. Returns the (possibly smaller) sample list."""
        k = getattr(self, "_max_file_samples_per_repo", None)
        if k is None or len(batch) <= k:
            return batch
        import random as _random

        seed = (
            int(self.cfg.get("seed", 42))
            + int(getattr(self, "epoch", 0))
            + (hash(root_node_id) & 0xFFFFFFFF)
        )
        return _random.Random(seed).sample(list(batch), k)

    def _handle_max_file_samples_per_repo(self, val) -> None:
        """Live-config handler for the per-repo file-sample cap (M1/M2 cost
        knob). ``null``/``None``/``<=0`` → unlimited (None); a positive int →
        clamp to K. :meth:`_subsample_repo_batch` reads
        ``self._max_file_samples_per_repo`` fresh per repo-batch, so a write to
        ``control.json`` takes effect on the next repo-batch with no restart —
        both opening up (raise / null) and clamping down."""
        old = getattr(self, "_max_file_samples_per_repo", None)
        if val is None:
            self._max_file_samples_per_repo = None
            logger.info(
                "live_max_file_samples_per_repo", old=old, new=None,
                meaning="unlimited",
            )
            return
        if isinstance(val, (int, float)):
            k = int(val)
            self._max_file_samples_per_repo = k if k > 0 else None
            logger.info(
                "live_max_file_samples_per_repo",
                old=old, new=self._max_file_samples_per_repo,
            )
            return
        logger.warning(
            "live_max_file_samples_per_repo_invalid",
            value=val, expected="None or int",
        )

    def _handle_per_repo_sample_group_size(self, val) -> None:
        """Live-config handler for the PASS-2 group-batch size G (clamped >=1).
        Read fresh per repo-batch in PASS 2, so a write to ``control.json``
        re-tunes occupancy↔memory on the next repo with no restart."""
        old = getattr(self, "_per_repo_sample_group_size", 1)
        if isinstance(val, (int, float)) and int(val) >= 1:
            self._per_repo_sample_group_size = int(val)
            logger.info(
                "live_per_repo_sample_group_size",
                old=old, new=self._per_repo_sample_group_size,
            )
            return
        logger.warning(
            "live_per_repo_sample_group_size_invalid",
            value=val, expected="int >= 1",
        )

    def _handle_per_repo_inner_loop(self, val) -> None:
        """Live toggle for the inner-loop mode flag. NOTE: the K-step driver is
        a pending integration (see report); until wired this only flips the flag
        + the inner-loop compute primitive's availability, it does NOT change the
        per-repo step path."""
        self._per_repo_inner_loop = bool(val)
        logger.info(
            "live_per_repo_inner_loop", new=self._per_repo_inner_loop,
            note="driver integration pending — flag is inert until wired",
        )

    def _handle_per_repo_option_a(self, val) -> None:
        """Reject the incoherent split-cadence optimizer path."""
        if bool(val):
            raise ValueError(
                "per_repo_option_a is removed; decoder and encoder must be "
                "updated from one coherent repository objective"
            )
        self._per_repo_option_a = False
        logger.info("live_per_repo_option_a", new=False)

    def _handle_option_a_max_subsets(self, val) -> None:
        """Live cap on Option A's subset count (0 = unlimited = all files).
        Read fresh per repo in _partition_option_a_subsets."""
        if isinstance(val, (int, float)) and int(val) >= 0:
            self._option_a_max_subsets = int(val)
            logger.info("live_option_a_max_subsets", new=int(val))
            return
        logger.warning("live_option_a_max_subsets_invalid", value=val)

    def _handle_drill_checkpoint_min_seqlen(self, val) -> None:
        """Live drill activation-checkpoint threshold (0 = off). Read fresh per
        drill in _run_l1_batch → effective immediately, no restart."""
        if isinstance(val, (int, float)) and int(val) >= 0:
            self._drill_checkpoint_min_seqlen = int(val)
            logger.info("live_drill_checkpoint_min_seqlen", new=int(val))
            return
        logger.warning("live_drill_checkpoint_min_seqlen_invalid", value=val)

    def _handle_per_repo_inner_subset_size(self, val) -> None:
        if isinstance(val, (int, float)) and int(val) >= 1:
            self._per_repo_inner_subset_size = int(val)
            logger.info("live_per_repo_inner_subset_size", new=int(val))
            return
        logger.warning("live_per_repo_inner_subset_size_invalid", value=val)

    def _handle_per_repo_max_inner_steps(self, val) -> None:
        if isinstance(val, (int, float)) and int(val) >= 1:
            self._per_repo_max_inner_steps = int(val)
            logger.info("live_per_repo_max_inner_steps", new=int(val))
            return
        logger.warning("live_per_repo_max_inner_steps_invalid", value=val)

    def _handle_inner_loop_warmup_steps(self, val) -> None:
        if isinstance(val, (int, float)) and int(val) >= 0:
            self._inner_loop_warmup_steps = int(val)
            logger.info("live_inner_loop_warmup_steps", new=int(val))
            return
        logger.warning("live_inner_loop_warmup_steps_invalid", value=val)

    def _handle_max_l0_encode_tokens(self, val) -> None:
        """Live cap on the per-leaf L0 encode token buffer (0 = off). Read fresh
        each encode in _live_l0_encode → effective on the next repo, no restart."""
        if isinstance(val, (int, float)) and int(val) >= 0:
            self._max_l0_encode_tokens = int(val)
            logger.info("live_max_l0_encode_tokens", new=int(val))
            return
        logger.warning("live_max_l0_encode_tokens_invalid", value=val)

    def _handle_max_decode_tokens(self, val) -> None:
        """Live cap on the rendered decode sequence (0 = off). Read fresh each
        decode in _encode_decode_group → effective immediately, no restart."""
        if isinstance(val, (int, float)) and int(val) >= 0:
            self._max_decode_tokens = int(val)
            logger.info("live_max_decode_tokens", new=int(val))
            return
        logger.warning("live_max_decode_tokens_invalid", value=val)

    def _handle_checkpoint_tree_encode(self, val) -> None:
        """Live toggle for FIX-2 per-node tree-encode checkpointing. Read fresh
        in _encode_tree_node_live → effective on the next repo's encode, no
        restart."""
        self._checkpoint_tree_encode = bool(val)
        logger.info("live_checkpoint_tree_encode", new=self._checkpoint_tree_encode)

    def _handle_tree_checkpoint_min_nodes(self, val) -> None:
        """Live SPEED threshold: only checkpoint the encode when a repo's tree
        has more than this many nodes. Read fresh in _compute_shared_repo_tree
        → effective on the next repo's encode, no restart. Tune so the largest
        un-checkpointed tree still fits (~40GB margin)."""
        if isinstance(val, (int, float)) and int(val) >= 0:
            self._tree_checkpoint_min_nodes = int(val)
            logger.info("live_tree_checkpoint_min_nodes", new=int(val))
            return
        logger.warning("live_tree_checkpoint_min_nodes_invalid", value=val)

    def _handle_tree_checkpoint_min_tokens(self, val) -> None:
        """Live token-based tree-checkpoint gate (0 = off). Forces the per-node
        checkpoint when a repo's total L0 leaf tokens exceed this — load-bearing
        for the full (uncapped-L0) tree. Read fresh in _compute_shared_repo_tree
        → effective on the next repo's encode."""
        if isinstance(val, (int, float)) and int(val) >= 0:
            self._tree_checkpoint_min_tokens = int(val)
            logger.info("live_tree_checkpoint_min_tokens", new=int(val))
            return
        logger.warning("live_tree_checkpoint_min_tokens_invalid", value=val)

    def _handle_decode_gc_min_seqlen(self, val) -> None:
        """Live SPEED threshold: only force decoder GC in the interleaved decode
        when the decode seqlen exceeds this. Propagated to every decoder so it's
        read fresh per forward → effective immediately. Tune so the longest
        un-GC'd decode still fits (~40GB margin)."""
        if isinstance(val, (int, float)) and int(val) >= 0:
            self._decode_gc_min_seqlen = int(val)
            self._propagate_decode_gc_min_seqlen()
            logger.info("live_decode_gc_min_seqlen", new=int(val))
            return
        logger.warning("live_decode_gc_min_seqlen_invalid", value=val)

    # --- Round-robin decoder routing --------------------------------------
    def _handle_qwen_decoder_prob(self, val) -> None:
        """Live P(qwen35) for round-robin decoder routing (0..1). Read fresh in
        _pick_decoder_family → effective on the next batch, no restart."""
        old = getattr(self, "_qwen_decoder_prob", 0.3)
        if isinstance(val, (int, float)) and 0.0 <= float(val) <= 1.0:
            self._qwen_decoder_prob = float(val)
            logger.info("live_qwen_decoder_prob", old=old, new=self._qwen_decoder_prob)
            return
        logger.warning(
            "live_qwen_decoder_prob_invalid", value=val, expected="float in [0, 1]",
        )

    # --- Survivorship aux losses ------------------------------------------
    def _handle_survivorship_aux(self, val) -> None:
        """Live master switch for ALL survivorship aux losses + the
        _pending_l{0,1}_outputs collection that feeds them. Read fresh via
        getattr each forward → effective on the next step."""
        self._survivorship_aux = bool(val)
        logger.info("live_survivorship_aux", new=self._survivorship_aux)

    # --- Projection-output norm-band regularizer (collapse fix) -----------
    def _handle_projection_norm_reg_weight(self, val) -> None:
        """Live weight for the projection norm-band regularizer (>=0). Read
        fresh each group in _projection_norm_reg_term → effective next step."""
        old = getattr(self, "_proj_norm_reg_weight", 0.0)
        if isinstance(val, (int, float)) and float(val) >= 0.0:
            self._proj_norm_reg_weight = float(val)
            logger.info(
                "live_projection_norm_reg_weight", old=old, new=self._proj_norm_reg_weight,
            )
            return
        logger.warning(
            "live_projection_norm_reg_weight_invalid", value=val, expected="float >= 0",
        )

    def _proj_norm_reg_embed_ref_norm(self, family: str) -> float | None:
        """Mean row-L2-norm of ``family``'s decoder ``embed_tokens.weight`` — the
        scalar the readable norm-band is measured against. Computed once per
        family and cached (the vocab-wide norm is a one-time cost)."""
        cache = self._proj_norm_reg_embed_ref_cache
        if family in cache:
            return cache[family]
        by_family = getattr(self, "_decoders_by_family", None)
        dec = by_family.get(family) if by_family else getattr(self, "decoder", None)
        if dec is None:
            return None
        try:
            w = dec.backbone.get_input_embeddings().weight
        except (AttributeError, TypeError):
            return None
        val = float(w.detach().float().norm(dim=-1).mean())
        cache[family] = val
        return val

    def _projection_norm_reg_term(self, reps: list[torch.Tensor]) -> torch.Tensor:
        """Hinge-band penalty keeping every projected/spliced survivor-rep's L2
        norm inside the ACTIVE decoder family's readable band. THE novel piece of
        the 2026-07-31 collapse fix: the 9164 git-repro reps drifted from ~4x
        embed-norm to 12-320x under-constrained full-backprop, losing content.

        Per family, ``target = target_ratio[family] * mean(embed_tokens row-L2-
        norm)`` (the reps are a co-trained code, NOT the embed_tokens manifold, so
        we anchor the NORM band only — never the direction). Permissive inside
        ``[target/tolerance, target*tolerance]``, quadratic on the runaway::

            loss = weight * mean( relu( |log(rep_norm / target)| - log(tol) )^2 )

        ``tol`` is per-family (``_proj_norm_reg_tolerances[family]``) with the
        scalar ``_proj_norm_reg_tolerance`` as fallback — so one family's band
        can be tightened without penalizing another that is in-band.

        Gradient flows from ``rep_norm`` back through ``projection_blocks[family]``
        into the L1/L0 backbones — that is the point (it constrains what the
        encoder PRODUCES). Only grad-carrying reps count (skips the zero-fallback
        splices). Returns a WEIGHTED float32 scalar (0 when disabled / eval / no
        grad-carrying reps). Also records the per-family mean rep/embed norm-ratio
        for the ``proj_norm_ratio_{family}`` step metric.
        """
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        if not getattr(self, "_proj_norm_reg_enabled", False):
            return zero
        # Training-only: eval reps do not require grad and must not be penalized.
        if not self.encoder.training:
            return zero
        weight = float(getattr(self, "_proj_norm_reg_weight", 0.0))
        if weight <= 0.0:
            return zero
        family = self._decoder_family
        embed_ref = self._proj_norm_reg_embed_ref_norm(family)
        target_ratio = self._proj_norm_reg_target_ratios.get(family)
        if embed_ref is None or not embed_ref > 0 or target_ratio is None:
            return zero
        target = float(target_ratio) * float(embed_ref)
        grad_reps = [
            r for r in reps
            if torch.is_tensor(r) and r.requires_grad and r.numel() > 0
        ]
        if not grad_reps:
            return zero
        flat = torch.cat([r.reshape(-1, r.shape[-1]) for r in grad_reps], dim=0)
        # clamp_min so a degenerate 0-norm rep can't produce a -inf log.
        per_rep_norm = flat.float().norm(dim=-1).clamp_min(1e-6)  # (S,)
        tol = self._proj_norm_reg_tolerances.get(
            family, self._proj_norm_reg_tolerance,
        )
        band = math.log(max(float(tol), 1.0 + 1e-6))
        log_ratio = (per_rep_norm / target).log()
        penalty = torch.relu(log_ratio.abs() - band).pow(2).mean()
        # Log-side: the rep/embed norm-ratio (comparable to the 4.2 / 0.9 band).
        self._proj_norm_ratio_accum.setdefault(family, []).append(
            float(per_rep_norm.mean().detach()) / float(embed_ref),
        )
        return weight * penalty

    def _proj_norm_reg_step_metrics(self) -> dict[str, float]:
        """Per-family mean rep/embed norm-ratio for the step metrics (so drift is
        greppable as ``proj_norm_ratio_qwen35`` / ``proj_norm_ratio_falcon_h1``).
        Reads + does not reset ``_proj_norm_ratio_accum`` (reset per step)."""
        out: dict[str, float] = {}
        for fam, ratios in (getattr(self, "_proj_norm_ratio_accum", None) or {}).items():
            if ratios:
                out[f"proj_norm_ratio_{fam}"] = float(sum(ratios) / len(ratios))
        return out

    def _aux_weight(self, level: str, name: str) -> float:
        """Resolve an aux-loss weight for ``level`` ∈ {l0, l1}.

        Precedence: a LIVE control.json override (applies to BOTH levels) >
        the per-level ``survivorship.{level}.{name}`` (``LevelLossCfg``) >
        the top-level ``survivorship.{name}`` global. Before 2026-08-22 the
        inline aux path read only the global, so the Stage A config's
        deliberate ``l0.decisiveness_loss_weight: 0.0`` was silently
        overridden by the global 0.05 — at p≈0.58 that term pushes the bulk
        UP and fights the ratio term's pull toward the target rate.
        """
        overrides = getattr(self, "_live_aux_overrides", None) or {}
        if name in overrides:
            return float(overrides[name])
        # Per-level value only when it was EXPLICITLY configured (a
        # default-constructed LevelLossCfg carries 0.0 and must not silence
        # the global); ``_level_explicit_aux`` is recorded at config parse.
        explicit = (getattr(self, "_level_explicit_aux", None) or {}).get(level, set())
        lvl_cfg = getattr(self, f"_surv_{level}", None)
        if name in explicit and lvl_cfg is not None and hasattr(lvl_cfg, name):
            return float(getattr(lvl_cfg, name))
        return float(getattr(self, f"_{name}", 0.0))

    def _handle_span_relevance_weight(self, val) -> None:
        """Live v5 gold-span relevance weight (>=0, both levels). Read fresh in
        _compute_survivorship_aux_losses → effective on the next step. Under
        exact_topk with standardized scores this is the term that drives
        ranking quality (ratio/decisiveness are near-constant there)."""
        old = getattr(self, "_span_relevance_weight", 0.0)
        if isinstance(val, (int, float)) and float(val) >= 0.0:
            self._span_relevance_weight = float(val)
            logger.info(
                "live_span_relevance_weight", old=old, new=self._span_relevance_weight,
            )
            return
        logger.warning("live_span_relevance_weight_invalid", value=val, expected="float >= 0")

    def _handle_ratio_loss_weight(self, val) -> None:
        """Live aggregate-ratio loss weight (>=0). Read fresh in
        _compute_survivorship_aux_losses → effective on the next step."""
        old = getattr(self, "_ratio_loss_weight", 0.1)
        if isinstance(val, (int, float)) and float(val) >= 0.0:
            self._ratio_loss_weight = float(val)
            self._live_aux_overrides = {
                **(getattr(self, "_live_aux_overrides", None) or {}),
                "ratio_loss_weight": float(val),
            }
            logger.info("live_ratio_loss_weight", old=old, new=self._ratio_loss_weight)
            return
        logger.warning("live_ratio_loss_weight_invalid", value=val, expected="float >= 0")

    def _handle_decisiveness_loss_weight(self, val) -> None:
        """Live decisiveness loss weight (>=0). Read fresh in
        _compute_survivorship_aux_losses → effective on the next step."""
        old = getattr(self, "_decisiveness_loss_weight", 0.05)
        if isinstance(val, (int, float)) and float(val) >= 0.0:
            self._decisiveness_loss_weight = float(val)
            self._live_aux_overrides = {
                **(getattr(self, "_live_aux_overrides", None) or {}),
                "decisiveness_loss_weight": float(val),
            }
            logger.info(
                "live_decisiveness_loss_weight",
                old=old, new=self._decisiveness_loss_weight,
            )
            return
        logger.warning(
            "live_decisiveness_loss_weight_invalid", value=val, expected="float >= 0",
        )

    def _handle_relevance_loss_weight(self, val) -> None:
        """Live relevance loss weight (>=0). Read fresh in
        _compute_survivorship_aux_losses → effective on the next step."""
        old = getattr(self, "_relevance_loss_weight", 0.05)
        if isinstance(val, (int, float)) and float(val) >= 0.0:
            self._relevance_loss_weight = float(val)
            logger.info(
                "live_relevance_loss_weight", old=old, new=self._relevance_loss_weight,
            )
            return
        logger.warning(
            "live_relevance_loss_weight_invalid", value=val, expected="float >= 0",
        )

    def _handle_relevance_gold_boost(self, val) -> None:
        """Live gold-article per-position target multiplier (>=0). Read fresh in
        _compute_survivorship_aux_losses → effective on the next step."""
        old = getattr(self, "_relevance_gold_boost", 1.5)
        if isinstance(val, (int, float)) and float(val) >= 0.0:
            self._relevance_gold_boost = float(val)
            logger.info(
                "live_relevance_gold_boost", old=old, new=self._relevance_gold_boost,
            )
            return
        logger.warning(
            "live_relevance_gold_boost_invalid", value=val, expected="float >= 0",
        )

    def _handle_relevance_distractor_damp(self, val) -> None:
        """Live distractor per-position target multiplier (>=0). Read fresh in
        _compute_survivorship_aux_losses → effective on the next step."""
        old = getattr(self, "_relevance_distractor_damp", 0.5)
        if isinstance(val, (int, float)) and float(val) >= 0.0:
            self._relevance_distractor_damp = float(val)
            logger.info(
                "live_relevance_distractor_damp",
                old=old, new=self._relevance_distractor_damp,
            )
            return
        logger.warning(
            "live_relevance_distractor_damp_invalid", value=val, expected="float >= 0",
        )

    def _handle_n_distractors(self, val) -> None:
        """Live count of distractor articles sampled per bgkit turn (>=0). Read
        fresh in _prepare_l1_turn → effective on the next sample (bounded by the
        articles available to sample from)."""
        old = getattr(self, "_n_distractors", 3)
        if isinstance(val, (int, float)) and int(val) >= 0:
            self._n_distractors = int(val)
            logger.info("live_n_distractors", old=old, new=self._n_distractors)
            return
        logger.warning("live_n_distractors_invalid", value=val, expected="int >= 0")

    # --- Training-time random ablation probabilities ----------------------
    def _handle_p_skip_bgkit(self, val) -> None:
        """Live training-time bgkit-skip ablation probability (0..1). Rolled
        fresh per sample in _build_decoder_segments_core → next sample."""
        old = getattr(self, "_p_skip_bgkit_training", 0.15)
        if isinstance(val, (int, float)) and 0.0 <= float(val) <= 1.0:
            self._p_skip_bgkit_training = float(val)
            logger.info("live_p_skip_bgkit", old=old, new=self._p_skip_bgkit_training)
            return
        logger.warning("live_p_skip_bgkit_invalid", value=val, expected="float in [0, 1]")

    def _handle_p_skip_topic(self, val) -> None:
        """Live training-time topic-skip ablation probability (0..1). Rolled
        fresh per sample in _build_decoder_segments_core → next sample."""
        old = getattr(self, "_p_skip_topic_training", 0.15)
        if isinstance(val, (int, float)) and 0.0 <= float(val) <= 1.0:
            self._p_skip_topic_training = float(val)
            logger.info("live_p_skip_topic", old=old, new=self._p_skip_topic_training)
            return
        logger.warning("live_p_skip_topic_invalid", value=val, expected="float in [0, 1]")

    def _handle_p_noise_bgkit(self, val) -> None:
        """Live training-time bgkit-noise ablation probability (0..1). Rolled
        fresh per sample in _build_decoder_segments_core → next sample."""
        old = getattr(self, "_p_noise_bgkit_training", 0.0)
        if isinstance(val, (int, float)) and 0.0 <= float(val) <= 1.0:
            self._p_noise_bgkit_training = float(val)
            logger.info("live_p_noise_bgkit", old=old, new=self._p_noise_bgkit_training)
            return
        logger.warning("live_p_noise_bgkit_invalid", value=val, expected="float in [0, 1]")

    # --- Retention ratios --------------------------------------------------
    def _handle_l1_retention(self, val) -> None:
        """Live base L1 retention ratio in (0, 1). Read fresh in
        _sample_l1_retention → effective on the next L1 encode. NOTE: the L1
        ratio-sampler anchor grid was derived from the setup-time value; when
        jitter/anchor sampling is enabled the anchor grid stays as built (only
        the base ratio moves) — a full restart re-derives the grid."""
        old = getattr(self, "_l1_retention", 0.15)
        if isinstance(val, (int, float)) and 0.0 < float(val) < 1.0:
            self._l1_retention = float(val)
            logger.info("live_l1_retention", old=old, new=self._l1_retention)
            return
        logger.warning("live_l1_retention_invalid", value=val, expected="float in (0, 1)")

    def _handle_recursive_l1_retention(self, val) -> None:
        """Live recursive-L1 (shared repo-tree) retention override. ``None``
        clears the override (revert to the configured ramp / sampled ratio); a
        float in (0, 1) pins both the ramp cfg and the scalar fallback. Read
        fresh in _recursive_l1_retention_now → effective on the next encode."""
        if val is None:
            self._recursive_l1_retention_cfg = None
            logger.info("live_recursive_l1_retention_cleared", reverting="ramp/sampled")
            return
        if isinstance(val, (int, float)) and 0.0 < float(val) < 1.0:
            self._recursive_l1_retention_cfg = float(val)
            self._recursive_l1_retention = float(val)
            logger.info("live_recursive_l1_retention", new=float(val))
            return
        logger.warning(
            "live_recursive_l1_retention_invalid",
            value=val, expected="None or float in (0, 1)",
        )

    def _handle_recursive_l0_retention(self, val) -> None:
        """Live recursive-L0 (shared repo-tree leaves) retention override.
        ``None`` clears (revert to default_l0_retention); a float in (0, 1) pins
        it. Read fresh in _recursive_l0_retention_now → effective on the next
        encode."""
        if val is None:
            self._recursive_l0_retention_cfg = None
            logger.info("live_recursive_l0_retention_cleared", reverting="default")
            return
        if isinstance(val, (int, float)) and 0.0 < float(val) < 1.0:
            self._recursive_l0_retention_cfg = float(val)
            logger.info("live_recursive_l0_retention", new=float(val))
            return
        logger.warning(
            "live_recursive_l0_retention_invalid",
            value=val, expected="None or float in (0, 1)",
        )

    def _handle_query_conditioned_drill_nodes(self, val) -> None:
        """Live toggle for the per-sample task-query drill-node re-encode.
        Read fresh in _prepare_sample_for_decode / _resolve_special_survivor →
        effective on the next sample. OFF restores exact legacy behavior
        (static node splices, legacy head ratio/checkpoint)."""
        old = bool(getattr(self, "_query_conditioned_drill_nodes", False))
        self._query_conditioned_drill_nodes = bool(val)
        logger.info(
            "live_query_conditioned_drill_nodes",
            old=old, new=self._query_conditioned_drill_nodes,
        )

    def _handle_drill_node_retention(self, val) -> None:
        """Live retention for the query-conditioned drill-node forwards.
        ``None`` clears (revert to the recursive L1 ramp); a float in (0, 1]
        pins it. Read fresh in _drill_node_retention_now → next drill."""
        if val is None:
            self._drill_node_retention_cfg = None
            logger.info("live_drill_node_retention_cleared", reverting="recursive ramp")
            return
        if isinstance(val, (int, float)) and 0.0 < float(val) <= 1.0:
            self._drill_node_retention_cfg = float(val)
            logger.info("live_drill_node_retention", new=float(val))
            return
        logger.warning(
            "live_drill_node_retention_invalid",
            value=val, expected="None or float in (0, 1]",
        )

    def _handle_drill_leaf_l0_retention(self, val) -> None:
        """Live retrieve-leaf drill L0 retention override. ``None`` clears
        (revert to the per-dataset l0_retention map); a float in (0, 1] pins
        it. Read fresh in _prepare_sample_for_decode → next sample."""
        if val is None:
            self._drill_leaf_l0_retention_cfg = None
            logger.info("live_drill_leaf_l0_retention_cleared", reverting="l0_retention map")
            return
        if isinstance(val, (int, float)) and 0.0 < float(val) <= 1.0:
            self._drill_leaf_l0_retention_cfg = float(val)
            logger.info("live_drill_leaf_l0_retention", new=float(val))
            return
        logger.warning(
            "live_drill_leaf_l0_retention_invalid",
            value=val, expected="None or float in (0, 1]",
        )

    def _handle_drill_leaf_l1_retention(self, val) -> None:
        """Live retrieve-leaf drill L1 retention override. ``None`` clears
        (revert to the recursive ramp / sampled L1); a float in (0, 1] pins
        it. Read fresh in the per-repo drivers + _run_l1_batch → next encode."""
        if val is None:
            self._drill_leaf_l1_retention_cfg = None
            logger.info(
                "live_drill_leaf_l1_retention_cleared", reverting="recursive ramp/sampled",
            )
            return
        if isinstance(val, (int, float)) and 0.0 < float(val) <= 1.0:
            self._drill_leaf_l1_retention_cfg = float(val)
            logger.info("live_drill_leaf_l1_retention", new=float(val))
            return
        logger.warning(
            "live_drill_leaf_l1_retention_invalid",
            value=val, expected="None or float in (0, 1]",
        )

    def _handle_recursive_general_prompt(self, val) -> None:
        """Live general (query-agnostic) compression prompt for the shared repo
        tree. Read + re-embedded fresh each call in
        _recursive_general_prompt_emb → effective on the next repo encode."""
        if isinstance(val, str) and val.strip():
            old = getattr(self, "_recursive_general_prompt", "")
            self._recursive_general_prompt = val
            logger.info("live_recursive_general_prompt", old=old, new=val)
            return
        logger.warning(
            "live_recursive_general_prompt_invalid", value=val, expected="non-empty str",
        )

    # --- Memory / speed toggles -------------------------------------------
    def _handle_checkpoint_encoder(self, val) -> None:
        """Live toggle for encoder activation checkpointing. Read fresh via
        getattr in _checkpointed_level → effective on the next forward."""
        self._checkpoint_encoder = bool(val)
        logger.info("live_checkpoint_encoder", new=self._checkpoint_encoder)

    def _handle_profile_timing(self, val) -> None:
        """Live toggle for per-repo per-component timing (adds a synchronize()
        per repo when on). Read fresh in _forward_backward_per_repo → next
        repo."""
        self._profile_timing = bool(val)
        logger.info("live_profile_timing", new=self._profile_timing)

    def _handle_ablation_probe_steps(self, val) -> None:
        """Live control for the READ-ONLY ablation-gap probe horizon. Read
        fresh in _forward_backward_option_a against self.global_step → takes
        effect on the next per-repo step."""
        self._ablation_probe_steps = int(val or 0)
        logger.info("live_ablation_probe_steps", new=self._ablation_probe_steps)

    # --- Per-repo SIZE FILTERS (full effect on next dataloader rebuild) ----
    def _handle_max_repo_leaf_tokens(self, val) -> None:
        """Live per-repo leaf-token SIZE FILTER (``None``/<=0 = off; int = drop
        repos whose window-0 subtree exceeds this many L0 leaf tokens). The
        filter is applied at DATALOADER-BUILD time (repo grouping), so a live
        write updates the attribute but only takes full effect on the next
        dataloader rebuild / restart."""
        old = getattr(self, "_max_repo_leaf_tokens", None)
        if val is None:
            self._max_repo_leaf_tokens = None
        elif isinstance(val, (int, float)):
            k = int(val)
            self._max_repo_leaf_tokens = k if k > 0 else None
        else:
            logger.warning(
                "live_max_repo_leaf_tokens_invalid", value=val, expected="None or int",
            )
            return
        logger.info(
            "live_max_repo_leaf_tokens", old=old, new=self._max_repo_leaf_tokens,
            note="applied on next dataloader rebuild/restart",
        )

    def _handle_max_repo_file_samples(self, val) -> None:
        """Live per-repo file-sample-count SIZE FILTER (``None``/<=0 = off; int =
        drop repos with more than this many (file, target) samples). Applied at
        DATALOADER-BUILD time (repo grouping) → full effect on the next
        dataloader rebuild / restart."""
        old = getattr(self, "_max_repo_file_samples", None)
        if val is None:
            self._max_repo_file_samples = None
        elif isinstance(val, (int, float)):
            k = int(val)
            self._max_repo_file_samples = k if k > 0 else None
        else:
            logger.warning(
                "live_max_repo_file_samples_invalid", value=val, expected="None or int",
            )
            return
        logger.info(
            "live_max_repo_file_samples", old=old, new=self._max_repo_file_samples,
            note="applied on next dataloader rebuild/restart",
        )

    def _handle_max_repo_tree_nodes(self, val) -> None:
        """Live per-repo subtree NODE-COUNT SIZE FILTER (``None``/<=0 = off; int =
        drop repos whose window-0 subtree exceeds this many nodes). Applied at
        DATALOADER-BUILD time (repo grouping) → full effect on the next
        dataloader rebuild / restart."""
        old = getattr(self, "_max_repo_tree_nodes", None)
        if val is None:
            self._max_repo_tree_nodes = None
        elif isinstance(val, (int, float)):
            k = int(val)
            self._max_repo_tree_nodes = k if k > 0 else None
        else:
            logger.warning(
                "live_max_repo_tree_nodes_invalid", value=val, expected="None or int",
            )
            return
        logger.info(
            "live_max_repo_tree_nodes", old=old, new=self._max_repo_tree_nodes,
            note="applied on next dataloader rebuild/restart",
        )

    def _dynamic_ckpt_allowed_modes(self) -> frozenset[str]:
        """KRKB can only run its managed backbones in ``full`` mode.

        Every encoder level forward is wrapped in an OUTER non-reentrant
        ``torch.utils.checkpoint`` (:meth:`_checkpointed_level`; likewise the
        drill / tree-node checkpoints). ``megatron`` installs a selective
        per-op save/recompute policy INSIDE that wrapper, and the outer
        recompute then fails strict recompute-match (``CheckpointError
        58-vs-55`` on the first-ever downshift, lognav v1 step 49,
        2026-08-21). ``off`` is likewise unvalidated under the wrapper. Until
        those modes are made recompute-safe the scheduler must not flip into
        them; the adaptive cache-flush tier stays fully active.
        """
        return frozenset({"full"})

    def _dynamic_ckpt_managed_models(self) -> list[tuple[str, torch.nn.Module]]:
        """Register the per-repo run's backbones with the memory-driven dynamic
        ckpt scheduler so it can flip their gradient-checkpointing MODE (not
        just the adaptive cache-flush) as system memory varies.

        Without this override the base default returns ``[]`` → only the
        adaptive CUDA cache-flush fires and the GC mode-flip never engages.
        Registers L0 + L1 encoder backbones + EVERY decoder backbone (both
        families under round-robin, so whichever family is active per
        microbatch is always in the managed set). Mirrors
        ``CommitEncodingTrainer._dynamic_ckpt_managed_models``.
        """
        models: list[tuple[str, torch.nn.Module]] = []
        enc = getattr(self, "encoder", None)
        if enc is not None:
            l0 = getattr(enc, "l0", None)
            if l0 is not None and getattr(l0, "backbone", None) is not None:
                models.append(("encoder.l0", l0.backbone))
            l1 = getattr(enc, "l1", None)
            if l1 is not None and getattr(l1, "backbone", None) is not None:
                models.append(("encoder.l1", l1.backbone))
        decs = getattr(self, "_decoders_by_family", None)
        if decs:
            for family, dec in decs.items():
                bb = getattr(dec, "backbone", None)
                if bb is not None:
                    models.append((f"decoder_{family}", bb))
        elif getattr(self, "decoder", None) is not None:
            bb = getattr(self.decoder, "backbone", None)
            if bb is not None:
                models.append(("decoder", bb))
        return models

    def _recursive_general_prompt_emb(self) -> torch.Tensor:
        """L0-input-space embedding of the GENERAL (query-agnostic)
        compression prompt fed to both L0 and L1 of the shared repo tree."""
        embed_tokens = self.encoder.l0.backbone.get_input_embeddings()
        text = getattr(
            self, "_recursive_general_prompt", DEFAULT_RECURSIVE_GENERAL_PROMPT,
        )
        ids = self.encoder_tokenizer.encode(text, add_special_tokens=False) or [0]
        return embed_tokens(
            torch.tensor(ids, dtype=torch.long, device=self.device),
        )

    def _repo_leaf_diff_count(self, dataset: str, root_node_id: str) -> int:
        """Number of leaf diffs L0-encoded for a repo's window-0 subtree — the
        count :meth:`_compute_shared_repo_tree` iterates (every leaf article is
        an L0 forward). Uses the browse tree only (cheap)."""
        tree = self._trees.get(dataset)
        if tree is None or root_node_id not in tree:
            return 0
        return len(tree.articles(root_node_id))

    def _repo_tree_node_count(self, dataset: str, root_node_id: str) -> int:
        """Total node count of a repo's window-0 subtree (interior + leaf-tag
        nodes) — the RETAINED-GRAPH memory driver: the shared tree runs one
        L0/L1 forward per node and (current path: the final reaccumulate;
        inner-loop: across K steps) holds that graph live. Heavy-tailed
        (p50≈68, p98≈257, max≈673 on git_commit_repro), so it isolates the
        OOM tail that uniform leaf-token size does not. Browse-tree-only
        traversal (cheap, no token load)."""
        tree = self._trees.get(dataset)
        if tree is None or root_node_id not in tree:
            return 0
        # Visited-set: a browse tree SHOULD be acyclic, but an id-collision bug
        # (or any future data issue) can introduce a cycle, and a naive DFS then
        # loops forever (this exact hang cost a debugging session). Count distinct
        # reachable nodes and never revisit.
        seen: set[str] = set()
        stack = [root_node_id]
        while stack:
            nid = stack.pop()
            if nid in seen or nid not in tree:
                continue
            seen.add(nid)
            stack.extend(tree.get(nid).children)
        return len(seen)

    def _repo_leaf_token_count(self, dataset: str, root_node_id: str) -> int:
        """Total L0-encoded TOKEN count of a repo's window-0 leaf diffs — the
        true memory driver of :meth:`_compute_shared_repo_tree` (L0 backbone
        activations scale with total tokens, not leaf count). Sums per-article
        token lengths via the live token store's CSR offsets (no token load).
        Returns 0 when the token store is unavailable (cached-L0 stages)."""
        tree = self._trees.get(dataset)
        store = getattr(self, "_token_store", None)
        if tree is None or store is None or root_node_id not in tree:
            return 0
        doc_ids = self._article_ids_to_document_ids(
            dataset, list(tree.articles(root_node_id)),
        )
        total = 0
        for doc_id in doc_ids:
            try:
                total += int(store.length(dataset, doc_id))
            except (KeyError, TypeError):
                continue
        return total

    def _compute_shared_repo_tree(
        self, dataset: str, root_node_id: str,
    ) -> tuple[dict, dict]:
        """Encode the WHOLE window-0 subtree ONCE for a repo with the GENERAL
        prompt, returning ``(memo, stats)``.

        ``memo`` maps every encoded node id → ``(projected_rep, l1_output)``
        (live, gradient-carrying); the browse turns of every file-sample in
        the repo then SPLICE ``memo[node][0]`` instead of re-encoding. The L0
        leaves are encoded LIVE under the recursive L0 ramp (``_recursive_l0_
        override``), bridged L0→L1, and consolidated up the tree under the
        recursive L1 ramp — all backed by the per-layer gradient-checkpointed
        L0/L1 backbones.  The autograd graph is kept alive by the returned
        ``memo`` so the file-samples' backward passes can flow gradient into
        the shared tree (see :meth:`_forward_backward_per_repo`).
        """
        # Tripwire: the shared-tree forward must run EXACTLY ONCE per
        # _forward_backward_per_repo. Detach-and-reaccumulate replaced the old
        # Nx-per-backward recompute; this counter lets tests/asserts catch a
        # regression that re-runs the forward.
        self._shared_tree_forward_count = getattr(
            self, "_shared_tree_forward_count", 0,
        ) + 1
        memo: dict[str, tuple] = {}
        stats: dict[str, int] = {"nodes": 0}
        tree = self._trees.get(dataset)
        if tree is None or root_node_id not in tree:
            return memo, stats
        q_emb = self._recursive_general_prompt_emb()
        self._recursive_l0_override = self._recursive_l0_retention_now()
        # Only checkpoint the encode for LARGE trees. Two thresholds (OR):
        #   tree_checkpoint_min_nodes — node count (cheap browse-tree traversal).
        #   tree_checkpoint_min_tokens — total L0 leaf-diff tokens. LOAD-BEARING
        #     for the FULL (uncapped-L0) tree: a LOW-node repo can still hold a
        #     huge initial-import leaf (~75-90k tokens); the node gate alone
        #     would skip it and retain the unbounded L0 forward. The token gate
        #     forces the per-node checkpoint so BOTH the encode forward AND the
        #     retained R stay bounded to ~one node/chunk regardless of tree
        #     shape. Small trees (the majority) skip recompute → fast.
        n_nodes_tree = self._repo_tree_node_count(dataset, root_node_id)
        min_nodes = getattr(self, "_tree_checkpoint_min_nodes", 64)
        min_tokens = int(getattr(self, "_tree_checkpoint_min_tokens", 0) or 0)
        n_leaf_tokens = (
            self._repo_leaf_token_count(dataset, root_node_id)
            if min_tokens > 0 else 0
        )
        prev_ckpt_active = getattr(self, "_tree_encode_ckpt_active", True)
        self._tree_encode_ckpt_active = (
            n_nodes_tree > min_nodes
            or (min_tokens > 0 and n_leaf_tokens > min_tokens)
        )
        encode_checkpointed = (
            getattr(self, "_checkpoint_tree_encode", False)
            and self._tree_encode_ckpt_active
        )
        try:
            self._encode_subtree(dataset, tree, root_node_id, q_emb, memo, stats)
        finally:
            self._recursive_l0_override = None
            self._tree_encode_ckpt_active = prev_ckpt_active
        logger.info(
            "phase2_kb_per_repo_shared_tree_encoded",
            dataset=dataset,
            root=root_node_id,
            n_nodes=stats["nodes"],
            tree_node_count=n_nodes_tree,
            tree_leaf_tokens=n_leaf_tokens,
            encode_checkpointed=encode_checkpointed,
            tree_checkpoint_min_nodes=min_nodes,
            tree_checkpoint_min_tokens=min_tokens,
            l0_retention=self._recursive_l0_retention_now(),
            l1_retention=self._recursive_l1_retention_now(),
        )
        # LOUD guard: a non-empty tree that encodes to 0 nodes means the whole-tree
        # full-backprop objective (the point of this run) got NO gradient for this
        # repo. With the min-survivors floor above this should be ~0; if it fires
        # the floor regressed or a new collapse path appeared — surface, don't hide.
        if stats.get("nodes", 0) == 0 and n_nodes_tree > 0:
            logger.warning(
                "phase2_kb_shared_tree_collapsed",
                dataset=dataset,
                root=root_node_id,
                tree_node_count=n_nodes_tree,
                tree_leaf_tokens=n_leaf_tokens,
                l0_retention=self._recursive_l0_retention_now(),
                l1_retention=self._recursive_l1_retention_now(),
                msg="whole-tree encode produced 0 nodes despite a non-empty "
                    "tree; repo contributes NO whole-tree gradient this step",
            )
        return memo, stats

    def _truncate_segments_to_gold_budget(
        self,
        segments: list,
        n_gold: int,
        answer_span: tuple[int, int] | None = None,
    ) -> list:
        """Hard-cut only the final answer span to its first ``n_gold`` tokens.

        Tool-call tokens are supervised too, but they are navigation prefix,
        not the gold file output.  Counting them against the answer cap made the
        retained answer length depend on trajectory depth. ``answer_span`` is in
        assembled concat coordinates and the answer is the final turn.
        """
        if n_gold <= 0 or answer_span is None:
            return segments
        answer_start, answer_end = answer_span
        if answer_end - answer_start <= n_gold:
            return segments
        cut_abs = answer_start + n_gold
        out: list = []
        cursor = 0
        for seg in segments:
            if isinstance(seg, TokenSegment):
                length = int(seg.token_ids.shape[-1])
            else:
                length = int(seg.embeddings.shape[-2])
            if cursor >= cut_abs:
                break
            if cursor + length <= cut_abs:
                out.append(seg)
                cursor += length
                continue
            local_cut = cut_abs - cursor
            if isinstance(seg, TokenSegment):
                out.append(
                    TokenSegment(
                        token_ids=seg.token_ids[..., :local_cut],
                        loss_mask=(
                            None if seg.loss_mask is None
                            else seg.loss_mask[..., :local_cut]
                        ),
                    )
                )
            else:
                raise RuntimeError("answer span unexpectedly intersects an embedding splice")
            break
        return out

    def _encode_decode_group(
        self,
        samples: list,
        timing: dict | None = None,
        l1_target_ratio: float | None = None,
        span_ce_accum: dict | None = None,
    ) -> tuple[torch.Tensor, int, int, int, int]:
        """Shared encode→decode core: per-sample prep + BUCKETED (FA4-varlen
        packed) L1 drill encode across the group + per-sample assemble+decode,
        summing the per-sample decoder losses into ONE group-loss tensor. Does
        NOT backward — the caller backwards (so it can normalise + free).

        Reused by both the regular cross-sample :meth:`_forward_backward` (whole
        batch) and the per-repo PASS 2 (per group of G), so both share identical
        packing/decode machinery rather than forking new packing code.

        - ``l1_target_ratio``: drill L1 retention. ``None`` → ``_run_l1_batch``
          samples it (regular path). Per-repo passes the recursive L1 ramp so
          the drill shares the curriculum/θ with the shared tree.
        - ``timing``: optional dict accumulating per-component perf_counter
          splits (prep / drill_encode / assemble / decode_fwd).
        - ``span_ce_accum``: PROBE-ONLY. When a dict is supplied (only the
          read-only :meth:`_run_ablation_gap_probe` passes one), the per-sample
          decode requests hidden states (``return_hidden_states=True``) and the
          per-position CE is aggregated into the dict over the trajectory's
          navigation (bgkit drill-id tool-call) and reconstruction (gold answer)
          spans via :meth:`_accumulate_span_ce`. Training callers pass ``None``
          and get the scalar-only decode path — the return contract is
          UNCHANGED for them.

        Returns ``(group_loss, total_tokens, n_done, n_turns, n_buckets)``.
        """
        def _tm(key: str, gpu: bool = False):
            if timing is None:
                return contextlib.nullcontext()
            return self._timed(timing, key, gpu=gpu)

        # Phase 1: per-sample render + L1-input prep (per-sample drill L0).
        with _tm("prep", gpu=True):
            preps: list[dict] = [
                self._prepare_sample_for_decode(s) for s in samples
            ]

        # Phase 2: flatten + bucket the group's non-None drill turns by
        # power-of-2 content length, run ONE packed L1 forward per bucket.
        flat: list[tuple[int, int, dict | None]] = []
        for s_idx, prep in enumerate(preps):
            for t_idx, turn in enumerate(prep["prepared_turns"]):
                flat.append((s_idx, t_idx, turn))
        # Mode-tagged turns (pure drill-down: head live-encode / node shared-tree
        # lookup) are NOT bucketable — they carry no packed content buffer, so
        # they are resolved per-turn below. Only plain leaf-drill dicts (with a
        # "content" buffer) are bucketed by content length.
        buckets: dict[int, list[tuple[int, int, dict]]] = {}
        for s_idx, t_idx, turn in flat:
            if turn is None:
                continue
            if isinstance(turn, dict) and "mode" in turn:
                continue
            n_content = int(turn["content"].size(0))
            bucket_key = max(0, (max(n_content, 1) - 1).bit_length())
            buckets.setdefault(bucket_key, []).append((s_idx, t_idx, turn))

        survivors_by_address: dict[tuple[int, int], torch.Tensor] = {}
        with _tm("drill_encode", gpu=True):
            for _bucket_key, items in buckets.items():
                bucket_out = self._run_l1_batch(
                    [p for _, _, p in items], target_ratio=l1_target_ratio,
                )
                for (s_idx, t_idx, _p), surv in zip(items, bucket_out, strict=True):
                    survivors_by_address[(s_idx, t_idx)] = surv
            # Resolve mode-tagged drill-down turns (head/node) per-turn.
            for s_idx, t_idx, turn in flat:
                if isinstance(turn, dict) and "mode" in turn:
                    survivors_by_address[(s_idx, t_idx)] = (
                        self._resolve_special_survivor(turn)
                    )
        # Zero fallback for None turns (computed lazily — only touch the encoder
        # when a None turn actually exists, so fully-stubbed tests don't need it).
        _zero_fallback: torch.Tensor | None = None
        for s_idx, t_idx, turn in flat:
            if turn is None:
                if _zero_fallback is None:
                    _zero_fallback = torch.zeros(
                        (1, self.encoder.active_projection_output_dim),
                        device=self.device, dtype=torch.bfloat16,
                    )
                survivors_by_address[(s_idx, t_idx)] = _zero_fallback

        # Phase 3: per-sample assemble + decode, summed into ONE group loss.
        group_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        total_tokens = 0
        n_done = 0
        n_turns = 0
        decode_cap = int(getattr(self, "_max_decode_tokens", 0) or 0)
        # Phase 3a: assemble every sample's segments (drop empties / zero-loss).
        batch_segments: list = []
        batch_meta: list = []  # (s_idx, trace, sample_tokens, n_prep_turns, decode_len)
        for s_idx, prep in enumerate(preps):
            per_turn = [
                survivors_by_address[(s_idx, t_idx)]
                for t_idx in range(len(prep["prepared_turns"]))
            ]
            with _tm("assemble"):
                segments, _trace = self._assemble_sample_segments(prep, per_turn)
            if not segments:
                continue
            # SINGLE-FORWARD BOUND: TRUNCATE the gold OUTPUT to the first
            # ``decode_cap`` tokens (hard cut — drop the tail + end-token + any
            # suffix), KEEPING the sample. The first portion of a file already
            # exercises extracting info from the tree, so reconstructing the
            # first-N gold is a valid target — and it bounds the decode seqlen
            # (≈ prefix + survivors + N) without dropping long files (no
            # short-file bias). The tree/input is NOT truncated. No-op when 0.
            if decode_cap > 0:
                segments = self._truncate_segments_to_gold_budget(
                    segments, decode_cap, _trace.answer_span,
                )
            sample_tokens = 0
            decode_len = 0
            for seg in segments:
                if isinstance(seg, TokenSegment) and seg.loss_mask is not None:
                    sample_tokens += int(seg.loss_mask.sum().item())
                tk = getattr(seg, "token_ids", None)
                if tk is not None:
                    decode_len += int(tk.reshape(-1).shape[0])
                else:
                    emb = getattr(seg, "embeddings", None)
                    if emb is not None:
                        decode_len += int(emb.reshape(-1, emb.shape[-1]).shape[0])
            if sample_tokens == 0:
                continue
            batch_segments.append(segments)
            batch_meta.append(
                (s_idx, _trace, sample_tokens, len(prep["prepared_turns"]), decode_len),
            )

        # Phase 3b: decode the group.
        #
        # G==1 (the default live-run path, per_repo_sample_group_size=1) uses the
        # proven per-file forward — NO packing, zero cross-sample risk.
        #
        # G>1 packs the whole group into ONE FA4-varlen forward (the batching
        # speedup) and computes per-file CE on each file's isolated hidden
        # slice. Cross-sample isolation is provided by
        # ``forward_interleaved_packed``: attention rides on cu_seqlens; the
        # stateful mixers reset via per-token seq_idx (Qwen3.5 DeltaNet
        # delta-rule + short-conv; Falcon-H1 Mamba conv + chunked scan, with
        # Falcon sample starts chunk-aligned because mamba_ssm's seq_idx
        # reset is exact only at chunk boundaries). Certified 2026-07-23 by
        # scripts/test_decode_batching_parity.py ("ALL PARITY CHECKS PASS"):
        # exact cross-grad + content-swap isolation probes at zero for both
        # families, eval+train, plus noise-calibrated per-file loss/grad
        # agreement vs the sequential reference. Re-run that gate before
        # trusting any change to the packed decode path.
        if batch_segments:
            want_hidden = span_ce_accum is not None
            with _tm("decode_fwd", gpu=True):
                if len(batch_segments) == 1:
                    outs = [
                        self.decoder.forward_interleaved_with_loss(
                            batch_segments[0], return_hidden_states=want_hidden,
                        )
                    ]
                else:
                    outs = self.decoder.forward_interleaved_packed(
                        batch_segments, return_hidden_states=want_hidden,
                    )
            for (_s_idx, _trace, sample_tokens, n_prep, decode_len), out in zip(
                batch_meta, outs, strict=True,
            ):
                if want_hidden:
                    sample_loss = out.loss
                    self._accumulate_span_ce(span_ce_accum, out, _trace)
                else:
                    sample_loss = out
                # per-file decode breakdown (decoder family + concat seqlen).
                if _MEM_BREAKDOWN:
                    self._mem_breakdown(
                        "after_decode_fwd",
                        decoder_family=getattr(self, "_decoder_family", None),
                        decode_seqlen=decode_len,
                        loss_tokens=sample_tokens,
                        n_drills=n_prep,
                    )
                group_loss = group_loss + sample_loss
                total_tokens += sample_tokens
                n_done += 1
                n_turns += n_prep

        # Projection-output NORM-BAND regularizer (2026-07-31 collapse fix).
        # FOLDED into the group loss so it backwards WITH the reconstruction
        # gradient through projection_blocks into the backbone (the callers do a
        # single per-group backward, freeing the drill graph — a deferred aux
        # backward would hit freed graphs). Computed over the group's spliced,
        # grad-carrying survivor reps (leaf-drill + drill-node + head; the
        # zero-fallback splices carry no grad and are skipped). No-op unless the
        # git-repro config enabled it. Scale note: with the run's default
        # per_repo_sample_group_size=1 each group is ONE sample, so summing
        # weight*penalty_group over the n_contrib groups and /n_contrib in the
        # Option-A backward yields exactly weight*mean_penalty (calibrated for
        # G=1; G>1 mildly under-weights — documented).
        # NB: check the (default-False) enable flag FIRST so a stub trainer
        # without a real ``self.encoder`` (unit tests) never touches it — and the
        # _projection_norm_reg_term re-checks self.encoder.training itself.
        if (
            getattr(self, "_proj_norm_reg_enabled", False)
            and n_done > 0
            and getattr(getattr(self, "encoder", None), "training", False)
        ):
            group_loss = group_loss + self._projection_norm_reg_term(
                list(survivors_by_address.values()),
            )
        return group_loss, total_tokens, n_done, n_turns, len(buckets)

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
        encoder call (packed varlen across turns — no padding tokens).
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

    @torch.no_grad()
    def generate_kb_turn(
        self,
        sample: KBSample,
        history: list,
        *,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate the next assistant turn from observed calls only.

        ``history`` contains canonical tool calls that were actually accepted
        and executed by the free-running evaluator. No future teacher call or
        gold answer is rendered into the context.
        """
        self._ensure_eval_shared_tree(sample)
        probe = replace(sample, trajectory=list(history))
        segments, _trace = self._build_decoder_segments_core(probe)
        topic_tags = (
            self._sample_tags_for(probe)
            if self.topic_embeddings is not None
            else None
        )
        prompt_ids = assistant_generation_prompt_ids(
            self.tokenizer,
            self._system_prompt_for(probe),
            probe.question,
            list(history),
            topic_knowledge_tags=topic_tags,
        ).to(self.device)
        if prompt_ids.numel() == 0:
            raise RuntimeError("chat template produced an empty generation prompt")
        segments.append(TokenSegment(token_ids=prompt_ids, loss=False))
        generated = self.decoder.generate_with_segments(
            segments,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        text = generated.content_text[0] if generated.content_text else ""
        # The decoder is trained to emit the template's end-of-turn glue
        # before the stop token (Falcon-H1: a newline); return content only.
        glue = assistant_turn_end_glue(
            self.tokenizer,
            [
                {"role": "system", "content": self._system_prompt_for(probe)},
                {"role": "user", "content": probe.question},
            ],
            [BGKIT_TOOL, BGKIT_TOPIC_KNOWLEDGE_TOOL],
        )
        if glue and text.endswith(glue):
            text = text[: -len(glue)]
        return text

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

    def _render_sample(self, sample: KBSample):
        """Render a sample's trajectory to ``(rendered, topic_block,
        topic_tags)`` — tokenization ONLY, no encoder/drill work and no
        side-effects (no θ accumulation, no ``_pending`` appends, no autograd
        graph). Shared by :meth:`_prepare_sample_for_decode` and the cheap
        per-repo contributing-count pass so the render logic stays single-source.
        """
        system_prompt = self._system_prompt_for(sample)
        # Build the topic embedding block up-front so we know whether to ask the
        # tokenizer to inject a topic-knowledge tool-call pair.
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
        return rendered, topic_block, topic_tags

    def _sample_contributing_token_count(self, sample: KBSample) -> int:
        """Cheap (render-only, NO encode, NO graph, NO side-effects) count of a
        sample's trainable loss-tokens. A file-sample contributes iff its
        rendered ``loss_mask`` has ≥1 set position (the gold ANSWER turn, plus
        any loss=True tool regions; browse/bgkit/topic sentinels are
        loss-masked-off so they don't count). Used by
        :meth:`_forward_backward_per_repo`'s count-then-stream Pass 1 to get the
        normaliser WITHOUT staging N encode graphs."""
        rendered, _tb, _tt = self._render_sample(sample)
        return int(rendered.loss_mask.sum().item())

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
        rendered, topic_block, _topic_tags = self._render_sample(sample)
        token_ids = rendered.token_ids.to(self.device)
        loss_mask = rendered.loss_mask.to(self.device)

        sentinel_ids = self.tokenizer.encode(BGKIT_SENTINEL, add_special_tokens=False)
        sentinel_len = len(sentinel_ids)
        topic_sentinel_ids = self.tokenizer.encode(
            BGKIT_TOPIC_SENTINEL, add_special_tokens=False,
        )
        topic_sentinel_len = len(topic_sentinel_ids)

        # Per-bgkit-turn CLASSIFIER (pure recursive drill-down, git_commit_repro).
        # Three kinds, ADDITIVE / GUARDED so browse-based datasets (KILT /
        # PubMedQA) and ordinary article drills are completely unaffected:
        #   - is_head=True             -> tag {"mode":"head"} : LIVE task-query
        #     recursive-L1 over the window node's shared-tree children.
        #   - query=="" and ids is a single REAL tree node
        #                              -> tag {"mode":"node"} : present that
        #     node's GENERIC shared-tree rep (memo/splice lookup, NO live encode;
        #     also the distractor-drill path, loss handled by loss_mask).
        #   - otherwise                -> _prepare_l1_turn(...) : leaf drill
        #     (retrieve a specific diff/article), UNCHANGED (incl. None handling).
        trees = getattr(self, "_trees", None) or {}
        dataset_tree = trees.get(sample.dataset_name)
        prepared_turns: list[dict | None] = []
        # Per-sample task query (carried by the head turn). Recursive full-backprop
        # leaf drills are compressed UNDER this query instead of query-agnostically
        # (the leaf's own ``query`` field is "" in the git-repro trajectories) so
        # the retrieved diff keeps the positions relevant to THIS reconstruction
        # (2026-07-30 recon fix). Non-git-repro datasets have no is_head turn, so
        # task_query stays "" and their leaves are unaffected.
        task_query = ""
        for turn in rendered.bgkit_turns:
            if bool(turn.args.get("is_head", False)):
                task_query = str(turn.args.get("query", ""))
                break
        # Flat retrieval leaves follow the configured L0 selection mode
        # (``training.selection_mode.l0``); the recursive full-backprop run
        # forces exact_topk for its tree leaves as before.
        leaf_l0_mode = (
            "exact_topk"
            if (
                getattr(self, "_recursive_l1_full_backprop", False)
                or getattr(self, "_selection_mode_l0", "threshold") == "exact_topk"
            )
            else "threshold"
        )
        # QUERY-CONDITIONED drill nodes (2026-07-31): tag EVERY node-mode turn
        # (on-path interiors AND wrong-sibling distractors — content-driven
        # rejection needs the distractor encoded under the same query) with the
        # per-sample task query so _resolve_special_survivor re-encodes it LIVE
        # via the generalized head machinery. Flag OFF (or no task query — non
        # git-repro datasets have no is_head turn) → the dict is EXACTLY the
        # legacy static-splice form.
        qc_drill = (
            bool(getattr(self, "_query_conditioned_drill_nodes", False))
            and task_query != ""
        )
        # Leaf L0 retention override (drill_leaf_retention.l0) — threaded
        # CONDITIONALLY so legacy-signature test doubles keep working when the
        # knob is unset.
        leaf_kwargs: dict = {"l0_selection_mode": leaf_l0_mode}
        _leaf_l0 = self._drill_leaf_l0_retention_now()
        if _leaf_l0 is not None:
            leaf_kwargs["l0_ratio"] = float(_leaf_l0)
        # CONTRACT: a ``head`` drill resolves its survivors from head-drill
        # infrastructure ONLY — the per-repo shared tree (git_commit_repro
        # full-backprop) or the offline L1-tree cache. Without either, every
        # head turn falls through to ``_drilldown_zero_survivor`` and the
        # decoder silently trains on ZERO reps while the encoder never runs
        # (this is exactly what happened to the 2026-08 widenet v1→v4 runs:
        # the flat writers tagged their single retrieval turn ``is_head``).
        # Fail loudly at sample-prep time instead of producing a fake run.
        head_infra = bool(getattr(self, "_per_repo_full_backprop", False)) or (
            getattr(self, "_l1_tree_cache", None) is not None
        )
        for turn in rendered.bgkit_turns:
            ids = list(turn.args.get("ids", []))
            query = str(turn.args.get("query", ""))
            is_head = bool(turn.args.get("is_head", False))
            if is_head and not head_infra:
                raise ValueError(
                    "bgkit turn tagged is_head=True for dataset "
                    f"{sample.dataset_name!r} (ids={ids[:2]}) but this trainer has "
                    "no head-drill infrastructure (no per-repo shared tree, no "
                    "L1-tree cache): the head drill would resolve to a ZERO "
                    "survivor on every sample. Flat single-retrieval datasets must "
                    "emit plain leaf drills {ids, query} (see "
                    "bgkit.data.flat_phase2_writer.flat_trajectory_row)."
                )
            if is_head:
                prepared_turns.append({
                    "mode": "head",
                    "node_id": str(ids[0]) if ids else "",
                    "query": query,
                    "dataset": sample.dataset_name,
                })
            elif (
                query == ""
                and len(ids) == 1
                and dataset_tree is not None
                and str(ids[0]) in dataset_tree
            ):
                node_entry = {
                    "mode": "node",
                    "node_id": str(ids[0]),
                    "dataset": sample.dataset_name,
                }
                if qc_drill:
                    node_entry["query"] = task_query
                prepared_turns.append(node_entry)
            else:
                # Leaf (retrieve) drill: fall back to the per-sample task query
                # when the turn carries none (git-repro leaves), and force
                # exact_topk L0 in the recursive full-backprop run.
                leaf_query = query or task_query
                # v5 gold span threaded CONDITIONALLY (legacy test doubles).
                _gs = getattr(sample, "gold_span", None)
                _gs_kwargs = {"gold_span": _gs} if _gs is not None else {}
                prepared_turns.append(
                    self._prepare_l1_turn(
                        sample.dataset_name, ids, leaf_query, **leaf_kwargs, **_gs_kwargs,
                    ),
                )

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
        sentinel_len_by_kind = {
            "topic": topic_sentinel_len,
            "bgkit": sentinel_len,
        }
        for start, kind, payload in splice_events:
            sentinel_tok_len = sentinel_len_by_kind[kind]
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
        bgkit_spans_concat = [_remap(span) for span in rendered.bgkit_call_spans]

        trace = _KBDecodeTrace(
            answer_span=answer_span_concat,
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
        # Aux OFF: no survivorship gradient at all (head stays frozen at its
        # pre-trained selection). _pending is empty in this mode anyway.
        if not getattr(self, "_survivorship_aux", True):
            return torch.zeros((), device=self.device, dtype=torch.float32), {}

        metrics: dict[str, float] = {}
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        # Per-term gradient attribution at the selector heads. Armed by
        # survivorship.grad_attribution_every (0 = off). See _attribute_terms.
        attrib: list[tuple[str, torch.Tensor]] = []
        # getattr: tests construct the trainer bypassing __init__.
        _attrib_every = int(getattr(self, "_aux_grad_attribution_every", 0) or 0)

        def _keep(name: str, term: torch.Tensor) -> torch.Tensor:
            if _attrib_every > 0 and term.requires_grad:
                attrib.append((name, term))
            return term

        def _head_scale_anchor(level: str, entry: dict):
            """Differentiable log-std anchor on ``base_raw`` for one entry.

            Returns None when the level does not use it. See
            LevelLossCfg.head_scale_anchor_weight for why this is mandatory
            under exact_topk: the z-score is scale-invariant, so without it the
            head's output scale is a gradient-free direction and Adam random-
            walks along it (l1_base_raw_std 129 -> 8431 in 30 steps).
            """
            cfg = getattr(self, f"_surv_{level}", None)
            w = float(getattr(cfg, "head_scale_anchor_weight", 0.0) or 0.0)
            if w <= 0.0:
                return None
            enc_out = entry.get("enc_out")
            raw = getattr(enc_out, "base_raw", None)
            if raw is None:
                raw = getattr(enc_out, "base_raw_for_util", None)
            cu = entry.get("cu_seqlens")
            if raw is None or cu is None or not raw.requires_grad:
                return None
            raw = raw.reshape(-1).float()
            n, n_seg = int(raw.shape[0]), int(cu.shape[0]) - 1
            if n <= 1 or n_seg <= 0:
                return None
            seg = segment_ids_from_cu(cu, n)
            mean = segment_mean(raw, seg, n_seg)
            centered = raw - mean[seg]
            var = segment_mean(centered * centered, seg, n_seg)
            std = torch.sqrt(var + 1e-6)
            target = float(getattr(cfg, "head_scale_anchor_target_std", 2.0) or 2.0)
            # log-space so the penalty is symmetric in over/under-shoot and
            # cannot explode when std is already far from target.
            return w * ((torch.log(std) - math.log(max(target, 1e-6))) ** 2).mean()

        def _record_head_spread(level: str, entry: dict) -> None:
            """Log the per-document std of the RAW head score.

            Under ``exact_topk`` the operator score is ``_segment_zscore`` =
            ``(raw - mean_doc) / sqrt(var_doc + 1e-6)``, so this std is the
            divisor — and d(score)/d(raw) scales as 1/std. A head whose output
            goes near-constant within a document therefore does not merely stop
            discriminating, it AMPLIFIES its own gradient without bound (the
            epsilon floors std at 1e-3, i.e. up to 1000x). That is the
            suspected mechanism behind grad_norm/l1_head running at ~6.7e3
            while every other group sits at O(1) (2026-08-27). Nothing logged
            this quantity, so the pathology was only visible as a grad-norm
            anomaly with no cause attached.
            """
            enc_out = entry.get("enc_out")
            raw = getattr(enc_out, "base_raw", None)
            if raw is None:
                raw = getattr(enc_out, "base_raw_for_util", None)
            cu = entry.get("cu_seqlens")
            if raw is None or cu is None:
                return
            raw = raw.detach().reshape(-1).float()
            n = int(raw.shape[0])
            n_seg = int(cu.shape[0]) - 1
            if n <= 0 or n_seg <= 0:
                return
            seg = segment_ids_from_cu(cu, n)
            mean = segment_mean(raw, seg, n_seg)
            centered = raw - mean[seg]
            var = segment_mean(centered * centered, seg, n_seg)
            std = torch.sqrt(var + 1e-6)
            metrics[f"{level}_base_raw_std"] = float(std.mean().item())
            metrics[f"{level}_base_raw_std_min"] = float(std.min().item())
            metrics[f"{level}_zscore_grad_gain"] = float((1.0 / std).mean().item())

        # ----------------------------------------------------------------
        # L0: ratio + decisiveness
        # Each entry is one _l0_for_articles call.  Packed: logits_for_op
        # is flat (N_l0,) with cu_seqlens marking per-article segments.
        # ----------------------------------------------------------------
        l0_ratio_losses: list[torch.Tensor] = []
        l0_decisive_losses: list[torch.Tensor] = []
        l0_span_losses: list[torch.Tensor] = []
        l0_scale_anchors: list[torch.Tensor] = []
        l0_span_survival: list[float] = []
        for entry in self._pending_l0_outputs:
            enc_out = entry["enc_out"]
            _record_head_spread("l0", entry)
            _anch = _head_scale_anchor("l0", entry)
            if _anch is not None:
                l0_scale_anchors.append(_anch)
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
            # v5 span-level relevance at L0: answer-span positions -> survive.
            _span = entry.get("span_mask")
            if (
                getattr(self, "_span_relevance_weight", 0.0) > 0.0
                and _span is not None
                and _span.numel() == probs_f.numel()
                and bool(_span.any())
            ):
                l0_span_losses.append(((1.0 - probs_f[_span]) ** 2).mean())
                _sm = getattr(enc_out, "survivor_mask", None)
                if _sm is not None and _sm.numel() == _span.numel():
                    l0_span_survival.append(_sm[_span].float().mean().item())
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
                + self._aux_weight("l0", "ratio_loss_weight") * l0_ratio_loss
                + self._aux_weight("l0", "decisiveness_loss_weight") * l0_decisive_loss
            )
            metrics["l0_ratio_loss"] = l0_ratio_loss.item()
            metrics["l0_decisiveness_loss"] = l0_decisive_loss.item()
        if l0_span_losses:
            l0_span_loss = torch.stack(l0_span_losses).mean()
            total = total + getattr(self, "_span_relevance_weight", 0.0) * l0_span_loss
            metrics["l0_span_loss"] = l0_span_loss.item()
            if l0_span_survival:
                metrics["l0_span_survival"] = sum(l0_span_survival) / len(l0_span_survival)

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
        l1_span_losses: list[torch.Tensor] = []
        l1_scale_anchors: list[torch.Tensor] = []
        l1_span_survival: list[float] = []
        for entry in self._pending_l1_outputs:
            enc_out = entry["enc_out"]
            _record_head_spread("l1", entry)
            _anch = _head_scale_anchor("l1", entry)
            if _anch is not None:
                l1_scale_anchors.append(_anch)
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
            pinned = entry.get("pinned")
            controllable = (
                ~pinned.reshape(-1)
                if pinned is not None
                else torch.ones_like(probs_f, dtype=torch.bool)
            )

            # Packed: all positions are valid.  Segment-aware per-article
            # reductions for ratio and decisiveness. Pinned ID positions are
            # forced survivors, not decisions made by the head, and therefore
            # stay outside every selector auxiliary.
            cu_seqlens = entry.get("cu_seqlens")
            if cu_seqlens is not None:
                n_arts = int(cu_seqlens.shape[0]) - 1
                seg_ids = segment_ids_from_cu(cu_seqlens, int(probs_f.shape[0]))
                ctrl_f = controllable.to(probs_f.dtype)
                ctrl_counts = segment_sum(ctrl_f, seg_ids, n_arts)
                valid_segments = ctrl_counts > 0
                if not valid_segments.any():
                    continue
                mean_probs = segment_sum(
                    probs_f * ctrl_f, seg_ids, n_arts,
                ) / ctrl_counts.clamp(min=1)
                l1_ratio_losses.append(
                    ((mean_probs[valid_segments] - target_ratio) ** 2).mean()
                )
                decisive = segment_sum(
                    4.0 * probs_f * (1.0 - probs_f) * ctrl_f,
                    seg_ids,
                    n_arts,
                ) / ctrl_counts.clamp(min=1)
                l1_decisive_losses.append(decisive[valid_segments].mean())
            else:
                # Single-segment fallback.
                selected_probs = probs_f[controllable]
                if selected_probs.numel() == 0:
                    continue
                l1_ratio_losses.append((selected_probs.mean() - target_ratio) ** 2)
                l1_decisive_losses.append(
                    (4.0 * selected_probs * (1.0 - selected_probs)).mean()
                )

            # Relevance loss: two per-group aggregate-ratio targets.
            # - Gold controllable content should survive at approximately
            #   gold_boost * target_ratio.
            # - Distractor controllable content should survive at
            #   distractor_damp * target_ratio
            #   (downsample but don't suppress to zero — distractor IDs may
            #   still be referenced in later bgkit calls).
            # relevance is flat (N_l1,) bool — unchanged if already flat.
            _span_l1 = entry.get("span_mask")
            if (
                getattr(self, "_span_relevance_weight", 0.0) > 0.0
                and _span_l1 is not None
                and _span_l1.numel() == probs_f.numel()
                and bool(_span_l1.any())
            ):
                l1_span_losses.append(((1.0 - probs_f[_span_l1.reshape(-1)]) ** 2).mean())
                _sm1 = getattr(enc_out, "survivor_mask", None)
                if _sm1 is not None and _sm1.numel() == _span_l1.numel():
                    l1_span_survival.append(_sm1[_span_l1.reshape(-1)].float().mean().item())
            if relevance is not None:
                relevance_flat = relevance.reshape(-1) & controllable
                gold_target = min(1.0, target_ratio * self._relevance_gold_boost)
                distractor_target = max(0.0, target_ratio * self._relevance_distractor_damp)
                per_group_losses: list[torch.Tensor] = []
                if relevance_flat.any():
                    gold_mean = probs_f[relevance_flat].mean()
                    per_group_losses.append((gold_mean - gold_target) ** 2)
                distractor_mask = ~relevance.reshape(-1) & controllable
                if distractor_mask.any():
                    distractor_mean = probs_f[distractor_mask].mean()
                    per_group_losses.append((distractor_mean - distractor_target) ** 2)
                if per_group_losses:
                    l1_relevance_losses.append(torch.stack(per_group_losses).mean())

        if l1_span_losses:
            l1_span_loss = torch.stack(l1_span_losses).mean()
            total = total + _keep(
                "l1_span", getattr(self, "_span_relevance_weight", 0.0) * l1_span_loss,
            )
            metrics["l1_span_loss"] = l1_span_loss.item()
            if l1_span_survival:
                metrics["l1_span_survival"] = sum(l1_span_survival) / len(l1_span_survival)
        if l1_ratio_losses:
            l1_ratio_loss = torch.stack(l1_ratio_losses).mean()
            l1_decisive_loss = torch.stack(l1_decisive_losses).mean()
            total = (
                total
                + _keep("l1_ratio", self._aux_weight("l1", "ratio_loss_weight") * l1_ratio_loss)
                + _keep(
                    "l1_decisiveness",
                    self._aux_weight("l1", "decisiveness_loss_weight") * l1_decisive_loss,
                )
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
                pinned = entry.get("pinned")
                controllable = (
                    ~pinned.reshape(-1)
                    if pinned is not None
                    else torch.ones_like(soft_gates, dtype=torch.bool)
                )
                ctrl_f = controllable.to(soft_gates.dtype)
                cu_seqlens = entry.get("cu_seqlens")
                if cu_seqlens is not None:
                    n_arts = int(cu_seqlens.shape[0]) - 1
                    seg_ids = segment_ids_from_cu(cu_seqlens, int(soft_gates.shape[0]))
                    soft_count = segment_sum(
                        soft_gates * ctrl_f, seg_ids, n_arts,
                    )
                    content_len = segment_sum(ctrl_f, seg_ids, n_arts)
                    if content_len.sum() == 0:
                        continue
                else:
                    soft_count = (soft_gates * ctrl_f).sum().unsqueeze(0)
                    content_len = torch.tensor(
                        [float(controllable.sum().item())],
                        dtype=soft_count.dtype, device=soft_count.device,
                    )
                    if content_len[0] == 0:
                        continue
                target_min = torch.clamp(
                    torch.ceil(content_len * self._surv_l1.min_survivors_floor_ratio),
                    min=float(self._surv_l1.min_survivors_absolute_min),
                )
                deficit = (1.0 - soft_count / target_min.clamp(min=1.0)).clamp(min=0.0)
                l1_min_surv_losses.append(
                    (deficit[content_len > 0] ** 2).mean()
                )
            if l1_min_surv_losses:
                l1_min_surv_loss = torch.stack(l1_min_surv_losses).mean()
                total = total + _keep(
                    "l1_min_survivors",
                    self._surv_l1.min_survivors_loss_weight * l1_min_surv_loss,
                )
                metrics["l1_min_survivors_loss"] = l1_min_surv_loss.item()

        if l1_relevance_losses:
            l1_relevance_loss = torch.stack(l1_relevance_losses).mean()
            total = total + _keep(
                "l1_relevance", self._relevance_loss_weight * l1_relevance_loss,
            )
            metrics["l1_relevance_loss"] = l1_relevance_loss.item()

        for _lvl, _terms in (("l0", l0_scale_anchors), ("l1", l1_scale_anchors)):
            if _terms:
                _t = torch.stack(_terms).mean()
                total = total + _t
                metrics[f"{_lvl}_head_scale_anchor_loss"] = float(_t.item())

        if attrib and _attrib_every > 0:
            step = int(getattr(self, "global_step", 0) or 0)
            if step % _attrib_every == 0:
                metrics.update(self._attribute_head_gradient(attrib))

        return total, metrics

    def _attribute_head_gradient(
        self, terms: list[tuple[str, torch.Tensor]],
    ) -> dict[str, float]:
        """Per-term gradient norm at the L1 selector head.

        ``grad_norm/l1_head`` reports the SUM over every auxiliary, so a head
        running at a median 6672 while l1_backbone sits at 0.40 tells you the
        head is being shouted at without telling you BY WHAT. This splits it:
        one ``gattr/<term>`` per weighted loss, so a term that dominates the
        span signal by orders of magnitude is visible directly.

        LIMITATION: the utility-grad BCE runs its own backward in
        ``_apply_utility_grad_bce_phase2`` and is NOT one of these terms, so
        the reported norms need not sum to grad_norm/l1_head. A large residual
        between the two is itself informative — it points at utility-grad.
        """
        params = [p for p in self.encoder.l1.head.parameters() if p.requires_grad]
        out: dict[str, float] = {}
        if not params:
            return out
        for name, term in terms:
            try:
                grads = torch.autograd.grad(
                    term, params, retain_graph=True, allow_unused=True,
                )
            except RuntimeError:
                # A term detached from the head contributes nothing; skip it
                # rather than aborting the whole attribution pass.
                continue
            sq = torch.zeros((), device=self.device, dtype=torch.float32)
            for g in grads:
                if g is not None:
                    sq = sq + (g.float() ** 2).sum()
            out[f"gattr/{name}"] = float(torch.sqrt(sq).item())
        return out

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
        # Aux OFF: no utility-grad BCE distillation (head frozen); nothing was
        # retained in _pending.
        if not getattr(self, "_survivorship_aux", True):
            return {}

        from bgkit.training.survivorship_helpers import (
            LevelLossCfg,
            utility_grad_bce_loss,
        )

        metrics: dict[str, float] = {}
        w_l0 = getattr(self, "_surv_l0", LevelLossCfg()).utility_grad_loss_weight
        w_l1 = getattr(self, "_surv_l1", LevelLossCfg()).utility_grad_loss_weight

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
                    # exact_topk's operator score IS the per-document z-score,
                    # so the BCE must run on it too or this term alone keeps a
                    # scale degree of freedom and inflates the head (2026-08-27).
                    standardize=(
                        getattr(self, "_selection_mode_l0", "threshold") == "exact_topk"
                    ),
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
                    # See the L0 site. L1 is where this actually bit: raw std
                    # 119 vs L0's 2.17, because L0 is anchored by
                    # moment_match_weight 0.05 and L1 sets it to 0.0.
                    standardize=(
                        getattr(self, "_selection_mode_l1", "threshold") == "exact_topk"
                    ),
                )
                if util_loss.requires_grad:
                    (util_loss * w_l1 / self._accum_steps).backward()
                grad_norms.append(float(content_grad.norm().item()))
                entry["enc_out"] = None
            if grad_norms:
                metrics["l1_content_grad_norm"] = sum(grad_norms) / len(grad_norms)

        return metrics

    def _freeze_decoder_embeddings(self) -> None:
        """Freeze each decoder's token embedding (and, when tied, its LM head).

        DEFAULT OFF — measured mildly HARMFUL. Two matched 700-step runs from
        the same base with the per-group-LR fix (26fc012) in place:

            freeze OFF   PPL 31.2 at steps 500/600/650/699 (base 30.4)
            freeze ON    PPL 41.8 / 41.2 / 41.1 at the same steps

        Freezing costs ~10 perplexity for no protection: the LR fix ALONE
        holds the decoder at the base's level through the window where the
        historical run reached 2425. Pinning the embedding evidently denies
        the decoder small beneficial adaptations. Enable it only with a
        specific reason and measure the cost.

        It was added on 2026-08-25 as the supposed cure for the collapse
        (decoder plain-text PPL 31 at the summarization base -> 671 at
        wide-net v6 -> 2585 at v7). That diagnosis was WRONG and the measured
        refutation is worth keeping:

        - The embedding is essentially IDENTICAL in the healthy base and the
          destroyed checkpoints — norm/pristine 0.798 / 0.797 / 0.797 and
          row-wise cosine 0.557 / 0.561 / 0.561 for base / v6 / v7. A model
          carrying that exact embedding scores PPL 31. The cosine-0.56
          rotation happened during PHASE 1 and is benign.
        - v6 -> v7 moved it by nothing (cosine 0.5605 -> 0.5606) while
          perplexity went 671 -> 2585.

        So freezing it prevents a drift that was not happening. It is kept
        because a 248,320-row tied matrix reshaped by ~107 loss-bearing
        tokens per sample is worth pinning on general principle, and because
        Phase 1 treats the decoder embedding as a fixed ANCHOR for the
        projection (Step 2.5, "projection embed-anchor repair") — but do NOT
        expect it to restore language health. The live hypothesis for that is
        the rep-scale operating point (wide-net splices reps at ~218x the
        embedding norm against git-repro's ~4x; git-repro moved the backbone
        FURTHER over 8700 steps and stayed at PPL 64), which
        ``eval/lm_health/*`` now measures every eval.

        ``training.freeze_decoder_embeddings: true`` opts IN.
        """
        if not bool(self.step_cfg.get("freeze_decoder_embeddings", False)):
            logger.warning("decoder_embeddings_trainable_opt_out")
            return
        frozen = 0
        for dec in self._all_decoders():
            backbone = getattr(dec, "backbone", dec)
            for mod in filter(None, (
                backbone.get_input_embeddings() if hasattr(backbone, "get_input_embeddings")
                else None,
                backbone.get_output_embeddings() if hasattr(backbone, "get_output_embeddings")
                else None,
            )):
                for p in mod.parameters():
                    if p.requires_grad:
                        p.requires_grad_(False)
                        frozen += p.numel()
        logger.info("decoder_embeddings_frozen", params=frozen)

    def _all_decoders(self) -> list:
        """Decoders to optimize/save: both families in round-robin, else one."""
        if getattr(self, "_decoders_by_family", None):
            return list(self._decoders_by_family.values())
        return [self.decoder]

    def _pick_decoder_family(self) -> str:
        """Pick the decoder family for the next batch (round-robin routing)."""
        import random

        if abs(self._qwen_decoder_prob - 0.5) > 1e-6:
            return "qwen35" if random.random() < self._qwen_decoder_prob else "falcon_h1"
        family = "qwen35" if (self._microbatch_counter % 2 == 0) else "falcon_h1"
        self._microbatch_counter += 1
        return family

    def _set_active_decoder(self, family: str) -> None:
        """Aim self.decoder / self.tokenizer / encoder projection at ``family``.

        Called before each batch (train) / sample (eval) so the encoder's active
        projection block, the decoder, and the tokenizer that renders the
        trajectory are all the same family.
        """
        self.decoder = self._decoders_by_family[family]
        self.tokenizer = self._tokenizer_by_family[family]
        self._decoder_family = family
        self.encoder.set_active_decoder_family(family)

    def _mem_breakdown(self, phase: str, **extra) -> None:
        """Flag-gated (``BGKIT_MEM_BREAKDOWN``) per-phase CUDA-memory probe for
        the per-repo full-backprop step. Emits ``resident_gb``
        (``memory_allocated``) + ``peak_gb`` (``max_memory_allocated`` SINCE the
        last probe — i.e. THIS phase's peak), then resets the peak counter so
        the next phase's peak is isolated. No-op (one bool check) when off."""
        if not _MEM_BREAKDOWN or not torch.cuda.is_available():
            return
        resident = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        logger.info(
            "phase2_kb_per_repo_memory_breakdown",
            phase=phase,
            resident_gb=round(resident, 3),
            phase_peak_gb=round(peak, 3),
            phase_peak_delta_gb=round(peak - resident, 3),
            **extra,
        )
        # The reset below isolates the NEXT phase's peak but would also make
        # the step-level ``mem/cuda_max_allocated_gb`` (read from the same
        # counter at log time) report only the last phase. Keep the true
        # max since the previous log for ``_step_peak_allocated_gb_hook``.
        self._mem_breakdown_peak_gb = max(
            float(getattr(self, "_mem_breakdown_peak_gb", 0.0)), float(peak),
        )
        torch.cuda.reset_peak_memory_stats()

    def _step_peak_allocated_gb_hook(self) -> float | None:
        """True per-step peak under the breakdown probe (see BaseTrainer)."""
        tracked = getattr(self, "_mem_breakdown_peak_gb", None)
        if tracked is None:
            return None
        self._mem_breakdown_peak_gb = 0.0
        return float(tracked)

    @contextlib.contextmanager
    def _timed(self, store: dict, key: str, *, gpu: bool = False):
        """Flag-gated per-component timer for the per-repo profile.

        No-op (zero overhead) unless ``self._profile_timing`` is True. When on,
        accumulates ``perf_counter`` seconds into ``store[key]`` (summed across
        repeated calls, e.g. the per-sample PASS-2 components). For GPU ops it
        ``torch.cuda.synchronize()``-es before start AND before stop so the
        measured time reflects real device work rather than async-launch return
        — this sync is ONLY taken under the flag, so the real run keeps full
        CPU/GPU overlap.
        """
        if not getattr(self, "_profile_timing", False):
            yield
            return
        sync = gpu and torch.cuda.is_available()
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync:
                torch.cuda.synchronize()
            store[key] = store.get(key, 0.0) + (time.perf_counter() - t0)

    def _forward_backward_per_repo(self, batch) -> dict[str, float]:
        """Per-repo forward + backward (git_commit_repro full-backprop).

        ``batch`` is one repo's window-0 file-samples (grouped by
        :class:`~bgkit.data.samplers.RepoGroupedBatchSampler`).

        DETACH-AND-REACCUMULATE — the shared-tree forward AND backward each run
        EXACTLY ONCE, not Nx (the previous retain_graph design re-ran the
        ~75k-token shared-tree backward once PER file-sample, exhausting unified
        memory on a normal-sized window):

          0. Encode the shared window-0 tree ONCE → live node reps ``R`` (in
             ``memo``, with grad to L0/L1/bridge). Build DETACHED leaf copies
             ``R_d[nid] = R[nid].detach().requires_grad_(True)``.
          PASS 1 (cheap, NO encode, NO graph): count CONTRIBUTING file-samples
             via :meth:`_sample_contributing_token_count` (render only) →
             ``n_contrib``.
          PASS 2 (ONE-AT-A-TIME): each sample's browse turns SPLICE ``R_d``
             (not ``R``), so ``(loss/(n_contrib*accum)).backward()`` (NO
             retain_graph) flows into ``R_d[nid].grad`` and into the drill's own
             L1/bridge/decoder params — the drill+decode graph frees each
             iteration; the shared-tree graph is NOT touched.
          FINAL: ONE ``torch.autograd.backward(R[used], R_d[used].grad)`` pushes
             the accumulated leaf gradients through the shared tree to
             L0/L1/bridge. Exact because ``dR/dparams`` is shared:
             ``Σ_i dLi/dparams = (Σ_i dLi/dR)·dR/dparams``.

        Memory: the shared-tree forward activations are freed during PASS 2
        (only ``R``/``R_d`` are held); the FINAL backward re-materialises them
        once. ``max_file_samples_per_repo`` bounds the per-sample drill+decode
        cost + data balance. Gradient flows to L0 (live) + L1 + ``l1l1_bridge``
        + the active decoder.
        """
        if not batch:
            return {"loss": torch.zeros((), device=self.device), "tokens": 0.0}

        if self._round_robin:
            self._set_active_decoder(self._pick_decoder_family())

        dataset = batch[0].dataset_name
        root_node_id = self._repo_group_key(batch[0])
        # m3: every reconstruction trajectory must start with an is_head bgkit
        # turn (the shared-subtree root / window node). An empty key collapses
        # cross-dataset in the sampler AND leaves _compute_shared_repo_tree with
        # no root — a silent no-op. Fail loudly so a future data regression is
        # caught.
        assert root_node_id, (
            "per-repo batch sample has no is_head bgkit turn (empty shared-tree "
            f"root key); dataset={dataset!r}. git_commit_repro trajectories must "
            "begin with an is_head bgkit turn (window node)."
        )

        # M1/M2 knob: optionally subsample the repo's file-samples to bound the
        # per-step cost (N tree-recomputes) + staged-graph memory. Re-seeded per
        # epoch so coverage rotates across epochs.
        batch = self._subsample_repo_batch(batch, root_node_id)

        self._pending_l0_outputs = []
        self._pending_l1_outputs = []
        self._step_sampled_l0_ratios = []
        self._step_sampled_l1_ratios = []
        self._proj_norm_ratio_accum = {}
        if self.topic_embeddings is not None:
            self.topic_embeddings.record_batch_usage(
                [self._sample_tags_for(s) for s in batch],
            )

        # Per-component timing (flag-gated; cheap no-op when off). The synced
        # brackets only fire under profile_timing so the real run keeps overlap.
        _prof = getattr(self, "_profile_timing", False)
        _t: dict[str, float] = {}
        _wall0 = time.perf_counter() if _prof else 0.0

        # (a) baseline: model + optimizer state + grads resident at entry
        #     (resets the peak counter so the encode peak below is isolated).
        self._mem_breakdown(
            "entry", repo=root_node_id, n_file_samples=len(batch),
            decoder_family=getattr(self, "_decoder_family", None),
        )

        # --- 0. Encode the shared window-0 tree ONCE (general prompt), then
        #        build DETACHED leaf copies the per-sample splices read from.
        with self._timed(_t, "shared_tree_encode", gpu=True):
            memo, _stats = self._compute_shared_repo_tree(dataset, root_node_id)
        self._shared_tree_memo = memo
        splice_reps: dict[str, torch.Tensor] = {
            nid: proj.detach().requires_grad_(True)
            for nid, (proj, _l1out) in memo.items()
            if proj is not None
        }
        self._shared_tree_splice_reps = splice_reps
        self._shared_tree_used_nodes = set()
        self._shared_tree_child_l1_reps = {}
        self._shared_tree_child_l1_used = set()
        self._per_repo_shared_tree_active = True
        # (b) after the shared-tree L0/L1 encode: the retained R graph delta
        #     (phase_peak_gb = the encode's transient peak; resident_gb -
        #     baseline = the retained R graph held for the final backward).
        self._mem_breakdown(
            "after_shared_tree_encode",
            n_tree_nodes=int(_stats.get("nodes", 0)),
            n_spliced=len(splice_reps),
        )

        total_loss_val = 0.0
        total_tokens = 0
        n_samples = 0
        n_turns_total = 0
        try:
            # --- PASS 1: cheap contributing-count (render only; NO encode, NO
            #     graph, NO θ side-effect). This is the normaliser.
            with self._timed(_t, "pass1_count"):
                n_contrib = sum(
                    1 for sample in batch
                    if self._sample_contributing_token_count(sample) > 0
                )
            if n_contrib == 0:
                return {"loss": torch.zeros((), device=self.device), "tokens": 0.0}

            # --- PASS 2: GROUP-BATCHED. Iterate the contributing samples in
            #     groups of G (per_repo_sample_group_size); per group run ONE
            #     packed L1 drill encode (bucketed) + per-sample decode summed
            #     into a group loss, then ONE group backward (NO retain_graph).
            #     Browse turns splice DETACHED R_d (via
            #     _recursive_browse_node_reps), so the group backward flows into
            #     R_d[nid].grad + the drill/decoder params, NOT the shared tree.
            #     The group's graph FREES after its backward (memory bounded by
            #     G + gradient checkpointing, NOT by N); the shared tree is never
            #     re-run here. The drill uses the recursive L1 ramp (one
            #     curriculum with the tree). The per-component timers (prep /
            #     drill_encode / assemble / decode_fwd) accumulate inside the
            #     group encode; sample_backward times the per-group backward.
            group_size = max(1, int(getattr(self, "_per_repo_sample_group_size", 1) or 1))
            # Retrieve-leaf drill L1: the drill_leaf_retention.l1 override when
            # set (query-conditioned-drill-nodes mode), else the recursive L1
            # ramp (legacy drill↔tree coupling). Threaded as l1_target_ratio to
            # _run_l1_batch — affects ONLY the bucketed leaf drills (head/node
            # turns resolve separately at drill_node_retention / tree ratios).
            l1_ramp = self._drill_leaf_l1_retention_now()
            n_groups = 0
            for start in range(0, len(batch), group_size):
                group = batch[start:start + group_size]
                group_loss, g_tokens, g_done, g_turns, _g_buckets = (
                    self._encode_decode_group(group, _t, l1_target_ratio=l1_ramp)
                )
                if g_done == 0:
                    continue
                # Normalise by the TRUE contributing count (m1) so the total
                # gradient is invariant to the group size.
                scaled = group_loss / (n_contrib * self._accum_steps)
                with self._timed(_t, "sample_backward", gpu=True):
                    scaled.backward()
                total_loss_val += float(group_loss.detach())
                total_tokens += g_tokens
                n_samples += g_done
                n_turns_total += g_turns
                n_groups += 1

            if n_samples == 0:
                # Every Pass-1 contributor degenerated to empty segments in
                # Pass 2 (rare). Nothing was backwarded.
                return {"loss": torch.zeros((), device=self.device), "tokens": 0.0}

            # --- 3. Survivorship aux losses over the accumulated L0 (shared
            #        tree leaves + per-sample drills) + L1 outputs. No-op when
            #        survivorship aux is OFF (returns zeros). When ON, this
            #        backward traverses the shared tree, so RETAIN the graph for
            #        the reaccumulate below.
            aux_loss, aux_metrics = self._compute_survivorship_aux_losses()
            aux_has_grad = bool(aux_loss.requires_grad)
            if aux_has_grad:
                (aux_loss / self._accum_steps).backward(retain_graph=True)

            # --- FINAL: ONE backward through the shared tree, feeding each used
            #     leaf's accumulated gradient. dR/dparams is shared, so this is
            #     exactly Σ_i dLi/dparams for L0/L1/bridge. retain_graph=False →
            #     frees the shared-tree graph here.
            used = [
                nid for nid in self._shared_tree_used_nodes
                if splice_reps.get(nid) is not None
                and splice_reps[nid].grad is not None
            ]
            # Drill-down HEAD children: reaccumulate each consumed child's
            # detached-L1-output gradient back into memo[c][1] in the SAME
            # tree backward (one traversal covers node-nav proj + head l1out).
            child_reps = self._shared_tree_child_l1_reps or {}
            child_used = [
                c for c in (self._shared_tree_child_l1_used or set())
                if child_reps.get(c) is not None
                and child_reps[c].grad is not None
            ]
            bwd_tensors = (
                [memo[nid][0] for nid in used]
                + [memo[c][1] for c in child_used]
            )
            bwd_grads = (
                [splice_reps[nid].grad for nid in used]
                + [child_reps[c].grad for c in child_used]
            )
            if bwd_tensors:
                with self._timed(_t, "final_tree_backward", gpu=True):
                    torch.autograd.backward(
                        tensors=bwd_tensors,
                        grad_tensors=bwd_grads,
                    )
            # (d) after the final tree backward: phase_peak_gb = the
            #     re-materialised shared-tree backward peak (the recompute of
            #     R's L0/L1 forward + grads through L0/L1/bridge).
            self._mem_breakdown("after_final_tree_backward", n_used_nodes=len(used))
        finally:
            self._per_repo_shared_tree_active = False
            self._shared_tree_memo = None
            self._shared_tree_splice_reps = None
            self._shared_tree_used_nodes = None
            self._shared_tree_child_l1_reps = None
            self._shared_tree_child_l1_used = None

        self._apply_utility_grad_bce_phase2()
        if self.topic_embeddings is not None:
            self.topic_embeddings.apply_gradient_averaging()

        if _prof:
            repo_wall_s = time.perf_counter() - _wall0
            # Rough GPU/CPU split: GPU = ops that touch the device (prep is
            # mixed — its CPU render is small vs the L0 forward — counted GPU).
            gpu_keys = (
                "shared_tree_encode", "prep", "drill_encode",
                "decode_fwd", "sample_backward", "final_tree_backward",
            )
            cpu_keys = ("pass1_count", "assemble")
            gpu_op_s = sum(_t.get(k, 0.0) for k in gpu_keys)
            cpu_op_s = sum(_t.get(k, 0.0) for k in cpu_keys)
            logger.info(
                "per_repo_timing",
                repo=root_node_id,
                n_file_samples=len(batch),
                n_contrib=n_contrib,
                n_groups=n_groups,
                group_size=group_size,
                repo_wall_s=round(repo_wall_s, 4),
                gpu_op_s=round(gpu_op_s, 4),
                cpu_op_s=round(cpu_op_s, 4),
                shared_tree_encode_s=round(_t.get("shared_tree_encode", 0.0), 4),
                pass1_count_s=round(_t.get("pass1_count", 0.0), 4),
                prep_s=round(_t.get("prep", 0.0), 4),
                drill_encode_s=round(_t.get("drill_encode", 0.0), 4),
                assemble_s=round(_t.get("assemble", 0.0), 4),
                decode_fwd_s=round(_t.get("decode_fwd", 0.0), 4),
                group_backward_s=round(_t.get("sample_backward", 0.0), 4),
                final_tree_backward_s=round(_t.get("final_tree_backward", 0.0), 4),
            )

        metrics_out = {
            "loss": torch.tensor(total_loss_val / n_samples, device=self.device),
            "tokens": float(total_tokens / n_samples),
            "live_l0": 1.0 if self._live_l0 else 0.0,
            "per_repo_file_samples": float(len(batch)),
            "per_repo_contributing_samples": float(n_contrib),
            "per_repo_backwarded_samples": float(n_samples),
            "per_repo_shared_tree_nodes": float(_stats.get("nodes", 0)),
            "l1_turns_per_sample": float(n_turns_total / n_samples),
            "recursive_l0_retention": self._recursive_l0_retention_now(),
            "recursive_l1_retention": self._recursive_l1_retention_now(),
        }
        metrics_out.update(aux_metrics)
        metrics_out.update(self._proj_norm_reg_step_metrics())
        return metrics_out

    # ------------------------------------------------------------------
    # Inner-loop per-repo COMPUTE primitive (gated; driver integration TBD)
    # ------------------------------------------------------------------

    def _encode_repo_tree_for_inner_loop(self, dataset: str, root_node_id: str):
        """Encode the shared window-0 tree ONCE for the inner loop and install
        the LIVE node reps as the browse-splice source (NOT detached), so each
        inner subset's backward flows gradient THROUGH the retained tree into
        the encoder (the intentional delayed/stale gradient). Returns
        ``(memo, stats)``. The caller RETAINS ``memo`` across the K inner steps
        (reuse stale) and calls :meth:`_free_inner_loop_tree` after the repo."""
        memo, stats = self._compute_shared_repo_tree(dataset, root_node_id)
        self._shared_tree_memo = memo
        # LIVE reps (not .detach()) — gradient flows through R into the encoder.
        self._shared_tree_splice_reps = {
            nid: proj for nid, (proj, _l1) in memo.items() if proj is not None
        }
        self._shared_tree_used_nodes = set()
        self._per_repo_shared_tree_active = True
        return memo, stats

    def _free_inner_loop_tree(self) -> None:
        """Drop the retained shared-tree refs after a repo's K inner steps so
        its graph is collected (frees the retained-graph memory floor)."""
        self._per_repo_shared_tree_active = False
        self._shared_tree_memo = None
        self._shared_tree_splice_reps = None
        self._shared_tree_used_nodes = None

    def _inner_loop_subset_backward(
        self, subset_samples: list, *, l1_target_ratio: float, normalizer: int,
    ) -> tuple[float, int, int, int]:
        """ONE inner step over a file-subset, backproping through the LIVE
        retained shared tree (splices read the live memo reps installed by
        :meth:`_encode_repo_tree_for_inner_loop`). The subset is processed in
        groups of G (``per_repo_sample_group_size``); each group backward uses
        ``retain_graph=True`` so the shared tree survives for the next inner
        step (memory bounded by G + the retained tree, NOT the whole repo). The
        DRIVER calls optimizer.step/zero_grad/scheduler/global_step++ after this
        returns (one inner step == one real optimizer step). Returns
        ``(loss_val, tokens, n_done, n_turns)``."""
        group_size = max(1, int(getattr(self, "_per_repo_sample_group_size", 1) or 1))
        denom = max(1, int(normalizer)) * self._accum_steps
        total_loss_val = 0.0
        total_tokens = 0
        n_done = 0
        n_turns = 0
        for start in range(0, len(subset_samples), group_size):
            group = subset_samples[start:start + group_size]
            group_loss, g_tok, g_done, g_turns, _ = self._encode_decode_group(
                group, l1_target_ratio=l1_target_ratio,
            )
            if g_done == 0:
                continue
            # retain_graph: keep the LIVE shared tree alive across groups AND
            # across the repo's subsequent inner steps; freed by
            # _free_inner_loop_tree after the repo.
            (group_loss / denom).backward(retain_graph=True)
            total_loss_val += float(group_loss.detach())
            total_tokens += g_tok
            n_done += g_done
            n_turns += g_turns
        return total_loss_val, total_tokens, n_done, n_turns

    def _partition_inner_subsets(self, samples: list) -> list[list]:
        """Partition contributing samples into inner-step subsets of size
        ``per_repo_inner_subset_size``, CAPPED at ``per_repo_max_inner_steps``
        subsets (bounds staleness). If there are more subsets than the cap, the
        remainder is DROPPED this repo-visit (documented — coverage rotates via
        the per-epoch reshuffle)."""
        s = max(1, int(getattr(self, "_per_repo_inner_subset_size", 16) or 1))
        k_cap = max(1, int(getattr(self, "_per_repo_max_inner_steps", 12) or 1))
        subsets = [samples[i:i + s] for i in range(0, len(samples), s)]
        return subsets[:k_cap]

    def _partition_option_a_subsets(self, samples: list) -> list[list]:
        """Partition into K = ceil(n/S) subsets of size S for Option A, with NO
        staleness K-cap (Option A steps the encoder ONCE per repo, so there is
        no staleness to bound) → ALL files are processed (the user's core
        motivation: no file cap). ``option_a_max_subsets`` (0 = unlimited)
        provides an OPTIONAL safety cap only."""
        s = max(1, int(getattr(self, "_per_repo_inner_subset_size", 16) or 1))
        subsets = [samples[i:i + s] for i in range(0, len(samples), s)]
        cap = int(getattr(self, "_option_a_max_subsets", 0) or 0)
        return subsets if cap <= 0 else subsets[:cap]

    def _forward_backward_inner_loop(self, batch) -> dict[str, float]:
        """Model-B inner-loop step: ONE subset-batch (the sampler already
        partitioned the repo into K consecutive subsets) → ONE outer optimizer
        step. The repo's shared tree is encoded ONCE on its FIRST subset
        (detected by the window-node root id changing) and REUSED — live,
        progressively stale — for the repo's subsequent subsets, freed on
        repo-change. Each subset backprops through the retained live tree
        (delayed gradient). The base loop does optimizer.step / scheduler /
        eval / save / global_step++ after this returns, so each inner step is a
        real global step. A mid-repo eval/save does NOT change the repo key, so
        the cached tree survives it and the next subset reuses it.
        """
        if not batch:
            return {"loss": torch.zeros((), device=self.device), "tokens": 0.0}

        dataset = batch[0].dataset_name
        root_key = self._repo_group_key(batch[0])
        assert root_key, (
            "inner-loop subset-batch has no is_head bgkit turn (empty "
            f"shared-tree root key); dataset={dataset!r}."
        )

        # Reset per-step survivorship-aux accumulators (drained by the base
        # loop's _post_optimizer_step dual-ascent each step). θ control state
        # (_surv_state_*) is NOT reset here — it accumulates via encode-time
        # _accumulate_theta_state and is drained per step by dual ascent.
        self._pending_l0_outputs = []
        self._pending_l1_outputs = []
        self._step_sampled_l0_ratios = []
        self._step_sampled_l1_ratios = []
        self._proj_norm_ratio_accum = {}

        repo_changed = root_key != self._inner_loop_repo_key
        if repo_changed:
            # Free the previous repo's retained tree, then encode THIS repo's
            # shared tree ONCE (live splice source). Round-robin: fix the
            # decoder family for the whole repo (the tree projection is
            # family-specific).
            self._free_inner_loop_tree()
            if self._round_robin:
                self._set_active_decoder(self._pick_decoder_family())
            # (a) baseline at a repo-change (after the prior repo's tree freed).
            self._mem_breakdown(
                "inner_entry", repo=root_key,
                decoder_family=getattr(self, "_decoder_family", None),
            )
            _memo, stats = self._encode_repo_tree_for_inner_loop(dataset, root_key)
            self._inner_loop_repo_key = root_key
            self._inner_loop_repo_steps = 0
            self._inner_loop_tree_nodes = int(stats.get("nodes", 0))
            # (b) after the shared-tree encode: retained LIVE R graph held
            #     across the repo's K inner steps (the inner-loop memory floor).
            self._mem_breakdown(
                "inner_after_shared_tree_encode",
                n_tree_nodes=int(stats.get("nodes", 0)),
            )
        else:
            self._inner_loop_repo_steps += 1

        # Retrieve-leaf drill L1: drill_leaf_retention.l1 override when set,
        # else the recursive L1 ramp (legacy) — see the Option A driver note.
        l1_ramp = self._drill_leaf_l1_retention_now()
        # subset_loss / S (S = this subset's sample count; last subset may be
        # short). One inner step == one optimizer step.
        normalizer = max(1, len(batch))
        loss_v, tokens, n_done, n_turns = self._inner_loop_subset_backward(
            batch, l1_target_ratio=l1_ramp, normalizer=normalizer,
        )
        # (d) after the subset's backward(s) through the retained live tree
        #     (the inner-loop analogue of the final tree backward).
        self._mem_breakdown(
            "inner_after_subset_backward",
            inner_repo_step=int(self._inner_loop_repo_steps),
            subset_samples=len(batch),
        )
        if n_done == 0:
            return {"loss": torch.zeros((), device=self.device), "tokens": 0.0}

        # Utility-grad BCE (no-op when survivorship aux is off, as in the
        # per-repo config). No extra survivorship-aux backward: the subset
        # backward already ran (retain_graph keeps the live tree); θ updates
        # via the base loop's _post_optimizer_step dual ascent.
        self._apply_utility_grad_bce_phase2()

        return {
            "loss": torch.tensor(loss_v / n_done, device=self.device),
            "tokens": float(tokens / n_done),
            "live_l0": 1.0 if self._live_l0 else 0.0,
            "inner_loop": 1.0,
            "inner_repo_step": float(self._inner_loop_repo_steps),
            "inner_subset_samples": float(len(batch)),
            "inner_contributing": float(n_done),
            "per_repo_shared_tree_nodes": float(
                getattr(self, "_inner_loop_tree_nodes", 0),
            ),
            "l1_turns_per_sample": float(n_turns / n_done),
            "recursive_l0_retention": self._recursive_l0_retention_now(),
            "recursive_l1_retention": self._recursive_l1_retention_now(),
        }

    # ------------------------------------------------------------------
    # Option A — crash-free amortized per-repo (decoder K-step / encoder 1-step)
    # ------------------------------------------------------------------

    def _option_a_group_indices(self) -> tuple[list[int], list[int]]:
        """Partition ``self.optimizer.param_groups`` into (decoder, encoder)
        index lists by param IDENTITY. A group is "decoder" iff any of its
        params is in ``_option_a_decoder_param_ids`` (groups are homogeneous —
        Muon's 2D/1D split keeps a source group's params together). Everything
        else is "encoder" (every param R's tree graph depends on)."""
        dec_ids = getattr(self, "_option_a_decoder_param_ids", frozenset())
        dec_idx: list[int] = []
        enc_idx: list[int] = []
        for i, group in enumerate(self.optimizer.param_groups):
            params = group.get("params", [])
            is_dec = any(id(p) in dec_ids for p in params)
            (dec_idx if is_dec else enc_idx).append(i)
        return dec_idx, enc_idx

    def _option_a_params_for_groups(self, indices: list[int]) -> list:
        """Flatten the params of the selected optimizer param-groups."""
        groups = self.optimizer.param_groups
        return [p for i in indices for p in groups[i].get("params", [])]

    def _option_a_step_groups(self, indices: list[int]) -> None:
        """Step ONLY the selected param-groups of the single optimizer.

        Temporarily narrows ``optimizer.param_groups`` to ``indices`` so
        ``optimizer.step()`` updates just those params (per-param Adam/Muon
        state is keyed by param, so a partial step is correct + checkpoint
        stays single-optimizer). Restored in ``finally``."""
        if not indices:
            return
        saved = self.optimizer.param_groups
        self.optimizer.param_groups = [saved[i] for i in indices]
        try:
            self.optimizer.step()
        finally:
            self.optimizer.param_groups = saved

    @staticmethod
    def _option_a_zero_grads(params: list) -> None:
        """Set ``.grad = None`` on the given params (selective zero — does NOT
        touch the encoder grads that must accumulate across subsets)."""
        for p in params:
            p.grad = None

    @staticmethod
    def _span_ce_sum_count(
        out, spans: list[tuple[int, int]],
    ) -> tuple[float, int]:
        """Per-position CE (sum, count) over the given concat-coordinate spans.

        ``out`` is the :class:`InterleavedForwardOutput` from
        ``forward_interleaved_with_loss(..., return_hidden_states=True)`` — B=1.
        ``spans`` are ``[start, end)`` TARGET-position ranges in the SAME
        coordinate space as ``out.token_ids`` / ``out.loss_mask`` (post-splice,
        already coordinate-shifted by :meth:`_assemble_sample_segments`, so NO
        re-shift is applied here). Spans are unioned; a span that runs past the
        decoded length (e.g. after gold-budget truncation) is clamped.

        Next-token shift: predicting the token at TARGET position ``p`` uses the
        hidden state at ``p-1``. Only positions with ``loss_mask[p] == True`` and
        ``1 <= p < S`` contribute. Returns ``(sum_ce, count)``; count 0 ⇒ no
        contributing positions.
        """
        import torch.nn.functional as F

        token_ids = out.token_ids  # (1, S) long
        loss_mask = out.loss_mask  # (1, S) bool
        hidden = out.hidden_states  # (1, S, D)
        s = int(token_ids.size(1))
        select = torch.zeros(s, dtype=torch.bool, device=hidden.device)
        for start, end in spans:
            lo = max(1, int(start))
            hi = min(s, int(end))
            if hi > lo:
                select[lo:hi] = True
        select = select & loss_mask[0].to(dtype=torch.bool)
        pos = select.nonzero(as_tuple=False).squeeze(-1)  # (M,) target positions
        if pos.numel() == 0:
            return 0.0, 0
        h_sel = hidden[0].index_select(0, pos - 1)  # (M, D) predictor hiddens
        logits = out.lm_head(h_sel).float()  # (M, V)
        targets = token_ids[0].index_select(0, pos)  # (M,)
        ce = F.cross_entropy(logits, targets, reduction="none")  # (M,)
        return float(ce.sum().item()), int(pos.numel())

    def _accumulate_span_ce(self, accum: dict, out, trace) -> None:
        """Split the decode's per-position CE into NAVIGATION (bgkit drill-id
        tool-call spans) and RECONSTRUCTION (gold answer span) buckets and add
        the (sum, count) into ``accum`` (keys ``nav_sum``/``nav_count``/
        ``recon_sum``/``recon_count``). Reuses the trace's already-shifted spans.
        """
        # nav_gap over DRILL ids: exclude the FIRST bgkit call (the head). The
        # head's nav tokens render BEFORE its own survivor is spliced, so its
        # prediction is causally survivor-INDEPENDENT (gap structurally 0), and
        # its query text is a copyable paraphrase of the question — both dilute
        # nav_gap toward 0. Only depth>=1 drills (child ids after the parent's
        # rep) can be rep-dependent, so measure those. (Measurement only — does
        # not change the training loss.)
        nav_spans = [
            span for turn, span in zip(
                trace.bgkit_turns, trace.bgkit_call_spans, strict=True,
            )
            if getattr(turn, "loss", True) and not turn.args.get("is_head")
        ]
        recon_spans = (
            [trace.answer_span] if trace.answer_span is not None else []
        )
        nav_sum, nav_count = self._span_ce_sum_count(out, nav_spans)
        recon_sum, recon_count = self._span_ce_sum_count(out, recon_spans)
        accum["nav_sum"] += nav_sum
        accum["nav_count"] += nav_count
        accum["recon_sum"] += recon_sum
        accum["recon_count"] += recon_count
        # DIAGNOSTIC: bucket by the artifact's structural tree depth, not by the
        # number of tool calls (multiple evidence branches and distractors make
        # call count unrelated to hierarchy depth). Reconstruction loss should
        # drop with depth because deeper drills retrieve more
        # of the real diff content, so reconstruction gets easier; a full-depth
        # drill that STILL has high recon_loss would be the concerning case (the
        # model has the diffs but can't use them). depth 1 == no_drill (recon from
        # the compressed head rep alone → hardest). Measurement only.
        depth = max(
            (
                int(turn.args.get("structural_depth", 0))
                for turn in trace.bgkit_turns
                if getattr(turn, "loss", True)
            ),
            default=0,
        )
        bd = accum.setdefault("by_depth", {}).setdefault(
            depth, {"recon_sum": 0.0, "recon_count": 0,
                    "nav_sum": 0.0, "nav_count": 0, "n": 0},
        )
        bd["recon_sum"] += recon_sum
        bd["recon_count"] += recon_count
        bd["nav_sum"] += nav_sum
        bd["nav_count"] += nav_count
        bd["n"] += 1

    def _run_ablation_gap_probe(
        self, group: list, l1_target_ratio: float | None,
    ) -> None:
        """READ-ONLY ablation-gap diagnostic for the git-repro decode path.

        Decodes ``group`` TWICE under ``torch.no_grad()`` — once normally, once
        with the compressed survivors forced to zero via
        :attr:`ABLATION_ZEROED` — and logs
        ``gap = loss_zeroed - loss_normal``. A gap ≈ 0 confirms the decoder is
        ignoring the survivors (both the per-turn bgkit drill survivors AND the
        recursive-L1 browse node-reps: ``ABLATION_ZEROED`` zeroes BOTH — the
        drill survivors at the bgkit-splice in :meth:`_assemble_sample_segments`
        and the browse node-reps at the browse-splice — because ``skip_survivors``
        is only set for ``TOPICS_ONLY``/``NEITHER``, so ZEROED takes the
        ``torch.zeros_like`` branch for each spliced payload).

        Zero training impact: no backward, no optimizer step. Every mutable
        training-state container the probe's encode/decode may touch (the θ
        dual-ascent accumulators, the ``_pending_*`` retention lists, the
        sampled-ratio lists, the shared-tree used-node set) plus the ablation
        mode is snapshotted and restored, so the real forward/backward that
        follows sees byte-identical state. Uses the SAME group-decode helper as
        training (:meth:`_encode_decode_group`) so layout/masking is identical.
        """
        from bgkit.training.survivorship_helpers import init_state

        saved_ablation = self._ablation_mode
        saved_pending_l0 = self._pending_l0_outputs
        saved_pending_l1 = self._pending_l1_outputs
        saved_ratios_l0 = self._step_sampled_l0_ratios
        saved_ratios_l1 = self._step_sampled_l1_ratios
        saved_used_nodes = self._shared_tree_used_nodes
        saved_child_used = self._shared_tree_child_l1_used
        saved_state_l0 = getattr(self, "_surv_state_l0", None)
        saved_state_l1 = getattr(self, "_surv_state_l1", None)
        try:
            # Redirect all encode-time accumulation into throwaway containers.
            self._pending_l0_outputs = []
            self._pending_l1_outputs = []
            self._step_sampled_l0_ratios = []
            self._step_sampled_l1_ratios = []
            self._shared_tree_used_nodes = (
                set(saved_used_nodes) if saved_used_nodes is not None else set()
            )
            self._shared_tree_child_l1_used = (
                set(saved_child_used) if saved_child_used is not None else set()
            )
            if saved_state_l0 is not None:
                self._surv_state_l0 = init_state()
            if saved_state_l1 is not None:
                self._surv_state_l1 = init_state()

            # Per-token-type CE accumulators (nav = bgkit drill-id tool-call
            # spans, recon = gold answer span) collected across the group's
            # samples inside _encode_decode_group's per-sample decode.
            accum_normal = {
                "nav_sum": 0.0, "nav_count": 0,
                "recon_sum": 0.0, "recon_count": 0,
            }
            accum_zeroed = {
                "nav_sum": 0.0, "nav_count": 0,
                "recon_sum": 0.0, "recon_count": 0,
            }
            with torch.no_grad():
                # Normal decode: _ablation_mode None == exactly what the real
                # Option A subset decode uses (no _training_ablation_override
                # on this path).
                self._ablation_mode = self.ABLATION_NONE
                loss_normal, _, done_n, _, _ = self._encode_decode_group(
                    group, None, l1_target_ratio=l1_target_ratio,
                    span_ce_accum=accum_normal,
                )
                # Survivors-zeroed decode of the SAME group.
                self._ablation_mode = self.ABLATION_ZEROED
                loss_zeroed, _, done_z, _, _ = self._encode_decode_group(
                    group, None, l1_target_ratio=l1_target_ratio,
                    span_ce_accum=accum_zeroed,
                )

            if done_n == 0 or done_z == 0:
                return

            def _mean(acc: dict, prefix: str) -> float:
                c = acc[f"{prefix}_count"]
                return (acc[f"{prefix}_sum"] / c) if c > 0 else float("nan")

            def _r(v: float) -> float | None:
                return round(v, 6) if v == v else None  # None for NaN

            ln = float(loss_normal.detach())
            lz = float(loss_zeroed.detach())
            nav_ln = _mean(accum_normal, "nav")
            nav_lz = _mean(accum_zeroed, "nav")
            recon_ln = _mean(accum_normal, "recon")
            recon_lz = _mean(accum_zeroed, "recon")
            nav_gap = nav_lz - nav_ln
            recon_gap = recon_lz - recon_ln
            logger.info(
                "ablation_probe",
                step=int(self.global_step),
                loss_normal=round(ln, 6),
                loss_zeroed=round(lz, 6),
                gap=round(lz - ln, 6),
                nav_loss_normal=_r(nav_ln),
                nav_loss_zeroed=_r(nav_lz),
                nav_gap=_r(nav_gap),
                recon_loss_normal=_r(recon_ln),
                recon_loss_zeroed=_r(recon_lz),
                recon_gap=_r(recon_gap),
                nav_tokens=int(accum_normal["nav_count"]),
                recon_tokens=int(accum_normal["recon_count"]),
                decoder_family=getattr(self, "_decoder_family", None),
            )
        finally:
            self._ablation_mode = saved_ablation
            self._pending_l0_outputs = saved_pending_l0
            self._pending_l1_outputs = saved_pending_l1
            self._step_sampled_l0_ratios = saved_ratios_l0
            self._step_sampled_l1_ratios = saved_ratios_l1
            self._shared_tree_used_nodes = saved_used_nodes
            self._shared_tree_child_l1_used = saved_child_used
            if saved_state_l0 is not None:
                self._surv_state_l0 = saved_state_l0
            if saved_state_l1 is not None:
                self._surv_state_l1 = saved_state_l1

    def _forward_backward_option_a(self, batch) -> dict[str, float]:
        """OPTION A — crash-free amortized per-repo training.

        Resolves the original inner-loop's autograd crash by SEPARATING the
        optimizer cadences so the encoder is backward'd + stepped exactly once
        while R's retained graph is alive:

          0. Encode the shared window-0 tree ONCE → live reps ``R`` (in
             ``memo``; FIX2/2b conditional checkpointing; retained for the
             final encoder backward). Build DETACHED snapshots
             ``R_d[nid] = R[nid].detach().requires_grad_(True)`` — the stale
             tree context the decoder reads (browse-splice source).
          1. Count CONTRIBUTING file-samples (render only) → normaliser.
          2. Partition the contributing samples into K subsets of size S
             (``per_repo_inner_subset_size``; K capped at
             ``per_repo_max_inner_steps``).
          3. PER SUBSET: decode each file (reading ``R_d``) → loss → backward.
             The backward accumulates (a) DECODER grads, (b) the per-file live
             DRILL's ENCODER grads, and (c) ``R_d.grad``. Then step ONLY the
             DECODER param-groups + null ONLY the decoder grads. The ENCODER
             grads are LEFT to accumulate across subsets (they are NOT stepped
             and NOT zeroed) → K decoder updates, encoder untouched.
          4. AFTER all K subsets: null the decoder grads, then ONE
             ``torch.autograd.backward(R[used], R_d[used].grad)`` pushes the
             summed leaf gradient through the shared tree, ADDING the
             shared-tree grad to the already-accumulated drill encoder grads.
             Step ONLY the ENCODER param-groups ONCE, then null all grads + free
             the tree. = 1 encoder update/repo.

        AUTOGRAD CRUX: the decoder step mutates DECODER params only; R's graph
        depends on ENCODER params (L0/L1/bridge/projection/LoRA — shared by the
        drill AND the tree), which are NOT stepped until step 4, so R's retained
        graph stays valid for the final backward → no in-place collision.

        Integration: Option A OWNS its optimizer cadence (selective param-group
        stepping on the single ``self.optimizer`` — keeps checkpoint + LR
        scheduling single-optimizer). The base loop's trailing
        ``clip_grad_norm`` / ``optimizer.step()`` no-op because all grads are
        None at return. ONE base-loop global step == ONE repo (K decoder + 1
        encoder update); eval/save cadence therefore counts REPOS.
        """
        if not batch:
            return {"loss": torch.zeros((), device=self.device), "tokens": 0.0}

        if self._round_robin:
            self._set_active_decoder(self._pick_decoder_family())

        dataset = batch[0].dataset_name
        root_node_id = self._repo_group_key(batch[0])
        assert root_node_id, (
            "Option A batch sample has no is_head bgkit turn (empty shared-tree "
            f"root key); dataset={dataset!r}."
        )
        # NO file subsample — Option A processes ALL of a repo's files across
        # K=ceil(n_contrib/S) subsets (the user's core motivation: no file cap).
        # Memory stays bounded by the per-file drill checkpoint + decode GC +
        # the per-node tree-encode checkpoint, NOT by capping files.

        self._pending_l0_outputs = []
        self._pending_l1_outputs = []
        self._step_sampled_l0_ratios = []
        self._step_sampled_l1_ratios = []
        self._proj_norm_ratio_accum = {}
        if self.topic_embeddings is not None:
            self.topic_embeddings.record_batch_usage(
                [self._sample_tags_for(s) for s in batch],
            )

        from bgkit.training.gradient_utils import clip_grad_norm

        _prof = getattr(self, "_profile_timing", False)
        _t: dict[str, float] = {}
        _wall0 = time.perf_counter() if _prof else 0.0
        max_grad_norm = float(getattr(self, "_max_grad_norm", 1.0))
        dec_idx, enc_idx = self._option_a_group_indices()
        dec_params = self._option_a_params_for_groups(dec_idx)
        enc_params = self._option_a_params_for_groups(enc_idx)

        self._mem_breakdown(
            "option_a_entry", repo=root_node_id, n_file_samples=len(batch),
            decoder_family=getattr(self, "_decoder_family", None),
        )

        # --- 0. Encode the shared tree ONCE → R (live), build detached R_d.
        with self._timed(_t, "shared_tree_encode", gpu=True):
            memo, _stats = self._compute_shared_repo_tree(dataset, root_node_id)
        self._shared_tree_memo = memo
        splice_reps: dict[str, torch.Tensor] = {
            nid: proj.detach().requires_grad_(True)
            for nid, (proj, _l1out) in memo.items()
            if proj is not None
        }
        self._shared_tree_splice_reps = splice_reps
        self._shared_tree_used_nodes = set()
        self._shared_tree_child_l1_reps = {}
        self._shared_tree_child_l1_used = set()
        self._per_repo_shared_tree_active = True
        self._mem_breakdown(
            "option_a_after_shared_tree_encode",
            n_tree_nodes=int(_stats.get("nodes", 0)),
            n_spliced=len(splice_reps),
        )

        total_loss_val = 0.0
        total_tokens = 0
        n_samples = 0
        n_turns_total = 0
        n_subsets = 0
        n_decoder_steps = 0
        try:
            # --- 1. Contributing count (render only; no encode/graph).
            with self._timed(_t, "pass1_count"):
                contributing = [
                    s for s in batch
                    if self._sample_contributing_token_count(s) > 0
                ]
            n_contrib = len(contributing)
            if n_contrib == 0:
                return {"loss": torch.zeros((), device=self.device), "tokens": 0.0}

            # --- 2. Partition into K=ceil(n/S) subsets (size S, UNCAPPED —
            #        process all files; encoder steps once so no staleness).
            subsets = self._partition_option_a_subsets(contributing)
            # Retrieve-leaf drill L1: the drill_leaf_retention.l1 override when
            # set (query-conditioned-drill-nodes mode), else the recursive L1
            # ramp (legacy drill↔tree coupling). Threaded as l1_target_ratio to
            # _run_l1_batch — affects ONLY the bucketed leaf drills (head/node
            # turns resolve separately at drill_node_retention / tree ratios).
            l1_ramp = self._drill_leaf_l1_retention_now()
            group_size = max(
                1, int(getattr(self, "_per_repo_sample_group_size", 1) or 1),
            )

            # READ-ONLY ablation-gap probe (first N steps). Decodes ONE
            # representative group twice (normal + survivors-zeroed) under
            # no_grad and logs the loss gap. Guarded so a probe failure never
            # kills the training step. Runs BEFORE the real subset loop while
            # the shared tree is alive; snapshots+restores all training state.
            if self.global_step < self._ablation_probe_steps and contributing:
                try:
                    self._run_ablation_gap_probe(
                        contributing[:group_size], l1_ramp,
                    )
                except Exception as exc:  # diagnostic must never kill a step
                    logger.warning(
                        "ablation_probe_failed",
                        step=int(self.global_step),
                        error=str(exc),
                    )

            # --- 3. PER SUBSET: decode reading R_d → backward → step DECODER.
            #        Backward PER GROUP (freeing each group's drill+decode graph
            #        immediately) and step the decoder ONCE per subset — so the
            #        within-subset activation peak is ONE group (group_size
            #        files, default 1), NOT all S files held at once. The
            #        per-group decoder grads accumulate additively across the
            #        subset (== one backward of the subset sum, bit-exact), and
            #        the drill encoder grads + R_d.grad accumulate across all
            #        subsets for the single deferred encoder backward.
            denom = n_contrib * self._accum_steps
            for subset in subsets:
                n_subsets += 1
                # Null ONLY decoder grads before this subset (encoder grads
                # accumulate across subsets; first subset's encoder grads were
                # nulled by the base loop's zero_grad at step entry).
                self._option_a_zero_grads(dec_params)
                subset_done = 0
                for start in range(0, len(subset), group_size):
                    group = subset[start:start + group_size]
                    with self._timed(_t, "subset_decode_fwd", gpu=True):
                        g_loss, g_tok, g_done, g_turns, _ = self._encode_decode_group(
                            group, _t, l1_target_ratio=l1_ramp,
                        )
                    if g_done == 0:
                        continue
                    # Per-group backward → frees this group's graph now (bounds
                    # the peak to ONE group). Per-repo-consistent normaliser
                    # (n_contrib): each decoder step is a properly-scaled partial
                    # of the repo mean; the encoder's accumulated grad over all
                    # subsets + the tree-backward is the clean per-repo mean
                    # (scale-invariant to K).
                    with self._timed(_t, "subset_backward", gpu=True):
                        (g_loss / denom).backward()
                    total_loss_val += float(g_loss.detach())
                    total_tokens += g_tok
                    subset_done += g_done
                    n_turns_total += g_turns
                if subset_done == 0:
                    continue
                # Step ONLY the decoder (encoder grads left to accumulate).
                with self._timed(_t, "decoder_step", gpu=True):
                    if dec_params:
                        clip_grad_norm(dec_params, max_norm=max_grad_norm)
                    self._option_a_step_groups(dec_idx)
                n_decoder_steps += 1
                n_samples += subset_done

            if n_samples == 0:
                return {"loss": torch.zeros((), device=self.device), "tokens": 0.0}

            # --- 4. Null decoder grads, then ONE encoder update via the final
            #        tree-backward (adds shared-tree grad to the accumulated
            #        drill encoder grads). R's graph is still valid because the
            #        encoder was never stepped during the subsets.
            self._option_a_zero_grads(dec_params)
            used = [
                nid for nid in self._shared_tree_used_nodes
                if splice_reps.get(nid) is not None
                and splice_reps[nid].grad is not None
            ]
            # Drill-down HEAD children: reaccumulate each consumed child's
            # detached-L1-output gradient back into memo[c][1] in the SAME
            # tree backward (node-nav proj + head l1out, one traversal).
            child_reps = self._shared_tree_child_l1_reps or {}
            child_used = [
                c for c in (self._shared_tree_child_l1_used or set())
                if child_reps.get(c) is not None
                and child_reps[c].grad is not None
            ]
            bwd_tensors = (
                [memo[nid][0] for nid in used]
                + [memo[c][1] for c in child_used]
            )
            bwd_grads = (
                [splice_reps[nid].grad for nid in used]
                + [child_reps[c].grad for c in child_used]
            )
            if bwd_tensors:
                with self._timed(_t, "final_encoder_backward", gpu=True):
                    torch.autograd.backward(
                        tensors=bwd_tensors,
                        grad_tensors=bwd_grads,
                    )
            if enc_params:
                clip_grad_norm(enc_params, max_norm=max_grad_norm)
            self._option_a_step_groups(enc_idx)
            # Null ALL grads so the base loop's trailing step no-ops.
            self._option_a_zero_grads(dec_params)
            self._option_a_zero_grads(enc_params)
            self._mem_breakdown(
                "option_a_after_final_encoder_backward", n_used_nodes=len(used),
            )
        finally:
            self._per_repo_shared_tree_active = False
            self._shared_tree_memo = None
            self._shared_tree_splice_reps = None
            self._shared_tree_used_nodes = None
            self._shared_tree_child_l1_reps = None
            self._shared_tree_child_l1_used = None

        # θ dual-ascent (drained by _post_optimizer_step). No survivorship-aux
        # backward in the per-repo config (aux off).
        self._apply_utility_grad_bce_phase2()
        if self.topic_embeddings is not None:
            self.topic_embeddings.apply_gradient_averaging()

        if _prof:
            repo_wall_s = time.perf_counter() - _wall0
            logger.info(
                "per_repo_timing",
                mode="option_a",
                repo=root_node_id,
                n_file_samples=len(batch),
                n_contrib=n_contrib,
                n_subsets=n_subsets,
                n_decoder_steps=n_decoder_steps,
                subset_size=int(self._per_repo_inner_subset_size),
                group_size=group_size,
                repo_wall_s=round(repo_wall_s, 4),
                shared_tree_encode_s=round(_t.get("shared_tree_encode", 0.0), 4),
                pass1_count_s=round(_t.get("pass1_count", 0.0), 4),
                subset_decode_fwd_s=round(_t.get("subset_decode_fwd", 0.0), 4),
                drill_encode_s=round(_t.get("drill_encode", 0.0), 4),
                decode_fwd_s=round(_t.get("decode_fwd", 0.0), 4),
                subset_backward_s=round(_t.get("subset_backward", 0.0), 4),
                decoder_step_s=round(_t.get("decoder_step", 0.0), 4),
                final_encoder_backward_s=round(
                    _t.get("final_encoder_backward", 0.0), 4,
                ),
            )

        return {
            "loss": torch.tensor(total_loss_val / n_samples, device=self.device),
            "tokens": float(total_tokens / n_samples),
            "live_l0": 1.0 if self._live_l0 else 0.0,
            "option_a": 1.0,
            "per_repo_file_samples": float(len(batch)),
            "per_repo_contributing_samples": float(n_contrib),
            "per_repo_backwarded_samples": float(n_samples),
            "option_a_n_subsets": float(n_subsets),
            "option_a_decoder_steps": float(n_decoder_steps),
            "per_repo_shared_tree_nodes": float(_stats.get("nodes", 0)),
            "l1_turns_per_sample": float(n_turns_total / max(1, n_samples)),
            "recursive_l0_retention": self._recursive_l0_retention_now(),
            "recursive_l1_retention": self._recursive_l1_retention_now(),
            # Drill-side retention visibility (== the recursive values when the
            # query-conditioned-drill overrides are unset).
            "drill_node_retention": self._drill_node_retention_now(),
            "drill_leaf_l1_retention": float(l1_ramp),
            # proj_norm_ratio_{family}: mean spliced-rep / embed norm-ratio (drift
            # canary — should track the 4.2 (qwen) / 0.9 (falcon) readable band).
            **self._proj_norm_reg_step_metrics(),
        }

    def _forward_backward(self, batch) -> dict[str, float]:
        """Cross-sample batched forward + backward.

        Every sample in the training batch has ~3-4 bgkit turns. We pool
        ALL turns from ALL samples, bucket them by content length (power
        of 2), and run each bucket as a single packed (varlen) encoder forward.
        Then each sample is assembled into segments using the pre-computed
        L1 outputs and the decoder forward runs per-sample. This collapses
        (batch_size * turns_per_sample) encoder launches into at most a
        handful of bucket forwards per training step.
        """
        if not batch:
            return {"loss": torch.zeros((), device=self.device), "tokens": 0.0}

        # PER-REPO full-backprop: a batch is one repo's file-samples. Encode
        # the shared window-0 tree ONCE and grad-accumulate the file-samples'
        # decoder losses through it (retain_graph). Distinct enough from the
        # cross-sample bucketed path to warrant its own method.
        if getattr(self, "_per_repo_full_backprop", False):
            # Option A (crash-free amortized): whole-repo batch → K decoder
            # updates (per subset, reading detached R_d) + ONE encoder update
            # (final tree-backward). Owns its optimizer cadence internally; the
            # base loop's trailing step no-ops (all grads None at return).
            if getattr(self, "_per_repo_option_a", False):
                return self._forward_backward_option_a(batch)
            # Inner-loop (Model B): post-warmup, the sampler emits subset-batches
            # and we reuse a once-encoded LIVE tree across the repo's K subsets.
            if getattr(self, "_inner_loop_active", False):
                return self._forward_backward_inner_loop(batch)
            # Warmup / one-step: whole repo → one detach-reaccumulate step.
            return self._forward_backward_per_repo(batch)

        # Round-robin: one decoder family for the whole batch. The L1 encode is
        # batched across all samples under the active projection family and then
        # decoded per-sample, so the family must be fixed for the step (matches
        # the summarization trainer's per-microbatch routing).
        if self._round_robin:
            self._set_active_decoder(self._pick_decoder_family())

        # Reset per-step accumulators for survivorship aux losses
        self._pending_l0_outputs: list[dict] = []
        self._pending_l1_outputs: list[dict] = []
        self._step_sampled_l0_ratios = []
        self._step_sampled_l1_ratios = []
        self._proj_norm_ratio_accum = {}

        # Stamp per-batch tag usage so the topic embeddings can divide
        # each tag parameter's gradient by the number of batch members
        # that referenced it (averaging instead of summing). No-op when
        # topic embeddings are disabled. Must be done before backward;
        # the divide itself happens in apply_gradient_averaging() below.
        if self.topic_embeddings is not None:
            self.topic_embeddings.record_batch_usage(
                [self._sample_tags_for(s) for s in batch],
            )

        # Phases 1-3 (per-sample prep + bucketed L1 + per-sample assemble+decode,
        # summed into one loss) are shared with the per-repo PASS 2 via
        # _encode_decode_group. The regular path encodes the WHOLE batch as one
        # group (l1_target_ratio=None → sampled L1 retention).
        # Per-phase wall-clock splits (prep / drill_encode / assemble /
        # decode_fwd from _encode_decode_group, + aux / backward / util here)
        # emitted as ``time/<phase>`` step metrics — the in-process profile
        # that decides where a slow step's time goes (perf playbook: measure
        # before hypothesising). GPU-synced splits; negligible vs a step.
        import time as _time

        step_timing: dict[str, float] = {}
        total_loss, total_tokens, n_samples, n_turns_total, n_buckets = (
            self._encode_decode_group(batch, timing=step_timing, l1_target_ratio=None)
        )

        if n_samples == 0:
            return {"loss": torch.zeros((), device=self.device), "tokens": 0.0}

        decoder_loss = total_loss / n_samples
        total_weighted = decoder_loss

        # Survivorship auxiliary losses over accumulated L0 + L1 outputs
        _t0 = _time.perf_counter()
        aux_loss, aux_metrics = self._compute_survivorship_aux_losses()
        if aux_loss.requires_grad or aux_loss.item() != 0.0:
            total_weighted = total_weighted + aux_loss
        step_timing["aux"] = step_timing.get("aux", 0.0) + (_time.perf_counter() - _t0)

        _t0 = _time.perf_counter()
        (total_weighted / self._accum_steps).backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_timing["backward"] = step_timing.get("backward", 0.0) + (
            _time.perf_counter() - _t0
        )

        # --- Utility-gradient BCE distillation (post-backward) ---
        # The main backward populated each LevelOutput's captured
        # ``post_head_content_grad`` via the level's backward hook.
        # Rebuild the top-k teacher from ``-(grad · value)`` and run a
        # small head-local backward per level.
        _t0 = _time.perf_counter()
        util_metrics = self._apply_utility_grad_bce_phase2()
        step_timing["util"] = step_timing.get("util", 0.0) + (_time.perf_counter() - _t0)

        # Average shared-tag gradients across the batch (see
        # TopicEmbeddingModule.apply_gradient_averaging). Only touches
        # parameters whose tags appeared in record_batch_usage above.
        if self.topic_embeddings is not None:
            self.topic_embeddings.apply_gradient_averaging()

        metrics_out = {
            "loss": decoder_loss.detach(),
            # Per decoder-family train loss (round-robin: one family per
            # microbatch; _average_metrics averages each key over the
            # microbatches it appears in). Without it a family-specific
            # divergence is invisible in the pooled `loss`.
            f"loss/{getattr(self, '_decoder_family', 'single') or 'single'}": (
                decoder_loss.detach()
            ),
            "tokens": float(total_tokens / n_samples),
            "l1_retention": self._l1_retention,
            "live_l0": 1.0 if self._live_l0 else 0.0,
            "l1_turns_per_sample": float(n_turns_total / n_samples),
            "l1_buckets_per_step": float(n_buckets),
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
        metrics_out.update(self._proj_norm_reg_step_metrics())
        for _k, _v in step_timing.items():
            metrics_out[f"time/{_k}"] = float(_v)

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

    def _accumulate_theta_state(self, level: str, enc_out, ratio: float) -> None:
        """Accumulate ONLY the dual-ascent θ control statistics (per-microbatch
        scalar keep-rate sum/count + target-ratio mass) for ``level`` directly
        into the per-level :class:`MicrobatchAggState`, WITHOUT retaining
        ``enc_out`` in ``_pending_l{0,1}_outputs``.

        Used when survivorship aux is disabled (``survivorship_aux: false``):
        θ keeps adapting to the retention ramp with zero activation retention.
        ``accumulate`` reads only zero-dim count scalars off ``enc_out`` (no
        autograd graph), so this is memory-free. The accumulated
        ``target_ratio`` mass lets :meth:`_run_dual_ascent` use the true
        batch-weighted target under a ramp without a pinned ``target_ratios``.
        """
        from bgkit.training.survivorship_helpers import accumulate

        state = getattr(
            self, "_surv_state_l0" if level == "l0" else "_surv_state_l1", None,
        )
        if state is None:
            return
        accumulate(state, enc_out, target_ratio=float(ratio))

    def _accumulate_theta_from_counts(
        self, level: str, counts: torch.Tensor, ratio: float,
    ) -> None:
        """θ accumulation from a detached ``[organic, controllable, valid]``
        counts tensor (FIX 2: lets the per-node L1 forward be checkpointed
        while θ is accumulated ONCE outside the checkpoint — no double-count on
        recompute). Wraps the counts in a minimal enc_out-shaped namespace and
        reuses :meth:`_accumulate_theta_state`'s ``accumulate`` semantics."""
        from types import SimpleNamespace
        org, ctrl, valid = (counts[0], counts[1], counts[2])
        self._accumulate_theta_state(
            level,
            SimpleNamespace(
                organic_count=org, controllable_count=ctrl, valid_count=valid,
            ),
            ratio,
        )

    def _theta_skip_levels(self) -> tuple[str, ...]:
        """Levels whose dual-ascent θ step is skipped (accumulator still drained).

        - ``l0`` when L0 is frozen (Stage B cached L0).
        - ``l1`` when L1 selection is ``exact_topk`` (BUG-1): the keep-rate is
          set by deterministic per-sample top-k at the target ratio, so θ is
          irrelevant — stepping it would chase a threshold that no longer gates
          selection. Skipping avoids drift; the accumulator is still drained so
          the keep-rate/mean-rate metrics keep logging.
        """
        skip: list[str] = []
        if not self._live_l0 or getattr(self, "_selection_mode_l0", "threshold") == "exact_topk":
            skip.append("l0")
        if getattr(self, "_selection_mode_l1", "threshold") == "exact_topk":
            skip.append("l1")
        return tuple(skip)

    def _post_optimizer_step(self, step: int) -> None:
        """Run dual-ascent θ + EMA μ updates per level using true-mean aggregation.

        Stage A: live L0 → update both L0 and L1.
        Stage B: cached L0 with L0 LoRA frozen → skip L0 updates.
        """
        from bgkit.training.survivorship_helpers import accumulate, maybe_unload_ice

        # Decoupled θ path: when survivorship aux is OFF the per-level state was
        # already accumulated per-microbatch at encode time (no _pending), so
        # just run dual-ascent (drains + resets the state). target_ratios=None
        # → the accumulated per-microbatch target-ratio mass drives the target
        # (correct under the L0/L1 retention ramps).
        if not getattr(self, "_survivorship_aux", True):
            self._run_dual_ascent(
                step,
                target_ratios=None,
                skip_levels=self._theta_skip_levels(),
            )
            unloaded = maybe_unload_ice(
                getattr(self, "_ice_teacher", None),
                step,
                getattr(self, "_max_warmup_step", 0),
            )
            if unloaded:
                logger.info("ice_teacher_unloaded", step=step)
            return

        # Accumulate pending L0/L1 outputs into per-level state, then apply
        # via the shared dual-ascent dispatch (CompressionCurriculumMixin).

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
            # No pending controllable counts (for example, an all-cached L0
            # batch). Resolve curriculum dictionaries at the current step
            # instead of calling float() on the raw config value.
            vals = [
                self._l0_retention_for(dataset)
                for dataset in self._l0_retention
            ]
            target_l0 = sum(vals) / len(vals) if vals else 0.10
        if l1_target_den > 0:
            target_l1 = l1_target_num / l1_target_den
        else:
            target_l1 = float(self._l1_retention)

        # L0 update skipped at Stage B (cached L0, L0 frozen); L1 always updates
        # UNLESS L1 runs exact-topk selection (BUG-1) — then θ is irrelevant.
        self._run_dual_ascent(
            step,
            target_ratios={"l0": target_l0, "l1": target_l1},
            skip_levels=self._theta_skip_levels(),
        )

        unloaded = maybe_unload_ice(
            getattr(self, "_ice_teacher", None),
            step,
            getattr(self, "_max_warmup_step", 0),
        )
        if unloaded:
            logger.info("ice_teacher_unloaded", step=step)

    def _add_step_metrics(self, metrics: dict[str, float]) -> None:
        """Attach post-step θ/μ updates (without clobbering base keys)."""
        self._inject_survivorship_metrics(metrics)  # CompressionCurriculumMixin
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

        When ``training.eval_ablation_modes`` is non-empty (e.g.
        ``[zeroed, noise]``), runs additional eval passes with each
        ablation mode applied and re-emits every metric under
        ``eval/ablation/{mode}/...``. Headline ``eval/...`` keys always
        reflect the no-ablation (``present``) pass. Each extra mode
        roughly doubles eval wall-clock; sized by ``max_eval_samples``.
        """
        self.model.eval()
        # Per-evaluate() bounded LRU of encoded repo trees. Keeping this small is
        # essential: each value contains GPU survivor tensors for a whole
        # history window. Cleared in finally.
        self._eval_tree_cache = {}
        try:
            with self._teacher_forced_decoders():
                metrics = self._eval_pass()
                # Generalization-gap probe: the SAME teacher-forced pass over a
                # fixed, evenly spaced slice of TRAIN samples, keyed
                # ``eval_train/...`` (per dataset + per decoder family). A
                # family whose eval loss sits far above its train-subset loss
                # is generalizing badly; one whose train-subset loss also sits
                # far above its logged training loss has an eval-PATH defect
                # (2026-08-22: Falcon-H1 eval 6.3 vs train ~2.6 — which?).
                n_train_probe = int(self.step_cfg.get("eval_train_subset_samples", 0) or 0)
                if n_train_probe > 0:
                    metrics.update({
                        k.replace("eval/", "eval/train_subset/", 1): v
                        for k, v in self._eval_pass(
                            loader=self._train_subset_batches(n_train_probe),
                        ).items()
                    })
            free_running_samples = int(
                self.step_cfg.get("eval_free_running_samples", 0) or 0
            )
            if free_running_samples > 0:
                # Generation needs the cached (eval-mode) decoder path.
                metrics.update(self._free_running_metrics_guarded(free_running_samples))
            extra_modes = list(self.step_cfg.get("eval_ablation_modes", []) or [])
            # Restore the mode evaluate() was ENTERED with, not None — the
            # standalone eval script may run the whole evaluation under an
            # explicit mode (e.g. oracle_span); restoring None here would
            # silently strip it from every pass after the sweep (2026-08-24).
            entering_mode = self._ablation_mode
            for mode in extra_modes:
                mode_str = str(mode)
                self.set_ablation_mode(mode_str)
                try:
                    with self._teacher_forced_decoders():
                        sub = self._eval_pass()
                finally:
                    self.set_ablation_mode(entering_mode)
                for k, v in sub.items():
                    if k.startswith("eval/"):
                        metrics[f"eval/ablation/{mode_str}/{k[len('eval/'):]}"] = v
            # recon/nav ablation GAPS: how much reconstruction / navigation CE
            # DEGRADES when the survivor reps are zeroed. gap ≈ 0 ⇒ reps decorative
            # (the shortcut); gap > 0 ⇒ reps load-bearing (the objective). Present
            # only when eval_gap_probe is on AND 'zeroed' is in eval_ablation_modes.
            base_r = metrics.get("eval/recon_loss")
            base_n = metrics.get("eval/nav_loss")
            z_r = metrics.get("eval/ablation/zeroed/recon_loss")
            z_n = metrics.get("eval/ablation/zeroed/nav_loss")
            if base_r is not None and z_r is not None:
                metrics["eval/recon_gap"] = z_r - base_r
            if base_n is not None and z_n is not None:
                metrics["eval/nav_gap"] = z_n - base_n
        finally:
            self._clear_eval_shared_tree()
            self.model.train()
        metrics.update(self._lm_health_metrics())
        return metrics

    def _lm_health_metrics(self) -> dict[str, float]:
        """Held-out plain-text CE for the active decoder(s).

        The wide-net runs took this decoder from PPL 33 to 2585 while eval
        loss, token accuracy, exact match AND a zeroed-rep ablation all
        looked merely mediocre — nothing in-distribution can see catastrophic
        forgetting, because every in-distribution metric is measured on the
        distribution being overfit. Watch DRIFT from the first eval's value,
        not an absolute threshold: the healthy starting point is
        lineage-specific (stock 15, summarization base 33).
        """
        n = int(self.step_cfg.get("lm_health_samples", 0) or 0)
        if n <= 0:
            return {}
        from bgkit.training.lm_health import lm_health_metrics, load_health_chunks

        try:
            if getattr(self, "_lm_health_chunks", None) is None:
                tokens_dir = self.step_cfg.get("lm_health_tokens_path", None) or (
                    self.cfg.training.data.file_tokens_path
                )
                self._lm_health_chunks = load_health_chunks(
                    tokens_dir,
                    n_docs=n,
                    seq_len=int(self.step_cfg.get("lm_health_seq_len", 1024)),
                )
            out: dict[str, float] = {}
            families = getattr(self, "_decoders_by_family", None) or {
                getattr(self, "_decoder_family", "decoder"): self.decoder
            }
            for family, dec in families.items():
                out.update(
                    lm_health_metrics(
                        dec, self._lm_health_chunks, self.device,
                        prefix=f"eval/lm_health/{family}",
                    )
                )
            return out
        except Exception as exc:  # diagnostics must never kill a run
            logger.warning("lm_health_failed", error=str(exc))
            return {}

    def _grad_norm_param_groups(self) -> dict[str, list]:
        """Pre-clip gradient-norm groups: backbones, the small survivorship
        heads/controllers, the L0→L1 bridge, projection blocks, decoders."""
        groups: dict[str, list] = {}
        enc = getattr(self, "encoder", None)
        for lvl in ("l0", "l1"):
            lc = getattr(enc, lvl, None) if enc is not None else None
            if lc is None:
                continue
            bb = getattr(lc, "backbone", None)
            if bb is not None:
                groups[f"{lvl}_backbone"] = [p for p in bb.parameters() if p.requires_grad]
            head_params: list = []
            for sub in ("head", "threshold"):
                m = getattr(lc, sub, None)
                if m is not None:
                    head_params += [p for p in m.parameters() if p.requires_grad]
            groups[f"{lvl}_head"] = head_params
            # The survive flag vector is added at EVERY survivor position, so
            # its gradient is a sum over thousands of positions — reported on
            # its own so it can't masquerade as the head's gradient.
            se = getattr(lc, "survive_embedding", None)
            if se is not None and getattr(se, "requires_grad", False):
                groups[f"{lvl}_survive_embedding"] = [se]
            br = getattr(lc, "auto_repro_head", None)
            if br is not None:
                groups[f"{lvl}_bridge"] = [p for p in br.parameters() if p.requires_grad]
        pb = getattr(enc, "projection_blocks", None) if enc is not None else None
        if pb is not None:
            groups["projection"] = [p for p in pb.parameters() if p.requires_grad]
        for fam, dec in (getattr(self, "_decoders_by_family", None) or {}).items():
            groups[f"decoder_{fam}"] = [p for p in dec.parameters() if p.requires_grad]
        return groups

    def _eval_family_for_index(self, index: int) -> str:
        """Decoder family for the ``index``-th eval sample under round-robin.

        Follows the TRAINING mix: ``qwen_decoder_prob >= 1`` → Qwen only,
        ``<= 0`` → Falcon only, otherwise deterministic 50/50 alternation.
        (2026-08-23: v6 trains Qwen-only, yet the eval alternated families,
        so half the pooled eval scored an untrained Falcon — eval/loss 10.0
        with Qwen at 1.88 — and half the eval wall-clock was wasted.)
        """
        prob = float(getattr(self, "_qwen_decoder_prob", 0.5))
        if prob >= 1.0 - 1e-9:
            return "qwen35"
        if prob <= 1e-9:
            return "falcon_h1"
        return "qwen35" if index % 2 == 0 else "falcon_h1"

    def _train_subset_batches(self, n: int) -> list[list]:
        """Fixed, evenly spaced ``n`` training samples as single-sample
        batches (the shape ``_eval_pass`` iterates) for the train-subset
        generalization-gap probe. Deterministic across evals so the numbers
        are comparable over the run."""
        ds = self.train_dataset
        total = len(ds)
        if total == 0 or n <= 0:
            return []
        n = min(n, total)
        stride = max(1, total // n)
        return [[ds[i]] for i in range(0, stride * n, stride)][:n]

    @contextlib.contextmanager
    def _teacher_forced_decoders(self):
        """Run the DECODER backbones in ``train()`` mode for teacher-forced
        eval (the encoder stays in eval mode; callers hold ``torch.no_grad``).

        HF "eval mode" assumes autoregressive inference with a cache.
        Falcon-H1's Mixer gates on ``self.training``: eval mode takes the
        unfused conv/scan path and additionally applies padding masking to
        the projected states — numerically DIFFERENT from the fused training
        path, which produced a 5-nat eval/train gap in the summarization
        trainer (2026-05-15) and, here, eval/loss 9.8 vs train ~1 with a zero
        reps-ablation gap at v5b step 250 (2026-08-22). Teacher-forced
        full-sequence eval must use the same forward as training; generation
        passes (which need the cache) keep eval mode. Neither decoder family
        has active dropout, so train mode changes nothing else.
        """
        decs = list(self._all_decoders())
        prev = [bool(d.training) for d in decs]
        for d in decs:
            d.train()
        try:
            yield
        finally:
            for d, was_training in zip(decs, prev, strict=True):
                d.train(was_training)

    def _free_running_metrics_guarded(self, max_samples: int) -> dict[str, float]:
        """The free-running probe is a DIAGNOSTIC: a defect in it must not
        take the training run down (2026-08-23: an exception inside it killed
        v6 between saves, 227 steps lost). Failures are logged with the
        traceback (`free_running_eval_failed`, visible to monitors) and
        reported as ``eval/kb/free_running/failed = 1``.
        """
        try:
            metrics = self._eval_free_running_pass(max_samples)
        except Exception:
            logger.exception("free_running_eval_failed", max_samples=max_samples)
            return {"eval/kb/free_running/failed": 1.0}
        metrics["eval/kb/free_running/failed"] = 0.0
        return metrics

    def _eval_free_running_pass(self, max_samples: int) -> dict[str, float]:
        """Generate autonomous tool calls and answers for a bounded eval slice.

        Unlike :meth:`_eval_pass`, this never renders future teacher calls or
        the gold answer. It is deliberately run once, without representation
        ablations, because generation is substantially more expensive than a
        teacher-forced forward.
        """
        from bgkit.eval.kb_trajectory_eval import evaluate_free_running_sample

        max_tool_calls = int(self.step_cfg.get("eval_free_running_max_tool_calls", 16))
        max_new_tokens = int(self.step_cfg.get("eval_free_running_max_new_tokens", 8192))
        fields = (
            "route_exact",
            "valid_navigation",
            "evidence_recall",
            "answer_token_f1",
            "answer_exact_match",
        )
        sums = dict.fromkeys(fields, 0.0)
        # Per decoder family AND per dataset (the generative needle gate is a
        # per-task question: fileneedle/grepset copy a line, lognav a log
        # line, swerecall a path — a pooled EM hides which one moves).
        group_sums: dict[str, dict[str, float]] = {}
        group_counts: dict[str, int] = {}
        invalid_reasons: dict[str, int] = {}
        invalid = 0
        seen = 0
        examples_logged = 0
        self._eval_shared_tree_key = None
        # Evenly spaced over the eval set: it is index-sorted (datasets
        # concatenated), so the first ``max_samples`` would all come from one
        # dataset (2026-08-23: 64/64 lognav).
        n_eval = len(self.eval_dataloader)
        stride = max(1, n_eval // max_samples) if max_samples > 0 else 1
        for batch_index, batch in enumerate(self.eval_dataloader):
            if batch_index % stride:
                continue
            for sample in batch:
                if seen >= max_samples:
                    break
                if self._round_robin:
                    # Class-qualified so partial test doubles (a ``_Stub`` that
                    # borrows this method) resolve the policy too.
                    self._set_active_decoder(KRKBTrainer._eval_family_for_index(self, seen))
                result = evaluate_free_running_sample(
                    self,
                    sample,
                    max_tool_calls=max_tool_calls,
                    max_new_tokens=max_new_tokens,
                )
                family = str(getattr(self, "_decoder_family", "single"))
                dataset = str(getattr(sample, "dataset_name", "") or "unknown")
                for group in (family, dataset):
                    group_sums.setdefault(group, dict.fromkeys(fields, 0.0))
                    group_counts[group] = group_counts.get(group, 0) + 1
                for field in fields:
                    value = float(result.get(field, 0.0))
                    sums[field] += value
                    group_sums[family][field] += value
                    group_sums[dataset][field] += value
                reason = str(result.get("invalid_reason") or "")
                if reason:
                    invalid += 1
                    key = reason.split(":", 1)[0]
                    invalid_reasons[key] = invalid_reasons.get(key, 0) + 1
                # A few raw generations per eval so a verdict (invalid OR a
                # zero-EM answer) can be read against what the model wrote.
                if examples_logged < 6 and (reason or seen % 8 == 0):
                    examples_logged += 1
                    logger.info(
                        "free_running_example",
                        dataset=dataset,
                        family=family,
                        invalid_reason=reason,
                        gold=str(result.get("gold_answer", ""))[:160],
                        pred=str(result.get("pred_answer", ""))[:160],
                        raw=str(result.get("raw_text", ""))[:200],
                    )
                seen += 1
            if seen >= max_samples:
                break

        prefix = "eval/kb/free_running"
        metrics = {
            f"{prefix}/{field}": sums[field] / seen if seen else 0.0
            for field in fields
        }
        metrics[f"{prefix}/invalid_rate"] = invalid / seen if seen else 0.0
        metrics[f"{prefix}/n_samples"] = float(seen)
        for key, count in invalid_reasons.items():
            metrics[f"{prefix}/invalid/{key}"] = count / seen if seen else 0.0
        for group, values in group_sums.items():
            count = group_counts[group]
            for field in fields:
                metrics[f"{prefix}/{group}/{field}"] = values[field] / count
            metrics[f"{prefix}/{group}/n_samples"] = float(count)
        return metrics

    def _ensure_eval_shared_tree(self, sample: KBSample) -> None:
        """EVAL: install the shared repo tree so recursive drill reps resolve.

        The coherent per-repo step encodes the window subtree once per group
        (:meth:`_compute_shared_repo_tree`) and installs ``_shared_tree_memo`` /
        ``_shared_tree_splice_reps``, so every ``bgkit`` drill splices a REAL
        survivor rep. Eval's single-sample decode path never did this, so every
        ``node`` / ``head`` drill fell through to
        :meth:`_drilldown_zero_survivor` (ZERO reps). That both starves the eval
        decode (understating quality) AND pins ``eval/recon_gap`` +
        ``eval/nav_gap`` at ~0 — a MEASUREMENT ARTIFACT, not a training collapse:
        the in-step ablation probe, which runs with the tree live, shows the
        reps are load-bearing (recon_gap ~0.1). Populate the same state here
        (under no_grad) so eval metrics + the ablation gaps reflect the real
        rep-using decode.

        Gated to the recursive per-repo full-backprop path (git_commit_repro);
        flat QA datasets keep the ``_run_l1_batch`` splice and are untouched.
        Memoized by root so consecutive same-repo eval samples reuse the encode.
        """
        if not getattr(self, "_per_repo_full_backprop", False):
            return
        root = self._repo_group_key(sample)
        if not root:
            return
        # Key by (root, decoder_family): under round-robin, _eval_pass alternates
        # the active decoder family PER SAMPLE (_set_active_decoder), and each
        # family's projection block emits survivors at THAT family's hidden dim
        # (qwen35 1024 vs falcon_h1 512). The shared tree reps are therefore
        # family-specific — memoizing by root alone reuses one family's reps
        # against the other family's decoder → the "survivor hidden dim != decoder
        # hidden dim" crash. A repo with both families in the eval set encodes its
        # tree once per family (still cheap vs no cache).
        family = getattr(self, "_decoder_family", None)
        key = (root, family)
        if getattr(self, "_eval_shared_tree_key", None) == key:
            return
        # Cross-ablation bounded LRU. The encoded tree is identical across
        # modes, but an unbounded cache retained every eval repository's GPU
        # tensors until evaluate() returned and could OOM before the first pass
        # finished. A small cache still helps repeated adjacent groups without
        # making memory scale with eval-set cardinality.
        cache = getattr(self, "_eval_tree_cache", None)
        if cache is not None and key in cache:
            memo, splice = cache.pop(key)
            cache[key] = (memo, splice)
        else:
            saved_count = getattr(self, "_shared_tree_forward_count", 0)
            with torch.no_grad():
                memo, _stats = self._compute_shared_repo_tree(
                    sample.dataset_name, root,
                )
            # The eval encode is not a training-step forward — don't inflate the
            # once-per-step tripwire counter.
            self._shared_tree_forward_count = saved_count
            splice = {
                nid: proj.detach()
                for nid, (proj, _l1out) in memo.items()
                if proj is not None
            }
            step_cfg = getattr(self, "step_cfg", {})
            max_cached = max(
                0, int(step_cfg.get("eval_tree_cache_max_groups", 1) or 0),
            )
            if cache is not None and max_cached > 0:
                cache[key] = (memo, splice)
                while len(cache) > max_cached:
                    oldest = next(iter(cache))
                    cache.pop(oldest)
        self._shared_tree_memo = memo
        self._shared_tree_splice_reps = splice
        self._shared_tree_used_nodes = set()
        # None → the head survivor reads memo[c][1] directly (the eval path); the
        # per-group reaccumulate bookkeeping is a training-backward concern only.
        self._shared_tree_child_l1_reps = None
        self._shared_tree_child_l1_used = None
        self._per_repo_shared_tree_active = True
        self._eval_shared_tree_key = key

    def _clear_eval_shared_tree(self) -> None:
        """Tear down any eval-installed shared repo tree (see
        :meth:`_ensure_eval_shared_tree`) so a following training step starts
        from a clean shared-tree state."""
        self._shared_tree_memo = None
        self._shared_tree_splice_reps = None
        self._shared_tree_used_nodes = None
        self._per_repo_shared_tree_active = False
        self._eval_shared_tree_key = None
        self._eval_tree_cache = None

    def _eval_pass(self, loader=None) -> dict[str, float]:
        """Single pass over the eval dataloader. Assumes the model is
        already in eval mode and any ablation_mode is set. Returns the
        standard ``eval/...`` metrics dict for whatever ablation_mode is
        currently active.

        ``loader`` overrides the sample source (an iterable of sample lists)
        — used by the train-subset generalization-gap pass in
        :meth:`evaluate`, whose keys are re-prefixed ``eval_train/``.
        """
        total_loss_weighted = 0.0
        total_tokens = 0
        n_samples = 0
        total_correct = 0
        answer_correct = 0
        answer_tokens = 0

        per_dataset_correct: dict[str, int] = {}
        per_dataset_total: dict[str, int] = {}
        per_dataset_em: dict[str, list[float]] = {}
        per_dataset_f1: dict[str, list[float]] = {}
        all_em: list[float] = []
        all_f1: list[float] = []

        # KB-harness trajectory metric accumulators. Tool-call ID
        # accuracy is micro-averaged across bgkit calls by summing the
        # per-call scores and dividing by total call counts; answer F1 is
        # macro-averaged across samples with an answer.
        kb_bgkit_sum = 0.0
        kb_bgkit_n = 0
        kb_f1_sum = 0.0
        kb_f1_n = 0
        self._eval_family_counter = 0
        # Per decoder-family breakdown (round-robin runs): a family-specific
        # eval-path defect is invisible in the pooled numbers.
        fam_loss_w: dict[str, float] = {}
        fam_tokens: dict[str, int] = {}
        fam_f1: dict[str, list[float]] = {}
        fam_em: dict[str, list[float]] = {}

        # Optional recon/nav CE split accumulator (eval/recon_loss + eval/nav_loss).
        # evaluate() diffs these across ablation modes to get eval/recon_gap + eval/nav_gap.
        gap_probe = bool(self.step_cfg.get("eval_gap_probe", False))
        span_accum = (
            {"nav_sum": 0.0, "nav_count": 0, "recon_sum": 0.0, "recon_count": 0}
            if gap_probe else None
        )

        # Fresh per pass: evaluate() calls _eval_pass once per ablation mode, so
        # the shared tree is re-installed each pass (reps are identical across
        # modes — the ablation is applied at splice, not encode). Torn down in
        # evaluate()'s finally via _clear_eval_shared_tree.
        self._eval_shared_tree_key = None
        for batch in (loader if loader is not None else self.eval_dataloader):
            for sample in batch:
                if self._round_robin:
                    # Deterministic family choice that FOLLOWS the training
                    # mix (see _eval_family_for_index); reproducible across runs.
                    self._set_active_decoder(
                        self._eval_family_for_index(self._eval_family_counter)
                    )
                    self._eval_family_counter += 1
                # Install the sample's shared repo tree so recursive drill reps
                # resolve to real survivors (git_commit_repro); no-op for flat
                # QA datasets. Without this every drill splices ZERO reps.
                self._ensure_eval_shared_tree(sample)
                result = self._eval_one_sample(sample, span_accum)
                if result is None:
                    continue
                sample_loss = result["loss"]
                sample_tokens = result["tokens"]
                total_loss_weighted += float(sample_loss) * sample_tokens
                total_tokens += sample_tokens
                total_correct += result["correct"]
                answer_correct += result["answer_correct"]
                answer_tokens += result["answer_tokens"]
                n_samples += 1
                all_em.append(result["em"])
                all_f1.append(result["f1"])
                ds = sample.dataset_name
                per_dataset_em.setdefault(ds, []).append(result["em"])
                per_dataset_f1.setdefault(ds, []).append(result["f1"])
                per_dataset_correct[ds] = per_dataset_correct.get(ds, 0) + result["correct"]
                per_dataset_total[ds] = per_dataset_total.get(ds, 0) + sample_tokens
                fam = str(getattr(self, "_decoder_family", "single") or "single")
                fam_loss_w[fam] = fam_loss_w.get(fam, 0.0) + float(sample_loss) * sample_tokens
                fam_tokens[fam] = fam_tokens.get(fam, 0) + sample_tokens
                fam_f1.setdefault(fam, []).append(result["f1"])
                fam_em.setdefault(fam, []).append(result["em"])

                # KB harness metric accumulation (tool-call ID
                # accuracy + answer F1). Already computed as part of
                # _eval_one_sample so no extra forward pass.
                tool_call = result.get("tool_call_id", None)
                if tool_call is not None:
                    n_bg = int(tool_call.get("n_bgkit", 0))
                    if n_bg:
                        kb_bgkit_sum += float(tool_call.get("bgkit", 0.0)) * n_bg
                        kb_bgkit_n += n_bg
                if sample.gold_answer:
                    kb_f1_sum += float(result["f1"])
                    kb_f1_n += 1

        if total_tokens == 0:
            return {
                "eval/loss": 0.0,
                "eval/answer_token_accuracy": 0.0,
                "eval/loss_token_accuracy": 0.0,
                "eval/exact_match": 0.0,
                "eval/token_f1": 0.0,
                "eval/n_samples": 0.0,
                "eval/kb/trajectory_step_accuracy": 0.0,
                "eval/kb/tool_call_id_accuracy/bgkit": 0.0,
                "eval/kb/tool_call_id_accuracy/overall": 0.0,
                "eval/kb/answer_token_f1": 0.0,
            }

        kb_bgkit_acc = kb_bgkit_sum / kb_bgkit_n if kb_bgkit_n else 0.0
        kb_overall_acc = kb_bgkit_acc
        kb_f1 = kb_f1_sum / kb_f1_n if kb_f1_n else 0.0

        metrics: dict[str, float] = {
            "eval/loss": total_loss_weighted / total_tokens,
            "eval/loss_token_accuracy": total_correct / total_tokens,
            "eval/answer_token_accuracy": (
                answer_correct / answer_tokens if answer_tokens else 0.0
            ),
            "eval/exact_match": sum(all_em) / len(all_em),
            "eval/token_f1": sum(all_f1) / len(all_f1),
            "eval/n_samples": float(n_samples),
            "eval/tokens_per_sample": float(total_tokens / max(n_samples, 1)),
            # KB trajectory metrics. Micro over tokens / calls; macro
            # over samples (for F1).
            "eval/kb/trajectory_step_accuracy": total_correct / total_tokens,
            "eval/kb/tool_call_id_accuracy/bgkit": kb_bgkit_acc,
            "eval/kb/tool_call_id_accuracy/overall": kb_overall_acc,
            "eval/kb/answer_token_f1": kb_f1,
            "eval/kb/n_bgkit_calls": float(kb_bgkit_n),
        }
        for ds, c in per_dataset_correct.items():
            t = per_dataset_total.get(ds, 0)
            if t:
                metrics[f"eval/{ds}/loss_token_accuracy"] = c / t
        for ds, scores in per_dataset_em.items():
            if scores:
                metrics[f"eval/{ds}/exact_match"] = sum(scores) / len(scores)
        for ds, scores in per_dataset_f1.items():
            if scores:
                metrics[f"eval/{ds}/token_f1"] = sum(scores) / len(scores)
        for fam, t in fam_tokens.items():
            if t:
                metrics[f"eval/family/{fam}/loss"] = fam_loss_w[fam] / t
                metrics[f"eval/family/{fam}/token_f1"] = sum(fam_f1[fam]) / len(fam_f1[fam])
                metrics[f"eval/family/{fam}/exact_match"] = sum(fam_em[fam]) / len(fam_em[fam])
                metrics[f"eval/family/{fam}/n_samples"] = float(len(fam_f1[fam]))
        # recon/nav CE split (present when eval_gap_probe). evaluate() diffs
        # these across ablation modes to form eval/recon_gap + eval/nav_gap.
        if span_accum is not None:
            rc = span_accum["recon_count"]
            nc = span_accum["nav_count"]
            if rc > 0:
                metrics["eval/recon_loss"] = span_accum["recon_sum"] / rc
                metrics["eval/recon_tokens"] = float(rc)
            if nc > 0:
                metrics["eval/nav_loss"] = span_accum["nav_sum"] / nc
                metrics["eval/nav_tokens"] = float(nc)
            # per-drill-depth recon/nav loss (does recon scale with drill depth?)
            for d, bd in sorted(span_accum.get("by_depth", {}).items()):
                if bd["recon_count"] > 0:
                    metrics[f"eval/recon_loss_by_depth/d{d:02d}"] = (
                        bd["recon_sum"] / bd["recon_count"]
                    )
                if bd["nav_count"] > 0:
                    metrics[f"eval/nav_loss_by_depth/d{d:02d}"] = (
                        bd["nav_sum"] / bd["nav_count"]
                    )
                metrics[f"eval/samples_by_depth/d{d:02d}"] = float(bd["n"])
        return metrics

    @torch.no_grad()
    def _eval_one_sample(self, sample: KBSample, span_accum: dict | None = None) -> dict | None:
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
            _is_scored_bgkit_turn,
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

        # Optional recon/nav CE split (for eval/recon_gap + eval/nav_gap): reuse
        # the same trace-span splitter as the training-step ablation probe.
        if span_accum is not None:
            self._accumulate_span_ce(span_accum, output, trace)

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
            answer_correct = 0
            answer_tokens = 0
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
                answer_correct = int((pred_ids == gold_ids).sum().item())
                answer_tokens = int(gold_ids.numel())
                pred_text = self.tokenizer.decode(
                    pred_ids, skip_special_tokens=True,
                )
                gold_text = self.tokenizer.decode(
                    gold_ids, skip_special_tokens=True,
                )
            else:
                pred_text = ""
                gold_text = str(sample.gold_answer)
                answer_correct = 0
                answer_tokens = 0
            em = exact_match(pred_text, [gold_text])
            f1 = token_f1(pred_text, [gold_text])

        # Per-call tool-ID accuracy, scored via the KB trajectory eval
        # helper functions so the scoring logic lives in one place.
        bgkit_scores: list[float] = []
        for turn, span in zip(
            trace.bgkit_turns, trace.bgkit_call_spans, strict=True,
        ):
            if not _is_scored_bgkit_turn(turn):
                continue
            bgkit_scores.append(
                _score_call_span(
                    preds, shift_m, span,
                    _tool_call_gold_ids(turn), self.tokenizer,
                )
            )
        n_bgkit = len(bgkit_scores)
        bgkit_acc = sum(bgkit_scores) / n_bgkit if n_bgkit else 0.0
        tool_call_id = {
            "bgkit": bgkit_acc,
            "overall": bgkit_acc,
            "n_bgkit": n_bgkit,
        }

        return {
            "loss": float(output.loss.item()),
            "tokens": tokens,
            "correct": correct,
            "answer_correct": answer_correct,
            "answer_tokens": answer_tokens,
            "em": em,
            "f1": f1,
            "pred_text": pred_text,
            "gold_text": gold_text,
            "tool_call_id": tool_call_id,
        }
