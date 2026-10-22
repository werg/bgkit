"""Structured config for the bgkit2 trainer. Loaded from YAML + CLI dotlist."""

from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import MISSING, OmegaConf


@dataclass
class LoraCfg:
    enabled: bool = False
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.0
    layers: list[int] = field(default_factory=lambda: list(range(8)))
    # Adapter delta at rep positions only (see models.decoder.LoraSpec).
    rep_gated: bool = False


@dataclass
class EncoderAdapterCfg:
    """``optim.encoder_mode: adapter`` (2026-09-22, user: "definitely we want to
    bring the encoder back"): a peft LoRA "slots" on the encoder backbone plus
    trainable COPIES of the head, survive embedding, prompt separator /
    conditioning, projection block, projection and interface norm
    (``BgKIT2Encoder.inject_slot_adapter``). Active for every ordinary slot and
    the compaction spine; OFF on the cached-tree path, whose base weights stay
    frozen, so cached tree roots remain exactly the base encoder's output.

    Rates follow the old encoder-LoRA runs (plans/reset_2026_09_03.md): rank
    32 / alpha 64 on every layer at 5e-5 ran P7b/P7c 4000 steps with 0
    non-finite steps and no collapse under a hard row-norm interface; 1e-4
    exploded at step 475 (P7, EMA interface). Opening the raw backbone
    collapsed the reps inside 1000 steps (P2-P5), hence an adapter. Default
    footprint is CONSERVATIVE (user 2026-09-22: "careful about memory ... but we
    do want the improving promptability and gradient flow"): the TOP 4 backbone
    layers only (P29/W3 trained layers 12-15 of 16) at rank 16 -- enough for the
    prompt to modulate the consolidated encoding and for gradient to reach it,
    far cheaper than the full stack. The part copies take ``optim.lr_head`` /
    ``lr_projection`` / ``lr_projection_block`` as in the unfrozen runs."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    # explicit layer indices; [] = the last ``top_layers`` layers (0 = every layer)
    layers: list[int] = field(default_factory=list)
    top_layers: int = 4
    targets: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj",
                                 "w1", "w2", "w3"]
    )
    lr: float = 5.0e-5  # the LoRA; the part copies use the optim.lr_* of their group
    # train the head / projection / interface copies (False = LoRA only; the
    # copies still exist so the slot path has its own, untouched-by-trees parts)
    train_parts: bool = True


@dataclass
class ModelCfg:
    encoder_id: str = "LiquidAI/LFM2.5-Encoder-350M"
    decoder_id: str = "LiquidAI/LFM2.5-350M-Base"
    # Tokenizer of everything the ENCODER reads (context slots, compression prompts, spine
    # leaves, gold spans, pre-tokenized corpora). None = the decoder's (one tokenizer; the
    # LFM2.5-350M decoders). Set it when the decoder's vocabulary differs (LFM2.5-8B-A1B:
    # "LiquidAI/LFM2.5-350M", the tokenizer the frozen encoder was trained with).
    ctx_tokenizer_id: str | None = None
    # Decoder scale-up (plans/decoder_scaleup_8b_2026_09_25.md): P8, a trained adapter from the
    # encoder's rep space (the embedding space of ``rep_space_decoder_id``, the decoder the
    # encoder was trained against) into this run's decoder, applied to every rep at splice time.
    decoder_space_adapter: bool = False
    rep_space_decoder_id: str = "LiquidAI/LFM2.5-350M"
    space_adapter_hidden_mult: int = 2
    # transformers experts implementation for MoE decoders (Lfm2Moe): None = the library default.
    decoder_experts_implementation: str | None = None
    # fused Triton RMSNorm on the decoder (models/fused_rmsnorm.py); 8B fwd+bwd 16k: 4.20 -> 3.46 s
    decoder_fused_rmsnorm: bool = False
    # render the decoder's chat template in non-thinking mode (data.templates.NO_THINK_BLOCK;
    # LFM2.5-8B-A1B opens every turn with a think block otherwise)
    decoder_no_think: bool = False
    # Save only the decoder's TRAINABLE tensors (LoRA, P8) in checkpoints: an 8B frozen base is
    # 16 GB per save. The base is re-read from decoder_id on load.
    decoder_save_trainable_only: bool = False
    head_layer: int = 6
    # Warm-start encoder + projection + interface norm from a bgkit2 checkpoint
    # directory (BaseTrainer format). The decoder/LoRA/optimizer start fresh.
    init_encoder_from: str | None = None
    # Backbone layers allowed to train. Empty = backbone fully frozen (only
    # head, survive embedding and projection learn). Never the embeddings.
    train_backbone_layers: list[int] = field(default_factory=list)
    head_inner_dim: int = 256
    projection_mult: int = 4
    # Contextual projection block (copy of the backbone's last attention
    # layer over the survivor set) ahead of the MLP projection.
    projection_block: bool = False
    # Zero-initialised linear skip from each survivor's encoder input
    # embedding into the projection input (token identity for free).
    projection_input_skip: bool = False
    # Interface = prenorm + fixed row norm only (no statistics, no recolour).
    interface_rms_only: bool = False
    # Learned separator embedding between the compression prompt and the
    # document (encoder.prompt_separator).
    prompt_separator: bool = False
    # Explicit prompt route (encoder.prompt_cond_in / prompt_cond_hook): the
    # pooled prompt added to every document position at the input and at
    # the head layer; zero-initialised. See the encoder note (P19o).
    prompt_conditioning: bool = False
    # Build encoder.repro, the projection from encoder output back into the
    # encoder's OWN input space, which a multi-level tree needs to feed one
    # level into the next (plans/multilevel_encoder_2026_09_16.md).
    repro_head: bool = False
    # Warm start: drop these state-dict prefixes so they start fresh
    # (e.g. ["projection.", "interface_norm."] when the interface contract
    # changes under a trained backbone).
    init_drop_prefixes: list[str] = field(default_factory=list)
    interface_momentum: float = 0.1
    interface_row_norm: bool = (
        False  # hard per-row rescale at the interface (see interface_norm.py)
    )
    # Parameter-free unit-RMS row normalisation BEFORE the interface EMA, so
    # the reference is bounded by construction (interface_norm.py, 2026-09-05).
    interface_prenorm: bool = False
    # Re-colour the standardised rep with the decoder's per-channel embedding
    # mean/std (fixed buffers) instead of unit variance: the reps land ON the
    # decoder's manifold (interface_norm.py, 2026-09-06).
    interface_match_moments: bool = False
    # Token-anchored residual reps (2026-09-06): rep_i = interface(proj(h_i))
    # + scale * E_dec[token_i]. Every survivor carries its own token embedding
    # by construction — the decoder reads it as it reads text — and the
    # encoder adds context on top. In every run so far x1 ≈ x4 ≈ x16 and the
    # within-document effective rank of the reps was ~2: survivors collapsed
    # to one document vector. `rep_token_residual_only` is the ablation arm
    # (plain token pruning: the selected tokens' embeddings, no encoder rep).
    # Warm-start ONLY the selection head (``head.*`` + ``survive_embedding``)
    # from another checkpoint's encoder state, after ``init_encoder_from``.
    init_head_from: str | None = None
    # Warm-start the DECODER state (the LoRA adapter) from a checkpoint —
    # continuing a stage-D run under a new curriculum / teacher.
    init_decoder_from: str | None = None
    rep_token_residual: bool = False
    rep_token_residual_only: bool = False
    rep_token_residual_scale: float = 1.0
    # Apply the straight-through gate to the token residual so the head gets
    # the decoder's gradient through the tokens it chose (P17: it degraded the
    # head — off by default; kept for the record).
    rep_token_gate: bool = False
    # Summary slots: m = ceil(frac * k) extra reps pooled over the DROPPED
    # tokens (encoder.py), survivor ratio scaled to keep the slot budget.
    summary_slot_frac: float = 0.0
    selection: str = "learned"
    min_survivors: int = 1
    gate_tau: float = 1.0
    lora: LoraCfg = field(default_factory=LoraCfg)
    # LoRA on the ENCODER backbone (in place; peft inject_adapter_in_model).
    # Full-layer backbone training diverged (P4); adapters bound the drift.
    # layers: [] = every layer.
    encoder_lora: LoraCfg = field(default_factory=lambda: LoraCfg(enabled=False, layers=[]))
    # The ENCODER ADAPTER of ``optim.encoder_mode: adapter`` (2026-09-22).
    encoder_adapter: EncoderAdapterCfg = field(default_factory=lambda: EncoderAdapterCfg())


@dataclass
class DataCfg:
    train_stores: list[str] = MISSING
    train_weights: list[float] | None = None
    eval_stores: list[str] = MISSING
    ctx_min: int = 64
    ctx_max: int = 1024
    cont_min: int = 32
    cont_max: int = 256
    p_continue: float = 0.5
    batch_size: int = 16
    # Length-bucketing: sort windows of this many batches by a cheap length
    # hint and cut them into batches of similar length (0 = plain order).
    length_bucket_window: int = 0
    # Padded token budget per training batch (rows x longest hint); 0 = fixed
    # batch_size. Lets 8k-token rows train in small batches beside 2k rows.
    max_batch_tokens: int = 0
    num_workers: int = 4
    epoch_size: int | None = None
    eval_samples: int = 256
    eval_batch_size: int = 16
    gen_samples: int = 48
    gen_max_new_tokens: int = 320
    probe_ratio: float = 1 / 16
    prompt_style: str = "plain"  # plain (base decoder) | chat (instruct decoder's own template)
    compression_prompts: bool = False  # hand the encoder a per-task compression prompt
    # Task stores (bgkit2.data.tasks parquet). Mixed with the autoencode
    # stream: P(task sample) = task_mix; within tasks by task_weights.
    task_stores: list[str] = field(default_factory=list)
    task_weights: list[float] | None = None
    # Stores evaluated but never trained on (external benchmarks, e.g.
    # BaBiLong): eval split only, no held-out phrasing arm.
    eval_only_stores: list[str] = field(default_factory=list)
    task_mix: float = 0.5
    task_max_ctx_tokens: int = 2048
    task_max_target_tokens: int = 384
    task_max_prefix_tokens: int = 1536  # decoder prefix (chat framing) cap; longer rows are skipped
    # Per-store override of task_max_prefix_tokens, keyed by the store's file
    # stem. A store whose PURPOSE is long-context anchoring should not be
    # trimmed to the global cap: read_replay_v2 is built to 7000 tokens (p90
    # 4,572) and 11.2% of its rows exceed a 4096 global cap, i.e. exactly the
    # long reading anchors the store exists to provide never train. Raising
    # the global cap instead would lengthen decoder sequences for every store.
    task_max_prefix_tokens_by_store: dict[str, int] = field(default_factory=dict)
    # Stage C: train on repo trees instead of the token corpus. A parquet from
    # scripts/bgkit2_build_repo_tree.py (paths + file_ids + question + answer);
    # the whole repo is encoded as a tree and the decoder answers with a path.
    repo_tree_store: str | None = None
    repo_tree_max_files: int = 24
    repo_tree_file_tokens: int = 512
    # Stage D: interleave tree-navigation steps with the corpus/task mix.
    # 0 = the tree store REPLACES the corpus (Stage C); k > 0 = every k-th
    # optimizer step draws a batch of trees instead, so the decoder learns to
    # read root encodings alongside everything else.
    repo_tree_every: int = 0   # k>0: a tree batch every k steps; 0: trees ARE the run; <0: trees are eval-only
    # Node reps per repo pre-computed by scripts/bgkit2_cache_repo_trees.py
    # (safetensors: "<repo>/reps", "<repo>/mask") = the BASE encoder's output.
    # Needs optim.encoder_mode frozen or adapter (the base stays frozen; the
    # adapter is off on this path); tree batches then cost nothing per step.
    repo_tree_root_cache: str = ""
    # Cap on rows x maxlen^2 for ONE extra-slot encoder call. The encoder is
    # bidirectional, so this bounds the attention-score allocation of a
    # multi-slot batch: P26 OOMed on a 23.84 GiB softmax at step ~800 because
    # every slot in the batch was padded to the longest one (36 rows x 16 heads
    # x 8192^2 = 72 GiB of scores). 2e8 keeps any chunk at ~6 GiB.
    extra_encode_token_sq_budget: int = 200_000_000
    task_eval_samples: int = 128  # per store, from its eval split
    task_gen_samples: int = 64  # generative answers per store per eval (short targets)
    task_gen_max_new_tokens: int = 96
    # Rendering-time phrasing diversity (bgkit2.data.framing_bank): the
    # decoder framing, encoder prompt and tool name are drawn per row from a
    # bank instead of the one stored phrasing. Off = the stored phrasing.
    framing_diversity: bool = False
    framing_p_system: float = 0.3
    framing_p_no_prompt: float = 0.1
    # With diversity on, each task store also gets a HELD-OUT-phrasing eval
    # arm (``task/<store>_heldout/*``, this many rows) = format transfer.
    task_eval_heldout_samples: int = 128
    # Stores (stems) that get a held-out phrasing arm; empty = every store.
    heldout_stores: list[str] = field(default_factory=list)
    # Stores (by stem) whose training rows also draw a SIBLING prompt —
    # another question about the same document (meta.doc) — for the
    # prompt-contrast loss (kd.prompt_contrast_weight).
    alt_prompt_stores: list[str] = field(default_factory=list)
    # Teacher-forced task-eval arms (compression factors; 1 = the x1 arm) and
    # the generative arms. Needle-style stores only test prompt conditioning
    # where the needle union exceeds the budget (64x/128x).
    task_eval_factors: list[int] = field(default_factory=lambda: [1, 4, 16])
    task_gen_factors: list[int] = field(default_factory=lambda: [4, 16])
    # ---- in-training eval budget (2026-09-11) ----
    # Stores (by task_eval key, e.g. "babilong", "qa_squad_heldout") skipped by
    # the trainer's own eval; the pause ablation still builds and scores them.
    train_eval_skip_stores: list[str] = field(default_factory=list)
    # Families generated on during the trainer's eval (empty = every store);
    # long free-form targets (compaction, chunks) always score EM 0 and cost
    # 96 generated tokens per row.
    task_gen_families: list[str] = field(default_factory=list)
    # Token cap per eval batch for the teacher-forced arms (0 = fixed
    # eval_batch_size). Rows are sorted by length and packed up to the cap,
    # so short stores run at 8-16 rows per batch while 8k rows stay alone.
    eval_batch_max_tokens: int = 0


@dataclass
class RatioCfg:
    r_max: float = 1 / 4
    r_start: float = 1 / 4
    r_end: float = 1 / 128
    descent_steps: int = 10_000
    r_max_start: float = 1.0
    ceiling_steps: int = 2000
    # Hold floor and ceiling at r_start / r_max_start for this many steps
    # before any descent (zero-compression alignment phase).
    hold_steps: int = 0
    # Gate that ends the hold early: at each eval, if ``hold_gate_metric``
    # minus ``hold_gate_reference`` (both eval keys) is <= ``hold_gate_max_gap``
    # the descents start at that step (logged ``ratio_hold_released``). With
    # no reference the metric alone must be <= max_gap. Empty = timer only.
    hold_gate_metric: str = ""
    hold_gate_reference: str = ""
    hold_gate_max_gap: float = 0.0
    # Adaptive floor: log-descent rate scaled by (r/r_start)**rate_power and
    # by a mastery multiplier read at each eval from the gap between the
    # reconstruct loss at the CURRENT FLOOR ratio and full text
    # (multiplier = clamp(mastery_target_gap / gap, mult_min, 1)).
    adaptive: bool = False
    rate_log2_per_step: float = 1e-3
    rate_power: float = 0.5
    mult_min: float = 0.1
    # Share of steps sampled from the band's top [band_top_lo, ceiling]
    # (keeps 4-8x trained while the floor descends; 0 = plain log-uniform).
    band_top_prob: float = 0.0
    band_top_lo: float = 0.125
    mastery_family: str = "reconstruct"
    mastery_target_gap: float = 0.3
    p_random_selection: float = 0.1
    # Per-family ratio bands {family: [lo, hi]}: rows of these families draw
    # their own log-uniform ratio each step instead of the band's sample.
    family_bands: dict[str, list[float]] = field(default_factory=dict)
    # Per-family ABSOLUTE rep budgets {family: [lo, hi]} (log-uniform per
    # sample; ratio = budget / document length, clamped to [1/1024, 1/2]).
    # For long documents a ratio hands the decoder thousands of reps it
    # cannot use (BABILong 8k: 4x is no better than 64x); a budget trains
    # the regime the reader works in. Takes precedence over family_bands.
    family_rep_budget: dict[str, list[int]] = field(default_factory=dict)
    # BenchPress-style multi-ratio training: encode ONCE per microbatch,
    # select at this many independently sampled ratios, run the decoder for
    # each and average the losses. 1 = one ratio per step.
    ratios_per_step: int = 1
    # Per-SLOT pricing of tool_slots rows (bgkit2.data.slot_pricing, 2026-09-22): the eval
    # harness prices each slot by its tool (read 1/8, web_search 1/4, memory index 4-24 reps,
    # ...) while the trainer gave every slot of a row the row's one ratio. "row" = that old
    # behaviour; "meta" = rows carrying meta.slot_reps (the harness's own n_reps) are encoded at
    # those counts; "harness" = meta, else prices inferred from the slots' compression prompts.
    # Unpriced slots (cached tree nodes, spines, unknown prompts) and unpriced rows keep the row
    # ratio (family_rep_budget / family_bands / the step's band). Training only: the eval arms
    # keep their fixed ratio sweeps.
    slot_reps_mode: str = "row"
    # log-uniform multiplier band around each slot's price, sampled per slot per step
    slot_reps_jitter: list[float] = field(default_factory=lambda: [0.5, 2.0])


@dataclass
class MultiLevelCfg:
    """Stage B: distil the encoder to accept COMPRESSED inputs.

    A frozen SNAPSHOT of the encoder encodes the document once (single level) at
    ``teacher_ratio``; the student encodes the SAME document twice — level 0 at a
    looser ratio, then level 1 through the repro bridge down to the teacher's own
    survivor set — and must land on the teacher's projection output. Masks are
    forced on both levels so the loss measures the BRIDGE, not a disagreement
    about which tokens survived. Ported from bgkit1 Step 4.7
    (src/bgkit/training/phase1/bridge_distill.py).

    The teacher is a snapshot, not the live encoder: a live teacher drifts as the
    student trains and the target moves under it.
    """

    enabled: bool = False
    teacher_checkpoint: str | None = None
    teacher_ratio: float = 0.20
    mse_weight: float = 1.0
    cos_weight: float = 1.0
    # Path A = the single-level encode (so the projection keeps handling that
    # distribution); Path B = the two-level encode. Mixed per step.
    path_a_prob: float = 0.5
    # Fraction of the teacher's DOOMED positions that survive level 0 and die at
    # level 1. 1.0 = level 0 keeps everyone and level 1 does all the compression;
    # sliding down moves load onto level 0 and makes level 1's job real.
    frac_extras_start: float = 1.0
    frac_extras_end: float = 0.31
    curriculum_steps: int = 2500
    # Path B over GROUPS of documents. 1 = the per-document port of Step 4.7
    # (level 1 re-compresses the same document), which never shows the upper
    # level the distribution it will actually serve: survivors concatenated
    # from DIFFERENT files. >1 = the teacher encodes the flat CONCATENATION of
    # each group single-level (skipping groups that overflow the encoder), the
    # student encodes the group as a tree, and the final survivor sets are
    # forced equal through the concat positions.
    group_size: int = 1
    # Tree encoding (Stage C and evaluation): per-level ratios and depth cap.
    ratio_leaf: float = 0.125
    ratio_node: float = 0.25
    max_depth: int = 3
    leaf_chunk: int = 32           # max files per encoder call when encoding a tree
    leaf_chunk_tokens: int = 32768 # max padded tokens (files x longest) per encoder call
    node_max_tokens: int = 8192    # a directory input above this is split into virtual sub-directories
    leaf_window: int = 0           # whole files: window size for files longer than this (0 = one window, cut by file_tokens)
    leaf_max_reps: int = 0         # bound a leaf's reps: long files compress harder instead of being cut
    # QUESTION-CONDITIONED node read-out (2026-09-21): the node the decoder reads is re-encoded from
    # its CACHED input (children's survivors; cache built with --node-inputs) with the question as
    # the encoder prompt; the tree below the node stays frozen-cached. Needs a node cache.
    prompted_readout: bool = False
    # PATH WALK: the tree task is asked level by level -- at each directory node
    # the decoder reads THAT node's encoding and names the next path component
    # -- instead of producing the whole path from the root (generated path exact
    # 2.3% on held-out repos, 2026-09-19).
    walk: bool = False
    grad_levels: int = -1          # levels from the root with a gradient; -1 = all, 0 = none
    # optim.encoder_mode adapter: trees of at most this many files are encoded
    # LIVE through the slot adapter with ``grad_levels`` (upper levels + root
    # train, leaves no_grad -- the 09-17 design); larger trees (e.g. the
    # repo_walk_large tier) keep the frozen BASE path and the node cache.
    # 0 = every tree through the cache / base path.
    live_tree_max_files: int = 0
    # How one level feeds the next in the tree: "repro" = through the trained
    # repro head (Stage A/B's premise); "identity" = survivors passed straight
    # through, no bridge. The fixed probe (2026-09-16) found identity within
    # +0.30 nats of the trained head, so Stage C runs BOTH as an ablation.
    bridge: str = "repro"


@dataclass
class AnchorCfg:
    """Embed-anchor auxiliary loss (the old Step 2.5 lesson): pull each
    survivor's projected rep toward the decoder's embedding of its own token,
    MSE + cosine, weight decaying linearly from ``weight`` to ``floor`` over
    ``steps``. Gives a frozen decoder something readable from step 0; decays
    because at low ratios a survivor must carry more than its own token."""

    weight: float = 1.0
    floor: float = 0.0
    steps: int = 4000
    # Stage D (user, 2026-09-08): once the decoder adapts, the anchor is a
    # constraint AGAINST distributed reps — phase it out on a timer from the
    # run's start (overrides the progress-driven decay when > 0).
    phase_out_steps: int = 0
    # Mean squared off-diagonal CORRELATION of the batch's valid reps (0 =
    # whitened, ~1 = one shared direction). Raises the rank the projection is
    # allowed to collapse to. 0 = off.
    decorrelation_weight: float = 0.0
    # REPRO-HEAD pre-warm (multi-level, Stage A of
    # plans/multilevel_encoder_2026_09_16.md): pull `encoder.reproduce(survivors)`
    # toward the ENCODER's own input embedding of the survivor's token — the
    # same objective bgkit1 used to pretrain auto_repro_head. Distinct from
    # `weight` above, which anchors the DECODER-facing projection. 0 = off, and
    # it is a no-op unless the encoder was built with repro_head=True.
    repro_weight: float = 0.0
    # Mean squared cosine between DISTINCT survivors of the same document
    # (0 = orthogonal rows, 1 = k copies of one vector). The cross-batch
    # decorrelation cannot see this: P7b reached within-doc shared fraction
    # 0.9998 with decorrelation at 0.12.
    within_doc_diversity_weight: float = 0.0


@dataclass
class KDCfg:
    """Knowledge distillation from the FULL-TEXT teacher: the same decoder with
    its adapter disabled, given the context's token embeddings spliced into
    the same slot the reps occupy. The teacher's answer distribution is
    context-dependent, so a student that ignores the reps cannot match it —
    the objective BenchPress trains with (there, a fine-tuned teacher)."""

    weight: float = 0.0
    temperature: float = 1.0
    # A bgkit2 checkpoint dir whose ``decoder`` state carries a LoRA adapter
    # trained on FULL TEXT (e.g. P8). When set, that adapter is loaded into a
    # FULL-TEXT STUDENT: train the decoder adapter with the context's own
    # token embeddings in the rep slot (no encoder forward, no reps) — the
    # matched ceiling for a rep-gated adapter (same adapter budget, same
    # positions) and the teacher for kd.teacher_checkpoint.
    full_text_student: bool = False
    # separate frozen copy of the decoder (same ``model.lora`` spec) and used
    # as the KD teacher instead of the adapter-off base: BenchPress-style
    # fine-tuned teacher. P12 showed KD-only imitates whatever the teacher can
    # do, and the base instruct decoder reads full text at SQuAD 2.63 while
    # P8's full-text adapter reached 1.11.
    teacher_checkpoint: str | None = None
    # The teacher's adapter spec. None = the student's ``model.lora`` (which
    # must then be enabled). Set it when the student trains WITHOUT a decoder
    # adapter (frozen-decoder arms) but the selection loss still needs the
    # fine-tuned teacher's attention.
    teacher_lora: LoraCfg | None = None
    # Selection supervision from the teacher's ATTENTION: the mass each
    # context token receives from the answer positions (heads x attention
    # layers averaged) is the target distribution; the head's z-scores are
    # trained with a listwise KL toward it. A prompt-conditioned "what the
    # teacher used" signal — the straight-through gate on token embeddings
    # (P17) only gave the head a scale-this-embedding gradient and degraded
    # it. Needs the teacher built with eager attention (automatic when > 0).
    selection_weight: float = 0.0
    selection_temperature: float = 1.0
    # Samples per eager teacher pass in attention_importance: the (b, H, T, T)
    # transient is the run's peak (chunk 4 at T~2.3k: 25 GB allocated / 78 GB
    # reserved, host available dipping to 36 GB on unified memory).
    selection_chunk: int = 2
    # Run the selection (teacher-attention) loss every N optimizer steps; the
    # eager-attention pass over the full text is the single heaviest pass.
    selection_every: int = 1
    # Run the KD teacher pass (a full frozen-decoder forward over the full
    # text, ~a fifth of the step) every N optimizer steps; CE alone otherwise.
    kd_every: int = 1
    # Where the head's supervision comes from. "attention": listwise KL to a
    # teacher's answer-to-context attention (an extra eager-attention pass
    # over the full text). "utility_grad": the old lineage's utility signal —
    # after the main backward, each survivor's first-order loss increase from
    # removing its rep, -(dL/drep · rep), becomes the listwise target; zero
    # extra passes, exploration over dropped positions comes from the
    # random-selection steps.
    selection_signal: str = "attention"
    # Rows longer than this (valid context tokens) are left out of the
    # attention-selection pass (0 = every row): the eager (H, T, T) pass on
    # an 8k row is the step's peak, and the needle / QA rows the signal is
    # for are shorter.
    selection_max_ctx_tokens: int = 0
    # Head raw-scale anchor (old lineage, helpers.py:236-254): under exact
    # top-k every head loss is on the per-document z-score, so the raw scale
    # is gradient-free and Adam amplifies noise there. Weight x mean over
    # documents of log(raw_std / head_scale_target)^2.
    head_scale_anchor_weight: float = 0.0
    head_scale_target: float = 1.0
    ce_weight: float = 1.0
    # Prompt contrast (P19l, 2026-09-10): the whole encoder trained for 6000
    # steps with the question in its prompt and learned == no-prompt ==
    # WRONG-prompt at every ratio — a query-independent encoding served every
    # question at once, so nothing pushed the prompt into the reps. Rows with
    # a sibling prompt (another question about the SAME document) are
    # encoded a second time under it, at the same ratio and selection, and
    # the true answer's per-sample nll under the sibling reps must exceed
    # the nll under the right reps by ``margin``:
    #   hinge = mean_i relu(margin - (nll_sibling_i - nll_right_i)).
    # The only way to satisfy it on the same document is prompt-specific
    # content. ``prompt_gap`` (nll_sibling - nll_right) is the live measure
    # of prompt use — the quantity the pause ablation reads offline.
    prompt_contrast_weight: float = 0.0
    prompt_contrast_margin: float = 1.0
    # Context-token budget for the contrast pass per step (0 = every row
    # with a sibling prompt); rows beyond it wait for another step.
    prompt_contrast_max_tokens: int = 6144
    # Symmetric (P19m): recompute the RIGHT-prompt arm in the contrast pass
    # too, so the gap (nll_sibling - nll_right) carries gradient through
    # both arms. The one-sided hinge against a detached nll_right (P19l) can
    # be satisfied by making the needle encoding worse in general — and was:
    # the needle loss rose a nat while the gap returned to zero.
    prompt_contrast_symmetric: bool = False
    # Paired (P19n): the sibling is a FULL sample (its own answer). Both
    # arms are recomputed and the 2x2 is scored: CE(B|reps_B) keeps the
    # sibling reps honest for the sibling's own answer, while the two
    # hinges push the crossed pairs nll(A|reps_B) and nll(B|reps_A) above
    # nll(A|reps_A) and nll(B|reps_B) by the margin. P19l/P19m (one hinge,
    # no CE on the sibling's answer) both opened a 0.3-nat gap by step 600
    # and closed it by step 900 while the needle loss rose ~1.5 nats: the
    # hinge was satisfied by making the sibling-prompt reps bad for
    # EVERYTHING, which the main CE then undid. With the sibling's own
    # answer in the pass, bad-for-everything costs.
    prompt_contrast_paired: bool = False
    # Oracle-window teacher (P21): rows whose store carries meta.gold_window
    # (the supporting facts / passage) give the KD teacher ONLY that window
    # in the slot instead of the whole document. At 8k the full-text teacher
    # is itself a poor reader, so its distribution is a poor target; the
    # window keeps the target strong at any length and asks the reps for
    # exactly "answer as if you had the relevant passage". Rows without a
    # window use the full context as before. The attention-selection target
    # skips windowed rows (its positions no longer map onto the document).
    teacher_gold_window: bool = False
    # Row-pooled CE (each row's mean nll averaged over rows) and per-family
    # loss multipliers — see BgKIT2Core.pooled_ce.
    per_row_ce: bool = False
    family_loss_weight: dict[str, float] = field(default_factory=dict)
    # ---- replay stream (2026-09-13, user: open the whole decoder without
    # losing it) ----
    # Rows of ``replay_family`` (framing "chat": a plain conversation, no
    # context slot) take no encoder pass. Their loss is self-distillation:
    # KL(base || student) on the assistant turn, the student being the
    # adapted decoder and the base the SAME decoder with its adapter
    # disabled — the model we want to preserve is the target, the datasets
    # only supply prompt coverage — plus ``replay_ce_weight`` x CE on the
    # stored turn. With ``model.lora.rep_gated: false`` the adapter applies
    # everywhere, so this stream is what keeps plain text and chat behaviour
    # from drifting; ``lm_health/*`` and ``task/<replay store>/*`` measure it.
    replay_family: str = "chat_replay"
    replay_kl_weight: float = 1.0
    replay_ce_weight: float = 0.2


@dataclass
class OptimCfg:
    lr_backbone: float = 5e-5
    lr_head: float = 1e-4
    lr_projection: float = 2e-4
    lr_projection_block: float = 1e-4
    lr_lora: float = 1e-4
    lr_space_adapter: float = 2e-4  # P8 (model.decoder_space_adapter)
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 500
    max_steps: int = 20_000
    grad_clip: float = 1.0
    grad_accum: int = 1
    # Decoder-first phase (P24, 2026-09-13): for the first N optimizer steps
    # the ENCODER (backbone, head, projection, projection block, interface)
    # takes no gradient — a fresh decoder adapter first learns to read the
    # warm-started reps as they are, then the two co-train. Without it the
    # untrained reader's gradient drags a converged encoder around (the
    # "realignment" is done by the reader, not by the reps). 0 = off.
    encoder_freeze_steps: int = 0
    lr_repro: float = 1e-4
    # Stage D: the encoder tree is FROZEN and OUT of the optimizer entirely —
    # no encoder param group, no optimizer state for it. Only the decoder
    # adapter trains. This is the memory constraint the multi-level design
    # exists to satisfy.
    freeze_encoder: bool = False
    # What of the encoder trains (2026-09-22): frozen | adapter | backbone.
    #   frozen   = the old ``freeze_encoder: true`` (only the decoder adapter and
    #              a compaction-spine adapter train);
    #   adapter  = ``model.encoder_adapter`` on every non-tree slot path; the
    #              base encoder (tree cache path) stays frozen;
    #   backbone = the per-group policy (train_backbone_layers, head,
    #              projection, ...), the pre-P28 mode.
    # None = derived from ``freeze_encoder`` (true -> frozen, false ->
    # backbone), so every existing config loads unchanged. Read it through
    # ``encoder_mode(cfg)``.
    encoder_mode: str | None = None
    # encoder_mode adapter: slot tokens per MICROBATCH encoded WITH gradient
    # (encoder activations are kept for backward only for these). Primary slots
    # always take gradient (the batch token cap bounds them) and count first;
    # the extra slots share the rest, the MOST RECENT slot of each row first
    # (round-robin over rows in a per-step random order), and the others are
    # encoded under no_grad -- every slot is still encoded LIVE with the current
    # weights (train == eval), only the gradient is sampled. Logged as
    # enc_grad/*. 0 = every slot takes gradient.
    encoder_grad_token_budget: int = 0
    # Stage A: train ONLY encoder.repro, everything else frozen. The optimizer
    # then holds a single small head instead of the whole encoder, which is the
    # memory point of the multi-level plan.
    train_only_repro: bool = False


@dataclass
class RunCfg:
    name: str = "run"
    out_dir: str = "./bgkit2_runs"
    seed: int = 0
    dtype: str = "bfloat16"
    device: str = "cuda"
    log_every: int = 10
    eval_every: int = 1000
    save_every: int = 1000
    keep_last: int = 3
    gradient_checkpointing: bool = True
    resume: bool = True
    wandb: bool = False
    wandb_project: str = "bgkit2"


@dataclass
class CompactionSpineCfg:
    """Cons-mode compaction (2026-09-22): a compaction slot is a SPINE
    cons(previous node, newly folded rounds) encoded node by node
    (``models.multilevel.encode_spines``) -- by the trainer for task rows whose
    slot carries a spine payload, and by the student server at eval."""

    # None = AUTO (default): on exactly when a configured task store carries
    # spine rows (trainer: scanned at setup, before the model is built) or the
    # checkpoint carries a spine adapter (student server) -- so a run that needs
    # it cannot forget it, and a run whose stores have none (and whose
    # checkpoints predate it) keeps its encoder state unchanged across resumes.
    # true / false force it. Rows with a spine slot raise when it is off.
    enabled: bool | None = None
    # adapter = a LoRA on the encoder backbone + a spine-only copy of the repro
    # bridge, active ONLY while encoding spine nodes (ordinary slots run the
    # base encoder bit-identically); frozen = the base encoder, no training.
    trainable: str = "adapter"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    layers: list[int] = field(default_factory=list)  # [] = every backbone layer
    targets: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj",
                                 "w1", "w2", "w3"]
    )
    train_bridge: bool = True
    lr: float = 5.0e-5
    # the top `grad_levels` CELLS (and their leaves) keep a gradient; older
    # cells run under no_grad
    grad_levels: int = 2
    # wide-cell layout (bgkit2.data.spine; the harness prices the slot with the
    # same numbers): leaves (raw folded rounds) at leaf_ratio, the cell-to-cell
    # pass at the gentler cell_ratio, at most max_depth cells, a cell input
    # above node_max_tokens closes the cell early
    leaf_ratio: float = 0.0625
    cell_ratio: float = 0.25
    max_depth: int = 4
    node_max_tokens: int = 8192


@dataclass
class Config:
    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    ratio: RatioCfg = field(default_factory=RatioCfg)
    anchor: AnchorCfg = field(default_factory=AnchorCfg)
    multilevel: MultiLevelCfg = field(default_factory=MultiLevelCfg)
    compaction_spine: CompactionSpineCfg = field(default_factory=CompactionSpineCfg)
    kd: KDCfg = field(default_factory=KDCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    run: RunCfg = field(default_factory=RunCfg)


def load_config(yaml_path: str | None, overrides: list[str] | None = None) -> Config:
    base = OmegaConf.structured(Config)
    if yaml_path:
        base = OmegaConf.merge(base, OmegaConf.load(yaml_path))
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(base)
    cfg: Config = OmegaConf.to_object(base)  # type: ignore[assignment]
    _validate(cfg)
    return cfg


# frozen_adapter (2026-09-25, decoder scale-up): the checkpoint's encoder adapter is INSTALLED
# (the encoder serves exactly the reps it was trained to) but nothing in the encoder trains.
ENCODER_MODES = ("frozen", "adapter", "backbone", "frozen_adapter")
ADAPTER_MODES = ("adapter", "frozen_adapter")


def encoder_mode(cfg: Config) -> str:
    """The effective ``optim.encoder_mode`` (``freeze_encoder: true`` is an
    alias for ``frozen``)."""
    o = getattr(cfg, "optim", None)  # (test stubs carry partial configs)
    m = getattr(o, "encoder_mode", None)
    if m is None:
        return "frozen" if bool(getattr(o, "freeze_encoder", False)) else "backbone"
    return str(m)


def _validate(cfg: Config) -> None:
    m = cfg.optim.encoder_mode
    if m is not None and m not in ENCODER_MODES:
        raise ValueError(f"optim.encoder_mode must be one of {ENCODER_MODES}; got {m!r}")
    if cfg.optim.freeze_encoder and m not in (None, "frozen"):
        raise ValueError(
            f"optim.freeze_encoder: true contradicts optim.encoder_mode: {m} "
            "(freeze_encoder is the alias of encoder_mode: frozen; drop it)")
    if encoder_mode(cfg) == "adapter" and cfg.model.encoder_lora.enabled:
        raise ValueError("encoder_mode adapter and model.encoder_lora are two encoder LoRAs; "
                         "use model.encoder_adapter")
    if int(cfg.multilevel.live_tree_max_files) > 0 and encoder_mode(cfg) not in ADAPTER_MODES:
        raise ValueError("multilevel.live_tree_max_files needs optim.encoder_mode: adapter")
    if cfg.model.selection not in {"learned", "random"}:
        raise ValueError("model.selection must be learned|random")
    if cfg.kd.selection_signal not in {"attention", "utility_grad"}:
        raise ValueError("kd.selection_signal must be attention|utility_grad")
    from bgkit2.data.slot_pricing import SLOT_REPS_MODES

    if cfg.ratio.slot_reps_mode not in SLOT_REPS_MODES:
        raise ValueError(f"ratio.slot_reps_mode must be one of {SLOT_REPS_MODES}")
    j = list(cfg.ratio.slot_reps_jitter)
    if len(j) != 2 or not 0 < float(j[0]) <= float(j[1]):
        raise ValueError("ratio.slot_reps_jitter must be [lo, hi] with 0 < lo <= hi")
    if cfg.ratio.ratios_per_step < 1:
        raise ValueError("ratio.ratios_per_step >= 1")
    if cfg.compaction_spine.trainable not in {"adapter", "frozen"}:
        raise ValueError("compaction_spine.trainable must be adapter|frozen")
    if not 0 <= cfg.ratio.p_random_selection <= 1:
        raise ValueError("ratio.p_random_selection in [0, 1]")
    if cfg.data.prompt_style not in {"plain", "chat"}:
        raise ValueError("data.prompt_style must be plain|chat")
    if cfg.data.task_stores and cfg.data.prompt_style != "chat":
        raise ValueError("task stores render chat framings; set data.prompt_style: chat")
    if not 0.0 <= cfg.data.task_mix <= 1.0:
        raise ValueError("data.task_mix in [0, 1]")
    if cfg.run.dtype not in {"bfloat16", "float32"}:
        raise ValueError("run.dtype must be bfloat16|float32")
    if cfg.optim.grad_accum < 1:
        raise ValueError("optim.grad_accum >= 1")


def to_yaml(cfg: Config) -> str:
    return OmegaConf.to_yaml(OmegaConf.structured(cfg))
