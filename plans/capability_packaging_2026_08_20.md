# BgKIT Capability Packaging Plan — 2026-08-20

**Status: ACTIVE — supersedes the Phase-2 KB browse/navigation training
direction and the Phase-2 → Phase-3 promotion path in
`docs/02_training_plan.md`.** That document remains accurate as a description
of implemented machinery; this plan redirects what we train next.

---

## 1. Why the pivot

The knowledge-retrieval formulation — teach the decoder to *navigate* a
compressed corpus by emitting `browse`/`bgkit` tool calls with learned node
IDs — is judged a dead end as formulated:

- **Reconstruction works; navigation never emerged.** Across the git-repro
  runs (through step 12000 with two ID redesigns — opaque BIP-39 IDs, then
  positional paths + file-name leaves), reconstruction from compressed
  representations trained well and the eval artifact bugs were all fixed, but
  navigation ID-accuracy never left zero. A 0.8B decoder learning a bespoke
  retrieval protocol over learned codes is fighting both its size and the
  fact that conventional retrieval over the same corpus is exact and cheap
  (`docs/00_status.md` already flagged single-passage QA as a poor target).
- **The valuable half is proven.** Every result we have says the *consumption*
  side works: dense splices carry real information (positive paired
  context-delta in the Phase-3 likelihood ablation; strong reconstruction at
  aggressive ratios; query-conditioned survivorship behaves under relevance
  supervision).

**The pivot: stop training the model to navigate compressed knowledge; train
it to *consume* compressed context everywhere it appears.** Package
soft-prompt injection + memory as the product primitive.

## 2. Proven assets we package (the "gains so far")

1. **L0/L1 compressor stack** — packed FA4 encoder, discrete survivorship,
   exact-ratio selection, threshold curves, L0→L1 bridge.
2. **Query-conditioned compression** — L1 query prompts + relevance
   supervision (gold-boost / distractor-damp). This *is* the "targeted
   compression prompt".
3. **Dense splice primitive** — compressed vectors framed between tokenized
   boundaries inside a causal decoder sequence; exercised end-to-end in
   Phase 2 and the Phase-3 prototype.
4. **Per-decoder-family projection blocks** (Qwen3.5-0.8B; Falcon-H1
   companion).
5. **Cached-L0 machinery** — content-fingerprinted manifests, blob-SHA
   dedup cache, persistent packed filesystem encoding
   (`encode_swe_repos.py`).
6. **Self-distillation design** (`plans/self-distillation.md`) — frozen
   same-architecture teacher, top-K KL, `TeacherContextBuilder`; the
   "compression rate" conditioning axis is exactly the consistency loss this
   plan needs.
7. **Training/ops infra** — packed samplers, dynamic memory scheduler, live
   config, checkpoint registry/NVMe recovery, free-running eval harness.

**Parked (not deleted):** browse trees, navigation-ID protocol, recursive
full-backprop tree training, KILT Stage-B-as-formulated. The browse-tree
datasets remain useful as *corpora* for Family B/C data generation.

## 3. One interface contract: the bgkit blob

All four task families below share a single decoder-facing format so one
trainer and one data schema cover everything. A **blob** is:

```
<bgkit_start> {header text} <splice: k dense vectors> <bgkit_end>
```

- **Framing tokens**: reuse the existing bgkit tool-response boundary tokens.
- **Header** (plaintext, ~10–40 tokens): provenance + intent, e.g.
  `compacted: turns 3–17 (2 file edits, 4 test runs)` or
  `compressed: build.log (48,210 lines) query="first failing test"` or
  `memory: session 2026-07-14` or `repo: fastapi @ a1b2c3 (1,412 files)`.
  Headers keep the model oriented, are trivially emittable by any harness,
  and give benchmarks a fair "the model knew what the blob was" footing.
- **Kinds**: `compaction | tool | memory | ambient` — a header word, not a
  new mechanism.
- **Loss discipline**: CE only on tokens after the blob (as Phase 2 does);
  the decoder only ever *reads* blobs. No IDs to emit, no tool grammar to
  learn — that entire failure surface is gone.

**Implemented 2026-08-20**: `src/bgkit/data/blob_format.py` (markers, header
templates, dedicated `BLOB_SPLICE_SENTINEL` distinct from the Phase-2 tool
sentinel, sentinel-position search) with unit tests incl. a BPE-stability
check against the real Qwen3.5 tokenizer
(`tests/unit/data/test_blob_format.py`).

## 4. Task Family A — Compaction (trajectory SFT)

**Capability:** the model behaves the same whether its older history is raw
text or a bgkit blob — under *aggressively frequent* compaction.

**Data — long trajectories from three sources:**
1. Agentic coding trajectories — the 2026-08 research found a windfall of
   public corpora far beyond what `download_swe_trajectories.py` assumed:
   **SWE-Zero (318k OpenHands trajectories), Toucan (1.5M tool-agentic
   trajectories over ~500 real MCP servers), SWE-rebench (67k), SWE-Hero
   (34k), SWE-smith (26k + synthetic-bug envs)** — see
   `docs/05_benchmark_targets.md` §3. Volume is not the constraint;
   filtering for quality/length is.
2. Long multi-turn tool-use dialogues: synthesized with the local vLLM
   teacher (GPT-OSS-20B) driving tools over our repos corpus.
3. Long-document processing chains from the summarization corpus.

**Sample construction:** for a trajectory `t1..tn`, sample compaction points;
replace prefix segments with blobs (per-segment blobs *and* a rolling merged
blob mode — both, so inference can use either); SFT on the continuation.
Each trajectory yields many samples (one per compaction point) and the
encoder forward covers only the compacted span, so aggressive frequency is
cheap at training time.

**Objectives:**
- CE on continuation tokens (primary).
- **Consistency self-distillation** (Phase P2): teacher = frozen decoder with
  the raw uncompacted history (when ≤ teacher cap), student = compacted;
  top-K KL on continuation tokens. This is the literal statement of
  "behavior preserved under compaction" and doubles as the headline
  fidelity metric.
- **Recall probes:** auto-generate continuations that *require* facts from
  the compacted region (file paths, variable names, earlier tool outputs,
  user-stated constraints, earlier decisions). Without probes, compaction
  training collapses to gist-preservation; probes force load-bearing detail
  through the bottleneck. Probes are held out as the eval suite too.

**Curricula:** ratio ramp obeys the continuous-descent rule (always down,
never reset); recency ramp (compact only old turns → everything but the
live window); blob-count ramp (1 blob → many sequential blobs).

## 5. Task Family B — Wide-net tool results (targeted compression)

**Capability:** huge tool output → few vectors, with a targeted compression
prompt; the needle survives.

**Data generators** (synthetic-over-real, all cheap to build):
1. **Large file read + needle**: real files from `processed_v2` with natural
   needles (function bodies, config values, specific lines) and planted
   needles; query prompt = what the agent was looking for.
2. **Search/grep result sets**: real greps over the repos corpus → hundreds
   of hits; QA over which hits matter, counts, file lists.
3. **Database rows / structured dumps**: synthesized tables + JSONL (point
   lookups, filters, aggregates); public table-QA sets where licensing
   allows.
4. **Logs**: LogHub corpora + synthesized logs; queries: first error,
   root-cause line, counts, time windows.
5. **Search-result pages**: MS MARCO passage data already in
   `mmap/phase2` — reuse directly.

**Training:** (query, giant blob) → answer. CE + the existing relevance
supervision on the survivorship head (gold-span boost). Include
distractor-heavy negatives and **"not present" cases** — an agent must be
able to trust a null answer from a compressed read.

This family is the Phase-2 QA machinery minus browse: the flat
single-bgkit trajectory path in `KRKBTrainer` handles it nearly unchanged.

## 6. Task Family C — Compressed memory

**Capability:** past sessions stored as blobs; injected as background in a
new session; correctly consulted and chronology-safe.

- **Conversational memory**: MSC / share / chronicles / perltqa / laps
  mmaps (partially built) — compress prior sessions per-session; queries in
  the current session reference earlier ones.
- **Agent-session memory**: prior trajectories on the same repo (the old
  Phase-3 §1.5 idea) — "what did we already try", "where was that bug".
- **Chronology**: headers carry timestamps; include conflict cases where a
  later memory overrides an earlier one (the chronology-safe memory target
  from `docs/00_status.md`).
- The user-follow-up-turn conditioning axis from `plans/self-distillation.md`
  applies directly here in P2.

## 7. Task Family D — Whole-repo ambient background

**Capability:** entire repo encoded once (cached L0 → L1) and spliced as
ambient background; the model consults it without being pointed at it.
This is the hardest family — it's the one that rhymes with the navigation
failure — so it is staged behind explicit-provenance SFT and gated.

- **D1 — SFT with oracle provenance (cheap):**
  - Repo-QA: synthetic QA generated from repos with the vLLM pipeline
    (the `generate_git_qa.py` machinery retargeted at code/state questions).
  - Cross-file completion: RepoBench/CrossCodeEval-shaped samples cut from
    our own corpus where the completion *provably* depends on another file,
    and that file is present only inside the ambient blob. Static analysis
    gives us the dependency; no navigation labels needed.
- **D2 — Self-distillation:** teacher = frozen Qwen with the actual
  dependency files in context (we know them from provenance); student =
  ambient blob only. Top-K KL on answer/completion tokens. This directly
  optimizes "use the background as if you had the files open".
- **D3 — RL (gated):** only if D2 plateaus short: RLVR with verifiable
  rewards (tests pass, exact-match completion) on repo tasks; GRPO-scale is
  feasible for a 0.8B decoder on the Spark. Do not start here.
- Keep the unseen-repository split discipline from the git-history work,
  and keep the paired likelihood ablation (`context_delta`) as the
  always-on cheap gate before any expensive D3 step.

## 8. Base checkpoint & first engineering decision

**DECIDED 2026-08-20: the summarization round-robin checkpoint.** (Bake-off
skipped by user decision; the git-repro encoder is rejected as nav-trained.)
Pin the exact checkpoint name + `run_name` in the new experiment config —
the same lineage the Phase-2 Stage-A preset already resolves from, so the
loading path is proven. Ship Qwen first; keep the Falcon-H1 projection
alive in round-robin where memory allows, but it is not on the release
critical path.

## 9. Staging

- **P0 (~1–2 wk): plumbing + eval-first.**
  - Unified blob schema + compaction sampler + Family A/B data generators.
  - Recall-probe generator and the probe eval suite.
  - Benchmark harness bring-up for the targets in
    `docs/05_benchmark_targets.md` (baseline numbers for our decoder with
    full context / truncation / text-summary compaction *before* training).
  - Serving spike: verify the vLLM embedding-injection path end-to-end.
    (Confirmed viable 2026-08-20: vLLM serves pre-computed embeddings via
    `--enable-prompt-embeds` on `/v1/completions` — request carries a
    serialized `(num_tokens, hidden)` tensor; Chat Completions support is an
    open RFC, vllm-project/vllm#39504. Our adapter must interleave token
    embeddings with projected survivor vectors client-side.)
  - Base-checkpoint bake-off (§8).
- **P1: joint SFT.** Mix A + B + C in one trainer (extend the flat
  `KRKBTrainer` path or, preferably, land the planned trainer split first —
  cache/encode/assembly/eval boundaries — and build the lean SFT trainer on
  those pieces). Continuous ratio descent; nightly probe + benchmark slices.
- **P2: distillation arm.** Consistency KL for A; D1 SFT joins the mix;
  D2 teacher pipeline on.
- **P3: package.** Benchmark runs, Pareto report (fidelity-KL and task
  scores vs. ratio), export bundle + serving adapter + agent-loop demo
  (auto-compress tool results > N tokens with the current query; auto-compact
  history) — the demo *is* the product story.

**Gates:** P1→P2: needle recall on held-out Family-B probes ≥ target at
ratio 0.02, and compaction probe-success drop vs. full-context ≤ agreed
budget at 10× compaction. P2→P3: consistency KL flat across 5+ sequential
blobs (no drift blow-up); D-family `context_delta` positive on unseen repos.
Compaction metrics adopt the published conventions: **Pass² multi-run
stability** (TRACE), tokens-per-resolved-task, peak live-window tokens, and
per-turn cost curves — see `docs/05_benchmark_targets.md` §3.

**Validating priors found in the 2026-08 research** (the bet is not
speculative): text-to-vector latent repo compression *exceeds full-context*
performance at 4–128× where text compression collapses at 7–12×
(arXiv:2604.13725); PISCO reaches 16× at 0–3% loss trained purely by
sequence-level distillation (our P2 mechanism); BenchPress shows
bidirectional compression encoders beat causal gist-tokens (our encoder is
bidirectional); ACON reports compaction helps *small* LMs most (+46%).

## 10. Risks

1. **Blob drift over long horizons** — many sequential compactions degrade
   coherence. Mitigation: rolling-merge blob mode trained from day one +
   blob-count curriculum + the P2 drift gate.
2. **0.8B reasoning ceiling** — pick benchmarks where recall dominates
   reasoning (the benchmark doc ranks by small-model feasibility).
3. **Ambient background stays decorative** (echo of nav failure) —
   Family D is staged, oracle-provenance-first, distillation-second,
   RL-only-if-gated, with `context_delta` as a cheap kill-switch.
4. **Serving adapter unknowns** — de-risked in P0 as a spike, not at
   release time.

## 11. Benchmarks

Full research in `docs/05_benchmark_targets.md` (2026-08-20). Program
summary (§5 there): **headline** = BABILong at 1M–10M tokens (sub-1B models
top its leaderboard, RAG documented as failing, soft-prompt-compressor niche
unoccupied) + LOCA-bench compaction curves; **credibility** = RULER
self-comparison per length; **comparability** = BenchPress + LongBench v1 +
the NQ/TriviaQA/HotpotQA soft-compression axis (xRAG/COCOM/PISCO/OSCAR);
**wide-net** = a self-built LogHub long-log QA benchmark (none exists —
doubles as the Family-B eval suite) + LongTableBench rebuttal;
**memory** = LongMemEval + BEAM-1M categories; **compaction** = AppWorld
with ACON/TRACE baselines (Pass² metric); **repo** = CrossCodeEval family
(published 0.5B anchors) + Long Code Arena huge bucket + SWE-QA/DeepCodeBench.
Universal protocol: fixed 0.8B decoder, arms = no-context / truncation /
LLMLingua-2 @ matched ratio / BM25-RAG @ matched budget / bgkit query-free /
bgkit query-conditioned (disclosed). Traps to avoid headlining: LongBench v2,
InfiniteBench realistic tasks, aggregation-bound suites.

## 12. P0 execution checklist (started 2026-08-20)

Status legend: [x] done · [~] running/partial · [ ] open · [B] blocked on the
external drive.

**Unblocked track:**

- [x] Blob format module + tests (`src/bgkit/data/blob_format.py`).
- [x] LogHub corpora → `bgkit-data-nvme/capability_packaging/loghub/`
      (13 Zenodo datasets ~450MB, BGL + HDFS_v1 extracted, + logpai/loghub
      2k-line samples repo).
- [x] BABILong 1k-samples eval set complete (2.9GB, splits 0k-128k) →
      `.../benchmarks/babilong/babilong-1k-samples/`; generator repo cloned
      (`babilong-repo/`) so training haystacks of arbitrary length can be
      generated locally — the 25GB pre-built train set
      (`RMT-team/babilong-train-5k-samples`) is deferred unless generation
      proves inconvenient.
- [~] SWE-Zero trajectories full snapshot (12GB) → `.../trajectories_ext/
      swe_zero/` (resumed after a session restart orphaned it; first shards
      landed and already schema-verified). Toucan SFT 500-row sample landed;
      median 9 turns / ~1K tokens, p90 ~5K → Toucan = tool-format
      diversity, NOT long-horizon stress; long-horizon compaction data
      comes from SWE-Zero + synthesized dialogues.
- [x] RULER repo cloned → `.../benchmarks/ruler/RULER`.
- [x] Compaction sampler IMPLEMENTED: `src/bgkit/data/compaction_sampler.py`
      + tests. Pure message-list surgery: anchor (system + task) preserved,
      turn-boundary-only cuts, per-segment vs merged blob modes, span-ref'd
      `source_ref`s (no text duplication), and recall-probe samples (path /
      symbol facts occurring only inside the compacted span). Smoke-verified
      on 30 real SWE-Zero trajectories -> 137 samples (25 probes), blob
      roundtrip checked. SWE-Zero schema: `trajectory` = OpenAI-style
      system/user/assistant(+tool_calls)/tool messages, median 55 msgs /
      ~25K tokens, p90 ~50K tokens; `model_patch` = ground-truth patch;
      per-row `license`. Still open: tokenize+loss-mask integration with the
      chat-template machinery — DONE 2026-08-21:
      `src/bgkit/data/blob_tokenize.py` (chat-template rendering w/
      OpenAI-style tool_calls normalization, target-only loss mask, blob
      sentinel spans; BPE-seam fallback) + tests; smoke on real SWE-Zero
      rows: 39/39 rendered, median 3.7K-token decoder sequences. REMAINING
      for the compaction trainer: source_ref->text resolution, encoder
      forward + splice, training loop (build after the lognav run frees the
      GPU).
- [x] LogHub log-QA generator: `src/bgkit/data/lognav_qa.py` +
      `scripts/build_lognav_qa.py` + tests. Question types: first_error /
      needle_token / count_keyword (flagged aggregation) / error_absent
      ("not present" negatives). Span-referenced windows (no text
      duplication), blob headers included. Smoke-verified at scale: 200
      samples over 40× ~1M-token windows of the full BGL log →
      `capability_packaging/lognav_qa/bgl_full_1mtok_qa.jsonl`. Still open:
      run over remaining LogHub systems + train/benchmark split with
      system-level holdout.
- [x] Recall-probe generator v1 (inside `compaction_sampler.py`): span-only
      facts (paths, def/class symbols) -> explicit probe question + exact
      answer. Extend later with tool-output values + user-constraint probes.
- [x] lognav v1 COMPLETE (2026-08-21, 2599 steps / ~45 min / ~0.87s/step):
      final held-out eval (zookeeper+apache) EM 0.295 / F1 0.533 /
      token-acc 0.517 / loss 1.71. Rapid learning to ~step 400, then EM
      pinned at exactly 59/200 — deterministic solved subset. Diagnosis:
      selection ceiling — survivorship aux was OFF (head frozen; θ
      bit-identical all run) so L1 selection never learned to keep needle
      lines; decode_seqlen math suggests actual retention ≈ target but
      UNVERIFIED under threshold-mode selection. Final ckpt
      `phase2_kb_step2598_...run-phase2_kb_lognav_v1`.
- [x] widenet v2 COMPLETE (2026-08-21, 3844 steps clean; ~1s/step):
      final held-out — fileneedle EM 0.275 / F1 0.634 / token-acc 0.595;
      lognav EM 0.269 / F1 0.529 (v1 lognav capability retained within
      ~2.6 EM pts while adding the second task). Final ckpt
      `phase2_kb_step3843_...run-phase2_kb_widenet_v2`. Cadence relaxed
      live at step ~900 (eval 250 / save 500). NOTE: trainer eval caps at
      256/640 eval rows (max_eval_samples applies even to explicit
      splits) — full-split numbers via eval_phase2_kb.py.
- [~] widenet v2 run details (launch record): lognav +
      fileneedle mix (9,017 train / 640 eval), continuing from the v1 final
      state (stage_a_checkpoint pinned explicitly — `auto` was rejected:
      it ranks by eval/loss across ALL stage-A phase2_kb ckpts incl.
      git-repro, cross-dataset losses incomparable). Deltas vs v1:
      survivorship_aux ON (utility-grad BCE trains the heads to keep
      answer-bearing lines) + `recursive_l1_tree.selection_mode_l1:
      exact_topk` (guaranteed ratios; head logits still rank survivors).
      Per-qtype breakdown of v1-final vs v2-final runs after v2 frees the
      GPU.
- [~] Baselines IN PROGRESS (2026-08-22): `scripts/baseline_babilong.py`
      written (HF generate, repo prompts/metrics, full + truncate arms;
      runs in-container via the new `/workspace/capability_packaging:ro`
      mount) — launches when the GPU frees. RULER: data prep works in the
      venv (needs `PATH=.venv/bin:$PATH` because prepare.py shells out to
      bare `python`; deps wonderwords/tenacity/scipy installed ad hoc);
      grid {niah_single_1,2, multikey_1, multiquery, qa_1} x {4K..32K} x 100
      generating; prediction via the `vllm-fast` service (Qwen3.5-0.8B
      :8091) after the GPU frees. `scripts/analyze_generative_eval.py` =
      reusable per-qtype generative breakdown.
- [ ] (superseded) Baseline benchmark runs (stock Qwen3.5-0.8B from HF cache;
      full-context / truncation / LLMLingua-2 arms) — GPU container work;
      NOTE: verify `flash_attn.cute`/quack import health in a fresh
      container first (known 2026-06-27 issue) or run baselines with plain
      SDPA attention, which these HF-only baselines permit.
- [ ] Serving spike: vLLM `--enable-prompt-embeds` end-to-end (stock model
      first; our projection output after the drive returns).

**P1 FIRST TRAINING RUN — LAUNCHED 2026-08-21 (lognav v1):**

- Dataset `lognav` built end-to-end and launched through the PROVEN
  `KRKBTrainer` flat single-bgkit path (fastest path to training; the blob
  trainer is built alongside): `scripts/build_lognav_phase2.py` → 960
  windows / 23.7M tokens (mmap `$DATA_DIR/mmap/phase2/lognav`), 3918 train
  + 200 eval trajectories (`$DATA_DIR/trajectories/lognav.parquet`,
  explicit split column, eval = held-out SYSTEMS zookeeper+apache), flat
  browse tree (994 nodes). Data note: most windows lack error-severity
  lines → first_error is rare (35) vs error_absent (925) — real
  distribution; revisit balance in v2.
- Config: `configs/experiment/phase2_kb_lognav.yaml` (inherits
  `/training/phase2_kb_stage_a`: live L0, direct L0/L1, GC-on, step-51945
  base) with datasets=[lognav], l0_retention.lognav=0.10,
  qwen_decoder_prob=0.80 (Qwen is the ship target — inverse of Stage A's
  Falcon bias; watch memory). Compose service `train-phase2-kb-lognav`
  (restart "no" during bring-up). run_name `phase2_kb_lognav_v1`.
- GPU container health RE-VERIFIED before launch: flash_attn +
  flash_attn.cute import clean (the 2026-06-27 quack blocker no longer
  reproduces); FlashQLA smoke passes (blackwell_sm121).

**Bring-up log (lognav v1):**

- Launch 1 (2026-08-21 ~09:13Z): healthy start — 4118 samples loaded,
  explicit split honored (3918/200), ~1s/step, loss 8.0→1.5 by step 30,
  CUDA peak ~10GB. CRASHED at step 49: the dynamic-ckpt scheduler made its
  first-ever DOWNSHIFT (full→megatron; free memory abundant for the first
  time in a KRKB run) and the live-L0 forward failed the non-reentrant
  strict recompute-match (CheckpointError 58-vs-55 — same class as the
  documented 666-vs-382 that made stage_a pin reentrant/per-layer GC).
  Latent since the 2026-06-29 mode-flip wiring; git-repro never downshifted
  (always memory-pressured).
- ALSO found: stale `${CHECKPOINT_DIR}/control.json` phase2_kb block (from
  git-repro) silently applied at step 0 (lr 5e-5, cadence, recursive keys).
  Removed the block (backup at
  `bgkit-data-nvme/capability_packaging/control.json.gitrepro_backup`);
  cadence now owned by the experiment config (eval 100 / save 200 /
  warmup 300).
- Fix for launch 2: `training.dynamic_ckpt.enabled: false` for this run
  (documented in the experiment config). **Engineering item (real fix):
  per-trainer allowed-mode restriction — the scheduler must never flip a
  model into a checkpointing mode its forward can't recompute; KRKB's valid
  set is {full} until megatron/off modes are made recompute-safe.**

**Per-qtype GENERATIVE eval of v2-final (2026-08-21, the pivotal finding):**

| capability (generative, held-out) | EM | F1 |
|---|---|---|
| not-present negatives (both tasks) | 0.96-1.00 | ~1.0 |
| code signatures | 0.10 | 0.43 |
| counts | 0.38 | - |
| verbatim log-line needles | 0.00 | 0.06 |
| assignments | 0.00 | 0.09 |

1. **Teacher-forced metrics flatter enormously** (EM 0.27 forced vs 0.00
   generative on needles — forced context permits local copying).
   Generation-based needle EM is the headline gate from now on
   (`eval_phase2_kb.py` post-run per checkpoint; note its report's
   `pred_answer` carries an `assistant`/`<think>` frame — normalize before
   scoring; trainer eval also caps at 256 of 640 eval rows).
2. **67x (L0 0.10 x L1 0.15) was the wrong STARTING ratio for verbatim
   retrieval** — gist/presence/structure survive, exact strings don't;
   consistent with the literature band (4-32x) and the RULER exact-needle
   warning. Curriculum problem, not physics.
3. **Response: ratio ladder** — v3a (~6x: L0 0.35 x L1 0.5, LAUNCHED
   2026-08-21 continuing from v2-final) -> v3b (~16x) -> v3c (67x), each
   stage gated on generative needle-EM recovery; the ladder doubles as the
   ratio-vs-quality Pareto curve for `docs/05_benchmark_targets.md`.
   (KRKB has no in-run retention ramp — staged runs implement the
   continuous-descent doctrine across stages.)

**Additional datasets built 2026-08-21 (post-mandate "carry on"):**

- [x] `swerecall` (Family A recall): 8,088 SWE-Zero trajectory spans /
      108M tokens / 8,403 recall probes (path/symbol/absent, held-out
      repos) via `scripts/build_swerecall_phase2.py` — the probe subset of
      compaction training with zero format compromise. Browse tree built.
- [x] `grepset` (Family B search results): 400 wide grep-output articles /
      2.2M tokens / 2,712 samples (which_file / quote_line / absent) via
      `scripts/build_grepset_phase2.py`; `repo_files.py` factored out as
      the shared bare-repo walker. Browse tree built.
- [x] v4 config + service staged (4-dataset mix from the v3a step-1019
      state; ratio pending the generative verdict).
- [~] v3a stopped early at step 1019 (graceful, rescue save worked):
      teacher-forced metrics FLAT at 6x (EM ~0.26) — forced ceiling is
      generalization-bound, not information-bound. Generative eval =
      decisive test; first attempt at 640 samples ran 2h (~200 done — 6x
      splices make generation ~15x slower; decode seqs ~13.6K); restarted
      at 192 samples. Eval-path warnings to investigate: sporadic zero-norm
      splices + `drilldown_zero_survivor count=400` in free-running
      (likely invalid generated IDs).

**Metric interpretation + v5 lever (2026-08-21, user calibration):**

- Verbatim recall under semantic compression is EXPECTED to degrade in
  general; generative verbatim needle-EM is therefore not a
  general-compression metric but **the query-conditioning capability
  metric** — the query names the target at encode time, so a
  well-conditioned encoder can protect that line verbatim while
  compressing the rest semantically. Report gist-F1 + presence/absence as
  the general metrics; verbatim needle-EM tracks pinpointing.
- Query conditioning is WIRED (verified: live path feeds the real question
  as the L0 prompt, L1 always query-conditioned) but arrived undertrained
  from the summarization lineage. Already pushing on it: per-window
  multi-query data (~4-5 queries/article), utility-grad BCE (v2+), v4
  volume. **v5 lever: gold-span relevance supervision** — generators know
  the answer line's token positions; emit them and boost those positions
  in the survivorship relevance loss under the query (data-schema +
  trainer change; the most direct pinpointing signal available).

**RATIO VERDICT + v4 (recorded 2026-08-22 after a session crash):**

- v3a generative eval (192 samples, 6x): verbatim needle-EM 0.000 on ALL
  needle slices — identical to 67x; gist/absence unchanged. RATIO IS NOT
  THE LEVER. The verbatim gap is generation/pinpointing (undertrained
  query conditioning), per the user's calibration; gold-span relevance
  supervision is the v5 lever. Ladder abandoned; v4 ran at 0.10/0.15.
- [x] widenet v4 COMPLETE (5,255 steps, clean; 4-dataset mix from v3a
      step-1019 at 0.10/0.15). Final held-out teacher-forced: fileneedle
      EM 0.310 / F1 0.666 (best yet), swerecall EM 0.214 / F1 0.768,
      grepset EM 0.154 / F1 0.552, lognav EM 0.200 / F1 0.483 (diluted),
      overall EM 0.234 / F1 0.668. Final ckpt
      `phase2_kb_step5254_..._run-phase2_kb_widenet_v4`.
- Ops: session crashed during v4 — monitors orphaned; recovered from
  logs/checkpoints. Fast ckpt dir at 288G, NVMe 72G free — prune old
  runs' non-final checkpoints after confirming HDD archival.

**v5 DESIGN — gold-span relevance supervision (2026-08-22, from code read):**

Current machinery: `_build_l1_turn_inputs` assembles the L1 content buffer
as `[pinned-id tokens | L0-survivor vectors]` per article and builds
`relevance_mask` = True over WHOLE gold articles; the L1 aux loss pushes
gold-group mean survival to `gold_boost*ratio` and distractors to
`damp*ratio`. L0 aux has NO relevance term. Single-gold-article flat
datasets therefore get zero positional signal about WHERE the answer is.

Change set (three pieces, all small):
1. **Data**: generators emit `gold_span_json = [start_tok, end_tok)` per
   sample — token offsets of the answer line INSIDE the tokenized article
   (search the full article tokenization, never re-encode substrings —
   BPE is not concat-distributive). lognav/fileneedle/grepset/swerecall
   all know the answer line at generation time.
2. **L0 relevance**: in `_l0_for_articles`, when a span is known for the
   gold article, add a per-position target: span tokens ->
   `gold_boost*ratio` (or a hard floor: span must survive), rest of the
   article -> ratio. This is the verbatim-survival lever — if the line
   dies at L0 nothing downstream can recover it.
3. **L1 relevance**: mark L1 positions whose L0-survivor SOURCE index lies
   inside the span (cumulative survivor count before span start, from the
   L0 survivor mask already available where `l0_flat` is gathered); boost
   those, keep other gold-article positions neutral, damp distractors.

Eval: generative needle-EM (the query-conditioning metric) before/after,
same 192-sample protocol. Expected: first nonzero verbatim recall.

**v5 IMPLEMENTED (2026-08-22, data + trainer):**

- Data: `src/bgkit/data/gold_span.py` (offset-mapping based, BPE-safe) +
  all four generators emit `gold_span_json`; writer column added.
  Generators also carry the error-centered lognav windows. Data-v2 rebuild
  of all four datasets pending (gated on the GPU eval finishing — the
  running eval reads the current artifacts).
- Trainer: `KBSample.gold_span`; `_l0_for_articles(gold_spans=...)` builds
  a flat L0 `span_mask` + stashes the survivor mask; `_prepare_l1_turn(
  gold_span=...)` maps span -> L1 survivor positions (k-th survivor row ==
  k-th True of the article's survivor mask) into `span_mask`; aux loss adds
  L0+L1 span terms `(1 - p_survive)^2` weighted by
  `survivorship.span_relevance_weight` (default 0 = off) + diagnostics
  `l0_span_survival`/`l1_span_survival`. Threaded conditionally for legacy
  test doubles; 54 trainer/dataset tests green.
- Config: `configs/experiment/phase2_kb_widenet_v5.yaml` (weight 0.5,
  from v4 final). Launch after data-v2 + GPU free.

**ZERO-REP DIAGNOSIS (2026-08-22) — v1→v4 were NOT real runs:**

While waiting on the v4 generative eval, the "investigate eval-path
warnings" item finally got its investigation. Evidence chain:

- v4 training log: `phase2_kb_drilldown_zero_survivor` count reached
  **82,400** over 5,255 steps; the rep-norm guard logged **1,650/1,650**
  sampled checks out of band (`n_rep_vectors=1, rep_norm_mean=0.0`) over
  82,500 decoder forwards (64,150 Qwen + 18,350 Falcon). Every decoder
  forward spliced exactly one ZERO vector.
- `train_step` lines: `theta_l0=-0.838` constant for 5,250 steps,
  `l1_buckets_per_step=0.0`, `tokens≈115` per sample → the encoder never
  ran; the decoder trained on question + answer only.
- Root cause: `flat_trajectory_row` copied the git-repro turn schema and
  tagged the single retrieval turn `is_head: true`. In
  `_prepare_sample_for_decode`, `is_head` routes unconditionally to the
  shared-tree **head** drill (`_shared_tree_head_survivor`), which for flat
  datasets has no memo, no splice dict and no L1-tree cache →
  `_drilldown_zero_survivor()`. `_prepare_l1_turn` (live query-conditioned
  L0→L1) was never called.
- Consequences: every v1→v4 metric (lognav EM .295/.20, fileneedle EM
  .31/F1 .67, swerecall F1 .77, grepset F1 .55) is a **question-only prior**
  (absent-qtype + guessable answers). Generative needle-EM 0.000 across
  v2/v3a/v4 was the reps being absent, not a ratio or a conditioning
  problem. The "ratio verdict" and the v3a probe conclusion are void. v5's
  span supervision would have been a silent no-op on this data (no
  `_prepare_l1_turn` call → no span mask).
- Fixes (root cause + global contract, no stopgap):
  1. `flat_phase2_writer.flat_trajectory_row` emits the plain leaf drill
     `{ids, query}` (the tags `is_head/drill_mode/structural_depth/
     is_retrieval` have no reader in the flat path).
  2. `KRKBTrainer._prepare_sample_for_decode` RAISES on an `is_head` turn
     when the trainer has neither the per-repo shared tree nor an L1-tree
     cache (head-drill infrastructure) — fails at the first sample instead
     of producing a fake run.
  3. `ReconstructionDecoder._maybe_guard_spliced_rep_norm` escalates: N
     consecutive sampled out-of-band checks (default 10 = 500 forwards) raise
     `RuntimeError`; `BGKIT_REP_NORM_GUARD_RAISE_AFTER=0` disables. The
     log-only warning was ignored for four runs.
  4. Tests: writer contract, trainer contract (raise / legal-with-cache),
     decoder streak raise + reset + disable.
- Run plan: v5 restarts from the UNTAINTED summarization base (no
  `stage_a_checkpoint`; v4's decoder drifted on question-only input, its
  encoder is byte-identical to base), `n_distractors: 0` (single-document
  tasks; distractors quadruple live-L0 cost on 10–40K-token windows),
  `max_steps: 4000` until the real step time is known. The v4 generative
  eval was stopped (it measured a zero-rep decoder). Baselines
  (BABILong/RULER) are unaffected — stock models, no bgkit.
- Lesson recorded in memory: zero-norm / zero-survivor warnings mean
  "reps ABSENT", a blocker, never deferrable noise.

**v5 LAUNCH LOG (2026-08-22):**

- 07:13Z first launch aborted: lognav still carried `is_head` (its builder
  inlined the turn instead of using `flat_trajectory_row`) — builder now
  uses the shared writer; lognav rebuilt (all four datasets verified
  `{ids, query}`; gold-span coverage lognav 62% / fileneedle 77% /
  grepset 85% / swerecall 80%, remainder = negatives + counts).
- 07:16Z launch: step 0 shows the encoder ALIVE for the first time on
  these data — `l1_buckets_per_step=3`, `l0_span_survival=0.28`,
  `l1_span_survival=0.89`, peak CUDA 33.8 GB, loss 7.72 (decoder meets a
  new task format). First steps ~30 s/step (fla autotune warm-up; re-measure
  at step 100).
- Guard recalibration: the Qwen splice norm from the 51945 base is 18.8 vs
  embed row-norm 0.50 (ratio ~38x; Falcon: same rep norm / 2.95 = 6.4x).
  The absolute 8x band came from the git-repro lineage (~4x, projection
  norm reg target 4.2 — NOT enabled in the Stage A configs). Guard now
  raises on DEGENERACY only: absolute floor (absent/collapsed) or >8x drift
  from the decoder's own first-readable reference ratio (the 4x→320x
  runaway class); the absolute band warns once, informationally.
  Restarted 07:23Z (rescue save at step 9, auto-resume).
- 07:29Z restart #3: sampled rep-norm INFO line (`spliced_rep_norm_sample`,
  every 50th forward) for trend visibility. Qwen ratio steady 36.4 → 37.2
  in-process (the 50.2 reading was sample variance).
- 07:39Z a zero splice on Qwen (1 vector) → traced to **swerecall: only
  3,200 of 8,088 spans were in its browse tree** (`build_browse_tree.py`
  flat path silently truncates at `--leaf-cap`; the setup coverage scan
  only checked the token store and reported `coverage_scan_ok`). ~60% of
  swerecall trajectories resolved to NO article → `None` turn → zero
  splice. Fixed: tree rebuilt with `--leaf-cap 1000` (0 missing across all
  four datasets), rebuild script carries the cap + a fatal
  trajectory→tree coverage assertion, and
  `_validate_trajectory_article_coverage` now RAISES on tree-unresolvable
  retrieval ids (tests added). Restarted 07:43Z (rescue save at step 31).
- Step time observed ~30–36 s/step over steps 0–31 (fla autotune warm-up
  included; re-measure at step 100). Peak CUDA 45.5 GB at step 10.
- 07:59Z STOPPED at step 43: 77 s/step, host available dipped to 18 GB,
  decoder sequences 17–78K tokens (`decode_seqlen`), rep counts up to
  16K for single articles. Direct diagnostic
  (`scripts/diag_flat_splice_counts.py`, runs the real
  `_prepare_l1_turn`/`_run_l1_batch` on train samples) on the untrained
  base: **L0 threshold mode keeps 0.0–0.3%** (61 of 21,559 tokens; several
  articles 0 survivors — the "confidently silent" failure that motivated
  exact_topk for tree leaves in July); the aux losses then flip the
  saturated head to ~100% within ~40 steps. Root cause #3 of the day: the
  `recursive_l1_tree.selection_mode_l1` parse was nested under
  `recursive_l1_tree.enabled`, so every widenet config's `exact_topk` was
  silently ignored — BOTH levels ran threshold mode, and the flat leaf L0
  mode was hard-wired to threshold unless the recursive flag was on.
  Fixed: first-class `training.selection_mode: {l0, l1}` parsed
  unconditionally (legacy key = L1 fallback), applied to the flat leaf L0
  path, θ dual ascent skipped for exact_topk levels, logged at setup
  (`phase2_kb_selection_modes`). Diagnostic with exact_topk: L0 keep ≈ the
  sampled ratio (0.02–0.32 jitter around 0.10), reps 0.6–10% of N,
  decode_len 750–2,500.
- 08:08Z **v5b** launched: cold start (run_name bump; the v5 rescue saves
  9/15/31/43 deleted — trained on 0%→100% splices), exact_topk both
  levels, `eval_ablation_modes: [zeroed]`. Watching: s/step, host
  available, rep-norm ratio, `l0/l1_span_survival` (now meaningful: the
  fraction of answer-span tokens that win top-k).
- 08:09Z host available 18 GB at step 0: `cuda_reserved 94 GB` vs 12 GB
  allocated — allocator-reserve bloat on unified memory with the adaptive
  flush off (`dynamic_ckpt.enabled: false`, the per-run dodge of the
  mode-flip crash). Interim: live `cuda_empty_cache_every_step: 1`
  (available → 102 GB). **Principled fix (user: "do it right")**: the filed
  per-trainer allowed-mode restriction — `BaseTrainer._dynamic_ckpt_allowed_modes()`
  (default all) + `_resolve_allowed_ckpt_mode` routing disallowed
  transitions to the nearest allowed mode or skipping them (logged);
  `KRKBTrainer` declares `{full}` because its outer non-reentrant level
  checkpoint cannot recompute under megatron's inner selective policy
  (the 58-vs-55 class). The scheduler is now ON with host defaults for
  KRKB (flush tier active, flips impossible by contract), the per-run
  `enabled: false` and the legacy control.json cadence are gone. Tests:
  `test_dynamic_ckpt_allowed_modes.py`. Restarted ~08:27Z (rescue save,
  auto-resume).
- Decision record (not a workaround): `exact_topk` is the operator for
  budgeted wide-net compression — deterministic retention at the requested
  ratio; the head learns RANKING through BCE/decisiveness/span terms.
  Threshold-θ selection remains the operator for ratio-free deployment and
  needs its own calibration pass on this domain.
- v5b progress before the restart: loss 7.7 → 4.3 (10) → 2.9 (20) → 1.6
  (25) at ~32 s/step; peak CUDA 31–47 GB.
- 08:30Z: θ-anchor probing (`anchor_sampling_prob: 0.30` over the full
  grid up to 0.95 — a θ(r) calibration mechanism) is now auto-disabled for
  exact_topk levels (`_anchor_free_if_topk`); it produced the 37K-rep decode
  outlier with no θ benefit. 08:33Z: per-phase step timing wired into the
  flat path (`time/prep|drill_encode|assemble|decode_fwd|aux|backward|util`,
  `profile_timing: true`) — the in-process profile that decides the speed
  lever (perf playbook: measure first). Final restart of the bring-up
  (resume from step 34).
- 09:00Z TIMING (per microbatch, GPU-synced, ~27 s/step): prep incl. live
  L0 1.2 s, L1 0.1 s, decode fwd 0.25 s, backward 4.4–5.5 s = **~6 s of
  model work**; `assemble` 5.7–8.4 s (trivial slicing code — absorbing
  something); ~13 s outside any timed phase. The quadratic-attention
  hypothesis is WRONG; ~3–4× of the step is Python/driver overhead.
  Attribution needs a stack profile: `py-spy` is installed but needs
  `sudo` (ptrace_scope=1) — command handed to the user; analysis pending.
- 09:04Z memory truth: pre-flush free dips to 11 GB at step 100 with 91 GB
  reclaimed — the per-step working set is ~90 GB, and the logged
  `cuda_max_allocated_gb≈40` was WRONG because the `BGKIT_MEM_BREAKDOWN`
  probe resets the peak counter per phase. Fixed via
  `BaseTrainer._step_peak_allocated_gb_hook` (KRKB reports the max over
  its probes since the last log). Safety: `BGKIT_CUDA_MEM_FRACTION=0.80`
  (`set_per_process_memory_fraction`, OOM not freeze) + 110 GB
  system-used abort rail. Microbatch budget (`max_microbatch_l0_tokens`
  52K) is the memory lever if needed; decided after the profile.
- 09:45Z **py-spy PROFILE (90 s, main thread fully accounted)**: backward
  63%, live-L0 prep 17%, post-backward sync 12%, decode fwd 4%, L1 1%,
  optimizer 1%, dynamic-ckpt 1%. `train_step` counts OPTIMIZER steps (4
  microbatches, grad-accum 4): 27 s/step = **6.8 s per microbatch, all
  model work**. My "13 s untimed / 8 s assemble" reading was wrong:
  `time/assemble` shows 5–10 s but has ZERO profile samples — an
  instrumentation artifact (open follow-up: `_timed` gpu=False scope vs
  the profile; not on the critical path). Conclusion: the step is bound
  by token volume (L0 backward-with-recompute over ≤52K-token
  microbatches), so the only speed lever is data volume per microbatch,
  which would mean dropping the long windows that define the wide-net
  task. Decision: keep the data; one epoch = 1,314 optimizer steps ≈ 10 h
  → `max_steps` set to 1,400 live via control.json (was 4,000 = 3 epochs);
  evals every 250 steps (~1.9 h) with the zeroed-rep ablation gap.
- 10:17Z **step-250 eval is a MEASUREMENT DEFECT**: eval/loss 9.84 (train
  ~1), headline == zeroed ablation (F1 0.175 vs 0.178), while the guard
  shows real in-band reps (258–526 vectors) spliced during eval. Root
  cause = the recorded Falcon-H1 asymmetry: `evaluate()` runs
  `model.eval()`, and the Mixer's eval-mode path (unfused conv/scan +
  padding masking) is numerically different from the training path (the
  summarization trainer's 5-nat gap, 2026-05-15, fixed there with
  `decoder.train()`; never applied to KRKB). Fix: `_teacher_forced_decoders()`
  context — decoders in train mode under no_grad for every teacher-forced
  pass (headline + ablation sweep; the eval script's `evaluate_sample`
  too), generation passes keep eval mode; plus `eval/family/{qwen35,
  falcon_h1}/{loss,token_f1,exact_match,n_samples}` so a family-specific
  defect can never hide in the pooled number again. Tests added.
  Restarted ~10:30Z; the first trustworthy eval is step 500.
- 10:36Z **validation eval at step 275** (pulled forward via live
  `eval_every`): pooled eval/loss 9.84 → **3.79**; reps load-bearing on
  exact answers (EM 0.051 vs zeroed 0.008; fileneedle 0.116 vs 0.023;
  lognav 0.063 vs 0.0). Per family — **Qwen**: loss 1.43, F1 0.557, EM
  0.102 (zeroed EM 0.016; loss gap ~0 this early); **Falcon-H1-90M**:
  loss 6.27, F1 0.247, EM 0 (zeroed 6.59). Falcon's zeroed loss is worse
  than v4's question-only ~2.7, and eval/train now share the exact same
  prep/splice/forward code, so the remaining question is whether Falcon's
  TRAIN loss is also ~6 (destabilized 90M companion at 20% share) or an
  eval-only residue. Added `loss/{family}` train metric (deploy on next
  restart) to settle it; if Falcon is genuinely diverged, the principled
  move is `qwen_decoder_prob` → 1.0 (live) for this run and a separate
  Falcon bring-up, not a pooled headline that averages in a broken family.
- 11:17Z per-family TRAIN loss (steps 300–355): Qwen 0.6–1.0, **Falcon
  1.6–3.7 (≈2.6)** → Falcon's eval 6.3 is a ~3.7-nat train/eval gap
  (Qwen's is 0.6). Exhausted path differences: G=1 training decodes via the
  same `forward_interleaved_with_loss`; decoders are in train mode for
  eval; attention dropout 0; no decode cap; same loss-token counts; 262K
  context. Remaining hypothesis = held-out generalization of the 90M
  companion. Settled by a first-class **train-subset generalization probe**
  (`eval_train_subset_samples: 64` → the same teacher-forced pass over a
  fixed slice of training samples, keyed `eval_train/...` per dataset +
  family). Restarted 11:21Z (resume 365), validation eval pulled to step
  370. Headline for this run = `eval/family/qwen35/*` until Falcon is
  understood.
- 11:32Z **step-370 eval + train-subset probe**: pooled 1.72; Qwen loss
  1.16 / F1 0.638 / EM 0.211 with train-subset F1 0.654 / EM 0.156 → no
  generalization gap; **Falcon** held-out 2.30 (from 6.27!) but
  train-subset **7.84** with F1 ≈ held-out (0.35 vs 0.32) → loss dominated
  by a few catastrophic tokens, unstable across evals/samples — a 90M
  companion pathology, not a path bug. DECISIONS (live): `qwen_decoder_prob
  1.0` (Falcon off for this run; per-sample-loss investigation filed) and
  `p_skip_bgkit 0.0` (the KR-era "answer without reps 15% of the time"
  roll directly rewards the shortcut on needle tasks).
- **The real finding**: Qwen's zeroed-rep ablation is now ≈ headline (EM
  0.203 vs 0.211; F1 0.634 vs 0.638; loss equal) — the question-only
  shortcut is re-emerging with REAL reps present, and the cause is
  upstream of the decoder: `l0_decisiveness_loss 0.99` → the L0 head's
  p≈0.5 everywhere → its exact_topk ranking is noise → `l0_span_survival`
  0.076 ≈ the 0.10 chance rate. The needle dies at L0, so the decoder has
  nothing to read and learns priors. Hypothesis: GLOBAL grad clipping
  (`max_grad_norm 1.0`, pre-clip norms 200–400 from the decoder/backbones)
  scales the head's gradient by ~1/300 — the head effectively trains at
  lr ~3e-7. Added per-group pre-clip `grad_norm/<group>` diagnostics
  (l0/l1 backbone, head, bridge, projection, decoders; every 25 steps) to
  measure it before changing the optimizer. If confirmed: per-group
  clipping (heads clipped on their own norm) is the principled fix.
- 11:45Z MECHANISM in code: `exact_topk` ranked on `logits_for_op =
  tanh(base_raw/T)` (level_compressor.py:495). With the head at the tanh
  floor (θ_l0≈−0.84, p≈0.46 ⇒ decisiveness 0.99; T_l0 = 2.24 from the
  cold-start calibration) the squashed scores collapse toward exact ties
  and top-k degenerates to arbitrary picks; tanh′ ≈ 0.1–0.2 also throttles
  every loss defined on it. FIX (shipped): exact_topk now ranks on
  `base_raw / T` — identical ordering wherever tanh is not saturated,
  strictly ordered where it is; θ and the loss probabilities stay in
  tanh space (threshold operator untouched). Regression test: saturated
  head with best scores at the end → survivors [4,5,6,7]. Also built
  `training.grad_clip_mode: per_group` (each `_grad_norm_param_groups`
  group clipped on its own norm; logged `grad_norm` unchanged = global
  pre-clip) as opt-in, to be enabled only if `grad_norm/<group>` confirms
  head starvation. No ICE-teacher load event was found in any captured
  log → the 200-step BCE re-anchor probably never ran (open check).
- 11:57Z **per-group norms at step 400**: total 4544 = `l1_head` 4556
  (decoder 5.3, l0_head 23, l0_backbone 0.23, l1_backbone 0.02,
  projection 0.17, bridge 0.003). The `l1_head` group included
  `survive_embedding` — a single flag vector summed over thousands of
  survivor positions — now reported as its own group. Clipping
  hypothesis REVISED: Adam/Muon updates are invariant to a uniform
  pre-clip scale, so global clipping does not starve the heads;
  `grad_clip_mode: per_group` stays opt-in (not enabled). Meanwhile the
  latest step-400 line shows the head IS learning the span (`l0_span_loss`
  0.020 ⇒ p≈0.86 on span tokens; decisiveness 0.99 → 0.49) while
  `l0_span_survival` stays ≈ chance — the saturation-tie signature
  exactly: the span is pushed to the tanh ceiling it shares with many
  other positions, and top-k over ties is arbitrary. Deploying the
  raw-score ranking now; success criterion = `l0_span_survival` rising
  well above 0.10 within ~100 steps, then the zeroed-ablation gap
  reopening at the step-500 eval.
- 12:25Z chat-template audit (user question): both decoders render via
  their own tokenizer's `apply_chat_template(tools=…)` — Qwen native
  `<tool_call>/<function=bgkit>` + `<tool_response>` user turn, answer loss
  span = full assistant turn incl. `<|im_start|>assistant … <|im_end|>`;
  Falcon native JSON `<tool_call>` + `<tool_response>`, its template's
  tool→assistant `<|im_end|>` bug patched in Phase 2 too
  (`falcon_chat_template_patched` logged; the renderer rejects the
  unpatched template). Falcon's instability is NOT template-related → stays
  a per-sample-loss investigation. First post-fix `l0_span_survival`
  readings (steps 450–465): 0.26 / 0.12 / 0.14 (pre-fix ~0.08–0.11) —
  directionally up, noisy; step-500 eval is the read.
- 12:45Z **step-500 eval** (ranking fix live ~80 steps): pooled loss 1.65,
  EM 0.141 vs zeroed **0.098** (gap reopened: Δ 0.043; loss gap 0.05 →
  0.13), lognav EM 0.170 vs 0.080, fileneedle 0.163 vs 0.140, swerecall/
  grepset unchanged; Qwen family loss 1.16 / F1 0.635 / EM 0.19. BUT
  `l0_span_survival` still 0.09–0.15 ≈ chance while `l0_span_loss` ≈ 0.02
  (span tokens at p≈0.86) and decisiveness ≈ 0.5 — i.e. the head marks
  ~half of ALL tokens as relevant; nothing in the L0 objective pushes
  non-span tokens down under exact_topk (L0 `ratio_loss_weight` 0 — the
  rate used to be owned by threshold dual-ascent, which exact_topk
  skips). Next: enable the aggregate ratio loss at L0 (negative pressure
  toward the 0.10 target) so span-vs-rest separates; the cleaner long-term
  objective for exact_topk is a ranking loss (span above the k-th score).
- 12:50Z: found that KRKB's inline aux reads the TOP-LEVEL
  `survivorship.ratio_loss_weight` (default 0.1, both levels) — the
  per-level `l0/l1.ratio_loss_weight` keys in the Stage A config are not
  consumed by the inline path (config-surface inconsistency; follow-up:
  make the inline path honor `LevelLossCfg` per level like the shared
  helper does). So the ratio loss was on at 0.1; raised to **0.5 live**
  (`ratio_loss_weight` in control.json) at step ~505. Watching
  `l0_ratio_loss`, decisiveness, and `l0_span_survival` over the next
  ~100 steps; step-750 eval is the next read.
- 12:55Z span-survival trend steps 450–520 (ranking fix live, ratio loss
  0.1): l0 0.26/0.12/0.14/0.09/0.15/0.14/0.27/0.10/0.05 (mean ≈ chance),
  decisiveness pinned 0.49 → the ranking fix alone is not enough; the
  head marks half the tokens as relevant, so the negative pressure is the
  missing piece (ratio 0.5 from step 505). Split groups at step 495:
  `grad_norm/l1_head` 77, `l1_survive_embedding` 4e-5 → the 4556 at step
  400 was the L1 head proper (gradients scale with Qwen's large hidden
  activations; benign under Adam/Muon).
- 13:40Z **L0 head FROZEN after the ratio loss** (steps 610–635):
  decisiveness 0.989, ratio 0.120, span 0.304 constant to 3 decimals;
  p≈0.46 everywhere ⇒ tanh(raw/T) = −1 for every token (the ratio term
  drove all raw scores into the floor; at the rail tanh′≈0 so no loss can
  move the head — the "confidently silent" collapse, uncatchable by
  min-survivors under fixed top-k counts). Structural cause: under
  exact_topk every loss is still defined on the tanh-squashed score.
  FIXES (shipped): (1) under exact_topk `logits_for_op = base_raw / T`
  (unsquashed; consumers = ranking + every aux loss; threshold mode
  untouched); (2) on resume, T is re-estimated from the LOADED head for
  exact_topk levels (`_pre_train_loop`; the setup probe ran on pre-restore
  weights and the checkpoint then overwrote T = 2.24 while raw scores
  drifted) and the probe now uses the first retrieval turn's ARTICLE
  tokens (2K prefix) instead of the question text; (3) diag script prints
  raw-score stats + rail fractions. Deploy = stop → diag → relaunch.
  Pre-existing, unrelated: 3 failures in `test_encoder_packed.py`
  (`EncoderOutput.survivor_mask` attribute; uncommitted test edits from
  an earlier session).
- 14:00Z DEPLOYED (restart from step 655). Post-restore recalibration on
  the TRAINED head: **T_l0 = 119.9, T_l1 = 89.8** (base head on article
  tokens: 2.96 / 6.90; the old checkpoint value 2.24 came from the
  question-text probe on pre-restore weights) — the heads' raw-score
  scale grew ~40× during training while T never moved, so `raw/T ≈ ±50`
  sat deep in the tanh rails. First post-fix step (660): decisiveness
  0.989 → 0.943, span loss 0.30 → 0.15, ratio loss re-based (0.27 in the
  new score space) — losses are no longer constant, i.e. the head has
  gradient again. (Diag caveat: `diag_flat_splice_counts.py` runs on the
  BASE weights — it never loads the run checkpoint — so its T/keep
  numbers describe the base head; resume-time values come from the
  training log.) Watching steps 700/750 + the step-750 eval.
- 14:40Z **first above-chance span survival**: `l0_span_survival` 0.06,
  0.07 (steps 690–695) → 0.26, 0.28, 0.21, 0.24, 0.31, 0.23, 0.11, 0.30
  (700–735) ≈ 2.5× chance and rising; span loss 0.16 (p_span ≈ 0.6),
  ratio loss 0.27 → 0.23 (mean p still ≈ 0.58, target 0.10),
  decisiveness 0.97 (not yet decisive). The head separates span from rest
  for the first time in the run. Next read: step-750 eval (zeroed gap).
- 14:48Z **step-750 eval** (95 steps after the re-normalization): reps
  are MORE load-bearing — Qwen EM 0.156 vs zeroed 0.094 (Δ 0.062; step
  500 pooled Δ 0.043), F1 0.578 vs 0.544; pooled EM 0.105 vs 0.051 (2×);
  fileneedle 0.163 vs 0.070 (2.3×), lognav 0.134 vs 0.071, swerecall
  0.054 vs 0.014. Absolute Qwen numbers dipped (loss 1.16 → 1.36, F1
  0.635 → 0.578) — expected transient: the survivor set changed abruptly
  at step 655 (new score space + T) and the decoder is re-adapting to the
  new rep distribution; train-subset Qwen loss 0.93 ≈ training loss.
  ~650 steps (~5.5 h) remain to max_steps 1400; next eval at 1000
  (~16:50Z). Read both curves: zeroed gap (mechanism) and absolute F1/EM
  (decoder recovery).
- 15:20Z windowed `l0_span_survival` 0.130 (660–705) → 0.241 (710–755)
  → 0.189 (760–805): above chance, plateauing; decisiveness flat 0.97 —
  the bulk at p≈0.58 vs span ≈0.6 is a thin margin. Found the inline aux
  path applying the GLOBAL `decisiveness_loss_weight` 0.05 to L0 although
  Stage A explicitly sets `l0.decisiveness_loss_weight: 0.0` (at p≈0.58
  that term pushes the bulk up against the ratio pull). Fixed: per-level
  weights honored (`_aux_weight`: live override > EXPLICITLY configured
  per-level > global; a default-constructed per-level cfg never silences
  the global — caught by `test_kr_kb_packed` doubles). Deployed 15:22Z
  (resume from 826); for this config the deployed and fixed precedence
  give the same weights (L0 decisiveness 0, L1 0.05, ratio 0.5 both via
  the live override). Restart discipline note: gate restarts on pytest's
  exit code, not a grep.
- 15:26Z post-restore recalibration: **T_l0 = 1030** (120 at step 655;
  2.96 on the base) — the L0 head's raw scale grew ~9× in 170 steps. The
  remaining structural hole: with an unsquashed score the ratio/span
  losses are satisfiable by INFLATING the scale (sigmoids saturate to
  {0,1} with 10% positives → ratio loss → 0) rather than by re-ordering,
  and saturated mis-ranked span tokens get no gradient → the ~0.2
  plateau. FIX (shipped): under exact_topk the operator score is the
  **per-document z-score** of `base_raw` (`_segment_zscore`: ranking
  unchanged, scale-free, O(1), differentiable; T irrelevant). Tests:
  ×1000 scale → identical scores and survivors; per-segment
  standardization; saturation regression keeps survivors [4..7]. Deploy
  via stop/relaunch (resume ~step 840).
- 15:31Z deployed z-scored exact_topk (resume 836; post-restore T_l0 had
  reached 1187 — the inflation ran to the end). Under z-scores ratio/
  decisiveness are ~constant (0.307 / 0.76 for a ~Gaussian score), so
  the span term alone drives rank quality: span loss 0.19 → 0.12 over
  100 steps; windows 0.20 / 0.22 / 0.17 span survival. Made
  `span_relevance_weight` live-tunable (handler + test; deploys next
  restart). Also seeded the flaky clone-evolves test.
- 16:55Z **step-1000 eval** — best so far and the right trend: Qwen loss
  1.27 (1.36 @750), F1 0.611 (0.578), EM 0.1875 (0.156); zeroed Qwen EM
  **0.070** (0.094 @750) → gap 2.7× (Δ 0.117; @500 0.043, @750 0.062);
  pooled EM 0.117 vs 0.039; swerecall 0.108 vs 0.0, fileneedle 0.163 vs
  0.070, lognav 0.125 vs 0.063. Train-subset Qwen 0.90 / F1 0.646 (no
  gap). The no-reps baseline FALLS while the headline rises: the decoder
  is moving from priors to reading the blob. Decision: no more
  intervention this run; eval @1250 (~18:55Z), end @1400 (~20:10Z), then
  the generative per-qtype eval on the final checkpoint (the headline
  gate) and the stock-model baselines while the GPU is free.
- 18:53Z **step-1250 eval** — monotone series: Qwen loss 1.255 (1.27),
  F1 0.624 (0.611), EM 0.195 (0.1875); zeroed Qwen EM 0.0625 (0.070) →
  gap 3.1× (Δ 0.133; 0.043 @500 → 0.062 @750 → 0.117 @1000 → 0.133).
  Pooled EM 0.125 vs 0.035; swerecall 0.122 vs 0.0; lognav 0.134 vs
  0.054; fileneedle 0.163 vs 0.070. Train-subset Qwen 0.87 / F1 0.648.
  Head: span loss 0.06–0.10 (p_span ≈ 0.7–0.75), `l0_span_survival`
  windows ≈ 0.2–0.27. Post-run chain armed
  (`scripts/post_run_widenet_v5b.sh`, detached): final-ckpt generative
  per-qtype eval vs the v2 zero-rep report → BABILong baseline.
- 20:07Z **v5b COMPLETE (step 1400, one epoch + change, ~12 h wall incl.
  ~9 restarts)**. Final teacher-forced (held-out, 256 samples):
  Qwen loss 1.255 / F1 0.624 / EM 0.195; zeroed-rep Qwen EM 0.0625, F1
  0.553, loss 1.273 → reps load-bearing 3.1× on EM. Per dataset EM
  (headline / zeroed): lognav 0.134 / 0.054, fileneedle 0.163 / 0.070,
  swerecall 0.122 / 0.0, grepset 0.037 / 0.0; token-F1 lognav 0.42,
  fileneedle 0.49, swerecall 0.59, grepset 0.37. Train-subset Qwen loss
  0.87 / F1 0.65 (no generalization gap). Head: `l0_span_survival`
  windows ≈ 0.2–0.3 (chance 0.10), span loss ≈ 0.06–0.10. The headline
  gate (generative per-qtype needle EM vs the v2 zero-rep reference) and
  the BABILong stock baseline run next in the detached chain.
  Context for the numbers: v1→v4's "EM 0.31" was question-only; this is
  the first run where the encoder trained at all, and it did so through
  three selection-operator fixes deployed mid-run (raw-score ranking @655,
  z-scored scores @836) — a clean re-run from step 0 with the final
  operator is the proper v6 baseline.
- 20:20Z **overnight chain (user: "autonomously keep on going")**:
  stage 1 (`post_run_widenet_v5b.sh`, running): generative eval of
  step-1399 → per-qtype analysis vs v2 → BABILong baseline. Stage 2
  (`post_chain_widenet_v6.sh`, detached, waits for POST-RUN DONE): RULER
  baseline for stock Qwen3.5-0.8B via `vllm-fast` (context length now
  env-parameterized `VLLM_MAX_MODEL_LEN_FAST`, 40960 for the 32K prompts;
  the predictor records empty predictions on 4xx instead of aborting) →
  scored per length with RULER's evaluate.py → vLLM stopped → **v6
  launched** (`configs/experiment/phase2_kb_widenet_v6.yaml` +
  `train-phase2-kb-widenet-v6`: cold start, z-scored exact_topk both
  levels, per-level aux (L0 decisiveness 0), ratio 0.5, span 0.5,
  Qwen-only, p_skip 0, zeroed-ablation + train-subset evals every 250,
  2 epochs = 2,630 steps ≈ 20 h). Outputs: `$FAST/eval_reports_widenet_v5b/`
  (+ `analysis_vs_v2.txt`), `$FAST/baselines/babilong_qwen35_0p8b/`,
  `$NVME/benchmarks/ruler/pred_qwen35_0p8b/L*/summary*.csv`, logs
  `$FAST/post_run_v5b.log`, `$FAST/post_chain_v6.log`.
- 22:57Z **v5b GENERATIVE per-qtype (the headline gate), 192 samples,
  vs the v2 zero-rep reference**: verbatim needle EM **0.000 everywhere**
  (fileneedle/needle 0/12, lognav/line_needle 0/78, grepset/quote_line
  0/18, swerecall path+symbol 0/6; needle F1 0.05–0.10); negatives
  REGRESSED (fileneedle/absent 0.75 vs 0.955; lognav/error_absent 0.50
  vs 1.0; lognav/count 0.125 vs 0.375). Needle generations are empty or
  degenerate (`''`, `'____…'`), not near-misses. Reading: the decoder
  now leans on the reps (teacher-forced gap 3.1×) but the needle isn't
  in them — span survival ≈0.25 at L0 × ≈0.2 at L1 ≈ 5% of a needle's
  tokens — and a model asked to copy what it cannot see emits filler.
  v2's perfect negatives were question-only priors, so their regression
  is the flip side of rep-dependence, not a loss of capability per se.
  DECISIONS: v6 `span_relevance_weight` 0.5 → **2.0** (the only
  rank-quality signal under the z-scored operator; live-tunable). OPEN
  CHECK for the next GPU window: generation-path parity — first-token
  argmax under `generate_with_segments` (eval mode + cache) vs the
  teacher-forced train-mode forward on the same needle sample, to rule
  out a Qwen eval-mode/cache asymmetry with 37× reps (same class as the
  Falcon finding). Note the teacher-forced EM of 0.195 is mostly
  negatives + partial credit; generative needle EM stays THE gate.
- 23:19Z **BABILong stock baseline** (Qwen3.5-0.8B, greedy, 100/cell,
  `$FAST/baselines/babilong_qwen35_0p8b/summary.json`): qa1 0.55 / 0.43 /
  0.45 / 0.53 / 0.48 / 0.52 / 0.48 (0k…32k), qa2 0.44 / 0.46 / 0.39 /
  0.23 / 0.23 / 0.14 / 0.15, qa3 0.29 / 0.31 / 0.25 / 0.31 / 0.29 / 0.35 /
  0.29. The "truncate" arm equals "full" at every length because its
  budget was the 32K max — follow-up: rerun the truncation arm at a short
  budget (e.g. 4K) as the "no long context" reference; the full-context
  rows stand as the stock reference for the bgkit-compacted comparison.
- 23:20Z stage 2: RULER is BLOCKED — `vllm-node:latest` (the GB10 vLLM
  image from the inference-server setup) is no longer present locally
  and is not pullable; the launcher times out on the health check and
  falls through to the v6 launch (by design). Follow-up: rebuild/restore
  the vLLM image (docs: `make vllm-server`, `VLLM_IMAGE_FAST`), then run
  `scripts/baseline_ruler_predict.py` + RULER `evaluate.py` per length
  (grid already generated under `$NVME/benchmarks/ruler/data`).
- 2026-08-23 01:31Z **v6 step-250 eval**: Qwen loss 1.88 / F1 0.504 / EM
  0.0625, zeroed ≈ headline (v5b's gap opened after 500). Head: L0 span
  survival already 0.19–0.36 at 150–250 (v5b ≈ 0.10 there) — the 4× span
  weight on the clean operator separates faster. Measurement defect
  found: the eval alternated Qwen/Falcon per sample although v6 trains
  Qwen-only → pooled eval/loss 10.0 (untrained Falcon half) and half the
  eval time wasted. Fixed: `_eval_family_for_index` follows the training
  mix (`qwen_decoder_prob` ≥1 → Qwen only; ≤0 → Falcon only; else
  alternate), used by `_eval_pass`, the train-subset probe and
  `scripts/eval_phase2_kb.py`. Restart at step ~260 to deploy.
- 01:38Z restarted (resume 258). Caught on the resume line: the schedule
  said `max_steps=1400` — v5b's stale `phase2_kb` control.json block
  (per-PHASE namespace) had overridden v6's 2630 at launch. Live block
  reset to `{max_steps: 2630}` (handler extends the horizon and the
  cosine schedule). Memory note filed
  (`feedback_control_json_per_phase_namespace`). Root fix landed the same
  hour: `LiveConfig(path, namespace=phase, run_namespace=run_name)` —
  run block overrides the phase block key-wise; inherited phase keys
  are logged at WARNING (`live_config_phase_block_inherited`) on the
  first poll. Tests in `test_live_config.py` (26 pass); training suite
  759 pass. Deploys with the next container start (the live v6 process
  still reads only the phase block). Also seeded the flaky
  `test_commit_encoding_trainer` fixture (non-finite smoke loss once).
- 03:33Z step-500 eval (Qwen-only policy confirmed: eval/loss 1.387 =
  family/qwen35, 256 samples): F1 0.615 / EM 0.19 (≈ v5b's FINAL at
  step 500); zeroed ablation F1 0.462 (loss gap small, 1.44 vs 1.39);
  train-subset loss 3.12 vs eval 1.39 with similar F1 → per-task
  breakdown: train-subset **fileneedle** loss-token acc 0.35 vs 0.80 on
  eval, tokens/sample 279 vs 139. ROOT CAUSE (data): 166/5539 fileneedle
  rows (3%) are "Quote the line that mentions X" on MINIFIED bundles —
  the gold "line" is the whole file (up to 116K chars; grepset has 5
  such rows, lognav/swerecall none). Token-weighted loss lets one such
  sample dominate a probe/step, and they are pure noise for training.
  FIX at the root: `repo_files.iter_repo_files` rejects minified
  (`looks_minified`: any line >1000 chars or avg >200) and
  generated-by-name (`*.min.*`, `*.bundle.*`) files for EVERY Family-B
  generator; `generate_file_samples(max_answer_chars=400)` and grepset's
  `--max-answer-chars 400` cap quote targets; the rebuild script gained
  `DATASETS=...` selection + a FATAL gold-answer-length tail check.
  Tests: `test_repo_files.py` (new, real pygit2 bare repo),
  `test_file_needle_qa.py`. v6 stopped 03:40Z (rescue save step 515);
  `DATASETS="fileneedle grepset" scripts/rebuild_widenet_data_v2.sh`
  running detached (old parquets backed up in the scratchpad); v6
  resumes from 515 on the clean data (data-only change → resume, not
  cold start; eval set = same held-out repos minus minified files, so
  eval numbers before/after are close but not identical).
- 03:58Z relaunched on clean data (resume 515; `max_steps` now in the
  `phase2_kb_widenet_v6` run block, `live_config_changed ...
  run_namespace=phase2_kb_widenet_v6` confirmed, no inherited-key
  warning). 05:54Z step-750 eval: probe artifact GONE — train-subset
  loss 0.79 (was 3.12), fileneedle loss-token acc 0.83 on both splits.
  Headline eval/loss 1.269 (↓1.387), F1 0.618, EM 0.23 (↑0.19); zeroed
  ablation F1 0.45 vs 0.62 (+0.17 F1 from reps; pooled loss gap still
  small, 1.33 vs 1.27 — loss is dominated by the rep-independent
  tool-call turn). Per-task F1: swerecall 0.76, fileneedle 0.71, lognav
  0.55, grepset 0.52. Train-subset F1 0.73 vs eval 0.62 = real ~0.11
  generalization gap (early; watch it).
- 06:00–07:00Z **the "generative needle EM 0.000" headline gate was never
  measured** — four measurement defects found by reading v5b's per-sample
  report (`eval_reports_widenet_v5b/eval_phase2_kb_stage_A.json`) and the
  templates on CPU:
  1. `parse_bgkit_call` accepted only JSON; Qwen3.5's template renders tool
     calls as XML (`<function=bgkit><parameter=ids>…`) — exactly what the
     decoder is trained on — so every well-formed Qwen call scored
     `malformed_tool_call` (50% of v5b rows) and no answer was generated.
     FIX: `_call_from_xml` (fail-closed: bgkit only, ids(+query) only).
  2. Falcon-H1's `eos_token` is `<|end_of_text|>` but its template closes
     turns with `<|im_end|>`: generation never stopped → `____pex____`
     garbage to 8192 tokens (the other 50%). FIX:
     `chat_template.end_of_turn_token_ids` (template-discovered marker;
     `marker_token_ids` admits added tokens only if they look like control
     tokens) used by `generate_with_single_splice`.
  3. `answer_span`/`bgkit_call_spans` included the assistant scaffold
     (`<|im_start|>assistant\n<think>\n\n</think>\n\n`): trainer F1 had a
     floor from the always-matching `assistant`/`think` tokens; the
     standalone eval compared scaffold-prefixed preds to bare gold. FIX in
     `tokenize_trajectory`: template-agnostic scaffold split
     (`assistant_turn_scaffold` via marker rendering); loss = content +
     end-of-turn emission (glue + marker; Falcon = `\n<|im_end|>`); spans
     = content only; `generate_kb_turn` strips the glue. **Training
     objective change**: ~13% of loss tokens (deterministic scaffold) no
     longer bear loss → train/eval loss levels shift at the restart.
  4. Flat rows' scope said "source file a.py" while the prompt says "start
     with the entrypoint ID named in the description" and the call carries
     `file:o/r:a.py@<blob8>` — unknowable sha → tool-call id accuracy
     structurally 0 (every eval so far), ~20 unlearnable loss tokens per
     sample, and the free-running evaluator rejected every flat call as
     unsurfaced. FIX: `flat_trajectory_row` names the id
     (``"...; entrypoint id: `<doc_id>`"`` — backtick-quoted after the
     chain's scope-id liveness gate caught 16 fileneedle rows whose paths
     contain spaces); `scope_entrypoint_ids` seeds the evaluator's
     available set; all four datasets regenerated.
  Plus: `_eval_free_running_pass` reports per-dataset EM/F1 +
  `invalid/<reason>` rates + `free_running_example` raw generations; v6
  config enables it (64 samples, 512 max tokens, 4 calls) → the generative
  gate is now observed every 250 steps. Tests: template tests run on BOTH
  real tokenizers (Qwen XML + Falcon JSON) and round-trip the decoded call
  span through the parser. Unit suite: 2257 pass; only
  `test_encoder_packed` ×3 fail and they fail identically at clean HEAD
  (pre-existing; stale parity fixture vs the current encoder — follow-up).
  Deployed via `scripts/redeploy_widenet_v6_scope_ids.sh` (stop → regen
  → liveness gates → relaunch, resume from the rescue save). Chain #1
  FATALed on the scope-id gate (16 fileneedle rows, spaces in paths) →
  quoted ids → chain #2 passed 0/N on all four datasets; v6 relaunched
  07:01Z resuming from step 812. READ THE NEXT EVALS WITH THIS IN MIND:
  loss/token-accuracy levels shift at step ~812 (scaffold tokens no
  longer bear loss; call ids now learnable), so compare trends within
  the post-812 segment; EM/F1 are now content-only (F1 loses its
  ~0.2 scaffold floor). Step-1000 eval (~08:40Z) is the first with
  `eval/kb/free_running/<dataset>/answer_exact_match`.
- 08:00Z (user: "why are we doing those hashes at all?") — agreed: the
  `@blob8` / `@digest` / UUID parts of flat doc ids never disambiguate
  (one blob per (repo, path) at HEAD; one grep article per repo; sessions
  are countable per repo) and cost 8–20 random tokens the decoder must
  copy into every tool call. New ids: `file:<repo>:<path>`,
  `grep:<repo>`, `swespan:<repo>#<n>:<a>-<b>`; lognav's
  `log:<src>:<file>:<range>` was already content-derived. Writer's
  uniqueness assertion stays as the guard. Deployed by
  `scripts/after_step_redeploy_v6.sh 1000` → redeploy chain right after
  the step-1000 eval/save (so that eval is the before-reference for the
  generative gate on hashed ids).
- 08:33Z step-1000 eval (content-only metrics, first post-812 basis):
  eval/loss 1.035, EM 0.234, F1 0.469 (F1 −0.15 vs step 750 = the
  removed scaffold floor, EM unchanged); zeroed ablation EM 0.098 / F1
  0.356 → reps load-bearing (2.4× EM). Per-task EM: fileneedle 0.36,
  lognav 0.26, swerecall 0.19, grepset 0.10. Train-subset loss 0.41 vs
  eval 1.04 (gap widening; EM 0.25 vs 0.23 though). Teacher-forced
  tool-call id accuracy 0 → **0.33** after 190 steps of scoped ids.
  FREE-RUNNING (first real reading): calls are well-formed XML (parser
  OK, 1.6% malformed) but 98% `unsurfaced_id` — the model INVENTS ids
  (`log:zulieu:Junkeeper.log:3651-3880` for a ZooKeeper log) while
  copying the `query` verbatim: the id-guessing habit from 800 steps of
  unlearnable ids; expect it to flip as scoped (now hash-free) ids train
  in. Probe defect fixed: the pass took the first 64 eval samples =
  all lognav (index-sorted eval subset) → now strides evenly across the
  eval set (`test_free_running_pass_slices_evenly_across_the_eval_set`);
  deploys with the hash-free relaunch.
- 09:07Z hash-free relaunch done (chain triggered on the step-1000 save
  after three watcher footguns — self-matching pkill, ANSI codes inside
  `docker logs` key=values, `pipefail`+`grep -q` — all recorded in
  memory `feedback_pgrep_count_footgun`); rescue save step 1023; all
  four datasets regenerated with readable ids, scope-id gate 0/N. Next
  eval step 1250 (~11:05Z): first strided (mixed-dataset) free-running
  reading; watch `free_running/invalid/unsurfaced_id` fall and
  `free_running/fileneedle/answer_exact_match` (the actual gate).
- 10:55Z step-1250 eval CRASHED the trainer (exit 1, last save 1023 →
  227 steps lost): the model copied a readable fileneedle id correctly
  for the first time (`file:aiortc/aioquic:tests/test_connection.py`),
  the evaluator recorded the accepted call with `is_head=True` (every
  first call), and `_prepare_sample_for_decode`'s flat-dataset head-drill
  guard raised. lognav calls were still invented ids
  (`log:zulieu:Junkeeper.log:…`) at 1250. FIXES: the evaluator records
  retrieval calls as plain `{ids, query}` leaf drills (head/structural
  tags only for tree-node calls) — `test_free_running_eval_accepts_ids_
  named_in_scope` / `..._tags_tree_node_calls_as_head_drills`; and the
  probe is isolated (`_free_running_metrics_guarded`: traceback logged
  as `free_running_eval_failed`, metric `free_running/failed=1`) so a
  diagnostic can never kill training again. v6 relaunched 10:59Z from
  1023; next eval step 1250 (~12:45Z).
- 12:45Z step-1250 eval: the guard held (training continued) but the
  probe failed again at the same guard — flat browse trees index the
  ARTICLE itself as a node (`is_article=True`), so `identifier in tree`
  classified the call as structural and re-added `is_head`. Fixed:
  only a non-article node is a structural call (test fixture now has an
  article node; verified on the real fileneedle tree). lognav examples
  at 1250: numeric range copied exactly, source name still mangled
  (`zulayers:Junkeeper.log`). Graceful restart ~12:50Z (rescue save);
  next eval step 1500 (~14:40Z) should finally report per-dataset
  free-running EM. Host-memory note: a transient dip to 6 GB available
  at 12:31Z (pre-flush spike; adaptive flush recovered to ~98 GB free
  within a minute) — watch for recurrence; raise
  `flush_when_free_below_gb` if it repeats.
- 13:26Z second dip (7 GB). Adaptive-flush lines show single steps
  reserving ~75 GB (free 98→23 within one step) → the dips are
  intra-step peaks on long samples. Live-set
  `cuda_empty_cache_every_step: 2` (run block; confirmed
  `live_cuda_empty_cache_cadence_update cadence=2`) so the baseline
  stays ~98 GB free between steps. If the <8 GB tripwire still fires,
  the lever is the per-sample length cap (`_max_l0_encode_tokens` /
  gold budget), not flushing. v6 resumed from 1250 at 12:51Z.
- 13:47Z third dip (5 GB) with the cadence flush on → not live tensors:
  per-step `cuda_max_allocated` 46–50 GB vs `cuda_max_reserved` 81–91
  GB. ROOT CAUSE: `.env` had `BGKIT_CUDA_MEM_FRACTION=0.80` (97 GB cap
  on the 121.6 GB device), so the caching allocator legitimately held
  up to 97 GB + 10 GB trainer RSS → ~5 GB left for the host (harness
  OOM risk; `garbage_collection_threshold:0.8` only bites relative to
  the cap). FIX: `.env` → 0.60 (73 GB cap; GC at 58 GB; ≥35 GB host
  headroom) — graceful restart ~13:55Z (rescue save). If the allocator
  now throws a CUDA OOM on a long sample, that is the signal to cap
  sample length rather than raise the fraction. The
  `cuda_empty_cache_every_step: 2` live setting stays (cheap).
- **14:58Z step-1500 eval — FIRST VALID GENERATIVE MEASUREMENT** (64
  strided samples, `free_running/failed=0`): valid call+answer
  swerecall 1.00 / grepset 0.90 / fileneedle 0.40 / lognav 0.27
  (overall 0.55; `unsurfaced_id` 98% → 45%); generative EM swerecall
  0.21 (F1 0.75), grepset 0.10, fileneedle 0.10 (≈0.25 conditional on
  a valid call; n=10), lognav 0.07; overall 0.109 — v5b's "0.000" was
  the broken path. Teacher-forced: EM 0.242 / F1 0.487, fileneedle EM
  0.38, zeroed EM 0.10, tool-call id acc 0.33 → 0.53, train-subset
  loss 0.37 vs eval 1.04 (gap steady). Conclusion: end-to-end
  generation works; the remaining bottleneck is id copying on
  lognav/fileneedle (lognav still mangles the rare source word while
  copying the numeric range exactly). No host-memory tripwire since
  the 0.60 cap. 17:00Z check: the id tokenizes IDENTICALLY in the
  backticked scope and in the JSON call (`z|ookeeper`), so it is not
  re-segmentation — lognav's eval split holds out whole log SOURCES,
  so `zookeeper` is a never-seen word; the model copies the per-sample
  digits (always unguessable → learned to copy) but emits a
  memorized-shaped word instead. I.e. the lognav gate is a test of
  copying novel rare tokens; expect it to lag the path-based datasets.
- 17:01Z step-1750 eval: plateau vs 1500 — EM 0.242 / F1 0.487 (same),
  id acc 0.54, free-running valid 0.53 / gen EM 0.094, zeroed EM 0.12,
  train-subset loss 0.36 vs eval 1.04. Per-dataset TF EMs identical to
  1500 to all digits → checked finer keys: per-dataset F1 and
  answer-token acc do move (grepset F1 .319→.333, swerecall .752→.724),
  so it is a real plateau of a coarse metric, not a frozen eval. LR is
  on the cosine tail (2630); remaining evals 2000/2250/2500 + final.
- 19:05Z step-2000: EM 0.246 / F1 0.496, gen EM 0.125 (grepset 0.20,
  swerecall 0.21, fileneedle 0.10, lognav 0.07), valid calls 0.53, id
  acc 0.535, zeroed EM 0.12, eval/loss 1.035 vs train-subset 0.347.
  Slow drift up; no memory incidents.
- 21:05Z step-2250 ≈ step-2000 on every key (EM 0.246, F1 0.495, gen EM
  0.125, id acc 0.535, valid 0.52) — converged on the cosine tail; the
  final numbers will come from the post-run 256-sample eval.
- 23:13Z step-2500: same plateau (EM 0.246, F1 0.496, gen EM 0.109, id
  acc 0.543). Run completes ~00:12Z → post-run chain
  (`post_run_widenet_v6.sh`) delivers the headline report.
- **2026-08-24 00:45Z v6 COMPLETE (2630 steps, exit 0) + final 256-sample
  generative report** (`eval_reports_widenet_v6/analysis_vs_v5b.txt`;
  cmp columns = v5b's broken-path numbers). Per-qtype genEM/genF1:
  fileneedle absent 1.00/1.00 · signature 0.125/0.414 · needle 0/0.150 ·
  assignment 0/0; grepset absent 1.00 · quote_line 0/0.305 ·
  which_file 0/0; lognav count 0.50 · error_absent 1.00 ·
  line_needle 0/0.093 (n=77); swerecall absent 0.889 · symbol 0.429 ·
  path 0/0. **Takeaway:** with valid tool calls and load-bearing reps
  (zeroed-EM gap 2×), the compressed context supports
  RECOGNITION/AGGREGATION generatively (absence 0.9–1.0, counts 0.5,
  symbols 0.43) but VERBATIM LONG-STRING EXTRACTION (needle lines,
  paths, quote_line) stays EM 0 with modest F1 at L0 0.10 × L1 0.15 —
  the capability to design for next (higher retention for wide-net tool
  reads, a copy mechanism, or span-anchored reps). Open follow-ups:
  RULER grid (blocked on vLLM image), Falcon per-sample-loss
  investigation.
- 01:00Z BABILong truncate-at-4K reference arm done (stock
  Qwen3.5-0.8B, 100 samples/row,
  `baselines/babilong_qwen35_0p8b_ctx4k`): qa1 0.53/0.50/0.41/0.35 at
  4k/8k/16k/32k (vs in-budget ~0.43–0.55); qa2 0.23/0.33/0.34/0.25;
  qa3 0.31/0.33/0.33/0.34 (qa3 barely above its 0k level of 0.29
  everywhere — near floor regardless). This is the "no long context"
  reference a bgkit-compressed arm has to beat, principally on qa1
  16k–32k (0.41/0.35) where truncation demonstrably loses the needle.
- **2026-08-24 next-steps session (user: "keep going autonomously")**:
  the RULER blocker is SOLVED at the root — `vllm-node` was never a
  registry image: it is built locally by eugr's `spark-vllm-docker`
  (checked out at `/home/werg/spark-vllm-docker`, build script default
  tag `vllm-node`), and the local copy was lost in a docker image
  prune. Rebuild running (`build-and-copy.sh`, prebuilt aarch64
  vLLM/FlashInfer wheels from the repo's GitHub releases; log
  `$FAST/vllm_node_rebuild.log`) → unblocks the RULER grid AND the P0
  `--enable-prompt-embeds` serving spike.
- **oracle_span ablation BUILT + RUNNING** (the decision experiment for
  the verbatim-EM-0 question): `+eval.ablation=oracle_span` on
  `eval_phase2_kb.py` forces the gold answer-span tokens to WIN the
  exact_topk selection at BOTH levels at the SAME rep budget (a
  +1e4 boost on a selection-only copy of the operator score:
  `must_keep_mask` through `LevelCompressor.forward` →
  `run_l1_and_project` → live-L0 build in `_live_l0_encode` (offsets
  against the final cap-truncated buffer) + `span_flat` in
  `_run_l1_batch`; survive_probs/aux losses see UNBOOSTED scores;
  fail-closed under threshold mode, forced_survivor_mask, and cached
  L0). Also fixed while wiring: the eval ablation sweep restored the
  mode to None instead of the ENTERING mode — it would have silently
  stripped oracle_span from every pass after `trainer.evaluate()` in
  the standalone script (same measurement-defect class as 2026-08-23).
  Tests: 4 new in `test_level_compressor.py` (forced-in at same budget,
  unboosted exported logits, three fail-closed raises) + 2 new in
  `test_kr_kb_eval_mode.py`; 27 pass. Running on the v6 final ckpt
  (256 samples, generative, report suffix `_oracle_span`). Read:
  oracle needle-EM ≫ headline ⇒ SELECTION is the bottleneck (levers:
  retention ladder — NOTE the v3a "ratio is not the lever" verdict is
  VOID, it was a zero-rep run — and/or a hard span floor / stronger
  span supervision); oracle EM ≈ headline ⇒ READ-OUT capacity is the
  bottleneck (span-anchored reps or a copy mechanism design pass).
- **ORACLE VERDICT (2026-08-24): READ-OUT, not selection.** The 256-sample
  oracle run returned per-qtype numbers identical to the headline to 3
  decimals — which first smelled like a silent no-op (the 2026-08-23
  lesson), so it was verified positively: (1) liveness instrumentation
  (`oracle_span_liveness` log in `_live_l0_encode`) shows the mask built
  and non-empty (e.g. fileneedle n_forced=10/6214, swerecall 17/24190)
  on a rerun (32-sample check; first check attempt died on hydra struct
  — use `++training.*` overrides for ad-hoc keys); (2) ROW-LEVEL diff of
  the two 256-sample reports: 58/256 teacher-forced and 24/256
  free-running predictions DID change under oracle — the reps really
  were different — yet answer-F1 moved on only 15 rows, ±0.03,
  symmetric, and needle EM stayed 0 on every qtype. CONCLUSION: with
  the gold span guaranteed to survive BOTH selection levels at the same
  budget, the current decoder still cannot emit the verbatim string —
  at L0 0.10 × L1 0.15 the projected reps do not retain token-identity
  for arbitrary strings this decoder can exploit. CAVEATS: (a)
  eval-time forcing is distribution-shifted vs the organically-trained
  decoder (a training-time hard span floor could in principle teach
  read-out — weaker prior after teacher-forced likelihood also failed
  to improve); (b) retention is still untested with live reps — the
  ladder point stays worthwhile and doubles as the Pareto curve.
  DESIGN OPTIONS now on the table for verbatim wide-net capability:
  (1) v7 retention point ~6× with live reps (short continuation run
  from v6-final, then generative eval); (2) hybrid splice / verbatim
  passthrough: selection emits a small budget of LITERAL token IDs
  (the head's top span picks) alongside the dense reps — the decoder
  can copy text it can actually see; (3) product-level answer:
  compressed read for recognition/aggregation + a second narrow RAW
  read for verbatim quotes (two-step agent pattern, no model change).
- **v7-ratio6x STAGED + diag-cleared** (config
  `phase2_kb_widenet_v7_ratio6x.yaml` + compose service): 1,000 steps at
  L0 0.35 × L1 0.5 continuing from v6-final. `diag_flat_splice_counts`
  on the real prep: l0_keep 0.32–0.38 (target 0.35, exact_topk live),
  rep/N ≈ 0.15–0.20 (theory 0.175), decode_len 970–5,014 (~2× v6).
  Launches when the RULER chain frees the GPU.
- **RULER bring-up defects (all fixed 2026-08-24)**: (1) the 08-22 "grid
  generated" claim was wrong — only niah_single_1 existed (essay-haystack
  tasks need `PaulGrahamEssays.json`, qa needs the SQuAD download;
  RULER's prepare.py swallows the child traceback and prints success —
  verify by files on disk, never by exit status); (2) predictor
  `stop=["\n\n"]` truncated 100% of predictions to "" at position 0
  (greedy Qwen3.5-0.8B opens every RULER base-prompt answer with
  "\n\n"; live A/B: stop → `''`, no-stop → `'\n\n5437923'` = correct
  needle). Fixed: no stop strings — RULER metrics are substring
  matches, trailing text is harmless; (3) RULER's evaluate.py imports
  the NeMo toolkit for jsonl IO — replaced with
  `scripts/score_ruler.py` (RULER's exact metric fns, plain jsonl).
  Full chain (downloads → regen 4 tasks × 4 lengths → predict → score)
  running detached; wrapper `run_ruler_baseline.sh` updated. Plus defect
  (4): greedy qa answers entered an UNBOUNDED <think> block (finish=
  length at 256 tokens, and think text can contain spurious substring
  matches) — protocol now appends the empty-think scaffold (Qwen's
  official non-thinking switch) to every prompt; A/B: plain 256 = still
  thinking, scaffold 64 = "France." clean.
- **RULER stock baseline COMPLETE (2026-08-24 12:05Z)** — Qwen3.5-0.8B,
  greedy, 100/cell, no-stop + empty-think scaffold, summaries under
  `$NVME/benchmarks/ruler/pred_qwen35_0p8b/L*/summary.csv`:
  | task | 4K | 8K | 16K | 32K |
  |---|---|---|---|---|
  | niah_single_1 | 100 | 100 | 100 | 100 |
  | niah_single_2 | 100 | 100 | 100 | 100 |
  | niah_multikey_1 | 100 | 100 | 100 | 100 |
  | niah_multiquery | 52.5 | 52 | 52 | 52.5 |
  | qa_1 | 71 | 67 | 69 | 70 |
  Reading: the stock 0.8B is length-FLAT on this grid to 32K — RULER
  niah is NOT where long-context degradation shows for this model
  (multiquery's ~52 is a partial-credit ceiling on 4-needle recall,
  flat in length; qa_1 ~69 flat). The self-comparison axis for a bgkit
  arm here is ratio-vs-fidelity at MATCHED length, while BABILong qa1
  16k–32k (truncate 0.41/0.35) stays the degradation showcase.
- **v7-ratio6x LAUNCHED** right after the grid (12:06Z),
  `train-phase2-kb-widenet-v7-ratio6x`, 1,000 steps from v6-final at
  L0 0.35 × L1 0.5. Watching bring-up: s/step at ~2× decode lengths,
  host-available, rep-norm band, l1_buckets>0, inherited-phase-block
  warning on the first live-config poll.
- **POSITIONING DECISION (user, 2026-08-24): semantic compression is
  the product; the verbatim boundary is acceptable.** The needle
  analysis showed reps behave as SEMANTIC memory (the model recovers
  `BindingLayout` from the reps but hallucinates the arbitrary shell —
  path/line/language; single symbols EM 0.43, full lines EM 0; copying
  from visible TOKENS is perfect) — and that is aligned with what
  bgkit is for. Consequences: (a) headline metrics = recognition /
  aggregation / absence / gist-F1 (per the 2026-08-21 calibration);
  verbatim needle-EM stays reported as the honest-limits slice, not a
  gate; (b) verbatim needs in the agent loop are served by the
  TWO-STEP pattern (compressed wide read → narrow raw re-read for
  exact quotes) — a harness behavior, no model change; the hybrid
  splice / copy mechanism is DEPRIORITIZED (revisit only if a
  benchmark demands in-context verbatim); (c) benchmark emphasis:
  BABILong (recognition-class answers) over RULER exact-needle slices;
  RULER stays as the budget-matched fidelity axis; (d) v7 is
  re-scoped from "verbatim rescue" to Pareto/boundary mapping — at
  what ratio does surface form start surviving; its eval still
  informs the auto-compress ratio heuristic; (e) next big build after
  v7 = Family A compaction trainer (semantic by nature — the best fit
  for what the reps actually are).
- **Family-A COMPACTION TRAINER BUILT (2026-08-24, while v7 trains)** —
  the three "missing pieces" collapsed to one (the data layer already
  resolves source_refs internally): `src/bgkit/training/
  blob_sft_trainer.py` (`BlobSFTTrainer(BaseTrainer)`, phase
  `blob_sft`): loads encoder + Qwen decoder from a Phase-2 wide-net
  ckpt (`training.init_checkpoint`, fail-closed on material key
  mismatch), ONE packed query-free encoder forward per sample over all
  blob contents (exact_topk both levels, `l0_ratio` 0.35 ×
  `l1_ratio` 0.5), `build_blob_segments` interleaves TokenSegment runs
  with EmbeddingSegment survivor runs at the sentinel spans (sentinel
  tokens dropped, loss mask travels with its run), CE via
  `forward_interleaved_with_loss`; per-layer encoder GC + reentrant
  decoder GC; device prefetch bypassed (nested-list samples). Eval:
  teacher-forced loss/token-acc per qtype, RECALL-PROBE exact match,
  and a zeroed-rep ablation gap every eval (the mandatory
  reps-load-bearing gate). Configs `training/blob_sft.yaml` +
  `experiment/blob_sft_v1.yaml` (SWE-Zero shards via the
  `/workspace/capability_packaging` mount; 60 shards × 200 rows × 4
  draws ≈ 48K samples; eval = held-out SHARD — repo-level split is a
  flagged follow-up), compose service `train-blob-sft-v1`, train.py
  dispatch. Tests: `test_blob_sft_pieces.py` (6 new) + blob suite = 18
  pass; ruff clean. NOT yet GPU-smoked (v7 owns the GPU) — smoke is
  the first queue item after the v7 post-run eval, then the
  `blob_sft_v1` launch.
- v7 pace: ~49 s/step (step 146 @ 13:41Z) → first eval ~14:40Z, run
  end ~03:00Z. Post-run chain ARMED (`scripts/post_run_widenet_v7.sh`,
  detached): training_complete → 256-sample generative eval of the
  final ckpt → `analysis_vs_v6.txt` (per-qtype vs the 67× report) —
  the Pareto/boundary read lands automatically. Queue after that:
  blob-SFT GPU smoke → launch; RULER 64K/128K scout (stock cliff
  hunt; model native context 262144).
- **v7-ratio6x COMPLETE (2026-08-25 00:06Z) — RATIO CONFIRMED NOT THE
  LEVER, this time with live reps.** 1,000 steps at 6× from v6-final;
  final TF EM 0.242 / loss 1.03 ≈ v6-final; zeroed-rep EM 0.0 for the
  ENTIRE run (total rep dependence; v6's zeroed arm could still guess
  0.12). 256-sample generative per-qtype vs v6
  (`eval_reports_widenet_v7_ratio6x/analysis_vs_v6.txt`): verbatim EM
  **0.000 on every needle qtype at 6× too** (needle F1 nudges:
  fileneedle 0.150→0.234, lognav 0.093→0.145; quote_line 0.305→0.222);
  recognition/aggregation IDENTICAL (absence 1.0, counts 0.5, symbols
  0.429, signature 0.125). Combined with the oracle-span result, the
  evidence chain is closed: selection is not the bottleneck (oracle),
  ratio is not the bottleneck (v7 live-reps), and reps demonstrably
  change generations without moving them toward the needle (row
  diffs) ⇒ verbatim long-string extraction from dense semantic reps
  is ARCHITECTURE-bound for this stack — fully consistent with the
  semantic-compression positioning. **Product insight (the flip
  side): 6× and 67× have near-identical capability profiles on the
  semantic slice — compression is effectively FREE on
  recognition/aggregation/gist between 6× and 67×, so wide-net tool
  reads can default to the HIGH ratio.** The Pareto curve on this
  family is flat where it matters and zero where it doesn't.
- Ops note: the post-run chain was never stalled — my two "slowdown"
  readings during the run were both wrong timeline anchors on my own
  probes (verify with timestamps, not memory of when a probe ran).
- 00:15Z queue executing: blob-SFT GPU smoke (run_name blob_sft_smoke,
  8 steps + tiny eval) → RULER 64K/128K scout → blob_sft_v1 launch.
- **Blob-SFT smoke PASSED (00:14Z)**: 8 steps end-to-end, real splices
  (n_reps 2298/sample, decode ~7.8K), evals + checkpoints + clean
  completion; step-8 loss 4.9 / zeroed_gap ≈ 0 = the expected cold
  signature of a new task format (the gap RISING during the real run
  is the gate). Smoke checkpoints removed.
- **RULER 64K/128K scout DONE (01:02Z): NO CLIFF.** niah_single_1 =
  100.0 at BOTH 65536 and 131072 (half the model's native 262K);
  qa_1 = 57/55 (gentle drift from ~69 at ≤32K). Qwen3.5-0.8B is
  simply strong at needle retrieval — RULER cannot furnish a
  "long-context rescued" story for this model at any tested length.
  RULER's role is locked as the budget-matched fidelity axis (can a
  ~2K-rep bgkit budget match the stock 55–100 at 128K?); BABILong
  remains the degradation showcase.
- **blob_sft_v1 LAUNCHED (01:04Z)** — first real Family-A compaction
  run (`train-blob-sft-v1`): 2,000 steps from v6-final, ~51K samples
  (64 SWE-Zero shards × 200 rows × 4 draws, 10% repo-holdout eval),
  6× blob encode, evals every 250 with recall-probe EM + zeroed-rep
  gap. Watch: the zeroed_gap trend (reps must become load-bearing)
  and probe EM (load-bearing detail through the bottleneck).
- **blob_sft_v1 bring-up (2026-08-25 02:13–03:26Z)**: loss 2.97 →
  1.27 (eval, step 500), token-acc 0.75 — the format trains fast.
  DEFECT found at the step-250 eval (measurement-gap class): the eval
  was 128/128 continuation — ZERO recall probes.
  `sample_trajectory` APPENDS the probe variant after the base
  sample; `BlobSFTDataset.__getitem__` took `[0]` unconditionally, so
  every probe was silently dropped and the designed gate did not
  exist. FIXED (dataset prefers the probe when present, ~25% of
  draws; regression test `test_recall_probes_appear_across_draws`);
  deployed at the step-500 save via graceful stop → relaunch →
  auto-resume from the step-501 rescue save. Also expected-but-noted:
  zeroed_gap ≈ 0 through step 500 — compaction CONTINUATIONS are
  largely predictable from the live window (agent next-steps), which
  is precisely why probes carry the Family-A signal. First probed
  eval = step 750.
- 03:30Z resume #1 CRASHED at the BaseTrainer resume-replay path:
  `_wrap_dataloader_iter` is called directly there (not via
  `_create_dataloader_iter`), so the prefetch bypass didn't apply and
  `_DevicePrefetcher` choked on list batches
  (`AttributeError: 'list' object has no attribute 'items'`). FIX:
  BlobSFTTrainer overrides `_wrap_dataloader_iter` too (the correct
  single choke point); regression test
  `test_wrap_dataloader_iter_never_prefetches`. Relaunched 03:4xZ,
  auto-resume from the step-501 rescue save; the replay of ~2K
  microbatches through the dataset takes some minutes before stepping
  resumes.

**Drive-blocked track ([B]):**

- [x] Base checkpoint RESCUED (2026-08-21, drive replugged + remounted):
      verified intact on HDD (8.9G: encoder.pt, decoder_qwen.pt,
      decoder_falcon.pt, metadata.json step=51945, optimizer state) and
      sha256-verified copy at `bgkit-data-nvme/capability_packaging/
      base_checkpoint/phase1_summarization_round_robin_step51945_20260624_060459`.
      Trainer configs may keep using the CHECKPOINT_DIR path; the NVMe copy
      is the disaster-recovery duplicate.
- [x] Registry backfilled 2026-08-21 (`bgkit-ckpt backfill`; step51945
      registered, status "interrupted" = mid-run save, expected). Still
      open: pin `phase1_checkpoint` when the P1 experiment config is
      written. Note: registry also contains
      `phase2_kb_hybrid_enc9164_dec51945` (hybrid git-repro-encoder +
      51945-decoder) — not our pick, but a candidate for a later encoder
      ablation.
- [~] Family B generators: **fileneedle IMPLEMENTED + BUILDING**
      (`src/bgkit/data/file_needle_qa.py` + tests +
      `scripts/build_fileneedle_phase2.py`, via the new shared
      `flat_phase2_writer.py`): real source files from the bare-repo corpus
      (pygit2 HEAD walk), qtypes signature/assignment/needle_token/absent,
      split = held-out repos; build running at idle IO priority alongside
      training. Still open: grep-set generator, MS MARCO reuse config,
      per-qtype eval breakdown (aggregate EM mixes formulaic negatives with
      needles — run `eval_phase2_kb.py` per-qtype post-run).
- [ ] (UNBLOCKED 2026-08-21) Family C memory mmaps — msc, share,
      chronicles, memory confirmed present in `mmap/phase2/`.

**Drive status 2026-08-20**: device absent from the USB bus entirely (all 12
xHCI root hubs enumerate, no storage device; fstab `nofail` skipped it at
the 2026-08-13 boot). Software recovery attempt = xhci-hcd platform driver
rebind (needs sudo; commands in session log). Remote reboot NOT advised:
GB10 GSP warm-reboot wedge risk requires physical cold power cycle to
recover — worse than a missing drive while the machine is physically
unattended.
