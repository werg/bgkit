# Training Workflows

This is the executable training map. Exact hyperparameters live in Hydra
configs; research alternatives live in `plans/`.

> **Direction change (2026-08-20):** the Phase-2 browse/navigation training
> direction and the Phase-2 → Phase-3 promotion path below are superseded by
> `plans/capability_packaging_2026_08_20.md` (compaction, wide-net tool-result
> compression, compressed memory, ambient repo background). This file remains
> the accurate map of implemented machinery.

## Phase 1: compression and reconstruction

Phase 1 develops the encoder/decoder representation contract through several
specialized trainers: joint block pretraining, decoder initialization, pruning
distillation, projection repair, compressed reconstruction, and summarization
round-robin handoff. The repository retains multiple historical experiment
presets, so a phase number alone is not enough to identify a result. Pin the
source checkpoint or record the intended `run_name`.

The active Phase-2 Qwen Stage-A preset currently starts from the pinned
summarization round-robin checkpoint in
`configs/training/phase2_kb_stage_a.yaml`.

## Phase 2: browse and dense retrieval trajectories

All active knowledge datasets use `KRKBTrainer` and the same trajectory schema.
The decoder observes text browse results, emits `browse`/`bgkit` calls, and
receives live L1 dense splices for `bgkit` turns.

### Stage A

`configs/experiment/phase2_kb_stage_a.yaml` runs L0 live and trains the enabled
encoder levels and decoder families. The current preset disables encoder LoRA
and directly trains L0/L1. Survivorship auxiliary losses are disabled, so the
inherited head policy is frozen while threshold control remains active.

### Prompt-fit

`configs/training/phase2_kb_prompt_fit.yaml` optionally learns per-dataset L0
prompts with the L0 backbone frozen. It uses the same Phase-1 base as Stage A
and overlays `stage_a_checkpoint: auto`. The auto resolver accepts plain Stage A
for prompt-fit.

### Cache build

Build L0 survivors with the exact Phase-1 and Phase-2 checkpoint pair that Stage
B will load:

```bash
uv run python scripts/precompute_l0_subset.py \
  --articles "$DATA_DIR/trajectory_sets/stage_b.jsonl" \
  --mmap-dir "$DATA_DIR/mmap/phase2" \
  --phase1-checkpoint "$PHASE1_CHECKPOINT" \
  --stage-a-checkpoint "$PROMPT_FIT_OR_STAGE_A_CHECKPOINT" \
  --output-dir "$DATA_DIR/l0_cache_kb" \
  --retention-json configs/phase2_kb/l0_retention.json \
  --lora-rank 32
```

The cache builder writes a content-fingerprinted manifest per dataset. Do not
copy or rebuild only the survivor arrays without their manifests.

### Stage B

`configs/experiment/phase2_kb_stage_b.yaml` resolves the Phase-1 base and then
prefers a completed prompt-fit checkpoint, falling back to plain Stage A. It
loads the complete Phase-2 state and verifies every cache manifest before data
loading. L0 remains cached/frozen; L1 and the decoder continue training.

Stage-B batching is budgeted by cached survivor rows
(`max_microbatch_l0_survivors`) rather than raw document tokens. Oversized
samples become singleton microbatches.

### Evaluation contract

`scripts/eval_phase2_kb.py` runs an autonomous loop that executes generated tool
calls, rejects IDs that were not exposed by an executed node, and reports route
completion, evidence recall, literal full-state exact match, and answer F1.
Teacher-forced token/tool metrics remain in the same report as diagnostics. The
git-reproduction preset also runs a small autonomous slice during periodic
checkpoint evaluation.

A production quality gate still needs acceptance thresholds and matched-decoder
BM25/dense/reranker baselines. Use repository-disjoint held-out data and retain
the present/zero/noise representation ablations when comparing checkpoints.

## Phase 3: SWE trajectory imitation prototype

`DistillationTrainer` currently performs CE imitation of external teacher
trajectory tokens. It does not implement the previously documented frozen
same-model teacher, hint mining, top-K KL, or hidden-state matching.

The working context source is compressed filesystem state:

```text
$DATA_DIR/swe_embeddings/
  blob_cache/<sha-prefix>/<blob-sha>.npy
  filesystem/
    manifest.parquet
    contexts/<repo>/<full-base-commit>/survivors.npy
```

Build it with:

```bash
uv run python scripts/encode_swe_repos.py \
  --checkpoint "$PHASE2_STAGE_B_CHECKPOINT" \
  --trajectories "$DATA_DIR/trajectories/openhands_filtered.jsonl" \
  --repos-dir "$DATA_DIR/swe_repos" \
  --output-dir "$DATA_DIR/swe_embeddings" \
  --batch-size 4
```

The encoder performs one packed forward per blob batch and persists the
content-addressed blob cache, so restarts reuse completed work. Phase 3 resolves
`phase2_checkpoint: auto` only among completed Stage-B checkpoints with an eval
loss. `phase2_run_name` can narrow the source.

`context_sources: [filesystem]` is the supported default. Git-history manifests
must be keyed by `(repo, base_commit)` to avoid future commits. Prior-session
manifests must provide numeric or ISO timestamps; sessions without comparable
timestamps are omitted rather than ordered lexicographically by SHA. Neither
optional producer is currently shipped.

Evaluation runs context and genuine no-context forwards on every identical
batch and reports `loss_with_context`, `loss_no_context`, `context_delta`, and
sample-level coverage. This is a paired likelihood ablation, not SWE-bench
patch execution accuracy.

## Promotion gates

Before expanding Phase 3 or deployment work:

1. Keep a small golden Stage-A → cache → Stage-B resume pipeline in tests.
2. Demonstrate a positive paired dense-context contribution on unseen data.
3. Add free-running Phase-2 tools and a matched RAG baseline.
4. Measure quality, latency, survivor count, and memory as a Pareto curve.
5. Only then add more decoder families, custom kernels, or serving adapters.
