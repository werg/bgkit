# Operations Runbook

## Environment

Copy/configure `.env` so `DATA_DIR`, `CHECKPOINT_DIR`, and model/cache locations
resolve consistently inside and outside Docker.

```bash
uv sync --extra dev --extra data
make test-unit
make lint
make typecheck
```

`make test-unit` explicitly hides CUDA and excludes `gpu` markers. Run GPU tests
through `make test-gpu`; do not use an unfiltered host `pytest tests/unit` as the
CPU gate. The unit gate is green. Repository-wide lint and typing still report
known legacy debt; use them as inventories and require touched files to avoid
adding new findings until those baselines are ratcheted into CI.

## Compose and inspect a training preset

```bash
uv run python scripts/train.py --cfg job +experiment=phase2_kb_stage_b
uv run python scripts/train.py +experiment=phase2_kb_stage_b
```

Use an experiment preset rather than supplying a training group alone; the
experiment sets a stable `run_name`, which is part of checkpoint identity.

## Phase-2 handoff checklist

1. Confirm the Stage-A or prompt-fit checkpoint has `phase: phase2_kb`, the
   expected `stage`, `run_name`, and eval metric in `metadata.json`/registry.
2. Resolve or pin the Phase-1 base and Phase-2 handoff.
3. Run `precompute_l0_subset.py` with that exact pair.
4. Verify every active dataset has `cache_manifest.json` with non-null
   `phase1_sha` and `stage_a_sha`.
5. Compose Stage B and confirm `stage_a_checkpoint`, direct/LoRA topology, dataset
   list, and retention values match cache generation.
6. Start Stage B. Manifest mismatch or a missing manifest is intentionally fatal.

The auto resolver prefers prompt-fit for Stage B and plain Stage A for the
prompt-fit stage. Use `stage_a_source_stages` only for an intentional custom
lineage.

## Phase-3 filesystem cache

`encode_swe_repos.py --output-dir "$DATA_DIR/swe_embeddings"` writes the
canonical `filesystem/manifest.parquet` and persistent `blob_cache/`. It is safe
to restart; already encoded blob SHAs are reused.

Only list a source under `training.context_sources` when its manifest exists.
Missing enabled sources fail setup. The old root-level `manifest.parquet` is
read with a migration warning but is no longer written.

## Checkpoints

```bash
bgkit-ckpt backfill
bgkit-ckpt list --phase phase2_kb
bgkit-ckpt show CHECKPOINT_NAME
```

Do not rename a checkpoint directory without updating the registry. Named runs
will not resume legacy checkpoints whose metadata lacks `run_name`. Explicit
`resume_checkpoint` remains authoritative and fails loudly if unreadable.

Fast NVMe and HDD archive pruning are run-scoped. If multiple processes use the
same `run_name`, they still share one retention set; use unique run names for
independent experiments.

## Evaluation interpretation

- Phase-2 `eval_phase2_kb.py`: free-running tool execution is the capability
  result; teacher-forced token/tool scores are diagnostics.
- Phase-3 trainer eval: paired token loss with and without context.
- Neither result is an autonomous SWE-bench score.

Record the checkpoint name, run name, config snapshot, dataset split, cache
fingerprints, ablation arm, and whether each reported metric was free-running or
teacher-forced.
