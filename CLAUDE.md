# BgKIT — Claude Code Instructions

## Hardware & Environment

This project runs on an **NVIDIA DGX Spark** (Blackwell GB10, ARM64, sm_121, 128 GB unified memory).

## Storage Configuration

**Required:** A `.env` file at the project root (gitignored) is the single source of truth for storage paths. Without it, all Python scripts, Make targets, and Docker services will fail fast with setup instructions.

```bash
cp .env.example .env   # then edit DATA_DIR / CHECKPOINT_DIR
```

| Variable | Description |
|---|---|
| `DATA_DIR` | Root data directory (repos, descriptions, structural, models, crawl DB) |
| `CHECKPOINT_DIR` | Training checkpoints and registry |

The `.env` file is read by: Python (`bgkit.env` via python-dotenv), Makefile (`-include .env`), shell scripts (source `.env`), and Docker Compose (native `.env` support). There are no fallback defaults anywhere — `.env` is it.

## GPU Work Must Use the Docker Container

Any work involving PyTorch, CUDA, or GPU computation **must** be run inside the NGC-based Docker container — not directly in the host venv. The container (`docker/Dockerfile`) is based on `nvcr.io/nvidia/pytorch:26.03-py3` and provides the correct ARM64 + CUDA 13.2 + sm_121 PyTorch build.

The Docker image contains only third-party dependencies. bgkit source is bind-mounted at runtime via docker-compose. The entrypoint verifies mounts exist before starting. Do not use bare `docker run` without mounting src/, configs/, and scripts/.

The container runs as the host user (uid 1000 by default) so files written to bind mounts (checkpoints, HF cache) are owned by `werg`, not root. Override with `DOCKER_UID`/`DOCKER_GID` env vars if needed.

**Build and run the training container:**
```bash
# Build deps image (only needed when pyproject.toml changes)
make docker-build-deps

# Run training (interactive — tails logs, Ctrl-C to detach)
make train-ice

# Run training (non-blocking — starts container and returns)
scripts/run-train.sh --no-follow train-ice

# Or via compose directly
docker compose -f docker/docker-compose.yaml up -d train-ice
```

**IMPORTANT for AI agents:** `make train-*` and `scripts/run-train.sh` without `--no-follow` will tail logs forever and block. Always use `scripts/run-train.sh --no-follow <service>` when launching training from a non-interactive context. Check status afterwards with `docker compose -f docker/docker-compose.yaml ps` and `docker compose -f docker/docker-compose.yaml logs --tail 30 <service>`.

**Why:** The host venv's PyTorch (pip cu130 wheels) lacks the full CUDA toolkit needed for JIT kernel compilation, TransformerEngine, and other GPU-accelerated libraries. The NGC container has all of this pre-validated.

## Host venv is for non-GPU work only

The local `.venv/` (managed by `uv`) is used for:
- Unit tests: `.venv/bin/pytest tests/unit -v`
- Linting/formatting: `uv run ruff check src/ tests/ scripts/`
- Type checking: `uv run mypy src/bgkit/`
- Data crawler: `.venv/bin/bgkit-data stats|discover|download`
- Any CPU-only data processing

Install locally with: `make install` or `uv sync --extra dev --extra data`
Full local dev (with torch + GPU packages): `make install-gpu` or `uv sync --group torch --extra gpu --extra dev --extra eval`

## Key Commands

| Task | Command |
|---|---|
| Install (CPU dev) | `make install` |
| Install (full + GPU) | `make install-gpu` |
| Run unit tests | `make test` or `.venv/bin/pytest tests/unit -v` |
| Lint | `make lint` |
| Format | `make format` |
| Build deps image | `make docker-build-deps` |
| Build data container | `make docker-build-data` |
| Train (in container) | `make train-ice` or `docker compose -f docker/docker-compose.yaml up train-ice` |
| Train (no log tail) | `scripts/run-train.sh --no-follow <service>` |
| Generate descriptions | `make generate-descriptions` |
| Generate QA pairs | `make generate-qa-pairs` |
| Convert QA to mmap | `make convert-qa-pairs` |
| Generate variant banks | `make generate-variants` (all 4 templates) |
| Full data pipeline | `make prepare-data-all` or `scripts/prepare-data.sh --with-descriptions --with-qa` |

## Dependency Management

- **Package manager:** `uv` (host), `pip` (Docker)
- **Core deps** (CPU-safe): transformers, safetensors, datasets, tokenizers, pygit2, hydra, etc.
- **GPU extra** (`[gpu]`): accelerate, peft, bitsandbytes, einops — requires torch
- **PyTorch in Docker:** NGC container provides torch. `pip install ".[gpu]"` sees it and skips reinstalling.
- **PyTorch locally:** Installed via the `torch` dependency group (`uv sync --group torch`).

## Training Runs & Checkpoints

Checkpoints are saved to `checkpoint_dir` (default: `./checkpoints`) with names like `{phase}_step{N}_{timestamp}`. A persistent `registry.json` in the checkpoint directory tracks all checkpoints ever saved, surviving pruning.

**Checkpoint registry CLI** (`bgkit-ckpt`):
- `bgkit-ckpt list [--phase X] [--status X] [--tag X]` — tabular summary
- `bgkit-ckpt show <name>` — full JSON detail
- `bgkit-ckpt best --phase X --metric X [--higher-is-better]` — find best checkpoint
- `bgkit-ckpt annotate <name> [--notes "..."] [--tag X]` — add notes/tags
- `bgkit-ckpt backfill` — populate registry from existing on-disk checkpoints

**Auto-resolution**: Set checkpoint paths to `auto` in experiment configs to automatically resolve the best checkpoint from the registry. Backfill runs first to catch any on-disk checkpoints not yet registered.

**Live hyperparameter control**: Write to `${CHECKPOINT_DIR}/control.json` while training runs to adjust LR, batch budget, compression ratio, eval cadence, and more — no restart required. See the full key table in the "Attention backend" section below.

| Dependency | Config Key | Source Phase | Target Phase | Metric |
|---|---|---|---|---|
| Joint Block → Step 1 | `bgkit_checkpoint` | `joint_block_pretrain` | `phase1_step1` | `eval/mse_repro` |
| Step 1 → Step 2 | `step1_checkpoint` | `phase1_step1` | `phase1_step2` | `eval/loss` |
| Step 2 → Step 2.5 | `bgkit_checkpoint` | `phase1_step2` | `phase1_step2p5` | `eval/loss` |
| Step 2.5 (or 2 fallback) → Step 3 | `bgkit_checkpoint` | `phase1_step2p5` → `phase1_step2` | `phase1_step3` | `eval/loss` |
| Step 3 (or 2.5/2 fallback) → Step 4 | `bgkit_checkpoint` | `phase1_step3` → `phase1_step2p5` → `phase1_step2` | `phase1_step4` | `eval/loss` |
| Step 4 → Step 5 | `step1_checkpoint` | `phase1_step4` | `phase1_step5` | `eval/loss` |
| Step 5 → Step 6 | `step1_checkpoint` | `phase1_step5` | `phase1_step6` | `eval/loss` |

**Step 2.5 (projection embed-anchor repair)**: Fixes the large-norm / orthogonal-direction drift between the encoder's projection output and the decoder's token-embedding manifold (diagnosed via `scripts/analyze_embedding_deviation.py` on 2026-04-17). Freezes compressor + decoder, retrains only `projection_block` (~19 M params) with MSE + cosine + log-norm loss against `decoder.embed_tokens(content_ids)` at `target_ratio=None`. `DecoderInitTrainer._resolve_bgkit_checkpoint` prefers `phase1_step2p5` for Step 3 and silently falls back to `phase1_step2` if no 2p5 checkpoint is registered — so this step is optional in the pipeline but, once present, is picked up automatically. Step 3 loads its encoder from Step 2.5 but sets `training.load_decoder_from_bgkit_checkpoint: false` — the Step 1 decoder (adapted to the pre-repair off-manifold projection) is discarded, and a fresh HF Qwen3.5-0.8B decoder is re-adapted via LoRA against the repaired projection under the Step 3 compression curriculum.

**Training phase pipeline**: Joint Block Pretrain → Phase 1 Steps 1-6 (compression pre-training on code; Step 3 is LoRA reconstruction with compression curriculum 0.95→0.10, Step 4 is QA-conditioned head supervision, Steps 5–6 are commit encoding / multi-objective compression) → Phase 2 (single-doc KR Steps 1-4, KB-scale KR Stages A/B/C, Track B git history, Track C user memory) → Phase 3 (agentic distillation from SWE-bench trajectories). The decoder is **Qwen3.5-0.8B throughout** — bgkit does not train any larger in-house target LLM.

**Survivorship head (2026-04-16 single-head)**: Phase 1 Step 3+ and Phase 2 use a **single head per level** inside the encoder (at layer 7 / pruned block 1):

- `head_base_l{0,1}` (name retained for checkpoint continuity; there is no "adapter" counterpart): BCE/moment-match anchor at L0; soft-attn-driven at L1.
- Operator-facing logit: `logits_for_op = tanh(base_raw / T)` where `T` is `head_tanh_temperature`. Auto-calibrated at sidecar load so `base_raw`'s std matches T (avoids brittle hardcoded temperature). Tanh saturation at ±1 is the sole structural guard against soft-attn inflating aggregate logit mass.
- The earlier two-head `base + (adapter_raw − μ)` composition + `AdapterMeanEMA` was removed 2026-04-16 once training stabilized after the LigerRMSNorm fix. A single head per level, trained by BCE + moment-match + soft-attn, is sufficient.

Selection is `logits_for_op > θ` against a single global threshold θ owned by `DualThresholdController` and updated externally by **dual ascent** on the aggregate keep-rate against the curriculum's target compression ratio. There is **no straight-through estimator** on the hard mask — head gradient flows only via BCE + moment-match + soft-attn.

The earlier ratio-embedding-at-layer-3 was **removed**. `target_ratio` is consumed by the operator (DualThresholdController), not by a learned embedding. See "Known limitations" below for the Phase 2 KB regression this introduces.

Hard flag embeddings (survive/doomed) propagate the decision to subsequent layers for consolidation. See `configs/model/survivorship_head.yaml` and the per-level `survivorship.l{0,1}` blocks in each training config.

**Per-kernel Liger toggles**: `use_liger_{rmsnorm,swiglu,rope,ce}` in `configs/compute/dgx_spark.yaml`. `use_liger_rmsnorm` is **off** by default — liger-kernel 0.7.x's LigerRMSNorm silently corrupts backward on Qwen3.5 (decoder loss jumps to the LM prior). SwiGLU + RoPE + fused linear-CE remain enabled.

**Survivorship training signals** (all stages):
- **BCE warmup** (L0, steps 0-`bce_warmup_steps`): direct ICE-teacher supervision on `base_raw` at a fixed `teacher_ratio` (0.10 by default). Cuts off hard at the warmup boundary; ICE is then unloadable via `ICETeacher.unload()` — no runtime ICE dependency post-warmup. When a pre-distilled sidecar is supplied, online BCE warmup is skipped entirely.
- **Moment match** (L0, permanent, `moment_match_start_step` gated): MSE between standardized 3rd+4th moments of `base_raw` and *fixed* reference moments pre-computed offline by `scripts/probe_ice_distribution.py` (saved as two floats in `$DATA_DIR/diagnostics/ice_reference_moments.json`).
- **Decisiveness loss** (L1 primary, L0 off by default): `mean(4·p·(1-p))` penalizes probabilities near 0.5. At L1 it provides the symmetry-break signal that BCE provides at L0.
- **Soft attention** (from `soft_attn_start_step`, weight 0.2 at L0, 0.3 at L1): runs a second decoder forward via `PrunedBidirectionalQwen35.forward_from_block(start_block=2)` on prob-gated layer-7 embeddings (boundary-matched to the hard forward). Gradient flows from decoder CE loss through `softattn_probs = σ(logits_for_op − θ)` back through tanh into the head.
- **Aggregate ratio loss** (default weight 0.0 — defer to dual-ascent θ): when used, `(mean(σ(logits_for_op − θ)) − target)²`.
- **Relevance loss** (Phase 2 only): per-group aggregate-ratio targets. Gold-article positions (IDs + content) target `gold_boost × target_ratio` (default 1.5× — upsample), distractor positions target `distractor_damp × target_ratio` (default 0.5× — mildly downsample).
- **Min-survivors loss** (L0 + L1, default weight 0.05): relative squared hinge on per-sample soft survivor count, `mean((max(0, 1 − soft_count/N_min))²)` with `N_min = max(min_survivors_absolute_min, ⌈floor_ratio × content_len⌉)` (default `floor_ratio=0.02`, `absolute_min=1`). Uses a larger-τ sigmoid (`min_survivors_tau=0.3`) so gradient flows through tanh saturation — targets the "confidently silent" head mode where all logits pile at the tanh floor, observed on license-header / boilerplate samples (see 2026-04-17 survivor analyzer). Implemented in `compute_survivorship_losses` (helper) AND inline in `KRKBTrainer._compute_survivorship_aux_losses` for Phase 2's per-article L0 / per-trajectory L1 paths. Monitored via `{level}_zero_survivor_rate`, `{level}_low_survivor_rate_lt5`, `{level}_median_survivors` diagnostics.

**Dual-ascent θ under gradient accumulation**: token-budget batching gives microbatches with variable valid/controllable counts. The trainer aggregates `(sum, count)` tuples per microbatch (via `accumulate(state, enc_out)` from `bgkit.training.survivorship_helpers`) and updates θ ONCE per optimizer step using the **true global mean** (NOT mean-of-means).

**Checkpoint registry note (legacy key filters)**: `BgKITEncoder.from_pretrained_with_state_dict` filters legacy `compressor.ratio_embedding.*` (pre-2026-04), `compressor.survivorship_head_l{0,1}.*` (pre-single-head), and short-lived `compressor.head_adapter_l{0,1}.*` / `compressor.adapter_mean_ema_l{0,1}.*` (befd361-era two-head) keys before loading. Remove these filters once Step 2 is re-run under the current architecture.

**Known limitations**:
- **Phase 2 KB ratio-conditioning regression** (pending separate design pass): removing the ratio embedding means the encoder backbone no longer adapts representation per ratio. For Phase 2 KB with per-query ratios in the Pareto sweep at [0.5, 0.1, 0.05, 0.02, 0.01], this likely hurts ablation numbers at extreme ratios. See `docs/survivorship_design.md §Phase 2 KB regression` for the constraint and possible mitigations.

## Attention backend (FA4 packed-attention, 2026-04-20 migration)

All encoder / decoder / DeltaNet attention runs via **FA4 varlen packed** on sm_121. No padded / `attention_mask` fallback exists — fallbacks were removed during the migration per the aggressive-removal directive.

**Packed data shape** (FA4 varlen convention, invariant across encoder + decoder + DeltaNet + losses):
- Flat `(N,)` or `(N, D)` over samples, `N = sum(L_i)`.
- `cu_seqlens: (B+1,)` int32 cumulative sequence lengths (`cu_seqlens[0] == 0`, `cu_seqlens[-1] == N`).
- `max_seqlen: int` for FA4 block selection.
- `position_ids: (N,)` int64, per-sample restart (each sample's positions go 0, 1, …, L_i − 1).
- No `attention_mask` at the attention boundary. Semantic masks (`loss_mask`, `valid_mask`) stay flat `(N,)`.
- Repo-variant compression has **two-level packing**: `cu_file_seqlens` (one segment per file across all repos) + `cu_repo_seqlens: (B+1,)` (indices INTO `cu_file_seqlens` marking repo boundaries).

**Helpers**: `src/bgkit/utils/packing.py` — `PackedBatch` + `segment_ids_from_cu`, `segment_mean`, `segment_sum`, `segment_max`, `lengths_from_cu`, `position_ids_from_cu`. Use these for any per-sample reduction on flat tensors.

**Sampler**: `PackedTokenBudgetSampler` in `src/bgkit/data/samplers.py` with budget `sum(L_i²) ≤ max_batch_tokens × reference_seq_len` (quadratic — reflects that FA4 varlen attention cost is `sum(L_i²)`, not `B × max_len²`). `QueryAwareBatchSampler` (Phase 2) is preserved. Old `TokenBudgetBatchSampler` / `LengthSortedBatchSampler` are deleted.

**Live tuning via `${CHECKPOINT_DIR}/control.json`**: Write a namespaced JSON file to adjust hyperparameters while training runs, without restarting. The trainer polls this file every step. Source of truth: `src/bgkit/training/live_config.py` + each trainer's `LIVE_CONFIG_FIELDS` / `LIVE_CONFIG_HANDLERS`. Keys handled in `BaseTrainer` (inherited by all trainers) are in the table below; trainer-specific keys are declared in each trainer's class dict.

Example control file (one block per active phase):
```json
{
  "phase1_step3": {
    "target_ratio": 0.15,
    "max_batch_tokens": 32768,
    "eval_every": 250
  },
  "phase1_step6": {
    "lr": 3e-5
  }
}
```

| Key | Trainer(s) | Effect |
|---|---|---|
| `lr` | All | Scales all param-group base LRs proportionally |
| `max_steps` | All | Extends the training horizon (must be > current step) |
| `warmup_steps` | All | Adjusts cosine-warmup window |
| `eval_every` | All | Changes eval cadence (steps) |
| `save_every` | All | Changes checkpoint cadence (steps) |
| `early_stopping_patience` | All | Adjusts early-stopping patience |
| `max_batch_tokens` | All Phase 1 + Phase 3 | Rebuilds `PackedTokenBudgetSampler` + train dataloader; preserves epoch cursor |
| `max_batch_tokens_eval` | All Phase 1 + Phase 3 | Rebuilds eval dataloader with new budget |
| `target_ratio` | DecoderInit, Compression, CommitEncoding | Override compression ratio (null to resume ramp) |
| `target_ratio_start` | DecoderInit, Compression, CommitEncoding | Ramp start value |
| `target_ratio_end` | DecoderInit, Compression, CommitEncoding | Ramp end value |
| `target_ratio_ramp_steps` | DecoderInit, Compression, CommitEncoding | Ramp length |
| `compression_introduction_step` | DecoderInit | Step at which compression is first enabled |
| `encoder_unfreeze_step` | DecoderInit | Step at which encoder unfreezes |
| `diagnostic_metrics_every_n_steps` | DecoderInit, Compression | Survivorship diagnostic cadence |
| `generation_eval_every` | DecoderInit | Generation-eval cadence (every Nth eval) |

`max_batch_tokens` / `max_batch_tokens_eval` rebuild is implemented in `BaseTrainer._rebuild_train_dataloader_with_budget` / `_rebuild_eval_dataloader_with_budget`. Trainers that support it stash `_train_lengths`, `_eval_lengths`, `_train_collate_fn`, `_num_workers`, `_pin_memory` in `setup()`. Phase 2 `KRKBTrainer` uses `QueryAwareBatchSampler` — not `PackedTokenBudgetSampler` — so it does not support `max_batch_tokens` live tuning.

**Collators**: `src/bgkit/data/collators.py` — `collate_token_ids`, `collate_chat_repro`, `collate_compression` (dispatches to file / repo variants), `collate_qa`. Names unchanged from pre-migration; output dicts are packed.

**FA4 SM12x native bootstrap**: `docker/entrypoint.sh::bootstrap_flash_attn_native()` detects `native_sm12x_owned_backend_available() = False` (aten-only, rejected by `require_sm12x_owned_backend()`), copies `/workspace/flash-attention` (bind-mount of `~/flash-attention` FA4 fork) to `${CHECKPOINT_DIR}/.flash-attn-native/repo`, runs `pip install -e . --no-build-isolation --user` inside container (one-time, ~10–30 min; cached by `find . -type f` SHA), and prepends the cache to `PYTHONPATH` so `flash_attn.flash_attn_interface` imports the rebuilt-against-container-libtorch FA2 .so. On SM12x it now also builds the optional `flash_attn.cute._sm12x_native` extension inside that cache when missing and, if the module imports successfully, exports `FLASH_ATTENTION_SM12X_USE_EXTENSION=1` automatically so bgkit prefers the extension path by default. **If the image is rebuilt, the bootstrap ships with it — verify by checking `$PYTHONPATH` starts with the cache repo path and logs `flash_attention_native_backend:extension` when the extension is available.**

**Parity fixtures** (`tests/fixtures/`): 6 padded-reference fixtures (encoder, decoder, DeltaNet, survivorship losses, Phase 2 losses, Step 3 smoke microbatch) captured via `scripts/capture_parity_fixtures.py` before the migration. Consumed by parity tests at `tests/unit/models/test_*_packed.py` and `tests/unit/training/test_survivorship_helpers_packed.py` / `test_kr_kb_packed.py`.

**Baseline for live validation** (`docs/baselines/phase1_step3_04_20_baseline.json`): 04-20 padded run `werg/bgkit/6wznpmwv` — step wall-clock 8.94 s/step, cuda_max_allocated 16.2 GB, final loss 0.24 at step 2599. Wave 4.5 validation compares packed runs against this.

**Memory cap for profiler** (lessons learned from a host OOM): `docker/docker-compose.yaml` profile services (`profile-phase1-step3` etc.) apply `mem_limit: 80g` + `memswap_limit: 80g`. Any ad-hoc `docker compose run` of the profiler outside the capped services can still OOM the host on unified memory — use the named service.

**Memory budget scope semantics** (as of 2026-04-22): `memory_budget_scope` cap checks use **delta-peak semantics** (cuda_peak_gb − cuda_pre_gb), not absolute peak. Cap values in `configs/compute/dgx_spark.yaml` now measure scope-local new allocation, not total allocation at scope exit. Prior absolute-peak caps trivially tripped when a scope entered with training state resident; the new delta semantics match the caps' intent and prevent false positives (commit `872c290`).

## flash-linear-attention (fla) on sm_121

Stock PyPI `fla==0.4.2` is installed in the training container via `pip install ".[gpu]"`, **but the local fork at `/home/werg/flash-linear-attention/` (branch `blackwell-sm121-compat`, rebased onto upstream `761fc0b`) is bind-mounted into all GPU services** at `/workspace/fla:ro` and prepended to `PYTHONPATH`, so `import fla` resolves to the fork (not dist-packages). The fork carries:

1. **Upstream `761fc0b` (PR #798) — persistent autotune cache** (latest upstream HEAD as of 04-26): `FLA_CACHE_RESULTS=1` (default on), backing JSON files under `~/.triton/cache/{sha}/{kernel}.autotune.json`. Survives container restarts; ends the "lucky autotune state" volatility we saw on 04-26. Plus all upstream history including PR #813 (fused gate kernel), #823 (b_dg backward opt), #849 (TileLang variant of #823).
2. **Broadened `IS_NVIDIA_BLACKWELL`** (already in upstream as `8b05e2f`): widened from `capability[0] == 10` to `>= 10` so sm_121 (capability (12, 1) on DGX Spark) qualifies as Blackwell. Activates the `global_scratch` allocator (a NullAllocator-deadlock fix on autotuned kernels), the `safe_dot` substitution, and any future Blackwell-gated paths.
3. **sm_121-tuned autotune configs + #790 audit** (our commits `a7ed16a` + `35b484a`): adds Blackwell extras (`num_warps∈{2,4}, num_stages=5`) and trims original tuning to drop configs that hit the `(num_warps > num_stages > 1)` trap from [fla #790](https://github.com/fla-org/flash-linear-attention/issues/790) / [triton #8695](https://github.com/triton-lang/triton/issues/8695) — those configs cause silent ~5% bf16 numerical corruption in `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` on Blackwell. Net config count after audit: 51 → ~16 sm_121-specific configs (correctness > exhaustive search).
4. **Opt-in `chunk_fwd_o_sm121` custom kernel** (commit `f1849a7`): default-off via `FLA_USE_SM121_CUSTOM_KERNEL=1`. UNVERIFIED — has no GPU parity test passed yet, AND the original draft was sized assuming 228 KB SMEM (true for H100, **not** sm_121 which has 101 KB). Almost certainly won't compile as-written; needs re-tile to BK=64, BV=128, num_stages=3 max. Leave off unless deliberately experimenting. See `plans/deltanet-custom-kernel.md` Phase 2 for re-tile plan.

**Hardware truth about sm_121 / GB10**: 24 SMs (vs H100's 132), **101 KB shared mem per SM** (NOT 228 KB — that's H100), 5th-gen tensor cores via `mma.sync.aligned.block_scale` (Ampere-style encoding, NOT `tcgen05`/TMEM/2-SM-MMA which are sm_100-only). Cluster size limited to 1×1×1. TMA via Triton 3.6 works on sm_121 (verified). FlashInfer's `gdn_prefill_sm100` and cuLA's KDA kernels target sm_100 features and don't run on sm_121.

Runtime patches that remain (in bgkit, not fla):
- `src/bgkit/utils/deltanet_patch.py` — clamps per-step gate to `>= -1.3` to prevent backward NaN on Qwen3.5 heads with extreme `A_log`/`dt_bias`, and wires `cu_seqlens` into `Qwen3_5GatedDeltaNet.forward`. No upstream replacement.
- `src/bgkit/utils/triton_patch.py` — sm_121-scoped autotuner `_bench` error catcher + `CompiledKernel._init_handles` retry. Defensive; low cost.
- `FLA_USE_TMA=1` in compose — TMA verified to work on sm_121 with Triton 3.6 (was defensively pinned to 0; re-enabled 2026-04-21).

After restart with the bind-mount, verify the fork is loaded:
```
docker exec <container> python -c \
  "import fla; from fla.utils import IS_NVIDIA_BLACKWELL; print(fla.__file__, IS_NVIDIA_BLACKWELL)"
# Expect: /workspace/fla/fla/__init__.py True
```

The fork is in-memory autotune only (no persistent cache); first ~50 steps after restart pay autotune-benchmarking cost as Triton tries each new config, then the winner is cached for the process lifetime.

| Task | Command |
|---|---|
| Backfill registry | `make ckpt-backfill` or `.venv/bin/bgkit-ckpt backfill` |
| List checkpoints | `.venv/bin/bgkit-ckpt list --phase phase1_step4` |
| Best checkpoint | `.venv/bin/bgkit-ckpt best --phase phase1_step6 --metric eval/loss` |

## Perf playbook for new training stages

When a stage feels slow on DGX Spark, **profile first, hypothesize second**:
see `docs/dgx_spark_perf_playbook.md`. The 04-26 phase1_step3 70 → 12.5
s/step investigation produced a sharp set of lessons:

1. **fla autotune fix is automatic** for all `gpu-common` services on
   restart (DeltaNet runs ~5× faster on sm_121 once Triton picks the new
   Blackwell autotune configs).
2. **Keep `gradient_checkpointing: true`** (full ckpt-on, the default).
   Two attempts to reduce ckpt cost on phase1_step3 both broke memory:
   `false` (peak 62 GB on long-tail sample) and `"selective"` (skip 18
   DeltaNet layers, peak 71 GB — DeltaNet's chunk-level recurrent state
   is much larger than the per-layer hidden state alone). The
   `"selective"` mode is still implemented in
   `bgkit.training.gradient_utils` for future experiments (e.g. skipping
   only a subset of DeltaNet layers) but is not the default. Don't
   enable without measuring `cuda_max_allocated` over 100+ steps.
3. **Don't raise `max_batch_tokens` past 16384** for Step-3-style
   workloads; DeltaNet kernel cost is `sum(L_i²)` and bigger microbatches
   lose. step5/step6 still use 32768/2 (pre-autotune tuning); worth
   re-evaluating to 16384/N on first run of those stages.

## Inference Server

### vLLM (primary)

Local LLM inference via vLLM in Docker, two-tier routing: primary model (GPT-OSS-20B, MXFP4 quantization via `vllm-node-mxfp4` image from eugr) for complex files, fast model (Qwen3.5-0.8B via `vllm-node` image) for config/test/simple files. Uses continuous batching, prefix caching, and FlashInfer MOE kernels for high throughput.

| Task | Command |
|---|---|
| Generate descriptions | `make generate-descriptions` |
| Download HF models | `make download-models-hf` |
| Start vLLM servers | `make vllm-server` |
| Stop vLLM servers | `make vllm-server-stop` |
| Tail logs | `make vllm-server-logs` |

Override model/params via env: `VLLM_IMAGE_PRIMARY`, `VLLM_IMAGE_FAST`, `VLLM_MODEL_PRIMARY`, `VLLM_MODEL_FAST`, `VLLM_GPU_UTIL_PRIMARY` (default 0.45), `VLLM_GPU_UTIL_FAST` (default 0.10), `VLLM_PORT_PRIMARY` (default 8090), `VLLM_PORT_FAST` (default 8091).

HF models are cached in `~/.cache/huggingface` (bind-mounted into container).

### llama-server (legacy)

Three-tier llama.cpp setup still available for GGUF models.

| Task | Command |
|---|---|
| Build llama image | `make docker-build-llama` |
| Download GGUF models | `make download-models` |
| Start server | `make llama-server` |
| Stop server | `make llama-server-stop` |
| Tail logs | `make llama-server-logs` |
| Benchmark/tune | `make llama-bench` |

### Python client

`bgkit.inference.LlamaClient`: async/sync HTTP client compatible with both vLLM and llama-server. Auto-detects backend via `/version` probe (vLLM returns version info, llama-server 404s). Handles concurrency control, retry on 503/429, warmup, and backend-specific behavior (e.g., `chat_template_kwargs` sent to llama-server but omitted for vLLM).

Configure via `InferenceConfig.backend_type`: `"auto"` (default, probes `/version`), `"vllm"`, or `"llama"`.

`generate_descriptions.py` uses two-tier routing (`--server-url-primary` / `--server-url-fast`). Old `--llama-url-*` flags still work as deprecated aliases.

`make generate-descriptions` is the single entry point: starts vLLM servers if needed, waits for health, then runs generation. Idempotent and resumable — skips repos that already have output files, safe to Ctrl-C and re-run.

## Code Quality

- Ruff for linting and formatting (line-length 100)
- pytest markers: `slow`, `gpu`, `integration`, `smoke`
- Pre-commit hooks configured (ruff)

## Execution Runbook — Remaining Pipeline Steps

All code is implemented. The items below are **execution tasks** that require trained checkpoints, GPU time, network access, or running services. They form a dependency chain — each step's gate condition must be met before starting it.

**When starting a new session**, check this list and execute the first unblocked item. Mark items done by changing `[ ]` to `[x]` with the date.

### Phase 2 Data Preparation (gate: network + disk + CPU)

- [ ] **Convert HF datasets to mmap** — Run for each dataset. CPU-only, no GPU needed. ~2-4 hours total.
  ```bash
  for ds in pubmedqa pubmedqa_artificial newsqa searchqa msmarco_passage narrativeqa; do
    python scripts/convert_hf_to_mmap.py $ds --output-dir $DATA_DIR/mmap/phase2/$ds
  done
  # KILT Wikipedia (large — ~5.9M articles):
  python scripts/convert_hf_to_mmap.py kilt_wikipedia --output-dir $DATA_DIR/mmap/phase2/kilt
  # KILT tasks (11 tasks):
  for task in kilt_nq kilt_hotpotqa kilt_fever kilt_zsre kilt_trex kilt_wow kilt_eli5 kilt_aidayago2 kilt_wned kilt_cweb kilt_triviaqa; do
    python scripts/convert_hf_to_mmap.py $task --output-dir $DATA_DIR/mmap/phase2/$task
  done
  ```

- [ ] **Convert memory datasets to mmap** — Requires HF access. CPU-only.
  ```bash
  for ds in msc share chronicles perltqa laps; do
    python scripts/convert_memory_datasets.py $ds --output-dir $DATA_DIR/mmap/phase2/$ds
  done
  ```

- [ ] **Extract dependency tags + build taxonomy** — CPU-only, scans local repos.
  ```bash
  python scripts/extract_dependency_tags.py repos --root-dir $DATA_DIR/repos --output-dir $DATA_DIR/taxonomy
  # After KILT/PubMedQA conversion, add their tags:
  python scripts/extract_dependency_tags.py kilt --metadata $DATA_DIR/mmap/phase2/kilt/metadata.parquet --output-dir $DATA_DIR/taxonomy
  python scripts/extract_dependency_tags.py pubmedqa --metadata $DATA_DIR/mmap/phase2/pubmedqa/metadata.parquet --output-dir $DATA_DIR/taxonomy
  ```

### Phase 2 Git QA Generation (gate: vLLM server running + repos available)

- [ ] **Generate git QA pairs** — Needs vLLM primary server on port 8090. Idempotent/resumable. GPU for inference, ~10-20 hours for 10K repos.
  ```bash
  make vllm-server  # start vLLM if not running
  python scripts/generate_git_qa.py \
    --repos-dir $DATA_DIR/repos \
    --output-dir $DATA_DIR/mmap/phase2/git_qa \
    --server-url http://localhost:8090/v1 \
    --workers 8
  ```

### Phase 1 Evaluation (gate: Phase 1 Step 6 checkpoint exists)

- [ ] **Run Eval 1 (post-Phase 1 baselines)** — GPU needed for encoder/decoder inference.
  ```bash
  python scripts/eval_phase1.py +eval.checkpoint=$CHECKPOINT_DIR/phase1_step6_best +eval.output_dir=$CHECKPOINT_DIR/eval_reports
  ```

### Phase 2 Training (gate: mmap datasets + Phase 1 checkpoint)

Phase 2 is unified around a single trainer (`KRKBTrainer`). The legacy
`KRTrainer` and the per-step `phase2_step{1-4}.yaml` configs have been
deleted — every Phase 2 dataset goes through the trajectory framework
now. Flat datasets (NewsQA, MS MARCO, SearchQA, git history, memory)
emit single-bgkit trajectories; hierarchical datasets (KILT via DBpedia
categories, PubMedQA via MeSH, NarrativeQA per book) emit
`browse → bgkit → answer` trajectories. Two stages:

- [ ] **Phase 2 Stage A** — live L0, L0 LoRA + L1 LoRA + decoder
      trainable. One epoch over the bootstrap mix
      (PubMedQA + NarrativeQA + git_history + memory). Produces the
      Stage A checkpoint that bakes the L0 LoRA into the cache for
      Stage B.
  ```bash
  scripts/run-train.sh --no-follow train-phase2-kb-stage-a
  ```
- [ ] **Phase 2 Stage B** — cached L0 (built from the Stage A
      checkpoint), L0 LoRA frozen, L1 LoRA + decoder train. Full-scale
      corpus including Wikipedia, all KILT/PubMedQA, MS MARCO,
      SearchQA, NewsQA, git history, memory.
  ```bash
  scripts/run-train.sh --no-follow train-phase2-kb-stage-b
  ```

### Phase 2 Evaluation (gate: Phase 2 checkpoints)

- [ ] **Per-stage eval** — Run after each stage completes.
  ```bash
  python scripts/eval_phase2_step.py training=phase2_kb_stage_a +eval.checkpoint=...
  ```
- [ ] **Comprehensive eval (Eval 3)** — Run after Stage B. Go/no-go gate for Phase 3.
  ```bash
  python scripts/eval_phase2_comprehensive.py +eval.checkpoint=$CHECKPOINT_DIR/phase2_kb_stage_b_best
  ```
- [ ] **Topic embedding ablation** — Run with topic embeddings enabled.
  ```bash
  python scripts/eval_topic_embeddings.py +eval.checkpoint=...
  ```
- [ ] **Retention-ratio Pareto sweep** — Uses pre-computed L0 sub-selection at [0.50, 0.10, 0.05, 0.02, 0.01]. Part of comprehensive eval.

### Phase 2 KB-Scale (gate: mmap datasets + Phase 1 checkpoint)

`KRKBTrainer` trains over browse + bgkit trajectories on every Phase 2
dataset (KILT Wikipedia, MS MARCO, PubMedQA, NewsQA, SearchQA,
NarrativeQA, the memory corpora, and git history). Hierarchical
datasets emit `browse → bgkit → answer` trajectories; flat datasets
emit single-bgkit trajectories with no browse step. Two stages:
**Stage A** (live L0, L0 LoRA trainable, one epoch over the bootstrap
mix) → **Stage B** (cached L0 re-computed from Stage A weights, L0 LoRA
frozen, L1 LoRA + decoder train, full corpus). Stage C from earlier
plans has been merged into Stage B.

End-to-end data-prep + trainer forward is covered by `tests/integration/test_phase2_kb_e2e.py`, which runs the pipeline on a toy 250-article corpus with stubbed encoder + decoder. Use it as a regression gate whenever any of the scripts below change:

```bash
make test-integration                       # runs all integration tests
.venv/bin/pytest tests/integration/test_phase2_kb_e2e.py -v  # just the KB e2e test
```

#### Data prep (per dataset)

Each dataset (`kilt_wikipedia`, `msmarco_passage`, `pubmedqa`, `newsqa`, `searchqa`, `narrativeqa`, `git_history`, `msc`, `share`, `chronicles`, `perltqa`, `laps`) runs the same five-step sequence. Substitute `$DS` for the dataset name below.

- [ ] **Build browse tree** (gate: `$DATA_DIR/mmap/phase2/$DS/metadata.parquet` exists)
  ```bash
  python scripts/build_browse_tree.py \
    --dataset $DS \
    --phase2-dir $DATA_DIR/mmap/phase2/$DS \
    --output-dir $DATA_DIR/browse_trees
  # → writes $DATA_DIR/browse_trees/$DS.parquet
  ```
  For datasets with an external hierarchy (KILT category DAG, PubMedQA MeSH tree), pre-compute the hierarchical paths and pass `--input paths.jsonl` instead of `--phase2-dir`.

- [ ] **Build per-dataset provenance JSONL** (gate: mmap exists with `provenance_json` or equivalent column)
  ```bash
  # KILT tasks (NQ, HotpotQA, FEVER, zsRE, T-REx, WoW, ELI5, etc.):
  python scripts/build_provenance_kilt.py --mmap-dir $DATA_DIR/mmap/phase2/$DS --output $DATA_DIR/provenance/$DS.jsonl
  # All other datasets have their own scripts:
  #   scripts/build_provenance_kilt_wikipedia.py — KILT Wikipedia corpus
  #   scripts/build_provenance_msmarco.py
  #   scripts/build_provenance_newsqa.py
  #   scripts/build_provenance_searchqa.py
  #   scripts/build_provenance_pubmedqa.py
  #   scripts/build_provenance_narrativeqa.py
  #   scripts/build_provenance_git_history.py
  #   scripts/build_provenance_memory.py
  # Each reads $DATA_DIR/mmap/phase2/$DS/metadata.parquet and emits a JSONL
  # with {question, gold_answer, gold_article_id, scope_template,
  # scope_description} rows compatible with build_teacher_trajectories.py.
  ```

- [ ] **Generate teacher trajectories** (gate: browse tree parquet + provenance JSONL)
  ```bash
  python scripts/build_teacher_trajectories.py \
    --input $DATA_DIR/provenance/$DS.jsonl \
    --dataset $DS \
    --browse-tree $DATA_DIR/browse_trees/$DS.parquet \
    --output-dir $DATA_DIR/trajectories \
    --exploration-fraction 0.20 --seed 17
  # → writes $DATA_DIR/trajectories/$DS.parquet in the KBTrajectoryDataset schema
  ```

- [ ] **Enumerate trajectory article set** (gate: trajectory parquet)
  ```bash
  python scripts/build_trajectory_set.py \
    --trajectory $DATA_DIR/trajectories/$DS.parquet \
    --browse-tree $DATA_DIR/browse_trees/$DS.parquet \
    --dataset $DS \
    --output $DATA_DIR/trajectory_sets/$DS.jsonl
  # → writes {dataset, article_id} rows for every article any trajectory touches
  ```
  Pass `--trajectory`/`--browse-tree`/`--dataset` multiple times to union several datasets into one article set for a combined pre-compute.

- [ ] **Pre-compute L0 for the trajectory subset** (gate: trajectory set JSONL + Phase 1 checkpoint)
  ```bash
  python scripts/precompute_l0_subset.py \
    --articles $DATA_DIR/trajectory_sets/$DS.jsonl \
    --mmap-dir $DATA_DIR/mmap/phase2 \
    --phase1-checkpoint $CHECKPOINT_DIR/phase1_step6_best \
    --output-dir $DATA_DIR/l0_cache_kb \
    --retention-json configs/phase2_kb/l0_retention.json \
    --lora-rank 32
  # → populates $DATA_DIR/l0_cache_kb/$DS/shard_NNNN/{survivors,offsets}.npy + index.parquet
  ```
  Omit `--stage-a-checkpoint` on the bootstrap pre-compute (before Stage A has trained); include it during the Stage A → B transition below.

#### Stage A training (live L0, L0 LoRA trainable)

- [ ] **Stage A** (gate: browse trees + trajectories for the Stage A subset + bootstrap L0 cache + Phase 1 checkpoint)
  ```bash
  scripts/run-train.sh --no-follow train-phase2-kb-stage-a
  # or direct:
  # docker compose -f docker/docker-compose.yaml run --rm train-phase2-kb-stage-a
  ```
  Stage A trains the L0 LoRA adapter in live mode (`live_l0: true`) on a KILT + PubMedQA subset. The L1 LoRA is also trainable. Produces a checkpoint at `$CHECKPOINT_DIR/phase2_kb_stage_a_best`.

#### Stage A → B transition

- [ ] **Re-build L0 cache using Stage A's LoRA weights** (gate: Stage A checkpoint)
  ```bash
  python scripts/precompute_l0_subset.py \
    --articles $DATA_DIR/trajectory_sets/stage_b.jsonl \
    --mmap-dir $DATA_DIR/mmap/phase2 \
    --phase1-checkpoint $CHECKPOINT_DIR/phase1_step6_best \
    --stage-a-checkpoint $CHECKPOINT_DIR/phase2_kb_stage_a_best \
    --output-dir $DATA_DIR/l0_cache_kb \
    --retention-json configs/phase2_kb/l0_retention.json \
    --lora-rank 32
  ```
  Passing `--stage-a-checkpoint` installs the Stage A LoRA router and loads its trained L0 weights before encoding, so the cached survivors reflect Stage A's text-adapted behavior instead of bare Phase 1. Without this flag the Stage A training is effectively discarded. The script appends new shards to `$DATA_DIR/l0_cache_kb/$DS/` and updates `index.parquet` idempotently.

#### Stage B training (cached L0, L0 LoRA frozen, L1 LoRA trainable)

- [ ] **Stage B** (gate: Stage A checkpoint + rebuilt L0 cache)
  ```bash
  scripts/run-train.sh --no-follow train-phase2-kb-stage-b
  ```
  Stage B disables live L0 (`live_l0: false`), freezes the L0 LoRA, and trains only the L1 LoRA + decoder head on the cached L0 survivors. Same dataset subset as Stage A but with the large KILT corpus fully enumerated.

#### Evaluation

- [ ] **KB-scale eval** (gate: Stage A or B checkpoint) — `scripts/eval_phase2_kb.py` reuses `_eval_one_sample` + `_build_decoder_segments_with_trace` from `KRKBTrainer` to score answer EM/F1, browse tool-call ID accuracy, bgkit tool-call ID accuracy, and trajectory step accuracy.
  ```bash
  python scripts/eval_phase2_kb.py +eval.checkpoint=$CHECKPOINT_DIR/phase2_kb_stage_b_best
  ```
  Also runs the ablation matrix (`zeroed`, `noise`, `no_topics`, `topics_only`, `neither`) via `KRKBTrainer.set_ablation_mode()`.

### Phase 3 Data Preparation (gate: network + repos)

- [ ] **Download SWE-bench trajectories**
  ```bash
  python scripts/download_swe_trajectories.py --datasets openhands --output-dir $DATA_DIR/trajectories
  ```
- [ ] **Filter trajectories** — CPU-only, fast.
  ```bash
  python scripts/filter_trajectories.py \
    --input $DATA_DIR/trajectories/openhands_trajectories.jsonl \
    --output $DATA_DIR/trajectories/openhands_filtered.jsonl
  ```

### Phase 3 Encoding (gate: Phase 2 best checkpoint + filtered trajectories + repos)

- [ ] **Encode SWE-bench repos** — Pre-compute BgKIT embeddings with blob-SHA dedup. ~24 GPU-hours.
  ```bash
  python scripts/encode_swe_repos.py \
    --checkpoint $CHECKPOINT_DIR/phase2_kb_stage_b_best \
    --trajectories $DATA_DIR/trajectories/openhands_filtered.jsonl \
    --repos-dir $DATA_DIR/swe_repos --output-dir $DATA_DIR/swe_embeddings
  ```

### Phase 3 Training (gate: encoded repos + filtered trajectories)

- [ ] **Phase 3 distillation** — Distillation from external SWE-bench teacher trajectories (OpenHands / Llama-70B / Claude Sonnet, etc.) into the Qwen3.5-0.8B student + BgKIT context. ~100K steps.

### Phase 3 Evaluation (gate: Phase 3 checkpoint + SWE-bench repos cloned)

- [ ] **SWE-bench eval (Lite)** — Interactive agent loop, ~300 instances. Start here for fast iteration.
  ```bash
  python scripts/eval_swebench.py generate \
    --checkpoint $CHECKPOINT_DIR/phase3_best \
    --repos-dir $DATA_DIR/swe_repos --subset lite
  python scripts/eval_swebench.py evaluate --predictions predictions.jsonl
  ```
- [ ] **SWE-bench eval (Verified)** — 500 instances, final numbers.
- [ ] **Knowledge source ablation** — With/without BgKIT context.
  ```bash
  python scripts/eval_swebench.py ablation \
    --checkpoint $CHECKPOINT_DIR/phase3_best \
    --repos-dir $DATA_DIR/swe_repos --output-dir ablation_results
  ```
- [ ] **Exploration-dropout sweep** — Train 3 models with p=0.5, 0.8, 1.0, evaluate all 3.
