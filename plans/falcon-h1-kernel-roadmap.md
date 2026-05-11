# Falcon-H1 Kernel Roadmap

## Posture

Do not optimize Falcon-H1 by sprinkling small fallbacks or local special cases
into the trainer. Kernel work should start from a Docker-native benchmark jig,
then move through a fork overlay that can be swapped into the training image
without rebuilding unrelated dependencies. Every candidate below must prove:

- the intended kernel path is called,
- fallback paths stay cold or fail loudly,
- forward and backward both pass,
- timings improve on GB10 / sm_121 under the actual container stack.

Current verified stack:

- NGC PyTorch 26.03, torch 2.11.0a0, CUDA 13.2
- Transformers 5.8.0
- Falcon-H1 decoder attention: PyTorch SDPA, configured flash-only on CUDA
- Falcon-H1 Mamba path: local `mamba-ssm` Python/Triton ops plus `causal-conv1d`
- Hugging Face `kernels`: intentionally absent; current artifact is ABI-broken
  on torch 2.11 / CUDA 13 / aarch64.

## To-Do

### 1. Fused Mamba Training Kernel

Target: `mamba_ssm.ops.triton.ssd_combined.mamba_split_conv1d_scan_combined`.

This is the main teacher-forced Falcon-H1 training hot path. It fuses causal
conv, scan, gated RMSNorm, and output projection. Start here before hand-tuning
cache/inference kernels.

Work items:

- Create a Docker-usable `mamba-ssm` fork overlay.
- Benchmark Falcon-H1 tiny and production-like shapes with forward+backward.
- Record kernel calls and fallback calls in the benchmark output.
- Inspect Triton configs / block sizes for Falcon-H1's stable shape family:
  bf16, attention head dim 64, Mamba head dim 32, Falcon-H1 Tiny hidden 512.
- Add sm_121-oriented candidate configs or shape-specialized dispatch only
  after the benchmark jig can compare them reproducibly.

### 2. Falcon Attention Kernel Path

Target: Falcon-H1's dense causal attention path.

Current correct path is PyTorch SDPA with flash enabled and math/mem-efficient
SDPA disabled. Native FlashAttention-2 is not viable today: the mounted build
rejects Falcon-H1 head dimension `<=64`. BgKIT FA4 is packed-varlen and is not
valid for Falcon's normal batched decoder attention.

Work items:

- Keep the family-aware resolver: Falcon-H1 uses SDPA, Qwen uses BgKIT FA4.
- Extend the benchmark jig to time SDPA flash-only separately from Mamba.
- Prototype a Falcon attention adapter only if profiling shows SDPA dominates:
  either pack dense `(B, H, S, D)` into FA4 varlen safely, or add a dense
  Falcon-specific FA4 path. Do not touch this before Mamba timings are known.

### 3. Cache / Inference Kernels

Targets:

- `causal_conv1d_fn`
- `causal_conv1d_update`
- `mamba_chunk_scan_combined`
- `selective_state_update`

These are used in eval/prefill and cached decode, not the main teacher-forced
training fast path. They matter once generation/eval throughput becomes a
bottleneck.

Work items:

- Benchmark prefill and one-token decode separately.
- Count all four calls in the benchmark output.
- Only fork/tune after the training fused path and dense attention are measured.

### 4. Falcon-Specific Fused Norm / MLP

Target: Falcon-H1 RMSNorm / gated norm / MLP helpers.

This is the lowest-priority kernel area until profiling shows it dominates.
Qwen's Liger patcher is intentionally not applied to Falcon-H1 because its
module assumptions are Qwen-specific.

Work items:

- Add profiler attribution for norm and MLP regions.
- Consider a Falcon-specific Liger-style patch only after Mamba and attention
  stop dominating.
- Keep this out of the trainer path until there is measured evidence.

## Performance Work Queue

These are the five broad follow-up items to work through before committing to a
deep hand-written kernel fork. Each item must run under Docker and either land a
change or produce evidence that the path is not worth pursuing yet.

### 1. Real Falcon Trainer Throughput

Status: in progress.

Acceptance criteria:

- `scripts/profile_training_step.py` supports Falcon phases through the same
  trainer dispatch as `scripts/train.py`.
- A compose service can run a short Falcon-H1 optimizer-step profile with real
  dataloader, optimizer, loss, LoRA, and decoder code.
- The report includes wall-clock step time and peak CUDA memory, not only model
  microbenchmarks.

### 2. Static Falcon / sm_121 Mamba Configs

Status: in progress.

Acceptance criteria:

- The mamba fork can choose known-good GB10 configs for Falcon-H1 hot Triton
  kernels before autotuning.
- The selection is guarded by env vars and falls back to normal autotune if a
  table entry does not match the candidate list exactly.
- Benchmarks report whether static config mode is enabled.

### 3. Full-Step Kernel Attribution

Status: in progress.

Acceptance criteria:

- The real trainer profile records call counts for Falcon Mamba kernels, SDPA,
  and fallback paths.
- The profiler report groups CUDA time into coarse buckets: Mamba, attention,
  cross entropy, optimizer, forward/backward wrapper, and uncategorized.
- The report is saved as JSON so we can compare runs after kernel edits.

### 4. Falcon Dense Attention Candidate

Status: in progress.

Acceptance criteria:

- The benchmark harness has a direct dense-attention candidate mode.
- It times PyTorch SDPA flash-only and attempts the available FlashAttention
  dense API if present.
- Failure is recorded as data; production remains on SDPA unless the candidate
  is both correct and faster.

### 5. Cache / Decode Sweep

Status: in progress.

Acceptance criteria:

- The benchmark harness can sweep prefill/decode sequence lengths and batch
  sizes in one Docker run.
- Output records the four cache/decode kernel call counts and timings per case.
- Any decode tuning is based on measured prefill/decode data, not training-only
  assumptions.

## First Milestone

The first milestone is not a hand edit. It is a repeatable Docker command that
loads the fork overlay and emits one JSON record per benchmark case with:

- environment versions,
- Falcon config shape,
- mode (`train`, `prefill`, `decode`),
- call counts for all Falcon kernel candidates,
- forward/backward timing,
- peak CUDA memory,
- pass/fail status for fallback guards.

Only after that should the fork start changing Triton configs or kernels.

Status:

- `docker compose ... benchmark-falcon-h1-kernels` is the reusable service for
  this check.
- `/home/werg/mamba-falcon-sm121` is mounted at
  `/workspace/mamba-falcon-sm121` and wins `import mamba_ssm` through
  `PYTHONPATH`.
- The all-mode smoke benchmark proves train, prefill, and decode use the
  intended kernels with `FalconH1Mixer.torch_forward` at zero calls.
- First fork-side change is an sm_121 autotune filter in
  `mamba_ssm.utils.determinism.autotune_configs`, enabled by
  `MAMBA_SM121_SAFE_AUTOTUNE=1`, to drop Falcon-H1 configs known to exceed the
  GB10 shared-memory cap before Triton spends cold-start time compiling them.
