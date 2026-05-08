# FlashQLA on DGX Spark / sm_121

BgKIT uses the training Docker image as the source of truth for FlashQLA work.
The host venv is not authoritative for CUDA, TileLang, FLA, or FlashAttention
behavior on GB10.

## Services

Run from the BgKIT repo root:

```bash
docker compose -f docker/docker-compose.yaml run --rm smoke-flashqla
docker compose -f docker/docker-compose.yaml run --rm parity-flashqla
docker compose -f docker/docker-compose.yaml run --rm profile-flashqla
docker compose -f docker/docker-compose.yaml run --rm shell-flashqla
```

Equivalent make targets are `make flashqla-smoke`, `make flashqla-parity`,
`make flashqla-profile`, and `make flashqla-shell`.

`smoke-flashqla` does not launch kernels. It reports CUDA limits, NVCC,
TileLang, FLA, FlashQLA, cache dirs, and the active FlashQLA chunk
architecture. On sm_121, the expected result after the Blackwell compatibility
refactor is `active_chunk_arch.name=blackwell_sm121`.

`parity-flashqla` launches fixed-length and packed-varlen GDN forward/backward
against FLA and FlashQLA. It is the correctness gate for the default
`BGKIT_GDN_BACKEND=flashqla` training path. The current sm_121 FlashQLA backend
is a compatibility path that delegates to FLA, so exact parity is expected.

`profile-flashqla` is a small development timing harness. It is intentionally
not a benchmark suite; it exists so sm_121 kernel changes have a stable compile
and timing entry point. While FlashQLA uses the compatibility backend, its
latency should be FLA-equivalent rather than H200-style FlashQLA speedup.

`shell-flashqla` starts a GPU shell with `/workspace/flashqla`,
`/workspace/fla`, `/workspace/flash-attention`, and `/workspace/bgkit/src` on
`PYTHONPATH`. Edit FlashQLA from the host checkout at `/home/werg/FlashQLA`.

## Container Contract

The image bakes in FlashQLA's JIT toolchain:

- `tilelang==0.1.8`
- `apache-tvm-ffi==0.1.9`
- CUDA/NVCC from the NGC PyTorch base image

The source trees remain bind-mounted:

- `/home/werg/bgkit` -> `/workspace/bgkit`
- `/home/werg/FlashQLA` -> `/workspace/flashqla`
- `/home/werg/flash-linear-attention` -> `/workspace/fla`
- `/home/werg/flash-attention` -> `/workspace/flash-attention`

TileLang and TVM caches are mounted under `/workspace/.cache/tilelang` and
`/workspace/.cache/tvm`, so JIT artifacts do not depend on a throwaway container
layer.

## Backend Policy

`BGKIT_GDN_BACKEND=flashqla` is the production default. On sm_121 today it
selects FlashQLA's Blackwell compatibility backend, which delegates to FLA while
native TileLang kernels are still being re-tiled. It is fail-fast: if FlashQLA
cannot be imported, training should stop rather than silently using another
kernel.

`BGKIT_GDN_BACKEND=auto` may be used for exploratory runs. It falls back to FLA
when FlashQLA is unavailable.

`BGKIT_GDN_BACKEND=fla` is the explicit escape hatch when we need to isolate a
FlashQLA regression.

## sm_121 Constraints

The current GB10 runtime reports compute capability 12.1. NVIDIA's Blackwell
limits relevant to this work are materially different from Hopper:

- maximum shared memory per thread block: about 99 KiB with opt-in
- shared memory per SM: about 100 KiB
- registers per SM: 64 Ki
- max resident threads per SM: 1536
- max resident blocks per SM: 24

FlashQLA's Hopper kernels exceed the 99 KiB/block sm_121 budget in the
forward, state-preparation, and backward kernels. FlashQLA now imports on
sm_121 through a compatibility backend, but native TileLang speedups still
require new Blackwell schedules under this shared-memory budget.

## Expected Migration Sequence

1. Rebuild the BgKIT training image.
2. Run `smoke-flashqla` and confirm CUDA/TileLang/FLA are healthy.
3. Run `parity-flashqla`; compatibility backend parity should pass exactly.
4. Run `profile-flashqla`; compatibility backend speed should be FLA-equivalent.
5. Refactor FlashQLA kernels under `/home/werg/FlashQLA` into native
   Blackwell/sm_121 schedules that keep per-block shared memory under 99 KiB.
6. Repeat parity/profile after each native-kernel replacement.
7. Keep `BGKIT_GDN_BACKEND=flashqla` as the default for BgKIT training, and use
   `BGKIT_GDN_BACKEND=fla` only as a temporary regression-isolation escape
   hatch.
