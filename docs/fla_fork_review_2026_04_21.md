# flash-linear-attention fork review — 2026-04-21

Investigation of whether the local fla fork at `/home/werg/flash-linear-attention/`
(branch `blackwell-sm121-compat`) is still needed given stock `fla==0.4.2` +
Triton 3.6.0 in the NGC 26.03 training container.

## Summary

**Decision: (A) with light touch-ups.** The fork's commit `f11bc2f`
(torch fallbacks for three Triton kernels claimed broken on sm_121) is
**obsolete**. Stock `fla 0.4.2` + Triton 3.6.0 in NGC 26.03 runs all three
kernels correctly on DGX Spark (GB10 / sm_121). The fork is not installed
or bind-mounted anywhere in the current pipeline — the container already
installs PyPI `fla 0.4.2` via the `[gpu]` extra.

- Archive the fork branch (keep it on disk, don't delete) for historical
  provenance. It still contains uncommitted work-in-progress torch
  fallbacks for `chunk_gated_delta_rule_fwd_h` and `recompute_w_u_fwd`;
  those are also not needed with current Triton 3.6.
- Keep all three bgkit-side patches. One is load-bearing
  (`deltanet_patch.py` gate clamp); two are defensive and cheap.
- Tighten Dockerfile comments to describe the current reality (sm_121 vs
  sm_100 Blackwell) and why stock fla works.

## f11bc2f kernel-replacement audit

`f11bc2f "Replace broken Triton kernels with torch fallbacks for Blackwell sm_121"`
replaced three call sites with pure-torch implementations.

| Kernel (file) | Claimed symptom | State today (fla 0.4.2 + Triton 3.6 + sm_121) |
|---|---|---|
| `chunk_local_cumsum_scalar_kernel` (`fla/ops/utils/cumsum.py`) | `tl.cumsum + tl.sum` codegen bug — `Triton Error [CUDA]: invalid argument`. Cited triton-lang/triton#3017 and fla#734. | **Works.** Tested in running container with `g` shape `(1,128,4)` — returns correct output. |
| `chunk_scaled_dot_kkt_fwd_kernel` (`fla/ops/common/chunk_scaled_dot_kkt.py`) | `tl.dot(k, tl.trans(k))` codegen bug. | **Works.** Full fwd OK on sm_121 with `bf16` k/beta and `fp32` g. |
| `solve_tril` merge kernels (`fla/ops/utils/solve_tril.py`) | Similar codegen issues. | **Works.** Tested on lower-triangular `(2, 4, 64, 64)` input, returns correct shape. |

End-to-end, `chunk_gated_delta_rule(q, k, v, g, beta)` fwd+bwd passes
without error on sm_121 under both fixed-length and cu_seqlens paths in
the running container. TMA forward also passes when
`FLA_USE_TMA=1` is set explicitly.

**Root cause of resolution:** NGC 26.03 ships **Triton 3.6.0**, which
fixes `triton-lang/triton#3017` (the `tl.cumsum+tl.sum` codegen bug) and
the `tl.dot(x, tl.trans(x))` codegen bug. These bugs were real when the
fork was created against an older Triton (likely 3.2–3.4 based on
`TRITON_ABOVE_3_4_0` / `TRITON_ABOVE_3_5_1` guards in fla's utils).

## bgkit-side patches — audit

### `src/bgkit/utils/deltanet_patch.py` — KEEP

Two concerns:

1. **Gate clamp (`g >= -1.3`)**: still required. Reproduced backward/forward
   NaN in the container with "mixed" gates (a few heads at `-5`/`-6`, others
   at normal magnitudes) — a scenario that matches pretrained Qwen3.5-0.8B's
   extreme-decay heads. Upstream fla has **no safeguard** for this —
   `chunk_gated_delta_rule` still uses `exp(g_cum[i] - g_cum[j])` without
   saturating, and the relevant issues (fla#389, fla#104, unslothai/unsloth#3155)
   are still open. The clamp costs near-zero and only changes
   semantics at gate magnitudes where the state is forgotten within a chunk
   anyway.

2. **cu_seqlens varlen path (Wave 1.3)**: still required. `chunk_gated_delta_rule`
   accepts `cu_seqlens` upstream but HF's `Qwen3_5GatedDeltaNet.forward` doesn't
   thread it through. The patch intercepts `forward(cu_seqlens=...)` and
   injects it into the call.

### `src/bgkit/utils/triton_patch.py` — KEEP

Two sub-patches, both scoped to sm_121 via `get_device_capability() == (12,1)`:

1. **Autotuner `_bench` RuntimeError catcher**: upstream only catches
   `OutOfResources`/`PTXASError`; on Blackwell / sm_121 some configs
   raise `CUDA error: invalid argument` at launch time, crashing the
   autotune run. This is scoped per-config and lets autotuning pick the
   working config. Cheap insurance; no cost when configs work.

2. **`CompiledKernel._init_handles` retry**: catches
   `"operation not permitted"` (cuModuleLoad inside a stream capture).
   Already complemented by `TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1` in
   compose, so in practice seldom fires, but retains a safety net if a
   user re-enables cudagraphs.

Neither patch has an upstream replacement. fla 0.4.2 doesn't contain a
per-config error catcher; it relies on config enumeration being
correct, which is false on sm_121.

### `FLA_USE_TMA=0` env var — KEEP (now matches upstream default)

Upstream fla 0.4.2 already defaults `IS_TMA_SUPPORTED` to `False` unless
`FLA_USE_TMA=1` explicitly. So setting `FLA_USE_TMA=0` is redundant but
defensive. Issue fla#609 described a logic inversion bug that would
enable TMA by default; that bug is gone from 0.4.2 source. I
verified in the running container: `IS_TMA_SUPPORTED = False` with no
env override.

Positive test: TMA forward actually **works** on sm_121 with Triton 3.6
when `FLA_USE_TMA=1` is set. So in principle we could enable it for
some speedup. Not doing so in this patch — deferred; low priority,
separate benchmarking pass.

## Post-0.4.2 upstream commits — relevance scan

63 commits on `upstream/main` since our fork base (`03a944d`). Filtered
for sm_121 / Blackwell / varlen / GDN relevance:

| SHA | Summary | Relevance to us |
|---|---|---|
| `8b05e2f` | Register default `global_scratch` allocator on Blackwell GPUs (`>= sm_100`). | **Not applicable.** sm_121 = capability (12, 1). `IS_NVIDIA_BLACKWELL = (capability[0] == 10)` in upstream — does **not** match sm_121. Our card never hit the NullAllocator crash and the fix wouldn't engage even if we took the commit. Upstream would need `>= 10` to cover sm_121; see "Would file upstream" below. |
| `02af88e` | Workaround pipeline issue in `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` on Blackwell (reduced num_stages). | Same gating issue as above — gated on `IS_NVIDIA_BLACKWELL` which is False for sm_121. Our running Qwen3.5 training does not show symptoms matching this workaround; probably the Triton 3.6 default stages work for us. |
| `27c2022` | Fix autotune hprams for Blackwell. | Same gating problem. |
| `1d4d88f` | Remove deprecated `head_first=` parameter from public ops. | **Breaking change for users of `head_first=True`.** bgkit does not pass `head_first` anywhere; verified via grep. Safe. |
| `6e039ee`, `04629e0`, `9b3c930` | cu_seqlens / varlen fixes in `fla/layers/attn.py`, `fla/ops/attn/parallel.py`, and other attention ops. | **Not applicable.** bgkit uses FA4 (`flash_attn.cute.flash_attn_varlen_func`) for attention, not fla's attention layer. The fixes are for fla-internal attention layers we don't touch. |
| `baf57d5` | Fuse kkt + solve_tril into one kernel. | Performance only; kkt/solve_tril already work for us. Would give a small speedup if we bumped fla. |
| `047b5df` | exp2 support across chunk kernels. | Performance only. |
| `12e2a70` | Fix `naive_chunk_linear_attn` cumsum dim (fixes #850). | Not used. |

**Conclusion: no upstream commit is load-bearing for our Phase 1 Step 3
training.** A fla bump would give us perf improvements (fused kkt,
exp2) but nothing correctness-critical.

## Would file upstream

`IS_NVIDIA_BLACKWELL` in `fla/utils.py` is defined as
`capability[0] == 10`. This excludes sm_121 (GB10 / Blackwell Ultra /
DGX Spark), which is also a Blackwell-class architecture. The
Blackwell-specific workarounds (`8b05e2f`, `02af88e`, `27c2022`)
therefore don't activate on sm_121 even though they likely should.

Suggested change: `capability[0] >= 10` (or `capability[0] in (10, 12)`).
The commit `8b05e2f` discussion on GitHub actually mentions "update to
`>= 10` for forward compatibility" — but the condition lives in the
BLACKWELL detection, not in the call site, and the definition was
pinned at `== 10`. This is probably an oversight.

Would-file: issue or PR `fla/utils.py: IS_NVIDIA_BLACKWELL should be
capability[0] in (10, 12) to cover sm_121 (DGX Spark)`.

Low urgency because (a) we're not hitting the symptoms those workarounds
address (at least on Triton 3.6), and (b) we can override locally if
needed. Document for later.

## Plan

1. **No code changes needed to start from.** Stock fla 0.4.2 in the
   container is correct for our workload.
2. **Documentation updates** in `docker/Dockerfile` and `CLAUDE.md` to
   make the decision discoverable to future agents.
3. **Archive fork branch**: leave on disk, do not delete — provides
   provenance and ready-to-reactivate state if Triton regresses.
4. **No bind-mount of the fork**, no entrypoint bootstrap for fla, no
   `pyproject.toml` changes.

## Verification

Tested in the live `docker-train-phase1-step3-1` container (which is
also running the phase1 step 3 training):

```text
fla 0.4.2
triton 3.6.0
cuda cap: (12, 1)
cuda name: NVIDIA GB10
IS_NVIDIA_BLACKWELL: False   # upstream misses us; this explains why we
IS_TMA_SUPPORTED: False      # don't need the Blackwell workarounds (Triton
                             # 3.6 emits good code without them).

chunk_local_cumsum_scalar OK
chunk_scaled_dot_kkt_fwd OK
solve_tril OK
chunk_gated_delta_rule fwd/bwd OK
chunk_gated_delta_rule varlen OK
chunk_gated_delta_rule TMA fwd/bwd OK (when FLA_USE_TMA=1)
```
