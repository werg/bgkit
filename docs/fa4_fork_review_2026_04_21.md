# FA4 fork review for the Phase 1 Step 3 wall-clock gap

Date: 2026-04-21
Scope: local FA4 fork at `~/flash-attention` (branch `sm120-fixes`), bgkit dispatch at
`src/bgkit/utils/attention_backend.py` and the packed layer rewrites in
`src/bgkit/models/bidirectional_qwen35.py` / `pruned_qwen35.py` / `decoder.py`.
Pairs with `docs/wall_clock_investigation_2026_04_21.md`.

## TL;DR

**The FA4 cute-DSL sm_120 kernels do not run at all on live training.** On sm_12x,
`flash_attn.cute.interface.flash_attn_varlen_func` hard-dispatches to
`flash_attn_varlen_sm12x_native` (interface.py:2010), which calls the
**FA2 C++ kernels** rebuilt against container libtorch by our entrypoint
bootstrap (`backend_kind="flash_attn"` path in `flash_attn/cute/native_sm12x.py`).
Our fork's `FlashAttentionForwardSm120` / `FlashAttentionBackwardSm120` subclasses,
the 99 KB SMEM can_implement checks, the GQA pack_gqa fix in `flash_fwd.py`,
the bwd AtomLayoutNdKV tile fix in `interface.py::_flash_attn_bwd`,
and the 2-CTA codegen regression workaround are all **unreachable from varlen
training**. They would only matter for padded non-varlen dispatch, which bgkit no
longer uses.

Confirmed from the running container (`docker exec docker-train-phase1-step3-1`):

```
PYTHONPATH=/workspace/checkpoints/.flash-attn-native/repo:/workspace/flash-attention:...
backend_kind = "flash_attn"
owned_backend_available = True
flash_attn_interface path = /workspace/checkpoints/.flash-attn-native/repo/flash_attn/flash_attn_interface.py
```

cuobjdump confirms `flash_attn_2_cuda.cpython-312-aarch64-linux-gnu.so` carries
sm_120 cubins for every generated kernel. These are the compiled FA2 SM80-style
kernels (MMA m16n8k16, cp.async) targeted at sm_120 — not FA4 cute DSL.

So:

- **Hypothesis #3 (FA4 cute-DSL per-shape JIT compile overhead): falsified for
  varlen training.** The cute DSL path is bypassed on sm_12x varlen. JIT compile
  caches in `flash_attn.cute.interface._flash_attn_fwd.compile_cache` never get
  touched during the step. FA2 dispatches statically at build time via
  `HEADDIM_SWITCH` / `BOOL_SWITCH(is_causal)` to a fixed cubin.
- **GQA repeat-interleave cost: NOT being paid in practice.** The integration
  memory note ("repeat-interleave K/V heads to match Q before dispatch, turning
  16:2 GQA into MHA") describes a historical workaround. In the current code the
  bgkit layer forwards pass Q (16 heads) and K/V (2 heads) directly to FA4's
  `flash_attn_varlen_func`, which routes to
  `flash_attn_varlen_sm12x_native`, which routes to the FA2 `varlen_fwd`
  C++ kernel — and FA2 has native GQA support. The model files contain NO
  `repeat_interleave` on the hot path (verified via grep in
  `src/bgkit/models/`). The backward path in
  `flash_attn.cute.interface._flash_attn_bwd:1301` force-disables pack_gqa but
  this branch is also unreachable at varlen dispatch.
- **cute-DSL tile/atom fixes for hd=256 sm_120 (can_implement, Q_in_regs, n_block
  shrink, AtomLayoutNdKV scaling): correct but inert at varlen training time.**
  They protect padded dispatch and unit tests; they cost us nothing and save us
  from a regression if padded dispatch is ever resurrected.

**Consequence for the wall-clock gap:** none of the kernel-level levers described
in the prompt's "Your investigation surface" apply. The FA4 fork is in reasonable
shape for the fact that our training is not actually using it.

## What IS running on sm_121, and where its perf comes from

Live dispatch at each attention call:

```
bgkit.models.*_qwen35::_packed_full_attention
  -> bgkit_flash_attention_4_forward  (src/bgkit/utils/attention_backend.py:80)
  -> flash_attn.cute.flash_attn_varlen_func (interface.py:1951)
  -> [arch//10 == 12 branch] flash_attn_varlen_sm12x_native  (interface.py:2011)
  -> FlashAttnVarlenSm12xNativeFunc.apply  (native_sm12x.py:378)
  -> flash_attn.flash_attn_interface._wrapped_flash_attn_varlen_forward
     (torch.ops.flash_attn._flash_attn_varlen_forward)
  -> flash_attn_2_cuda.varlen_fwd  (C++, sm_120 cubin)
```

Backward path is symmetric through `_wrapped_flash_attn_varlen_backward`.

Key observations on the FA2 kernel for our shape (head_dim=256, bf16, GQA 16:2,
sm_120 with 99 KB SMEM per block):

1. **Forward tile is suboptimal but correct.**
   `csrc/flash_attn/src/flash_fwd_launch_template.h::run_mha_fwd_hdim256`
   picks its tile purely on `max_smem_per_block`:
   - A100 (164 KB): uses `128×64, 8 warps` tile. SMEM = `2*256*(128+128) = 128 KB`.
   - H100 (228 KB): uses `64×64, 4 warps` tile (smaller tile, more CTAs/SM).
   - **sm_120 (99 KB): falls into the `64×64, 4 warps` branch** via
     `max_smem_per_sm < 4*256*(64+128) = 192 KB` (sm_120 is 100 KB, so the
     condition trivially holds). SMEM = `2*256*(64+128) = 96 KB`. Fits in 99 KB.
   The 64×64 tile is the H100 config at lower concurrency; it is NOT the
   A100 config. No middle-ground configuration (e.g. `128×32, 8 warps, 96 KB`)
   is registered in the launcher. This is a potential lever worth 5–15% on fwd.
2. **Backward tile is the more suboptimal case.**
   `csrc/flash_attn/src/flash_bwd_launch_template.h::run_mha_bwd_hdim256`
   for sm_120 falls into the "sm86/sm89" branch: `64×32, 8 warps,
   V_in_regs=true, Is_Q_in_regs=true, no double buffering, Is_dropout disabled`.
   This is half the N tile of the SM80/A100 path (`64×64, double-buffered`) and
   does not overlap global loads with compute. For a compute-bound kernel like
   Qwen3.5 decoder bwd (head_dim=256 is the MAC-heaviest case in FA2), this is a
   substantial hit. Plausibly 1.3–1.6× slower than SM80 at equivalent SMs.
3. **No native-sm_121 specialization.** sm_121 is treated as sm_120 in every
   codepath. GB10 actually has identical tensor-core throughput to sm_120
   (same MMA ISA), so this is probably fine — but we have not verified.
4. **FA4 cute-DSL SM100 variant exists but is for Blackwell datacenter.**
   `flash_fwd_sm100.py` with 2-CTA / TMA / CLC scheduler is only reachable if
   `_get_device_arch() // 10 in [10, 11]`. It's 500+ LOC of advanced tiling that
   sm_120 does not map onto cleanly (no 2-CTA TMA on GB10).

## Component-by-component assessment

### Tile layout (cute-DSL sm_120 subclass) — IRRELEVANT at varlen

`flash_fwd_sm120.py` / `flash_bwd_sm120.py` exist only to answer
`can_implement` for the cute-DSL compile-time planner, and the interface's
`if arch//10 == 12` branch in `_flash_attn_fwd` selects tiles that fit the 99 KB
budget (128×64 for hd≤64, 128×64 Q_in_regs for hd>128). **For varlen dispatch on
sm_12x this code is dead.** Leave it alone; it's correct and covers the padded
dispatch path that we currently don't use.

### Tile layout (FA2 C++ hd=256 sm_120) — material, hard to fix safely

`flash_fwd_launch_template.h` for hd=256 does not register a mid-sized tile
(e.g. `128×32, 8 warps, 96 KB`). Adding one would require:
- New kernel_traits entry.
- New cubin compiled into `flash_attn_2_cuda.so`, bumping bootstrap build time
  by ~3–5 minutes per focused-build tile.
- Re-benchmarking the dispatch heuristic (currently only checks SMEM; would need
  a sm_120-specific fast path).

**Out of scope for this 2-hour budget.** Recommend tracking as future work.

### GQA workaround — CORRECT, no cost paid in current code

The memory-described repeat-interleave is NOT active. The current encoder / decoder
forward (`src/bgkit/models/bidirectional_qwen35.py::_packed_full_attention`) passes
`q (N, 16, 256)` + `k/v (N, 2, 256)` directly to `bgkit_flash_attention_4_forward`,
which routes to FA2's `varlen_fwd` which supports native GQA at the kernel level
(the kernel reads each KV head qhead_per_kvhead=8 times). There is no 8×
KV-duplication in memory and no bandwidth penalty.

The FA4 cute-DSL path DOES force `pack_gqa=False` in bwd
(`interface.py:1301`), but again this branch is dead for varlen. The statement
in the integration memory should be considered **out of date** — updating below.

### JIT / compilation caching — IRRELEVANT at varlen

`flash_attn.cute.cache_utils.JITCache` / `JITPersistentCache` caches compiled
cute-DSL kernels. The `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED` env var (default
off) enables disk persistence. For our shape the compile key does NOT contain
`max_seqlen` or `batch_size` — it contains only dtype, head_dim, qhead_per_kvhead,
causal, pack_gqa, tile_m/n, arch, etc., all of which are fixed for our workload.
So even if varlen dispatched to cute DSL, there would be ~2 compile keys at most
(causal vs non-causal), not one per microbatch.

**But it doesn't matter because varlen bypasses cute-DSL entirely.**

### Scheduler / varlen overheads — not an issue

`FlashAttnVarlenSm12xNativeFunc.forward` applies a `(8 - head_dim % 8) % 8`
padding to make head_dim %8 == 0. For head_dim=256 the padding is 0, so it's a
no-op. The `.contiguous()` guard in `maybe_contiguous` is fine.

The FA2 varlen kernel itself has per-sequence tile scheduling in the CUDA
kernel; at our shapes (sum(L²) ~134M, max_seqlen up to 4096) the scheduler
picks reasonable grid sizes. No evidence of scheduler pathology.

### arch=Arch.sm_80 pinning — correct for the cute-DSL case, inert for varlen

Our pin `self.arch = Arch.sm_80` in `FlashAttentionForwardSm120.__init__` forces
the cute-DSL runtime to emit SM80 cp.async rather than SM90 TMA. For padded
dispatch this is the right call — we don't have SM90 TMA hardware on GB10. For
varlen dispatch this is irrelevant (FA2 C++ path).

### Partial RoPE interaction — minor but not hot

Qwen3.5 has partial_rotary_factor=0.25, so only head_dim_rotary=64 of head_dim=256
gets RoPE. The cost is a few elementwise mults + two cats in `apply_rotary_pos_emb`
(`transformers/models/qwen3_5/modeling_qwen3_5.py`). Our packed caller performs a
transpose+squeeze dance (`bidirectional_qwen35.py::_packed_full_attention:168-173`)
to adapt the `(N, H, Dh)` packed tensors to the `(B, H, L, Dh)` layout HF's
`apply_rotary_pos_emb` expects. This creates a full q+k copy per layer (~60 MB
per layer at our shapes × 6 full-attn layers × 2 encoder+decoder ≈ 720 MB of
memory traffic per step). On unified memory this might not register because
host DRAM is the backing store for CUDA anyway, but it's untidy.

An alternative is to squeeze cos/sin once to `(N, 1, Dr)` and apply the rotary
math directly in packed layout without the transpose. This is a modest code
change (~30 LOC of hand-rolled RoPE), breaking correctness parity with HF by
implementation identity. **Deferring** — needs a parity fixture comparison and
is not the dominant factor.

### `_sm12x_native` scaffold — no point pursuing

The `flash_attn/cute/native_sm12x/` directory has a `varlen_stub.cu` that
`throw_not_implemented`s. Building it out as a proper kernel would duplicate
the FA2 C++ varlen path we're already using (rebuilt against container libtorch
by the entrypoint bootstrap). No clear win.

## Where the 2.5× wall-clock gap actually lives

Per `docs/wall_clock_investigation_2026_04_21.md`: NOT in the attention kernel.
The attention FLOPs per step are actually **half** the baseline (12.7 G vs 24.6 G).
The dominant factors are:

1. **Quadratic budget inflates per-microbatch worst case** — a single
   L=4096 sample consumes 1/8 of the 134M L²-budget, producing 8-sample
   microbatches whose attention cost equals 23-sample average microbatches.
   → Config-only fix: cut `max_batch_tokens` and/or cap `content_max_seq_len`.
2. **Decoder seq = ~2× content seq** — prefix+survivors+suffix roughly doubles
   the decoder sequence length vs content. Quadratic budget is effectively 4×
   looser on decoder.
3. **Muon optimizer cost now paid every step** (baseline had grad_accum=2).
4. **Allocator pressure at 40 GB peak** on unified memory — suggestive.

None of these are FA2/FA4 kernel issues.

## Prioritized remediation — what's actually worth doing

### Tier 1 (zero-risk, this sweep)

1. **Memoize `require_sm12x_owned_backend()`** — each attention layer calls it on
   the hot path. The unoptimized version does `torch.cuda.get_device_capability()`
   (~8 µs/call) + `importlib` lookup on every FA forward; at 48 attention layers
   per step × ~400 steps that's ~150 ms/run, negligible but trivially avoidable.
   Now an atomic bool check after first success.
   **Implemented** in this sweep (`src/bgkit/utils/attention_backend.py`).
2. **Correct the integration memory** — remove the out-of-date
   "repeat-interleave K/V heads to match Q" statement from the FA4 integration
   memory. It is not what the current code does.
   **Implemented** in this sweep.

### Tier 2 (out of scope but documented)

3. **FA2 hd=256 sm_120 forward tile retune** — investigate `128×32, 8 warps,
   96 KB` as an alternative to the H100-style `64×64, 4 warps`. Estimated
   10–20% forward speedup at hd=256. **Blocked by:** new cubin (~3–5 min
   bootstrap build), shape-specific heuristic, microbench harness. Defer to
   Wave 5.
4. **FA2 hd=256 sm_120 backward tile retune** — investigate `64×64, 8 warps,
   double-buffered` if SMEM can be squeezed (tight: need to remove V_in_regs).
   Potentially 20–40% bwd speedup. **Blocked by:** same + correctness risk
   (the current `V_in_regs=true` is there specifically because smem is tight).
5. **Partial RoPE in packed layout** — avoid the `transpose+squeeze+contiguous`
   per-layer dance. 30 LOC change in `bidirectional_qwen35.py`, parity fixture
   comparison required. Estimated 1–3% wall-clock. **Out of scope** — worth
   doing when rewriting the RoPE application for Phase 2 anyway.
6. **Kineto trace of one live microbatch** — per the investigation, needed to
   partition the 14 s profiler microbatch across FA2 fwd / FA2 bwd / MLP /
   norm / RoPE / DeltaNet. Without it we are guessing at relative kernel costs.
   Requires one low-contention run with `torch.profiler.profile(...)` on 3
   microbatches. **Do this first if the config-only fix doesn't close the gap.**

### Tier 3 (long-term)

7. **`_sm12x_native` C++ extension** — the scaffold. Not worth building; the
   current FA2 bootstrap is functionally equivalent.
8. **Bring the cute-DSL sm_120 kernel up to varlen parity** — would let us
   benefit from CLC scheduler, 2-CTA, TMA epilogue. But sm_120 has no 2-CTA and
   no TMA, so the parity would bring nothing. Don't do this.

## What is explicitly NOT broken in the fork

- `FlashAttentionForwardSm120.can_implement` — correct for 99 KB SMEM budget.
- `interface.py::_flash_attn_fwd` sm_120 branch — tile sizing matches can_implement.
- `interface.py::_flash_attn_bwd` sm_120 branch — head-dim aware n_block_size
  shrink with AtomLayoutNdKV scaling, needed for correct dK/dV at hd≥192.
- `utils.atomic_add_fp32` signature-switch shim for libs-base vs libs-cu13.
- `FlashAttentionForwardSm80.__call__` pack_gqa_layout apply on Q/O/LSE when
  pack_gqa=True — fixed the earlier MLIR "unable to compute crd2idx" crash.
- Entrypoint bootstrap — correctly installs an FA2 .so with sm_120 cubins and
  prepends to PYTHONPATH.

## Concrete deliverables in this sweep

- `src/bgkit/utils/attention_backend.py`: memoize owned-backend check.
- `tests/unit/test_attention_backend.py`: autouse fixture to reset the memo
  between tests.
- `docs/fa4_fork_review_2026_04_21.md`: this document.
- `~/.claude/projects/-home-werg-bgkit/memory/project_fa4_integration.md`:
  correction (see below).

## Answer to the prompt's "is the FA4 fork actually in reasonable shape"

**Yes.** The cute-DSL sm_120 subclass + interface branch are all correct, the
tile/atom fixes are load-bearing for padded dispatch and unit tests, and the
atomic_add_fp32 shim is the right thing. The one real performance lever at the
kernel level is the FA2 hd=256 sm_120 tile layout — and that's not the FA4 fork,
that's legacy FA2 C++ in our repo's `csrc/`. Pursuing it is a separate project
with its own build-and-validate cycle.

For the Phase 1 Step 3 live wall-clock gap specifically, the FA4 fork is not
the dominant factor. The leverage is in config (`max_batch_tokens` and
`content_max_seq_len`) and in visibility (add kineto trace and per-step
tokens/samples/max-L metrics). Those are tracked in the wall-clock
investigation doc.

## Expected wall-clock impact of this sweep

**~0 directly.** The owned-backend memoization saves ~150 ms/run (not /step).
The real wins remain the config changes (cut max_batch_tokens from 65536 to
32768 + grad_accum=2) called out in the wall-clock investigation, which are
expected to cut step time by ~40%.

## Blocked / future work

- **FA2 C++ hd=256 tile retune for sm_120** — blocked by correctness risk,
  bootstrap build cost, need for a microbench harness. Track separately.
- **Live kineto trace** — needed to definitively attribute the per-stage wall
  clock. Do this when the config-only lever is exhausted.
- **Cute-DSL varlen for sm_12x** — would require implementing a new sm_120 varlen
  interface path in `flash_attn.cute.interface`. Not clearly better than the FA2
  C++ path; would duplicate effort. Not worth pursuing until we have evidence
  the FA2 path is bottlenecked at a level the cute DSL can outrun.
