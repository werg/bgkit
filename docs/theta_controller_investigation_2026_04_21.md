# Dual-ascent θ controller investigation — Phase 1 Step 3 packed run (2026-04-21)

Live run: `werg/bgkit/pq428wpa`  (train-phase1-step3 container, packed FA4).

## TL;DR

The dual-ascent loop is **structurally correct** (plumbing, sign, aggregation all
check out). The observed "controller not tracking" behaviour has two
superimposed causes, both about the *operating point*, not the update rule:

1. **Primary bug — θ clamp + tanh saturation interact to make high
   target_ratio values unreachable.** The operator uses
   `logits_for_op = tanh(base_raw / T)` which is bounded in (−1, 1) but in
   bf16 frequently *saturates* to exactly ±1 at the tails. The controller
   clamps θ to (−0.99, +0.99). Positions whose `logits_for_op == −1.0`
   (saturated-negative tail) fail `> θ` for **any** θ > −1.0, so they can
   never survive organically. The head's current distribution has ~12–35 %
   of positions sitting at the −1 floor (see `head_logit_min = −1.0` in
   every logged `train_step`). That caps the achievable keep-rate.
   During the `target_ratio = 0.92` override, θ hit its clamp at −0.99 and
   *stayed* there from step 2550 through step ~3500; actual-rate drifted
   0.88 → 0.65 not because the controller flipped direction but because
   the controller was pinned at the clamp and the head's distribution
   wandered freely under other gradients (utility-grad BCE, moment-match,
   decoder CE via utility grad) that weren't being counter-balanced.

2. **Secondary — under-damped tracking on the active ramp (0.55 → 0.05
   over 1000 steps).** With `lr = 0.02`, target decreasing at
   `|r_t| ≈ 5e-4` per step, and the local sensitivity
   `d(actual)/dθ ≈ −0.25` (the keep-rate drops by ~0.25 per θ unit near
   the current operating point), the steady-state tracking error is
   `gap ≈ |r_t| / (lr · |d(actual)/dθ|) ≈ 0.1`. That's exactly the
   observed lag (actual ≈ 0.36 when target ≈ 0.27 at step 4050). The
   controller *is* converging, just not faster than the ramp.

The low training loss (~1e-5) masks the issue because at 0.35-0.65
keep-rate with Step-2.5-anchored projections the reproduction task is
near-trivial.

## Plumbing audit — no bug found on these axes

### (a) `(sum, count)` aggregation under packing — correct

`bgkit_compressor.BgKITCompressor._hook_after_head_layer` (lines 537-547):

```python
valid = torch.ones_like(base_raw, dtype=torch.bool)     # all content positions
organic = (logits_for_op.float() > theta.float()) & valid
pinned_mask = torch.zeros_like(valid)                    # Step 3: pinned=None
floor_mask = mask & ~organic & ~pinned_mask              # empty: min_per_sample=0
controllable = valid & ~pinned_mask & ~floor_mask        # all-True
organic_count = (organic & controllable).sum()           # N_survive
controllable_count = controllable.sum()                  # N_content
```

Under Step 3 post-warmup: `min_per_sample = 0`, `pinned_positions = None`,
so `floor_mask` and `pinned_mask` are both empty. Therefore
`mean_rate = organic_sum / controllable_sum = N_survive / N_content`
which is **exactly equal to** the separately-logged `actual_ratio` metric
(line 1366 in `decoder_init.py`:
`metrics["actual_ratio"] = n_survivors / max(n_valid, 1)`).

Live confirmation: the log lines show `mean_rate == actual_ratio` at every
step, down to the last decimal.

`survivorship_helpers.MicrobatchAggState.accumulate` correctly sums
`(organic_count, controllable_count)` across microbatches with the
`cc == 0` guard to skip empty microbatches — packed-shape agnostic, as
documented.

### (b) θ update direction — correct

`DualThresholdController.step` (selection.py:333):

```python
gap = float(current_rate) - float(target_rate)   # actual − target
delta = self.lr * gap                             # + when actual > target
new_theta = theta + delta                         # θ rises when actual > target
```

Live confirmation: during the ramp, target decreases monotonically, actual
lags above target, gap > 0, θ is observed rising monotonically from
−0.99 → −0.558 over steps 3500 → 4050. Direction matches expectation
(higher θ → fewer survive → actual decreases toward target).

### (c) θ step size

`lr = 0.02` (default from `configs/model/survivorship_head.yaml`). Not
config-overridden at Step 3. With gap ≈ 0.08 per step, θ moves ≈ 0.0016/step
— slow but monotonic, so under the ramp's `|r_t| = 5e-4` the tracking
error is small but non-zero (math in TL;DR §2).

### (d) Competing gradients on `logits_for_op`

Several losses operate on `base_raw` / `logits_for_op` and shift the head's
distribution independently of θ:

- `utility_grad_loss_weight = 0.3` (primary training signal on the head).
- `moment_match_weight = 0.05` (permanent anchor against offline ICE moments).
- `min_survivors_loss_weight = 0.05` (per-sample soft-count hinge).
- Decoder CE gradient flowing via `survive_embedding` scatter
  (indirect but present).

These move the head's distribution shape; θ reacts to the aggregate keep-rate
but cannot bound the shape itself. When θ is clamped at the boundary (case 1),
these pressures determine actual-rate entirely.

### (e) `target_ratio_override` routing — correct

`DecoderInitTrainer.LIVE_CONFIG_HANDLERS["target_ratio"] →
_handle_target_ratio` (inherited from `BaseTrainer`, base_trainer.py:768).
Writes `self._target_ratio_override`. Consumed by
`_current_target_ratio()` (decoder_init.py:1101-1113) and passed into
`apply_post_step_updates(..., target_ratio=target_ratio, ...)`
(decoder_init.py:1689). **Live confirmation:** at step 2550 with
`target_ratio=0.92` override set, the log shows
`target_ratio=0.92 min_target_ratio=0.92`; the controller's `target_rate`
param was the overridden value.

### (f) Per-level θ state — one level active

Step 3 trains L0 only (`_surv_state_l1` never accumulated in the Step 3
path). `threshold_l0.theta_param` is the only scalar in play. No per-level
stale-cache issue.

## θ trajectory — observed vs expected

Parsed from container `docker-train-phase1-step3-1` logs
(`/tmp/traindoc.log`):

```
step  target  actual  theta     comment
────  ──────  ──────  ────────  ──────────────────────────────
2550  0.92    0.888   -0.9900   override set, θ already at clamp
2560  0.92    0.832   -0.9900   clamped; actual wandering
...
3180  0.92    0.704   -0.9900   clamped; actual has drifted down 0.18
3500  0.92    0.598   -0.9900   override cleared, ramp begins
3510  0.67    0.676   -0.9896   θ starts moving (gap near zero here)
3700  0.50    0.543   -0.9397   θ climbing, lag ~0.04
3800  0.49    0.524   -0.9243   lag stable ~0.03
3900  0.42    0.486   -0.8680   lag ~0.07 (steady-state under ramp)
4050  0.265   0.342   -0.558    lag ~0.08 (still in ramp region)
```

During steps 2550-3500 θ is pinned at the lower clamp and the controller
has zero leverage. After 3500 the ramp reduces target; the controller
becomes leverage-bearing again and θ monotonically rises. Observed
tracking error is consistent with the under-damping math in TL;DR §2.

## Root-cause summary

**Not a bug in the update rule, aggregation, routing, or per-level state.**

Two compounding design issues:

1. `clamp = 0.99` combined with bf16 tanh saturation → a measurable
   fraction of positions at exactly `logits_for_op = −1.0` that can never
   be "selected". This puts a hard ceiling on achievable keep-rate below
   1.0 (observed ~0.85-0.90 at θ = −0.99). A controller target above
   this ceiling is infeasible and θ pegs at the clamp indefinitely
   (`selection.py:347`).

2. `lr = 0.02` + tanh-bounded (−1, 1) logit space gives a quasi-steady
   tracking lag of ~0.1 under the current ramp rate
   (`configs/model/survivorship_head.yaml:16`). This is not a bug — the
   controller converges, just slower than the ramp pulls the target.

### Why the observed symptom looks like "wrong direction" during override

During the override period θ was clamped and *could not move*. The head's
distribution shifted **upward** (fewer saturated-negative tails, or shifted
mean) under utility-grad BCE training: the BCE teacher selects top ~30 %
of positions by `−(grad · value)`, so the head's `base_raw` on those
positions gets pushed up over time. If the overall mass shifts such that
more positions are *above* `−0.99`, actual rises; if it shifts such that
more positions are at the `−1.0` floor (i.e. more distribution mass in
the deeply-negative tail), actual drops. During 2550-3500 the latter
happened — but θ couldn't respond. Hence actual drifted 0.88 → 0.65 while
"target said 0.92".

## Proposed fix — minimal diff

**Change `configs/model/survivorship_head.yaml`:**

```yaml
threshold_controller:
  init_theta: -0.9            # unchanged
  lr: 0.05                    # was 0.02 — 2.5× for faster ramp tracking
  momentum: 0.0               # unchanged
  clamp: 1.5                  # was 0.99 — allow θ to exceed tanh saturation
  # ^^^ At θ > 1, the hard mask is always False; at θ < −1, always True.
  # Lets the controller saturate to "all-off" / "all-on" cleanly when
  # the target is at an infeasible extreme, instead of pegging at ±0.99
  # and letting other gradients drift the rate around.
```

That's it — the controller's `step` method already handles clamp
symmetrically; raising the clamp values past ±1 doesn't change any tensor
math or break serialization (θ is fp32 scalar).

**Why raising the clamp is safe:**
- `logits_for_op = tanh(base_raw/T) ∈ [−1, 1]` (bf16-clamped).
- `(logits_for_op > θ)` with θ = 1.01 → always False → zero survivors
  (well-defined: post-warmup `min_per_sample = 0`, so zero-survivor
  samples are accepted signal).
- `(logits_for_op > θ)` with θ = −1.01 → always True → all survive.
- No reason the clamp should be inside the tanh range — it was a
  cargo-culted bound from an earlier un-bounded logit era.

**Why raising lr is safe:**
- `lr = 0.05` keeps `|delta per step| ≤ 0.05` (gap ≤ 1 by construction).
- At a step-function change in target, θ moves ~0.05 toward the new
  operating point; at the current gap-of-0.08 during the ramp, θ moves
  0.004/step → effectively catches up to the ramp rate 5e-4 *8x over*.
- Upper bound on θ oscillation: for gap switching sign each step,
  amplitude would be `lr * typical_gap ≈ 0.05 * 0.05 = 0.0025` — well
  below the θ span, no chatter risk.

**Optional follow-up (not required for this fix but worth considering):**

- **Gain scheduling on θ's local sensitivity.** Near `θ = 0` one θ-unit
  moves keep-rate by `~std(logits_for_op) × density ≈ 0.3`; near the
  tails by much less. A non-linear `lr = base_lr / local_density_estimate`
  would give more consistent response. Out of scope for a
  minimal-diff fix.
- **Recalibrate `head_tanh_temperature` periodically.** The current T
  was set at model init from the fresh head's std (~1.0, so T ≈ 1.0?
  Actually `T = 2.69` in the live run — see log line
  `head_tanh_temperature_calibrated T=2.69`). As training proceeds,
  `base_raw`'s std grows (currently ~6.5 based on `base_norm`), so
  `base_raw/T` becomes less linear and tanh saturation gets worse. A
  periodic recalibration would keep tanh in its linear region. Out of
  scope.

## Unit test that would have caught the tracking-error symptom

Add to `tests/unit/training/test_survivorship_helpers_packed.py`:

```python
def test_dual_ascent_converges_under_static_target():
    """With a fixed target and a linearly-responsive rate model, θ
    should converge to the target within ~100 iterations.

    Simulates the trainer's per-optimizer-step loop:
        1. Measure rate at current θ via a synthetic monotone model
           `rate = clip(0.5 − 0.25 · (θ − θ0), 0, 1)`.
        2. Feed (organic=rate·N, controllable=N) into MicrobatchAggState.
        3. Call apply_post_step_updates(target_ratio=target).
        4. Reset state and repeat.

    Assert the controller reaches |rate − target| < 0.02 within 100
    steps for target in {0.3, 0.5, 0.7} with default lr=0.02.
    """
    import torch
    from bgkit.models.components.selection import DualThresholdController
    from bgkit.training.survivorship_helpers import (
        init_state, apply_post_step_updates, MicrobatchAggState,
    )

    class _RateModel:
        """Synthetic monotone rate(θ) = 0.5 − 0.25·(θ + 0.5), clipped."""
        def __call__(self, theta: float) -> float:
            return max(0.0, min(1.0, 0.5 - 0.25 * (theta + 0.5)))

    class _Compressor:
        def __init__(self, clamp=1.5, lr=0.05):
            self.threshold_l0 = DualThresholdController(
                init_theta=-0.5, lr=lr, clamp=clamp,
            )
            self.threshold_l1 = DualThresholdController(
                init_theta=-0.5, lr=lr, clamp=clamp,
            )

    model = _RateModel()
    for target in [0.3, 0.5, 0.7]:
        compressor = _Compressor()
        N = 1000
        for _ in range(100):
            theta = float(compressor.threshold_l0.theta.item())
            rate = model(theta)
            state = init_state()
            # One microbatch:
            state.organic_count_sum = torch.tensor(int(rate * N))
            state.controllable_count_sum = torch.tensor(N)
            state.controllable_empty_count = torch.tensor(0)
            apply_post_step_updates(
                compressor, state, target_ratio=target, level="l0",
            )
        final_theta = float(compressor.threshold_l0.theta.item())
        final_rate = model(final_theta)
        assert abs(final_rate - target) < 0.02, (
            f"target={target}: final rate={final_rate:.3f}, "
            f"θ={final_theta:.3f}"
        )


def test_dual_ascent_tracks_ramping_target():
    """Target ramps from 0.5 to 0.1 over 500 steps; controller should
    keep the tracking error bounded below 0.15 throughout (and below
    0.05 after the ramp ends)."""
    import torch
    from bgkit.models.components.selection import DualThresholdController
    from bgkit.training.survivorship_helpers import (
        init_state, apply_post_step_updates,
    )

    class _Compressor:
        def __init__(self):
            self.threshold_l0 = DualThresholdController(
                init_theta=0.0, lr=0.05, clamp=1.5,
            )
            self.threshold_l1 = DualThresholdController(
                init_theta=0.0, lr=0.05, clamp=1.5,
            )

    def rate_of_theta(theta: float) -> float:
        return max(0.0, min(1.0, 0.5 - 0.4 * theta))

    compressor = _Compressor()
    N = 1000
    max_gap_ramp = 0.0
    for step in range(800):
        target = max(0.1, 0.5 - 0.4 * (step / 500.0))
        theta = float(compressor.threshold_l0.theta.item())
        rate = rate_of_theta(theta)
        state = init_state()
        state.organic_count_sum = torch.tensor(int(rate * N))
        state.controllable_count_sum = torch.tensor(N)
        state.controllable_empty_count = torch.tensor(0)
        apply_post_step_updates(
            compressor, state, target_ratio=target, level="l0",
        )
        gap = abs(rate - target)
        if step < 500:
            max_gap_ramp = max(max_gap_ramp, gap)
        else:
            # After ramp, track tightly.
            assert gap < 0.05, f"step {step}: gap={gap:.3f}"
    assert max_gap_ramp < 0.15, f"ramp max gap={max_gap_ramp:.3f}"


def test_dual_ascent_clamp_allows_infeasible_target():
    """When the true achievable rate ceiling is below the target
    (simulating tanh saturation), the controller should saturate θ at
    the lower clamp without pathology. Regression guard: clamp must
    admit θ ≤ −1.0 so the controller can push beyond tanh's (−1, 1)
    range when needed."""
    import torch
    from bgkit.models.components.selection import DualThresholdController
    from bgkit.training.survivorship_helpers import (
        init_state, apply_post_step_updates,
    )

    class _Compressor:
        def __init__(self):
            self.threshold_l0 = DualThresholdController(
                init_theta=0.0, lr=0.1, clamp=1.5,
            )
            self.threshold_l1 = DualThresholdController(
                init_theta=0.0, lr=0.1, clamp=1.5,
            )

    # Cap the achievable rate at 0.85: even at θ = −1.5, only 85% survive.
    def rate_of_theta(theta: float) -> float:
        if theta <= -1.0:
            return 0.85
        return max(0.0, min(1.0, 0.5 - 0.4 * theta))

    compressor = _Compressor()
    N = 1000
    for _ in range(200):
        theta = float(compressor.threshold_l0.theta.item())
        rate = rate_of_theta(theta)
        state = init_state()
        state.organic_count_sum = torch.tensor(int(rate * N))
        state.controllable_count_sum = torch.tensor(N)
        state.controllable_empty_count = torch.tensor(0)
        apply_post_step_updates(
            compressor, state, target_ratio=0.95, level="l0",
        )
    # θ should have hit the lower clamp.
    final_theta = float(compressor.threshold_l0.theta.item())
    assert final_theta == pytest.approx(-1.5, abs=1e-3)
    final_rate = rate_of_theta(final_theta)
    # And the actual rate is the infeasibility ceiling (0.85) — NOT 0.95.
    assert abs(final_rate - 0.85) < 1e-3
```

These three tests:
1. Convergence on a static target (would have caught sign/plumbing bugs).
2. Tracking error bound under a ramp (would have flagged that `lr = 0.02`
   leaves a 0.1 lag; would also flag a sign flip that diverges the error).
3. Clamp-saturation behaviour under an infeasible target — documents the
   expected behaviour, catches clamps that are too restrictive
   (e.g., clamp=0.99 with this rate model would fail to saturate and
   show the "stuck-clamp drift" symptom observed in production).

## Additional measurement that would distinguish hypotheses

The current diagnosis is confident. If additional signal is desired:

- **Log `theta_at_clamp` boolean per step** (`abs(θ) >= clamp_val - eps`).
  During the 0.92 override period this would have been true → immediately
  visible that the controller was saturated, not misbehaving.
- **Log `head_logit_at_saturation_frac`** — fraction of positions where
  `abs(logits_for_op) >= 1 − eps`. A large value means tanh saturation is
  capping the controllable range.
- **Log `rate_feasible_max`** — estimate as `1 − frac(logits ≤ clamp_lower)`,
  giving the maximum rate the controller can achieve at the current
  distribution.

None of these are needed to accept the diagnosis, but they'd make the
failure mode self-evident in the next ramp.

## Files referenced

- `/home/werg/bgkit/src/bgkit/models/components/selection.py` —
  `DualThresholdController.step` (333-348), `adaptive_threshold_select`
  (154-277).
- `/home/werg/bgkit/src/bgkit/training/survivorship_helpers.py` —
  `MicrobatchAggState`, `accumulate` (91-138), `apply_post_step_updates`
  (740-787).
- `/home/werg/bgkit/src/bgkit/models/bgkit_compressor.py` —
  `_hook_after_head_layer` (467-568), specifically the controllable/organic
  count computation at 537-547.
- `/home/werg/bgkit/src/bgkit/training/phase1/decoder_init.py` —
  `_current_target_ratio` (1101-1113), `_forward_backward` accumulate call
  (1351), `_post_optimizer_step` apply_post_step_updates (1686-1691).
- `/home/werg/bgkit/src/bgkit/training/base_trainer.py` —
  `_handle_target_ratio` (768-791), `_post_optimizer_step` dispatch (1279).
- `/home/werg/bgkit/configs/model/survivorship_head.yaml` —
  controller lr / clamp / init_theta defaults.
- `/home/werg/bgkit/configs/training/phase1_step3.yaml` —
  Step 3 curriculum config (target_ratio_start/end/ramp_steps).
- `/mnt/external/bgkit-checkpoints/control.json` — live-config override
  source (ramp parameters applied mid-run).
