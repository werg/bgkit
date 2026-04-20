# Padded-era training baselines

Captured pre-FA4-packed-attention migration for Wave 4.5 live-validation comparison.

## phase1_step3_04_20_baseline.json

Source: wandb run `werg/bgkit/6wznpmwv` (04-20, crashed at step 2599 during gen_eval swap cascade — see plan `~/.claude/plans/cozy-waddling-graham.md` §Context).

Baseline headline numbers:

| Metric | Value | Wave 4.5 target |
|---|---|---|
| step wall-clock (s/step) | 8.94 | ≤ 3.0 (≥ 3× speedup from packing + acc-step reduction) |
| cuda_max_allocated_gb (at step 2550) | 16.23 | 50–85 (utilization target) |
| system_used_gb (at step 2550) | 12.58 | ≥ 23 acceptable |
| final loss @ step 2599 | 0.2402 | match within noise |
| eval/loss @ step 2550 | 0.3766 | match within noise |
| eval/compression_ratio @ step 2550 | 0.8552 | match (same curriculum) |

Trajectory samples (every 10 training steps, every ~260 for eval) stored in JSON for direct comparison to new packed runs.

## Usage for Wave 4.5 validation

A packed Phase 1 Step 3 smoke run of ~100-200 steps should match this trajectory's loss/grad_norm/ratio shape at the same absolute step indices. Because the packed run will use different `gradient_accumulation_steps` (target 1 vs. baseline 8), expect effective-batch-size differences to introduce ~5-10% trajectory noise; reject only gross divergence.
