# Step 5 Best Current Training Setup

This is the current recommended setup after the Qwen3.5 decoder kernel
exploration, with the actual goal preserved: Step 5 should train the decoder.
The frozen-decoder setup is useful as a speed diagnostic, but it is not the
default training objective.

## Canonical Config

Use `configs/config.yaml` as the shared decoder-training contract, with
`configs/training/phase1_step5.yaml` carrying only Step 5-specific curriculum
and data settings:

- `training.decoder_lora.enabled: false`
- `training.freeze.decoder: false`
- `training.decoder_ce_impl: cce`
- `training.decoder_ce_strict: true`
- `training.decoder_gradient_checkpointing: false`
- `training.decoder_layerwise_split.mode: "0"`
- `training.decoder_layerwise_split.packed_deltanet: false`
- all `training.frozen_decoder_kernels.*` diagnostic kernel switches default to
  `false`

The Docker training environment should keep these runtime defaults:

- `BGKIT_GDN_BACKEND=fla`
- `BGKIT_DECODER_CE_IMPL=cce`
- `FLA_GDR_SAVE_INTERMEDIATES=0`
- `FLA_GDR_SAVE_LOCAL_ATTENTION=1`
- `FLA_GDR_RECOMPUTE_WY_DW=1`
- `FLA_GDR_FUSE_WY_DG_CUMSUM=1`
- `FLA_GDR_FUSE_DQKG_WY=1`
- `FLA_GDR_FUSE_GATE_BWD=1`

Do not set any of the lower-level `BGKIT_FROZEN_DELTANET_*`,
`BGKIT_DELTANET_RAW_GATE_IN_KERNEL`, `BGKIT_QWEN35_LAYERWISE_SPLIT`, or
`FLA_GDR_FUSE_DQKG_WY_VARLEN` flags for ordinary Step 5 training unless running
a named A/B experiment.

## Training Command

Resume Step 5 with the normal training service/config path:

```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm train-phase1-step5
```

For a fixed-batch smoke/profile before a long run, use the profiling harness and
the same config. Always check for active GPU processes first and stop any
training container gracefully before benchmarking.

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
docker compose --env-file .env -f docker/docker-compose.yaml run --rm \
  train-phase1-step5 \
  scripts/profile_training_step.py \
    +experiment=phase1_step5 \
    +profile.warmup_steps=0 \
    +profile.steps=1 \
    +profile.fixed_batches=true \
    +profile.repeat_first_fixed_step=true \
    +profile.capture_profiler=false
```

## Why This Setup

The best current Step 5 path that still trains the decoder is full-decoder
training with LoRA disabled, CCE, and the stock optimized FLA GDR path. The
strongest real fixed-batch kernel signal was `FLA_GDR_SAVE_INTERMEDIATES=0`,
which reduced peak memory and improved wall time versus saving GDR
intermediates.

The layerwise Qwen split is off by default because it deliberately runs the
prefix under `no_grad`. That can be appropriate for frozen-decoder or
prefix-heavy diagnostics, but it is not the default when the objective is to
train decoder weights from the full packed splice.

Decoder activation checkpointing is explicitly off for now. On the fixed Step 5
gate, both selective (`megatron`) and standard PyTorch decoder checkpointing
hit a saved-tensor count mismatch during trainable Qwen3.5 packed backward.
The safe current split is encoder-side checkpointing only, with decoder
activations kept explicit until the packed decoder path is checkpoint-stable.

The corrected full-decoder gate was rerun from
`phase1_step5_step6116_20260509_160443` with `contract_ok=true`: all
`752,393,024` decoder parameters were trainable and present in the optimizer.
On the fixed `152,431` target-token batch, the current default
`FLA_GDR_SAVE_INTERMEDIATES=0` completed in `70,674 ms`, `2,157 target/s`, and
`54.88 GiB` peak
(`profile_training_step_phase1_step5_20260518_132029.json`). The direct A/B
with `FLA_GDR_SAVE_INTERMEDIATES=1` was worse: `73,896 ms`,
`2,063 target/s`, and `61.21 GiB` peak
(`profile_training_step_phase1_step5_20260518_132613.json`).

The diagnostic kernel paths remain default-off because they failed to beat the
same-runtime fixed Step 5 gate, even when they improved isolated layer or
synthetic decoder timing. In particular:

- channel-last/raw-gate is correct and slightly faster synthetically, but only
  sub-percent on the full decoder gate;
- packed DeltaNet split drifts in survivor gradients;
- residual/custom-core DeltaNet wrappers improve or reduce memory in isolated
  cases, but lose or remain noise-level on fixed Step 5;
- LoRA is out of scope for the current default because we want full decoder
  training, not adapter-only training.

The next ambitious kernel direction is below the current Python wrapper
boundary: a lower-level Qwen3.5 linear-attention training contract that keeps
qkv channel-last through conv/L2Norm/GDR and owns the residual input-gradient
path without adding extra materialization or Python autograd boundaries.
