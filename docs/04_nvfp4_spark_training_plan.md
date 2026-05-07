# NVFP4 Spark Training Plan

Date: 2026-05-07

Atlas gets NVFP4 working on DGX Spark by avoiding TransformerEngine's dynamic
FP4 conversion path. TE currently emits device-side errors on our sm_121 image:
the FP4 `cvt` PTX instructions require an architecture-specific build. Atlas
instead packs weights at load time with software quantization and runs custom
W4A16 kernels.

## Atlas Pattern To Imitate

- Detect quantized tensors through model metadata and tensor suffixes such as
  `.weight_packed`, `.weight_scale_inv`, and `.weight_scale`.
- For raw BF16 weights, run a global absmax pass, then pack BF16 to NVFP4:
  packed uint8 weights shaped like `[N, K / 2]`, E4M3 scale bytes shaped like
  `[N, K / 16]`, and one second-level scalar scale.
- Use software E4M3 and E2M1 conversion rather than hardware FP4 PTX conversion.
  Dequantization is logically `E2M1(nibble) * E4M3(scale) * scale2`.
- Transpose packed weights and scales for coalesced reads in the GEMM kernels.
- Use custom W4A16 matmul kernels rather than TE Linear.

The Atlas source is AGPL-3.0-only, so BgKIT should reimplement the technique
from the interface and measured behavior rather than copying code.

## Training-Specific Scope

BgKIT's first useful NVFP4 target is frozen-base LoRA training, not full
fine-tuning. In that regime each packed base linear needs:

- forward: `Y = X @ W4.T`
- activation backward: `dX = dY @ W4`
- no base-weight gradient
- separate ordinary BF16/FP32 LoRA A/B gradients

That is much smaller than a general trainable FP4 Linear because it avoids W4
wgrad entirely. It also matches the reason the current `enable_nvfp4()` path
already freezes converted base layers.

## Proposed BgKIT Implementation

1. Add an independent `bgkit.quant.nvfp4` packer with CPU and CUDA entry
   points. Start with a reference CPU implementation for exact tests, then add
   a CUDA packer once the format is locked.
2. Add a `FrozenNVFP4Linear` module that stores `weight_packed`, `scale_e4m3`,
   `scale2`, dimensions, and optional bias. It should expose BF16 forward and
   BF16 `dX` through a custom autograd function, returning `None` for packed
   weight gradients.
3. Convert only LoRA base layers at first. Priority order for the Qwen decoder:
   MLP `gate_proj`, `up_proj`, `down_proj`, then attention/SSM projections once
   parity and speed are proven.
4. Implement the first W4A16 kernel for the dense MLP shapes in the decoder.
   The kernel can be independent of the Atlas license while using the same
   schedule idea: packed/coalesced K reads, BF16 activations, FP32 accumulate,
   BF16 output.
5. Add benchmark switches beside `--decoder-nvfp4` so TE and BgKIT-native
   NVFP4 are never confused:
   `--decoder-nvfp4-backend te|native-frozen`.

## Validation Gates

- Unit-test pack/dequant against a BF16 reference on small shapes, including
  odd K rejection and scale saturation behavior.
- Linear forward max-error and cosine checks against BF16 weights.
- Autograd check for `dX`; packed weights must not receive gradients.
- Decoder benchmark with LoRA enabled and frozen base weights:
  compare BF16, TE experiment, and native-frozen NVFP4 at seq256 and seq2048.
- Only make `native-frozen` a default after it beats BF16 on full decoder
  forward plus backward, not just a microbenchmark.
