#!/usr/bin/env python3
"""Benchmark packed-splice embedding assembly backward overhead.

This isolates the boundary created by
``ReconstructionDecoder.forward_with_single_splice``:

    [prefix token embeddings | survivor embeddings | suffix token embeddings]

In the no-LoRA frozen-decoder contract, prefix/suffix token embeddings are
frozen and only the survivor embedding tensor needs gradients. The full decoder
still has to propagate input gradients through all sequence rows, but this
benchmark checks whether the splice assembly node itself is worth replacing by
a custom autograd function that gathers only survivor-row gradients.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Case:
    batch_size: int
    prefix_len: int
    survivor_len: int
    suffix_len: int
    hidden_size: int
    dtype: torch.dtype


class _PackedSpliceAssemblyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        prefix_all: torch.Tensor,
        survivor_embeddings: torch.Tensor,
        suffix_all: torch.Tensor,
        prefix_lens: tuple[int, ...],
        survivor_lens: tuple[int, ...],
        suffix_lens: tuple[int, ...],
    ) -> torch.Tensor:
        hidden_size = int(survivor_embeddings.shape[-1])
        total_len = sum(prefix_lens) + sum(survivor_lens) + sum(suffix_lens)
        out = survivor_embeddings.new_empty(total_len, hidden_size)
        survivor_ranges: list[tuple[int, int, int, int]] = []
        out_off = 0
        prefix_off = 0
        survivor_off = 0
        suffix_off = 0
        for prefix_len, survivor_len, suffix_len in zip(
            prefix_lens,
            survivor_lens,
            suffix_lens,
            strict=True,
        ):
            if prefix_len:
                out[out_off : out_off + prefix_len] = prefix_all[
                    prefix_off : prefix_off + prefix_len
                ]
                out_off += prefix_len
                prefix_off += prefix_len
            if survivor_len:
                out[out_off : out_off + survivor_len] = survivor_embeddings[
                    survivor_off : survivor_off + survivor_len
                ]
                survivor_ranges.append(
                    (out_off, out_off + survivor_len, survivor_off, survivor_off + survivor_len)
                )
                out_off += survivor_len
                survivor_off += survivor_len
            if suffix_len:
                out[out_off : out_off + suffix_len] = suffix_all[
                    suffix_off : suffix_off + suffix_len
                ]
                out_off += suffix_len
                suffix_off += suffix_len
        ctx.survivor_shape = tuple(survivor_embeddings.shape)
        ctx.survivor_ranges = tuple(survivor_ranges)
        return out.unsqueeze(0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_flat = grad_output.squeeze(0)
        grad_survivors = torch.empty(
            ctx.survivor_shape,
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        for out_start, out_end, surv_start, surv_end in ctx.survivor_ranges:
            grad_survivors[surv_start:surv_end] = grad_flat[out_start:out_end]
        return None, grad_survivors, None, None, None, None


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _build_inputs(case: Case, device: torch.device):
    prefix_lens = (case.prefix_len,) * case.batch_size
    survivor_lens = (case.survivor_len,) * case.batch_size
    suffix_lens = (case.suffix_len,) * case.batch_size
    prefix_total = sum(prefix_lens)
    survivor_total = sum(survivor_lens)
    suffix_total = sum(suffix_lens)
    prefix_all = torch.randn(
        prefix_total,
        case.hidden_size,
        device=device,
        dtype=case.dtype,
    )
    survivor_embeddings = torch.randn(
        survivor_total,
        case.hidden_size,
        device=device,
        dtype=case.dtype,
        requires_grad=True,
    )
    suffix_all = torch.randn(
        suffix_total,
        case.hidden_size,
        device=device,
        dtype=case.dtype,
    )
    return prefix_all, survivor_embeddings, suffix_all, prefix_lens, survivor_lens, suffix_lens


def _assemble_cat(
    prefix_all: torch.Tensor,
    survivor_embeddings: torch.Tensor,
    suffix_all: torch.Tensor,
    prefix_lens: tuple[int, ...],
    survivor_lens: tuple[int, ...],
    suffix_lens: tuple[int, ...],
) -> torch.Tensor:
    sample_embeds: list[torch.Tensor] = []
    prefix_off = 0
    survivor_off = 0
    suffix_off = 0
    for prefix_len, survivor_len, suffix_len in zip(
        prefix_lens,
        survivor_lens,
        suffix_lens,
        strict=True,
    ):
        emb_pre = prefix_all[prefix_off : prefix_off + prefix_len]
        surv = survivor_embeddings[survivor_off : survivor_off + survivor_len]
        emb_suf = suffix_all[suffix_off : suffix_off + suffix_len]
        sample_embeds.append(torch.cat([emb_pre, surv, emb_suf], dim=0))
        prefix_off += prefix_len
        survivor_off += survivor_len
        suffix_off += suffix_len
    return torch.cat(sample_embeds, dim=0).unsqueeze(0)


def _assemble_custom(
    prefix_all: torch.Tensor,
    survivor_embeddings: torch.Tensor,
    suffix_all: torch.Tensor,
    prefix_lens: tuple[int, ...],
    survivor_lens: tuple[int, ...],
    suffix_lens: tuple[int, ...],
) -> torch.Tensor:
    return _PackedSpliceAssemblyFunction.apply(
        prefix_all,
        survivor_embeddings,
        suffix_all,
        prefix_lens,
        survivor_lens,
        suffix_lens,
    )


def _time_path(
    fn,
    inputs,
    grad_out: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    prefix_all, survivor_embeddings, suffix_all, prefix_lens, survivor_lens, suffix_lens = inputs
    for _ in range(warmup):
        survivor_embeddings.grad = None
        out = fn(
            prefix_all,
            survivor_embeddings,
            suffix_all,
            prefix_lens,
            survivor_lens,
            suffix_lens,
        )
        out.backward(grad_out)
    torch.cuda.synchronize()

    fwd_times: list[float] = []
    total_times: list[float] = []
    for _ in range(iters):
        survivor_embeddings.grad = None
        start_fwd = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)
        end_total = torch.cuda.Event(enable_timing=True)
        start_fwd.record()
        out = fn(
            prefix_all,
            survivor_embeddings,
            suffix_all,
            prefix_lens,
            survivor_lens,
            suffix_lens,
        )
        end_fwd.record()
        out.backward(grad_out)
        end_total.record()
        torch.cuda.synchronize()
        fwd_times.append(start_fwd.elapsed_time(end_fwd))
        total_times.append(start_fwd.elapsed_time(end_total))
    return statistics.median(fwd_times), statistics.median(total_times)


def _check_parity(case: Case, device: torch.device) -> None:
    inputs = _build_inputs(case, device)
    cat_survivors = inputs[1]
    custom_survivors = cat_survivors.detach().clone().requires_grad_(True)
    custom_inputs = (
        inputs[0],
        custom_survivors,
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
    )
    cat_out = _assemble_cat(*inputs)
    custom_out = _assemble_custom(*custom_inputs)
    torch.testing.assert_close(custom_out, cat_out)
    grad_out = torch.randn_like(cat_out)
    cat_out.backward(grad_out)
    custom_out.backward(grad_out)
    torch.testing.assert_close(custom_survivors.grad, cat_survivors.grad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefix-len", type=int, default=0)
    parser.add_argument("--survivor-len", type=int, default=32)
    parser.add_argument("--suffix-len", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(0)
    case = Case(
        batch_size=args.batch_size,
        prefix_len=args.prefix_len,
        survivor_len=args.survivor_len,
        suffix_len=args.suffix_len,
        hidden_size=args.hidden_size,
        dtype=_dtype(args.dtype),
    )
    _check_parity(case, device)
    inputs = _build_inputs(case, device)
    total_len = case.batch_size * (case.prefix_len + case.survivor_len + case.suffix_len)
    grad_out = torch.randn(1, total_len, case.hidden_size, device=device, dtype=case.dtype)
    cat_fwd, cat_total = _time_path(
        _assemble_cat,
        inputs,
        grad_out,
        warmup=args.warmup,
        iters=args.iters,
    )
    custom_fwd, custom_total = _time_path(
        _assemble_custom,
        inputs,
        grad_out,
        warmup=args.warmup,
        iters=args.iters,
    )
    print(
        "splice_assembly "
        f"B={case.batch_size} prefix={case.prefix_len} survivor={case.survivor_len} "
        f"suffix={case.suffix_len} hidden={case.hidden_size} dtype={args.dtype}"
    )
    print(f"cat:    fwd={cat_fwd:.4f} ms total={cat_total:.4f} ms")
    print(f"custom: fwd={custom_fwd:.4f} ms total={custom_total:.4f} ms")
    print(
        "delta:  "
        f"fwd={custom_fwd - cat_fwd:+.4f} ms total={custom_total - cat_total:+.4f} ms"
    )


if __name__ == "__main__":
    main()
