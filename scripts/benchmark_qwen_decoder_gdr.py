#!/usr/bin/env python3
"""Benchmark Qwen3.5 decoder forward/backward speed on real text.

This is a small training jig for DeltaNet/GDR kernel work. It loads the same
Qwen decoder backbone configured for BgKIT, applies BgKIT's DeltaNet patch, and
times causal-LM forward/backward passes over tokenized text.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import textwrap
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_TEXT = """
BgKIT compresses long code and document contexts into a compact memory that a
decoder can query while reconstructing or answering questions. The benchmark
batch repeats this text to create stable Qwen3.5 DeltaNet sequence lengths for
forward and backward timing on the target hardware.
"""


def _set_env_default(name: str, value: str) -> None:
    if os.environ.get(name) is None:
        os.environ[name] = value


def _set_toggle(name: str, value: str) -> None:
    if value == "default":
        return
    elif value == "on":
        os.environ[name] = "1"
    elif value == "off":
        os.environ[name] = "0"
    else:
        raise ValueError(f"Unknown toggle value for {name}: {value}")


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


_GPU_TELEMETRY_FIELDS = (
    "power.draw",
    "clocks.current.graphics",
    "clocks.current.sm",
    "utilization.gpu",
    "temperature.gpu",
)


def _coerce_gpu_telemetry_value(value: str) -> float | str | None:
    value = value.strip()
    if not value or value == "[N/A]":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def _gpu_telemetry() -> dict[str, float | str | None]:
    query = ",".join(_GPU_TELEMETRY_FIELDS)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception as exc:
        return {"error": str(exc)}
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    values = [item.strip() for item in line.split(",")]
    return {
        field: _coerce_gpu_telemetry_value(values[idx])
        if idx < len(values)
        else None
        for idx, field in enumerate(_GPU_TELEMETRY_FIELDS)
    }


def _start_gpu_telemetry_sampler(
    interval_s: float,
) -> tuple[threading.Event, threading.Thread, list[dict[str, float | str | None]]]:
    samples: list[dict[str, float | str | None]] = []
    stop_event = threading.Event()

    def _sample_loop() -> None:
        while not stop_event.is_set():
            sample = _gpu_telemetry()
            sample["timestamp_s"] = time.time()
            samples.append(sample)
            stop_event.wait(interval_s)

    thread = threading.Thread(target=_sample_loop, name="gpu-telemetry", daemon=True)
    thread.start()
    return stop_event, thread, samples


def _summarize_gpu_telemetry(
    samples: list[dict[str, float | str | None]],
) -> dict[str, float | int | None]:
    def _numeric_values(field: str) -> list[float]:
        return [float(sample[field]) for sample in samples if isinstance(sample.get(field), float)]

    summary: dict[str, float | int | None] = {"samples": len(samples)}
    for field in _GPU_TELEMETRY_FIELDS:
        values = _numeric_values(field)
        summary[f"{field}.max"] = max(values) if values else None
        summary[f"{field}.median"] = statistics.median(values) if values else None
    return summary


def _read_text(args: argparse.Namespace) -> str:
    if args.text_file is not None:
        return Path(args.text_file).read_text(encoding="utf-8")
    if args.text:
        return args.text
    return DEFAULT_TEXT


def _dense_repeated_text(tokenizer, text: str, seq_len: int) -> str:
    unit = text.strip()
    repeated = unit
    while len(tokenizer(repeated, truncation=False)["input_ids"]) < seq_len:
        repeated = repeated + "\n\n" + unit
    return repeated


def _build_token_batch(
    tokenizer,
    text: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    repeated = _dense_repeated_text(tokenizer, text, seq_len)
    encoded = tokenizer(
        repeated,
        max_length=seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].repeat(batch_size, 1).to(device)
    attention_mask = encoded["attention_mask"].repeat(batch_size, 1).to(device)
    return input_ids, attention_mask


def _build_hf_batch(
    tokenizer,
    text: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_ids, attention_mask = _build_token_batch(
        tokenizer=tokenizer,
        text=text,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _build_packed_splice_batch(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    survivor_len: int = 0,
    train_survivors: bool = False,
) -> dict[str, object]:
    input_ids, attention_mask = _build_token_batch(
        tokenizer=tokenizer,
        text=text,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )
    if not torch.all(attention_mask):
        raise RuntimeError("packed-splice benchmark expects dense, unpadded token batches.")
    if seq_len < 2:
        raise ValueError("packed-splice benchmark requires --seq-len >= 2.")

    if hasattr(model, "_get_inner_model_and_head"):
        inner_model, _lm_head = model._get_inner_model_and_head()
    else:
        backbone = getattr(model, "backbone", model)
        inner_model = backbone.model if hasattr(backbone, "model") else backbone
    embed = inner_model.get_input_embeddings()
    hidden_dim = int(embed.weight.shape[-1])
    survivor_total = batch_size * int(survivor_len)
    survivor_embeddings = torch.empty(
        survivor_total,
        hidden_dim,
        dtype=embed.weight.dtype,
        device=device,
    )
    if survivor_total:
        survivor_embeddings.normal_(mean=0.0, std=0.02)
    survivor_embeddings.requires_grad_(bool(train_survivors))
    survivor_cu = torch.arange(
        0,
        survivor_total + 1,
        int(survivor_len) if int(survivor_len) > 0 else 1,
        dtype=torch.int32,
        device=device,
    )
    if survivor_len == 0:
        survivor_cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    survivor_cu_cpu = [b * int(survivor_len) for b in range(batch_size + 1)]
    packed_seq_len = int(seq_len) + int(survivor_len)
    packed_cu = torch.arange(
        0,
        (batch_size + 1) * packed_seq_len,
        packed_seq_len if packed_seq_len > 0 else 1,
        dtype=torch.int32,
        device=device,
    )
    packed_position_ids = (
        torch.arange(packed_seq_len, dtype=torch.long, device=device)
        .repeat(batch_size)
        .contiguous()
    )
    return {
        "survivor_embeddings": survivor_embeddings,
        "survivor_cu_seqlens": survivor_cu,
        "survivor_cu_seqlens_cpu": survivor_cu_cpu,
        "packed_cu_seqlens": packed_cu,
        "packed_position_ids": packed_position_ids,
        "prefix_ids": [row[:1].contiguous() for row in input_ids],
        "suffix_ids": [row[1:].contiguous() for row in input_ids],
    }


def _ms_stats(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    return {
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def _run_step(
    model: torch.nn.Module,
    forward_loss: Callable[[], torch.Tensor],
    zero_extra_grads: Callable[[], None] | None = None,
) -> tuple[float, float, float, float]:
    model.zero_grad(set_to_none=True)
    if zero_extra_grads is not None:
        zero_extra_grads()
    start = torch.cuda.Event(enable_timing=True)
    fwd_done = torch.cuda.Event(enable_timing=True)
    bwd_done = torch.cuda.Event(enable_timing=True)
    start.record()
    loss = forward_loss()
    fwd_done.record()
    loss.backward()
    bwd_done.record()
    torch.cuda.synchronize()
    return (
        start.elapsed_time(fwd_done),
        fwd_done.elapsed_time(bwd_done),
        start.elapsed_time(bwd_done),
        float(loss.detach().cpu()),
    )


def _zero_existing_grads_(
    model: torch.nn.Module,
    extra_grad_tensors: tuple[torch.Tensor, ...],
) -> None:
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()
    for tensor in extra_grad_tensors:
        if tensor.grad is not None:
            tensor.grad.zero_()


def _clear_grads_to_none_(
    model: torch.nn.Module,
    extra_grad_tensors: tuple[torch.Tensor, ...],
) -> None:
    model.zero_grad(set_to_none=True)
    for tensor in extra_grad_tensors:
        tensor.grad = None


def _capture_cuda_graph_step(
    model: torch.nn.Module,
    forward_loss: Callable[[], torch.Tensor],
    *,
    extra_grad_tensors: tuple[torch.Tensor, ...] = (),
    eager_warmup: int = 2,
) -> tuple[torch.cuda.CUDAGraph, list[torch.Tensor | None]]:
    """Capture one static total step for benchmark-only replay timing."""

    capture_stream = torch.cuda.current_stream()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(capture_stream)
    with torch.cuda.stream(warmup_stream):
        for _ in range(max(int(eager_warmup), 1)):
            _clear_grads_to_none_(model, extra_grad_tensors)
            loss = forward_loss()
            loss.backward()
            del loss
    capture_stream.wait_stream(warmup_stream)

    static_loss: list[torch.Tensor | None] = [None]
    graph = torch.cuda.CUDAGraph()
    _zero_existing_grads_(model, extra_grad_tensors)
    with torch.cuda.graph(graph):
        _zero_existing_grads_(model, extra_grad_tensors)
        static_loss[0] = forward_loss()
        static_loss[0].backward()
    torch.cuda.synchronize()
    return graph, static_loss


def _run_cuda_graph_replay(
    graph: torch.cuda.CUDAGraph,
    static_loss: list[torch.Tensor | None],
) -> tuple[float, float, float, float]:
    start = torch.cuda.Event(enable_timing=True)
    done = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    done.record()
    torch.cuda.synchronize()
    loss_tensor = static_loss[0]
    if loss_tensor is None:
        raise RuntimeError("CUDA graph replay did not expose a captured loss tensor")
    total = start.elapsed_time(done)
    return 0.0, 0.0, total, float(loss_tensor.detach().cpu())


def _count_gdr_layers(model: torch.nn.Module) -> int:
    return sum(
        1
        for module in model.modules()
        if hasattr(module, "chunk_gated_delta_rule") and hasattr(module, "A_log")
    )


def _active_backend(requested: str, resolved: str | None) -> str:
    if resolved is not None:
        return resolved
    if requested == "fla":
        return "fla"
    return "<unresolved>"


def _cce_requested(impl: str) -> bool:
    return impl not in {"auto", "chunked", "frozen_chunked", "liger"}


class _ProfiledFrozenLinearDxFunction(torch.autograd.Function):
    """Diagnostic frozen-linear autograd with per-module CUDA event timing."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        name: str,
        stats: dict[str, dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]],
    ) -> torch.Tensor:
        ctx.save_for_backward(weight)
        ctx.x_shape = tuple(x.shape)
        ctx.name = name
        ctx.stats = stats
        if x.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = F.linear(x, weight, bias)
            end.record()
            stats.setdefault(name, {"fwd": [], "bwd": []})["fwd"].append((start, end))
            return out
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (weight,) = ctx.saved_tensors
        grad_out = grad_output.reshape(-1, grad_output.shape[-1])
        if grad_out.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            grad_x = grad_out.matmul(weight)
            end.record()
            ctx.stats.setdefault(ctx.name, {"fwd": [], "bwd": []})["bwd"].append(
                (start, end)
            )
        else:
            grad_x = grad_out.matmul(weight)
        return grad_x.reshape(ctx.x_shape), None, None, None, None


class _ProfiledFrozenLinear(nn.Module):
    """Linear wrapper used only by --profile-linears."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        name: str,
        stats: dict[str, dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]],
    ) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight
        self.bias = base.bias
        self._bgkit_profile_name = name
        self._bgkit_profile_stats = stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.requires_grad or (
            self.bias is not None and self.bias.requires_grad
        ):
            return F.linear(x, self.weight, self.bias)
        if not x.requires_grad:
            return F.linear(x, self.weight, self.bias)
        return _ProfiledFrozenLinearDxFunction.apply(
            x,
            self.weight,
            self.bias,
            self._bgkit_profile_name,
            self._bgkit_profile_stats,
        )


def _all_qwen_projection_names() -> tuple[str, ...]:
    return (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    )


def _install_profiled_frozen_linears(
    root: nn.Module,
    *,
    target_names: tuple[str, ...],
) -> tuple[dict[str, dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]], int]:
    stats: dict[str, dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]] = {}
    targets = set(target_names)
    count = 0
    for parent_name, parent in list(root.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name not in targets or not isinstance(child, nn.Linear):
                continue
            if child.weight.requires_grad or (
                child.bias is not None and child.bias.requires_grad
            ):
                continue
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            setattr(
                parent,
                child_name,
                _ProfiledFrozenLinear(child, name=full_name, stats=stats),
            )
            count += 1
    return stats, count


def _clear_profiled_linear_stats(
    stats: dict[str, dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]],
) -> None:
    for item in stats.values():
        item["fwd"].clear()
        item["bwd"].clear()


def _elapsed_event_pairs(
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]],
) -> float:
    return sum(start.elapsed_time(end) for start, end in events)


def _summarize_profiled_linears(
    stats: dict[str, dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]],
    *,
    topk: int,
) -> dict[str, object]:
    path_rows: list[dict[str, object]] = []
    family: dict[str, dict[str, float | int]] = {}
    for name, item in stats.items():
        fwd_ms = _elapsed_event_pairs(item["fwd"])
        bwd_ms = _elapsed_event_pairs(item["bwd"])
        fwd_calls = len(item["fwd"])
        bwd_calls = len(item["bwd"])
        path_rows.append(
            {
                "name": name,
                "family": name.rsplit(".", 1)[-1],
                "fwd_ms": fwd_ms,
                "bwd_ms": bwd_ms,
                "total_ms": fwd_ms + bwd_ms,
                "fwd_calls": fwd_calls,
                "bwd_calls": bwd_calls,
            }
        )
        local = name.rsplit(".", 1)[-1]
        bucket = family.setdefault(
            local,
            {"fwd_ms": 0.0, "bwd_ms": 0.0, "total_ms": 0.0, "fwd_calls": 0, "bwd_calls": 0},
        )
        bucket["fwd_ms"] = float(bucket["fwd_ms"]) + fwd_ms
        bucket["bwd_ms"] = float(bucket["bwd_ms"]) + bwd_ms
        bucket["total_ms"] = float(bucket["total_ms"]) + fwd_ms + bwd_ms
        bucket["fwd_calls"] = int(bucket["fwd_calls"]) + fwd_calls
        bucket["bwd_calls"] = int(bucket["bwd_calls"]) + bwd_calls

    path_rows.sort(key=lambda row: float(row["total_ms"]), reverse=True)
    family_rows = [
        {"family": name, **values}
        for name, values in sorted(
            family.items(),
            key=lambda pair: float(pair[1]["total_ms"]),
            reverse=True,
        )
    ]
    return {
        "families": family_rows[:topk],
        "paths": path_rows[:topk],
    }


def _contains_cuda_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.is_cuda)
    if isinstance(value, (tuple, list)):
        return any(_contains_cuda_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_cuda_tensor(item) for item in value.values())
    return False


def _install_gdr_internal_timers() -> dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]:
    """Wrap FLA GDR Python entry points with CUDA event timers."""

    import fla.ops.gated_delta_rule.chunk as gdr_chunk

    names = (
        "chunk_gated_delta_rule_fwd_intra",
        "chunk_gated_delta_rule_fwd_h",
        "chunk_fwd_o",
        "chunk_fwd_o_sm121",
        "chunk_bwd_dv_local",
        "chunk_gated_delta_rule_bwd_dhu",
        "fused_dqkg_wy_bwd",
        "chunk_bwd_dqkg_fullk",
        "chunk_bwd_dqkwg",
        "prepare_wy_repr_bwd",
        "gdn_gate_bwd",
        "chunk_local_cumsum",
        "recompute_w_u_fwd",
        "l2norm_fwd",
        "l2norm_bwd_pair",
    )
    stats: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        name: [] for name in names
    }
    for name in names:
        original = getattr(gdr_chunk, name, None)
        if original is None or getattr(original, "_bgkit_profiled_gdr_internal", False):
            continue

        def _wrap(fn, label):
            def _profiled(*args, **kwargs):
                if not (_contains_cuda_tensor(args) or _contains_cuda_tensor(kwargs)):
                    return fn(*args, **kwargs)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = fn(*args, **kwargs)
                end.record()
                stats[label].append((start, end))
                return out

            _profiled._bgkit_profiled_gdr_internal = True
            return _profiled

        setattr(gdr_chunk, name, _wrap(original, name))
    return stats


def _clear_gdr_internal_timers(
    stats: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]],
) -> None:
    for events in stats.values():
        events.clear()


def _summarize_gdr_internal_timers(
    stats: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, events in stats.items():
        if not events:
            continue
        total_ms = _elapsed_event_pairs(events)
        rows.append({"name": name, "calls": len(events), "total_ms": total_ms})
    rows.sort(key=lambda row: float(row["total_ms"]), reverse=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Torch RNG seed for deterministic synthetic benchmark inputs.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--backend", choices=["fla", "flashqla", "auto"], default="fla")
    parser.add_argument(
        "--loss-path",
        choices=["packed-splice", "hf"],
        default="packed-splice",
        help="Use bgkit's packed decoder loss path, or raw HF causal-LM loss.",
    )
    parser.add_argument("--ce-chunk-size", type=int, default=2048)
    parser.add_argument(
        "--ce-impl",
        choices=[
            "auto",
            "chunked",
            "frozen_chunked",
            "liger",
            "cce",
            "cce_static",
            "cce_exact",
            "cce_kahan_full",
            "cce_kahan_full_c",
            "cce_kahan_full_e",
            "cce_kahan_full_c_full_e",
            "torch_compile",
        ],
        default=os.environ.get("BGKIT_DECODER_CE_IMPL", "auto"),
        help="Decoder LM CE implementation for the packed-splice path.",
    )
    parser.add_argument(
        "--ce-strict",
        action="store_true",
        help="Fail instead of falling back when an explicit CCE implementation is unavailable.",
    )
    parser.add_argument(
        "--decoder-lora",
        action="store_true",
        help="Apply bgkit's default decoder LoRA setup.",
    )
    parser.add_argument(
        "--freeze-decoder",
        action="store_true",
        help=(
            "Freeze decoder weights and benchmark gradients flowing back to "
            "the packed survivor embeddings. Requires --loss-path packed-splice "
            "and --survivor-len > 0."
        ),
    )
    parser.add_argument(
        "--survivor-len",
        type=int,
        default=0,
        help=(
            "Synthetic survivor embedding count per sample for packed-splice "
            "benchmarks. Use with --freeze-decoder to measure dLoss/dSurvivors."
        ),
    )
    parser.add_argument(
        "--train-survivors",
        action="store_true",
        help=(
            "Require gradients for synthetic survivor embeddings even when "
            "decoder weights are trainable."
        ),
    )
    parser.add_argument(
        "--omit-packed-splice-metadata",
        action="store_true",
        help=(
            "Diagnostic: remove prebuilt survivor/packed metadata from the "
            "synthetic packed-splice batch so forward_with_single_splice rebuilds it."
        ),
    )
    parser.add_argument(
        "--frozen-mlp-fusion",
        action="store_true",
        help=(
            "Enable BgKIT's frozen no-LoRA Qwen MLP autograd path. Only valid "
            "with --freeze-decoder."
        ),
    )
    parser.add_argument(
        "--frozen-mlp-swiglu-fusion",
        action="store_true",
        help=(
            "Patch only frozen Qwen MLP SwiGLU activation backward. Only valid "
            "with --freeze-decoder."
        ),
    )
    parser.add_argument(
        "--frozen-mlp-residual-fusion",
        action="store_true",
        help=(
            "Enable BgKIT's frozen Qwen layer post-attention RMSNorm+MLP "
            "residual autograd path. Only valid with --freeze-decoder."
        ),
    )
    parser.add_argument(
        "--frozen-linear-dx",
        action="store_true",
        help=(
            "Wrap frozen decoder linears with an input-gradient-only autograd "
            "path. Only valid with --freeze-decoder."
        ),
    )
    parser.add_argument(
        "--frozen-linear-dx-targets",
        choices=["core", "all-qwen"],
        default="core",
        help=(
            "Projection names covered by --frozen-linear-dx. 'core' keeps the "
            "original attention/MLP set; 'all-qwen' also covers Qwen3.5 "
            "DeltaNet input/output projections."
        ),
    )
    parser.add_argument(
        "--fused-attention-qkv",
        action="store_true",
        help=(
            "Patch frozen Qwen3.5 full-attention blocks to compute q/k/v with "
            "one concatenated projection. Only valid with --freeze-decoder."
        ),
    )
    parser.add_argument(
        "--fused-deltanet-zba",
        action="store_true",
        help=(
            "Patch frozen Qwen3.5 DeltaNet blocks to compute z/b/a with one "
            "concatenated projection. Only valid with --freeze-decoder."
        ),
    )
    parser.add_argument(
        "--fused-deltanet-input-bundle",
        action="store_true",
        help=(
            "Patch frozen Qwen3.5 DeltaNet blocks to compute qkv/z/b/a with "
            "one concatenated projection. Only valid with --freeze-decoder."
        ),
    )
    parser.add_argument(
        "--frozen-deltanet-core-bwd",
        action="store_true",
        help=(
            "Patch frozen Qwen3.5 DeltaNet modules with BgKIT's direct_split "
            "core backward experiment. Diagnostic only: warmed real-decoder "
            "seq512/seq2048 A/B loses to the stock frozen FLA path even after "
            "direct norm-tail/no-weight-grad and raw-gate dproj cleanup. "
            "Only valid with --freeze-decoder."
        ),
    )
    parser.add_argument(
        "--frozen-deltanet-residual-bwd",
        action="store_true",
        help=(
            "Patch frozen Qwen3.5 linear-attention decoder layers with a wider "
            "input-RMSNorm + DeltaNet + residual dX-only autograd path. "
            "Diagnostic only; only valid with --freeze-decoder."
        ),
    )
    parser.add_argument(
        "--frozen-deltanet-residual-mlp-bwd",
        action="store_true",
        help=(
            "With --frozen-deltanet-residual-bwd, also route the same "
            "linear-attention layer through the frozen post-attention "
            "RMSNorm+MLP residual autograd path."
        ),
    )
    parser.add_argument(
        "--frozen-deltanet-channel-last-conv",
        action="store_true",
        help=(
            "Patch frozen Qwen3.5 DeltaNet modules to keep the qkv causal conv "
            "channel-last while leaving the stock FLA GDR graph intact. "
            "Diagnostic only; only valid with --freeze-decoder."
        ),
    )
    parser.add_argument("--lora-implementation", choices=["peft", "native"], default="peft")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=64.0)
    parser.add_argument(
        "--lora-targets",
        choices=["all", "attention", "mlp"],
        default="all",
        help="LoRA target-module set for trainable decoder benchmarks.",
    )
    parser.add_argument(
        "--lora-fused",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use BgKIT's fused frozen-base native LoRA autograd path. "
            "Only affects --lora-implementation native; default follows "
            "BGKIT_DECODER_LORA_FUSED."
        ),
    )
    parser.add_argument(
        "--peft-fused-backward",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Patch PEFT LoRA Linear modules to use BgKIT's fused frozen-base "
            "autograd path. Only affects --lora-implementation peft; default "
            "follows BGKIT_DECODER_PEFT_FUSED_BACKWARD."
        ),
    )
    parser.add_argument(
        "--peft-fuse-gate-up",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use BgKIT's PEFT gate/up paired LoRA backward fusion. Only affects "
            "--lora-implementation peft; default follows "
            "BGKIT_DECODER_PEFT_FUSE_GATE_UP."
        ),
    )
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--decoder-nvfp4",
        action="store_true",
        help="Convert decoder base linears to an NVFP4 backend.",
    )
    parser.add_argument(
        "--decoder-nvfp4-backend",
        choices=["te", "native-frozen"],
        default="te",
        help="NVFP4 backend; native-frozen is BgKIT's packed frozen-base reference path.",
    )
    parser.add_argument("--save-intermediates", choices=["default", "on", "off"], default="default")
    parser.add_argument(
        "--save-local-attention",
        choices=["default", "on", "off"],
        default="default",
    )
    parser.add_argument("--recompute-wy-dw", choices=["default", "on", "off"], default="default")
    parser.add_argument("--fuse-wy-dg-cumsum", choices=["default", "on", "off"], default="default")
    parser.add_argument("--fuse-dqkg-wy", choices=["default", "on", "off"], default="default")
    parser.add_argument("--fuse-gate-bwd", choices=["default", "on", "off"], default="default")
    parser.add_argument("--packed-dproj-bwd", choices=["default", "on", "off"], default="default")
    parser.add_argument("--fuse-kkt-wu", choices=["default", "on", "off"], default="default")
    parser.add_argument("--state-dkdg", choices=["default", "on", "off"], default="default")
    parser.add_argument("--fullk-dqkg", choices=["default", "on", "off"], default="default")
    parser.add_argument(
        "--deltanet-raw-gate-in-kernel",
        choices=["default", "on", "off"],
        default="default",
        help=(
            "Pass raw DeltaNet a-projection gates to FLA and let the GDR gate "
            "kernel apply A_log/dt_bias/clamp. Only meaningful on the patched "
            "packed DeltaNet path."
        ),
    )
    parser.add_argument("--sm121-output", choices=["default", "on", "off"], default="off")
    parser.add_argument("--dqkwg-warps", type=int, default=None)
    parser.add_argument("--dqkwg-bk", type=int, default=None)
    parser.add_argument("--dqkwg-bv", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-liger", action="store_true")
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary.")
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help=(
            "Diagnostic mode: capture one static frozen packed-splice step in "
            "a CUDA graph and time graph replay total latency. This reports "
            "total step time only and is not a training integration."
        ),
    )
    parser.add_argument(
        "--compile-forward-loss",
        action="store_true",
        help=(
            "Diagnostic mode: wrap the final zero-argument loss closure in "
            "torch.compile before timing. This tests whole packed-splice "
            "forward/backward graph compilation after decoder patches are "
            "installed."
        ),
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
        help="torch.compile mode used with --compile-forward-loss.",
    )
    parser.add_argument(
        "--compile-backend",
        default="inductor",
        help="torch.compile backend used with --compile-forward-loss.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile one extra step after timing.",
    )
    parser.add_argument(
        "--profile-linears",
        action="store_true",
        help=(
            "Diagnostic mode: wrap frozen Qwen projection linears with a "
            "timed input-gradient-only autograd function and report path/family "
            "CUDA event totals over measured steps. This changes the graph and "
            "is not a speed candidate."
        ),
    )
    parser.add_argument("--profile-linears-topk", type=int, default=20)
    parser.add_argument(
        "--profile-gdr-internals",
        action="store_true",
        help=(
            "Diagnostic mode: time FLA GDR internal Python entry points with "
            "CUDA events over measured steps. This changes Python call paths "
            "and is not a speed candidate."
        ),
    )
    parser.add_argument("--profile-topk", type=int, default=30)
    parser.add_argument("--profile-trace", default=None, help="Optional Chrome trace output path.")
    parser.add_argument(
        "--gpu-telemetry",
        action="store_true",
        help=(
            "Sample nvidia-smi power/clocks/utilization in a background "
            "thread during timing. Diagnostic only; useful for detecting "
            "throttled runs."
        ),
    )
    parser.add_argument(
        "--gpu-telemetry-interval",
        type=float,
        default=0.5,
        help="Seconds between background nvidia-smi samples when --gpu-telemetry is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    if args.survivor_len < 0:
        raise ValueError("--survivor-len must be non-negative.")
    if args.freeze_decoder and args.loss_path != "packed-splice":
        raise ValueError("--freeze-decoder currently requires --loss-path packed-splice.")
    if args.freeze_decoder and args.survivor_len <= 0:
        raise ValueError("--freeze-decoder requires --survivor-len > 0.")
    if args.freeze_decoder and args.decoder_lora:
        raise ValueError(
            "--freeze-decoder is the no-LoRA dLoss/dSurvivors contract; "
            "drop --decoder-lora or benchmark a trainable decoder."
        )
    if args.frozen_mlp_fusion and not args.freeze_decoder:
        raise ValueError("--frozen-mlp-fusion requires --freeze-decoder.")
    if args.frozen_mlp_swiglu_fusion and not args.freeze_decoder:
        raise ValueError("--frozen-mlp-swiglu-fusion requires --freeze-decoder.")
    if args.frozen_mlp_residual_fusion and not args.freeze_decoder:
        raise ValueError("--frozen-mlp-residual-fusion requires --freeze-decoder.")
    if (
        int(bool(args.frozen_mlp_fusion))
        + int(bool(args.frozen_mlp_swiglu_fusion))
        + int(bool(args.frozen_mlp_residual_fusion))
        > 1
    ):
        raise ValueError("Frozen MLP fusion modes are mutually exclusive.")
    if args.frozen_linear_dx and not args.freeze_decoder:
        raise ValueError("--frozen-linear-dx requires --freeze-decoder.")
    if args.profile_linears and not args.freeze_decoder:
        raise ValueError("--profile-linears requires --freeze-decoder.")
    if args.cuda_graph:
        if args.loss_path != "packed-splice" or not args.freeze_decoder:
            raise ValueError("--cuda-graph requires frozen --loss-path packed-splice.")
        if args.train_survivors:
            raise ValueError("--cuda-graph is only supported for synthetic survivor grads.")
        if args.profile or args.profile_linears or args.profile_gdr_internals:
            raise ValueError("--cuda-graph cannot be combined with profiler modes.")
    if args.compile_forward_loss:
        if args.cuda_graph:
            raise ValueError("--compile-forward-loss cannot be combined with --cuda-graph.")
        if args.profile or args.profile_linears or args.profile_gdr_internals:
            raise ValueError("--compile-forward-loss cannot be combined with profiler modes.")
    if args.fused_attention_qkv and not args.freeze_decoder:
        raise ValueError("--fused-attention-qkv requires --freeze-decoder.")
    if args.fused_deltanet_zba and not args.freeze_decoder:
        raise ValueError("--fused-deltanet-zba requires --freeze-decoder.")
    if args.fused_deltanet_input_bundle and not args.freeze_decoder:
        raise ValueError("--fused-deltanet-input-bundle requires --freeze-decoder.")
    if args.frozen_deltanet_core_bwd and not args.freeze_decoder:
        raise ValueError("--frozen-deltanet-core-bwd requires --freeze-decoder.")
    if args.frozen_deltanet_residual_bwd and not args.freeze_decoder:
        raise ValueError("--frozen-deltanet-residual-bwd requires --freeze-decoder.")
    if args.frozen_deltanet_residual_mlp_bwd and not args.frozen_deltanet_residual_bwd:
        raise ValueError(
            "--frozen-deltanet-residual-mlp-bwd requires "
            "--frozen-deltanet-residual-bwd."
        )
    if args.frozen_deltanet_channel_last_conv and not args.freeze_decoder:
        raise ValueError("--frozen-deltanet-channel-last-conv requires --freeze-decoder.")
    if args.fused_deltanet_zba and args.fused_deltanet_input_bundle:
        raise ValueError(
            "--fused-deltanet-zba and --fused-deltanet-input-bundle are mutually exclusive."
        )
    if args.frozen_deltanet_core_bwd and (
        args.fused_deltanet_zba or args.fused_deltanet_input_bundle
    ):
        raise ValueError(
            "--frozen-deltanet-core-bwd is mutually exclusive with fused DeltaNet "
            "projection wrappers."
        )
    if args.frozen_deltanet_residual_bwd and (
        args.frozen_deltanet_core_bwd
        or args.fused_deltanet_zba
        or args.fused_deltanet_input_bundle
    ):
        raise ValueError(
            "--frozen-deltanet-residual-bwd is mutually exclusive with other "
            "DeltaNet forward/core wrappers."
        )
    if args.frozen_deltanet_channel_last_conv and (
        args.frozen_deltanet_core_bwd
        or args.frozen_deltanet_residual_bwd
        or args.fused_deltanet_zba
        or args.fused_deltanet_input_bundle
    ):
        raise ValueError(
            "--frozen-deltanet-channel-last-conv is mutually exclusive with other "
            "DeltaNet forward wrappers."
        )
    if args.frozen_deltanet_residual_mlp_bwd and args.frozen_mlp_residual_fusion:
        raise ValueError(
            "--frozen-deltanet-residual-mlp-bwd already owns the linear-attention "
            "MLP residual and cannot be combined with --frozen-mlp-residual-fusion."
        )

    os.environ["BGKIT_GDN_BACKEND"] = args.backend
    _set_toggle(
        "BGKIT_FROZEN_DELTANET_RESIDUAL_MLP_BWD",
        "on" if args.frozen_deltanet_residual_mlp_bwd else "default",
    )
    _set_toggle("FLA_GDR_SAVE_INTERMEDIATES", args.save_intermediates)
    _set_toggle("FLA_GDR_SAVE_LOCAL_ATTENTION", args.save_local_attention)
    _set_toggle("FLA_GDR_RECOMPUTE_WY_DW", args.recompute_wy_dw)
    _set_toggle("FLA_GDR_FUSE_WY_DG_CUMSUM", args.fuse_wy_dg_cumsum)
    _set_toggle("FLA_GDR_FUSE_DQKG_WY", args.fuse_dqkg_wy)
    _set_toggle("FLA_GDR_FUSE_GATE_BWD", args.fuse_gate_bwd)
    _set_toggle("FLA_GDR_PACKED_DPROJ_BWD", args.packed_dproj_bwd)
    _set_toggle("FLA_GDR_FUSE_KKT_WU", args.fuse_kkt_wu)
    _set_toggle("FLA_GDR_STATE_DKDG", args.state_dkdg)
    _set_toggle("FLA_GDR_FULLK_DQKG", args.fullk_dqkg)
    _set_toggle("BGKIT_DELTANET_RAW_GATE_IN_KERNEL", args.deltanet_raw_gate_in_kernel)
    _set_toggle("FLA_USE_SM121_CUSTOM_KERNEL", args.sm121_output)
    if args.ce_strict:
        os.environ["BGKIT_DECODER_CE_STRICT"] = "1"
    if args.dqkwg_warps is not None:
        os.environ["FLA_DQKWG_TL_NUM_WARPS"] = str(args.dqkwg_warps)
    if args.dqkwg_bk is not None:
        os.environ["FLA_DQKWG_TL_BK"] = str(args.dqkwg_bk)
    if args.dqkwg_bv is not None:
        os.environ["FLA_DQKWG_TL_BV"] = str(args.dqkwg_bv)
    _set_env_default("TOKENIZERS_PARALLELISM", "false")

    from bgkit.utils.deltanet_patch import patch_gated_delta_rule_numerics
    from bgkit.utils.gdn_backend import describe_backend_environment, resolved_backend_name

    patch_gated_delta_rule_numerics(model=None)
    gdr_internal_stats = (
        _install_gdr_internal_timers() if args.profile_gdr_internals else {}
    )

    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
    )
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    patch_gated_delta_rule_numerics(model=backbone)

    if args.loss_path == "packed-splice":
        from bgkit.models.decoder import ReconstructionDecoder

        hidden_dim = backbone.get_input_embeddings().weight.shape[1]
        model = ReconstructionDecoder(backbone, hidden_dim=hidden_dim)
        if args.decoder_lora:
            lora_target_modules = {
                "all": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "mlp": ["gate_proj", "up_proj", "down_proj"],
            }[args.lora_targets]
            lora_config = {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "implementation": args.lora_implementation,
                "target_modules": lora_target_modules,
            }
            if args.lora_fused is not None:
                lora_config["fused"] = bool(args.lora_fused)
            if args.peft_fused_backward is not None:
                lora_config["peft_fused_backward"] = bool(args.peft_fused_backward)
            if args.peft_fuse_gate_up is not None:
                lora_config["peft_fuse_gate_up"] = bool(args.peft_fuse_gate_up)
            model.apply_lora(lora_config)
        if args.decoder_nvfp4:
            if args.decoder_nvfp4_backend == "native-frozen":
                model.enable_native_frozen_nvfp4()
            else:
                model.enable_nvfp4()
        if hasattr(model, "set_lm_ce_impl"):
            model.set_lm_ce_impl(args.ce_impl)
        if args.freeze_decoder:
            model.requires_grad_(False)
        profiled_linear_stats = {}
        profiled_linear_count = 0
        if args.profile_linears:
            profiled_linear_stats, profiled_linear_count = _install_profiled_frozen_linears(
                model.backbone,
                target_names=_all_qwen_projection_names(),
            )
        fused_attention_qkv_count = (
            model.enable_fused_attention_qkv() if args.fused_attention_qkv else 0
        )
        fused_deltanet_zba_count = (
            model.enable_fused_deltanet_zba() if args.fused_deltanet_zba else 0
        )
        fused_deltanet_input_bundle_count = (
            model.enable_fused_deltanet_input_bundle()
            if args.fused_deltanet_input_bundle
            else 0
        )
        frozen_deltanet_core_bwd_count = (
            model.enable_frozen_deltanet_core_bwd()
            if args.frozen_deltanet_core_bwd
            else 0
        )
        frozen_deltanet_residual_bwd_count = (
            model.enable_frozen_deltanet_residual_bwd()
            if args.frozen_deltanet_residual_bwd
            else 0
        )
        frozen_deltanet_channel_last_conv_count = (
            model.enable_frozen_deltanet_channel_last_conv()
            if args.frozen_deltanet_channel_last_conv
            else 0
        )
        frozen_linear_dx_targets = None
        if args.frozen_linear_dx_targets == "all-qwen":
            frozen_linear_dx_targets = (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_b",
                "in_proj_a",
                "out_proj",
            )
        frozen_linear_dx_count = (
            model.enable_frozen_linear_dx(target_modules=frozen_linear_dx_targets)
            if args.frozen_linear_dx
            else 0
        )
        frozen_mlp_fused_count = (
            model.enable_frozen_mlp_fusion() if args.frozen_mlp_fusion else 0
        )
        frozen_mlp_swiglu_fused_count = (
            model.enable_frozen_mlp_swiglu_fusion()
            if args.frozen_mlp_swiglu_fusion
            else 0
        )
        frozen_mlp_residual_fused_count = (
            model.enable_frozen_mlp_residual_fusion()
            if args.frozen_mlp_residual_fusion
            else 0
        )
    else:
        if args.decoder_lora:
            raise ValueError("--decoder-lora currently requires --loss-path packed-splice.")
        if args.decoder_nvfp4:
            raise ValueError("--decoder-nvfp4 currently requires --loss-path packed-splice.")
        if args.freeze_decoder:
            raise ValueError("--freeze-decoder currently requires --loss-path packed-splice.")
        model = backbone
        fused_attention_qkv_count = 0
        fused_deltanet_zba_count = 0
        fused_deltanet_input_bundle_count = 0
        frozen_deltanet_core_bwd_count = 0
        frozen_deltanet_residual_bwd_count = 0
        frozen_deltanet_channel_last_conv_count = 0
        frozen_linear_dx_count = 0
        frozen_mlp_fused_count = 0
        frozen_mlp_swiglu_fused_count = 0
        frozen_mlp_residual_fused_count = 0
        profiled_linear_stats = {}
        profiled_linear_count = 0

    if args.gradient_checkpointing:
        target = model.backbone if hasattr(model, "backbone") else model
        target.gradient_checkpointing_enable()
        if hasattr(target.config, "use_cache"):
            target.config.use_cache = False
    else:
        target = model.backbone if hasattr(model, "backbone") else model
        if hasattr(target.config, "use_cache"):
            target.config.use_cache = False

    if args.use_liger:
        from bgkit.utils.liger_integration import apply_liger_to_qwen35

        apply_liger_to_qwen35(
            model,
            patch_rmsnorm=False,
            patch_swiglu=True,
            patch_rope=True,
        )
        if hasattr(model, "enable_liger_ce"):
            model.enable_liger_ce(True)

    model.train()
    text = _read_text(args)
    if args.loss_path == "packed-splice":
        batch = _build_packed_splice_batch(
            model=model,
            tokenizer=tokenizer,
            text=text,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=device,
            survivor_len=args.survivor_len,
            train_survivors=args.freeze_decoder or args.train_survivors,
        )
        if args.omit_packed_splice_metadata:
            batch.pop("survivor_cu_seqlens_cpu", None)
            batch.pop("packed_cu_seqlens", None)
            batch.pop("packed_position_ids", None)
        survivor_embeddings = batch["survivor_embeddings"]
        survivor_requires_grad = (
            isinstance(survivor_embeddings, torch.Tensor)
            and bool(survivor_embeddings.requires_grad)
        )

        def zero_extra_grads() -> None:
            if isinstance(survivor_embeddings, torch.Tensor):
                survivor_embeddings.grad = None

        def forward_loss() -> torch.Tensor:
            return model.forward_with_single_splice(**batch, chunk_size=args.ce_chunk_size)

    else:
        batch = _build_hf_batch(
            tokenizer=tokenizer,
            text=text,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=device,
        )

        def forward_loss() -> torch.Tensor:
            return model(**batch).loss

        zero_extra_grads = None
        survivor_requires_grad = False

    torch_compile_forward_loss_error = None
    if args.compile_forward_loss:
        try:
            forward_loss = torch.compile(
                forward_loss,
                backend=args.compile_backend,
                mode=None if args.compile_mode == "default" else args.compile_mode,
                fullgraph=False,
            )
        except Exception as exc:
            torch_compile_forward_loss_error = repr(exc)
            raise

    env = describe_backend_environment()
    active_backend = _active_backend(args.backend, resolved_backend_name())
    n_gdr = _count_gdr_layers(model)
    tokens = args.batch_size * args.seq_len
    env_save_intermediates = os.environ.get("FLA_GDR_SAVE_INTERMEDIATES", "<default>")
    env_save_local_attention = os.environ.get("FLA_GDR_SAVE_LOCAL_ATTENTION", "<default>")
    env_recompute_wy_dw = os.environ.get("FLA_GDR_RECOMPUTE_WY_DW", "<default>")
    env_fuse_wy_dg_cumsum = os.environ.get("FLA_GDR_FUSE_WY_DG_CUMSUM", "<default>")
    env_fuse_dqkg_wy = os.environ.get("FLA_GDR_FUSE_DQKG_WY", "<default>")
    env_fuse_dqkg_wy_varlen = os.environ.get(
        "FLA_GDR_FUSE_DQKG_WY_VARLEN",
        "<default>",
    )
    env_fuse_gate_bwd = os.environ.get("FLA_GDR_FUSE_GATE_BWD", "<default>")
    env_packed_dproj_bwd = os.environ.get("FLA_GDR_PACKED_DPROJ_BWD", "<default>")
    env_packed_dproj_contiguous_returns = os.environ.get(
        "FLA_GDR_PACKED_DPROJ_CONTIGUOUS_RETURNS",
        "<default>",
    )
    env_gate_param_grads = os.environ.get("FLA_GDR_GATE_PARAM_GRADS", "<default>")
    env_fused_dqkg_wy_warps = os.environ.get(
        "FLA_GDR_FUSED_DQKG_WY_WARPS",
        "<default>",
    )
    env_fused_dqkg_wy_stages = os.environ.get(
        "FLA_GDR_FUSED_DQKG_WY_STAGES",
        "<default>",
    )
    env_fused_gate_raw_dg_native_dtype = os.environ.get(
        "FLA_GDR_FUSED_GATE_RAW_DG_NATIVE_DTYPE",
        "<default>",
    )
    env_dproj_fuse_qk_l2norm_bwd = os.environ.get(
        "FLA_GDR_DPROJ_FUSE_QK_L2NORM_BWD",
        "<default>",
    )
    env_fuse_qk_l2norm_bwd = os.environ.get(
        "FLA_GDR_FUSE_QK_L2NORM_BWD",
        "<default>",
    )
    env_inplace_qk_l2norm_bwd = os.environ.get(
        "FLA_GDR_INPLACE_QK_L2NORM_BWD",
        "<default>",
    )
    env_pair_qk_l2norm_fwd = os.environ.get(
        "FLA_GDR_PAIR_QK_L2NORM_FWD",
        "<default>",
    )
    env_fuse_kkt_wu = os.environ.get("FLA_GDR_FUSE_KKT_WU", "<default>")
    env_state_dkdg = os.environ.get("FLA_GDR_STATE_DKDG", "<default>")
    env_state_dkdg_fullv = os.environ.get("FLA_GDR_STATE_DKDG_FULLV", "<default>")
    env_state_dkdg_bv = os.environ.get("FLA_GDR_STATE_DKDG_BV", "<default>")
    env_fullk_dqkg = os.environ.get("FLA_GDR_FULLK_DQKG", "<default>")
    env_fuse_fwd_h_o = os.environ.get("FLA_GDR_FUSE_FWD_H_O", "<default>")
    env_deltanet_raw_gate_in_kernel = os.environ.get(
        "BGKIT_DELTANET_RAW_GATE_IN_KERNEL",
        "<default>",
    )
    env_sm121_output = os.environ.get("FLA_USE_SM121_CUSTOM_KERNEL", "<default>")
    env_dqkwg_warps = os.environ.get("FLA_DQKWG_TL_NUM_WARPS", "<default>")
    env_dqkwg_bk = os.environ.get("FLA_DQKWG_TL_BK", "<default>")
    env_dqkwg_bv = os.environ.get("FLA_DQKWG_TL_BV", "<default>")
    env_ce_strict = os.environ.get("BGKIT_DECODER_CE_STRICT", "<default>")
    env_force_liger_ce = os.environ.get("BGKIT_FORCE_LIGER_CE", "<default>")
    env_disable_liger_ce = os.environ.get("BGKIT_DISABLE_LIGER_CE", "<default>")
    env_lora_triton_dx = os.environ.get("BGKIT_DECODER_LORA_TRITON_DX", "<default>")
    env_mlp_base_dx = os.environ.get("BGKIT_DECODER_MLP_BASE_DX", "<default>")
    env_mlp_swiglu_triton_fwd = os.environ.get(
        "BGKIT_DECODER_MLP_SWIGLU_TRITON_FWD",
        "<default>",
    )
    env_mlp_quack = os.environ.get("BGKIT_DECODER_MLP_QUACK", "<default>")
    env_mlp_quack_tuned = os.environ.get("BGKIT_DECODER_MLP_QUACK_TUNED", "<default>")
    env_mlp_quack_min_rows = os.environ.get(
        "BGKIT_DECODER_MLP_QUACK_MIN_ROWS",
        "<default>",
    )
    env_frozen_deltanet_channel_last_conv = os.environ.get(
        "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV",
        "<default>",
    )
    env_frozen_deltanet_channel_last_conv_dx = os.environ.get(
        "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX",
        "<default>",
    )
    env_frozen_deltanet_channel_last_conv_backend = os.environ.get(
        "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_BACKEND",
        "<default>",
    )
    env_frozen_deltanet_channel_last_bundle_input = os.environ.get(
        "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_BUNDLE_INPUT",
        "<default>",
    )
    env_frozen_deltanet_channel_last_pre_l2norm = os.environ.get(
        "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_PRE_L2NORM",
        "<default>",
    )
    env_frozen_deltanet_channel_last_reset_conv = os.environ.get(
        "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_RESET_CONV",
        "<default>",
    )
    env_frozen_deltanet_fused_qkv_conv_l2norm = os.environ.get(
        "BGKIT_FROZEN_DELTANET_FUSED_QKV_CONV_L2NORM",
        "<default>",
    )
    env_frozen_deltanet_stock_fused_qkv_conv_l2norm = os.environ.get(
        "BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_L2NORM",
        "<default>",
    )
    env_frozen_deltanet_stock_fused_qkv_conv_l2norm_dx = os.environ.get(
        "BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_L2NORM_DX",
        "<default>",
    )
    env_frozen_deltanet_stock_fused_qkv_conv_split_dx = os.environ.get(
        "BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_SPLIT_DX",
        "<default>",
    )
    env_frozen_deltanet_triton_qkv_split = os.environ.get(
        "BGKIT_FROZEN_DELTANET_TRITON_QKV_SPLIT",
        "<default>",
    )
    env_frozen_deltanet_triton_qkv_l2norm_split = os.environ.get(
        "BGKIT_FROZEN_DELTANET_TRITON_QKV_L2NORM_SPLIT",
        "<default>",
    )
    env_frozen_deltanet_triton_ba_dx = os.environ.get(
        "BGKIT_FROZEN_DELTANET_TRITON_BA_DX",
        "<default>",
    )
    env_frozen_deltanet_addmm_z_dx = os.environ.get(
        "BGKIT_FROZEN_DELTANET_ADDMM_Z_DX",
        "<default>",
    )
    env_frozen_deltanet_bundle_dx = os.environ.get(
        "BGKIT_FROZEN_DELTANET_BUNDLE_DX",
        "<default>",
    )
    env_frozen_deltanet_triton_proj_dx = os.environ.get(
        "BGKIT_FROZEN_DELTANET_TRITON_PROJ_DX",
        "<default>",
    )
    env_frozen_deltanet_triton_zba_dx = os.environ.get(
        "BGKIT_FROZEN_DELTANET_TRITON_ZBA_DX",
        "<default>",
    )
    env_frozen_deltanet_wide_dproj_dx = os.environ.get(
        "BGKIT_FROZEN_DELTANET_WIDE_DPROJ_DX",
        "<default>",
    )
    env_frozen_deltanet_wide_dproj_scratch_qkv = os.environ.get(
        "BGKIT_FROZEN_DELTANET_WIDE_DPROJ_SCRATCH_QKV",
        "<default>",
    )
    env_frozen_deltanet_original_fwd_recompute_bwd = os.environ.get(
        "BGKIT_FROZEN_DELTANET_ORIGINAL_FWD_RECOMPUTE_BWD",
        "<default>",
    )
    env_frozen_deltanet_residual_mlp_bwd = os.environ.get(
        "BGKIT_FROZEN_DELTANET_RESIDUAL_MLP_BWD",
        "<default>",
    )
    env_frozen_deltanet_input_rmsnorm_dx = os.environ.get(
        "BGKIT_FROZEN_DELTANET_INPUT_RMSNORM_DX",
        "<default>",
    )
    env_frozen_deltanet_core_timers = os.environ.get(
        "BGKIT_FROZEN_DELTANET_CORE_TIMERS",
        "<default>",
    )
    env_frozen_deltanet_zba_dx_block_m = os.environ.get(
        "BGKIT_FROZEN_DELTANET_ZBA_DX_BLOCK_M",
        "<default>",
    )
    env_frozen_deltanet_zba_dx_block_k = os.environ.get(
        "BGKIT_FROZEN_DELTANET_ZBA_DX_BLOCK_K",
        "<default>",
    )
    env_frozen_deltanet_zba_dx_block_z = os.environ.get(
        "BGKIT_FROZEN_DELTANET_ZBA_DX_BLOCK_Z",
        "<default>",
    )
    env_qkv_conv_l2norm_block_rows = os.environ.get(
        "BGKIT_QKV_CONV_L2NORM_BLOCK_ROWS",
        "<default>",
    )
    env_qkv_conv_l2norm_dx_block_rows = os.environ.get(
        "BGKIT_QKV_CONV_L2NORM_DX_BLOCK_ROWS",
        "<default>",
    )
    lora_fused_display = (
        args.lora_fused
        if args.lora_fused is not None
        else os.environ.get("BGKIT_DECODER_LORA_FUSED", "<default>")
    )
    peft_fused_backward_display = (
        args.peft_fused_backward
        if args.peft_fused_backward is not None
        else os.environ.get("BGKIT_DECODER_PEFT_FUSED_BACKWARD", "<default>")
    )
    cce_available: bool | None = None
    expected_ce_path = args.ce_impl
    liger_ce_estimated_chunks: int | None = None
    liger_ce_max_internal_chunks = os.environ.get(
        "BGKIT_LIGER_CE_MAX_INTERNAL_CHUNKS",
        "64",
    )
    liger_ce_should_use_fused: bool | None = None
    if _cce_requested(args.ce_impl):
        from bgkit.utils.cce_integration import is_cut_cross_entropy_available

        cce_available = is_cut_cross_entropy_available()
        expected_ce_path = "cut_cross_entropy" if cce_available else "chunked_fallback"
    elif args.ce_impl == "frozen_chunked":
        expected_ce_path = "frozen_chunked"
    elif args.ce_impl in {"auto", "liger"}:
        use_liger_ce = args.ce_impl == "liger" or (args.ce_impl == "auto" and args.use_liger)
        if not use_liger_ce:
            expected_ce_path = "chunked"
        else:
            from bgkit.utils.liger_integration import (
                _estimated_liger_ce_chunks,
                _should_use_liger_ce,
                is_liger_available,
            )

            if args.loss_path == "packed-splice":
                ce_tokens = args.batch_size * (args.seq_len + args.survivor_len) - 1
            else:
                ce_tokens = args.batch_size * args.seq_len - 1
            lm_head = model.backbone.lm_head if hasattr(model, "backbone") else model.lm_head
            hidden_dim = int(lm_head.weight.shape[1])
            vocab_size = int(lm_head.weight.shape[0])
            liger_ce_estimated_chunks = _estimated_liger_ce_chunks(
                num_tokens=ce_tokens,
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
            )
            liger_ce_should_use_fused = _should_use_liger_ce(
                num_tokens=ce_tokens,
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
            )
            if not is_liger_available():
                expected_ce_path = "chunked_liger_unavailable"
            elif liger_ce_should_use_fused:
                expected_ce_path = "liger_fused"
            else:
                expected_ce_path = "chunked_liger_guard"
    print(
        textwrap.dedent(
            f"""
            qwen_decoder_gdr_benchmark
              model={args.model}
              device={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}
              dtype={args.dtype} batch={args.batch_size} seq_len={args.seq_len} tokens/step={tokens}
              loss_path={args.loss_path} ce_impl={args.ce_impl} expected_ce_path={expected_ce_path}
              ce_chunk_size={args.ce_chunk_size} cce_available={cce_available}
              liger_ce_estimated_chunks={liger_ce_estimated_chunks}
              liger_ce_max_internal_chunks={liger_ce_max_internal_chunks}
              liger_ce_should_use_fused={liger_ce_should_use_fused}
              requested_backend={args.backend} active_backend={active_backend}
              gdr_layers={n_gdr}
              FLA_GDR_SAVE_INTERMEDIATES={env_save_intermediates}
              FLA_GDR_SAVE_LOCAL_ATTENTION={env_save_local_attention}
              FLA_GDR_RECOMPUTE_WY_DW={env_recompute_wy_dw}
              FLA_GDR_FUSE_WY_DG_CUMSUM={env_fuse_wy_dg_cumsum}
              FLA_GDR_FUSE_DQKG_WY={env_fuse_dqkg_wy}
              FLA_GDR_FUSE_DQKG_WY_VARLEN={env_fuse_dqkg_wy_varlen}
              FLA_GDR_FUSE_GATE_BWD={env_fuse_gate_bwd}
              FLA_GDR_PACKED_DPROJ_BWD={env_packed_dproj_bwd}
              FLA_GDR_PACKED_DPROJ_CONTIGUOUS_RETURNS={env_packed_dproj_contiguous_returns}
              FLA_GDR_GATE_PARAM_GRADS={env_gate_param_grads}
              FLA_GDR_FUSED_DQKG_WY_WARPS={env_fused_dqkg_wy_warps}
              FLA_GDR_FUSED_DQKG_WY_STAGES={env_fused_dqkg_wy_stages}
              FLA_GDR_FUSED_GATE_RAW_DG_NATIVE_DTYPE={env_fused_gate_raw_dg_native_dtype}
              FLA_GDR_DPROJ_FUSE_QK_L2NORM_BWD={env_dproj_fuse_qk_l2norm_bwd}
              FLA_GDR_FUSE_QK_L2NORM_BWD={env_fuse_qk_l2norm_bwd}
              FLA_GDR_INPLACE_QK_L2NORM_BWD={env_inplace_qk_l2norm_bwd}
              FLA_GDR_PAIR_QK_L2NORM_FWD={env_pair_qk_l2norm_fwd}
              FLA_GDR_FUSE_KKT_WU={env_fuse_kkt_wu}
              FLA_GDR_STATE_DKDG={env_state_dkdg}
              FLA_GDR_STATE_DKDG_FULLV={env_state_dkdg_fullv}
              FLA_GDR_STATE_DKDG_BV={env_state_dkdg_bv}
              FLA_GDR_FULLK_DQKG={env_fullk_dqkg}
              FLA_GDR_FUSE_FWD_H_O={env_fuse_fwd_h_o}
              BGKIT_DELTANET_RAW_GATE_IN_KERNEL={env_deltanet_raw_gate_in_kernel}
              FLA_USE_SM121_CUSTOM_KERNEL={env_sm121_output}
              FLA_DQKWG_TL_NUM_WARPS={env_dqkwg_warps}
              FLA_DQKWG_TL_BK={env_dqkwg_bk}
              FLA_DQKWG_TL_BV={env_dqkwg_bv}
              BGKIT_DECODER_CE_STRICT={env_ce_strict}
              BGKIT_FORCE_LIGER_CE={env_force_liger_ce}
              BGKIT_DISABLE_LIGER_CE={env_disable_liger_ce}
              BGKIT_DECODER_LORA_TRITON_DX={env_lora_triton_dx}
              BGKIT_DECODER_MLP_BASE_DX={env_mlp_base_dx}
              BGKIT_DECODER_MLP_SWIGLU_TRITON_FWD={env_mlp_swiglu_triton_fwd}
              BGKIT_DECODER_MLP_QUACK={env_mlp_quack}
              BGKIT_DECODER_MLP_QUACK_TUNED={env_mlp_quack_tuned}
              BGKIT_DECODER_MLP_QUACK_MIN_ROWS={env_mlp_quack_min_rows}
              BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV={env_frozen_deltanet_channel_last_conv}
              BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX={env_frozen_deltanet_channel_last_conv_dx}
              BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_BACKEND={env_frozen_deltanet_channel_last_conv_backend}
              BGKIT_FROZEN_DELTANET_CHANNEL_LAST_BUNDLE_INPUT={env_frozen_deltanet_channel_last_bundle_input}
              BGKIT_FROZEN_DELTANET_CHANNEL_LAST_PRE_L2NORM={env_frozen_deltanet_channel_last_pre_l2norm}
              BGKIT_FROZEN_DELTANET_CHANNEL_LAST_RESET_CONV={env_frozen_deltanet_channel_last_reset_conv}
              BGKIT_FROZEN_DELTANET_FUSED_QKV_CONV_L2NORM={env_frozen_deltanet_fused_qkv_conv_l2norm}
              BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_L2NORM={env_frozen_deltanet_stock_fused_qkv_conv_l2norm}
              BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_L2NORM_DX={env_frozen_deltanet_stock_fused_qkv_conv_l2norm_dx}
              BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_SPLIT_DX={env_frozen_deltanet_stock_fused_qkv_conv_split_dx}
              BGKIT_FROZEN_DELTANET_TRITON_QKV_SPLIT={env_frozen_deltanet_triton_qkv_split}
              BGKIT_FROZEN_DELTANET_TRITON_QKV_L2NORM_SPLIT={env_frozen_deltanet_triton_qkv_l2norm_split}
              BGKIT_FROZEN_DELTANET_TRITON_BA_DX={env_frozen_deltanet_triton_ba_dx}
              BGKIT_FROZEN_DELTANET_ADDMM_Z_DX={env_frozen_deltanet_addmm_z_dx}
              BGKIT_FROZEN_DELTANET_BUNDLE_DX={env_frozen_deltanet_bundle_dx}
              BGKIT_FROZEN_DELTANET_TRITON_PROJ_DX={env_frozen_deltanet_triton_proj_dx}
              BGKIT_FROZEN_DELTANET_TRITON_ZBA_DX={env_frozen_deltanet_triton_zba_dx}
              BGKIT_FROZEN_DELTANET_WIDE_DPROJ_DX={env_frozen_deltanet_wide_dproj_dx}
              BGKIT_FROZEN_DELTANET_WIDE_DPROJ_SCRATCH_QKV={env_frozen_deltanet_wide_dproj_scratch_qkv}
              BGKIT_FROZEN_DELTANET_ORIGINAL_FWD_RECOMPUTE_BWD={env_frozen_deltanet_original_fwd_recompute_bwd}
              BGKIT_FROZEN_DELTANET_RESIDUAL_MLP_BWD={env_frozen_deltanet_residual_mlp_bwd}
              BGKIT_FROZEN_DELTANET_INPUT_RMSNORM_DX={env_frozen_deltanet_input_rmsnorm_dx}
              BGKIT_FROZEN_DELTANET_CORE_TIMERS={env_frozen_deltanet_core_timers}
              BGKIT_FROZEN_DELTANET_ZBA_DX_BLOCK_M={env_frozen_deltanet_zba_dx_block_m}
              BGKIT_FROZEN_DELTANET_ZBA_DX_BLOCK_K={env_frozen_deltanet_zba_dx_block_k}
              BGKIT_FROZEN_DELTANET_ZBA_DX_BLOCK_Z={env_frozen_deltanet_zba_dx_block_z}
              BGKIT_QKV_CONV_L2NORM_BLOCK_ROWS={env_qkv_conv_l2norm_block_rows}
              BGKIT_QKV_CONV_L2NORM_DX_BLOCK_ROWS={env_qkv_conv_l2norm_dx_block_rows}
              gradient_checkpointing={args.gradient_checkpointing} use_liger={args.use_liger}
              seed={args.seed}
              decoder_lora={args.decoder_lora} lora_impl={args.lora_implementation}
              lora_r={args.lora_r} lora_alpha={args.lora_alpha} lora_targets={args.lora_targets}
              freeze_decoder={args.freeze_decoder} survivor_len={args.survivor_len}
              train_survivors={args.train_survivors}
              survivor_requires_grad={survivor_requires_grad}
              fused_attention_qkv={args.fused_attention_qkv}
              fused_attention_qkv_count={fused_attention_qkv_count}
              fused_deltanet_zba={args.fused_deltanet_zba}
              fused_deltanet_zba_count={fused_deltanet_zba_count}
              fused_deltanet_input_bundle={args.fused_deltanet_input_bundle}
              fused_deltanet_input_bundle_count={fused_deltanet_input_bundle_count}
              frozen_deltanet_core_bwd={args.frozen_deltanet_core_bwd}
              frozen_deltanet_core_bwd_count={frozen_deltanet_core_bwd_count}
              frozen_deltanet_residual_bwd={args.frozen_deltanet_residual_bwd}
              frozen_deltanet_residual_bwd_count={frozen_deltanet_residual_bwd_count}
              frozen_deltanet_residual_mlp_bwd={args.frozen_deltanet_residual_mlp_bwd}
              frozen_deltanet_channel_last_conv={args.frozen_deltanet_channel_last_conv}
              frozen_deltanet_channel_last_conv_count={frozen_deltanet_channel_last_conv_count}
              frozen_linear_dx={args.frozen_linear_dx}
              frozen_linear_dx_targets={args.frozen_linear_dx_targets}
              frozen_linear_dx_count={frozen_linear_dx_count}
              profile_linears={args.profile_linears}
              profiled_linear_count={profiled_linear_count}
              profile_gdr_internals={args.profile_gdr_internals}
              compile_forward_loss={args.compile_forward_loss}
              compile_backend={args.compile_backend}
              compile_mode={args.compile_mode}
              frozen_mlp_fusion={args.frozen_mlp_fusion}
              frozen_mlp_fused_count={frozen_mlp_fused_count}
              frozen_mlp_swiglu_fusion={args.frozen_mlp_swiglu_fusion}
              frozen_mlp_swiglu_fused_count={frozen_mlp_swiglu_fused_count}
              frozen_mlp_residual_fusion={args.frozen_mlp_residual_fusion}
              frozen_mlp_residual_fused_count={frozen_mlp_residual_fused_count}
              lora_fused={lora_fused_display}
              peft_fused_backward={peft_fused_backward_display}
              peft_fuse_gate_up={
                  args.peft_fuse_gate_up
                  if args.peft_fuse_gate_up is not None
                  else os.environ.get("BGKIT_DECODER_PEFT_FUSE_GATE_UP", "<default>")
              }
              lora_dropout={args.lora_dropout}
              decoder_nvfp4={args.decoder_nvfp4} decoder_nvfp4_backend={args.decoder_nvfp4_backend}
            """
        ).strip()
    )
    print(f"backend_env={json.dumps(env, sort_keys=True)}")

    fwd_ms: list[float] = []
    bwd_ms: list[float] = []
    total_ms: list[float] = []
    losses: list[float] = []
    gpu_telemetry_samples: list[dict[str, float | str | None]] = []
    gpu_telemetry_stop: threading.Event | None = None
    gpu_telemetry_thread: threading.Thread | None = None
    if args.gpu_telemetry:
        gpu_telemetry_stop, gpu_telemetry_thread, gpu_telemetry_samples = (
            _start_gpu_telemetry_sampler(max(float(args.gpu_telemetry_interval), 0.05))
        )
    torch.cuda.reset_peak_memory_stats()
    cuda_graph: torch.cuda.CUDAGraph | None = None
    cuda_graph_loss: list[torch.Tensor | None] | None = None
    if args.cuda_graph:
        graph_extra_tensors: tuple[torch.Tensor, ...] = ()
        if (
            args.loss_path == "packed-splice"
            and isinstance(survivor_embeddings, torch.Tensor)
            and bool(survivor_embeddings.requires_grad)
        ):
            graph_extra_tensors = (survivor_embeddings,)
        try:
            cuda_graph, cuda_graph_loss = _capture_cuda_graph_step(
                model,
                forward_loss,
                extra_grad_tensors=graph_extra_tensors,
            )
        except Exception as exc:
            raise RuntimeError(
                "CUDA graph capture failed for this decoder benchmark shape"
            ) from exc
    try:
        for step in range(args.warmup + args.steps):
            if step == args.warmup and profiled_linear_stats:
                _clear_profiled_linear_stats(profiled_linear_stats)
            if step == args.warmup and gdr_internal_stats:
                _clear_gdr_internal_timers(gdr_internal_stats)
            if (
                step == args.warmup
                and args.frozen_deltanet_core_bwd
                and hasattr(model, "reset_frozen_deltanet_core_bwd_stats")
            ):
                model.reset_frozen_deltanet_core_bwd_stats()
            if args.cuda_graph:
                if cuda_graph is None or cuda_graph_loss is None:
                    raise RuntimeError("CUDA graph mode was selected but not captured")
                fwd, bwd, total, loss = _run_cuda_graph_replay(
                    cuda_graph,
                    cuda_graph_loss,
                )
            else:
                fwd, bwd, total, loss = _run_step(
                    model,
                    forward_loss,
                    zero_extra_grads=zero_extra_grads,
                )
            if step >= args.warmup:
                fwd_ms.append(fwd)
                bwd_ms.append(bwd)
                total_ms.append(total)
                losses.append(loss)
            print(
                f"step={step:03d} phase={'warmup' if step < args.warmup else 'measure'} "
                f"loss={loss:.4f} fwd_ms={fwd:.3f} bwd_ms={bwd:.3f} total_ms={total:.3f}",
                flush=True,
            )
    finally:
        if gpu_telemetry_stop is not None:
            gpu_telemetry_stop.set()
        if gpu_telemetry_thread is not None:
            gpu_telemetry_thread.join(timeout=2.0)

    frozen_deltanet_core_bwd_stats = (
        model.frozen_deltanet_core_bwd_stats()
        if args.frozen_deltanet_core_bwd
        and hasattr(model, "frozen_deltanet_core_bwd_stats")
        else None
    )
    summary = {
        "model": args.model,
        "backend": args.backend,
        "active_backend": _active_backend(args.backend, resolved_backend_name()),
        "dtype": args.dtype,
        "loss_path": args.loss_path,
        "ce_impl": args.ce_impl,
        "ce_strict": args.ce_strict,
        "ce_chunk_size": args.ce_chunk_size,
        "compile_forward_loss": args.compile_forward_loss,
        "compile_backend": args.compile_backend,
        "compile_mode": args.compile_mode,
        "compile_forward_loss_error": torch_compile_forward_loss_error,
        "cce_available": cce_available,
        "expected_ce_path": expected_ce_path,
        "liger_ce_estimated_chunks": liger_ce_estimated_chunks,
        "liger_ce_max_internal_chunks": liger_ce_max_internal_chunks,
        "liger_ce_should_use_fused": liger_ce_should_use_fused,
        "decoder_lora": args.decoder_lora,
        "freeze_decoder": args.freeze_decoder,
        "survivor_len": args.survivor_len,
        "train_survivors": args.train_survivors,
        "survivor_requires_grad": survivor_requires_grad,
        "omit_packed_splice_metadata": args.omit_packed_splice_metadata,
        "fused_attention_qkv": args.fused_attention_qkv,
        "fused_attention_qkv_count": fused_attention_qkv_count,
        "fused_deltanet_zba": args.fused_deltanet_zba,
        "fused_deltanet_zba_count": fused_deltanet_zba_count,
        "fused_deltanet_input_bundle": args.fused_deltanet_input_bundle,
        "fused_deltanet_input_bundle_count": fused_deltanet_input_bundle_count,
        "frozen_deltanet_core_bwd": args.frozen_deltanet_core_bwd,
        "frozen_deltanet_core_bwd_count": frozen_deltanet_core_bwd_count,
        "frozen_deltanet_core_bwd_stats": frozen_deltanet_core_bwd_stats,
        "frozen_deltanet_residual_bwd": args.frozen_deltanet_residual_bwd,
        "frozen_deltanet_residual_bwd_count": frozen_deltanet_residual_bwd_count,
        "frozen_deltanet_residual_mlp_bwd": args.frozen_deltanet_residual_mlp_bwd,
        "frozen_deltanet_channel_last_conv": args.frozen_deltanet_channel_last_conv,
        "frozen_deltanet_channel_last_conv_count": frozen_deltanet_channel_last_conv_count,
        "frozen_linear_dx": args.frozen_linear_dx,
        "frozen_linear_dx_targets": args.frozen_linear_dx_targets,
        "frozen_linear_dx_count": frozen_linear_dx_count,
        "profile_linears": args.profile_linears,
        "profiled_linear_count": profiled_linear_count,
        "profile_gdr_internals": args.profile_gdr_internals,
        "cuda_graph": args.cuda_graph,
        "cuda_graph_total_only": args.cuda_graph,
        "gpu_telemetry": args.gpu_telemetry,
        "gpu_telemetry_interval": args.gpu_telemetry_interval,
        "frozen_mlp_fusion": args.frozen_mlp_fusion,
        "frozen_mlp_fused_count": frozen_mlp_fused_count,
        "frozen_mlp_swiglu_fusion": args.frozen_mlp_swiglu_fusion,
        "frozen_mlp_swiglu_fused_count": frozen_mlp_swiglu_fused_count,
        "frozen_mlp_residual_fusion": args.frozen_mlp_residual_fusion,
        "frozen_mlp_residual_fused_count": frozen_mlp_residual_fused_count,
        "lora_implementation": args.lora_implementation,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_targets": args.lora_targets,
        "lora_fused": args.lora_fused
        if args.lora_fused is not None
        else os.environ.get("BGKIT_DECODER_LORA_FUSED"),
        "peft_fused_backward": args.peft_fused_backward
        if args.peft_fused_backward is not None
        else os.environ.get("BGKIT_DECODER_PEFT_FUSED_BACKWARD"),
        "peft_fuse_gate_up": args.peft_fuse_gate_up
        if args.peft_fuse_gate_up is not None
        else os.environ.get("BGKIT_DECODER_PEFT_FUSE_GATE_UP"),
        "lora_dropout": args.lora_dropout,
        "decoder_nvfp4": args.decoder_nvfp4,
        "decoder_nvfp4_backend": args.decoder_nvfp4_backend,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "tokens_per_step": tokens,
        "steps": args.steps,
        "warmup": args.warmup,
        "seed": args.seed,
        "fwd_ms": _ms_stats(fwd_ms),
        "bwd_ms": _ms_stats(bwd_ms),
        "total_ms": _ms_stats(total_ms),
        "tokens_per_second": tokens / (statistics.median(total_ms) / 1000.0),
        "loss_mean": statistics.mean(losses),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "gdr_layers": n_gdr,
        "env": {
            "FLA_GDR_SAVE_INTERMEDIATES": env_save_intermediates,
            "FLA_GDR_SAVE_LOCAL_ATTENTION": env_save_local_attention,
            "FLA_GDR_RECOMPUTE_WY_DW": env_recompute_wy_dw,
            "FLA_GDR_FUSE_WY_DG_CUMSUM": env_fuse_wy_dg_cumsum,
            "FLA_GDR_FUSE_DQKG_WY": env_fuse_dqkg_wy,
            "FLA_GDR_FUSE_DQKG_WY_VARLEN": env_fuse_dqkg_wy_varlen,
            "FLA_GDR_FUSE_GATE_BWD": env_fuse_gate_bwd,
            "FLA_GDR_PACKED_DPROJ_BWD": env_packed_dproj_bwd,
            "FLA_GDR_PACKED_DPROJ_CONTIGUOUS_RETURNS": (
                env_packed_dproj_contiguous_returns
            ),
            "FLA_GDR_GATE_PARAM_GRADS": env_gate_param_grads,
            "FLA_GDR_FUSED_DQKG_WY_WARPS": env_fused_dqkg_wy_warps,
            "FLA_GDR_FUSED_DQKG_WY_STAGES": env_fused_dqkg_wy_stages,
            "FLA_GDR_FUSED_GATE_RAW_DG_NATIVE_DTYPE": env_fused_gate_raw_dg_native_dtype,
            "FLA_GDR_DPROJ_FUSE_QK_L2NORM_BWD": env_dproj_fuse_qk_l2norm_bwd,
            "FLA_GDR_FUSE_QK_L2NORM_BWD": env_fuse_qk_l2norm_bwd,
            "FLA_GDR_INPLACE_QK_L2NORM_BWD": env_inplace_qk_l2norm_bwd,
            "FLA_GDR_PAIR_QK_L2NORM_FWD": env_pair_qk_l2norm_fwd,
            "FLA_GDR_FUSE_KKT_WU": env_fuse_kkt_wu,
            "FLA_GDR_STATE_DKDG": env_state_dkdg,
            "FLA_GDR_STATE_DKDG_FULLV": env_state_dkdg_fullv,
            "FLA_GDR_STATE_DKDG_BV": env_state_dkdg_bv,
            "FLA_GDR_FULLK_DQKG": env_fullk_dqkg,
            "FLA_GDR_FUSE_FWD_H_O": env_fuse_fwd_h_o,
            "BGKIT_DELTANET_RAW_GATE_IN_KERNEL": env_deltanet_raw_gate_in_kernel,
            "FLA_USE_SM121_CUSTOM_KERNEL": env_sm121_output,
            "FLA_DQKWG_TL_NUM_WARPS": env_dqkwg_warps,
            "FLA_DQKWG_TL_BK": env_dqkwg_bk,
            "FLA_DQKWG_TL_BV": env_dqkwg_bv,
            "BGKIT_DECODER_CE_STRICT": env_ce_strict,
            "BGKIT_FORCE_LIGER_CE": env_force_liger_ce,
            "BGKIT_DISABLE_LIGER_CE": env_disable_liger_ce,
            "BGKIT_DECODER_LORA_TRITON_DX": env_lora_triton_dx,
            "BGKIT_DECODER_MLP_BASE_DX": env_mlp_base_dx,
            "BGKIT_DECODER_MLP_SWIGLU_TRITON_FWD": env_mlp_swiglu_triton_fwd,
            "BGKIT_DECODER_MLP_QUACK": env_mlp_quack,
            "BGKIT_DECODER_MLP_QUACK_TUNED": env_mlp_quack_tuned,
            "BGKIT_DECODER_MLP_QUACK_MIN_ROWS": env_mlp_quack_min_rows,
            "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV": env_frozen_deltanet_channel_last_conv,
            "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX": env_frozen_deltanet_channel_last_conv_dx,
            "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_BACKEND": (
                env_frozen_deltanet_channel_last_conv_backend
            ),
            "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_BUNDLE_INPUT": (
                env_frozen_deltanet_channel_last_bundle_input
            ),
            "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_PRE_L2NORM": (
                env_frozen_deltanet_channel_last_pre_l2norm
            ),
            "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_RESET_CONV": (
                env_frozen_deltanet_channel_last_reset_conv
            ),
            "BGKIT_FROZEN_DELTANET_FUSED_QKV_CONV_L2NORM": (
                env_frozen_deltanet_fused_qkv_conv_l2norm
            ),
            "BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_L2NORM": (
                env_frozen_deltanet_stock_fused_qkv_conv_l2norm
            ),
            "BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_L2NORM_DX": (
                env_frozen_deltanet_stock_fused_qkv_conv_l2norm_dx
            ),
            "BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_SPLIT_DX": (
                env_frozen_deltanet_stock_fused_qkv_conv_split_dx
            ),
            "BGKIT_FROZEN_DELTANET_TRITON_QKV_SPLIT": env_frozen_deltanet_triton_qkv_split,
            "BGKIT_FROZEN_DELTANET_TRITON_QKV_L2NORM_SPLIT": (
                env_frozen_deltanet_triton_qkv_l2norm_split
            ),
            "BGKIT_FROZEN_DELTANET_TRITON_BA_DX": env_frozen_deltanet_triton_ba_dx,
            "BGKIT_FROZEN_DELTANET_ADDMM_Z_DX": env_frozen_deltanet_addmm_z_dx,
            "BGKIT_FROZEN_DELTANET_BUNDLE_DX": env_frozen_deltanet_bundle_dx,
            "BGKIT_FROZEN_DELTANET_TRITON_PROJ_DX": env_frozen_deltanet_triton_proj_dx,
            "BGKIT_FROZEN_DELTANET_TRITON_ZBA_DX": env_frozen_deltanet_triton_zba_dx,
            "BGKIT_FROZEN_DELTANET_WIDE_DPROJ_DX": env_frozen_deltanet_wide_dproj_dx,
            "BGKIT_FROZEN_DELTANET_WIDE_DPROJ_SCRATCH_QKV": (
                env_frozen_deltanet_wide_dproj_scratch_qkv
            ),
            "BGKIT_FROZEN_DELTANET_ORIGINAL_FWD_RECOMPUTE_BWD": (
                env_frozen_deltanet_original_fwd_recompute_bwd
            ),
            "BGKIT_FROZEN_DELTANET_RESIDUAL_MLP_BWD": (
                env_frozen_deltanet_residual_mlp_bwd
            ),
            "BGKIT_FROZEN_DELTANET_INPUT_RMSNORM_DX": (
                env_frozen_deltanet_input_rmsnorm_dx
            ),
            "BGKIT_FROZEN_DELTANET_CORE_TIMERS": env_frozen_deltanet_core_timers,
            "BGKIT_FROZEN_DELTANET_ZBA_DX_BLOCK_M": env_frozen_deltanet_zba_dx_block_m,
            "BGKIT_FROZEN_DELTANET_ZBA_DX_BLOCK_K": env_frozen_deltanet_zba_dx_block_k,
            "BGKIT_FROZEN_DELTANET_ZBA_DX_BLOCK_Z": env_frozen_deltanet_zba_dx_block_z,
            "BGKIT_QKV_CONV_L2NORM_BLOCK_ROWS": env_qkv_conv_l2norm_block_rows,
            "BGKIT_QKV_CONV_L2NORM_DX_BLOCK_ROWS": (
                env_qkv_conv_l2norm_dx_block_rows
            ),
        },
    }
    if gpu_telemetry_samples:
        summary["gpu_telemetry_samples"] = gpu_telemetry_samples
        summary["gpu_telemetry_summary"] = _summarize_gpu_telemetry(gpu_telemetry_samples)
    if profiled_linear_stats:
        torch.cuda.synchronize()
        linear_profile = _summarize_profiled_linears(
            profiled_linear_stats,
            topk=args.profile_linears_topk,
        )
        summary["profiled_linears"] = linear_profile
        print("profiled_linears=" + json.dumps(linear_profile, sort_keys=True))
    if gdr_internal_stats:
        torch.cuda.synchronize()
        gdr_profile = _summarize_gdr_internal_timers(gdr_internal_stats)
        summary["profiled_gdr_internals"] = gdr_profile
        print("profiled_gdr_internals=" + json.dumps(gdr_profile, sort_keys=True))
    print("summary=" + json.dumps(summary, sort_keys=True))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))

    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        print("profile_start=1", flush=True)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            fwd, bwd, total, loss = _run_step(
                model,
                forward_loss,
                zero_extra_grads=zero_extra_grads,
            )
        print(
            f"profiled_step loss={loss:.4f} fwd_ms={fwd:.3f} "
            f"bwd_ms={bwd:.3f} total_ms={total:.3f}"
        )
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.profile_topk))
        if args.profile_trace:
            prof.export_chrome_trace(args.profile_trace)
            print(f"profile_trace={args.profile_trace}")


if __name__ == "__main__":
    main()
