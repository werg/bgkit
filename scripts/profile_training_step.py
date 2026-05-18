#!/usr/bin/env python
"""Profile one real BgKIT optimizer step from a resumed checkpoint.

This is a step-level timing and optional kernel profiler, not a memory sweep.
It restores the selected training phase, runs a configurable number of warmup
optimizer steps to get JIT compilation out of the measured path, then measures
one or more real optimizer steps. Set ``+profile.capture_profiler=true`` when a
full ``torch.profiler`` table or Chrome trace is needed; the default keeps
timing overhead low.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import sys
import time
import types
from collections import Counter
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from bgkit.training.base_trainer import _average_metrics
from bgkit.training.checkpoint_registry import resolve_latest_checkpoint
from bgkit.training.gradient_utils import clip_grad_norm
from bgkit.training.scheduling import cosine_with_warmup
from bgkit.utils.deltanet_patch import (
    patch_fused_rms_norm_gated_for_sm121,
    patch_gated_delta_rule_numerics,
)
from bgkit.utils.logging import setup_logging
from bgkit.utils.reproducibility import set_seed
from bgkit.utils.step_watchdog import install_step_watchdog
from bgkit.utils.triton_alloc_patch import patch_triton_allocator
from bgkit.utils.triton_patch import patch_triton_autotuner


class _ProfiledFrozenDecoderLinearFunction(torch.autograd.Function):
    """Diagnostic frozen-linear autograd with CUDA event timing."""

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
            ctx.stats.setdefault(ctx.name, {"fwd": [], "bwd": []})["bwd"].append((start, end))
        else:
            grad_x = grad_out.matmul(weight)
        return grad_x.reshape(ctx.x_shape), None, None, None, None


class _ProfiledFrozenDecoderLinear(nn.Module):
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
        if self.weight.requires_grad or (self.bias is not None and self.bias.requires_grad):
            return F.linear(x, self.weight, self.bias)
        if not x.requires_grad:
            return F.linear(x, self.weight, self.bias)
        return _ProfiledFrozenDecoderLinearFunction.apply(
            x,
            self.weight,
            self.bias,
            self._bgkit_profile_name,
            self._bgkit_profile_stats,
        )


_DECODER_LINEAR_PROFILE_TARGETS = {
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
}


def _create_trainer(cfg: DictConfig):
    phase = cfg.get("training", {}).get("phase", None)
    if phase is None:
        raise ValueError("No training phase specified.")

    if phase == "joint_block_pretrain":
        from bgkit.training.joint_block_trainer import JointBlockTrainer

        return JointBlockTrainer(cfg)
    if phase == "phase1_step1":
        from bgkit.training.phase1.decoder_init import DecoderInitTrainer

        return DecoderInitTrainer(cfg)
    if phase == "phase1_step2":
        from bgkit.training.distillation.pruning_distill import PruningDistillTrainer

        return PruningDistillTrainer(cfg)
    if phase == "phase1_step2p5":
        from bgkit.training.phase1.projection_repair import ProjectionRepairTrainer

        return ProjectionRepairTrainer(cfg)
    if phase in ("phase1_step3", "phase1_step4"):
        from bgkit.training.phase1.decoder_init import DecoderInitTrainer

        return DecoderInitTrainer(cfg)
    if phase in ("phase1_falcon_dense_seed", "phase1_falcon_forced_adapt"):
        from bgkit.training.phase1.projection_seed_falcon import (
            FalconProjectionSeedTrainer,
        )

        return FalconProjectionSeedTrainer(cfg)
    if phase in ("phase1_step4p7", "phase1_step4p7_v2", "phase1_step4p7_v3"):
        from bgkit.training.phase1.bridge_distill import BridgeDistillTrainer

        return BridgeDistillTrainer(cfg)
    if phase == "phase1_step5":
        from bgkit.training.phase1.commit_encoding import CommitEncodingTrainer

        return CommitEncodingTrainer(cfg)
    if phase in ("phase1_step6", "phase1_falcon_l0", "phase1_falcon_l1"):
        from bgkit.training.phase1.compression import CompressionTrainer

        return CompressionTrainer(cfg)
    if phase in (
        "phase2",
        "phase2_kb",
        "phase2_kb_stage_a",
        "phase2_kb_stage_b",
        "phase2_kb_stage_a_falcon",
        "phase2_kb_stage_b_falcon",
    ):
        from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

        return KRKBTrainer(cfg)
    if phase == "phase3":
        from bgkit.training.phase3.distillation_trainer import DistillationTrainer

        return DistillationTrainer(cfg)
    raise NotImplementedError(f"Training phase {phase!r} not supported.")


def _resolve_resume_checkpoint(cfg: DictConfig) -> Path | None:
    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    resume_path = cfg.get("resume_checkpoint", None)
    if resume_path == "none":
        return None
    if resume_path is not None:
        return Path(str(resume_path))

    phase = cfg.get("training", {}).get("phase", None)
    if phase:
        resolved = resolve_latest_checkpoint(checkpoint_dir, phase)
        if resolved is not None:
            return resolved

    last_file = checkpoint_dir / ".last_checkpoint"
    if last_file.exists():
        raw = last_file.read_text().strip()
        if raw:
            path = Path(raw)
            if path.exists():
                return path

    return None


def _prepare_trainer(cfg: DictConfig):
    trainer = _create_trainer(cfg)
    trainer.setup()

    checkpoint = _resolve_resume_checkpoint(cfg)
    is_resuming = checkpoint is not None
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint)
        trainer.global_step += 1

    trainer._sync_epoch(trainer.epoch)
    trainer._pre_train_loop()
    _apply_initial_length_filters(trainer, cfg)

    dataloader_iter = None
    cursor_restored = False
    if is_resuming and trainer._microbatches_in_epoch > 0:
        batch_sampler = getattr(trainer.train_dataloader, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "set_batch_cursor"):
            batch_sampler.set_batch_cursor(trainer._microbatches_in_epoch)
            cursor_restored = True

    if is_resuming and trainer._microbatches_in_epoch > 0 and not cursor_restored:
        raw_iter = trainer._create_dataloader_iter(use_prefetch=False)
        skipped = 0
        try:
            for _ in range(int(trainer._microbatches_in_epoch)):
                next(raw_iter)
                skipped += 1
            dataloader_iter = trainer._wrap_dataloader_iter(raw_iter)
        except StopIteration:
            trainer.epoch += 1
            trainer._microbatches_in_epoch = 0
            trainer._sync_epoch(trainer.epoch)

    if dataloader_iter is None:
        dataloader_iter = trainer._create_dataloader_iter()
    return trainer, dataloader_iter, checkpoint


def _apply_initial_length_filters(trainer, cfg: DictConfig) -> None:
    """Mirror BaseTrainer.train()'s initial length filters for profile runs."""

    tcfg = cfg.training
    if not hasattr(trainer, "_max_batch_tokens"):
        return
    min_len = int(tcfg.get("min_sample_length", 0) or 0)
    max_len = int(tcfg.get("max_sample_length", 0) or 0)
    if min_len > 0 and hasattr(trainer, "_handle_min_sample_length"):
        trainer._handle_min_sample_length(min_len)
    if max_len > 0 and hasattr(trainer, "_handle_max_sample_length"):
        trainer._handle_max_sample_length(max_len)


def _schedule_values(trainer, cfg: DictConfig) -> tuple[int, float, int]:
    tcfg = cfg.training
    reset_schedule = bool(tcfg.get("reset_schedule", False))
    if trainer._schedule_params is not None and not reset_schedule:
        max_steps = int(trainer._schedule_params["max_steps"])
        base_lr = float(trainer._schedule_params["base_lr"])
        warmup_steps = int(trainer._schedule_params["warmup_steps"])
        if int(tcfg.max_steps) > max_steps:
            max_steps = int(tcfg.max_steps)
        saved_per_group = trainer._schedule_params.get("per_group_base_lrs")
        if saved_per_group is not None and len(saved_per_group) == len(
            trainer.optimizer.param_groups
        ):
            for pg, lr in zip(trainer.optimizer.param_groups, saved_per_group, strict=True):
                pg["base_lr"] = float(lr)
    else:
        max_steps = int(tcfg.max_steps)
        base_lr = float(tcfg.lr)
        warmup_steps = int(tcfg.warmup_steps)
    trainer._schedule_params = {
        "max_steps": max_steps,
        "base_lr": base_lr,
        "warmup_steps": warmup_steps,
    }
    return max_steps, base_lr, warmup_steps


def _next_batch(trainer, dataloader_iter):
    try:
        batch = next(dataloader_iter)
        trainer._microbatches_in_epoch += 1
        return batch, dataloader_iter
    except StopIteration:
        trainer.epoch += 1
        trainer._microbatches_in_epoch = 0
        trainer._sync_epoch(trainer.epoch)
        dataloader_iter = trainer._create_dataloader_iter()
        batch = next(dataloader_iter)
        trainer._microbatches_in_epoch += 1
        return batch, dataloader_iter


def _batch_profile_stats(batch: dict, prefix: str) -> dict[str, float]:
    stats: dict[str, float] = {}

    content = batch.get("content_token_ids")
    if content is not None:
        stats[f"{prefix}_content_tokens"] = float(content.numel())
    target = batch.get("target_token_ids")
    if target is not None:
        stats[f"{prefix}_target_tokens"] = float(target.numel())
    prompt = batch.get("prompt_token_ids")
    if prompt is not None:
        stats[f"{prefix}_prompt_tokens"] = float(prompt.numel())

    target_cu = batch.get("target_cu_seqlens")
    if target_cu is not None and target_cu.numel() > 1:
        target_lengths = target_cu[1:] - target_cu[:-1]
        stats[f"{prefix}_samples"] = float(target_lengths.numel())
        stats[f"{prefix}_max_target_len"] = float(target_lengths.max().item())
        stats[f"{prefix}_target_l2_cost"] = float(
            (target_lengths.to(torch.float64) ** 2).sum().item(),
        )

    return stats


def _install_decoder_linear_profiler(
    trainer,
) -> tuple[dict[str, dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]], int]:
    decoder = getattr(trainer, "decoder", None)
    root = getattr(decoder, "backbone", decoder)
    if root is None:
        return {}, 0

    stats: dict[str, dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]] = {}
    count = 0
    for parent_name, parent in list(root.named_modules()):
        parent_has_frozen_mlp_boundary = bool(
            getattr(parent, "_bgkit_frozen_mlp_forward", False)
            or getattr(parent, "_bgkit_frozen_mlp_swiglu_forward", False)
        )
        for child_name, child in list(parent.named_children()):
            if child_name not in _DECODER_LINEAR_PROFILE_TARGETS:
                continue
            if parent_has_frozen_mlp_boundary and child_name in {
                "gate_proj",
                "up_proj",
                "down_proj",
            }:
                continue
            if not isinstance(child, nn.Linear):
                continue
            if child.weight.requires_grad or (child.bias is not None and child.bias.requires_grad):
                continue
            if isinstance(child, _ProfiledFrozenDecoderLinear):
                continue
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            setattr(
                parent,
                child_name,
                _ProfiledFrozenDecoderLinear(child, name=full_name, stats=stats),
            )
            count += 1
    return stats, count


def _decoder_layer_family(module: nn.Module) -> str:
    layer_type = str(getattr(module, "layer_type", "unknown"))
    if layer_type == "linear_attention":
        return "deltanet_layer"
    if layer_type == "full_attention":
        return "full_attention_layer"
    return f"{layer_type}_layer"


def _profiled_decoder_layer_forward(module: nn.Module, *args, **kwargs):
    original_forward = getattr(module, "_bgkit_original_profile_layer_forward", None)
    if original_forward is None:
        raise RuntimeError("decoder layer profiler installed without original forward")

    name = str(getattr(module, "_bgkit_profile_layer_name", module.__class__.__name__))
    family = str(getattr(module, "_bgkit_profile_layer_family", "decoder_layer"))
    stats = getattr(module, "_bgkit_profile_layer_stats", None)
    if stats is None:
        return original_forward(*args, **kwargs)

    uses_cuda = _contains_cuda_tensor(args) or _contains_cuda_tensor(kwargs)
    if uses_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = original_forward(*args, **kwargs)
        end.record()
        stats.setdefault(name, {"family": family, "fwd": [], "bwd": []})["fwd"].append((start, end))
    else:
        out = original_forward(*args, **kwargs)

    tensor = out if isinstance(out, torch.Tensor) else None
    if tensor is None and isinstance(out, (tuple, list)) and out:
        first = out[0]
        tensor = first if isinstance(first, torch.Tensor) else None

    if tensor is not None and tensor.is_cuda and tensor.requires_grad:
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        pending = getattr(module, "_bgkit_profile_layer_pending_bwd_events", None)
        if pending is None:
            pending = []
            module._bgkit_profile_layer_pending_bwd_events = pending

        def _record_bwd_start(grad):
            bwd_start.record()
            pending.append((bwd_start, bwd_end))
            return grad

        tensor.register_hook(_record_bwd_start)

    return out


def _decoder_layer_backward_hook(module: nn.Module, _grad_input, _grad_output) -> None:
    name = str(getattr(module, "_bgkit_profile_layer_name", module.__class__.__name__))
    family = str(getattr(module, "_bgkit_profile_layer_family", "decoder_layer"))
    stats = getattr(module, "_bgkit_profile_layer_stats", None)
    pending = getattr(module, "_bgkit_profile_layer_pending_bwd_events", None)
    if stats is None or not pending:
        return
    start, end = pending.pop(0)
    end.record()
    stats.setdefault(name, {"family": family, "fwd": [], "bwd": []})["bwd"].append((start, end))


def _install_decoder_layer_profiler(
    trainer,
) -> tuple[dict[str, dict[str, object]], int]:
    decoder = getattr(trainer, "decoder", None)
    root = getattr(decoder, "backbone", decoder)
    if root is None:
        return {}, 0

    stats: dict[str, dict[str, object]] = {}
    count = 0
    for name, module in list(root.named_modules()):
        if getattr(module, "_bgkit_profile_layer_forward", False):
            count += 1
            continue
        if not (
            hasattr(module, "input_layernorm")
            and hasattr(module, "post_attention_layernorm")
            and hasattr(module, "mlp")
            and hasattr(module, "layer_type")
        ):
            continue
        module._bgkit_original_profile_layer_forward = module.forward
        module._bgkit_profile_layer_forward = True
        module._bgkit_profile_layer_name = name or module.__class__.__name__
        module._bgkit_profile_layer_family = _decoder_layer_family(module)
        module._bgkit_profile_layer_stats = stats
        module._bgkit_profile_layer_pending_bwd_events = []
        module._bgkit_profile_layer_backward_handle = module.register_full_backward_hook(
            _decoder_layer_backward_hook,
        )
        module.forward = types.MethodType(_profiled_decoder_layer_forward, module)
        count += 1
    return stats, count


def _clear_decoder_linear_profile_stats(
    stats: dict[str, dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]],
) -> None:
    for item in stats.values():
        item["fwd"].clear()
        item["bwd"].clear()


def _clear_decoder_layer_profile_stats(stats: dict[str, dict[str, object]]) -> None:
    for item in stats.values():
        item["fwd"].clear()
        item["bwd"].clear()


def _reset_frozen_deltanet_core_stats(trainer) -> None:
    decoder = getattr(trainer, "decoder", None)
    reset = getattr(decoder, "reset_frozen_deltanet_core_bwd_stats", None)
    if reset is not None:
        reset()


def _frozen_deltanet_core_stats(trainer) -> dict[str, object] | None:
    decoder = getattr(trainer, "decoder", None)
    stats = getattr(decoder, "frozen_deltanet_core_bwd_stats", None)
    if stats is None:
        return None
    report = stats()
    if not report or int(report.get("patched_modules", 0) or 0) <= 0:
        return None
    return report


def _elapsed_event_pairs(events: list[tuple[torch.cuda.Event, torch.cuda.Event]]) -> float:
    return sum(start.elapsed_time(end) for start, end in events)


def _summarize_decoder_linear_profile(
    stats: dict[str, dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]],
    *,
    topk: int,
) -> dict[str, object]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    paths: list[dict[str, object]] = []
    families: dict[str, dict[str, float | int]] = {}
    for name, item in stats.items():
        fwd_ms = _elapsed_event_pairs(item["fwd"])
        bwd_ms = _elapsed_event_pairs(item["bwd"])
        fwd_calls = len(item["fwd"])
        bwd_calls = len(item["bwd"])
        family = name.rsplit(".", 1)[-1]
        total_ms = fwd_ms + bwd_ms
        paths.append(
            {
                "name": name,
                "family": family,
                "fwd_ms": fwd_ms,
                "bwd_ms": bwd_ms,
                "total_ms": total_ms,
                "fwd_calls": fwd_calls,
                "bwd_calls": bwd_calls,
            }
        )
        bucket = families.setdefault(
            family,
            {
                "fwd_ms": 0.0,
                "bwd_ms": 0.0,
                "total_ms": 0.0,
                "fwd_calls": 0,
                "bwd_calls": 0,
            },
        )
        bucket["fwd_ms"] = float(bucket["fwd_ms"]) + fwd_ms
        bucket["bwd_ms"] = float(bucket["bwd_ms"]) + bwd_ms
        bucket["total_ms"] = float(bucket["total_ms"]) + total_ms
        bucket["fwd_calls"] = int(bucket["fwd_calls"]) + fwd_calls
        bucket["bwd_calls"] = int(bucket["bwd_calls"]) + bwd_calls

    paths.sort(key=lambda row: float(row["total_ms"]), reverse=True)
    family_rows = [
        {"family": family, **values}
        for family, values in sorted(
            families.items(),
            key=lambda pair: float(pair[1]["total_ms"]),
            reverse=True,
        )
    ]
    return {"families": family_rows[:topk], "paths": paths[:topk]}


def _summarize_decoder_layer_profile(
    stats: dict[str, dict[str, object]],
    *,
    topk: int,
) -> dict[str, object]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    paths: list[dict[str, object]] = []
    families: dict[str, dict[str, float | int]] = {}
    for name, item in stats.items():
        fwd_events = item.get("fwd", [])
        bwd_events = item.get("bwd", [])
        fwd_ms = _elapsed_event_pairs(fwd_events)
        bwd_ms = _elapsed_event_pairs(bwd_events)
        fwd_calls = len(fwd_events)
        bwd_calls = len(bwd_events)
        family = str(item.get("family", "decoder_layer"))
        total_ms = fwd_ms + bwd_ms
        total_calls = fwd_calls + bwd_calls
        paths.append(
            {
                "name": name,
                "family": family,
                "fwd_ms": fwd_ms,
                "bwd_ms": bwd_ms,
                "total_ms": total_ms,
                "fwd_calls": fwd_calls,
                "bwd_calls": bwd_calls,
                "fwd_ms_per_call": fwd_ms / fwd_calls if fwd_calls else 0.0,
                "bwd_ms_per_call": bwd_ms / bwd_calls if bwd_calls else 0.0,
                "total_ms_per_event": total_ms / total_calls if total_calls else 0.0,
            }
        )
        bucket = families.setdefault(
            family,
            {
                "fwd_ms": 0.0,
                "bwd_ms": 0.0,
                "total_ms": 0.0,
                "fwd_calls": 0,
                "bwd_calls": 0,
            },
        )
        bucket["fwd_ms"] = float(bucket["fwd_ms"]) + fwd_ms
        bucket["bwd_ms"] = float(bucket["bwd_ms"]) + bwd_ms
        bucket["total_ms"] = float(bucket["total_ms"]) + total_ms
        bucket["fwd_calls"] = int(bucket["fwd_calls"]) + fwd_calls
        bucket["bwd_calls"] = int(bucket["bwd_calls"]) + bwd_calls

    paths.sort(key=lambda row: float(row["total_ms"]), reverse=True)
    family_rows = [
        {
            "family": family,
            **values,
            "fwd_ms_per_call": float(values["fwd_ms"]) / int(values["fwd_calls"])
            if int(values["fwd_calls"])
            else 0.0,
            "bwd_ms_per_call": float(values["bwd_ms"]) / int(values["bwd_calls"])
            if int(values["bwd_calls"])
            else 0.0,
        }
        for family, values in sorted(
            families.items(),
            key=lambda pair: float(pair[1]["total_ms"]),
            reverse=True,
        )
    ]
    return {"families": family_rows[:topk], "paths": paths[:topk]}


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
    stats: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {name: [] for name in names}
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
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    rows: list[dict[str, object]] = []
    for name, events in stats.items():
        if not events:
            continue
        total_ms = _elapsed_event_pairs(events)
        rows.append({"name": name, "calls": len(events), "total_ms": total_ms})
    rows.sort(key=lambda row: float(row["total_ms"]), reverse=True)
    return rows


def _accumulate_batch_stats(total: dict[str, float], stats: dict[str, float]) -> None:
    for key, value in stats.items():
        if key.endswith("_max_content_len") or key.endswith("_max_target_len"):
            total[key] = max(total.get(key, 0.0), value)
        else:
            total[key] = total.get(key, 0.0) + value


def _attach_wall_throughput_metrics(metrics: dict[str, float]) -> None:
    wall_ms = float(metrics.get("wall_ms", 0.0) or 0.0)
    if wall_ms <= 0.0:
        return
    seconds = wall_ms / 1000.0
    for source_key, output_key in (
        ("batch_target_tokens", "batch_target_tokens_per_s"),
        ("batch_content_tokens", "batch_content_tokens_per_s"),
        ("batch_prompt_tokens", "batch_prompt_tokens_per_s"),
        ("batch_samples", "batch_samples_per_s"),
        ("batch_target_l2_cost", "batch_target_l2_cost_per_s"),
    ):
        value = metrics.get(source_key)
        if value is not None:
            metrics[output_key] = float(value) / seconds


def _stable_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=12).hexdigest()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _cu_lengths_summary(key: str, value: torch.Tensor) -> dict[str, int | str] | None:
    if value.ndim != 1 or value.numel() < 2:
        return None
    if "cu_seqlens" not in key and not key.startswith("cu_") and not key.endswith("_cu"):
        return None

    cu = value.detach().to(device="cpu", dtype=torch.int64)
    lengths = (cu[1:] - cu[:-1]).tolist()
    if not lengths:
        return None
    total = int(sum(lengths))
    l2_cost = int(sum(int(length) * int(length) for length in lengths))
    return {
        "count": len(lengths),
        "total": total,
        "min": int(min(lengths)),
        "max": int(max(lengths)),
        "l2_cost": l2_cost,
        "lengths_hash": _stable_json_hash(lengths),
    }


def _batch_bucket_signature(batch: dict) -> dict:
    """Return a graph-bucket signature for one microbatch.

    CUDA graphs need tensor metadata to be stable for each replay slot. Sequence
    lengths are summarized separately because some current code still branches
    on ``cu_seqlens`` values on the Python side; those hashes tell us when a
    nominally shape-static bucket is still value-dynamic.
    """

    tensors: list[dict[str, object]] = []
    cu_lengths: dict[str, dict[str, int | str]] = {}
    non_tensors: list[dict[str, str]] = []
    for key in sorted(batch):
        value = batch[key]
        if isinstance(value, torch.Tensor):
            tensors.append(
                {
                    "key": key,
                    "shape": [int(dim) for dim in value.shape],
                    "dtype": _dtype_name(value.dtype),
                    "device": str(value.device),
                }
            )
            summary = _cu_lengths_summary(key, value)
            if summary is not None:
                cu_lengths[key] = summary
        else:
            non_tensors.append({"key": key, "type": type(value).__name__})
    return {
        "hash": _stable_json_hash({"tensors": tensors, "cu_lengths": cu_lengths}),
        "tensors": tensors,
        "cu_lengths": cu_lengths,
        "non_tensors": non_tensors,
    }


def _batch_bucket_signatures(step_batches: list[list[dict]]) -> list[list[dict]]:
    return [
        [_batch_bucket_signature(batch) for batch in micro_batches]
        for micro_batches in step_batches
    ]


def _summarize_static_buckets(
    step_signatures: list[list[dict]],
    *,
    max_buckets: int = 64,
) -> dict:
    micro_keys = [
        json.dumps(signature, sort_keys=True, separators=(",", ":"))
        for step in step_signatures
        for signature in step
    ]
    micro_counter = Counter(micro_keys)
    sorted_micro_keys = sorted(micro_counter, key=lambda key: (-micro_counter[key], key))
    micro_id_by_key = {key: idx for idx, key in enumerate(sorted_micro_keys)}

    optimizer_step_keys = [
        tuple(json.dumps(signature, sort_keys=True, separators=(",", ":")) for signature in step)
        for step in step_signatures
    ]
    optimizer_counter = Counter(optimizer_step_keys)
    sorted_optimizer_keys = sorted(
        optimizer_counter,
        key=lambda key: (-optimizer_counter[key], [micro_id_by_key[item] for item in key]),
    )

    micro_buckets = []
    for key in sorted_micro_keys[:max_buckets]:
        signature = json.loads(key)
        micro_buckets.append(
            {
                "bucket_id": micro_id_by_key[key],
                "count": micro_counter[key],
                "hash": signature["hash"],
                "signature": signature,
            }
        )

    optimizer_step_buckets = []
    for key in sorted_optimizer_keys[:max_buckets]:
        micro_bucket_ids = [micro_id_by_key[item] for item in key]
        optimizer_step_buckets.append(
            {
                "count": optimizer_counter[key],
                "micro_bucket_ids": micro_bucket_ids,
                "hash": _stable_json_hash(micro_bucket_ids),
            }
        )

    return {
        "optimizer_steps": len(step_signatures),
        "microbatches": len(micro_keys),
        "unique_microbatch_buckets": len(micro_counter),
        "unique_optimizer_step_buckets": len(optimizer_counter),
        "reported_microbatch_buckets": len(micro_buckets),
        "reported_optimizer_step_buckets": len(optimizer_step_buckets),
        "omitted_microbatch_buckets": max(0, len(micro_counter) - len(micro_buckets)),
        "omitted_optimizer_step_buckets": max(
            0,
            len(optimizer_counter) - len(optimizer_step_buckets),
        ),
        "microbatch_buckets": micro_buckets,
        "optimizer_step_buckets": optimizer_step_buckets,
    }


def _compact_static_bucket_summary(report: dict | None) -> dict | None:
    if report is None:
        return None
    keys = (
        "optimizer_steps",
        "microbatches",
        "unique_microbatch_buckets",
        "unique_optimizer_step_buckets",
        "omitted_microbatch_buckets",
        "omitted_optimizer_step_buckets",
    )
    return {key: report[key] for key in keys}


def _move_batch_to_device(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: _move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [_move_batch_to_device(value, device) for value in batch]
    return batch


def _move_step_batches_to_device(
    step_batches: list[list[dict]],
    device: torch.device,
) -> list[list[dict]]:
    return [
        [_move_batch_to_device(batch, device) for batch in micro_batches]
        for micro_batches in step_batches
    ]


def _fixed_step_batches(
    prefetched_batches: list[list[dict]] | None,
    replay_idx: int,
    *,
    repeat_first_fixed_step: bool,
) -> list[dict] | None:
    if prefetched_batches is None:
        return None
    if repeat_first_fixed_step:
        return prefetched_batches[0]
    return prefetched_batches[replay_idx]


def _optimizer_step_profile_stats(
    micro_batches: list[dict],
    prefix: str,
) -> dict[str, float]:
    stats: dict[str, float] = {}
    for batch in micro_batches:
        _accumulate_batch_stats(stats, _batch_profile_stats(batch, prefix))
    return stats


def _module_param_counts(module: nn.Module | None) -> dict[str, int]:
    if module is None:
        return {
            "total_tensors": 0,
            "total_params": 0,
            "trainable_tensors": 0,
            "trainable_params": 0,
        }
    total_tensors = 0
    total_params = 0
    trainable_tensors = 0
    trainable_params = 0
    for param in module.parameters():
        total_tensors += 1
        n = int(param.numel())
        total_params += n
        if param.requires_grad:
            trainable_tensors += 1
            trainable_params += n
    return {
        "total_tensors": total_tensors,
        "total_params": total_params,
        "trainable_tensors": trainable_tensors,
        "trainable_params": trainable_params,
    }


def _trainable_contract_profile(trainer) -> dict[str, object]:
    """Summarize the trainable-parameter contract used by the profiled step."""

    encoder = getattr(trainer, "encoder", None)
    decoder = getattr(trainer, "decoder", None)
    components = {
        "encoder": _module_param_counts(encoder),
        "decoder": _module_param_counts(decoder),
    }

    id_to_component: dict[int, str] = {}
    id_to_numel: dict[int, int] = {}
    for component, module in (("encoder", encoder), ("decoder", decoder)):
        if module is None:
            continue
        for param in module.parameters():
            ident = id(param)
            id_to_component[ident] = component
            id_to_numel[ident] = int(param.numel())

    optimizer = getattr(trainer, "optimizer", None)
    optimizer_param_ids: set[int] = set()
    optimizer_groups: list[dict[str, object]] = []
    optimizer_by_component = {"encoder": 0, "decoder": 0, "unknown": 0}
    if optimizer is not None:
        for index, group in enumerate(optimizer.param_groups):
            group_components = {"encoder": 0, "decoder": 0, "unknown": 0}
            group_params = list(group.get("params", []))
            for param in group_params:
                ident = id(param)
                optimizer_param_ids.add(ident)
                component = id_to_component.get(ident, "unknown")
                n = id_to_numel.get(ident, int(param.numel()))
                group_components[component] += n
                optimizer_by_component[component] += n
            optimizer_groups.append(
                {
                    "index": index,
                    "use_muon": bool(group.get("use_muon", False)),
                    "lr": float(group.get("lr", 0.0)),
                    "base_lr": float(group.get("base_lr", group.get("lr", 0.0))),
                    "tensors": len(group_params),
                    "params": sum(group_components.values()),
                    "components": group_components,
                }
            )

    decoder_trainable_ids = {
        id(param)
        for param in decoder.parameters()
        if getattr(param, "requires_grad", False)
    } if decoder is not None else set()
    missing_decoder_optimizer_ids = decoder_trainable_ids - optimizer_param_ids
    missing_decoder_optimizer_params = sum(
        id_to_numel.get(ident, 0) for ident in missing_decoder_optimizer_ids
    )

    tcfg = getattr(getattr(trainer, "cfg", None), "training", {}) or {}
    freeze_cfg = tcfg.get("freeze", {}) if hasattr(tcfg, "get") else {}
    lora_cfg = tcfg.get("decoder_lora", {}) if hasattr(tcfg, "get") else {}
    freeze_decoder = bool(freeze_cfg.get("decoder", False)) if hasattr(freeze_cfg, "get") else False
    lora_enabled = bool(lora_cfg.get("enabled", False)) if hasattr(lora_cfg, "get") else False

    decoder_trainable_params = int(components["decoder"]["trainable_params"])
    decoder_optimizer_params = int(optimizer_by_component["decoder"])
    if freeze_decoder:
        contract_ok = decoder_trainable_params == 0 and decoder_optimizer_params == 0
    elif lora_enabled:
        contract_ok = decoder_trainable_params > 0 and missing_decoder_optimizer_params == 0
    else:
        contract_ok = decoder_trainable_params > 0 and missing_decoder_optimizer_params == 0

    return {
        "freeze_decoder": freeze_decoder,
        "decoder_lora_enabled": lora_enabled,
        "contract_ok": contract_ok,
        "components": components,
        "optimizer": {
            "groups": optimizer_groups,
            "params_by_component": optimizer_by_component,
            "missing_decoder_optimizer_params": int(missing_decoder_optimizer_params),
            "missing_decoder_optimizer_tensors": len(missing_decoder_optimizer_ids),
        },
    }


def _fixed_step_batch_stats(
    fixed_stats: list[dict[str, float]] | None,
    replay_idx: int,
    *,
    repeat_first_fixed_step: bool,
) -> dict[str, float] | None:
    if fixed_stats is None:
        return None
    if repeat_first_fixed_step:
        return fixed_stats[0]
    return fixed_stats[replay_idx]


def _cuda_graph_forward_backward_probe(
    trainer,
    fixed_batches: list[list[dict]] | None,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"enabled": True, "status": "skipped", "reason": "cuda_unavailable"}
    if not fixed_batches:
        return {"enabled": True, "status": "skipped", "reason": "fixed_batches_required"}

    batch = _move_batch_to_device(fixed_batches[0][0], trainer.device)
    result: dict[str, object] = {
        "enabled": True,
        "status": "failed",
        "microbatches": 1,
        "bucket_hash": _batch_bucket_signature(batch)["hash"],
    }
    try:
        trainer.optimizer.zero_grad()
        torch.cuda.synchronize()

        trainer._forward_backward(batch)
        trainer.optimizer.zero_grad()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            trainer._forward_backward(batch)
        graph.replay()
        torch.cuda.synchronize()
        trainer.optimizer.zero_grad()

        result["status"] = "captured"
    except Exception as exc:  # pragma: no cover - depends on CUDA/kernel runtime.
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:1000]
        with contextlib.suppress(Exception):
            trainer.optimizer.zero_grad()
    return result


def _run_optimizer_step(
    trainer,
    dataloader_iter,
    *,
    step: int,
    max_steps: int,
    warmup_steps: int,
    base_lr: float,
    accum_steps: int,
    fixed_batches: list[dict] | None = None,
    fixed_batch_stats: dict[str, float] | None = None,
) -> tuple[object, dict[str, float]]:
    if fixed_batches is not None and len(fixed_batches) != accum_steps:
        raise ValueError(
            f"fixed batch replay expected {accum_steps} microbatches, got {len(fixed_batches)}"
        )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.global_step = step
    trainer._pre_step_hook()
    if trainer._dataloader_invalidated:
        dataloader_iter = trainer._create_dataloader_iter()
        trainer._dataloader_invalidated = False

    for pg in trainer.optimizer.param_groups:
        group_base = pg.get("base_lr", base_lr)
        pg["lr"] = cosine_with_warmup(step, max_steps, warmup_steps, group_base)
    trainer._post_lr_schedule(step)

    trainer.optimizer.zero_grad()
    accum_metrics = []
    batch_stats: dict[str, float] = dict(fixed_batch_stats or {})
    for micro_idx in range(accum_steps):
        with torch.profiler.record_function(f"microbatch_{micro_idx:02d}/fetch"):
            if fixed_batches is None:
                batch, dataloader_iter = _next_batch(trainer, dataloader_iter)
                _accumulate_batch_stats(
                    batch_stats,
                    _batch_profile_stats(batch, "batch"),
                )
            else:
                batch = fixed_batches[micro_idx]
                if fixed_batch_stats is None:
                    _accumulate_batch_stats(
                        batch_stats,
                        _batch_profile_stats(batch, "batch"),
                    )
        with torch.profiler.record_function(f"microbatch_{micro_idx:02d}/forward_backward"):
            accum_metrics.append(trainer._forward_backward(batch))

    with torch.profiler.record_function("clip_grad_norm"):
        grad_norm = clip_grad_norm(trainer.trainable_parameters())
    if not math.isfinite(grad_norm):
        raise RuntimeError(f"NaN/Inf grad_norm at profiled step {step}: {grad_norm}")

    with torch.profiler.record_function("optimizer_step"):
        trainer.optimizer.step()
    with torch.profiler.record_function("post_optimizer_step"):
        trainer._post_optimizer_step(step)

    if _empty_cache_every_step(trainer.cfg) and torch.cuda.is_available():
        with torch.profiler.record_function("cuda_empty_cache"):
            torch.cuda.empty_cache()

    metrics = _average_metrics(accum_metrics)
    metrics.update(batch_stats)
    metrics["grad_norm"] = grad_norm
    metrics["lr"] = trainer.optimizer.param_groups[0]["lr"]
    if len(trainer.optimizer.param_groups) > 1:
        metrics["lr_min"] = min(pg["lr"] for pg in trainer.optimizer.param_groups)
    if torch.cuda.is_available():
        metrics["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        metrics["reserved_memory_gib"] = torch.cuda.max_memory_reserved() / 1024**3
    trainer._add_step_metrics(metrics)
    return dataloader_iter, metrics


def _prefetch_optimizer_step_batches(
    trainer,
    dataloader_iter,
    *,
    optimizer_steps: int,
    accum_steps: int,
) -> tuple[list[list[dict]], object, dict[str, float]]:
    step_batches: list[list[dict]] = []
    total_stats: dict[str, float] = {}
    for _ in range(optimizer_steps):
        micro_batches: list[dict] = []
        for _micro_idx in range(accum_steps):
            batch, dataloader_iter = _next_batch(trainer, dataloader_iter)
            micro_batches.append(batch)
            _accumulate_batch_stats(
                total_stats,
                _batch_profile_stats(batch, "prefetch"),
            )
        step_batches.append(micro_batches)
    return step_batches, dataloader_iter, total_stats


def _empty_cache_every_step(cfg: DictConfig) -> bool:
    training_val = cfg.training.get("cuda_empty_cache_every_step", None)
    if training_val is not None:
        return bool(training_val)
    compute_cfg = cfg.get("compute", {}) or {}
    return bool(compute_cfg.get("cuda_empty_cache_every_step", False))


@contextmanager
def _falcon_kernel_counters() -> Iterable[dict[str, int]]:
    counts = {
        "mixer_torch_forward": 0,
        "mixer_cuda_kernels_forward": 0,
        "mamba_split_conv1d_scan_combined": 0,
        "mamba_chunk_scan_combined": 0,
        "causal_conv1d_fn": 0,
        "causal_conv1d_update": 0,
        "selective_state_update": 0,
        "scaled_dot_product_attention": 0,
    }
    try:
        import transformers.models.falcon_h1.modeling_falcon_h1 as falcon_h1
    except Exception:
        yield counts
        return

    originals = {
        "torch_forward": getattr(falcon_h1.FalconH1Mixer, "torch_forward", None),
        "cuda_kernels_forward": getattr(
            falcon_h1.FalconH1Mixer,
            "cuda_kernels_forward",
            None,
        ),
        "mamba_split_conv1d_scan_combined": getattr(
            falcon_h1,
            "mamba_split_conv1d_scan_combined",
            None,
        ),
        "mamba_chunk_scan_combined": getattr(
            falcon_h1,
            "mamba_chunk_scan_combined",
            None,
        ),
        "causal_conv1d_fn": getattr(falcon_h1, "causal_conv1d_fn", None),
        "causal_conv1d_update": getattr(falcon_h1, "causal_conv1d_update", None),
        "selective_state_update": getattr(falcon_h1, "selective_state_update", None),
        "scaled_dot_product_attention": F.scaled_dot_product_attention,
    }

    def counted_torch_forward(self, *args: Any, **kwargs: Any) -> Any:
        counts["mixer_torch_forward"] += 1
        with torch.profiler.record_function("falcon_mamba/mixer_torch_forward"):
            return originals["torch_forward"](self, *args, **kwargs)

    def counted_cuda_forward(self, *args: Any, **kwargs: Any) -> Any:
        counts["mixer_cuda_kernels_forward"] += 1
        with torch.profiler.record_function("falcon_mamba/mixer_cuda_kernels_forward"):
            return originals["cuda_kernels_forward"](self, *args, **kwargs)

    def wrap_global(name: str) -> Callable[..., Any] | None:
        original = originals[name]
        if original is None:
            return None

        def counted(*args: Any, **kwargs: Any) -> Any:
            counts[name] += 1
            with torch.profiler.record_function(f"falcon_mamba/{name}"):
                return original(*args, **kwargs)

        return counted

    def counted_sdpa(*args: Any, **kwargs: Any) -> Any:
        counts["scaled_dot_product_attention"] += 1
        with torch.profiler.record_function("falcon_attention/scaled_dot_product_attention"):
            return originals["scaled_dot_product_attention"](*args, **kwargs)

    if originals["torch_forward"] is not None:
        falcon_h1.FalconH1Mixer.torch_forward = counted_torch_forward
    if originals["cuda_kernels_forward"] is not None:
        falcon_h1.FalconH1Mixer.cuda_kernels_forward = counted_cuda_forward
    for name in (
        "mamba_split_conv1d_scan_combined",
        "mamba_chunk_scan_combined",
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "selective_state_update",
    ):
        if originals[name] is not None:
            setattr(falcon_h1, name, wrap_global(name))
    F.scaled_dot_product_attention = counted_sdpa

    try:
        yield counts
    finally:
        if originals["torch_forward"] is not None:
            falcon_h1.FalconH1Mixer.torch_forward = originals["torch_forward"]
        if originals["cuda_kernels_forward"] is not None:
            falcon_h1.FalconH1Mixer.cuda_kernels_forward = originals["cuda_kernels_forward"]
        for name in (
            "mamba_split_conv1d_scan_combined",
            "mamba_chunk_scan_combined",
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "selective_state_update",
        ):
            if originals[name] is not None:
                setattr(falcon_h1, name, originals[name])
        F.scaled_dot_product_attention = originals["scaled_dot_product_attention"]


def _reset_counts(counts: dict[str, int]) -> None:
    for key in counts:
        counts[key] = 0


def _event_time_us(event: object, attr: str) -> float:
    return float(
        getattr(
            event,
            attr,
            getattr(event, attr.replace("cuda", "device"), 0.0),
        )
        or 0.0
    )


def _profile_bucket_for_key(key: str) -> str:
    lowered = key.lower()
    if (
        "falcon_mamba/" in lowered
        or "mamba" in lowered
        or "causal_conv1d" in lowered
        or "selective_state" in lowered
        or "chunk_scan" in lowered
        or "chunk_state" in lowered
        or "state_passing" in lowered
        or "ssd_" in lowered
    ):
        return "mamba"
    if (
        "falcon_attention/" in lowered
        or "scaled_dot_product_attention" in lowered
        or "flash_attention" in lowered
        or "attention" in lowered
        or "fmha" in lowered
    ):
        return "attention"
    if (
        "cross_entropy" in lowered
        or "linear_cross_entropy" in lowered
        or "nll_loss" in lowered
        or "log_softmax" in lowered
    ):
        return "cross_entropy"
    if "optimizer_step" in lowered or "adam" in lowered or "muon" in lowered:
        return "optimizer"
    if "forward_backward" in lowered or "microbatch_" in lowered:
        return "forward_backward"
    if "/fetch" in lowered or "dataloader" in lowered:
        return "fetch"
    return "uncategorized"


def _profile_bucket_summary(prof, *, topk: int = 30) -> dict[str, object]:
    buckets: dict[str, dict[str, float]] = {}
    top_events: list[dict[str, object]] = []
    for event in prof.key_averages():
        key = str(getattr(event, "key", ""))
        cuda_us = _event_time_us(event, "cuda_time_total")
        self_cuda_us = _event_time_us(event, "self_cuda_time_total")
        cpu_us = float(getattr(event, "cpu_time_total", 0.0) or 0.0)
        count = int(getattr(event, "count", 0) or 0)
        bucket = _profile_bucket_for_key(key)
        target = buckets.setdefault(
            bucket,
            {
                "cuda_ms": 0.0,
                "self_cuda_ms": 0.0,
                "cpu_ms": 0.0,
                "events": 0.0,
            },
        )
        target["cuda_ms"] += cuda_us / 1000.0
        target["self_cuda_ms"] += self_cuda_us / 1000.0
        target["cpu_ms"] += cpu_us / 1000.0
        target["events"] += float(count)
        top_events.append(
            {
                "key": key,
                "bucket": bucket,
                "cuda_ms": cuda_us / 1000.0,
                "self_cuda_ms": self_cuda_us / 1000.0,
                "cpu_ms": cpu_us / 1000.0,
                "count": count,
            }
        )
    total_cuda_ms = sum(value["cuda_ms"] for value in buckets.values())
    for value in buckets.values():
        value["cuda_pct"] = 100.0 * value["cuda_ms"] / total_cuda_ms if total_cuda_ms > 0 else 0.0
    top_events.sort(key=lambda item: (-float(item["cuda_ms"]), str(item["key"])))
    return {
        "total_cuda_ms": total_cuda_ms,
        "buckets": buckets,
        "top_events": top_events[:topk],
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    patch_triton_allocator()
    patch_triton_autotuner()
    patch_gated_delta_rule_numerics()
    patch_fused_rms_norm_gated_for_sm121()
    install_step_watchdog(
        timeout_seconds=float(os.environ.get("BGKIT_STEP_TIMEOUT", "180.0")),
        poll_seconds=5.0,
    )
    setup_logging()
    set_seed(cfg.get("seed", 42))

    if torch.cuda.is_available():
        frac = float(os.environ.get("BGKIT_CUDA_MEM_FRACTION", "0.25"))
        torch.cuda.set_per_process_memory_fraction(frac)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    profile_cfg = cfg.get("profile", {}) or {}
    warmup_profile_steps = int(profile_cfg.get("warmup_steps", 1))
    measured_steps = int(profile_cfg.get("steps", 1))
    topk = int(profile_cfg.get("topk", 60))
    record_shapes = bool(profile_cfg.get("record_shapes", False))
    profile_memory = bool(profile_cfg.get("profile_memory", False))
    capture_profiler = bool(profile_cfg.get("capture_profiler", False))
    fixed_batches = bool(profile_cfg.get("fixed_batches", False))
    repeat_first_fixed_step = bool(profile_cfg.get("repeat_first_fixed_step", False))
    static_device_batches = bool(profile_cfg.get("static_device_batches", False))
    static_bucket_summary = bool(profile_cfg.get("static_bucket_summary", fixed_batches))
    static_bucket_max_buckets = int(profile_cfg.get("static_bucket_max_buckets", 64))
    cuda_graph_probe = bool(profile_cfg.get("cuda_graph_probe", False))
    profile_decoder_linears = bool(profile_cfg.get("decoder_linears", False))
    decoder_linears_topk = int(profile_cfg.get("decoder_linears_topk", 40))
    profile_gdr_internals = bool(profile_cfg.get("gdr_internals", False))
    profile_decoder_layers = bool(profile_cfg.get("decoder_layers", False))
    decoder_layers_topk = int(profile_cfg.get("decoder_layers_topk", 40))
    trace_path_raw = profile_cfg.get("trace_path", None)

    trainer, dataloader_iter, checkpoint = _prepare_trainer(cfg)
    decoder_linear_stats: dict[str, dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]] = {}
    decoder_linear_count = 0
    if profile_decoder_linears:
        decoder_linear_stats, decoder_linear_count = _install_decoder_linear_profiler(trainer)
    decoder_layer_stats: dict[str, dict[str, object]] = {}
    decoder_layer_count = 0
    if profile_decoder_layers:
        decoder_layer_stats, decoder_layer_count = _install_decoder_layer_profiler(trainer)
    gdr_internal_stats: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]
    gdr_internal_stats = _install_gdr_internal_timers() if profile_gdr_internals else {}
    trainable_contract = _trainable_contract_profile(trainer)
    max_steps, base_lr, warmup_steps = _schedule_values(trainer, cfg)
    accum_steps = trainer._validate_accum_steps(
        cfg.training.get("gradient_accumulation_steps", 1),
    )
    trainer._accum_steps = accum_steps

    start_step = int(trainer.global_step)
    # Fixed-batch prefetch must see the same curriculum-routed sampler as the
    # measured step. Step 5 switches from the stage-0 loader to the stage-1
    # frozen/unfrozen budgets in _pre_step_hook(), so run it before prefetching.
    trainer.global_step = start_step
    trainer._pre_step_hook()
    if trainer._dataloader_invalidated:
        dataloader_iter = trainer._create_dataloader_iter()
        trainer._dataloader_invalidated = False

    prefetched_batches: list[list[dict]] | None = None
    prefetched_batch_stats: list[dict[str, float]] | None = None
    prefetch_stats: dict[str, float] = {}
    static_bucket_report: dict | None = None
    if fixed_batches:
        prefetch_optimizer_steps = (
            1 if repeat_first_fixed_step else warmup_profile_steps + measured_steps
        )
        prefetched_batches, dataloader_iter, prefetch_stats = _prefetch_optimizer_step_batches(
            trainer,
            dataloader_iter,
            optimizer_steps=prefetch_optimizer_steps,
            accum_steps=accum_steps,
        )
        if static_device_batches and torch.cuda.is_available():
            prefetched_batches = _move_step_batches_to_device(
                prefetched_batches,
                trainer.device,
            )
        prefetched_batch_stats = [
            _optimizer_step_profile_stats(step_batches, "batch")
            for step_batches in prefetched_batches
        ]
        if static_bucket_summary:
            static_bucket_report = _summarize_static_buckets(
                _batch_bucket_signatures(prefetched_batches),
                max_buckets=static_bucket_max_buckets,
            )
        print(
            json.dumps(
                {
                    "event": "profile_fixed_batches_prefetched",
                    "optimizer_steps": len(prefetched_batches),
                    "accum_steps": accum_steps,
                    "repeat_first_fixed_step": repeat_first_fixed_step,
                    "static_device_batches": static_device_batches,
                    "static_bucket_summary": _compact_static_bucket_summary(static_bucket_report),
                    **prefetch_stats,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "event": "profile_training_step_start",
                "phase": cfg.training.phase,
                "checkpoint": str(checkpoint) if checkpoint is not None else None,
                "start_step": start_step,
                "warmup_steps": warmup_profile_steps,
                "profile_steps": measured_steps,
                "accum_steps": accum_steps,
                "capture_profiler": capture_profiler,
                "fixed_batches": fixed_batches,
                "repeat_first_fixed_step": repeat_first_fixed_step,
                "static_device_batches": static_device_batches,
                "static_bucket_summary": static_bucket_summary,
                "cuda_graph_probe": cuda_graph_probe,
                "profile_decoder_linears": profile_decoder_linears,
                "profile_decoder_linears_count": decoder_linear_count,
                "profile_decoder_layers": profile_decoder_layers,
                "profile_decoder_layers_count": decoder_layer_count,
                "profile_gdr_internals": profile_gdr_internals,
                "trainable_contract": trainable_contract,
                "ce_impl": os.environ.get("BGKIT_DECODER_CE_IMPL", "<unset>"),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    step = start_step
    warmup_metrics: list[dict[str, float]] = []
    replay_idx = 0
    prof = None

    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    measured_metrics: list[dict[str, float]] = []
    with _falcon_kernel_counters() as kernel_counts:
        for _ in range(warmup_profile_steps):
            t0 = time.perf_counter()
            dataloader_iter, metrics = _run_optimizer_step(
                trainer,
                dataloader_iter,
                step=step,
                max_steps=max_steps,
                warmup_steps=warmup_steps,
                base_lr=base_lr,
                accum_steps=accum_steps,
                fixed_batches=_fixed_step_batches(
                    prefetched_batches,
                    replay_idx,
                    repeat_first_fixed_step=repeat_first_fixed_step,
                ),
                fixed_batch_stats=_fixed_step_batch_stats(
                    prefetched_batch_stats,
                    replay_idx,
                    repeat_first_fixed_step=repeat_first_fixed_step,
                ),
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            metrics = dict(metrics)
            metrics["wall_ms"] = (time.perf_counter() - t0) * 1000.0
            _attach_wall_throughput_metrics(metrics)
            metrics["step"] = step
            warmup_metrics.append(metrics)
            print(json.dumps({"event": "warmup_step", **metrics}, sort_keys=True), flush=True)
            step += 1
            replay_idx += 1

        if decoder_linear_stats:
            _clear_decoder_linear_profile_stats(decoder_linear_stats)
        if decoder_layer_stats:
            _clear_decoder_layer_profile_stats(decoder_layer_stats)
        if gdr_internal_stats:
            _clear_gdr_internal_timers(gdr_internal_stats)
        _reset_frozen_deltanet_core_stats(trainer)
        _reset_counts(kernel_counts)
        if capture_profiler:
            with profile(
                activities=activities,
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=False,
            ) as prof:
                for _ in range(measured_steps):
                    t0 = time.perf_counter()
                    dataloader_iter, metrics = _run_optimizer_step(
                        trainer,
                        dataloader_iter,
                        step=step,
                        max_steps=max_steps,
                        warmup_steps=warmup_steps,
                        base_lr=base_lr,
                        accum_steps=accum_steps,
                        fixed_batches=_fixed_step_batches(
                            prefetched_batches,
                            replay_idx,
                            repeat_first_fixed_step=repeat_first_fixed_step,
                        ),
                        fixed_batch_stats=_fixed_step_batch_stats(
                            prefetched_batch_stats,
                            replay_idx,
                            repeat_first_fixed_step=repeat_first_fixed_step,
                        ),
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    metrics = dict(metrics)
                    metrics["wall_ms"] = (time.perf_counter() - t0) * 1000.0
                    _attach_wall_throughput_metrics(metrics)
                    metrics["step"] = step
                    measured_metrics.append(metrics)
                    print(
                        json.dumps({"event": "profiled_step", **metrics}, sort_keys=True),
                        flush=True,
                    )
                    step += 1
                    replay_idx += 1
                    prof.step()
        else:
            for _ in range(measured_steps):
                t0 = time.perf_counter()
                dataloader_iter, metrics = _run_optimizer_step(
                    trainer,
                    dataloader_iter,
                    step=step,
                    max_steps=max_steps,
                    warmup_steps=warmup_steps,
                    base_lr=base_lr,
                    accum_steps=accum_steps,
                    fixed_batches=_fixed_step_batches(
                        prefetched_batches,
                        replay_idx,
                        repeat_first_fixed_step=repeat_first_fixed_step,
                    ),
                    fixed_batch_stats=_fixed_step_batch_stats(
                        prefetched_batch_stats,
                        replay_idx,
                        repeat_first_fixed_step=repeat_first_fixed_step,
                    ),
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                metrics = dict(metrics)
                metrics["wall_ms"] = (time.perf_counter() - t0) * 1000.0
                _attach_wall_throughput_metrics(metrics)
                metrics["step"] = step
                measured_metrics.append(metrics)
                print(
                    json.dumps({"event": "profiled_step", **metrics}, sort_keys=True),
                    flush=True,
                )
                step += 1
                replay_idx += 1
        measured_kernel_counts = dict(kernel_counts)

    print(
        json.dumps(
            {"event": "profile_kernel_counts", **measured_kernel_counts},
            sort_keys=True,
        ),
        flush=True,
    )

    if prof is not None:
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=topk)
        print(table)
        profiler_buckets = _profile_bucket_summary(prof, topk=topk)
        print(
            json.dumps(
                {"event": "profile_bucket_summary", **profiler_buckets},
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        table = ""
        profiler_buckets = {
            "enabled": False,
            "reason": "capture_profiler_false",
            "total_cuda_ms": 0.0,
            "buckets": [],
            "top_events": [],
        }
        print(
            json.dumps(
                {"event": "profile_bucket_summary", **profiler_buckets},
                sort_keys=True,
            ),
            flush=True,
        )

    decoder_linear_profile: dict[str, object] | None = None
    if decoder_linear_stats:
        decoder_linear_profile = _summarize_decoder_linear_profile(
            decoder_linear_stats,
            topk=decoder_linears_topk,
        )
        print(
            json.dumps(
                {
                    "event": "profile_decoder_linears",
                    "profiled_linear_count": decoder_linear_count,
                    **decoder_linear_profile,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    decoder_layer_profile: dict[str, object] | None = None
    if decoder_layer_stats:
        decoder_layer_profile = _summarize_decoder_layer_profile(
            decoder_layer_stats,
            topk=decoder_layers_topk,
        )
        print(
            json.dumps(
                {
                    "event": "profile_decoder_layers",
                    "profiled_layer_count": decoder_layer_count,
                    **decoder_layer_profile,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    gdr_internal_profile: list[dict[str, object]] | None = None
    if gdr_internal_stats:
        gdr_internal_profile = _summarize_gdr_internal_timers(gdr_internal_stats)
        print(
            json.dumps(
                {
                    "event": "profile_gdr_internals",
                    "items": gdr_internal_profile,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    frozen_deltanet_core_profile = _frozen_deltanet_core_stats(trainer)
    if frozen_deltanet_core_profile is not None:
        print(
            json.dumps(
                {
                    "event": "profile_frozen_deltanet_core",
                    **frozen_deltanet_core_profile,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    cuda_graph_probe_report: dict[str, object] | None = None
    if cuda_graph_probe:
        cuda_graph_probe_report = _cuda_graph_forward_backward_probe(
            trainer,
            prefetched_batches,
        )
        print(
            json.dumps(
                {"event": "cuda_graph_forward_backward_probe", **cuda_graph_probe_report},
                sort_keys=True,
            ),
            flush=True,
        )

    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = checkpoint_dir / f"profile_training_step_{cfg.training.phase}_{ts}.json"
    trace_path = Path(str(trace_path_raw)) if trace_path_raw else None
    if trace_path is not None and prof is not None:
        prof.export_chrome_trace(str(trace_path))
    elif trace_path is not None:
        print(
            json.dumps(
                {
                    "event": "profile_trace_skipped",
                    "reason": "capture_profiler_false",
                    "trace_path": str(trace_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    # Profile-report schema (top-level keys; consumers should treat this dict
    # as forward-compatible — new keys may appear, existing keys keep meaning):
    #   schema_version      str   — bump when an existing key's meaning changes
    #   phase               str   — cfg.training.phase
    #   checkpoint          str?  — resolved input checkpoint, or null
    #   start_step          int   — global step at warmup start
    #   end_step_exclusive  int   — global step after the final measured step
    #   warmup_metrics      list  — per-warmup-step metric dicts
    #   measured_metrics    list  — per-measured-step metric dicts (timing,
    #                               cuda_max_allocated_gib, peak_memory_gib —
    #                               peak resets PER STEP, not run-wide)
    #   topk                list  — top-K profiler events by CUDA self time
    #   profile_table       str   — full torch profiler text table
    #   profile_buckets     dict  — coarse buckets (attention/mamba/mlp/ce/optimizer/uncategorized)
    #   decoder_linear_profile dict? — opt-in frozen decoder linear event totals
    #   decoder_layer_profile dict? — opt-in whole decoder-layer event totals
    #   gdr_internal_profile list? — opt-in FLA GDR internal event totals
    #   frozen_deltanet_core_profile dict? — opt-in custom/fallback core counts
    #   trainable_contract dict — trainable/optimizer parameter counts used to
    #                             verify that decoder-training profiles really
    #                             include decoder weights in the optimizer
    #   falcon_kernel_counts dict — per-kernel call counts captured via the
    #                               functional-namespace monkey-patch in
    #                               _collect_kernel_counts; keys: mamba_split_*,
    #                               causal_conv1d_*, scaled_dot_product_attention, ...
    #   trace_path          str?  — chrome trace path, or null
    #   training_shape      dict  — config knobs that affect timing
    #   env                 dict  — captured environment variables
    report = {
        "schema_version": "1",
        "phase": cfg.training.phase,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "start_step": start_step,
        "end_step_exclusive": step,
        "warmup_metrics": warmup_metrics,
        "measured_metrics": measured_metrics,
        "topk": topk,
        "profile_table": table,
        "profile_buckets": profiler_buckets,
        "decoder_linear_profile": decoder_linear_profile,
        "decoder_layer_profile": decoder_layer_profile,
        "gdr_internal_profile": gdr_internal_profile,
        "frozen_deltanet_core_profile": frozen_deltanet_core_profile,
        "trainable_contract": trainable_contract,
        "falcon_kernel_counts": measured_kernel_counts,
        "trace_path": str(trace_path) if trace_path is not None else None,
        "training_shape": {
            "decoder_lora": OmegaConf.to_container(
                cfg.training.get("decoder_lora", {}),
                resolve=True,
            ),
            "decoder_layerwise_split": OmegaConf.to_container(
                cfg.training.get("decoder_layerwise_split", {}),
                resolve=True,
            ),
            "frozen_decoder_kernels": OmegaConf.to_container(
                cfg.training.get("frozen_decoder_kernels", {}),
                resolve=True,
            ),
            "capture_profiler": capture_profiler,
            "profile_decoder_linears": profile_decoder_linears,
            "profile_decoder_linears_count": decoder_linear_count,
            "profile_decoder_layers": profile_decoder_layers,
            "profile_decoder_layers_count": decoder_layer_count,
            "profile_gdr_internals": profile_gdr_internals,
            "fixed_batches": fixed_batches,
            "repeat_first_fixed_step": repeat_first_fixed_step,
            "static_device_batches": static_device_batches,
            "prefetch_stats": prefetch_stats,
            "static_bucket_summary": static_bucket_report,
            "cuda_graph_probe": cuda_graph_probe_report,
            "gradient_accumulation_steps": int(
                cfg.training.get("gradient_accumulation_steps", 1),
            ),
            "max_batch_tokens": int(cfg.training.get("max_batch_tokens", 0)),
            "max_sample_length": int(cfg.training.get("max_sample_length", 0)),
        },
        "env": {
            "BGKIT_DECODER_CE_IMPL": os.environ.get("BGKIT_DECODER_CE_IMPL"),
            "BGKIT_GDN_BACKEND": os.environ.get("BGKIT_GDN_BACKEND"),
            "BGKIT_QWEN35_LAYERWISE_SPLIT": os.environ.get("BGKIT_QWEN35_LAYERWISE_SPLIT"),
            "BGKIT_QWEN35_LAYERWISE_SPLIT_MIN_PREFIX": os.environ.get(
                "BGKIT_QWEN35_LAYERWISE_SPLIT_MIN_PREFIX"
            ),
            "BGKIT_QWEN35_LAYERWISE_SPLIT_MIN_RATIO": os.environ.get(
                "BGKIT_QWEN35_LAYERWISE_SPLIT_MIN_RATIO"
            ),
            "BGKIT_QWEN35_LAYERWISE_SPLIT_PACKED_DELTANET": os.environ.get(
                "BGKIT_QWEN35_LAYERWISE_SPLIT_PACKED_DELTANET"
            ),
            "FLA_GDR_FUSE_GATE_BWD": os.environ.get("FLA_GDR_FUSE_GATE_BWD"),
            "FLA_GDR_FUSE_DQKG_WY": os.environ.get("FLA_GDR_FUSE_DQKG_WY"),
            "FLA_GDR_FUSE_DQKG_WY_VARLEN": os.environ.get("FLA_GDR_FUSE_DQKG_WY_VARLEN"),
            "FLA_GDR_PACKED_DPROJ_BWD": os.environ.get("FLA_GDR_PACKED_DPROJ_BWD"),
            "FLA_GDR_PAIR_QK_L2NORM_FWD": os.environ.get("FLA_GDR_PAIR_QK_L2NORM_FWD"),
            "FLA_GDR_RECOMPUTE_WY_DW": os.environ.get("FLA_GDR_RECOMPUTE_WY_DW"),
            "FLA_GDR_SAVE_INTERMEDIATES": os.environ.get("FLA_GDR_SAVE_INTERMEDIATES"),
            "FLA_GDR_SAVE_LOCAL_ATTENTION": os.environ.get("FLA_GDR_SAVE_LOCAL_ATTENTION"),
            "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV": os.environ.get(
                "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV"
            ),
            "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX": os.environ.get(
                "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX"
            ),
            "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_BACKEND": os.environ.get(
                "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_BACKEND"
            ),
            "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_BUNDLE_INPUT": os.environ.get(
                "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_BUNDLE_INPUT"
            ),
            "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_PRE_L2NORM": os.environ.get(
                "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_PRE_L2NORM"
            ),
            "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_RESET_CONV": os.environ.get(
                "BGKIT_FROZEN_DELTANET_CHANNEL_LAST_RESET_CONV"
            ),
            "BGKIT_FROZEN_DELTANET_FUSED_QKV_CONV_L2NORM": os.environ.get(
                "BGKIT_FROZEN_DELTANET_FUSED_QKV_CONV_L2NORM"
            ),
            "BGKIT_DELTANET_RAW_GATE_IN_KERNEL": os.environ.get(
                "BGKIT_DELTANET_RAW_GATE_IN_KERNEL"
            ),
            "BGKIT_FROZEN_DELTANET_TRITON_QKV_SPLIT": os.environ.get(
                "BGKIT_FROZEN_DELTANET_TRITON_QKV_SPLIT"
            ),
            "BGKIT_FROZEN_DELTANET_TRITON_BA_DX": os.environ.get(
                "BGKIT_FROZEN_DELTANET_TRITON_BA_DX"
            ),
            "BGKIT_FROZEN_DELTANET_TRITON_ZBA_DX": os.environ.get(
                "BGKIT_FROZEN_DELTANET_TRITON_ZBA_DX"
            ),
            "BGKIT_FROZEN_DELTANET_WIDE_DPROJ_DX": os.environ.get(
                "BGKIT_FROZEN_DELTANET_WIDE_DPROJ_DX"
            ),
            "BGKIT_FROZEN_DELTANET_WIDE_DPROJ_SCRATCH_QKV": os.environ.get(
                "BGKIT_FROZEN_DELTANET_WIDE_DPROJ_SCRATCH_QKV"
            ),
            "BGKIT_FROZEN_DELTANET_BUNDLE_DX": os.environ.get("BGKIT_FROZEN_DELTANET_BUNDLE_DX"),
            "BGKIT_FROZEN_DELTANET_CORE_BWD_MAX_SEQ_LEN": os.environ.get(
                "BGKIT_FROZEN_DELTANET_CORE_BWD_MAX_SEQ_LEN"
            ),
            "BGKIT_FROZEN_DELTANET_CORE_BWD_MIN_SEQ_LEN": os.environ.get(
                "BGKIT_FROZEN_DELTANET_CORE_BWD_MIN_SEQ_LEN"
            ),
            "BGKIT_DECODER_MLP_SWIGLU_FUSION": os.environ.get("BGKIT_DECODER_MLP_SWIGLU_FUSION"),
            "BGKIT_DECODER_MLP_SWIGLU_TRITON_FWD": os.environ.get(
                "BGKIT_DECODER_MLP_SWIGLU_TRITON_FWD"
            ),
            "MAMBA_SM121_SAFE_AUTOTUNE": os.environ.get("MAMBA_SM121_SAFE_AUTOTUNE"),
            "MAMBA_SM121_STATIC_CONFIGS": os.environ.get("MAMBA_SM121_STATIC_CONFIGS"),
            "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"profile_report={report_path}", flush=True)


if __name__ == "__main__":
    main()
