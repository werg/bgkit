#!/usr/bin/env python
"""Profile one real BgKIT optimizer step from a resumed checkpoint.

This is a kernel-level profiler, not a memory sweep. It restores the selected
training phase, runs a configurable number of warmup optimizer steps to get JIT
compilation out of the measured path, then captures one or more real optimizer
steps with ``torch.profiler``.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import hydra
import torch
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
    if phase in ("phase1_step4p7", "phase1_step4p7_v2", "phase1_step4p7_v3"):
        from bgkit.training.phase1.bridge_distill import BridgeDistillTrainer

        return BridgeDistillTrainer(cfg)
    if phase == "phase1_step5":
        from bgkit.training.phase1.commit_encoding import CommitEncodingTrainer

        return CommitEncodingTrainer(cfg)
    if phase == "phase1_step6":
        from bgkit.training.phase1.compression import CompressionTrainer

        return CompressionTrainer(cfg)
    if phase in ("phase2", "phase2_kb"):
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

    last_file = checkpoint_dir / ".last_checkpoint"
    if last_file.exists():
        raw = last_file.read_text().strip()
        if raw:
            path = Path(raw)
            if path.exists():
                return path

    phase = cfg.get("training", {}).get("phase", None)
    if phase:
        return resolve_latest_checkpoint(checkpoint_dir, phase)
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
        if (
            saved_per_group is not None
            and len(saved_per_group) == len(trainer.optimizer.param_groups)
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


def _accumulate_batch_stats(total: dict[str, float], stats: dict[str, float]) -> None:
    for key, value in stats.items():
        if key.endswith("_max_content_len") or key.endswith("_max_target_len"):
            total[key] = max(total.get(key, 0.0), value)
        else:
            total[key] = total.get(key, 0.0) + value


def _run_optimizer_step(
    trainer,
    dataloader_iter,
    *,
    step: int,
    max_steps: int,
    warmup_steps: int,
    base_lr: float,
    accum_steps: int,
) -> tuple[object, dict[str, float]]:
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
    batch_stats: dict[str, float] = {}
    for micro_idx in range(accum_steps):
        with torch.profiler.record_function(f"microbatch_{micro_idx:02d}/fetch"):
            batch, dataloader_iter = _next_batch(trainer, dataloader_iter)
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

    if trainer.cfg.training.get("cuda_empty_cache_every_step", True) and torch.cuda.is_available():
        with torch.profiler.record_function("cuda_empty_cache"):
            torch.cuda.empty_cache()

    metrics = _average_metrics(accum_metrics)
    metrics.update(batch_stats)
    metrics["grad_norm"] = grad_norm
    metrics["lr"] = trainer.optimizer.param_groups[0]["lr"]
    if len(trainer.optimizer.param_groups) > 1:
        metrics["lr_min"] = min(pg["lr"] for pg in trainer.optimizer.param_groups)
    trainer._add_step_metrics(metrics)
    return dataloader_iter, metrics


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
    trace_path_raw = profile_cfg.get("trace_path", None)

    trainer, dataloader_iter, checkpoint = _prepare_trainer(cfg)
    max_steps, base_lr, warmup_steps = _schedule_values(trainer, cfg)
    accum_steps = trainer._validate_accum_steps(
        cfg.training.get("gradient_accumulation_steps", 1),
    )
    trainer._accum_steps = accum_steps

    start_step = int(trainer.global_step)
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
                "ce_impl": os.environ.get("BGKIT_DECODER_CE_IMPL", "<unset>"),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    step = start_step
    warmup_metrics: list[dict[str, float]] = []
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
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metrics = dict(metrics)
        metrics["wall_ms"] = (time.perf_counter() - t0) * 1000.0
        metrics["step"] = step
        warmup_metrics.append(metrics)
        print(json.dumps({"event": "warmup_step", **metrics}, sort_keys=True), flush=True)
        step += 1

    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    measured_metrics: list[dict[str, float]] = []
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
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            metrics = dict(metrics)
            metrics["wall_ms"] = (time.perf_counter() - t0) * 1000.0
            metrics["step"] = step
            measured_metrics.append(metrics)
            print(json.dumps({"event": "profiled_step", **metrics}, sort_keys=True), flush=True)
            step += 1
            prof.step()

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=topk)
    print(table)

    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = checkpoint_dir / f"profile_training_step_{cfg.training.phase}_{ts}.json"
    trace_path = Path(str(trace_path_raw)) if trace_path_raw else None
    if trace_path is not None:
        prof.export_chrome_trace(str(trace_path))

    report = {
        "phase": cfg.training.phase,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "start_step": start_step,
        "end_step_exclusive": step,
        "warmup_metrics": warmup_metrics,
        "measured_metrics": measured_metrics,
        "topk": topk,
        "profile_table": table,
        "trace_path": str(trace_path) if trace_path is not None else None,
        "training_shape": {
            "decoder_lora": OmegaConf.to_container(
                cfg.training.get("decoder_lora", {}),
                resolve=True,
            ),
            "gradient_accumulation_steps": int(
                cfg.training.get("gradient_accumulation_steps", 1),
            ),
            "max_batch_tokens": int(cfg.training.get("max_batch_tokens", 0)),
            "max_sample_length": int(cfg.training.get("max_sample_length", 0)),
        },
        "env": {
            "BGKIT_DECODER_CE_IMPL": os.environ.get("BGKIT_DECODER_CE_IMPL"),
            "BGKIT_GDN_BACKEND": os.environ.get("BGKIT_GDN_BACKEND"),
            "FLA_GDR_FUSE_GATE_BWD": os.environ.get("FLA_GDR_FUSE_GATE_BWD"),
            "FLA_GDR_FUSE_DQKG_WY": os.environ.get("FLA_GDR_FUSE_DQKG_WY"),
            "FLA_GDR_RECOMPUTE_WY_DW": os.environ.get("FLA_GDR_RECOMPUTE_WY_DW"),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"profile_report={report_path}", flush=True)


if __name__ == "__main__":
    main()
