#!/usr/bin/env python
"""Memory-utilization profiler for packed training microbatches.

Runs one packed microbatch (forward + backward) at progressively larger
``max_batch_tokens`` values and records:

    * ``torch.cuda.max_memory_allocated`` post-forward + post-backward
    * System used delta (``/proc/meminfo`` MemAvailable delta, GB)
    * Wall-clock per microbatch
    * Effective samples per microbatch (distribution over the dataset)

The measurement goes through
``memory_budget_scope("profile_step", cap_gb=None)`` so values land in the
usual log format.

Usage (inside the training container)::

    python scripts/profile_packed_memory.py training=phase1_step3 \
        compute=dgx_spark \
        +profile.budgets='[8192,16384,32768,49152,65536]' \
        +profile.gradient_checkpointing_variants='[true,false]'

The script iterates both ``max_batch_tokens`` and
``gradient_checkpointing`` variants, resets CUDA peaks between runs, and
writes a markdown-style summary + a JSON report to
``$CHECKPOINT_DIR/profile_packed_memory_<phase>_<timestamp>.json``.

If a phase's setup needs a checkpoint that's missing, the run is
``skip``-ed with a clear reason — profiling continues for the remaining
budgets / variants.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

# Allow running without an editable install (mirrors scripts/train.py).
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from bgkit.utils.deltanet_patch import patch_gated_delta_rule_numerics
from bgkit.utils.logging import setup_logging
from bgkit.utils.memory_budget import (
    collect_memory_diagnostics,
    memory_budget_scope,
)
from bgkit.utils.reproducibility import set_seed
from bgkit.utils.triton_alloc_patch import patch_triton_allocator
from bgkit.utils.triton_patch import patch_triton_autotuner

DEFAULT_BUDGETS: tuple[int, ...] = (4096, 8192, 16384, 24576)
DEFAULT_GC_VARIANTS: tuple[bool, ...] = (True,)


def _mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            meminfo = {
                k.strip(): v.strip()
                for k, v in (line.split(":", 1) for line in f)
            }
        avail_kb = int(meminfo["MemAvailable"].split()[0])
        return avail_kb * 1024 / 1e9
    except (OSError, KeyError, ValueError, IndexError):
        return 0.0


def _extract_microbatch_count(batch) -> int:
    """Best-effort count of samples in a packed microbatch."""
    if isinstance(batch, dict):
        for key in (
            "content_cu_seqlens",
            "cu_seqlens",
            "cu_file_seqlens",
            "cu_repo_seqlens",
        ):
            if key in batch and torch.is_tensor(batch[key]):
                return int(batch[key].shape[0]) - 1
        # Generic "len" hints
        for key in ("languages", "splice_lengths", "splice_starts"):
            if key in batch and hasattr(batch[key], "__len__"):
                return len(batch[key])
    if isinstance(batch, (list, tuple)):
        return len(batch)
    return 1


def _create_trainer_from_cfg(cfg: DictConfig):
    """Mirrors ``scripts/train.py::_create_trainer`` so the profiler uses
    the exact same wiring as a real training run."""
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
    raise NotImplementedError(f"Training phase {phase!r} not supported by profiler.")


def _apply_overrides(
    cfg: DictConfig,
    *,
    max_batch_tokens: int,
    gradient_checkpointing: bool,
) -> None:
    """Mutate ``cfg`` in place to apply the per-run overrides."""
    with open_dict(cfg):
        cfg.training.max_batch_tokens = int(max_batch_tokens)
        # Eval budget doesn't matter for the profiler (we read the train loader),
        # but keep it consistent so nothing chokes on reading the value.
        cfg.training.max_batch_tokens_eval = int(max_batch_tokens)
        cfg.compute.gradient_checkpointing = bool(gradient_checkpointing)


def _reset_cuda_state() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _run_one_microbatch(
    cfg: DictConfig,
    *,
    max_batch_tokens: int,
    gradient_checkpointing: bool,
) -> dict:
    """Set up the trainer, pull one microbatch, run fwd+bwd, record stats."""
    _apply_overrides(
        cfg,
        max_batch_tokens=max_batch_tokens,
        gradient_checkpointing=gradient_checkpointing,
    )

    result: dict = {
        "max_batch_tokens": int(max_batch_tokens),
        "gradient_checkpointing": bool(gradient_checkpointing),
        "status": "unknown",
    }

    try:
        trainer = _create_trainer_from_cfg(cfg)
    except Exception as exc:
        result["status"] = "skip"
        result["skip_reason"] = f"trainer-construction: {type(exc).__name__}: {exc}"
        return result

    try:
        trainer.setup()
    except FileNotFoundError as exc:
        result["status"] = "skip"
        result["skip_reason"] = f"setup-missing-file: {exc}"
        return result
    except Exception as exc:
        result["status"] = "skip"
        result["skip_reason"] = f"setup-failed: {type(exc).__name__}: {exc}"
        return result

    try:
        batch_iter = iter(trainer.train_dataloader)
        batch = next(batch_iter)
    except StopIteration:
        result["status"] = "skip"
        result["skip_reason"] = "train-dataloader-empty"
        del trainer
        _reset_cuda_state()
        return result
    except Exception as exc:
        result["status"] = "skip"
        result["skip_reason"] = f"batch-fetch: {type(exc).__name__}: {exc}"
        del trainer
        _reset_cuda_state()
        return result

    sample_count = _extract_microbatch_count(batch)

    _reset_cuda_state()

    sys_pre_gb = collect_memory_diagnostics().get("mem/system_used_gb", 0.0)
    avail_pre_gb = _mem_available_gb()

    with memory_budget_scope("profile_step", cap_gb=None) as stats:
        t0 = time.perf_counter()
        try:
            trainer.optimizer.zero_grad()
            # _forward_backward runs a complete fwd + bwd (no optimizer.step).
            fb_metrics = trainer._forward_backward(batch)
            # Materialize metrics so any lazy cuda work flushes.
            import contextlib

            for v in fb_metrics.values():
                if hasattr(v, "item"):
                    with contextlib.suppress(Exception):
                        v.item()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_fwbw = time.perf_counter() - t0
            # Peak after backward is what matters for training-step headroom.
            cuda_peak_post_bwd = (
                torch.cuda.max_memory_allocated() / 1e9
                if torch.cuda.is_available()
                else 0.0
            )
        except torch.cuda.OutOfMemoryError as exc:
            result["status"] = "oom"
            result["error"] = f"CUDA OOM: {exc}"
            del trainer, batch
            _reset_cuda_state()
            return result
        except Exception as exc:
            import traceback

            result["status"] = "error"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc().splitlines()[-25:]
            del trainer, batch
            _reset_cuda_state()
            return result

    sys_post_gb = collect_memory_diagnostics().get("mem/system_used_gb", 0.0)
    avail_post_gb = _mem_available_gb()

    result.update(
        {
            "status": "ok",
            "sample_count": int(sample_count),
            "wall_ms": round(t_fwbw * 1000.0, 2),
            "cuda_peak_gb": round(stats.cuda_peak_gb, 3),
            "cuda_peak_post_bwd_gb": round(cuda_peak_post_bwd, 3),
            "cuda_pre_gb": round(stats.cuda_pre_gb, 3),
            "cuda_post_gb": round(stats.cuda_post_gb, 3),
            "system_pre_gb": round(sys_pre_gb, 3),
            "system_post_gb": round(sys_post_gb, 3),
            "system_delta_gb": round(sys_post_gb - sys_pre_gb, 3),
            "mem_available_pre_gb": round(avail_pre_gb, 3),
            "mem_available_post_gb": round(avail_post_gb, 3),
            "mem_available_delta_gb": round(avail_pre_gb - avail_post_gb, 3),
        },
    )

    del trainer, batch
    _reset_cuda_state()
    return result


def _format_markdown_table(rows: list[dict]) -> str:
    header = (
        "| max_batch_tokens | grad_ckpt | status | samples | wall_ms | "
        "cuda_peak_gb | cuda_peak_post_bwd_gb | sys_delta_gb | note |"
    )
    sep = "|---:|:---:|:---:|---:|---:|---:|---:|---:|:---|"
    lines = [header, sep]
    for r in rows:
        note = r.get("skip_reason") or r.get("error") or ""
        lines.append(
            "| {mbt} | {gc} | {status} | {n} | {wall} | {peak} | "
            "{peak_b} | {sysd} | {note} |".format(
                mbt=r.get("max_batch_tokens", "-"),
                gc="T" if r.get("gradient_checkpointing") else "F",
                status=r.get("status", "-"),
                n=r.get("sample_count", "-"),
                wall=r.get("wall_ms", "-"),
                peak=r.get("cuda_peak_gb", "-"),
                peak_b=r.get("cuda_peak_post_bwd_gb", "-"),
                sysd=r.get("system_delta_gb", "-"),
                note=str(note)[:80],
            ),
        )
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    patch_triton_allocator()
    patch_triton_autotuner()
    patch_gated_delta_rule_numerics()
    setup_logging()
    set_seed(cfg.get("seed", 42))

    phase = cfg.get("training", {}).get("phase", "unknown")

    profile_cfg = cfg.get("profile", {}) or {}
    budgets = list(
        profile_cfg.get("budgets", DEFAULT_BUDGETS),
    )
    gc_variants = [
        bool(v) for v in profile_cfg.get(
            "gradient_checkpointing_variants", DEFAULT_GC_VARIANTS,
        )
    ]
    num_microbatches = int(profile_cfg.get("num_microbatches", 1))

    print(f"[profile] phase={phase}")
    print(f"[profile] budgets={budgets}")
    print(f"[profile] gradient_checkpointing_variants={gc_variants}")
    print(f"[profile] num_microbatches={num_microbatches}")

    results: list[dict] = []
    for gc_flag in gc_variants:
        for budget in budgets:
            print(
                f"\n[profile] --- run: max_batch_tokens={budget} "
                f"gradient_checkpointing={gc_flag} ---",
            )
            # Sample ``num_microbatches`` to stabilize timing + capture
            # sample-count variance. We build a fresh trainer per
            # microbatch to avoid any warm-up effects bleeding across
            # budget boundaries.
            per_run_samples: list[int] = []
            final_result: dict | None = None
            for _i in range(num_microbatches):
                cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                # Hydra gives us a mutable copy but ``struct`` is set.
                OmegaConf.set_struct(cfg_copy, True)
                r = _run_one_microbatch(
                    cfg_copy,
                    max_batch_tokens=budget,
                    gradient_checkpointing=gc_flag,
                )
                if r.get("status") == "ok":
                    per_run_samples.append(int(r.get("sample_count", 0)))
                final_result = r
                # Bail on hard failures — no point retrying OOM at the same budget.
                if r.get("status") != "ok":
                    break
            if final_result is None:
                continue
            if per_run_samples:
                final_result["sample_count_median"] = int(
                    np.median(per_run_samples),
                )
                final_result["sample_count_min"] = int(min(per_run_samples))
                final_result["sample_count_max"] = int(max(per_run_samples))
            results.append(final_result)

            # If this budget OOM'd at gc=False, larger budgets will too.
            if final_result.get("status") == "oom":
                print(
                    f"[profile] OOM at budget={budget} gc={gc_flag} — "
                    "skipping larger budgets in this variant.",
                )
                break

    # --- Output ---
    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = checkpoint_dir / f"profile_packed_memory_{phase}_{ts}.json"
    markdown_path = checkpoint_dir / f"profile_packed_memory_{phase}_{ts}.md"

    summary = {
        "phase": phase,
        "timestamp": ts,
        "budgets": budgets,
        "gradient_checkpointing_variants": gc_variants,
        "results": results,
    }
    report_path.write_text(json.dumps(summary, indent=2))
    markdown_path.write_text(_format_markdown_table(results))

    print("\n" + "=" * 60)
    print(f"[profile] phase={phase}")
    print(_format_markdown_table(results))
    print("=" * 60)
    print(f"[profile] wrote {report_path}")
    print(f"[profile] wrote {markdown_path}")


def _bootstrap_without_hydra_fallback() -> None:
    """Allow ``python scripts/profile_packed_memory.py --help`` outside hydra.

    Not strictly needed, but mirrors patterns used elsewhere in scripts/.
    """


if __name__ == "__main__":
    main()
