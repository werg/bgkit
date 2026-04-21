#!/usr/bin/env python
"""Memory-utilization profiler for Phase 2 KB-scale trajectory batches.

Phase 2 uses ``KRKBTrainer``, which differs from Phase 1 trainers in two
key ways:

1. The dataloader yields lists of ``KBSample`` objects (identity collation
   via ``_collate_kb``), not token-budget-sized packed dicts.  There is no
   ``TokenBudgetBatchSampler`` — batch size is a plain integer
   ``cfg.batch_size``.

2. Memory per step is driven by the number of trajectory steps per sample
   and the number of articles per bgkit turn, not by a raw token count.
   The dominant controllable knobs are:
   - ``batch_size``: number of KBSample objects per training step.
   - ``max_samples_per_dataset_default``: caps how many trajectories are
     loaded from each dataset (indirectly sets how complex the trajectory
     mix is).

The profiler sweeps ``batch_size`` over a configurable range, runs one
``_forward_backward`` per size, and records CUDA peak, system delta, and
wall-clock.  It also prints per-trajectory structural stats (number of
bgkit turns, number of browse turns, approx article count) so the operator
can correlate memory with workload shape.

Usage (inside the training container)::

    python scripts/profile_packed_memory_phase2.py \\
        +experiment=phase2_kb_stage_a \\
        +profile.batch_sizes='[1,2,4,8]' \\
        +profile.gradient_checkpointing_variants='[true,false]'

    # or stage B:
    python scripts/profile_packed_memory_phase2.py \\
        +experiment=phase2_kb_stage_b \\
        +profile.batch_sizes='[1,2,4]'

The script writes a markdown summary + JSON report to
``$CHECKPOINT_DIR/profile_packed_memory_phase2_{stage}_{ts}.json``.

Do NOT run while live training is holding the GPU.
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

DEFAULT_BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8)
DEFAULT_GC_VARIANTS: tuple[bool, ...] = (True,)


def _mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            meminfo = {k.strip(): v.strip() for k, v in (line.split(":", 1) for line in f)}
        avail_kb = int(meminfo["MemAvailable"].split()[0])
        return avail_kb * 1024 / 1e9
    except (OSError, KeyError, ValueError, IndexError):
        return 0.0


def _inspect_kb_batch(batch) -> dict:
    """Extract structural stats from a list[KBSample] batch.

    Returns a dict with:
    - n_samples: number of KBSample objects
    - total_bgkit_turns: sum of bgkit tool-call turns across samples
    - total_browse_turns: sum of browse tool-call turns across samples
    - total_traj_steps: total trajectory steps (all turn kinds)
    """
    if not isinstance(batch, list):
        return {"n_samples": 1, "total_bgkit_turns": 0, "total_browse_turns": 0,
                "total_traj_steps": 0}
    n_bgkit = 0
    n_browse = 0
    n_steps = 0
    for sample in batch:
        traj = getattr(sample, "trajectory", None) or []
        n_steps += len(traj)
        for turn in traj:
            kind = getattr(turn, "kind", None)
            if kind is not None:
                name = kind.name if hasattr(kind, "name") else str(kind)
                if "BGKIT" in name:
                    n_bgkit += 1
                elif "BROWSE" in name:
                    n_browse += 1
    return {
        "n_samples": len(batch),
        "total_bgkit_turns": n_bgkit,
        "total_browse_turns": n_browse,
        "total_traj_steps": n_steps,
    }


def _reset_cuda_state() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _apply_overrides(
    cfg: DictConfig,
    *,
    batch_size: int,
    gradient_checkpointing: bool,
) -> None:
    """Mutate ``cfg`` in place to apply per-run overrides."""
    with open_dict(cfg):
        cfg.batch_size = int(batch_size)
        cfg.compute.gradient_checkpointing = bool(gradient_checkpointing)
        # Eval budget: set to 1 so we don't waste memory on the eval split
        # sizing path during setup. The profiler only exercises train_dataloader.
        if not hasattr(cfg.training, "max_eval_samples"):
            cfg.training.max_eval_samples = 16
        else:
            cfg.training.max_eval_samples = 16


def _run_one_batch(
    cfg: DictConfig,
    *,
    batch_size: int,
    gradient_checkpointing: bool,
) -> dict:
    """Set up the KRKBTrainer, pull one trajectory batch, run fwd+bwd."""
    _apply_overrides(
        cfg,
        batch_size=batch_size,
        gradient_checkpointing=gradient_checkpointing,
    )

    phase = cfg.get("training", {}).get("phase", "phase2_kb")
    stage = cfg.get("training", {}).get("stage", "A")

    result: dict = {
        "batch_size": int(batch_size),
        "gradient_checkpointing": bool(gradient_checkpointing),
        "phase": phase,
        "stage": stage,
        "status": "unknown",
    }

    try:
        from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

        trainer = KRKBTrainer(cfg)
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

    # Structural stats BEFORE forward (no GPU needed)
    batch_stats = _inspect_kb_batch(batch)
    result.update(batch_stats)

    _reset_cuda_state()

    sys_pre_gb = collect_memory_diagnostics().get("mem/system_used_gb", 0.0)
    avail_pre_gb = _mem_available_gb()

    with memory_budget_scope("profile_phase2_step", cap_gb=None) as stats:
        t0 = time.perf_counter()
        try:
            trainer.optimizer.zero_grad()
            fb_metrics = trainer._forward_backward(batch)
            # Flush any lazy CUDA work
            import contextlib

            for v in fb_metrics.values():
                if hasattr(v, "item"):
                    with contextlib.suppress(Exception):
                        v.item()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_fwbw = time.perf_counter() - t0
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
            "wall_ms": round(t_fwbw * 1000.0, 2),
            "wall_ms_per_sample": round(t_fwbw * 1000.0 / max(batch_size, 1), 2),
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
            "fb_loss": float(fb_metrics.get("loss", 0.0)),
            "fb_tokens": float(fb_metrics.get("tokens", 0.0)),
        },
    )

    del trainer, batch
    _reset_cuda_state()
    return result


def _format_markdown_table(rows: list[dict]) -> str:
    header = (
        "| batch_size | gc | status | samples | bgkit_turns | browse_turns |"
        " wall_ms | wall_ms/samp | cuda_peak_gb | cuda_peak_post_bwd_gb |"
        " sys_delta_gb | note |"
    )
    sep = (
        "|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|"
    )
    lines = [header, sep]
    for r in rows:
        note = r.get("skip_reason") or r.get("error") or ""
        lines.append(
            "| {bs} | {gc} | {status} | {n} | {bt} | {brt} |"
            " {wall} | {wallps} | {peak} | {peak_b} | {sysd} | {note} |".format(
                bs=r.get("batch_size", "-"),
                gc="T" if r.get("gradient_checkpointing") else "F",
                status=r.get("status", "-"),
                n=r.get("n_samples", "-"),
                bt=r.get("total_bgkit_turns", "-"),
                brt=r.get("total_browse_turns", "-"),
                wall=r.get("wall_ms", "-"),
                wallps=r.get("wall_ms_per_sample", "-"),
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

    phase = cfg.get("training", {}).get("phase", "phase2_kb")
    stage = cfg.get("training", {}).get("stage", "A")
    label = f"{phase}_stage{stage}"

    profile_cfg = cfg.get("profile", {}) or {}
    batch_sizes = list(profile_cfg.get("batch_sizes", DEFAULT_BATCH_SIZES))
    gc_variants = [
        bool(v) for v in profile_cfg.get(
            "gradient_checkpointing_variants", DEFAULT_GC_VARIANTS,
        )
    ]
    num_runs = int(profile_cfg.get("num_runs", 1))

    print(f"[profile_phase2] phase={phase} stage={stage}")
    print(f"[profile_phase2] batch_sizes={batch_sizes}")
    print(f"[profile_phase2] gradient_checkpointing_variants={gc_variants}")
    print(f"[profile_phase2] num_runs={num_runs}")

    results: list[dict] = []
    for gc_flag in gc_variants:
        for bs in batch_sizes:
            print(
                f"\n[profile_phase2] --- run: batch_size={bs} "
                f"gradient_checkpointing={gc_flag} ---",
            )

            per_run_wall: list[float] = []
            final_result: dict | None = None
            for _i in range(num_runs):
                cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                OmegaConf.set_struct(cfg_copy, True)
                r = _run_one_batch(
                    cfg_copy,
                    batch_size=bs,
                    gradient_checkpointing=gc_flag,
                )
                if r.get("status") == "ok":
                    per_run_wall.append(float(r.get("wall_ms", 0.0)))
                final_result = r
                if r.get("status") != "ok":
                    break

            if final_result is None:
                continue
            if per_run_wall:
                final_result["wall_ms_median"] = round(float(np.median(per_run_wall)), 2)
                final_result["wall_ms_min"] = round(float(min(per_run_wall)), 2)
                final_result["wall_ms_max"] = round(float(max(per_run_wall)), 2)
            results.append(final_result)

            # OOM at this batch_size → larger sizes will also OOM, bail early.
            if final_result.get("status") == "oom":
                print(
                    f"[profile_phase2] OOM at batch_size={bs} gc={gc_flag} — "
                    "skipping larger batch sizes in this variant.",
                )
                break

    # --- Output ---
    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = checkpoint_dir / f"profile_packed_memory_phase2_{label}_{ts}.json"
    markdown_path = checkpoint_dir / f"profile_packed_memory_phase2_{label}_{ts}.md"

    summary = {
        "phase": phase,
        "stage": stage,
        "label": label,
        "timestamp": ts,
        "batch_sizes": batch_sizes,
        "gradient_checkpointing_variants": gc_variants,
        "results": results,
    }
    report_path.write_text(json.dumps(summary, indent=2))
    markdown_path.write_text(_format_markdown_table(results))

    print("\n" + "=" * 72)
    print(f"[profile_phase2] {label}")
    print(_format_markdown_table(results))
    print("=" * 72)
    print(f"[profile_phase2] wrote {report_path}")
    print(f"[profile_phase2] wrote {markdown_path}")


if __name__ == "__main__":
    main()
