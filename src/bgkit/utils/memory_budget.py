"""Scoped memory-budget accounting for phase-boundary enforcement.

On a unified-memory device (GB10 / DGX Spark: one 128 GB pool for CPU
and GPU) training, evaluation, and generation all compete for the
same pool.  Without a budget contract, the first subsystem to
overshoot triggers Linux swap; once the Python heap is paged out,
every subsequent access faults and the run thrashes to a halt.

:func:`memory_budget_scope` makes that contract explicit.  Every
expensive scope declares its peak budget, reclaims the allocator's
reserved-but-unused blocks at entry, captures the in-scope peak, and
raises informatively if the peak blows the cap.  The complement:
periodic polls (the existing ``memory_abort_system_used_gb`` rail)
can miss sub-step transients; a scoped peak measurement cannot.

Typical use::

    with memory_budget_scope("gen_eval", cap_gb=30):
        metrics = self._run_generation_eval()

The scope also returns a :class:`MemoryScopeStats` for callers that
want to *derive* operation sizing from the measured budget (e.g.
``max_new_tokens`` from available GB + model shape) rather than
hard-coding magic numbers::

    with memory_budget_scope("gen_eval", cap_gb=30) as stats:
        max_new = derive_max_new_tokens(cap_gb=stats.cap_gb, ...)
        ...

Entry side-effects (both enabled by default):

* ``gc.collect()`` — drop Python-side reference cycles before the
  next phase allocates.
* ``torch.cuda.empty_cache()`` — return the allocator's reserved
  blocks to the OS pool.  Critical on unified memory: reserved
  blocks that aren't currently in use are still physical pages
  competing with the next phase's claim.
* ``torch.cuda.reset_peak_memory_stats()`` — zero the peak tracker
  so the reading on exit reflects only in-scope allocation.

Exit side-effects:

* Peak is read via ``torch.cuda.max_memory_allocated()`` (accurate
  over the whole scope duration; a periodic poll is not needed).
* If ``cap_gb`` is set and the peak exceeded it, raises
  :class:`MemoryBudgetExceeded` with the scope name, declared cap,
  observed peak, and pre/post system memory.  The run aborts
  *before* the swap/OOM cascade — a legible failure, not silent
  thrashing.
* Host-side memory (``system_used_gb``, ``proc_rss_gb``) is logged
  pre/post but is *not* the basis for the cap: the unified-pool
  CUDA peak is the better proxy for the transient, and host RSS
  takes minutes to recover from pagecache behavior so post-scope
  RSS lags the actual reclaim.

The scope is cheap (on the order of a single GC pass + a
cudaDeviceSynchronize-free peak read) and safe to nest — each scope
tracks its own peak via the reset.
"""

from __future__ import annotations

import gc
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import structlog

logger = structlog.get_logger()


def collect_memory_diagnostics() -> dict[str, float]:
    """Collect memory-usage metrics for leak/fragmentation diagnosis.

    On DGX Spark / GB10 the GPU and CPU share one 128 GB pool, so a
    training-side leak surfaces as system-memory thrashing rather
    than a discrete OOM.  This helper exposes both the torch
    allocator's view (allocated / reserved, current / peak) and the
    kernel's view (process RSS / VSZ, system-wide MemAvailable) so a
    single log event captures all angles of the unified pool.

    Returns floats (GB for memory) in a metrics-friendly shape.  Cheap
    enough to call every log step; the torch allocator queries are
    O(1) internal reads.
    """
    import torch

    out: dict[str, float] = {}
    if torch.cuda.is_available():
        out["mem/cuda_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        out["mem/cuda_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        out["mem/cuda_max_allocated_gb"] = (
            torch.cuda.max_memory_allocated() / 1e9
        )
        out["mem/cuda_max_reserved_gb"] = (
            torch.cuda.max_memory_reserved() / 1e9
        )
    try:
        with open("/proc/self/statm") as f:
            fields = f.read().split()
        page_size = os.sysconf("SC_PAGESIZE")
        out["mem/proc_rss_gb"] = int(fields[1]) * page_size / 1e9
        out["mem/proc_vsz_gb"] = int(fields[0]) * page_size / 1e9
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open("/proc/meminfo") as f:
            meminfo = {
                k.strip(): v.strip()
                for k, v in (line.split(":", 1) for line in f)
            }
        avail_kb = int(meminfo["MemAvailable"].split()[0])
        total_kb = int(meminfo["MemTotal"].split()[0])
        out["mem/system_available_gb"] = avail_kb * 1024 / 1e9
        out["mem/system_used_gb"] = (total_kb - avail_kb) * 1024 / 1e9
    except (OSError, KeyError, ValueError, IndexError):
        pass
    return out


class MemoryBudgetExceeded(RuntimeError):
    """Raised when a scoped operation's peak allocation blows its cap.

    Prefer this over silent swap / OOM-kill: the caller gets the scope
    name, declared cap, and observed peak so they can either raise the
    cap, shrink the operation, or redesign the phase boundary.
    """


@dataclass
class MemoryScopeStats:
    """Snapshot of a memory-budget scope.

    All numeric fields are GB.  ``cuda_peak_gb`` is the in-scope peak
    of ``torch.cuda.memory_allocated()``; ``cuda_pre_gb`` /
    ``cuda_post_gb`` are instantaneous readings at entry / exit.  The
    ``system_*`` fields are host-level readings (unified pool usage).
    """

    name: str
    cap_gb: float | None
    cuda_pre_gb: float = 0.0
    cuda_post_gb: float = 0.0
    cuda_peak_gb: float = 0.0
    system_pre_gb: float = 0.0
    system_post_gb: float = 0.0
    rss_pre_gb: float = 0.0
    rss_post_gb: float = 0.0
    extra: dict = field(default_factory=dict)


def _reclaim() -> None:
    """Best-effort reclamation of allocator caches.

    ``gc.collect()`` drops Python-side ref cycles; ``empty_cache``
    returns reserved-but-unused CUDA blocks to the pool.  On unified
    memory those reserved blocks are physical pages that the next
    phase cannot otherwise see.
    """
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cuda_allocated_gb() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


def _cuda_peak_gb() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def _reset_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@contextmanager
def memory_budget_scope(
    name: str,
    *,
    cap_gb: float | None = None,
    reclaim_on_enter: bool = True,
    reclaim_on_exit: bool = True,
) -> Iterator[MemoryScopeStats]:
    """Enter a memory-budgeted scope.  See module docstring for rationale.

    Args:
        name: Stable scope name; appears in log events and exception
            messages.  Use a subsystem-level name (``"evaluate"``,
            ``"gen_eval"``, ``"save_checkpoint"``) rather than a
            call-site name so that enforcement semantics match the
            architectural phase boundary.
        cap_gb: Maximum in-scope CUDA peak allocation, in GB.  None
            disables enforcement (scope still logs).  If the peak
            exceeds the cap, :class:`MemoryBudgetExceeded` is raised
            at exit.
        reclaim_on_enter: Run ``gc.collect()`` and
            ``torch.cuda.empty_cache()`` before entering.  Default
            True — composable scopes require this, so the prior
            phase's reserved cache doesn't leak into the new
            phase's budget.
        reclaim_on_exit: Same on exit.  Default True so the next
            phase sees a clean baseline.

    Yields:
        :class:`MemoryScopeStats` whose fields are populated on exit.
        Callers can read ``stats.cap_gb`` mid-scope to derive
        operation sizing, but allocation-measurement fields are
        filled only after the scope exits.
    """
    pre_diag = collect_memory_diagnostics()
    stats = MemoryScopeStats(
        name=name,
        cap_gb=cap_gb,
        cuda_pre_gb=pre_diag.get("mem/cuda_allocated_gb", 0.0),
        system_pre_gb=pre_diag.get("mem/system_used_gb", 0.0),
        rss_pre_gb=pre_diag.get("mem/proc_rss_gb", 0.0),
    )

    if reclaim_on_enter:
        _reclaim()
    _reset_peak()

    logger.info(
        "mem_scope_enter",
        scope=name,
        cap_gb=cap_gb,
        cuda_pre_gb=round(stats.cuda_pre_gb, 3),
        system_pre_gb=round(stats.system_pre_gb, 3),
        rss_pre_gb=round(stats.rss_pre_gb, 3),
    )

    try:
        yield stats
    finally:
        stats.cuda_peak_gb = _cuda_peak_gb()
        stats.cuda_post_gb = _cuda_allocated_gb()

        if reclaim_on_exit:
            _reclaim()

        post_diag = collect_memory_diagnostics()
        stats.system_post_gb = post_diag.get("mem/system_used_gb", 0.0)
        stats.rss_post_gb = post_diag.get("mem/proc_rss_gb", 0.0)

        logger.info(
            "mem_scope_exit",
            scope=name,
            cap_gb=cap_gb,
            cuda_peak_gb=round(stats.cuda_peak_gb, 3),
            cuda_delta_gb=round(
                stats.cuda_peak_gb - stats.cuda_pre_gb, 3,
            ),
            cuda_post_gb=round(stats.cuda_post_gb, 3),
            system_pre_gb=round(stats.system_pre_gb, 3),
            system_post_gb=round(stats.system_post_gb, 3),
            system_delta_gb=round(
                stats.system_post_gb - stats.system_pre_gb, 3,
            ),
            rss_delta_gb=round(
                stats.rss_post_gb - stats.rss_pre_gb, 3,
            ),
        )

        # Delta-peak semantics: how much NEW memory did this scope allocate
        # above its entry baseline. Absolute-peak semantics blamed the scope
        # for pre-existing training state resident at scope entry (e.g.,
        # gen_eval entering with 30 GB of training state already on GPU
        # trivially exceeded any sane scope cap even when the scope itself
        # allocated 240 MB). The cap now measures scope-local cost.
        cuda_delta_gb = stats.cuda_peak_gb - stats.cuda_pre_gb
        if cap_gb is not None and cuda_delta_gb > cap_gb:
            raise MemoryBudgetExceeded(
                f"scope={name!r} blew cuda-delta budget: "
                f"delta={cuda_delta_gb:.2f} GB > cap={cap_gb:.2f} GB "
                f"(pre={stats.cuda_pre_gb:.2f} GB, "
                f"peak={stats.cuda_peak_gb:.2f} GB, "
                f"system_pre={stats.system_pre_gb:.2f} GB, "
                f"system_post={stats.system_post_gb:.2f} GB). "
                f"Either raise the cap, shrink the operation, or "
                f"redesign the phase boundary so concurrent live "
                f"footprints don't sum past the physical ceiling.",
            )
