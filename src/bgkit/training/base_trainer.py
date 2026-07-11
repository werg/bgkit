"""Base trainer: wandb logging, LR scheduling, checkpointing.

Custom training loops — too many heterogeneous training phases for
HF Trainer or Lightning. No Accelerate for now (ICE trains on one GPU
with bf16 autocast). Add Accelerate later for Phase 1/2.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import os
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import structlog
import torch
from omegaconf import OmegaConf

from bgkit.training.checkpoint_manager import CheckpointManager
from bgkit.training.checkpoint_registry import (
    CheckpointRegistry,
    RegistryEntry,
    normalize_checkpoint_name,
    resolve_latest_checkpoint,
)
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.training.gradient_utils import clip_grad_norm
from bgkit.training.interruption import GracefulInterruptor
from bgkit.training.live_config import LiveConfig
from bgkit.training.scheduling import cosine_with_warmup
from bgkit.utils.memory_budget import (
    collect_memory_diagnostics as _collect_memory_diagnostics,
)
from bgkit.utils.memory_budget import (
    memory_budget_scope,
)

logger = structlog.get_logger()


class _DevicePrefetcher:
    """Prefetch dataloader batches to GPU on a background CUDA stream.

    Overlaps host→device transfer with ongoing GPU compute so the next
    batch is ready by the time the current forward/backward finishes.
    Existing ``.to(device)`` calls in ``_forward_backward`` become no-ops
    since the tensors are already on the target device.
    """

    def __init__(self, iterator, device):
        import torch

        self.iterator = iterator
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._next_batch = None
        # ``_exhausted`` distinguishes "underlying iterator returned
        # StopIteration" (genuine end-of-epoch) from "staged batch was dropped
        # externally" (``_release_training_transients`` nulls ``_next_batch``
        # before an eval/save scope to free its device memory). Without this
        # flag a dropped staged batch looks identical to exhaustion, so the
        # next ``__next__`` raised StopIteration and the train loop spuriously
        # rolled the epoch over — resetting an ``sort_samples_ascending``
        # dataloader back to its smallest samples on EVERY save/eval (the
        # 2026-06-10 post-save loss spike + survivor collapse).
        self._exhausted = False
        self._prefetch()

    def _to_device(self, batch):
        import torch

        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _prefetch(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self._next_batch = None
            self._exhausted = True
            return
        if self.stream is None:
            self._next_batch = batch
            return
        import torch

        with torch.cuda.stream(self.stream):
            self._next_batch = self._to_device(batch)

    def __next__(self):
        # The staged batch may have been dropped externally (a memory-budget
        # scope called _release_training_transients). Re-stage it rather than
        # treating it as end-of-epoch — only a real StopIteration from the
        # underlying iterator (``_exhausted``) ends the epoch.
        if self._next_batch is None and not self._exhausted:
            self._prefetch()
        if self.stream is not None:
            import torch

            torch.cuda.current_stream().wait_stream(self.stream)
        if self._next_batch is None:
            raise StopIteration
        batch = self._next_batch
        self._prefetch()
        return batch


def _coerce_empty_cache_cadence(training_val: Any, compute_default: Any) -> int:
    """Resolve ``cuda_empty_cache_every_step`` into an integer cadence.

    Accepts bool or int from either training-scope cfg or compute-scope
    cfg fallback:
        False / 0 / None → 0 (off)
        True / 1         → 1 (every step)
        N (int > 1)      → N (every N steps)

    Used by both ``BaseTrainer.train`` startup and the live-config branch
    so behavior is identical on first read and on hot updates.
    """
    val = training_val if training_val is not None else compute_default
    if isinstance(val, bool):
        return 1 if val else 0
    try:
        n = int(val)
    except (TypeError, ValueError):
        return 0
    return max(0, n)


def _average_metrics(accum_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average metrics across accumulation micro-batches.

    Numeric values (including 0-dim CUDA tensors from _forward_backward)
    are averaged.  A single ``.item()`` call at the end converts any
    remaining tensors to Python floats — this is the only GPU sync point,
    avoiding per-micro-batch synchronisation inside the accumulation loop.
    """
    if len(accum_metrics) == 1:
        m = accum_metrics[0]
        return {
            k: v.item() if hasattr(v, "item") else v
            for k, v in m.items()
        }

    # Union of keys across all microbatches — keys only present in some
    # microbatches (e.g. ``loss_qa`` when the InterleavingDataLoader fed a
    # mix of recon + QA batches in this accumulation window) average over
    # the subset they appear in. Previously we only took the first
    # microbatch's keys, silently dropping per-source loss tracking.
    result: dict[str, float] = {}
    keys: set[str] = set()
    for m in accum_metrics:
        keys.update(m.keys())
    for key in keys:
        values = [m[key] for m in accum_metrics if key in m]
        if values and isinstance(values[0], (int, float)):
            result[key] = sum(values) / len(values)
        elif values and hasattr(values[0], "item"):
            avg = values[0].detach().float() if hasattr(values[0], "detach") else values[0]
            for value in values[1:]:
                avg = avg + (value.detach().float() if hasattr(value, "detach") else value)
            result[key] = (avg / len(values)).item()
        else:
            result[key] = values[-1] if values else 0.0
    return result


class BaseTrainer(ABC):
    """Base class for all BgKIT trainers.

    Provides:
    - Training loop with LR scheduling
    - WandB logging
    - Checkpoint save/load with phase metadata

    Subclasses can declare ``LIVE_CONFIG_FIELDS`` to enable live tuning
    of hyperparameters via a JSON control file (written to
    ``checkpoints/control.json`` by default).  The dict maps
    control-file key → instance attribute name.  Simple numeric fields
    are applied automatically; override ``apply_live_config`` for
    fields needing custom validation.

    Extensibility hooks (override in subclasses):
    - ``_pre_step_hook()``: called at the top of each training step
      (before LR schedule). Use for dataloader rebuilds, curriculum changes.
    - ``_post_optimizer_step(step)``: called after optimizer.step().
      Use for per-step bookkeeping like bidi warmup.
    - ``_add_step_metrics(metrics)``: add trainer-specific metrics to
      the step dict before logging (e.g. bidi_alpha).
    - ``_build_training_state(...)``: build training state dict for
      checkpointing (override to add curriculum fields).
    - ``_create_dataloader_iter()``: create the dataloader iterator.
      Override to disable DevicePrefetcher or customize iteration.
    - ``_pre_train_loop()``: called after all setup but before the
      training loop starts. Use for resume-time rebuilds.
    """

    LIVE_CONFIG_FIELDS: ClassVar[dict[str, str]] = {}

    #: Registry of live-config handler methods for keys that need custom
    #: validation (e.g. nullable fields, range checks).  Maps control-file
    #: key → unbound method name on ``self``.  Merged across the MRO so
    #: subclasses add handlers without rewriting ``apply_live_config``.
    LIVE_CONFIG_HANDLERS: ClassVar[dict[str, str]] = {
        "max_batch_tokens": "_handle_max_batch_tokens",
        "max_batch_tokens_eval": "_handle_max_batch_tokens_eval",
        "min_sample_length": "_handle_min_sample_length",
        "max_sample_length": "_handle_max_sample_length",
        "max_grad_norm": "_handle_max_grad_norm",
    }

    #: Steps between structured log messages. Override in subclass.
    _log_every: int = 10

    #: Whether to use DevicePrefetcher for async batch transfer.
    _use_device_prefetcher: bool = True

    def __init__(self, cfg):
        self.cfg = cfg
        self.global_step = 0
        self.epoch = 0
        self._last_checkpoint_path: str | None = None
        self._schedule_params: dict[str, float] | None = None
        self._training_state: dict | None = None
        self._input_sources: dict[str, str] | None = None
        # Map of component name → source path (or None for cold start).
        # Populated by trainers via ``register_checkpoint_source`` during
        # ``setup()``; emitted as a hard-to-miss banner by
        # ``_log_startup_banner`` immediately after ``setup()`` returns.
        # The banner forces the operator to confirm checkpoint provenance
        # at every launch, which catches "I set the wrong config key and
        # the trainer silently ran on random weights" class of bugs.
        # Components that intentionally cold-start should register
        # ``None`` so the banner explicitly flags it instead of staying
        # silent.
        self._startup_sources: dict[str, str | None] = {}
        self._startup_extras: dict[str, str | int | float | None] = {}
        self._startup_notes: list[str] = []
        self._accum_steps = 1
        self._dataloader_invalidated = False  # set True in _pre_step_hook to force re-iter
        # Microbatches already consumed from the current epoch's dataloader.
        # Persisted and restored on resume so we don't rewind the iterator
        # back to batch 0 every restart -- critical for length-sorted
        # samplers where early-epoch batches are systematically short and
        # out-of-distribution for a model mid-trained on longer samples
        # (diagnosed 2026-04-19 when resuming Step 3 step2000 caused the
        # trained head to over-compress short content, spiking loss and
        # degrading eval through the re-adaptation).
        self._microbatches_in_epoch: int = 0
        # Optimizer type: set from config, overridden by _create_optimizer()
        self._optimizer_type: str = cfg.training.get("optimizer", "muon")
        self._muon_exclude_set: frozenset[int] = frozenset()

    @abstractmethod
    def setup(self) -> None:
        """Create model, optimizer, dataloader. Called before train()."""

    @abstractmethod
    def _forward_backward(self, batch) -> dict[str, float]:
        """Forward pass + scaled backward. No optimizer ops.

        Subclasses implement this. Must:
        - Compute loss
        - Call (loss / self._accum_steps).backward()
        - Return dict with unscaled metrics (e.g. {"loss": loss.item()})
        - NOT call optimizer.zero_grad(), optimizer.step(), or clip_grad_norm
        """

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        """Run evaluation. Returns dict of metrics."""

    def trainable_parameters(self) -> list:
        """Parameters for gradient clipping. Override in subclasses."""
        return [p for p in self.model.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # Startup banner registration (called from trainer.setup())
    # ------------------------------------------------------------------

    def register_checkpoint_source(
        self, component: str, source: str | None,
    ) -> None:
        """Record where a component's weights came from for the startup banner.

        Pass ``source=None`` to flag a cold start (pristine HF weights /
        random init) — the banner marks it loudly so the operator can
        confirm the cold start is intentional. Components: typically
        ``encoder``, ``decoder``, ``optimizer_state``, and any
        phase-specific extras like ``teacher_encoder``.
        """
        self._startup_sources[component] = source

    def register_startup_extra(
        self, key: str, value: str | int | float | None,
    ) -> None:
        """Record a phase-specific value (target_ratio, anchor count, etc.).

        Surfaces in the startup banner under "Phase-specific".
        """
        self._startup_extras[key] = value

    def register_startup_note(self, note: str) -> None:
        """Add a free-form note for the startup banner.

        Use for things like 'decoder cold-start (intentional, no Qwen
        ckpt exists for current encoder)'. Keep each note short — they
        list verbatim under "Notes:".
        """
        self._startup_notes.append(note)

    def _log_startup_banner(self) -> None:
        """Emit the loud "PLEASE CHECK" banner. Called by ``train()`` after
        ``setup()`` completes, so every trainer (current and future) gets
        the banner for free.

        Backward-compat: if a trainer hasn't been ported to call
        ``register_checkpoint_source`` yet, fall back to populating from
        ``_input_sources`` (the existing lineage convention) so the
        banner is still informative on older trainers.
        """
        from bgkit.training.startup_banner import log_startup_banner

        sources = dict(self._startup_sources)
        notes = list(self._startup_notes)
        # Back-compat fallback: trainers that haven't been updated still
        # populate _input_sources from their _resolve_*_checkpoint
        # methods. Carry those over so the banner is informative.
        if not sources and self._input_sources:
            sources = {k: v for k, v in self._input_sources.items()}
            notes.append(
                "trainer has not yet registered explicit checkpoint "
                "sources — banner populated from legacy _input_sources",
            )

        encoder_src = (
            sources.pop("encoder", None)
            or sources.pop("step1", None)
            or sources.pop("bgkit", None)
        )
        decoder_src = sources.pop("decoder", None)
        opt_src = sources.pop("optimizer_state", None)
        log_startup_banner(
            phase=str(self.cfg.training.get("phase", "<unknown>")),
            run_name=str(self.cfg.get("run_name", "<unnamed>")),
            encoder_source=encoder_src,
            decoder_source=decoder_src,
            optimizer_state_source=opt_src,
            extras={**sources, **self._startup_extras},
            notes=notes,
        )

    def _post_step(self, step: int) -> None:
        """Hook called after each optimizer step. Override for per-step bookkeeping.

        .. deprecated:: Use ``_post_optimizer_step`` instead.
        """
        del step

    def _post_optimizer_step(self, step: int) -> None:
        """Hook called after optimizer.step(). Override for per-step bookkeeping."""
        self._post_step(step)

    def _pre_step_hook(self) -> None:
        """Hook called at the top of each training step (before LR schedule).

        Override for dataloader rebuilds, curriculum transitions, etc.
        Return value is ignored.
        """
        return None

    def _post_lr_schedule(self, step: int) -> None:
        """Hook called after the LR schedule is applied, before the accumulation loop.

        Override to apply per-param-group LR adjustments (e.g. local warmup ramps
        for newly added param groups at stage transitions).
        """
        del step

    def _add_step_metrics(self, metrics: dict[str, float]) -> None:
        """Add trainer-specific metrics to the step dict before logging.

        Override to inject metrics like bidi_alpha, compression ratio, etc.
        Modify ``metrics`` in place.
        """
        del metrics

    def _build_training_state(
        self,
        es_best: float | None,
        es_evals_without_improvement: int,
        wandb_run,
    ) -> dict:
        """Build training state dict for checkpointing.

        Override to add curriculum or trainer-specific fields.
        """
        return {
            "es_best": es_best,
            "es_evals_without_improvement": es_evals_without_improvement,
            "wandb_run_id": wandb_run.id if wandb_run is not None else None,
            "microbatches_in_epoch": self._microbatches_in_epoch,
        }

    def _wrap_dataloader_iter(self, iterator, *, use_prefetch: bool | None = None):
        """Wrap a raw train-dataloader iterator for runtime consumption.

        ``use_prefetch=False`` is used for resume-time skipping so we do not
        pay host->device copies for microbatches that will be discarded.
        """
        if use_prefetch is None:
            use_prefetch = self._use_device_prefetcher
        if not use_prefetch:
            it = iterator
        else:
            device = getattr(self, "device", None)
            if device is not None and hasattr(device, "type"):
                it = _DevicePrefetcher(iterator, device)
            else:
                it = iterator
        self._active_dataloader_iter = it
        return it

    def _create_dataloader_iter(self, *, use_prefetch: bool | None = None):
        """Create an iterator over the train dataloader.

        Default wraps in _DevicePrefetcher for async GPU transfer.
        Override to disable prefetching (e.g. to save memory); make
        sure overrides also assign ``self._active_dataloader_iter`` so
        :meth:`_release_training_transients` can clear the prefetched
        batch at phase boundaries.
        """
        return self._wrap_dataloader_iter(
            iter(self.train_dataloader), use_prefetch=use_prefetch,
        )

    def _pre_train_loop(self) -> None:
        """Called after all train() setup but before the loop starts.

        Override for resume-time rebuilds (e.g. L1 dataloader rebuild).
        """
        return None

    def _dynamic_ckpt_managed_models(self) -> list[tuple[str, torch.nn.Module]]:
        """Models managed by the memory-driven dynamic ckpt scheduler.

        Subclasses override to return ``[(label, model), ...]`` pairs whose
        gradient-checkpointing mode the scheduler is allowed to flip at
        runtime. Default returns ``[]`` (scheduler is a no-op).

        Each registered model must support either HF v5+ ``GradientCheckpointingLayer``
        semantics or our ``PrunedBidirectionalQwen35.gradient_checkpointing_{enable,disable}``
        API. ``set_gradient_checkpointing_mode`` in
        ``bgkit.training.gradient_utils`` handles both.
        """
        return []

    def _detect_unified_memory(self) -> bool:
        """True iff the CUDA device's reported total ~= system MemTotal.

        On a unified-memory device (GB10 / DGX Spark) the GPU and host share
        one physical pool. ``torch.cuda.mem_get_info()[0]`` then returns
        Linux ``MemFree``, which is depressed by the kernel's page cache.
        Reclaimable page-cache pages are NOT counted as "free" even though
        the kernel will gladly evict them when CUDA asks. The correct
        signal on unified memory is ``MemAvailable`` (free + reclaimable).
        """
        import torch as _t
        if not _t.cuda.is_available():
            return False
        try:
            cuda_total = _t.cuda.get_device_properties(0).total_memory
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        sys_total = int(line.split()[1]) * 1024
                        break
                else:
                    return False
            return cuda_total >= 0.8 * sys_total
        except (OSError, ValueError, IndexError, AttributeError):
            return False

    def _get_free_gb_signal(self) -> float:
        """Free-memory signal for the dynamic_ckpt scheduler.

        Discrete GPU: ``torch.cuda.mem_get_info()[0]`` (VRAM free).
        Unified memory: ``MemAvailable`` (system free + reclaimable cache),
        because the kernel evicts page cache on demand for CUDA allocations,
        so mmap pages do not actually contend for our budget.
        """
        import torch as _t
        free_bytes, _ = _t.cuda.mem_get_info()
        if getattr(self, "_unified_memory", False):
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            return int(line.split()[1]) * 1024 / 1e9
            except (OSError, ValueError, IndexError):
                pass
        return free_bytes / 1e9

    def _init_dynamic_ckpt_scheduler(self) -> None:
        """Resolve config + initialize per-step state for the memory-driven
        ckpt scheduler.

        Reads (in priority order): ``training.dynamic_ckpt`` (per-phase
        override) > ``compute.memory.dynamic_ckpt`` (host-level default) >
        built-in defaults. Default ``enabled: true`` host-wide; trainers that
        haven't registered any models in ``_dynamic_ckpt_managed_models``
        get a quiet no-op.
        """
        from collections import deque

        tcfg = self.cfg.training
        compute_mem = (self.cfg.get("compute", {}) or {}).get("memory", {}) or {}
        compute_dyn = compute_mem.get("dynamic_ckpt", {}) or {}
        train_dyn = tcfg.get("dynamic_ckpt", {}) or {}
        self._unified_memory = self._detect_unified_memory()

        def _get(key, default):
            v = train_dyn.get(key, None) if hasattr(train_dyn, "get") else None
            if v is None:
                v = (
                    compute_dyn.get(key, default)
                    if hasattr(compute_dyn, "get")
                    else default
                )
            return v

        self._dyn_ckpt_enabled = bool(_get("enabled", True))
        self._dyn_ckpt_window = int(_get("window", 50))
        # Signal: ``torch.cuda.mem_get_info()[0] / 1e9`` = free GB the CUDA
        # allocator can still claim. Stating thresholds as "free room
        # remaining" makes the policy adaptive to total host memory and
        # consistent with the abort semantics (abort = no room).
        # All thresholds are in GB of free memory the trigger requires:
        #   * upshift to megatron when free < megatron_upshift_when_free_below_gb
        #   * upshift to full when free < full_upshift_when_free_below_gb
        #   * downshift requires the worst-case (min) free over the window
        #     to be above target_threshold + downshift_margin_gb (hysteresis)
        self._dyn_megatron_upshift_when_free_below = float(
            _get("megatron_upshift_when_free_below_gb", 15.0)
        )
        self._dyn_full_upshift_when_free_below = float(
            _get("full_upshift_when_free_below_gb", 8.0)
        )
        self._dyn_downshift_margin = float(_get("downshift_margin_gb", 5.0))
        self._dyn_ckpt_min_steps_in_mode = int(_get("min_steps_in_mode", 50))
        # Adaptive CUDA cache flush. ``flush_when_free_below_gb`` is the
        # outer ceiling - flush only triggers when room is genuinely scarce.
        # ``flush_min_slack_gb`` skips the sync when there's nothing to
        # actually reclaim (slack = reserved - allocated; if slack ~= 0 the
        # pool is fully in-use and ``empty_cache()`` returns nothing).
        self._dyn_flush_when_free_below = float(
            _get("flush_when_free_below_gb", 20.0)
        )
        self._dyn_flush_min_slack = float(_get("flush_min_slack_gb", 3.0))
        self._dyn_cache_clear_cooldown = int(_get("cache_clear_cooldown_steps", 5))
        self._last_cache_clear_step = -10**9

        self._dyn_ckpt_models = self._dynamic_ckpt_managed_models()
        self._dyn_ckpt_window_data: deque = deque(maxlen=self._dyn_ckpt_window)
        self._dyn_ckpt_steps_in_mode = 0

        # Initial mode mirrors the resolved gradient_checkpointing config so
        # the runtime mode and the static config agree on first step.
        from bgkit.training.gradient_utils import (
            _resolve_gradient_checkpointing_mode,
        )
        resolved = _resolve_gradient_checkpointing_mode(self.cfg)
        if resolved is False or resolved is None:
            self._ckpt_mode = "off"
        elif resolved == "megatron":
            self._ckpt_mode = "megatron"
        else:
            self._ckpt_mode = "full"

        if self._dyn_ckpt_enabled and self._dyn_ckpt_models:
            logger.info(
                "dynamic_ckpt_scheduler_armed",
                mode=self._ckpt_mode,
                managed=[label for label, _ in self._dyn_ckpt_models],
                signal="mem_available_gb" if self._unified_memory else "cuda_free_gb",
                unified_memory=self._unified_memory,
                flush_when_free_below_gb=self._dyn_flush_when_free_below,
                flush_min_slack_gb=self._dyn_flush_min_slack,
                megatron_upshift_when_free_below_gb=self._dyn_megatron_upshift_when_free_below,
                full_upshift_when_free_below_gb=self._dyn_full_upshift_when_free_below,
                downshift_margin_gb=self._dyn_downshift_margin,
                window=self._dyn_ckpt_window,
            )

    def _dynamic_ckpt_step(self, step: int) -> None:
        """Memory-driven scheduler tick. Called once per optimizer step.

        Signal: ``torch.cuda.mem_get_info()[0] / 1e9`` = free GB the
        allocator can still claim. Thresholds are stated as "free room
        remaining" so the same config adapts across hosts of different
        total memory and remains consistent with the abort semantics
        (abort = no room left).

        Two-tier response:

        1. **Adaptive cache flush** (cheap first response): when free is
           tight (``free_gb < flush_when_free_below_gb``) AND there's
           slack worth recovering (``reserved - allocated > flush_min_slack_gb``)
           AND cooldown has elapsed, call ``empty_cache()``. The slack
           guard skips the sync when the pool is fully in-use and a flush
           would return nothing.
        2. **Mode flip** (expensive, requires managed models): upshift on
           first breach (no hysteresis); downshift requires the window's
           **min** free to stay above ``target_threshold + downshift_margin``
           for ``min_steps_in_mode`` consecutive steps.
        """
        if not getattr(self, "_dyn_ckpt_enabled", False):
            return
        import torch as _t
        if not _t.cuda.is_available():
            return
        # CRITICAL: ``free_pre`` is the actual memory pressure before any
        # rescue. Use this for ALL mode-flip decisions. Using post-flush
        # free would hide the pressure (flush rescues, mode flip then sees
        # no problem, scheduler stays in off-mode forever while flush keeps
        # rescuing every step). Pre-flush free reveals the true working-set
        # demand. ``_get_free_gb_signal`` returns MemAvailable on unified
        # memory so reclaimable page cache (mmap data files) doesn't fake
        # pressure that isn't there.
        free_pre_gb = self._get_free_gb_signal()

        # Tier 1: adaptive cache flush. Slack-driven, cooldown-free.
        if free_pre_gb < self._dyn_flush_when_free_below:
            slack_gb = (
                _t.cuda.memory_reserved() - _t.cuda.memory_allocated()
            ) / 1e9
            if slack_gb >= self._dyn_flush_min_slack:
                _t.cuda.empty_cache()
                self._last_cache_clear_step = step
                new_free_gb = self._get_free_gb_signal()
                logger.info(
                    "cuda_cache_cleared_adaptive",
                    step=step,
                    pre_free_gb=free_pre_gb,
                    post_free_gb=new_free_gb,
                    reclaimed_gb=new_free_gb - free_pre_gb,
                    slack_gb=slack_gb,
                )

        # Tier 2: mode flip — only when managed models are registered.
        # IMPORTANT: window + threshold checks use ``free_pre_gb`` (pre-flush),
        # not the post-flush value. Otherwise frequent flushes mask sustained
        # pressure and the scheduler never upshifts.
        if not getattr(self, "_dyn_ckpt_models", None):
            return

        self._dyn_ckpt_window_data.append(free_pre_gb)
        self._dyn_ckpt_steps_in_mode += 1

        # Upshift: snap back on memory pressure (low free), no hysteresis.
        if (
            self._ckpt_mode == "off"
            and free_pre_gb < self._dyn_megatron_upshift_when_free_below
        ):
            self._apply_ckpt_mode(
                "megatron", step, "free_below_threshold", free_pre_gb,
            )
            return
        if (
            self._ckpt_mode == "megatron"
            and free_pre_gb < self._dyn_full_upshift_when_free_below
        ):
            self._apply_ckpt_mode(
                "full", step, "free_below_threshold", free_pre_gb,
            )
            return

        # Downshift: only after the window is full and we've dwelled enough.
        # Window-MIN over pre-flush readings: the worst-case sustained
        # pressure. If even the worst sample stays well above the
        # downshift target, the workload genuinely fits the safer mode.
        if len(self._dyn_ckpt_window_data) < self._dyn_ckpt_window:
            return
        if self._dyn_ckpt_steps_in_mode < self._dyn_ckpt_min_steps_in_mode:
            return

        worst_recent_free = min(self._dyn_ckpt_window_data)
        if (
            self._ckpt_mode == "full"
            and worst_recent_free
            > self._dyn_full_upshift_when_free_below + self._dyn_downshift_margin
        ):
            self._apply_ckpt_mode(
                "megatron", step, "free_above_threshold", free_pre_gb,
            )
        elif (
            self._ckpt_mode == "megatron"
            and worst_recent_free
            > self._dyn_megatron_upshift_when_free_below + self._dyn_downshift_margin
        ):
            self._apply_ckpt_mode(
                "off", step, "free_above_threshold", free_pre_gb,
            )

    def _apply_ckpt_mode(
        self, target: str, step: int, reason: str, free_gb: float,
    ) -> None:
        """Flip the mode across all managed models. Idempotent on no-change."""
        from bgkit.training.gradient_utils import set_gradient_checkpointing_mode
        if target == self._ckpt_mode:
            return
        prev = self._ckpt_mode
        for label, model in self._dyn_ckpt_models:
            try:
                set_gradient_checkpointing_mode(model, target)
            except Exception as exc:
                logger.warning(
                    "ckpt_mode_transition_failed",
                    component=label,
                    target=target,
                    error=str(exc),
                )
                return
        self._ckpt_mode = target
        self._dyn_ckpt_steps_in_mode = 0
        logger.info(
            "ckpt_mode_transition",
            step=step,
            from_mode=prev,
            to_mode=target,
            reason=reason,
            cuda_free_gb=free_gb,
        )

    @staticmethod
    def _validate_accum_steps(value) -> int:
        """Validate gradient_accumulation_steps config value."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"gradient_accumulation_steps must be int >= 1, got {value}")
        return value

    def _create_optimizer(
        self,
        param_groups: list[dict],
        default_lr: float,
        exclude_from_muon: frozenset[int] | None = None,
    ) -> object:
        """Create optimizer based on config: 'muon', 'adamw8bit', or 'adamw'.

        For muon: splits params into Muon (2D+) and AdamW (1D/embedding).
        For adamw8bit: tries bnb with explicit CUDA probe, raises on failure.
        For adamw: standard torch.optim.AdamW.

        Args:
            param_groups: List of param group dicts (standard PyTorch format).
            default_lr: Fallback LR for the optimizer constructor.
            exclude_from_muon: Set of param ``id()``s that should use AdamW
                even if they are 2D+ (e.g. embedding tables, lm_head).
        """
        import torch

        optimizer_type = self.cfg.training.get("optimizer", "muon")
        self._optimizer_type = optimizer_type
        self._muon_exclude_set = exclude_from_muon or frozenset()

        if optimizer_type == "muon":
            from bgkit.training.muon import Muon

            muon_ns_steps = self.cfg.training.get("muon_ns_steps", None)
            if muon_ns_steps is not None:
                muon_ns_steps = int(muon_ns_steps)
                if muon_ns_steps < 1:
                    raise ValueError(
                        f"training.muon_ns_steps must be >= 1, got {muon_ns_steps}"
                    )
                for group in param_groups:
                    group.setdefault("ns_steps", muon_ns_steps)

            split_groups = self._split_for_muon(param_groups, self._muon_exclude_set)
            optimizer = Muon(split_groups)
            muon_count = sum(
                sum(p.numel() for p in g["params"])
                for g in split_groups
                if g.get("use_muon")
            )
            adam_count = sum(
                sum(p.numel() for p in g["params"])
                for g in split_groups
                if not g.get("use_muon")
            )
            logger.info(
                "optimizer_created",
                type="muon",
                muon_params=muon_count,
                adam_params=adam_count,
                groups=len(split_groups),
                ns_steps=muon_ns_steps if muon_ns_steps is not None else 5,
            )
            return optimizer

        if optimizer_type == "adamw8bit":
            try:
                import bitsandbytes as bnb
            except ImportError as e:
                raise ImportError(
                    "optimizer: adamw8bit requires bitsandbytes. "
                    "Install with pip install bitsandbytes, or use "
                    "optimizer: adamw or optimizer: muon."
                ) from e

            # CUDA probe: verify bnb kernels actually work on this GPU
            device = getattr(self, "device", torch.device("cpu"))
            try:
                _test_p = torch.nn.Parameter(
                    torch.zeros(1, device=device, dtype=torch.bfloat16)
                )
                _test_opt = bnb.optim.AdamW8bit([_test_p], lr=1e-3)
                _test_p.grad = torch.ones_like(_test_p)
                _test_opt.step()
                del _test_opt, _test_p
            except Exception as e:
                raise RuntimeError(
                    f"bitsandbytes CUDA probe failed: {e}. "
                    "This GPU (sm_121 / Blackwell) may not be supported by bnb. "
                    "Use optimizer: adamw or optimizer: muon instead."
                ) from e

            optimizer = bnb.optim.AdamW8bit(param_groups, lr=default_lr)
            total = sum(sum(p.numel() for p in g["params"]) for g in param_groups)
            logger.info("optimizer_created", type="adamw8bit", params=total)
            return optimizer

        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(param_groups, lr=default_lr)
            total = sum(sum(p.numel() for p in g["params"]) for g in param_groups)
            logger.info("optimizer_created", type="adamw", params=total)
            return optimizer

        raise ValueError(
            f"Unknown optimizer type: {optimizer_type!r}. "
            "Supported: 'adamw', 'adamw8bit', 'muon'."
        )

    @staticmethod
    def _lora_param_ids(model) -> frozenset[int]:
        """Return param IDs of LoRA factors (``lora_A``, ``lora_B``) in ``model``.

        These MUST be excluded from Muon: Muon's Newton-Schulz orthogonalization
        plus its ``sqrt(d_out/d_in)`` rectangular rescaler is wrong for low-rank
        factors. For typical Qwen3.5 LoRA shapes (``r=16``, intermediate=3584),
        the rescaler over-amplifies ``lora_B`` (3584,16) updates by ~15x while
        leaving ``lora_A`` (16,1024) at 1x. Empirically (diagnosed 2026-05-09
        on Step 5 checkpoints): ``lora_B`` weight-norm drift was ~100x that
        of ``lora_A`` — effective rank-1 collapse — driving eval/loss creep
        while train loss stayed flat.

        LoRA factors should run through AdamW. Apply this helper alongside any
        per-trainer embed_tokens / lm_head exclusions.
        """
        ids = set()
        if model is None:
            return frozenset()
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # PEFT names contain ``.lora_A.`` or ``.lora_B.`` when the LoRA
            # factor is a child Module; native fused implementations may use
            # bare ``lora_a``/``lora_b`` parameter attrs.
            n = name.lower()
            lora_suffix = n.endswith((".lora_a.weight", ".lora_b.weight"))
            native_attr = name.split(".")[-1] in {"lora_a", "lora_b"}
            if ".lora_a." in n or ".lora_b." in n or lora_suffix or native_attr:
                ids.add(id(p))
        return frozenset(ids)

    def _split_for_muon(
        self, param_groups: list[dict], exclude_ids: frozenset[int] = frozenset()
    ) -> list[dict]:
        """Split param groups by ndim for Muon. Preserves all group metadata.

        Params with ndim >= 2 and not in exclude_ids get ``use_muon=True``.
        Everything else gets ``use_muon=False``.

        Output order: for each input group, the Muon subgroup comes first,
        then the AdamW subgroup. This preserves a stable ordering whether
        groups are built all at once or added incrementally (critical for
        optimizer state restore on checkpoint resume).
        """
        result: list[dict] = []
        for group in param_groups:
            meta = {k: v for k, v in group.items() if k != "params"}
            muon_ps = [
                p
                for p in group["params"]
                if p.ndim >= 2 and id(p) not in exclude_ids
            ]
            adam_ps = [
                p
                for p in group["params"]
                if p.ndim < 2 or id(p) in exclude_ids
            ]
            if muon_ps:
                result.append({"params": muon_ps, "use_muon": True, **meta})
            if adam_ps:
                result.append({"params": adam_ps, "use_muon": False, **meta})
        return result

    def _add_param_group_to_optimizer(self, group: dict) -> None:
        """Add a param group to the optimizer, applying Muon split if needed.

        For Muon optimizers, splits the group by ndim and adds up to two
        sub-groups. For AdamW/AdamW8bit, passes through unchanged.
        """
        if self._optimizer_type == "muon":
            split = self._split_for_muon([group], self._muon_exclude_set)
            for sg in split:
                self.optimizer.add_param_group(sg)
        else:
            self.optimizer.add_param_group(group)

    def train_step(self, batch) -> dict[str, float]:
        """Complete training step: zero_grad + forward_backward + clip + step.

        Public API for tests and standalone use. The train() loop calls
        _forward_backward() directly for accumulation support.
        """
        self.optimizer.zero_grad()
        metrics = {
            k: v.item() if hasattr(v, "item") else v
            for k, v in self._forward_backward(batch).items()
        }
        grad_norm = clip_grad_norm(
            self.trainable_parameters(),
            max_norm=getattr(self, "_max_grad_norm", 1.0),
        )
        if not math.isfinite(grad_norm):
            raise RuntimeError(
                f"NaN/Inf grad_norm in train_step (grad_norm={grad_norm}). "
                "This usually indicates a numerical stability issue."
            )
        self.optimizer.step()
        metrics["grad_norm"] = grad_norm
        return metrics

    def _named_parameters_for_optimizer(self):
        """Yield ``(name, param)`` pairs for every param the optimizer tracks.

        Used by the name-keyed optimizer state-dict save/load path so a
        resume can survive param-group topology changes (LoRA toggling,
        freeze flips, LoRA adapters added/removed, etc.). PyTorch's native
        ``optimizer.load_state_dict`` requires exact param-group count
        match and silently bails otherwise — we saw this fail when the
        LoRA freeze fix changed Step 3's decoder topology between save
        and load, discarding 2 000 steps of Muon momentum.

        Default walks ``self.model.named_parameters()`` with a ``model.``
        prefix. Subclasses with multi-module structures (e.g.
        encoder + decoder + projection) override to yield from each
        module with distinct prefixes. Names must be **stable across code
        changes** — anchored to module structure, not to anything that
        could renumber between runs.
        """
        if hasattr(self, "model") and isinstance(self.model, torch.nn.Module):
            for name, param in self.model.named_parameters():
                yield f"model.{name}", param
        else:
            raise NotImplementedError(
                f"{type(self).__name__} has no ``self.model`` attribute; "
                "override ``_named_parameters_for_optimizer()`` to yield "
                "(name, param) pairs across all trainable modules.",
            )

    def _build_optimizer_state_by_name(self) -> dict:
        """Produce a ``{param_name: per_param_state}`` dict from the optimizer.

        PyTorch's optimizer keys its per-parameter state by ``id(param)``,
        stable only within one process. Serializing the native state
        produces ``param_groups`` with integer ``params`` indices that
        depend on build order — fatal under any topology change. This
        method re-keys by stable module-path name instead, so the saved
        state survives param-group rebuilds.

        Asserts every optimizer-tracked param is reachable by
        ``_named_parameters_for_optimizer`` — if any aren't, their
        momentum is lost on the next resume and we want to know now.
        """
        opt_param_ids = {
            id(p) for pg in self.optimizer.param_groups for p in pg["params"]
        }
        seen_opt_ids: set[int] = set()
        state_by_name: dict = {}
        for name, param in self._named_parameters_for_optimizer():
            if id(param) in opt_param_ids:
                seen_opt_ids.add(id(param))
            if param in self.optimizer.state:
                # Copy the per-param state dict so the saved tensor isn't
                # aliased to the live optimizer buffer (torch.save will
                # serialize whatever we hand it, but an aliased dict is
                # fragile if the optimizer mutates mid-serialization).
                #
                # Downcast the large float buffers (momentum / moments — the
                # dominant ~12 GB of this run's Muon state for 1.586 B params)
                # to bf16 on disk to ~halve the checkpoint write volume. The
                # slow USB-HDD flush of that volume is what left the dirty-page
                # backlog that stalled the post-save CUDA allocation; halving it
                # halves the fsync time. Scalars like the integer ``step``
                # (numel == 1) and non-float tensors are left intact so they
                # round-trip EXACTLY. bf16 momentum/moments is standard practice
                # (running averages tolerate it; the value is upcast back to
                # fp32 on load) and convergence-neutral. ``.to(bf16)`` also
                # returns a fresh tensor, so the live buffer isn't aliased.
                state_by_name[name] = {
                    k: (
                        v.to(torch.bfloat16)
                        if isinstance(v, torch.Tensor)
                        and v.is_floating_point()
                        and v.numel() > 1
                        else v
                    )
                    for k, v in self.optimizer.state[param].items()
                }
        unreachable = len(opt_param_ids) - len(seen_opt_ids)
        if unreachable:
            raise RuntimeError(
                f"_build_optimizer_state_by_name: {unreachable} optimizer-tracked "
                f"param(s) have no name in _named_parameters_for_optimizer. Their "
                "state will not be saved and momentum will be lost on resume. "
                "Override _named_parameters_for_optimizer to yield every param "
                "the optimizer touches.",
            )
        return state_by_name

    def _legacy_optimizer_fallback(self, state_dicts: dict) -> bool:
        """TRANSITIONAL: load pre-refactor ``optimizer.pt`` state if present.

        Added 2026-04-19 so the in-flight Phase 1 Step 3 run can still be
        resumed from one of its pre-step-4000 checkpoints (which only
        have the legacy ``optimizer.pt`` artifact) if the container
        stalls before the next save under the new name-keyed format.
        Success prints ``optimizer_state_loaded_via_legacy_fallback``;
        failure prints ``optimizer_state_legacy_load_failed`` and
        returns False so the caller logs the standard
        ``_missing_using_fresh_moments`` warning and proceeds.

        Discard this method + its callers once all live checkpoints on
        disk are in the name-keyed format. At that point the block
        becomes dead code that silently masks a "someone resumed a very
        old checkpoint" bug.

        Returns True iff legacy state was found and loaded successfully.
        """
        legacy = state_dicts.get("optimizer")
        if legacy is None:
            return False
        try:
            self.optimizer.load_state_dict(legacy)
        except (ValueError, KeyError, RuntimeError) as exc:
            logger.warning(
                "optimizer_state_legacy_load_failed",
                error=str(exc)[:200],
                hint="param-group topology differs from legacy save; "
                "fresh moments",
            )
            return False
        logger.info(
            "optimizer_state_loaded_via_legacy_fallback",
            hint="transitional path; remove after 2026-04-19 refactor's "
            "checkpoints are all on disk",
        )
        return True

    def _optimizer_state_lookup_names(self, name: str) -> tuple[str, ...]:
        """Return saved-state names that may correspond to current ``name``.

        Decoder LoRA can be represented by PEFT or BgKIT's native wrapper.
        Their parameter paths differ even though they point to the same logical
        adapter tensors. Keep lookup aliases here so resumes preserve optimizer
        moments when configs move between the two implementations.
        """

        prefixes = ("decoder.backbone.", "model.backbone.")
        if not name.startswith(prefixes):
            return (name,)

        names = [name]
        seen = {name}

        def add(candidate: str) -> None:
            if candidate not in seen:
                seen.add(candidate)
                names.append(candidate)

        idx = 0
        while idx < len(names):
            current = names[idx]
            for prefix in prefixes:
                peft_prefix = f"{prefix}base_model.model."
                if current.startswith(peft_prefix):
                    add(prefix + current[len(peft_prefix) :])
                elif current.startswith(prefix):
                    add(peft_prefix + current[len(prefix) :])

            for lora_name in ("lora_A", "lora_B"):
                native_suffix = f".{lora_name}"
                peft_suffix = f".{lora_name}.default.weight"
                if current.endswith(peft_suffix):
                    add(current[: -len(".default.weight")])
                elif current.endswith(native_suffix):
                    add(f"{current}.default.weight")
            idx += 1

        return tuple(names)

    def _packed_optimizer_state_from_parts(
        self,
        param: torch.nn.Parameter,
        state_by_name: dict,
        source_names: tuple[str, ...],
    ) -> tuple[dict | None, tuple[str, ...]]:
        source_states = tuple(state_by_name.get(source_name) for source_name in source_names)
        if not all(isinstance(source_state, dict) for source_state in source_states):
            return None, ()

        packed: dict = {}
        common_keys = set(source_states[0])
        for source_state in source_states[1:]:
            common_keys &= set(source_state)
        for key in common_keys:
            values = tuple(source_state[key] for source_state in source_states)
            if all(isinstance(value, torch.Tensor) for value in values):
                tensors = tuple(value for value in values if isinstance(value, torch.Tensor))
                first = tensors[0]
                if (
                    first.ndim >= 1
                    and all(tensor.shape[1:] == first.shape[1:] for tensor in tensors)
                    and sum(tensor.shape[0] for tensor in tensors) == param.shape[0]
                    and first.shape[1:] == param.shape[1:]
                ):
                    packed[key] = torch.cat(tensors, dim=0)
                elif first.ndim == 0 and all(tensor.shape == first.shape for tensor in tensors):
                    stacked = torch.stack(tensors)
                    packed[key] = stacked.max()
                elif all(torch.equal(first, tensor) for tensor in tensors[1:]):
                    packed[key] = first
            elif all(value == values[0] for value in values[1:]):
                packed[key] = values[0]

        if not packed:
            return None, ()
        return packed, source_names

    def _packed_falcon_optimizer_state(
        self,
        name: str,
        param: torch.nn.Parameter,
        state_by_name: dict,
    ) -> tuple[dict | None, tuple[str, ...]]:
        """Merge old Falcon split optimizer moments for packed params."""
        if ".gate_up_proj." in name:
            return self._packed_optimizer_state_from_parts(
                param,
                state_by_name,
                (
                    name.replace(".gate_up_proj.", ".gate_proj."),
                    name.replace(".gate_up_proj.", ".up_proj."),
                ),
            )
        if ".qkv_proj." in name:
            return self._packed_optimizer_state_from_parts(
                param,
                state_by_name,
                (
                    name.replace(".qkv_proj.", ".q_proj."),
                    name.replace(".qkv_proj.", ".k_proj."),
                    name.replace(".qkv_proj.", ".v_proj."),
                ),
            )
        return None, ()

    @staticmethod
    def _restore_opt_tensor(v, device, target_float_dtype=torch.float32):
        """Move a saved optimizer-state value to ``device`` and cast bf16/fp16
        float buffers to ``target_float_dtype`` — the inverse of the bf16
        downcast applied in :meth:`_build_optimizer_state_by_name`.

        ``target_float_dtype`` must be the dtype a FRESH (unresumed) optimizer
        would use for this buffer (see :meth:`_fresh_opt_state_float_dtype`):

        - **Muon / fp32-master optimizers** keep fp32 state (the fp32
          ``master_param`` MUST stay fp32; momentum/exp_avg are re-upcast to
          fp32 inside ``Muon.step``). Default fp32 preserves this exactly.
        - **Plain torch AdamW on a bf16 model** creates its moments as
          ``zeros_like(param)`` → bf16, and does NOT self-heal. Upcasting them
          to fp32 on resume makes ``exp_avg.lerp_(grad, …)`` mix fp32 state with
          a bf16 grad → ``RuntimeError: expected dtype float for 'end'``. Passing
          ``param.dtype`` (bf16) keeps the resumed state matching a fresh run.

        Non-tensors and already-target-dtype buffers pass through unchanged, so
        loading stays backward-compatible (older fp32 checkpoints round-trip)."""
        if not isinstance(v, torch.Tensor):
            return v
        v = v.to(device)
        if v.is_floating_point() and v.dtype in (torch.bfloat16, torch.float16):
            v = v.to(target_float_dtype)
        return v

    def _fresh_opt_state_float_dtype(self, param: torch.nn.Parameter) -> torch.dtype:
        """The float dtype a FRESH optimizer state would use for ``param`` — the
        target for :meth:`_restore_opt_tensor`. Plain ``torch.optim.AdamW``
        builds moments as ``zeros_like(param)`` (== ``param.dtype``); Muon keeps
        an fp32 master + self-heals its moments to fp32, so fp32 is both correct
        and required there. Default fp32 for any other / unknown optimizer keeps
        the historical behavior and is safe (matches pre-fix loads)."""
        if getattr(self, "_optimizer_type", "muon") == "adamw":
            return param.dtype
        return torch.float32

    def _restore_optimizer_state_by_name(self, state_by_name: dict) -> None:
        """Install name-keyed optimizer state into the current optimizer.

        For each ``(name, param)`` the current optimizer tracks, if
        ``name`` appears in ``state_by_name`` we copy the saved
        per-parameter state into ``self.optimizer.state[param]``. Params
        with no saved state keep fresh moments. Names present in the save
        but absent from the current topology are silently dropped. Logs
        counts for visibility.

        Saved state tensors may be on a different device than the live
        params (e.g. a migration script that ran on CPU produced the
        ``optimizer_state_by_name.pt`` file). Move each tensor in the
        per-param state onto the live param's device before installing.
        Without this, the first Muon/Adam update crashes with "Expected
        all tensors to be on the same device" when the momentum buffer
        is on CPU and the grad is on cuda:0.

        Counts split into ``in_optimizer`` (param is currently in an
        optimizer param group — these are the ones that actually train)
        and ``frozen`` (param is reachable by name but isn't in the
        optimizer right now — restoring state is harmless and preserves
        momentum across freeze→unfreeze cycles, but it doesn't reflect
        any training cost). Asserts that every optimizer-tracked param
        is reachable by ``_named_parameters_for_optimizer`` — if any
        aren't, save/restore would silently lose state for them.
        """

        opt_param_ids = {
            id(p) for pg in self.optimizer.param_groups for p in pg["params"]
        }
        n_opt = len(opt_param_ids)

        matched_in_opt = 0
        matched_frozen = 0
        new_in_opt = 0
        new_frozen = 0
        matched_saved_names: set[str] = set()
        for name, param in self._named_parameters_for_optimizer():
            in_opt = id(param) in opt_param_ids
            saved_name = next(
                (
                    candidate
                    for candidate in self._optimizer_state_lookup_names(name)
                    if candidate in state_by_name
                ),
                None,
            )
            if saved_name is not None:
                matched_saved_names.add(saved_name)
                saved = state_by_name[saved_name]
                device = param.device
                tgt_dtype = self._fresh_opt_state_float_dtype(param)
                moved = {
                    k: self._restore_opt_tensor(v, device, tgt_dtype)
                    for k, v in saved.items()
                }
                self.optimizer.state[param] = moved
                if in_opt:
                    matched_in_opt += 1
                else:
                    matched_frozen += 1
            else:
                packed_state, packed_source_names = self._packed_falcon_optimizer_state(
                    name, param, state_by_name
                )
                if packed_state is not None:
                    matched_saved_names.update(packed_source_names)
                    device = param.device
                    tgt_dtype = self._fresh_opt_state_float_dtype(param)
                    moved = {
                        k: self._restore_opt_tensor(v, device, tgt_dtype)
                        for k, v in packed_state.items()
                    }
                    self.optimizer.state[param] = moved
                    if in_opt:
                        matched_in_opt += 1
                    else:
                        matched_frozen += 1
                elif in_opt:
                    new_in_opt += 1
                else:
                    new_frozen += 1
        skipped_from_save = len(state_by_name) - len(matched_saved_names)
        logger.info(
            "optimizer_state_restored_by_name",
            matched_in_optimizer=matched_in_opt,
            new_in_optimizer=new_in_opt,
            matched_frozen=matched_frozen,
            new_frozen=new_frozen,
            skipped_from_save=skipped_from_save,
        )

        accounted = matched_in_opt + new_in_opt
        if accounted != n_opt:
            raise RuntimeError(
                f"Optimizer tracks {n_opt} param tensors but only {accounted} of "
                f"them are reachable via _named_parameters_for_optimizer. The "
                f"unreachable {n_opt - accounted} param(s) will silently lose "
                "optimizer state across save/restore. Override "
                "_named_parameters_for_optimizer in your trainer to yield every "
                "param the optimizer touches.",
            )

    def _graceful_shutdown_save(
        self,
        *,
        checkpoint_dir: Path,
        registry,
        ckpt_manager,
        wandb_run,
        es_best,
        es_evals_without_improvement,
        step: int,
        interruptor,
        already_saved: bool,
    ) -> None:
        """Write a rescue checkpoint on SIGTERM/SIGINT.

        Called from every graceful-shutdown detection point (mid-accumulation,
        or at the end-of-step check). Two guarantees that keep the save inside a
        short ``docker stop`` grace window:

        * **Fast path only.** ``save_checkpoint`` routes through
          ``_write_checkpoint``, which writes to the NVMe ``_fast_checkpoint_dir``
          when configured. The NVMe copy is authoritative (resume prefers it), so
          we never block the grace window fsync'ing the ~15 GB checkpoint to the
          slow HDD.
        * **No HDD drain here.** We do NOT ``wait_idle`` on the async archiver.
          ``self._graceful_shutdown`` is set so the outer ``finally`` bounds its
          drain too. Anything not yet copied to the HDD is recovered on next
          startup by ``archive_pending_into``.

        The step watchdog is paused around the serialize so a slow fsync isn't
        mistaken for a hang and ``os._exit``'d mid-write (2026-06-10 regression).
        """
        self._graceful_shutdown = True
        if not already_saved:
            self._training_state = self._build_training_state(
                es_best, es_evals_without_improvement, wandb_run,
            )
            parent = self._registry_parent()
            self._release_training_transients()
            from bgkit.utils.step_watchdog import pause as _wd_pause
            from bgkit.utils.step_watchdog import resume as _wd_resume

            _wd_pause()
            try:
                with memory_budget_scope("save_checkpoint_shutdown"):
                    ckpt_path = self.save_checkpoint(checkpoint_dir)
            finally:
                _wd_resume()
            registry.register(self._build_registry_entry(
                ckpt_path, None, wandb_run,
                status="interrupted",
                parent_checkpoint=parent,
            ))
            if ckpt_manager is not None:
                ckpt_manager.record(ckpt_path, step, None)
                ckpt_manager.prune()
            (checkpoint_dir / ".last_checkpoint").write_text(str(ckpt_path))
        logger.info(
            "graceful_shutdown_complete",
            step=step,
            signal=interruptor.received_signal.name
            if interruptor.received_signal
            else None,
        )

    def save_checkpoint(
        self, checkpoint_dir: Path, metrics: dict[str, float] | None = None
    ) -> Path:
        """Save checkpoint with phase metadata and lineage."""
        # Capture per-param-group base_lrs into schedule_params so they
        # survive restart. Without this, live `lr` bumps that scale per-
        # group base_lrs (encoder vs decoder) are lost on restart — the
        # optimizer is rebuilt from yaml in setup() and the live handler
        # then no-ops because new == saved-global. See 04-26 perf notes.
        opt = getattr(self, "optimizer", None)
        if opt is not None and self._schedule_params is not None:
            self._schedule_params["per_group_base_lrs"] = [
                float(pg.get("base_lr", self._schedule_params.get("base_lr", 0.0)))
                for pg in opt.param_groups
            ]
        metadata = CheckpointMetadata(
            phase=self.cfg.training.phase,
            step=self.global_step,
            epoch=self.epoch,
            parent_checkpoint=self._last_checkpoint_path,
            metrics=metrics,
            schedule_params=self._schedule_params,
            training_state=self._training_state,
            optimizer_type=self._optimizer_type,
            run_name=self.cfg.get("run_name", None),
        )
        return self._write_checkpoint(
            checkpoint_dir,
            metadata,
            model=self.model.state_dict(),
            optimizer_state_by_name=self._build_optimizer_state_by_name(),
        )

    def _write_checkpoint(
        self, checkpoint_dir: Path, metadata: CheckpointMetadata, **state_dicts
    ) -> Path:
        """Write a checkpoint, routing through the NVMe fast-dir if configured.

        SINGLE source of truth for NVMe routing + async archive. Both the
        default ``save_checkpoint`` and any trainer override (which may save a
        different set of state dicts, e.g. encoder + two decoders) MUST go
        through here — otherwise an override silently writes to the slow HDD and
        skips archival (the 2026-06-10 routing bug). When ``_fast_checkpoint_dir``
        is set, the checkpoint is written there (fast fsync, no spinning-disk
        dirty-page spike) and queued for async copy to the HDD archive.
        """
        fast_dir = getattr(self, "_fast_checkpoint_dir", None)
        write_dir = fast_dir if fast_dir is not None else checkpoint_dir
        ckpt_path = save_checkpoint(write_dir, metadata, **state_dicts)
        self._last_checkpoint_path = str(ckpt_path)
        archiver = getattr(self, "_archiver", None)
        if fast_dir is not None and archiver is not None:
            archiver.enqueue(ckpt_path, fast_dir)
        return ckpt_path

    def _check_optimizer_type_compat(self, metadata: CheckpointMetadata) -> None:
        """Raise if saved optimizer type doesn't match current config."""
        saved = metadata.optimizer_type
        if saved is None:
            return  # old checkpoint, no type info — allow
        if saved != self._optimizer_type:
            raise RuntimeError(
                f"Optimizer type mismatch: checkpoint was saved with "
                f"'{saved}' but current config uses '{self._optimizer_type}'. "
                f"Cannot resume — start a fresh run or change the optimizer config."
            )

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load checkpoint and restore training state.

        Subclasses customize by overriding the hooks below instead of
        rewriting this method:

        * :meth:`_restore_model_state` — load weights (default loads
          ``state_dicts["model"]`` into ``self.model``).
        * :meth:`_restore_training_state` — restore subclass-specific
          fields from the ``training_state`` dict (e.g. curriculum
          overrides, stage number).  Called only when training_state is
          present in the checkpoint.
        * :meth:`_post_weight_load_hook` — run after weights + step are
          restored, before optimizer state is loaded.  The canonical
          place to rebuild the optimizer when trainable parameters
          depend on restored state (e.g. distillation stage, freeze
          schedule keyed off ``global_step``).
        * :meth:`_log_restore` — final log line.  Override when the
          subclass wants to include extra fields (stage, ratio, ...).
        """
        metadata, state_dicts = load_checkpoint(checkpoint_path)
        self._check_optimizer_type_compat(metadata)

        self._restore_model_state(state_dicts)

        self.global_step = metadata.step
        self.epoch = metadata.epoch
        self._last_checkpoint_path = str(checkpoint_path)
        if metadata.schedule_params is not None:
            self._schedule_params = metadata.schedule_params
        if metadata.training_state is not None:
            self._training_state = metadata.training_state
            self._microbatches_in_epoch = int(
                metadata.training_state.get("microbatches_in_epoch", 0),
            )
            self._restore_training_state(metadata.training_state)

        self._post_weight_load_hook()

        if "optimizer_state_by_name" in state_dicts:
            self._restore_optimizer_state_by_name(
                state_dicts["optimizer_state_by_name"],
            )
        elif not self._legacy_optimizer_fallback(state_dicts):
            logger.warning(
                "optimizer_state_missing_using_fresh_moments",
                hint="checkpoint predates the name-keyed optimizer state "
                "refactor; resuming with fresh moments",
            )

        self._log_restore()

    # ------------------------------------------------------------------
    # load_checkpoint hooks
    # ------------------------------------------------------------------

    def _restore_model_state(self, state_dicts: dict) -> None:
        """Load model weights from ``state_dicts``.

        Default implementation loads ``state_dicts["model"]`` into
        ``self.model``.  Override when the trainer holds a different
        set of modules (encoder-only, encoder + decoder, student /
        teacher pair, ...).
        """
        self.model.load_state_dict(state_dicts["model"])

    def _restore_training_state(self, training_state: dict) -> None:
        """Restore subclass-specific fields from ``training_state``.

        Default: no-op.  Called only when ``training_state`` is present
        in the checkpoint.  The base class has already stashed the dict
        on ``self._training_state`` and restored
        ``_microbatches_in_epoch`` before this hook fires.
        """
        del training_state

    def _post_weight_load_hook(self) -> None:
        """Run after weights + step + training_state are restored,
        before the optimizer state is loaded.

        Default: no-op.  The canonical place to rebuild the optimizer
        when trainable parameters depend on restored state (e.g. a
        distillation stage from ``training_state`` or a freeze schedule
        keyed off ``global_step``).
        """
        return None

    def _log_restore(self) -> None:
        """Emit the restore log line.  Override to include extra
        fields (stage, ratio, ...) alongside ``step``.
        """
        logger.info("restored_from_checkpoint", step=self.global_step)

    def _sync_epoch(self, epoch: int) -> None:
        """Propagate epoch to batch sampler and dataset for shuffling/variant diversity."""
        batch_sampler = getattr(self.train_dataloader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
        dataset = getattr(self.train_dataloader, "dataset", None)
        # Unwrap Subset → underlying dataset
        inner = getattr(dataset, "dataset", dataset)
        if hasattr(inner, "set_epoch"):
            inner.set_epoch(epoch)

    def apply_live_config(self, changes: dict) -> None:
        """Apply trainer-specific live config changes.

        Dispatch is two-stage:

        1. Keys in :attr:`LIVE_CONFIG_HANDLERS` (merged across MRO) are
           dispatched to the named method as ``self.<method>(value)``.
           Use for fields that need custom validation or that accept
           non-numeric values (e.g. ``None`` to clear an override).
        2. Keys in :attr:`LIVE_CONFIG_FIELDS` that are NOT covered by a
           handler fall through to the declarative numeric path:
           ``setattr(self, attr, value)`` with numeric coercion.

        Subclasses with dynamic / prefix-based keys (e.g. PruningDistill's
        ``stage_N_steps``) can still override ``apply_live_config``; call
        ``super().apply_live_config(changes)`` to preserve registry
        dispatch.
        """
        handlers: dict[str, str] = {}
        for cls in reversed(type(self).__mro__):
            handlers.update(getattr(cls, "LIVE_CONFIG_HANDLERS", {}))

        fields: dict[str, str] = {}
        for cls in reversed(type(self).__mro__):
            fields.update(getattr(cls, "LIVE_CONFIG_FIELDS", {}))

        for key, method_name in handlers.items():
            if key not in changes:
                continue
            getattr(self, method_name)(changes[key])

        for key, attr in fields.items():
            if key not in changes or key in handlers:
                continue
            val = changes[key]
            old = getattr(self, attr, None)
            if not isinstance(val, (int, float)):
                logger.warning("live_config_type_error", key=key, value=val, expected="numeric")
                continue
            setattr(self, attr, type(old)(val) if old is not None else float(val))
            logger.info("live_config_update", key=key, attr=attr, old=old, new=val)

    def _handle_target_ratio(self, val: float | int | None) -> None:
        """Live-config handler for ``target_ratio`` override.

        Sets ``self._target_ratio_override`` to ``None`` (resume the
        curriculum ramp) or to a validated float in ``(0, 1)``.  Shared
        across trainers that expose a compression-ratio override
        (DecoderInit, Compression, CommitEncoding).
        """
        if val is None:
            self._target_ratio_override = None
            logger.info("live_target_ratio_cleared", resuming="ramp")
            return
        if isinstance(val, (int, float)) and 0 < val < 1:
            self._target_ratio_override = float(val)
            logger.info(
                "live_target_ratio_update",
                target_ratio=self._target_ratio_override,
            )
            return
        logger.warning(
            "live_target_ratio_invalid",
            value=val,
            expected="None or float in (0, 1)",
        )

    # Subclasses with multi-level samplers (e.g. KRKBTrainer with separate
    # L0 / L1 sampler configs) override this list so the live handler
    # rebuilds every relevant dataclass.
    RATIO_SAMPLER_CFG_ATTRS: ClassVar[tuple[str, ...]] = (
        "_target_ratio_sampler_cfg",
    )

    def _handle_ratio_sampling_window_above(self, val: float | int) -> None:
        """Live-config handler for ``target_ratio_sampling_window_above``.

        Rebuilds every ``RatioSamplerConfig`` listed in
        ``RATIO_SAMPLER_CFG_ATTRS`` with the new ``window_above``. The
        config is a frozen dataclass so we use ``dataclasses.replace``.
        """
        if not isinstance(val, (int, float)) or float(val) < 0:
            logger.warning(
                "live_ratio_sampling_window_above_invalid",
                value=val,
                expected="non-negative float",
            )
            return
        new_val = float(val)
        for attr in self.RATIO_SAMPLER_CFG_ATTRS:
            cfg = getattr(self, attr, None)
            if cfg is None:
                continue
            old = cfg.window_above
            setattr(self, attr, dataclasses.replace(cfg, window_above=new_val))
            logger.info(
                "live_ratio_sampling_window_above_update",
                attr=attr,
                old=old,
                new=new_val,
            )

    def _handle_ratio_sampling_enabled(self, val: bool | int | float) -> None:
        """Live-config handler for ``sample_target_ratio_during_training``.

        Flips ``enabled`` on every ``RatioSamplerConfig`` listed in
        ``RATIO_SAMPLER_CFG_ATTRS``. When disabled, ``sample_ratio``
        returns the curriculum floor unchanged — useful to remove the
        per-microbatch shape variance that defeats CUDA caching-allocator
        block reuse.
        """
        if not isinstance(val, (bool, int, float)):
            logger.warning(
                "live_ratio_sampling_enabled_invalid",
                value=val,
                expected="bool",
            )
            return
        new_val = bool(val)
        for attr in self.RATIO_SAMPLER_CFG_ATTRS:
            cfg = getattr(self, attr, None)
            if cfg is None:
                continue
            old = cfg.enabled
            setattr(self, attr, dataclasses.replace(cfg, enabled=new_val))
            logger.info(
                "live_ratio_sampling_enabled_update",
                attr=attr,
                old=old,
                new=new_val,
            )

    def _handle_ratio_sampling_anchor_prob(self, val: float | int) -> None:
        """Live-config handler for ``target_ratio_anchor_sampling_prob``.

        Probability per microbatch of trying anchor-grid sampling
        (snap to one of ``RatioSamplerConfig.anchor_grid``) instead of
        uniform sampling within the window. Only effective when at least
        one anchor falls inside ``[floor, floor + window_above]`` —
        otherwise the anchor branch is a no-op and the ratio is sampled
        uniformly anyway.
        """
        if not isinstance(val, (int, float)) or not 0.0 <= float(val) <= 1.0:
            logger.warning(
                "live_ratio_anchor_sampling_prob_invalid",
                value=val,
                expected="float in [0, 1]",
            )
            return
        new_val = float(val)
        for attr in self.RATIO_SAMPLER_CFG_ATTRS:
            cfg = getattr(self, attr, None)
            if cfg is None:
                continue
            old = cfg.anchor_sampling_prob
            setattr(self, attr, dataclasses.replace(cfg, anchor_sampling_prob=new_val))
            logger.info(
                "live_ratio_anchor_sampling_prob_update",
                attr=attr,
                old=old,
                new=new_val,
            )

    def _rebuild_train_dataloader_with_budget(self, new_budget: int) -> None:
        """Rebuild the train dataloader with a new token budget.

        Subclasses must have set ``_train_lengths``, ``_train_collate_fn``,
        ``_num_workers``, and ``_pin_memory`` in their ``setup()`` method,
        and must expose ``train_dataset`` and ``train_sampler`` attributes.
        The method preserves the current ``_microbatches_in_epoch`` cursor
        so the next step continues from the same logical position in the epoch.
        Invalidates the dataloader iterator via ``_dataloader_invalidated``.

        No-op if any required attribute is missing (trainer was set up without
        the caching pattern — logs a warning instead).
        """
        required = ("_train_lengths", "_train_collate_fn", "_num_workers", "_pin_memory",
                    "train_dataset", "train_sampler")
        for attr in required:
            if not hasattr(self, attr):
                logger.warning(
                    "live_max_batch_tokens_rebuild_skipped",
                    reason=f"missing attribute {attr!r}",
                    trainer=type(self).__name__,
                )
                return

        import numpy as np
        from torch.utils.data import DataLoader, Subset

        from bgkit.data.samplers import PackedTokenBudgetSampler

        # Snapshot the unfiltered originals on first rebuild so we can
        # re-derive the filtered Subset whenever min_sample_length changes.
        # The filter operates on *content* lengths (encoder input size) when
        # available; the sampler's budget still uses `_train_lengths` (which
        # may include decoder/template overhead).
        if not hasattr(self, "_train_dataset_full"):
            self._train_dataset_full = self.train_dataset
            self._train_lengths_full = np.asarray(self._train_lengths)
            content_lengths = getattr(self, "_train_content_lengths", None)
            self._train_content_lengths_full = (
                np.asarray(content_lengths)
                if content_lengths is not None
                else self._train_lengths_full
            )

        min_len = int(getattr(self, "_min_sample_length", 0) or 0)
        max_len = int(getattr(self, "_max_sample_length", 0) or 0)
        if min_len > 0 or max_len > 0:
            content = self._train_content_lengths_full
            mask = np.ones(len(content), dtype=bool)
            if min_len > 0:
                mask &= content >= min_len
            if max_len > 0:
                mask &= content <= max_len
            valid_idx = np.where(mask)[0]
            if len(valid_idx) == 0:
                logger.warning(
                    "live_sample_length_filters_all",
                    min_sample_length=min_len,
                    max_sample_length=max_len,
                    max_length_observed=int(content.max()) if len(content) else 0,
                    min_length_observed=int(content.min()) if len(content) else 0,
                )
                return
            ds = Subset(self._train_dataset_full, valid_idx.tolist())
            lengths = self._train_lengths_full[valid_idx]
        else:
            ds = self._train_dataset_full
            lengths = self._train_lengths_full

        old_budget = getattr(self.train_sampler, "_max_batch_tokens", None)
        cursor = self._microbatches_in_epoch

        seed = getattr(self.train_sampler, "_seed", None)
        epoch = getattr(self.train_sampler, "_epoch", 0)
        cost_multiplier = float(
            getattr(self, "_sampler_cost_multiplier", 1.0) or 1.0
        )
        budget_mode = str(getattr(self, "_sampler_budget_mode", "packed_quadratic"))

        # If the train_dataloader is currently wrapped (e.g. by
        # ``_InterleavingDataLoader`` in DecoderInitTrainer for QA mixing),
        # preserve the wrapper attributes so we can re-wrap after rebuild.
        # Without this, the rebuild silently strips the QA secondary loader
        # and training falls back to 100% primary — a subtle correctness
        # bug, not an obvious crash.
        wrapper_state = None
        existing_loader = self.train_dataloader
        if hasattr(existing_loader, "_secondary") and hasattr(existing_loader, "_ratio"):
            wrapper_state = {
                "type": type(existing_loader),
                "secondary": existing_loader._secondary,
                "ratio": existing_loader._ratio,
            }

        self.train_dataset = ds
        self._train_lengths = lengths
        self.train_sampler = PackedTokenBudgetSampler(
            ds,
            lengths=lengths,
            max_batch_tokens=new_budget,
            shuffle=True,
            seed=seed,
            cost_multiplier=cost_multiplier,
            budget_mode=budget_mode,
        )
        # Restore epoch so shuffle order is deterministic on resume.
        self.train_sampler.set_epoch(epoch)
        # Preserve cursor so the next step continues from the same position.
        self.train_sampler.set_batch_cursor(cursor)

        primary_loader = DataLoader(
            ds,
            batch_sampler=self.train_sampler,
            collate_fn=self._train_collate_fn,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )
        if wrapper_state is not None:
            self.train_dataloader = wrapper_state["type"](
                primary=primary_loader,
                secondary=wrapper_state["secondary"],
                secondary_ratio=wrapper_state["ratio"],
            )
        else:
            self.train_dataloader = primary_loader
        self._dataloader_invalidated = True

        logger.info(
            "live_max_batch_tokens_update",
            old=old_budget,
            new=new_budget,
            cursor_preserved=cursor,
            min_sample_length=min_len,
            max_sample_length=max_len,
            n_samples=len(ds),
            cost_multiplier=cost_multiplier,
            budget_mode=budget_mode,
        )

    def _rebuild_eval_dataloader_with_budget(self, new_budget: int) -> None:
        """Rebuild the eval dataloader with a new token budget.

        Subclasses must have set ``_eval_lengths``, ``_train_collate_fn`` (reused
        for eval), ``_num_workers``, and ``_pin_memory`` in their ``setup()``
        method, and must expose ``eval_dataset`` attribute.

        No-op if any required attribute is missing.
        """
        required = ("_eval_lengths", "_train_collate_fn", "_num_workers", "_pin_memory",
                    "eval_dataset", "_max_batch_tokens_eval")
        for attr in required:
            if not hasattr(self, attr):
                logger.warning(
                    "live_max_batch_tokens_eval_rebuild_skipped",
                    reason=f"missing attribute {attr!r}",
                    trainer=type(self).__name__,
                )
                return

        from torch.utils.data import DataLoader

        from bgkit.data.samplers import PackedTokenBudgetSampler

        old_budget = self._max_batch_tokens_eval
        cost_multiplier = float(
            getattr(
                self,
                "_sampler_eval_cost_multiplier",
                getattr(self, "_sampler_cost_multiplier", 1.0),
            )
            or 1.0
        )
        budget_mode = str(
            getattr(
                self,
                "_sampler_eval_budget_mode",
                getattr(self, "_sampler_budget_mode", "packed_quadratic"),
            )
        )
        eval_sampler = PackedTokenBudgetSampler(
            self.eval_dataset,
            lengths=self._eval_lengths,
            max_batch_tokens=new_budget,
            shuffle=False,
            cost_multiplier=cost_multiplier,
            budget_mode=budget_mode,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_sampler=eval_sampler,
            collate_fn=self._train_collate_fn,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )
        logger.info(
            "live_max_batch_tokens_eval_update",
            old=old_budget,
            new=new_budget,
            cost_multiplier=cost_multiplier,
            budget_mode=budget_mode,
        )

    @staticmethod
    def _resolve_eval_batch_budget(tcfg, max_batch_tokens: int) -> int:
        """Resolve the eval-sampler token budget with a principled default.

        Packed eval has no backward pass, so its CUDA peak at a given
        ``max_batch_tokens`` is roughly half that of training at the same
        budget. When a phase config does not override ``max_batch_tokens_eval``,
        default to ``2 * max_batch_tokens`` — this keeps eval's CUDA peak
        comparable to training's while letting eval pack more samples per
        microbatch (faster eval, better statistics, same memory headroom).

        A phase can still set an explicit ``max_batch_tokens_eval`` in its
        training config to pin a specific value (e.g., a phase with eval on
        a different dataset distribution or at a tighter budget for testing).
        """
        explicit = tcfg.get("max_batch_tokens_eval", None)
        if explicit is not None:
            return int(explicit)
        return 2 * int(max_batch_tokens)

    def _handle_max_grad_norm(self, val) -> None:
        """Live-config handler for ``max_grad_norm``.

        Validates the value and updates ``_max_grad_norm``. Takes effect
        immediately on the next optimizer step via ``clip_grad_norm`` (no
        rebuild needed).
        """
        try:
            new_val = float(val)
        except (TypeError, ValueError):
            logger.warning(
                "live_max_grad_norm_invalid",
                value=val,
                expected="positive float",
            )
            return
        if not (new_val > 0) or new_val != new_val:  # rejects 0, negatives, NaN
            logger.warning(
                "live_max_grad_norm_invalid",
                value=val,
                expected="positive float",
            )
            return
        old = getattr(self, "_max_grad_norm", 1.0)
        if old == new_val:
            return
        self._max_grad_norm = new_val
        logger.info(
            "live_max_grad_norm_update",
            old=float(old),
            new=new_val,
        )

    def _handle_max_batch_tokens(self, val) -> None:
        """Live-config handler for ``max_batch_tokens``.

        Validates the value, updates ``_max_batch_tokens``, and triggers a
        train-dataloader rebuild via :meth:`_rebuild_train_dataloader_with_budget`.
        """
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            logger.warning(
                "live_max_batch_tokens_invalid",
                value=val,
                expected="positive int",
            )
            return
        old = getattr(self, "_max_batch_tokens", None)
        if old == val:
            return
        self._max_batch_tokens = val
        self._rebuild_train_dataloader_with_budget(val)

    def _handle_max_batch_tokens_eval(self, val) -> None:
        """Live-config handler for ``max_batch_tokens_eval``.

        Validates the value, updates ``_max_batch_tokens_eval``, and triggers an
        eval-dataloader rebuild via :meth:`_rebuild_eval_dataloader_with_budget`.
        """
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            logger.warning(
                "live_max_batch_tokens_eval_invalid",
                value=val,
                expected="positive int",
            )
            return
        old = getattr(self, "_max_batch_tokens_eval", None)
        if old == val:
            return
        self._max_batch_tokens_eval = val
        self._rebuild_eval_dataloader_with_budget(val)

    def evaluate_ablation_hook(
        self, sub_batch, enc_out, sub_tokens,
    ) -> tuple[float, float, str]:
        """First-class ablation hook for eval (always-on when conditions match).

        Subclasses override to compute an alternative "supremum" / floor
        loss alongside the normal eval loss - typically by substituting
        an idealized input at some pipeline stage (e.g. perfect projection,
        gold encoder output, etc.). The gap ``eval/loss - eval/ablation_*_loss``
        measures the cost of that pipeline stage.

        Called once per eval sub-batch from the trainer's eval loop with:

        - ``sub_batch``: the sliced packed batch the normal forward used,
          carrying whatever extra companion fields (pair_ids, gold
          embeddings, etc.) the override needs.
        - ``enc_out``: the trainer-specific encoder output from the
          normal forward (so the override can reuse survivor cu_seqlens
          / shapes without re-running the encoder).
        - ``sub_tokens``: the token denominator used for the normal
          loss on this sub-batch — passed in so the override returns
          a token-weighted sum on the same scale.

        Returns ``(ablation_loss_sum, ablation_tokens, suffix)``:

        - If conditions for ablation are not met on this sub-batch
          (e.g. companion data missing, config flag off), return
          ``(0.0, 0.0, "")`` and the caller skips accumulation.
        - Otherwise return the loss x tokens sum, the token count, and
          the metric-name suffix (e.g. ``"perfect_projection"``). The
          eval loop accumulates across sub-batches and the suffix is
          used to emit ``eval/ablation_<suffix>_loss`` and
          ``eval/<suffix>_quality_gap`` once at the end.

        Default no-op returns ``(0.0, 0.0, "")``.
        """
        return 0.0, 0.0, ""

    def _apply_sample_length_filter_to_eval(self) -> None:
        """Apply min/max sample-length filter to ``eval_dataset`` too.

        2026-05-15: previously the sample-length filter applied only to
        the TRAIN dataset; eval saw the unfiltered random_split which
        included samples below ``min_sample_length``. With min=256 the
        train set has 256+ token samples; eval was avg ~44 token
        samples — different distributions, making eval loss look much
        worse than train. Eval must use the SAME filter to give a
        comparable loss measurement.
        """
        import numpy as np
        from torch.utils.data import DataLoader, Subset

        from bgkit.data.samplers import PackedTokenBudgetSampler

        if not hasattr(self, "_eval_lengths"):
            return  # subclass doesn't support eval rebuild

        # Snapshot unfiltered eval state once.
        if not hasattr(self, "_eval_dataset_full"):
            self._eval_dataset_full = self.eval_dataset
            self._eval_lengths_full = np.asarray(self._eval_lengths)
            # Try to get content lengths from the compression_dataset
            # via the subset's indices, same way setup() did.
            try:
                cd = getattr(self, "compression_dataset", None)
                if cd is not None and hasattr(self.eval_dataset, "indices"):
                    self._eval_content_lengths_full = np.array(
                        [cd.content_token_length(i) for i in self.eval_dataset.indices],
                        dtype=np.int64,
                    )
                else:
                    self._eval_content_lengths_full = self._eval_lengths_full
            except Exception:
                self._eval_content_lengths_full = self._eval_lengths_full

        min_len = int(getattr(self, "_min_sample_length", 0) or 0)
        max_len = int(getattr(self, "_max_sample_length", 0) or 0)
        if min_len > 0 or max_len > 0:
            content = self._eval_content_lengths_full
            mask = np.ones(len(content), dtype=bool)
            if min_len > 0:
                mask &= content >= min_len
            if max_len > 0:
                mask &= content <= max_len
            valid_idx = np.where(mask)[0]
            if len(valid_idx) == 0:
                logger.warning(
                    "eval_sample_length_filter_drops_all",
                    min_sample_length=min_len,
                    max_sample_length=max_len,
                )
                return
            ds = Subset(self._eval_dataset_full, valid_idx.tolist())
            lengths = self._eval_lengths_full[valid_idx]
        else:
            ds = self._eval_dataset_full
            lengths = self._eval_lengths_full

        self.eval_dataset = ds
        self._eval_lengths = lengths
        budget = getattr(self, "_max_batch_tokens_eval", None)
        cost_multiplier = float(
            getattr(
                self, "_sampler_eval_cost_multiplier",
                getattr(self, "_sampler_cost_multiplier", 1.0),
            ) or 1.0
        )
        budget_mode = str(
            getattr(
                self,
                "_sampler_eval_budget_mode",
                getattr(self, "_sampler_budget_mode", "packed_quadratic"),
            )
        )
        if budget is None:
            return
        eval_sampler = PackedTokenBudgetSampler(
            self.eval_dataset,
            lengths=self._eval_lengths,
            max_batch_tokens=int(budget),
            shuffle=False,
            cost_multiplier=cost_multiplier,
            budget_mode=budget_mode,
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_sampler=eval_sampler,
            collate_fn=self._train_collate_fn,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )
        logger.info(
            "eval_sample_length_filter_applied",
            min_sample_length=min_len,
            max_sample_length=max_len,
            kept=len(self._eval_lengths),
            dropped=int(
                len(self._eval_lengths_full) - len(self._eval_lengths),
            ),
        )

    def _handle_min_sample_length(self, val) -> None:
        """Live-config handler for ``min_sample_length``.

        Filters training samples shorter than ``val`` tokens by wrapping
        the dataset in a ``Subset`` and rebuilding sampler + dataloader
        via :meth:`_rebuild_train_dataloader_with_budget`. ``val=0``
        disables the filter (returns to the unfiltered dataset).
        Also re-applies the filter to ``eval_dataset`` so train and
        eval see the same length distribution.
        """
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            logger.warning(
                "live_min_sample_length_invalid",
                value=val,
                expected="non-negative int",
            )
            return
        old = int(getattr(self, "_min_sample_length", 0) or 0)
        if old == val:
            return
        self._min_sample_length = val
        budget = getattr(self, "_max_batch_tokens", None)
        if budget is None:
            logger.warning(
                "live_min_sample_length_skipped",
                reason="trainer has no _max_batch_tokens (rebuild path unavailable)",
            )
            return
        self._rebuild_train_dataloader_with_budget(int(budget))
        self._apply_sample_length_filter_to_eval()
        logger.info("live_min_sample_length_update", old=old, new=val)

    def _handle_max_sample_length(self, val) -> None:
        """Live-config handler for ``max_sample_length`` — guard against
        OOM-inducing pathologically long samples.

        Symmetric to :meth:`_handle_min_sample_length`. Filters training
        samples LONGER than ``val`` content tokens by adding an upper
        bound to the same Subset-based filter. ``val=0`` disables the
        upper bound (returns to "no max"). Useful when expanding the
        corpus to looser file types brings in huge data dumps that the
        decoder can't fit at high target_ratio.
        """
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            logger.warning(
                "live_max_sample_length_invalid",
                value=val,
                expected="non-negative int",
            )
            return
        old = int(getattr(self, "_max_sample_length", 0) or 0)
        if old == val:
            return
        self._max_sample_length = val
        budget = getattr(self, "_max_batch_tokens", None)
        if budget is None:
            logger.warning(
                "live_max_sample_length_skipped",
                reason="trainer has no _max_batch_tokens (rebuild path unavailable)",
            )
            return
        self._rebuild_train_dataloader_with_budget(int(budget))
        self._apply_sample_length_filter_to_eval()
        logger.info("live_max_sample_length_update", old=old, new=val)

    # ------------------------------------------------------------------
    # Memory accounting: everything comes from ``compute.memory`` — a
    # hardware-level concern, not a per-phase one. Backcompat: old
    # ``training.memory_*`` keys are still honored with a one-shot warn
    # so in-flight resume configs don't break.
    # ------------------------------------------------------------------

    _memory_legacy_warned = False

    def _memory_cfg(self) -> dict:
        """Return the resolved ``compute.memory`` config block.

        Pulls from ``cfg.compute.memory`` (the canonical location — a
        single block inherited by every training config on this box).
        Reads legacy ``training.memory_*`` scalars as a fallback so
        existing resume configs keep working; warns once when any are
        found so the operator can migrate them.
        """
        compute = self.cfg.get("compute", None)
        mem = compute.get("memory", None) if compute else None
        out = dict(mem) if mem else {}

        tcfg = self.cfg.training
        legacy: dict = {}
        if "memory_log_every" in tcfg:
            legacy["log_every_steps"] = int(tcfg.memory_log_every)
        if "memory_abort_system_used_gb" in tcfg:
            legacy["system_abort_gb"] = float(
                tcfg.memory_abort_system_used_gb,
            )
        if "memory_budget" in tcfg:
            mb = tcfg.memory_budget
            budgets = dict(out.get("scope_budgets", {}) or {})
            for scope in ("evaluate", "gen_eval", "save_checkpoint"):
                k = f"{scope}_cap_gb"
                if k in mb and mb.get(k) is not None:
                    budgets[f"{scope}_gb"] = float(mb.get(k))
            if budgets:
                legacy["scope_budgets"] = budgets

        if legacy and not BaseTrainer._memory_legacy_warned:
            logger.warning(
                "memory_cfg_legacy_training_keys",
                keys=sorted(legacy.keys()),
                hint="move to configs/compute/<host>.yaml under memory: "
                "— training.memory_* is a backcompat shim, not the "
                "canonical location",
            )
            BaseTrainer._memory_legacy_warned = True
        for k, v in legacy.items():
            out.setdefault(k, v)

        return out

    def _release_training_transients(self) -> None:
        """Drop training-side tensors before entering a memory-budgeted scope.

        Called immediately before every ``evaluate`` / ``gen_eval`` /
        ``save_checkpoint`` scope entry in :meth:`train`.  Frees:

        * Optimizer gradients — one tensor per trainable param, live
          between ``optimizer.step()`` and the next step's
          ``zero_grad()``.  Under full-model training these sum to
          several GB of otherwise-idle memory during a 5-minute eval.
        * Any training batch held on device by
          :class:`_DevicePrefetcher` (``_next_batch``).  The prefetcher
          stages the next batch at the end of each ``__next__``; if
          eval fires right after a micro-batch the staged tensor sits
          through the entire eval.  Re-prefetch costs one extra
          host→device transfer, cheap compared to the hold.

        The scope's own ``gc.collect`` + ``empty_cache`` then reclaim
        the freed blocks back to the OS.

        Subclasses should override and call ``super()`` to add their
        own phase-boundary transients (e.g. survivorship state, cached
        ICE teacher).  No-op on trainers that haven't reached
        :meth:`setup` yet.
        """
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        it = getattr(self, "_active_dataloader_iter", None)
        if it is not None and hasattr(it, "_next_batch"):
            it._next_batch = None

    def _scope_cap(self, scope: str) -> float | None:
        """Read a :func:`memory_budget_scope` cap for ``scope``.

        Looks up ``compute.memory.scope_budgets.<scope>_gb``.  ``None``
        (or absent) disables enforcement — the scope still logs peak
        but does not raise.  Subsystems that need a fixed cap regardless
        of config can override this method.
        """
        budgets = self._memory_cfg().get("scope_budgets", None)
        if not budgets:
            return None
        val = budgets.get(f"{scope}_gb", None)
        if val is None:
            return None
        return float(val)

    def _registry_parent(self) -> str | None:
        """Return normalized parent checkpoint name, or None."""
        if self._last_checkpoint_path:
            return normalize_checkpoint_name(self._last_checkpoint_path)
        return None

    def _build_registry_entry(
        self,
        ckpt_path: Path,
        metrics: dict[str, float] | None,
        wandb_run,
        status: str = "completed",
        parent_checkpoint: str | None = None,
    ) -> RegistryEntry:
        """Build a RegistryEntry for a saved checkpoint.

        Args:
            parent_checkpoint: Dir name of the previous checkpoint. Must be
                captured *before* ``save_checkpoint()`` which mutates
                ``self._last_checkpoint_path``.
        """
        config_snapshot = None
        with contextlib.suppress(Exception):
            config_snapshot = OmegaConf.to_container(self.cfg.training, resolve=True)

        disk_size = None
        if ckpt_path.exists():
            disk_size = sum(f.stat().st_size for f in ckpt_path.rglob("*") if f.is_file())

        return RegistryEntry(
            name=ckpt_path.name,
            phase=self.cfg.training.phase,
            step=self.global_step,
            epoch=self.epoch,
            timestamp=datetime.now(UTC).isoformat(),
            status=status,
            on_disk=True,
            metrics=metrics,
            config_snapshot=config_snapshot,
            wandb_run_id=wandb_run.id if wandb_run is not None else None,
            disk_size_bytes=disk_size,
            parent_checkpoint=parent_checkpoint,
            input_sources=self._input_sources,
            run_name=self.cfg.get("run_name", None),
        )

    @staticmethod
    def _normalize_eval_metrics(metrics: dict[str, float]) -> dict[str, float]:
        """Return eval metrics with exactly one ``eval/`` prefix."""

        return {
            key if str(key).startswith("eval/") else f"eval/{key}": value
            for key, value in metrics.items()
        }

    def train(self) -> None:
        """Main training loop.

        Expects scalar ``training.max_steps``, ``training.lr``, and
        ``training.warmup_steps``.  Phase configs with multi-step or
        per-component LR schedules (phase1_step4, phase2) must override
        this method with their own loop.

        Supports early stopping via ``training.early_stopping`` config:
        - ``enabled`` (bool, default False): set True to enable
        - ``patience`` (int, default 5): evals without improvement before stopping
        - ``min_delta`` (float, default 0.001): minimum improvement to reset patience
        - ``metric`` (str, default "eval/loss"): eval metric to track (must be present
          in evaluate() results; lower is better)
        """
        # Arm SIGTERM/SIGINT handling BEFORE setup. setup() + resume can spend
        # minutes loading the Phase-1 checkpoint, verifying caches, and
        # restoring ~12 GB of Muon optimizer state; a `docker stop` in that
        # window used to hit python's default disposition and SIGKILL mid-setup
        # (exit 137) with no clean exit. The SAME interruptor is reused by the
        # training loop below (which no longer installs its own).
        self._graceful_shutdown = False
        interruptor = GracefulInterruptor()
        interruptor.install()
        self.setup()
        # NOTE: the loud "STARTUP CHECKPOINT SUMMARY" banner is emitted below,
        # AFTER the resume-checkpoint load — not here. Emitting it right after
        # setup() (its historical position) meant it ran BEFORE resume applied,
        # so a resume run always printed "FRESH / RANDOM INIT" for
        # encoder/decoder/optimizer even though the resume checkpoint was about
        # to overwrite them. That structural lie made a perfectly good resume
        # look like a cold start (2026-07-03 phase2_kb git-repro incident).

        tcfg = self.cfg.training
        if (
            not hasattr(tcfg, "max_steps")
            or not hasattr(tcfg, "lr")
            or not isinstance(tcfg.lr, (int, float))
        ):
            phase = getattr(tcfg, "phase", "<unknown>")
            raise TypeError(
                f"BaseTrainer.train() requires scalar training.max_steps and "
                f"training.lr, but phase '{phase}' uses a different schema. "
                f"Override train() in the phase-specific trainer."
            )
        eval_every = tcfg.eval_every
        save_every = tcfg.save_every
        empty_cache_cadence = _coerce_empty_cache_cadence(
            tcfg.get("cuda_empty_cache_every_step", None),
            (self.cfg.get("compute", {}) or {}).get(
                "cuda_empty_cache_every_step", False,
            ),
        )
        checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))

        # Optional NVMe fast-write + async HDD archive. When configured, save
        # writes the checkpoint to the fast (NVMe) dir — ~15 s fsync, no
        # spinning-disk dirty-page backlog / post-save spike — then a background
        # daemon copies it to the HDD archive (throttled fsync) so the full
        # checkpoint history is preserved. Resume prefers the NVMe copy; it is a
        # host bind-mount so it survives container restarts. keep_last_n caps the
        # NVMe footprint; the HDD keeps everything.
        fast_ckpt = self.cfg.get("fast_checkpoint_dir", None) or os.environ.get(
            "BGKIT_FAST_CHECKPOINT_DIR"
        )
        self._fast_checkpoint_dir = None
        self._archiver = None
        phase_for_ckpt = getattr(tcfg, "phase", None) or self.cfg.training.phase
        if fast_ckpt:
            from bgkit.training.checkpoint_archiver import (
                CheckpointArchiver,
                archive_pending_into,
            )

            self._fast_checkpoint_dir = Path(fast_ckpt)
            self._fast_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            # Recover any NVMe-only checkpoints from a prior crash onto the HDD.
            archive_pending_into(
                checkpoint_dir,
                self._fast_checkpoint_dir,
                phase_for_ckpt,
                run_name=self.cfg.get("run_name", None),
            )
            self._archiver = CheckpointArchiver(
                archive_dir=checkpoint_dir,
                phase=phase_for_ckpt,
                run_name=self.cfg.get("run_name", None),
                keep_last_n=int(self.cfg.get("fast_checkpoint_keep_last_n", 3)),
                archive_keep_last_n=self.cfg.get("archive_keep_last_n", None),
            )
            logger.info(
                "checkpoint_fast_dir_enabled",
                fast_dir=str(self._fast_checkpoint_dir),
                archive_dir=str(checkpoint_dir),
            )

        # Checkpoint registry
        registry = CheckpointRegistry(checkpoint_dir)

        # Early stopping config (disabled by default; enable per-phase in YAML)
        es_cfg = tcfg.get("early_stopping", {})
        if isinstance(es_cfg, bool):
            es_enabled = es_cfg
            es_cfg = {}
        else:
            es_enabled = es_cfg.get("enabled", False) if es_cfg else False
        es_patience = es_cfg.get("patience", 5) if es_cfg else 5
        es_min_delta = es_cfg.get("min_delta", 0.001) if es_cfg else 0.001
        es_metric = es_cfg.get("metric", "eval/loss") if es_cfg else "eval/loss"
        es_best: float | None = None
        es_evals_without_improvement = 0

        # Resume from checkpoint: explicit path, "none" to disable, or auto-resolve
        resume_path = self.cfg.get("resume_checkpoint", None)
        # Auto-resume is RUN-SCOPED: it exists to resume the SAME run after an
        # interruption. Phase alone is NOT a safe scope — many runs share a
        # phase (all Phase 2 KB runs use phase="phase2_kb"), so a phase-only
        # resolve could grab a DIFFERENT run's checkpoint and crash on the
        # strict state-dict load (e.g. a checkpoint predating a new module).
        # Passing run_name restricts resolution to this run's own checkpoints;
        # if none exist (or they predate the run_name field), we cold-start.
        # NOTE: cross-phase handoffs (Step1→Step2 etc.) use SEPARATE explicit
        # checkpoint configs (bgkit_checkpoint, step1_checkpoint) — NOT this
        # auto-resume path — and are unaffected.
        auto_resolved_resume = False
        if resume_path == "none":
            resume_path = None  # explicitly disabled
        elif resume_path is None:
            phase = getattr(tcfg, "phase", None)
            if phase:
                auto_resolved = resolve_latest_checkpoint(
                    checkpoint_dir,
                    phase,
                    fast_dir=self._fast_checkpoint_dir,
                    run_name=self.cfg.get("run_name", None),
                )
                if auto_resolved is not None:
                    resume_path = str(auto_resolved)
                    auto_resolved_resume = True
                    logger.info(
                        "auto_resume_resolved",
                        checkpoint=resume_path,
                        phase=phase,
                        run_name=self.cfg.get("run_name", None),
                    )
        is_resuming = False
        resume_step: int | None = None
        if resume_path is not None and auto_resolved_resume:
            # Belt-and-suspenders: an auto-resolved checkpoint can still be
            # incompatible (e.g. a same-run checkpoint saved before a model
            # change). A strict state-dict load would crash; instead log and
            # cold-start. Explicit resume paths still raise below — the user
            # asked for that specific checkpoint, so a mismatch is a real error.
            try:
                self.load_checkpoint(Path(resume_path))
            except RuntimeError as exc:
                logger.warning(
                    "auto_resume_load_failed_cold_start",
                    checkpoint=resume_path,
                    error=str(exc),
                    hint="auto-resolved checkpoint is incompatible with the "
                    "current model (missing/unexpected keys); cold-starting "
                    "this run instead of crashing",
                )
                resume_path = None
                self.global_step = 0
                self.epoch = 0
                self._schedule_params = None
                self._training_state = None
                self._last_checkpoint_path = None
        elif resume_path is not None:
            # EXPLICIT resume path (user pointed resume_checkpoint at a specific
            # checkpoint). This is the resume-the-same-run path. Two guarantees:
            #
            # * OBSERVABLE: log BEFORE the load. load_checkpoint only logs
            #   ``checkpoint_loaded`` on COMPLETION, and a ~10 GB model +
            #   optimizer read from the spinning HDD can take many minutes; the
            #   silent gap used to be indistinguishable from "resume skipped"
            #   and let an operator kill a run mid-load thinking it cold-started.
            # * LOUD ON FAILURE: an explicit resume that fails to load is a real
            #   error — the user asked for THIS checkpoint. Never swallow it into
            #   a silent cold start (that is reserved for AUTO-resolved resumes
            #   above, which are best-effort). Re-raise with a greppable log.
            logger.info("resume_checkpoint_loading", path=resume_path)
            try:
                self.load_checkpoint(Path(resume_path))
            except Exception as exc:
                logger.error(
                    "resume_checkpoint_load_failed",
                    checkpoint=resume_path,
                    error=str(exc),
                    hint="explicit resume_checkpoint failed to load — refusing "
                    "to silently cold-start. Fix the path/checkpoint or set "
                    "resume_checkpoint=none to intentionally start fresh.",
                )
                raise
        if resume_path is not None:
            # Checkpoint was saved after step completed, so resume from next step
            self.global_step += 1
            is_resuming = True
            resume_step = self.global_step
            # The resume checkpoint supersedes whatever setup() loaded as the
            # base (e.g. a Phase-1 checkpoint) for every component it carries.
            # Reflect that in the startup banner (emitted just below) so the
            # summary is truthful instead of showing setup()'s base — or, for
            # trainers that register no sources at all, "FRESH / RANDOM INIT".
            self.register_checkpoint_source("encoder", resume_path)
            self.register_checkpoint_source("decoder", resume_path)
            self.register_checkpoint_source("optimizer_state", resume_path)
            self.register_startup_note(
                f"RESUMED from {resume_path} at step {self.global_step} "
                "(supersedes any setup() base for the components above)",
            )
            # Restore early stopping state
            if self._training_state is not None:
                es_best = self._training_state.get("es_best")
                es_evals_without_improvement = self._training_state.get(
                    "es_evals_without_improvement", 0
                )
            logger.info(
                "resuming_training",
                from_step=self.global_step,
                es_best=es_best,
                es_evals_without_improvement=es_evals_without_improvement,
            )

        # Loud, hard-to-miss checkpoint provenance summary so the operator can
        # confirm what ACTUALLY loaded — emitted here, after resume, so a resume
        # run reports its resume source instead of setup()'s pre-resume base.
        # Catches the "wrong config key, silent pristine encoder" class of bugs
        # AND the "resume silently didn't apply" class.
        self._log_startup_banner()

        # LR schedule params: use restored values from checkpoint if available,
        # otherwise use current config. This ensures schedule continuity on resume
        # even if the config file changed between runs.
        # Set reset_schedule: true to force using config values (e.g. when
        # switching training modes and wanting a fresh schedule).
        reset_schedule = tcfg.get("reset_schedule", False)
        if self._schedule_params is not None and not reset_schedule:
            max_steps = int(self._schedule_params["max_steps"])
            base_lr = self._schedule_params["base_lr"]
            warmup_steps = int(self._schedule_params["warmup_steps"])
            logger.info(
                "schedule_restored_from_checkpoint",
                max_steps=max_steps,
                base_lr=base_lr,
                warmup_steps=warmup_steps,
            )
            # Allow config to extend max_steps beyond the original schedule
            if tcfg.max_steps > max_steps:
                max_steps = tcfg.max_steps
                logger.info("max_steps_extended", max_steps=max_steps)
            # Restore per-param-group base_lrs if saved (preserves live `lr`
            # ratio scaling across restarts — e.g. encoder vs decoder LoRA
            # rates that diverge from the global base_lr).
            saved_per_group = self._schedule_params.get("per_group_base_lrs")
            if saved_per_group is not None:
                opt = getattr(self, "optimizer", None)
                if opt is None:
                    logger.warning(
                        "schedule_per_group_base_lr_skip",
                        reason="optimizer_not_yet_built",
                    )
                elif len(saved_per_group) != len(opt.param_groups):
                    logger.warning(
                        "schedule_per_group_base_lr_count_mismatch",
                        saved=len(saved_per_group),
                        current=len(opt.param_groups),
                        action="ignored — likely yaml param-group structure changed",
                    )
                else:
                    for pg, lr in zip(opt.param_groups, saved_per_group, strict=True):
                        pg["base_lr"] = float(lr)
                    logger.info(
                        "schedule_per_group_base_lr_restored",
                        count=len(saved_per_group),
                        values=[round(v, 6) for v in saved_per_group],
                    )
        else:
            max_steps = tcfg.max_steps
            base_lr = tcfg.lr
            warmup_steps = tcfg.warmup_steps
            if reset_schedule and self._schedule_params is not None:
                logger.info(
                    "schedule_reset_from_config",
                    max_steps=max_steps,
                    base_lr=base_lr,
                    warmup_steps=warmup_steps,
                )

        # Store schedule params for checkpointing. per_group_base_lrs is
        # populated in save_checkpoint() right before serialization so we
        # capture the latest live-scaled values without needing to track
        # them every step.
        self._schedule_params = {
            "max_steps": max_steps,
            "base_lr": base_lr,
            "warmup_steps": warmup_steps,
        }

        # Optional wandb init (resume previous run if checkpoint had a wandb run ID)
        wandb_run = None
        wandb_run_id = (
            self._training_state.get("wandb_run_id") if self._training_state else None
        )
        if self.cfg.get("wandb", {}).get("enabled", False):
            try:
                import wandb

                wandb_kwargs = dict(
                    project=self.cfg.wandb.get("project", "bgkit"),
                    entity=self.cfg.wandb.get("entity", None),
                    name=self.cfg.get("run_name", None),
                    tags=list(self.cfg.wandb.get("tags", [])),
                    config=OmegaConf.to_container(self.cfg, resolve=True),
                )
                reset_wandb = tcfg.get("reset_wandb", False)
                if is_resuming and wandb_run_id is not None and not reset_wandb:
                    wandb_kwargs["id"] = wandb_run_id
                    wandb_kwargs["resume"] = "allow"
                    logger.info("wandb_resuming_run", run_id=wandb_run_id)
                elif reset_wandb:
                    logger.info("wandb_fresh_run", old_run_id=wandb_run_id)
                wandb_run = wandb.init(**wandb_kwargs)
            except ImportError:
                logger.warning("wandb_not_installed")

        # Sync sampler + dataset epoch before first iter (needed after resume)
        self._sync_epoch(self.epoch)

        # Subclass hook for resume-time setup (e.g. L1 dataloader rebuild)
        self._pre_train_loop()

        # Memory-driven dynamic ckpt scheduler (default-on host-wide via
        # compute config). Initializes after _pre_train_loop so subclass
        # registration of managed models — done in setup() or _pre_train_loop —
        # is visible. Cheap no-op when no models registered.
        self._init_dynamic_ckpt_scheduler()

        # Restore the dataloader cursor on resume so the model sees
        # samples from roughly where it was trained, not epoch 0 batch 0.
        # Without this, a length-sorted sampler (or any sampler where
        # batch i is correlated with training dynamics) re-enters a
        # stale sub-distribution on every resume. See
        # ``self._microbatches_in_epoch`` docstring for the diagnosis.
        #
        # Two paths:
        #
        # * Sampler supports ``set_batch_cursor`` — the sampler builds
        #   its batch list and starts iteration at the stored logical
        #   cursor.  No replay, no wasted CPU.  This is the fast path.
        # * Sampler lacks cursor support — fall back to replay: pull and
        #   discard microbatches one by one on CPU (no prefetch, no
        #   host→device copies).  Kept for legacy samplers and for
        #   ``_InterleavingDataLoader`` until it too exposes cursor
        #   state.
        #
        # The logical cursor is the trainer's
        # ``_microbatches_in_epoch``, **not** an iterator-internal
        # position: ``_DevicePrefetcher`` stages one batch ahead, so a
        # raw iterator cursor would be off by one.
        dataloader_iter = None
        cursor_restored = False
        if is_resuming and self._microbatches_in_epoch > 0:
            batch_sampler = getattr(self.train_dataloader, "batch_sampler", None)
            if batch_sampler is not None and hasattr(batch_sampler, "set_batch_cursor"):
                batch_sampler.set_batch_cursor(self._microbatches_in_epoch)
                logger.info(
                    "dataloader_resume_cursor_restored",
                    batch_cursor=self._microbatches_in_epoch,
                    epoch=self.epoch,
                )
                cursor_restored = True

        if is_resuming and self._microbatches_in_epoch > 0 and not cursor_restored:
            skip_target = int(self._microbatches_in_epoch)
            logger.info(
                "dataloader_resume_skip_start",
                microbatches_to_skip=skip_target,
                epoch=self.epoch,
                reason="batch_sampler has no set_batch_cursor",
            )
            skipped = 0
            try:
                raw_dataloader_iter = self._create_dataloader_iter(use_prefetch=False)
                for _ in range(skip_target):
                    next(raw_dataloader_iter)
                    skipped += 1
                    if skipped % 250 == 0 or skipped == skip_target:
                        logger.info(
                            "dataloader_resume_skip_progress",
                            skipped=skipped,
                            target=skip_target,
                            epoch=self.epoch,
                        )
                dataloader_iter = self._wrap_dataloader_iter(raw_dataloader_iter)
            except StopIteration:
                # Edge case: epoch ended exactly at save; roll over.
                self.epoch += 1
                self._microbatches_in_epoch = 0
                self._sync_epoch(self.epoch)
                dataloader_iter = self._create_dataloader_iter()
            logger.info(
                "dataloader_resume_skip_done",
                skipped=skipped,
                target=skip_target,
                epoch=self.epoch,
            )

        if dataloader_iter is None:
            dataloader_iter = self._create_dataloader_iter()

        accum_steps = self._validate_accum_steps(
            tcfg.get("gradient_accumulation_steps", 1)
        )
        self._accum_steps = accum_steps
        # Per-phase override for gradient clipping. Default 1.0 preserves
        # historical behavior. Phases that rely on large-magnitude
        # auxiliary losses (e.g. symmetric forced-survivor BCE in
        # phase1_falcon_l0) need a higher cap so the auxiliary gradient
        # direction survives clipping.
        self._max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))

        # Resume warmup: optional linear ramp from near-zero to scheduled LR
        # over N steps after a resume. DEFAULT 0 (disabled) — the by-name
        # optimizer state restore already preserves momentum for every param
        # that received a gradient pre-checkpoint, and curriculum-aware
        # trainers (e.g. CommitEncodingTrainer's stage 0→1 dual-loader
        # schedule) handle their own warmup for newly-gradient-receiving
        # params. The blanket LR ramp doesn't distinguish "params with stale
        # momentum" from "params that just got their state restored", so
        # leaving it on punishes both groups equally for no protective
        # benefit on the second.
        #
        # Set ``training.resume_warmup_steps`` > 0 explicitly when:
        #   - a topology change introduces new param groups that will receive
        #     gradients immediately (e.g. mid-training LoRA adapter swap)
        #   - the LR schedule changed substantially (extended max_steps,
        #     base_lr bumped) and you want to avoid a one-step LR jump
        # Otherwise leave at 0 — see plans/l0-l1-rebuild-blockers.md
        # post-reboot resume notes for the diagnosis.
        resume_warmup_steps = int(tcfg.get("resume_warmup_steps", 0))
        if resume_warmup_steps > 0 and resume_step is not None:
            resume_warmup_end = resume_step + resume_warmup_steps
            logger.info(
                "resume_warmup_enabled",
                resume_step=resume_step,
                warmup_steps=resume_warmup_steps,
                warmup_end=resume_warmup_end,
            )
        else:
            resume_warmup_end = None

        logger.info(
            "training_start",
            max_steps=max_steps,
            lr=base_lr,
            start_step=self.global_step,
            early_stopping=es_enabled,
            gradient_accumulation_steps=accum_steps,
        )

        # Apply YAML-default min/max_sample_length once at training-loop
        # start. The handlers no-op gracefully if the trainer lacks the
        # rebuild infrastructure (e.g. KRKBTrainer with
        # QueryAwareBatchSampler).
        initial_min_len = int(tcfg.get("min_sample_length", 0) or 0)
        if initial_min_len > 0 and hasattr(self, "_max_batch_tokens"):
            self._handle_min_sample_length(initial_min_len)
        initial_max_len = int(tcfg.get("max_sample_length", 0) or 0)
        if initial_max_len > 0 and hasattr(self, "_max_batch_tokens"):
            self._handle_max_sample_length(initial_max_len)

        # Live config (file-based HP control)
        # Clear stale control file so ad-hoc changes don't carry across runs.
        control_file = self.cfg.get("control_file", None)
        if control_file is None:
            control_file = checkpoint_dir / "control.json"
        control_path = Path(control_file)
        phase = getattr(tcfg, "phase", None)
        live_config = LiveConfig(control_path, namespace=phase)

        # Checkpoint pruning
        prune_cfg = tcfg.get("checkpoint_pruning", {})
        prune_enabled = prune_cfg.get("enabled", False) if prune_cfg else False
        if prune_enabled:
            ckpt_manager = CheckpointManager(
                keep_best=prune_cfg.get("keep_best", 3),
                keep_latest=prune_cfg.get("keep_latest", 2),
                metric=prune_cfg.get("metric", es_metric),
                lower_is_better=prune_cfg.get("lower_is_better", True),
                phase=tcfg.phase,
                run_name=self.cfg.get("run_name", None),
                registry=registry,
            )
            ckpt_manager.load_existing(checkpoint_dir)
        else:
            ckpt_manager = None

        last_eval_metrics: dict[str, float] | None = None
        last_eval_step = -1

        # A `docker stop` arrived during setup/resume: nothing new was trained
        # (the last checkpoint is already on disk), so exit cleanly instead of
        # letting the SIGKILL land. This is why the interruptor is armed before
        # setup() above.
        if interruptor.should_stop:
            logger.warning(
                "graceful_shutdown_during_setup",
                signal=interruptor.received_signal.name
                if interruptor.received_signal
                else None,
            )
            interruptor.restore()
            if wandb_run is not None:
                wandb_run.finish()
            return

        stopped_early = False
        try:
            # Reuse the interruptor armed before setup() — do NOT install a new
            # one (which would reset the captured signal). nullcontext keeps the
            # loop body's indentation; restore() happens in the finally below.
            with contextlib.nullcontext():
                step = self.global_step
                while step < max_steps:
                    self.global_step = step

                    # Subclass hook (dataloader rebuild, curriculum, etc.)
                    self._pre_step_hook()
                    if self._dataloader_invalidated:
                        dataloader_iter = self._create_dataloader_iter()
                        self._dataloader_invalidated = False

                    # LR schedule
                    for pg in self.optimizer.param_groups:
                        group_base = pg.get("base_lr", base_lr)
                        lr = cosine_with_warmup(
                            step, max_steps, warmup_steps, group_base
                        )
                        # Apply resume warmup ramp (linear 0→1 multiplier)
                        if (
                            resume_warmup_end is not None
                            and step < resume_warmup_end
                        ):
                            ramp = (step - resume_step + 1) / resume_warmup_steps
                            lr *= ramp
                        pg["lr"] = lr

                    self._post_lr_schedule(step)

                    # Accumulation loop
                    self.optimizer.zero_grad()
                    accum_metrics = []
                    for _micro in range(accum_steps):
                        # Detect a `docker stop` BEFORE launching the next
                        # microbatch. A full-backprop git-repro step accumulates
                        # many microbatches and a single one can be seconds; the
                        # end-of-step check alone can miss a short grace window
                        # (e.g. `docker stop -t 85`). Bail here, skip the
                        # optimizer step (partial grads), and rescue-save now.
                        if interruptor.should_stop:
                            self._graceful_shutdown_save(
                                checkpoint_dir=checkpoint_dir,
                                registry=registry,
                                ckpt_manager=ckpt_manager,
                                wandb_run=wandb_run,
                                es_best=es_best,
                                es_evals_without_improvement=(
                                    es_evals_without_improvement
                                ),
                                step=step,
                                interruptor=interruptor,
                                already_saved=False,
                            )
                            return
                        try:
                            batch = next(dataloader_iter)
                            self._microbatches_in_epoch += 1
                        except StopIteration:
                            self.epoch += 1
                            self._microbatches_in_epoch = 0
                            self._sync_epoch(self.epoch)
                            dataloader_iter = self._create_dataloader_iter()
                            batch = next(dataloader_iter)
                            self._microbatches_in_epoch += 1
                        micro_metrics = self._forward_backward(batch)
                        accum_metrics.append(micro_metrics)

                    grad_norm = clip_grad_norm(
                        self.trainable_parameters(),
                        max_norm=self._max_grad_norm,
                    )

                    if not math.isfinite(grad_norm):
                        raise RuntimeError(
                            f"NaN/Inf grad_norm at step {step} (grad_norm={grad_norm}). "
                            "This usually indicates a numerical stability issue."
                        )
                    self.optimizer.step()
                    self._post_optimizer_step(step)
                    # Memory-driven dynamic ckpt scheduler — flips
                    # gradient-checkpointing mode based on observed
                    # system_used_gb, not curriculum ratio. No-op when
                    # the trainer hasn't registered managed models.
                    self._dynamic_ckpt_step(step)
                    # Heartbeat for the step-level deadlock watchdog. Imported
                    # at use to avoid a hot-path import (cached after first call).
                    from bgkit.utils.step_watchdog import heartbeat as _hb
                    _hb()

                    # Legacy fallback: unconditional cadence flush. The
                    # adaptive flush in ``_dynamic_ckpt_step`` is the
                    # primary mechanism (default-on); this branch only
                    # fires when the user explicitly sets
                    # ``cuda_empty_cache_every_step`` to a positive value
                    # AND wants the cadence behavior regardless of
                    # measured memory pressure.
                    if empty_cache_cadence > 0 and (self.global_step % empty_cache_cadence == 0):
                        import torch as _t
                        if _t.cuda.is_available():
                            _t.cuda.empty_cache()

                    metrics = _average_metrics(accum_metrics)
                    metrics["grad_norm"] = grad_norm
                    metrics["lr"] = self.optimizer.param_groups[0]["lr"]
                    if len(self.optimizer.param_groups) > 1:
                        metrics["lr_min"] = min(
                            pg["lr"] for pg in self.optimizer.param_groups
                        )
                    self._add_step_metrics(metrics)

                    # Memory diagnostics — cheap O(1) allocator queries +
                    # one /proc read. Logged alongside train_step every
                    # ``compute.memory.log_every_steps`` so a slow leak or
                    # allocator-fragmentation curve is visible in wandb
                    # long before the unified-memory pool thrashes.
                    mem_cfg = self._memory_cfg()
                    mem_log_every = int(mem_cfg.get("log_every_steps", 50))
                    if mem_log_every > 0 and step % mem_log_every == 0:
                        mem_diag = _collect_memory_diagnostics()
                        metrics.update(mem_diag)

                        # Safety rail — save a final checkpoint and hard-
                        # exit if the unified pool gets dangerously full.
                        # Complements memory_budget_scope (which bounds
                        # named scopes); this rail is the safety net for
                        # slow leaks in the training hot-loop that don't
                        # live inside any scope. 0 disables.
                        abort_threshold = float(
                            mem_cfg.get("system_abort_gb", 110.0),
                        )
                        sys_used = mem_diag.get("mem/system_used_gb", 0.0)
                        if abort_threshold > 0 and sys_used >= abort_threshold:
                            logger.error(
                                "memory_abort_triggered",
                                system_used_gb=sys_used,
                                threshold_gb=abort_threshold,
                                step=step,
                                hint="training halted before Linux OOM "
                                "could wedge the host; investigate the "
                                "mem/* wandb curves",
                            )
                            # Emergency save — best effort, no metrics
                            # evaluation (may itself allocate).
                            try:
                                self._training_state = self._build_training_state(
                                    es_best, es_evals_without_improvement,
                                    wandb_run,
                                )
                                self.save_checkpoint(checkpoint_dir, metrics=None)
                            except Exception as save_exc:
                                logger.warning(
                                    "memory_abort_save_failed",
                                    error=str(save_exc)[:200],
                                )
                            raise SystemExit(
                                f"Memory abort: system_used={sys_used:.1f} "
                                f">= {abort_threshold:.1f} GB",
                            )

                    # Log
                    if step % self._log_every == 0:
                        logger.info("train_step", step=step, **metrics)
                    if wandb_run is not None:
                        wandb_run.log(metrics, step=step)

                    # Eval
                    if eval_every > 0 and step > 0 and step % eval_every == 0:
                        self._release_training_transients()
                        from bgkit.utils.step_watchdog import pause as _wd_pause
                        from bgkit.utils.step_watchdog import resume as _wd_resume

                        with memory_budget_scope(
                            "evaluate", cap_gb=self._scope_cap("evaluate"),
                        ):
                            _wd_pause()
                            try:
                                eval_metrics = self.evaluate()
                            finally:
                                _wd_resume()
                        eval_metrics = self._normalize_eval_metrics(eval_metrics)
                        logger.info("eval", step=step, **eval_metrics)
                        if wandb_run is not None:
                            wandb_run.log(eval_metrics, step=step)

                        last_eval_metrics = eval_metrics
                        last_eval_step = step

                        # Early stopping check
                        if es_enabled:
                            if es_metric not in eval_metrics:
                                raise KeyError(
                                    f"Early stopping metric '{es_metric}' not "
                                    f"found in eval results. Available: "
                                    f"{sorted(eval_metrics.keys())}. "
                                    f"Check training.early_stopping.metric "
                                    f"config."
                                )
                            current_val = eval_metrics[es_metric]
                            if (
                                es_best is None
                                or current_val < es_best - es_min_delta
                            ):
                                es_best = current_val
                                es_evals_without_improvement = 0
                            else:
                                es_evals_without_improvement += 1
                                if es_evals_without_improvement >= es_patience:
                                    logger.info(
                                        "early_stopping",
                                        step=step,
                                        metric=es_metric,
                                        best=es_best,
                                        patience=es_patience,
                                    )
                                    stopped_early = True
                                    break

                    # Live config polling
                    changes = live_config.poll()
                    if changes:
                        # Apply LR changes
                        if "lr" in changes:
                            new_lr = changes["lr"]
                            old_base_lr = base_lr
                            if (
                                isinstance(new_lr, (int, float))
                                and new_lr > 0
                                and old_base_lr > 0
                            ):
                                ratio = new_lr / old_base_lr
                                base_lr = new_lr
                                self._schedule_params["base_lr"] = base_lr
                                for pg in self.optimizer.param_groups:
                                    pg["base_lr"] = (
                                        pg.get("base_lr", old_base_lr) * ratio
                                    )
                                logger.info(
                                    "live_lr_update",
                                    old_lr=old_base_lr,
                                    new_lr=base_lr,
                                    ratio=ratio,
                                )

                        # Apply early stopping patience
                        if "early_stopping_patience" in changes:
                            new_patience = changes["early_stopping_patience"]
                            if isinstance(new_patience, int) and new_patience > 0:
                                es_patience = new_patience
                                logger.info(
                                    "live_es_patience_update",
                                    patience=es_patience,
                                )

                        # Apply eval/save frequency and max_steps
                        if "eval_every" in changes:
                            val = changes["eval_every"]
                            if isinstance(val, int) and val > 0:
                                eval_every = val
                                logger.info("live_eval_every_update", eval_every=val)
                        if "save_every" in changes:
                            val = changes["save_every"]
                            if isinstance(val, int) and val > 0:
                                save_every = val
                                logger.info("live_save_every_update", save_every=val)
                        if "cuda_empty_cache_every_step" in changes:
                            val = changes["cuda_empty_cache_every_step"]
                            new_cadence = _coerce_empty_cache_cadence(val, 0)
                            if new_cadence != empty_cache_cadence:
                                empty_cache_cadence = new_cadence
                                logger.info(
                                    "live_cuda_empty_cache_cadence_update",
                                    cadence=new_cadence,
                                )
                        if "max_steps" in changes:
                            val = changes["max_steps"]
                            if isinstance(val, int) and val > step:
                                max_steps = val
                                self._schedule_params["max_steps"] = val
                                logger.info("live_max_steps_update", max_steps=val)
                        if "warmup_steps" in changes:
                            val = changes["warmup_steps"]
                            if isinstance(val, int) and val >= 0:
                                warmup_steps = val
                                self._schedule_params["warmup_steps"] = val
                                logger.info("live_warmup_steps_update", warmup_steps=val)

                        # Apply trainer-specific changes (loss weights, etc.)
                        self.apply_live_config(changes)

                    # Checkpoint
                    saved_this_step = False
                    if save_every > 0 and step > 0 and step % save_every == 0:
                        self._training_state = self._build_training_state(
                            es_best, es_evals_without_improvement, wandb_run,
                        )
                        step_metrics = (
                            last_eval_metrics if last_eval_step == step else None
                        )
                        parent = self._registry_parent()
                        self._release_training_transients()
                        from bgkit.utils.step_watchdog import pause as _wd_pause
                        from bgkit.utils.step_watchdog import resume as _wd_resume

                        with memory_budget_scope(
                            "save_checkpoint",
                            cap_gb=self._scope_cap("save_checkpoint"),
                        ):
                            _wd_pause()
                            try:
                                ckpt_path = self.save_checkpoint(
                                    checkpoint_dir, metrics=step_metrics
                                )
                                # Unified-memory: the save's transient blocks
                                # (e.g. bf16 optimizer copies) and the prior
                                # step's freed activations sit in the CUDA
                                # allocator's RESERVED pool. The scope's own
                                # reclaim skips empty_cache() when free>20GB, so
                                # on this run (free ~64GB) they're never returned
                                # to the single shared pool — and the first
                                # post-save step then re-demands pages ON TOP of
                                # them, spiking toward OOM. Flush explicitly here
                                # so those bytes go back to the pool before the
                                # next step. Safe: watchdog paused, and this is a
                                # full step away from the next FLA kernel launch
                                # (the sm_121 cudaFree-near-launch concern that
                                # gated the scope flush does not apply here).
                                import gc as _gc
                                _gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            finally:
                                _wd_resume()
                        registry.register(self._build_registry_entry(
                            ckpt_path, step_metrics, wandb_run,
                            parent_checkpoint=parent,
                        ))
                        if ckpt_manager is not None:
                            ckpt_manager.record(ckpt_path, step, step_metrics)
                            ckpt_manager.prune()
                        (checkpoint_dir / ".last_checkpoint").write_text(
                            str(ckpt_path)
                        )
                        saved_this_step = True

                    # Graceful shutdown check (end of step). Mid-accumulation
                    # detection above handles the long-step case; this catches a
                    # signal that arrived during eval / the periodic save.
                    if interruptor.should_stop:
                        self._graceful_shutdown_save(
                            checkpoint_dir=checkpoint_dir,
                            registry=registry,
                            ckpt_manager=ckpt_manager,
                            wandb_run=wandb_run,
                            es_best=es_best,
                            es_evals_without_improvement=(
                                es_evals_without_improvement
                            ),
                            step=step,
                            interruptor=interruptor,
                            already_saved=saved_this_step,
                        )
                        return

                    step += 1

                # Final eval + checkpoint
                self._release_training_transients()
                from bgkit.utils.step_watchdog import pause as _wd_pause
                from bgkit.utils.step_watchdog import resume as _wd_resume

                with memory_budget_scope(
                    "evaluate", cap_gb=self._scope_cap("evaluate"),
                ):
                    _wd_pause()
                    try:
                        eval_metrics = self.evaluate()
                    finally:
                        _wd_resume()
                eval_metrics = self._normalize_eval_metrics(eval_metrics)
                logger.info("final_eval", **eval_metrics)
                self._training_state = self._build_training_state(
                    es_best, es_evals_without_improvement, wandb_run,
                )
                parent = self._registry_parent()
                self._release_training_transients()
                with memory_budget_scope(
                    "save_checkpoint",
                    cap_gb=self._scope_cap("save_checkpoint"),
                ):
                    ckpt_path = self.save_checkpoint(
                        checkpoint_dir, metrics=eval_metrics
                    )
                registry.register(self._build_registry_entry(
                    ckpt_path, eval_metrics, wandb_run,
                    parent_checkpoint=parent,
                ))
                if ckpt_manager is not None:
                    ckpt_manager.record(
                        ckpt_path, self.global_step, eval_metrics
                    )
                    ckpt_manager.prune()
                (checkpoint_dir / ".last_checkpoint").write_text(str(ckpt_path))
        finally:
            interruptor.restore()
            if getattr(self, "_archiver", None) is not None:
                if getattr(self, "_graceful_shutdown", False):
                    # On a `docker stop`, do NOT block the grace window draining
                    # the ~15 GB checkpoint to the slow (~27 MB/s USB) HDD — an
                    # unbounded wait_idle() here was a root cause of the SIGKILL:
                    # even after the fast NVMe rescue-save completed, `return`
                    # hit this finally and blocked minutes on the HDD copy. The
                    # NVMe copy is authoritative (resume prefers it) and
                    # archive_pending_into lands it on the HDD at next startup.
                    self._archiver.wait_idle(timeout=5.0)
                else:
                    # Training done normally — drain all pending archives to the
                    # HDD so the full checkpoint history is durable before exit.
                    self._archiver.wait_idle()
            if wandb_run is not None:
                wandb_run.finish()

        if stopped_early:
            logger.info("training_complete_early_stop", total_steps=self.global_step)
        else:
            logger.info("training_complete", total_steps=max_steps)
