"""Base trainer: wandb logging, LR scheduling, checkpointing.

Custom training loops — too many heterogeneous training phases for
HF Trainer or Lightning. No Accelerate for now (ICE trains on one GPU
with bf16 autocast). Add Accelerate later for Phase 1/2.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

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
            return
        if self.stream is None:
            self._next_batch = batch
            return
        import torch

        with torch.cuda.stream(self.stream):
            self._next_batch = self._to_device(batch)

    def __next__(self):
        if self.stream is not None:
            import torch

            torch.cuda.current_stream().wait_stream(self.stream)
        if self._next_batch is None:
            raise StopIteration
        batch = self._next_batch
        self._prefetch()
        return batch


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

    result: dict[str, float] = {}
    keys = accum_metrics[0].keys()
    for key in keys:
        values = [m[key] for m in accum_metrics if key in m]
        if values and isinstance(values[0], (int, float)):
            result[key] = sum(values) / len(values)
        elif values and hasattr(values[0], "item"):
            result[key] = (sum(v.item() for v in values)) / len(values)
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

    def _post_step(self, step: int) -> None:
        """Hook called after each optimizer step. Override for per-step bookkeeping.

        .. deprecated:: Use ``_post_optimizer_step`` instead.
        """

    def _post_optimizer_step(self, step: int) -> None:
        """Hook called after optimizer.step(). Override for per-step bookkeeping."""
        self._post_step(step)

    def _pre_step_hook(self) -> None:
        """Hook called at the top of each training step (before LR schedule).

        Override for dataloader rebuilds, curriculum transitions, etc.
        Return value is ignored.
        """

    def _post_lr_schedule(self, step: int) -> None:
        """Hook called after the LR schedule is applied, before the accumulation loop.

        Override to apply per-param-group LR adjustments (e.g. local warmup ramps
        for newly added param groups at stage transitions).
        """

    def _add_step_metrics(self, metrics: dict[str, float]) -> None:
        """Add trainer-specific metrics to the step dict before logging.

        Override to inject metrics like bidi_alpha, compression ratio, etc.
        Modify ``metrics`` in place.
        """

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
        grad_norm = clip_grad_norm(self.trainable_parameters())
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
        """
        state_by_name: dict = {}
        for name, param in self._named_parameters_for_optimizer():
            if param in self.optimizer.state:
                # Copy the per-param state dict so the saved tensor isn't
                # aliased to the live optimizer buffer (torch.save will
                # serialize whatever we hand it, but an aliased dict is
                # fragile if the optimizer mutates mid-serialization).
                state_by_name[name] = dict(self.optimizer.state[param])
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
        """
        import torch

        matched = 0
        new = 0
        for name, param in self._named_parameters_for_optimizer():
            if name in state_by_name:
                saved = state_by_name[name]
                device = param.device
                moved = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in saved.items()
                }
                self.optimizer.state[param] = moved
                matched += 1
            else:
                new += 1
        skipped = len(state_by_name) - matched
        logger.info(
            "optimizer_state_restored_by_name",
            matched=matched,
            new=new,
            skipped=skipped,
        )

    def save_checkpoint(
        self, checkpoint_dir: Path, metrics: dict[str, float] | None = None
    ) -> Path:
        """Save checkpoint with phase metadata and lineage."""
        metadata = CheckpointMetadata(
            phase=self.cfg.training.phase,
            step=self.global_step,
            epoch=self.epoch,
            parent_checkpoint=self._last_checkpoint_path,
            metrics=metrics,
            schedule_params=self._schedule_params,
            training_state=self._training_state,
            optimizer_type=self._optimizer_type,
        )
        ckpt_path = save_checkpoint(
            checkpoint_dir,
            metadata,
            model=self.model.state_dict(),
            optimizer_state_by_name=self._build_optimizer_state_by_name(),
        )
        self._last_checkpoint_path = str(ckpt_path)
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

    def _post_weight_load_hook(self) -> None:
        """Run after weights + step + training_state are restored,
        before the optimizer state is loaded.

        Default: no-op.  The canonical place to rebuild the optimizer
        when trainable parameters depend on restored state (e.g. a
        distillation stage from ``training_state`` or a freeze schedule
        keyed off ``global_step``).
        """

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

        from torch.utils.data import DataLoader

        from bgkit.data.samplers import PackedTokenBudgetSampler

        old_budget = getattr(self.train_sampler, "_max_batch_tokens", None)
        cursor = self._microbatches_in_epoch

        seed = getattr(self.train_sampler, "_seed", None)
        epoch = getattr(self.train_sampler, "_epoch", 0)
        self.train_sampler = PackedTokenBudgetSampler(
            self.train_dataset,
            lengths=self._train_lengths,
            max_batch_tokens=new_budget,
            shuffle=True,
            seed=seed,
        )
        # Restore epoch so shuffle order is deterministic on resume.
        self.train_sampler.set_epoch(epoch)
        # Preserve cursor so the next step continues from the same position.
        self.train_sampler.set_batch_cursor(cursor)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=self._train_collate_fn,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )
        self._dataloader_invalidated = True

        logger.info(
            "live_max_batch_tokens_update",
            old=old_budget,
            new=new_budget,
            cursor_preserved=cursor,
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
        eval_sampler = PackedTokenBudgetSampler(
            self.eval_dataset,
            lengths=self._eval_lengths,
            max_batch_tokens=new_budget,
            shuffle=False,
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
        )

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
        self.setup()

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
        checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))

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
        if resume_path == "none":
            resume_path = None  # explicitly disabled
        elif resume_path is None:
            phase = getattr(tcfg, "phase", None)
            if phase:
                auto_resolved = resolve_latest_checkpoint(checkpoint_dir, phase)
                if auto_resolved is not None:
                    resume_path = str(auto_resolved)
                    logger.info("auto_resume_resolved", checkpoint=resume_path, phase=phase)
        is_resuming = False
        resume_step: int | None = None
        if resume_path is not None:
            self.load_checkpoint(Path(resume_path))
            # Checkpoint was saved after step completed, so resume from next step
            self.global_step += 1
            is_resuming = True
            resume_step = self.global_step
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

        # Store schedule params for checkpointing
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

        # Resume warmup: linear ramp from near-zero to scheduled LR over N steps
        # after resuming from a checkpoint. Helps optimizer re-stabilize when
        # training regime changed (new param groups, extended schedule, etc.).
        resume_warmup_steps = int(tcfg.get("resume_warmup_steps", 200))
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
                registry=registry,
            )
            ckpt_manager.load_existing(checkpoint_dir)
        else:
            ckpt_manager = None

        last_eval_metrics: dict[str, float] | None = None
        last_eval_step = -1

        stopped_early = False
        try:
            with GracefulInterruptor() as interruptor:
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

                    grad_norm = clip_grad_norm(self.trainable_parameters())

                    if not math.isfinite(grad_norm):
                        raise RuntimeError(
                            f"NaN/Inf grad_norm at step {step} (grad_norm={grad_norm}). "
                            "This usually indicates a numerical stability issue."
                        )
                    self.optimizer.step()
                    self._post_optimizer_step(step)

                    # Release cached-but-unused CUDA allocator blocks back
                    # to the OS each optimizer step. The DGX Spark's
                    # unified-memory allocator, combined with variable-
                    # shape batches from a shuffled-order sampler, was
                    # observed to accumulate fragmentation (monotonic
                    # system-memory growth over ~30 steps leading to a
                    # whole-machine stall, 2026-04-19). ``empty_cache``
                    # forces a defragmentation pass; cost is ~ms scale
                    # compared to multi-second steps. Gate via
                    # ``training.cuda_empty_cache_every_step`` in case a
                    # future workload hits an allocator path where this
                    # hurts more than it helps.
                    if tcfg.get("cuda_empty_cache_every_step", True):
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
                        with memory_budget_scope(
                            "evaluate", cap_gb=self._scope_cap("evaluate"),
                        ):
                            eval_metrics = self.evaluate()
                        eval_metrics = {
                            f"eval/{k}": v for k, v in eval_metrics.items()
                        }
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
                        with memory_budget_scope(
                            "save_checkpoint",
                            cap_gb=self._scope_cap("save_checkpoint"),
                        ):
                            ckpt_path = self.save_checkpoint(
                                checkpoint_dir, metrics=step_metrics
                            )
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

                    # Graceful shutdown check
                    if interruptor.should_stop:
                        if not saved_this_step:
                            self._training_state = self._build_training_state(
                                es_best, es_evals_without_improvement, wandb_run,
                            )
                            parent = self._registry_parent()
                            self._release_training_transients()
                            # Graceful shutdown: do NOT enforce the cap — we
                            # want the rescue-save to succeed even under
                            # memory pressure that may be causing the
                            # shutdown.  Still scope it for logging/reclaim.
                            with memory_budget_scope("save_checkpoint_shutdown"):
                                ckpt_path = self.save_checkpoint(checkpoint_dir)
                            registry.register(self._build_registry_entry(
                                ckpt_path, None, wandb_run,
                                status="interrupted",
                                parent_checkpoint=parent,
                            ))
                            if ckpt_manager is not None:
                                ckpt_manager.record(ckpt_path, step, None)
                                ckpt_manager.prune()
                            (checkpoint_dir / ".last_checkpoint").write_text(
                                str(ckpt_path)
                            )
                        logger.info(
                            "graceful_shutdown_complete",
                            step=step,
                            signal=interruptor.received_signal.name
                            if interruptor.received_signal
                            else None,
                        )
                        return

                    step += 1

                # Final eval + checkpoint
                self._release_training_transients()
                with memory_budget_scope(
                    "evaluate", cap_gb=self._scope_cap("evaluate"),
                ):
                    eval_metrics = self.evaluate()
                eval_metrics = {
                    f"eval/{k}": v for k, v in eval_metrics.items()
                }
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
            if wandb_run is not None:
                wandb_run.finish()

        if stopped_early:
            logger.info("training_complete_early_stop", total_steps=self.global_step)
        else:
            logger.info("training_complete", total_steps=max_steps)
