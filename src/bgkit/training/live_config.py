"""File-based live hyperparameter control during training."""

from __future__ import annotations

import json
from pathlib import Path

import structlog

logger = structlog.get_logger()


class LiveConfig:
    """Watches a JSON control file for mid-run hyperparameter changes.

    The control file is shared across all training phases and runs. A run
    reads two sections and merges them, run block over phase block:

    * ``namespace`` — the training phase (``phase1_step6``, ``phase2_kb``),
      the long-standing operator interface. It persists across runs of the
      same phase, so keys left there by a previous run apply to the next one
      at its first poll. Those inherited keys are logged at WARNING on the
      first poll (``live_config_phase_block_inherited``) so a stale block is
      visible at launch instead of silently overriding the experiment config
      (2026-08-23: v6 inherited v5b's ``max_steps``).
    * ``run_namespace`` — the run name (``phase2_kb_widenet_v6``). Keys here
      belong to exactly one run and override the phase block key-wise.

    Control file format::

        {
            "phase1_step6": {"eval_every": 500, "save_every": 500, "lr": 5e-5},
            "phase2_kb": {"eval_every": 250},
            "phase2_kb_widenet_v6": {"max_steps": 2630}
        }

    Only keys that changed since the last poll are returned.
    """

    def __init__(
        self,
        path: Path | None,
        namespace: str | None = None,
        run_namespace: str | None = None,
    ) -> None:
        self._path = Path(path) if path is not None else None
        self._namespace = namespace
        self._run_namespace = run_namespace if run_namespace != namespace else None
        self._last_mtime: float = 0.0
        self._last_values: dict = {}
        self._polled_once = False

    def _section(self, top: dict, key: str | None) -> dict | None:
        """Return ``top[key]`` as a dict, ``{}`` if absent, ``None`` if malformed."""
        if not key or key not in top:
            return {}
        values = top[key]
        if not isinstance(values, dict):
            logger.warning("live_config_namespace_not_dict", namespace=key, path=str(self._path))
            return None
        return values

    def poll(self) -> dict:
        """Check the control file and return changed keys for this namespace.

        Returns:
            Dict of keys whose values changed since last poll.
            Empty dict if file is missing, unchanged, or malformed.
        """
        if self._path is None:
            return {}

        try:
            stat = self._path.stat()
        except FileNotFoundError:
            return {}

        if stat.st_mtime == self._last_mtime:
            return {}

        try:
            raw = self._path.read_text()
            top = json.loads(raw)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("live_config_parse_error", path=str(self._path), error=str(exc))
            return {}

        if not isinstance(top, dict):
            logger.warning("live_config_not_dict", path=str(self._path))
            return {}

        self._last_mtime = stat.st_mtime

        if self._namespace:
            phase_values = self._section(top, self._namespace)
            run_values = self._section(top, self._run_namespace)
            if phase_values is None or run_values is None:
                return {}
            values = dict(phase_values)
            values.update(run_values)
            inherited = sorted(k for k in phase_values if k not in run_values)
        else:
            # No namespace (legacy): use the top-level dict directly
            values = top
            inherited = []

        first_poll = not self._polled_once
        self._polled_once = True
        if first_poll and inherited:
            # Keys from the phase block applying to a freshly started process:
            # either intended operator defaults or a stale block from the
            # previous run of this phase. Surface them at launch.
            logger.warning(
                "live_config_phase_block_inherited",
                namespace=self._namespace,
                run_namespace=self._run_namespace,
                keys=inherited,
            )

        changed = {k: v for k, v in values.items() if self._last_values.get(k) != v}
        self._last_values = values

        if changed:
            logger.info(
                "live_config_changed",
                namespace=self._namespace,
                run_namespace=self._run_namespace,
                changes=changed,
            )

        return changed
