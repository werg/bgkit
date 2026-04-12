"""File-based live hyperparameter control during training."""

from __future__ import annotations

import json
from pathlib import Path

import structlog

logger = structlog.get_logger()


class LiveConfig:
    """Watches a JSON control file for mid-run hyperparameter changes.

    The control file is shared across all training phases. Each phase reads
    only its own section, keyed by ``namespace`` (typically the training
    phase name like ``phase1_step5``). Unrelated phases' settings are ignored.

    Control file format::

        {
            "phase1_step5": {"eval_every": 500, "save_every": 500, "lr": 5e-5},
            "phase1_step4": {"lr": 3e-5}
        }

    Only keys that changed since the last poll are returned.
    """

    def __init__(self, path: Path | None, namespace: str | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._namespace = namespace
        self._last_mtime: float = 0.0
        self._last_values: dict = {}

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

        # Extract this phase's section; ignore other phases
        if self._namespace and self._namespace in top:
            values = top[self._namespace]
            if not isinstance(values, dict):
                logger.warning(
                    "live_config_namespace_not_dict",
                    namespace=self._namespace,
                    path=str(self._path),
                )
                return {}
        elif self._namespace:
            # Namespace specified but not present in file — nothing for us
            values = {}
        else:
            # No namespace (legacy): use the top-level dict directly
            values = top

        changed = {k: v for k, v in values.items() if self._last_values.get(k) != v}
        self._last_values = values

        if changed:
            logger.info("live_config_changed", namespace=self._namespace, changes=changed)

        return changed
