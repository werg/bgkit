"""File-based live hyperparameter control during training."""

from __future__ import annotations

import json
from pathlib import Path

import structlog

logger = structlog.get_logger()


class LiveConfig:
    """Watches a JSON control file for mid-run hyperparameter changes.

    The control file is optional; if it doesn't exist, ``poll()`` returns ``{}``.
    Only keys that changed since the last poll are returned.

    Control file format (any subset of keys)::

        {"lr": 5e-5, "w_repro": 0.8, "early_stopping_patience": 10}
    """

    def __init__(self, path: Path | None) -> None:
        self._path = Path(path) if path is not None else None
        self._last_mtime: float = 0.0
        self._last_values: dict = {}

    def poll(self) -> dict:
        """Check the control file and return changed keys.

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
            values = json.loads(raw)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("live_config_parse_error", path=str(self._path), error=str(exc))
            # Don't update _last_mtime so we re-read on next poll even if
            # the file is corrected without an mtime change.
            return {}

        if not isinstance(values, dict):
            logger.warning("live_config_not_dict", path=str(self._path))
            return {}

        self._last_mtime = stat.st_mtime

        changed = {k: v for k, v in values.items() if self._last_values.get(k) != v}
        self._last_values = values

        if changed:
            logger.info("live_config_changed", changes=changed)

        return changed
