#!/usr/bin/env python
"""Swap one checkpoint pin for another in a compose file, or refuse.

WHY THIS IS A SEPARATE, TESTED FILE. It used to be a heredoc inside
``restart-train.sh`` guarded by ``if old not in s: refuse``. That guard does
not catch an EMPTY pin, because ``"" in s`` is always true -- so for a service
with no ``+resume_checkpoint=`` (a cold-start service, which is most of them)
it fell through to ``s.replace("", new)``, which inserts the replacement
between every character of the file. On 2026-08-31 that destroyed
docker-compose.yaml: 56226 "occurrences", 71 services, recovered only because
the file was committed.

A service with no pin does not need one. Auto-resume is run-scoped and, since
the config-fingerprint check, refuses to continue under a changed model
config -- so the right behaviour for a pinless service is to leave the file
alone and let auto-resume resolve it, not to invent a pin.
"""

from __future__ import annotations

import sys
from pathlib import Path


class RepinRefusedError(Exception):
    """The substitution is not safe to perform."""


def repin(text: str, old: str, new: str) -> tuple[str, int]:
    """Return ``(new_text, occurrences)``.

    Raises :class:`RepinRefusedError` rather than performing a substitution that
    cannot be what the caller meant.
    """
    if not new.strip():
        raise RepinRefusedError("refusing: the new pin is empty")
    if not old.strip():
        raise RepinRefusedError(
            "refusing: the current pin is empty. A service with no "
            "+resume_checkpoint= is resolved by run-scoped auto-resume; "
            "there is nothing to swap."
        )
    if old == new:
        return text, 0
    if old not in text:
        raise RepinRefusedError(f"refusing: current pin {old!r} not found verbatim")
    return text.replace(old, new), text.count(old)


def main() -> int:
    old, new, compose_file = sys.argv[1], sys.argv[2], sys.argv[3]
    path = Path(compose_file)
    try:
        updated, n = repin(path.read_text(), old, new)
    except RepinRefusedError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if n == 0:
        print(f"pin already at {new}")
        return 0
    path.write_text(updated)
    print(f"pin -> {new}" + (f"  ({n} occurrences)" if n > 1 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
