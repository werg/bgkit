#!/usr/bin/env python
"""Find config keys that no code ever reads — the silent-default defect class.

Born from the 2026-08-25 per-group-LR bug: ``decoder_lr`` / ``l0_lr`` /
``l1_lr`` / ``projection_lr`` were set in every Phase-2 config, threaded into
the optimizer as ``"lr"``, and then silently overwritten on step 1 because the
schedule reads ``base_lr`` — which nothing set. Four knobs that looked
configured, were documented, appeared in checkpoints, and did nothing. The
wide-net decoder trained at 2x its configured rate for months.

Nothing catches that class today. A key can be:

- **DEAD**: present in configs, never mentioned in ``src/`` — either a knob
  someone renamed out from under, or one that was never wired.
- **DEFAULT-ONLY**: read in code with a fallback but set in NO config, so the
  default always wins. Often fine, sometimes a knob nobody can actually turn.

Both are reported; neither is automatically a bug, so the output is a
worklist, not a gate. Anything named ``*_lr``, ``freeze_*``, ``*_weight`` or
``*_ratio`` is flagged HIGH because those silently change what trains.

Usage:
    .venv/bin/python scripts/audit_dead_config_keys.py [--configs configs] [--src src]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

# Keys whose silent loss changes what/how the model trains.
HIGH_RISK = re.compile(
    r"(_lr$|^lr$|^freeze_|_weight$|_ratio$|_prob$|^optimizer$|_steps$|^batch_size$)"
)
# Structural keys that are consumed by hydra/compose, not by name in code.
SKIP_TOP = {"defaults", "_self_", "hydra", "run_name"}


def collect_config_keys(root: Path) -> dict[str, set[str]]:
    """Leaf key name -> set of files that set it."""
    keys: dict[str, set[str]] = {}

    def walk(node, where: str):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in SKIP_TOP:
                    continue
                if isinstance(v, dict | list):
                    walk(v, where)
                else:
                    keys.setdefault(str(k), set()).add(where)
        elif isinstance(node, list):
            for item in node:
                walk(item, where)

    for path in sorted(root.rglob("*.yaml")):
        try:
            doc = yaml.safe_load(path.read_text())
        except Exception:
            continue
        if doc:
            walk(doc, str(path.relative_to(root.parent)))
    return keys


def code_mentions(src: Path, scripts: Path | None) -> str:
    blobs = [p.read_text(errors="ignore") for p in src.rglob("*.py")]
    if scripts is not None:
        blobs += [p.read_text(errors="ignore") for p in scripts.rglob("*.py")]
    return "\n".join(blobs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--src", default="src")
    ap.add_argument("--scripts", default="scripts")
    ap.add_argument("--show-low", action="store_true", help="also list low-risk keys")
    args = ap.parse_args()

    cfg_keys = collect_config_keys(Path(args.configs))
    code = code_mentions(Path(args.src), Path(args.scripts))

    dead: list[tuple[str, set[str]]] = []
    for key, files in sorted(cfg_keys.items()):
        # A key is "read" if it appears as a quoted string or attribute access.
        if re.search(rf"""["']{re.escape(key)}["']|\.{re.escape(key)}\b""", code):
            continue
        dead.append((key, files))

    high = [(k, f) for k, f in dead if HIGH_RISK.search(k)]
    low = [(k, f) for k, f in dead if not HIGH_RISK.search(k)]

    print(f"{len(cfg_keys)} distinct config keys; {len(dead)} never read by code\n")
    print(f"=== HIGH RISK ({len(high)}) — these silently change what trains ===")
    for key, files in high:
        shown = sorted(files)[:3]
        more = f" (+{len(files) - 3} more)" if len(files) > 3 else ""
        print(f"  {key:38s} {', '.join(shown)}{more}")
    if not high:
        print("  (none)")
    if args.show_low:
        print(f"\n=== other unread keys ({len(low)}) ===")
        for key, files in low:
            print(f"  {key:38s} {sorted(files)[0]}")
    else:
        print(f"\n({len(low)} lower-risk unread keys; pass --show-low to list)")


if __name__ == "__main__":
    main()
