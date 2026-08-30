#!/usr/bin/env python
"""What did training ACTUALLY move between two checkpoints?

WHY THIS EXISTS. On 2026-08-29 the question "did Phase-2 training walk the
projection off the decoder's embedding manifold, or was it always there?" had
been open for a day. Every attempt to answer it needed a GPU window, and the
GPU probe silently skipped the Phase-1 base on a checkpoint-layout KeyError.

The question turned out to be answerable on CPU in two minutes by comparing
weights directly:

    l1.backbone                 0.0000
    l0.backbone                 0.0001
    projection_blocks.qwen35    0.0039   (output_norm.weight BIT-IDENTICAL)
    l0.head                     0.1727
    DECODER qwen35              0.0120   (uniform across every layer)

That killed the manifold hypothesis outright — a checkpoint with rep_gain
2.03-2.95 nats and one with 0.01 have the SAME projection, so nothing about the
representation drifted and only the reader changed. It also showed the encoder
backbones are effectively frozen in Phase 2, which reframed what the run is
even training.

Generalised from that one-off because "which components did this stage move,
and by how much?" is the first question for any regression across a training
boundary, and answering it should never again require writing a bespoke script.

Groups are formed by collapsing layer indices (``layers.7.`` -> ``layers.N.``)
so per-layer tensors aggregate into one comparable row, and reported as a
relative Frobenius change ``||B - A|| / ||A||`` summed within the group.

Usage (CPU, no GPU window, safe to run alongside training):

    python scripts/compare_checkpoint_weights.py \\
        --a $CHECKPOINT_DIR/phase1_summarization_round_robin_step51945_... \\
        --b $CHECKPOINT_DIR/phase2_kb_step2599_... \\
        --filter projection --top 20
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bgkit.training.checkpointing import load_checkpoint, normalize_model_state


def _flat_model_state(path: Path) -> dict:
    """Load either checkpoint layout as one flat prefixed state dict."""
    _meta, state = load_checkpoint(path)
    return normalize_model_state(state)["model"]


def _group_of(key: str, depth: int) -> str:
    """Collapse layer indices so per-layer tensors aggregate into one row."""
    k = re.sub(r"\.\d+\.", ".N.", key)
    parts = k.split(".")
    return ".".join(parts[:depth]) if depth > 0 else k


def _is_weight(key: str, t) -> bool:
    """Exclude counters and other non-weight state from the comparison.

    THIS BIT THE FIRST USE OF THIS TOOL. ``encoder`` state carries a ``_step``
    tensor whose value is a step count, so its norm was 156006 against real
    weight tensors at norm 20-105. Summed in quadrature it dominated its group
    by ~10^6, and since a step counter differs by ~0 in relative terms it drove
    the whole group's rel_change to ~0.0000 — which I reported as "the encoder
    is untouched by Phase 2". Excluding non-float and scalar-ish state, the same
    backbones differ substantially. A magnitude-weighted aggregate is only
    meaningful over tensors that are actually parameters.
    """
    if not hasattr(t, "dtype") or not t.dtype.is_floating_point:
        return False
    if t.ndim == 0 or t.numel() <= 1:
        return False
    return not key.split(".")[-1].startswith("_")   # _step, _count, ...


def compare(a: dict, b: dict, depth: int) -> tuple[dict, list]:
    """Return (per-group {norm_a, delta, rel}, missing-key report)."""
    groups: dict[str, dict[str, float]] = {}
    missing: list[str] = []
    skipped_non_weight: list[str] = []
    for k, va in a.items():
        if not _is_weight(k, va):
            skipped_non_weight.append(k)
            continue
        vb = b.get(k)
        if vb is None:
            # Tolerate the LoRA re-key: on-disk `q_proj.weight` becomes
            # `q_proj.base_layer.weight` once adapters are installed.
            vb = b.get(k.replace(".weight", ".base_layer.weight"))
        if vb is None or not hasattr(vb, "shape") or va.shape != vb.shape:
            missing.append(k)
            continue
        x = va.float()
        y = vb.float()
        g = groups.setdefault(_group_of(k, depth), {"norm_a": 0.0, "delta": 0.0, "n": 0})
        g["norm_a"] += float(x.norm()) ** 2
        g["delta"] += float((y - x).norm()) ** 2
        g["n"] += 1
    for g in groups.values():
        g["norm_a"] = g["norm_a"] ** 0.5
        g["delta"] = g["delta"] ** 0.5
        g["rel"] = g["delta"] / max(g["norm_a"], 1e-12)
    if skipped_non_weight:
        print(f"(excluded {len(skipped_non_weight)} non-weight tensors, "
              f"e.g. {skipped_non_weight[:3]})")
    return groups, missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="baseline checkpoint dir")
    ap.add_argument("--b", required=True, help="later checkpoint dir")
    ap.add_argument("--depth", type=int, default=2,
                    help="key components to group by (0 = per-tensor)")
    ap.add_argument("--filter", default="", help="only groups matching this substring")
    ap.add_argument("--top", type=int, default=40, help="rows to print")
    ap.add_argument("--json", default="", help="also write the full result here")
    args = ap.parse_args()

    a = _flat_model_state(Path(args.a))
    b = _flat_model_state(Path(args.b))
    groups, missing = compare(a, b, args.depth)

    rows = [(g, v) for g, v in groups.items() if args.filter in g]
    rows.sort(key=lambda kv: -kv[1]["rel"])

    print(f"\n{'group':<52}{'rel_change':>12}{'||A||':>12}{'tensors':>9}")
    print("-" * 85)
    for g, v in rows[: args.top]:
        print(f"{g:<52}{v['rel']:12.4f}{v['norm_a']:12.3f}{int(v['n']):9d}")
    if len(rows) > args.top:
        print(f"... {len(rows) - args.top} more groups not shown (raise --top)")
    if missing:
        # Never silent: an unmatched key is a real difference between the two
        # checkpoints, not something to quietly drop from the comparison.
        print(f"\n{len(missing)} key(s) in A had no shape-matching counterpart in B, e.g.:")
        for k in missing[:5]:
            print(f"    {k}")

    print("\nrel_change = ||B - A|| / ||A|| within the group.")
    print("~0.000 means the stage did not train that component at all.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"groups": groups, "missing": missing}, indent=2, default=str,
        ))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
