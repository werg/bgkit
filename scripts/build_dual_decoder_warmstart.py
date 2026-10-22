#!/usr/bin/env python3
"""Build a merged starting checkpoint for round-robin Qwen+Falcon training.

Inputs:
  --encoder-source       Path to ckpt whose ``encoder.pt`` carries the joint
                         encoder backbone + heads + thresholds + both
                         ``projection_blocks.qwen35`` and
                         ``projection_blocks.falcon_h1`` we want to start from.
                         (Typically the latest Qwen realign ckpt, which
                         inherited both projections during its setup.)
  --decoder-qwen-source  Path to ckpt whose ``decoder_merged.pt`` (preferred)
                         or ``decoder.pt`` holds a fully-trained Qwen3.5
                         decoder paired with that encoder.
  --decoder-falcon-source
                         Path to ckpt whose ``decoder_merged.pt`` /
                         ``decoder.pt`` holds a fully-trained Falcon-H1
                         decoder paired with that encoder.
  --output               Output ckpt dir name (created under CHECKPOINT_DIR).

Output structure::

    ${CHECKPOINT_DIR}/<output>/
        encoder.pt          (copied from encoder source)
        decoder_qwen.pt     (canonical, no LoRA wrapping)
        decoder_falcon.pt   (canonical, no LoRA wrapping)
        metadata.json

The merged ckpt is the ``step1_checkpoint`` for
``phase1_summarization_round_robin`` and any future round-robin task.

This script does NOT validate that the two decoders are actually paired
with the encoder source — that's an operator concern. The startup banner
in the trainer will show all three sources so the operator can confirm.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"checkpoint dir not found: {p}")
    return p


def _resolve_decoder_pt(ckpt_dir: Path) -> Path:
    """Pick ``decoder_merged.pt`` if present, else ``decoder.pt``.

    Both are equally valid as 'the decoder state'; merged is the
    LoRA-merged canonical version that loads cleanly into a non-LoRA
    decoder. round-robin training doesn't use LoRA, so we want merged.
    """
    merged = ckpt_dir / "decoder_merged.pt"
    if merged.exists():
        return merged
    raw = ckpt_dir / "decoder.pt"
    if raw.exists():
        return raw
    # Some ckpts (e.g. earlier round-robin runs) save dual decoders as
    # decoder_qwen.pt / decoder_falcon.pt directly — handle the case
    # where the caller passes a ckpt dir with those instead.
    for name in ("decoder_qwen.pt", "decoder_falcon.pt"):
        p = ckpt_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"no decoder.pt / decoder_merged.pt under {ckpt_dir}",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder-source", required=True)
    ap.add_argument("--decoder-qwen-source", required=True)
    ap.add_argument("--decoder-falcon-source", required=True)
    ap.add_argument(
        "--output", required=True,
        help="Output dir NAME (created under $CHECKPOINT_DIR).",
    )
    ap.add_argument(
        "--checkpoint-dir", default=os.environ.get("CHECKPOINT_DIR"),
        help="Override CHECKPOINT_DIR.",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.checkpoint_dir:
        print("ERROR: CHECKPOINT_DIR not set and --checkpoint-dir not given.")
        return 1
    ckpt_root = Path(args.checkpoint_dir)
    enc_src = _resolve(args.encoder_source)
    dec_q_src = _resolve(args.decoder_qwen_source)
    dec_f_src = _resolve(args.decoder_falcon_source)
    dst = ckpt_root / args.output
    if dst.exists():
        if not args.force:
            print(f"ERROR: {dst} exists. Use --force to overwrite.")
            return 1
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # Copy encoder.pt — by reference / hard-copy.
    enc_pt = enc_src / "encoder.pt"
    if not enc_pt.exists():
        print(f"ERROR: {enc_pt} missing")
        return 1
    print(f"Copying encoder: {enc_pt} -> {dst / 'encoder.pt'}")
    shutil.copy(enc_pt, dst / "encoder.pt")

    # Copy decoder_qwen
    dec_q_pt = _resolve_decoder_pt(dec_q_src)
    print(f"Copying decoder_qwen: {dec_q_pt} -> {dst / 'decoder_qwen.pt'}")
    shutil.copy(dec_q_pt, dst / "decoder_qwen.pt")

    # Copy decoder_falcon
    dec_f_pt = _resolve_decoder_pt(dec_f_src)
    print(f"Copying decoder_falcon: {dec_f_pt} -> {dst / 'decoder_falcon.pt'}")
    shutil.copy(dec_f_pt, dst / "decoder_falcon.pt")

    # Minimal metadata so the registry / trainer treat it as a real ckpt.
    metadata = {
        "phase": "manual_dual_decoder_warmstart",
        "step": 0,
        "epoch": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "manual",
        "notes": (
            f"merged from encoder={enc_src.name}, "
            f"decoder_qwen={dec_q_src.name}/{dec_q_pt.name}, "
            f"decoder_falcon={dec_f_src.name}/{dec_f_pt.name}"
        ),
        "parent_checkpoint": str(enc_src),
        "input_sources": {
            "encoder": enc_src.name,
            "decoder_qwen": dec_q_src.name,
            "decoder_falcon": dec_f_src.name,
        },
    }
    with open(dst / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nWrote merged warmstart: {dst}")
    print("Inputs (please verify they are the right paired states):")
    print(f"  encoder         : {enc_src}")
    print(f"  decoder_qwen    : {dec_q_src}  (file: {dec_q_pt.name})")
    print(f"  decoder_falcon  : {dec_f_src}  (file: {dec_f_pt.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
