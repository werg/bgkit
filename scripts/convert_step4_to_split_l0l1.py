"""Convert a legacy Phase 1 Step 4 checkpoint into the split-L0/L1 layout.

Reads the latest ``phase1_step4`` checkpoint from the registry, runs the
legacy migration via ``BgKITEncoder.from_pretrained_legacy_step4_checkpoint``,
and saves the result as a multi-artifact split-layout checkpoint
(``l0.pt``, ``l1.pt``, ``projection_block.pt``, ``decoder.pt``,
``metadata.json``) under the registry phase ``phase1_step4_split``.

After conversion the new ``encoder.l0.head`` and ``encoder.l1.head`` carry
the trained Step-4 head weights (transferred unchanged — heads still fire
at the block-1 hook in the new architecture). ``encoder.l0.survive_embedding``
and ``encoder.l1.survive_embedding`` are also transferred. The L1 backbone
is a deepcopy of L0's backbone at construction.

Usage::

    .venv/bin/python scripts/convert_step4_to_split_l0l1.py \\
        [--source PHASE_NAME] [--checkpoint-name NAME] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from datetime import UTC, datetime

from bgkit.env import get_checkpoint_dir
from bgkit.models.encoder import BgKITEncoder
from bgkit.training.checkpoint_registry import CheckpointRegistry, RegistryEntry
from bgkit.training.checkpointing import load_checkpoint

logger = logging.getLogger("convert_step4_to_split_l0l1")


def _save_split_artifacts(
    out_dir: Path,
    encoder: BgKITEncoder,
    decoder_state: dict | None,
    src_metadata,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    enc_sd = encoder.state_dict()
    l0_sd = {k[len("l0."):]: v for k, v in enc_sd.items() if k.startswith("l0.")}
    l1_sd = {k[len("l1."):]: v for k, v in enc_sd.items() if k.startswith("l1.")}
    proj_sd = {
        k[len("projection_block."):]: v
        for k, v in enc_sd.items()
        if k.startswith("projection_block.")
    }

    torch.save(l0_sd, out_dir / "l0.pt")
    torch.save(l1_sd, out_dir / "l1.pt")
    torch.save(proj_sd, out_dir / "projection_block.pt")
    if decoder_state is not None:
        torch.save(decoder_state, out_dir / "decoder.pt")

    metadata = {
        "phase": "phase1_step4_split",
        "source_phase": getattr(src_metadata, "phase", None),
        "source_step": getattr(src_metadata, "step", None),
        "source_metrics": getattr(src_metadata, "metrics", None),
        "note": (
            "Converted from a legacy Step-4 checkpoint via "
            "BgKITEncoder.from_pretrained_legacy_step4_checkpoint. "
            "L0/L1 backbones share initial state (deepcopy at construction). "
            "L0/L1 heads + survive_embedding TRANSFERRED from the legacy "
            "compressor.head_base_l{0,1} / compressor.survive_embedding "
            "(no re-initialisation needed — heads still fire at the block-1 hook)."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="phase1_step4",
        help="Source registry phase name (default: phase1_step4)",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        help="Specific checkpoint name to convert (overrides --source latest()). "
             "Use this when the latest checkpoint is interrupted/incomplete and "
             "you want a known-good earlier one.",
    )
    parser.add_argument(
        "--target-phase",
        default="phase1_step4_split",
        help="Output registry phase name (default: phase1_step4_split)",
    )
    parser.add_argument(
        "--backbone-name",
        default="Qwen/Qwen3.5-0.8B-Base",
        help="HF model id used to build the fresh encoder skeleton.",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=1024,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the source checkpoint and run the migration in-memory; "
             "do not write the output artifact.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    checkpoint_dir = get_checkpoint_dir()
    registry = CheckpointRegistry(checkpoint_dir)
    registry.backfill(checkpoint_dir)
    if args.checkpoint_name is not None:
        src_entry = registry.get(args.checkpoint_name)
        if src_entry is None:
            logger.error("No checkpoint registered under name %r", args.checkpoint_name)
            return 1
    else:
        src_entry = registry.latest(phase=args.source)
        if src_entry is None:
            logger.error("No checkpoint registered under phase %r", args.source)
            return 1

    src_path = checkpoint_dir / src_entry.name
    logger.info("loading source checkpoint: %s", src_path)
    metadata, state_dicts = load_checkpoint(src_path)

    if "encoder" not in state_dicts:
        logger.error("source checkpoint missing 'encoder' key")
        return 1

    encoder = BgKITEncoder.from_pretrained_legacy_step4_checkpoint(
        args.backbone_name,
        state_dicts["encoder"],
        hidden_dim=args.hidden_dim,
        torch_dtype=torch.bfloat16,
    )
    decoder_state = state_dicts.get("decoder", None)

    out_dir = checkpoint_dir / f"{args.target_phase}_from_{src_path.name}"
    if args.dry_run:
        logger.info("dry-run: would write to %s", out_dir)
        return 0

    _save_split_artifacts(out_dir, encoder, decoder_state, metadata)
    logger.info("wrote split-layout checkpoint: %s", out_dir)

    entry = RegistryEntry(
        name=out_dir.name,
        phase=args.target_phase,
        step=int(getattr(metadata, "step", 0) or 0),
        epoch=int(getattr(metadata, "epoch", 0) or 0),
        timestamp=datetime.now(UTC).isoformat(),
        metrics=getattr(metadata, "metrics", None) or {},
        parent_checkpoint=src_path.name,
        notes="Converted from legacy Step-4 checkpoint to split-L0/L1 layout.",
        tags=["converted"],
    )
    registry.register(entry)
    logger.info("registered %s under phase %r", out_dir.name, args.target_phase)
    return 0


if __name__ == "__main__":
    sys.exit(main())
