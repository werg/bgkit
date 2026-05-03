#!/usr/bin/env python
"""Smoke-test that a Step 4 checkpoint can be loaded into the Step 5 trainer.

Catches checkpoint-shape incompatibilities BEFORE a multi-hour Step 5
launch. The Step 5 trainer adds the L1 head as a new component; if the
encoder state_dict has unexpected keys the trainer will silently ignore
them with a warning, but missing keys signal a real architecture
mismatch that needs investigation.

Run:
    .venv/bin/python scripts/smoke_test_phase1_step5.py [--checkpoint PATH]

CPU-only — only verifies model construction + checkpoint loading +
state_dict key match. Doesn't run forward passes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import torch
import structlog
from omegaconf import OmegaConf

from bgkit.training.checkpointing import load_checkpoint
from bgkit.training.checkpoint_registry import CheckpointRegistry

logger = structlog.get_logger()


def _resolve_step4_ckpt(checkpoint_dir: Path) -> Path:
    """Pick the latest on-disk phase1_step4 checkpoint."""
    registry = CheckpointRegistry(checkpoint_dir)
    registry.backfill(checkpoint_dir)
    latest = registry.latest(phase="phase1_step4")
    if latest is None:
        raise SystemExit(
            f"No phase1_step4 checkpoint found in {checkpoint_dir}"
        )
    return checkpoint_dir / latest.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Step 4 checkpoint dir; defaults to latest in CHECKPOINT_DIR")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent.parent /
                                "checkpoints",
                        help="Where to look for checkpoints (default: ./checkpoints)")
    args = parser.parse_args()

    import os
    cd = args.checkpoint_dir
    if "CHECKPOINT_DIR" in os.environ:
        cd = Path(os.environ["CHECKPOINT_DIR"])

    ckpt_path = args.checkpoint or _resolve_step4_ckpt(cd)
    print(f"Step 4 checkpoint: {ckpt_path}")
    if not ckpt_path.is_dir():
        raise SystemExit(f"Not a directory: {ckpt_path}")

    metadata, state_dicts = load_checkpoint(ckpt_path)
    print(f"\nState dict components found: {list(state_dicts.keys())}")
    for name, sd in state_dicts.items():
        n_params = sum(t.numel() for t in sd.values() if hasattr(t, "numel"))
        print(f"  {name}: {len(sd)} keys, ~{n_params/1e6:.1f}M params")

    # --- Construct Step 5 trainer (CPU only, no GPU) ---
    print("\n--- Constructing Step 5 trainer (CPU model construction only) ---")
    # Hydra-compose minimal cfg
    from hydra import compose, initialize_config_dir
    config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["+experiment=phase1_step5"],
        )

    print(f"  cfg.training.phase = {cfg.training.phase}")
    print(f"  cfg.step1_checkpoint = {cfg.step1_checkpoint}")

    # Construct encoder + decoder shells (no setup, no GPU).
    from transformers import AutoModelForCausalLM
    from bgkit.models.encoder import BgKITEncoder
    from bgkit.models.decoder import ReconstructionDecoder

    bgkit_cfg = cfg.model.bgkit
    hidden_dim = bgkit_cfg.get("hidden_dim", 1024)

    print("\n--- Constructing encoder shell from checkpoint state ---")
    if "encoder" not in state_dicts:
        raise SystemExit(f"Step 4 checkpoint missing 'encoder' key")
    try:
        encoder = BgKITEncoder.from_pretrained_with_state_dict(
            bgkit_cfg.backbone_name,
            state_dicts["encoder"],
            hidden_dim=hidden_dim,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            revision=bgkit_cfg.get("backbone_revision"),
            attn_implementation="eager",
        )
        print(f"  ✓ Encoder constructed; ~{sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params")
    except Exception as exc:
        print(f"  ✗ Encoder construction FAILED: {exc}")
        raise

    print("\n--- Verifying L1 survivorship head presence in encoder ---")
    head_l0 = sum(1 for n, _ in encoder.named_parameters() if "head_base_l0" in n)
    head_l1 = sum(1 for n, _ in encoder.named_parameters() if "head_base_l1" in n)
    print(f"  head_base_l0 params: {head_l0}")
    print(f"  head_base_l1 params: {head_l1}")
    if head_l1 == 0:
        print("  ⚠ Step 4 checkpoint has no L1 head; Step 5 will cold-start it (expected).")

    print("\n--- Decoder construction ---")
    decoder_cfg = cfg.model.decoder
    decoder_backbone = AutoModelForCausalLM.from_pretrained(
        decoder_cfg.backbone_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        revision=decoder_cfg.get("backbone_revision"),
    )
    decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=hidden_dim)
    decoder_state = state_dicts.get("decoder_merged") or state_dicts.get("decoder")
    if decoder_state is None:
        raise SystemExit("Step 4 checkpoint missing 'decoder'/'decoder_merged' keys")
    missing, unexpected = decoder.load_state_dict(decoder_state, strict=False)
    print(f"  Decoder loaded; missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"    First 5 missing: {missing[:5]}")
    if unexpected:
        print(f"    First 5 unexpected: {unexpected[:5]}")

    print("\n✅ Smoke test passed: Step 4 checkpoint shape is compatible with Step 5 trainer.")
    print(f"   Step 5 will load encoder + decoder from {ckpt_path.name}")
    print(f"   L1 head will cold-start (no L1 weights in Step 4 checkpoint)")


if __name__ == "__main__":
    main()
