#!/usr/bin/env python
"""Standalone Family-A blob-SFT eval (teacher-forced + zeroed-rep ablation).

Runs :meth:`BlobSFTTrainer.evaluate` on a given checkpoint without the
training loop (and without the resume-replay cost of relaunching a completed
run). Reports the full metric set including the probe zeroed-gap — the
"is recall read from the reps or guessed from priors" gate.

Usage (GPU container):
    python scripts/eval_blob_sft.py +experiment=blob_sft_v1 \
        "+eval.checkpoint=/workspace/checkpoints_fast/<final>" \
        training.max_eval_samples=256 \
        "+eval.output_json=/workspace/checkpoints_fast/eval_reports_blob_sft_v1.json"
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from bgkit.utils.logging import setup_logging


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    if cfg.training.phase != "blob_sft":
        raise ValueError(
            f"eval_blob_sft.py expects training.phase=blob_sft, got {cfg.training.phase!r}"
        )
    from bgkit.training.blob_sft_trainer import BlobSFTTrainer

    trainer = BlobSFTTrainer(cfg)
    trainer.setup()
    eval_cfg = cfg.get("eval", {}) or {}
    ckpt = eval_cfg.get("checkpoint")
    if ckpt:
        trainer.load_checkpoint(Path(str(ckpt)))
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2, default=str))
    out = eval_cfg.get("output_json")
    if out:
        Path(str(out)).parent.mkdir(parents=True, exist_ok=True)
        Path(str(out)).write_text(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
