#!/usr/bin/env python
"""Topic embedding ablation study.

Ablation experiments:
  (a) Compressed context + topic embeddings
  (b) Compressed context only
  (c) Topic embeddings only
  (d) Neither

Monitors per-tag embedding norms for divergence/stagnation detection.

Usage:
    python scripts/eval_topic_embeddings.py \
        +eval.checkpoint=checkpoints/phase2_step3_best \
        +eval.output_dir=eval_reports/topic_embeddings
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import structlog
import torch
from omegaconf import DictConfig

from bgkit.utils.logging import setup_logging

logger = structlog.get_logger()

_CONDITIONS = {
    "compressed_plus_topics": {"use_compressed": True, "use_topics": True},
    "compressed_only": {"use_compressed": True, "use_topics": False},
    "topics_only": {"use_compressed": False, "use_topics": True},
    "neither": {"use_compressed": False, "use_topics": False},
}


def _load_trainer(cfg: DictConfig):
    """Load a KRTrainer with checkpoint weights."""
    from bgkit.training.phase2.kr_trainer import KRTrainer

    trainer = KRTrainer(cfg)
    trainer.setup()

    eval_cfg = cfg.get("eval", {})
    checkpoint_path = eval_cfg.get("checkpoint")
    if checkpoint_path:
        from bgkit.training.checkpointing import load_checkpoint

        logger.info("loading_checkpoint", path=str(checkpoint_path))
        _metadata, state_dicts = load_checkpoint(Path(str(checkpoint_path)))
        model_state = state_dicts.get("model", {})
        if model_state:
            trainer.model.load_state_dict(model_state, strict=False)
            logger.info("checkpoint_loaded", keys=len(model_state))

    return trainer


def _evaluate_condition(
    trainer,
    condition_cfg: dict[str, bool],
    max_samples: int = 500,
) -> dict[str, float]:
    """Evaluate the trainer under a specific ablation condition.

    Temporarily disables topic embeddings and/or replaces compressed context
    with zeros, then runs the standard evaluate() loop.
    """
    use_compressed = condition_cfg["use_compressed"]
    use_topics = condition_cfg["use_topics"]

    # Save originals for restoration
    original_topic_embeddings = trainer.topic_embeddings
    original_compose_prompt = trainer._compose_prompt

    def _ablated_compose_prompt(batch):
        """Compose prompt with ablation modifications applied."""
        # Get the full prompt (compressed context + topic embeddings)
        prompt, mask = original_compose_prompt(batch)

        if not use_compressed and not use_topics:
            # Neither: zero out everything
            prompt = torch.zeros_like(prompt)
        elif not use_compressed and use_topics:
            # Topics only: zero out the compressed context portion,
            # keep topic embedding positions
            if original_topic_embeddings is not None:
                # Topic embeddings are appended at the end of the prompt.
                # Determine how many positions are from topics.
                topic_positions = 0
                expanded = [
                    original_topic_embeddings.taxonomy.expand_tags(tags)
                    for tags in batch["tags"]
                ]
                for tags in expanded:
                    n_tags = sum(
                        1
                        for tag in tags
                        if original_topic_embeddings._key(tag)
                        in original_topic_embeddings.embeddings
                    )
                    topic_positions = max(
                        topic_positions,
                        n_tags * original_topic_embeddings.positions_per_tag,
                    )
                # Zero out everything except the last topic_positions
                if topic_positions > 0 and prompt.size(1) > topic_positions:
                    compressed_len = prompt.size(1) - topic_positions
                    prompt[:, :compressed_len] = 0.0
                else:
                    # No topic positions found; zero everything
                    prompt = torch.zeros_like(prompt)
            else:
                # No topic embeddings module at all; zero everything
                prompt = torch.zeros_like(prompt)
        elif use_compressed and not use_topics and original_topic_embeddings is not None:
            # Compressed only: zero out topic embedding positions at the end
            expanded = [
                original_topic_embeddings.taxonomy.expand_tags(tags)
                for tags in batch["tags"]
            ]
            topic_positions = 0
            for tags in expanded:
                n_tags = sum(
                    1
                    for tag in tags
                    if original_topic_embeddings._key(tag)
                    in original_topic_embeddings.embeddings
                )
                topic_positions = max(
                    topic_positions,
                    n_tags * original_topic_embeddings.positions_per_tag,
                )
            # Zero out the topic embedding positions at the end
            if topic_positions > 0 and prompt.size(1) > topic_positions:
                prompt[:, -topic_positions:] = 0.0
        # else: use_compressed and use_topics -- no modification

        return prompt, mask

    # Patch the trainer's compose method for this condition
    trainer._compose_prompt = _ablated_compose_prompt

    try:
        # Run the trainer's built-in evaluate()
        metrics = trainer.evaluate()
    finally:
        # Restore original method
        trainer._compose_prompt = original_compose_prompt

    return metrics


def _analyze_topic_norms(trainer) -> dict[str, float]:
    """Extract per-tag embedding norms from a live trainer's topic embeddings."""
    norms = {}
    if trainer.topic_embeddings is None:
        return norms

    for tag in trainer.topic_embeddings.taxonomy.tags:
        key = trainer.topic_embeddings._key(tag)
        if key in trainer.topic_embeddings.embeddings:
            param = trainer.topic_embeddings.embeddings[key]
            norms[tag] = float(param.data.norm().item())

    return norms


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()

    eval_cfg = cfg.get("eval", {})
    checkpoint_path = eval_cfg.get("checkpoint")
    if not checkpoint_path:
        logger.error("No checkpoint specified. Use +eval.checkpoint=<path>")
        sys.exit(1)

    output_dir = Path(eval_cfg.get("output_dir", "eval_reports/topic_embeddings"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trainer with checkpoint
    logger.info("loading_trainer")
    trainer = _load_trainer(cfg)

    max_samples = int(eval_cfg.get("max_eval_samples", 500))

    report = {
        "checkpoint": str(checkpoint_path),
        "conditions": {},
        "has_topic_embeddings": trainer.topic_embeddings is not None,
    }

    # Run ablation for each condition
    for condition_name, condition_cfg in _CONDITIONS.items():
        logger.info("topic_ablation", condition=condition_name, **condition_cfg)
        try:
            metrics = _evaluate_condition(
                trainer, condition_cfg, max_samples=max_samples,
            )
            report["conditions"][condition_name] = {
                "config": condition_cfg,
                "metrics": metrics,
            }
            logger.info(
                "topic_ablation_result",
                condition=condition_name,
                loss=metrics.get("eval/loss"),
                accuracy=metrics.get("eval/answer_token_accuracy"),
            )
        except Exception as exc:
            logger.warning(
                "topic_ablation_failed", condition=condition_name, error=str(exc),
            )
            report["conditions"][condition_name] = {
                "config": condition_cfg,
                "status": "error",
                "error": str(exc),
            }

    # Analyze per-tag norms from the live model
    logger.info("analyzing_topic_norms")
    norms = _analyze_topic_norms(trainer)
    report["tag_norms"] = norms
    if norms:
        norm_values = list(norms.values())
        report["tag_norm_stats"] = {
            "count": len(norms),
            "mean": sum(norm_values) / len(norm_values),
            "max": max(norm_values),
            "min": min(norm_values),
            "std": float(torch.tensor(norm_values).std().item()) if len(norm_values) > 1 else 0.0,
        }
    else:
        report["tag_norm_stats"] = {"count": 0}

    # Compute deltas between conditions for quick comparison
    conditions = report["conditions"]
    if (
        "compressed_plus_topics" in conditions
        and "compressed_only" in conditions
        and "metrics" in conditions["compressed_plus_topics"]
        and "metrics" in conditions["compressed_only"]
    ):
        both_m = conditions["compressed_plus_topics"]["metrics"]
        comp_m = conditions["compressed_only"]["metrics"]
        report["topic_delta"] = {
            "loss_delta": (
                both_m.get("eval/loss", 0) - comp_m.get("eval/loss", 0)
            ),
            "accuracy_delta": (
                both_m.get("eval/answer_token_accuracy", 0)
                - comp_m.get("eval/answer_token_accuracy", 0)
            ),
            "interpretation": (
                "negative loss_delta = topics help; "
                "positive accuracy_delta = topics help"
            ),
        }

    # Write report
    report_path = output_dir / "topic_embedding_ablation.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("topic_eval_complete", report_path=str(report_path))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
