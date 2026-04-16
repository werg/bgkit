#!/usr/bin/env python
"""Run mandatory ablation suite: survivors present vs zeroed vs noise.

Usage:
    python scripts/run_ablation.py \
        +eval.checkpoint=checkpoints/step_5000

Exit code 1 if ablation gap < min_ablation_gap from config.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import structlog
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from bgkit.data.collators import collate_chat_repro
from bgkit.data.datasets.chat_repro_dataset import ChatReproDataset
from bgkit.data.datasets.mmap_token_dataset import MmapTokenDataset
from bgkit.data.samplers import TokenBudgetBatchSampler
from bgkit.eval.ablations import (
    AblationCondition,
    compute_ablation_gap,
    run_ablation_suite,
)
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.training.checkpointing import load_checkpoint
from bgkit.utils.attention_backend import resolve_attention_implementation
from bgkit.utils.logging import setup_logging

logger = structlog.get_logger()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_cfg = cfg.get("eval", {})
    checkpoint_path = eval_cfg.get("checkpoint")
    if not checkpoint_path:
        logger.error("No checkpoint specified. Use +eval.checkpoint=<path>")
        sys.exit(1)

    ablation_cfg = cfg.get("eval", {}).get("ablation", {})
    max_examples = ablation_cfg.get("num_examples", 500)
    include_generation = ablation_cfg.get("include_generation_metrics", False)
    min_ablation_gap = cfg.get("eval", {}).get(
        "quality_gates", {},
    ).get("phase2", {}).get("min_ablation_gap", 0.10)
    attention_impl = resolve_attention_implementation(
        cfg.compute.get("attention_implementation", "auto")
    )

    # Parse configured conditions (e.g., ["present", "zeroed", "noise"])
    condition_names = ablation_cfg.get("conditions", None)
    if condition_names:
        name_to_condition = {c.value: c for c in AblationCondition}
        conditions = []
        for name in condition_names:
            if name not in name_to_condition:
                logger.warning("unknown_ablation_condition", name=name)
                continue
            conditions.append(name_to_condition[name])
    else:
        conditions = list(AblationCondition)

    if not conditions:
        logger.error("No valid ablation conditions configured. Check configs/eval/ablation.yaml")
        sys.exit(1)

    # Load models
    bgkit_cfg = cfg.model.bgkit
    hidden_dim = bgkit_cfg.get("hidden_dim", 1024)

    # Load checkpoint first — encoder architecture is auto-detected from keys
    _, state_dicts = load_checkpoint(Path(checkpoint_path))

    if "encoder" not in state_dicts:
        raise ValueError(
            f"Checkpoint missing 'encoder' key: {checkpoint_path}. "
            f"Found keys: {list(state_dicts.keys())}"
        )
    encoder = BgKITEncoder.from_pretrained_with_state_dict(
        bgkit_cfg.backbone_name,
        state_dicts["encoder"],
        hidden_dim=hidden_dim,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        revision=bgkit_cfg.get("backbone_revision"),
        attn_implementation=attention_impl,
    )
    encoder.to(device)
    encoder.requires_grad_(False)
    encoder.eval()

    decoder_cfg = cfg.model.decoder
    decoder_backbone = AutoModelForCausalLM.from_pretrained(
        decoder_cfg.backbone_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        revision=decoder_cfg.get("backbone_revision"),
        attn_implementation=attention_impl,
    )
    decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=hidden_dim)
    decoder.to(device)
    decoder.eval()
    decoder.load_state_dict(state_dicts["decoder"])

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        decoder_cfg.backbone_name,
        trust_remote_code=True,
        revision=decoder_cfg.get("backbone_revision"),
    )

    # Dataset
    data_dir = cfg.data.tokens.input_dir
    max_seq_len = cfg.data.tokens.get("max_seq_len", 8192)
    variant_bank_path = cfg.data.tokens.variant_bank_path
    inner_dataset = MmapTokenDataset(data_dir, max_seq_len=max_seq_len)
    chat_dataset = ChatReproDataset(
        inner_dataset, tokenizer=tokenizer, variant_bank_path=variant_bank_path,
    )

    max_batch_tokens = 65536
    if cfg.get("training"):
        max_batch_tokens = cfg.training.get("max_batch_tokens", max_batch_tokens)
    eval_lengths = chat_dataset.lengths
    eval_sampler = TokenBudgetBatchSampler(eval_lengths, max_batch_tokens, shuffle=False)
    eval_dataloader = DataLoader(
        chat_dataset, batch_sampler=eval_sampler, collate_fn=collate_chat_repro,
    )

    suffix_ids = chat_dataset.suffix_ids if include_generation else None

    # Run ablation suite
    results = run_ablation_suite(
        decoder=decoder,
        encoder=encoder,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        device=device,
        conditions=conditions,
        max_examples=max_examples,
        include_generation_metrics=include_generation,
        suffix_ids=suffix_ids,
    )

    # Compute and print gaps
    gaps = compute_ablation_gap(results)

    print("\n=== Ablation Results ===")
    for r in results:
        print(f"\n{r.condition.value}:")
        print(json.dumps(r.metrics, indent=2))

    print("\n=== Ablation Gaps ===")
    print(json.dumps(gaps, indent=2))

    # Optionally log to wandb
    if cfg.get("wandb", {}).get("enabled", False):
        try:
            import wandb

            wandb.init(
                project=cfg.wandb.get("project", "bgkit"),
                name=f"ablation-{Path(checkpoint_path).stem}",
                config=dict(cfg),
            )
            for r in results:
                wandb.log({f"ablation/{r.condition.value}/{k}": v for k, v in r.metrics.items()})
            wandb.log({f"ablation/{k}": v for k, v in gaps.items()})
            wandb.finish()
        except Exception:
            logger.warning("wandb_logging_failed", exc_info=True)

    # Check gap threshold
    failed = False
    for gap_name, gap_value in gaps.items():
        if gap_value < min_ablation_gap:
            logger.error(
                "ablation_gap_too_small",
                gap=gap_name,
                value=gap_value,
                threshold=min_ablation_gap,
            )
            failed = True

    if failed:
        print(f"\nFAILED: Ablation gap below threshold ({min_ablation_gap})")
        sys.exit(1)
    else:
        print(f"\nPASSED: All ablation gaps above threshold ({min_ablation_gap})")
        sys.exit(0)


if __name__ == "__main__":
    main()
