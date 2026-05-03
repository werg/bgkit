#!/usr/bin/env python
"""Run with-vs-without-bgkit ablation on QA-conditioned data.

Mirrors scripts/run_ablation.py but uses QAChatReproDataset so the
loss is measured on answer tokens (the actual signal step 4 trains
against), not on reconstruction targets.

Usage:
    python scripts/run_ablation_qa.py +eval.checkpoint=<path>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import structlog
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bgkit.data.collators import collate_chat_repro
from bgkit.data.datasets.mmap_token_dataset import MmapTokenDataset
from bgkit.data.datasets.qa_chat_repro_dataset import QAChatReproDataset
from bgkit.data.datasets.qa_conditioned_dataset import MmapQAConditionedDataset
from bgkit.data.samplers import PackedTokenBudgetSampler
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

    ablation_cfg = eval_cfg.get("ablation", {})
    max_examples = ablation_cfg.get("num_examples", 200)
    attention_impl = resolve_attention_implementation(
        cfg.compute.get("attention_implementation", "auto")
    )

    bgkit_cfg = cfg.model.bgkit
    hidden_dim = bgkit_cfg.get("hidden_dim", 1024)

    _, state_dicts = load_checkpoint(Path(checkpoint_path))
    if "encoder" not in state_dicts:
        raise ValueError(f"Checkpoint missing 'encoder': {checkpoint_path}")
    encoder = BgKITEncoder.from_pretrained_with_state_dict(
        bgkit_cfg.backbone_name,
        state_dicts["encoder"],
        hidden_dim=hidden_dim,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        revision=bgkit_cfg.get("backbone_revision"),
        attn_implementation=attention_impl,
    )
    encoder.to(device).requires_grad_(False).eval()

    decoder_cfg = cfg.model.decoder
    decoder_backbone = AutoModelForCausalLM.from_pretrained(
        decoder_cfg.backbone_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        revision=decoder_cfg.get("backbone_revision"),
        attn_implementation=attention_impl,
    )
    decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=hidden_dim)
    decoder.to(device).eval()
    decoder_state = state_dicts.get("decoder_merged") or state_dicts.get("decoder")
    if decoder_state is None:
        raise ValueError(f"Checkpoint missing 'decoder'/'decoder_merged'")
    decoder.load_state_dict(decoder_state)

    tokenizer = AutoTokenizer.from_pretrained(
        decoder_cfg.backbone_name,
        trust_remote_code=True,
        revision=decoder_cfg.get("backbone_revision"),
    )

    # QA-conditioned dataset (same construction as DecoderInitTrainer).
    data_dir = cfg.data.tokens.input_dir
    qa_data_dir = cfg.data.qa_data_dir
    max_seq_len = cfg.data.tokens.get("max_seq_len", 8192)
    inner_dataset = MmapTokenDataset(
        data_dir, max_seq_len=max_seq_len, include_metadata=True,
    )
    qa_mmap = MmapQAConditionedDataset(qa_data_dir, max_seq_len=2048)
    qa_full = QAChatReproDataset(
        qa_mmap, inner_dataset, tokenizer, seed=42,
    )

    # Subset to first `max_examples` to keep eval fast and deterministic.
    n = min(max_examples * 4, len(qa_full))  # headroom for sampler packing
    eval_ds = Subset(qa_full, list(range(n)))

    import numpy as np
    eval_lengths = qa_full.lengths[:n]
    eval_sampler = PackedTokenBudgetSampler(
        dataset=None, lengths=eval_lengths,
        max_batch_tokens=16384, shuffle=False,
    )
    eval_dataloader = DataLoader(
        eval_ds, batch_sampler=eval_sampler, collate_fn=collate_chat_repro,
    )

    results = run_ablation_suite(
        decoder=decoder, encoder=encoder,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer, device=device,
        conditions=list(AblationCondition),
        max_examples=max_examples,
        include_generation_metrics=False,
    )

    logger.info("ablation_done", n_results=len(results))
    for r in results:
        logger.info(
            "ablation_result",
            condition=r.condition.value,
            loss=r.loss,
            perplexity=r.perplexity,
        )

    gaps = compute_ablation_gap(results)
    logger.info("ablation_gaps", **gaps)

    out = {
        "checkpoint": checkpoint_path,
        "max_examples": max_examples,
        "results": [
            {
                "condition": r.condition.value,
                "loss": float(r.loss),
                "perplexity": float(r.perplexity),
            }
            for r in results
        ],
        "gaps": {k: float(v) for k, v in gaps.items()},
    }
    print("\n" + json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
