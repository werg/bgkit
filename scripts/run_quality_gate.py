#!/usr/bin/env python
"""Phase transition quality gate check.

Usage:
    python scripts/run_quality_gate.py \
        +eval.checkpoint=checkpoints/step_5000 \
        +eval.step=1

This script computes reconstruction_loss and parse_success_rate.
Only steps "1" and "2" (default: "1") are currently supported.

Steps "3a", "3b", and "all" require text_repro_loss and/or
description_quality (target LM inference + description data from
Phase 1 Step 3+) and will be rejected until the corresponding eval
paths are implemented.

Exit code 0 if all checked gates pass, 1 if any fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import structlog
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from bgkit.data.collators import collate_chat_repro
from bgkit.data.datasets.auto_repro_dataset import AutoReproDataset
from bgkit.data.datasets.chat_repro_dataset import ChatReproDataset
from bgkit.data.samplers import TokenBudgetBatchSampler
from bgkit.eval.metrics.reconstruction import parse_success_rate
from bgkit.eval.quality_gates import check_phase1_gates
from bgkit.models.bgkit_compressor import BgKITCompressor
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.training.checkpointing import load_checkpoint
from bgkit.training.objectives.data_reconstruction import data_reconstruction_loss
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

    step = str(eval_cfg.get("step", "1"))
    supported_steps = {"1", "2"}
    if step not in supported_steps:
        logger.error(
            "unsupported_step",
            step=step,
            supported=sorted(supported_steps),
            hint="Steps 3a/3b/all require text_repro_loss and description_quality, "
            "which this script does not yet compute.",
        )
        sys.exit(1)

    max_generation_samples = eval_cfg.get("generation_samples", 200)
    max_eval_examples = eval_cfg.get("max_eval_examples", 0)  # 0 = all

    # Load models
    bgkit_cfg = cfg.model.bgkit
    hidden_dim = bgkit_cfg.get("hidden_dim", 1024)
    bgkit_backbone = AutoModel.from_pretrained(
        bgkit_cfg.backbone_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        revision=bgkit_cfg.get("backbone_revision"),
        attn_implementation="sdpa",
    )
    bgkit_model = BgKITCompressor(bgkit_backbone, hidden_dim=hidden_dim)
    bgkit_model.to(device)
    bgkit_model.requires_grad_(False)
    bgkit_model.eval()

    decoder_cfg = cfg.model.decoder
    decoder_backbone = AutoModelForCausalLM.from_pretrained(
        decoder_cfg.backbone_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        revision=decoder_cfg.get("backbone_revision"),
        attn_implementation="sdpa",
    )
    decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=hidden_dim)
    decoder.to(device)
    decoder.eval()

    # Load checkpoint weights
    _, state_dicts = load_checkpoint(Path(checkpoint_path))
    if "bgkit_model" in state_dicts:
        bgkit_model.load_state_dict(state_dicts["bgkit_model"])
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
    variant_bank_path = cfg.data.get(
        "variant_bank_path", "data/prompt_variants/file_read_repro.json",
    )
    inner_dataset = AutoReproDataset(data_dir, max_seq_len=max_seq_len)
    chat_dataset = ChatReproDataset(
        inner_dataset, tokenizer=tokenizer, variant_bank_path=variant_bank_path,
    )

    max_batch_tokens = cfg.get("batch_size", 32) * 2048  # fallback from top-level config
    if cfg.get("training"):
        max_batch_tokens = cfg.training.get("max_batch_tokens", max_batch_tokens)
    eval_lengths = chat_dataset.lengths
    eval_sampler = TokenBudgetBatchSampler(eval_lengths, max_batch_tokens, shuffle=False)
    eval_dataloader = DataLoader(
        chat_dataset, batch_sampler=eval_sampler, collate_fn=collate_chat_repro,
    )

    # Compute metrics: loss over full eval data, generation on a subset
    total_loss = 0.0
    total_tokens = 0.0
    generated_texts: list[str] = []
    generated_languages: list[str] = []
    gen_examples_seen = 0
    eval_examples_seen = 0

    suffix_ids = chat_dataset.suffix_ids.to(device)

    with torch.no_grad():
        for batch in eval_dataloader:
            if max_eval_examples > 0 and eval_examples_seen >= max_eval_examples:
                break

            token_ids = batch["token_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            content_token_ids = batch["content_token_ids"].to(device)
            content_attention_mask = batch["content_attention_mask"].to(device)
            compression_prompt_ids = batch["compression_prompt_ids"].to(device)
            compression_prompt_mask = batch["compression_prompt_mask"].to(device)

            bgkit_embed = bgkit_model.backbone.get_input_embeddings()
            content_emb = bgkit_embed(content_token_ids)
            prompt_emb = bgkit_embed(compression_prompt_ids)

            bgkit_out = bgkit_model.backbone(
                inputs_embeds=torch.cat([
                    prompt_emb,
                    bgkit_model.prompt_separator_embedding.unsqueeze(0).unsqueeze(0).expand(
                        content_emb.size(0), 1, -1,
                    ),
                    content_emb,
                ], dim=1),
                attention_mask=torch.cat([
                    compression_prompt_mask,
                    torch.ones(content_emb.size(0), 1, dtype=torch.bool, device=device),
                    content_attention_mask,
                ], dim=1),
            )
            prompt_len = compression_prompt_ids.size(1) + 1
            survivors = bgkit_out.last_hidden_state[:, prompt_len:, :]

            # Teacher-forced loss (all batches)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = decoder(
                    survivor_embeddings=survivors,
                    target_ids=token_ids,
                    target_attention_mask=attention_mask,
                    survivor_attention_mask=content_attention_mask,
                )
                loss = data_reconstruction_loss(
                    logits, token_ids, attention_mask, loss_mask=loss_mask,
                )
            batch_tokens = loss_mask[:, 1:].sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            eval_examples_seen += token_ids.size(0)

            # Generation for parse success rate (capped subset)
            if gen_examples_seen < max_generation_samples:
                prefix_ids = batch["prefix_ids"].to(device)
                prefix_attention_mask = batch["prefix_attention_mask"].to(device)
                gen_output = decoder.generate(
                    survivor_embeddings=survivors,
                    survivor_attention_mask=content_attention_mask,
                    prefix_ids=prefix_ids,
                    prefix_attention_mask=prefix_attention_mask,
                    suffix_ids=suffix_ids,
                    tokenizer=tokenizer,
                    max_new_tokens=2048,
                    temperature=0.0,
                )
                generated_texts.extend(gen_output.content_text)
                generated_languages.extend(batch["languages"])
                gen_examples_seen += token_ids.size(0)

    reconstruction_loss_val = total_loss / max(total_tokens, 1)
    psr = (
        parse_success_rate(generated_texts, languages=generated_languages)
        if generated_texts else 0.0
    )

    # Load gate thresholds from config
    gate_cfg = cfg.get("eval", {}).get("quality_gates", {}).get("phase1", {})

    gates = check_phase1_gates(
        step=step,
        reconstruction_loss=reconstruction_loss_val,
        parse_success_rate=psr,
        max_reconstruction_loss=gate_cfg.get("max_reconstruction_loss", 2.0),
        min_parse_success_rate=gate_cfg.get("min_parse_success_rate", 0.8),
        max_text_repro_loss=gate_cfg.get("max_text_repro_loss", 3.0),
        min_description_quality=gate_cfg.get("min_description_quality", 0.5),
    )

    # Print results
    print("\n=== Quality Gate Results ===")
    print(f"Step: {step}")
    print(f"Reconstruction loss: {reconstruction_loss_val:.4f}")
    print(f"Parse success rate: {psr:.1%}")
    print()

    any_failed = False
    for gate in gates:
        if gate.skipped:
            print(f"  [{' SKIP '}] {gate.gate_name}")
        elif gate.passed:
            print(f"  [  OK  ] {gate.gate_name}: {gate.message}")
        else:
            print(f"  [ FAIL ] {gate.gate_name}: {gate.message}")
            any_failed = True

    if any_failed:
        print("\nResult: FAILED — one or more gates did not pass.")
        sys.exit(1)
    else:
        print("\nResult: PASSED — all checked gates passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
