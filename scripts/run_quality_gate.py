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
from transformers import AutoModelForCausalLM, AutoTokenizer

from bgkit.data.collators import collate_chat_repro
from bgkit.data.datasets.chat_repro_dataset import ChatReproDataset
from bgkit.data.datasets.mmap_token_dataset import MmapTokenDataset
from bgkit.data.samplers import PackedTokenBudgetSampler
from bgkit.eval.ablations import _build_decoder_segments_from_batch
from bgkit.eval.metrics.reconstruction import parse_success_rate
from bgkit.eval.quality_gates import check_phase1_gates
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
    attention_impl = resolve_attention_implementation(
        cfg.compute.get("attention_implementation", "auto")
    )

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

    max_batch_tokens = cfg.get("batch_size", 32) * 2048  # fallback from top-level config
    if cfg.get("training"):
        max_batch_tokens = cfg.training.get("max_batch_tokens", max_batch_tokens)
    eval_lengths = chat_dataset.lengths
    eval_sampler = PackedTokenBudgetSampler(
        dataset=None,
        lengths=eval_lengths,
        max_batch_tokens=max_batch_tokens,
        shuffle=False,
    )
    eval_dataloader = DataLoader(
        chat_dataset, batch_sampler=eval_sampler, collate_fn=collate_chat_repro,
    )

    # Compute metrics: loss over full eval data, generation on a subset.
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

            content_token_ids = batch["content_token_ids"].to(device)
            content_cu = batch["content_cu_seqlens"].to(device)
            content_position_ids = batch["content_position_ids"].to(device)
            prompt_ids = batch["compression_prompt_ids"].to(device)
            prompt_cu = batch["compression_prompt_cu_seqlens"].to(device)
            prompt_position_ids = batch["compression_prompt_position_ids"].to(device)
            loss_mask_flat = batch["loss_mask"].to(device)
            batch_size = int(content_cu.shape[0]) - 1

            bgkit_embed = encoder.compressor.backbone.get_input_embeddings()
            content_emb = bgkit_embed(content_token_ids)
            prompt_emb = bgkit_embed(prompt_ids)

            enc_out = encoder(
                content_embeddings=content_emb,
                content_cu_seqlens=content_cu,
                content_position_ids=content_position_ids,
                prompt_embeddings=prompt_emb,
                prompt_cu_seqlens=prompt_cu,
                prompt_position_ids=prompt_position_ids,
                target_ratio=None,
            )
            survivors = enc_out.survivor_embeddings
            survivor_cu = enc_out.survivor_cu_seqlens

            # Teacher-forced packed decoder forward.
            prefix_list, suffix_list, flat_loss_mask = _build_decoder_segments_from_batch(
                batch, device, survivor_cu
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = decoder.forward_with_single_splice(
                    survivor_embeddings=survivors,
                    survivor_cu_seqlens=survivor_cu,
                    prefix_ids=prefix_list,
                    suffix_ids=suffix_list,
                    loss_mask=flat_loss_mask,
                )
            batch_tokens = loss_mask_flat.sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            eval_examples_seen += batch_size

            # Generation for parse success rate (capped subset).
            if gen_examples_seen < max_generation_samples:
                for b in range(batch_size):
                    if gen_examples_seen >= max_generation_samples:
                        break
                    k_start = int(survivor_cu[b].item())
                    k_end = int(survivor_cu[b + 1].item())
                    surv_b = survivors[k_start:k_end]
                    cu_b = torch.tensor([0, k_end - k_start], dtype=torch.int32, device=device)
                    gen_output = decoder.generate_with_single_splice(
                        survivor_embeddings=surv_b,
                        survivor_cu_seqlens=cu_b,
                        prefix_ids=prefix_list[b],
                        suffix_ids=suffix_ids,
                        tokenizer=tokenizer,
                        max_new_tokens=2048,
                        temperature=0.0,
                    )
                    generated_texts.extend(gen_output.content_text)
                    gen_examples_seen += 1
                if "languages" in batch:
                    generated_languages.extend(batch["languages"][:batch_size])

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
        print("\nResult: FAILED -- one or more gates did not pass.")
        sys.exit(1)
    else:
        print("\nResult: PASSED -- all checked gates passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
