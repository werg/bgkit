#!/usr/bin/env python
"""Post-Phase-1 baseline evaluation.

Orchestrates Eval 1: compression curves, ablations, RAG baseline comparison,
per-language parse success, embedding health, and description quality.

Usage:
    python scripts/eval_phase1.py \
        +eval.checkpoint=checkpoints/phase1_step5_best \
        +eval.output_dir=eval_reports
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
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
from bgkit.eval.baselines.rag_baseline import RAGBaseline
from bgkit.eval.metrics.compression import compute_compression_curve
from bgkit.eval.metrics.description import description_quality_score
from bgkit.eval.metrics.embedding_health import embedding_drift_metrics
from bgkit.eval.metrics.reconstruction import parse_success_rate
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.training.checkpointing import load_checkpoint
from bgkit.utils.attention_backend import resolve_attention_implementation
from bgkit.utils.logging import setup_logging

logger = structlog.get_logger()


def _load_models(cfg: DictConfig, device: torch.device):
    """Load encoder, decoder, and ICE from checkpoint."""
    checkpoint_path = cfg.eval.checkpoint
    logger.info("loading_checkpoint", path=checkpoint_path)

    bgkit_cfg = cfg.model.bgkit
    hidden_dim = bgkit_cfg.get("hidden_dim", 1024)
    attention_impl = resolve_attention_implementation(
        cfg.compute.get("attention_implementation", "auto")
    )

    _metadata, state_dicts = load_checkpoint(Path(str(checkpoint_path)))

    # Load encoder
    if "encoder" in state_dicts:
        encoder = BgKITEncoder.from_pretrained_with_state_dict(
            bgkit_cfg.backbone_name,
            state_dicts["encoder"],
            hidden_dim=hidden_dim,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            revision=bgkit_cfg.get("backbone_revision"),
            attn_implementation=attention_impl,
        )
    elif "model" in state_dicts:
        model_state = state_dicts["model"]
        encoder_state = {
            k.replace("encoder.", "", 1): v
            for k, v in model_state.items() if k.startswith("encoder.")
        }
        if encoder_state:
            encoder = BgKITEncoder.from_pretrained_with_state_dict(
                bgkit_cfg.backbone_name,
                encoder_state,
                hidden_dim=hidden_dim,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                revision=bgkit_cfg.get("backbone_revision"),
                attn_implementation=attention_impl,
            )
        else:
            encoder = BgKITEncoder.from_pretrained(
                bgkit_cfg.backbone_name,
                hidden_dim=hidden_dim,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                revision=bgkit_cfg.get("backbone_revision"),
                attn_implementation=attention_impl,
            )
    else:
        raise ValueError(
            f"Checkpoint missing encoder state: {checkpoint_path}. "
            f"Found keys: {list(state_dicts.keys())}"
        )
    encoder.to(device).eval()
    encoder.requires_grad_(False)

    # Load decoder
    decoder_cfg = cfg.model.decoder
    decoder_backbone = AutoModelForCausalLM.from_pretrained(
        decoder_cfg.backbone_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        revision=decoder_cfg.get("backbone_revision"),
        attn_implementation=attention_impl,
    )
    decoder = ReconstructionDecoder(
        decoder_backbone,
        hidden_dim=hidden_dim,
    )
    if "decoder" in state_dicts:
        decoder.load_state_dict(state_dicts["decoder"], strict=False)
    elif "model" in state_dicts:
        decoder_state = {
            k.replace("decoder.", "", 1): v
            for k, v in state_dicts["model"].items() if k.startswith("decoder.")
        }
        if decoder_state:
            decoder.load_state_dict(decoder_state, strict=False)
    decoder.to(device).eval()

    return encoder, decoder


def _eval_at_ratio(
    encoder, decoder, dataloader, device, ratio: float, tokenizer=None,
) -> dict[str, float]:
    """Run evaluation at a specific compression ratio."""
    import math

    total_loss = 0.0
    total_batches = 0
    all_pred_texts: list[str] = []
    all_ref_texts: list[str] = []
    all_languages: list[str] = []

    bgkit_embed = encoder.compressor.backbone.get_input_embeddings()

    with torch.no_grad():
        for batch in dataloader:
            content_ids = batch["content_token_ids"].to(device)
            content_mask = batch["content_attention_mask"].to(device)
            target_ids = batch["token_ids"].to(device)
            target_mask = batch["attention_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            compression_prompt_ids = batch["compression_prompt_ids"].to(device)
            compression_prompt_mask = batch["compression_prompt_mask"].to(device)

            # Encode content
            content_emb = bgkit_embed(content_ids)
            prompt_emb = bgkit_embed(compression_prompt_ids)

            # Build survivor mask for the given ratio
            batch_size, _seq_len = content_ids.shape
            survivor_masks = torch.zeros_like(content_mask, dtype=torch.bool)
            for i in range(batch_size):
                length = int(content_mask[i].sum())
                keep = max(1, math.ceil(length * ratio))
                # Uniform subsampling (no ICE scores needed for eval)
                indices = torch.linspace(
                    0, length - 1, steps=keep, device=device,
                ).round().long()
                survivor_masks[i, indices] = True

            enc_out = encoder(
                input_embeddings=content_emb,
                survivor_mask=survivor_masks,
                attention_mask=content_mask,
                prompt_embeddings=prompt_emb,
                prompt_attention_mask=compression_prompt_mask,
            )
            survivors = enc_out.survivor_embeddings

            # Compute loss
            loss = decoder.forward_with_loss(
                survivors,
                target_ids,
                target_mask,
                content_mask,
                loss_mask=loss_mask,
            )
            total_loss += loss.item()
            total_batches += 1

            # Generate predictions for parse success / description quality
            if tokenizer is not None and len(all_pred_texts) < 200:
                logits = decoder(survivors, target_ids, target_mask, content_mask)
                predictions = logits[:, :-1].argmax(dim=-1)
                for i in range(predictions.size(0)):
                    mask_i = loss_mask[i, 1:]
                    if mask_i.any():
                        pred_tokens = predictions[i][mask_i.bool()].cpu().tolist()
                        tgt_tokens = target_ids[i, 1:][mask_i.bool()].cpu().tolist()
                        all_pred_texts.append(
                            tokenizer.decode(pred_tokens, skip_special_tokens=True),
                        )
                        all_ref_texts.append(
                            tokenizer.decode(tgt_tokens, skip_special_tokens=True),
                        )
                if "languages" in batch:
                    all_languages.extend(batch["languages"])

    avg_loss = total_loss / max(total_batches, 1)

    parse_rate = 0.0
    desc_quality = 0.0
    if all_pred_texts:
        import contextlib

        with contextlib.suppress(Exception):
            parse_rate = parse_success_rate(
                all_pred_texts,
                languages=all_languages[:len(all_pred_texts)] if all_languages else None,
            )
        with contextlib.suppress(Exception):
            desc_quality = description_quality_score(all_pred_texts, all_ref_texts)

    return {
        "reconstruction_loss": avg_loss,
        "parse_success_rate": parse_rate,
        "description_quality": desc_quality,
    }


def _run_compression_curve(encoder, decoder, dataloader, device, tokenizer) -> list[dict]:
    """Evaluate quality across compression ratios."""
    ratios = [0.50, 0.25, 0.10, 0.05, 0.02]

    def eval_fn(ratio: float) -> dict[str, float]:
        return _eval_at_ratio(encoder, decoder, dataloader, device, ratio, tokenizer)

    points = compute_compression_curve(eval_fn, ratios=ratios)
    return [asdict(p) for p in points]


def _run_embedding_health(encoder, dataloader, device) -> dict[str, float]:
    """Check embedding drift and health."""
    all_embeddings = []
    bgkit_embed = encoder.compressor.backbone.get_input_embeddings()

    with torch.no_grad():
        for batch in dataloader:
            content_ids = batch["content_token_ids"].to(device)
            content_mask = batch["content_attention_mask"].to(device)

            content_emb = bgkit_embed(content_ids)
            enc_out = encoder(
                input_embeddings=content_emb,
                survivor_mask=None,
                attention_mask=content_mask,
            )
            # Collect uncompressed embeddings (full pass, no survivor selection)
            flat = enc_out.survivor_embeddings[content_mask].detach().cpu()
            all_embeddings.append(flat)
            if sum(e.size(0) for e in all_embeddings) >= 512:
                break

    if not all_embeddings:
        return {}
    embeddings = torch.cat(all_embeddings, dim=0)[:512]
    token_emb = bgkit_embed.weight.detach().cpu()
    return embedding_drift_metrics(embeddings, token_emb)


def _run_rag_baseline(
    eval_cfg: DictConfig,
    encoder,
    decoder,
    tokenizer,
    dataloader,
    device,
) -> dict[str, float]:
    """Run RAG baseline comparison.

    Indexes repo files from eval samples, retrieves context with RAG,
    feeds to the decoder, and compares loss against BgKIT-compressed context.
    """
    rag = RAGBaseline(
        embedding_model_name=eval_cfg.get("rag_embedding_model", "all-MiniLM-L6-v2"),
        reranker_model_name=eval_cfg.get("rag_reranker_model", None),
    )
    top_k = int(eval_cfg.get("rag_top_k", 5))
    max_samples = int(eval_cfg.get("rag_max_samples", 100))

    # Collect files from the eval dataset for indexing
    files: dict[str, str] = {}
    batch_data: list[dict] = []
    samples_seen = 0

    with torch.no_grad():
        for batch in dataloader:
            if samples_seen >= max_samples:
                break
            content_ids = batch["content_token_ids"]
            batch_size = content_ids.size(0)
            for i in range(batch_size):
                if samples_seen >= max_samples:
                    break
                mask_i = batch["content_attention_mask"][i].bool()
                ids_i = content_ids[i][mask_i].tolist()
                text = tokenizer.decode(ids_i, skip_special_tokens=True)
                file_key = f"sample_{samples_seen}"
                files[file_key] = text
                batch_data.append({
                    "batch": batch,
                    "sample_idx": i,
                    "file_key": file_key,
                })
                samples_seen += 1

    if not files:
        return {"rag_baseline": "no_eval_data"}

    # Index all eval files
    rag.index_repository(files)

    # For each sample: retrieve RAG context, feed to decoder, compute loss
    bgkit_embed = encoder.compressor.backbone.get_input_embeddings()
    rag_total_loss = 0.0
    bgkit_total_loss = 0.0
    count = 0

    with torch.no_grad():
        for entry in batch_data:
            batch = entry["batch"]
            i = entry["sample_idx"]

            target_ids = batch["token_ids"][i : i + 1].to(device)
            target_mask = batch["attention_mask"][i : i + 1].to(device)
            loss_mask = batch["loss_mask"][i : i + 1].to(device)

            # Decode the file content as the query for retrieval
            content_mask_i = batch["content_attention_mask"][i].bool()
            content_text = tokenizer.decode(
                batch["content_token_ids"][i][content_mask_i].tolist(),
                skip_special_tokens=True,
            )

            # RAG: retrieve and tokenize context
            rag_context = rag.retrieve_text(
                content_text[:200],  # Use first ~200 chars as query
                top_k=top_k,
            )
            rag_ids = tokenizer.encode(
                rag_context, return_tensors="pt", truncation=True, max_length=2048,
            ).to(device)
            rag_emb = decoder.backbone.get_input_embeddings()(rag_ids)
            rag_mask = torch.ones(1, rag_emb.size(1), dtype=torch.bool, device=device)

            rag_loss = decoder.forward_with_loss(
                rag_emb, target_ids, target_mask, rag_mask, loss_mask=loss_mask,
            )
            rag_total_loss += rag_loss.item()

            # BgKIT: compress and use survivors
            content_ids_dev = batch["content_token_ids"][i : i + 1].to(device)
            content_mask_dev = batch["content_attention_mask"][i : i + 1].to(device)
            prompt_ids = batch["compression_prompt_ids"][i : i + 1].to(device)
            prompt_mask = batch["compression_prompt_mask"][i : i + 1].to(device)

            content_emb = bgkit_embed(content_ids_dev)
            prompt_emb = bgkit_embed(prompt_ids)

            enc_out = encoder(
                input_embeddings=content_emb,
                survivor_mask=None,
                attention_mask=content_mask_dev,
                prompt_embeddings=prompt_emb,
                prompt_attention_mask=prompt_mask,
            )
            survivors = enc_out.survivor_embeddings

            bgkit_loss = decoder.forward_with_loss(
                survivors, target_ids, target_mask, content_mask_dev, loss_mask=loss_mask,
            )
            bgkit_total_loss += bgkit_loss.item()
            count += 1

    if count == 0:
        return {"rag_baseline": "no_samples_processed"}

    rag_avg = rag_total_loss / count
    bgkit_avg = bgkit_total_loss / count

    return {
        "rag_loss": rag_avg,
        "bgkit_loss": bgkit_avg,
        "delta_loss": rag_avg - bgkit_avg,
        "bgkit_wins": float(bgkit_avg < rag_avg),
        "n_samples": float(count),
        "embedding_model": rag.embedding_model_name,
        "reranker_model": rag.reranker_model_name or "none",
        "rag_top_k": float(top_k),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_cfg = cfg.get("eval", {})
    checkpoint_path = eval_cfg.get("checkpoint")
    if not checkpoint_path:
        logger.error("No checkpoint specified. Use +eval.checkpoint=<path>")
        sys.exit(1)

    output_dir = Path(eval_cfg.get("output_dir", "eval_reports"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    encoder, decoder = _load_models(cfg, device)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.decoder.backbone_name,
        trust_remote_code=True,
        revision=cfg.model.decoder.get("backbone_revision"),
    )

    # Load eval dataset
    data_cfg = cfg.data.tokens
    token_dataset = MmapTokenDataset(
        str(data_cfg.input_dir),
        max_seq_len=int(data_cfg.get("max_seq_len", 8192)),
    )
    eval_dataset = ChatReproDataset(
        token_dataset,
        tokenizer=tokenizer,
        variant_bank_path=data_cfg.variant_bank_path,
    )

    # Use a subset for eval
    max_eval = int(eval_cfg.get("max_eval_samples", 500))
    if len(eval_dataset) > max_eval:
        from torch.utils.data import Subset

        indices = list(range(max_eval))
        eval_dataset = Subset(eval_dataset, indices)

    max_batch_tokens = 65536
    if cfg.get("training"):
        max_batch_tokens = cfg.training.get("max_batch_tokens", max_batch_tokens)

    from bgkit.data.samplers import TokenBudgetBatchSampler

    # Get lengths for sampler
    if hasattr(eval_dataset, "lengths"):
        eval_lengths = eval_dataset.lengths
    else:
        inner = getattr(eval_dataset, "dataset", eval_dataset)
        if hasattr(inner, "lengths"):
            if hasattr(eval_dataset, "indices"):
                eval_lengths = inner.lengths[eval_dataset.indices]
            else:
                eval_lengths = inner.lengths
        else:
            eval_lengths = [4096] * len(eval_dataset)

    eval_sampler = TokenBudgetBatchSampler(eval_lengths, max_batch_tokens, shuffle=False)
    dataloader = DataLoader(
        eval_dataset,
        batch_sampler=eval_sampler,
        collate_fn=collate_chat_repro,
        num_workers=0,
    )

    report = {"checkpoint": str(checkpoint_path), "device": str(device)}

    # 1. Compression curve
    logger.info("eval_compression_curve")
    report["compression_curve"] = _run_compression_curve(
        encoder, decoder, dataloader, device, tokenizer,
    )

    # 2. Embedding health
    logger.info("eval_embedding_health")
    report["embedding_health"] = _run_embedding_health(encoder, dataloader, device)

    # 3. RAG baseline comparison
    logger.info("eval_rag_baseline")
    report["rag_baseline"] = _run_rag_baseline(
        eval_cfg, encoder, decoder, tokenizer, dataloader, device,
    )

    # Write report
    report_path = output_dir / "eval_phase1.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("eval_phase1_complete", report_path=str(report_path))
    print(json.dumps(report, indent=2, default=str))

    # WandB logging
    if eval_cfg.get("wandb", False):
        try:
            import wandb

            wandb.init(project=cfg.get("wandb_project", "bgkit"), name="eval_phase1")
            flat = {}
            for point in report.get("compression_curve", []):
                ratio = point["compression_ratio"]
                for k, v in point.items():
                    if k != "compression_ratio" and isinstance(v, (int, float)):
                        flat[f"eval/compression/{k}_at_{ratio}"] = v
            for k, v in report.get("embedding_health", {}).items():
                if isinstance(v, (int, float)):
                    flat[f"eval/embedding/{k}"] = v
            for k, v in report.get("rag_baseline", {}).items():
                if isinstance(v, (int, float)):
                    flat[f"eval/rag/{k}"] = v
            wandb.log(flat)
            wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
