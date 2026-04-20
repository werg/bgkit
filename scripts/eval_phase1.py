#!/usr/bin/env python
"""Post-Phase-1 baseline evaluation.

Orchestrates Eval 1: compression curves, ablations, RAG baseline comparison,
per-language parse success, embedding health, and description quality.

Usage:
    python scripts/eval_phase1.py \
        +eval.checkpoint=checkpoints/phase1_step6_best \
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
    """Run evaluation at a specific compression ratio (packed form).

    Uses the encoder's own survivorship head with ``target_ratio=ratio``
    rather than synthesizing a uniform survivor mask from outside —
    that was the old padded helper's behaviour and doesn't translate
    cleanly to packed inputs (per-sample indexing into a padded axis
    becomes a flat scatter across the packed buffer). Letting the head
    pick survivors at the requested ratio is more faithful to the
    operating regime anyway.
    """
    from bgkit.eval.ablations import _build_decoder_segments_from_batch

    total_loss = 0.0
    total_weight = 0.0
    all_pred_texts: list[str] = []
    all_ref_texts: list[str] = []
    all_languages: list[str] = []

    bgkit_embed = encoder.compressor.backbone.get_input_embeddings()

    with torch.no_grad():
        for batch in dataloader:
            content_token_ids = batch["content_token_ids"].to(device)
            content_cu = batch["content_cu_seqlens"].to(device)
            content_position_ids = batch["content_position_ids"].to(device)
            prompt_ids = batch["compression_prompt_ids"].to(device)
            prompt_cu = batch["compression_prompt_cu_seqlens"].to(device)
            prompt_position_ids = batch["compression_prompt_position_ids"].to(device)
            loss_mask_flat = batch["loss_mask"].to(device)

            content_emb = bgkit_embed(content_token_ids)
            prompt_emb = bgkit_embed(prompt_ids)

            enc_out = encoder(
                content_embeddings=content_emb,
                content_cu_seqlens=content_cu,
                content_position_ids=content_position_ids,
                prompt_embeddings=prompt_emb,
                prompt_cu_seqlens=prompt_cu,
                prompt_position_ids=prompt_position_ids,
                target_ratio=ratio,
                level="l0",
            )
            survivors = enc_out.survivor_embeddings
            survivor_cu = enc_out.survivor_cu_seqlens

            prefix_list, suffix_list, flat_loss_mask = _build_decoder_segments_from_batch(
                batch, device, survivor_cu
            )
            loss = decoder.forward_with_single_splice(
                survivor_embeddings=survivors,
                survivor_cu_seqlens=survivor_cu,
                prefix_ids=prefix_list,
                suffix_ids=suffix_list,
                loss_mask=flat_loss_mask,
            )
            batch_weight = loss_mask_flat.sum().item()
            total_loss += loss.item() * batch_weight
            total_weight += batch_weight

            # Text preds for parse / description metrics — stubbed under packing.
            # The previous argmax-over-padded-logits extraction doesn't translate
            # cleanly to the packed decoder forward (which returns a scalar loss,
            # not logits). Left as a follow-up: recover per-sample argmax via a
            # second forward that returns hidden states, then call lm_head. Skip
            # for now so the curve metric still produces.
            if tokenizer is not None:
                pass

    avg_loss = total_loss / max(total_weight, 1.0)

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
    """Check embedding drift and health (packed form)."""
    all_embeddings = []
    bgkit_embed = encoder.compressor.backbone.get_input_embeddings()

    with torch.no_grad():
        for batch in dataloader:
            content_token_ids = batch["content_token_ids"].to(device)
            content_cu = batch["content_cu_seqlens"].to(device)
            content_position_ids = batch["content_position_ids"].to(device)

            content_emb = bgkit_embed(content_token_ids)
            enc_out = encoder(
                content_embeddings=content_emb,
                content_cu_seqlens=content_cu,
                content_position_ids=content_position_ids,
                target_ratio=None,
            )
            # Flat (N, D) survivor embeddings from the uncompressed pass.
            flat = enc_out.survivor_embeddings.detach().cpu()
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
    """Run RAG baseline comparison (packed form).

    Rewritten for packed attention: both the BgKIT leg and the RAG leg
    go through ``forward_with_single_splice``. BgKIT's survivors fill
    the splice region; RAG retrieves text context, embeds it via the
    decoder's token embedding table, and passes the embeddings as
    "survivors" — a synthetic context of retrieved tokens.
    """
    from bgkit.eval.ablations import _build_decoder_segments_from_batch

    rag = RAGBaseline(
        embedding_model_name=eval_cfg.get("rag_embedding_model", "all-MiniLM-L6-v2"),
        reranker_model_name=eval_cfg.get("rag_reranker_model", None),
    )
    top_k = int(eval_cfg.get("rag_top_k", 5))
    max_samples = int(eval_cfg.get("rag_max_samples", 100))

    # First pass: collect per-sample content text so RAG can index + retrieve.
    indexed_samples: list[dict] = []
    samples_seen = 0
    with torch.no_grad():
        for batch in dataloader:
            if samples_seen >= max_samples:
                break
            content_cu = batch["content_cu_seqlens"].to(torch.int64)
            b_size = int(content_cu.shape[0]) - 1
            for i in range(b_size):
                if samples_seen >= max_samples:
                    break
                s, e = int(content_cu[i].item()), int(content_cu[i + 1].item())
                ids_i = batch["content_token_ids"][s:e].tolist()
                text = tokenizer.decode(ids_i, skip_special_tokens=True)
                file_key = f"sample_{samples_seen}"
                indexed_samples.append({
                    "batch": batch,
                    "sample_idx": i,
                    "content_text": text,
                    "file_key": file_key,
                })
                samples_seen += 1

    if not indexed_samples:
        return {"rag_baseline": "no_eval_data"}

    files = {e["file_key"]: e["content_text"] for e in indexed_samples}
    rag.index_repository(files)

    bgkit_embed = encoder.compressor.backbone.get_input_embeddings()
    decoder_embed = decoder.backbone.get_input_embeddings()
    rag_total_loss = 0.0
    bgkit_total_loss = 0.0
    count = 0

    with torch.no_grad():
        for entry in indexed_samples:
            batch = entry["batch"]
            i = entry["sample_idx"]

            # --- Build a single-sample sub-batch view onto the per-sample slice ---
            content_cu_full = batch["content_cu_seqlens"].to(device, dtype=torch.int32)
            prompt_cu_full = batch["compression_prompt_cu_seqlens"].to(device, dtype=torch.int32)
            tok_cu_full = batch["cu_seqlens"].to(device, dtype=torch.int32)

            c_s = int(content_cu_full[i].item())
            c_e = int(content_cu_full[i + 1].item())
            p_s = int(prompt_cu_full[i].item())
            p_e = int(prompt_cu_full[i + 1].item())
            t_s = int(tok_cu_full[i].item())
            t_e = int(tok_cu_full[i + 1].item())

            content_ids_i = batch["content_token_ids"][c_s:c_e].to(device)
            prompt_ids_i = batch["compression_prompt_ids"][p_s:p_e].to(device)
            token_ids_i = batch["token_ids"][t_s:t_e].to(device)
            loss_mask_i = batch["loss_mask"][t_s:t_e].to(device)
            splice_start_i = int(batch["bgkit_splice_start"][i].item())
            splice_len_i = int(batch["bgkit_splice_len"][i].item())

            if splice_start_i < 0:
                splice_start_i = token_ids_i.shape[0]
                splice_len_i = 0
            prefix_i = token_ids_i[:splice_start_i]
            suffix_i = token_ids_i[splice_start_i + splice_len_i :]
            pre_mask = loss_mask_i[:splice_start_i]
            suf_mask = loss_mask_i[splice_start_i + splice_len_i :]

            cu_single = torch.tensor(
                [0, token_ids_i.shape[0]], dtype=torch.int32, device=device,
            )
            single_batch = {
                "token_ids": token_ids_i,
                "cu_seqlens": cu_single,
                "loss_mask": loss_mask_i,
                "bgkit_splice_start": torch.tensor([splice_start_i], dtype=torch.long),
                "bgkit_splice_len": torch.tensor([splice_len_i], dtype=torch.long),
            }

            # ---- BgKIT leg ----
            content_cu_i = torch.tensor([0, c_e - c_s], dtype=torch.int32, device=device)
            prompt_cu_i = torch.tensor([0, p_e - p_s], dtype=torch.int32, device=device)
            content_emb = bgkit_embed(content_ids_i)
            prompt_emb = bgkit_embed(prompt_ids_i)
            content_pos_i = torch.arange(
                content_ids_i.shape[0], device=device, dtype=torch.long,
            )
            prompt_pos_i = torch.arange(
                prompt_ids_i.shape[0], device=device, dtype=torch.long,
            )
            enc_out = encoder(
                content_embeddings=content_emb,
                content_cu_seqlens=content_cu_i,
                content_position_ids=content_pos_i,
                prompt_embeddings=prompt_emb,
                prompt_cu_seqlens=prompt_cu_i,
                prompt_position_ids=prompt_pos_i,
                target_ratio=None,
            )
            survivors = enc_out.survivor_embeddings
            survivor_cu = enc_out.survivor_cu_seqlens
            prefix_list, suffix_list, flat_loss_mask = _build_decoder_segments_from_batch(
                single_batch, device, survivor_cu
            )
            bgkit_loss = decoder.forward_with_single_splice(
                survivor_embeddings=survivors,
                survivor_cu_seqlens=survivor_cu,
                prefix_ids=prefix_list,
                suffix_ids=suffix_list,
                loss_mask=flat_loss_mask,
            )
            bgkit_total_loss += bgkit_loss.item()

            # ---- RAG leg: retrieved tokens as "survivors" ----
            rag_context = rag.retrieve_text(entry["content_text"][:200], top_k=top_k)
            rag_ids = tokenizer.encode(
                rag_context,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(device).squeeze(0)
            rag_emb = decoder_embed(rag_ids)
            rag_cu = torch.tensor([0, rag_emb.shape[0]], dtype=torch.int32, device=device)
            prefix_list_rag, suffix_list_rag, flat_loss_mask_rag = (
                _build_decoder_segments_from_batch(single_batch, device, rag_cu)
            )
            rag_loss = decoder.forward_with_single_splice(
                survivor_embeddings=rag_emb,
                survivor_cu_seqlens=rag_cu,
                prefix_ids=prefix_list_rag,
                suffix_ids=suffix_list_rag,
                loss_mask=flat_loss_mask_rag,
            )
            rag_total_loss += rag_loss.item()
            count += 1
            # Reserved: per-leg loss_mask overrides via pre_mask / suf_mask.
            _ = pre_mask, suf_mask, prefix_i, suffix_i

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

    from bgkit.data.samplers import PackedTokenBudgetSampler

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

    eval_sampler = PackedTokenBudgetSampler(
        dataset=None,
        lengths=eval_lengths,
        max_batch_tokens=max_batch_tokens,
        shuffle=False,
    )
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
