"""Mandatory ablation suite: survivors present vs zeroed vs noise.

This is the project's kill switch, not optional evaluation. Run after
every training stage. If the gap between present and zeroed/noise is
negligible, stop and re-evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import structlog
import torch
from torch.utils.data import DataLoader

from bgkit.eval.metrics.reconstruction import parse_success_rate
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder

logger = structlog.get_logger()


def _split_packed_segments(
    flat: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> list[torch.Tensor]:
    """Split a flat packed ``(N, ...)`` tensor into per-sample slices."""
    cu_list = cu_seqlens.to(torch.int64).tolist()
    return [flat[cu_list[i] : cu_list[i + 1]] for i in range(len(cu_list) - 1)]


def _build_decoder_segments_from_batch(
    batch: dict,
    device: torch.device,
    survivor_cu_seqlens: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Derive ``(prefix_ids, suffix_ids, flat_loss_mask)`` from a packed chat-repro batch.

    The flat ``loss_mask`` is laid out over the assembled
    ``[prefix_b | survivors_b | suffix_b]`` segments that
    ``ReconstructionDecoder.forward_with_single_splice`` expects. Survivor
    positions receive loss_mask=False (decoder cannot predict embeddings).
    """
    token_ids_flat = batch["token_ids"].to(device)
    tok_cu = batch["cu_seqlens"].to(device)
    loss_mask_flat = batch["loss_mask"].to(device).to(torch.bool)
    splice_start = batch["bgkit_splice_start"].to(device)
    splice_len = batch["bgkit_splice_len"].to(device)

    batch_size = int(tok_cu.shape[0]) - 1
    tok_cu_list = tok_cu.to(torch.int64).tolist()
    surv_cu_list = survivor_cu_seqlens.to(torch.int64).tolist()

    prefix_ids: list[torch.Tensor] = []
    suffix_ids: list[torch.Tensor] = []
    per_segment: list[torch.Tensor] = []
    for b in range(batch_size):
        sample_tokens = token_ids_flat[tok_cu_list[b] : tok_cu_list[b + 1]]
        sample_loss = loss_mask_flat[tok_cu_list[b] : tok_cu_list[b + 1]]
        ss = int(splice_start[b].item())
        sl = int(splice_len[b].item())
        if ss < 0:
            ss = sample_tokens.shape[0]
            sl = 0
        prefix_ids.append(sample_tokens[:ss])
        suffix_ids.append(sample_tokens[ss + sl :])

        pre_mask = sample_loss[:ss]
        suf_mask = sample_loss[ss + sl :]
        k_i = surv_cu_list[b + 1] - surv_cu_list[b]
        surv_mask = torch.zeros(k_i, dtype=torch.bool, device=device)
        per_segment.append(torch.cat([pre_mask, surv_mask, suf_mask], dim=0))

    return prefix_ids, suffix_ids, torch.cat(per_segment, dim=0)


class AblationCondition(Enum):
    SURVIVORS_PRESENT = "present"
    SURVIVORS_ZEROED = "zeroed"
    SURVIVORS_NOISE = "noise"


@dataclass
class AblationResult:
    """Result of a single ablation run."""

    condition: AblationCondition
    metrics: dict[str, float]


def _modify_survivors(
    survivors: torch.Tensor,
    condition: AblationCondition,
) -> torch.Tensor:
    """Apply ablation modification to survivor embeddings."""
    if condition == AblationCondition.SURVIVORS_PRESENT:
        return survivors
    elif condition == AblationCondition.SURVIVORS_ZEROED:
        return torch.zeros_like(survivors)
    elif condition == AblationCondition.SURVIVORS_NOISE:
        return torch.randn_like(survivors) * survivors.std()
    else:
        raise ValueError(f"Unknown ablation condition: {condition}")


def run_ablation_suite(
    decoder: ReconstructionDecoder,
    encoder: BgKITEncoder,
    eval_dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    conditions: list[AblationCondition] | None = None,
    max_examples: int = 500,
    include_generation_metrics: bool = False,
    suffix_ids: torch.Tensor | None = None,
) -> list[AblationResult]:
    """Run the mandatory ablation suite.

    Tests model performance with:
    - Survivors present (normal operation)
    - Survivors zeroed (all survivor embeddings set to zero)
    - Survivors noise (random Gaussian noise in place of survivors)

    The gap between present and zeroed/noise is the value signal.

    Args:
        decoder: Trained reconstruction decoder.
        encoder: Frozen BgKIT encoder.
        eval_dataloader: Evaluation dataloader yielding collated batches.
        tokenizer: Tokenizer for decoding (needed for generation metrics).
        device: Torch device.
        conditions: Which conditions to test (default: all three).
        max_examples: Maximum evaluation examples per condition.
        include_generation_metrics: Whether to run generation + parse_success_rate
            (expensive). Default False -- primary signal is loss gap.
        suffix_ids: Constant 1D suffix token IDs for generation. Required when
            include_generation_metrics is True.

    Returns:
        List of AblationResults.
    """
    if conditions is None:
        conditions = list(AblationCondition)

    decoder.eval()
    encoder.eval()
    results = []

    for condition in conditions:
        logger.info("ablation_condition_start", condition=condition.value)
        total_loss = 0.0
        total_tokens = 0.0
        generated_texts: list[str] = []
        generated_languages: list[str] = []
        examples_seen = 0

        for batch in eval_dataloader:
            if examples_seen >= max_examples:
                break

            # --- packed batch contents ---
            content_token_ids = batch["content_token_ids"].to(device)
            content_cu = batch["content_cu_seqlens"].to(device)
            content_position_ids = batch["content_position_ids"].to(device)
            prompt_ids = batch["compression_prompt_ids"].to(device)
            prompt_cu = batch["compression_prompt_cu_seqlens"].to(device)
            prompt_position_ids = batch["compression_prompt_position_ids"].to(device)
            loss_mask_flat = batch["loss_mask"].to(device)

            bgkit_embed = encoder.compressor.backbone.get_input_embeddings()
            content_emb = bgkit_embed(content_token_ids)
            prompt_emb = bgkit_embed(prompt_ids)

            with torch.no_grad():
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

                # Apply ablation modification.
                survivors = _modify_survivors(survivors, condition)

                # Teacher-forced loss.
                prefix_list, suffix_list, flat_loss_mask = _build_decoder_segments_from_batch(
                    batch, device, survivor_cu
                )
                batch_size = int(content_cu.shape[0]) - 1
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

                # Optional generation metrics.
                if include_generation_metrics and suffix_ids is not None:
                    # For generation we need per-sample prefix_ids and a
                    # constant suffix terminator. Here we take prefix_list
                    # (sample 0) since the decoder only processes one
                    # sample per call under sequential generation.
                    for b in range(batch_size):
                        k_start = int(survivor_cu[b].item())
                        k_end = int(survivor_cu[b + 1].item())
                        surv_b = survivors[k_start:k_end]
                        cu_b = torch.tensor([0, k_end - k_start], dtype=torch.int32, device=device)
                        gen_output = decoder.generate_with_single_splice(
                            survivor_embeddings=surv_b,
                            survivor_cu_seqlens=cu_b,
                            prefix_ids=prefix_list[b],
                            suffix_ids=suffix_ids.to(device),
                            tokenizer=tokenizer,
                            max_new_tokens=2048,
                            temperature=0.0,
                        )
                        generated_texts.extend(gen_output.content_text)
                    if "languages" in batch:
                        generated_languages.extend(batch["languages"])

            examples_seen += batch_size

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        metrics: dict[str, float] = {
            "loss": avg_loss,
            "perplexity": perplexity,
        }
        if include_generation_metrics and generated_texts:
            metrics["parse_success_rate"] = parse_success_rate(
                generated_texts, languages=generated_languages,
            )

        results.append(AblationResult(condition=condition, metrics=metrics))
        logger.info(
            "ablation_condition_done",
            condition=condition.value,
            loss=avg_loss,
            perplexity=perplexity,
        )

    return results


def compute_ablation_gap(results: list[AblationResult]) -> dict[str, float]:
    """Compute present-vs-zeroed and present-vs-noise gaps.

    A positive gap means the 'present' condition is better (lower loss),
    i.e., zeroed/noise hurt performance -- the expected outcome if BgKIT
    survivors carry signal.

    Returns:
        Dict with gap values. Both should be positive if survivors are useful.
    """
    by_condition = {r.condition: r.metrics for r in results}

    gaps: dict[str, float] = {}
    present_loss = by_condition.get(AblationCondition.SURVIVORS_PRESENT, {}).get("loss")

    if present_loss is not None:
        zeroed_loss = by_condition.get(AblationCondition.SURVIVORS_ZEROED, {}).get("loss")
        noise_loss = by_condition.get(AblationCondition.SURVIVORS_NOISE, {}).get("loss")

        if zeroed_loss is not None:
            gaps["present_vs_zeroed_loss_gap"] = zeroed_loss - present_loss
        if noise_loss is not None:
            gaps["present_vs_noise_loss_gap"] = noise_loss - present_loss

    return gaps
