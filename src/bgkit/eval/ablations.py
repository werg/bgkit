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


def _resolve_splice_metadata(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(splice_starts, splice_lengths)`` for decoder injection.

    Chat-formatted phase-1 data carries explicit splice metadata.
    Simpler QA-style datasets do not, but their ``prefix_ids`` correspond to
    the token prefix before the answer-bearing region, so a zero-width splice
    at ``len(prefix_ids)`` preserves the intended geometry.
    """
    if "bgkit_splice_start" in batch and "bgkit_splice_len" in batch:
        return (
            batch["bgkit_splice_start"].to(device),
            batch["bgkit_splice_len"].to(device),
        )
    prefix_mask = batch.get("prefix_attention_mask")
    if prefix_mask is not None:
        starts = prefix_mask.to(device).sum(dim=1, dtype=torch.long)
        lengths = torch.zeros_like(starts)
        return starts, lengths
    prefix_ids = batch.get("prefix_ids")
    if prefix_ids is not None:
        starts = torch.full(
            (prefix_ids.size(0),), prefix_ids.size(1),
            dtype=torch.long, device=device,
        )
        lengths = torch.zeros_like(starts)
        return starts, lengths
    raise ValueError("Batch does not provide BgKIT splice metadata or prefix_ids")


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

            token_ids = batch["token_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            content_token_ids = batch["content_token_ids"].to(device)
            content_attention_mask = batch["content_attention_mask"].to(device)
            compression_prompt_ids = batch["compression_prompt_ids"].to(device)
            compression_prompt_mask = batch["compression_prompt_mask"].to(device)

            # Compute survivors via BgKIT encoder
            bgkit_embed = encoder.compressor.backbone.get_input_embeddings()
            content_emb = bgkit_embed(content_token_ids)
            prompt_emb = bgkit_embed(compression_prompt_ids)

            with torch.no_grad():
                enc_out = encoder(
                    input_embeddings=content_emb,
                    survivor_mask=None,
                    attention_mask=content_attention_mask,
                    prompt_embeddings=prompt_emb,
                    prompt_attention_mask=compression_prompt_mask,
                )
                survivors = enc_out.survivor_embeddings

                # Apply ablation modification
                survivors = _modify_survivors(survivors, condition)

                # Teacher-forced loss
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    splice_starts, splice_lengths = _resolve_splice_metadata(batch, device)
                    loss = decoder.forward_with_single_splice(
                        survivor_embeddings=survivors,
                        survivor_attention_mask=content_attention_mask,
                        token_ids=token_ids,
                        token_attention_mask=attention_mask,
                        splice_starts=splice_starts,
                        splice_lengths=splice_lengths,
                        loss_mask=loss_mask,
                    )

                batch_tokens = loss_mask.sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

                # Optional generation metrics
                if include_generation_metrics and suffix_ids is not None:
                    prefix_ids = batch["prefix_ids"].to(device)
                    prefix_attention_mask = batch["prefix_attention_mask"].to(device)
                    splice_starts, splice_lengths = _resolve_splice_metadata(batch, device)
                    gen_output = decoder.generate_with_single_splice(
                        survivor_embeddings=survivors,
                        survivor_attention_mask=content_attention_mask,
                        prefix_ids=prefix_ids,
                        prefix_attention_mask=prefix_attention_mask,
                        splice_starts=splice_starts,
                        splice_lengths=splice_lengths,
                        suffix_ids=suffix_ids.to(device),
                        tokenizer=tokenizer,
                        max_new_tokens=2048,
                        temperature=0.0,
                    )
                    generated_texts.extend(gen_output.content_text)
                    generated_languages.extend(batch["languages"])

            examples_seen += token_ids.size(0)

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
