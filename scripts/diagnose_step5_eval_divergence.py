#!/usr/bin/env python
"""Phase 1 Step 5 train/eval divergence diagnostic.

Reproduces the trainer's setup() pipeline (encoder + decoder load, dataset
construction, train/eval random_split with the same seed) WITHOUT building
an optimizer or any training-side state. Then iterates the eval dataloader
sample-by-sample and records:

  * present_loss     -- decoder loss with real survivors
  * zeroed_loss      -- decoder loss with zeroed survivors (encoder ablated)
  * sample_length    -- token count of each sample as the dataset reports it
  * batch_index      -- which packed eval batch the sample landed in
  * sample_index     -- the dataset-relative index inside the eval split

Optional sweeps (per env var):
  * BGKIT_DIAG_RATIO_SWEEP=1      -- run a 7-point ratio response curve over
                                     a 30-sample slice (small N to keep
                                     under 5 minutes when sharing GPU).
  * BGKIT_DIAG_TRAIN_HOLDOUT=1    -- evaluate `loss_zeroed` on a 200-sample
                                     train slice so we can compare with the
                                     eval `loss_zeroed` (decoder overfit
                                     check).
  * BGKIT_DIAG_LIMIT=N            -- limit the present/zeroed pass to the
                                     first N batches.

Output: a JSON report under
  ${CHECKPOINT_DIR}/_diagnostics/eval_divergence_<ts>.json
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer

from bgkit.data.collators import collate_compression
from bgkit.data.datasets.commit_encoding_dataset import CommitEncodingDataset
from bgkit.data.samplers import PackedTokenBudgetSampler
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.training.checkpointing import load_checkpoint
from bgkit.utils.attention_backend import resolve_attention_implementation
from bgkit.utils.packing import position_ids_from_cu

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Trainer helpers reproduced inline so we don't have to instantiate the
# CommitEncodingTrainer (which builds an optimizer + Muon state + ICE etc).
# Mirrors `_encode_two_level` / `_decoder_forward_single_splice` exactly.
# ---------------------------------------------------------------------------


def _encode_two_level(
    encoder: BgKITEncoder,
    batch: dict,
    target_ratio_l0: float,
    target_ratio_l1: float,
    device: torch.device,
):
    file_ids = batch["content_token_ids"].to(device)
    cu_file = batch["cu_file_seqlens"].to(device)
    content_position_ids = batch["content_position_ids"].to(device)
    prompt_ids = batch["prompt_token_ids"].to(device)
    prompt_cu = batch["prompt_cu_seqlens"].to(device)
    prompt_position_ids = position_ids_from_cu(prompt_cu, int(prompt_ids.shape[0]))
    cu_repo = batch["cu_repo_seqlens"].to(device)

    embed = encoder.l0.backbone.get_input_embeddings()
    content_emb = embed(file_ids)
    prompt_emb = embed(prompt_ids)

    l0_out = encoder.l0(
        content_embeddings=content_emb,
        content_cu_seqlens=cu_file,
        content_position_ids=content_position_ids,
        prompt_embeddings=prompt_emb,
        prompt_cu_seqlens=prompt_cu,
        prompt_position_ids=prompt_position_ids,
        target_ratio=target_ratio_l0,
        utility_grad_active=False,
    )

    l0_survivors = l0_out.survivor_embeddings
    l0_surv_cu = l0_out.survivor_cu_seqlens
    cu_repo_64 = cu_repo.to(torch.int64)
    cu_file_64 = l0_surv_cu.to(torch.int64)
    l1_input_cu = cu_file_64[cu_repo_64].to(torch.int32)

    bridged = encoder.l0.auto_reproduce(l0_survivors)
    n_l1 = int(bridged.shape[0])
    l1_pos = position_ids_from_cu(l1_input_cu, n_l1)

    l1_out = encoder.l1(
        content_embeddings=bridged,
        content_cu_seqlens=l1_input_cu,
        content_position_ids=l1_pos,
        target_ratio=target_ratio_l1,
        utility_grad_active=False,
    )

    proj_input = l1_out.survivor_embeddings
    proj_cu = l1_out.survivor_cu_seqlens
    proj_pos = position_ids_from_cu(proj_cu, int(proj_input.shape[0]))
    from bgkit.utils.packing import lengths_from_cu
    proj_max = int(lengths_from_cu(proj_cu).max().item()) if proj_cu.numel() else 0
    proj_out = encoder.projection_block(
        proj_input,
        cu_seqlens=proj_cu,
        max_seqlen=proj_max,
        position_ids=proj_pos,
        survivor_mask=None,
    )
    return proj_out.projected_embeddings, proj_cu, l0_out, l1_out


def _decoder_forward_single_splice(
    decoder: ReconstructionDecoder,
    survivors: torch.Tensor,
    survivor_cu_seqlens: torch.Tensor,
    batch: dict,
    device: torch.device,
) -> torch.Tensor:
    target_ids_flat = batch["target_token_ids"].to(device)
    target_cu = batch["target_cu_seqlens"].to(device)
    splice_start = batch["bgkit_splice_start"].to(device)
    splice_len = batch["bgkit_splice_len"].to(device)
    loss_mask_flat = batch.get("target_loss_mask")
    if loss_mask_flat is not None:
        loss_mask_flat = loss_mask_flat.to(device).to(torch.bool)

    batch_size = int(target_cu.shape[0]) - 1
    tok_cu_list = target_cu.to(torch.int64).tolist()
    surv_cu_list = survivor_cu_seqlens.to(torch.int64).tolist()

    prefix_ids: list[torch.Tensor] = []
    suffix_ids: list[torch.Tensor] = []
    per_segment_loss_masks: list[torch.Tensor] = []
    for b in range(batch_size):
        sample_start = int(tok_cu_list[b])
        sample_end = int(tok_cu_list[b + 1])
        sample_tokens = target_ids_flat[sample_start:sample_end]
        splice_b_start = int(splice_start[b].item())
        splice_b_len = int(splice_len[b].item())
        if splice_b_start < 0:
            splice_b_start = sample_tokens.shape[0]
            splice_b_len = 0
        pre = sample_tokens[:splice_b_start]
        suf = sample_tokens[splice_b_start + splice_b_len:]
        prefix_ids.append(pre)
        suffix_ids.append(suf)

        k_i = int(surv_cu_list[b + 1]) - int(surv_cu_list[b])
        surv_mask = torch.zeros(k_i, dtype=torch.bool, device=device)
        if loss_mask_flat is not None:
            sample_loss = loss_mask_flat[sample_start:sample_end]
            pre_mask = sample_loss[:splice_b_start]
            suf_mask = sample_loss[splice_b_start + splice_b_len:]
        else:
            pre_mask = torch.zeros(pre.shape[0], dtype=torch.bool, device=device)
            suf_mask = torch.ones(suf.shape[0], dtype=torch.bool, device=device)
        per_segment_loss_masks.append(torch.cat([pre_mask, surv_mask, suf_mask], dim=0))

    flat_loss_mask = (
        torch.cat(per_segment_loss_masks, dim=0)
        if per_segment_loss_masks else None
    )
    return decoder.forward_with_single_splice(
        survivor_embeddings=survivors,
        survivor_cu_seqlens=survivor_cu_seqlens,
        prefix_ids=prefix_ids,
        suffix_ids=suffix_ids,
        loss_mask=flat_loss_mask,
    )


def _per_sample_decoder_loss(
    decoder: ReconstructionDecoder,
    survivors: torch.Tensor,
    survivor_cu_seqlens: torch.Tensor,
    batch: dict,
    device: torch.device,
) -> tuple[list[float], list[int]]:
    """Run the decoder one sample at a time so we get per-sample loss values.

    Returns (per_sample_loss, per_sample_loss_token_count).
    """
    losses: list[float] = []
    token_counts: list[int] = []
    target_ids_flat = batch["target_token_ids"]
    target_cu = batch["target_cu_seqlens"].to(torch.int64)
    splice_start = batch["bgkit_splice_start"]
    splice_len = batch["bgkit_splice_len"]
    loss_mask_flat = batch.get("target_loss_mask")

    surv_cu_list = survivor_cu_seqlens.to(torch.int64).tolist()
    batch_size = int(target_cu.shape[0]) - 1

    for b in range(batch_size):
        s_lo = int(target_cu[b].item())
        s_hi = int(target_cu[b + 1].item())
        sample_tokens = target_ids_flat[s_lo:s_hi]

        # Build a one-sample mini-batch view by slicing.
        single = {
            "target_token_ids": sample_tokens,
            "target_cu_seqlens": torch.tensor([0, sample_tokens.shape[0]], dtype=torch.int32),
            "bgkit_splice_start": splice_start[b:b + 1],
            "bgkit_splice_len": splice_len[b:b + 1],
        }
        if loss_mask_flat is not None:
            single["target_loss_mask"] = loss_mask_flat[s_lo:s_hi]

        s_lo_surv = int(surv_cu_list[b])
        s_hi_surv = int(surv_cu_list[b + 1])
        single_surv = survivors[s_lo_surv:s_hi_surv]
        single_surv_cu = torch.tensor([0, s_hi_surv - s_lo_surv], dtype=torch.int32)

        loss = _decoder_forward_single_splice(
            decoder, single_surv, single_surv_cu, single, device
        )
        if loss_mask_flat is not None:
            tcount = int(loss_mask_flat[s_lo:s_hi].sum().item())
        else:
            tcount = int(sample_tokens.shape[0])
        losses.append(float(loss.item()))
        token_counts.append(tcount)
    return losses, token_counts


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


@dataclass
class DiagModels:
    encoder: BgKITEncoder
    decoder: ReconstructionDecoder
    tokenizer: object


def _load_models_for_step5(
    cfg: DictConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> DiagModels:
    bgkit_cfg = cfg.model.bgkit
    decoder_cfg = cfg.model.decoder
    hidden_dim = bgkit_cfg.get("hidden_dim", 1024)
    attention_impl = resolve_attention_implementation(
        cfg.compute.get("attention_implementation", "auto")
    )

    logger.info("loading_checkpoint", path=str(checkpoint_path))
    metadata, state_dicts = load_checkpoint(checkpoint_path)
    logger.info("checkpoint_metadata", step=metadata.step, metrics=metadata.metrics)

    # Read curriculum-controller init_target_ratio / init_theta from config.
    ctrl_src = cfg.model.get("threshold_controller", {})
    tcfg = cfg.training
    stage0_l0_start = float(tcfg.get("stage0_l0_ratio_start", 0.9))
    threshold_controller_cfg = {
        "init_theta": float(ctrl_src.get("init_theta", 1.0 - 2.0 * stage0_l0_start)),
        "lr": float(ctrl_src.get("lr", 0.02)),
        "momentum": float(ctrl_src.get("momentum", 0.0)),
        "clamp": float(ctrl_src.get("clamp", 0.99)),
        "anchor_ratios": list(ctrl_src.get("anchor_ratios", [])) or None,
        "ratio_space": str(ctrl_src.get("ratio_space", "log")),
        "init_target_ratio": stage0_l0_start,
        "default_query_ratio": stage0_l0_start,
    }
    encoder = BgKITEncoder.from_pretrained_with_state_dict(
        bgkit_cfg.backbone_name,
        state_dicts["encoder"],
        hidden_dim=hidden_dim,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        revision=bgkit_cfg.get("backbone_revision"),
        attn_implementation=attention_impl,
        bidi_warmup_steps=tcfg.get("bidi_warmup_steps", 0),
        threshold_controller_cfg=threshold_controller_cfg,
    )
    encoder.to(device).eval()
    encoder.requires_grad_(False)

    # Decoder: the trainer applies LoRA after instantiation. The checkpoint's
    # decoder state dict reflects the LoRA-wrapped model; build the same way.
    decoder_backbone = AutoModelForCausalLM.from_pretrained(
        decoder_cfg.backbone_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        revision=decoder_cfg.get("backbone_revision"),
        attn_implementation=attention_impl,
        device_map=device,
    )
    decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=hidden_dim)

    lora_cfg = tcfg.get("decoder_lora", {})
    if lora_cfg.get("enabled", False):
        decoder.apply_lora(lora_cfg)

    # Prefer 'decoder' key (LoRA-wrapped state dict) over 'decoder_merged'
    # for an exact match to the live trainer.
    decoder_sd = state_dicts.get("decoder", state_dicts.get("decoder_merged"))
    if decoder_sd is not None:
        decoder.load_state_dict(decoder_sd)
    decoder.to(device).eval()
    decoder.requires_grad_(False)

    # Liger patches — keep them aligned with trainer behavior.
    if tcfg.get("use_liger", True):
        from bgkit.utils.liger_integration import apply_liger_to_qwen35

        patch_rmsnorm = bool(tcfg.get("use_liger_rmsnorm", False))
        patch_swiglu = bool(tcfg.get("use_liger_swiglu", True))
        patch_rope = bool(tcfg.get("use_liger_rope", True))
        use_liger_ce = bool(tcfg.get("use_liger_ce", True))
        apply_liger_to_qwen35(
            encoder, patch_rmsnorm=patch_rmsnorm,
            patch_swiglu=patch_swiglu, patch_rope=patch_rope,
        )
        apply_liger_to_qwen35(
            decoder, patch_rmsnorm=patch_rmsnorm,
            patch_swiglu=patch_swiglu, patch_rope=patch_rope,
        )
        if use_liger_ce:
            decoder.enable_liger_ce(True)

    tokenizer = AutoTokenizer.from_pretrained(
        decoder_cfg.backbone_name,
        trust_remote_code=True,
        revision=decoder_cfg.get("backbone_revision"),
    )
    return DiagModels(encoder=encoder, decoder=decoder, tokenizer=tokenizer)


def _build_dataset_and_split(
    cfg: DictConfig,
    tokenizer,
):
    tcfg = cfg.training
    data_cfg = tcfg.data
    seed = cfg.get("seed", 42)
    # Variant bank — load same way the trainer does.
    from bgkit.training.base_trainer import BaseTrainer  # noqa: F401
    # Replicate `_load_variant_bank` minimal behavior.
    variant_dir = Path(data_cfg.get("prompt_variants_dir", "configs/prompt_variants"))
    variant_path = variant_dir / "commit_encoding.json"
    if not variant_path.exists():
        # fall back: trainer will reject empty banks; just provide a single
        # default.
        variant_bank = [{"system_prompt": "", "user_prompt": "{content}"}]
    else:
        variant_bank = json.loads(variant_path.read_text())

    from bgkit.data.chat_template import TOOL_CONFIGS

    config = TOOL_CONFIGS["commit_encoding"]
    dataset = CommitEncodingDataset(
        data_dir=data_cfg.commit_encoding_dir,
        tokenizer=tokenizer,
        variant_bank=variant_bank,
        config=config,
        max_diff_tokens_per_file=data_cfg.get("max_diff_tokens_per_file", 4096),
        max_files_per_commit=data_cfg.get("max_files_per_commit", 16),
        max_message_tokens=data_cfg.get("max_message_tokens", 256),
        seed=seed,
    )

    max_eval_samples = tcfg.get("max_eval_samples", 5000)
    total = len(dataset)
    eval_size = min(max(1, int(total * 0.1)), max_eval_samples)
    train_size = total - eval_size
    # CRITICAL: random_split requires a `generator` to be deterministic.
    # The trainer in commit_encoding.py:280 does NOT pass a generator —
    # it relies on the global torch RNG state. To reproduce, we mirror the
    # same: do random_split with the default generator. We seed first via
    # torch.manual_seed(seed).
    torch.manual_seed(seed)
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    return dataset, train_dataset, eval_dataset, total


# ---------------------------------------------------------------------------
# Diagnostic passes
# ---------------------------------------------------------------------------


def _length_distribution(lengths: np.ndarray) -> dict:
    return {
        "n": int(lengths.size),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p10": float(np.percentile(lengths, 10)),
        "p25": float(np.percentile(lengths, 25)),
        "p75": float(np.percentile(lengths, 75)),
        "p90": float(np.percentile(lengths, 90)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "min": int(lengths.min()),
        "std": float(lengths.std()),
    }


def _per_sample_eval_pass(
    models: DiagModels,
    eval_dataloader: DataLoader,
    eval_indices: list[int],
    target_ratio_l0: float,
    target_ratio_l1: float,
    device: torch.device,
    do_zeroed: bool,
    limit_batches: int | None,
) -> list[dict]:
    """Single forward pass over the eval dataloader, recording per-sample loss.

    Each batch packs B samples; we run two encoder forward passes (present +
    optionally zeroed) then break the per-sample decoder loss out by
    iterating over `b` inside _per_sample_decoder_loss.
    """
    results: list[dict] = []
    eval_indices_arr = np.asarray(eval_indices)
    sample_cursor = 0  # tracks position into eval_indices_arr in iteration order

    for batch_idx, batch in enumerate(eval_dataloader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        if batch_idx % 25 == 0:
            logger.info("diag_progress", batch=batch_idx)

        # Determine sample-level dataset indices for this batch:
        # PackedTokenBudgetSampler is shuffle=False for eval, but it does
        # bucket-based reordering. We can't easily know the batch indices
        # without modifying the sampler. Instead we reconstruct lengths
        # from cu_repo_seqlens which equals batch size.
        cu_repo = batch["cu_repo_seqlens"]
        batch_size = int(cu_repo.shape[0]) - 1

        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            survivors, surv_cu, l0_out, l1_out = _encode_two_level(
                models.encoder, batch,
                target_ratio_l0=target_ratio_l0,
                target_ratio_l1=target_ratio_l1,
                device=device,
            )

            present_losses, present_tcounts = _per_sample_decoder_loss(
                models.decoder, survivors, surv_cu, batch, device,
            )

            zeroed_losses: list[float] | None = None
            if do_zeroed:
                zeroed_survivors = torch.zeros_like(survivors)
                zeroed_losses, _ = _per_sample_decoder_loss(
                    models.decoder, zeroed_survivors, surv_cu, batch, device,
                )

        # Per-sample lengths from cu_file_seqlens within the batch.
        cu_file = batch["cu_file_seqlens"].to(torch.int64).tolist()
        cu_repo_list = cu_repo.to(torch.int64).tolist()
        # cu_repo holds indices INTO cu_file. So sample b's content tokens
        # span file boundaries cu_repo_list[b] .. cu_repo_list[b+1] (exclusive).
        # Total content tokens for sample b = cu_file[cu_repo_list[b+1]] - cu_file[cu_repo_list[b]]
        for b in range(batch_size):
            f_lo = cu_repo_list[b]
            f_hi = cu_repo_list[b + 1]
            content_tokens = cu_file[f_hi] - cu_file[f_lo]
            n_files = f_hi - f_lo
            row = {
                "batch_idx": batch_idx,
                "in_batch_idx": b,
                "content_tokens": int(content_tokens),
                "n_files": int(n_files),
                "loss_target_tokens": int(present_tcounts[b]),
                "present_loss": float(present_losses[b]),
            }
            if zeroed_losses is not None:
                row["zeroed_loss"] = float(zeroed_losses[b])
                row["gap_zeroed"] = float(zeroed_losses[b] - present_losses[b])
            results.append(row)
            sample_cursor += 1

        # Free.
        if hasattr(l0_out, "release"):
            l0_out.release()
        if hasattr(l1_out, "release"):
            l1_out.release()

    return results


def _ratio_response_curve(
    models: DiagModels,
    dataset_subset,
    sample_indices: list[int],
    ratios_l0: list[float],
    target_ratio_l1: float,
    device: torch.device,
) -> dict:
    """For a small set of samples, compute present/zeroed loss at each ratio.

    Returns a dict {"sample_idx": [...], "by_ratio": {ratio: [{loss, ...}]}}.
    """
    # Build a tiny dataloader that emits these samples one at a time.
    out: dict = {"sample_indices": list(sample_indices), "ratios": list(ratios_l0), "data": []}
    # Use a Subset over the eval subset.
    sub = Subset(dataset_subset, sample_indices)
    # batch_size=1 dataloader using the standard collate.
    loader = DataLoader(
        sub, batch_size=1, shuffle=False,
        collate_fn=collate_compression, num_workers=0,
    )

    samples_data: list[dict] = []
    for sample_pos, batch in enumerate(loader):
        per_ratio: dict = {}
        for ratio in ratios_l0:
            with torch.autocast(
                "cuda", dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                survivors, surv_cu, l0_out, l1_out = _encode_two_level(
                    models.encoder, batch,
                    target_ratio_l0=ratio,
                    target_ratio_l1=target_ratio_l1,
                    device=device,
                )
                p_losses, _ = _per_sample_decoder_loss(
                    models.decoder, survivors, surv_cu, batch, device,
                )
            per_ratio[f"{ratio:.4f}"] = float(p_losses[0])
            if hasattr(l0_out, "release"):
                l0_out.release()
            if hasattr(l1_out, "release"):
                l1_out.release()
        samples_data.append({
            "subset_pos": sample_pos,
            "subset_idx": sample_indices[sample_pos],
            "loss_by_ratio": per_ratio,
        })
    out["data"] = samples_data
    return out


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point — uses Hydra `compose` directly (no decorator)."""
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["+experiment=phase1_step5"],
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("diag_start", device=str(device))

    # Where's the checkpoint?
    ckpt_path_str = os.environ.get("BGKIT_DIAG_CHECKPOINT")
    if ckpt_path_str is None:
        raise RuntimeError(
            "BGKIT_DIAG_CHECKPOINT env var must be set to a phase1_step5 checkpoint dir"
        )
    ckpt_path = Path(ckpt_path_str)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ---- Load models ----
    models = _load_models_for_step5(cfg, ckpt_path, device)

    # ---- Build dataset + same train/eval split ----
    dataset, train_dataset, eval_dataset, total = _build_dataset_and_split(
        cfg, models.tokenizer,
    )
    logger.info(
        "split_built", total=total, train=len(train_dataset), eval=len(eval_dataset),
    )

    # ---- Length distributions ----
    train_lengths = np.array(
        [dataset.token_length(i) for i in train_dataset.indices], dtype=np.int64,
    )
    eval_lengths = np.array(
        [dataset.token_length(i) for i in eval_dataset.indices], dtype=np.int64,
    )

    # ---- Build eval dataloader (same config as trainer) ----
    tcfg = cfg.training
    max_batch_tokens_eval = int(tcfg.get(
        "max_batch_tokens_eval",
        int(tcfg.get("stage0_max_batch_tokens", 4096)),
    ))
    # Reuse trainer's eval-side default: same budget as train, but no shuffle.
    eval_sampler = PackedTokenBudgetSampler(
        eval_dataset,
        lengths=eval_lengths,
        max_batch_tokens=max_batch_tokens_eval,
        shuffle=False,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_sampler=eval_sampler,
        collate_fn=collate_compression,
        num_workers=0,
    )

    # ---- Resolve current curriculum state from ckpt step ----
    metadata_step = json.loads((ckpt_path / "metadata.json").read_text())["step"]
    # _curriculum_state but inlined (we don't have the trainer instance).
    auto_repro_warmup = int(tcfg.get("auto_repro_warmup_steps", 0))
    head_warmup = int(tcfg.get("head_warmup_steps", 0))
    stage0_end = int(tcfg.get("stage0_end_step", 3000))
    stage0_l0_start = float(tcfg.get("stage0_l0_ratio_start", 0.9))
    stage0_l0_end = float(tcfg.get("stage0_l0_ratio_end", 0.15))
    stage1_l0 = float(tcfg.get("stage1_target_ratio_l0", 0.30))
    stage0_l1 = float(tcfg.get("stage0_target_ratio_l1", 0.33))
    stage1_l1_ramp_start = int(tcfg.get("stage1_l1_ramp_start_step", 3000))
    stage1_l1_ramp_end = int(tcfg.get("stage1_l1_ramp_end_step", 5000))
    stage1_l1_start = float(tcfg.get("stage1_target_ratio_l1_start", 0.33))
    stage1_l1_end = float(tcfg.get("stage1_target_ratio_l1_end", 0.33))

    step = int(metadata_step)
    if step < auto_repro_warmup:
        ratio_l0, ratio_l1 = 1.0, 1.0
    elif step < stage0_end:
        offset = step - auto_repro_warmup
        span = max(stage0_end - auto_repro_warmup, 1)
        t = max(0.0, min(1.0, offset / span))
        ratio_l0 = stage0_l0_start + t * (stage0_l0_end - stage0_l0_start)
        ratio_l1 = stage0_l1
    else:
        ratio_l0 = stage1_l0
        if step < stage1_l1_ramp_start:
            ratio_l1 = stage1_l1_start
        elif step >= stage1_l1_ramp_end:
            ratio_l1 = stage1_l1_end
        else:
            tt = (step - stage1_l1_ramp_start) / max(
                stage1_l1_ramp_end - stage1_l1_ramp_start, 1,
            )
            ratio_l1 = stage1_l1_start + tt * (stage1_l1_end - stage1_l1_start)

    # The training_state in the checkpoint may carry a target_ratio_l1
    # override (e.g. 1.0). Check + apply.
    training_state = json.loads((ckpt_path / "metadata.json").read_text()).get("training_state", {})
    override_l0 = training_state.get("target_ratio_l0_override")
    override_l1 = training_state.get("target_ratio_l1_override")
    if override_l0 is not None:
        ratio_l0 = float(override_l0)
    if override_l1 is not None:
        ratio_l1 = float(override_l1)

    # Honor live-tunable overrides written to ${CHECKPOINT_DIR}/control.json.
    # These are the keys that affect the curriculum schedule, NOT just the
    # active-target-ratio values. Recompute ratio_l0 if the schedule has been
    # extended via stage0_end_step / etc.
    ckpt_dir = Path(os.environ.get("CHECKPOINT_DIR", "/workspace/checkpoints"))
    control_path = ckpt_dir / "control.json"
    if control_path.exists():
        try:
            ctrl_all = json.loads(control_path.read_text())
            ctrl = ctrl_all.get("phase1_step5", {})
            ctrl_stage0_end = ctrl.get("stage0_end_step")
            if ctrl_stage0_end is not None and step < int(ctrl_stage0_end):
                # Recompute ratio_l0 using extended span.
                offset = step - auto_repro_warmup
                span = max(int(ctrl_stage0_end) - auto_repro_warmup, 1)
                t = max(0.0, min(1.0, offset / span))
                ratio_l0 = stage0_l0_start + t * (stage0_l0_end - stage0_l0_start)
            ctrl_t_l1 = ctrl.get("target_ratio_l1")
            if ctrl_t_l1 is not None:
                ratio_l1 = float(ctrl_t_l1)
            ctrl_t_l0 = ctrl.get("target_ratio_l0")
            if ctrl_t_l0 is not None:
                ratio_l0 = float(ctrl_t_l0)
            logger.info("control_json_applied", control=ctrl)
        except Exception as e:
            logger.warning("control_json_parse_failed", error=str(e))

    # Allow explicit override for the diagnostic itself.
    env_ratio_l0 = os.environ.get("BGKIT_DIAG_RATIO_L0")
    env_ratio_l1 = os.environ.get("BGKIT_DIAG_RATIO_L1")
    if env_ratio_l0 is not None:
        ratio_l0 = float(env_ratio_l0)
    if env_ratio_l1 is not None:
        ratio_l1 = float(env_ratio_l1)

    logger.info(
        "curriculum_resolved",
        step=step,
        ratio_l0=float(ratio_l0),
        ratio_l1=float(ratio_l1),
        override_l0=override_l0,
        override_l1=override_l1,
    )

    # ---- Pass 1: present + zeroed per-sample over full eval set ----
    limit_env = os.environ.get("BGKIT_DIAG_LIMIT")
    limit_batches = int(limit_env) if limit_env else None
    logger.info("starting_per_sample_pass", limit_batches=limit_batches)
    t0 = time.time()
    per_sample = _per_sample_eval_pass(
        models, eval_dataloader, eval_dataset.indices,
        target_ratio_l0=float(ratio_l0),
        target_ratio_l1=float(ratio_l1),
        device=device,
        do_zeroed=True,
        limit_batches=limit_batches,
    )
    elapsed = time.time() - t0
    logger.info("per_sample_pass_done", n=len(per_sample), elapsed_s=elapsed)

    # ---- Optional: ratio response curve on 30 samples ----
    ratio_curve = None
    if os.environ.get("BGKIT_DIAG_RATIO_SWEEP", "0") == "1":
        rng = np.random.default_rng(123)
        n_curve = 30
        sample_indices = rng.choice(
            len(eval_dataset), size=min(n_curve, len(eval_dataset)), replace=False,
        ).tolist()
        # Sweep ratios of interest.
        ratios = [0.95, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15]
        logger.info("starting_ratio_sweep", n_samples=len(sample_indices), ratios=ratios)
        ratio_curve = _ratio_response_curve(
            models, eval_dataset, sample_indices,
            ratios_l0=ratios, target_ratio_l1=float(ratio_l1), device=device,
        )

    # ---- Optional: train holdout zeroed loss ----
    train_holdout = None
    if os.environ.get("BGKIT_DIAG_TRAIN_HOLDOUT", "0") == "1":
        rng = np.random.default_rng(7)
        n_holdout = 200
        train_indices = rng.choice(
            len(train_dataset), size=min(n_holdout, len(train_dataset)),
            replace=False,
        ).tolist()
        sub = Subset(train_dataset, train_indices)
        sub_lengths = np.array(
            [dataset.token_length(train_dataset.indices[i]) for i in train_indices],
            dtype=np.int64,
        )
        sub_sampler = PackedTokenBudgetSampler(
            sub, lengths=sub_lengths, max_batch_tokens=max_batch_tokens_eval,
            shuffle=False,
        )
        sub_loader = DataLoader(
            sub, batch_sampler=sub_sampler, collate_fn=collate_compression,
            num_workers=0,
        )
        logger.info("starting_train_holdout", n=n_holdout)
        train_holdout_per_sample = _per_sample_eval_pass(
            models, sub_loader,
            eval_indices=[train_dataset.indices[i] for i in train_indices],
            target_ratio_l0=float(ratio_l0),
            target_ratio_l1=float(ratio_l1),
            device=device,
            do_zeroed=True,
            limit_batches=None,
        )
        train_holdout = train_holdout_per_sample

    # ---- Aggregate + save ----
    present_losses_arr = np.array([r["present_loss"] for r in per_sample])
    target_token_arr = np.array([r["loss_target_tokens"] for r in per_sample])
    content_token_arr = np.array([r["content_tokens"] for r in per_sample])
    weighted_present = float(
        (present_losses_arr * target_token_arr).sum()
        / max(target_token_arr.sum(), 1)
    )

    summary = {
        "checkpoint": str(ckpt_path),
        "step": step,
        "target_ratio_l0_used": float(ratio_l0),
        "target_ratio_l1_used": float(ratio_l1),
        "split": {
            "total": total,
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
            "seed": cfg.get("seed", 42),
            "train_length_distribution": _length_distribution(train_lengths),
            "eval_length_distribution": _length_distribution(eval_lengths),
        },
        "per_sample_summary": {
            "n": len(per_sample),
            "weighted_present_loss": weighted_present,
            "present_loss_p10": float(np.percentile(present_losses_arr, 10)),
            "present_loss_p50": float(np.percentile(present_losses_arr, 50)),
            "present_loss_p90": float(np.percentile(present_losses_arr, 90)),
            "present_loss_p99": float(np.percentile(present_losses_arr, 99)),
            "present_loss_max": float(present_losses_arr.max()),
            "present_loss_mean": float(present_losses_arr.mean()),
        },
    }
    if "zeroed_loss" in (per_sample[0] if per_sample else {}):
        zeroed_arr = np.array([r["zeroed_loss"] for r in per_sample])
        gap_arr = np.array([r["gap_zeroed"] for r in per_sample])
        n_negative_gap = int((gap_arr < 0).sum())
        n_tiny_gap = int((gap_arr < 0.05).sum())
        weighted_zeroed = float(
            (zeroed_arr * target_token_arr).sum()
            / max(target_token_arr.sum(), 1)
        )
        summary["per_sample_summary"].update({
            "weighted_zeroed_loss": weighted_zeroed,
            "weighted_gap": weighted_zeroed - weighted_present,
            "gap_p10": float(np.percentile(gap_arr, 10)),
            "gap_p50": float(np.percentile(gap_arr, 50)),
            "gap_p90": float(np.percentile(gap_arr, 90)),
            "gap_min": float(gap_arr.min()),
            "gap_max": float(gap_arr.max()),
            "gap_negative_count": n_negative_gap,
            "gap_tiny_count": n_tiny_gap,
        })
        # Top-10 worst (highest present_loss).
        worst_idx = np.argsort(-present_losses_arr)[:10]
        summary["worst_present_loss"] = [
            {
                **per_sample[int(i)],
                "rank": int(rank + 1),
            }
            for rank, i in enumerate(worst_idx)
        ]
        # Top-10 most negative gap.
        worst_gap_idx = np.argsort(gap_arr)[:10]
        summary["worst_gap_zeroed"] = [
            {
                **per_sample[int(i)],
                "rank": int(rank + 1),
            }
            for rank, i in enumerate(worst_gap_idx)
        ]

    if ratio_curve is not None:
        summary["ratio_curve"] = ratio_curve

    if train_holdout is not None:
        train_arr = np.array([r["zeroed_loss"] for r in train_holdout])
        train_present_arr = np.array([r["present_loss"] for r in train_holdout])
        train_token_arr = np.array(
            [r["loss_target_tokens"] for r in train_holdout]
        )
        summary["train_holdout"] = {
            "n": len(train_holdout),
            "weighted_present_loss": float(
                (train_present_arr * train_token_arr).sum()
                / max(train_token_arr.sum(), 1)
            ),
            "weighted_zeroed_loss": float(
                (train_arr * train_token_arr).sum()
                / max(train_token_arr.sum(), 1)
            ),
        }

    summary["per_sample_records"] = per_sample

    out_dir = Path(os.environ.get("CHECKPOINT_DIR", "/workspace/checkpoints")) / "_diagnostics"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"eval_divergence_{int(time.time())}.json"
    out_file.write_text(json.dumps(summary, indent=2))
    logger.info("diag_saved", path=str(out_file))

    # ---- Print summary table ----
    print("\n" + "=" * 78)
    print("STEP 5 EVAL-DIVERGENCE DIAGNOSTIC SUMMARY")
    print("=" * 78)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Step: {step}")
    print(f"Curriculum ratio_l0 used: {ratio_l0:.4f}, ratio_l1 used: {ratio_l1:.4f}")
    print()
    print("Split composition:")
    print(f"  train: n={len(train_dataset)}  mean={train_lengths.mean():.1f}  "
          f"median={np.median(train_lengths):.1f}  p90={np.percentile(train_lengths, 90):.1f}  "
          f"p99={np.percentile(train_lengths, 99):.1f}  max={train_lengths.max()}")
    print(f"  eval:  n={len(eval_dataset)}  mean={eval_lengths.mean():.1f}  "
          f"median={np.median(eval_lengths):.1f}  p90={np.percentile(eval_lengths, 90):.1f}  "
          f"p99={np.percentile(eval_lengths, 99):.1f}  max={eval_lengths.max()}")
    print()
    s = summary["per_sample_summary"]
    print("Per-sample present-loss:")
    print(f"  n={s['n']}  weighted={s['weighted_present_loss']:.4f}  "
          f"mean={s['present_loss_mean']:.4f}")
    print(f"  p10={s['present_loss_p10']:.4f}  p50={s['present_loss_p50']:.4f}  "
          f"p90={s['present_loss_p90']:.4f}  p99={s['present_loss_p99']:.4f}  "
          f"max={s['present_loss_max']:.4f}")
    if "weighted_zeroed_loss" in s:
        print()
        print("Per-sample zeroed-loss / gap:")
        print(f"  weighted_zeroed={s['weighted_zeroed_loss']:.4f}  "
              f"weighted_gap={s['weighted_gap']:.4f}")
        print(f"  gap_min={s['gap_min']:.4f}  gap_p10={s['gap_p10']:.4f}  "
              f"gap_p50={s['gap_p50']:.4f}  gap_p90={s['gap_p90']:.4f}")
        print(f"  gap_negative_count={s['gap_negative_count']}/{s['n']}  "
              f"gap_tiny<0.05={s['gap_tiny_count']}/{s['n']}")

    print()
    print("Top-10 worst (highest present_loss):")
    for r in summary.get("worst_present_loss", []):
        print(f"  rank {r['rank']}: batch {r['batch_idx']}/{r['in_batch_idx']}  "
              f"content_toks={r['content_tokens']}  files={r['n_files']}  "
              f"target_toks={r['loss_target_tokens']}  "
              f"present={r['present_loss']:.4f}  "
              f"zeroed={r.get('zeroed_loss', float('nan')):.4f}  "
              f"gap={r.get('gap_zeroed', float('nan')):.4f}")
    print()
    print("Top-10 most negative gap_zeroed (encoder hurts most):")
    for r in summary.get("worst_gap_zeroed", []):
        print(f"  rank {r['rank']}: batch {r['batch_idx']}/{r['in_batch_idx']}  "
              f"content_toks={r['content_tokens']}  files={r['n_files']}  "
              f"target_toks={r['loss_target_tokens']}  "
              f"present={r['present_loss']:.4f}  "
              f"zeroed={r.get('zeroed_loss', float('nan')):.4f}  "
              f"gap={r.get('gap_zeroed', float('nan')):.4f}")

    if train_holdout is not None:
        th = summary["train_holdout"]
        print()
        print("Train holdout (200 samples):")
        print(f"  weighted_present={th['weighted_present_loss']:.4f}  "
              f"weighted_zeroed={th['weighted_zeroed_loss']:.4f}  "
              f"gap={th['weighted_zeroed_loss'] - th['weighted_present_loss']:.4f}")

    if ratio_curve is not None:
        print()
        print(f"Ratio response curve (30 eval samples, ratios={ratio_curve['ratios']}):")
        # Show mean across samples per ratio.
        ratios = ratio_curve['ratios']
        for ratio in ratios:
            key = f"{ratio:.4f}"
            losses = [d["loss_by_ratio"][key] for d in ratio_curve["data"]]
            print(f"  ratio={ratio:.2f}  mean_loss={np.mean(losses):.4f}  "
                  f"p90={np.percentile(losses, 90):.4f}  "
                  f"max={np.max(losses):.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
