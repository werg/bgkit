#!/usr/bin/env python3
"""Offline distillation: ICE → head_base_l0 (tanh-bounded operator regime).

Runs a dedicated BCE-with-logits pass against ICE top-k teacher targets to
initialize ``compressor.head_base_l0`` before Phase 1 Step 3 starts. Frees
Step 3 from doing this work while simultaneously trying to teach the
decoder to use BgKIT output.

What's trained:
    Only ``compressor.head_base_l0.*``. Everything else (pruned backbone,
    projection block, adapter head, flag embeddings, etc.) is frozen. The
    script taps the pruned backbone at block 1 (≡ layer 7 of the unpruned
    Qwen3.5-0.8B) via ``return_intermediates=True`` and feeds that hidden
    state into ``head_base_l0`` for BCE against ICE's top-k mask.

Architecture note: ``SurvivorshipHead.forward`` returns raw (pre-tanh)
logits. Tanh is applied at composition inside the compressor hook so the
operator-facing logit is bounded. During distillation we train on raw
logits so BCE-with-logits is numerically stable. At runtime the trained
head outputs a raw distribution that tanh bounds naturally — if BCE
targets drive raw logits into ~±3, tanh saturates lightly at ±0.99, which
is the sweet spot for the θ ∈ (-0.99, 0.99) operator.

What's saved:
    A standalone sidecar checkpoint at ``$CHECKPOINT_DIR/survivorship_head_base_l0_YYYYMMDD_HHMMSS/``
    containing ONLY head_base_l0 weights, loadable into a Step 3 encoder
    via a partial state-dict update. The encoder state dict from the Step 2
    checkpoint is NOT modified.

Usage:
    python scripts/pretrain_survivorship_head.py \
        --step2-checkpoint $CHECKPOINT_DIR/phase1_step2_step19999_20260323_132922 \
        --ice-checkpoint $CHECKPOINT_DIR/ice_step89999_20260306_213825 \
        --data-dir $DATA_DIR/mmap/code \
        --output-dir $CHECKPOINT_DIR \
        --teacher-ratio 0.10 --max-steps 3000 --max-batch-tokens 16384

The script intentionally does NOT touch the head_adapter_l0 or the
threshold controller — these are initialized fresh by Step 3. It only
provides a warm start for the ICE-aligned ranking signal that BCE would
have installed online.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


logger = logging.getLogger("pretrain_survivorship_head")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step2-checkpoint", type=Path, required=True,
                        help="Phase 1 Step 2 checkpoint (pruned encoder).")
    parser.add_argument("--ice-checkpoint", type=Path, required=True,
                        help="Frozen ICE checkpoint used as teacher.")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="MmapTokenDataset directory.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Parent directory for saved sidecar checkpoint.")
    parser.add_argument("--teacher-ratio", type=float, default=0.10,
                        help="ICE top-k ratio for the teacher mask.")
    parser.add_argument("--max-steps", type=int, default=3000,
                        help="Number of optimizer steps.")
    parser.add_argument("--max-batch-tokens", type=int, default=16384,
                        help="Token budget per microbatch.")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="AdamW LR for head_base_l0. This is small head, "
                             "can afford higher LR than Step 3's encoder_lr.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--backbone", type=str,
                        default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--ice-input-dim", type=int, default=1024)
    parser.add_argument("--ice-hidden-dim", type=int, default=192)
    parser.add_argument("--ice-num-layers", type=int, default=3)
    parser.add_argument("--ice-kernel-size", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    _setup_logging()
    args = parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, random_split

    from bgkit.data.datasets.mmap_token_dataset import MmapTokenDataset
    from bgkit.data.samplers import TokenBudgetBatchSampler
    from bgkit.models.encoder import BgKITEncoder
    from bgkit.models.ice_teacher import ICETeacher
    from bgkit.training.checkpointing import load_checkpoint
    from bgkit.utils.attention_backend import resolve_attention_implementation

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    # --- Encoder (frozen except head_base_l0) ---
    logger.info("loading_step2_checkpoint path=%s", args.step2_checkpoint)
    _metadata, state_dicts = load_checkpoint(args.step2_checkpoint)
    if "encoder" not in state_dicts:
        logger.error("step2 checkpoint missing 'encoder' key: %s",
                     list(state_dicts.keys()))
        return 2

    encoder = BgKITEncoder.from_pretrained_with_state_dict(
        args.backbone,
        state_dicts["encoder"],
        hidden_dim=args.hidden_dim,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=resolve_attention_implementation(),
    ).to(device)

    # Freeze everything; only head_base_l0 gets grad.
    encoder.requires_grad_(False)
    encoder.eval()
    head = encoder.compressor.head_base_l0
    head.requires_grad_(True)
    head.train()

    head_params = [p for p in head.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in head_params)
    logger.info("trainable_params head_base_l0=%d", n_trainable)

    # --- ICE teacher ---
    logger.info("loading_ice path=%s", args.ice_checkpoint)
    embed_tokens = encoder.compressor.backbone.get_input_embeddings()
    ice_teacher = ICETeacher(
        args.ice_checkpoint, embed_tokens,
        input_dim=args.ice_input_dim,
        hidden_dim=args.ice_hidden_dim,
        num_layers=args.ice_num_layers,
        kernel_size=args.ice_kernel_size,
    ).to(device)
    ice_teacher.eval()

    # --- Dataset ---
    logger.info("loading_dataset dir=%s", args.data_dir)
    full_dataset = MmapTokenDataset(
        str(args.data_dir), max_seq_len=args.max_seq_len,
        include_metadata=False,
    )
    if len(full_dataset) == 0:
        logger.error("dataset_empty dir=%s", args.data_dir)
        return 3
    eval_size = min(max(1, int(len(full_dataset) * 0.02)), 500)
    train_size = len(full_dataset) - eval_size
    train_dataset, eval_dataset = random_split(
        full_dataset, [train_size, eval_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    lengths_all = full_dataset.lengths
    train_lengths = lengths_all[np.array(train_dataset.indices)]
    eval_lengths = lengths_all[np.array(eval_dataset.indices)]

    train_sampler = TokenBudgetBatchSampler(
        train_lengths, args.max_batch_tokens, shuffle=True, seed=args.seed,
    )
    eval_sampler = TokenBudgetBatchSampler(
        eval_lengths, args.max_batch_tokens, shuffle=False,
    )

    def _collate(batch):
        """Pad to longest in-batch for token_ids + attn_mask."""
        ids_list = []
        mask_list = []
        for item in batch:
            if isinstance(item, dict):
                tid = item["token_ids"]
            else:
                tid = item
            ids_list.append(tid)
            mask_list.append(torch.ones_like(tid))
        max_len = max(t.size(0) for t in ids_list)
        padded_ids = torch.zeros(len(ids_list), max_len, dtype=ids_list[0].dtype)
        padded_mask = torch.zeros(len(ids_list), max_len, dtype=torch.bool)
        for i, (tid, m) in enumerate(zip(ids_list, mask_list, strict=True)):
            n = tid.size(0)
            padded_ids[i, :n] = tid
            padded_mask[i, :n] = True
        return {"token_ids": padded_ids, "attention_mask": padded_mask}

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=_collate,
        num_workers=args.num_workers,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_sampler=eval_sampler,
        collate_fn=_collate,
        num_workers=args.num_workers,
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        head_params, lr=args.lr, weight_decay=args.weight_decay,
    )

    # --- Training loop ---
    # Reach the encoder's layer-7 (pruned: block 1) hidden state by running
    # only the compressor backbone up to that point. We can't reuse the head
    # hook because we're training head_base_l0 itself; instead, we tap
    # after-block-1 via `forward_from_block` convention. The simplest and
    # safest approach is to run compressor.forward(...) WITHOUT a target
    # ratio (compression_off → hook doesn't fire) and capture the output
    # block-1 hidden via `return_intermediates=True`, then feed that into
    # head_base_l0 ourselves.

    def _run_one(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute BCE loss + two diagnostic stats.

        Returns (loss, base_raw.detach(), teacher_mask.detach()).
        """
        token_ids = batch["token_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Teacher target (no grad).
        with torch.no_grad():
            teacher = ice_teacher.teacher_mask(
                token_ids, attention_mask.int(), args.teacher_ratio,
            )

        # Run backbone up to block 1, then apply head_base_l0.
        # Pruned backbone: block 1 output = hidden after 2nd block.
        # Use return_intermediates to collect post-block states.
        bgkit_embed = embed_tokens
        input_emb = bgkit_embed(token_ids)

        with torch.no_grad():
            # target_ratio=None disables the hook; we need manual block-1 tap.
            # PrunedBidirectionalQwen35.forward(...) with return_intermediates
            # gives us hidden after each block.
            backbone = encoder.compressor.backbone
            backbone_out = backbone(
                inputs_embeds=input_emb,
                attention_mask=attention_mask,
                return_intermediates=True,
            )
            # intermediates[1] = hidden after block 1 (for pruned),
            # which matches where the head hook fires.
            intermediates = backbone_out.hidden_states
            if intermediates is None or len(intermediates) < 2:
                raise RuntimeError(
                    "backbone did not return enough intermediate states; "
                    f"got {len(intermediates) if intermediates is not None else 0}",
                )
            layer7 = intermediates[1]

        # Ensure head input dtype matches head param dtype. Layer7 may be bf16.
        head_in = layer7.to(head.head[0].weight.dtype)
        base_raw = head(head_in)  # (B, L)

        # BCE (stable).
        bce_per_pos = torch.nn.functional.binary_cross_entropy_with_logits(
            base_raw.float(), teacher.float(), reduction="none",
        )
        valid_f = attention_mask.float()
        bce = (bce_per_pos * valid_f).sum() / valid_f.sum().clamp(min=1)
        return bce, base_raw.detach(), teacher.detach()

    def _run_eval() -> dict[str, float]:
        head.eval()
        losses = []
        tpr_num = 0  # teacher positives correctly predicted (by base_raw > 0)
        tpr_den = 0
        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                if i >= 20:
                    break
                loss, base, teacher = _run_one(batch)
                losses.append(float(loss.item()))
                preds = (base.float() > 0.0).to(teacher.dtype)
                tpr_num += int(((preds * teacher).sum()).item())
                tpr_den += int(teacher.sum().item())
        head.train()
        out = {
            "eval_loss": sum(losses) / max(len(losses), 1),
        }
        if tpr_den > 0:
            out["eval_teacher_recall_at_0"] = tpr_num / tpr_den
        return out

    logger.info(
        "starting_distillation max_steps=%d teacher_ratio=%.3f lr=%g",
        args.max_steps, args.teacher_ratio, args.lr,
    )
    step = 0
    running_loss = 0.0
    running_n = 0
    train_iter = iter(train_loader)
    started_at = dt.datetime.utcnow()
    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        for _micro in range(args.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            loss, _, _ = _run_one(batch)
            (loss / args.grad_accum).backward()
            running_loss += float(loss.item())
            running_n += 1
        torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
        optimizer.step()
        step += 1
        if step % args.log_every == 0:
            avg = running_loss / max(running_n, 1)
            logger.info("step=%d loss=%.4f", step, avg)
            running_loss = 0.0
            running_n = 0
        if step % args.eval_every == 0 or step == args.max_steps:
            eval_metrics = _run_eval()
            logger.info("eval step=%d %s", step, eval_metrics)

    elapsed = dt.datetime.utcnow() - started_at
    logger.info("done elapsed=%s", elapsed)

    # --- Save sidecar ---
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"survivorship_head_base_l0_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    head_state = {
        f"compressor.head_base_l0.{k}": v.detach().cpu()
        for k, v in head.state_dict().items()
    }
    torch.save(head_state, out_dir / "head_base_l0.pt")
    (out_dir / "meta.json").write_text(json.dumps({
        "step2_checkpoint": str(args.step2_checkpoint),
        "ice_checkpoint": str(args.ice_checkpoint),
        "teacher_ratio": args.teacher_ratio,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "backbone": args.backbone,
        "timestamp_utc": timestamp,
    }, indent=2))
    logger.info("saved_sidecar path=%s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
