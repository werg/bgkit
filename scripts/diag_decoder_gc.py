"""Diagnose whether decoder gradient checkpointing corrupts gradients.

The 2026-06-10 finding: enabling decoder GC (reentrant) bounded memory but the
loss degraded 0.8 -> 3.7 (both decoders), implying wrong gradients flowing back
to the shared encoder. Hypothesis: GC recomputes the decoder forward in
backward, but the decoder kernels (FLA Gated-DeltaNet for Qwen3.5, Mamba for
Falcon-H1) are NON-DETERMINISTIC, so recompute != original forward -> wrong
grad. If true, NO checkpointing mode is viable and we must lower the ratio.

Three measurements on the REAL encode+decode path, all with identical RNG seed
(so any difference is kernel non-determinism, not dropout):
  A1, A2) forward+backward TWICE with GC OFF   -> baseline determinism
  B)      forward+backward with GC ON reentrant -> compare grad to A1
Gradient measured w.r.t. the (detached, leaf) survivor embeddings.

Verdict:
  A1~=A2 and B~=A1 -> GC CORRECT (loss regression was elsewhere; GC usable).
  A1~=A2 but B!=A1 -> GC recompute-path bug (reentrant / checkpoint mechanism).
  A1!=A2           -> non-deterministic kernels; GC can't match -> lower ratio.

Run inside the training container (GPU). Single batch; small footprint.
"""

from __future__ import annotations

import argparse

import torch
from hydra import compose, initialize_config_dir

from bgkit.training.gradient_utils import _enable_gradient_checkpointing_mode
from bgkit.training.phase1.summarization_round_robin import (
    SummarizationRoundRobinTrainer,
)

CONFIGS = "/workspace/bgkit/configs"


def _seed():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


def _build_loss_mask(prefix_ids, suffix_masks, per_group):
    full = []
    for i, (pre, sm) in enumerate(zip(prefix_ids, suffix_masks, strict=True)):
        zeros_pre = torch.zeros(pre.shape[0], dtype=torch.bool, device=sm.device)
        zeros_surv = torch.zeros(int(per_group[i]), dtype=torch.bool, device=sm.device)
        full.append(torch.cat([zeros_pre, zeros_surv, sm]))
    return torch.cat(full)


def _run_fwd_bwd(decoder, se_values, group_cu, prefix_ids, suffix_ids, loss_mask):
    """One forward+backward on a FRESH survivor-embedding leaf; return (loss, grad)."""
    _seed()
    se = se_values.detach().clone().requires_grad_(True)
    out = decoder.forward_with_single_splice(
        survivor_embeddings=se,
        survivor_cu_seqlens=group_cu,
        prefix_ids=prefix_ids,
        suffix_ids=suffix_ids,
        loss_mask=loss_mask,
    )
    loss = out.loss if hasattr(out, "loss") else out
    decoder.zero_grad(set_to_none=True)
    loss.backward()
    return float(loss.item()), se.grad.detach().clone()


def _cmp(name, g_ref, g_test):
    diff = (g_test - g_ref).abs()
    denom = g_ref.abs().clamp(min=1e-8)
    rel = (diff / denom).mean().item()
    print(
        f"  [{name}] max_abs_diff={diff.max().item():.3e} "
        f"mean_abs_diff={diff.mean().item():.3e} mean_rel_diff={rel:.3e} "
        f"ref_absmean={g_ref.abs().mean().item():.3e}",
        flush=True,
    )
    return diff.max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="falcon_h1", choices=["falcon_h1", "qwen35"])
    ap.add_argument("--step", type=int, default=11030)
    args = ap.parse_args()

    overrides = [
        "+experiment=phase1_summarization_round_robin",
        "++training.max_eval_samples=8",
        "++compute.num_workers=0",
        "++compute.pin_memory=false",
    ]
    with initialize_config_dir(version_base=None, config_dir=CONFIGS):
        cfg = compose(config_name="config", overrides=overrides)

    print(f"[diag] building trainer, family={args.family} step={args.step}", flush=True)
    trainer = SummarizationRoundRobinTrainer(cfg)
    trainer.setup()
    trainer.global_step = args.step
    fam = args.family
    trainer.encoder.eval()
    trainer.encoder.set_active_decoder_family(fam)
    decoder = trainer.decoder_qwen if fam == "qwen35" else trainer.decoder_falcon

    # One eval batch through the real encode path.
    batch = next(iter(trainer.eval_dataloader))
    prefix_ids, suffix_ids, suffix_masks, comp_prompt_ids = trainer._build_chat_inputs(fam, batch)
    with torch.no_grad():
        enc_out, group_cu, per_group, ratio, ratio_l1, *_ = trainer._encode_batch(
            batch, comp_prompt_ids,
        )
    loss_mask = _build_loss_mask(prefix_ids, suffix_masks, per_group)
    se_values = enc_out.survivor_embeddings.detach().clone()
    enc_out.release()
    print(
        f"[diag] survivors={se_values.shape[0]} dim={se_values.shape[1]} "
        f"ratio={ratio} groups={len(per_group)}",
        flush=True,
    )

    # Decoder in train() mode = the real training path (Falcon Mamba gates on it;
    # GC only checkpoints in train()).
    decoder.train()

    # --- A) GC OFF, twice (baseline determinism) ---
    decoder.backbone.gradient_checkpointing_disable()
    lossA1, gradA1 = _run_fwd_bwd(decoder, se_values, group_cu, prefix_ids, suffix_ids, loss_mask)
    lossA2, gradA2 = _run_fwd_bwd(decoder, se_values, group_cu, prefix_ids, suffix_ids, loss_mask)
    print(f"[diag] GC-OFF lossA1={lossA1:.5f} lossA2={lossA2:.5f}", flush=True)
    print("[diag] === A2 vs A1 (baseline forward+backward determinism) ===", flush=True)
    det_max = _cmp("det", gradA1, gradA2)

    # --- B) GC ON (reentrant) ---
    _enable_gradient_checkpointing_mode(decoder.backbone, "reentrant")
    lossB, gradB = _run_fwd_bwd(decoder, se_values, group_cu, prefix_ids, suffix_ids, loss_mask)
    print(f"[diag] GC-REENTRANT lossB={lossB:.5f}", flush=True)
    print("[diag] === B vs A1 (GC-on vs GC-off gradients) ===", flush=True)
    gc_max = _cmp("gc", gradA1, gradB)

    print("\n[diag] ============ VERDICT ============", flush=True)
    print(f"[diag] baseline determinism max|A2-A1| = {det_max:.3e}", flush=True)
    print(f"[diag] GC vs no-GC      max|B-A1|  = {gc_max:.3e}", flush=True)
    print(f"[diag] loss: A1={lossA1:.5f} A2={lossA2:.5f} B={lossB:.5f}", flush=True)
    tol = 1e-3
    if det_max > tol:
        print("[diag] -> NON-DETERMINISTIC kernels (A2!=A1). GC cannot match the "
              "original forward -> ANY checkpointing corrupts grads. FIX: lower ratio "
              "(no GC).", flush=True)
    elif gc_max > tol:
        print("[diag] -> Kernels deterministic, but GC grad DIFFERS from no-GC. "
              "Recompute-path bug (reentrant impl / checkpoint mechanism) -> fixable.",
              flush=True)
    else:
        print("[diag] -> GC grads MATCH no-GC. Decoder GC is CORRECT; the loss "
              "regression came from something else -> re-examine.", flush=True)


if __name__ == "__main__":
    main()
