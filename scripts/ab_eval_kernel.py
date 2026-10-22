"""A/B eval for the summarization round-robin run.

Two modes of investigation, sharing one checkpoint-load path:

1. **Survivor-embedding ablation** (default) — the representation-quality probe.
   For a checkpoint, run the deterministic eval set several times, each time
   transforming the *survivor embeddings* (the compressed distributed
   representation) before they reach the decoder:
     - real    : untouched (baseline CE)
     - zero     : survivor embeddings zeroed — does the decoder use them at all,
                  or is it reconstructing from the prefix/its own prior?
     - shuffle  : survivor rows globally permuted (same per-group counts, wrong
                  content) — does the *specific* content matter, or just that
                  *some* vectors are present?
     - noise    : Gaussian noise matched to the survivor std — does it need
                  structured content vs any vector of the right magnitude?
   The CE delta of each ablation vs `real` measures how load-bearing the
   compressed representation actually is. Small deltas => the representation is
   not informative / the decoder shortcuts; large deltas => it is doing the work.

2. **Kernel A/B** (--ckpts default-list) — original purpose: compare fixed
   checkpoints under the current fla kernel to detect a survivor-selection shift
   from the 06-08 forward-kernel change.

Run inside the training container (shares GPU); keep --max-eval small to bound
the concurrent memory footprint.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from bgkit.training.checkpointing import load_checkpoint
from bgkit.training.phase1.summarization_round_robin import (
    SummarizationRoundRobinTrainer,
)

CONFIGS = "/workspace/bgkit/configs"

# (label, checkpoint_path, global_step_to_simulate) — kernel-A/B fallback list.
DEFAULT_CKPTS = [
    (
        "step3040_PREFIX",
        "/workspace/checkpoints/phase1_summarization_round_robin_step3040_20260607_200112",
        3040,
    ),
    (
        "step5000_POSTFIX",
        "/workspace/checkpoints/phase1_summarization_round_robin_step5000_20260608_134303",
        5000,
    ),
]


def _ablate_survivors(se: torch.Tensor, mode: str, seed: int) -> torch.Tensor:
    """Transform the packed survivor embeddings (N_surv_total, D) for an ablation.

    Group boundaries (survivor_cu_seqlens) and per-group counts are unchanged —
    only the *content* of the embeddings is altered, isolating "is the compressed
    representation informative" from "how many / which positions survive".
    """
    if mode == "real":
        return se
    torch.manual_seed(seed)
    if mode == "zero":
        return torch.zeros_like(se)
    if mode == "noise":
        # zero-mean Gaussian matched to the global survivor std (preserve dtype)
        std = se.float().std().clamp(min=1e-6).to(se.dtype)
        return torch.randn_like(se) * std
    if mode == "shuffle":
        # global row permutation: each group keeps its count but gets wrong content
        perm = torch.randperm(se.shape[0], device=se.device)
        return se[perm]
    raise ValueError(f"unknown ablation mode: {mode}")


@torch.no_grad()
def eval_checkpoint(
    trainer: SummarizationRoundRobinTrainer,
    step: int,
    ablate: str = "real",
    seed: int = 0,
) -> dict:
    """Mirror of trainer.evaluate() but tallies survivors + applies a survivor
    ablation transform before the decoder splice."""
    trainer.global_step = step
    trainer.encoder.eval()
    trainer.decoder_qwen.eval()
    trainer.decoder_falcon.eval()
    agg = {
        "qwen35": {"loss": 0.0, "n": 0, "surv": 0, "src": 0, "ratio": None},
        "falcon_h1": {"loss": 0.0, "n": 0, "surv": 0, "src": 0, "ratio": None},
    }
    batch_idx = 0
    for batch in trainer.eval_dataloader:
        family = "qwen35" if (batch_idx % 2 == 0) else "falcon_h1"
        batch_idx += 1
        trainer.encoder.set_active_decoder_family(family)
        decoder = (
            trainer.decoder_qwen if family == "qwen35" else trainer.decoder_falcon
        )
        prefix_ids, suffix_ids, suffix_masks, comp_prompt_ids = trainer._build_chat_inputs(
            family, batch,
        )
        enc_out, group_cu, per_group, ratio, ratio_l1, *_ = trainer._encode_batch(
            batch, comp_prompt_ids,
        )
        full_masks = []
        for sample_i, (pre, sm) in enumerate(
            zip(prefix_ids, suffix_masks, strict=True)
        ):
            zeros_pre = torch.zeros(pre.shape[0], dtype=torch.bool, device=sm.device)
            zeros_surv = torch.zeros(
                int(per_group[sample_i]), dtype=torch.bool, device=sm.device
            )
            full_masks.append(torch.cat([zeros_pre, zeros_surv, sm]))
        survivor_embeddings = _ablate_survivors(
            enc_out.survivor_embeddings, ablate, seed + batch_idx
        )
        out = decoder.forward_with_single_splice(
            survivor_embeddings=survivor_embeddings,
            survivor_cu_seqlens=group_cu,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            loss_mask=torch.cat(full_masks),
        )
        loss_val = out.loss if hasattr(out, "loss") else out
        src_per_group = [sum(int(d.shape[0]) for d in docs) for docs in batch["source_docs"]]
        a = agg[family]
        a["loss"] += float(loss_val.item())
        a["n"] += 1
        a["surv"] += int(sum(per_group))
        a["src"] += int(sum(src_per_group))
        e2e = ratio * (ratio_l1 if (ratio_l1 is not None and ratio_l1 > 0) else 1.0)
        a["ratio"] = e2e
        enc_out.release()

    res = {}
    for fam, a in agg.items():
        if a["n"] == 0:
            continue
        res[fam] = {
            "loss": a["loss"] / a["n"],
            "mean_surv_per_group": a["surv"] / max(1, a["n"]),
            "achieved_keep": a["surv"] / max(1, a["src"]),
            "target_ratio_e2e": a["ratio"],
            "n_batches": a["n"],
        }
    if "qwen35" in res and "falcon_h1" in res:
        res["loss"] = 0.5 * (res["qwen35"]["loss"] + res["falcon_h1"]["loss"])
    elif res:
        res["loss"] = next(iter(res.values()))["loss"]
    return res


def _latest_ckpt() -> str | None:
    base = os.environ.get("CHECKPOINT_DIR", "/workspace/checkpoints")
    cands = [
        c
        for c in glob.glob(
            os.path.join(base, "phase1_summarization_round_robin_step*")
        )
        if os.path.isdir(c)
    ]
    if not cands:
        return None

    def stepnum(p: str) -> int:
        m = re.search(r"step(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1

    return max(cands, key=stepnum)


def _stepnum(path: str) -> int:
    m = re.search(r"step(\d+)", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=32)
    ap.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="checkpoint dir to ablate; default = latest summarization checkpoint",
    )
    ap.add_argument(
        "--step",
        type=int,
        default=None,
        help="curriculum step to simulate; default = parsed from ckpt name",
    )
    ap.add_argument(
        "--modes",
        type=str,
        default="real,zero,shuffle,noise",
        help="comma list of survivor-ablation modes",
    )
    ap.add_argument(
        "--kernel-ab",
        action="store_true",
        help="run the original fixed-checkpoint kernel A/B instead of ablation",
    )
    args = ap.parse_args()

    overrides = [
        "+experiment=phase1_summarization_round_robin",
        f"++training.max_eval_samples={args.max_eval}",
        "++compute.num_workers=0",
        "++compute.pin_memory=false",
    ]
    with initialize_config_dir(version_base=None, config_dir=CONFIGS):
        cfg = compose(config_name="config", overrides=overrides)

    print(f"[ab_eval] building trainer (max_eval={args.max_eval}) ...", flush=True)
    trainer = SummarizationRoundRobinTrainer(cfg)
    trainer.setup()
    f, _ = torch.cuda.mem_get_info()
    print(f"[ab_eval] post-setup cuda_free={f/1e9:.1f}GB", flush=True)

    if args.kernel_ab:
        for label, path, step in DEFAULT_CKPTS:
            print(f"\n[ab_eval] === {label}  (ckpt step={step}) ===", flush=True)
            _meta, sds = load_checkpoint(Path(path))
            trainer._restore_model_state(sds)
            res = eval_checkpoint(trainer, step, ablate="real")
            for fam in ("qwen35", "falcon_h1"):
                r = res[fam]
                print(
                    f"  {fam:10s} loss={r['loss']:.4f}  "
                    f"mean_surv/grp={r['mean_surv_per_group']:.1f}  "
                    f"achieved_keep={r['achieved_keep']:.4f}  "
                    f"target_e2e={r['target_ratio_e2e']:.4f}",
                    flush=True,
                )
            print(f"  COMBINED loss={res['loss']:.4f}", flush=True)
        return

    # Survivor-embedding ablation on a single checkpoint.
    ckpt = args.ckpt or _latest_ckpt()
    if ckpt is None:
        raise SystemExit("no checkpoint found; pass --ckpt")
    step = args.step if args.step is not None else _stepnum(ckpt)
    print(f"\n[ab_eval] ablation target: {os.path.basename(ckpt)} (step={step})", flush=True)
    _meta, sds = load_checkpoint(Path(ckpt))
    trainer._restore_model_state(sds)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = {}
    for mode in modes:
        res = eval_checkpoint(trainer, step, ablate=mode, seed=1234)
        results[mode] = res
        line = f"  [{mode:8s}] combined_loss={res['loss']:.4f}"
        for fam in ("qwen35", "falcon_h1"):
            if fam in res:
                line += f"  {fam}={res[fam]['loss']:.4f}"
        if mode == "real" and "qwen35" in res:
            line += (
                f"  | keep={res['qwen35']['achieved_keep']:.3f}"
                f"/target={res['qwen35']['target_ratio_e2e']:.3f}"
                f"  surv/grp={res['qwen35']['mean_surv_per_group']:.0f}"
            )
        print(line, flush=True)
        f, _ = torch.cuda.mem_get_info()
        print(f"    [mem] cuda_free={f/1e9:.1f}GB", flush=True)

    if "real" in results:
        base = results["real"]["loss"]
        print("\n[ab_eval] === CE delta vs real (representation informativeness) ===", flush=True)
        for mode in modes:
            if mode == "real":
                continue
            d = results[mode]["loss"] - base
            print(
                f"  {mode:8s}: Δ={d:+.4f}  ({results[mode]['loss']:.4f} vs {base:.4f})",
                flush=True,
            )
        print(
            "\n  Interpretation: small Δ on zero/shuffle => representation is NOT "
            "load-bearing (decoder shortcuts). Large Δ => it carries the signal.",
            flush=True,
        )


if __name__ == "__main__":
    main()
