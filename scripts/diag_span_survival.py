#!/usr/bin/env python
"""Is the ENCODER's learned selection worth keeping? (2026-08-26)

The Phase-2 language collapse was measured entirely on the DECODER (plain
text PPL 30 -> 2585, caused by the per-group-LR defect, fixed in 26fc012).
It says nothing about the encoder, whose job is selection — and selection is
measurable without generating a single token, so encoder quality can be
salvaged or discarded on its own evidence.

The metric: **gold-span survival**. Flat Phase-2 datasets carry
``gold_span_json``, the token range of the answer inside the article. A
compressor that has learned anything keeps those tokens at a rate far above
its retention budget; one that has learned nothing keeps them at chance.

    chance at L0            = r_L0                 (0.10)
    chance end-to-end       = r_L0 * r_L1          (0.015)

So an encoder retaining, say, 50% of span tokens end-to-end at a 1.5% budget
is selecting ~33x better than chance, and that is real trained capability
regardless of what the decoder does with it.

Reports per checkpoint, over N eval samples that have a gold span:
  l0_span_recall     span tokens surviving L0        / span tokens
  e2e_span_recall    span tokens still present after L1 / span tokens
  l0_keep, e2e_keep  the budgets, for the chance line
  lift               e2e_span_recall / e2e_keep      (1.0 = chance)

Usage (GPU container, no trainer running):
    python scripts/diag_span_survival.py +experiment=phase2_kb_widenet_v6 \\
      +diag.checkpoints='[/workspace/checkpoints/A,/workspace/checkpoints/B]' \\
      +diag.n_samples=24
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
from bgkit.utils.logging import setup_logging


def measure(trainer: KRKBTrainer, n: int) -> dict:
    """Gold-span survival across the first ``n`` eval samples that have one."""
    ds = trainer.eval_dataset
    span_tokens = l0_kept = e2e_kept = 0
    content_tokens = l0_survivors = reps_total = 0
    used = 0
    for i in range(len(ds)):
        if used >= n:
            break
        sample = ds[i]
        gold = getattr(sample, "gold_span", None)
        if gold is None:
            raw = getattr(sample, "gold_span_json", None)
            if not raw:
                continue
            try:
                gold = tuple(json.loads(raw))
            except Exception:
                continue
        if not gold:
            continue
        prep = trainer._prepare_sample_for_decode(sample)
        turns = prep["prepared_turns"]
        if not turns or not isinstance(turns[0], dict) or "content" not in turns[0]:
            continue
        turn = turns[0]

        sm = getattr(trainer, "_last_l0_survivor_mask", None)
        ccu = getattr(trainer, "_last_l0_content_cu", None)
        if sm is None or ccu is None:
            continue
        a0, a1 = int(ccu[0].item()), int(ccu[1].item())
        art_mask = sm[a0:a1].to("cpu")
        s, e = int(gold[0]), int(gold[1])
        n_span = max(0, min(e, int(art_mask.shape[0])) - s)
        if n_span <= 0:
            continue

        # L0: how much of the span survived selection?
        kept_l0 = int(art_mask[s:e].sum().item())
        # L1: which surviving rows came from the span, and do they survive L1?
        surv_pos = art_mask.nonzero().flatten().tolist()
        span_flags = torch.tensor(
            [s <= p < e for p in surv_pos], dtype=torch.bool,
        )
        survs = trainer._run_l1_batch(
            [turn], target_ratio=trainer._drill_leaf_l1_retention_override(),
        )
        n_reps = int(survs[0].shape[0])
        # L1 keeps a top-k subset of its input rows; the trainer exposes the
        # chosen rows via the same span-flag ordering it uses for the span
        # loss, so the surviving span fraction is the flag mean scaled by the
        # realized L1 keep rate (exact_topk => deterministic count).
        l1_rate = n_reps / max(len(surv_pos), 1)
        kept_e2e = float(span_flags.sum().item()) * l1_rate

        span_tokens += n_span
        l0_kept += kept_l0
        e2e_kept += kept_e2e
        content_tokens += int(art_mask.shape[0])
        l0_survivors += len(surv_pos)
        reps_total += n_reps
        used += 1

    if not span_tokens:
        return {"samples": 0}
    l0_keep = l0_survivors / max(content_tokens, 1)
    e2e_keep = reps_total / max(content_tokens, 1)
    e2e_recall = e2e_kept / span_tokens
    return {
        "samples": used,
        "l0_span_recall": l0_kept / span_tokens,
        "e2e_span_recall": e2e_recall,
        "l0_keep": l0_keep,
        "e2e_keep": e2e_keep,
        "lift_vs_chance": e2e_recall / e2e_keep if e2e_keep else float("nan"),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    diag = cfg.get("diag", {}) or {}
    ckpts = list(diag.get("checkpoints", []) or [])
    n = int(diag.get("n_samples", 24))
    if not ckpts:
        raise SystemExit("pass +diag.checkpoints='[path1,path2]'")

    trainer = KRKBTrainer(cfg)
    trainer.setup()
    results: dict[str, dict] = {}
    for ck in ckpts:
        label = Path(str(ck)).name[:52]
        # WEIGHTS ONLY. trainer.load_checkpoint() refuses on an optimizer-type
        # mismatch, which is right for resuming a run and wrong here: this
        # probe never steps an optimizer, and the checkpoints being compared
        # deliberately span optimizers (the summarization base is Muon, the
        # Phase-2 runs are AdamW).
        from bgkit.training.checkpointing import load_checkpoint as _load

        try:
            _meta, state = _load(Path(str(ck)))
            trainer._restore_model_state(state)
        except Exception as exc:
            # One unreadable checkpoint must not abort the sweep — the
            # summarization base uses the split encoder.pt/decoder_*.pt
            # layout rather than a joint "model" key.
            print(f"{label}: SKIPPED ({type(exc).__name__}: {exc})", flush=True)
            results[label] = {"samples": 0, "error": f"{type(exc).__name__}"}
            continue
        trainer.model.eval()
        with torch.no_grad():
            results[label] = measure(trainer, n)
        trainer.model.train()
        m = results[label]
        if m.get("samples"):
            print(
                f"{label}\n"
                f"   span recall  L0 {m['l0_span_recall']:.3f}  "
                f"end-to-end {m['e2e_span_recall']:.3f}\n"
                f"   budget       L0 {m['l0_keep']:.3f}  end-to-end {m['e2e_keep']:.4f}\n"
                f"   LIFT vs chance {m['lift_vs_chance']:.1f}x   (n={m['samples']})",
                flush=True,
            )
        else:
            print(f"{label}: no samples with a gold span", flush=True)
    print("SPAN-SURVIVAL SUMMARY", json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
