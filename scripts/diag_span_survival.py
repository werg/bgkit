#!/usr/bin/env python
"""Is the ENCODER's learned selection worth keeping? (2026-08-26, rebuilt 08-27)

The Phase-2 language collapse was measured entirely on the DECODER (plain
text PPL 30 -> 2585, caused by the per-group-LR defect, fixed in 26fc012).
It says nothing about the encoder, whose job is selection — and selection is
measurable without generating a single token, so encoder quality can be
salvaged or discarded on its own evidence.

The metric: **gold-span survival**. Flat Phase-2 datasets carry
``gold_span_json``, the token range of the answer inside the article (verified
end-to-end by ``scripts/verify_gold_span_alignment.py``: 600/600 spans decode
to the gold answer). A compressor that has learned anything keeps those tokens
at a rate above its retention budget; one that has learned nothing keeps them
at chance:

    chance at L0            = r_L0                 (0.10)
    chance end-to-end       = r_L0 * r_L1          (0.015)

THE FIRST VERSION OF THIS SCRIPT REPORTED CHANCE FOR EVERY CHECKPOINT, AND
WAS WRONG. Three defects, all of which this version fixes, because each is a
trap any future selection probe can fall into:

1. **It measured one dataset.** It walked the eval set in order and stopped at
   the first ``n`` samples carrying a span. ``lognav`` sorts first and its
   spans are ~5x longer than every other dataset's (median 72 tokens vs
   11-18), so "the encoder" was really "lognav, the hardest case". This
   version STRATIFIES: an equal quota per dataset, reported per dataset.
2. **It pooled by token.** Summing kept/total across samples lets a handful of
   long spans own the number. The in-training diagnostic averages per SAMPLE,
   so the two could not be compared, and they disagreed ~5x. This version
   reports BOTH, labelled — ``per_sample`` is the one comparable to
   ``l0_span_survival`` in the training logs.
3. **Its end-to-end number was an assumption, not a measurement.** It scaled
   the L0 recall by the uniform L1 keep rate, so ``e2e_span_recall`` was
   algebraically ``l0_recall * l1_rate`` and its "lift" was just the L0 lift
   restated. This version reads the REAL L1 survivor mask via the
   ``_last_l1_*`` stash added to ``_run_l1_batch``.

Reports per checkpoint, per dataset, and by span-length bucket:
  l0_span_recall   fraction of span tokens surviving L0   (per_sample + pooled)
  e2e_span_recall  fraction still present after L1        (measured)
  lift             recall / budget                        (1.0 = chance)

Usage (GPU container, no trainer running):
    python scripts/diag_span_survival.py +experiment=phase2_kb_widenet_v6 \\
      +diag.checkpoints='[/workspace/checkpoints/A,/workspace/checkpoints/B]' \\
      +diag.per_dataset=40
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
from bgkit.utils.logging import setup_logging

# Span-length buckets. lognav's ~72-token spans behave differently from
# fileneedle's ~12-token ones, and pooling them hid exactly that.
BUCKETS = ((0, 16), (16, 32), (32, 64), (64, 1 << 30))


def _bucket(n: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= n < hi:
            return f"{lo}-{hi if hi < (1 << 30) else 'inf'}"
    return "?"


class _Acc:
    """Accumulates both weightings so they can never be conflated again."""

    __slots__ = (
        "content_tok",
        "e2e_rates",
        "e2e_tok",
        "l0_rates",
        "l0_surv",
        "l0_tok",
        "l1_surv",
        "span_tok",
    )

    def __init__(self) -> None:
        self.l0_rates: list[float] = []
        self.e2e_rates: list[float] = []
        self.span_tok = self.l0_tok = self.e2e_tok = 0
        self.content_tok = self.l0_surv = self.l1_surv = 0

    def add(self, n_span: int, kept_l0: int, kept_e2e: int,
            n_content: int, n_l0: int, n_l1: int) -> None:
        self.l0_rates.append(kept_l0 / n_span)
        self.e2e_rates.append(kept_e2e / n_span)
        self.span_tok += n_span
        self.l0_tok += kept_l0
        self.e2e_tok += kept_e2e
        self.content_tok += n_content
        self.l0_surv += n_l0
        self.l1_surv += n_l1

    def report(self) -> dict:
        n = len(self.l0_rates)
        if not n:
            return {"samples": 0}
        l0_keep = self.l0_surv / max(self.content_tok, 1)
        e2e_keep = self.l1_surv / max(self.content_tok, 1)
        per_l0 = sum(self.l0_rates) / n
        per_e2e = sum(self.e2e_rates) / n
        return {
            "samples": n,
            # Comparable to the training log's l0_span_survival / l1_span_survival.
            "l0_span_recall_per_sample": per_l0,
            "e2e_span_recall_per_sample": per_e2e,
            # Token-pooled: dominated by the longest spans, kept for contrast.
            "l0_span_recall_pooled": self.l0_tok / self.span_tok,
            "e2e_span_recall_pooled": self.e2e_tok / self.span_tok,
            "l0_keep": l0_keep,
            "e2e_keep": e2e_keep,
            "l0_lift": per_l0 / l0_keep if l0_keep else float("nan"),
            "e2e_lift": per_e2e / e2e_keep if e2e_keep else float("nan"),
            "mean_span_tokens": self.span_tok / n,
        }


def measure(trainer: KRKBTrainer, per_dataset: int) -> dict:
    """Gold-span survival, stratified across datasets."""
    ds = trainer.eval_dataset
    overall = _Acc()
    by_ds: dict[str, _Acc] = defaultdict(_Acc)
    by_len: dict[str, _Acc] = defaultdict(_Acc)
    quota: dict[str, int] = defaultdict(int)

    for i in range(len(ds)):
        sample = ds[i]
        name = str(getattr(sample, "dataset_name", "?"))
        if quota[name] >= per_dataset:
            continue
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
        # Gold article is index 0: _prepare_l1_turn builds all_ids as
        # article_ids + distractors.
        a0, a1 = int(ccu[0].item()), int(ccu[1].item())
        art_mask = sm[a0:a1].to("cpu")
        s, e = int(gold[0]), int(gold[1])
        e = min(e, int(art_mask.shape[0]))
        n_span = max(0, e - s)
        if n_span <= 0:
            continue
        kept_l0 = int(art_mask[s:e].sum().item())

        # Real L1 pass, then read the REAL L1 survivor mask (not an assumed
        # uniform rate). The L1 content buffer for turn 0 starts at 0 and its
        # rows are the L0 survivors in content order, so the span's L1 rows are
        # the positions where the L0 survivor came from inside [s, e).
        trainer._run_l1_batch(
            [turn], target_ratio=trainer._drill_leaf_l1_retention_override(),
        )
        l1_mask = getattr(trainer, "_last_l1_survivor_mask", None)
        l1_cu = getattr(trainer, "_last_l1_content_cu", None)
        surv_pos = art_mask.nonzero().flatten().tolist()
        n_l1_total = 0
        kept_e2e = 0
        if l1_mask is not None and l1_cu is not None:
            t0, t1 = int(l1_cu[0].item()), int(l1_cu[1].item())
            turn_mask = l1_mask[t0:t1].to("cpu")
            n_l1_total = int(turn_mask.sum().item())
            # The turn's content buffer INTERLEAVES pinned-ID tokens with L0
            # survivors per article, so content row k is NOT L0 survivor k.
            # ``survivor_mask`` marks the survivor rows; the gold article is
            # segment 0, so its survivors are the first len(surv_pos) of them.
            sv_rows = (
                turn["survivor_mask"].to("cpu").nonzero().flatten().tolist()
                if turn.get("survivor_mask") is not None
                else list(range(int(turn_mask.shape[0])))
            )
            for k, pos in enumerate(surv_pos):
                if k >= len(sv_rows):
                    break
                row = sv_rows[k]
                if s <= pos < e and row < int(turn_mask.shape[0]) and bool(turn_mask[row]):
                    kept_e2e += 1

        args = (n_span, kept_l0, kept_e2e, int(art_mask.shape[0]),
                len(surv_pos), n_l1_total)
        overall.add(*args)
        by_ds[name].add(*args)
        by_len[_bucket(n_span)].add(*args)
        quota[name] += 1

    return {
        "overall": overall.report(),
        "by_dataset": {k: v.report() for k, v in sorted(by_ds.items())},
        "by_span_length": {k: v.report() for k, v in sorted(by_len.items())},
    }


def _fmt(tag: str, m: dict) -> str:
    if not m.get("samples"):
        return f"   {tag:22s} (no samples)"
    return (
        f"   {tag:22s} n={m['samples']:4d}  span~{m['mean_span_tokens']:5.1f}tok  "
        f"L0 {m['l0_span_recall_per_sample']:.3f} (pooled {m['l0_span_recall_pooled']:.3f}) "
        f"lift {m['l0_lift']:5.2f}x   "
        f"E2E {m['e2e_span_recall_per_sample']:.3f} lift {m['e2e_lift']:5.2f}x"
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    diag = cfg.get("diag", {}) or {}
    ckpts = list(diag.get("checkpoints", []) or [])
    per_dataset = int(diag.get("per_dataset", diag.get("n_samples", 40)))
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
            results[label] = {"error": f"{type(exc).__name__}"}
            continue
        trainer.model.eval()
        with torch.no_grad():
            res = measure(trainer, per_dataset)
        trainer.model.train()
        results[label] = res
        print(f"\n{label}", flush=True)
        print(_fmt("OVERALL", res["overall"]), flush=True)
        for name, m in res["by_dataset"].items():
            print(_fmt(f"ds:{name}", m), flush=True)
        for name, m in res["by_span_length"].items():
            print(_fmt(f"span:{name}tok", m), flush=True)
    print("\nSPAN-SURVIVAL SUMMARY", json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
