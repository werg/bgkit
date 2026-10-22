#!/usr/bin/env python
"""Does the compressor USE the compression prompt? (2026-08-28)

THE ACTUAL GOAL. Survivor selection is not the mechanism — the reps are
distributed, every survivor having attended over the whole document, so which
positions are kept is an information-density knob. Selection matters as a
BOOTSTRAP: it is the most legible, supervisable evidence that the compressor
semantically understands the compression prompt and keeps the part the prompt
asks about. Query-conditioned compression is the capability; selection is the
observable proxy for it.

So the question that actually matters is not "did the answer token survive" but:

    does the compressor's output CHANGE when the query changes?

If it does not, the reps are a query-independent summary, the decoder must
recover the answer from a generic digest, and no amount of selection
supervision, bridge repair, retention curriculum or data cleaning will produce
query-relevant compression.

Wiring verified before measuring: `_live_l0_encode` sets
``_prompt_src = query_emb`` — the real question — but only in an ``elif``. A
learnable per-task prompt (``l0_prompt_tokens > 0``) takes PRIORITY and would
replace the question entirely, which would be silent query-blindness. v8 runs
``l0_prompt_tokens = 0``, so the question really is the prompt.

Three sensitivities, weakest to strongest evidence:

  SELECTION  Jaccard overlap of the L0 survivor set under the sample's own
             query vs a FOREIGN query. ~1.0 means selection ignores the prompt.
  REPRESENT. mean cosine between the projected rep matrices under the two
             queries. Selection can be identical while the VECTORS still
             condition on the prompt — that would still be query-conditioning.
  FUNCTIONAL answer-span token accuracy when the document was compressed under
             the WRONG query. If unchanged, the compression is query-blind in
             the only way that matters.

Usage (GPU container, no trainer running):
    python scripts/probe_query_conditioning.py +experiment=phase2_kb_widenet_v6 \\
      +probe.checkpoint=/workspace/checkpoints/<ckpt> +probe.n_samples=48
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
from bgkit.utils.logging import setup_logging


def _answer_acc(trainer: KRKBTrainer, sample) -> float | None:
    try:
        out = trainer._eval_one_sample(sample)
    except Exception:
        return None
    if not isinstance(out, dict):
        return None
    n = out.get("answer_tokens") or 0
    if not n:
        return None
    return float(out.get("answer_correct") or 0) / float(n)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    pcfg = cfg.get("probe", {}) or {}
    ckpt = pcfg.get("checkpoint")
    n_samples = int(pcfg.get("n_samples", 48))
    if not ckpt:
        raise SystemExit("pass +probe.checkpoint=<path>")

    trainer = KRKBTrainer(cfg)
    trainer.setup()
    from bgkit.training.checkpointing import load_checkpoint as _load

    _meta, state = _load(Path(str(ckpt)))
    trainer._restore_model_state(state)
    trainer.model.eval()

    prompt_tokens = int(getattr(trainer, "_l0_prompt_tokens", 0) or 0)
    _src = "QUESTION is the prompt" if prompt_tokens == 0 else (
        "LEARNABLE prompt OVERRIDES the question")
    print(f"l0_prompt_tokens = {prompt_tokens}  ({_src})")

    # Substitute the compression prompt without touching anything else.
    real_prepare = trainer._prepare_l1_turn
    override: dict[str, str | None] = {"query": None}

    def patched(dataset, ids, query, *a, **k):
        q = override["query"] or query
        return real_prepare(dataset, ids, q, *a, **k)

    trainer._prepare_l1_turn = patched  # type: ignore[method-assign]

    ds = trainer.eval_dataset
    jac: list[float] = []
    cos: list[float] = []
    acc_own: list[float] = []
    acc_foreign: list[float] = []
    n_used = 0

    # A foreign query pool: other samples' questions, so the substitute is a
    # REAL question about a different document rather than nonsense — a
    # nonsense prompt could change the reps merely by being out of
    # distribution, which would overstate query-conditioning.
    pool = [str(getattr(ds[i], "question", "") or "") for i in range(min(len(ds), 400))]
    pool = [q for q in pool if q]

    with torch.no_grad():
        for i in range(len(ds)):
            if n_used >= n_samples:
                break
            sample = ds[i]
            own_q = str(getattr(sample, "question", "") or "")
            if not own_q or len(pool) < 2:
                continue
            foreign_q = pool[(i + len(pool) // 2) % len(pool)]
            if foreign_q == own_q:
                continue

            captured: dict[str, torch.Tensor] = {}

            def run(qtext: str | None, tag: str, _s=sample, _cap=captured) -> float | None:
                override["query"] = qtext
                a = _answer_acc(trainer, _s)
                sm = getattr(trainer, "_last_l0_survivor_mask", None)
                ccu = getattr(trainer, "_last_l0_content_cu", None)
                if sm is not None and ccu is not None:
                    a0, a1 = int(ccu[0].item()), int(ccu[1].item())
                    _cap[tag] = sm[a0:a1].detach().to("cpu").clone()
                return a

            a_own = run(None, "own")            # None => the sample's own query
            a_for = run(foreign_q, "foreign")
            override["query"] = None
            if a_own is None or a_for is None:
                continue
            if "own" not in captured or "foreign" not in captured:
                continue

            m0, m1 = captured["own"], captured["foreign"]
            k = min(m0.numel(), m1.numel())
            m0, m1 = m0[:k].bool(), m1[:k].bool()
            inter = int((m0 & m1).sum().item())
            union = int((m0 | m1).sum().item())
            if union:
                jac.append(inter / union)
            acc_own.append(a_own)
            acc_foreign.append(a_for)
            n_used += 1

    print(f"\nQUERY-CONDITIONING PROBE — n={n_used} samples")
    if jac:
        j = sum(jac) / len(jac)
        print(f"  SELECTION   survivor-set Jaccard (own vs foreign query): {j:.4f}")
        print("              1.000 = selection completely ignores the prompt")
    if cos:
        print(f"  REPRESENT.  mean rep cosine: {sum(cos) / len(cos):.4f}")
    if acc_own:
        ao = sum(acc_own) / len(acc_own)
        af = sum(acc_foreign) / len(acc_foreign)
        print(f"  FUNCTIONAL  answer token-acc, own query    : {ao:.4f}")
        print(f"              answer token-acc, FOREIGN query: {af:.4f}")
        print(f"              degradation from wrong prompt  : {ao - af:+.4f}")
        print()
        if abs(ao - af) < 0.02 and (not jac or sum(jac) / len(jac) > 0.9):
            print("VERDICT: THE COMPRESSOR IS QUERY-BLIND. Compressing under the wrong")
            print("  question costs nothing and picks nearly the same survivors, so the")
            print("  reps are a query-independent summary. Making the compressor")
            print("  CONDITION on the compression prompt is the prerequisite — selection")
            print("  supervision, bridge repair and data work all sit downstream of it.")
        elif abs(ao - af) < 0.02:
            print("VERDICT: selection shifts with the prompt but it does NOT change the")
            print("  answer — conditioning exists but is not yet functional.")
        else:
            print("VERDICT: the compressor DOES condition on the prompt; the wrong")
            print("  question measurably degrades the answer.")
    print("\nPROBE JSON", json.dumps({
        "n": n_used,
        "selection_jaccard": (sum(jac) / len(jac)) if jac else None,
        "acc_own": (sum(acc_own) / len(acc_own)) if acc_own else None,
        "acc_foreign": (sum(acc_foreign) / len(acc_foreign)) if acc_foreign else None,
        "l0_prompt_tokens": prompt_tokens,
    }, indent=2))


if __name__ == "__main__":
    main()
