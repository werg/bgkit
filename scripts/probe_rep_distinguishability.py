#!/usr/bin/env python
"""Do the reps distinguish one document from another at all?

WHY THIS PROBE, AND WHY NOW. The full-text ceiling arm (2026-08-30) measured
that reps carry ~4% of what the raw document carries (token-F1 grepset
0.386 vs 0.752 full text). The retention sweep then showed that 33x more
reps does NOT help: 522 -> 3570 rep vectors leaves F1 flat, while 1265 raw
token vectors reach 0.752. Budget is not the constraint and neither is
selection. That combination has exactly one shape: the rep vectors are
largely REDUNDANT with each other and largely IDENTICAL across documents --
a near-constant summary, not a document-specific code.

This probe tests that directly, with no training and no decoder in the loop,
so it cannot be confounded by decoder adaptation (which is what made the
base-vs-v8 nats comparison useless) or by answer-set harshness (which is what
makes EM a poor read).

THE MEASUREMENT. Per document, split the vectors into two disjoint halves by
row parity, mean-pool each half, L2-normalise. Then ask: does half A of
document i retrieve half B of document i out of M candidates? A representation
that carries document-specific content retrieves itself. One that emits the
same summary for every document scores at chance (1/M).

Run at three stages of the same forward, so a failure localises:

  raw       decoder embeddings of the raw document tokens (the upper bound the
            full-text arm actually spends)
  l1_input  the packed L1 input: L0 survivors + pinned id-token embeddings
            (i.e. everything L0 and the bridge have already decided)
  reps      the final projected L1 survivors -- what the decoder is spliced

  raw high, l1_input high, reps at chance -> the collapse is L1 + projection.
  raw high, l1_input at chance            -> the collapse is L0 or earlier.
  all three high                          -> content is present and distinct;
                                             the 4% is a readability/training
                                             problem, not a content problem,
                                             and the manifold story is dead.

Also reported per stage:
  eff_rank_across  participation ratio of the M pooled vectors (how many
                   directions the corpus actually uses; M means fully spread,
                   ~1 means one vector for every document)
  eff_rank_within  mean participation ratio of one document's own vectors
                   (why extra budget bought nothing: if this saturates, the
                   k+1-th rep is a copy of the first k)
  mean_offdiag_cos mean cosine between different documents' pooled vectors

Usage (GPU container, no trainer running):
    python scripts/probe_rep_distinguishability.py +experiment=phase2_kb_widenet_v8 \\
      +diag.checkpoint=/workspace/checkpoints/<ckpt> +diag.n_samples=96
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from bgkit.training.checkpointing import load_checkpoint, restore_model_state_lenient
from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
from bgkit.utils.logging import setup_logging


def _participation_ratio(x: torch.Tensor) -> float:
    """Effective number of directions used by the rows of ``x``.

    ``(sum lambda)^2 / sum lambda^2`` over the covariance eigenvalues, which
    equals the row count for a perfectly isotropic spread and 1.0 when every
    row lies on a single direction. Rows are centred first: an uncentred
    matrix whose rows are "one shared mean + tiny noise" reports rank ~1 for
    the mean alone, which is the thing being tested, not an artefact to keep.
    """
    if x.shape[0] < 2:
        return float("nan")
    xc = (x - x.mean(dim=0, keepdim=True)).float()
    sv = torch.linalg.svdvals(xc)
    lam = sv.pow(2)
    denom = lam.pow(2).sum().clamp_min(1e-30)
    return float((lam.sum().pow(2) / denom).item())


def _shared_energy(x: torch.Tensor) -> dict:
    """How much of a document's vectors is ONE shared vector?

    ``shared_frac`` is ||mean||^2 / mean(||row||^2): the fraction of the
    representation's energy that every survivor of the document carries
    identically. At 0.999 the k survivors are one vector plus 0.1% content,
    which is what an effective rank of ~1.0 means in energy terms and why a
    33x larger retention budget bought nothing.
    """
    if x.shape[0] < 2:
        return {}
    xf = x.float()
    mu = xf.mean(dim=0)
    energy = xf.pow(2).sum(dim=-1).mean().clamp_min(1e-30)
    return {
        "shared_frac": float((mu.pow(2).sum() / energy).item()),
        "mean_vec": mu,
    }


def _halves(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Mean-pool disjoint parity halves, L2-normalised. None if too few rows."""
    if x.shape[0] < 4:
        return None
    a = x[0::2].float().mean(dim=0)
    b = x[1::2].float().mean(dim=0)
    return (
        a / a.norm().clamp_min(1e-6),
        b / b.norm().clamp_min(1e-6),
    )


def _retrieval(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Top-1 / MRR for matching each row of ``a`` to its own row of ``b``."""
    m = a.shape[0]
    sim = a @ b.T  # (M, M)
    order = sim.argsort(dim=-1, descending=True)
    gold = torch.arange(m, device=sim.device).unsqueeze(-1)
    rank = (order == gold).float().argmax(dim=-1) + 1
    off = sim.clone()
    off.fill_diagonal_(float("nan"))
    return {
        "n_docs": m,
        "top1": float((rank == 1).float().mean().item()),
        "chance_top1": 1.0 / m,
        "mrr": float((1.0 / rank.float()).mean().item()),
        "median_rank": float(rank.float().median().item()),
        "mean_selfsim": float(sim.diagonal().mean().item()),
        "mean_offdiag_cos": float(torch.nanmean(off).item()),
    }


def collect(trainer: KRKBTrainer, n_samples: int) -> dict:
    ds = trainer.eval_dataset
    dec = getattr(trainer, "decoder", None) or next(iter(trainer.decoders.values()))
    dec_hidden = int(dec.backbone.get_input_embeddings().weight.shape[-1])

    stages: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
        "raw": [], "l1_input": [], "reps": [],
    }
    within: dict[str, list[float]] = {"raw": [], "l1_input": [], "reps": []}
    shared: dict[str, list[float]] = {"raw": [], "l1_input": [], "reps": []}
    doc_means: dict[str, list[torch.Tensor]] = {"raw": [], "l1_input": [], "reps": []}
    datasets: list[str] = []
    used = 0
    raw_failures = 0

    with torch.no_grad():
        for i in range(len(ds)):
            if used >= n_samples:
                break
            sample = ds[i]
            try:
                prep = trainer._prepare_sample_for_decode(sample)
            except Exception:
                continue
            turns = prep["prepared_turns"]
            if not turns or not isinstance(turns[0], dict) or "content" not in turns[0]:
                continue
            turn = turns[0]
            try:
                reps = trainer._run_l1_batch(
                    [turn], target_ratio=trainer._drill_leaf_l1_retention_override(),
                )[0].detach()
            except Exception:
                continue
            content = turn["content"].detach()

            # The raw-document upper bound needs the turn's article ids, which
            # live on the rendered trajectory, not on the prepared turn.
            raw = None
            bgkit_turns = getattr(prep["rendered"], "bgkit_turns", None) or []
            if bgkit_turns:
                try:
                    raw = trainer._full_text_payload(
                        prep["sample"].dataset_name,
                        bgkit_turns[0].args.get("ids"),
                        dec_hidden,
                        reps,
                    ).detach()
                except Exception:
                    raw_failures += 1

            per_stage = {"reps": reps, "l1_input": content, "raw": raw}
            pooled = {k: (_halves(v) if v is not None else None)
                      for k, v in per_stage.items()}
            # Require every stage present so the three columns are measured on
            # the SAME set of documents -- otherwise "raw beats reps" could be
            # a difference in which documents each column saw.
            if any(p is None for p in pooled.values()):
                continue
            for k, p in pooled.items():
                stages[k].append(p)
                within[k].append(_participation_ratio(per_stage[k]))
                se = _shared_energy(per_stage[k])
                if se:
                    shared[k].append(se["shared_frac"])
                    doc_means[k].append(se["mean_vec"].cpu())
            datasets.append(getattr(sample, "dataset_name", "?"))
            used += 1

    out: dict = {"samples": used, "raw_payload_failures": raw_failures}
    corpus_means: dict[str, torch.Tensor] = {}
    if used < 4:
        return out
    for k, pairs in stages.items():
        a = torch.stack([p[0] for p in pairs])
        b = torch.stack([p[1] for p in pairs])
        r = _retrieval(a, b)
        r["eff_rank_across"] = _participation_ratio(a)
        wv = [w for w in within[k] if w == w]
        r["eff_rank_within"] = (sum(wv) / len(wv)) if wv else None
        sv = shared[k]
        r["shared_frac_within_doc"] = (sum(sv) / len(sv)) if sv else None
        if doc_means[k]:
            dm = torch.stack(doc_means[k])
            gm = dm.mean(dim=0)
            gmu = gm / gm.norm().clamp_min(1e-6)
            dmu = dm / dm.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            # How much of that shared component is shared ACROSS documents
            # too: 1.0 means every document emits the same vector.
            r["doc_mean_cos_to_corpus_mean"] = float((dmu @ gmu).mean().item())
            r["corpus_mean_norm_over_doc_mean_norm"] = float(
                (gm.norm() / dm.norm(dim=-1).mean().clamp_min(1e-6)).item(),
            )
            corpus_means[k] = gm
        out[k] = r
    # WHICH vector is the shared one? ``survive_embedding`` is a single
    # learned parameter scattered at every surviving position, and its
    # gradient is a sum over all of them (it needs its own grad-norm group for
    # exactly that reason). If it grew, it swamps the position-specific
    # content and every survivor becomes the same vector -- the concrete
    # mechanism behind an effective rank of 1.
    enc = trainer.encoder
    out["survive_embedding"] = {}
    for lvl in ("l0", "l1"):
        vec = getattr(getattr(enc, lvl, None), "survive_embedding", None)
        if vec is None:
            continue
        v = vec.detach().float().cpu()
        entry = {"norm": float(v.norm().item())}
        vu = v / v.norm().clamp_min(1e-6)
        for stage, gm in corpus_means.items():
            if gm.shape[-1] != v.shape[-1]:
                continue
            gmu = gm / gm.norm().clamp_min(1e-6)
            entry[f"cos_to_{stage}_corpus_mean"] = float((vu @ gmu).item())
            entry[f"{stage}_corpus_mean_norm"] = float(gm.norm().item())
        out["survive_embedding"][lvl] = entry

    # Per-dataset retrieval too: the families differ enough that a pooled
    # number can hide one clean family behind two contaminated ones.
    by_ds: dict[str, list[int]] = {}
    for idx, name in enumerate(datasets):
        by_ds.setdefault(name, []).append(idx)
    out["by_dataset"] = {}
    for name, idxs in by_ds.items():
        if len(idxs) < 4:
            continue
        out["by_dataset"][name] = {
            k: _retrieval(
                torch.stack([stages[k][j][0] for j in idxs]),
                torch.stack([stages[k][j][1] for j in idxs]),
            )
            for k in stages
        }
        out["by_dataset"][name]["n"] = len(idxs)
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    diag = cfg.get("diag", {}) or {}
    ckpt = diag.get("checkpoint")
    n = int(diag.get("n_samples", 96))
    if not ckpt:
        raise SystemExit("pass +diag.checkpoint=/path/to/checkpoint")

    trainer = KRKBTrainer(cfg)
    trainer.setup()
    _meta, state = load_checkpoint(Path(str(ckpt)))
    restore_model_state_lenient(trainer, state)
    trainer.model.eval()

    res = collect(trainer, n)
    print(f"\ncheckpoint: {ckpt}")
    print(f"documents:  {res.get('samples')}")
    hdr = f"{'stage':<10}{'top1':>8}{'chance':>8}{'MRR':>8}{'med_rank':>10}" \
          f"{'rank_across':>13}{'rank_within':>13}{'offdiag_cos':>13}" \
          f"{'shared_frac':>13}{'doc~corpus':>12}"
    print(hdr)
    for k in ("raw", "l1_input", "reps"):
        r = res.get(k)
        if not r:
            continue
        print(f"{k:<10}{r['top1']:8.3f}{r['chance_top1']:8.3f}{r['mrr']:8.3f}"
              f"{r['median_rank']:10.1f}{r['eff_rank_across']:13.2f}"
              f"{(r['eff_rank_within'] or float('nan')):13.2f}"
              f"{r['mean_offdiag_cos']:13.3f}"
              f"{(r.get('shared_frac_within_doc') or float('nan')):13.5f}"
              f"{(r.get('doc_mean_cos_to_corpus_mean') or float('nan')):12.5f}")
    print()
    print("top1 at chance for 'reps' while 'raw'/'l1_input' are high means the")
    print("reps are a near-constant summary: they do not identify their own")
    print("document, which is why 33x the budget bought nothing.")
    print()
    print("shared_frac = fraction of a document's rep energy carried IDENTICALLY")
    print("by every one of its survivors; doc~corpus = how much of that shared")
    print("part is the SAME vector for every document.")
    blob = json.dumps(res, indent=2, default=str)
    dest = diag.get("out")
    if dest:
        Path(str(dest)).write_text(blob)
        print(f"wrote {dest}")
    print("\nDISTINGUISHABILITY JSON", blob)


if __name__ == "__main__":
    main()
