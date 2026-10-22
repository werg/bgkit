#!/usr/bin/env python
"""Does L1's INPUT still contain the answer-span signal? (2026-08-28)

L1 discards the answer. Its span discrimination sits at +0.138 sd above the
document mean where L0 reaches +0.517 sd, and a head with ZERO discrimination
would score +0.000 — so L1 is barely above chance and has stayed there for
thousands of steps, through every intervention tried at the head.

Before fixing the head again, establish whether the head COULD succeed. L1
never sees tokens: it sees L0's survivor embeddings pushed through
``encoder.l0.auto_repro_head`` (the bridge), interleaved with pinned article-ID
embeddings. If the bridge does not preserve whatever distinguishes an
answer-span token from its neighbours, no head loss can recover it and every
head-side fix is doomed by construction.

The test is a LINEAR PROBE, deliberately weak: fit logistic regression on the
bridged vectors to classify span vs non-span, and compare against the same
probe on L0's PRE-bridge survivor embeddings. Linear is the point — if a
linear map can find the span, a trained MLP head certainly can, so a failure
here is a strong statement about the representation rather than the classifier.

    L0 AUC high, L1 AUC ~0.5  -> the BRIDGE destroys the signal. Fix the
                                 bridge (or bypass it); head losses are futile.
    both AUCs high            -> the information survives; L1's head/loss is
                                 the problem and head-side work is justified.
    both AUCs ~0.5            -> nothing distinguishes span tokens even at L0's
                                 output, i.e. the span supervision target is
                                 not linearly present anywhere in the pipeline.

Reports AUC + accuracy at the class-balanced operating point, per level, with
the positive rate so a degenerate all-negative probe cannot masquerade as
success.

Usage (GPU container, no trainer running):
    python scripts/probe_l1_bridge_span_signal.py +experiment=phase2_kb_widenet_v6 \\
      +probe.checkpoint=/workspace/checkpoints/<ckpt> +probe.per_dataset=40
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


def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based AUC (Mann-Whitney U). No sklearn dependency."""
    pos = int(labels.sum().item())
    neg = int(labels.numel() - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    rank_sum = ranks[labels.bool()].sum().item()
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def content_recoverability(reps: torch.Tensor, emb: torch.Tensor,
                           gold_ids: list[int]) -> tuple[float, float]:
    """Can the ANSWER'S IDENTITY be read out of the reps, not just its location?

    H3, and a correction to how the span-membership AUC was reported. A probe
    that separates "came from the answer region" from "did not" can succeed on
    positional or stylistic cues alone — it does NOT show the reps encode WHICH
    tokens the answer contains. Membership is not content.

    This asks the stronger question directly: project each rep onto the
    decoder's token-embedding matrix and see whether the gold answer's tokens
    rank near the top. Reported against a random-token floor, since in high
    dimension a raw similarity is meaningless on its own.

    Returns (mean_gold_percentile, chance_percentile=0.5).
    """
    if reps.numel() == 0 or not gold_ids:
        return float("nan"), 0.5
    r = reps.float()
    r = r / r.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    e = emb.float()
    e = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    # Best similarity any rep gives each vocabulary item.
    best = torch.full((e.shape[0],), -1.0)
    for s_ in range(0, e.shape[0], 32768):
        blk = e[s_ : s_ + 32768].to(r.device)
        sim = (r @ blk.T).max(dim=0).values.cpu()
        best[s_ : s_ + 32768] = sim
    order = torch.argsort(best, descending=True)
    rank_of = torch.empty_like(order)
    rank_of[order] = torch.arange(order.numel())
    v = float(e.shape[0])
    pct = [1.0 - (float(rank_of[t].item()) / v) for t in gold_ids if 0 <= t < e.shape[0]]
    return (sum(pct) / len(pct) if pct else float("nan")), 0.5


def _fit_linear_probe(feats: torch.Tensor, y: torch.Tensor, steps: int = 300):
    """Class-balanced logistic regression, returns (auc, acc, pos_rate).

    ``torch.enable_grad()`` is REQUIRED: the caller collects features under
    ``torch.no_grad()`` (correctly — the encoder forward needs no graph), and
    that context also disables grad for THIS probe's own parameters, so the
    first fit died with "element 0 of tensors does not require grad".
    """
    x = feats.float()
    x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)
    n_pos = float(y.sum().item())
    n_neg = float(y.numel() - n_pos)
    if n_pos < 5 or n_neg < 5:
        return float("nan"), float("nan"), n_pos / max(y.numel(), 1)
    # Balance the classes through the loss, not by resampling, so every
    # example is used and the AUC is computed on the full set.
    w_pos = (n_pos + n_neg) / (2.0 * n_pos)
    w_neg = (n_pos + n_neg) / (2.0 * n_neg)
    weights = torch.where(y.bool(), torch.full_like(y, w_pos, dtype=torch.float32),
                          torch.full_like(y, w_neg, dtype=torch.float32))
    with torch.enable_grad():
        lin = torch.nn.Linear(x.shape[1], 1)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
        yf = y.float().unsqueeze(1)
        for _ in range(steps):
            opt.zero_grad()
            logits = lin(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yf, weight=weights.unsqueeze(1),
            )
            loss.backward()
            opt.step()
    with torch.no_grad():
        s = lin(x).squeeze(1)
    auc = _auc(s.double(), y)
    acc = float(((s > 0).float() == y.float()).float().mean().item())
    return auc, acc, n_pos / y.numel()


def collect(trainer: KRKBTrainer, per_dataset: int) -> dict:
    ds = trainer.eval_dataset
    feats: dict[str, list[torch.Tensor]] = defaultdict(list)
    labels: dict[str, list[torch.Tensor]] = defaultdict(list)
    quota: dict[str, int] = defaultdict(int)
    per_ds_rows: dict[str, dict[str, list]] = defaultdict(
        lambda: {"l0": [], "l1": [], "y": []}
    )

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
        a0, a1 = int(ccu[0].item()), int(ccu[1].item())
        art_mask = sm[a0:a1].to("cpu")
        s, e = int(gold[0]), int(gold[1])
        e = min(e, int(art_mask.shape[0]))
        if e <= s:
            continue

        # Which L0 survivors came from inside the answer span? Same convention
        # the trainer uses to build its own L1 span mask.
        surv_pos = art_mask.nonzero().flatten().tolist()
        span_flags = torch.tensor([s <= p < e for p in surv_pos], dtype=torch.long)
        if int(span_flags.sum().item()) == 0:
            continue

        # L1 input = the turn's content buffer. survivor_mask marks the rows
        # that are bridged L0 survivors (the rest are pinned ID embeddings);
        # the gold article is segment 0 so its survivors come first.
        content = turn["content"].detach().to("cpu").float()
        svm = turn.get("survivor_mask")
        if svm is None:
            continue
        sv_rows = svm.to("cpu").nonzero().flatten().tolist()
        k = min(len(sv_rows), int(span_flags.numel()))
        if k < 2:
            continue
        l1_vecs = content[torch.tensor(sv_rows[:k], dtype=torch.long)]

        # L0 PRE-bridge survivor embeddings, same rows, same order.
        l0_src = getattr(trainer, "_probe_l0_survivors", None)
        if l0_src is None:
            continue
        l0_vecs = l0_src[:k].detach().to("cpu").float()

        y = span_flags[:k]
        # H3: membership != content. Keep the projected reps and the gold answer
        # ids so content_recoverability can ask whether the ANSWER'S IDENTITY is
        # readable, not merely which rows sat in the answer region.
        try:
            gold_txt = str(getattr(sample, "gold_answer", "") or "")
            if gold_txt:
                gid = trainer.tokenizer.encode(gold_txt, add_special_tokens=False)
                projected = trainer._run_l1_batch(
                    [turn], target_ratio=trainer._drill_leaf_l1_retention_override(),
                )[0].detach()
                store = getattr(trainer, "_probe_content_items", None)
                if store is None:
                    store = {}
                    trainer._probe_content_items = store
                store.setdefault(name, []).append((projected, gid))
        except Exception:
            pass
        per_ds_rows[name]["l0"].append(l0_vecs)
        per_ds_rows[name]["l1"].append(l1_vecs)
        per_ds_rows[name]["y"].append(y)
        feats["l0"].append(l0_vecs)
        feats["l1"].append(l1_vecs)
        labels["all"].append(y)
        quota[name] += 1

    out: dict = {}
    if labels["all"]:
        y = torch.cat(labels["all"])
        for lvl in ("l0", "l1"):
            mat = torch.cat(feats[lvl])
            n = min(mat.shape[0], y.shape[0])
            auc, acc, pos = _fit_linear_probe(mat[:n], y[:n])
            out[f"overall_{lvl}"] = {
                "auc": auc, "acc": acc, "pos_rate": pos, "n": int(n),
            }
    for name, rows in per_ds_rows.items():
        if not rows["y"]:
            continue
        y = torch.cat(rows["y"])
        for lvl in ("l0", "l1"):
            mat = torch.cat(rows[lvl])
            n = min(mat.shape[0], y.shape[0])
            auc, acc, pos = _fit_linear_probe(mat[:n], y[:n])
            out[f"{name}_{lvl}"] = {
                "auc": auc, "acc": acc, "pos_rate": pos, "n": int(n),
            }
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    pcfg = cfg.get("probe", {}) or {}
    ckpt = pcfg.get("checkpoint")
    per_dataset = int(pcfg.get("per_dataset", 40))
    if not ckpt:
        raise SystemExit("pass +probe.checkpoint=<path>")

    trainer = KRKBTrainer(cfg)
    trainer.setup()

    from bgkit.training.checkpointing import load_checkpoint as _load

    _meta, state = _load(Path(str(ckpt)))
    trainer._restore_model_state(state)
    trainer.model.eval()

    # Capture L0's PRE-bridge survivors: _prepare_l1_turn bridges them before
    # they reach the L1 content buffer, so without this hook the "L0" arm would
    # be the same post-bridge vectors and the comparison would be vacuous.
    orig = trainer._l0_for_articles

    def _wrapped(*a, **k):
        flat, cu = orig(*a, **k)
        trainer._probe_l0_survivors = flat.detach()
        return flat, cu

    trainer._l0_for_articles = _wrapped  # type: ignore[method-assign]

    with torch.no_grad():
        res = collect(trainer, per_dataset)

    print("\nLINEAR SPAN PROBE — can a linear map find the answer span?")
    print(f"{'slice':<22}{'AUC':>8}{'acc':>8}{'pos_rate':>10}{'n':>8}")
    for key in sorted(res):
        m = res[key]
        print(f"{key:<22}{m['auc']:8.3f}{m['acc']:8.3f}{m['pos_rate']:10.3f}{m['n']:8d}")
    print()
    print("AUC 0.5 = no linear signal. Compare l0 (pre-bridge) vs l1 (post-bridge):")
    print("  l0 high, l1 ~0.5 -> THE BRIDGE destroys it; head-side fixes are futile")
    print("  both high        -> signal survives; L1's head/loss is the problem")
    print("  both ~0.5        -> the span target is not linearly present anywhere")
    # H3 readout: where do the gold answer's tokens rank when the reps are
    # projected onto the decoder's embedding matrix? 0.5 = chance.
    dec = getattr(trainer, "decoder", None)
    if dec is None:
        dec = next(iter(trainer.decoders.values()))
    emb_w = dec.backbone.get_input_embeddings().weight.detach()
    print("\nH3 CONTENT RECOVERABILITY (is the ANSWER readable, not just located?)")
    print(f"{'dataset':<16}{'gold_pctile':>13}{'chance':>9}{'n':>6}")
    for name, items in sorted(
        (getattr(trainer, "_probe_content_items", None) or {}).items()
    ):
        if not items:
            continue
        vals = []
        for reps, gid in items[:24]:
            v, _ = content_recoverability(reps, emb_w, gid)
            if v == v:
                vals.append(v)
        if vals:
            print(f"{name:<16}{sum(vals) / len(vals):13.3f}{0.5:9.2f}{len(vals):6d}")
    print("  ~0.5 = the answer's tokens are NO closer to the reps than random")
    print("  tokens: the reps locate the span but do NOT encode its content, and")
    print("  the projection/manifold story is not the whole explanation.")
    print("\nPROBE JSON", json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
