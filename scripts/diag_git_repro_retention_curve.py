#!/usr/bin/env python
"""Diagnostic: gold-content delivery vs (L0, L1) drill retention — git-repro.

For the "less compression in the drill-down" test the coordinator needs:

1. L0 θ diagnostics: is the drill L0 (threshold mode) starved because the
   frozen head's logits leave no θ that realizes the target (BUG-1 family)?
   Reports θ(r) at several ratios, the drill-doc logit distribution vs θ, and
   the realized threshold keep-rate vs target.

2. The delivery curve: per full-drill sample, run the REAL retrieval-leaf
   pipeline (pinned fcid + live L0 of the diff + L1 re-compress, task query
   injected as compression prompt at BOTH levels, quota fix in source) over a
   retention grid:
       L0: threshold@ramp (current), exact_topk@0.05, exact_topk@0.10
       L1: exact_topk 0.05 / 0.15 / 0.25
   and measure what actually survives to the answer's context:
       - delivered content tokens (count + fraction of the diff)
       - gold-blob line hit-rate of the delivered text
       - recon CE with the delivered tokens spliced as TEXT before the answer
         (decoder-readable proxy; the embedding channel needs retraining, see
         finding (h) in diag_git_repro_retrieve_splice).

Run:
  docker compose --env-file .env -f docker/docker-compose.yaml run --rm -T \
    train-phase2-kb-git-repro-fullbackprop \
    python scripts/diag_git_repro_retention_curve.py \
    +experiment=phase2_kb_git_repro_fullbackprop
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import structlog
import torch
from omegaconf import DictConfig

from bgkit.models.decoder import TokenSegment
from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
from bgkit.utils.logging import setup_logging
from bgkit.utils.packing import position_ids_from_cu

logger = structlog.get_logger()

DEFAULT_CKPT = (
    "/workspace/checkpoints_fast/"
    "phase2_kb_step8859_20260729_175947_588332_run-phase2_kb_git_repro_fullbackprop"
)

L0_GRID = (("thresh", None), ("topk", 0.05), ("topk", 0.10))
L1_GRID = (0.05, 0.15, 0.25)


def seg_len(seg) -> int:
    if isinstance(seg, TokenSegment):
        return int(seg.token_ids.reshape(-1).shape[0])
    return int(seg.embeddings.reshape(-1, seg.embeddings.shape[-1]).shape[0])


def classify_turns(trainer, sample) -> list[str]:
    tree = trainer._trees.get(sample.dataset_name)
    kinds = []
    for t in sample.trajectory:
        if t.kind != "bgkit":
            continue
        ids = list(t.args.get("ids", []))
        q = str(t.args.get("query", ""))
        if bool(t.args.get("is_head", False)):
            kinds.append("head")
        elif q == "" and len(ids) == 1 and tree is not None and str(ids[0]) in tree:
            kinds.append("node")
        else:
            kinds.append("retrieve")
    return kinds


def head_task_query(sample) -> str:
    for t in sample.trajectory:
        if t.kind == "bgkit" and bool(t.args.get("is_head", False)):
            return str(t.args.get("query", ""))
    return ""


def retrieve_doc_ids(trainer, sample) -> list[str]:
    ds = sample.dataset_name
    tree = trainer._trees.get(ds)
    docs: list[str] = []
    for t in sample.trajectory:
        if t.kind != "bgkit" or bool(t.args.get("is_head", False)):
            continue
        ids = list(t.args.get("ids", []))
        q = str(t.args.get("query", ""))
        if q == "" and len(ids) == 1 and tree is not None and str(ids[0]) in tree:
            continue
        docs.extend(trainer._resolve_article_ids(ds, ids))
    return docs


def recon_ce(trainer, segments, span) -> float:
    out = trainer.decoder.forward_interleaved_with_loss(
        segments, return_hidden_states=True,
    )
    s, c = KRKBTrainer._span_ce_sum_count(out, [span])
    del out
    return s / max(c, 1)


def insert_before_answer(segments, span, tok_ids):
    offs = 0
    idx = len(segments) - 1
    for i, seg in enumerate(segments):
        length = seg_len(seg)
        if offs <= span[0] < offs + length:
            idx = i
            break
        offs += length
    ins = TokenSegment(
        token_ids=tok_ids.unsqueeze(0),
        loss_mask=torch.zeros(
            (1, tok_ids.shape[0]), dtype=torch.bool, device=tok_ids.device,
        ),
    )
    return (
        list(segments[:idx]) + [ins] + list(segments[idx:]),
        (span[0] + int(tok_ids.shape[0]), span[1] + int(tok_ids.shape[0])),
    )


def gold_line_hit_rate(text: str, gold: str, min_sub: int = 16) -> float:
    lines = [ln.strip() for ln in gold.splitlines() if len(ln.strip()) >= min_sub]
    if not lines:
        return -1.0
    hit = 0
    for ln in lines:
        found = any(
            ln[i:i + min_sub] in text
            for i in range(0, max(1, len(ln) - min_sub + 1), min_sub)
        )
        hit += bool(found)
    return round(hit / len(lines), 3)


# ---------------------------------------------------------------------------
# Leaf pipeline (explicit, mirrors _prepare_l1_turn + _run_l1_batch)
# ---------------------------------------------------------------------------


def l0_run(trainer, ds: str, doc_id: str, q_emb, mode: str, ratio):
    """Run live L0 on one doc. mode 'thresh' uses the current θ path at the
    ramp ratio (None -> sampled = base); 'topk' uses exact_topk at ``ratio``.
    Returns (positions, survivor_rows, out)."""
    out, _cu, used_ratio = trainer._live_l0_encode(
        ds, [doc_id], query_emb=q_emb,
        ratio=ratio if mode == "topk" else None,
        selection_mode="exact_topk" if mode == "topk" else "threshold",
    )
    mask = out.survivor_mask.bool()
    pos = mask.nonzero(as_tuple=True)[0].tolist()
    return pos, out.survivor_embeddings, out, used_ratio


def l1_keep_positions(trainer, ds, doc_id, q_emb, pos, surv_rows, r_l1):
    """Real leaf L1: [pinned fcid ids | bridged L0 rows] -> exact_topk@r_l1.
    Returns positions (into the doc tokens) whose rows survive L1."""
    dev = trainer.device
    embed_tokens = trainer.encoder.l0.backbone.get_input_embeddings()
    pin_ids = trainer.encoder_tokenizer.encode(
        f" {doc_id}", add_special_tokens=False,
    ) or [0]
    pin_t = torch.tensor(pin_ids, dtype=torch.long, device=dev)
    id_emb = embed_tokens(pin_t)
    bridged = trainer.encoder.l0.auto_reproduce(surv_rows).to(id_emb.dtype)
    content = torch.cat([id_emb, bridged], dim=0)
    pinned = torch.zeros(int(content.shape[0]), dtype=torch.bool, device=dev)
    pinned[: len(pin_ids)] = True
    cu = torch.tensor([0, int(content.shape[0])], dtype=torch.int32, device=dev)
    qcu = torch.tensor([0, int(q_emb.shape[0])], dtype=torch.int32, device=dev)
    qpos = position_ids_from_cu(qcu, int(q_emb.shape[0]))
    l1_out, _p, _c = trainer.encoder.run_l1_and_project(
        l1_input_embeddings=content,
        l1_input_cu_seqlens=cu,
        target_ratio_l1=float(r_l1),
        content_group_cu_seqlens=None,
        prompt_embeddings_l1=q_emb.to(content.dtype),
        prompt_cu_seqlens_l1=qcu,
        prompt_position_ids_l1=qpos,
        pinned_positions_l1=pinned,
        selection_mode_l1="exact_topk",
    )
    mask = l1_out.survivor_mask.bool()
    content_mask = mask[len(pin_ids):]  # rows aligned with pos
    kept = [pos[i] for i in range(len(pos)) if bool(content_mask[i])]
    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    assert cfg.training.phase == "phase2_kb"
    trainer = KRKBTrainer(cfg)
    trainer.setup()
    ckpt = os.environ.get("DIAG_CKPT", DEFAULT_CKPT)
    trainer.load_checkpoint(Path(ckpt))
    logger.info("checkpoint_loaded", step=trainer.global_step)
    trainer.model.eval()
    trainer._eval_tree_cache = {}
    trainer._eval_shared_tree_key = None
    if trainer._round_robin:
        trainer._set_active_decoder(
            os.environ.get("DIAG_FAMILY", "qwen35"),
        )

    n_samples = int(os.environ.get("DIAG_N_FULL", "3"))
    scan_cap = int(os.environ.get("DIAG_SCAN_CAP", "4000"))

    picked: list = []
    scanned = 0
    for batch in trainer.eval_dataloader:
        for s in batch:
            scanned += 1
            if s.dataset_name != "git_commit_repro":
                continue
            kinds = classify_turns(trainer, s)
            if sum(1 for k in kinds if k == "retrieve") >= 2:
                root = trainer._repo_group_key(s)
                if not any(trainer._repo_group_key(p) == root for p in picked):
                    picked.append(s)
        if scanned >= scan_cap or len(picked) >= n_samples:
            break
    if not picked:
        raise RuntimeError("no full-drill samples found")
    logger.info("samples_picked", n=len(picked), scanned=scanned)

    report: dict = {"checkpoint": ckpt, "samples": []}

    with torch.no_grad():
        # ---- (1) L0 θ diagnostics ----
        theta_diag = {
            f"theta@{r}": round(
                float(trainer.encoder.l0.threshold.theta_for_ratio(r)), 4,
            )
            for r in (0.05, 0.10, 0.15, 0.25)
        }
        report["l0_theta"] = theta_diag
        print("L0 theta(r):", json.dumps(theta_diag))

        for sample in picked:
            ds = sample.dataset_name
            task_query = head_task_query(sample)
            q_emb = trainer._head_query_emb(task_query)
            gold = str(sample.gold_answer)
            docs = retrieve_doc_ids(trainer, sample)
            res: dict = {
                "question": sample.question[:110],
                "n_docs": len(docs),
                "doc_tokens": [
                    int(trainer._token_store.length(ds, d)) for d in docs
                ],
            }

            # Baseline segments for the CE splice (current behavior).
            trainer._ensure_eval_shared_tree(sample)
            prep = trainer._prepare_sample_for_decode(sample)
            survs = trainer._run_l1_batch(prep["prepared_turns"])
            seg_a, tr_a = trainer._assemble_sample_segments(prep, survs)
            span = tr_a.answer_span
            if span is None:
                continue
            res["ce_baseline"] = round(recon_ce(trainer, seg_a, span), 4)

            # Full-text reference.
            full_text = "\n\n".join(
                trainer.encoder_tokenizer.decode(
                    trainer._token_store.get(ds, d).tolist(),
                )
                for d in docs
            )
            ids = trainer.tokenizer.encode(full_text, add_special_tokens=False)[:3072]
            tok_t = torch.tensor(ids, dtype=torch.long, device=trainer.device)
            seg_x, span_x = insert_before_answer(seg_a, span, tok_t)
            res["ce_fulltext"] = round(recon_ce(trainer, seg_x, span_x), 4)
            res["gold_hit_fulltext"] = gold_line_hit_rate(full_text, gold)

            # ---- L0 stage per grid mode (per doc) ----
            l0_cache: dict = {}
            l0_stats: dict = {}
            for mode, r0 in L0_GRID:
                key = f"{mode}@{r0 if r0 is not None else 'ramp'}"
                per_doc = []
                stats = []
                for d in docs:
                    pos, rows, out, used_r = l0_run(trainer, ds, d, q_emb, mode, r0)
                    per_doc.append((d, pos, rows))
                    n = int(out.survivor_mask.shape[0])
                    entry = {
                        "doc_len": n,
                        "kept": len(pos),
                        "keep_rate": round(len(pos) / max(n, 1), 4),
                        "ratio_requested": round(float(used_r), 4),
                    }
                    if mode == "thresh":
                        logits = out.logits_for_op.float()
                        theta = float(out.theta) if out.theta is not None else None
                        entry["theta"] = round(theta, 4) if theta is not None else None
                        entry["frac_logits_gt_theta"] = round(
                            float((logits > theta).float().mean()), 4,
                        ) if theta is not None else None
                        entry["frac_logits_saturated"] = round(
                            float((logits.abs() > 0.95).float().mean()), 4,
                        )
                    stats.append(entry)
                l0_cache[key] = per_doc
                l0_stats[key] = stats
            res["l0_stage"] = l0_stats

            # ---- L1 stage grid + delivery metrics ----
            grid = {}
            for l0_key, per_doc in l0_cache.items():
                for r1 in L1_GRID:
                    kept_texts = []
                    n_kept = 0
                    n_total = 0
                    for d, pos, rows in per_doc:
                        toks = trainer._token_store.get(ds, d)
                        n_total += int(toks.shape[0])
                        if not pos:
                            continue
                        kept = l1_keep_positions(
                            trainer, ds, d, q_emb, pos, rows, r1,
                        )
                        n_kept += len(kept)
                        kept_texts.append(
                            trainer.encoder_tokenizer.decode(
                                [int(toks[p]) for p in kept],
                            )
                        )
                    delivered = "\n".join(kept_texts)
                    ids = trainer.tokenizer.encode(
                        delivered, add_special_tokens=False,
                    )[:3072]
                    if ids:
                        tok_t = torch.tensor(
                            ids, dtype=torch.long, device=trainer.device,
                        )
                        seg_x, span_x = insert_before_answer(seg_a, span, tok_t)
                        ce = round(recon_ce(trainer, seg_x, span_x), 4)
                    else:
                        ce = None
                    grid[f"l0={l0_key},l1={r1}"] = {
                        "delivered_tokens": n_kept,
                        "delivered_frac_of_diff": round(n_kept / max(n_total, 1), 4),
                        "gold_line_hit": gold_line_hit_rate(delivered, gold),
                        "ce_token_splice": ce,
                    }
            res["delivery_grid"] = grid
            report["samples"].append(res)
            print("\n" + "=" * 100)
            print(json.dumps(res, indent=2, default=str))

    out_path = Path("/workspace/checkpoints/diag_git_repro_retention_curve.json")
    try:
        out_path.write_text(json.dumps(report, indent=2, default=str))
        logger.info("report_written", path=str(out_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("report_write_failed", err=str(exc))


if __name__ == "__main__":
    main()
