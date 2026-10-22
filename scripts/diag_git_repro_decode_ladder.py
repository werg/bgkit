#!/usr/bin/env python
"""Decode-ladder sweep: WHICH layer blocks git-repro recon? (2026-07-30)

Recon stayed flat (~1.7, depth-flat) after the recon-fix retrain (quota fix +
leaf exact_topk L0 + query-conditioned leaf + 0.30/0.30 retention pins,
step 8859 -> 9164). This ladder isolates the failing layer on the step-9164
checkpoint (and compares 8859 on the key rung):

  1  current            real leaf path as configured (NOTE: drill L0 ratio
                        comes from the TOP-LEVEL l0_retention ramp, which still
                        ends at 0.05 — the 0.30 pin only covers the recursive
                        tree. Measured as-is.)
  1b current-intended   drill L0 forced to 0.30 (what the test was meant to run)
  2  text-oracle        raw diff text tokens before the answer
  3  emb-oracle-full    leaf survivors at ratio 1.0 THROUGH THE SAME PATH
                        (L0 1.0 -> L1 1.0 -> projection): lossless embedding
                        channel. THE key rung.
  4  l0-only-full       encoder.forward(target_ratio_l0=None, target_ratio_l1=
                        None): all doc rows -> projection, L1 skipped.
                        Discriminates the L1 stage.
  5  overlap            gold-line hit + delivered-token fraction of the CURRENT
                        selection (as-configured and intended-0.30)
  6  ckpt compare       rungs 1 + 3 on the step-8859 checkpoint

Run:
  docker compose --env-file .env -f docker/docker-compose.yaml run --rm -T \
    train-phase2-kb-git-repro-fullbackprop \
    python scripts/diag_git_repro_decode_ladder.py \
    +experiment=phase2_kb_git_repro_fullbackprop
"""

from __future__ import annotations

import contextlib
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

CKPT_NEW = (
    "/workspace/checkpoints_fast/"
    "phase2_kb_step9164_20260730_091047_791250_run-phase2_kb_git_repro_fullbackprop"
)
CKPT_OLD = (
    "/workspace/checkpoints_fast/"
    "phase2_kb_step8859_20260729_175947_588332_run-phase2_kb_git_repro_fullbackprop"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def retrieve_turn_info(trainer, sample) -> list[tuple[int, list[str], str]]:
    """(bgkit_turn_index, ids, leaf_query) for every retrieve turn."""
    ds = sample.dataset_name
    tree = trainer._trees.get(ds)
    task_query = head_task_query(sample)
    out = []
    k = -1
    for t in sample.trajectory:
        if t.kind != "bgkit":
            continue
        k += 1
        ids = list(t.args.get("ids", []))
        q = str(t.args.get("query", ""))
        if bool(t.args.get("is_head", False)):
            continue
        if q == "" and len(ids) == 1 and tree is not None and str(ids[0]) in tree:
            continue
        out.append((k, ids, q or task_query))
    return out


def recon_ce(trainer, segments, span) -> tuple[float, int]:
    out = trainer.decoder.forward_interleaved_with_loss(
        segments, return_hidden_states=True,
    )
    s, c = KRKBTrainer._span_ce_sum_count(out, [span])
    del out
    return s / max(c, 1), c


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


@contextlib.contextmanager
def forced_l0_ratio(trainer, ratio: float | None):
    """Force _sample_l0_retention_for to ``ratio`` (None = leave as-is)."""
    if ratio is None:
        yield
        return
    orig = trainer._sample_l0_retention_for
    trainer._sample_l0_retention_for = lambda ds: float(ratio)
    try:
        yield
    finally:
        trainer._sample_l0_retention_for = orig


def rebuild_leaf_turns(trainer, sample, prep, l0_ratio: float | None) -> list:
    """Copy prepared_turns with leaf turns rebuilt under a forced L0 ratio
    (exact_topk + task-query, i.e. the new real path)."""
    ds = sample.dataset_name
    infos = {k: (ids, q) for k, ids, q in retrieve_turn_info(trainer, sample)}
    out = []
    with forced_l0_ratio(trainer, l0_ratio):
        for k, t in enumerate(prep["prepared_turns"]):
            if k in infos and t is not None and not (
                isinstance(t, dict) and "mode" in t
            ):
                ids, q = infos[k]
                out.append(trainer._prepare_l1_turn(
                    ds, ids, q, l0_selection_mode="exact_topk",
                ))
            else:
                out.append(t)
    return out


def leaf_content_stats(turns, survs) -> list[dict]:
    rows = []
    for i, t in enumerate(turns):
        if t is not None and not (isinstance(t, dict) and "mode" in t):
            rows.append({
                "turn": i,
                "N": int(t["content"].shape[0]),
                "pin": int(t["pinned"].sum()),
                "l0_rows": int(t["survivor_mask"].sum()),
                "out_rows": int(survs[i].shape[0]),
            })
    return rows


def l0_only_full_survivor(trainer, ds: str, doc_id: str, q_emb) -> torch.Tensor:
    """encoder.forward with L0 ratio None (all survive) + L1 skipped:
    projection of ALL doc rows — the Phase-1-style lossless single level."""
    dev = trainer.device
    toks = trainer._token_store.get(ds, doc_id).to(dev)
    embed_tokens = trainer.encoder.l0.backbone.get_input_embeddings()
    emb = embed_tokens(toks)
    cu = torch.tensor([0, int(emb.shape[0])], dtype=torch.int32, device=dev)
    pos = position_ids_from_cu(cu, int(emb.shape[0]))
    qcu = torch.tensor([0, int(q_emb.shape[0])], dtype=torch.int32, device=dev)
    qpos = position_ids_from_cu(qcu, int(q_emb.shape[0]))
    out = trainer.encoder.forward(
        content_embeddings=emb.to(torch.bfloat16),
        content_cu_seqlens=cu,
        content_position_ids=pos,
        prompt_embeddings=q_emb.to(torch.bfloat16),
        prompt_cu_seqlens=qcu,
        prompt_position_ids=qpos,
        target_ratio_l0=None,
        target_ratio_l1=None,
    )
    # EncoderOutput.survivor_embeddings = the PROJECTED survivors (decoder-ready).
    return out.survivor_embeddings


def selection_overlap(trainer, sample, l0_ratio: float, l1_ratio: float) -> dict:
    """Explicit leaf pipeline: which diff tokens survive L0(topk@l0_ratio,
    task query) then L1(topk@l1_ratio, pinned fcid, quota-fixed)."""
    ds = sample.dataset_name
    q_emb = trainer._head_query_emb(head_task_query(sample))
    gold = str(sample.gold_answer)
    embed_tokens = trainer.encoder.l0.backbone.get_input_embeddings()
    dev = trainer.device
    kept_texts = []
    n_kept = 0
    n_total = 0
    for _k, ids, _q in retrieve_turn_info(trainer, sample):
        for did in trainer._resolve_article_ids(ds, ids):
            toks = trainer._token_store.get(ds, did)
            n_total += int(toks.shape[0])
            out, _cu, _r = trainer._live_l0_encode(
                ds, [did], query_emb=q_emb, ratio=l0_ratio,
                selection_mode="exact_topk",
            )
            mask = out.survivor_mask.bool()
            pos_l0 = mask.nonzero(as_tuple=True)[0].tolist()
            if not pos_l0:
                continue
            pin_ids = trainer.encoder_tokenizer.encode(
                f" {did}", add_special_tokens=False,
            ) or [0]
            id_emb = embed_tokens(
                torch.tensor(pin_ids, dtype=torch.long, device=dev),
            )
            bridged = trainer.encoder.l0.auto_reproduce(
                out.survivor_embeddings,
            ).to(id_emb.dtype)
            content = torch.cat([id_emb, bridged], dim=0)
            pinned = torch.zeros(
                int(content.shape[0]), dtype=torch.bool, device=dev,
            )
            pinned[: len(pin_ids)] = True
            cu = torch.tensor(
                [0, int(content.shape[0])], dtype=torch.int32, device=dev,
            )
            qcu = torch.tensor(
                [0, int(q_emb.shape[0])], dtype=torch.int32, device=dev,
            )
            qpos = position_ids_from_cu(qcu, int(q_emb.shape[0]))
            l1_out, _p, _c = trainer.encoder.run_l1_and_project(
                l1_input_embeddings=content,
                l1_input_cu_seqlens=cu,
                target_ratio_l1=float(l1_ratio),
                content_group_cu_seqlens=None,
                prompt_embeddings_l1=q_emb.to(content.dtype),
                prompt_cu_seqlens_l1=qcu,
                prompt_position_ids_l1=qpos,
                pinned_positions_l1=pinned,
                selection_mode_l1="exact_topk",
            )
            cmask = l1_out.survivor_mask.bool()[len(pin_ids):]
            kept = [pos_l0[i] for i in range(len(pos_l0)) if bool(cmask[i])]
            n_kept += len(kept)
            kept_texts.append(
                trainer.encoder_tokenizer.decode([int(toks[p]) for p in kept]),
            )
    text = "\n".join(kept_texts)
    return {
        "delivered_tokens": n_kept,
        "delivered_frac_of_diff": round(n_kept / max(n_total, 1), 4),
        "gold_line_hit": gold_line_hit_rate(text, gold),
        "example": text[:200],
    }


# ---------------------------------------------------------------------------
# Per-sample ladder
# ---------------------------------------------------------------------------


def ladder(trainer, sample, family: str, tag: str, key_rungs_only: bool = False) -> dict:
    ds = sample.dataset_name
    res: dict = {
        "ckpt": tag, "family": family,
        "question": sample.question[:100],
        "depth": len(classify_turns(trainer, sample)),
    }
    trainer._ensure_eval_shared_tree(sample)
    prep = trainer._prepare_sample_for_decode(sample)
    turns = prep["prepared_turns"]
    l1_ramp = float(trainer._recursive_l1_retention_now())
    res["drill_l0_ratio_as_configured"] = round(
        float(trainer._l0_retention_for(ds)), 4,
    )
    res["drill_l1_ratio"] = round(l1_ramp, 4)

    def is_leaf(i):
        return turns[i] is not None and not (
            isinstance(turns[i], dict) and "mode" in turns[i]
        )

    # 1 current (as configured; drill L1 at the recursive ramp like training)
    survs_1 = trainer._run_l1_batch(turns, target_ratio=l1_ramp)
    seg_1, tr_1 = trainer._assemble_sample_segments(prep, survs_1)
    span = tr_1.answer_span
    if span is None:
        res["skip"] = "no answer"
        return res
    ce, n_tok = recon_ce(trainer, seg_1, span)
    res["answer_tokens"] = n_tok
    res["ce"] = {"1_current": round(ce, 4)}
    res["leaf_stats_current"] = leaf_content_stats(turns, survs_1)

    # 3 embedding-oracle-full (same path, ratio 1.0 end-to-end at the leaf)
    turns_3 = rebuild_leaf_turns(trainer, sample, prep, l0_ratio=1.0)
    survs_3_all = trainer._run_l1_batch(turns_3, target_ratio=1.0)
    survs_3 = [
        survs_3_all[i] if is_leaf(i) else survs_1[i] for i in range(len(turns))
    ]
    seg, tr = trainer._assemble_sample_segments(prep, survs_3)
    ce3, _ = recon_ce(trainer, seg, tr.answer_span)
    res["ce"]["3_emb_oracle_full"] = round(ce3, 4)
    res["leaf_rows_emb_oracle"] = [
        int(survs_3_all[i].shape[0]) for i in range(len(turns)) if is_leaf(i)
    ]

    if key_rungs_only:
        return res

    # 1b current-intended (drill L0 forced 0.30)
    turns_1b = rebuild_leaf_turns(trainer, sample, prep, l0_ratio=0.30)
    survs_1b = trainer._run_l1_batch(turns_1b, target_ratio=l1_ramp)
    seg, tr = trainer._assemble_sample_segments(prep, survs_1b)
    ce1b, _ = recon_ce(trainer, seg, tr.answer_span)
    res["ce"]["1b_current_intended_l0_030"] = round(ce1b, 4)
    res["leaf_stats_intended"] = leaf_content_stats(turns_1b, survs_1b)

    # 2 text-oracle
    texts = []
    for _k, ids, _q in retrieve_turn_info(trainer, sample):
        for did in trainer._resolve_article_ids(ds, ids):
            texts.append(trainer.encoder_tokenizer.decode(
                trainer._token_store.get(ds, did).tolist(),
            ))
    if texts:
        blob = "\n\n".join(texts)
        idsx = trainer.tokenizer.encode(blob, add_special_tokens=False)[:3072]
        tok_t = torch.tensor(idsx, dtype=torch.long, device=trainer.device)
        seg, span2 = insert_before_answer(seg_1, span, tok_t)
        ce2, _ = recon_ce(trainer, seg, span2)
        res["ce"]["2_text_oracle"] = round(ce2, 4)

    # 4 L0-only-full (L1 skipped, all rows -> projection)
    q_emb = trainer._head_query_emb(head_task_query(sample))
    survs_4 = list(survs_1)
    ok4 = True
    for k, ids, _q in retrieve_turn_info(trainer, sample):
        parts = []
        for did in trainer._resolve_article_ids(ds, ids):
            p = l0_only_full_survivor(trainer, ds, did, q_emb)
            if p is None:
                ok4 = False
                break
            parts.append(p)
        if not ok4:
            break
        if parts:
            survs_4[k] = torch.cat(parts, dim=0)
    if ok4:
        seg, tr = trainer._assemble_sample_segments(prep, survs_4)
        ce4, _ = recon_ce(trainer, seg, tr.answer_span)
        res["ce"]["4_l0_only_full"] = round(ce4, 4)

    # 5 selection overlap (as-configured L0 ratio and intended 0.30)
    l0_now = float(trainer._l0_retention_for(ds))
    res["overlap_as_configured"] = selection_overlap(trainer, sample, l0_now, l1_ramp)
    res["overlap_intended_030"] = selection_overlap(trainer, sample, 0.30, l1_ramp)

    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    assert cfg.training.phase == "phase2_kb"
    trainer = KRKBTrainer(cfg)
    trainer.setup()
    trainer.load_checkpoint(Path(CKPT_NEW))
    logger.info("checkpoint_loaded", step=trainer.global_step, which="9164")
    trainer.model.eval()

    families = os.environ.get("DIAG_FAMILIES", "qwen35,falcon_h1").split(",")
    n_samples = int(os.environ.get("DIAG_N_FULL", "6"))
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
                if sum(1 for p in picked
                       if trainer._repo_group_key(p) == root) < 1:
                    picked.append(s)
        if scanned >= scan_cap or len(picked) >= n_samples:
            break
    if not picked:
        raise RuntimeError("no full-drill samples found")
    logger.info("samples_picked", n=len(picked), scanned=scanned)

    report: dict = {"ckpt_new": CKPT_NEW, "ckpt_old": CKPT_OLD, "rows": []}

    def run_pass(tag: str, key_rungs_only: bool):
        trainer._eval_tree_cache = {}
        trainer._eval_shared_tree_key = None
        with torch.no_grad():
            for family in [f.strip() for f in families if f.strip()]:
                if trainer._round_robin:
                    trainer._set_active_decoder(family)
                trainer._eval_tree_cache = {}
                trainer._eval_shared_tree_key = None
                for s in picked:
                    r = ladder(trainer, s, family, tag, key_rungs_only)
                    report["rows"].append(r)
                    print("\n" + "=" * 100)
                    print(json.dumps(r, indent=2, default=str))

    run_pass("9164", key_rungs_only=False)

    # rung 6: 8859 on the key rungs (1 + 3)
    trainer.load_checkpoint(Path(CKPT_OLD))
    logger.info("checkpoint_loaded", step=trainer.global_step, which="8859")
    trainer.model.eval()
    run_pass("8859", key_rungs_only=True)

    # ---- aggregate table (token-weighted mean CE per rung x family x ckpt) ----
    aggs: dict = {}
    for r in report["rows"]:
        if "ce" not in r:
            continue
        key = (r["ckpt"], r["family"])
        a = aggs.setdefault(key, {})
        for rung, v in r["ce"].items():
            s, w = a.get(rung, (0.0, 0))
            a[rung] = (s + v * r["answer_tokens"], w + r["answer_tokens"])
    print("\n" + "#" * 100)
    for (ck, fam), a in sorted(aggs.items()):
        tbl = {k: round(s / max(w, 1), 4) for k, (s, w) in sorted(a.items())}
        print(f"LADDER [{ck}/{fam}]:", json.dumps(tbl))
        report.setdefault("agg", {})[f"{ck}/{fam}"] = tbl
    # overlap aggregate
    for which in ("overlap_as_configured", "overlap_intended_030"):
        rows = [r[which] for r in report["rows"] if which in r]
        if rows:
            print(f"OVERLAP {which}: mean_gold_hit=" + str(round(
                sum(x["gold_line_hit"] for x in rows if x["gold_line_hit"] >= 0)
                / max(1, sum(1 for x in rows if x["gold_line_hit"] >= 0)), 3,
            )) + " mean_frac=" + str(round(
                sum(x["delivered_frac_of_diff"] for x in rows) / len(rows), 4,
            )))

    out_path = Path("/workspace/checkpoints/diag_git_repro_decode_ladder.json")
    try:
        out_path.write_text(json.dumps(report, indent=2, default=str))
        logger.info("report_written", path=str(out_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("report_write_failed", err=str(exc))


if __name__ == "__main__":
    main()
