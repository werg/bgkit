#!/usr/bin/env python
"""Comprehensive decode-ladder part 2 — remaining stage isolations (2026-07-30).

Complements diag_git_repro_decode_ladder.py (rungs: current/1b/text-oracle/
emb-oracle-full[R3]/L0-only-full[S1]/overlap/ckpt-compare) and
diag_git_repro_prefix_splice.py (X1/F3 prefix framing). This script adds, on
the step-9164 checkpoint, live-trajectory framing unless stated:

  F1   retrieve reps zeroed (no-context ceiling, live slot)
  F2s  raw diff TEXT swapped into the retrieve SLOT (TokenSegment replaces the
       EmbeddingSegment — position controlled, modality varied)
  R2   leaf L1=0.50 (retention sweep mid-point; R1=0.30 and R3=1.0 in part 1)
  S2   L0-only ratio 0.30 -> projection in the retrieve slot (L1 skipped)
  S3   L0 ratio 1.0 -> L1 ratio 0.30 (full L0 into L1; separates L0- vs L1-loss)
  S4   bridge OFF: L1 fed RAW L0 survivors (no auto_repro_head) at 0.30/0.30
  Sel2 RANDOM 30% of L0 rows (vs exact_topk 30% = part-1 rung 1b): does the
       CHOICE matter, or are reps distributed?
  X2   R3 survivors MOVED to immediately before the answer (slot zeroed)
  Pid  projection SKIPPED: raw L1 survivor rows spliced (qwen only, dims match)
  E1   Phase-1-regime control: the SAME diff as a standalone doc, L0-only
       exact_topk@0.10 -> projection, PREFIX framing, gold = THE DIFF ITSELF
       (does the known-good regime reconstruct diffs at all?). References:
       E1r10 (ratio 1.0) and E1nc (no context) floors/ceilings.
  E2   argmax preview of E1@0.10 (what the decoder reads out of the reps)

Run after part 1 finishes (one GPU job at a time):
  docker compose --env-file .env -f docker/docker-compose.yaml run --rm -T \
    train-phase2-kb-git-repro-fullbackprop \
    python scripts/diag_git_repro_ladder2.py \
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

from bgkit.models.decoder import EmbeddingSegment, TokenSegment
from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
from bgkit.utils.logging import setup_logging
from bgkit.utils.packing import position_ids_from_cu

logger = structlog.get_logger()

CKPT = os.environ.get(
    "DIAG_CKPT",
    "/workspace/checkpoints_fast/"
    "phase2_kb_step9164_20260730_091047_791250_run-phase2_kb_git_repro_fullbackprop",
)


# ---------------------------------------------------------------------------
# helpers (shared shape with part 1)
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


def retrieve_turn_info(trainer, sample):
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


@contextlib.contextmanager
def forced_l0_ratio(trainer, ratio: float | None):
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


def is_leaf_i(turns, i) -> bool:
    return turns[i] is not None and not (
        isinstance(turns[i], dict) and "mode" in turns[i]
    )


def emb_segment_turn_order(prep) -> list[int]:
    """bgkit-turn index of each EMB segment, in splice order (no topic block
    for git-repro; splice events sorted by rendered position == turn order)."""
    return list(range(len(prep["prepared_turns"])))


def swap_segments(trainer, prep, survs, replacements: dict, span):
    """Assemble with ``survs`` then replace the EMB segment of turn k with
    ``replacements[k]`` (a Segment). Returns (segments, shifted span)."""
    segments, trace = trainer._assemble_sample_segments(prep, survs)
    span = trace.answer_span
    out = []
    emb_i = 0
    delta = 0
    order = emb_segment_turn_order(prep)
    offs = 0
    for seg in segments:
        length = seg_len(seg)
        if isinstance(seg, EmbeddingSegment):
            turn_k = order[emb_i]
            emb_i += 1
            if turn_k in replacements:
                new = replacements[turn_k]
                out.append(new)
                if offs + length <= span[0]:
                    delta += seg_len(new) - length
                offs += length
                continue
        out.append(seg)
        offs += length
    return out, (span[0] + delta, span[1] + delta)


def token_segment(trainer, text: str, cap: int = 3072) -> TokenSegment:
    ids = trainer.tokenizer.encode(text, add_special_tokens=False)[:cap]
    t = torch.tensor(ids, dtype=torch.long, device=trainer.device)
    return TokenSegment(
        token_ids=t.unsqueeze(0),
        loss_mask=torch.zeros((1, len(ids)), dtype=torch.bool, device=trainer.device),
    )


def insert_seg_before_answer(segments, span, new_seg):
    offs = 0
    idx = len(segments) - 1
    for i, seg in enumerate(segments):
        length = seg_len(seg)
        if offs <= span[0] < offs + length:
            idx = i
            break
        offs += length
    n = seg_len(new_seg)
    return (
        list(segments[:idx]) + [new_seg] + list(segments[idx:]),
        (span[0] + n, span[1] + n),
    )


def l0_only_survivor(trainer, ds, doc_id, q_emb, ratio: float | None):
    """encoder.forward, L1 skipped. ratio None => all rows survive; else
    exact_topk at ratio."""
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
        target_ratio_l0=ratio,
        target_ratio_l1=None,
        selection_mode_l0="exact_topk",
    )
    return out.survivor_embeddings


def leaf_l1_survivor(
    trainer, ds, doc_id, q_emb, l0_rows, l1_ratio, *, bridge=True, pin=True,
    return_l1_raw=False,
):
    """Manual leaf L1 from given L0 rows. Mirrors _run_l1_batch for one doc."""
    dev = trainer.device
    embed_tokens = trainer.encoder.l0.backbone.get_input_embeddings()
    pieces = []
    pinned_l = []
    if pin:
        pin_ids = trainer.encoder_tokenizer.encode(
            f" {doc_id}", add_special_tokens=False,
        ) or [0]
        id_emb = embed_tokens(torch.tensor(pin_ids, dtype=torch.long, device=dev))
        pieces.append(id_emb.to(torch.bfloat16))
        pinned_l += [True] * len(pin_ids)
    rows = (
        trainer.encoder.l0.auto_reproduce(l0_rows) if bridge else l0_rows
    ).to(torch.bfloat16)
    pieces.append(rows)
    pinned_l += [False] * int(rows.shape[0])
    content = torch.cat(pieces, dim=0)
    pinned = torch.tensor(pinned_l, dtype=torch.bool, device=dev)
    cu = torch.tensor([0, int(content.shape[0])], dtype=torch.int32, device=dev)
    qcu = torch.tensor([0, int(q_emb.shape[0])], dtype=torch.int32, device=dev)
    qpos = position_ids_from_cu(qcu, int(q_emb.shape[0]))
    l1_out, proj_out, proj_cu = trainer.encoder.run_l1_and_project(
        l1_input_embeddings=content,
        l1_input_cu_seqlens=cu,
        target_ratio_l1=float(l1_ratio),
        content_group_cu_seqlens=None,
        prompt_embeddings_l1=q_emb.to(torch.bfloat16),
        prompt_cu_seqlens_l1=qcu,
        prompt_position_ids_l1=qpos,
        pinned_positions_l1=pinned,
        selection_mode_l1="exact_topk",
    )
    if return_l1_raw:
        return l1_out.survivor_embeddings
    return proj_out.projected_embeddings


# ---------------------------------------------------------------------------
# E1/E2: Phase-1-regime standalone control
# ---------------------------------------------------------------------------


def phase1_regime_control(trainer, ds, doc_id, ratio) -> dict:
    """Standalone doc, L0-only exact_topk@ratio -> projection, PREFIX framing,
    gold = THE DIFF TEXT. Returns CE (+ argmax preview at ratio 0.10)."""
    dev = trainer.device
    diff_text = trainer.encoder_tokenizer.decode(
        trainer._token_store.get(ds, doc_id).tolist(),
    )
    q_emb = trainer._head_query_emb("reconstruct the document")
    prompt = "\nReconstruct the document:\n"
    p_ids = trainer.tokenizer.encode(prompt, add_special_tokens=False)
    g_ids = trainer.tokenizer.encode(diff_text, add_special_tokens=False)[:2048]
    segs = []
    ctx = 0
    if ratio is not None or ratio is None:  # context always present here
        surv = l0_only_survivor(trainer, ds, doc_id, q_emb, ratio)
        segs.append(EmbeddingSegment(embeddings=surv.unsqueeze(0)))
        ctx = int(surv.shape[0])
    p_t = torch.tensor(p_ids, dtype=torch.long, device=dev)
    g_t = torch.tensor(g_ids, dtype=torch.long, device=dev)
    segs.append(TokenSegment(
        token_ids=p_t.unsqueeze(0),
        loss_mask=torch.zeros((1, len(p_ids)), dtype=torch.bool, device=dev),
    ))
    segs.append(TokenSegment(
        token_ids=g_t.unsqueeze(0),
        loss_mask=torch.ones((1, len(g_ids)), dtype=torch.bool, device=dev),
    ))
    span = (ctx + len(p_ids), ctx + len(p_ids) + len(g_ids))
    out = trainer.decoder.forward_interleaved_with_loss(
        segs, return_hidden_states=True,
    )
    s, c = KRKBTrainer._span_ce_sum_count(out, [span])
    preds = out.argmax_predictions()[0]
    lo = max(0, span[0] - 1)
    hi = min(preds.shape[0], span[0] - 1 + 60)
    preview = trainer.tokenizer.decode(
        preds[lo:hi].tolist(), skip_special_tokens=True,
    )
    del out
    return {
        "ce": round(s / max(c, 1), 4),
        "n_rows": ctx,
        "preview": preview[:150],
        "diff_head": diff_text[:150],
    }


def phase1_nocontext(trainer, ds, doc_id) -> float:
    dev = trainer.device
    diff_text = trainer.encoder_tokenizer.decode(
        trainer._token_store.get(ds, doc_id).tolist(),
    )
    p_ids = trainer.tokenizer.encode(
        "\nReconstruct the document:\n", add_special_tokens=False,
    )
    g_ids = trainer.tokenizer.encode(diff_text, add_special_tokens=False)[:2048]
    segs = [
        TokenSegment(
            token_ids=torch.tensor(p_ids, device=dev).unsqueeze(0),
            loss_mask=torch.zeros((1, len(p_ids)), dtype=torch.bool, device=dev),
        ),
        TokenSegment(
            token_ids=torch.tensor(g_ids, device=dev).unsqueeze(0),
            loss_mask=torch.ones((1, len(g_ids)), dtype=torch.bool, device=dev),
        ),
    ]
    span = (len(p_ids), len(p_ids) + len(g_ids))
    out = trainer.decoder.forward_interleaved_with_loss(
        segs, return_hidden_states=True,
    )
    s, c = KRKBTrainer._span_ce_sum_count(out, [span])
    del out
    return round(s / max(c, 1), 4)


# ---------------------------------------------------------------------------
# per-sample ladder part 2
# ---------------------------------------------------------------------------


def ladder2(trainer, sample, family: str) -> dict:
    ds = sample.dataset_name
    res: dict = {"family": family, "question": sample.question[:100]}
    trainer._ensure_eval_shared_tree(sample)
    prep = trainer._prepare_sample_for_decode(sample)
    turns = prep["prepared_turns"]
    l1_ramp = float(trainer._recursive_l1_retention_now())
    q_emb_task = trainer._head_query_emb(head_task_query(sample))
    infos = retrieve_turn_info(trainer, sample)

    survs_cur = trainer._run_l1_batch(turns, target_ratio=l1_ramp)
    seg0, tr0 = trainer._assemble_sample_segments(prep, survs_cur)
    span0 = tr0.answer_span
    if span0 is None:
        res["skip"] = "no answer"
        return res
    ce_cur, n_tok = recon_ce(trainer, seg0, span0)
    res["answer_tokens"] = n_tok
    ce: dict = {"X3_current_R1": round(ce_cur, 4)}

    zero1 = torch.zeros(
        (1, trainer.encoder.active_projection_output_dim),
        device=trainer.device, dtype=torch.bfloat16,
    )

    # F1: retrieve reps zeroed
    sv = [zero1 if is_leaf_i(turns, i) else s for i, s in enumerate(survs_cur)]
    seg, tr = trainer._assemble_sample_segments(prep, sv)
    ce["F1_retrieve_zeroed"], _ = recon_ce(trainer, seg, tr.answer_span)

    # F2s: diff TEXT swapped into the retrieve SLOT
    reps = {}
    for k, ids, _q in infos:
        texts = [
            trainer.encoder_tokenizer.decode(
                trainer._token_store.get(ds, d).tolist(),
            )
            for d in trainer._resolve_article_ids(ds, ids)
        ]
        reps[k] = token_segment(trainer, "\n".join(texts), cap=1024)
    seg, spanx = swap_segments(trainer, prep, survs_cur, reps, span0)
    ce["F2s_text_in_slot"], _ = recon_ce(trainer, seg, spanx)

    # R2: leaf L1 = 0.50
    sv_r2_all = trainer._run_l1_batch(turns, target_ratio=0.50)
    sv = [sv_r2_all[i] if is_leaf_i(turns, i) else survs_cur[i]
          for i in range(len(turns))]
    seg, tr = trainer._assemble_sample_segments(prep, sv)
    ce["R2_leaf_l1_050"], _ = recon_ce(trainer, seg, tr.answer_span)

    # S2: L0-only @0.30 -> projection, in slot
    sv = list(survs_cur)
    for k, ids, _q in infos:
        parts = [
            l0_only_survivor(trainer, ds, d, q_emb_task, 0.30)
            for d in trainer._resolve_article_ids(ds, ids)
        ]
        if parts:
            sv[k] = torch.cat(parts, dim=0)
    seg, tr = trainer._assemble_sample_segments(prep, sv)
    ce["S2_l0only_030_in_slot"], _ = recon_ce(trainer, seg, tr.answer_span)

    # S3: L0@1.0 -> L1@0.30 (full L0 into L1)
    turns_s3 = rebuild_leaf_turns(trainer, sample, prep, l0_ratio=1.0)
    sv_s3_all = trainer._run_l1_batch(turns_s3, target_ratio=0.30)
    sv = [sv_s3_all[i] if is_leaf_i(turns, i) else survs_cur[i]
          for i in range(len(turns))]
    seg, tr = trainer._assemble_sample_segments(prep, sv)
    ce["S3_l0full_l1_030"], _ = recon_ce(trainer, seg, tr.answer_span)

    # S4: bridge OFF (raw L0 rows into L1), L0 0.30 / L1 0.30
    sv = list(survs_cur)
    for k, ids, q in infos:
        parts = []
        for d in trainer._resolve_article_ids(ds, ids):
            out, _cu, _r = trainer._live_l0_encode(
                ds, [d], query_emb=q_emb_task, ratio=0.30,
                selection_mode="exact_topk",
            )
            parts.append(leaf_l1_survivor(
                trainer, ds, d, q_emb_task, out.survivor_embeddings,
                0.30, bridge=False,
            ))
        if parts:
            sv[k] = torch.cat(parts, dim=0)
    seg, tr = trainer._assemble_sample_segments(prep, sv)
    ce["S4_bridge_off"], _ = recon_ce(trainer, seg, tr.answer_span)

    # Sel2: RANDOM 30% of L0 rows (vs exact_topk 30%)
    g = torch.Generator(device="cpu").manual_seed(17)
    sv = list(survs_cur)
    for k, ids, q in infos:
        parts = []
        for d in trainer._resolve_article_ids(ds, ids):
            out, _cu, _r = trainer._live_l0_encode(
                ds, [d], query_emb=q_emb_task, ratio=1.0,
            )
            rows = out.survivor_embeddings
            n = int(rows.shape[0])
            keep = max(1, int(round(0.30 * n)))
            idx = torch.randperm(n, generator=g)[:keep].sort().values.to(rows.device)
            parts.append(leaf_l1_survivor(
                trainer, ds, d, q_emb_task, rows[idx], 0.30, bridge=True,
            ))
        if parts:
            sv[k] = torch.cat(parts, dim=0)
    seg, tr = trainer._assemble_sample_segments(prep, sv)
    ce["Sel2_random30"], _ = recon_ce(trainer, seg, tr.answer_span)

    # X2: R3 lossless survivors MOVED to just before the answer (slot zeroed)
    turns_r3 = rebuild_leaf_turns(trainer, sample, prep, l0_ratio=1.0)
    sv_r3_all = trainer._run_l1_batch(turns_r3, target_ratio=1.0)
    full_leaf = torch.cat(
        [sv_r3_all[i] for i in range(len(turns)) if is_leaf_i(turns, i)], dim=0,
    )
    sv = [zero1 if is_leaf_i(turns, i) else survs_cur[i]
          for i in range(len(turns))]
    seg, tr = trainer._assemble_sample_segments(prep, sv)
    seg, spanx = insert_seg_before_answer(
        seg, tr.answer_span, EmbeddingSegment(embeddings=full_leaf.unsqueeze(0)),
    )
    ce["X2_R3_before_answer"], _ = recon_ce(trainer, seg, spanx)
    res["x2_rows"] = int(full_leaf.shape[0])

    # Pid: projection skipped (raw L1 survivors) — only when dims match
    if trainer.encoder.active_projection_output_dim == int(
        trainer.encoder.l1.hidden_dim,
    ):
        sv = list(survs_cur)
        for k, ids, q in infos:
            parts = []
            for d in trainer._resolve_article_ids(ds, ids):
                out, _cu, _r = trainer._live_l0_encode(
                    ds, [d], query_emb=q_emb_task, ratio=0.30,
                    selection_mode="exact_topk",
                )
                parts.append(leaf_l1_survivor(
                    trainer, ds, d, q_emb_task, out.survivor_embeddings,
                    0.30, bridge=True, return_l1_raw=True,
                ).to(torch.bfloat16))
            if parts:
                sv[k] = torch.cat(parts, dim=0)
        seg, tr = trainer._assemble_sample_segments(prep, sv)
        ce["Pid_no_projection"], _ = recon_ce(trainer, seg, tr.answer_span)
    else:
        res["Pid_note"] = "dims mismatch — projection not separable"

    res["ce"] = {k: round(float(v), 4) for k, v in ce.items()}

    # E1/E2: Phase-1-regime standalone control on the first 2 docs
    e1 = []
    docs = []
    for _k, ids, _q in infos:
        docs.extend(trainer._resolve_article_ids(ds, ids))
    for d in docs[:2]:
        row = {
            "doc": d[-60:],
            "E1nc_no_context": phase1_nocontext(trainer, ds, d),
            "E1_l0only_010": phase1_regime_control(trainer, ds, d, 0.10),
            "E1r10_l0only_full": phase1_regime_control(trainer, ds, d, None),
        }
        e1.append(row)
    res["E1"] = e1
    return res


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    assert cfg.training.phase == "phase2_kb"
    trainer = KRKBTrainer(cfg)
    trainer.setup()
    trainer.load_checkpoint(Path(CKPT))
    logger.info("checkpoint_loaded", step=trainer.global_step)
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
    logger.info("samples_picked", n=len(picked), scanned=scanned)

    report: dict = {"checkpoint": CKPT, "rows": []}
    with torch.no_grad():
        for family in [f.strip() for f in families if f.strip()]:
            if trainer._round_robin:
                trainer._set_active_decoder(family)
            trainer._eval_tree_cache = {}
            trainer._eval_shared_tree_key = None
            for s in picked:
                r = ladder2(trainer, s, family)
                report["rows"].append(r)
                print("\n" + "=" * 100)
                print(json.dumps(r, indent=2, default=str))

    # aggregate
    for family in [f.strip() for f in families if f.strip()]:
        rows = [r for r in report["rows"]
                if r["family"] == family and "ce" in r]
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r["ce"]})
        agg = {}
        for k in keys:
            vals = [(r["ce"][k], r["answer_tokens"]) for r in rows if k in r["ce"]]
            tot = sum(w for _, w in vals)
            agg[k] = round(sum(v * w for v, w in vals) / max(tot, 1), 4)
        # E1 aggregate (per-doc mean)
        e1rows = [d for r in rows for d in r.get("E1", [])]
        if e1rows:
            agg["E1nc_no_context"] = round(
                sum(d["E1nc_no_context"] for d in e1rows) / len(e1rows), 4,
            )
            agg["E1_l0only_010"] = round(
                sum(d["E1_l0only_010"]["ce"] for d in e1rows) / len(e1rows), 4,
            )
            agg["E1r10_l0only_full"] = round(
                sum(d["E1r10_l0only_full"]["ce"] for d in e1rows) / len(e1rows), 4,
            )
        print(f"\nLADDER2 AGG [{family}]:", json.dumps(agg, indent=2))
        report.setdefault("agg", {})[family] = agg

    out = Path("/workspace/checkpoints/diag_git_repro_ladder2.json")
    try:
        out.write_text(json.dumps(report, indent=2, default=str))
        logger.info("report_written", path=str(out))
    except Exception as exc:  # noqa: BLE001
        logger.warning("report_write_failed", err=str(exc))


if __name__ == "__main__":
    main()
