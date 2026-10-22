#!/usr/bin/env python
"""Diagnostic: is git-repro drill-down QUERY-CONDITIONED, and does it matter?

Follow-up to diag_git_repro_retrieve_splice.py (quota fix now applied in
selection.py). The intended architecture (user): every drill instance
re-compresses the drilled node's CHILD SURVIVORS under a PER-QUERY compression
prompt (~0.08 retention is fine because the query keeps the RIGHT 8%). The
code today: only the HEAD drill is query-conditioned; deeper node drills
replay the static general-prompt memo rep; retrieve leaves live-encode with an
EMPTY query ("" -> token-0 prompt).

Measurements per full-drill eval sample (eval mode, no_grad):

  A    baseline (current behavior, sampled drill ratios)
  A08  leaf drills at ratio 0.08, query still ""            (ratio control)
  B08  leaf drills at ratio 0.08, query = the sample's HEAD task_query
       (query-conditioned L0 + L1 at the retrieval leaf)
  C    node drills re-encoded LIVE from memo child survivors under the task
       query (the head mechanism generalized to every drill site), ratio 0.08
  D    B08 + C (the user's intended architecture end-to-end)
  Token-splice selection probe (decoder-readable, isolates SELECTION quality
  from the decoder's untrained embedding-reading):
  T1   full raw diff text before the answer (upper bound)
  T2   only the diff tokens surviving L0@0.08 with the CURRENT empty prompt
  T3   only the diff tokens surviving L0@0.08 with the TASK-QUERY prompt
  plus L0 survivor-set Jaccard (empty vs task query) and gold-line hit rates.

Run (container, trainer stopped):
  docker compose --env-file .env -f docker/docker-compose.yaml run --rm -T \
    train-phase2-kb-git-repro-fullbackprop \
    python scripts/diag_git_repro_query_conditioning.py \
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

logger = structlog.get_logger()

DEFAULT_CKPT = (
    "/workspace/checkpoints_fast/"
    "phase2_kb_step8859_20260729_175947_588332_run-phase2_kb_git_repro_fullbackprop"
)

DRILL_RATIO = float(os.environ.get("DIAG_DRILL_RATIO", "0.08"))


# ---------------------------------------------------------------------------
# Shared helpers (mirrors diag_git_repro_retrieve_splice.py)
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


def recon_ce(trainer, segments, span) -> tuple[float, int]:
    out = trainer.decoder.forward_interleaved_with_loss(
        segments, return_hidden_states=True,
    )
    s, c = KRKBTrainer._span_ce_sum_count(out, [span])
    del out
    return (s / max(c, 1), c)


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
    new = list(segments[:idx]) + [ins] + list(segments[idx:])
    n = int(tok_ids.shape[0])
    return new, (span[0] + n, span[1] + n)


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


# ---------------------------------------------------------------------------
# L0 selection probe (which diff tokens survive under which prompt)
# ---------------------------------------------------------------------------


def l0_survivor_token_ids(
    trainer, dataset: str, doc_id: str, query: str | None, ratio: float,
) -> tuple[list[int], list[int]]:
    """Run live L0 on one doc at ``ratio`` (exact_topk) with the given query
    prompt. Returns ``(survivor_positions, survivor_token_ids)``."""
    embed_tokens = trainer.encoder.l0.backbone.get_input_embeddings()
    if query:
        q_ids = trainer.encoder_tokenizer.encode(query, add_special_tokens=False) or [0]
    else:
        q_ids = [0]  # mirrors _prepare_l1_turn's encode("") or [0]
    q_emb = embed_tokens(
        torch.tensor(q_ids, dtype=torch.long, device=trainer.device),
    )
    out, _cu, _r = trainer._live_l0_encode(
        dataset, [doc_id], query_emb=q_emb, ratio=ratio,
        selection_mode="exact_topk",
    )
    mask = out.survivor_mask.bool().cpu()
    toks = trainer._token_store.get(dataset, doc_id).cpu()
    pos = mask.nonzero(as_tuple=True)[0].tolist()
    return pos, [int(toks[p]) for p in pos]


def gold_line_hit_rate(text: str, gold: str, min_sub: int = 16) -> float:
    lines = [ln.strip() for ln in gold.splitlines() if len(ln.strip()) >= min_sub]
    if not lines:
        return -1.0
    hit = 0
    for ln in lines:
        # any >=min_sub-char slice of the line present in the survivor text
        found = any(
            ln[i:i + min_sub] in text
            for i in range(0, max(1, len(ln) - min_sub + 1), min_sub)
        )
        hit += bool(found)
    return round(hit / len(lines), 3)


# ---------------------------------------------------------------------------
# Variant survivor builders
# ---------------------------------------------------------------------------


def is_leaf_turn(t) -> bool:
    return t is not None and not (isinstance(t, dict) and "mode" in t)


def is_node_turn(t) -> bool:
    return isinstance(t, dict) and t.get("mode") == "node"


def leaf_variant_turns(trainer, sample, prep, query: str) -> list:
    """Copy prepared_turns with every LEAF (retrieve) turn rebuilt via
    _prepare_l1_turn under ``query`` (query-conditioned L0 + L1 prompt)."""
    ds = sample.dataset_name
    tree = trainer._trees.get(ds)
    rendered = prep["rendered"]
    out = []
    for k, turn in enumerate(rendered.bgkit_turns):
        t = prep["prepared_turns"][k]
        if is_leaf_turn(t):
            ids = list(turn.args.get("ids", []))
            out.append(trainer._prepare_l1_turn(ds, ids, query))
        else:
            out.append(t)
    del tree
    return out


def node_query_survivor(trainer, sample, node_id: str, query: str) -> torch.Tensor:
    """Query-conditioned LIVE re-encode of a drilled node from its children's
    memo survivors (the head mechanism generalized), leaf-tags via
    _encode_leaf_subtree."""
    ds = sample.dataset_name
    tree = trainer._trees.get(ds)
    node = tree.get(node_id)
    if node.children:
        return trainer._shared_tree_head_survivor(node_id, query, ds)
    # commit leaf-tag: re-encode its articles' L0 under the query
    q_emb = trainer._head_query_emb(query)
    proj, _l1 = trainer._encode_leaf_subtree(ds, node, q_emb)
    if proj is None:
        return trainer._drilldown_zero_survivor()
    return proj


# ---------------------------------------------------------------------------
# Per-sample analysis
# ---------------------------------------------------------------------------


def analyze(trainer, sample, family: str) -> dict:
    ds = sample.dataset_name
    res: dict = {
        "family": family,
        "question": sample.question[:120],
        "turn_kinds": classify_turns(trainer, sample),
    }
    task_query = head_task_query(sample)
    res["task_query"] = task_query[:160]

    trainer._ensure_eval_shared_tree(sample)
    prep = trainer._prepare_sample_for_decode(sample)
    turns = prep["prepared_turns"]

    # ---- A: baseline (current behavior) ----
    survs_a = trainer._run_l1_batch(turns)
    seg_a, tr_a = trainer._assemble_sample_segments(prep, survs_a)
    span = tr_a.answer_span
    if span is None:
        res["skip"] = "no answer span"
        return res
    ce_a, n_tok = recon_ce(trainer, seg_a, span)

    # ---- A08: leaves at DRILL_RATIO, still query-"" ----
    survs_a08 = trainer._run_l1_batch(turns, target_ratio=DRILL_RATIO)
    seg, tr = trainer._assemble_sample_segments(prep, survs_a08)
    ce_a08, _ = recon_ce(trainer, seg, tr.answer_span)

    # ---- B08: leaves at DRILL_RATIO with the task query ----
    turns_b = leaf_variant_turns(trainer, sample, prep, task_query)
    survs_b = trainer._run_l1_batch(turns_b, target_ratio=DRILL_RATIO)
    seg, tr = trainer._assemble_sample_segments(prep, survs_b)
    ce_b08, _ = recon_ce(trainer, seg, tr.answer_span)

    # leaf survivor accounting A08 vs B08
    def leaf_stats(vturns, vsurvs):
        rows = []
        for i, t in enumerate(vturns):
            if is_leaf_turn(t):
                rows.append({
                    "turn": i,
                    "N": int(t["content"].shape[0]),
                    "pin": int(t["pinned"].sum()),
                    "l0_rows": int(t["survivor_mask"].sum()),
                    "out_rows": int(vsurvs[i].shape[0]),
                })
        return rows

    res["leaf_stats_a08"] = leaf_stats(turns, survs_a08)
    res["leaf_stats_b08"] = leaf_stats(turns_b, survs_b)

    # ---- C: node drills re-encoded under the task query (ratio 0.08) ----
    saved_cfg = getattr(trainer, "_recursive_l1_retention_cfg", None)
    trainer._recursive_l1_retention_cfg = DRILL_RATIO
    try:
        survs_c = list(survs_a)
        n_node = 0
        for i, t in enumerate(turns):
            if is_node_turn(t):
                survs_c[i] = node_query_survivor(
                    trainer, sample, str(t["node_id"]), task_query,
                )
                n_node += 1
        seg, tr = trainer._assemble_sample_segments(prep, survs_c)
        ce_c, _ = recon_ce(trainer, seg, tr.answer_span)

        # ---- D: B08 leaves + C nodes ----
        survs_d = list(survs_b)
        for i, t in enumerate(turns):
            if is_node_turn(t):
                survs_d[i] = survs_c[i]
        seg, tr = trainer._assemble_sample_segments(prep, survs_d)
        ce_d, _ = recon_ce(trainer, seg, tr.answer_span)
    finally:
        trainer._recursive_l1_retention_cfg = saved_cfg
    res["n_node_drills_reencoded"] = n_node
    res["node_rows_static_vs_query"] = [
        (int(survs_a[i].shape[0]), int(survs_c[i].shape[0]))
        for i, t in enumerate(turns) if is_node_turn(t)
    ]

    res["recon_ce"] = {
        "answer_tokens": n_tok,
        "A_baseline": round(ce_a, 4),
        "A08_leaf_ratio08_noquery": round(ce_a08, 4),
        "B08_leaf_ratio08_taskquery": round(ce_b08, 4),
        "C_nodes_taskquery": round(ce_c, 4),
        "D_leaves_and_nodes_taskquery": round(ce_d, 4),
    }

    # ---- Token-splice selection probe (T1/T2/T3) ----
    docs = retrieve_doc_ids(trainer, sample)
    gold = str(sample.gold_answer)
    if docs:
        full_texts, t2_texts, t3_texts, jaccards = [], [], [], []
        for did in docs:
            toks = trainer._token_store.get(ds, did)
            full_texts.append(trainer.encoder_tokenizer.decode(toks.tolist()))
            pos2, tid2 = l0_survivor_token_ids(trainer, ds, did, None, DRILL_RATIO)
            pos3, tid3 = l0_survivor_token_ids(
                trainer, ds, did, task_query, DRILL_RATIO,
            )
            t2_texts.append(trainer.encoder_tokenizer.decode(tid2))
            t3_texts.append(trainer.encoder_tokenizer.decode(tid3))
            s2, s3 = set(pos2), set(pos3)
            if s2 | s3:
                jaccards.append(len(s2 & s3) / len(s2 | s3))
        res["l0_selection_jaccard_noq_vs_q"] = [round(j, 3) for j in jaccards]

        def ce_with_text(texts, tag):
            blob = "\n\n[retrieved diffs]\n" + "\n\n".join(texts) + "\n[/retrieved diffs]\n"
            ids = trainer.tokenizer.encode(blob, add_special_tokens=False)[:3072]
            tok_t = torch.tensor(ids, dtype=torch.long, device=trainer.device)
            seg_x, span_x = insert_before_answer(seg_a, span, tok_t)
            ce, _ = recon_ce(trainer, seg_x, span_x)
            return round(ce, 4), len(ids), gold_line_hit_rate(blob, gold)

        res["T1_fulltext"], res["T1_tokens"], res["T1_gold_hit"] = ce_with_text(
            full_texts, "T1",
        )
        res["T2_l0_08_noquery"], res["T2_tokens"], res["T2_gold_hit"] = ce_with_text(
            t2_texts, "T2",
        )
        res["T3_l0_08_taskquery"], res["T3_tokens"], res["T3_gold_hit"] = ce_with_text(
            t3_texts, "T3",
        )
        res["survivor_text_example_noq"] = t2_texts[0][:220]
        res["survivor_text_example_q"] = t3_texts[0][:220]
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
    ckpt = os.environ.get("DIAG_CKPT", DEFAULT_CKPT)
    trainer.load_checkpoint(Path(ckpt))
    logger.info("checkpoint_loaded", step=trainer.global_step)
    trainer.model.eval()
    trainer._eval_tree_cache = {}
    trainer._eval_shared_tree_key = None

    families = os.environ.get("DIAG_FAMILIES", "qwen35,falcon_h1").split(",")
    n_full = int(os.environ.get("DIAG_N_FULL", "4"))
    scan_cap = int(os.environ.get("DIAG_SCAN_CAP", "4000"))

    picked: list = []
    scanned = 0
    for batch in trainer.eval_dataloader:
        for s in batch:
            scanned += 1
            if s.dataset_name != "git_commit_repro":
                continue
            kinds = classify_turns(trainer, s)
            if sum(1 for k in kinds if k == "retrieve") >= 1 and "node" in kinds:
                root = trainer._repo_group_key(s)
                if sum(1 for p in picked
                       if trainer._repo_group_key(p) == root) < 2:
                    picked.append(s)
        if scanned >= scan_cap or len(picked) >= n_full:
            break
    if not picked:
        raise RuntimeError("no full-drill samples found")
    logger.info("samples_picked", n=len(picked), scanned=scanned)

    report: dict = {"checkpoint": ckpt, "drill_ratio": DRILL_RATIO, "samples": []}
    with torch.no_grad():
        for family in [f.strip() for f in families if f.strip()]:
            if trainer._round_robin:
                trainer._set_active_decoder(family)
            trainer._eval_tree_cache = {}
            trainer._eval_shared_tree_key = None
            for s in picked:
                r = analyze(trainer, s, family)
                report["samples"].append(r)
                print("\n" + "=" * 100)
                print(json.dumps(r, indent=2, default=str))

    # aggregate (token-weighted) per family
    for family in [f.strip() for f in families if f.strip()]:
        rows = [r for r in report["samples"]
                if r["family"] == family and "recon_ce" in r]
        agg = {}
        keys = set()
        for r in rows:
            keys |= set(r["recon_ce"])
            for k in ("T1_fulltext", "T2_l0_08_noquery", "T3_l0_08_taskquery"):
                if k in r:
                    keys.add(k)
        for k in sorted(keys - {"answer_tokens"}):
            vals = []
            for r in rows:
                v = r["recon_ce"].get(k, r.get(k))
                if v is not None:
                    vals.append((float(v), r["recon_ce"]["answer_tokens"]))
            if vals:
                tot = sum(w for _, w in vals)
                agg[k] = round(sum(v * w for v, w in vals) / max(tot, 1), 4)
        print(f"\nAGG token-weighted recon CE [{family}]:", json.dumps(agg, indent=2))
        report.setdefault("agg", {})[family] = agg

    out = Path("/workspace/checkpoints/diag_git_repro_query_conditioning.json")
    try:
        out.write_text(json.dumps(report, indent=2, default=str))
        logger.info("report_written", path=str(out))
    except Exception as exc:  # noqa: BLE001
        logger.warning("report_write_failed", err=str(exc))


if __name__ == "__main__":
    main()
