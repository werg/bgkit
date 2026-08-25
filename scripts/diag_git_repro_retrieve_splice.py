#!/usr/bin/env python
"""Diagnostic: does the git-repro RETRIEVE drill splice actual diff content?

Investigates the depth-flat eval recon_loss (~1.5 at every drill depth) on
phase2_kb_git_repro_fullbackprop. Hypothesis chain:

  H1 (wiring): the retrieve turn splices only the pinned file-change-ID token
      rows; ZERO diff-content survivors reach the decoder because
      ``exact_ratio_topk_select`` deducts the pinned count from the total keep
      quota: ``remaining = clamp(ceil(valid*ratio) - n_pinned, 0)`` — at ratio
      0.05 with 20-40 pinned ID tokens the content quota is 0.
  H2 (same mechanism up the tree): every ``encode_tree_node`` call pins the
      children's (long, path-shaped) ID tokens, so interior/commit node reps
      are ID-only too.

What it does (eval mode, no_grad, one repo's eval samples):
  1. Dumps the decoder segment layout for full-drill samples.
  2. Per drill turn: content length, n_pinned, L0-survivor rows, output rows,
     and the pinned/content split of the surviving rows.
  3. Instruments exact_ratio_topk_select during the shared-tree encode to log
     (n_valid, n_pinned, n_selected, n_content_selected) per call.
  4. Ablations: recon CE with (a) all reps, (b) retrieve splices zeroed,
     (c) tree (head+node) splices zeroed, (d) all zeroed.
  5. Oracle: raw retrieved-diff TEXT spliced as a loss-masked TokenSegment
     right before the answer -> recon CE.
  6. Quota fix preview (first full-drill sample per family): re-runs drill +
     tree with pinned OUTSIDE the keep quota
     (``remaining = ceil(controllable * ratio)``) and re-measures recon CE.

Run (container, trainer stopped):
  docker compose --env-file .env -f docker/docker-compose.yaml run --rm \
    train-phase2-kb-git-repro-fullbackprop \
    python scripts/diag_git_repro_retrieve_splice.py \
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

from bgkit.models.components.selection import (
    exact_ratio_topk_select as TRUE_ORIG_TOPK,
)
from bgkit.models.decoder import EmbeddingSegment, TokenSegment
from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
from bgkit.utils.logging import setup_logging
from bgkit.utils.packing import position_ids_from_cu

logger = structlog.get_logger()

DEFAULT_CKPT = (
    "/workspace/checkpoints_fast/"
    "phase2_kb_step8859_20260729_175947_588332_run-phase2_kb_git_repro_fullbackprop"
)

# ---------------------------------------------------------------------------
# exact_ratio_topk_select instrumentation + quota-fix variant
# ---------------------------------------------------------------------------

TOPK_STATS: list[dict] = []
_PATCH_TARGETS = ("bgkit.models.level_compressor", "bgkit.models.components.selection")


def _set_topk(fn) -> None:
    import importlib

    for name in _PATCH_TARGETS:
        mod = importlib.import_module(name)
        mod.exact_ratio_topk_select = fn


def make_probe(tag: str):
    def probe(logits, valid_mask, target_ratio, cu_seqlens, pinned=None, min_per_sample=0):
        out = TRUE_ORIG_TOPK(
            logits, valid_mask, target_ratio, cu_seqlens,
            pinned=pinned, min_per_sample=min_per_sample,
        )
        with torch.no_grad():
            pin = (
                (pinned & valid_mask)
                if pinned is not None
                else torch.zeros_like(valid_mask)
            )
            TOPK_STATS.append({
                "tag": tag,
                "ratio": float(target_ratio),
                "n_valid": int(valid_mask.sum()),
                "n_pinned": int(pin.sum()),
                "n_selected": int(out.mask.sum()),
                "n_content_selected": int((out.mask & ~pin).sum()),
                "n_segs": int(cu_seqlens.shape[0]) - 1,
            })
        return out

    return probe


def fixed_topk(logits, valid_mask, target_ratio, cu_seqlens, pinned=None, min_per_sample=0):
    """Quota-fix variant: pinned sit OUTSIDE the keep quota.

    Selection over the CONTROLLABLE (non-pinned) positions picks
    ceil(controllable * ratio) per segment; the pinned mask is OR'd back in.
    """
    if pinned is None:
        return TRUE_ORIG_TOPK(
            logits, valid_mask, target_ratio, cu_seqlens,
            pinned=None, min_per_sample=min_per_sample,
        )
    pin = pinned & valid_mask
    controllable = valid_mask & ~pin
    sub = TRUE_ORIG_TOPK(
        logits, controllable, target_ratio, cu_seqlens,
        pinned=None, min_per_sample=min_per_sample,
    )
    out = TRUE_ORIG_TOPK(
        logits, valid_mask, target_ratio, cu_seqlens,
        pinned=pinned, min_per_sample=min_per_sample,
    )
    object.__setattr__(out, "mask", (sub.mask | pin) & valid_mask)
    return out


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


def recon_ce(trainer, segments, span) -> tuple[float, int]:
    out = trainer.decoder.forward_interleaved_with_loss(
        segments, return_hidden_states=True,
    )
    s, c = KRKBTrainer._span_ce_sum_count(out, [span])
    del out
    return (s / max(c, 1), c)


def leaf_turn_survivor_split(trainer, t: dict) -> dict:
    """Re-run ONE leaf drill's L1 forward and split survivors pinned/content."""
    dev = trainer.device
    content = t["content"]
    pinned = t["pinned"]
    l0_mask = t["survivor_mask"]
    q = t["query_emb"]
    bridged = content.clone()
    if bool(l0_mask.any()):
        bridged[l0_mask] = trainer.encoder.l0.auto_reproduce(
            content[l0_mask],
        ).to(content.dtype)
    cu = torch.tensor([0, int(content.shape[0])], dtype=torch.int32, device=dev)
    qcu = torch.tensor([0, int(q.shape[0])], dtype=torch.int32, device=dev)
    qpos = position_ids_from_cu(qcu, int(q.shape[0]))
    ratio = trainer._sample_l1_retention()  # eval -> deterministic base
    l1_out, _proj_out, _proj_cu = trainer.encoder.run_l1_and_project(
        l1_input_embeddings=bridged,
        l1_input_cu_seqlens=cu,
        target_ratio_l1=ratio,
        content_group_cu_seqlens=None,
        prompt_embeddings_l1=q,
        prompt_cu_seqlens_l1=qcu,
        prompt_position_ids_l1=qpos,
        pinned_positions_l1=pinned,
        selection_mode_l1=getattr(trainer, "_selection_mode_l1", "threshold"),
    )
    mask = l1_out.survivor_mask
    return {
        "l1_ratio": ratio,
        "n_content_positions": int(content.shape[0]),
        "n_pinned_id_tokens": int(pinned.sum()),
        "n_l0_survivor_rows": int(l0_mask.sum()),
        "n_survivors_out": int(mask.sum()),
        "n_pinned_survivors": int((mask & pinned).sum()),
        "n_content_survivors": int((mask & l0_mask).sum()),
    }


def build_oracle_tokens(trainer, sample, cap_tokens: int = 3072) -> torch.Tensor | None:
    """Decoder-tokenized raw diff text of every retrieve turn's documents."""
    ds = sample.dataset_name
    tree = trainer._trees.get(ds)
    texts: list[str] = []
    for t in sample.trajectory:
        if t.kind != "bgkit" or bool(t.args.get("is_head", False)):
            continue
        ids = list(t.args.get("ids", []))
        q = str(t.args.get("query", ""))
        if q == "" and len(ids) == 1 and tree is not None and str(ids[0]) in tree:
            continue  # nav drill
        try:
            doc_ids = trainer._resolve_article_ids(ds, ids)
        except Exception as exc:  # noqa: BLE001
            logger.warning("oracle_resolve_failed", ids=ids, err=str(exc))
            continue
        for did in doc_ids:
            try:
                toks = trainer._token_store.get(ds, did)
            except Exception as exc:  # noqa: BLE001
                logger.warning("oracle_token_fetch_failed", doc=did, err=str(exc))
                continue
            texts.append(trainer.encoder_tokenizer.decode(toks.tolist()))
    if not texts:
        return None
    blob = "\n\n[retrieved diffs]\n" + "\n\n".join(texts) + "\n[/retrieved diffs]\n"
    ids = trainer.tokenizer.encode(blob, add_special_tokens=False)[:cap_tokens]
    return torch.tensor(ids, dtype=torch.long, device=trainer.device)


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


def dump_segments(trainer, segments, trace, prep) -> list[dict]:
    """Segment table; EMB rows labeled by their originating splice event."""
    # Reconstruct the splice-event order exactly as _assemble_sample_segments:
    # optional topic block first (by rendered position), then bgkit turns in
    # rendered order — all sorted by rendered start position.
    rendered = prep["rendered"]
    events: list[tuple[int, str]] = []
    if prep["topic_block"] is not None and rendered.topic_sentinel_position is not None:
        events.append((rendered.topic_sentinel_position, "topic"))
    turns = prep["prepared_turns"]
    for k, start in enumerate(rendered.bgkit_sentinel_positions):
        t = turns[k]
        if t is None:
            label = "none(zero)"
        elif isinstance(t, dict) and "mode" in t:
            label = f"{t['mode']}:{str(t.get('node_id', ''))[-48:]}"
        else:
            label = (
                f"retrieve(N={int(t['content'].shape[0])},"
                f"pin={int(t['pinned'].sum())})"
            )
        events.append((start, label))
    events.sort(key=lambda e: e[0])

    rows = []
    offs = 0
    emb_i = 0
    a_start = trace.answer_span[0] if trace.answer_span else -1
    for i, seg in enumerate(segments):
        length = seg_len(seg)
        row = {
            "i": i, "start": offs, "len": length,
            "before_answer": bool(offs + length <= a_start) if a_start >= 0 else None,
        }
        if isinstance(seg, TokenSegment):
            row["type"] = "TOK"
            flat = seg.token_ids.reshape(-1)[:24]
            row["preview"] = trainer.tokenizer.decode(
                flat.tolist(),
            )[:90].replace("\n", "\\n")
        else:
            row["type"] = "EMB"
            row["turn"] = events[emb_i][1] if emb_i < len(events) else "?"
            emb_i += 1
        rows.append(row)
        offs += length
    return rows


# ---------------------------------------------------------------------------
# Per-sample analysis
# ---------------------------------------------------------------------------


def analyze_sample(trainer, sample, family: str, do_fix_preview: bool) -> dict:
    res: dict = {"family": family, "question": sample.question[:140]}
    kinds = classify_turns(trainer, sample)
    res["turn_kinds"] = kinds
    res["depth"] = len(kinds)

    trainer._ensure_eval_shared_tree(sample)

    prep = trainer._prepare_sample_for_decode(sample)
    turns = prep["prepared_turns"]

    # ---- Memo-keyset vs drill/retrieve ids (coordinator check #1) ----
    splice_reps = getattr(trainer, "_shared_tree_splice_reps", None) or {}
    ds = sample.dataset_name
    tree = trainer._trees.get(ds)
    memo_rows = []
    for t in sample.trajectory:
        if t.kind != "bgkit":
            continue
        ids = list(t.args.get("ids", []))
        row = {"ids": ids, "is_head": bool(t.args.get("is_head", False))}
        for one in ids:
            row.setdefault("in_tree", []).append(bool(tree is not None and one in tree))
            row.setdefault("in_memo", []).append(one in splice_reps)
            if tree is not None and one not in tree:
                # retrieve id: does it resolve to a real document?
                try:
                    docs = trainer._resolve_article_ids(ds, [one])
                    row.setdefault("resolved_docs", []).append(docs)
                    row.setdefault("doc_tokens", []).extend(
                        int(trainer._token_store.length(ds, d)) for d in docs
                    )
                except Exception as exc:  # noqa: BLE001
                    row.setdefault("resolve_error", []).append(str(exc)[:120])
        memo_rows.append(row)
    res["id_resolution"] = memo_rows
    res["memo_size"] = len(splice_reps)

    # Per-turn drill introspection (leaf turns only).
    leaf_info = []
    for i, t in enumerate(turns):
        if isinstance(t, dict) and "mode" not in t:
            info = leaf_turn_survivor_split(trainer, t)
            info["turn_index"] = i
            leaf_info.append(info)
    res["leaf_turns"] = leaf_info

    survs = trainer._run_l1_batch(turns)
    res["splice_rows_per_turn"] = [int(s.shape[0]) for s in survs]
    res["splice_mean_row_norm_per_turn"] = [
        round(float(s.float().norm(dim=-1).mean()), 3) for s in survs
    ]

    segments, trace = trainer._assemble_sample_segments(prep, survs)
    res["segments"] = dump_segments(trainer, segments, trace, prep)
    span = trace.answer_span
    res["answer_span"] = list(span) if span else None
    if span is None:
        return res

    # ---- answer_span alignment check (coordinator check #4) ----
    concat_ids: list[int] = []
    for seg in segments:
        if isinstance(seg, TokenSegment):
            concat_ids.extend(seg.token_ids.reshape(-1).tolist())
        else:
            concat_ids.extend([0] * seg_len(seg))
    span_text = trainer.tokenizer.decode(
        concat_ids[span[0]:span[1]], skip_special_tokens=True,
    )
    gold = str(sample.gold_answer)
    res["answer_alignment"] = {
        "span_text_head": span_text[:100],
        "gold_head": gold[:100],
        "span_startswith_gold_head": span_text.strip()[:60] == gold.strip()[:60],
        "span_len_tokens": span[1] - span[0],
    }

    def is_leaf(i):
        return turns[i] is not None and not (
            isinstance(turns[i], dict) and "mode" in turns[i]
        )

    def is_special(i):
        return isinstance(turns[i], dict) and "mode" in turns[i]

    ce_a, n_tok = recon_ce(trainer, segments, span)
    sv_b = [torch.zeros_like(s) if is_leaf(i) else s for i, s in enumerate(survs)]
    seg_b, tr_b = trainer._assemble_sample_segments(prep, sv_b)
    ce_b, _ = recon_ce(trainer, seg_b, tr_b.answer_span)
    sv_c = [torch.zeros_like(s) if is_special(i) else s for i, s in enumerate(survs)]
    seg_c, tr_c = trainer._assemble_sample_segments(prep, sv_c)
    ce_c, _ = recon_ce(trainer, seg_c, tr_c.answer_span)
    sv_d = [torch.zeros_like(s) for s in survs]
    seg_d, tr_d = trainer._assemble_sample_segments(prep, sv_d)
    ce_d, _ = recon_ce(trainer, seg_d, tr_d.answer_span)

    res["recon_ce"] = {
        "answer_tokens": n_tok,
        "a_baseline": round(ce_a, 4),
        "b_retrieve_zeroed": round(ce_b, 4),
        "c_tree_zeroed": round(ce_c, 4),
        "d_all_zeroed": round(ce_d, 4),
    }

    oracle_toks = build_oracle_tokens(trainer, sample)
    if oracle_toks is not None:
        seg_e, span_e = insert_before_answer(segments, span, oracle_toks)
        ce_e, _ = recon_ce(trainer, seg_e, span_e)
        res["recon_ce"]["e_oracle_rawdiff"] = round(ce_e, 4)
        res["recon_ce"]["oracle_tokens"] = int(oracle_toks.shape[0])
        # ---- gold vs retrieved-diff overlap (coordinator check #3) ----
        diff_text = trainer.tokenizer.decode(
            oracle_toks.tolist(), skip_special_tokens=True,
        )
        gold_lines = [
            ln.strip() for ln in str(sample.gold_answer).splitlines() if ln.strip()
        ]
        if gold_lines:
            hit = sum(1 for ln in gold_lines if ln in diff_text)
            res["gold_diff_line_overlap"] = round(hit / len(gold_lines), 3)
        # (g) oracle text with ALL reps zeroed — is text alone sufficient?
        seg_g, span_g = insert_before_answer(seg_d, tr_d.answer_span, oracle_toks)
        ce_g, _ = recon_ce(trainer, seg_g, span_g)
        res["recon_ce"]["g_oracle_no_reps"] = round(ce_g, 4)

    # (h) drill L1 at ratio 1.0 — retrieval leaves keep EVERY row (pinned +
    # all L0 content). Ceiling of what the encoder-space delivery channel can
    # give the CURRENT decoder without retraining.
    if any(is_leaf(i) for i in range(len(turns))):
        survs_h_leaf = trainer._run_l1_batch(turns, target_ratio=1.0)
        sv_h = [
            survs_h_leaf[i] if is_leaf(i) else s for i, s in enumerate(survs)
        ]
        seg_h, tr_h = trainer._assemble_sample_segments(prep, sv_h)
        ce_h, _ = recon_ce(trainer, seg_h, tr_h.answer_span)
        res["recon_ce"]["h_drill_ratio1"] = round(ce_h, 4)
        res["drill_ratio1_rows"] = [
            int(survs_h_leaf[i].shape[0]) for i in range(len(turns)) if is_leaf(i)
        ]

    if do_fix_preview:
        import bgkit.models.level_compressor as lc_mod

        prev = lc_mod.exact_ratio_topk_select  # probe (restored after)
        try:
            _set_topk(fixed_topk)
            trainer._eval_tree_cache = {}
            trainer._eval_shared_tree_key = None
            trainer._ensure_eval_shared_tree(sample)
            prep_f = trainer._prepare_sample_for_decode(sample)
            survs_f = trainer._run_l1_batch(prep_f["prepared_turns"])
            seg_f, tr_f = trainer._assemble_sample_segments(prep_f, survs_f)
            ce_f, _ = recon_ce(trainer, seg_f, tr_f.answer_span)
            res["recon_ce"]["f_quota_fix"] = round(ce_f, 4)
            res["splice_rows_per_turn_fixed"] = [int(s.shape[0]) for s in survs_f]
            fixed_leaf = []
            for i, t in enumerate(prep_f["prepared_turns"]):
                if isinstance(t, dict) and "mode" not in t:
                    info = leaf_turn_survivor_split(trainer, t)
                    info["turn_index"] = i
                    fixed_leaf.append(info)
            res["leaf_turns_fixed"] = fixed_leaf
        finally:
            _set_topk(prev)
            # Drop the fixed tree so later samples re-encode unfixed.
            trainer._eval_tree_cache = {}
            trainer._eval_shared_tree_key = None
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
    logger.info("loading_checkpoint", path=ckpt)
    trainer.load_checkpoint(Path(ckpt))
    logger.info("checkpoint_loaded", step=trainer.global_step)

    trainer.model.eval()
    trainer._eval_tree_cache = {}
    trainer._eval_shared_tree_key = None

    families = os.environ.get("DIAG_FAMILIES", "qwen35,falcon_h1").split(",")
    n_full = int(os.environ.get("DIAG_N_FULL", "6"))
    n_nodrill = int(os.environ.get("DIAG_N_NODRILL", "2"))
    scan_cap = int(os.environ.get("DIAG_SCAN_CAP", "4000"))
    do_fix = os.environ.get("DIAG_FIX_PREVIEW", "1") == "1"

    # ---- pick samples: full-drill samples across (multiple) repo roots ----
    picked_full: list = []
    picked_nodrill: list = []
    full_roots: set[str] = set()
    nodrill_roots: set[str] = set()
    scanned = 0
    for batch in trainer.eval_dataloader:
        for s in batch:
            scanned += 1
            if s.dataset_name != "git_commit_repro":
                continue
            kinds = classify_turns(trainer, s)
            root = trainer._repo_group_key(s)
            n_retrieve = sum(1 for k in kinds if k == "retrieve")
            if n_retrieve >= 1 and len(picked_full) < n_full:
                # at most 2 per root for tree variety
                if sum(1 for p in picked_full
                       if trainer._repo_group_key(p) == root) < 2:
                    picked_full.append(s)
                    full_roots.add(root)
            elif kinds == ["head"] and len(picked_nodrill) < n_nodrill:
                if root not in nodrill_roots:
                    picked_nodrill.append(s)
                    nodrill_roots.add(root)
        if scanned >= scan_cap or (
            len(picked_full) >= n_full and len(picked_nodrill) >= n_nodrill
        ):
            break
    if not picked_full:
        raise RuntimeError(
            f"No full-drill git_commit_repro eval sample in first {scanned}"
        )
    logger.info(
        "samples_picked", roots=sorted(full_roots), n_full=len(picked_full),
        n_nodrill=len(picked_nodrill), scanned=scanned,
    )

    report: dict = {"checkpoint": ckpt, "roots": sorted(full_roots), "samples": []}
    with torch.no_grad():
        for family in [f.strip() for f in families if f.strip()]:
            if trainer._round_robin:
                trainer._set_active_decoder(family)
            trainer._eval_tree_cache = {}
            trainer._eval_shared_tree_key = None

            TOPK_STATS.clear()
            _set_topk(make_probe(f"tree+drill/{family}"))
            try:
                for j, s in enumerate(picked_full + picked_nodrill):
                    r = analyze_sample(
                        trainer, s, family,
                        do_fix_preview=do_fix and j == 0,
                    )
                    report["samples"].append(r)
                    print("\n" + "=" * 100)
                    print(json.dumps(r, indent=2, default=str))
            finally:
                _set_topk(TRUE_ORIG_TOPK)

            tot = list(TOPK_STATS)
            if tot:
                agg = {
                    "family": family,
                    "n_topk_calls": len(tot),
                    "sum_valid": sum(x["n_valid"] for x in tot),
                    "sum_pinned": sum(x["n_pinned"] for x in tot),
                    "sum_selected": sum(x["n_selected"] for x in tot),
                    "sum_content_selected": sum(
                        x["n_content_selected"] for x in tot
                    ),
                    "calls_with_zero_content": sum(
                        1 for x in tot if x["n_content_selected"] == 0
                    ),
                    "ratios": sorted({round(x["ratio"], 4) for x in tot}),
                }
                print("\nTOPK PROBE AGG:", json.dumps(agg, indent=2))
                report.setdefault("topk_agg", []).append(agg)

    # ---- aggregate CE table per family ----
    for family in [f.strip() for f in families if f.strip()]:
        rows = [
            r for r in report["samples"]
            if r["family"] == family and "recon_ce" in r
        ]
        keys = sorted({k for r in rows for k in r["recon_ce"]})
        agg = {}
        for k in keys:
            vals = [
                (r["recon_ce"][k], r["recon_ce"]["answer_tokens"])
                for r in rows if k in r["recon_ce"]
            ]
            if vals and k != "answer_tokens" and not k.startswith("oracle"):
                tot_tok = sum(v[1] for v in vals)
                agg[k] = round(sum(v[0] * v[1] for v in vals) / max(tot_tok, 1), 4)
        print(f"\nAGG token-weighted recon CE [{family}]:", json.dumps(agg, indent=2))
        report.setdefault("agg_recon_ce", {})[family] = agg

    out_path = Path("/workspace/checkpoints/diag_git_repro_retrieve_splice.json")
    try:
        out_path.write_text(json.dumps(report, indent=2, default=str))
        logger.info("report_written", path=str(out_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("report_write_failed", err=str(exc))


if __name__ == "__main__":
    main()
