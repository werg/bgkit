#!/usr/bin/env python
"""DECISIVE A-vs-B split for the git-repro recon plateau (recon_gap ~= 0.25).

ONE question: does the reconstruction wall come from
  (A) the reps NOT conveying the retrieved diff (compression too lossy), or
  (B) the reps DO convey the diff but the decoder can't APPLY it to rebuild
      the file (the file-reconstruction reasoning is the wall)?

For a handful of real git_commit_repro eval samples and BOTH decoder families,
using the SAME leaf/trajectory reps (encoded at the run's leaf retention
l0=0.63/l1=0.63 via _run_l1_batch None-default), same chat/tool template, and
ONLY the decode target differing:

  Test A  reps -> DIFF : target = the retrieved diff text (verbatim). Can the
          decoder recover the diff that lives in its reps?
  Test B  reps -> FILE : target = the gold file blob (the real task; the ~0.25
          we see).

recon_gap = CE(all reps zeroed) - CE(reps present)   [matches eval
_run_ablation_gap_probe: ABLATION_ZEROED zeros every spliced survivor; recon
span = the answer span].

For BOTH tests we also report the TEXT-ORACLE ceiling: reps zeroed, the
diff-as-TEXT inserted in the slot right before the answer:
  Test A oracle = CE(diff-text -> diff)  (echo/copy sanity: must be ~0; a
      broken template -> pathological CE, esp. for falcon welded to its tool
      format -> the whole run is worthless)
  Test B oracle = CE(diff-text -> file)  (the ~1 nat achievable ceiling)

Interpretation:
  A-gap HIGH, B-gap LOW (~0.25) -> reps convey the diff; decoder can't apply
     it -> the WALL is the file-reconstruction REASONING (simplify the task).
  A-gap ALSO LOW (~0.25)        -> reps don't convey the diff -> the WALL is
     COMPRESSION (less-lossy leaf / richer channel).

Reuses the trainer's own encode+splice+decode (guarantees the CORRECT
bgkit_tool_template + falcon chat template). Only the answer TokenSegment is
swapped between A and B; the reps + every other segment are byte-identical.

Run (container, trainer STOPPED, GPU free):
  docker compose --env-file .env -f docker/docker-compose.yaml run --rm --no-deps \
    train-phase2-kb-git-repro-fullbackprop \
    python /workspace/scripts/diag_git_repro_A_vs_B.py \
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
    "phase2_kb_step286_20260801_214749_232269_run-phase2_kb_git_repro_reanchor51945"
)

# Cap the diff target/slot tokens (decoder-tokenized). Bounds decode length +
# memory; recon_gap is a PER-TOKEN mean so a cap different from the file
# target's max_decode_tokens (4096) is apples-to-apples.
DIFF_CAP = int(os.environ.get("DIAG_DIFF_CAP", "3072"))


# ---------------------------------------------------------------------------
# Helpers (mirror diag_git_repro_retrieve_splice.py / _query_conditioning.py)
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


def ordered_diffs(trainer, sample) -> list[str]:
    """Retrieved diff texts (oldest -> newest), retrieve turns only."""
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
        for d in trainer._resolve_article_ids(ds, ids):
            texts.append(
                trainer.encoder_tokenizer.decode(
                    trainer._token_store.get(ds, d).tolist(),
                ),
            )
    return texts


def recon_ce(trainer, segments, span) -> tuple[float, int]:
    out = trainer.decoder.forward_interleaved_with_loss(
        segments, return_hidden_states=True,
    )
    s, c = KRKBTrainer._span_ce_sum_count(out, [span])
    del out
    return (s / max(c, 1), c)


def swap_answer_target(segments, span, new_ids, device):
    """Return (new_segments, new_span): identical to ``segments`` except the
    TokenSegment that fully contains ``span`` has that span's tokens replaced
    by ``new_ids`` (loss-masked True), the surrounding template kept
    (loss-masked False). Only the answer content changes; reps + every other
    segment are untouched. Returns (None, None) if no single segment contains
    the span (should not happen — the answer is the final TokenSegment)."""
    offs = 0
    out: list = []
    new_span = None
    for seg in segments:
        length = seg_len(seg)
        if (
            new_span is None
            and isinstance(seg, TokenSegment)
            and offs <= span[0]
            and span[1] <= offs + length
        ):
            local_s = span[0] - offs
            local_e = span[1] - offs
            tok = seg.token_ids.reshape(-1)
            pre = tok[:local_s]
            post = tok[local_e:]
            if pre.shape[0] > 0:
                out.append(TokenSegment(
                    token_ids=pre.unsqueeze(0),
                    loss_mask=torch.zeros(
                        (1, pre.shape[0]), dtype=torch.bool, device=device,
                    ),
                ))
            ans_start = offs + int(pre.shape[0])
            out.append(TokenSegment(
                token_ids=new_ids.unsqueeze(0),
                loss_mask=torch.ones(
                    (1, new_ids.shape[0]), dtype=torch.bool, device=device,
                ),
            ))
            new_span = (ans_start, ans_start + int(new_ids.shape[0]))
            if post.shape[0] > 0:
                out.append(TokenSegment(
                    token_ids=post.unsqueeze(0),
                    loss_mask=torch.zeros(
                        (1, post.shape[0]), dtype=torch.bool, device=device,
                    ),
                ))
        else:
            out.append(seg)
        offs += length
    return (out, new_span) if new_span is not None else (None, None)


def insert_before_answer(segments, span, tok_ids):
    """Insert a loss-masked TokenSegment right before the segment containing
    ``span[0]`` (the text-oracle slot); shift the span by +len(tok_ids)."""
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


# ---------------------------------------------------------------------------
# Per-sample analysis
# ---------------------------------------------------------------------------


def analyze(trainer, sample, family: str) -> dict:
    ds = sample.dataset_name
    dev = trainer.device
    res: dict = {
        "family": family,
        "question": sample.question[:120],
        "turn_kinds": classify_turns(trainer, sample),
    }

    trainer._ensure_eval_shared_tree(sample)
    prep = trainer._prepare_sample_for_decode(sample)
    turns = prep["prepared_turns"]

    # ---- reps (IDENTICAL for A and B), at the run's leaf retention ----
    survs = trainer._run_l1_batch(turns)
    survs_z = [torch.zeros_like(s) for s in survs]
    seg_present, tr = trainer._assemble_sample_segments(prep, survs)
    seg_zeroed, tr_z = trainer._assemble_sample_segments(prep, survs_z)
    file_span = tr.answer_span
    if file_span is None:
        res["skip"] = "no answer span"
        return res
    res["splice_rows_per_turn"] = [int(s.shape[0]) for s in survs]

    # ---- diff target (decoder-tokenized retrieved diff text, oldest first) --
    diffs = ordered_diffs(trainer, sample)
    diff_text = "\n\n".join(diffs)
    diff_ids = torch.tensor(
        trainer.tokenizer.encode(diff_text, add_special_tokens=False)[:DIFF_CAP],
        dtype=torch.long, device=dev,
    )
    res["n_retrieve_diffs"] = len(diffs)
    res["diff_target_tokens"] = int(diff_ids.shape[0])
    res["file_target_tokens"] = int(file_span[1] - file_span[0])
    if diff_ids.shape[0] == 0:
        res["skip"] = "empty diff target"
        return res

    # =====================  TEST B : reps -> FILE  =========================
    ce_b_pres, ntok_b = recon_ce(trainer, seg_present, file_span)
    ce_b_zero, _ = recon_ce(trainer, seg_zeroed, file_span)
    # text-oracle: reps zeroed + diff-as-text in the slot -> file
    seg_bo, span_bo = insert_before_answer(seg_zeroed, file_span, diff_ids)
    ce_b_orac, _ = recon_ce(trainer, seg_bo, span_bo)

    # =====================  TEST A : reps -> DIFF  =========================
    segA_pres, spanA = swap_answer_target(seg_present, file_span, diff_ids, dev)
    segA_zero, spanA_z = swap_answer_target(seg_zeroed, file_span, diff_ids, dev)
    if spanA is None or spanA_z is None:
        res["skip"] = "answer span not contained in one TokenSegment"
        return res
    ce_a_pres, ntok_a = recon_ce(trainer, segA_pres, spanA)
    ce_a_zero, _ = recon_ce(trainer, segA_zero, spanA_z)
    # text-oracle: reps zeroed + diff-as-text in the slot -> diff (== COPY;
    # must be ~0). Slot text is the SAME tokens as the target -> echo sanity.
    seg_ao, span_ao = insert_before_answer(segA_zero, spanA_z, diff_ids)
    ce_a_orac, _ = recon_ce(trainer, seg_ao, span_ao)

    res["A_reps_to_diff"] = {
        "ce_present": round(ce_a_pres, 4),
        "ce_zeroed": round(ce_a_zero, 4),
        "recon_gap": round(ce_a_zero - ce_a_pres, 4),
        "text_oracle_ce": round(ce_a_orac, 4),
        "target_tokens": ntok_a,
    }
    res["B_reps_to_file"] = {
        "ce_present": round(ce_b_pres, 4),
        "ce_zeroed": round(ce_b_zero, 4),
        "recon_gap": round(ce_b_zero - ce_b_pres, 4),
        "text_oracle_ce": round(ce_b_orac, 4),
        "target_tokens": ntok_b,
    }
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    assert cfg.training.phase == "phase2_kb", cfg.training.phase
    trainer = KRKBTrainer(cfg)
    trainer.setup()
    ckpt = os.environ.get("DIAG_CKPT", DEFAULT_CKPT)
    logger.info("loading_checkpoint", path=ckpt)
    trainer.load_checkpoint(Path(ckpt))
    logger.info("checkpoint_loaded", step=trainer.global_step)
    trainer.model.eval()
    trainer._eval_tree_cache = {}
    trainer._eval_shared_tree_key = None

    families = [
        f.strip()
        for f in os.environ.get("DIAG_FAMILIES", "qwen35,falcon_h1").split(",")
        if f.strip()
    ]
    n_full = int(os.environ.get("DIAG_N_FULL", "6"))
    scan_cap = int(os.environ.get("DIAG_SCAN_CAP", "6000"))

    # ---- pick full-drill samples (>=1 retrieve), <=2 per repo root ----
    picked: list = []
    scanned = 0
    for batch in trainer.eval_dataloader:
        for s in batch:
            scanned += 1
            if s.dataset_name != "git_commit_repro":
                continue
            kinds = classify_turns(trainer, s)
            if sum(1 for k in kinds if k == "retrieve") >= 1:
                root = trainer._repo_group_key(s)
                if sum(1 for p in picked
                       if trainer._repo_group_key(p) == root) < 2:
                    picked.append(s)
        if scanned >= scan_cap or len(picked) >= n_full:
            break
    if not picked:
        raise RuntimeError(f"no full-drill git_commit_repro sample in {scanned}")
    logger.info("samples_picked", n=len(picked), scanned=scanned)

    report: dict = {"checkpoint": ckpt, "step": int(trainer.global_step),
                    "diff_cap": DIFF_CAP, "samples": []}
    with torch.no_grad():
        for family in families:
            if trainer._round_robin:
                trainer._set_active_decoder(family)
            trainer._eval_tree_cache = {}
            trainer._eval_shared_tree_key = None
            for s in picked:
                r = analyze(trainer, s, family)
                report["samples"].append(r)
                print("\n" + "=" * 100)
                print(json.dumps(r, indent=2, default=str))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ---- token-weighted aggregate: rows = {A,B} x family ----
    print("\n" + "#" * 100)
    print("AGGREGATE  (token-weighted CE; recon_gap = ce_zeroed - ce_present)")
    agg: dict = {}
    for r in report["samples"]:
        for test in ("A_reps_to_diff", "B_reps_to_file"):
            if test not in r:
                continue
            e = r[test]
            w = e["target_tokens"]
            key = (r["family"], test)
            a = agg.setdefault(key, {
                "w": 0, "pres": 0.0, "zero": 0.0, "orac": 0.0, "n": 0,
                "gaps": [], "oracs": [],
            })
            a["w"] += w
            a["pres"] += e["ce_present"] * w
            a["zero"] += e["ce_zeroed"] * w
            a["orac"] += e["text_oracle_ce"] * w
            a["gaps"].append(e["recon_gap"])
            a["oracs"].append(e["text_oracle_ce"])
            a["n"] += 1
    header = (
        f"{'family':10s} {'test':16s} {'CE_present':>11s} {'CE_zeroed':>10s} "
        f"{'recon_gap':>10s} {'text_oracle':>12s} {'n':>3s}"
    )
    print(header)
    print("-" * len(header))
    for (fam, test), a in sorted(agg.items()):
        w = max(a["w"], 1)
        row = {
            "family": fam,
            "test": test,
            "ce_present": round(a["pres"] / w, 4),
            "ce_zeroed": round(a["zero"] / w, 4),
            "recon_gap": round((a["zero"] - a["pres"]) / w, 4),
            "text_oracle_ce": round(a["orac"] / w, 4),
            "n_samples": a["n"],
        }
        report.setdefault("agg", {})[f"{fam}/{test}"] = row
        print(
            f"{fam:10s} {test:16s} {row['ce_present']:>11.4f} "
            f"{row['ce_zeroed']:>10.4f} {row['recon_gap']:>10.4f} "
            f"{row['text_oracle_ce']:>12.4f} {a['n']:>3d}"
        )

    out = Path("/workspace/checkpoints/diag_git_repro_A_vs_B.json")
    try:
        out.write_text(json.dumps(report, indent=2, default=str))
        logger.info("report_written", path=str(out))
    except Exception as exc:  # noqa: BLE001
        logger.warning("report_write_failed", err=str(exc))


if __name__ == "__main__":
    main()
