#!/usr/bin/env python
"""Splice-POSITION test: same lossless leaf survivors, decode-PREFIX framing.

Runs ONLY if the decode ladder shows (3) emb-oracle-full >> (2) text-oracle:
the rep pipeline fails to convey the diff even lossless, mid-trajectory. This
test moves the SAME ratio-1.0 leaf survivors to a Phase-1-style decode PREFIX
([survivors | minimal prompt | answer], no nav turns, no tool framing) to
separate:

  - prefix-emb CE ~ text CE  -> channel FINE; mid-trajectory POSITION/framing
    is the problem (answer can't exploit reps across the nav turns).
  - prefix-emb CE still ~1.7 -> the embedding channel genuinely cannot convey
    the diff on this path (encode/project/decode-read bug).

Controls, all in the SAME stripped prefix framing (per family, step-9164):
  P0  no context      [prompt | answer]
  P1  text prefix     [diff text | prompt | answer]
  P2  emb prefix      [ratio-1.0 leaf survivors (L0->L1->proj) | prompt | answer]
  P3  emb prefix L0-only (target_ratio_l1=None, all rows -> projection)

Run:
  docker compose --env-file .env -f docker/docker-compose.yaml run --rm -T \
    train-phase2-kb-git-repro-fullbackprop \
    python scripts/diag_git_repro_prefix_splice.py \
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


def retrieve_docs(trainer, sample) -> list[str]:
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


@contextlib.contextmanager
def forced_l0_ratio(trainer, ratio: float):
    orig = trainer._sample_l0_retention_for
    trainer._sample_l0_retention_for = lambda ds: float(ratio)
    try:
        yield
    finally:
        trainer._sample_l0_retention_for = orig


def lossless_leaf_survivors(trainer, sample) -> torch.Tensor:
    """Ratio-1.0 leaf survivors through the REAL path (pin + L0(1.0) -> L1(1.0)
    -> projection), concatenated over all retrieve turns."""
    ds = sample.dataset_name
    task_query = head_task_query(sample)
    parts = []
    with forced_l0_ratio(trainer, 1.0):
        for t in sample.trajectory:
            if t.kind != "bgkit" or bool(t.args.get("is_head", False)):
                continue
            ids = list(t.args.get("ids", []))
            q = str(t.args.get("query", ""))
            tree = trainer._trees.get(ds)
            if q == "" and len(ids) == 1 and tree is not None and str(ids[0]) in tree:
                continue
            turn = trainer._prepare_l1_turn(
                ds, ids, q or task_query, l0_selection_mode="exact_topk",
            )
            if turn is None:
                continue
            surv = trainer._run_l1_batch([turn], target_ratio=1.0)[0]
            parts.append(surv)
    if not parts:
        return torch.zeros(
            (1, trainer.encoder.active_projection_output_dim),
            device=trainer.device, dtype=torch.bfloat16,
        )
    return torch.cat(parts, dim=0)


def l0_only_survivors(trainer, sample) -> torch.Tensor:
    """All doc rows -> projection, L1 skipped (encoder.forward, ratios None)."""
    ds = sample.dataset_name
    q_emb = trainer._head_query_emb(head_task_query(sample))
    dev = trainer.device
    embed_tokens = trainer.encoder.l0.backbone.get_input_embeddings()
    parts = []
    for did in retrieve_docs(trainer, sample):
        toks = trainer._token_store.get(ds, did).to(dev)
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
        parts.append(out.survivor_embeddings)
    if not parts:
        return torch.zeros(
            (1, trainer.encoder.active_projection_output_dim),
            device=trainer.device, dtype=torch.bfloat16,
        )
    return torch.cat(parts, dim=0)


def prefix_ce(trainer, sample, context) -> tuple[float, int]:
    """CE over the gold answer in the stripped prefix framing.

    context: None | torch.Tensor(text token ids) | EmbeddingSegment tensor.
    Layout: [context?][prompt tokens (no loss)][gold tokens (loss)].
    """
    dev = trainer.device
    prompt = (
        "\nQuestion: " + str(sample.question)
        + "\nAnswer with the full file content:\n"
    )
    p_ids = trainer.tokenizer.encode(prompt, add_special_tokens=False)
    g_ids = trainer.tokenizer.encode(
        str(sample.gold_answer), add_special_tokens=False,
    )[:4096]
    segments = []
    ctx_len = 0
    if isinstance(context, torch.Tensor) and context.dtype == torch.long:
        segments.append(TokenSegment(
            token_ids=context.unsqueeze(0),
            loss_mask=torch.zeros((1, context.shape[0]), dtype=torch.bool, device=dev),
        ))
        ctx_len = int(context.shape[0])
    elif isinstance(context, torch.Tensor):
        segments.append(EmbeddingSegment(embeddings=context.unsqueeze(0)))
        ctx_len = int(context.shape[0])
    p_t = torch.tensor(p_ids, dtype=torch.long, device=dev)
    g_t = torch.tensor(g_ids, dtype=torch.long, device=dev)
    segments.append(TokenSegment(
        token_ids=p_t.unsqueeze(0),
        loss_mask=torch.zeros((1, len(p_ids)), dtype=torch.bool, device=dev),
    ))
    segments.append(TokenSegment(
        token_ids=g_t.unsqueeze(0),
        loss_mask=torch.ones((1, len(g_ids)), dtype=torch.bool, device=dev),
    ))
    span = (ctx_len + len(p_ids), ctx_len + len(p_ids) + len(g_ids))
    out = trainer.decoder.forward_interleaved_with_loss(
        segments, return_hidden_states=True,
    )
    s, c = KRKBTrainer._span_ce_sum_count(out, [span])
    del out
    return s / max(c, 1), c


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    assert cfg.training.phase == "phase2_kb"
    trainer = KRKBTrainer(cfg)
    trainer.setup()
    # DIAG_SKIP_LOAD=1: evaluate the summarization-base weights that setup()
    # already loaded (phase1_checkpoint) — the pre-git-repro ancestor whose
    # decoder provably reconstructed from projected reps. Control for "did
    # git-repro training DESTROY the splice-reading channel?"
    if os.environ.get("DIAG_SKIP_LOAD", "0") != "1":
        trainer.load_checkpoint(Path(CKPT))
        logger.info("checkpoint_loaded", step=trainer.global_step)
    else:
        logger.info("using_summarization_base_weights")
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
            for s in picked:
                ds = s.dataset_name
                # contexts
                texts = [
                    trainer.encoder_tokenizer.decode(
                        trainer._token_store.get(ds, d).tolist(),
                    )
                    for d in retrieve_docs(trainer, s)
                ]
                text_ids = trainer.tokenizer.encode(
                    "\n\n".join(texts), add_special_tokens=False,
                )[:3072]
                text_t = torch.tensor(
                    text_ids, dtype=torch.long, device=trainer.device,
                )
                emb_full = lossless_leaf_survivors(trainer, s)
                emb_l0 = l0_only_survivors(trainer, s)

                ce0, n_tok = prefix_ce(trainer, s, None)
                ce1, _ = prefix_ce(trainer, s, text_t)
                ce2, _ = prefix_ce(trainer, s, emb_full)
                ce3, _ = prefix_ce(trainer, s, emb_l0)
                row = {
                    "family": family,
                    "question": s.question[:100],
                    "answer_tokens": n_tok,
                    "P0_no_context": round(ce0, 4),
                    "P1_text_prefix": round(ce1, 4),
                    "P2_emb_prefix_l0l1_proj": round(ce2, 4),
                    "P3_emb_prefix_l0_only_proj": round(ce3, 4),
                    "emb_rows_full": int(emb_full.shape[0]),
                    "emb_rows_l0only": int(emb_l0.shape[0]),
                    "text_tokens": len(text_ids),
                }
                report["rows"].append(row)
                print(json.dumps(row, indent=2))

    # aggregate
    for family in [f.strip() for f in families if f.strip()]:
        rows = [r for r in report["rows"] if r["family"] == family]
        if not rows:
            continue
        agg = {}
        for k in ("P0_no_context", "P1_text_prefix",
                  "P2_emb_prefix_l0l1_proj", "P3_emb_prefix_l0_only_proj"):
            tot = sum(r["answer_tokens"] for r in rows)
            agg[k] = round(
                sum(r[k] * r["answer_tokens"] for r in rows) / max(tot, 1), 4,
            )
        print(f"\nPREFIX AGG [{family}]:", json.dumps(agg))
        report.setdefault("agg", {})[family] = agg

    out = Path("/workspace/checkpoints/diag_git_repro_prefix_splice.json")
    try:
        out.write_text(json.dumps(report, indent=2, default=str))
        logger.info("report_written", path=str(out))
    except Exception as exc:  # noqa: BLE001
        logger.warning("report_write_failed", err=str(exc))


if __name__ == "__main__":
    main()
