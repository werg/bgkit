#!/usr/bin/env python
"""REAL-PATH control: is the dead rep channel a git-repro wiring regression?

Two experiments, per decoder family, on base-51945 (the weights setup() loads
— the known-good summarization ancestor) and the step-9164 git-repro weights:

W  WIRING A/B — identical [prefix | reps | suffix] + identical loss mask
   through BOTH decoder APIs:
     A: decoder.forward_with_single_splice   (the Phase-1/summarization path)
     B: decoder.forward_interleaved_with_loss (the git-repro splice path)
   CE must match to numerical noise if the interleaved machinery is sound.

C  REAL-PATH reconstruction — the ACTUAL Phase-1 verbatim-repro regime:
   - chat inputs via summarization_round_robin._build_chat_inputs_for_sample
     (tokenize_with_sentinel + TOOL_CONFIGS["file_read_repro"] + the
     compression.py probe variant — the literal Phase-1 template),
   - encoder.forward with the ChatML compression prompt (as the summarization
     _encode_batch does), ratios (0.30,None) / (0.30,0.30) / (1.0,None),
   - decoder.forward_with_single_splice with the suffix-target loss mask.
   Controls per point: reps present / reps zeroed / no reps (K=0).
   Docs: git diffs (task content) + gold blobs (code files — the literal
   Phase-1 file_read_repro distribution, harness validation).

Run:
  docker compose --env-file .env -f docker/docker-compose.yaml run --rm -T \
    train-phase2-kb-git-repro-fullbackprop \
    python scripts/diag_git_repro_realpath_control.py \
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

from bgkit.data.chat_template import TOOL_CONFIGS
from bgkit.models.decoder import EmbeddingSegment, TokenSegment
from bgkit.training.phase1.summarization_round_robin import (
    _build_chat_inputs_for_sample,
)
from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
from bgkit.utils.logging import setup_logging
from bgkit.utils.packing import position_ids_from_cu

logger = structlog.get_logger()

CKPT_NEW = os.environ.get(
    "DIAG_CKPT",
    "/workspace/checkpoints_fast/"
    "phase2_kb_step9164_20260730_091047_791250_run-phase2_kb_git_repro_fullbackprop",
)

# The literal Phase-1 verbatim-repro variant (compression.py probe / the
# document_verbatim_repro bank shape).
VARIANT = {
    "system_prompt": (
        "You are an AI coding assistant with access to the "
        "bgkit_read_file tool for reading file contents."
    ),
    "user_prompt": "Read the file `{file_path}`",
    "compression_prompt": "Return the file contents verbatim",
    "response_prefix": "Here are the contents of `{file_path}`:",
}

RATIOS = ((0.30, None), (0.30, 0.30), (1.0, None))
DOC_CAP = 2048  # encoder-token cap per control doc


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


def pick_docs(trainer, n_diffs: int = 3, n_blobs: int = 2, scan_cap: int = 3000):
    """(label, text) docs: retrieve diffs + gold blobs from full-drill samples."""
    docs: list[tuple[str, str]] = []
    seen_roots: set[str] = set()
    scanned = 0
    for batch in trainer.eval_dataloader:
        for s in batch:
            scanned += 1
            if s.dataset_name != "git_commit_repro":
                continue
            kinds = classify_turns(trainer, s)
            if "retrieve" not in kinds:
                continue
            root = trainer._repo_group_key(s)
            if root in seen_roots:
                continue
            seen_roots.add(root)
            ds = s.dataset_name
            tree = trainer._trees.get(ds)
            # last retrieve doc = the target-commit diff
            for t in reversed(s.trajectory):
                if t.kind != "bgkit" or bool(t.args.get("is_head", False)):
                    continue
                ids = list(t.args.get("ids", []))
                q = str(t.args.get("query", ""))
                if q == "" and len(ids) == 1 and tree is not None and ids[0] in tree:
                    continue
                dids = trainer._resolve_article_ids(ds, ids)
                if dids and sum(1 for d in docs if d[0].startswith("diff")) < n_diffs:
                    text = trainer.encoder_tokenizer.decode(
                        trainer._token_store.get(ds, dids[0]).tolist()[:DOC_CAP],
                    )
                    docs.append((f"diff:{dids[0][-40:]}", text))
                break
            if sum(1 for d in docs if d[0].startswith("blob")) < n_blobs:
                gold = str(s.gold_answer)
                if 200 < len(gold) < 6000:
                    docs.append((f"blob:{s.question[:40]}", gold))
        if scanned >= scan_cap or (
            sum(1 for d in docs if d[0].startswith("diff")) >= n_diffs
            and sum(1 for d in docs if d[0].startswith("blob")) >= n_blobs
        ):
            break
    return docs


def encode_doc(trainer, text: str, comp_prompt_ids: torch.Tensor, r0, r1):
    """encoder.forward exactly like summarization _encode_batch (1 group, 1 doc)."""
    dev = trainer.device
    enc_ids = torch.tensor(
        trainer.encoder_tokenizer.encode(text, add_special_tokens=False)[:DOC_CAP],
        dtype=torch.long, device=dev,
    )
    embed = trainer.encoder.l0.backbone.get_input_embeddings()
    cu = torch.tensor([0, int(enc_ids.shape[0])], dtype=torch.int32, device=dev)
    pos = position_ids_from_cu(cu, int(enc_ids.shape[0]))
    cp = comp_prompt_ids.to(device=dev, dtype=torch.long)
    pcu = torch.tensor([0, int(cp.shape[0])], dtype=torch.int32, device=dev)
    ppos = position_ids_from_cu(pcu, int(cp.shape[0]))
    group_cu = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    enc_out = trainer.encoder(
        content_embeddings=embed(enc_ids),
        content_cu_seqlens=cu,
        content_position_ids=pos,
        prompt_embeddings=embed(cp),
        prompt_cu_seqlens=pcu,
        prompt_position_ids=ppos,
        target_ratio_l0=r0,
        target_ratio_l1=r1,
        content_group_cu_seqlens=group_cu if r1 is not None else None,
        prompt_embeddings_l1=embed(cp) if r1 is not None else None,
        prompt_cu_seqlens_l1=pcu if r1 is not None else None,
        prompt_position_ids_l1=ppos if r1 is not None else None,
    )
    return enc_out.survivor_embeddings  # projected, decoder-ready


def splice_ce(trainer, survivors, prefix, suffix, suffix_mask) -> float:
    dev = trainer.device
    k = int(survivors.shape[0])
    cu = torch.tensor([0, k], dtype=torch.int32, device=dev)
    lm = torch.cat([
        torch.zeros(int(prefix.shape[0]), dtype=torch.bool, device=dev),
        torch.zeros(k, dtype=torch.bool, device=dev),
        suffix_mask.to(dev),
    ])
    out = trainer.decoder.forward_with_single_splice(
        survivor_embeddings=survivors,
        survivor_cu_seqlens=cu,
        prefix_ids=[prefix],
        suffix_ids=[suffix],
        loss_mask=lm,
    )
    loss = out.loss if hasattr(out, "loss") else out
    return float(loss.item())


def interleaved_ce(trainer, survivors, prefix, suffix, suffix_mask) -> float:
    dev = trainer.device
    segs = [
        TokenSegment(
            token_ids=prefix.unsqueeze(0),
            loss_mask=torch.zeros(
                (1, int(prefix.shape[0])), dtype=torch.bool, device=dev,
            ),
        ),
        EmbeddingSegment(embeddings=survivors.unsqueeze(0)),
        TokenSegment(
            token_ids=suffix.unsqueeze(0),
            loss_mask=suffix_mask.to(dev).unsqueeze(0),
        ),
    ]
    loss = trainer.decoder.forward_interleaved_with_loss(segs)
    return float(loss.item())


def run_pass(trainer, docs, tag, families, report):
    cfg = TOOL_CONFIGS["file_read_repro"]
    for family in families:
        trainer._set_active_decoder(family)
        for label, text in docs:
            tgt = torch.tensor(
                trainer.tokenizer.encode(text, add_special_tokens=False)[:DOC_CAP],
                dtype=torch.long,
            )
            prefix, suffix, sm, comp = _build_chat_inputs_for_sample(
                trainer.tokenizer, VARIANT, cfg,
                group_id="src/reconstruct/target.txt",
                target_ids=tgt.numpy(),
                device=trainer.device,
                encoder_tokenizer=trainer.encoder_tokenizer,
            )
            row = {
                "ckpt": tag, "family": family, "doc": label,
                "target_tokens": int(tgt.shape[0]),
                "ratios": {},
            }
            for r0, r1 in RATIOS:
                surv = encode_doc(trainer, text, comp, r0, r1)
                k = int(surv.shape[0])
                ce_reps = splice_ce(trainer, surv, prefix, suffix, sm)
                ce_zero = splice_ce(
                    trainer, torch.zeros_like(surv), prefix, suffix, sm,
                )
                none_surv = torch.zeros(
                    (0, surv.shape[-1]), dtype=surv.dtype, device=surv.device,
                )
                ce_none = splice_ce(trainer, none_surv, prefix, suffix, sm)
                entry = {
                    "K": k,
                    "ce_reps": round(ce_reps, 4),
                    "ce_zeroed": round(ce_zero, 4),
                    "ce_none": round(ce_none, 4),
                    "rep_gain_vs_zeroed": round(ce_zero - ce_reps, 4),
                }
                # W: wiring A/B on the (0.30, None) point
                if (r0, r1) == (0.30, None):
                    ce_b = interleaved_ce(trainer, surv, prefix, suffix, sm)
                    entry["ce_interleaved_same_inputs"] = round(ce_b, 4)
                    entry["api_delta"] = round(ce_b - ce_reps, 4)
                row["ratios"][f"l0={r0},l1={r1}"] = entry
            report["rows"].append(row)
            print("\n" + "=" * 100)
            print(json.dumps(row, indent=2, default=str))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    assert cfg.training.phase == "phase2_kb"
    trainer = KRKBTrainer(cfg)
    trainer.setup()
    trainer.model.eval()

    families = [
        f.strip()
        for f in os.environ.get("DIAG_FAMILIES", "qwen35,falcon_h1").split(",")
        if f.strip()
    ]
    docs = pick_docs(trainer)
    logger.info("docs_picked", labels=[d[0] for d in docs])

    report: dict = {"ckpt_new": CKPT_NEW, "rows": []}
    with torch.no_grad():
        run_pass(trainer, docs, "base51945", families, report)
        trainer.load_checkpoint(Path(CKPT_NEW))
        logger.info("checkpoint_loaded", step=trainer.global_step)
        trainer.model.eval()
        run_pass(trainer, docs, "9164", families, report)

    # aggregate: mean rep_gain per (ckpt, family, ratio)
    agg: dict = {}
    for r in report["rows"]:
        for rk, e in r["ratios"].items():
            key = f'{r["ckpt"]}/{r["family"]}/{rk}'
            a = agg.setdefault(
                key, {"gain": [], "reps": [], "zero": [], "none": [], "api": []},
            )
            a["gain"].append(e["rep_gain_vs_zeroed"])
            a["reps"].append(e["ce_reps"])
            a["zero"].append(e["ce_zeroed"])
            a["none"].append(e["ce_none"])
            if "api_delta" in e:
                a["api"].append(e["api_delta"])
    print("\n" + "#" * 100)
    for key, a in sorted(agg.items()):
        n = len(a["gain"])
        row = {
            "mean_ce_reps": round(sum(a["reps"]) / n, 4),
            "mean_ce_zeroed": round(sum(a["zero"]) / n, 4),
            "mean_ce_none": round(sum(a["none"]) / n, 4),
            "mean_rep_gain": round(sum(a["gain"]) / n, 4),
        }
        if a["api"]:
            row["mean_api_delta"] = round(sum(a["api"]) / len(a["api"]), 4)
        print(f"REALPATH [{key}]:", json.dumps(row))
        report.setdefault("agg", {})[key] = row

    out = Path("/workspace/checkpoints/diag_git_repro_realpath_control.json")
    try:
        out.write_text(json.dumps(report, indent=2, default=str))
        logger.info("report_written", path=str(out))
    except Exception as exc:  # noqa: BLE001
        logger.warning("report_write_failed", err=str(exc))


if __name__ == "__main__":
    main()
