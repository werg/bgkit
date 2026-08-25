#!/usr/bin/env python
"""DECISIVE control: can the KNOWN-GOOD 51945 encoder compress+reproduce a git
DIFF (and a gold target FILE) as well as it reproduces in-distribution prose/code?

Motivation: the A-vs-B diagnostic concluded "diffs don't survive lossy
compression" — but it read reps from the STEP-286 git-repro checkpoint through
the RECURSIVE-L1 TREE pipeline (_run_l1_batch: L1 LoRA + l1l1 bridge + nav +
splice, all trained on the suspect task). That confounds "diffs are
incompressible" with "the git-repro tree pipeline / step-286 training mangles
the reps". This script removes BOTH confounds: it uses the 51945 base encoder
(matched, load-bearing +2 on summarization) via the PROVEN single-shot
_encode_batch -> forward_with_single_splice path (NO tree, NO navigation), and
verbatim-reproduces the content.

For each content item and BOTH decoder families, at several retention operating
points (including the git-repro leaf retention l0=0.63 x l1=0.63 ~= 0.40 kept):

  rep_gain = CE(reps zeroed) - CE(reps present)   [load-bearing measure]

Content types:
  * git_diff       : real leaf diffs from the git_commit_repro token store
  * git_multi_diff : several concatenated diffs (mimics a retrieve turn)
  * git_gold_file  : real gold target files (trajectory gold_answer) — the
                     ACTUAL reconstruction target, reproduced verbatim
  * summ_source    : in-distribution summarization SOURCE (arxiv/multi_news),
                     reproduced verbatim (the content the encoder was trained on)
  * summ_summary   : canonical summary reproduction (machinery-health anchor —
                     this is the path that scored +2)

Interpretation:
  git_* rep_gain HEALTHY and ~ summ_source at matched retention
      -> diffs/files ARE compressible; the git-repro WALL is the tree pipeline
         / step-286 training, i.e. A BUG (not content physics).
  git_* rep_gain << summ_source at matched retention
      -> content genuinely matters (the original A-vs-B verdict survives).

Run (container, trainer STOPPED, GPU free):
  docker compose --env-file .env -f docker/docker-compose.yaml run --rm --no-deps \
    train-phase2-kb-git-repro-fullbackprop \
    python /workspace/scripts/diag_git_repro_verbatim_control.py \
    +experiment=phase1_summarization_round_robin \
    step1_checkpoint=/workspace/checkpoints/phase1_summarization_round_robin_step51945_20260624_060459 \
    training.max_total_source_tokens=4096
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import numpy as np
import structlog
import torch
from omegaconf import DictConfig

from bgkit.training.phase1.summarization_round_robin import (
    SummarizationRoundRobinTrainer,
)
from bgkit.utils.logging import setup_logging

logger = structlog.get_logger()

# (l0_ratio, l1_ratio_or_None). 0.999 ~ near-lossless (all survive).
# (0.63, 0.63) == the git_commit_repro leaf retention (the operating point in
# question). (0.316, 0.316) == the summarization curriculum END the 51945
# checkpoint was trained at (most in-distribution).
RATIOS = (
    (0.999, None),
    (0.63, 0.63),
    (0.5, None),
    (0.316, 0.316),
)

CONTENT_JSON = os.environ.get(
    "DIAG_CONTENT_JSON", "/workspace/scripts/git_repro_content.json",
)
SRC_CAP = int(os.environ.get("DIAG_SRC_CAP", "3500"))   # encoder source tokens
TGT_CAP = int(os.environ.get("DIAG_TGT_CAP", "3072"))   # decoder target tokens


def build_loss_mask(prefix_ids, per_group, suffix_masks, device):
    full = []
    for i, (pre, sm) in enumerate(zip(prefix_ids, suffix_masks, strict=True)):
        n_pre = int(pre.shape[0])
        n_surv = int(per_group[i])
        full.append(
            torch.cat([
                torch.zeros(n_pre, dtype=torch.bool, device=device),
                torch.zeros(n_surv, dtype=torch.bool, device=device),
                sm.to(device),
            ])
        )
    return torch.cat(full)


def ce_with_slot(decoder, survivors, group_cu, prefix_ids, suffix_ids, loss_mask):
    out = decoder.forward_with_single_splice(
        survivor_embeddings=survivors,
        survivor_cu_seqlens=group_cu,
        prefix_ids=prefix_ids,
        suffix_ids=suffix_ids,
        loss_mask=loss_mask,
    )
    loss = out.loss if hasattr(out, "loss") else out
    return float(loss.item())


def make_batch(trainer, family, text, label):
    """Synthetic single-group batch: encoder source = enc(text); decoder target
    = dec(text) VERBATIM (loss-masked). Uses a valid summ template (multi_news)
    so _build_chat_inputs works; only the target tokens are ours."""
    enc_ids = np.asarray(
        trainer.encoder_tokenizer.encode(text, add_special_tokens=False)[:SRC_CAP],
        dtype=np.int64,
    )
    dec_tok = trainer.tokenizer_qwen if family == "qwen35" else trainer.tokenizer_falcon
    tgt = np.asarray(
        dec_tok.encode(text, add_special_tokens=False)[:TGT_CAP], dtype=np.int64,
    )
    return {
        "source_docs": [[enc_ids]],
        "targets_qwen": [tgt],
        "targets_falcon": [tgt],
        "dataset_names": ["multi_news"],
        "group_ids": [f"verbatim/{label}"],
    }, int(enc_ids.shape[0]), int(tgt.shape[0])


@torch.no_grad()
def run_item(trainer, family, label, ctype, text, report):
    decoder = trainer.decoder_qwen if family == "qwen35" else trainer.decoder_falcon
    dev = trainer.device
    embed_dec = decoder._get_inner_model_and_head()[0].get_input_embeddings()
    embed_norm_mean = float(embed_dec.weight.detach().float().norm(dim=-1).mean())

    batch, n_src, n_tgt = make_batch(trainer, family, text, label)
    prefix_ids, suffix_ids, suffix_masks, _comp = trainer._build_chat_inputs(family, batch)
    row = {
        "family": family, "label": label, "ctype": ctype,
        "src_tokens": n_src, "tgt_tokens": n_tgt,
        "embed_norm_mean": round(embed_norm_mean, 3), "ratios": {},
    }
    for r0, r1 in RATIOS:
        trainer._target_ratio_start = trainer._target_ratio_end = r0
        trainer.global_step = 0
        if r1 is None:
            trainer._l1_introduction_step = 10**9
        else:
            trainer._l1_introduction_step = 0
            trainer._target_ratio_l1_start = trainer._target_ratio_l1_end = r1
        # No query conditioning (comp=None) -> cleanest channel-capacity test.
        enc_out, group_cu, per_group, _, _, _ = trainer._encode_batch(batch, None)
        survivors = enc_out.survivor_embeddings
        K = int(survivors.shape[0])
        lm = build_loss_mask(prefix_ids, per_group, suffix_masks, dev)
        ce_reps = ce_with_slot(decoder, survivors, group_cu, prefix_ids, suffix_ids, lm)
        ce_zero = ce_with_slot(
            decoder, torch.zeros_like(survivors), group_cu, prefix_ids, suffix_ids, lm,
        )
        if K > 0:
            vnorm = survivors.detach().float().norm(dim=-1, keepdim=True)
            surv_norm_mean = float(vnorm.mean())
        else:
            surv_norm_mean = 0.0
        row["ratios"][f"l0={r0},l1={r1}"] = {
            "K": K,
            "kept_frac": round(K / max(n_src, 1), 3),
            "ce_reps": round(ce_reps, 4),
            "ce_zeroed": round(ce_zero, 4),
            "rep_gain": round(ce_zero - ce_reps, 4),
            "surv_norm_mean": round(surv_norm_mean, 3),
            "norm_ratio": round(surv_norm_mean / max(embed_norm_mean, 1e-6), 3),
        }
        enc_out.release()
    report["rows"].append(row)
    print("\n" + "=" * 100)
    print(json.dumps(row, indent=2, default=str))


@torch.no_grad()
def run_summary_anchor(trainer, family, batch, report):
    """Canonical summary reproduction (the +2 path) — machinery-health anchor."""
    decoder = trainer.decoder_qwen if family == "qwen35" else trainer.decoder_falcon
    dev = trainer.device
    prefix_ids, suffix_ids, suffix_masks, comp = trainer._build_chat_inputs(family, batch)
    row = {"family": family, "label": "summ_summary_anchor", "ctype": "summ_summary",
           "dataset": batch["dataset_names"][0], "ratios": {}}
    for r0, r1 in ((0.63, 0.63), (0.316, 0.316)):
        trainer._target_ratio_start = trainer._target_ratio_end = r0
        trainer.global_step = 0
        trainer._l1_introduction_step = 0
        trainer._target_ratio_l1_start = trainer._target_ratio_l1_end = r1
        enc_out, group_cu, per_group, _, _, _ = trainer._encode_batch(batch, comp)
        survivors = enc_out.survivor_embeddings
        lm = build_loss_mask(prefix_ids, per_group, suffix_masks, dev)
        ce_reps = ce_with_slot(decoder, survivors, group_cu, prefix_ids, suffix_ids, lm)
        ce_zero = ce_with_slot(
            decoder, torch.zeros_like(survivors), group_cu, prefix_ids, suffix_ids, lm,
        )
        row["ratios"][f"l0={r0},l1={r1}"] = {
            "K": int(survivors.shape[0]),
            "ce_reps": round(ce_reps, 4),
            "ce_zeroed": round(ce_zero, 4),
            "rep_gain": round(ce_zero - ce_reps, 4),
        }
        enc_out.release()
    report["rows"].append(row)
    print("\n" + "=" * 100)
    print(json.dumps(row, indent=2, default=str))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    assert cfg.training.phase == "phase1_summarization_round_robin", cfg.training.phase
    trainer = SummarizationRoundRobinTrainer(cfg)
    trainer.setup()
    trainer.encoder.eval()
    trainer.decoder_qwen.eval()
    trainer.decoder_falcon.eval()

    content = json.load(open(CONTENT_JSON))
    # Build the content work-list: (label, ctype, text).
    items: list[tuple[str, str, str]] = []
    for e in content.get("diffs", []):
        items.append((f"diff_{e['n_tok_qwen']}", "git_diff", e["text"]))
    for i, e in enumerate(content.get("multi_diff", [])):
        items.append((f"multidiff_{i}", "git_multi_diff", e["text"]))
    for e in content.get("gold_files", []):
        items.append((f"goldfile_{e['n_tok_qwen']}", "git_gold_file", e["text"]))

    # In-distribution verbatim control: decode a few summ eval SOURCES to text.
    n_summ = int(os.environ.get("DIAG_N_SUMM", "4"))
    max_src = int(cfg.training.get("max_total_source_tokens", 4096))
    summ_batches = []
    for flat_i in trainer._eval_flat_idx[: n_summ * 5]:
        b = trainer._collate([int(flat_i)])
        n = sum(len(d) for d in b["source_docs"][0])
        if n > max_src:
            continue
        summ_batches.append(b)
        text = trainer.encoder_tokenizer.decode(
            np.concatenate([np.asarray(d) for d in b["source_docs"][0]]).tolist(),
        )
        items.append((f"summsrc_{b['dataset_names'][0]}_{len(summ_batches)}",
                      "summ_source", text))
        if len(summ_batches) >= n_summ:
            break

    families = [
        f.strip() for f in os.environ.get("DIAG_FAMILIES", "qwen35,falcon_h1").split(",")
        if f.strip()
    ]
    logger.info("work_list", n_items=len(items), n_summ_anchor=len(summ_batches),
                families=families)
    report = {
        "step1_checkpoint": str(cfg.get("step1_checkpoint")),
        "ratios": [f"l0={a},l1={b}" for a, b in RATIOS],
        "rows": [],
    }
    for family in families:
        trainer.encoder.set_active_decoder_family(family)
        # machinery-health anchor first
        for b in summ_batches[:2]:
            run_summary_anchor(trainer, family, b, report)
        for label, ctype, text in items:
            run_item(trainer, family, label, ctype, text, report)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- aggregate mean rep_gain per (family, ctype, ratio) ----
    print("\n" + "#" * 100)
    print("AGGREGATE mean rep_gain by (family, ctype, ratio)")
    agg: dict = {}
    for r in report["rows"]:
        for rk, e in r["ratios"].items():
            key = (r["family"], r["ctype"], rk)
            a = agg.setdefault(key, {"gain": [], "kept": [], "nr": []})
            a["gain"].append(e["rep_gain"])
            if "kept_frac" in e:
                a["kept"].append(e["kept_frac"])
            if "norm_ratio" in e:
                a["nr"].append(e["norm_ratio"])
    header = f"{'family':10s} {'ctype':16s} {'ratio':20s} {'n':>2s} {'mean_rep_gain':>13s} {'mean_kept':>9s} {'mean_nr':>8s}"
    print(header)
    print("-" * len(header))
    for (fam, ct, rk), a in sorted(agg.items()):
        n = len(a["gain"])
        mg = round(sum(a["gain"]) / n, 4)
        mk = round(sum(a["kept"]) / len(a["kept"]), 3) if a["kept"] else None
        mn = round(sum(a["nr"]) / len(a["nr"]), 3) if a["nr"] else None
        report.setdefault("agg", {})[f"{fam}/{ct}/{rk}"] = {
            "n": n, "mean_rep_gain": mg, "mean_kept": mk, "mean_norm_ratio": mn,
        }
        print(f"{fam:10s} {ct:16s} {rk:20s} {n:>2d} {mg:>13.4f} "
              f"{(mk if mk is not None else -1):>9.3f} {(mn if mn is not None else -1):>8.3f}")

    out = Path("/workspace/checkpoints/diag_git_repro_verbatim_control.json")
    try:
        out.write_text(json.dumps(report, indent=2, default=str))
        logger.info("report_written", path=str(out))
    except Exception as exc:  # noqa: BLE001
        logger.warning("report_write_failed", err=str(exc))


if __name__ == "__main__":
    main()
