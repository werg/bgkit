#!/usr/bin/env python
"""CORRECTED verbatim-reconstruction control (supersedes diag_git_repro_verbatim_control).

The verbatim_control harness wrapped a VERBATIM target inside the SUMMARIZATION
tool template ("summarize this..."), then graded exact copy — a prompt/target
mismatch that produced pathological CE (4-9 nats) and a bogus "summarizer can't
copy" story. WRONG. The 51945 decoders (both Qwen AND the 90M Falcon) were
trained to decompress VERBATIM as part of the mix. This harness uses the REAL
reconstruction prompt (`file_read_repro`: "Return the file contents verbatim")
so we test the capability the decoder actually has.

Same known-good 51945 encoder + PROVEN single-shot path (_encode_batch ->
forward_with_single_splice). For each content item and BOTH families, at several
retention operating points including the git-repro leaf retention (0.63 x 0.63):

  ce_reps      : CE reconstructing the content from the COMPRESSED reps
  ce_zeroed    : CE with reps zeroed (prompt-only lower bound — should be HIGH)
  ce_oracle    : CE with the FULL uncompressed source embeddings in the slot
                 (echo/copy ceiling — should be ~0 for verbatim)
  rep_gain     : ce_zeroed - ce_reps                    (load-bearing)
  rep_ratio    : rep_gain / (ce_zeroed - ce_oracle)     (fraction of the raw-text
                 information the reps deliver; 1.0 == reps carry everything)

Content: git leaf DIFFS + gold target FILES (from the git_commit_repro store /
trajectory), and in-distribution summarization SOURCE (code/prose) as reference.

Interpretation:
  git_* ce_reps LOW + rep_gain HIGH + rep_ratio ~ summ_source
      -> diffs reconstruct verbatim FINE on the good encoder+decoder; the
         git-repro wall is the TREE PIPELINE / step-286 training (a BUG).
  git_* ce_reps HIGH + rep_gain LOW  (while summ_source reconstructs fine)
      -> diff content genuinely resists this compression channel.

Run (container, trainer STOPPED, GPU free):
  docker compose --env-file .env -f docker/docker-compose.yaml run --rm --no-deps \
    -e DIAG_CONTENT_JSON=/workspace/bgkit/scripts/git_repro_content.json \
    train-phase2-kb-git-repro-fullbackprop \
    python /workspace/bgkit/scripts/diag_git_repro_recon_control.py \
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

from bgkit.data.chat_template import TOOL_CONFIGS, tokenize_with_sentinel
from bgkit.training.phase1.summarization_round_robin import (
    SummarizationRoundRobinTrainer,
)
from bgkit.utils.logging import setup_logging

logger = structlog.get_logger()

# The verbatim-reconstruction variant + tool config (== the training prompt).
RECON_VARIANT = {
    "system_prompt": (
        "You are an AI coding assistant with access to the "
        "bgkit_read_file tool for reading file contents."
    ),
    "user_prompt": "Read the file `{file_path}`",
    "compression_prompt": "Return the file contents verbatim",
    "response_prefix": "Here are the contents of `{file_path}`:",
}
RECON_CONFIG = TOOL_CONFIGS["file_read_repro"]

RATIOS = (
    (0.999, None),
    (0.63, 0.63),
    (0.316, 0.316),
)

CONTENT_JSON = os.environ.get(
    "DIAG_CONTENT_JSON", "/workspace/bgkit/scripts/git_repro_content.json",
)
SRC_CAP = int(os.environ.get("DIAG_SRC_CAP", "3000"))
TGT_CAP = int(os.environ.get("DIAG_TGT_CAP", "2048"))


def build_loss_mask(prefix_ids, per_group, suffix_masks, device):
    full = []
    for i, (pre, sm) in enumerate(zip(prefix_ids, suffix_masks, strict=True)):
        n_pre = int(pre.shape[0])
        n_surv = int(per_group[i])
        full.append(torch.cat([
            torch.zeros(n_pre, dtype=torch.bool, device=device),
            torch.zeros(n_surv, dtype=torch.bool, device=device),
            sm.to(device),
        ]))
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


def build_recon_inputs(trainer, family, text, label, language):
    dev = trainer.device
    dec_tok = trainer.tokenizer_qwen if family == "qwen35" else trainer.tokenizer_falcon
    target_t = torch.as_tensor(
        dec_tok.encode(text, add_special_tokens=False)[:TGT_CAP], dtype=torch.long,
    )
    out = tokenize_with_sentinel(
        dec_tok, RECON_VARIANT, RECON_CONFIG,
        file_path=label, language=language,
        content_token_ids=target_t, encoder_tokenizer=trainer.encoder_tokenizer,
    )
    tok = out["token_ids"]
    lm = out["loss_mask"]
    ss = int(out["bgkit_splice_start"].item())
    sl = int(out["bgkit_splice_len"].item())
    prefix = tok[:ss].to(dev)
    suffix = tok[ss + sl:].to(dev)
    suffix_mask = lm[ss + sl:].to(torch.bool).to(dev)
    comp = out["compression_prompt_ids"].to(torch.long).to(dev)
    return prefix, suffix, suffix_mask, comp, target_t


@torch.no_grad()
def run_item(trainer, family, label, ctype, text, language, report):
    decoder = trainer.decoder_qwen if family == "qwen35" else trainer.decoder_falcon
    dev = trainer.device
    embed_dec = decoder._get_inner_model_and_head()[0].get_input_embeddings()
    embed_norm_mean = float(embed_dec.weight.detach().float().norm(dim=-1).mean())

    prefix, suffix, suffix_mask, comp, _ = build_recon_inputs(
        trainer, family, text, label, language,
    )
    enc_ids = np.asarray(
        trainer.encoder_tokenizer.encode(text, add_special_tokens=False)[:SRC_CAP],
        dtype=np.int64,
    )
    n_src = int(enc_ids.shape[0])
    batch = {
        "source_docs": [[enc_ids]],
        "dataset_names": ["multi_news"],
        "group_ids": [label],
    }
    row = {
        "family": family, "label": label, "ctype": ctype,
        "src_tokens": n_src, "tgt_tokens": int(suffix_mask.sum().item()),
        "embed_norm_mean": round(embed_norm_mean, 3), "ratios": {},
    }

    # ---- text-oracle: full uncompressed source embeddings in the slot ----
    dec_tok = trainer.tokenizer_qwen if family == "qwen35" else trainer.tokenizer_falcon
    src_dec = torch.as_tensor(
        dec_tok.encode(text, add_special_tokens=False)[:SRC_CAP], dtype=torch.long,
        device=dev,
    )
    oracle_emb = embed_dec(src_dec).to(dtype=torch.bfloat16)
    gc_o = torch.tensor([0, int(src_dec.shape[0])], dtype=torch.int32, device=dev)
    lm_o = build_loss_mask([prefix], [int(src_dec.shape[0])], [suffix_mask], dev)
    ce_oracle = ce_with_slot(decoder, oracle_emb, gc_o, [prefix], [suffix], lm_o)
    row["ce_oracle"] = round(ce_oracle, 4)

    for r0, r1 in RATIOS:
        trainer._target_ratio_start = trainer._target_ratio_end = r0
        trainer.global_step = 0
        if r1 is None:
            trainer._l1_introduction_step = 10**9
        else:
            trainer._l1_introduction_step = 0
            trainer._target_ratio_l1_start = trainer._target_ratio_l1_end = r1
        # Query-condition the encoder on the SAME "Return verbatim" prompt.
        enc_out, group_cu, per_group, _, _, _ = trainer._encode_batch(batch, [comp])
        survivors = enc_out.survivor_embeddings
        K = int(survivors.shape[0])
        lm = build_loss_mask([prefix], per_group, [suffix_mask], dev)
        ce_reps = ce_with_slot(decoder, survivors, group_cu, [prefix], [suffix], lm)
        ce_zero = ce_with_slot(
            decoder, torch.zeros_like(survivors), group_cu, [prefix], [suffix], lm,
        )
        surv_norm = (
            float(survivors.detach().float().norm(dim=-1).mean()) if K > 0 else 0.0
        )
        denom = max(ce_zero - ce_oracle, 1e-6)
        row["ratios"][f"l0={r0},l1={r1}"] = {
            "K": K, "per_group": int(per_group[0]), "kept_frac": round(K / max(n_src, 1), 3),
            "ce_reps": round(ce_reps, 4),
            "ce_zeroed": round(ce_zero, 4),
            "rep_gain": round(ce_zero - ce_reps, 4),
            "rep_ratio": round((ce_zero - ce_reps) / denom, 3),
            "norm_ratio": round(surv_norm / max(embed_norm_mean, 1e-6), 3),
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
    items: list[tuple[str, str, str, str]] = []  # label, ctype, text, language
    for e in content.get("diffs", []):
        items.append((f"diff_{e['n_tok_qwen']}", "git_diff", e["text"], "diff"))
    for i, e in enumerate(content.get("multi_diff", [])):
        items.append((f"multidiff_{i}", "git_multi_diff", e["text"], "diff"))
    for e in content.get("gold_files", []):
        items.append((f"goldfile_{e['n_tok_qwen']}", "git_gold_file", e["text"], "text"))

    # In-distribution reference: reconstruct a few summ eval SOURCES verbatim.
    n_summ = int(os.environ.get("DIAG_N_SUMM", "4"))
    max_src = int(cfg.training.get("max_total_source_tokens", 4096))
    got = 0
    for flat_i in trainer._eval_flat_idx[: n_summ * 6]:
        b = trainer._collate([int(flat_i)])
        if sum(len(d) for d in b["source_docs"][0]) > max_src:
            continue
        text = trainer.encoder_tokenizer.decode(
            np.concatenate([np.asarray(d) for d in b["source_docs"][0]]).tolist(),
        )
        items.append((f"summsrc_{b['dataset_names'][0]}_{got}", "summ_source", text, "text"))
        got += 1
        if got >= n_summ:
            break

    families = [
        f.strip() for f in os.environ.get("DIAG_FAMILIES", "qwen35,falcon_h1").split(",")
        if f.strip()
    ]
    logger.info("work_list", n_items=len(items), families=families)
    report = {"step1_checkpoint": str(cfg.get("step1_checkpoint")), "rows": []}
    for family in families:
        trainer.encoder.set_active_decoder_family(family)
        for label, ctype, text, lang in items:
            run_item(trainer, family, label, ctype, text, lang, report)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # aggregate: mean over items per (family, ctype, ratio)
    print("\n" + "#" * 100)
    print("AGGREGATE (verbatim RECONSTRUCTION prompt): rep_gain / rep_ratio by (family,ctype,ratio)")
    agg: dict = {}
    orac: dict = {}
    for r in report["rows"]:
        ok = (r["family"], r["ctype"])
        orac.setdefault(ok, []).append(r["ce_oracle"])
        for rk, e in r["ratios"].items():
            a = agg.setdefault((r["family"], r["ctype"], rk),
                               {"gain": [], "ratio": [], "reps": [], "zero": [], "nr": []})
            a["gain"].append(e["rep_gain"]); a["ratio"].append(e["rep_ratio"])
            a["reps"].append(e["ce_reps"]); a["zero"].append(e["ce_zeroed"])
            a["nr"].append(e["norm_ratio"])
    hdr = (f"{'family':10s} {'ctype':15s} {'ratio':18s} {'n':>2s} {'ce_reps':>8s} "
           f"{'ce_zero':>8s} {'rep_gain':>8s} {'rep_ratio':>9s} {'nr':>6s}")
    print(hdr); print("-" * len(hdr))
    for (fam, ct, rk), a in sorted(agg.items()):
        n = len(a["gain"])
        rec = {
            "n": n,
            "ce_reps": round(sum(a["reps"]) / n, 3),
            "ce_zeroed": round(sum(a["zero"]) / n, 3),
            "rep_gain": round(sum(a["gain"]) / n, 3),
            "rep_ratio": round(sum(a["ratio"]) / n, 3),
            "norm_ratio": round(sum(a["nr"]) / n, 3),
        }
        report.setdefault("agg", {})[f"{fam}/{ct}/{rk}"] = rec
        print(f"{fam:10s} {ct:15s} {rk:18s} {n:>2d} {rec['ce_reps']:>8.3f} "
              f"{rec['ce_zeroed']:>8.3f} {rec['rep_gain']:>8.3f} {rec['rep_ratio']:>9.3f} "
              f"{rec['norm_ratio']:>6.2f}")
    print("\nmean ce_oracle (verbatim echo ceiling; should be ~0):")
    for (fam, ct), v in sorted(orac.items()):
        print(f"  {fam:10s} {ct:15s} {round(sum(v)/len(v),4)}")

    out = Path("/workspace/checkpoints/diag_git_repro_recon_control.json")
    try:
        out.write_text(json.dumps(report, indent=2, default=str))
        logger.info("report_written", path=str(out))
    except Exception as exc:  # noqa: BLE001
        logger.warning("report_write_failed", err=str(exc))


if __name__ == "__main__":
    main()
