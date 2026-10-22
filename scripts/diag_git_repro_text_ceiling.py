#!/usr/bin/env python
"""TEXT CEILING for the real git-repro task: diffs as TEXT -> full file recon.

Separates "task is hard" from "rep channel is dead": give the decoders the
SAME evidence the trajectory retrieves (the touching-commit diffs, oldest
first) as PLAIN TEXT in a normal chat prompt, gold = the file blob at the
target commit, and measure teacher-forced CE + GREEDY GENERATION quality.

Diff sets per sample:
  n  no diffs (query-only LM-prior floor)
  a  target-commit diff only (the last touching diff)
  b  the 3 most-recent touching diffs
  c  ALL touching diffs (oldest first)
  e  echo sanity (subset): the gold file itself as evidence -> should copy

Passes: base51945 (summarization decoders, pre-git-repro; CE all sets + gen on
a small subset) -> load step-9164 -> full pass (CE all sets; generation for
sets n/a/c on a subset). Both decoder families.

Metrics: CE on gold span; teacher-forced token-acc; generation line-level
P/R/F1 vs gold (non-empty stripped lines), exact match, prefix-token match;
2-3 dumped generations.

Run:
  docker compose --env-file .env -f docker/docker-compose.yaml run --rm -T \
    train-phase2-kb-git-repro-fullbackprop \
    python scripts/diag_git_repro_text_ceiling.py \
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

CKPT_NEW = os.environ.get(
    "DIAG_CKPT",
    "/workspace/checkpoints_fast/"
    "phase2_kb_step9164_20260730_091047_791250_run-phase2_kb_git_repro_fullbackprop",
)

CE_GOLD_CAP = 3072
DIFF_TOKEN_CAP = 6000
GEN_MAX_NEW = 900


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
    """Touching-diff texts, oldest -> newest (trajectory retrieve order)."""
    ds = sample.dataset_name
    tree = trainer._trees.get(ds)
    texts = []
    for t in sample.trajectory:
        if t.kind != "bgkit" or bool(t.args.get("is_head", False)):
            continue
        ids = list(t.args.get("ids", []))
        q = str(t.args.get("query", ""))
        if q == "" and len(ids) == 1 and tree is not None and str(ids[0]) in tree:
            continue
        for d in trainer._resolve_article_ids(ds, ids):
            texts.append(trainer.encoder_tokenizer.decode(
                trainer._token_store.get(ds, d).tolist(),
            ))
    return texts


def build_prompt_ids(trainer, sample, diffs: list[str] | None) -> torch.Tensor:
    sys_msg = (
        "You are a precise code reconstruction engine. Reconstruct file "
        "contents exactly, character for character."
    )
    user = str(sample.question)
    if diffs is not None and diffs:
        # Cap total diff tokens, keep the NEWEST diffs (drop oldest overflow).
        kept: list[str] = []
        total = 0
        for d in reversed(diffs):
            n = len(trainer.tokenizer.encode(d, add_special_tokens=False))
            if total + n > DIFF_TOKEN_CAP and kept:
                break
            kept.append(d)
            total += n
        kept = list(reversed(kept))
        blocks = "\n\n".join(
            f"### Diff {i + 1} of {len(kept)} (oldest first):\n{d}"
            for i, d in enumerate(kept)
        )
        user += (
            "\n\nHere are the diffs that touched the file, oldest first:\n\n"
            + blocks
            + "\n\nOutput ONLY the complete file content as of the target commit."
        )
    else:
        user += "\n\nOutput ONLY the complete file content as of the target commit."
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user},
    ]
    s1 = trainer.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )
    ids = trainer.tokenizer.encode(s1, add_special_tokens=False)
    return torch.tensor(ids, dtype=torch.long, device=trainer.device)


def ce_and_tf_acc(trainer, prompt_ids, gold_ids) -> tuple[float, float, int]:
    dev = trainer.device
    segs = [
        TokenSegment(
            token_ids=prompt_ids.unsqueeze(0),
            loss_mask=torch.zeros(
                (1, prompt_ids.shape[0]), dtype=torch.bool, device=dev,
            ),
        ),
        TokenSegment(
            token_ids=gold_ids.unsqueeze(0),
            loss_mask=torch.ones(
                (1, gold_ids.shape[0]), dtype=torch.bool, device=dev,
            ),
        ),
    ]
    span = (
        int(prompt_ids.shape[0]),
        int(prompt_ids.shape[0]) + int(gold_ids.shape[0]),
    )
    out = trainer.decoder.forward_interleaved_with_loss(
        segs, return_hidden_states=True,
    )
    s, c = KRKBTrainer._span_ce_sum_count(out, [span])
    preds = out.argmax_predictions()  # (1, S-1)
    tgt = out.token_ids[:, 1:]
    m = torch.zeros_like(out.loss_mask[:, 1:])
    lo = max(0, span[0] - 1)
    hi = min(m.shape[1], span[1] - 1)
    m[:, lo:hi] = True
    m = m & out.loss_mask[:, 1:]
    acc = float(((preds == tgt) & m).sum().item()) / max(1, int(m.sum().item()))
    del out
    return s / max(c, 1), acc, c


def greedy_generate(trainer, prompt_ids, max_new: int) -> str:
    d = trainer.decoder
    zero_surv = torch.zeros(
        (0, d.hidden_dim), device=trainer.device, dtype=torch.bfloat16,
    )
    cu = torch.tensor([0, 0], dtype=torch.int32, device=trainer.device)
    out = d.generate_with_single_splice(
        survivor_embeddings=zero_surv,
        survivor_cu_seqlens=cu,
        prefix_ids=prompt_ids,
        suffix_ids=torch.zeros(0, dtype=torch.long, device=trainer.device),
        tokenizer=trainer.tokenizer,
        max_new_tokens=max_new,
        temperature=0.0,
    )
    return out.content_text[0] if out.content_text else ""


def line_prf(gen: str, gold: str) -> tuple[float, float, float]:
    g_lines = {ln.strip() for ln in gold.splitlines() if ln.strip()}
    p_lines = {ln.strip() for ln in gen.splitlines() if ln.strip()}
    if not g_lines or not p_lines:
        return 0.0, 0.0, 0.0
    inter = len(g_lines & p_lines)
    p = inter / len(p_lines)
    r = inter / len(g_lines)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


def prefix_match_frac(trainer, gen: str, gold: str) -> float:
    a = trainer.tokenizer.encode(gen, add_special_tokens=False)
    b = trainer.tokenizer.encode(gold, add_special_tokens=False)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return round(i / n, 3)


def run_pass(trainer, picked, tag, families, gen_samples, report):
    for family in families:
        if trainer._round_robin:
            trainer._set_active_decoder(family)
        for si, s in enumerate(picked):
            diffs = ordered_diffs(trainer, s)
            gold = str(s.gold_answer)
            gold_ids = torch.tensor(
                trainer.tokenizer.encode(gold, add_special_tokens=False)[:CE_GOLD_CAP],
                dtype=torch.long, device=trainer.device,
            )
            sets = {
                "n_none": None,
                "a_target_only": diffs[-1:],
                "b_recent3": diffs[-3:],
                "c_all": diffs,
            }
            if si < 3:
                sets["e_echo"] = [gold]
            row = {
                "ckpt": tag, "family": family,
                "question": s.question[:100],
                "n_diffs": len(diffs),
                "gold_tokens": int(gold_ids.shape[0]),
            }
            do_gen = si < gen_samples
            for name, dset in sets.items():
                prompt_ids = build_prompt_ids(trainer, s, dset)
                ce, tf_acc, _ = ce_and_tf_acc(trainer, prompt_ids, gold_ids)
                entry = {
                    "ce": round(ce, 4),
                    "tf_acc": round(tf_acc, 4),
                    "prompt_tokens": int(prompt_ids.shape[0]),
                }
                if do_gen and name in ("n_none", "a_target_only", "c_all"):
                    gen = greedy_generate(
                        trainer, prompt_ids,
                        min(int(gold_ids.shape[0]) + 64, GEN_MAX_NEW),
                    )
                    gold_cmp = trainer.tokenizer.decode(
                        gold_ids[: GEN_MAX_NEW].tolist(),
                    )
                    p, r, f = line_prf(gen, gold_cmp)
                    entry.update({
                        "gen_line_p": p, "gen_line_r": r, "gen_line_f1": f,
                        "gen_exact": gen.strip() == gold_cmp.strip(),
                        "gen_prefix_match": prefix_match_frac(trainer, gen, gold_cmp),
                        "gen_len_chars": len(gen),
                    })
                    if si < 2 and name == "c_all":
                        entry["gen_dump"] = gen[:600]
                        entry["gold_dump"] = gold_cmp[:600]
                row[name] = entry
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
    n_samples = int(os.environ.get("DIAG_N_FULL", "14"))
    gen_samples = int(os.environ.get("DIAG_GEN_SAMPLES", "6"))
    base_gen_samples = int(os.environ.get("DIAG_BASE_GEN_SAMPLES", "2"))
    scan_cap = int(os.environ.get("DIAG_SCAN_CAP", "8000"))
    do_base = os.environ.get("DIAG_BASE_PASS", "1") == "1"

    # ---- pick samples: spread over n_retrieve and gold size ----
    cands: list = []
    scanned = 0
    for batch in trainer.eval_dataloader:
        for s in batch:
            scanned += 1
            if s.dataset_name != "git_commit_repro":
                continue
            kinds = classify_turns(trainer, s)
            n_r = sum(1 for k in kinds if k == "retrieve")
            if n_r >= 1:
                cands.append((n_r, len(str(s.gold_answer)), s))
        if scanned >= scan_cap or len(cands) >= n_samples * 6:
            break
    if not cands:
        raise RuntimeError("no full-drill samples")
    # spread: sort by n_retrieve then gold size; take evenly spaced,
    # max 2 per repo root
    cands.sort(key=lambda x: (x[0], x[1]))
    picked: list = []
    step = max(1, len(cands) // n_samples)
    for i in range(0, len(cands), step):
        s = cands[i][2]
        root = trainer._repo_group_key(s)
        if sum(1 for p in picked if trainer._repo_group_key(p) == root) < 2:
            picked.append(s)
        if len(picked) >= n_samples:
            break
    logger.info(
        "samples_picked", n=len(picked), scanned=scanned,
        n_retrieve=[c[0] for c in cands[::step]][: len(picked)],
    )

    report: dict = {"ckpt_new": CKPT_NEW, "rows": []}
    with torch.no_grad():
        if do_base:
            run_pass(trainer, picked, "base51945", families,
                     base_gen_samples, report)
        trainer.load_checkpoint(Path(CKPT_NEW))
        logger.info("checkpoint_loaded", step=trainer.global_step)
        trainer.model.eval()
        run_pass(trainer, picked, "9164", families, gen_samples, report)

    # ---- aggregates: token-weighted CE + mean gen metrics ----
    aggs: dict = {}
    for r in report["rows"]:
        for name in ("n_none", "a_target_only", "b_recent3", "c_all", "e_echo"):
            if name not in r:
                continue
            e = r[name]
            key = (r["ckpt"], r["family"], name)
            a = aggs.setdefault(
                key,
                {"ce_s": 0.0, "w": 0, "tf_s": 0.0,
                 "gen_f1": [], "gen_pm": [], "gen_ex": []},
            )
            a["ce_s"] += e["ce"] * r["gold_tokens"]
            a["tf_s"] += e["tf_acc"] * r["gold_tokens"]
            a["w"] += r["gold_tokens"]
            if "gen_line_f1" in e:
                a["gen_f1"].append(e["gen_line_f1"])
                a["gen_pm"].append(e["gen_prefix_match"])
                a["gen_ex"].append(1.0 if e["gen_exact"] else 0.0)
    print("\n" + "#" * 100)
    for (ck, fam, name), a in sorted(aggs.items()):
        row = {
            "ce": round(a["ce_s"] / max(a["w"], 1), 4),
            "tf_acc": round(a["tf_s"] / max(a["w"], 1), 4),
        }
        if a["gen_f1"]:
            row["gen_line_f1"] = round(sum(a["gen_f1"]) / len(a["gen_f1"]), 3)
            row["gen_prefix_match"] = round(
                sum(a["gen_pm"]) / len(a["gen_pm"]), 3,
            )
            row["gen_exact_rate"] = round(
                sum(a["gen_ex"]) / len(a["gen_ex"]), 3,
            )
            row["n_gen"] = len(a["gen_f1"])
        print(f"TEXTCEIL [{ck}/{fam}/{name}]:", json.dumps(row))
        report.setdefault("agg", {})[f"{ck}/{fam}/{name}"] = row

    out = Path("/workspace/checkpoints/diag_git_repro_text_ceiling.json")
    try:
        out.write_text(json.dumps(report, indent=2, default=str))
        logger.info("report_written", path=str(out))
    except Exception as exc:  # noqa: BLE001
        logger.warning("report_write_failed", err=str(exc))


if __name__ == "__main__":
    main()
