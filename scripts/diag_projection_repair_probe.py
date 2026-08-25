#!/usr/bin/env python
"""DECIDING experiment: does projection-ONLY manifold repair on the 9164 encoder
restore rep-usage?  (verdict A / B / C)

Direct extension of scripts/diag_verbatim_decisive.py — same checkpoint loading
(SummarizationRoundRobinTrainer.setup), same arxiv summarization content +
reconstruction task, same per-family projection routing + norm_ratio/rep_gain
measurement — so before/after numbers are apples-to-apples with the
matched-51945 and un-repaired-9164 numbers already reported.

Flow (ONE process, apples-to-apples):
  1. setup() with the MATCHED 51945 ckpt -> measure MATCHED reference
     (norm_ratio, cos@all-survive, rep_gain) on held-out eval samples.
  2. Swap in the 9164 (hybrid) encoder.pt (keep the 51945 decoders) ->
     measure 9164-RAW (confirm norm 12-35x, gain ~0).
  3. REPAIR: freeze everything except encoder.projection_blocks['qwen35'];
     Adam on the embed-anchor loss (MSE + cos + log-norm vs
     decoder_qwen.embed_tokens(content_ids)) at target_ratio_l0=None
     (all-survive, survivor i <-> content token i 1:1), on TRAIN samples.
  4. Re-measure 9164-REPAIRED on the SAME held-out eval samples.
  5. Verdict:
     (A) anchor converges (norm->~1x, cos high) AND rep_gain jumps toward
         matched  -> manifold was the problem; SAVE repaired encoder.
     (B) anchor converges but rep_gain stays ~0 -> reps informationally weak.
     (C) anchor does NOT converge -> 9164 backbone survivors degraded.

QWEN is the decisive family: the encoder content is Qwen-tokenized, so the
embed-anchor to decoder_qwen.embed_tokens(content_ids) is exactly defined; the
L0/L1 backbone (which produces the survivors) is SHARED across families, so this
answers the core "do the reps carry content" question for both.  Falcon's
projection-only embed-anchor is ill-posed under cross-tokenizer round-robin
(qwen content-ids are out-of-vocab for the falcon table, no 1:1 alignment), so
falcon is measured (matched/raw) but not repaired.

Run (trainer stopped, GPU free):
  docker compose --env-file .env -f docker/docker-compose.yaml run --rm --no-deps -T \
    train-phase2-kb-git-repro-fullbackprop \
    python scripts/diag_projection_repair_probe.py \
    +experiment=phase1_summarization_round_robin \
    step1_checkpoint=/workspace/checkpoints/phase1_summarization_round_robin_step51945_20260624_060459 \
    training.max_total_source_tokens=3072
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import hydra
import structlog
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from bgkit.training.checkpointing import load_checkpoint
from bgkit.training.phase1.summarization_round_robin import (
    SummarizationRoundRobinTrainer,
)
from bgkit.utils.logging import setup_logging
from bgkit.utils.packing import position_ids_from_cu

logger = structlog.get_logger()

HYBRID = os.environ.get(
    "DIAG_HYBRID", "/workspace/checkpoints/phase2_kb_hybrid_enc9164_dec51945"
)
REPAIRED_OUT = os.environ.get(
    "DIAG_REPAIRED_OUT",
    "/workspace/checkpoints/phase2_kb_hybrid_enc9164repaired_dec51945",
)
DOC_CAP = 3072
N_EVAL = int(os.environ.get("DIAG_N_EVAL", "6"))
N_TRAIN = int(os.environ.get("DIAG_N_TRAIN", "96"))
REPAIR_STEPS = int(os.environ.get("DIAG_REPAIR_STEPS", "600"))
REPAIR_LR = float(os.environ.get("DIAG_REPAIR_LR", "3e-4"))

# L0-only (directly repaired path) + L0->L1 (transfer). Compressed ratios are
# the meaningful test (all-survive trivially reconstructs post-repair).
RATIOS = ((0.999, None), (0.316, None), (0.1, None), (0.5, 0.5), (0.316, 0.316))


def direct_encode(trainer, content_ids, comp_ids, r0, r1):
    """Single-segment encoder forward with EXPLICIT ratios (supports r0=None =
    all-survive). Mirrors realpath_control.encode_doc / projection_repair."""
    dev = trainer.device
    embed = trainer.encoder.l0.backbone.get_input_embeddings()
    n = int(content_ids.shape[0])
    cu = torch.tensor([0, n], dtype=torch.int32, device=dev)
    pos = position_ids_from_cu(cu, n)
    cp = comp_ids.to(device=dev, dtype=torch.long)
    pcu = torch.tensor([0, int(cp.shape[0])], dtype=torch.int32, device=dev)
    ppos = position_ids_from_cu(pcu, int(cp.shape[0]))
    group_cu = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    return trainer.encoder(
        content_embeddings=embed(content_ids),
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


def ce_with_slot(decoder, survivors, K, prefix, suffix, suffix_mask, dev):
    lm = torch.cat([
        torch.zeros(int(prefix.shape[0]), dtype=torch.bool, device=dev),
        torch.zeros(K, dtype=torch.bool, device=dev),
        suffix_mask.to(dev),
    ])
    cu = torch.tensor([0, K], dtype=torch.int32, device=dev)
    out = decoder.forward_with_single_splice(
        survivor_embeddings=survivors, survivor_cu_seqlens=cu,
        prefix_ids=[prefix], suffix_ids=[suffix], loss_mask=lm,
    )
    return float((out.loss if hasattr(out, "loss") else out).item())


def content_ids_of(batch, dev):
    return torch.cat(
        [torch.as_tensor(d, dtype=torch.long) for d in batch["source_docs"][0]]
    )[:DOC_CAP].to(dev)


@torch.no_grad()
def measure(trainer, family, samples, tag, report):
    trainer.encoder.set_active_decoder_family(family)
    trainer.encoder.eval()
    decoder = trainer.decoder_qwen if family == "qwen35" else trainer.decoder_falcon
    dev = trainer.device
    embed_dec = decoder._get_inner_model_and_head()[0].get_input_embeddings()
    embed_norm = float(embed_dec.weight.detach().float().norm(dim=-1).mean())

    for si, batch in enumerate(samples):
        cids = content_ids_of(batch, dev)
        prefix, suffix, masks, comp = trainer._build_chat_inputs(family, batch)
        prefix, suffix, mask = prefix[0], suffix[0], masks[0]
        comp0 = comp[0]

        # cos + norm @ all-survive L0 (1:1 alignment with content).
        # cos-to-embed is only defined for qwen (encoder content is Qwen-tokenized;
        # qwen ids are out-of-vocab / semantically wrong for the falcon table).
        enc_all = direct_encode(trainer, cids, comp0, None, None)
        proj_all = enc_all.survivor_embeddings.float()
        m = min(proj_all.shape[0], cids.shape[0])
        if family == "qwen35":
            tgt = embed_dec(cids[:m]).float()
            cos_all = float(F.cosine_similarity(proj_all[:m], tgt[:m], dim=-1).mean())
        else:
            cos_all = None
        nr_all = float(proj_all[:m].norm(dim=-1).mean() / max(embed_norm, 1e-6))
        enc_all.release()

        row = {"tag": tag, "family": family, "sample": si,
               "cos_allsurvive": (round(cos_all, 4) if cos_all is not None else None),
               "norm_ratio_allsurvive": round(nr_all, 3),
               "embed_norm": round(embed_norm, 4), "ratios": {}}
        for r0, r1 in RATIOS:
            enc = direct_encode(trainer, cids, comp0, r0, r1)
            surv = enc.survivor_embeddings
            K = int(surv.shape[0])
            ce_reps = ce_with_slot(decoder, surv, K, prefix, suffix, mask, dev)
            ce_zero = ce_with_slot(decoder, torch.zeros_like(surv), K,
                                   prefix, suffix, mask, dev)
            nr = float(surv.detach().float().norm(dim=-1).mean() / max(embed_norm, 1e-6))
            row["ratios"][f"l0={r0},l1={r1}"] = {
                "K": K, "ce_reps": round(ce_reps, 4), "ce_zeroed": round(ce_zero, 4),
                "rep_gain": round(ce_zero - ce_reps, 4), "norm_ratio": round(nr, 3),
            }
            enc.release()
        report["rows"].append(row)
    trainer.encoder.eval()


def _heldout_ce_gain(trainer, decoder, family, eval_samples, r0, r1, dev, n=3):
    """Mean rep_gain = CE(zeroed) - CE(reps) over a few held-out samples."""
    gains, reps_l, zero_l = [], [], []
    with torch.no_grad():
        for batch in eval_samples[:n]:
            cids = content_ids_of(batch, dev)
            prefix, suffix, masks, comp = trainer._build_chat_inputs(family, batch)
            enc = direct_encode(trainer, cids, comp[0], r0, r1)
            surv = enc.survivor_embeddings
            K = int(surv.shape[0])
            cer = ce_with_slot(decoder, surv, K, prefix[0], suffix[0], masks[0], dev)
            cez = ce_with_slot(decoder, torch.zeros_like(surv), K,
                               prefix[0], suffix[0], masks[0], dev)
            enc.release()
            gains.append(cez - cer); reps_l.append(cer); zero_l.append(cez)
    return (round(sum(gains) / len(gains), 3), round(sum(reps_l) / len(reps_l), 3),
            round(sum(zero_l) / len(zero_l), 3))


def repair_qwen(trainer, train_samples, eval_samples, report):
    """DIRECT test: train ONLY encoder.projection_blocks['qwen35'] to minimize
    the actual DECODER reconstruction CE (summary target) at the matched sweet
    spot (l0=0.5,l1=0.5) — everything else frozen. This is exactly "restore
    rep-usage": if the projection can present the frozen 9164 backbone survivors
    so the decoder reads them, rep_gain recovers (A); if not, the survivors are
    informationally weak / degraded (B/C). Held-out rep_gain tracked throughout."""
    family = "qwen35"
    trainer.encoder.set_active_decoder_family(family)
    decoder = trainer.decoder_qwen
    dev = trainer.device
    r0, r1 = 0.5, 0.5

    trainer.encoder.requires_grad_(False)
    decoder.requires_grad_(False)
    proj = trainer.encoder.projection_blocks[family]
    proj.requires_grad_(True)
    proj.train()
    params = [p for p in proj.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    logger.info("repair_start", family=family, mode="decoder_ce", proj_params=n_params,
                steps=REPAIR_STEPS, lr=REPAIR_LR, ratio=f"{r0},{r1}",
                n_train=len(train_samples))
    opt = torch.optim.Adam(params, lr=REPAIR_LR)

    log = []
    for step in range(REPAIR_STEPS):
        batch = train_samples[step % len(train_samples)]
        cids = content_ids_of(batch, dev)
        prefix, suffix, masks, comp = trainer._build_chat_inputs(family, batch)
        # grad flows CE -> surv(projection output) -> projection params ONLY
        # (backbone + bridge + decoder all frozen, inputs carry no requires_grad).
        enc = direct_encode(trainer, cids, comp[0], r0, r1)
        surv = enc.survivor_embeddings
        K = int(surv.shape[0])
        lm = torch.cat([
            torch.zeros(int(prefix[0].shape[0]), dtype=torch.bool, device=dev),
            torch.zeros(K, dtype=torch.bool, device=dev),
            masks[0].to(dev),
        ])
        cu = torch.tensor([0, K], dtype=torch.int32, device=dev)
        out = decoder.forward_with_single_splice(
            survivor_embeddings=surv, survivor_cu_seqlens=cu,
            prefix_ids=[prefix[0]], suffix_ids=[suffix[0]], loss_mask=lm,
        )
        loss = out.loss if hasattr(out, "loss") else out
        opt.zero_grad()
        loss.backward()
        opt.step()
        enc.release()
        if step % 100 == 0 or step == REPAIR_STEPS - 1:
            proj.eval()
            g, cr, cz = _heldout_ce_gain(trainer, decoder, family, eval_samples, r0, r1, dev)
            proj.train()
            rec = {"step": step, "train_ce": round(float(loss), 4),
                   "heldout_rep_gain": g, "heldout_ce_reps": cr, "heldout_ce_zero": cz}
            log.append(rec)
            logger.info("repair_step", **rec)
    proj.eval()
    trainer.encoder.eval()
    report["repair_log"] = log
    return log


def agg(report, tag):
    out = {}
    for r in report["rows"]:
        if r["tag"] != tag:
            continue
        for rk, e in r["ratios"].items():
            key = f'{r["family"]}/{rk}'
            a = out.setdefault(key, {"reps": [], "zero": [], "gain": [], "nr": []})
            a["reps"].append(e["ce_reps"]); a["zero"].append(e["ce_zeroed"])
            a["gain"].append(e["rep_gain"]); a["nr"].append(e["norm_ratio"])
    res = {}
    for k, a in out.items():
        n = len(a["gain"])
        res[k] = {"reps": round(sum(a["reps"]) / n, 3), "zero": round(sum(a["zero"]) / n, 3),
                  "gain": round(sum(a["gain"]) / n, 3), "nr": round(sum(a["nr"]) / n, 2)}
    # per-family all-survive cos/norm
    cos = {}
    for r in report["rows"]:
        if r["tag"] != tag:
            continue
        c = cos.setdefault(r["family"], {"cos": [], "nr": []})
        if r["cos_allsurvive"] is not None:
            c["cos"].append(r["cos_allsurvive"])
        c["nr"].append(r["norm_ratio_allsurvive"])
    coss = {}
    for f, v in cos.items():
        coss[f] = {
            "cos_allsurvive": (round(sum(v["cos"]) / len(v["cos"]), 3) if v["cos"] else None),
            "norm_ratio_allsurvive": round(sum(v["nr"]) / len(v["nr"]), 2),
        }
    return res, coss


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    assert cfg.training.phase == "phase1_summarization_round_robin"
    trainer = SummarizationRoundRobinTrainer(cfg)
    trainer.setup()
    trainer.encoder.eval()
    trainer.decoder_qwen.eval()
    trainer.decoder_falcon.eval()
    dev = trainer.device

    eval_samples, train_samples = [], []
    for flat_i in trainer._eval_flat_idx:
        b = trainer._collate([int(flat_i)])
        if sum(len(d) for d in b["source_docs"][0]) <= DOC_CAP and \
           b["dataset_names"][0] == "arxiv_s2orc":
            eval_samples.append(b)
        if len(eval_samples) >= N_EVAL:
            break
    for flat_i in trainer._train_flat_idx:
        b = trainer._collate([int(flat_i)])
        if sum(len(d) for d in b["source_docs"][0]) <= DOC_CAP and \
           b["dataset_names"][0] == "arxiv_s2orc":
            train_samples.append(b)
        if len(train_samples) >= N_TRAIN:
            break
    logger.info("samples", n_eval=len(eval_samples), n_train=len(train_samples))

    report = {"matched": cfg.get("step1_checkpoint"), "hybrid": HYBRID, "rows": []}

    # 1. MATCHED reference (enc51945 + dec51945)
    for fam in ("qwen35", "falcon_h1"):
        measure(trainer, fam, eval_samples, "matched51945", report)

    # 2. Swap in the 9164 (hybrid) encoder; keep the 51945 decoders.
    _meta, sds = load_checkpoint(Path(HYBRID))
    missing, unexpected = trainer.encoder.load_state_dict(sds["encoder"], strict=False)
    logger.info("swapped_to_9164_encoder",
                missing=len(missing), unexpected=len(unexpected),
                missing_sample=[k for k in missing][:4])
    trainer.encoder.eval()
    for fam in ("qwen35", "falcon_h1"):
        measure(trainer, fam, eval_samples, "raw9164", report)

    # 3. Projection-only repair (qwen) via direct decoder-CE.
    repair_qwen(trainer, train_samples, eval_samples, report)

    # 4. Re-measure repaired (qwen decisive; falcon unchanged -> also measure).
    for fam in ("qwen35", "falcon_h1"):
        measure(trainer, fam, eval_samples, "repaired9164", report)

    # ---- verdict ----
    print("\n" + "#" * 100)
    ref, refcos = {}, {}
    for tag in ("matched51945", "raw9164", "repaired9164"):
        ref[tag], refcos[tag] = agg(report, tag)
    report["agg"] = ref
    report["agg_allsurvive"] = refcos
    print("ALL-SURVIVE manifold alignment (cos to embed_tokens, norm_ratio):")
    for tag in ("matched51945", "raw9164", "repaired9164"):
        print(f"  {tag:14s} {json.dumps(refcos[tag])}")
    print("\nBEFORE -> AFTER  (reps CE / zeroed CE / rep_gain / norm_ratio) per family/ratio:")
    keys = sorted(set(ref["matched51945"]) | set(ref["raw9164"]) | set(ref["repaired9164"]))
    for k in keys:
        line = {t: ref[t].get(k) for t in ("matched51945", "raw9164", "repaired9164")}
        print(f"  [{k:22s}]")
        for t in ("matched51945", "raw9164", "repaired9164"):
            e = ref[t].get(k)
            if e:
                print(f"      {t:14s} reps={e['reps']:.3f} zero={e['zero']:.3f} "
                      f"GAIN={e['gain']:+.3f} nr={e['nr']}")

    # verdict: DECODER-CE repair trains projection at qwen l0=0.5,l1=0.5.
    KEY = "qwen35/l0=0.5,l1=0.5"
    rep_gain_after = ref["repaired9164"][KEY]["gain"]
    matched_gain = ref["matched51945"][KEY]["gain"]
    raw_gain = ref["raw9164"][KEY]["gain"]
    rlog = report.get("repair_log", [])
    first_ce = rlog[0]["train_ce"] if rlog else None
    last_ce = rlog[-1]["train_ce"] if rlog else None
    heldout_gain_traj = [r["heldout_rep_gain"] for r in rlog]
    # Did the projection manage to drive the decoder CE down on TRAIN?
    train_fit = (last_ce is not None and last_ce < 1.5)
    if rep_gain_after >= 0.5 * matched_gain and rep_gain_after > 0.5:
        verdict = ("A (projection-only CE-repair RESTORES rep-usage -> manifold/"
                   "projection was the problem; restart-with-anchor justified; SAVE)")
    elif train_fit:
        verdict = ("B (projection CAN drive train CE down but held-out rep_gain "
                   "stays ~0 -> reps informationally weak / do not generalize; "
                   "manifold fix NOT sufficient)")
    else:
        verdict = ("C (projection-only CE-repair CANNOT even fit -> 9164 backbone "
                   "survivors degraded; clean restart from 51945 base)")
    report["verdict"] = verdict
    report["key_ratio"] = KEY
    report["matched_gain"] = matched_gain
    report["raw_gain"] = raw_gain
    report["repaired_gain"] = rep_gain_after
    report["repair_train_ce_first_last"] = [first_ce, last_ce]
    report["heldout_gain_trajectory"] = heldout_gain_traj
    report["repaired_cos_allsurvive"] = refcos["repaired9164"]["qwen35"]["cos_allsurvive"]
    print(f"\n>>> VERDICT: {verdict}")
    print(f"    {KEY}: matched gain={matched_gain:+.3f}  raw={raw_gain:+.3f}  "
          f"repaired={rep_gain_after:+.3f}")
    print(f"    repair train_ce {first_ce} -> {last_ce}; "
          f"held-out gain trajectory: {heldout_gain_traj}")

    # save repaired encoder if verdict A (non-destructive new path)
    if verdict.startswith("A"):
        outp = Path(REPAIRED_OUT)
        outp.mkdir(parents=True, exist_ok=True)
        torch.save(trainer.encoder.state_dict(), outp / "encoder.pt")
        for f in ("decoder_qwen.pt", "decoder_falcon.pt", "metadata.json"):
            src = Path(HYBRID) / f
            if src.exists():
                shutil.copy2(src, outp / f)
        report["repaired_encoder_path"] = str(outp)
        logger.info("saved_repaired_encoder", path=str(outp))
        print(f"    SAVED repaired encoder -> {outp}")

    out = Path("/workspace/checkpoints/diag_projection_repair_probe.json")
    try:
        out.write_text(json.dumps(report, indent=2, default=str))
        logger.info("report_written", path=str(out))
    except Exception as exc:  # noqa: BLE001
        logger.warning("report_write_failed", err=str(exc))


if __name__ == "__main__":
    main()
