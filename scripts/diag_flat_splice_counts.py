#!/usr/bin/env python
"""Diagnostic: per-stage survivor counts on the FLAT live-L0 path.

For a handful of train samples of the configured experiment, run the real
``_prepare_sample_for_decode`` → ``_run_l1_batch`` chain and print, per sample:
article tokens N, L1-input content rows (pinned ids + L0 survivors), L0 keep
rate, spliced rep rows, and the decoder sequence length the trainer would
build. Answers "is the configured retention actually applied?" without a
training step. Run inside the GPU container:

    python scripts/diag_flat_splice_counts.py +experiment=phase2_kb_widenet_v5 \
        +diag.n_samples=6
"""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
from bgkit.utils.logging import setup_logging


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    trainer = KRKBTrainer(cfg)
    trainer.setup()
    n = int((cfg.get("diag", {}) or {}).get("n_samples", 6))
    trainer.model.train()  # training-mode routing (aux stashes are harmless here)
    ds = trainer.train_dataset
    step = max(1, len(ds) // n)
    rows = []
    with torch.no_grad():
        for i in range(0, len(ds), step):
            if len(rows) >= n:
                break
            sample = ds[i]
            if getattr(trainer, "_round_robin", False):
                trainer._set_active_decoder("qwen35")
            prep = trainer._prepare_sample_for_decode(sample)
            turns = prep["prepared_turns"]
            turn = turns[0] if turns else None
            art_id = sample.trajectory[0].args["ids"][0]
            n_tok = int(trainer._token_store.length(sample.dataset_name, art_id))
            if turn is None or not isinstance(turn, dict) or "content" not in turn:
                rows.append((sample.dataset_name, n_tok, "NONE-TURN", None, None, None, None))
                continue
            content_rows = int(turn["content"].shape[0])
            pinned = int(turn["pinned"].sum().item())
            l0_surv = int(turn["survivor_mask"].sum().item())
            # L0 ENCODES gold + distractors, so the survivor count is over ALL
            # of them; dividing by the gold article's length alone reads as a
            # 4x-too-high keep rate (2026-08-25: a configured 0.10 looked like
            # 0.40 in a config that had inherited n_distractors=3).
            n_l0_articles = 1
            n_l0_tokens = n_tok
            # L0 head score health: raw score stats + tanh-rail saturation
            # (the 2026-08-22 collapse: every token at tanh(raw/T) == -1).
            l0_out = None
            pend = getattr(trainer, "_pending_l0_outputs", None) or []
            if pend:
                l0_out = pend[-1].get("enc_out")
            if l0_out is not None and getattr(l0_out, "base_raw", None) is not None:
                br = l0_out.base_raw.detach().float()
                temp = float(trainer.encoder.l0.head_tanh_temperature)
                sq = torch.tanh(br / temp)
                floor_frac = (sq <= -0.999).float().mean()
                ceil_frac = (sq >= 0.999).float().mean()
                print(
                    f"  L0 head: base_raw mean={br.mean():.2f} std={br.std():.2f} "
                    f"min={br.min():.1f} max={br.max():.1f} T={temp:.2f} "
                    f"tanh rail frac: floor={floor_frac:.3f} ceil={ceil_frac:.3f}",
                    flush=True,
                )
                trainer._pending_l0_outputs = []
            content_cu = getattr(trainer, "_last_l0_content_cu", None)
            if content_cu is not None and int(content_cu.numel()) > 1:
                n_l0_articles = int(content_cu.numel()) - 1
                n_l0_tokens = int(content_cu[-1].item())
            ratio_l1 = trainer._drill_leaf_l1_retention_override()
            survs = trainer._run_l1_batch([turn], target_ratio=ratio_l1)
            reps = int(survs[0].shape[0])
            segs, _trace = trainer._assemble_sample_segments(prep, survs)
            dec_len = 0
            for seg in segs:
                tk = getattr(seg, "token_ids", None)
                if tk is not None:
                    dec_len += int(tk.reshape(-1).shape[0])
                else:
                    emb = getattr(seg, "embeddings", None)
                    if emb is not None:
                        dec_len += int(emb.reshape(-1, emb.shape[-1]).shape[0])
            rows.append((sample.dataset_name, n_tok, content_rows, pinned, l0_surv, reps, dec_len))
            thr = getattr(getattr(trainer.encoder, "l0", None), "threshold", None)
            theta = getattr(thr, "theta", None)
            theta_s = f"{float(theta):.4f}" if theta is not None else "?"
            print(
                f"DIAG {sample.dataset_name}: N={n_tok} l1_content_rows={content_rows} "
                f"(pinned={pinned}, l0_survivors={l0_surv}, "
                f"l0_keep={l0_surv / max(n_l0_tokens, 1):.3f} over "
                f"{n_l0_articles} article(s)/{n_l0_tokens} tok) "
                f"reps={reps} (rep/N={reps / max(n_tok, 1):.3f}) decode_len={dec_len} "
                f"l0_cfg_ratio={trainer._l0_retention_for(sample.dataset_name):.3f} "
                f"l0_mode={getattr(trainer, '_selection_mode_l0', '?')} "
                f"l1_ratio={ratio_l1} theta_l0={theta_s}",
                flush=True,
            )
    print("DIAG DONE", flush=True)


if __name__ == "__main__":
    main()
