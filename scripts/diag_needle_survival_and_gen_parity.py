#!/usr/bin/env python
"""Diagnostic for the verbatim-needle gate (2026-08-23).

For N held-out samples that carry a gold span, report per sample:

* **needle survival**: fraction of the gold answer-span tokens that survive
  L0 selection, and of those, L1 (from the trainer's stashed survivor masks
  and the prepared turn's ``span_mask``) — the direct "is the needle in the
  reps at all" measure;
* **generation-path parity**: the first answer token's argmax under the
  teacher-forced train-mode forward vs. the first token produced by
  ``generate_with_segments`` (eval mode + cache) on the SAME segments — a
  mismatch beyond the ±1 expected from dtype/kernel differences indicates an
  eval-mode/cache asymmetry in the decoder path (the Falcon-H1 class).

Run in the GPU container with a trained checkpoint:

    python scripts/diag_needle_survival_and_gen_parity.py +experiment=phase2_kb_widenet_v6 \
        +eval.checkpoint=/workspace/checkpoints_fast/<ckpt> +diag.n_samples=12
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from bgkit.models.decoder import TokenSegment
from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
from bgkit.utils.logging import setup_logging


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    trainer = KRKBTrainer(cfg)
    trainer.setup()
    ckpt = (cfg.get("eval", {}) or {}).get("checkpoint")
    if ckpt:
        trainer.load_checkpoint(Path(str(ckpt)))
    n = int((cfg.get("diag", {}) or {}).get("n_samples", 12))
    trainer.model.eval()
    # The L0/L1 outputs (survivor masks) are stashed in ``_pending_l*_outputs``
    # only while the ENCODER is in train mode (no dropout anywhere; we run
    # under no_grad, so no checkpointing either). Decoders are switched per
    # pass below (train mode for teacher forcing, eval mode for generation).
    trainer.encoder.train()
    if getattr(trainer, "_round_robin", False):
        trainer._set_active_decoder("qwen35")
    ds = trainer.eval_dataset
    done = 0
    surv_l0: list[float] = []
    surv_l1: list[float] = []
    parity: list[bool] = []
    with torch.no_grad():
        for i in range(len(ds)):
            if done >= n:
                break
            sample = ds[i]
            span = getattr(sample, "gold_span", None)
            if span is None:
                continue
            prep = trainer._prepare_sample_for_decode(sample)
            turns = prep["prepared_turns"]
            if not turns or not isinstance(turns[0], dict) or "content" not in turns[0]:
                continue
            turn = turns[0]
            # L0: span tokens that survived (from the stashed masks).
            sm = getattr(trainer, "_last_l0_survivor_mask", None)
            cu = getattr(trainer, "_last_l0_content_cu", None)
            s, e = int(span[0]), int(span[1])
            l0_frac = float("nan")
            if sm is not None and cu is not None:
                a0, a1 = int(cu[0].item()), int(cu[1].item())
                art = sm[a0:a1]
                s_c, e_c = max(0, min(s, a1 - a0)), max(0, min(e, a1 - a0))
                if e_c > s_c:
                    l0_frac = float(art[s_c:e_c].float().mean().item())
            # L1: of the span tokens that reached L1 input, how many L1 kept.
            with trainer._teacher_forced_decoders():
                survs = trainer._run_l1_batch([turn], target_ratio=None)
            l1_frac = float("nan")
            span_mask_l1 = turn.get("span_mask")
            pend = getattr(trainer, "_pending_l1_outputs", None) or []
            if span_mask_l1 is not None and int(span_mask_l1.sum()) > 0 and pend:
                out = pend[-1]["enc_out"]
                mask = getattr(out, "survivor_mask", None)
                if mask is not None and mask.numel() == span_mask_l1.numel():
                    l1_frac = float(mask[span_mask_l1.to(mask.device)].float().mean().item())
            trainer._pending_l1_outputs = []
            trainer._pending_l0_outputs = []
            # Teacher-forced first answer token (train-mode decoder forward).
            segs, trace = trainer._assemble_sample_segments(prep, survs)
            with trainer._teacher_forced_decoders():
                out_tf = trainer.decoder.forward_interleaved_with_loss(
                    segs, return_hidden_states=True,
                )
            preds = out_tf.argmax_predictions()
            a_start = trace.answer_span[0] if trace.answer_span else None
            tf_first = int(preds[0, max(0, a_start - 1)].item()) if a_start else -1
            # Generation path (eval mode + cache): segments up to the answer span.
            gen_first = -1
            try:
                cut = a_start
                gen_segs = []
                cursor = 0
                for seg in segs:
                    tk = getattr(seg, "token_ids", None)
                    if tk is not None:
                        seg_len = int(tk.reshape(-1).shape[0])
                        if cut is not None and cursor + seg_len > cut:
                            keep = max(0, cut - cursor)
                            if keep > 0:
                                gen_segs.append(TokenSegment(token_ids=tk[..., :keep], loss=False))
                            break
                        gen_segs.append(seg)
                        cursor += seg_len
                    else:
                        emb = getattr(seg, "embeddings", None)
                        gen_segs.append(seg)
                        cursor += int(emb.reshape(-1, emb.shape[-1]).shape[0])
                g = trainer.decoder.generate_with_segments(
                    gen_segs, tokenizer=trainer.tokenizer, max_new_tokens=1, temperature=0.0,
                )
                ids = getattr(g, "token_ids", None)
                if ids is not None and len(ids) and len(ids[0]):
                    gen_first = int(ids[0][0])
            except Exception as exc:  # diagnostic must report, not crash
                print(f"  gen error: {exc!r}", flush=True)
            same = tf_first == gen_first
            surv_l0.append(l0_frac)
            surv_l1.append(l1_frac)
            parity.append(same)
            print(
                f"DIAG {sample.dataset_name}: span_len={e - s} l0_span_survival={l0_frac:.2f} "
                f"l1_span_survival_of_l0={l1_frac:.2f} reps={int(survs[0].shape[0])} "
                f"tf_first={tf_first} gen_first={gen_first} parity={same}",
                flush=True,
            )
            done += 1
    import math

    def _mean(xs):
        xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
        return sum(xs) / len(xs) if xs else float("nan")

    print(
        f"SUMMARY n={done} mean_l0_span_survival={_mean(surv_l0):.3f} "
        f"mean_l1_span_survival_of_l0={_mean(surv_l1):.3f} "
        f"gen_parity={sum(parity)}/{len(parity)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
