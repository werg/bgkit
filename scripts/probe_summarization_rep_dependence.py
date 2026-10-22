#!/usr/bin/env python
"""Did the summarization task actually USE the compressed reps?

THE QUESTION THIS SETTLES. The Phase-2 investigation has been framed as "a
trained capability decayed": the base could read reps, wide-net training
destroyed it. That framing rests on a rep_gain of 2.03-2.95 nats attributed to
the base. On 2026-08-29 that anchor did not survive checking:

  * the artifact it came from is not on disk, so it cannot be verified;
  * it was measured on GIT-REPRO, a file-state reconstruction task, not on
    summarization — and CLAUDE.md records git-repro's recon_gap as ~0.1 nats,
    which does not reconcile with 2-3;
  * ``summarization_round_robin.py`` contains ZERO occurrences of
    ablation/zeroed/rep_gain, and the base checkpoint records ``metrics: null``.
    The run that produced the base never measured rep dependence at all.

So there may never have been a measured "reps are used" state to decay FROM.
This probe measures it directly, on the base checkpoint, on the summarization
task it was actually trained on.

WHY PER-TOKEN AND NOT POOLED. The task is structurally rep-requiring — the
source document appears ONLY as spliced reps, there is no leak. But under
teacher forcing most target tokens are predictable from the preceding target
tokens alone (syntax, discourse, entities already introduced), so pooled CE can
fall a long way with the reps unread. Measured on the Phase-2 replay task the
same day: CE fell 2.42 -> 2.17 while the rep gap SHRANK. A pooled mean cannot
distinguish "reps are useless" from "reps are load-bearing for 2% of tokens" —
those have the same mean and demand opposite conclusions. So the primary read
is the DISTRIBUTION of the per-token gap.

READING THE RESULT:
  thin tail  (p99 ~ 0, top-decile ~ 0)  reps were never load-bearing here, and
                                        the whole "capability decayed" frame is
                                        wrong — nothing regressed.
  heavy tail (p99 >> mean)              reps ARE load-bearing for a minority of
                                        content-introducing tokens, the pooled
                                        number was hiding it, and rep_gain as
                                        currently computed is the wrong metric.

Usage (GPU container, no trainer running):
    python scripts/probe_summarization_rep_dependence.py \\
      +experiment=phase1_summarization_round_robin \\
      +probe.checkpoint=/workspace/checkpoints/phase1_summarization_round_robin_step51945_... \\
      +probe.n_batches=24
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from bgkit.eval.ablations import rescale_to_embed_norm
from bgkit.eval.rep_dependence import (
    per_token_ce,
    split_by_source_overlap,
    summarize_gap,
)
from bgkit.training.checkpointing import load_checkpoint, normalize_model_state
from bgkit.training.phase1.summarization_round_robin import (
    SummarizationRoundRobinTrainer,
)
from bgkit.utils.logging import setup_logging


def _score_batch(trainer, batch, family: str, zero_reps: bool, rescale: bool = False):
    """One teacher-forced forward; returns (ce, positions, target_ids, source_ids).

    Mirrors ``SummarizationRoundRobinTrainer.evaluate`` exactly — same
    _build_chat_inputs, same _encode_batch, same loss-mask assembly — so the
    only difference between the two arms is whether the survivor embeddings are
    zeroed. Any divergence from the trainer's own path would make the number
    unattributable.
    """
    trainer.encoder.set_active_decoder_family(family)
    decoder = trainer.decoder_qwen if family == "qwen35" else trainer.decoder_falcon
    prefix_ids, suffix_ids, suffix_masks, comp_prompt_ids = trainer._build_chat_inputs(
        family, batch,
    )
    enc_out, group_cu, per_group, _, _, _ = trainer._encode_batch(batch, comp_prompt_ids)
    try:
        full_masks = []
        for i, (pre, sm) in enumerate(zip(prefix_ids, suffix_masks, strict=True)):
            zeros_pre = torch.zeros(pre.shape[0], dtype=torch.bool, device=sm.device)
            zeros_surv = torch.zeros(
                int(per_group[i]), dtype=torch.bool, device=sm.device,
            )
            full_masks.append(torch.cat([zeros_pre, zeros_surv, sm]))

        survivors = enc_out.survivor_embeddings
        if zero_reps:
            survivors = torch.zeros_like(survivors)
        elif rescale:
            # IS THE INTERFERENCE MECHANICAL OR INFORMATIONAL? The reps sit at
            # ~500x the decoder's embedding norm, and the sweep found CE WITH
            # reps rising monotonically with how many are spliced (3.3586 at
            # l0=0.10 -> 3.4268 at 0.63) while the zeroed arm stayed flat. That
            # is dose-dependent HARM. Two explanations: high-norm vectors
            # disrupt the decoder's attention/normalisation (mechanical), or the
            # content genuinely misleads (informational). Rescaling to the
            # embedding norm changes ONLY the magnitude and leaves direction —
            # i.e. all the information — untouched.
            survivors = rescale_to_embed_norm(
                survivors,
                trainer.decoder_qwen.backbone.get_input_embeddings().weight,
            )

        out = decoder.forward_with_single_splice(
            survivor_embeddings=survivors,
            survivor_cu_seqlens=group_cu,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            loss_mask=torch.cat(full_masks),
            return_hidden_states=True,
        )
        ce, pos = per_token_ce(
            out.hidden_states, out.token_ids, out.loss_mask, out.lm_head,
        )
        tok = out.token_ids
        if tok.dim() == 2:
            tok = tok[0]
        targets = tok[1:][pos] if pos.numel() else pos
        return ce.detach().cpu(), pos.detach().cpu(), targets.detach().cpu()
    finally:
        enc_out.release()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    probe = cfg.get("probe", {}) or {}
    ckpt = probe.get("checkpoint")
    n_batches = int(probe.get("n_batches", 24))
    family = str(probe.get("family", "qwen35"))
    if not ckpt:
        raise SystemExit("pass +probe.checkpoint=/workspace/checkpoints/...")

    trainer = SummarizationRoundRobinTrainer(cfg)
    trainer.setup()
    _meta, state = load_checkpoint(Path(str(ckpt)))
    trainer._restore_model_state(normalize_model_state(state))

    # PIN THE RETENTION EXPLICITLY. _encode_batch reads
    # _current_target_ratio(self.global_step), and a freshly-set-up trainer sits
    # at global_step 0, i.e. target_ratio_START. The first run of this probe
    # therefore measured at an operating point that was never pinned, never
    # logged, and not comparable to anything — the same class of defect this
    # probe exists to expose. Override and PRINT what was used.
    r0 = probe.get("ratio_l0", None)
    r1 = probe.get("ratio_l1", "__unset__")
    if r0 is not None:
        trainer._current_target_ratio = lambda _v=float(r0): _v
    if r1 != "__unset__":
        _v1 = None if r1 in (None, "none", "null", "None") else float(r1)
        trainer._current_target_ratio_l1 = lambda _v=_v1: _v
    eff_l0 = trainer._current_target_ratio()
    eff_l1 = trainer._current_target_ratio_l1()
    print(f"\nRETENTION IN EFFECT: l0={eff_l0}  l1={eff_l1}"
          f"   (l1=None bypasses the bridge AND L1 entirely)")

    trainer.encoder.eval()
    trainer.decoder_qwen.eval()
    trainer.decoder_falcon.eval()

    all_reps, all_zero, all_tgt, all_src_hit = [], [], [], []
    done = 0
    with torch.no_grad():
        for batch in trainer.eval_dataloader:
            if done >= n_batches:
                break
            try:
                ce_r, pos_r, tgt = _score_batch(
                    trainer, batch, family, zero_reps=False,
                    rescale=bool(probe.get('rescale_reps', False)),
                )
                ce_z, pos_z, _ = _score_batch(trainer, batch, family, zero_reps=True)
            except Exception as exc:  # a probe must not be the thing that fails
                print(f"  batch skipped ({type(exc).__name__}: {str(exc)[:100]})")
                continue
            if ce_r.numel() == 0 or ce_r.numel() != ce_z.numel():
                continue
            if not torch.equal(pos_r, pos_z):
                # Same masks in both arms; if this ever trips the two CE
                # vectors are not aligned and the gap would be meaningless.
                print("  batch skipped (position mismatch between arms)")
                continue
            all_reps.append(ce_r)
            all_zero.append(ce_z)
            all_tgt.append(tgt)
            src = torch.as_tensor(
                [t for docs in batch["source_docs"] for d in docs for t in d],
                dtype=torch.long,
            )
            all_src_hit.append(split_by_source_overlap(tgt, src))
            done += 1

    if not all_reps:
        raise SystemExit("no batches scored")

    ce_r = torch.cat(all_reps)
    ce_z = torch.cat(all_zero)
    stats = summarize_gap(ce_r, ce_z)

    print(f"\n=== SUMMARIZATION REP DEPENDENCE — {Path(str(ckpt)).name} ===")
    print(f"family={family}  batches={done}  l0={eff_l0}  l1={eff_l1}  "
          f"rescale_reps={bool(probe.get('rescale_reps', False))}\n")
    print(stats.render())

    hit = torch.cat(all_src_hit)
    if bool(hit.any()) and bool((~hit).any()):
        in_src = summarize_gap(ce_r[hit], ce_z[hit])
        no_src = summarize_gap(ce_r[~hit], ce_z[~hit])
        print("SECONDARY (heuristic — target-token id also occurs in the source;")
        print("weak proxy: function words appear in both, paraphrase in neither)")
        print(f"  in-source     n={in_src.n_tokens:<7} gap_mean={in_src.gap_mean:+.4f} "
              f"p99={in_src.gap_p99:+.4f}")
        print(f"  not-in-source n={no_src.n_tokens:<7} gap_mean={no_src.gap_mean:+.4f} "
              f"p99={no_src.gap_p99:+.4f}")

    print("\nA thin tail (p99 ~ mean ~ 0) means the reps were never load-bearing")
    print("on this task — nothing decayed, and the Phase-2 framing needs rewriting.")
    print("A heavy tail means they ARE, for a content-token minority that the")
    print("pooled rep_gain metric averages away.")

    out_path = probe.get("out")
    if out_path:
        Path(str(out_path)).write_text(json.dumps(stats.__dict__, indent=2, default=str))
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
