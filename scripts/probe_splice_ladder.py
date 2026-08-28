#!/usr/bin/env python
"""Is the SPLICE CHANNEL usable at all, or is only the compression bad?

At the end of a full 2600-step run, zeroing the reps changed nothing:
exact_match 0.164 -> 0.160, token_f1 0.477 -> 0.474, loss 0.428 -> 0.429. A
weakly-informative representation would still cost something when deleted. A
delta of ~0 across all three is the signature of a channel the decoder is not
reading.

Everything checked so far says the plumbing is nominally intact:
  - the zeroed ablation really does zero (torch.zeros_like on the splice path)
  - the splice does not detach, so decoder loss can reach survivor VALUES
  - decoder gradient reaches the encoder (l0/l1_content_grad_norm 0.062/0.0149)

So the question is no longer "is a wire cut" but "can the decoder use ANY
embedding spliced at that position?" No experiment run so far separates:

    (a) the splice channel is broken   — position, attention, or scale means
        spliced embeddings are unreadable no matter what they contain; versus
    (b) the compression is bad         — the channel works, but L0/L1 put the
        wrong content into it.

THE LADDER. Identical decoder, identical prompt, only the spliced payload
changes:

  rung 0  ZEROS              floor
  rung 1  REAL reps          what training produces
  rung 2  GOLD-SPAN token embeddings, spliced through the SAME sentinel
          (embed_tokens of the answer's own tokens — no encoder, no
          compression, no projection)
  rung 3  GOLD-SPAN as literal TEXT in the prompt (no splice at all)

Reading:
  rung 2 ~ rung 0  -> THE SPLICE CHANNEL IS BROKEN. The decoder cannot read
                      embeddings at that position even when they are the
                      literal answer. Fixing selection/compression is futile;
                      the bug is in splice position, attention, or scale.
  rung 2 >> rung 1  -> the channel works and carries the answer when handed it
                      verbatim; the compression pipeline is what fails.
  rung 3 >> rung 2  -> the channel is lossy even for perfect content: spliced
                      embeddings are worse than the same tokens as text, which
                      indicts the splice representation itself.

Scored by teacher-forced CE and token accuracy over the ANSWER SPAN ONLY, so
no generation/parsing confounds enter.

Usage (GPU container, no trainer running):
    python scripts/probe_splice_ladder.py +experiment=phase2_kb_widenet_v6 \\
      +probe.checkpoint=/workspace/checkpoints/<ckpt> +probe.n_samples=48
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
from bgkit.utils.logging import setup_logging


def _answer_metrics(trainer: KRKBTrainer, sample) -> tuple[float, float] | None:
    """Teacher-forced (CE, token-accuracy) over the answer span only."""
    try:
        out = trainer._eval_one_sample(sample)
    except Exception:
        return None
    if not isinstance(out, dict):
        return None
    # _eval_one_sample returns loss over ALL loss-bearing positions plus
    # answer_correct/answer_tokens restricted to the answer turn. Use the
    # ANSWER-restricted accuracy and EM: pooled loss is dominated by the
    # tool-call scaffold, which is rep-independent by construction and would
    # mask exactly the effect being measured.
    ce = out.get("loss")
    n = out.get("answer_tokens") or 0
    if ce is None or not n:
        return None
    acc = float(out.get("answer_correct") or 0) / float(n)
    return float(ce), acc


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    pcfg = cfg.get("probe", {}) or {}
    ckpt = pcfg.get("checkpoint")
    n_samples = int(pcfg.get("n_samples", 48))
    if not ckpt:
        raise SystemExit("pass +probe.checkpoint=<path>")

    trainer = KRKBTrainer(cfg)
    trainer.setup()
    from bgkit.training.checkpointing import load_checkpoint as _load

    _meta, state = _load(Path(str(ckpt)))
    trainer._restore_model_state(state)
    trainer.model.eval()

    # self.decoder is None in round-robin (dict) mode; fall back to the
    # ModuleDict's active family.
    dec = getattr(trainer, "decoder", None)
    if dec is None:
        dec = next(iter(trainer.decoders.values()))
    embed = dec.backbone.get_input_embeddings()

    # RANDOM-SELECTION CONTROL. bgkit's survivors are taken from the post-norm
    # LAST-block output, and the head fires mid-backbone so the remaining blocks
    # consolidate under the survival signal — i.e. every survivor has attended
    # over the whole document. The representation is DISTRIBUTED, so WHICH
    # positions are kept should be an information-density knob, not the
    # load-bearing mechanism, and random selection at the same budget should do
    # nearly as well.
    #
    # If that holds, gold-span survival (and the span_relevance_weight
    # supervision built on it) is measuring the wrong thing, and the whole
    # L0-vs-L1 span-discrimination thread is a red herring.
    #
    # Implemented by making the selector heads emit random scores: exact_topk
    # then picks uniformly at random at the SAME budget, with everything else —
    # backbone, bridge, projection, splice — untouched.
    def _randomize_heads(enable: bool) -> None:
        for level in ("l0", "l1"):
            lc = getattr(trainer.encoder, level, None)
            head = getattr(lc, "head", None)
            if head is None:
                continue
            if enable:
                if not hasattr(head, "_orig_forward"):
                    head._orig_forward = head.forward
                head.forward = (  # type: ignore[method-assign]
                    lambda hs, _h=head: torch.randn(
                        hs.shape[:-1], device=hs.device, dtype=torch.float32
                    ).squeeze(0)
                )
            elif hasattr(head, "_orig_forward"):
                head.forward = head._orig_forward  # type: ignore[method-assign]

    real_run_l1 = trainer._run_l1_batch
    mode = {"rung": "real"}

    def patched(prepared, target_ratio=None):
        survivors = real_run_l1(prepared, target_ratio=target_ratio)
        if mode["rung"] == "real":
            return survivors
        if mode["rung"] == "zeros":
            return [torch.zeros_like(s) for s in survivors]
        if mode["rung"] == "real_rescaled":
            # THE MANIFOLD TEST. Gold token embeddings work (acc 0.626); the
            # same splice with real reps does not (0.271 vs 0.266 for zeros).
            # The reps sit at ~139x the token-embedding norm in a projected
            # space, and CLAUDE.md documents Phase-1 Step 2.5 as an entire
            # repair stage for exactly this "large-norm / orthogonal-direction
            # drift between the projection output and the decoder's
            # token-embedding manifold". If simply matching the norm recovers
            # accuracy, the drift is the bug and projection_block is what needs
            # repair — not selection, not the bridge, not the data.
            tgt = embed.weight.detach().norm(dim=-1).mean()
            out = []
            for sv in survivors:
                cur = sv.detach().norm(dim=-1).mean().clamp(min=1e-6)
                out.append((sv * (tgt / cur)).to(sv.dtype))
            return out
        if mode["rung"] == "gold_tokens":
            ids = mode.get("gold_ids")
            if not ids:
                return [torch.zeros_like(s) for s in survivors]
            vec = embed(torch.tensor(ids, dtype=torch.long, device=trainer.device))
            vec = vec.to(survivors[0].dtype)
            # Same count contract as a real splice: one payload per turn.
            return [vec] + [torch.zeros_like(s) for s in survivors[1:]]
        return survivors

    trainer._run_l1_batch = patched  # type: ignore[method-assign]

    ds = trainer.eval_dataset
    res: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    used = 0
    with torch.no_grad():
        for i in range(len(ds)):
            if used >= n_samples:
                break
            sample = ds[i]
            gold = str(getattr(sample, "gold_answer", "") or "")
            if not gold:
                continue
            gold_ids = trainer.tokenizer.encode(gold, add_special_tokens=False)
            if not gold_ids:
                continue
            ok = True
            for rung in ("zeros", "real", "random_sel", "real_rescaled", "gold_tokens"):
                mode["rung"] = "real" if rung == "random_sel" else rung
                mode["gold_ids"] = gold_ids
                _randomize_heads(rung == "random_sel")
                m = _answer_metrics(trainer, sample)
                _randomize_heads(False)
                if m is None:
                    ok = False
                    break
                res[rung]["ce"].append(m[0])
                res[rung]["acc"].append(m[1])
            if ok:
                used += 1

    mode["rung"] = "real"
    print(f"\nSPLICE LADDER — n={used} samples, answer span only")
    print(f"{'rung':<26}{'CE':>9}{'tok_acc':>10}")
    order = [("zeros", "0 zeros (floor)"),
             ("random_sel", "1a RANDOM selection"),
             ("real", "1b trained selection"),
             ("real_rescaled", "1c real reps @ embed norm"),
             ("gold_tokens", "2 gold-span token embeds")]
    base_acc = None
    for key, label in order:
        ce = res[key]["ce"]
        acc = res[key]["acc"]
        if not ce:
            print(f"{label:<26}{'-':>9}{'-':>10}")
            continue
        mce, macc = sum(ce) / len(ce), sum(acc) / len(acc)
        print(f"{label:<26}{mce:9.4f}{macc:10.4f}")
        if key == "zeros":
            base_acc = macc
    print()
    # Does selection matter at all? The distributed-representation claim.
    if res["random_sel"]["acc"] and res["real"]["acc"]:
        rnd = sum(res["random_sel"]["acc"]) / len(res["random_sel"]["acc"])
        tr = sum(res["real"]["acc"]) / len(res["real"]["acc"])
        print(f"SELECTION: trained {tr:.4f} vs random {rnd:.4f}  (delta {tr - rnd:+.4f})")
        if abs(tr - rnd) < 0.02:
            print("  -> selection is NOT load-bearing: random does as well. The reps")
            print("     are distributed, so gold-span survival and the")
            print("     span_relevance_weight supervision measure the wrong thing.")
        else:
            print("  -> selection DOES matter at this budget.")
        print()
    if res["gold_tokens"]["acc"] and base_acc is not None:
        g = sum(res["gold_tokens"]["acc"]) / len(res["gold_tokens"]["acc"])
        r = (sum(res["real"]["acc"]) / len(res["real"]["acc"])) if res["real"]["acc"] else 0.0
        print(f"gold-vs-zeros token-acc lift: {g - base_acc:+.4f}")
        print(f"real-vs-zeros token-acc lift: {r - base_acc:+.4f}")
        print()
        if g - base_acc < 0.02:
            print("VERDICT: THE SPLICE CHANNEL IS BROKEN. Handing the decoder the")
            print("  literal answer tokens at the splice position does not help it.")
            print("  Selection and compression work cannot fix this — the bug is in")
            print("  splice position, attention, or embedding scale.")
        elif g - base_acc > 3 * max(r - base_acc, 1e-6):
            print("VERDICT: the channel WORKS; the compression pipeline is what fails.")
            print("  Perfect content through the same splice is far better than what")
            print("  L0/L1 produce, so effort belongs in selection/compression.")
        else:
            print("VERDICT: channel and compression are comparably limited — the")
            print("  splice carries little even with perfect content.")
    print("\nLADDER JSON", json.dumps(
        {k: {m: (sum(v) / len(v) if v else None) for m, v in d.items()}
         for k, d in res.items()}, indent=2))


if __name__ == "__main__":
    main()
