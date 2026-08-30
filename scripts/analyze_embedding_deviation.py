#!/usr/bin/env python
"""How far is the projection output from the decoder's token-embedding manifold?

RESTORED 2026-08-28. This diagnostic existed, caught exactly this failure on
2026-04-17 (it is what motivated Phase-1 Step 2.5, "projection embed-anchor
repair"), and was DELETED in commit ce3afb2 during the FA4 packed-attention
migration. With the guard gone the drift returned unmonitored — the same shape
as the missing no-context dataset gate.

WHY IT MATTERS NOW. The splice ladder (2026-08-28) measured, same decoder, same
prompt, only the spliced payload changing:

    zeros 0.266 | real reps 0.271 | RANDOM selection 0.270 | gold tokens 0.626

So the splice channel is fine — the decoder reads spliced embeddings perfectly
when they are its OWN token embeddings — and the real reps are worth ~1% of
that. Meanwhile a linear probe finds the answer span inside those same reps at
AUC 0.92-0.99. The information is present and linearly accessible; it is simply
not in a form the DECODER can read. The obvious suspect is that the projection
output no longer lives on the decoder's embedding manifold: reps have been
observed at ~139x the token-embedding norm (git-repro's, which reconstruct
fine, sit at ~4x).

WHAT IT DISTINGUISHES. Run across the lineage and the answer to "did training
break it, or was it always broken?" falls out:

  base aligned, Phase-2 drifted  -> Phase-2 training walked the projection off
                                    the manifold. Nothing pushed back because
                                    the objective never required the decoder to
                                    read the reps (57% of loss tokens were a
                                    rep-independent copy task).
  base ALSO drifted              -> the drift predates Phase 2. The
                                    summarization lineage never had Step 2.5
                                    applied, so the projection was never
                                    anchored and Phase 2 inherited it.

Metrics, per checkpoint:
  norm_ratio     mean ||projection|| / mean ||embed_tokens||   (1.0 = matched)
  cos_to_embed   mean cosine between each rep and its NEAREST token embedding
                 (direction: is it pointing anywhere the decoder recognises?)
  cos_random     mean cosine against random token embeddings
  cos_nearest_for_random_vector
                 THE CORRECT FLOOR: the nearest-embedding cosine achieved by a
                 RANDOM direction. cos_near is a MAX over ~150k tokens, so its
                 null must be a max too — a mean-against-random-tokens floor
                 (cos_random) makes any max look impressive. Read
                 nearest_vs_random_ratio: <= 1.0 means no directional alignment

Usage (GPU container, no trainer running):
    python scripts/analyze_embedding_deviation.py +experiment=phase2_kb_widenet_v6 \\
      +diag.checkpoints='[/workspace/checkpoints/A,/workspace/checkpoints/B]' \\
      +diag.n_samples=16
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from bgkit.training.checkpointing import restore_model_state_lenient
from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
from bgkit.utils.logging import setup_logging


def measure(trainer: KRKBTrainer, n_samples: int) -> dict:
    dec = getattr(trainer, "decoder", None)
    if dec is None:
        dec = next(iter(trainer.decoders.values()))
    emb = dec.backbone.get_input_embeddings().weight.detach().float()
    emb_norm = emb.norm(dim=-1)
    emb_unit = emb / emb_norm.clamp(min=1e-6).unsqueeze(-1)
    mean_emb_norm = float(emb_norm.mean().item())

    ds = trainer.eval_dataset
    norms: list[float] = []
    cos_near: list[float] = []
    cos_rand: list[float] = []
    cos_near_rand: list[float] = []
    used = 0
    g = torch.Generator(device="cpu").manual_seed(0)

    with torch.no_grad():
        for i in range(len(ds)):
            if used >= n_samples:
                break
            sample = ds[i]
            try:
                prep = trainer._prepare_sample_for_decode(sample)
                turns = prep["prepared_turns"]
                if not turns or not isinstance(turns[0], dict) or "content" not in turns[0]:
                    continue
                survs = trainer._run_l1_batch(
                    [turns[0]], target_ratio=trainer._drill_leaf_l1_retention_override(),
                )
            except Exception:
                continue
            reps = survs[0].detach().float()
            if reps.numel() == 0 or reps.shape[0] < 2:
                continue
            r_norm = reps.norm(dim=-1)
            norms.extend(r_norm.tolist())
            r_unit = reps / r_norm.clamp(min=1e-6).unsqueeze(-1)
            # Nearest-embedding cosine, in chunks so the vocab matmul fits.
            best = torch.full((reps.shape[0],), -1.0, device=reps.device)
            for s in range(0, emb_unit.shape[0], 32768):
                blk = emb_unit[s : s + 32768].to(reps.device)
                best = torch.maximum(best, (r_unit @ blk.T).max(dim=-1).values)
            cos_near.extend(best.tolist())
            idx = torch.randint(0, emb_unit.shape[0], (reps.shape[0],), generator=g)
            rnd = emb_unit[idx].to(reps.device)
            cos_rand.extend((r_unit * rnd).sum(dim=-1).tolist())

            # THE FLOOR FOR A *NEAREST* STATISTIC MUST ALSO BE A NEAREST
            # STATISTIC. ``cos_to_random_embed`` is a MEAN against random
            # tokens; ``cos_to_nearest_embed`` is a MAX over ~150k of them.
            # Comparing them is comparing a max to a mean, and the max cosine
            # of even a RANDOM direction over a vocabulary that large is well
            # above zero in high dimension. Read against the wrong floor, v8's
            # 0.159 looked like "weak but real alignment" when it may be pure
            # extreme-value noise. So: draw random unit vectors of the same
            # dimension and take THEIR nearest-embedding cosine.
            rv = torch.randn(reps.shape[0], reps.shape[-1], generator=g)
            rv = (rv / rv.norm(dim=-1, keepdim=True).clamp(min=1e-6)).to(reps.device)
            rbest = torch.full((reps.shape[0],), -1.0, device=reps.device)
            for s in range(0, emb_unit.shape[0], 32768):
                blk = emb_unit[s : s + 32768].to(reps.device)
                rbest = torch.maximum(rbest, (rv @ blk.T).max(dim=-1).values)
            cos_near_rand.extend(rbest.tolist())
            used += 1

    if not norms:
        return {"samples": 0}
    mean_rep_norm = sum(norms) / len(norms)
    return {
        "samples": used,
        "n_reps": len(norms),
        "mean_rep_norm": mean_rep_norm,
        "mean_embed_norm": mean_emb_norm,
        "norm_ratio": mean_rep_norm / max(mean_emb_norm, 1e-6),
        "cos_to_nearest_embed": sum(cos_near) / len(cos_near),
        "cos_to_random_embed": sum(cos_rand) / len(cos_rand),
        # The correct floor: nearest-embedding cosine for a RANDOM direction.
        "cos_nearest_for_random_vector": (
            sum(cos_near_rand) / len(cos_near_rand) if cos_near_rand else None
        ),
        # What the number actually means. <= 1.0 means the reps point no closer
        # to any token than chance does: no directional alignment at all.
        "nearest_vs_random_ratio": (
            (sum(cos_near) / len(cos_near))
            / max(sum(cos_near_rand) / len(cos_near_rand), 1e-6)
            if cos_near_rand else None
        ),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    diag = cfg.get("diag", {}) or {}
    ckpts = list(diag.get("checkpoints", []) or [])
    n = int(diag.get("n_samples", 16))
    if not ckpts:
        raise SystemExit("pass +diag.checkpoints='[path1,path2]'")

    trainer = KRKBTrainer(cfg)
    trainer.setup()
    from bgkit.training.checkpointing import load_checkpoint as _load

    results: dict[str, dict] = {}
    print(f"{'checkpoint':<46}{'norm_ratio':>12}{'cos_near':>10}{'cos_rand':>10}")
    for ck in ckpts:
        label = Path(str(ck)).name[:44]
        try:
            _meta, state = _load(Path(str(ck)))
            # Lenient + disclosed: the Phase-1 base predates l1l1_bridge,
            # which is provably off this path (recursive_l1_tree is None).
            restore_model_state_lenient(trainer, state)
        except Exception as exc:
            print(f"{label:<46}  SKIPPED ({type(exc).__name__}: {str(exc)[:120]})")
            results[label] = {"error": type(exc).__name__}
            continue
        trainer.model.eval()
        m = measure(trainer, n)
        results[label] = m
        if not m.get("samples"):
            print(f"{label:<46}  no samples")
            continue
        print(f"{label:<46}{m['norm_ratio']:12.2f}"
              f"{m['cos_to_nearest_embed']:10.3f}{m['cos_to_random_embed']:10.3f}")

    print()
    print("norm_ratio 1.0 = reps match the token-embedding scale the decoder expects.")
    print("cos_near must be read AGAINST cos_rand: in high dimension a cosine only")
    print("means something relative to the random floor.")
    print()
    print("A large norm_ratio with cos_near ~ cos_rand is the Step-2.5 pathology:")
    print("the projection output is both too large AND pointing nowhere the decoder")
    print("recognises. Repair is projection_block (MSE + cosine + log-norm against")
    print("decoder.embed_tokens), NOT selection, NOT the bridge, NOT the data.")
    print("\nDEVIATION JSON", json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
