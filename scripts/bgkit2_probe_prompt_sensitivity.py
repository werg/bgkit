#!/usr/bin/env python
"""How much does the compression prompt change what the encoder emits?

For rows of a multi-question store (a document with several questions), encode
the SAME document under two sibling prompts at a ratio and compare:
  jaccard    overlap of the selected positions
  cos_shared mean cosine between the reps at positions selected under both
  rel_diff   ||reps_A - reps_B|| / ||reps_A|| at shared positions
A prompt-blind encoder gives 1.0 / 1.0 / 0.0. CPU is enough for a 350M
encoder (run with CUDA_VISIBLE_DEVICES= next to a live trainer).

  scripts/bgkit2_probe_prompt_sensitivity.py --experiment bgkit2_p19n_paired_contrast \
      --checkpoint <dir> [--checkpoint <dir> ...] --store <parquet> --n 12 --ratios 0.25,0.015625
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--checkpoint", action="append", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ratios", default="0.25,0.015625")
    ap.add_argument("--max-ctx-tokens", type=int, default=2048)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir

    from bgkit.training.checkpointing import load_checkpoint
    from bgkit2.data.tasks import ChatRenderer, TaskDataset
    from bgkit2.models.loading import load_tokenizer
    from bgkit2.training.bgkit2_trainer import bgkit2_config_from_hydra
    from bgkit2.training.core import BgKIT2Core

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[f"+experiment={args.experiment}"])
    cfg2 = bgkit2_config_from_hydra(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(cfg2.model.decoder_id)
    from bgkit2.data.templates import set_no_think_template
    if cfg2.model.decoder_no_think:
        set_no_think_template(tok)
    renderer = ChatRenderer.from_tokenizer(tok)
    ds = TaskDataset(
        args.store,
        renderer,
        split=args.split,
        max_ctx_tokens=args.max_ctx_tokens,
        sibling_prompts=True,
        seed=10_009,
    )
    rows = []
    i = 0
    while len(rows) < args.n and i < min(ds.n, 50 * args.n):
        s = ds[i]
        i += 1
        if s.sibling is not None and s.prompt_ids is not None:
            rows.append(s)
    ratios = [float(r) for r in args.ratios.split(",")]
    report: dict = {
        "experiment": args.experiment,
        "store": args.store,
        "n": len(rows),
        "checkpoints": {},
    }
    for ck in args.checkpoint:
        core = BgKIT2Core()
        core.cfg2, core.device = cfg2, device
        core.amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        core.tok = tok
        core.encoder = core.build_encoder()
        core.decoder = core.build_decoder()
        _meta, state = load_checkpoint(Path(ck))
        core.resolve_encoder_adapter(state["encoder"])  # served as trained
        core.finish_models()
        core.load_encoder_state(core.encoder, state["encoder"])
        core.encoder.eval()
        core.encoder.to(device)
        per_ratio: dict = {}
        for r in ratios:
            jac, cos, rel = [], [], []
            for s in rows:
                ids = s.ctx_ids[None].to(device)
                mask = torch.ones_like(ids)
                outs = []
                for p in (s.prompt_ids, s.sibling.prompt_ids):
                    pid = p[None].to(device)
                    with torch.no_grad(), core.autocast():
                        o = core.encoder(
                            ids, mask, ratio=r, prompt_ids=pid, prompt_mask=torch.ones_like(pid)
                        )
                        o = core.finish_reps(ids, o)
                    outs.append(o)
                a, b = outs
                pa = a.positions[0][a.rep_mask[0].bool()].tolist()
                pb = b.positions[0][b.rep_mask[0].bool()].tolist()
                sa, sb = set(pa), set(pb)
                jac.append(len(sa & sb) / max(1, len(sa | sb)))
                shared = sorted(sa & sb)
                if shared:
                    ia = {p: k for k, p in enumerate(pa)}
                    ib = {p: k for k, p in enumerate(pb)}
                    ra = torch.stack([a.reps[0, ia[p]].float() for p in shared])
                    rb = torch.stack([b.reps[0, ib[p]].float() for p in shared])
                    cos.append(float(torch.nn.functional.cosine_similarity(ra, rb, dim=-1).mean()))
                    rel.append(
                        float(((ra - rb).norm(dim=-1) / ra.norm(dim=-1).clamp_min(1e-6)).mean())
                    )
            per_ratio[f"x{round(1 / r)}"] = {
                "jaccard": sum(jac) / len(jac),
                "cos_shared": sum(cos) / max(1, len(cos)),
                "rel_diff": sum(rel) / max(1, len(rel)),
            }
            print(
                f"[{Path(ck).name[-40:]}] x{round(1 / r)}: "
                + json.dumps(per_ratio[f"x{round(1 / r)}"]),
                flush=True,
            )
        report["checkpoints"][ck] = per_ratio
        del core
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
