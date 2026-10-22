#!/usr/bin/env python
"""Matched ablations of ONE bgkit2 checkpoint on the task eval sets: the same
rows, the same decoder (adapter included), the same budget — only the way the
rep slot is filled changes.

  learned    the checkpoint as trained (head-selected positions, encoder reps)
  random     random positions at the same budget (selection="random")
  noprompt   no compression prompt to the encoder
  wrongprompt the next row's prompt (another question about the same
             document on multi-question stores) — a model that uses the
             prompt must lose here even where noprompt ties
  pruning    token EMBEDDINGS at the head-selected positions instead of reps
             (model.rep_token_residual_only) — the P15b/P18 baseline arm

Every arm reports the full ladder task_eval_metrics produces (noctx / x1 / x4
/ x16 / full_text / gen_x4 ...), so the noctx and full_text rows are shared and
only the rep arms differ.

Usage (GPU container, no trainer running):
  scripts/bgkit2_eval_ablations.py --checkpoint /workspace/checkpoints/<ckpt> \
      --experiment bgkit2_p19f_chunks_diverse --arms learned,random,noprompt,pruning \
      --out /workspace/bgkit2_data/eval/ablations_<tag>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARMS = ("learned", "random", "noprompt", "wrongprompt", "pruning")


def build_core(cfg, checkpoint: Path, device: torch.device):
    from bgkit.training.checkpointing import load_checkpoint
    from bgkit2.models.loading import load_tokenizer
    from bgkit2.training.bgkit2_trainer import bgkit2_config_from_hydra
    from bgkit2.training.core import BgKIT2Core

    core = BgKIT2Core()
    core.cfg2 = bgkit2_config_from_hydra(cfg)
    core.device = device
    core.amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    core.tok = load_tokenizer(core.cfg2.model.decoder_id)
    from bgkit2.data.templates import set_no_think_template
    if core.cfg2.model.decoder_no_think:
        set_no_think_template(core.tok)
    core.encoder = core.build_encoder()
    core.decoder = core.build_decoder()
    _meta, state = load_checkpoint(checkpoint)
    core.resolve_encoder_adapter(state["encoder"])  # served as trained
    core.finish_models()
    core.load_encoder_state(core.encoder, state["encoder"])
    if "decoder" in state:
        core.decoder.load_state_dict(state["decoder"])
    core.encoder.eval()
    core.decoder.eval()
    from bgkit2.data.autoencode import Collator
    from bgkit2.data.templates import Templates

    # Mirrors BgKIT2Trainer.setup: the trainer owns these on the GPU path.
    core.templates = Templates.from_tokenizer(core.tok, style=core.cfg2.data.prompt_style)
    core.collate = Collator(core.templates)
    core.select_rng = __import__("random").Random(0)
    core.ratio_schedule = core.build_ratio_schedule(0)
    return core


def set_arm(core, arm: str) -> None:
    core.eval_selection = "random" if arm == "random" else None
    core.eval_no_prompt = arm == "noprompt"
    core.eval_wrong_prompt = arm == "wrongprompt"
    core.cfg2.model.rep_token_residual_only = arm == "pruning"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--stores", default="", help="comma list of store stems; default all")
    ap.add_argument(
        "--task-eval-samples",
        type=int,
        default=None,
        help="rows per eval store (overrides the config; e.g. 384 for babilong = 32 per family)",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[f"+experiment={args.experiment}"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = build_core(cfg, Path(args.checkpoint), device)
    if args.task_eval_samples:
        core.cfg2.data.task_eval_samples = int(args.task_eval_samples)
    # The trainer's in-process eval may restrict generation to a few families
    # and skip stores (eval budget); the pause ablation scores everything.
    core.cfg2.data.task_gen_families = []
    core.cfg2.data.train_eval_skip_stores = []
    _train, _evald, task_eval = core.build_datasets(seed=0)
    wanted = [s for s in args.stores.split(",") if s]
    stores = {k: v for k, v in task_eval.items() if not wanted or k in wanted}
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a}; choose from {ARMS}")
    report: dict = {"checkpoint": args.checkpoint, "experiment": args.experiment, "arms": {}}
    t0 = time.time()
    for arm in arms:
        set_arm(core, arm)
        report["arms"][arm] = {}
        for name, ds in stores.items():
            with torch.no_grad():
                core.last_gen_examples = []
                m = core.task_eval_metrics(ds)
            report["arms"][arm][name] = m
            if getattr(core, "last_gen_examples", None):
                report.setdefault("examples", {}).setdefault(arm, {})[name] = core.last_gen_examples
            keys = [k for k in ("noctx/loss", "x4/loss", "x16/loss", "full_text/loss") if k in m]
            print(
                f"[{arm}] {name}: "
                + " ".join(f"{k.split('/')[0]} {m[k]:.3f}" for k in keys)
                + " ".join(
                    f" {k} {v:.3f}"
                    for k, v in m.items()
                    if k.startswith("gen_x4/") and k.endswith(("exact_match", "token_f1"))
                )
                + f"  ({time.time() - t0:.0f}s)",
                flush=True,
            )
    set_arm(core, "learned")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"out": args.out, "arms": arms, "stores": list(stores)}))


if __name__ == "__main__":
    main()
