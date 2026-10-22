#!/usr/bin/env python
"""Evaluate a bgkit2 checkpoint on BenchPress (arXiv 2510.20797) — their data,
their templates, their metrics, their teacher-normalised score.

Arms, all with the SAME decoder (the checkpoint's rep-gated adapter is
active only at slot positions, so ``full`` and ``noctx`` are the base model):
  full      the passage as text inside the template (BenchPress's teacher;
            the UNADAPTED base reading the full passage)
  fullslot  the passage's token embeddings IN THE SLOT — the adapter reads
            them exactly as it reads reps: the matched ceiling for the reps
            arm (the trainer's full_text arm)
  noctx     the template with the passage removed (the floor)
  bgkit     the passage compressed at ratio r into reps spliced where the
            template's context placeholder sits (encoder prompt = the question)

Usage (inside the GPU container; see compose service eval-bgkit2-benchpress):
  scripts/bgkit2_eval_benchpress.py --checkpoint /workspace/checkpoints_fast/<ckpt_dir> \
      --experiment bgkit2_p1_prompted --datasets short --n-per-dataset 200 \
      --ratios 4,16,128 --out /workspace/bgkit2_data/eval/benchpress_<run>.json
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
for cand in (
    Path("/workspace/capability_packaging/benchmarks/benchpress/src"),
    Path("/home/werg/bgkit-data-nvme/capability_packaging/benchmarks/benchpress/src"),
):
    if cand.exists():
        sys.path.insert(0, str(cand))
        break

from bgkit2.data.templates import SPLICE_SENTINEL, set_no_think_template

DEFAULT_DATA = (
    "/workspace/bgkit2_data/benchmarks/benchpress_data"
    if Path("/workspace/bgkit2_data").exists()
    else "/home/werg/bgkit-data-nvme/bgkit2/benchmarks/benchpress_data"
)


def split_template(template: str, question: str) -> tuple[str, str]:
    """(text before the context placeholder, text after it) with the question
    filled in and the answer placeholder removed."""
    from benchpress.templates import ANSWER_PLACEHOLDER, CONTEXT_PLACEHOLDER, QUERY_PLACEHOLDER

    if template.count(CONTEXT_PLACEHOLDER) != 1:
        raise ValueError("template must contain the context placeholder exactly once")
    pre, post = template.split(CONTEXT_PLACEHOLDER)
    fill = lambda s: s.replace(QUERY_PLACEHOLDER, question).replace(ANSWER_PLACEHOLDER, "")  # noqa: E731
    return fill(pre), fill(post)


def user_content(arm: str, template: str, context: str, question: str) -> str:
    pre, post = split_template(template, question)
    if arm == "full":
        return pre + context + post
    if arm == "noctx":
        return pre + post
    if arm in ("bgkit", "fullslot"):
        return pre + SPLICE_SENTINEL + post
    raise ValueError(arm)


class Evaluator:
    def __init__(self, cfg, checkpoint: Path, device: torch.device) -> None:
        from bgkit.training.checkpointing import load_checkpoint
        from bgkit2.data.tasks import ChatRenderer
        from bgkit2.models.loading import load_tokenizer
        from bgkit2.training.bgkit2_trainer import bgkit2_config_from_hydra
        from bgkit2.training.core import BgKIT2Core

        self.cfg2 = bgkit2_config_from_hydra(cfg)
        self.device = device
        self.amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        core = BgKIT2Core()
        core.cfg2, core.device, core.amp_dtype = self.cfg2, device, self.amp_dtype
        self.tok = load_tokenizer(self.cfg2.model.decoder_id)
        if self.cfg2.model.decoder_no_think:
            set_no_think_template(self.tok)
        core.tok = self.tok
        self.encoder = core.build_encoder()
        self.decoder = core.build_decoder()
        core.encoder, core.decoder = self.encoder, self.decoder
        _meta, state = load_checkpoint(checkpoint)
        core.resolve_encoder_adapter(state["encoder"])  # served as trained
        core.finish_models()
        core.load_encoder_state(self.encoder, state["encoder"])
        if "decoder" in state:
            self.decoder.load_state_dict(state["decoder"])
        self.encoder.eval()
        self.decoder.eval()
        self.renderer = ChatRenderer.from_tokenizer(self.tok)
        self.core = core

    def _prompt_ids(self, content: str) -> tuple[torch.Tensor, torch.Tensor | None]:
        ids = self.renderer._apply([{"role": "user", "content": content}])
        if SPLICE_SENTINEL in content:
            prefix, instr = self.renderer._split(ids)
            return prefix, instr
        return torch.tensor(ids, dtype=torch.long), None

    @torch.no_grad()
    def generate(self, items: list[dict], arm: str, ratio: float | None, max_new: int) -> list[str]:
        prefixes, instrs, ctx_ids, prompts = [], [], [], []
        for it in items:
            content = user_content(arm, it["template"], it["context"], it["question"])
            p, i = self._prompt_ids(content)
            prefixes.append(p)
            instrs.append(i if i is not None else torch.zeros(0, dtype=torch.long))
            ctx_ids.append(self.renderer.encode(it["context"], self.cfg2.data.task_max_ctx_tokens))
            prompts.append(
                self.renderer.encode(f"Compress the passage to answer: {it['question']}\n")
            )
        reps = rep_mask = None
        if arm == "fullslot":
            lmax = max(c.shape[0] for c in ctx_ids)
            ids = torch.zeros(len(items), lmax, dtype=torch.long)
            mask = torch.zeros(len(items), lmax, dtype=torch.long)
            for k, c in enumerate(ctx_ids):
                ids[k, : c.shape[0]] = c
                mask[k, : c.shape[0]] = 1
            with torch.no_grad():
                reps = self.decoder.embed_tokens(ids.to(self.device))
            rep_mask = mask.to(self.device)
        if arm == "bgkit":
            lmax = max(c.shape[0] for c in ctx_ids)
            pmax = max(p.shape[0] for p in prompts)
            ids = torch.zeros(len(items), lmax, dtype=torch.long)
            mask = torch.zeros(len(items), lmax, dtype=torch.long)
            pid = torch.zeros(len(items), pmax, dtype=torch.long)
            pmask = torch.zeros(len(items), pmax, dtype=torch.long)
            for k, (c, p) in enumerate(zip(ctx_ids, prompts, strict=True)):
                ids[k, : c.shape[0]] = c
                mask[k, : c.shape[0]] = 1
                pid[k, pmax - p.shape[0] :] = p
                pmask[k, pmax - p.shape[0] :] = 1
            with self.core.autocast():
                out = self.encoder(
                    ids.to(self.device),
                    mask.to(self.device),
                    ratio=ratio,
                    prompt_ids=pid.to(self.device),
                    prompt_mask=pmask.to(self.device),
                )
                out = self.core.finish_reps(ids.to(self.device), out)
            reps, rep_mask = out.reps, out.rep_mask
        with self.core.autocast():
            batch = self.decoder.build_batch(
                prefixes, reps, rep_mask, instrs, pad_side="left", loss_on_suffix=False
            )
            gen = self.decoder.generate(
                batch, max_new_tokens=max_new, eos_token_id=self.renderer.eos_id
            )
        outs = []
        for k in range(len(items)):
            g = gen[k].tolist()
            if self.renderer.eos_id in g:
                g = g[: g.index(self.renderer.eos_id)]
            g = [t for t in g if t != self.renderer.pad_id]
            outs.append(self.tok.decode(g, skip_special_tokens=True).strip())
        return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--experiment", default="bgkit2_p1_prompted")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--datasets", default="short", help="short | mid_range | comma list")
    ap.add_argument("--n-per-dataset", type=int, default=200)
    ap.add_argument("--max-context-tokens", type=int, default=1024)
    ap.add_argument("--ratios", default="4,16,128")
    ap.add_argument("--arms", default="full,fullslot,noctx,bgkit")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import benchpress
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[f"+experiment={args.experiment}"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ev = Evaluator(cfg, Path(args.checkpoint), device)

    if args.datasets in ("short", "mid_range", "all"):
        ds = benchpress.load(
            path=args.data, subset=args.datasets, max_context_tokens=args.max_context_tokens
        )
    else:
        ds = benchpress.load(
            path=args.data,
            datasets=args.datasets.split(","),
            max_context_tokens=args.max_context_tokens,
        )
    by_name: dict[str, list[dict]] = {}
    for row in ds:
        lst = by_name.setdefault(row["dataset_name"], [])
        if len(lst) >= args.n_per_dataset:
            continue
        lst.append(
            {
                "id": row["id"],
                "context": row["context"],
                "question": row["question"],
                "answers": list(row["answers"]) or [row["answer"]],
                "template": benchpress.sample_template(
                    row["dataset_name"], row["context"], row["question"]
                ),
            }
        )
    ratios = [float(r) for r in args.ratios.split(",")]
    arms = args.arms.split(",")
    report: dict = {"checkpoint": args.checkpoint, "experiment": args.experiment, "datasets": {}}
    t0 = time.time()
    for name, items in by_name.items():
        rep = {"n": len(items), "arms": {}}
        refs = [it["answers"] for it in items]
        runs: list[tuple[str, float | None]] = []
        for arm in arms:
            if arm == "bgkit":
                runs += [("bgkit", 1.0 / r) for r in ratios]
            else:
                runs.append((arm, None))
        for arm, ratio in runs:
            preds: list[str] = []
            for a in range(0, len(items), args.batch_size):
                preds += ev.generate(
                    items[a : a + args.batch_size], arm, ratio, args.max_new_tokens
                )
            scores = benchpress.aggregate(benchpress.evaluate(preds, refs))
            key = arm if ratio is None else f"bgkit_x{round(1 / ratio)}"
            rep["arms"][key] = {
                **scores,
                "examples": [
                    {"pred": p, "gold": r[0]} for p, r in list(zip(preds, refs, strict=True))[:5]
                ],
            }
            print(
                f"[{name}] {key}: F1 {scores['F1']:.3f} EM {scores['EM']:.3f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
        if "full" in rep["arms"] and "noctx" in rep["arms"]:
            for key, sc in rep["arms"].items():
                if key.startswith("bgkit"):
                    sc["teacher_normalized_F1"] = benchpress.teacher_normalized_score(
                        sc["F1"], rep["arms"]["full"]["F1"], rep["arms"]["noctx"]["F1"]
                    )
                    if "fullslot" in rep["arms"]:
                        # normalised against the MATCHED ceiling (same adapter)
                        sc["fullslot_normalized_F1"] = benchpress.teacher_normalized_score(
                            sc["F1"], rep["arms"]["fullslot"]["F1"], rep["arms"]["noctx"]["F1"]
                        )
        report["datasets"][name] = rep
    # macro averages
    macro: dict[str, dict[str, float]] = {}
    for rep in report["datasets"].values():
        for key, sc in rep["arms"].items():
            m = macro.setdefault(key, {"F1": 0.0, "EM": 0.0, "n_datasets": 0})
            m["F1"] += sc["F1"]
            m["EM"] += sc["EM"]
            m["n_datasets"] += 1
    for m in macro.values():
        n = max(m["n_datasets"], 1)
        m["F1"] /= n
        m["EM"] /= n
    report["macro"] = macro
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"macro": macro, "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
