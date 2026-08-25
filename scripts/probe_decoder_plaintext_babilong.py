#!/usr/bin/env python
"""Can the fine-tuned decoder still answer BABILong in PLAIN TEXT?

The 2026-08-25 BABILong arms scored 0.00 at every length AND at retention
1.0 (no compression, 38-token contexts), with every generation collapsing to
the same degenerate ``babilong: Table_StandReportReceiver-3000-…`` string.
Length, compression and selection are therefore all ruled out as the cause.
What has NOT been ruled out is the decoder itself: the wide-net runs
fine-tuned it hard on code-retrieval trajectories, and its own in-distribution
metrics are shaky (``free_running/invalid_rate`` 0.47, almost all
``unsurfaced_id``).

This probe removes bgkit from the loop entirely. It loads the checkpoint's
``decoders.<family>.backbone.*`` weights into a stock HF Qwen3.5 and runs
BABILong exactly as ``scripts/baseline_babilong.py`` does — context inline as
TOKENS, no encoder, no splice, same prompts, same ``compare_answers``.

- Scores near the stock model (qa1 0k 0.55) → the decoder is fine and the
  failure is in the splice/representation path.
- Scores near zero → the fine-tuning has cost the decoder general
  instruction-following, and no amount of prose training DATA fixes that;
  the training mix is the thing to change.

Usage (GPU container, no trainer running):
    python scripts/probe_decoder_plaintext_babilong.py \
      --checkpoint /workspace/checkpoints_fast/<v6> \
      --data-dir $BL/babilong-1k-samples --repo-dir $BL/babilong-repo \
      --tasks qa1 qa2 --lengths 0k 1k 4k --n-samples 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="phase2 checkpoint dir (model.pt)")
    ap.add_argument("--family", default="qwen35")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--tasks", nargs="+", default=["qa1", "qa2"])
    ap.add_argument("--lengths", nargs="+", default=["0k", "1k", "4k"])
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out-json", default=None)
    ap.add_argument(
        "--stock-too",
        action="store_true",
        help="also run the pristine HF weights in the same process, as the control",
    )
    args = ap.parse_args()

    sys.path.insert(0, args.repo_dir)
    from babilong.metrics import TASK_LABELS, compare_answers
    from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def build(load_ckpt: bool):
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        if load_ckpt:
            sd = torch.load(
                str(Path(args.checkpoint) / "model.pt"),
                map_location="cpu", mmap=True, weights_only=True,
            )
            prefix = f"decoders.{args.family}.backbone."
            trimmed = {
                k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)
            }
            if not trimmed:
                raise SystemExit(f"no keys under {prefix!r}; found {sorted(sd)[:3]}")
            missing, unexpected = model.load_state_dict(trimmed, strict=False)
            print(
                f"loaded {len(trimmed)} tensors from checkpoint "
                f"(missing={len(missing)}, unexpected={len(unexpected)})",
                flush=True,
            )
            if unexpected:
                print("  unexpected sample:", list(unexpected)[:5], flush=True)
            if missing:
                print("  missing sample:", list(missing)[:5], flush=True)
        return model.cuda().eval()

    results: dict[str, dict] = {}
    arms = [("finetuned", True)] + ([("stock", False)] if args.stock_too else [])
    for arm, load_ckpt in arms:
        model = build(load_ckpt)
        for task in args.tasks:
            cfg = DEFAULT_PROMPTS[task]
            for length in args.lengths:
                path = Path(args.data_dir) / length / f"{task}-00000-of-00001.parquet"
                if not path.exists():
                    continue
                rows = pq.read_table(path).to_pylist()[: args.n_samples]
                hits = 0
                examples: list[str] = []
                for row in rows:
                    prompt = get_formatted_input(
                        row["input"], row["question"], cfg["examples"],
                        cfg["instruction"], cfg["post_prompt"], template=DEFAULT_TEMPLATE,
                    )
                    ids = tok(prompt, return_tensors="pt", add_special_tokens=True)
                    ids = {k: v.cuda() for k, v in ids.items()}
                    with torch.no_grad():
                        gen = model.generate(
                            **ids, max_new_tokens=args.max_new_tokens,
                            do_sample=False, pad_token_id=tok.eos_token_id,
                        )
                    out = tok.decode(
                        gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True
                    )
                    hits += bool(
                        compare_answers(row["target"], out, row["question"], TASK_LABELS[task])
                    )
                    if len(examples) < 3:
                        examples.append(out[:120])
                key = f"{arm}/{task}/{length}"
                results[key] = {"n": len(rows), "acc": hits / max(len(rows), 1)}
                print(key, json.dumps(results[key]), "| e.g.", examples, flush=True)
        del model
        torch.cuda.empty_cache()

    print("SUMMARY", json.dumps(results, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
