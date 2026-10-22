#!/usr/bin/env python
"""Stock-model BABILong baseline (the "before" numbers for the headline benchmark).

Runs a plain HF causal LM (default Qwen/Qwen3.5-0.8B) over the local
``babilong-1k-samples`` parquet splits using the BABILong repo's own prompt
templates and ``compare_answers`` metric (docs/05_benchmark_targets.md §2).
Two arms per length: ``full`` (whole context if it fits ``--max-context``)
and ``truncate`` (keep the LAST ``--max-context`` tokens of the context —
the naive budget baseline bgkit competes against).

Usage (inside the GPU container):
    python scripts/baseline_babilong.py \
      --data-dir /workspace/capability_packaging/benchmarks/babilong/babilong-1k-samples \
      --repo-dir /workspace/capability_packaging/benchmarks/babilong/babilong-repo \
      --out-dir /workspace/checkpoints_fast/baselines/babilong_qwen35_0p8b \
      --tasks qa1 qa2 qa3 --lengths 0k 1k 2k 4k 8k 16k 32k --n-samples 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--repo-dir", required=True, help="babilong repo (for prompts/metrics)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", nargs="+", default=["qa1", "qa2", "qa3"])
    ap.add_argument("--lengths", nargs="+", default=["0k", "1k", "2k", "4k", "8k", "16k", "32k"])
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--max-context", type=int, default=32_768, help="token budget for the arms")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--use-chat-template", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.repo_dir)
    from babilong.metrics import TASK_LABELS, compare_answers
    from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        .cuda()
        .eval()
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for task in args.tasks:
        cfg = DEFAULT_PROMPTS[task]
        for length in args.lengths:
            path = Path(args.data_dir) / length / f"{task}-00000-of-00001.parquet"
            if not path.exists():
                print(f"skip missing {path}")
                continue
            rows = pq.read_table(path).to_pylist()[: args.n_samples]
            arm_results = {"full": [], "truncate": []}
            t0 = time.time()
            for row in rows:
                ctx_ids = tok.encode(row["input"], add_special_tokens=False)
                fits = len(ctx_ids) <= args.max_context
                for arm in ("full", "truncate"):
                    if arm == "full" and not fits:
                        arm_results[arm].append(None)  # cannot run: doesn't fit
                        continue
                    ctx = row["input"] if fits else tok.decode(ctx_ids[-args.max_context :])
                    prompt = get_formatted_input(
                        ctx,
                        row["question"],
                        cfg["examples"],
                        cfg["instruction"],
                        cfg["post_prompt"],
                        template=DEFAULT_TEMPLATE,
                    )
                    if args.use_chat_template:
                        prompt = tok.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    ids = tok(
                        prompt, return_tensors="pt", add_special_tokens=not args.use_chat_template
                    )
                    ids = {k: v.cuda() for k, v in ids.items()}
                    with torch.no_grad():
                        gen = model.generate(
                            **ids,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            pad_token_id=tok.eos_token_id,
                        )
                    output = tok.decode(
                        gen[0, ids["input_ids"].shape[1] :], skip_special_tokens=True
                    )
                    ok = compare_answers(row["target"], output, row["question"], TASK_LABELS[task])
                    arm_results[arm].append(bool(ok))
                    if arm == "full" and fits:
                        # truncate arm is identical when it fits — reuse
                        arm_results["truncate"].append(bool(ok))
                        break
            key = f"{task}/{length}"
            summary[key] = {}
            for arm, res in arm_results.items():
                valid = [r for r in res if r is not None]
                summary[key][arm] = {
                    "n": len(valid),
                    "acc": (sum(valid) / len(valid)) if valid else None,
                }
            summary[key]["seconds"] = round(time.time() - t0, 1)
            print(key, json.dumps(summary[key]), flush=True)
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("DONE ->", out_dir / "summary.json")


if __name__ == "__main__":
    main()
