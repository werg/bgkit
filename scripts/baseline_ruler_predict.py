#!/usr/bin/env python
"""RULER predictions for a stock model via an OpenAI-compatible vLLM server.

RULER's own clients target its custom ``serve_vllm.py`` endpoint or named
OpenAI models; our ``vllm-fast`` compose service exposes the standard
``/v1/completions`` API, so this thin predictor writes RULER's prediction
jsonl (``index,input,outputs,pred``) for ``scripts/eval/evaluate.py``.

Usage (host venv; vllm-fast must be up on :8091):
    .venv/bin/python scripts/baseline_ruler_predict.py \
      --data-root $NVME/capability_packaging/benchmarks/ruler/data \
      --out-root  $NVME/capability_packaging/benchmarks/ruler/pred_qwen35_0p8b \
      --model Qwen/Qwen3.5-0.8B --base-url http://localhost:8091/v1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


def complete(base_url: str, model: str, prompt: str, max_tokens: int, stop: list[str]) -> str:
    r = requests.post(
        f"{base_url}/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stop": stop or None,
        },
        timeout=600,
    )
    if 400 <= r.status_code < 500:
        # A prompt longer than the server's max_model_len is a REAL failure of
        # the full-context arm (the model cannot see the input) — record an
        # empty prediction (scored wrong) instead of aborting the whole grid.
        print(f"WARN {r.status_code}: {r.text[:160]} -> empty prediction", flush=True)
        return ""
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--base-url", default="http://localhost:8091/v1")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    for val in sorted(data_root.glob("L*/*/validation.jsonl")):
        length_dir, task = val.parent.parent.name, val.parent.name
        out = Path(args.out_root) / length_dir / f"{task}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = 0
        if out.exists():
            done = sum(1 for _ in out.open())
        rows = [json.loads(line) for line in val.open()]
        if args.limit:
            rows = rows[: args.limit]
        if done >= len(rows):
            print(f"skip {length_dir}/{task} (done)")
            continue
        t0 = time.time()
        with out.open("a") as fh:
            for row in rows[done:]:
                # Protocol (2026-08-24, applied uniformly to every task):
                # 1. NO stop strings — greedy Qwen3.5-0.8B opens RULER base-
                #    prompt answers with "\n\n"; stop=["\n\n"] truncated 100%
                #    of predictions to "" at position 0. Metrics are substring
                #    matches, so trailing text is harmless.
                # 2. Thinking DISABLED via the empty-think scaffold appended
                #    to the prompt (Qwen's official non-thinking switch): on
                #    qa tasks the raw prompt sent the model into an unbounded
                #    <think> block (finish=length at 256 with no answer, and
                #    think text can contain spurious ref substrings); with the
                #    scaffold it answers directly ("France.", 64 tokens).
                prompt = row["input"] + "\n\n<think>\n\n</think>\n\n"
                pred = complete(args.base_url, args.model, prompt, args.max_tokens, None)
                fh.write(
                    json.dumps(
                        {
                            "index": row["index"],
                            "input": row["input"][:200],
                            "outputs": row["outputs"],
                            "pred": pred,
                        }
                    )
                    + "\n"
                )
        print(f"{length_dir}/{task}: {len(rows)} preds in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
