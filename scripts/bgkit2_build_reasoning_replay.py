#!/usr/bin/env python
"""Build the bgkit2 ``reasoning_replay`` task store (2026-09-26): public reasoning traces as
replay rows (family ``chat_replay``, framing ``chat_think``), so the 8B decoder's thinking
mode is anchored while its LoRA trains in non-thinking mode.

LFM2.5-8B-A1B is a reasoning model. The bgkit stores carry no reasoning at all (every task
row is a direct answer, rendered in non-thinking mode by ``templates.set_no_think_template``),
so without these rows nothing constrains what the always-on LoRA does to the model's
thinking. Each row is one user problem and one assistant turn carrying its reasoning
(``reasoning_content``) and answer; ``ChatRenderer`` renders it in the model's own thinking
format, ``<think>\\n{reasoning}\\n</think>\\n{answer}`` (measured from the base model's own
output), and the replay loss self-distils the turn to the adapter-off base -- the traces
only supply prompt coverage in thinking mode; the target distribution is the base's own.

Source: open-thoughts/OpenThoughts-114k, ``metadata`` config (Apache-2.0; DeepSeek-R1
traces over math / code / science / puzzles; ``deepseek_reasoning`` and ``deepseek_solution``
are separate fields). Rows whose rendered length exceeds ``--max-tokens`` (decoder tokens)
are dropped; domains are balanced by a per-domain cap. The eval split is decided by a hash
of the problem BEFORE sampling, so eval rows are never training rows.

  .venv/bin/python scripts/bgkit2_build_reasoning_replay.py \\
      --shards $B/raw/openthoughts/metadata --out $B/tasks/reasoning_replay_v1.arrow
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgkit2.data.tasks import COLUMNS, ChatRenderer, write_task_store_arrow
from bgkit2.data.templates import set_no_think_template


def _split_of(key: str, eval_frac: float) -> str:
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if h < eval_frac else "train"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="directory of the metadata parquet shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--decoder", default="/home/werg/bgkit-data-nvme/models_hf/LFM2.5-8B-A1B")
    ap.add_argument("--max-tokens", type=int, default=6000, help="rendered prefix + target")
    ap.add_argument("--per-domain", type=int, default=4000)
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.decoder)
    set_no_think_template(tok)  # as the trainer loads it; chat_think rows opt back in
    renderer = ChatRenderer.from_tokenizer(tok)

    rng = random.Random(args.seed)
    by_dom: dict[str, list[dict]] = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{args.shards}/*.parquet")):
        for r in pq.read_table(
            f, columns=["problem", "deepseek_reasoning", "deepseek_solution", "domain", "source"]
        ).to_pylist():
            if r["deepseek_reasoning"] and r["deepseek_solution"] and r["problem"]:
                by_dom[r["domain"] or "other"].append(r)
    stats = collections.Counter()
    rows = []
    for dom, items in sorted(by_dom.items()):
        rng.shuffle(items)
        kept = 0
        for r in items:
            if kept >= args.per_domain:
                break
            instr = [{"role": "user", "content": r["problem"].strip()}]
            turn = [{
                "role": "assistant",
                "content": "\n" + r["deepseek_solution"].strip(),
                "reasoning_content": "\n" + r["deepseek_reasoning"].strip() + "\n",
            }]
            try:
                prefix, _ = renderer.render("chat_think", json.dumps(instr))
                tgt, _mask = renderer.render_chunk(instr, turn, enable_thinking=True)
            except ValueError:
                stats["unrenderable"] += 1
                continue
            n = int(prefix.shape[0] + tgt.shape[0])
            if n > args.max_tokens:
                stats["too_long"] += 1
                continue
            text = tok.decode(tgt)
            if not text.startswith("<think>\n") or "\n</think>\n" not in text:
                raise ValueError(f"not rendered in thinking format: {text[:80]!r}")
            rows.append({
                "family": "chat_replay",
                "framing": "chat_think",
                "instruction": json.dumps(instr),
                "target": json.dumps(turn),
                "split": _split_of(r["problem"], args.eval_frac),
                "meta": json.dumps({"source": f"openthoughts/{r['source']}", "domain": dom,
                                    "tokens": n}),
            })
            kept += 1
            stats[f"kept/{dom}"] += 1
            stats["tokens"] += n
    rng.shuffle(rows)
    cols = {c: [str(r.get(c, "")) for r in rows] for c in COLUMNS}
    write_task_store_arrow(args.out, pa.table({c: pa.array(v, type=pa.string())
                                               for c, v in cols.items()}))
    splits = collections.Counter(r["split"] for r in rows)
    print(json.dumps({"rows": len(rows), "splits": dict(splits),
                      "mean_tokens": round(stats["tokens"] / max(len(rows), 1)),
                      **{k: v for k, v in stats.items() if k != "tokens"}}))


if __name__ == "__main__":
    main()
