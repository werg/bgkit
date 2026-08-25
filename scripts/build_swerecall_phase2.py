#!/usr/bin/env python
"""Build the ``swerecall`` Phase-2 flat dataset from SWE-Zero trajectories.

Family A recall-through-the-bottleneck training in the proven flat schema
(`plans/capability_packaging_2026_08_20.md` §4): each article is a compacted
trajectory *span* (contiguous agent turns rendered as plain text); each
sample asks a recall probe whose answer is a fact that occurs only inside
that span (file path / symbol), or a "not in this span" negative. This is
the probe subset of compaction training with zero format compromise — the
faithful blob-ordering trainer is built separately.

Split = held-out repos (trajectories grouped by their `repo` field).

Usage:
    .venv/bin/python scripts/build_swerecall_phase2.py \
      --swezero-dir /home/werg/bgkit-data-nvme/capability_packaging/trajectories_ext/swe_zero \
      --mmap-out $DATA_DIR/mmap/phase2 --traj-out $DATA_DIR/trajectories
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from bgkit.data.compaction_sampler import (
    build_probe_sample,
    build_sample,
    detect_anchor_len,
    turn_boundaries,
)
from bgkit.data.flat_phase2_writer import flat_trajectory_row, write_flat_phase2_dataset
from bgkit.data.gold_span import answer_span_from_offsets, article_offsets, span_to_json

DATASET_NAME = "swerecall"


def render_span_text(messages: list[dict]) -> str:
    """Plain-text rendering of a message span (what the encoder compresses)."""
    parts = []
    for m in messages:
        role = m.get("role", "?")
        content = str(m.get("content") or "")
        tc = m.get("tool_calls") or []
        call_txt = ""
        if tc:
            names = ",".join(str((c.get("function") or {}).get("name", "")) for c in tc)
            call_txt = f" [tool_calls: {names}]"
        parts.append(f"<{role}>{call_txt}\n{content}")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--swezero-dir", required=True)
    ap.add_argument("--mmap-out", required=True)
    ap.add_argument("--traj-out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--n-shards", type=int, default=2)
    ap.add_argument("--rows-per-shard", type=int, default=1500)
    ap.add_argument("--probes-per-trajectory", type=int, default=3)
    ap.add_argument("--max-span-tokens", type=int, default=45_000)
    ap.add_argument("--eval-repo-fraction", type=float, default=0.08)
    ap.add_argument("--negative-fraction", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rng = random.Random(args.seed)

    shard_paths = sorted(Path(args.swezero_dir, "data").glob("train-*.parquet"))[: args.n_shards]
    doc_ids: list[str] = []
    doc_tokens: list[np.ndarray] = []
    traj_rows: list[dict] = []
    stats: dict[str, int] = {}
    seen_docs: set[str] = set()
    session_counts: dict[str, int] = {}
    fact_pool: list[str] = []

    for shard in shard_paths:
        pf = pq.ParquetFile(shard)
        n_taken = 0
        for batch in pf.iter_batches(batch_size=200):
            for row in batch.to_pylist():
                if n_taken >= args.rows_per_shard:
                    break
                n_taken += 1
                msgs = row["trajectory"]
                msgs = json.loads(msgs) if isinstance(msgs, str) else msgs
                repo = str(row.get("repo") or "unknown")
                digest = int(hashlib.sha256(repo.encode()).hexdigest(), 16)
                split = "eval" if (digest % 1000) < int(args.eval_repo_fraction * 1000) else "train"
                anchor = detect_anchor_len(msgs)
                bounds = turn_boundaries(msgs, anchor)
                if len(bounds) < 4:
                    continue
                # Readable session key (repo + running index) instead of the
                # source UUID: ids are copied into every tool call by the
                # decoder, and a 36-char UUID is ~20 tokens of noise.
                session_counts[repo] = session_counts.get(repo, 0) + 1
                tid = f"{repo}#{session_counts[repo]}"
                for _ in range(args.probes_per_trajectory):
                    target_index = rng.choice(bounds[2:])
                    base = build_sample(
                        msgs,
                        trajectory_id=tid,
                        target_index=target_index,
                        live_window_turns=rng.choice((1, 2)),
                        n_blobs=1,
                        mode="merged",
                    )
                    if base is None:
                        continue
                    probe = build_probe_sample(base, msgs, rng)
                    if probe is None:
                        continue
                    a, b = probe.compacted_span
                    span_text = render_span_text(msgs[a:b])
                    ids = tokenizer.encode(span_text, add_special_tokens=False)
                    if not (512 <= len(ids) <= args.max_span_tokens):
                        continue
                    doc_id = f"swespan:{tid}:{a}-{b}"
                    if doc_id not in seen_docs:
                        seen_docs.add(doc_id)
                        doc_ids.append(doc_id)
                        doc_tokens.append(np.asarray(ids, dtype=np.int32))
                    kind, fact = probe.probe_fact
                    question = probe.prefix_messages[-1]["content"]
                    answer = fact
                    qtype = f"recall_{kind}"
                    # negatives: ask about a fact from a DIFFERENT trajectory
                    if fact_pool and rng.random() < args.negative_fraction:
                        neg = rng.choice(fact_pool)
                        if neg not in span_text:
                            question = (
                                "Earlier in this session (now compacted) — was the file or "
                                f"symbol '{neg}' mentioned? Answer yes with the exact "
                                "mention, or state it was not mentioned."
                            )
                            answer = f"'{neg}' was not mentioned in the compacted context."
                            qtype = "recall_absent"
                    span = (
                        None
                        if qtype == "recall_absent"
                        else answer_span_from_offsets(
                            article_offsets(tokenizer, span_text), span_text, answer
                        )
                    )
                    traj_rows.append(
                        flat_trajectory_row(
                            dataset_name=DATASET_NAME,
                            doc_id=doc_id,
                            question=question,
                            answer=answer,
                            scope_description="compacted agent-session history",
                            split=split,
                            group_id=repo,
                            gold_span_json=span_to_json(span),
                        )
                    )
                    stats[f"{split}:{qtype}"] = stats.get(f"{split}:{qtype}", 0) + 1
                    fact_pool.append(fact)
                    if len(fact_pool) > 3000:
                        del fact_pool[:1500]
            else:
                continue
            break

    manifest = write_flat_phase2_dataset(
        dataset_name=DATASET_NAME,
        doc_ids=doc_ids,
        doc_tokens=doc_tokens,
        traj_rows=traj_rows,
        mmap_out=args.mmap_out,
        traj_out=args.traj_out,
        tokenizer_name=args.tokenizer,
    )
    print(
        f"mmap: {manifest['row_count']} spans, {manifest['total_tokens']:,} tokens; "
        f"trajectories: {len(traj_rows)}"
    )
    print("by split:qtype:", json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
