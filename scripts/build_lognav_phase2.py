#!/usr/bin/env python
"""Build the ``lognav`` Phase-2 flat dataset from raw LogHub logs.

Family B (wide-net tool results) training data in the format the existing
``KRKBTrainer`` flat single-bgkit path consumes
(`plans/capability_packaging_2026_08_20.md` §5 + §12 checklist):

- ``{mmap_out}/lognav/``: article store — one article per log *window*
  (tokens.npy/offsets.npy/metadata.parquet/manifest.json).
- ``{traj_out}/lognav.parquet``: flat 2-turn trajectories
  (bgkit retrieval call + answer) with an explicit ``split`` column —
  eval = held-out log *systems*, so generalization is cross-system.

Windows are span-subsampled uniformly across each file (not head-biased).
Windows above ``--max-window-tokens`` are dropped (live-L0 budget).

Usage:
    .venv/bin/python scripts/build_lognav_phase2.py \
      --loghub-dir /home/werg/bgkit-data-nvme/capability_packaging/loghub \
      --mmap-out $DATA_DIR/mmap/phase2 \
      --traj-out $DATA_DIR/trajectories
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bgkit.data.flat_phase2_writer import flat_trajectory_row
from bgkit.data.gold_span import answer_span_from_offsets, article_offsets, span_to_json
from bgkit.data.lognav_qa import generate_window_samples, iter_error_windows, iter_windows
from bgkit.data.mmap_writer import build_csr_offsets, write_mmap_artifacts

DATASET_NAME = "lognav"

# (system, relative log path, split). Held-out SYSTEMS give the eval split —
# generalization is across log formats, not just across windows.
SOURCES: list[tuple[str, str, str]] = [
    ("bgl", "BGL_extracted/BGL.log", "train"),
    ("hdfs", "HDFS_v1_extracted/HDFS.log", "train"),
    ("ssh", "SSH_extracted/SSH.log", "train"),
    ("openstack", "OpenStack_extracted/openstack_normal1.log", "train"),
    ("openstack", "OpenStack_extracted/openstack_abnormal.log", "train"),
    ("mac", "Mac_extracted/Mac.log", "train"),
    ("healthapp", "HealthApp_extracted/HealthApp.log", "train"),
    ("hpc", "HPC_extracted/HPC.log", "train"),
    ("linux", "Linux_extracted/Linux.log", "train"),
    ("proxifier", "Proxifier_extracted/Proxifier.log", "train"),
    ("zookeeper", "Zookeeper_extracted/Zookeeper.log", "eval"),
    ("apache", "Apache_extracted/Apache.log", "eval"),
]

WINDOW_CHARS = (30_000, 60_000, 120_000)


def subsample_spans(spans: list[tuple[int, int]], k: int, rng: random.Random) -> list:
    if len(spans) <= k:
        return spans
    idx = sorted(rng.sample(range(len(spans)), k))
    return [spans[i] for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loghub-dir", required=True)
    ap.add_argument("--mmap-out", required=True, help="phase2 mmap root")
    ap.add_argument("--traj-out", required=True, help="trajectories dir")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--max-windows-per-config", type=int, default=40)
    ap.add_argument("--max-window-tokens", type=int, default=45_000)
    ap.add_argument("--error-window-fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    loghub = Path(args.loghub_dir)
    rng = random.Random(args.seed)

    # window doc_id -> (token ids); trajectory rows reference doc_ids.
    doc_ids: list[str] = []
    doc_tokens: list[np.ndarray] = []
    traj_rows: list[dict] = []
    stats: dict[str, int] = {}

    for system, rel, split in SOURCES:
        path = loghub / rel
        if not path.exists():
            print(f"SKIP missing {path}")
            continue
        lines = path.read_text(errors="replace").splitlines()
        for wchars in WINDOW_CHARS:
            spans = iter_windows(lines, window_chars=wchars)
            spans = subsample_spans(spans, args.max_windows_per_config, rng)
            # error-centered windows guarantee first_error supply (2026-08-21
            # fix: uniform windows gave 35 first_error vs 925 error_absent)
            n_err = max(1, int(len(spans) * args.error_window_fraction))
            spans = spans + iter_error_windows(
                lines, window_chars=wchars, max_windows=n_err, rng=rng
            )
            for span in spans:
                window_text = "\n".join(lines[span[0] : span[1]])
                ids = tokenizer.encode(window_text, add_special_tokens=False)
                if len(ids) > args.max_window_tokens or len(ids) < 256:
                    continue
                samples = generate_window_samples(
                    lines,
                    span,
                    dataset=system,
                    path=str(path),
                    rng=rng,
                    include_counts=True,
                )
                if not samples:
                    continue
                doc_id = samples[0].source_ref
                doc_ids.append(doc_id)
                doc_tokens.append(np.asarray(ids, dtype=np.int32))
                offsets = article_offsets(tokenizer, window_text)
                for s in samples:
                    span = (
                        None
                        if s.qtype in ("error_absent", "count_keyword")
                        else answer_span_from_offsets(offsets, window_text, s.answer)
                    )
                    # Shared writer = the single source of the leaf-drill
                    # contract (plain {ids, query}; NO is_head — see
                    # flat_trajectory_row docstring, 2026-08-22).
                    traj_rows.append(
                        flat_trajectory_row(
                            dataset_name=DATASET_NAME,
                            doc_id=doc_id,
                            question=s.question,
                            answer=s.answer,
                            scope_description=f"raw log file {path.name}",
                            split=split,
                            group_id=f"{system}:{path.name}",
                            gold_span_json=span_to_json(span),
                        )
                    )
                    stats[f"{split}:{s.qtype}"] = stats.get(f"{split}:{s.qtype}", 0) + 1

    assert doc_ids, "no windows produced"
    assert len(set(doc_ids)) == len(doc_ids), "duplicate window doc_ids"

    lengths = np.asarray([len(t) for t in doc_tokens], dtype=np.int64)
    tokens = np.concatenate(doc_tokens).astype(np.int32)
    offsets = build_csr_offsets(lengths)
    metadata = pa.table(
        {
            "id": pa.array(range(len(doc_ids)), type=pa.int64()),
            "document_id": pa.array(doc_ids, type=pa.string()),
            "dataset_name": pa.array([DATASET_NAME] * len(doc_ids), type=pa.string()),
            "tag_list_json": pa.array(["[]"] * len(doc_ids), type=pa.string()),
        }
    )
    mmap_dir = Path(args.mmap_out) / DATASET_NAME
    manifest = write_mmap_artifacts(
        mmap_dir,
        tokens,
        offsets,
        manifest_extra={"dataset": DATASET_NAME, "tokenizer": args.tokenizer},
        metadata_table=metadata,
    )
    print(
        f"mmap: {manifest['row_count']} windows, {manifest['total_tokens']:,} tokens -> {mmap_dir}"
    )

    traj_table = pa.table(
        {
            name: pa.array([r[name] for r in traj_rows])
            for name in (
                "dataset_name",
                "scope_template",
                "scope_description",
                "topic_list_json",
                "question",
                "gold_answer",
                "trajectory_json",
                "split",
                "group_id",
                "gold_span_json",
            )
        }
    )
    traj_path = Path(args.traj_out) / f"{DATASET_NAME}.parquet"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(traj_table, traj_path)
    print(f"trajectories: {len(traj_rows)} rows -> {traj_path}")
    print("by split:qtype:", json.dumps(stats, indent=None, sort_keys=True))


if __name__ == "__main__":
    main()
