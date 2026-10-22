#!/usr/bin/env python3
"""Convert QA pair JSONL to memory-mappable numpy arrays.

Reads per-repo JSONL files from data/qa_pairs/{owner}/{repo}/qa_pairs.jsonl,
tokenizes questions and answers, and writes mmap output:
  - tokens.npy / offsets.npy           — answer token sequences (decoder targets)
  - question_tokens.npy / question_offsets.npy — question token sequences
  - metadata.parquet                    — join keys + provenance
  - manifest.json                       — dataset stats

Usage:
    python scripts/convert_qa_pairs_to_npy.py \
        --input-dir data/qa_pairs/ \
        --output-dir data/mmap/qa_conditioned/ \
        --tokenizer Qwen/Qwen3.5-0.8B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without editable install
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import numpy as np
import pyarrow as pa

from bgkit.data.mmap_writer import (
    build_csr_offsets,
    collect_jsonl_files,
    infer_repo_path,
    write_mmap_artifacts,
)

# Metadata columns written to metadata.parquet.
#
# ``commit_ts`` (Unix seconds, int64) and ``question_type`` (str) are
# optional-but-preserved columns used by the git_history dataset wrapper
# and by build_browse_tree.py to construct the year-bucketed browse tree.
# They default to 0 / "" when missing so non-git pipelines stay
# backwards-compatible.
META_COLUMNS = [
    "repo_path", "file_path", "commit_sha", "commit_ts",
    "question", "question_type", "category",
    "model_id", "generation_tier", "prompt_version",
]


def convert(input_dir: Path, output_dir: Path, tokenizer_name: str, max_tokens: int) -> dict:
    """Read QA JSONL files, tokenize, write mmap output. Returns manifest."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    jsonl_files = collect_jsonl_files(input_dir)
    if not jsonl_files:
        print(f"ERROR: No .jsonl files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(jsonl_files)} JSONL files in {input_dir}")

    # Accumulate data
    answer_chunks: list[np.ndarray] = []
    answer_lengths: list[int] = []
    question_chunks: list[np.ndarray] = []
    question_lengths: list[int] = []
    # Optional: per-row sorted answer-position indices into the source file's
    # token stream, produced by scripts/filter_qa_with_positions.py. Stored
    # CSR-style alongside answer/question tokens. When present in input
    # records, all rows must have it (we hard-fail mid-stream otherwise so
    # training doesn't silently get partial supervision).
    position_index_chunks: list[np.ndarray] = []
    position_index_lengths: list[int] = []
    saw_positions = False
    saw_no_positions = False
    meta: dict[str, list] = {col: [] for col in META_COLUMNS}
    skipped = 0

    for fi, jsonl_path in enumerate(jsonl_files):
        repo_path = infer_repo_path(jsonl_path.parent, input_dir)

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                question = record.get("question", "")
                answer = record.get("answer", "")
                if not question or not answer or not answer.strip():
                    continue

                # Tokenize answer (decoder target)
                answer_ids = tokenizer.encode(answer, add_special_tokens=False)
                if len(answer_ids) > max_tokens:
                    skipped += 1
                    continue

                # Tokenize question (compression prompt)
                question_ids = tokenizer.encode(question, add_special_tokens=False)

                answer_arr = np.array(answer_ids, dtype=np.int32)
                question_arr = np.array(question_ids, dtype=np.int32)

                answer_chunks.append(answer_arr)
                answer_lengths.append(len(answer_arr))
                question_chunks.append(question_arr)
                question_lengths.append(len(question_arr))

                # Optional answer-position annotations.
                pos_indices = record.get("answer_position_indices")
                if pos_indices is not None:
                    if saw_no_positions:
                        sys.exit(
                            "ERROR: input dir mixes records with and without "
                            "'answer_position_indices'. Re-run the filter to "
                            "produce a uniform set, or convert separately.",
                        )
                    saw_positions = True
                    pos_arr = np.array(pos_indices, dtype=np.int32)
                    position_index_chunks.append(pos_arr)
                    position_index_lengths.append(len(pos_arr))
                else:
                    if saw_positions:
                        sys.exit(
                            "ERROR: input dir mixes records with and without "
                            "'answer_position_indices'.",
                        )
                    saw_no_positions = True

                # Metadata
                meta["repo_path"].append(record.get("repo_path", repo_path))
                meta["file_path"].append(record.get("file_path", ""))
                meta["commit_sha"].append(record.get("commit_sha", ""))
                # Unix seconds; 0 means unknown (older records without this
                # field, or non-git QA sources).
                commit_ts = record.get("commit_ts", 0)
                try:
                    meta["commit_ts"].append(int(commit_ts) if commit_ts else 0)
                except (TypeError, ValueError):
                    meta["commit_ts"].append(0)
                meta["question"].append(question[:500])  # Truncate for storage
                meta["question_type"].append(record.get("question_type", ""))
                meta["category"].append(record.get("category", ""))
                meta["model_id"].append(record.get("model_id", ""))
                meta["generation_tier"].append(record.get("generation_tier", ""))
                meta["prompt_version"].append(record.get("prompt_version", 0))

        if (fi + 1) % 500 == 0 or (fi + 1) == len(jsonl_files):
            print(f"  Processed {fi + 1}/{len(jsonl_files)} JSONL files "
                  f"({len(answer_lengths)} QA pairs so far)")

    if not answer_chunks:
        print("ERROR: No QA pairs found", file=sys.stderr)
        sys.exit(1)

    # Concatenate token arrays
    answer_tokens = np.concatenate(answer_chunks)
    question_tokens = np.concatenate(question_chunks)

    # Build CSR offsets
    answer_offsets = build_csr_offsets(np.array(answer_lengths, dtype=np.int64))
    question_offsets = build_csr_offsets(np.array(question_lengths, dtype=np.int64))

    total_rows = len(answer_lengths)
    total_answer_tokens = int(answer_offsets[-1])
    total_question_tokens = int(question_offsets[-1])

    print(f"\n  {total_rows} QA pairs")
    print(f"  {total_answer_tokens} answer tokens ({answer_tokens.nbytes / 1e6:.1f} MB)")
    print(f"  {total_question_tokens} question tokens ({question_tokens.nbytes / 1e6:.1f} MB)")
    print(f"  {skipped} skipped (over {max_tokens} tokens)")

    # Build metadata table
    meta_columns: dict[str, pa.Array] = {}
    for col in META_COLUMNS:
        if col == "prompt_version":
            meta_columns[col] = pa.array(meta[col], type=pa.int32())
        elif col == "commit_ts":
            meta_columns[col] = pa.array(meta[col], type=pa.int64())
        else:
            meta_columns[col] = pa.array(meta[col], type=pa.string())

    meta_table = pa.table(meta_columns)

    extra_arrays = {
        "question_tokens.npy": question_tokens,
        "question_offsets.npy": question_offsets,
    }
    manifest_extras = {
        "dataset_type": "qa_conditioned",
        "skipped_over_max_tokens": skipped,
        "max_tokens": max_tokens,
        "tokenizer": tokenizer_name,
        "source_jsonl_count": len(jsonl_files),
        "total_question_tokens": total_question_tokens,
        "has_answer_positions": saw_positions,
    }

    if saw_positions:
        position_indices = np.concatenate(position_index_chunks)
        position_offsets = build_csr_offsets(
            np.array(position_index_lengths, dtype=np.int64),
        )
        extra_arrays["answer_position_indices.npy"] = position_indices
        extra_arrays["answer_position_offsets.npy"] = position_offsets
        total_pos = int(position_offsets[-1])
        manifest_extras["total_answer_position_indices"] = total_pos
        print(f"  {total_pos} answer-position indices "
              f"({position_indices.nbytes / 1e6:.1f} MB)")

    manifest = write_mmap_artifacts(
        output_dir, answer_tokens, answer_offsets,
        manifest_extra=manifest_extras,
        metadata_table=meta_table,
        extra_arrays=extra_arrays,
    )

    print(f"\n  Wrote {output_dir}/tokens.npy ({answer_tokens.nbytes / 1e9:.2f} GB)")
    print(f"  Wrote {output_dir}/question_tokens.npy ({question_tokens.nbytes / 1e6:.1f} MB)")
    print(f"  Wrote {output_dir}/offsets.npy, question_offsets.npy")
    if saw_positions:
        print(f"  Wrote {output_dir}/answer_position_indices.npy, answer_position_offsets.npy")
    print(f"  Wrote {output_dir}/metadata.parquet, manifest.json")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert QA pair JSONL to memory-mappable numpy arrays."
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory containing {owner}/{repo}/qa_pairs.jsonl files",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for mmap artifacts",
    )
    parser.add_argument(
        "--tokenizer", type=str, default="Qwen/Qwen3.5-0.8B",
        help="HuggingFace tokenizer name (default: Qwen/Qwen3.5-0.8B)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Skip answers exceeding this many tokens (default: 2048)",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"ERROR: {args.input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    convert(args.input_dir, args.output_dir, args.tokenizer, args.max_tokens)


if __name__ == "__main__":
    main()
