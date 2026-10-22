#!/usr/bin/env python3
"""Tokenize FineWeb-Edu into shard parquets matching the processed_v2 schema.

Streams HuggingFaceFW/fineweb-edu (default config: sample-10BT) with the
Qwen3.5-0.8B-Base tokenizer and writes one parquet shard per
``--docs-per-shard`` documents. Output layout:

    {output_dir}/
        manifest.jsonl
        tokens/
            shard_NNNNN.parquet   # columns: repo_path, file_path, language,
                                  #          token_ids (list<int32>),
                                  #          commit_sha

This is the FineWeb-Edu analog of bgkit.data.corpus_stats.process_corpus.
The shards drop straight into the existing pipeline:

    scripts/convert_tokens_to_npy.py --input-dir <out>/tokens
    scripts/convert_tokens_to_falcon_mmap.py --input-dir <out>/tokens \\
        --output-dir <out>/tokens_falcon_h1

Resume: pass --start-shard N to continue after a partial run. The dataset
stream is also fast-forwarded by --skip-docs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def _write_shard(shard_path: Path, rows: list[dict]) -> None:
    table = pa.table({
        "repo_path": pa.array([r["repo_path"] for r in rows], type=pa.string()),
        "file_path": pa.array([r["file_path"] for r in rows], type=pa.string()),
        "language": pa.array([r["language"] for r in rows], type=pa.string()),
        "token_ids": pa.array(
            [np.array(r["token_ids"], dtype=np.int32) for r in rows],
            type=pa.list_(pa.int32()),
        ),
        "commit_sha": pa.array(
            [r["commit_sha"] for r in rows], type=pa.string(),
        ),
    })
    pq.write_table(table, shard_path, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument(
        "--target-tokens", type=int, default=10_000_000_000,
        help="Stop after this many Qwen tokens written.",
    )
    parser.add_argument(
        "--max-tokens-per-doc", type=int, default=32_768,
        help="Truncate any doc longer than this many Qwen tokens.",
    )
    parser.add_argument("--docs-per-shard", type=int, default=5000)
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Tokenizer batch_encode call size.",
    )
    parser.add_argument(
        "--start-shard", type=int, default=0,
        help="Resume: name new shards starting here.",
    )
    parser.add_argument(
        "--skip-docs", type=int, default=0,
        help="Resume: fast-forward this many docs from the stream head.",
    )
    args = parser.parse_args()

    out = args.output_dir
    tokens_dir = out / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True,
    )

    print(
        f"Streaming {args.dataset_id} :: {args.config} :: {args.split} "
        f"(target {args.target_tokens:,} tokens, "
        f"start_shard={args.start_shard}, skip_docs={args.skip_docs})",
        file=sys.stderr,
    )
    ds = load_dataset(
        args.dataset_id, args.config, split=args.split, streaming=True,
    )
    if args.skip_docs:
        ds = ds.skip(args.skip_docs)

    repo_path_id = f"{args.dataset_id}/{args.config}"
    shard_idx = args.start_shard
    shard_rows: list[dict] = []
    docs_in_shard = 0
    total_tokens = 0
    total_docs = 0
    batch_texts: list[str] = []
    batch_meta: list[tuple[str, str | None]] = []

    def flush_batch():
        nonlocal docs_in_shard, total_tokens, total_docs, shard_idx
        if not batch_texts:
            return
        # add_special_tokens=False matches process_corpus; we don't want
        # BOS/EOS injected since concatenation is done at use time.
        encs = tokenizer(
            batch_texts, add_special_tokens=False, return_attention_mask=False,
        )["input_ids"]
        for (doc_id, url), ids in zip(batch_meta, encs, strict=True):
            if not ids:
                continue
            ids = ids[: args.max_tokens_per_doc]
            tok_count = len(ids)
            shard_rows.append({
                "repo_path": repo_path_id,
                "file_path": str(doc_id),
                "language": "en",
                "token_ids": ids,
                "commit_sha": "",
            })
            mf.write(json.dumps({
                "doc_id": str(doc_id),
                "url": url,
                "token_count": tok_count,
            }) + "\n")
            docs_in_shard += 1
            total_docs += 1
            total_tokens += tok_count
            if docs_in_shard >= args.docs_per_shard:
                shard_path = tokens_dir / f"shard_{shard_idx:05d}.parquet"
                _write_shard(shard_path, shard_rows)
                shard_idx += 1
                shard_rows.clear()
                docs_in_shard = 0
        batch_texts.clear()
        batch_meta.clear()

    with open(manifest_path, "a") as mf:
        pbar = tqdm(ds, desc="FineWeb-Edu", unit="doc", mininterval=2.0)
        for rec in pbar:
            text = rec.get("text") or ""
            if not text:
                continue
            doc_id = rec.get("id") or f"doc_{total_docs}"
            url = rec.get("url")
            batch_texts.append(text)
            batch_meta.append((doc_id, url))
            if len(batch_texts) >= args.batch_size:
                flush_batch()
                pbar.set_postfix(
                    docs=total_docs,
                    tokens=f"{total_tokens:,}",
                    shards=shard_idx,
                )
            if total_tokens >= args.target_tokens:
                break

        flush_batch()
        if shard_rows:
            shard_path = tokens_dir / f"shard_{shard_idx:05d}.parquet"
            _write_shard(shard_path, shard_rows)
            shard_idx += 1

    print(
        f"\nWrote {shard_idx - args.start_shard} new shards "
        f"(indices {args.start_shard}..{shard_idx - 1}), "
        f"{total_docs} docs, {total_tokens:,} tokens",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
