#!/usr/bin/env python3
"""Tokenize a multi-doc summarization corpus into the BgKIT shard schema.

Produces one parquet shard per ``--rows-per-shard`` examples. Each row is one
summarization "group":

    columns:
        group_id:          str                — unique ID within dataset
        dataset:           str                — e.g. 'multi_news', 'arxiv_s2orc'
        doc_token_ids:     list<list<int32>>  — Qwen tokens, one list per source doc
        target_token_ids:  list<int32>        — Falcon-H1 tokens for the summary/abstract
        num_docs:          int32              — convenience copy of len(doc_token_ids)

Per-doc and per-group caps are enforced (default: 4096 Qwen tokens per doc,
16 docs per group, 1024 Falcon tokens per target). Groups with fewer than
``--min-docs`` source docs after filtering are dropped. The target column
uses the **decoder** tokenizer (Falcon-H1-Tiny by default) because the
decoder is the only consumer of the target tokens.

Three input regimes are supported via ``--dataset``:

    multi_news   ${DATA_DIR}/multi_news_v1/raw/data/{train,val,test}.{src.cleaned,tgt}
    arxiv_s2orc  ${DATA_DIR}/arxiv_v1/raw/data/*.parquet  (S2ORC schema)
    pmc_oa_md    ${DATA_DIR}/pubmed_v1/raw/data/*.parquet (markdown text)

Output layout::

    ${output_dir}/
        manifest.json
        shard_NNNNN.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from bgkit.data.section_parser import (
    extract_pmc_abstract,
    split_multi_news,
    split_pmc_markdown,
    split_s2orc_arxiv,
    strip_pmc_frontmatter,
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("process_summarization_corpus")


# ----------------------------------------------------------------------------
# Per-dataset readers — each yields {group_id, source_docs, target_text}
# ----------------------------------------------------------------------------


def iter_multi_news(raw_dir: Path, *, split: str = "train") -> Iterator[dict]:
    # alexfabbri/multi_news raw dump puts files under data/.
    data_dir = raw_dir / "data" if (raw_dir / "data").is_dir() else raw_dir
    src_path = data_dir / f"{split}.src.cleaned"
    tgt_path = data_dir / f"{split}.tgt"
    if not src_path.exists() or not tgt_path.exists():
        raise FileNotFoundError(
            f"multi_news raw files missing: expected {src_path} and {tgt_path}"
        )
    with src_path.open() as fs, tgt_path.open() as ft:
        for idx, (src_line, tgt_line) in enumerate(zip(fs, ft)):
            articles = split_multi_news(src_line)
            target = tgt_line.replace("NEWLINE_CHAR", "\n").strip()
            if target.startswith("– "):
                target = target[2:]  # leading bullet often present
            if not target:
                continue
            yield {
                "group_id": f"multi_news/{split}/{idx:07d}",
                "source_docs": articles,
                "target_text": target,
            }


def iter_s2orc_arxiv(raw_dir: Path) -> Iterator[dict]:
    files = sorted(raw_dir.glob("data/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no S2ORC parquet files under {raw_dir}/data/")
    for shard_path in files:
        pf = pq.ParquetFile(shard_path)
        for batch in pf.iter_batches(batch_size=512, columns=["arxivid", "text", "abstract"]):
            tbl = batch.to_pydict()
            for arxivid, text, abstract in zip(
                tbl["arxivid"], tbl["text"], tbl["abstract"]
            ):
                if not text or not abstract:
                    continue
                sections = split_s2orc_arxiv(text)
                # Use section bodies as source docs, skipping empty ones and
                # the "preamble" chunk if it's empty / just the title.
                docs = [body for _heading, body in sections if body and len(body) >= 200]
                if not docs:
                    continue
                yield {
                    "group_id": f"arxiv_s2orc/{arxivid or shard_path.stem}",
                    "source_docs": docs,
                    "target_text": abstract.strip(),
                }


def iter_pmc_markdown(raw_dir: Path) -> Iterator[dict]:
    files = sorted(raw_dir.glob("data/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no PMC parquet files under {raw_dir}/data/")
    for shard_path in files:
        pf = pq.ParquetFile(shard_path)
        for batch in pf.iter_batches(batch_size=512, columns=["text"]):
            for text in batch.column("text").to_pylist():
                if not text:
                    continue
                abstract = extract_pmc_abstract(text)
                if not abstract:
                    continue
                # Strip frontmatter and split; drop the abstract section
                # from source docs so the target isn't leaked.
                clean = strip_pmc_frontmatter(text)
                sections = split_pmc_markdown(clean, strip_frontmatter=False)
                docs = []
                for heading, body in sections:
                    if not body or len(body) < 200:
                        continue
                    # Skip sections whose heading mentions "abstract"
                    if "abstract" in heading.lower():
                        continue
                    docs.append(body)
                if not docs:
                    continue
                # Use the title (line 1 of frontmatter or first heading)
                # as part of the group_id; fall back to byte hash.
                yield {
                    "group_id": f"pmc_oa_md/{shard_path.stem}/{abs(hash(text)) % 10**10:010d}",
                    "source_docs": docs,
                    "target_text": abstract,
                }


_READERS = {
    "multi_news": iter_multi_news,
    "arxiv_s2orc": iter_s2orc_arxiv,
    "pmc_oa_md": iter_pmc_markdown,
}


# ----------------------------------------------------------------------------
# Sharded parquet writer
# ----------------------------------------------------------------------------


_SCHEMA = pa.schema([
    pa.field("group_id", pa.string()),
    pa.field("dataset", pa.string()),
    pa.field("doc_token_ids", pa.list_(pa.list_(pa.int32()))),
    pa.field("target_token_ids", pa.list_(pa.int32())),
    pa.field("num_docs", pa.int32()),
])


class ShardedWriter:
    """Buffer rows in memory and flush parquet shards on a row threshold."""

    def __init__(self, out_dir: Path, dataset_name: str, rows_per_shard: int):
        self.out_dir = out_dir
        self.dataset = dataset_name
        self.rows_per_shard = rows_per_shard
        self.shard_idx = 0
        self.buffer: list[dict] = []
        self.total_rows = 0
        self.total_source_tokens = 0
        self.total_target_tokens = 0
        out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, row: dict) -> None:
        self.buffer.append(row)
        self.total_source_tokens += sum(len(d) for d in row["doc_token_ids"])
        self.total_target_tokens += len(row["target_token_ids"])
        if len(self.buffer) >= self.rows_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        table = pa.Table.from_pylist(self.buffer, schema=_SCHEMA)
        path = self.out_dir / f"shard_{self.shard_idx:05d}.parquet"
        pq.write_table(table, path, compression="zstd")
        self.total_rows += len(self.buffer)
        log.info(
            "wrote_shard idx=%d path=%s rows=%d cum_rows=%d cum_src_tokens=%d cum_tgt_tokens=%d",
            self.shard_idx, path.name, len(self.buffer), self.total_rows,
            self.total_source_tokens, self.total_target_tokens,
        )
        self.shard_idx += 1
        self.buffer = []

    def close(self) -> dict:
        self.flush()
        manifest = {
            "schema_version": 1,
            "dataset": self.dataset,
            "shard_count": self.shard_idx,
            "row_count": self.total_rows,
            "total_source_tokens": self.total_source_tokens,
            "total_target_tokens": self.total_target_tokens,
            "encoder_tokenizer": None,  # filled in by caller
            "decoder_tokenizer": None,  # filled in by caller
            "rows_per_shard": self.rows_per_shard,
        }
        return manifest


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tokenize a multi-doc summarization corpus into BgKIT shard format."
    )
    ap.add_argument(
        "--dataset", required=True, choices=sorted(_READERS),
        help="Which dataset to process.",
    )
    ap.add_argument(
        "--raw-dir", type=Path, required=False, default=None,
        help="Path to raw data root (default: ${DATA_DIR}/<corpus>_v1/raw).",
    )
    ap.add_argument(
        "--output-dir", type=Path, required=False, default=None,
        help="Output dir (default: ${DATA_DIR}/<corpus>_v1/processed).",
    )
    ap.add_argument(
        "--encoder-tokenizer", default="Qwen/Qwen3.5-0.8B-Base",
        help="HuggingFace tokenizer for the encoder (source docs).",
    )
    ap.add_argument(
        "--decoder-tokenizer", default="tiiuae/Falcon-H1-Tiny-90M-Instruct",
        help="HuggingFace tokenizer for the decoder (target text).",
    )
    ap.add_argument("--max-doc-tokens", type=int, default=4096)
    ap.add_argument("--max-target-tokens", type=int, default=1024)
    ap.add_argument("--max-docs-per-group", type=int, default=16)
    ap.add_argument("--min-docs-per-group", type=int, default=2)
    ap.add_argument("--rows-per-shard", type=int, default=2000)
    ap.add_argument(
        "--max-groups", type=int, default=None,
        help="Cap total groups for smoke tests (default: no cap).",
    )
    ap.add_argument(
        "--mn-splits", nargs="+", default=["train"],
        help="MultiNews splits to ingest (train/val/test).",
    )
    args = ap.parse_args()

    data_root = Path(os.environ["DATA_DIR"])
    corpus_dir = {
        "multi_news": "multi_news_v1",
        "arxiv_s2orc": "arxiv_v1",
        "pmc_oa_md": "pubmed_v1",
    }[args.dataset]
    raw_dir = args.raw_dir or data_root / corpus_dir / "raw"
    output_dir = args.output_dir or data_root / corpus_dir / "processed"

    log.info("loading tokenizers...")
    enc_tok = AutoTokenizer.from_pretrained(args.encoder_tokenizer, trust_remote_code=True)
    dec_tok = AutoTokenizer.from_pretrained(args.decoder_tokenizer, trust_remote_code=True)

    log.info(
        "dataset=%s raw_dir=%s output_dir=%s caps[doc=%d target=%d docs=%d-%d] rows_per_shard=%d max_groups=%s",
        args.dataset, raw_dir, output_dir,
        args.max_doc_tokens, args.max_target_tokens,
        args.min_docs_per_group, args.max_docs_per_group,
        args.rows_per_shard, args.max_groups,
    )

    writer = ShardedWriter(output_dir, args.dataset, args.rows_per_shard)

    if args.dataset == "multi_news":
        readers = [iter_multi_news(raw_dir, split=s) for s in args.mn_splits]

        def reader():
            for r in readers:
                yield from r
    else:
        reader = _READERS[args.dataset]
        reader_iter = reader(raw_dir)
        reader = lambda: reader_iter  # noqa: E731

    n_in = 0
    n_filtered = 0
    n_emitted = 0
    t0 = time.time()
    last_log = t0

    for example in reader():
        n_in += 1
        docs = example["source_docs"]
        target = example["target_text"]
        if not target.strip():
            n_filtered += 1
            continue

        # Cap doc count (keep first N — these are usually ordered most→least relevant)
        docs = docs[: args.max_docs_per_group]
        if len(docs) < args.min_docs_per_group:
            n_filtered += 1
            continue

        # Tokenize source docs with encoder tokenizer
        doc_token_ids: list[list[int]] = []
        for d in docs:
            ids = enc_tok.encode(d, add_special_tokens=False)
            if len(ids) > args.max_doc_tokens:
                ids = ids[: args.max_doc_tokens]
            if len(ids) >= 16:  # skip near-empty docs
                doc_token_ids.append(ids)
        if len(doc_token_ids) < args.min_docs_per_group:
            n_filtered += 1
            continue

        # Tokenize target with decoder tokenizer
        target_ids = dec_tok.encode(target, add_special_tokens=False)
        if len(target_ids) > args.max_target_tokens:
            target_ids = target_ids[: args.max_target_tokens]
        if len(target_ids) < 8:
            n_filtered += 1
            continue

        writer.add({
            "group_id": example["group_id"],
            "dataset": args.dataset,
            "doc_token_ids": doc_token_ids,
            "target_token_ids": target_ids,
            "num_docs": len(doc_token_ids),
        })
        n_emitted += 1
        if args.max_groups and n_emitted >= args.max_groups:
            log.info("hit --max-groups cap (%d); stopping", args.max_groups)
            break

        now = time.time()
        if now - last_log > 30:
            rate = n_emitted / (now - t0)
            log.info(
                "progress: in=%d filt=%d out=%d rate=%.1f/s elapsed=%.0fs",
                n_in, n_filtered, n_emitted, rate, now - t0,
            )
            last_log = now

    manifest = writer.close()
    manifest.update({
        "encoder_tokenizer": args.encoder_tokenizer,
        "decoder_tokenizer": args.decoder_tokenizer,
        "max_doc_tokens": args.max_doc_tokens,
        "max_target_tokens": args.max_target_tokens,
        "max_docs_per_group": args.max_docs_per_group,
        "min_docs_per_group": args.min_docs_per_group,
        "examples_seen": n_in,
        "examples_filtered": n_filtered,
        "elapsed_seconds": time.time() - t0,
    })
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(
        "DONE dataset=%s in=%d filt=%d out=%d shards=%d → %s",
        args.dataset, n_in, n_filtered, n_emitted, manifest["shard_count"], manifest_path,
    )


if __name__ == "__main__":
    main()
