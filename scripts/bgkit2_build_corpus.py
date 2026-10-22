#!/usr/bin/env python
"""Build an LFM2.5-tokenised token store from a Qwen-tokenised parquet corpus.

Every prepared corpus on disk stores Qwen3.5 token ids and no text, so we
decode with the Qwen tokenizer and re-encode with the LFM2.5 tokenizer
(byte-level BPE both; round-trip is exact for valid UTF-8 text). CPU only,
one worker per shard.

Examples (host venv):

  # prose: FineWeb-Edu shards 0..149 -> train, 970..979 -> eval
  scripts/bgkit2_build_corpus.py --source /home/werg/bgkit-data-nvme/fineweb_edu_v1/tokens \\
      --shards 0:150 --out /home/werg/bgkit-data-nvme/bgkit2/corpus/fineweb_train --workers 12
  scripts/bgkit2_build_corpus.py --source /home/werg/bgkit-data-nvme/fineweb_edu_v1/tokens \\
      --shards 970:980 --out /home/werg/bgkit-data-nvme/bgkit2/corpus/fineweb_eval --max-docs 4000

  # code: processed_v2 shards
  scripts/bgkit2_build_corpus.py --source $DATA_DIR/processed_v2/tokens --shards 0:20 \\
      --out /home/werg/bgkit-data-nvme/bgkit2/corpus/code_train --workers 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgkit2.data.token_store import TOKEN_DTYPE, TokenStoreWriter

QWEN_ID = "Qwen/Qwen3.5-0.8B"
LFM_ID = "LiquidAI/LFM2.5-350M-Base"

_qwen = None
_lfm = None


def _tokenizers():
    global _qwen, _lfm
    if _qwen is None:
        from transformers import AutoTokenizer

        _qwen = AutoTokenizer.from_pretrained(QWEN_ID)
        _lfm = AutoTokenizer.from_pretrained(LFM_ID)
    return _qwen, _lfm


def convert_shard(
    shard_path: str, max_doc_tokens: int, min_doc_tokens: int, max_docs: int | None
) -> tuple[str, list[np.ndarray], int]:
    import pyarrow.parquet as pq

    qwen, lfm = _tokenizers()
    table = pq.read_table(shard_path, columns=["token_ids"])
    out: list[np.ndarray] = []
    dropped = 0
    col = table.column("token_ids")
    n = table.num_rows if max_docs is None else min(table.num_rows, max_docs)
    batch_texts: list[str] = []

    def flush() -> None:
        nonlocal dropped
        if not batch_texts:
            return
        enc = lfm(batch_texts, add_special_tokens=False)["input_ids"]
        for ids in enc:
            if len(ids) < min_doc_tokens:
                dropped += 1
                continue
            out.append(np.asarray(ids[:max_doc_tokens], dtype=TOKEN_DTYPE))
        batch_texts.clear()

    for i in range(n):
        ids = col[i].as_py()
        if not ids:
            dropped += 1
            continue
        batch_texts.append(qwen.decode(ids, skip_special_tokens=True))
        if len(batch_texts) >= 256:
            flush()
    flush()
    return shard_path, out, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source", required=True, help="dir with shard_NNNNN.parquet files (token_ids column)"
    )
    ap.add_argument("--shards", required=True, help="range a:b over sorted shard files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-doc-tokens", type=int, default=8192)
    ap.add_argument("--min-doc-tokens", type=int, default=64)
    ap.add_argument("--max-docs", type=int, default=None, help="cap documents per shard")
    args = ap.parse_args()

    shards = sorted(Path(args.source).glob("shard_*.parquet"))
    a, b = (int(x) for x in args.shards.split(":"))
    shards = shards[a:b]
    if not shards:
        sys.exit(f"no shards in {args.source}[{a}:{b}]")
    out = Path(args.out)
    if (out / "meta.json").exists():
        sys.exit(f"{out} already exists; refusing to overwrite")
    writer = TokenStoreWriter(
        out,
        meta={
            "source": str(args.source),
            "shards": [s.name for s in shards],
            "tokenizer": LFM_ID,
            "decoded_from": QWEN_ID,
            "max_doc_tokens": args.max_doc_tokens,
            "min_doc_tokens": args.min_doc_tokens,
        },
    )
    t0 = time.time()
    total_dropped = 0
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                convert_shard, str(s), args.max_doc_tokens, args.min_doc_tokens, args.max_docs
            ): s
            for s in shards
        }
        for fut in as_completed(futs):
            name, docs, dropped = fut.result()
            for d in docs:
                writer.add(d)
            total_dropped += dropped
            done += 1
            print(
                f"[{done}/{len(shards)}] {Path(name).name}: {len(docs)} docs, dropped {dropped}; "
                f"total docs {writer.n_docs}; {time.time() - t0:.0f}s",
                flush=True,
            )
    writer.close()
    meta = json.loads((out / "meta.json").read_text())
    print(
        json.dumps({**meta, "dropped": total_dropped, "seconds": round(time.time() - t0)}, indent=2)
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    main()
