#!/usr/bin/env python
"""Fetch plain-text filler documents (for a later ``babi_filler`` rebuild).

Writes JSONL files, one ``{"text": ..., "source": ...}`` per line, documents of
2,000-60,000 characters, UTF-8, no markup, to ``--out-dir``:

  pg19.jsonl         PG-19 train (parquet mirror ``emozilla/pg19`` — the
                     ``deepmind/pg19`` repo is a loading script, which
                     ``datasets`` >= 4 refuses): N books, each split into
                     non-overlapping 40,000-char chunks after skipping the
                     first 5% (front matter), at most 12 chunks per book.
  fineweb_edu.jsonl  HuggingFaceFW/fineweb-edu ``sample-10BT``: the first N
                     documents of 2,000-40,000 chars.
  wikitext.jsonl     Salesforce/wikitext ``wikitext-103-raw-v1`` train: lines
                     joined into articles at `` = Title = `` headers (header
                     markup stripped, tokenisation artefacts `` @-@ `` and
                     `` , `` de-tokenised); N articles of 2,000+ chars.

Everything is streamed; nothing is materialised beyond one document.

Example:
  scripts/bgkit2_fetch_filler.py --out-dir $B/filler
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

OUT_DIR = "/home/werg/bgkit-data-nvme/bgkit2/filler"
MAX_DOC = 60_000
MIN_DOC = 2_000


def _open_out(path: Path, overwrite: bool):
    if path.exists() and not overwrite:
        sys.exit(f"refusing to overwrite existing {path} (pass --overwrite)")
    return open(path, "w", encoding="utf-8")


def _next_paragraph_break(text: str, pos: int) -> int:
    """Index just after the first blank line at or after ``pos`` (``len(text)``
    when there is none), so a chunk cut there starts on a paragraph."""
    j = text.find("\n\n", pos)
    if j < 0:
        return len(text)
    while j < len(text) and text[j] == "\n":
        j += 1
    return j


def _emit(fh, text: str, source: str, stats: dict) -> None:
    fh.write(json.dumps({"text": text, "source": source}, ensure_ascii=False) + "\n")
    stats["docs"] += 1
    stats["chars"] += len(text)


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_PG19_REDACTION_RE = re.compile(r" ?<DW\d+>")
_GUTENBERG_END_RE = re.compile(r"\n\**\s*End of (the )?Project Gutenberg", re.IGNORECASE)


# ----------------------------------------------------------------------------
def fetch_pg19(args, out: Path) -> dict:
    from datasets import load_dataset

    ds = load_dataset(args.pg19_repo, split="train", streaming=True)
    stats = {"docs": 0, "chars": 0, "books": 0}
    t0 = time.time()
    with _open_out(out, args.overwrite) as fh:
        for r in ds:
            if stats["books"] >= args.pg19_books:
                break
            # PG-19 ships ``<DWnn>`` redaction placeholders in place of some
            # words (229 of the first 300 train books); they are not text.
            text = _clean(_PG19_REDACTION_RE.sub("", str(r.get("text") or "")))
            end = _GUTENBERG_END_RE.search(text)
            if end:  # trailing licence boilerplate
                text = text[: end.start()]
            start = int(len(text) * 0.05)
            # Chunk edges sit on paragraph breaks (2026-09-14): fixed offsets
            # cut books mid-word, and 46% / 52% of the books-corpus documents
            # then began / ended mid-sentence.
            start = _next_paragraph_break(text, start)
            body = text[start:]
            if len(body) < MIN_DOC:
                continue
            stats["books"] += 1
            title = str(r.get("short_book_title") or r.get("book_title") or r.get("book_id") or "")
            k = 0
            for _n_chunks in range(args.pg19_max_chunks):
                if k >= len(body):
                    break
                end = _next_paragraph_break(body, min(len(body), k + args.pg19_chunk))
                chunk = body[k:end].strip()
                k = end
                if len(chunk) < MIN_DOC:
                    break
                _emit(fh, chunk[:MAX_DOC], f"pg19:{title}", stats)
            if stats["books"] % 25 == 0:
                print(
                    f"pg19: {stats['books']} books, {stats['docs']} docs, "
                    f"{stats['chars'] / 1e6:.1f}M chars; {time.time() - t0:.0f}s",
                    flush=True,
                )
    return stats


def fetch_fineweb_edu(args, out: Path) -> dict:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    stats = {"docs": 0, "chars": 0, "scanned": 0}
    t0 = time.time()
    with _open_out(out, args.overwrite) as fh:
        for r in ds:
            if stats["docs"] >= args.fineweb_docs:
                break
            stats["scanned"] += 1
            text = _clean(str(r.get("text") or ""))
            if not (MIN_DOC <= len(text) <= args.fineweb_max_chars):
                continue
            _emit(fh, text, f"fineweb_edu:{r.get('id', '')}", stats)
            if stats["docs"] % 1000 == 0:
                print(
                    f"fineweb_edu: {stats['docs']} docs of {stats['scanned']} scanned, "
                    f"{stats['chars'] / 1e6:.1f}M chars; {time.time() - t0:.0f}s",
                    flush=True,
                )
    return stats


_TITLE_RE = re.compile(r"^\s*=\s+([^=].*?)\s+=\s*$")  # " = Title = "
_HEADING_RE = re.compile(r"^\s*(=\s*){2,}\s*(.*?)\s*(\s*=){2,}\s*$")  # " = = Sub = = "


def _detok(line: str) -> str:
    line = line.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    line = re.sub(r" ([,.;:!?%)\]}])", r"\1", line)
    line = re.sub(r"([(\[{$]) ", r"\1", line)
    line = re.sub(r" ([’']) ?(s|t|re|ve|ll|d|m)\b", r"\1\2", line)  # noqa: RUF001
    line = re.sub(r"(\w) ([’'])(s|t|re|ve|ll|d|m)\b", r"\1\2\3", line)  # noqa: RUF001
    line = line.replace(" n't", "n't")
    line = re.sub(r'" (.*?) "', r'"\1"', line)
    return re.sub(r"[ \t]{2,}", " ", line).strip()


def fetch_wikitext(args, out: Path) -> dict:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    stats = {"docs": 0, "chars": 0, "articles": 0}
    t0 = time.time()
    title: str | None = None
    parts: list[str] = []

    def flush(fh) -> None:
        if title is None:
            return
        stats["articles"] += 1
        if stats["articles"] % 500 == 0:
            print(
                f"wikitext: {stats['articles']} articles seen, {stats['docs']} kept, "
                f"{stats['chars'] / 1e6:.1f}M chars; {time.time() - t0:.0f}s",
                flush=True,
            )
        body = "\n\n".join(p for p in parts if p)
        text = _clean(title + "\n\n" + body)
        if len(text) >= MIN_DOC:
            if len(text) > MAX_DOC:
                cut = text.rfind("\n\n", 0, MAX_DOC)
                text = text[: cut if cut > MIN_DOC else MAX_DOC].strip()
            _emit(fh, text, f"wikitext:{title}", stats)

    with _open_out(out, args.overwrite) as fh:
        for r in ds:
            if stats["docs"] >= args.wikitext_articles:
                break
            line = r["text"]
            m = _TITLE_RE.match(line)
            if m:
                flush(fh)
                title, parts = _detok(m.group(1)), []
                continue
            if title is None:
                continue
            h = _HEADING_RE.match(line)
            if h:
                parts.append(_detok(h.group(2)))
                continue
            s = line.strip()
            if s:
                parts.append(_detok(s))
        if stats["docs"] < args.wikitext_articles:
            flush(fh)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--sources", default="pg19,fineweb_edu,wikitext")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--pg19-repo", default="emozilla/pg19")
    ap.add_argument("--pg19-books", type=int, default=300)
    ap.add_argument("--pg19-chunk", type=int, default=40_000)
    ap.add_argument("--pg19-max-chunks", type=int, default=12)
    ap.add_argument("--fineweb-docs", type=int, default=8000)
    ap.add_argument("--fineweb-max-chars", type=int, default=40_000)
    ap.add_argument("--wikitext-articles", type=int, default=4000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fetchers = {
        "pg19": (fetch_pg19, "pg19.jsonl"),
        "fineweb_edu": (fetch_fineweb_edu, "fineweb_edu.jsonl"),
        "wikitext": (fetch_wikitext, "wikitext.jsonl"),
    }
    summary = {}
    for src in args.sources.split(","):
        fn, name = fetchers[src]
        t0 = time.time()
        st = fn(args, out_dir / name)
        summary[src] = st
        print(
            f"== {src}: {st['docs']} docs, {st['chars']:,} chars -> {out_dir / name} "
            f"({time.time() - t0:.0f}s) {st}",
            flush=True,
        )
    print(json.dumps(summary, indent=1))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
    # datasets' streaming reader aborts the interpreter at finalisation
    # (PyGILState_Release, datasets 4.8.4 on this venv); outputs are closed.
    os._exit(0)
