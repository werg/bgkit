#!/usr/bin/env python
"""Build a per-dataset browse tree parquet.

Two input modes:

1. ``--phase2-dir``: point at an existing Phase 2 mmap dataset
   (``$DATA_DIR/mmap/phase2/{dataset}``). The script reads
   ``metadata.parquet`` directly — ``document_id`` becomes the browse-tree
   article ID (matching the L0 cache + bgkit tool key), and
   ``tag_list_json`` supplies per-article tags. Each flat tag becomes a
   one-level path under root: ``root → <tag> → <article>``. This is the
   right default for datasets whose own metadata is the authoritative
   taxonomy (memory datasets with ``memory_type``, git_history with
   ``repo_path``, etc.). For datasets with an external taxonomy (KILT
   Wikipedia category DAG, PubMedQA MeSH tree, NewsQA year/month), use
   ``--input`` instead and pre-compute the hierarchical paths offline.

2. ``--input``: a JSONL file with rows
   ``{"article_id", "tag_path", ["document_id"]}`` where ``tag_path`` is
   an ordered hierarchical path root → leaf. Use this when an external
   taxonomy provides the hierarchy that flat ``tag_list_json`` cannot.

   The optional ``document_id`` column carries the canonical mmap key
   that the L0 cache + ``ArticleTokenStore`` index. When present, the
   browse tree's node id (``article_id``) is human-readable (a title)
   and differs from the mmap key: the trainer and downstream pre-compute
   scripts translate ``article_id → document_id`` via a per-dataset
   sidecar file this script emits alongside the browse tree parquet.

Either ``--phase2-dir`` or ``--input`` must be provided.

Outputs:
    {output_dir}/{dataset}.parquet           BrowseTree serialized format
    {output_dir}/{dataset}_title_to_doc_id.json
        Title → mmap document_id lookup, emitted whenever the input
        rows carried a ``document_id`` field distinct from
        ``article_id``. Consumed by the Phase 2 KB trainer
        (:mod:`bgkit.training.phase2.kr_kb_trainer`) and by
        :mod:`scripts.build_trajectory_set` to map browse-tree IDs back
        to L0-cache / token-store keys. Missing sidecar means the
        browse tree IDs are already mmap keys (the ``--phase2-dir`` path
        or legacy numeric-id flows).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
from pathlib import Path

import pyarrow.parquet as pq

from bgkit.data.tagging import BrowseTreeBuilder, TaggingConfig

_log = logging.getLogger(__name__)


def _ingest_jsonl(
    builder: BrowseTreeBuilder, input_path: Path,
) -> dict[str, str]:
    """Ingest rows into the browse-tree builder and collect a
    title → document_id map.

    Each row is ``{"article_id": ..., "tag_path": [...], "document_id": ...}``
    where ``document_id`` is optional. When it is present and distinct
    from ``article_id``, the returned dict records the mapping so the
    caller can emit a sidecar JSON. When absent, the row contributes
    ``article_id → article_id`` (identity) — harmless because the
    downstream sidecar logic skips emission when every entry is an
    identity mapping.
    """
    title_to_doc: dict[str, str] = {}
    with input_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            article_id = str(row["article_id"])
            doc_id = str(row.get("document_id") or article_id)
            builder.add_article(
                article_id=article_id,
                tag_path=list(row.get("tag_path", [])),
            )
            # Last-writer-wins when the same article_id appears twice —
            # matches how BrowseTreeBuilder handles duplicate article ids
            # (second add_article appends to the same leaf tag's article
            # list). Production hierarchy builders disambiguate titles
            # before writing the JSONL, so collisions here are rare.
            title_to_doc[article_id] = doc_id
    return title_to_doc


def _ingest_phase2_mmap(builder: BrowseTreeBuilder, phase2_dir: Path) -> None:
    """Read ``metadata.parquet`` and feed ``(document_id, tag_path)`` rows.

    The tag path is derived from ``tag_list_json`` — each flat tag becomes
    a one-level path ``[tag]`` under root. Articles with multiple tags are
    attached to the first one (a heuristic; use ``--input`` with a
    hand-computed hierarchy for datasets where multi-tag is semantically
    important).
    """
    meta_path = phase2_dir / "metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Phase 2 metadata missing: {meta_path}. "
            "Run scripts/convert_hf_to_mmap.py first."
        )
    table = pq.read_table(
        meta_path, columns=["document_id", "tag_list_json"],
    ).to_pylist()
    for row in table:
        doc_id = str(row["document_id"])
        raw_tags = row.get("tag_list_json") or "[]"
        try:
            tags = json.loads(raw_tags)
        except (TypeError, json.JSONDecodeError):
            tags = []
        # Use the first tag as the parent; articles with no tags land
        # directly under a synthetic "untagged" leaf.
        tag_path = [str(tags[0])] if tags else ["untagged"]
        builder.add_article(article_id=doc_id, tag_path=tag_path)


def _year_from_unix_seconds(ts: int) -> str | None:
    """Return ``"year_YYYY"`` label for a Unix timestamp, or ``None`` if junk.

    Uses UTC so the bucket boundary is deterministic and matches what the
    decoder sees when it reasons about commit chronology. Zero/negative
    timestamps (the "unknown" sentinel written by older converters) map
    to ``None`` so callers can fall back to a non-year bucket.
    """
    if ts is None or ts <= 0:
        return None
    try:
        year = _dt.datetime.fromtimestamp(int(ts), tz=_dt.UTC).year
    except (OverflowError, OSError, ValueError):
        return None
    # Clamp against clearly-bogus timestamps (e.g. year 1970 from default 0,
    # or far-future sentinels). Git has existed since ~2005 so anything
    # before 2000 is almost certainly a broken commit object.
    if year < 2000 or year > 2100:
        return None
    return f"year_{year}"


def _ingest_git_history_mmap(builder: BrowseTreeBuilder, phase2_dir: Path) -> None:
    """Specialized git_history ingester: ``root → org → org/repo → year``.

    Each commit becomes a leaf article under its ``repo_path``/``year_YYYY``
    group, where the year is derived from the ``commit_ts`` (Unix seconds,
    UTC) column in ``metadata.parquet``. The decoder can then
    ``browse(id="org/repo")`` to see a chronological outline of available
    years, then drill into ``"org/repo/year_2023"`` to list that year's
    commits, then ``bgkit(ids=["org/repo/year_2023/<sha>"], query=...)``.

    A repo with 1000 commits across 10 years produces 10 year-leaves of
    ~100 commits each — the shape ``BrowseTreeBuilder`` is designed for.
    Years with >100 commits trigger the automatic alphabetical
    sub-division into ``year_YYYY/~A`` etc., which is semantically
    meaningful ("year, then alpha bucket") rather than a random hash.

    When the ``commit_ts`` column is absent (older converters) OR a
    particular row's timestamp is unknown/invalid, that article falls
    back to the flat ``[org, "org/repo"]`` path for backwards
    compatibility. A warning is emitted once per conversion when the
    column is entirely missing.
    """
    meta_path = phase2_dir / "metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"git_history metadata missing: {meta_path}. "
            "Run scripts/generate_git_qa.py + "
            "scripts/convert_qa_pairs_to_npy.py first."
        )

    schema = pq.read_schema(meta_path)
    wanted = ["document_id", "repo_path", "commit_sha", "commit_ts"]
    columns = [c for c in wanted if c in schema.names]
    if "repo_path" not in columns:
        raise RuntimeError(
            f"git_history metadata at {meta_path} is missing the "
            "repo_path column — cannot build a repo-rooted browse tree."
        )
    has_commit_ts = "commit_ts" in columns
    if not has_commit_ts:
        _log.warning(
            "git history browse tree falls back to flat commit structure — "
            "run scripts/generate_git_qa.py with the new output schema "
            "(adds commit_ts) to enable year-bucketed navigation."
        )

    table = pq.read_table(meta_path, columns=columns).to_pylist()
    for row in table:
        doc_id = str(row["document_id"])
        repo = str(row.get("repo_path") or "unknown_repo")

        # Split "org/repo" into [org, repo]. The BrowseTreeBuilder
        # prepends each parent's id, so the leaf names we feed in are
        # just the bare org / repo-name / year components; the builder
        # assembles them into ids like "torvalds", "torvalds/linux",
        # "torvalds/linux/year_2020".
        parts = [p for p in repo.split("/") if p]
        if len(parts) >= 2:
            org_name = parts[0]
            repo_leaf = "/".join(parts[1:])  # may contain nested subgroups
            base_path: list[str] = [org_name, repo_leaf]
        elif parts:
            base_path = [parts[0]]
        else:
            base_path = ["unknown_repo"]

        tag_path = list(base_path)
        if has_commit_ts and len(base_path) >= 2:
            year_label = _year_from_unix_seconds(row.get("commit_ts"))
            if year_label is not None:
                tag_path = [*base_path, year_label]
        builder.add_article(article_id=doc_id, tag_path=tag_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--input",
        type=Path,
        help="JSONL with {article_id, tag_path} rows for datasets with "
        "an externally-supplied hierarchy.",
    )
    parser.add_argument(
        "--phase2-dir",
        type=Path,
        help="Phase 2 mmap dir ({DATA_DIR}/mmap/phase2/{dataset}) whose "
        "metadata.parquet drives the build directly.",
    )
    parser.add_argument("--leaf-cap", type=int, default=100)
    parser.add_argument("--fanout-cap", type=int, default=100)
    args = parser.parse_args()

    if not args.input and not args.phase2_dir:
        parser.error("one of --input or --phase2-dir is required")
    if args.input and args.phase2_dir:
        parser.error("--input and --phase2-dir are mutually exclusive")

    builder = BrowseTreeBuilder(TaggingConfig(
        dataset=args.dataset,
        leaf_cap=args.leaf_cap,
        fanout_cap=args.fanout_cap,
    ))
    title_to_doc: dict[str, str] = {}
    if args.input:
        title_to_doc = _ingest_jsonl(builder, args.input)
    elif args.dataset == "git_history":
        _ingest_git_history_mmap(builder, args.phase2_dir)
    else:
        _ingest_phase2_mmap(builder, args.phase2_dir)

    tree = builder.build()
    out = args.output_dir / f"{args.dataset}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.save(out)
    print(f"wrote {out} — {len(tree)} nodes")

    # Emit the title → document_id sidecar whenever the input JSONL
    # provided document_id fields that differ from the browse-tree node
    # ids. The trainer loads this sidecar at setup time to translate
    # decoder-emitted human-readable ids into the canonical mmap key
    # used by the L0 cache and ArticleTokenStore. When every entry is
    # identity (or no document_id was supplied), we skip the sidecar —
    # the absence itself signals "browse-tree ids are already mmap keys".
    nontrivial = any(
        title != doc_id for title, doc_id in title_to_doc.items()
    )
    if nontrivial:
        sidecar_path = args.output_dir / f"{args.dataset}_title_to_doc_id.json"
        with sidecar_path.open("w") as f:
            json.dump(title_to_doc, f, sort_keys=True)
        print(
            f"wrote {sidecar_path} — {len(title_to_doc)} title → document_id entries"
        )


if __name__ == "__main__":
    main()
