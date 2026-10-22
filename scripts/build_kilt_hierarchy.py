#!/usr/bin/env python3
"""Build a hierarchical tag_path file for KILT Wikipedia articles.

KILT's Wikipedia dump attaches a *flat* list of categories to each article
(e.g. ``"Academy Awards,Best Art Direction Academy Award winners"``) — just
leaf-level category names, no hierarchy. To get a real browse tree we need
the Wikipedia category DAG (each category's parent categories).

We use the **DBpedia SKOS categories** dump, which is a pre-extracted form
of Wikipedia's category→parent DAG serialized as N-triples. It's about
52 MB compressed, which is dramatically smaller than the 8 GB+ Wikipedia
``categorylinks.sql.gz`` dump and has already been tidied into the SKOS
``broader`` vocabulary. The canonical top of the Wikipedia category DAG is
``Category:Main_topic_classifications`` (39 direct children like "Science",
"History", "Mathematics", "Sports", ...).

Algorithm:

1. Download (or reuse) ``categories_lang=en_skos.ttl.bz2`` from DBpedia.
2. Stream-parse the triples into:
   - ``child -> list[parent]`` via ``skos:broader``
   - ``uri -> prefLabel`` via ``skos:prefLabel``
   - ``label -> uri`` (inverse of prefLabel, for normalizing KILT's
     display-name categories back onto URIs).
3. BFS from ``Category:Main_topic_classifications`` through ``broader^-1``
   (i.e. treat parent-of as the forward edge and walk descendants) to
   compute, for every reachable category, a *shortest path from root*.
   Cache it as an ordered list of prefLabels.
4. Stream the KILT Wikipedia HF dataset. For each article:
   a. Parse its ``categories`` field into a list of display names.
   b. Normalize each name to a category URI via ``label -> uri``.
   c. Look up each URI's cached root-path. Pick the **shortest** non-empty
      path so article hooks attach near the top where useful ancestors
      exist; within ties pick alphabetically for determinism.
   d. Write ``{"article_id": wikipedia_title,
                "document_id": wikipedia_id,
                "tag_path": [root->leaf]}``.
      ``article_id`` is the human-readable title that the browse tree
      uses as its node ID (so the decoder can memorize and emit it at
      training/inference time); ``document_id`` is the canonical numeric
      mmap key the L0 cache and ``ArticleTokenStore`` continue to use.
      The downstream browse-tree builder turns this pair into a
      title → document_id sidecar file so the trainer can translate
      decoder-emitted titles back to mmap keys.
5. Articles with no matching category fall through to
   ``["Wikipedia", "Uncategorized"]``.
6. Per-run title collisions (different wikipedia_ids sharing a title)
   are disambiguated by suffixing ``" (disambiguation N)"`` so the
   browse tree still has unique node IDs.

Run:
    python scripts/build_kilt_hierarchy.py \
        --output $DATA_DIR/taxonomies/kilt_wikipedia_hierarchy.jsonl
"""

from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from collections.abc import Iterator
from pathlib import Path

_DBPEDIA_SKOS_URL = (
    "https://downloads.dbpedia.org/repo/dbpedia/generic/"
    "categories/2022.12.01/categories_lang=en_skos.ttl.bz2"
)

_KILT_JSONL_URL = (
    "https://huggingface.co/datasets/orionweller/kilt_wikipedia_split/resolve/main/train_0.jsonl"
)

_ROOT_URI = "http://dbpedia.org/resource/Category:Main_topic_classifications"
_ROOT_LABEL = "Wikipedia"

# Regex to parse an N-triple line. SKOS category lines look like:
#   <subj_uri> <pred_uri> <obj_uri_or_literal> .
_TRIPLE_RE = re.compile(
    r"^<([^>]+)>\s+<([^>]+)>\s+(<[^>]+>|\"(?:[^\"\\]|\\.)*\"(?:@\w+)?)\s*\.\s*$"
)

_PRED_BROADER = "http://www.w3.org/2004/02/skos/core#broader"
_PRED_PREFLABEL = "http://www.w3.org/2004/02/skos/core#prefLabel"


def download_skos(target: Path) -> Path:
    """Fetch the DBpedia English SKOS category dump into ``target``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[kilt] downloading {_DBPEDIA_SKOS_URL} -> {target}", file=sys.stderr)
    # Stream to disk in 4 MB chunks — 52 MB file, keeps memory flat.
    with (
        urllib.request.urlopen(_DBPEDIA_SKOS_URL, timeout=120) as resp,
        target.open("wb") as f,
    ):
        while True:
            chunk = resp.read(4 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    print(f"[kilt] wrote {target.stat().st_size:,} bytes", file=sys.stderr)
    return target


def _decode_literal(raw: str) -> str:
    """Decode an N-triple literal ``"foo"@en`` -> ``foo``.

    Strips language tag, unescapes ``\\"`` and ``\\\\``. Good enough for
    Wikipedia category labels.
    """
    m = re.match(r'^"((?:[^"\\]|\\.)*)"', raw)
    if not m:
        return raw
    body = m.group(1)
    return body.replace('\\"', '"').replace("\\\\", "\\")


def parse_skos(
    skos_path: Path,
) -> tuple[dict[str, set[str]], dict[str, str], dict[str, str]]:
    """Parse the SKOS .ttl.bz2 file into three maps.

    Returns:
        child_to_parents: ``child_uri -> set[parent_uri]`` from ``skos:broader``.
        uri_to_label:     ``uri -> prefLabel`` (human-readable name).
        label_to_uri:     ``lowercase label -> uri`` (inverse of above).
            When the same label maps to multiple URIs we keep the first one
            seen; that's fine for our purposes because prefLabels are
            mostly unique.
    """
    child_to_parents: dict[str, set[str]] = defaultdict(set)
    uri_to_label: dict[str, str] = {}
    label_to_uri: dict[str, str] = {}

    n_lines = 0
    n_broader = 0
    n_pref = 0
    print(f"[kilt] parsing {skos_path} (bz2 stream)", file=sys.stderr)
    with bz2.open(skos_path, "rt", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            m = _TRIPLE_RE.match(line)
            if not m:
                continue
            subj, pred, obj = m.group(1), m.group(2), m.group(3)
            if pred == _PRED_BROADER:
                if not obj.startswith("<"):
                    continue
                parent = obj[1:-1]
                child_to_parents[subj].add(parent)
                n_broader += 1
            elif pred == _PRED_PREFLABEL:
                label = _decode_literal(obj)
                if label:
                    uri_to_label[subj] = label
                    key = label.lower()
                    label_to_uri.setdefault(key, subj)
                    n_pref += 1
            if n_lines % 1_000_000 == 0:
                print(
                    f"[kilt]   parsed {n_lines:,} lines "
                    f"({n_broader:,} broader, {n_pref:,} prefLabel)",
                    file=sys.stderr,
                )

    print(
        f"[kilt] total {n_lines:,} lines, {len(uri_to_label):,} categories, "
        f"{n_broader:,} broader edges",
        file=sys.stderr,
    )
    return dict(child_to_parents), uri_to_label, label_to_uri


def compute_root_paths(
    child_to_parents: dict[str, set[str]],
    uri_to_label: dict[str, str],
    root_uri: str = _ROOT_URI,
    max_depth: int = 8,
) -> dict[str, list[str]]:
    """For every reachable category URI, compute a shortest path *from root*.

    We walk from ``root_uri`` through ``broader^-1`` (reverse the parent
    edges). ``child_to_parents`` gives child->parent, so the descendant
    map is the inverse: every ``(child, parent)`` pair contributes an edge
    ``parent -> child`` in the BFS. Paths are stored as ordered lists of
    ``prefLabel`` strings, root first, leaf last.

    ``max_depth`` caps the BFS; Wikipedia categories can recurse through
    cycles via "By country", "By year", etc., so we cut off to keep the
    reachable set bounded and paths interpretable.
    """
    # Build inverse adjacency: parent -> set of children.
    parent_to_children: dict[str, list[str]] = defaultdict(list)
    for child, parents in child_to_parents.items():
        for p in parents:
            parent_to_children[p].append(child)

    paths: dict[str, list[str]] = {}
    root_label = uri_to_label.get(root_uri, _ROOT_LABEL)
    paths[root_uri] = [root_label]

    # BFS. When we first reach a URI, the path we already have is the
    # shortest one (BFS property). Subsequent shorter discoveries can't
    # happen.
    queue: deque[tuple[str, int]] = deque([(root_uri, 0)])
    while queue:
        uri, depth = queue.popleft()
        if depth >= max_depth:
            continue
        current_path = paths[uri]
        for child in parent_to_children.get(uri, ()):
            if child in paths:
                continue
            label = uri_to_label.get(child)
            if not label:
                continue
            paths[child] = [*current_path, label]
            queue.append((child, depth + 1))

    print(
        f"[kilt] BFS reached {len(paths):,} categories from {root_label} (max depth {max_depth})",
        file=sys.stderr,
    )
    return paths


def _normalize_label_to_uri_key(name: str) -> str:
    """Lowercase + strip whitespace so KILT display names match prefLabels."""
    return name.strip().lower()


def _label_to_uri_fallback(
    name: str,
    label_to_uri: dict[str, str],
) -> str | None:
    """Try a few ways to find ``Category:<name>`` in the URI map.

    KILT's raw category field uses human-readable names identical to the
    Wikipedia display title (e.g. ``"Academy Awards"``). DBpedia prefLabels
    also use the display form (``"Academy Awards"@en``), so a direct
    case-insensitive lookup handles most articles. For the rest we fall
    back to constructing a synthetic URI and seeing if it's in the DAG.
    """
    key = _normalize_label_to_uri_key(name)
    uri = label_to_uri.get(key)
    if uri:
        return uri
    # Try constructing the URI directly. Wikipedia URIs use underscores
    # and URL encoding.
    slug = name.strip().replace(" ", "_")
    candidate = "http://dbpedia.org/resource/Category:" + urllib.parse.quote(slug, safe="_,()")
    # We don't have a "reachable" set at this level; just return the
    # candidate URI and let the caller check it against the root-path map.
    return candidate


def _find_extended_path(
    start_uri: str,
    child_to_parents: dict[str, set[str]],
    uri_to_label: dict[str, str],
    root_paths: dict[str, list[str]],
    max_steps: int = 6,
) -> list[str] | None:
    """Walk upward from ``start_uri`` through ``skos:broader`` until an
    ancestor with a cached root path is found. Then return
    ``root_path + [intermediate_labels..., start_label]``.

    This handles "orphan" categories (like ``1995 albums``) whose own
    ``broader`` edge is missing from the DBpedia SKOS dump: we fall back
    to BFS through the parent graph to find an ancestor that IS in the
    tree. The returned path is still root->leaf ordered.
    """
    start_label = uri_to_label.get(start_uri)
    if not start_label:
        return None
    # Already in the tree? Return the cached path directly.
    if start_uri in root_paths:
        return root_paths[start_uri]

    # BFS up the parent DAG, tracking the reverse chain back to start.
    visited: set[str] = {start_uri}
    # Queue entries are (uri, ordered_chain_from_uri_down_to_start_exclusive).
    # ordered_chain_from_uri_down_to_start_exclusive is the list of labels
    # *between* ``uri`` and ``start_uri`` (not including either endpoint).
    queue: deque[tuple[str, list[str]]] = deque([(start_uri, [])])
    steps = 0
    while queue and steps < max_steps:
        steps += 1
        for _ in range(len(queue)):
            uri, tail = queue.popleft()
            parents = child_to_parents.get(uri, set())
            for parent in parents:
                if parent in visited:
                    continue
                visited.add(parent)
                parent_label = uri_to_label.get(parent)
                if not parent_label:
                    continue
                if parent in root_paths:
                    # Compose: root_path + reversed(tail) + [start_label]
                    base = root_paths[parent]
                    mid = list(reversed(tail))
                    return [*base, *mid, start_label]
                new_tail = [*tail, parent_label]
                queue.append((parent, new_tail))
    return None


def pick_tag_path(
    raw_categories: list[str],
    label_to_uri: dict[str, str],
    root_paths: dict[str, list[str]],
    child_to_parents: dict[str, set[str]],
    uri_to_label: dict[str, str],
) -> list[str] | None:
    """Choose the best tag path for a KILT article.

    For each raw category:
      1. Resolve the KILT display name to a DBpedia URI via
         ``label_to_uri`` (case-insensitive).
      2. If the URI has a cached root-path, use that directly.
      3. Otherwise walk upward through ``skos:broader`` until an
         ancestor with a cached root-path is found; extend that root
         path with the intermediate labels. This handles orphan
         categories (like "1995 albums") whose own ``broader`` edge is
         missing from the DBpedia dump.
    Among the article's resolved candidates, pick the **shortest** root
    path so articles anchor near the top-level buckets; within ties,
    alphabetical for determinism.
    """
    candidates: list[tuple[int, list[str]]] = []
    for raw in raw_categories:
        uri = _label_to_uri_fallback(raw, label_to_uri)
        if uri is None:
            continue
        path = root_paths.get(uri)
        if path is None:
            path = _find_extended_path(
                uri,
                child_to_parents,
                uri_to_label,
                root_paths,
            )
        if path:
            candidates.append((len(path), path))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: (pair[0], pair[1]))
    return candidates[0][1]


def download_kilt_jsonl(target: Path) -> Path:
    """Fetch the raw KILT Wikipedia JSONL file from HF directly.

    This is ~2.16 GB and is much faster than streaming via the
    ``datasets`` library for our use case — we only need two columns
    (``wikipedia_id`` and ``categories``) and can skip the rest of the
    record with a cheap JSON parse.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[kilt] downloading {_KILT_JSONL_URL} -> {target}", file=sys.stderr)
    req = urllib.request.Request(
        _KILT_JSONL_URL,
        headers={"User-Agent": "bgkit/taxonomy-builder"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        total = 0
        with target.open("wb") as f:
            while True:
                chunk = resp.read(4 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                if total % (256 * 1024 * 1024) < 4 * 1024 * 1024:
                    print(f"[kilt]   downloaded {total // (1024 * 1024)} MB", file=sys.stderr)
    print(f"[kilt] wrote {target.stat().st_size:,} bytes", file=sys.stderr)
    return target


def iter_kilt_records(
    limit: int = 0,
    kilt_jsonl: Path | None = None,
) -> Iterator[tuple[str, str, list[str]]]:
    """Yield ``(wikipedia_id, wikipedia_title, category_list)`` from the
    KILT Wikipedia split.

    If ``kilt_jsonl`` points to a cached local file, use it directly. This
    is ~5x faster than the datasets library streaming mode because we skip
    the text/anchors/history fields entirely.

    KILT Wikipedia is pre-split by paragraph, so the same ``wikipedia_id``
    can appear in dozens of consecutive rows. We de-dupe on the fly.
    """
    seen: set[str] = set()

    def _normalize(rec: dict) -> tuple[str, str, list[str]] | None:
        wid = rec.get("wikipedia_id")
        if wid is None:
            return None
        wid = str(wid)
        if wid in seen:
            return None
        seen.add(wid)
        title = rec.get("wikipedia_title")
        # Fall back to wikipedia_id when the title is missing so the row
        # still participates in the browse tree (under its numeric id).
        title_str = str(title).strip() if title else wid
        if not title_str:
            title_str = wid
        raw = rec.get("categories") or ""
        if isinstance(raw, list):
            cats = [str(c).strip() for c in raw if c]
        else:
            cats = [c.strip() for c in str(raw).split(",") if c.strip()]
        return wid, title_str, cats

    n = 0
    if kilt_jsonl is not None and kilt_jsonl.exists():
        print(f"[kilt] reading {kilt_jsonl}", file=sys.stderr)
        with kilt_jsonl.open("r", encoding="utf-8") as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    rec = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                out = _normalize(rec)
                if out is None:
                    continue
                yield out
                n += 1
                if limit and n >= limit:
                    return
        return

    from datasets import load_dataset

    print("[kilt] streaming orionweller/kilt_wikipedia_split", file=sys.stderr)
    ds = load_dataset(
        "orionweller/kilt_wikipedia_split",
        split="train",
        streaming=True,
    )
    for rec in ds:
        out = _normalize(rec)
        if out is None:
            continue
        yield out
        n += 1
        if limit and n >= limit:
            return


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--skos-file", type=Path, default=None)
    p.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Max BFS depth from Main_topic_classifications (default: 6).",
    )
    p.add_argument("--limit", type=int, default=0, help="Stop after N articles (0 = all).")
    p.add_argument(
        "--kilt-jsonl",
        type=Path,
        default=None,
        help="Local copy of train_0.jsonl. If missing, download from HF.",
    )
    p.add_argument(
        "--skip-kilt-download",
        action="store_true",
        help="Use the streaming datasets library instead of downloading the "
        "2.16 GB JSONL file. Slower but uses less disk.",
    )
    args = p.parse_args()

    skos_path = args.skos_file
    if skos_path is None or not skos_path.exists():
        default = Path("/tmp/kilt_tax") / "skos.ttl.bz2"
        skos_path = default if default.exists() else download_skos(default)

    child_to_parents, uri_to_label, label_to_uri = parse_skos(skos_path)
    root_paths = compute_root_paths(
        child_to_parents,
        uri_to_label,
        max_depth=args.max_depth,
    )

    kilt_jsonl = args.kilt_jsonl
    if not args.skip_kilt_download:
        if kilt_jsonl is None:
            kilt_jsonl = Path("/tmp/kilt_tax") / "train_0.jsonl"
        if not kilt_jsonl.exists():
            download_kilt_jsonl(kilt_jsonl)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_fallback_empty = 0  # articles with no category field at all
    n_fallback_unmatched = 0  # had categories but none reached root via BFS
    # Track title occurrences so we can disambiguate collisions on the fly.
    # KILT's wikipedia_title is *mostly* unique but redirects + sandbox
    # pages can share a human-readable label. The browse tree needs every
    # article to resolve to a unique node id, so we suffix
    # ``" (disambiguation N)"`` on the second+ occurrence.
    title_counts: dict[str, int] = {}
    n_disambiguated = 0
    with args.output.open("w") as out:
        for wid, title, cats in iter_kilt_records(
            limit=args.limit, kilt_jsonl=kilt_jsonl,
        ):
            tag_path = pick_tag_path(
                cats,
                label_to_uri,
                root_paths,
                child_to_parents,
                uri_to_label,
            )
            if tag_path is None:
                tag_path = ["Wikipedia", "Uncategorized"]
                if cats:
                    n_fallback_unmatched += 1
                else:
                    n_fallback_empty += 1

            occ = title_counts.get(title, 0)
            title_counts[title] = occ + 1
            if occ == 0:
                article_id = title
            else:
                article_id = f"{title} (disambiguation {occ})"
                n_disambiguated += 1

            out.write(
                json.dumps({
                    "article_id": article_id,
                    "document_id": wid,
                    "tag_path": tag_path,
                }) + "\n"
            )
            n_written += 1
            if n_written % 50000 == 0:
                print(
                    f"[kilt] {n_written:,} rows "
                    f"(empty={n_fallback_empty:,}, unmatched={n_fallback_unmatched:,}, "
                    f"disambig={n_disambiguated:,})",
                    file=sys.stderr,
                )

    n_fallback = n_fallback_empty + n_fallback_unmatched
    hierarchical = n_written - n_fallback
    print(
        f"[kilt] done: {n_written:,} rows -> {args.output}\n"
        f"       hierarchical: {hierarchical:,}\n"
        f"       fallback (no categories): {n_fallback_empty:,}\n"
        f"       fallback (categories not in DAG): {n_fallback_unmatched:,}\n"
        f"       disambiguated titles: {n_disambiguated:,}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
