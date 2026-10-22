#!/usr/bin/env python3
"""Build a hierarchical tag_path file for PubMedQA articles using the MeSH tree.

Downloads the NLM MeSH ASCII tree file (``mtreesYYYY.bin``) and parses it into
a tree-number <-> term map. Each MeSH term carries one or more tree numbers
(MeSH is a DAG, not a strict tree) like ``C04.588.945.418.365.930`` where each
dot-separated prefix is an ancestor. The top level is a single letter
(``A``..``Z``) that maps to one of the 16 canonical MeSH categories
("Anatomy", "Organisms", "Diseases", ...).

For each PubMedQA article, this script:

1. Loads the raw PubMedQA dataset from HF (``qiaojin/PubMedQA``).
2. Extracts the mesh term list from ``record['context']['meshes']``.
3. Picks the mesh term with the shortest tree number as the canonical hook
   (most generic — closer to the root — so paths stay meaningful).
4. Decodes that tree number into a named ancestor chain by looking up each
   dot-separated prefix.
5. Writes ``{"article_id": title_or_pubid,
             "document_id": pubid,
             "tag_path": [root -> leaf]}``.
   ``article_id`` is the browse-tree node id: human-readable where we can
   obtain a title, and the numeric pubid otherwise. PubMedQA rows on HF
   do not carry a canonical article title field, so this script falls
   back to ``pubid`` and emits a warning summarizing how many rows used
   the fallback. ``document_id`` is always the mmap key.

Articles with no MeSH terms fall through to ``["Medical", "Uncategorized"]``.

Per-run title collisions (different pubids sharing a title) are
disambiguated by suffixing ``" (disambiguation N)"`` so the browse tree
has unique node IDs.

Run:
    python scripts/build_mesh_hierarchy.py --output $DATA_DIR/taxonomies/pubmedqa_hierarchy.jsonl

Optional flags:
    --mesh-file PATH   Reuse an existing mtrees*.bin file instead of downloading.
    --mesh-year YEAR   Override the MeSH year (default: 2025).
    --splits LIST      Comma-separated PubMedQA splits (default: ``labeled,artificial``).
    --limit N          Stop after N articles (useful for smoke-testing).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

# Canonical names for the 16 top-level MeSH letters. These match the section
# headings of the MeSH tree and are what users expect at the top of a browse
# tree. The letters come from the first character of each tree number.
_MESH_LETTER_ROOTS: dict[str, str] = {
    "A": "Anatomy",
    "B": "Organisms",
    "C": "Diseases",
    "D": "Chemicals and Drugs",
    "E": "Analytical, Diagnostic and Therapeutic Techniques, and Equipment",
    "F": "Psychiatry and Psychology",
    "G": "Phenomena and Processes",
    "H": "Disciplines and Occupations",
    "I": "Anthropology, Education, Sociology, and Social Phenomena",
    "J": "Technology, Industry, and Agriculture",
    "K": "Humanities",
    "L": "Information Science",
    "M": "Named Groups",
    "N": "Health Care",
    "V": "Publication Characteristics",
    "Z": "Geographicals",
}

_DEFAULT_MESH_YEAR = 2025
_MESH_URL_TEMPLATE = "https://nlmpubs.nlm.nih.gov/projects/mesh/{year}/meshtrees/mtrees{year}.bin"

# Letter priority for picking the "main topic" mesh term of an article.
# Lower number = higher preference. PubMedQA articles always carry B
# (Organisms: Humans, Animals) and M (Named Groups: Male, Female) tags;
# these bury the actual subject if we just pick the deepest tree number.
# Prefer disease (C), chemical/drug (D), analytical technique (E), psychology
# (F), phenomena (G) first. Fall back to A anatomy, then demographic tags.
_LETTER_PRIORITY: dict[str, int] = {
    "C": 0,  # Diseases
    "D": 1,  # Chemicals and Drugs
    "F": 2,  # Psychiatry and Psychology
    "E": 3,  # Analytical/Diagnostic/Therapeutic Techniques
    "G": 4,  # Phenomena and Processes
    "A": 5,  # Anatomy
    "N": 6,  # Health Care
    "H": 7,  # Disciplines
    "I": 8,  # Social phenomena
    "J": 9,  # Technology/industry
    "K": 10,  # Humanities
    "L": 11,  # Information Science
    "B": 12,  # Organisms (down-ranked: Humans/Animals swamp everything)
    "M": 13,  # Named Groups (Male, Adult, etc.)
    "V": 14,  # Publication characteristics
    "Z": 15,  # Geographical
}


def download_mesh_file(year: int, target: Path) -> Path:
    """Fetch the NLM MeSH ASCII tree file into ``target``.

    Uses urllib directly so we don't add a new runtime dep. ~2.7 MB file.
    """
    url = _MESH_URL_TEMPLATE.format(year=year)
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mesh] downloading {url} -> {target}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    target.write_bytes(data)
    print(f"[mesh] wrote {len(data):,} bytes", file=sys.stderr)
    return target


def parse_mesh_tree(mesh_path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Parse ``mtrees*.bin`` into two maps.

    Returns:
        tree_to_term: ``tree_number -> term name`` (e.g. ``A01.236 -> "Breast"``).
        term_to_trees: ``term name -> list[tree_number]`` (a MeSH term can appear
            multiple times in the DAG).
    """
    tree_to_term: dict[str, str] = {}
    term_to_trees: dict[str, list[str]] = defaultdict(list)
    with mesh_path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n").lstrip(" ")
            if not line:
                continue
            # Format: "<Term Name>;<Tree Number>". The term may contain commas.
            if ";" not in line:
                continue
            term, tree = line.rsplit(";", 1)
            term = term.strip()
            tree = tree.strip()
            if not term or not tree:
                continue
            tree_to_term[tree] = term
            term_to_trees[term].append(tree)
    return tree_to_term, dict(term_to_trees)


def decode_tree_number(
    tree: str,
    tree_to_term: dict[str, str],
) -> list[str]:
    """Turn a tree number like ``C04.588.180`` into a named root->leaf chain.

    Walks every dot-separated prefix, looks up the term for each, and
    prepends the canonical MeSH letter root. Any prefix that's missing a
    term (rare) is kept as the raw tree number so the caller can tell.
    """
    segments = tree.split(".")
    path: list[str] = []
    root_letter = tree[0]
    root_name = _MESH_LETTER_ROOTS.get(root_letter)
    if root_name:
        path.append(root_name)

    accum: list[str] = []
    for seg in segments:
        accum.append(seg)
        prefix = ".".join(accum)
        term = tree_to_term.get(prefix)
        if term:
            path.append(term)
        else:
            # Unknown prefix -- keep as-is so we don't silently corrupt paths.
            path.append(prefix)
    return path


def pick_canonical_tree(
    meshes: list[str],
    term_to_trees: dict[str, list[str]],
) -> list[str] | None:
    """Pick the best named tree path for an article from its mesh list.

    Heuristic: rank every tree number by ``(letter_priority, -depth, tree)``.
    Lowest ``letter_priority`` wins first (so Diseases/Chemicals beat
    Organisms/Demographics), then deepest tree in that letter, then
    alphabetical on tree number for determinism. Returns the raw tree
    number segments or ``None`` if no MeSH term is in the tree.
    """
    candidates: list[tuple[int, int, str]] = []
    for term in meshes:
        trees = term_to_trees.get(term)
        if not trees:
            continue
        for t in trees:
            letter = t[0]
            prio = _LETTER_PRIORITY.get(letter, 99)
            depth = len(t.split("."))
            candidates.append((prio, -depth, t))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2].split(".")


def build_tag_path(
    meshes: list[str],
    tree_to_term: dict[str, str],
    term_to_trees: dict[str, list[str]],
) -> list[str]:
    """Return the root->leaf tag path for a PubMedQA article.

    Falls back to ``["Medical", "Uncategorized"]`` when no mesh term has a
    tree number. That still gives the browse tree *something* to work with.
    """
    segs = pick_canonical_tree(meshes, term_to_trees)
    if segs is None:
        return ["Medical", "Uncategorized"]
    tree = ".".join(segs)
    return decode_tree_number(tree, tree_to_term)


def iter_pubmedqa_records(splits: list[str]):
    """Yield ``(pubid, title_or_none, mesh_list)`` tuples from PubMedQA.

    The PubMedQA HF dataset uses per-split *configs* (``pqa_labeled``,
    ``pqa_artificial``, ``pqa_unlabeled``) each with a single ``train`` split.
    So we pass ``name=`` and always use ``split='train'``.

    PubMedQA rows do not expose a dedicated article-title field on HF
    (the canonical columns are ``pubid``, ``question``, ``context``,
    ``long_answer``, ``final_decision``). We still probe several
    commonly-seen alternatives in case future dataset versions add one;
    rows without any title field yield ``None`` and the caller treats
    that as "fall back to pubid".
    """
    from datasets import load_dataset

    title_candidates = ("title", "article_title", "paper_title")

    for split_name in splits:
        config = f"pqa_{split_name}"
        print(f"[pubmedqa] loading config {config}", file=sys.stderr)
        ds = load_dataset(
            "qiaojin/PubMedQA",
            name=config,
            split="train",
            streaming=True,
        )
        for rec in ds:
            pubid = rec.get("pubid")
            if pubid is None:
                continue
            ctx = rec.get("context") or {}
            meshes = ctx.get("meshes") if isinstance(ctx, dict) else None
            if not isinstance(meshes, list):
                meshes = []
            title: str | None = None
            for key in title_candidates:
                value = rec.get(key)
                if isinstance(value, str) and value.strip():
                    title = value.strip()
                    break
            yield str(pubid), title, [str(m) for m in meshes if m]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mesh-file", type=Path, default=None)
    p.add_argument("--mesh-year", type=int, default=_DEFAULT_MESH_YEAR)
    p.add_argument(
        "--splits",
        type=str,
        default="labeled,artificial",
        help="Comma-separated list of PubMedQA split suffixes.",
    )
    p.add_argument("--limit", type=int, default=0, help="Stop after N articles (0 = all).")
    args = p.parse_args()

    mesh_path = args.mesh_file
    if mesh_path is None or not mesh_path.exists():
        default = Path("/tmp/mesh") / f"mtrees{args.mesh_year}.bin"
        mesh_path = default if default.exists() else download_mesh_file(args.mesh_year, default)

    print(f"[mesh] parsing {mesh_path}", file=sys.stderr)
    tree_to_term, term_to_trees = parse_mesh_tree(mesh_path)
    print(
        f"[mesh] {len(tree_to_term):,} tree numbers, {len(term_to_trees):,} unique terms",
        file=sys.stderr,
    )

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_fallback = 0
    n_total = 0
    n_title_fallback = 0
    n_disambiguated = 0
    title_counts: dict[str, int] = {}
    with args.output.open("w") as out:
        for pubid, title, meshes in iter_pubmedqa_records(splits):
            n_total += 1
            tag_path = build_tag_path(meshes, tree_to_term, term_to_trees)
            if tag_path == ["Medical", "Uncategorized"]:
                n_fallback += 1

            if title is None:
                # PubMedQA HF rows don't carry a canonical title field;
                # fall back to the numeric pubid. This keeps the pipeline
                # working but means the decoder will have to emit digits
                # for drill-down on PubMedQA. A separate title-extraction
                # pass (e.g. an NCBI pubmed lookup) is needed to unlock
                # true human-readable IDs for this dataset.
                base_id = pubid
                n_title_fallback += 1
            else:
                base_id = title
            occ = title_counts.get(base_id, 0)
            title_counts[base_id] = occ + 1
            if occ == 0:
                article_id = base_id
            else:
                article_id = f"{base_id} (disambiguation {occ})"
                n_disambiguated += 1

            out.write(
                json.dumps({
                    "article_id": article_id,
                    "document_id": pubid,
                    "tag_path": tag_path,
                }) + "\n"
            )
            n_written += 1
            if n_written % 10000 == 0:
                print(f"[pubmedqa] {n_written:,} rows", file=sys.stderr)
            if args.limit and n_written >= args.limit:
                break

    if n_title_fallback and n_title_fallback == n_total:
        print(
            "[pubmedqa] WARNING: none of the PubMedQA rows exposed a title "
            "field — every article_id falls back to the numeric pubid. "
            "The decoder will see digit-tokenized IDs for drill-down. To "
            "unlock human-readable PubMedQA titles, add a title-enrichment "
            "pass (e.g. an NCBI E-Utilities lookup) before this step.",
            file=sys.stderr,
        )
    elif n_title_fallback:
        print(
            f"[pubmedqa] WARNING: {n_title_fallback:,}/{n_total:,} rows had "
            "no title field and fell back to numeric pubid as article_id.",
            file=sys.stderr,
        )
    print(
        f"[pubmedqa] done: {n_written:,} rows ({n_fallback:,} fallback, "
        f"{n_total - n_fallback:,} hierarchical, "
        f"disambig={n_disambiguated:,})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
