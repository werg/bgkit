#!/usr/bin/env python3
"""Filter synthetic QA pairs and annotate per-pair answer-position masks.

For each `(question, answer, repo_path, file_path, commit_sha)` record:

1. Re-extract the source file at the recorded commit via pygit2.
2. Tokenize the source with character offset mapping.
3. Extract long-enough identifiers from the answer (`[A-Za-z_][A-Za-z0-9_]{3,}`).
4. For each identifier, find all char-spans in the source and map them to
   the covering source-token indices.
5. Keep the pair iff at least ``--min-identifier-hits`` distinct identifiers
   appear in the source AND the resulting position set has at least
   ``--min-positions`` tokens.
6. Write a filtered JSONL with two new fields per row:
   - ``answer_position_indices``: sorted list[int] of source token positions
   - ``answer_position_meta``: {n_identifier_hits, n_positions, source_token_count}

The downstream converter (``convert_qa_pairs_to_npy.py``) consumes these
fields to build the position-mask mmap that the Phase 1 Step 4
(QA-conditioned head supervision) trainer uses to supervise the
survivorship head directly.

The 2026-04-17 audit found that the existing 114k synthetic QA pairs are
0% extractive (no answer is a literal span). About 25-35% are meaningfully
grounded in the source (identifiers in the answer also appear in the
file). This script materializes that grounded subset and turns it into
position-annotated training data.

Usage:
    python scripts/filter_qa_with_positions.py \\
        --input-dir $DATA_DIR/qa_pairs/ \\
        --repos-dir $DATA_DIR/repos/ \\
        --output-dir $DATA_DIR/qa_pairs_filtered/ \\
        --tokenizer Qwen/Qwen3.5-0.8B \\
        --min-identifier-hits 2 \\
        --min-positions 1
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from pathlib import Path

# Allow running without editable install
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import pygit2

# Identifier regex: starts with letter/underscore, ≥4 chars total. The
# 4-char floor cuts generic short words (`if`, `for`, `def`, `set`, `get`)
# that appear everywhere in source code and would create huge position
# sets dominated by structural noise rather than answer-specific content.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")

# Common English / programming stop words that pass the 4-char regex but
# shouldn't count as content-anchoring identifiers. Filtered post-regex.
_STOP_IDENTS = frozenset({
    "this", "that", "these", "those", "there", "their", "they",
    "with", "from", "into", "have", "been", "were", "will", "shall",
    "would", "could", "should", "might", "must",
    "function", "method", "class", "object", "value", "values", "type",
    "types", "string", "strings", "number", "true", "false", "null",
    "none", "return", "returns", "param", "params", "args", "kwargs",
    "self", "cls", "init", "main",
    "code", "file", "files", "line", "lines", "block", "blocks",
    "data", "list", "dict", "tuple", "array", "input", "output",
    "print", "true", "false", "test", "tests", "case", "cases",
    "name", "names", "path", "paths", "key", "keys", "item", "items",
    "field", "fields", "size", "length", "count", "result", "results",
    "is", "the", "and", "or", "for", "to", "of", "in", "on", "at", "as",
    "be", "by", "an", "if", "it", "its",
})


def _build_repo_lookup(repos_dir: Path) -> dict[str, Path]:
    """Map ``owner/repo`` → repo path on disk."""
    lookup: dict[str, Path] = {}
    if not repos_dir.is_dir():
        return lookup
    for owner in repos_dir.iterdir():
        if not owner.is_dir():
            continue
        for repo in owner.iterdir():
            if (repo / ".git").exists() or (repo / "HEAD").exists():
                lookup[f"{owner.name}/{repo.name}"] = repo
    return lookup


def _read_blob_at_commit(
    repo: pygit2.Repository, commit_sha: str, file_path: str,
) -> bytes | None:
    """Walk the tree at ``commit_sha`` to ``file_path``; return blob bytes or None."""
    try:
        commit = repo.get(commit_sha)
        if commit is None or commit.type_str != "commit":
            return None
        tree = commit.tree
        for part in file_path.split("/"):
            if not part:
                continue
            try:
                entry = tree[part]
            except KeyError:
                return None
            obj = repo.get(entry.id)
            if obj is None:
                return None
            if obj.type_str == "tree":
                tree = obj
            elif obj.type_str == "blob":
                return obj.data
            else:
                return None
        return None
    except (pygit2.GitError, ValueError, KeyError):
        return None


def _is_code_shaped(ident: str) -> bool:
    """True iff ``ident`` looks like a code identifier (not a plain English word).

    Code-shape signals:
      - contains an underscore (``snake_case``, ``CONSTANT_NAME``)
      - contains a digit (``arg2``, ``utf8``)
      - CamelCase: lowercase letter immediately followed by uppercase
      - ALL_CAPS of length ≥ 4 (acronyms / constants)

    Plain-English answer paraphrases (``Android``, ``possible``, ``build``,
    ``configuration``) fail the test, so they don't count toward the
    grounding signal even if they happen to appear in the file.
    """
    if "_" in ident:
        return True
    if any(c.isdigit() for c in ident):
        return True
    if ident.isupper() and len(ident) >= 4:
        return True
    # CamelCase: lowercase-followed-by-uppercase at any internal boundary.
    for i in range(len(ident) - 1):
        if ident[i].islower() and ident[i + 1].isupper():
            return True
    return False


def _extract_grounding_identifiers(answer: str) -> tuple[list[str], list[str]]:
    """Extract grounding identifiers; return ``(all_idents, code_idents)``.

    All identifiers pass the regex + stop-word filter. ``code_idents`` is
    the subset that is code-shaped (per :func:`_is_code_shaped`). The
    main filter requires AT LEAST ONE code-shaped hit so plain-English
    paraphrases of the question can't ground a pair on their own.
    """
    candidates = _IDENT_RE.findall(answer)
    seen: set[str] = set()
    all_idents: list[str] = []
    for c in candidates:
        if c.lower() in _STOP_IDENTS:
            continue
        if c in seen:
            continue
        seen.add(c)
        all_idents.append(c)
    code_idents = [c for c in all_idents if _is_code_shaped(c)]
    return all_idents, code_idents


def _positions_for_identifier(
    source_text: str,
    ident: str,
    token_starts: list[int],
    token_ends: list[int],
) -> set[int]:
    """Return source-token indices covering any occurrence of ``ident``."""
    out: set[int] = set()
    pos = 0
    ident_len = len(ident)
    while True:
        char_pos = source_text.find(ident, pos)
        if char_pos < 0:
            break
        char_end = char_pos + ident_len
        # First token whose end is strictly after char_pos.
        first_tok = bisect.bisect_right(token_ends, char_pos)
        for tok_idx in range(first_tok, len(token_starts)):
            if token_starts[tok_idx] >= char_end:
                break
            out.add(tok_idx)
        # Advance past this occurrence (avoid infinite loops on overlapping matches).
        pos = char_pos + max(1, ident_len)
    return out


def _process_record(
    record: dict,
    repo_lookup: dict[str, Path],
    tokenizer,
    min_identifier_hits: int,
    min_positions: int,
    max_source_tokens: int,
) -> tuple[dict | None, str]:
    """Try to annotate one QA record. Returns (annotated_record_or_None, reason)."""
    repo_path = record.get("repo_path", "").strip()
    file_path = record.get("file_path", "").strip()
    commit_sha = record.get("commit_sha", "").strip()
    answer = record.get("answer", "")

    if not repo_path or not file_path or not commit_sha or not answer:
        return None, "missing_fields"

    # Locate the repo on disk.
    repo_dir = repo_lookup.get(repo_path)
    if repo_dir is None:
        return None, "repo_not_found"

    try:
        repo = pygit2.Repository(str(repo_dir))
    except pygit2.GitError:
        return None, "repo_open_failed"

    blob_bytes = _read_blob_at_commit(repo, commit_sha, file_path)
    if blob_bytes is None:
        return None, "blob_not_found"

    try:
        source_text = blob_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None, "binary_blob"

    if not source_text.strip():
        return None, "empty_source"

    # Tokenize source with char offsets. Use a fast tokenizer; truncate to
    # ``max_source_tokens`` to bound work on huge files.
    enc = tokenizer(
        source_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_source_tokens,
    )
    offsets = enc["offset_mapping"]
    if not offsets:
        return None, "tokenize_empty"
    token_starts = [s for s, _ in offsets]
    token_ends = [e for _, e in offsets]
    source_token_count = len(token_starts)

    # Extract identifiers + map to positions.
    all_idents, code_idents = _extract_grounding_identifiers(answer)
    if not all_idents:
        return None, "no_identifiers"
    if not code_idents:
        return None, "no_code_identifiers"

    # Position set comes from ALL identifier matches (code-shaped or not),
    # since we want the head to attend to anything the answer references.
    # The acceptance criterion uses the code-shaped count: at least one
    # code-shaped identifier must appear in the source so we know the
    # answer is talking about THIS file specifically rather than
    # paraphrasing the question.
    position_set: set[int] = set()
    n_hits = 0
    n_code_hits = 0
    code_set = set(code_idents)
    for ident in all_idents:
        pos_for_ident = _positions_for_identifier(
            source_text, ident, token_starts, token_ends,
        )
        if pos_for_ident:
            n_hits += 1
            if ident in code_set:
                n_code_hits += 1
            position_set.update(pos_for_ident)

    if n_code_hits == 0:
        return None, "no_code_hits_in_source"
    if n_hits < min_identifier_hits:
        return None, "insufficient_hits"
    if len(position_set) < min_positions:
        return None, "insufficient_positions"

    annotated = dict(record)
    annotated["answer_position_indices"] = sorted(position_set)
    annotated["answer_position_meta"] = {
        "n_identifier_hits": n_hits,
        "n_code_identifier_hits": n_code_hits,
        "n_positions": len(position_set),
        "source_token_count": source_token_count,
    }
    return annotated, "kept"


def filter_qa(
    input_dir: Path,
    repos_dir: Path,
    output_dir: Path,
    tokenizer_name: str,
    min_identifier_hits: int,
    min_positions: int,
    max_source_tokens: int,
    max_files: int | None = None,
) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if not tokenizer.is_fast:
        raise RuntimeError(
            f"Tokenizer {tokenizer_name!r} must be a fast tokenizer to provide "
            "offset_mapping; got slow tokenizer.",
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building repo lookup from {repos_dir} ...")
    repo_lookup = _build_repo_lookup(repos_dir)
    print(f"  found {len(repo_lookup)} repos")

    jsonl_files = sorted(input_dir.glob("*/*/qa_pairs.jsonl"))
    if max_files is not None:
        jsonl_files = jsonl_files[:max_files]
    print(f"Processing {len(jsonl_files)} JSONL files")

    counters = {
        "total": 0, "kept": 0,
        "missing_fields": 0, "repo_not_found": 0, "repo_open_failed": 0,
        "blob_not_found": 0, "binary_blob": 0, "empty_source": 0,
        "tokenize_empty": 0, "no_identifiers": 0, "no_code_identifiers": 0,
        "no_code_hits_in_source": 0,
        "insufficient_hits": 0, "insufficient_positions": 0,
    }

    for fi, jsonl_path in enumerate(jsonl_files):
        # Mirror the input directory structure under output_dir.
        rel = jsonl_path.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        kept_rows: list[dict] = []

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                counters["total"] += 1
                annotated, reason = _process_record(
                    record, repo_lookup, tokenizer,
                    min_identifier_hits=min_identifier_hits,
                    min_positions=min_positions,
                    max_source_tokens=max_source_tokens,
                )
                counters[reason] += 1
                if annotated is not None:
                    kept_rows.append(annotated)

        if kept_rows:
            tmp = out_path.with_suffix(".jsonl.partial")
            with open(tmp, "w", encoding="utf-8") as f:
                for r in kept_rows:
                    f.write(json.dumps(r) + "\n")
            tmp.rename(out_path)

        if (fi + 1) % 200 == 0 or (fi + 1) == len(jsonl_files):
            keep_pct = 100.0 * counters["kept"] / max(counters["total"], 1)
            print(
                f"  [{fi + 1}/{len(jsonl_files)}] total={counters['total']} "
                f"kept={counters['kept']} ({keep_pct:.1f}%)",
            )

    summary_path = output_dir / "filter_summary.json"
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "tokenizer": tokenizer_name,
        "min_identifier_hits": min_identifier_hits,
        "min_positions": min_positions,
        "max_source_tokens": max_source_tokens,
        "counters": counters,
        "keep_rate": counters["kept"] / max(counters["total"], 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {summary_path}")
    print(f"  Total: {counters['total']:,}")
    print(f"  Kept:  {counters['kept']:,} ({100*counters['kept']/max(counters['total'],1):.1f}%)")
    print("  Skip reasons:")
    for k, v in sorted(counters.items(), key=lambda x: -x[1]):
        if k in ("total", "kept"):
            continue
        if v == 0:
            continue
        print(f"    {k}: {v:,}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="QA pairs root, e.g. $DATA_DIR/qa_pairs/",
    )
    parser.add_argument(
        "--repos-dir", type=Path, required=True,
        help="Git repos root, e.g. $DATA_DIR/repos/",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where to write filtered JSONL (mirrors input directory layout)",
    )
    parser.add_argument(
        "--tokenizer", type=str, default="Qwen/Qwen3.5-0.8B",
        help="HuggingFace fast tokenizer (default: Qwen/Qwen3.5-0.8B)",
    )
    parser.add_argument(
        "--min-identifier-hits", type=int, default=2,
        help="Drop pairs where fewer than N distinct grounded identifiers "
             "appear in the source file (default: 2).",
    )
    parser.add_argument(
        "--min-positions", type=int, default=1,
        help="Drop pairs whose answer-position set has fewer than N source "
             "tokens (default: 1).",
    )
    parser.add_argument(
        "--max-source-tokens", type=int, default=8192,
        help="Cap source-file tokenization length (default: 8192).",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="For dev runs: process only the first N JSONL files.",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"ERROR: input dir not found: {args.input_dir}")
    if not args.repos_dir.is_dir():
        sys.exit(f"ERROR: repos dir not found: {args.repos_dir}")

    filter_qa(
        input_dir=args.input_dir,
        repos_dir=args.repos_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        min_identifier_hits=args.min_identifier_hits,
        min_positions=args.min_positions,
        max_source_tokens=args.max_source_tokens,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
