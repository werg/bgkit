"""File-read needle QA over source files (Family B, wide-net tool results).

Given a source file's text (a simulated large `read_file` tool result),
generate needle questions whose answers are exact lines/values from the
file, plus explicit "not defined here" negatives
(`plans/capability_packaging_2026_08_20.md` §5.1).

Question types:

- ``signature``   — quote the full definition line of a symbol that is
  defined exactly once in the file (def/class/function/fn heuristics for
  py/js/ts/go/rust/java).
- ``assignment``  — value assigned to an UPPER_CASE constant defined once.
- ``needle_token``— quote the unique line mentioning a rare identifier
  (generic fallback, shares :func:`bgkit.data.lognav_qa.rare_id_tokens`'s
  spirit but code-tokenized).
- ``absent``      — a plausible symbol NOT present in this file (drawn from
  a sibling file by the caller): the answer states it is not defined here.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from bgkit.data.blob_format import render_header

_DEF_PATTERNS = [
    re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", re.M),  # python
    re.compile(r"^\s*class\s+([A-Za-z_]\w*)[\s(:]", re.M),  # python/js/ts/java
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$]\w*)\s*\(", re.M),  # js/ts
    re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\(", re.M),  # go
    re.compile(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_]\w*)", re.M),  # rust
]
_CONST_RE = re.compile(r"^\s*([A-Z][A-Z0-9_]{3,})\s*[:=]\s*(.+?)\s*$", re.M)
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{5,}")


@dataclass(frozen=True)
class FileNeedleSample:
    qtype: str
    question: str
    answer: str
    blob_header: str


def defined_symbols(text: str) -> dict[str, str]:
    """symbol -> its full definition line, for symbols defined exactly once."""
    counts: dict[str, list[str]] = {}
    lines = text.splitlines()
    joined = "\n".join(lines)
    for pat in _DEF_PATTERNS:
        for m in pat.finditer(joined):
            line = joined[: m.end()].rsplit("\n", 1)[-1] + joined[m.end() :].split("\n", 1)[0]
            counts.setdefault(m.group(1), []).append(line.strip())
    # keep symbols whose NAME occurs as a definition exactly once
    return {s: ls[0] for s, ls in counts.items() if len(ls) == 1}


def constant_assignments(text: str) -> dict[str, str]:
    """CONST_NAME -> assigned value string, for constants assigned exactly once."""
    hits: dict[str, list[str]] = {}
    for m in _CONST_RE.finditer(text):
        hits.setdefault(m.group(1), []).append(m.group(2).rstrip(",;"))
    return {k: v[0] for k, v in hits.items() if len(v) == 1}


def unique_identifier_lines(text: str, *, max_hits: int = 200) -> dict[str, str]:
    """identifier -> the single line containing it (identifier appears once)."""
    lines = text.splitlines()
    counts: dict[str, int] = {}
    first_line: dict[str, str] = {}
    for ln in lines:
        for w in set(_WORD_RE.findall(ln)):
            counts[w] = counts.get(w, 0) + 1
            if w not in first_line:
                first_line[w] = ln
    out: dict[str, str] = {}
    for w, c in counts.items():
        if c == 1 and len(out) < max_hits:
            out[w] = first_line[w].strip()
    return out


def generate_file_samples(
    text: str,
    *,
    source_label: str,
    rng: random.Random,
    absent_symbols: list[str] | None = None,
    max_per_type: int = 2,
    max_answer_chars: int = 400,
) -> list[FileNeedleSample]:
    """Needle samples for ``text``. Candidate answers longer than
    ``max_answer_chars`` (data tables, embedded blobs, minified lines) are
    not needles — the task is to locate and copy ONE human-scale line — and
    are excluded before sampling."""
    n_lines = text.count("\n") + 1
    samples: list[FileNeedleSample] = []

    def short(table: dict[str, str]) -> dict[str, str]:
        return {k: v for k, v in table.items() if len(v) <= max_answer_chars}

    def make(qtype: str, question: str, answer: str) -> FileNeedleSample:
        return FileNeedleSample(
            qtype=qtype,
            question=question,
            answer=answer,
            blob_header=render_header(
                "tool", source=source_label, stats=f"{n_lines} lines", query=question
            ),
        )

    defs = short(defined_symbols(text))
    for sym in rng.sample(sorted(defs), min(max_per_type, len(defs))):
        samples.append(
            make(
                "signature",
                f"Quote the full line where `{sym}` is defined in this file.",
                defs[sym],
            )
        )

    consts = short(constant_assignments(text))
    for k in rng.sample(sorted(consts), min(max_per_type, len(consts))):
        samples.append(
            make("assignment", f"What value is assigned to `{k}` in this file?", consts[k])
        )

    uniq = short(unique_identifier_lines(text))
    # avoid re-asking symbols already covered
    pool = sorted(set(uniq) - set(defs) - set(consts))
    for w in rng.sample(pool, min(max_per_type, len(pool))):
        samples.append(
            make("needle_token", f"Quote the line in this file that mentions `{w}`.", uniq[w])
        )

    if absent_symbols:
        candidates = [s for s in absent_symbols if s not in text]
        if candidates:
            sym = rng.choice(candidates)
            samples.append(
                make(
                    "absent",
                    f"Is `{sym}` defined in this file? If so, quote its definition line.",
                    f"No — `{sym}` is not defined in this file.",
                )
            )
    return samples
