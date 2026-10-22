"""Anchored cross-reference QA: short, high-cardinality answers over source files.

WHY A NEW FORMULATION. Every existing wide-net family fails guessability on
some axis, measured on the shipped data 2026-09-03:

    family        unique ans   top-20 covers   question-echo   cross-doc
    lognav             48.5%          52.5%           0.066       0.025
    grepset            94.3%          11.2%           0.142       0.051
    fileneedle         98.1%           6.7%           0.227       0.012
    reconstruct       100.0%           4.9%           0.117       0.012

The failures are structural, not parametric. lognav's answers are a closed
vocabulary (``SEVERITIES`` is a 9-element enum; three answers cover 100% of
several question forms). fileneedle's questions NAME the symbol its answers
contain, so echoing the prompt scores 0.227 -- and its zeroed arm scored
0.241, i.e. the no-document model was doing exactly that. reconstruct is
unguessable but its 124-character answers proved unlearnable at this scale,
plateauing at 0.063 against its own 0.117 floor.

Underneath sits one tension: SHORT answers are guessable from a prior, LONG
answers are unreproducible under compression. v10 and v11 each hit one horn.

THE ESCAPE. Name an ANCHOR in the question and ask for something ADJACENT to
it. The answer is a symbol name -- ~15 characters, so it survives 5%
retention -- drawn from a per-document vocabulary, so no prior covers it, and
never present in the question, so echo cannot reach it. Measured on the same
corpus: 97.7% unique answers, top-20 covers 18.0%, echo 0.043, cross-doc
0.044.

TWO FORMS, both requiring two locations rather than one span:

``next_def``       anchor is a definition LINE; answer is the name of the
                   definition that follows it. Locate, then read forward.
``enclosing_def``  anchor is a token occurring exactly once; answer is the
                   name of the definition containing it. Locate, then scan
                   back.

Both are one-location-plus-a-direction, which is deliberate: the
query-blindness result (survivor-set Jaccard 0.967 own-vs-foreign query) says
a family whose answers all sit in one predictable region lets a
query-independent policy satisfy every question at once.

CORPUS-LEVEL FILTERS ARE NOT OPTIONAL and cannot run per document. A
generator seeing one file cannot know that ``__init__`` answers a thousand
others; measured, corpus-common answers put enclosing_def's top-20 at 66.4%.
And a single large function contains many rare tokens, so one file otherwise
emits a dozen rows sharing an answer -- which kept it at 68% even after the
corpus filter. Both filters live in :func:`filter_candidates`, which the
builder applies across the whole corpus.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from bgkit.data.blob_format import render_header

_DEF = re.compile(
    r"^\s*(?:(?:async\s+)?def|class|(?:export\s+)?(?:async\s+)?function|func|"
    r"(?:pub\s+)?fn)\s+([A-Za-z_$][\w$]*)",
    re.M,
)
_WORD = re.compile(r"[A-Za-z_][\w]{5,}")


@dataclass(frozen=True)
class XrefSample:
    qtype: str
    question: str
    answer: str
    blob_header: str
    # The answer is a symbol NAME that occurs in the file, so a gold span is
    # available; the builder resolves it against the tokenized article.
    span_text: str


def generate_xref_samples(
    text: str,
    *,
    source_label: str,
    rng: random.Random,
    max_per_type: int = 3,
    max_anchor_chars: int = 160,
    max_answer_chars: int = 60,
) -> list[XrefSample]:
    """Anchored cross-reference questions for one source file.

    Emits candidates only. The corpus-level guessability filters
    (:func:`filter_candidates`) must be applied by the caller across all
    documents, or the answer distribution collapses onto names like
    ``__init__``.
    """
    n_lines = text.count("\n") + 1

    def make(qtype: str, question: str, answer: str) -> XrefSample:
        return XrefSample(
            qtype=qtype,
            question=question,
            answer=answer,
            span_text=answer,
            blob_header=render_header(
                "tool", source=source_label, stats=f"{n_lines} lines", query=question,
            ),
        )

    defs = [(m.start(), m.group(1)) for m in _DEF.finditer(text)]
    if not defs:
        return []
    def_counts = Counter(name for _, name in defs)
    out: list[XrefSample] = []

    # NEXT DEFINITION. The answer must be unique as a definition so "which one"
    # has a single correct reply; the ANCHOR need not be, and requiring both
    # halved the yield for no measured benefit.
    pairs = []
    for i in range(len(defs) - 1):
        pos, name = defs[i]
        nxt = defs[i + 1][1]
        if nxt == name or def_counts[nxt] != 1 or len(nxt) > max_answer_chars:
            continue
        line = text[:pos].split("\n")[-1] + text[pos:].split("\n")[0]
        line = line.strip()
        if not line or len(line) > max_anchor_chars or nxt in line:
            continue
        pairs.append((line, nxt))
    rng.shuffle(pairs)
    for line, nxt in pairs[:max_per_type]:
        out.append(make(
            "next_def",
            f"In this file, which function or class is defined immediately "
            f"after the line `{line}`?",
            nxt,
        ))

    # ENCLOSING DEFINITION. Anchor is a token occurring exactly once anywhere
    # in the file and never as a definition name, so the question cannot name
    # its own answer.
    counts = Counter(_WORD.findall(text))
    rare = [t for t, c in counts.items() if c == 1 and def_counts.get(t, 0) == 0]
    rng.shuffle(rare)
    picked = 0
    for tok in rare:
        if picked >= max_per_type:
            break
        pos = text.find(tok)
        before = [(p, n) for p, n in defs if p < pos]
        if not before:
            continue
        name = before[-1][1]
        if len(name) > max_answer_chars or name in tok or tok in name:
            continue
        out.append(make(
            "enclosing_def",
            f"In this file, `{tok}` appears inside exactly one function or "
            f"class. What is its name?",
            name,
        ))
        picked += 1
    return out


def filter_candidates(
    rows: list[dict],
    *,
    max_doc_share: float = 0.01,
    per_doc_answer_cap: int = 1,
) -> list[dict]:
    """Drop guessable rows using information only the whole corpus has.

    Each row needs ``answer`` and ``doc_id``. Two filters, both measured
    necessary:

    - an answer appearing across more than ``max_doc_share`` of documents is
      supplied by a prior rather than by the document (``__init__``, ``main``,
      ``render``): corpus-common answers put enclosing_def's top-20 at 66.4%;
    - an answer may appear at most ``per_doc_answer_cap`` times per document,
      because one large function contains many rare tokens and otherwise emits
      a dozen rows sharing its name -- still 68% after the first filter alone.

    Together they took the combined family to 97.7% unique answers with the
    top 20 covering 18.0%.
    """
    docs_per_answer: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        docs_per_answer[r["answer"]].add(r["doc_id"])
    n_docs = len({r["doc_id"] for r in rows}) or 1
    limit = max(2, int(n_docs * max_doc_share))

    seen: dict[tuple[str, str], int] = defaultdict(int)
    kept: list[dict] = []
    for r in rows:
        if len(docs_per_answer[r["answer"]]) > limit:
            continue
        key = (r["doc_id"], r["answer"])
        if seen[key] >= per_doc_answer_cap:
            continue
        seen[key] += 1
        kept.append(r)
    return kept
