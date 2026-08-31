"""Verbatim span reconstruction over source files -- the information-forcing task.

WHY THIS FAMILY EXISTS. Every Phase-2 wide-net run collapses the encoder's
output to rank ~1 per document within a few hundred steps, from the Phase-1
base's 12.97. The base kept its rank because its task was summarization: the
decoder could not produce the target without the document. The wide-net QA
families do not have that property. Measured:

  - the union of all answer spans per document is 0.4-3.1% while the L0 budget
    is 10%, so "keep everything answer-shaped" satisfies every question at
    once and query-INDEPENDENT selection is optimal (own-vs-foreign survivor
    Jaccard 0.967);
  - reps deliver ~4% of what the raw document delivers (grepset 1.6%,
    fileneedle 0%);
  - and it collapsed anyway with the rep-independent tool-call copy already
    down-weighted 5x, so the ANSWER half is rep-independent too.

Short answers to questions about a named document are largely guessable. Rank
1 is the cheapest solution to an objective that never asks for content -- not
a bug to debug, a property of the task family.

THE DESIGN, and every piece of it is load-bearing:

- The answer is a VERBATIM span of the document, long enough that a prior
  cannot cover it. Nothing in the prompt contains it; the document reaches the
  decoder only as spliced survivors.
- The span is RANDOM per row and DIFFERENT across a document's rows, so the
  union over questions approaches the whole file. No fixed subset of positions
  satisfies them all, which is exactly what defeats the query-blind policy.
- Spans are anchored so the task does not become line-counting. ``head`` and
  ``tail`` need no anchor at all; ``after_anchor`` names one line that occurs
  exactly once and asks for what FOLLOWS it, so the anchor is revealed and the
  target is not.
- Capacity is not the constraint: 10% of a 2k-token document is ~200 survivor
  vectors of width 1024, far above the entropy of any span asked for here. The
  constraint is whether the encoder CHOOSES to carry it.

A previous reconstruction attempt (``git_commit_repro``) sat at recon_gap ~0
for thousands of steps because a browse plaintext-copy path let the decoder
copy the answer instead of reading the reps. Hence the rule that decides this
family's validity: THE PROMPT MUST NEVER CONTAIN THE TARGET. The head/tail
questions name no content at all; the anchor question names exactly one line
and asks for the following ones.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from bgkit.data.blob_format import render_header

# A span made mostly of blank or punctuation-only lines is reproducible
# without having read anything.
_SUBSTANTIVE = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True)
class ReconstructSample:
    qtype: str
    question: str
    answer: str
    blob_header: str
    # Substring of the file the answer quotes, for gold-span extraction. Always
    # the answer itself here -- every answer in this family IS a quotation.
    span_text: str


def _substantive(lines: list[str], min_ratio: float = 0.6) -> bool:
    """Enough real content that the span is not guessable boilerplate."""
    if not lines:
        return False
    hits = sum(1 for ln in lines if _SUBSTANTIVE.search(ln))
    return hits >= max(1, int(len(lines) * min_ratio))


def _span_text(lines: list[str], start: int, count: int) -> str | None:
    chunk = lines[start : start + count]
    if len(chunk) < count or not _substantive(chunk):
        return None
    return "\n".join(chunk)


def generate_reconstruct_samples(
    text: str,
    *,
    source_label: str,
    rng: random.Random,
    n_samples: int = 4,
    min_lines: int = 4,
    max_lines: int = 12,
    min_chars: int = 120,
    max_chars: int = 900,
) -> list[ReconstructSample]:
    """Span-reconstruction questions for ``text``.

    ``n_samples`` spans are drawn WITHOUT overlap where possible, so a
    document's questions together cover a large and varying part of it. A span
    outside ``[min_chars, max_chars]`` is rejected: too short is guessable,
    too long dominates the decoder's loss budget and crowds out every other
    family in the mix.
    """
    lines = text.split("\n")
    n_lines = len(lines)
    if n_lines < min_lines * 2:
        return []

    def make(qtype: str, question: str, answer: str) -> ReconstructSample:
        return ReconstructSample(
            qtype=qtype,
            question=question,
            answer=answer,
            span_text=answer,
            blob_header=render_header(
                "tool", source=source_label, stats=f"{n_lines} lines", query=question,
            ),
        )

    def ok(span: str | None) -> bool:
        return span is not None and min_chars <= len(span) <= max_chars

    samples: list[ReconstructSample] = []
    used: set[int] = set()

    # HEAD and TAIL leak nothing at all: the position is named, not the
    # content. One of each at most, so the family is not dominated by the two
    # spans every document shares a shape for.
    for qtype, start_of in (("head", lambda k: 0), ("tail", lambda k: n_lines - k)):
        k = rng.randint(min_lines, max_lines)
        start = start_of(k)
        span = _span_text(lines, start, k) if start >= 0 else None
        if ok(span):
            where = "first" if qtype == "head" else "last"
            samples.append(make(
                qtype,
                f"Reproduce the {where} {k} lines of this file, exactly as written.",
                span,
            ))
            used.update(range(start, start + k))

    # ANCHORED spans: name one line that occurs exactly once, ask for what
    # follows it. The anchor is revealed; the target is not.
    once = [
        i for i, ln in enumerate(lines)
        if ln.strip() and _SUBSTANTIVE.search(ln) and text.count(ln) == 1
        and len(ln) <= 200
    ]
    rng.shuffle(once)
    for i in once:
        if len(samples) >= n_samples:
            break
        k = rng.randint(min_lines, max_lines)
        if any(j in used for j in range(i + 1, i + 1 + k)):
            continue
        span = _span_text(lines, i + 1, k)
        if not ok(span):
            continue
        samples.append(make(
            "after_anchor",
            f"This file contains the line `{lines[i].strip()}`. Reproduce the "
            f"{k} lines that immediately follow it, exactly as written.",
            span,
        ))
        used.update(range(i + 1, i + 1 + k))

    return samples[:n_samples]
