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
- Spans are anchored so the task does not become line-counting. ``tail``
  needs no anchor at all; ``after_anchor`` names one line that occurs exactly
  once and asks for what FOLLOWS it, so the anchor is revealed and the target
  is not. There is no ``head``: the first lines of a source file are its most
  predictable region and a correct answer there is weak evidence of reading.
- Licence headers and comment blocks are rejected wherever they land. A model
  reproduces an Apache header from its prior, not from the document.
- Capacity is not the constraint: 10% of a 2k-token document is ~200 survivor
  vectors of width 1024, far above the entropy of any span asked for here. The
  constraint is whether the encoder CHOOSES to carry it.

A previous reconstruction attempt (``git_commit_repro``) sat at recon_gap ~0
for thousands of steps because a browse plaintext-copy path let the decoder
copy the answer instead of reading the reps. Hence the rule that decides this
family's validity: THE PROMPT MUST NEVER CONTAIN THE TARGET. The tail
question names no content at all; the anchor question names exactly one line
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

# Neither is a licence header. Spot-checking the first build on real repo
# files turned up spans like "/****...\n * Copyright 2011, " -- an Apache
# header a language model reproduces from its prior, with no reading
# involved. That is the same "answerable without the document" defect this
# family exists to remove, so it is filtered wherever it lands.
_COMMENT_START = re.compile(r"^\s*(#|//|/\*|\*|<!--|--\s|;|%)")
_BOILERPLATE = re.compile(
    r"copyright|licen[sc]ed under|all rights reserved|"
    r"permission is hereby granted|apache licen[sc]e|mit licen[sc]e|"
    r"gnu general public|spdx-licen[sc]e-identifier|redistribution and use",
    re.I,
)


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


def _boilerplate(lines: list[str]) -> bool:
    """Reproducible from a language prior rather than from the document."""
    text = "\n".join(lines)
    if _BOILERPLATE.search(text):
        return True
    real = [ln for ln in lines if ln.strip()]
    if not real:
        return True
    comments = sum(1 for ln in real if _COMMENT_START.match(ln))
    return comments > len(real) * 0.5


def _span_text(lines: list[str], start: int, count: int) -> str | None:
    chunk = lines[start : start + count]
    if len(chunk) < count or not _substantive(chunk) or _boilerplate(chunk):
        return None
    return "\n".join(chunk)


def generate_reconstruct_samples(
    text: str,
    *,
    source_label: str,
    rng: random.Random,
    n_samples: int = 4,
    min_lines: int = 2,
    max_lines: int = 5,
    min_chars: int = 60,
    max_chars: int = 300,
) -> list[ReconstructSample]:
    """Span-reconstruction questions for ``text``.

    ``n_samples`` spans are drawn WITHOUT overlap where possible, so a
    document's questions together cover a large and varying part of it. A span
    outside ``[min_chars, max_chars]`` is rejected: too short is guessable,
    too long dominates the decoder's loss budget and crowds out every other
    family in the mix.

    THE DEFAULTS WERE HALVED after v10. The first cut asked for 4-12 lines
    (p50 266 chars) and after 750 steps the model's GENERATIVE token_f1 on
    this family was 0.025 against a measured cross-document floor of 0.021 --
    it had learned nothing, while lognav's tripled to 0.589 over the same
    steps. Two things made it near-impossible rather than merely hard:
    the span length, and a retention curriculum composing to ~1.5% effective
    (l0 0.10 x l1 0.15) by the end of the ramp. Reproducing 266 characters
    verbatim from 1.5% of a document is not a test of whether the reps carry
    content.

    60-300 characters is still far above the guessing floor -- random spans
    from different files score 0.021 against each other -- and is a length a
    model can actually reach. An unlearnable task measures nothing: a rep_gain
    computed between two arms that are both at floor reads zero for a reason
    that has nothing to do with the reps.
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

    # TAIL leaks nothing at all: the position is named, not the content. There
    # is deliberately no HEAD counterpart -- the first lines of a source file
    # are its single most predictable region (shebang, licence, imports), so a
    # correct answer there is weak evidence of having read anything. Spot
    # checks on real repos produced Apache headers and bundler preambles.
    k = rng.randint(min_lines, max_lines)
    start = n_lines - k
    span = _span_text(lines, start, k) if start >= 0 else None
    if ok(span):
        samples.append(make(
            "tail",
            f"Reproduce the last {k} lines of this file, exactly as written.",
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
