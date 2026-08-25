"""Gold answer spans as token offsets inside a tokenized article (v5 lever).

Generators know the exact answer substring inside the article text. Emitting
its TOKEN span lets the trainer apply span-level relevance supervision
(`plans/capability_packaging_2026_08_20.md` "v5 DESIGN"). Token offsets are
derived from the fast tokenizer's offset mapping over the FULL article
encoding — never by re-encoding the substring (BPE is not
concat-distributive).
"""

from __future__ import annotations

import json


def char_span_to_token_span(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int] | None:
    """Map a character span ``[char_start, char_end)`` to ``[tok_start, tok_end)``.

    ``offsets`` is the tokenizer's ``offset_mapping`` for the full text. A
    token is included if it overlaps the character span. Returns None if no
    token overlaps (empty or out-of-range span).
    """
    tok_start = None
    tok_end = None
    for i, (a, b) in enumerate(offsets):
        if b <= a:  # zero-width (special) tokens
            continue
        if b > char_start and a < char_end:
            if tok_start is None:
                tok_start = i
            tok_end = i + 1
    if tok_start is None or tok_end is None:
        return None
    return tok_start, tok_end


def answer_token_span(
    tokenizer, article_text: str, answer: str, *, occurrence: int = 0
) -> tuple[int, int] | None:
    """Token span of the ``occurrence``-th exact occurrence of ``answer`` in
    ``article_text`` (0-based), or None if absent. Uses one full encoding with
    offsets; returns offsets consistent with ``tokenizer.encode(article_text,
    add_special_tokens=False)``.
    """
    if not answer:
        return None
    start = -1
    for _ in range(occurrence + 1):
        start = article_text.find(answer, start + 1)
        if start < 0:
            return None
    enc = tokenizer(article_text, add_special_tokens=False, return_offsets_mapping=True)
    return char_span_to_token_span(enc["offset_mapping"], start, start + len(answer))


def article_offsets(tokenizer, article_text: str) -> list[tuple[int, int]]:
    """Offset mapping of the full article encoding (compute ONCE per article)."""
    enc = tokenizer(article_text, add_special_tokens=False, return_offsets_mapping=True)
    return [tuple(o) for o in enc["offset_mapping"]]


def answer_span_from_offsets(
    offsets: list[tuple[int, int]], article_text: str, answer: str
) -> tuple[int, int] | None:
    """Token span of the first occurrence of ``answer`` using precomputed offsets."""
    if not answer:
        return None
    start = article_text.find(answer)
    if start < 0:
        return None
    return char_span_to_token_span(offsets, start, start + len(answer))


def span_to_json(span: tuple[int, int] | None) -> str | None:
    return None if span is None else json.dumps([int(span[0]), int(span[1])])
