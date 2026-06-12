"""Heuristic section splitters for multi-doc summarization corpora.

Three input regimes are supported:

* ``split_multi_news``: source articles joined by ``" ||||| "`` (the
  alexfabbri/multi_news raw format). Returns the article list verbatim,
  with the ``NEWLINE_CHAR`` literal sentinel replaced by real newlines.

* ``split_s2orc_arxiv``: S2ORC-parsed arXiv full text. A heading is a
  short standalone block (<=200 chars, single line, no terminal
  sentence punctuation) that either starts with a numbered/lettered
  prefix (``I.``, ``1.``, ``A.``, ``1.1``, ``Appendix``, ``Chapter``,
  ``Part``) or is ALL CAPS or Title Case. Consecutive headings with no
  body between them collapse into a single section header (so a
  section→subsection nest yields one new section, not two).

* ``split_pmc_markdown``: Markdown with ATX headers (``# Heading``,
  ``## Subheading``, etc.). Same nesting-collapse behavior. The leading
  YAML frontmatter block ``---\\n...\\n---`` is stripped.

All splitters return ``list[tuple[heading: str, body: str]]``. The body
may be empty for headings that bottom out into a subheading with no
content of its own; downstream code should filter empty-body sections.

Abstract extraction lives in ``extract_pmc_abstract`` (case-insensitive
match on ``# Abstract`` / ``## Abstract`` / etc. headers in PMC
markdown). S2ORC arXiv has the abstract in a separate parquet column,
so no extraction is needed there. MultiNews has the summary in a
parallel ``.tgt`` file.
"""

from __future__ import annotations

import re

__all__ = [
    "split_multi_news",
    "split_s2orc_arxiv",
    "split_pmc_markdown",
    "extract_pmc_abstract",
    "strip_pmc_frontmatter",
]


# ---- MultiNews --------------------------------------------------------------

_MN_SEP = "|||||"


def split_multi_news(src_line: str) -> list[str]:
    """Split one ``.src.cleaned`` row into its source articles.

    The dataset uses the literal string ``NEWLINE_CHAR`` for in-article
    newlines; we restore those. Empty articles are dropped.
    """
    articles = src_line.split(_MN_SEP)
    cleaned = []
    for a in articles:
        a = a.replace("NEWLINE_CHAR", "\n").strip()
        if a:
            cleaned.append(a)
    return cleaned


# ---- S2ORC arXiv ------------------------------------------------------------

_HEADING_PREFIX = re.compile(
    r"^(?:[IVXLC]+\.|\d+(?:\.\d+)*\.?|[A-Z]\.|Chapter|Appendix|Part|Section)\s+\w"
)


def _is_heading_block(block: str) -> bool:
    if "\n" in block or not (0 < len(block) <= 200):
        return False
    if block[-1] in ".?!":
        return False
    if _HEADING_PREFIX.match(block):
        return True
    if block.isupper() and len(block) >= 4:
        return True
    words = block.split()
    if not (2 <= len(words) <= 12):
        return False
    capitalized = sum(1 for w in words if w and w[0].isupper())
    lowercase_chars = sum(1 for c in block if c.isalpha() and c.islower())
    if capitalized >= max(1, len(words) - 1) and lowercase_chars < len(block) * 0.45:
        return True
    return False


def split_s2orc_arxiv(text: str) -> list[tuple[str, str]]:
    """Heuristic section split for S2ORC-parsed arXiv full text."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    sections: list[tuple[str, str]] = []
    cur_heading = ""
    cur_body: list[str] = []
    for block in blocks:
        if _is_heading_block(block):
            if cur_body:
                sections.append((cur_heading, "\n\n".join(cur_body)))
                cur_body = []
                cur_heading = block
            else:
                cur_heading = (cur_heading + "\n" + block) if cur_heading else block
        else:
            cur_body.append(block)
    if cur_heading or cur_body:
        sections.append((cur_heading, "\n\n".join(cur_body)))
    return sections


# ---- PMC Markdown -----------------------------------------------------------

_PMC_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_MD_HEADER = re.compile(r"(?m)^(#{1,6}\s+.+)$")


def strip_pmc_frontmatter(text: str) -> str:
    """Strip the leading ``---\\n...\\n---\\n`` YAML block, if present."""
    m = _PMC_FRONTMATTER.match(text)
    return text[m.end():] if m else text


def split_pmc_markdown(text: str, *, strip_frontmatter: bool = True) -> list[tuple[str, str]]:
    """Split PMC markdown into ``[(heading, body), ...]``.

    Splits on ATX headers (``#`` ... ``######``). YAML frontmatter is
    stripped by default. The first chunk before any header (the title
    line + any preamble) is returned with ``heading=""``.
    """
    if strip_frontmatter:
        text = strip_pmc_frontmatter(text)
    parts = _MD_HEADER.split(text)
    sections: list[tuple[str, str]] = []
    cur_heading = ""
    cur_body: list[str] = []
    for piece in parts:
        piece = piece.strip()
        if not piece:
            continue
        if _MD_HEADER.match(piece):
            if cur_body:
                sections.append((cur_heading, "\n\n".join(cur_body)))
                cur_body = []
                cur_heading = piece
            else:
                cur_heading = (cur_heading + "\n" + piece) if cur_heading else piece
        else:
            cur_body.append(piece)
    if cur_heading or cur_body:
        sections.append((cur_heading, "\n\n".join(cur_body)))
    return sections


_ABSTRACT_HEADER = re.compile(
    r"(?im)^#{1,6}\s+abstract\b[^\n]*\n(.+?)(?=^#{1,6}\s|\Z)",
    re.DOTALL,
)


def extract_pmc_abstract(text: str) -> str | None:
    """Find the first ``# Abstract`` / ``## Abstract`` / etc. section
    body in a PMC markdown document. Returns the stripped body or
    ``None`` if no abstract section is present.
    """
    m = _ABSTRACT_HEADER.search(text)
    if not m:
        return None
    body = m.group(1).strip()
    return body or None
