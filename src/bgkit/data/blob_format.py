"""Unified BgKIT blob format — the single decoder-facing contract for
compressed-context injection.

Part of the capability-packaging plan
(``plans/capability_packaging_2026_08_20.md`` §3). A *blob* is the only way
compressed content appears to the decoder, across all four task families
(compaction, wide-net tool results, memory, ambient background)::

    <bgkit_blob kind=tool>
    compressed: build.log (48,210 lines) query="first failing test"
    <<<BGKIT_BLOB_SURVIVORS_9c4e77d1>>>
    </bgkit_blob>

Three layers:

1. **Markers** (``<bgkit_blob kind=...>`` / ``</bgkit_blob>``) — plain text,
   independent of any chat template, so a blob can live inside a tool-role
   message (Family B), replace a span of history (Family A), sit in the
   system region (memory / ambient), or appear anywhere else. The chat
   templating machinery in :mod:`bgkit.data.bgkit_tool_template` is untouched.
2. **Header** — one short plaintext line of provenance + intent, rendered by
   :func:`render_header`. Headers keep the decoder oriented ("what is this
   blob?") and are trivially emittable by any real harness.
3. **Splice sentinel** — a single unique string replaced at forward time by
   a run of projected survivor embeddings, using the same embedding-space
   splice primitive Phase 2 uses for :data:`~bgkit.data.bgkit_tool_template.
   BGKIT_SENTINEL`. Blobs use their own sentinel so trainers can distinguish
   blob splices from legacy Phase-2 tool splices in mixed data.

The decoder only ever *reads* blobs: blob framing tokens are never
loss-supervised as decoder output, and no ID or tool grammar is associated
with them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

BlobKind = Literal["compaction", "tool", "memory", "ambient"]

BLOB_KINDS: tuple[BlobKind, ...] = ("compaction", "tool", "memory", "ambient")

# Long-random sentinel so tokenization collisions are effectively zero.
# Deliberately distinct from BGKIT_SENTINEL (Phase-2 tool splices).
BLOB_SPLICE_SENTINEL = "<<<BGKIT_BLOB_SURVIVORS_9c4e77d1>>>"

BLOB_OPEN_TEMPLATE = "<bgkit_blob kind={kind}>"
BLOB_CLOSE = "</bgkit_blob>"

_BLOB_RE = re.compile(
    r"<bgkit_blob kind=(?P<kind>[a-z]+)>\n"
    r"(?P<header>[^\n]*)\n" + re.escape(BLOB_SPLICE_SENTINEL) + r"\n</bgkit_blob>",
)


@dataclass(frozen=True)
class BgkitBlob:
    """One blob: what the decoder sees (kind + header) plus the encode spec.

    ``source_ref`` is an opaque handle for the *content* to compress — a
    dataset row id, a file path, a turn-range key. Data generators resolve it
    to text; the trainer resolves it to survivor embeddings. It never reaches
    the decoder as text.

    ``query`` is the targeted-compression prompt for query-conditioned
    encoding (Family B). ``None`` means query-free encoding.
    """

    kind: BlobKind
    header: str
    source_ref: str
    query: str | None = None

    def render(self) -> str:
        """Render the decoder-facing text with the splice sentinel inside."""
        if self.kind not in BLOB_KINDS:
            raise ValueError(f"unknown blob kind: {self.kind!r}")
        if "\n" in self.header:
            raise ValueError("blob header must be a single line")
        open_tag = BLOB_OPEN_TEMPLATE.format(kind=self.kind)
        return f"{open_tag}\n{self.header}\n{BLOB_SPLICE_SENTINEL}\n{BLOB_CLOSE}"


def render_header(
    kind: BlobKind,
    *,
    source: str | None = None,
    span: str | None = None,
    stats: str | None = None,
    query: str | None = None,
    timestamp: str | None = None,
) -> str:
    """Render the canonical one-line header per kind.

    Templates (capability-packaging plan §3):

    - compaction: ``compacted: turns 3-17 (2 file edits, 4 test runs)``
    - tool:       ``compressed: build.log (48,210 lines) query="first failing test"``
    - memory:     ``memory: session 2026-07-14``
    - ambient:    ``repo: fastapi @ a1b2c3 (1,412 files)``
    """
    if kind == "compaction":
        head = f"compacted: {span or 'earlier context'}"
        return f"{head} ({stats})" if stats else head
    if kind == "tool":
        head = f"compressed: {source or 'tool output'}"
        if stats:
            head += f" ({stats})"
        if query:
            head += f' query="{query}"'
        return head
    if kind == "memory":
        head = f"memory: {source or 'prior session'}"
        return f"{head} @ {timestamp}" if timestamp else head
    if kind == "ambient":
        head = f"repo: {source or 'background corpus'}"
        return f"{head} ({stats})" if stats else head
    raise ValueError(f"unknown blob kind: {kind!r}")


def parse_blobs(text: str) -> list[tuple[BlobKind, str]]:
    """Extract ``(kind, header)`` for every well-formed blob in ``text``.

    Data-generation validation helper: generators render trajectories to text
    and assert the blobs they intended are the blobs a consumer would find.
    """
    out: list[tuple[BlobKind, str]] = []
    for m in _BLOB_RE.finditer(text):
        kind = m.group("kind")
        if kind not in BLOB_KINDS:
            raise ValueError(f"blob with unknown kind {kind!r} in text")
        out.append((kind, m.group("header")))  # type: ignore[arg-type]
    return out


def blob_sentinel_positions(input_ids: list[int], sentinel_ids: list[int]) -> list[int]:
    """Start index of every occurrence of the tokenized sentinel.

    ``sentinel_ids`` must be ``tokenizer.encode(BLOB_SPLICE_SENTINEL,
    add_special_tokens=False)`` computed once by the caller. The sentinel is
    replaced by a variable-length run of survivor embeddings at forward time,
    exactly like the Phase-2 splice; consumers use these positions to build
    the embedding-space splice plan.

    Note the BPE caution: the sentinel must be tokenized as part of the full
    rendered sequence only if its boundary characters can merge with
    neighbours — blobs always place the sentinel on its own line, and the
    newline framing makes the encoding stable (asserted in unit tests).
    """
    n, m = len(input_ids), len(sentinel_ids)
    if m == 0:
        raise ValueError("sentinel_ids must be non-empty")
    return [i for i in range(n - m + 1) if input_ids[i : i + m] == sentinel_ids]
