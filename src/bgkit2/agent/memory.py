"""Two-stage AGENTIC memory (2026-09-22): conversational memory as an always-on INDEX plus
question-focused re-reads.

* The INDEX: every past session is one slot in the opening user turn, encoded with the fixed
  generic prompt ``MEMORY_INDEX_PROMPT`` at a high compression factor (``index_reps``). Its
  label line (``[S3] 2023/05/20 (Sat) 08:23``) is verbatim text, so the session id the agent
  passes to ``memory_read`` is never something it has to decode from reps.
* ``memory_search(query, focus)``: BM25 over the sessions (``web.WebCorpus``), top-k as
  ``[id] date`` + a short snippet, a COMPRESSIBLE slot exactly like ``web_search`` (prompt
  ``memory search <query>: <focus>``, the same gentle ratio 4): the agent copies session ids out
  of its reps as it copies page ids out of web_search's (user, 2026-09-22). ``focus`` joins the
  query in ranking and in choosing each hit's snippet. (zoom and web_find are encoded too since
  2026-09-22; only their opt-in control paths are verbatim.)
* ``memory_read(session, focus)``: the whole session as a COMPRESSIBLE slot with the encoder
  prompt ``session <date>: <focus>`` -- the prompt-conditioned encoding that carries bgkit's
  results (1.3-2.0 nats; the wrong or no prompt loses most of it).

Why per-session index slots and not one spine-like slot: (1) caching -- an index slot is a pure
function of (session text, generic prompt, reps), so a memory that grows by one session encodes
one new slot and every question over the same memory reuses the rest; a cons spine re-encodes the
cells above any change and its top cell is a single opaque node; (2) addressability -- each slot
sits under its own verbatim id + date, so "which session" is read off the label next to the reps
rather than recovered from a pooled node; (3) it is the layout the memory stores already train
(one slot per dated session, ``memory_qa`` / ``memory_recall``), so the agentic rows only change
the prompt and the rep count of those slots; (4) the decoder cost stays bounded by the rep clamp
(``MEMORY_INDEX_MAX_REPS`` per session; ~50 sessions -> <= 1.2k positions).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from bgkit2.data.memory_framing import (
    MEMORY_INDEX_PROMPT,
    MEMORY_INDEX_TOOL,
    MEMORY_SNIPPET_WORDS,
    index_label,
    index_reps,
    memory_agent_user,
    session_date,
)

from .context import Conversation, Slot, split_by_tokens
from .web import Page, WebCorpus, snippet

_SID = re.compile(r"^\[?\s*[sS]?(?:ession\s*)?(\d+)\s*\]?$")


@dataclass(frozen=True)
class MemorySession:
    id: str  # "S1", "S2", ... in haystack order
    date: str
    text: str  # the whole session, header line included

    @property
    def body(self) -> str:
        head, _, rest = self.text.partition("\n")
        return rest if head.lstrip().startswith("[") and rest else self.text


class MemoryStore:
    """A memory: its sessions in order, BM25 over them, lookup by id."""

    def __init__(self, sessions: list[MemorySession]) -> None:
        self.sessions = list(sessions)
        self.by_id = {s.id: s for s in self.sessions}
        if len(self.by_id) != len(self.sessions):
            raise ValueError("session ids are not unique")
        self.corpus = WebCorpus([Page(s.id, s.date, "", s.body) for s in self.sessions])

    @classmethod
    def from_texts(cls, texts: list[str]) -> MemoryStore:
        """Sessions S1..Sn from dated session texts (``[session on DATE]\\n...`` or LoCoMo's
        ``[session k, DATE]\\n...``: the benchmark segments and the memory stores' slots)."""
        return cls([MemorySession(f"S{i + 1}", session_date(t), t) for i, t in enumerate(texts)])

    def __len__(self) -> int:
        return len(self.sessions)

    def resolve(self, session) -> MemorySession | None:
        """A session by id; tolerant of ``s3`` / ``3`` / ``[S3]`` / ``session 3``."""
        key = str(session).strip()
        if key in self.by_id:
            return self.by_id[key]
        m = _SID.match(key)
        return self.by_id.get(f"S{int(m.group(1))}") if m else None

    def search(self, query: str, k: int = 5) -> list[tuple[MemorySession, float]]:
        return [(self.by_id[p.id], s) for p, s in self.corpus.search(query, k=k)]


def index_entries(store: MemoryStore, count: Callable[[str], int],
                  slot_max_tokens: int | None = None) -> list[tuple[str, Slot]]:
    """(label, slot) per session under the generic index prompt; a session longer than
    ``slot_max_tokens`` becomes consecutive ``(part i/n)`` slots (no silent truncation)."""
    out: list[tuple[str, Slot]] = []
    for s in store.sessions:
        label = index_label(s.id, s.date)
        n = count(s.text)
        parts = (split_by_tokens(s.text, int(slot_max_tokens), count)
                 if slot_max_tokens and n > slot_max_tokens else [s.text])
        for i, part in enumerate(parts):
            npart = count(part) if len(parts) > 1 else n
            lab = f"{label} (part {i + 1}/{len(parts)})" if len(parts) > 1 else label
            out.append((lab, Slot(part, MEMORY_INDEX_PROMPT, MEMORY_INDEX_TOOL, npart,
                                  index_reps(npart))))
    return out


def add_memory_index(conv: Conversation, store: MemoryStore, question: str,
                     slot_max_tokens: int | None = None) -> list[Slot]:
    """The opening user turn: the memory index (one labelled slot per session) + the question."""
    before, after = memory_agent_user(question)
    return conv.add_user_with_slots(
        before, index_entries(store, conv.renderer.count, slot_max_tokens), after)


def memory_opening(store: MemoryStore, question: str, slot_max_tokens: int | None = None):
    """``run_episode(opening=...)`` for a memory episode."""

    def opening(conv: Conversation, _task) -> None:
        add_memory_index(conv, store, question, slot_max_tokens)

    return opening


def search_text(store: MemoryStore, query: str, focus: str = "", k: int = 5,
                snippet_words: int = MEMORY_SNIPPET_WORDS) -> str:
    """``memory_search``'s result text (compressed as a slot by the harness)."""
    hits = store.search(f"{query} {focus}".strip(), k=k)
    if not hits:
        return f"no sessions match {query!r}"
    q = f"{query} {focus}".strip()
    return "\n\n".join(f"{index_label(s.id, s.date)}\n{snippet(s.body, q, snippet_words)}"
                       for s, _ in hits)


def memory_tool_specs() -> list[dict]:
    from .tools import _FOCUS, ToolSpec, _schema

    return [
        ToolSpec("memory_search",
                 "Search the past sessions. Returns the best-matching session ids with their "
                 "dates and a short snippet.",
                 _schema({"query": ("string", "search terms"), "focus": _FOCUS}, ["query"]),
                 compressible=True).as_openai(),
        ToolSpec("memory_read",
                 "Read one past session by its id (e.g. S3, from the memory index or "
                 "memory_search).",
                 _schema({"session": ("string", "session id, e.g. S3"), "focus": _FOCUS},
                         ["session"]),
                 compressible=True).as_openai(),
    ]
