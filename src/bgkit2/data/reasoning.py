"""Per-turn reasoning in agent-trajectory messages (opt-in; builders default to dropping it).

LFM2.5-8B-A1B's chat template renders an assistant message's ``reasoning_content`` string as
``<think>{reasoning}</think>{content}`` for turns after the last user message. Sources carry
reasoning in several shapes: a ``reasoning_content`` / ``reasoning`` / ``thinking`` field,
``<think>...</think>`` or ``<thinking>...</thinking>`` blocks inside ``content`` (or a bare
``...</think>`` prefix whose opener the source stripped), and OpenHands' ``think`` tool call
(its ``thought`` argument). :func:`split_reasoning` moves the first two into one string;
the ``think`` call is merged by the tool-slot remapper, which owns tool calls.

:func:`focus_from_reasoning` derives a slot's compression FOCUS from the reasoning of the turn
that issued the read: the last one or two sentences that say what is being looked for.
"""

from __future__ import annotations

import re

REASONING_FIELDS = ("reasoning_content", "reasoning", "thinking")
_TAG_RE = re.compile(r"<(think|thinking)>(.*?)</\1>", re.DOTALL)
_BARE_CLOSE_RE = re.compile(r"^(.*?)</(think|thinking)>", re.DOTALL)


def join_reasoning(parts: list[str]) -> str:
    """Non-empty stripped parts, in order, blank-line separated."""
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def split_reasoning(m: dict) -> tuple[str, str]:
    """``(reasoning, visible content)`` of a message: the reasoning fields first (they precede
    the visible text), then every think/thinking block of ``content`` in order. The visible
    content has the blocks removed and is stripped."""
    parts = [str(m.get(k) or "") for k in REASONING_FIELDS]
    content = str(m.get("content") or "")
    bare = _BARE_CLOSE_RE.match(content)
    if bare and not re.search(r"<(think|thinking)>", bare.group(1)):
        parts.append(bare.group(1))
        content = content[bare.end() :]
    parts += [g[1] for g in _TAG_RE.findall(content)]
    content = _TAG_RE.sub("", content)
    return join_reasoning(parts), content.strip()


def move_reasoning(m: dict) -> dict:
    """A copy of ``m`` with its reasoning in ``reasoning_content`` (absent when there is none),
    the other reasoning fields removed and ``content`` the visible text only. Non-assistant
    messages are returned unchanged."""
    if m.get("role") != "assistant":
        return m
    reasoning, content = split_reasoning(m)
    out = {k: v for k, v in m.items() if k not in REASONING_FIELDS}
    out["content"] = content
    if reasoning:
        out["reasoning_content"] = reasoning
    return out


# Sentences that say what the turn is after ("let me find where X is set", "I need to check
# how Y handles Z").
_LOOKING_RE = re.compile(
    r"\b(look|looking|find|search|check|see|inspect|examine|locate|understand|verify|"
    r"confirm|where|how|which|what|need|want|explore|investigate|figure out|let me|let's|"
    r"view|read|open)\b",
    re.IGNORECASE,
)
_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def focus_from_reasoning(reasoning: str, max_chars: int = 200) -> str:
    """A slot focus from the issuing turn's reasoning: the last one or two sentences that
    mention what is being looked for (the last sentence when none does), whitespace-collapsed
    and capped at ``max_chars`` on a word boundary. Empty when there is no reasoning."""
    sents = [" ".join(s.split()) for s in _SENT_RE.split(reasoning or "")]
    sents = [s for s in sents if s]
    if not sents:
        return ""
    hits = [s for s in sents if _LOOKING_RE.search(s)] or sents[-1:]
    two = " ".join(hits[-2:])
    focus = two if len(two) <= max_chars else hits[-1]
    if len(focus) > max_chars:
        cut = focus[:max_chars].rsplit(" ", 1)[0]
        focus = (cut or focus[:max_chars]).rstrip(" ,;:")
    return focus
