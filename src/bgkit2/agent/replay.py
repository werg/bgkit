"""Replaying a recorded trajectory through the harness's own ``Conversation`` (2026-09-26).

Two store builders turn EXTERNAL trajectories into training rows that must look exactly like an
eval episode of the harness at the same context budget: the web SFT builder
(``scripts/bgkit2_build_web_sft_store.py``) and the coding ``tool_slots`` remap
(``scripts/bgkit2_remap_tool_slots.py --compact-history``). Both replay the trajectory message by
message into ``bgkit2.agent.context.Conversation`` and call ``Conversation.compact`` before every
assistant turn, as ``runner.run_episode`` does, so the oldest rounds fold into the ONE compaction
slot (cons mode: the spine of ``compaction_payload`` / ``compaction_leaves``) exactly when and how
the harness would fold them.

    FastRenderer     ``TemplateRenderer``'s interface with per-message cached token counts
                     (100-turn trajectories replay in milliseconds, same numbers)
    replay_history   add recorded turns to a Conversation, compacting before each assistant turn
    spine_meta       depth / leaf token counts of a cons compaction slot (row meta)
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from bgkit2.data.templates import SPLICE_SENTINEL

from .context import Conversation, ToolCall, compaction_leaves, compaction_payload


class _Counted(str):
    """A rendering whose token count is already known (FastRenderer.render)."""

    n: int = 0


class FastRenderer:
    """``TemplateRenderer``'s interface (render / count) for replaying 100-turn trajectories
    cheaply, with the same numbers. ``render`` returns only the token COUNT (as a ``_Counted``
    string): the LFM template renders every message independently (no ``</think>`` in our
    contents), so the rendering's count is bos + the per-message chunk counts (cached) + the
    generation prompt -- the builder asserts it against the real rendering on sampled rows. A
    message's ``reasoning_content`` (``--keep-reasoning``) is counted in every assistant turn,
    as LFM2.5-8B-A1B's template renders it (every turn after the last user message; the
    350M's keeps only the last turn's, so the check renders with ``preserve_thinking``).
    ``count`` of a string that EXTENDS the last counted string at a newline (the compaction
    slot growing by one folded round) adds the count of the new part: the pre-tokenizer splits
    at the newline, so this is exact in practice (also asserted)."""

    def __init__(self, tok) -> None:
        self.tok = tok
        self.sentinel = tok.convert_tokens_to_ids(SPLICE_SENTINEL)
        self.bos_text = tok.bos_token or ""
        self.bos = self._raw(self.bos_text)
        self.gen = self._raw("<|im_start|>assistant\n")
        self._chunk: dict[str, int] = {}
        self._recent: list[tuple[str, int]] = []

    def _raw(self, s: str) -> int:
        ids = self.tok(s, add_special_tokens=False)["input_ids"]
        return sum(1 for i in ids if i != self.sentinel)

    def _msg_count(self, m: dict, tools: list[dict] | None) -> int:
        key = json.dumps([m, tools if m.get("role") == "system" else None], sort_keys=True)
        n = self._chunk.get(key)
        if n is None:
            kw = {"tools": tools} if (tools and m.get("role") == "system") else {}
            s = self.tok.apply_chat_template([m], tokenize=False, add_generation_prompt=False, **kw)
            s = s[len(self.bos_text) :] if s.startswith(self.bos_text) else s
            n = self._chunk[key] = self._raw(s)
            if len(self._chunk) > 50_000:
                self._chunk.clear()
        return n

    def render(self, messages: list[dict], tools: list[dict], add_generation_prompt: bool) -> str:
        n = self.bos + sum(self._msg_count(m, tools) for m in messages)
        out = _Counted("")
        out.n = n + (self.gen if add_generation_prompt else 0)
        return out

    def count(self, text: str) -> int:
        if isinstance(text, _Counted):
            return text.n
        for s, n in self._recent:
            if len(text) > len(s) and text.startswith(s) and text[len(s)] == "\n":
                got = n + self._raw(text[len(s) :])
                break
        else:
            got = self._raw(text)
        if len(text) > 2000:
            self._recent = [(text, got), *self._recent[:3]]
        return got


def spine_meta(text: str, renderer) -> dict:
    """Depth and leaf token counts of a cons compaction slot (empty for flat / none)."""
    from bgkit2.data.spine import spine_cells

    sp = compaction_payload(text) if text else None
    if sp is None:
        return {"spine_leaves": 0, "spine_depth": 0, "spine_leaf_tokens": []}
    counts = [renderer.count(x) for x in compaction_leaves(sp)]
    cells = spine_cells(counts)
    return {
        "spine_leaves": len(counts),
        "spine_depth": len(cells),
        "spine_leaves_per_cell": [len(c) for c in cells],
        "spine_leaf_tokens": counts,
    }


@dataclass
class Turn:
    """One recorded message to replay. ``role`` is "assistant" (``content``, ``calls``,
    ``reasoning``), "tool" (``content`` = the result text; ``slot`` True = a compressible result
    with the encoder ``prompt``, priced under ``tool``) or "user" (``content``)."""

    role: str
    content: str = ""
    calls: tuple[ToolCall, ...] = ()
    reasoning: str | None = None
    slot: bool = False
    prompt: str = ""
    tool: str = ""


def replay_history(
    conv: Conversation,
    turns: list[Turn],
    *,
    budget: int,
    keep_recent: int = 1,
    compressed: bool = True,
) -> int:
    """Append ``turns`` (everything after the conversation head) to ``conv`` the way
    ``run_episode`` builds an episode: ``conv.compact(budget, keep_recent, compressed)`` before
    every assistant turn, and once more after the last turn (the prompt of the NEXT turn, i.e.
    the row's target). Returns the number of compact() calls that folded something."""
    folded = 0
    for t in turns:
        if t.role == "assistant":
            folded += int(conv.compact(budget, keep_recent=keep_recent, compressed=compressed))
            conv.add_assistant(t.content, list(t.calls), reasoning=t.reasoning)
        elif t.role == "tool":
            conv.add_tool_result(t.content, t.slot, t.prompt if t.slot else "", t.tool)
        elif t.role == "user":
            conv.add_user(t.content)
        else:
            raise ValueError(f"cannot replay a {t.role!r} message")
    folded += int(conv.compact(budget, keep_recent=keep_recent, compressed=compressed))
    return folded
