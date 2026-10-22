"""Conversation builder: slots, tool_slots rendering, compaction, tool-call syntax.

LFM2.5 tool-call syntax (rendered by the tokenizer's ``chat_template.jinja``,
verified on ``LiquidAI/LFM2.5-350M`` 2026-09-14)
--------------------------------------------------------------------------

The tool list goes into the SYSTEM message as JSON, an assistant turn carries
its calls as a Pythonic call list between two special tokens, and every tool
result is its own ``tool`` turn::

    <|startoftext|><|im_start|>system
    You are a coding agent.
    List of tools: [{"type": "function", "function": {"name": "search", ...}}]<|im_end|>
    <|im_start|>user
    Fix the bug.<|im_end|>
    <|im_start|>assistant
    <|tool_call_start|>[search(query='def foo', focus='where it is')]<|tool_call_end|><|im_end|>
    <|im_start|>tool
    src/a.py:3: def foo():<|im_end|>
    <|im_start|>assistant
    Found it.<|im_end|>

Rules, from the template source:

* ``<|tool_call_start|>`` / ``<|tool_call_end|>`` are single special tokens
  (ids 10 / 11); ``<|im_start|>`` = 6, ``<|im_end|>`` = 7 (also the eos).
* Several calls in one turn: ``[a(x='1'), b(y=2)]`` — comma-space separated
  inside one bracket pair; assistant ``content`` (if any) is emitted BEFORE the
  call block with no separator: ``Let me look.<|tool_call_start|>[read(...)]``.
* Arguments are keyword-only. Strings are single-quoted with ``\\``, ``'``,
  newline and CR escaped (``\\\\``, ``\\'``, ``\\n``, ``\\r``); ints / floats /
  bools are printed bare (``start=3``); lists and dicts as JSON.
* ``tool_calls[i]["function"]["arguments"]`` MUST be a mapping — a JSON string
  raises inside the template.
* Tool results render as ``<|im_start|>tool\\n{content}<|im_end|>\\n``; the tool
  name is not shown (results follow the calls in order). The
  ``<|tool_response_start|>`` tokens exist in the vocabulary but the template
  does not use them.
* No tools in the message list -> no "List of tools" line, otherwise unchanged.

``render_tool_calls`` reproduces the macro in Python (used for the full-text
prompt and checked against the real template in the tests);
``parse_tool_calls`` is its inverse, run on the model's raw generation.

Slots
-----
A *slot* is the span of a tool result's text in the rendered conversation that
the compressed model receives as reps instead. ``Conversation`` keeps the
full-text messages and, for every compressible tool result, a ``Slot`` (text +
compression prompt). ``messages_with_slots()`` renders the same conversation
with ``SPLICE_SENTINEL`` in place of each slot's text — exactly the
``tool_slots`` framing of ``bgkit2.data.tasks`` (contexts JSON list, prompts
JSON list, one sentinel per slot) — so ``to_task_row()`` is a trajectory row
the training side reads unchanged.

Compaction: when the rendered prefix exceeds a token budget, the oldest turns
after the task message are folded into ONE compaction slot (their plain text
joined, prompt ``compaction: <task summary>``) that lives in a user message
``[Earlier conversation compacted]\\n<sentinel>`` right after the task.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from bgkit2.data.spine import spine_top_reps
from bgkit2.data.templates import SPLICE_SENTINEL

TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
COMPACTION_HEADER = "[Earlier conversation compacted]"
COMPACTION_TOOL = "compaction"
COMPACTION_MODES = ("flat", "cons")


def compaction_payload(text: str) -> dict | None:
    """A cons-mode compaction slot's spine ``{"compaction": 1, "child": <node|None>,
    "leaf": <folded rounds>}``, or None for any other slot text."""
    if not text.startswith('{"compaction"'):
        return None
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) and d.get(COMPACTION_TOOL) == 1 and "leaf" in d else None


def split_by_tokens(text: str, cap: int, count) -> list[str]:
    """``text`` as consecutive parts of at most ``cap`` tokens (``count``),
    cut at line boundaries where possible, else inside a line at the largest
    character prefix that fits. The parts concatenate back to ``text``."""
    if count(text) <= cap:
        return [text]
    parts: list[str] = []
    rest = text
    while rest:
        if count(rest) <= cap:
            parts.append(rest)
            break
        lo, hi = 1, len(rest)  # largest prefix length L with count(rest[:L]) <= cap
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count(rest[:mid]) <= cap:
                lo = mid
            else:
                hi = mid - 1
        cut = rest.rfind("\n", 0, lo)
        cut = cut + 1 if cut > 0 else lo
        parts.append(rest[:cut])
        rest = rest[cut:]
    return parts


def system_with_tool_list(system: str, tools: list[dict]) -> str:
    """``system`` + the tool list exactly as the LFM chat template appends it
    for ``tools=`` -- for ``chat`` rows, whose framing renders no tools."""
    if not tools:
        return system
    body = ", ".join(t if isinstance(t, str) else json.dumps(t, ensure_ascii=False)
                     for t in tools)
    return f"{system}\nList of tools: [{body}]" if system else f"List of tools: [{body}]"


def compaction_leaves(payload: dict) -> list[str]:
    """The spine's leaves, OLDEST first."""
    out: list[str] = []
    node: dict | None = payload
    while node is not None:
        out.append(node["leaf"])
        node = node.get("child")
    return out[::-1]

# Plan problem 3: fixed per-tool ratios first (search 16x, file 8x, docs 16x,
# overview 32x, compaction 32x).
DEFAULT_RATIOS: dict[str, int] = {
    "read": 8,
    "search": 16,
    "docs": 16,
    "bash": 16,
    "overview": 32,
    COMPACTION_TOOL: 32,
    # web tools (the web eval's ratios): search hits and find-in-page carry the exact names /
    # dates / numbers the answer is copied from -> gentle 4x; a page read with a focus 16x.
    "web_search": 4,
    "web_open": 16,
    # exact-quote tools through bgkit (user, 2026-09-22): web_find re-encodes the WHOLE page,
    # zoom the WHOLE file, both prompted with what to find -> 4x. zoom(exact=true) is inline
    # verbatim text, never a slot, so it has no ratio.
    "web_find": 4,
    "zoom": 4,
}


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)

    def as_message(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "arguments": dict(self.args)}}

    @classmethod
    def from_message(cls, d: dict) -> ToolCall:
        f = d.get("function", d)
        args = f.get("arguments") or {}
        if isinstance(args, str):
            args = json.loads(args)
        return cls(str(f["name"]), dict(args))


@dataclass
class RatioPolicy:
    """Rep count of a slot: ``ceil(n_tokens / ratio[tool])``, at least 1."""

    ratios: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_RATIOS))
    default: int = 16

    def ratio(self, tool: str) -> int:
        return int(self.ratios.get(tool, self.default))

    def reps(self, n_tokens: int, tool: str) -> int:
        return max(1, math.ceil(n_tokens / self.ratio(tool)))


from ..data.overview import (  # noqa: E402,F401 - re-exported for the harness
    OVERVIEW_HEADER,
    OVERVIEW_TOOL,
    overview_marker,
    overview_repo,
)


@dataclass
class Slot:
    text: str
    prompt: str
    tool: str
    n_tokens: int
    n_reps: int
    # Teacher-side digest (the "digest sub-agent" scheme, 2026-09-14): a
    # prompted plain-text compression of ``text`` that the TEACHER reads in
    # place of the full result; the student reads reps of ``text``. None when
    # no digest was produced (the full-text arm / the student's own runs).
    digest: str | None = None


# ---------------------------------------------------------------------------
# Tool-call syntax
# ---------------------------------------------------------------------------


def _format_arg(v: object) -> str:
    if isinstance(v, str):
        s = v.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r")
        return f"'{s}'"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v)
    return str(v)


def render_tool_calls(calls: list[ToolCall]) -> str:
    """``<|tool_call_start|>[name(k='v', n=3), ...]<|tool_call_end|>`` — the
    template's ``render_tool_calls`` macro."""
    body = ", ".join(
        f"{c.name}({', '.join(f'{k}={_format_arg(v)}' for k, v in c.args.items())})" for c in calls
    )
    return f"{TOOL_CALL_START}[{body}]{TOOL_CALL_END}"


_CALL_BLOCK = re.compile(
    re.escape(TOOL_CALL_START) + r"\s*(.*?)\s*(?:" + re.escape(TOOL_CALL_END) + r"|$)", re.DOTALL
)


def _literal(node: ast.expr) -> object:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"argument is not a literal: {ast.dump(node)}") from exc


def parse_tool_calls(raw: str) -> tuple[str, list[ToolCall], str | None]:
    """Split a generation into (content, calls, error).

    ``content`` is the text outside the call block with ``<|im_end|>`` stripped;
    ``error`` names a syntax problem (then ``calls`` is empty) so the loop can
    report it back to the model instead of crashing. A missing
    ``<|tool_call_end|>`` (generation cut off) is tolerated.
    """
    raw = raw.replace(IM_END, "")
    m = _CALL_BLOCK.search(raw)
    if m is None:
        return raw.strip(), [], None
    content = (raw[: m.start()] + raw[m.end() :]).strip()
    body = m.group(1).strip()
    try:
        tree = ast.parse(body, mode="eval").body
        nodes = tree.elts if isinstance(tree, ast.List) else [tree]
        calls: list[ToolCall] = []
        for n in nodes:
            if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Name):
                raise ValueError("expected name(arg=value, ...)")
            if n.args:
                raise ValueError(f"{n.func.id}: positional arguments are not allowed")
            args = {}
            for kw in n.keywords:
                if kw.arg is None:
                    raise ValueError(f"{n.func.id}: **kwargs is not allowed")
                args[kw.arg] = _literal(kw.value)
            calls.append(ToolCall(n.func.id, args))
    except (SyntaxError, ValueError) as exc:
        return content, [], f"could not parse tool call {body!r}: {exc}"
    return content, calls, None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class Renderer(Protocol):
    """Chat-template rendering + token counting (sentinels never counted)."""

    def render(
        self, messages: list[dict], tools: list[dict], add_generation_prompt: bool
    ) -> str: ...

    def count(self, text: str) -> int: ...


class TemplateRenderer:
    """The real thing: an HF tokenizer with the LFM2.5 template."""

    def __init__(self, tok) -> None:
        self.tok = tok
        self.sentinel_id = int(tok.convert_tokens_to_ids(SPLICE_SENTINEL))

    def render(self, messages: list[dict], tools: list[dict], add_generation_prompt: bool) -> str:
        return self.tok.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def count(self, text: str) -> int:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return sum(1 for i in ids if i != self.sentinel_id)


class PlainRenderer:
    """Tokenizer-free stand-in with the same turn structure; one token per
    ``chars_per_token`` characters. For CPU tests and rough accounting."""

    def __init__(self, chars_per_token: float = 4.0) -> None:
        self.cpt = chars_per_token

    def render(self, messages: list[dict], tools: list[dict], add_generation_prompt: bool) -> str:
        msgs = list(messages)
        system = ""
        if msgs and msgs[0].get("role") == "system":
            system = str(msgs[0].get("content") or "")
            msgs = msgs[1:]
        if tools:
            system += ("\n" if system else "") + "List of tools: " + json.dumps(tools)
        out = "<|startoftext|>"
        if system:
            out += f"{IM_START}system\n{system}{IM_END}\n"
        for m in msgs:
            out += f"{IM_START}{m['role']}\n{m.get('content') or ''}"
            if m.get("tool_calls"):
                out += render_tool_calls([ToolCall.from_message(c) for c in m["tool_calls"]])
            out += f"{IM_END}\n"
        if add_generation_prompt:
            out += f"{IM_START}assistant\n"
        return out

    def count(self, text: str) -> int:
        return math.ceil(len(text.replace(SPLICE_SENTINEL, "")) / self.cpt)


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


@dataclass
class _Msg:
    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    slot: Slot | None = None  # the content is a compressible tool result
    # Text around the slot INSIDE the message. A tool result is all slot; the
    # repo overview shares the opening user turn with the task instruction:
    # ``before + <overview> + after``.
    slot_before: str = ""
    slot_after: str = ""
    # Hidden reasoning of an assistant turn (teachers such as K2 Horizon
    # require it back in the history as ``reasoning_content``).
    reasoning: str | None = None
    # further slots of ONE over-long tool result, split at build time into
    # consecutive parts ("<prompt> (part i/n)"); the message renders one
    # sentinel per part. Empty for every ordinary message.
    more_slots: list[Slot] = field(default_factory=list)
    # A message of SEVERAL labelled slots (the memory index): the verbatim
    # label line in front of each slot, one per slot (``slot`` + ``more_slots``).
    # Empty for every other message.
    slot_labels: list[str] = field(default_factory=list)

    def _labelled(self, bodies: list[str]) -> str:
        entries = "\n".join(f"{lab}\n{b}" for lab, b in zip(self.slot_labels, bodies, strict=True))
        return f"{self.slot_before}{entries}{self.slot_after}"

    def full(self, teacher_view: bool = False) -> dict:
        content = self.content
        if teacher_view and self.slot is not None and self.slot.digest is not None:
            content = self.slot.digest
        if self.slot is not None and self.slot_labels:
            content = self._labelled([
                s.digest if (teacher_view and s.digest is not None) else s.text
                for s in (self.slot, *self.more_slots)
            ])
        elif self.slot is not None:
            content = f"{self.slot_before}{content}{self.slot_after}"
        d: dict = {"role": self.role, "content": content}
        if self.tool_calls:
            d["tool_calls"] = [c.as_message() for c in self.tool_calls]
        if self.reasoning is not None:
            d["reasoning_content"] = self.reasoning
        return d

    def slotted(self) -> dict:
        d = self.full()
        if self.slot is not None and self.slot_labels:
            d["content"] = self._labelled([SPLICE_SENTINEL] * (1 + len(self.more_slots)))
        elif self.slot is not None:
            sent = "\n".join([SPLICE_SENTINEL] * (1 + len(self.more_slots)))
            d["content"] = (
                f"{COMPACTION_HEADER}\n{SPLICE_SENTINEL}"
                if self.slot.tool == COMPACTION_TOOL
                else f"{self.slot_before}{sent}{self.slot_after}"
            )
        return d

    def plain(self) -> str:
        """The message as plain text (what a compaction slot absorbs)."""
        body = self.content
        if self.tool_calls:
            body += render_tool_calls(self.tool_calls)
        return f"{self.role}: {body}"


def _history_form(d: dict) -> dict:
    """An assistant turn that called tools carries no prose in the HISTORY (2026-09-23). The LFM2.5
    chat template already drops that prose from every tool-calling assistant turn except the most
    recent one, so a row whose prefix kept it was not prefix-consistent with its own target
    (render_chunk: 43/120 LFM-teacher rows unrenderable) and the model saw its last turn's prose
    only sometimes. The history now always omits it; a TARGET turn keeps its reasoning."""
    if d.get("role") == "assistant" and d.get("tool_calls") and d.get("content"):
        d = dict(d, content="")
    return d


class Conversation:
    """Message list + slots; renders full-text or with sentinels."""

    def __init__(
        self,
        system: str,
        tools: list[dict],
        renderer: Renderer,
        ratios: RatioPolicy | None = None,
        task_summary: str = "",
        compaction_mode: str = "flat",
        slot_max_tokens: int | None = None,
        compaction_leaf_max_tokens: int = 7680,
        spine_layout: dict | None = None,
        full_view_compaction_tokens: int | None = None,
    ) -> None:
        if compaction_mode not in COMPACTION_MODES:
            raise ValueError(f"compaction_mode must be one of {COMPACTION_MODES}")
        self.compaction_mode = compaction_mode
        # No silent truncation (2026-09-22): a compressible result longer than
        # ``slot_max_tokens`` (the encoder's max_ctx) becomes several consecutive
        # slots here, so train rows and eval episodes split it identically.
        # None = the historical behaviour (one slot; the encoder side cuts).
        self.slot_max_tokens = int(slot_max_tokens) if slot_max_tokens else None
        # cons mode: a fold's text above this many tokens becomes several nodes
        self.compaction_leaf_max_tokens = int(compaction_leaf_max_tokens)
        # the wide-cell layout the encoder builds from the leaves (must equal the
        # trainer's compaction_spine.* settings): prices the slot's decoder reps
        self.spine_layout = dict(spine_layout or {})
        # The FULL-TEXT view of the compaction slot (2026-09-23): without a cap it renders every
        # folded round verbatim, so compact() -- which measures the full view for a text model --
        # could never shorten a text model's prompt (the LFM teacher overflowed 16k and then 32k).
        # With a cap, a text model sees the TAIL of the folded history ("[earlier history
        # truncated]"): classic truncation compaction, the honest text baseline to bgkit's encoded
        # spine. The student's slotted view is unaffected.
        self.full_view_compaction_tokens = (
            int(full_view_compaction_tokens) if full_view_compaction_tokens else None
        )
        self.tools = list(tools)
        self.renderer = renderer
        self.ratios = ratios or RatioPolicy()
        self.task_summary = task_summary
        self._msgs: list[_Msg] = []
        if system:
            self._msgs.append(_Msg("system", system))
        self._head = len(self._msgs)  # messages never folded by compaction
        self.n_compactions = 0

    # -- building -----------------------------------------------------------
    def add_user_with_overview(
        self, instruction: str, repo: str, listing: str, n_reps: int
    ) -> Slot:
        """The opening user turn with an ALWAYS-ON repo overview in front of the
        task: ``OVERVIEW_HEADER <overview> \n\n <instruction>``.

        The student reads the overview as the cached ROOT reps of the repo's
        whole-tree encoding (``n_reps`` of them); the slot's context is only a
        marker naming the repo (``overview_marker``) -- nothing is encoded at
        run time. A text-only teacher, and the full-text arm, read ``listing``
        (the repo's file paths) in the same position. The turn sits in the
        conversation head, so compaction never folds it."""
        slot = Slot(overview_marker(repo, n_reps), "", OVERVIEW_TOOL, self.renderer.count(listing),
                    int(n_reps), digest=listing)
        self._msgs.append(_Msg("user", listing, slot=slot,
                               slot_before=OVERVIEW_HEADER, slot_after=f"\n\n{instruction}"))
        self.repo = repo
        if self._head == len(self._msgs) - 1:
            self._head = len(self._msgs)
        return slot

    def add_user_with_slots(
        self, before: str, entries: list[tuple[str, Slot]], after: str
    ) -> list[Slot]:
        """The opening user turn with SEVERAL always-on labelled slots:
        ``before + label_0 \\n <slot_0> \\n label_1 \\n <slot_1> ... + after``
        (the two-stage memory INDEX: one slot per past session under the
        generic index prompt, its label the session id + date the agent passes
        to ``memory_read``). The slots are built by the caller (their rep
        counts are the caller's policy, e.g. ``memory_framing.index_reps``).
        The turn is part of the conversation head: compaction never folds it."""
        if not entries:
            raise ValueError("add_user_with_slots needs at least one slot")
        labels = [lab for lab, _ in entries]
        slots = [s for _, s in entries]
        msg = _Msg("user", "", slot=slots[0], slot_before=before, slot_after=after,
                   more_slots=slots[1:], slot_labels=labels)
        msg.content = msg._labelled([s.text for s in slots])
        self._msgs.append(msg)
        if self._head == len(self._msgs) - 1:
            self._head = len(self._msgs)
        return slots

    def add_user(self, text: str) -> None:
        self._msgs.append(_Msg("user", text))
        if self._head == len(self._msgs) - 1:
            self._head = len(self._msgs)  # the task message stays

    def add_assistant(
        self,
        content: str,
        tool_calls: list[ToolCall] | None = None,
        reasoning: str | None = None,
    ) -> None:
        self._msgs.append(_Msg("assistant", content, list(tool_calls or []), reasoning=reasoning))

    def add_tool_result(
        self,
        text: str,
        compressible: bool,
        prompt: str,
        tool: str = "",
        digest: str | None = None,
    ) -> Slot | None:
        """Append a tool turn; a compressible one becomes a slot. ``digest``
        is what a teacher reads in place of ``text`` (see Slot.digest)."""
        slot = None
        more: list[Slot] = []
        if compressible:
            n = self.renderer.count(text)
            if self.slot_max_tokens and n > self.slot_max_tokens:
                parts = split_by_tokens(text, self.slot_max_tokens, self.renderer.count)
                k = len(parts)
                made = []
                for i, part in enumerate(parts):
                    npart = self.renderer.count(part)
                    made.append(Slot(part, f"{prompt} (part {i + 1}/{k})".strip(), tool, npart,
                                     self.ratios.reps(npart, tool)))
                slot, more = made[0], made[1:]
            else:
                slot = Slot(text, prompt, tool, n, self.ratios.reps(n, tool), digest=digest)
        self._msgs.append(_Msg("tool", text, slot=slot, more_slots=more))
        return slot

    def add_cached_tool_result(
        self, listing: str, marker: str, n_reps: int, prompt: str, tool: str
    ) -> Slot:
        """A tool turn whose slot is a CACHED tree node (``browse``): the
        student reads the node's reps (``marker`` names it), a text-only
        teacher and the full-text arm read ``listing``."""
        slot = Slot(marker, prompt, tool, self.renderer.count(listing), int(n_reps),
                    digest=listing)
        self._msgs.append(_Msg("tool", listing, slot=slot))
        return slot

    # -- views --------------------------------------------------------------
    @property
    def slots(self) -> list[Slot]:
        return [s for m in self._msgs if m.slot is not None for s in (m.slot, *m.more_slots)]

    # When set, ``messages_full`` / ``render_full`` show each digested slot's
    # DIGEST instead of the raw tool result — the teacher's view. The student
    # views (``messages_with_slots``, ``contexts``) are unaffected.
    teacher_view: bool = False

    def messages_full(self) -> list[dict]:
        out = []
        for m in self._msgs:
            d = m.full(self.teacher_view)
            cap = self.full_view_compaction_tokens
            if cap and m.slot is not None and m.slot.tool == COMPACTION_TOOL:
                d["content"] = self._tail(str(d["content"]), cap)
            out.append(_history_form(d))
        return out

    TRUNCATED_HEAD = "[earlier history truncated]\n"

    def _tail(self, text: str, cap: int) -> str:
        """The longest suffix of ``text`` within ``cap`` tokens, marked as truncated."""
        if self.renderer.count(text) <= cap:
            return text
        lo, hi = 0, len(text)  # smallest start i with count(text[i:]) <= cap
        while lo < hi:
            mid = (lo + hi) // 2
            if self.renderer.count(text[mid:]) <= cap:
                hi = mid
            else:
                lo = mid + 1
        return self.TRUNCATED_HEAD + text[lo:]

    def digests(self) -> list[str | None]:
        return [s.digest for s in self.slots]

    def messages_with_slots(self) -> list[dict]:
        return [_history_form(m.slotted()) for m in self._msgs]

    def contexts(self) -> list[str]:
        return [s.text for s in self.slots]

    def prompts(self) -> list[str]:
        return [s.prompt for s in self.slots]

    def render_full(self, add_generation_prompt: bool = True) -> str:
        return self.renderer.render(self.messages_full(), self.tools, add_generation_prompt)

    def render_slots(self, add_generation_prompt: bool = True) -> str:
        return self.renderer.render(self.messages_with_slots(), self.tools, add_generation_prompt)

    def prefix_tokens(self, compressed: bool) -> int:
        """Decoder positions of the prompt: every slot counts as its reps when
        ``compressed``, as its full text otherwise. Measured on the actual
        rendering, so it can differ from ``full - sum(n_tokens - n_reps)`` by
        up to one token per slot (tokenizer boundary / rounding effects)."""
        if not compressed:
            return self.renderer.count(self.render_full())
        return self.renderer.count(self.render_slots()) + sum(s.n_reps for s in self.slots)

    def slot_tokens(self) -> int:
        return sum(s.n_tokens for s in self.slots)

    # -- compaction ---------------------------------------------------------
    def _compaction_index(self) -> int | None:
        i = self._head
        if i < len(self._msgs) and self._msgs[i].role == "user":
            slot = self._msgs[i].slot
            if slot is not None and slot.tool == COMPACTION_TOOL:
                return i
        return None

    def _rounds(self) -> list[tuple[int, int]]:
        """(start, end) of each foldable round: an assistant message plus the
        tool / user messages that follow it, after the head and the
        compaction message."""
        first = self._head + (1 if self._compaction_index() is not None else 0)
        rounds: list[tuple[int, int]] = []
        for i in range(first, len(self._msgs)):
            if self._msgs[i].role == "assistant" or not rounds:
                rounds.append((i, i + 1))
            else:
                rounds[-1] = (rounds[-1][0], i + 1)
        return rounds

    def compact(self, budget: int, keep_recent: int = 1, compressed: bool = True) -> bool:
        """Fold the oldest rounds into the compaction slot until the prefix
        (measured in the ``compressed`` view, or the full/teacher view) fits
        ``budget`` or only ``keep_recent`` rounds remain. Returns whether
        anything was folded.

        ``compaction_mode`` "flat" (default): the slot's text is the previous
        compaction's TEXT plus the newly folded rounds -- one ever-growing slot.
        "cons": every compaction event appends the newly folded rounds as LEAVES
        (``compaction_payload`` / ``compaction_leaves``); the encoder groups the
        leaves into at most 4 wide cells (``bgkit2.data.spine``: leaves at 1/16,
        cell-to-cell at 1/4), so the slot's cost is the top cell's reps
        (``spine_top_reps``) and stays bounded however long the trajectory."""
        if self.compaction_mode == "cons":
            return self._compact_cons(budget, keep_recent, compressed)
        folded = False
        while self.prefix_tokens(compressed=compressed) > budget:
            rounds = self._rounds()
            if len(rounds) <= keep_recent:
                break
            start, end = rounds[0]
            text = "\n".join(m.plain() for m in self._msgs[start:end])
            ci = self._compaction_index()
            if ci is None:
                self._msgs.insert(self._head, _Msg("user", ""))
                ci = self._head
                start, end = start + 1, end + 1
                prev = ""
            else:
                existing = self._msgs[ci].slot
                prev = existing.text if existing is not None else ""
            joined = f"{prev}\n{text}" if prev else text
            n = self.renderer.count(joined)
            self._msgs[ci].content = joined
            self._msgs[ci].slot = Slot(
                joined,
                f"{COMPACTION_TOOL}: {self.task_summary}",
                COMPACTION_TOOL,
                n,
                self.ratios.reps(n, COMPACTION_TOOL),
            )
            del self._msgs[start:end]
            folded = True
        self.n_compactions += int(folded)
        return folded

    def _compact_cons(self, budget: int, keep_recent: int, compressed: bool) -> bool:
        pieces: list[list[str]] = []  # folded rounds (oldest first), each a list of messages
        child: dict | None = None
        prev_plain = ""
        ci = self._compaction_index()
        if ci is not None and self._msgs[ci].slot is not None:
            child = json.loads(self._msgs[ci].slot.text)
            prev_plain = self._msgs[ci].content
        while self.prefix_tokens(compressed=compressed) > budget:
            rounds = self._rounds()
            if len(rounds) <= keep_recent:
                break
            start, end = rounds[0]
            pieces.append([m.plain() for m in self._msgs[start:end]])
            ci = self._compaction_index()
            if ci is None:
                self._msgs.insert(self._head, _Msg("user", ""))
                ci = self._head
                start, end = start + 1, end + 1
            node = child
            for leaf in self._pack_leaves(pieces):
                node = {COMPACTION_TOOL: 1, "child": node, "leaf": leaf}
            # decoder cost = the TOP CELL's reps of the wide-cell layout
            # (bgkit2.data.spine), the same layout the encoder builds
            counts = [self._leaf_count(x) for x in compaction_leaves(node)]
            n = sum(counts)
            reps = spine_top_reps(counts, **self.spine_layout)
            plain = "\n".join("\n".join(p) for p in pieces)
            # the full-text / teacher view reads the folded history as text
            self._msgs[ci].content = f"{prev_plain}\n{plain}" if prev_plain else plain
            self._msgs[ci].slot = Slot(
                json.dumps(node, ensure_ascii=False),
                f"{COMPACTION_TOOL}: {self.task_summary}",
                COMPACTION_TOOL,
                n,
                reps,
            )
            del self._msgs[start:end]
        self.n_compactions += int(bool(pieces))
        return bool(pieces)

    def _leaf_count(self, leaf: str) -> int:
        cache = self.__dict__.setdefault("_leaf_counts", {})
        n = cache.get(leaf)
        if n is None:
            n = cache[leaf] = self.renderer.count(leaf)
        return n

    def _pack_leaves(self, rounds: list[list[str]]) -> list[str]:
        """The folded rounds as leaves of at most ``compaction_leaf_max_tokens``:
        whole rounds while they fit, an over-long round at message boundaries, an
        over-long message at token boundaries (``split_by_tokens``)."""
        cap = self.compaction_leaf_max_tokens
        count = self.renderer.count
        units: list[str] = []
        for msgs in rounds:
            text = "\n".join(msgs)
            if count(text) <= cap:
                units.append(text)
                continue
            for m in msgs:
                units.extend([m] if count(m) <= cap else split_by_tokens(m, cap, count))
        leaves: list[str] = []
        cur = ""
        for u in units:
            cand = f"{cur}\n{u}" if cur else u
            if cur and count(cand) > cap:
                leaves.append(cur)
                cur = u
            else:
                cur = cand
        if cur:
            leaves.append(cur)
        return leaves

    # -- export -------------------------------------------------------------
    def to_task_row(
        self,
        target: list[dict] | str | None = None,
        family: str = "agent_tool_slots",
        split: str = "eval",
        meta: dict | None = None,
    ) -> dict:
        """A ``tool_slots`` task-store row (framing ``chat`` when there is no
        slot yet). ``target`` is the assistant chunk that follows (a message
        list) or plain text; None leaves it empty (a logged prefix)."""
        slots = self.slots
        tgt = json.dumps(target) if isinstance(target, list) else (target or "")
        msgs = self.messages_with_slots()
        if not slots and self.tools:
            # the chat framing renders no ``tools=``: write the list into the
            # system message so the row renders the tokens the eval prompt had
            if msgs and msgs[0]["role"] == "system":
                msgs[0] = dict(msgs[0], content=system_with_tool_list(msgs[0]["content"],
                                                                      self.tools))
            else:
                msgs.insert(0, {"role": "system",
                                "content": system_with_tool_list("", self.tools)})
        return {
            "family": family,
            "framing": "tool_slots" if slots else "chat",
            "context": json.dumps(self.contexts()) if slots else "",
            "prompt": json.dumps(self.prompts()) if slots else "",
            "instruction": json.dumps(msgs),
            "tool_name": "",
            "tool_args": "{}",
            "target": tgt,
            "split": split,
            "meta": dict(
                meta or {},
                **({"repo": self.repo} if getattr(self, "repo", "") else {}),
                tools=self.tools,
                n_slots=len(slots),
                digests=self.digests() if slots else [],
            ),
        }


CountFn = Callable[[str], int]
