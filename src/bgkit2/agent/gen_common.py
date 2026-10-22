"""Shared pieces of the synthetic trajectory generators (repo_qa /
localisation questions and real-fix edits): key-term extraction, the
EXECUTION-VERIFIED discovery call, and the scripted model that replays a
planned turn list through the real harness.

Kept in one place because the grounding rule is the thing that makes these
rows honest: the first tool result must NAME the file the agent then opens,
and `find -name <basename>` may only be used when the task text itself names
that file (audits of 2026-09-14; see plans/rep_recipe_port_2026_09_07.md).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from bgkit2.agent.context import ToolCall
from bgkit2.agent.tools import ToolRegistry

_STOPWORDS = (
    "the a an of to in for and or with from that this into when where which while "
    "fix fixed fixes bug bugs issue error errors handle handling support add added "
    "remove removed update updated use using make made not now also should would"
)


STOP = set(_STOPWORDS.split())


BACKTICK = re.compile(r"`([^`\n]{2,80})`")


WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{4,}")


SEARCH_TEXT = [
    "I'll search the repository for `{t}` first.",
    "Let me look for `{t}` across the repository.",
    "Starting with a search for `{t}`.",
    "`{t}` is the obvious thing to look for.",
    "First I need to find where `{t}` appears.",
]


GREP_TEXT = [
    "Let me grep for what the question points at:\n{c}",
    "A quick grep should locate it:\n{c}",
    "I'll search the sources directly:\n{c}",
    "Let me find the relevant lines:\n{c}",
]


def key_terms(text: str) -> list[str]:
    """Search terms from a question / commit message: backticked identifiers
    first, then long words that are not stop words (longest first)."""
    out = [m.strip() for m in BACKTICK.findall(text)]
    words = [w for w in WORD.findall(text) if w.lower() not in STOP]
    out += sorted(set(words), key=lambda w: -len(w))
    seen: list[str] = []
    for t in out:
        if t not in seen:
            seen.append(t)
    return seen


def def_line(text: str, name: str) -> int | None:
    m = re.search(rf"^[ \t]*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b", text, re.M)
    return text[: m.start()].count("\n") + 1 if m else None


def call_text(c: ToolCall, rng: random.Random) -> str:
    """A neutral sentence introducing a discovery call."""
    if c.name == "search":
        return rng.choice(SEARCH_TEXT).format(t=c.args["query"])
    return rng.choice(GREP_TEXT).format(c=c.args.get("cmd", ""))


MISS_TEXT = [
    "Nothing useful there. Let me try a different angle:",
    "That was not it; searching for something more specific:",
    "No luck. Another search:",
    "That did not surface the right file. Trying again:",
]


def miss_call(
    tools: ToolRegistry, terms: list[str], path: str, focus: str, rng: random.Random
) -> tuple[ToolCall, str] | None:
    """A first call that HONESTLY misses: its result is non-empty but does NOT
    name ``path``, so the episode shows a real dead end and a real recovery.
    The student otherwise only ever sees searches that work (2026-09-14)."""
    cands = [ToolCall("search", {"query": t, "focus": focus}) for t in terms[:4]]
    cands += [
        ToolCall("bash", {"cmd": f"grep -rn {json.dumps(t)} --include='*.py' . | head -20"})
        for t in terms[:4]
    ]
    rng.shuffle(cands)
    for c in cands:
        try:
            res = tools.call(c.name, c.args)
        except Exception:
            continue
        text = res.text.strip()
        if not text or path in res.text or len(text) < 40 or "no matches" in text[:40].lower():
            continue
        return c, call_text(c, rng)
    return None


BAD_READ_TEXT = [
    "Let me read `{p}` directly.",
    "I'll start by opening `{p}`.",
    "`{p}` sounds like the file; reading it.",
]
RECOVER_TEXT = [
    "That path does not exist — I need to find where the file actually lives.",
    "No such file; let me locate it properly.",
    "Wrong path. Searching for it instead:",
    "It is not at the repository root. Finding the real path:",
]


def bad_path_call(
    tools: ToolRegistry, path: str, focus: str, rng: random.Random, task_text: str = ""
) -> tuple[ToolCall, str] | None:
    """A read that GENUINELY FAILS, on the evidence file's bare basename.

    P26's first agent eval (step 6092): 31% of localisation and 38% of repo_qa
    first-turn reads passed a bare basename ("interpreter.py" for
    "src/pipx/interpreter.py"), the tool errored, no slot was created, and the
    model then repeated the identical failing call until the loop guard fired
    (37 of 60 episodes). Every synthetic trajectory had opened with a discovery
    call that SUCCEEDS, so the student had never seen a failed call, let alone
    a recovery. None when the basename happens to resolve (nothing to recover
    from) or the tool does not error.
    """
    base = Path(path).name
    if base == path:
        return None  # already at the repo root: reading it would succeed
    # Only open with a guess the task text actually supports. 75% of
    # localisation commit messages never name the file (measured: 3,769 of
    # 5,000), so on those this shape taught "open by reading a word you cannot
    # know" — and the step-12069 agent eval showed exactly that: 249 read calls,
    # 0 searches, paths like "pydantic" lifted straight out of the message.
    # Same principle as the find -name rule: never teach an action whose
    # argument is not derivable from the task.
    if task_text and base.rsplit(".", 1)[0].lower() not in task_text.lower():
        return None
    call = ToolCall("read", {"path": base, "focus": focus})
    try:
        res = tools.call("read", call.args)
    except Exception:
        return call, rng.choice(BAD_READ_TEXT).format(p=base)
    if res.ok is False or res.text.lower().startswith("error"):
        return call, rng.choice(BAD_READ_TEXT).format(p=base)
    return None


def discovery_call(
    tools: ToolRegistry,
    terms: list[str],
    path: str,
    focus: str,
    question: str,
    rng: random.Random,
) -> tuple[ToolCall, str] | None:
    """The first call, verified by RUNNING it: the result must name ``path``.
    Tries search / grep on each key term (order shuffled per task), then
    `find -name <basename>` — but ONLY when the question itself names that file,
    since otherwise the call presupposes the answer (audit 2026-09-14: 28 of
    13,701 episodes grepped for a filename the question never mentioned).
    None when nothing surfaces the file."""
    base = Path(path).name
    cands: list[ToolCall] = []
    # `ls` on the ROOT only. Listing the answer's own directory would
    # presuppose knowing it, the clairvoyance defect from the 2026-09-14 audit.
    # It is shuffled in with the term searches rather than always placed first,
    # so the student does not learn "always ls" as a reflex, and the
    # verification below drops it whenever the listing is truncated before the
    # target file.
    # `overview` first when it surfaces the file: its result is COMPRESSIBLE, so
    # the map becomes an encoder slot and locating the file becomes a
    # compression task rather than a guess. `ls` (bare paths) is the fallback.
    if rng.random() < 0.6:
        cands.append(ToolCall("overview", {"focus": focus}))
    if rng.random() < 0.4:
        cands.append(ToolCall("ls", {"focus": focus}))
    for term in terms[:3]:
        pair = [
            ToolCall("search", {"query": term, "focus": focus}),
            ToolCall("bash", {"cmd": f"grep -rn {json.dumps(term)} --include='*.py' . | head -40"}),
        ]
        rng.shuffle(pair)
        cands += pair
    if base.rsplit(".", 1)[0].lower() in question.lower():
        cands.append(ToolCall("bash", {"cmd": f"find . -name {json.dumps(base)} | head -20"}))
    for c in cands:
        try:
            res = tools.call(c.name, c.args)
        except Exception:
            continue
        if path in res.text:  # the file the agent reads next must be named here
            return c, call_text(c, rng)
    return None


class Scripted:
    def __init__(self, turns: list[tuple[str, str]]):
        self.turns, self.i = turns, 0

    def __call__(self, _prompt: str) -> str:
        text, call = self.turns[min(self.i, len(self.turns) - 1)]
        self.i += 1
        return f"{text}{call}<|im_end|>"
