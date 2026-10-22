#!/usr/bin/env python
"""Remap the EXTERNAL tool_slots store into our own tool schema.

The 39k rows built from OpenHands / SWE-Zero / OpenResearcher trajectories call
`str_replace_editor`, `execute_bash`, `think`, `task_tracker`, `finish`, and
their `meta.tools` holds bare tool NAMES rather than JSON schemas — so the chat
template refuses every row (`Tools should either be a JSON schema...`) and the
store would have crashed the P26 dataloader on its first batch (caught by the
pre-launch check, 2026-09-15).

Beyond the crash, training on foreign call names would teach the student to
emit tools our harness does not have. This pass rewrites every assistant call
into the harness schema:

    str_replace_editor view        -> read(path) or zoom(path, start, end) with view_range
    str_replace_editor str_replace     -> edit(path, old, new)
    str_replace_editor create/insert   -> write(path, text)   (create only)
    execute_bash                       -> bash(cmd)
    think / task_tracker / finish      -> dropped (no equivalent; the turn's text is kept)

and sets `meta.tools` to the real registry schemas. A row is kept only when
every call in its instruction AND target maps, and the sentinel count still
matches the context list.

Exact-quote tools go through bgkit (2026-09-22, user): every `view_range` read
becomes an ENCODED `zoom` -- its result a compressible slot prompted
"zoom <path> lines <a>-<b>: <purpose>" (priced at 4x by the harness), including
right before an edit: predicting an edit's verbatim `old` from reps is the
capability mid-training builds, so no row carries `exact`. The trajectory holds
only the VIEWED lines (str_replace_editor's `cat -n` of the range, sometimes
already elided by the source store with "[...]"), not the whole file, so the
slot is that text and the prompt names the range. An inline view result becomes
a slot (``zoom_slotted``); error results ("ERROR: Invalid `view_range`...")
stay inline. With --web, `browser_find` results become encoded `web_find` slots
the same way (the find display is the page text the trajectory has; prompt
"find <pattern> in <page>: <purpose>"; "No `find` results" -> the harness's
inline "not found"), and every web slot's prompt head is rebuilt the harness
way (the synthesised prompts named "bash browser_search {...}").
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shlex
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlparse

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bgkit2.agent.tools import ToolRegistry, zoom_prompt
from bgkit2.agent.web import web_find_prompt
from bgkit2.data.reasoning import focus_from_reasoning as derive_focus
from bgkit2.data.reasoning import join_reasoning, move_reasoning, split_reasoning
from bgkit2.data.tasks import has_reasoning, write_task_store
from bgkit2.data.templates import SPLICE_SENTINEL

DROP = {"think", "task_tracker", "finish"}
# Tools whose result a `focus` steers (ToolRegistry's compressible readers): with
# --focus-from-reasoning a focus-less call of one gets its focus from the issuing turn's reasoning.
READ_TOOLS = {
    "read", "zoom", "search", "docs", "repo_tree", "web_search", "web_open", "web_find",
    "memory_search", "memory_read",
}
WEB = False  # --web: map OpenResearcher's browser_* calls onto the offline web tools (agent/web.py)
# --ext-harnesses: the SWE-agent and OpenCode (Nemotron-SFT-SWE) tool sets as well:
#   SWE-agent  submit                          -> dropped (a no-op like finish)
#   OpenCode   read(filePath)                  -> read(path)
#              read(filePath, offset, limit)   -> zoom(path, offset, offset + limit - 1)
#              grep(pattern, path, include)    -> search(query=pattern)
#              glob(pattern, path)             -> bash('rg --files <path> -g "<pattern>"')
#              list(path)                      -> ls(dir=path)
#              edit(filePath, oldString, newString) -> edit(path, old, new)
#              write(filePath, content)        -> write(path, content)
#              bash(command, workdir)          -> bash("cd <workdir> && <command>")
#              todowrite / todoread / task     -> dropped (planning notes; the `task` sub-agent
#                                                 results in the source are empty)
# and `create` -> write(path, content): the registry's write takes `content` (the legacy
# mapping's `text` is kept for the default path, so existing builds stay byte-identical).
EXT = False
EXT_DROP = {"submit", "todowrite", "todoread", "task"}
OPENCODE_READ_LIMIT = 2000  # OpenCode's read default line limit


def as_args(a: object) -> dict:
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


def drop_set() -> set[str]:
    """The no-op calls folded into the turn (with their results removed)."""
    return DROP | EXT_DROP if EXT else DROP


def _int(x, default: int) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _dq(x: str) -> str:
    """A shell word in DOUBLE quotes where that is exact (the 8B chat template renders call
    arguments in single quotes without escaping them), else ``shlex.quote``."""
    if x and not re.search(r'["$`\\!]', x):
        return x if re.fullmatch(r"[\w./-]+", x) else f'"{x}"'
    return shlex.quote(x)


def remap_ext(name: str, args: dict) -> tuple[str, dict] | None | bool:
    """The --ext-harnesses (OpenCode) calls; False = not one of them (fall through)."""
    path = str(args.get("filePath", args.get("path", ""))).strip()
    if name == "read" and "filePath" in args:
        if not path:
            return None
        if args.get("offset") is not None or args.get("limit") is not None:
            start = max(1, _int(args.get("offset"), 1))
            end = start + max(1, _int(args.get("limit"), OPENCODE_READ_LIMIT)) - 1
            return ("zoom", {"path": path, "start": start, "end": end})
        return ("read", {"path": path})
    if name == "grep":
        pat = str(args.get("pattern", "")).strip()
        return ("search", {"query": pat}) if pat else None
    if name == "glob":
        pat = str(args.get("pattern", "")).strip()
        if not pat:
            return None
        where = _dq(path) if path else "."
        return ("bash", {"cmd": f"rg --files {where} -g {_dq(pat)}"})
    if name == "list":
        return ("ls", {"dir": path} if path else {})
    if name == "edit" and "filePath" in args:
        old, new = args.get("oldString"), args.get("newString")
        if not path or not isinstance(old, str) or not old or not isinstance(new, str):
            return None
        return ("edit", {"path": path, "old": old, "new": new})
    if name == "write" and "filePath" in args:
        text = args.get("content")
        return ("write", {"path": path, "content": text}) if path and isinstance(text, str) \
            else None
    if name == "bash" and args.get("workdir"):
        cmd = str(args.get("command", args.get("cmd", ""))).strip()
        if not cmd:
            return None
        return ("bash", {"cmd": f"cd {_dq(str(args['workdir']))} && {cmd}"})
    return False


def remap(name: str, args: dict) -> tuple[str, dict] | None:
    """Our (tool, args), or None when the call has no equivalent."""
    if name in drop_set():
        return None
    if EXT:
        got = remap_ext(name, args)
        if got is not False:
            return got
    if name in ("execute_bash", "bash"):
        cmd = str(args.get("command", args.get("cmd", ""))).strip()
        return ("bash", {"cmd": cmd}) if cmd else None
    if WEB and name == "browser_search":
        q = str(args.get("query", "")).strip()
        return ("web_search", {"query": q}) if q else None
    if WEB and name == "browser_open":
        # OpenResearcher opens by result index (id) within a cursor; our web_open takes a page id.
        # The trajectory's own tool result carries the page text, so the id is a replay label.
        pid = args.get("id", args.get("url", ""))
        return ("web_open", {"id": str(pid)}) if str(pid) != "" else None
    if WEB and name == "browser_find":
        pat = str(args.get("pattern", "")).strip()
        return ("web_find", {"pattern": pat}) if pat else None
    if name in ("str_replace_editor", "str_replace_based_edit_tool"):
        cmd = str(args.get("command", "view"))
        path = str(args.get("path", "")).strip()
        if not path:
            return None
        if cmd == "view":
            vr = args.get("view_range")
            if isinstance(vr, list) and len(vr) == 2:
                try:
                    return ("zoom", {"path": path, "start": int(vr[0]), "end": int(vr[1])})
                except Exception:
                    return None
            return ("read", {"path": path})
        if cmd == "str_replace":
            old, new = args.get("old_str"), args.get("new_str")
            if not isinstance(old, str) or not old or not isinstance(new, str):
                return None
            return ("edit", {"path": path, "old": old, "new": new})
        if cmd == "create":
            text = args.get("file_text")
            key = "content" if EXT else "text"
            return ("write", {"path": path, key: text}) if isinstance(text, str) else None
        return None  # insert / undo_edit: no equivalent
    return None


def fold_text(name: str, args: dict) -> str:
    """A dropped no-op call still carries the model's words: keep them as text
    so the trajectory stays coherent instead of losing the whole row."""
    if name == "think":
        return str(args.get("thought", "")).strip()
    if name == "finish":
        return str(args.get("message", "")).strip()
    return ""


def turn_reasoning(m: dict) -> str:
    """All reasoning of an assistant message: its reasoning fields and think blocks, then the
    ``thought`` of each ``think`` call (for messages not yet remapped)."""
    thoughts = []
    for c in m.get("tool_calls") or []:
        fn = c.get("function", c) if isinstance(c, dict) else {}
        if fn.get("name") == "think":
            thoughts.append(str(as_args(fn.get("arguments", {})).get("thought", "")))
    return join_reasoning([split_reasoning(m)[0], *thoughts])


def _with_focus(prompt: str, a: dict, reasoning: str) -> str:
    """``prompt: <focus from reasoning>`` when the call has no focus of its own."""
    if str(a.get("focus") or "").strip():
        return prompt
    focus = derive_focus(reasoning)
    return f"{prompt}: {focus}" if focus else prompt


def slot_prompts(msgs: list[dict], focus_from_reasoning: bool = False) -> list[str]:
    """The compression prompt for each SENTINEL tool result, rebuilt from the
    remapped call that produced it: "<tool> <path|query|cmd>", the shape
    ToolRegistry._prompt emits when a call carries no focus. The store's own
    prompts are the EXTERNAL call text ("str_replace_editor: view /workspace"),
    which after remapping names a tool the conversation no longer has and says
    nothing about what to keep — and the P24 final ablation measured the prompt
    as worth 1.3-3.3 nats against 0.02-0.13 for the learned selection, so an
    empty prompt on 19k multi-slot rows is not a cosmetic problem.

    ``focus_from_reasoning`` (--focus-from-reasoning): a READ_TOOLS call with no ``focus``
    gets ``: <focus>`` derived from the issuing turn's reasoning
    (``bgkit2.data.reasoning.focus_from_reasoning``); unchanged when the turn has none."""
    out: list[str] = []
    pending: list[dict] = []
    reasoning = ""
    for m in msgs:
        if m.get("role") == "assistant":
            pending = list(m.get("tool_calls") or [])
            reasoning = turn_reasoning(m) if focus_from_reasoning else ""
            continue
        if m.get("role") != "tool":
            continue
        call = pending.pop(0) if pending else None
        if SPLICE_SENTINEL not in str(m.get("content", "")):
            continue
        if call is None:
            out.append("")
            continue
        out.append(call_prompt(call, reasoning))
    return out


def call_prompt(call: dict, reasoning: str = "") -> str:
    """The rebuilt compression prompt of one (remapped) call's result (see slot_prompts)."""
    fn = call.get("function", call)
    a = fn.get("arguments", {}) if isinstance(fn.get("arguments"), dict) else {}
    name = fn.get("name")
    if name == "zoom":
        p = zoom_prompt(str(a.get("path", "")), int(a.get("start", 0)), int(a.get("end", 0)))
    elif name == "web_search":
        p = f"web search {a.get('query', '')}".strip()
    elif name == "web_find":
        p = web_find_prompt(str(a.get("pattern", "")), "")
    else:
        head = a.get("path") or a.get("query") or a.get("cmd") or a.get("dir") or ""
        p = f"{fn.get('name', '')} {head}".strip()
    if reasoning and name in READ_TOOLS:
        p = _with_focus(p, a, reasoning)
    return p


CAT_N = "Here's the result of running `cat -n`"
_FIND_IN = re.compile(r"Find results for text: `.*?` in `(.*?)`")


def _purpose(prompt: str, name: str, args: dict) -> str:
    """The synthesised purpose of a slot prompt, without its head ("read <path>: <purpose>",
    "bash <external tool> {json}: <purpose>" -- bgkit2_synth_slot_prompts.py's heads)."""
    p = str(prompt or "").strip()
    path = str(args.get("path", ""))
    if path and p.startswith(f"read {path}: "):
        return p[len(f"read {path}: ") :].strip()
    if p.startswith(f"bash {name} "):
        m = re.match(r"^bash \S+ \{.*?\}: ", p)
        return p[m.end() :].strip() if m else ""
    return p


def _page_title(text: str) -> str:
    """A page title from a gpt-oss page display header ``[k] <title|Doc N> (<url>)``."""
    m = re.match(r"^\[\d+\] (.*?)(?: \((\S+)\))?\s*$", text.split("\n", 1)[0])
    if not m:
        return ""
    title, url = m.group(1).strip(), (m.group(2) or "").strip()
    if re.fullmatch(r"Doc \S+", title) and url:
        seg = [x for x in unquote(urlparse(url).path).split("/") if x]
        t = " ".join(re.sub(r"[_\-+]+", " ", re.sub(r"\.(html?|php|aspx?|pdf)$", "", seg[-1],
                                                   flags=re.I)).split()) if seg else ""
        return t if t and not t.isdigit() else (urlparse(url).netloc or url)
    return title


def _web_head(name: str, args: dict, text: str) -> str | None:
    """The harness prompt head of a remapped web result (None: not a web call)."""
    if name == "browser_search":
        return f"web search {str(args.get('query', '')).strip()}"
    if name == "browser_open":
        return f"page {_page_title(text) or str(args.get('id', args.get('url', ''))).strip()}"
    if name == "browser_find":
        m = _FIND_IN.search(text.split("\n", 1)[0])
        return web_find_prompt(str(args.get("pattern", "")).strip(), m.group(1) if m else "")
    return None


def encode_exact_quote_results(ins_raw: str, ctx: list[str], prompts: list[str] | None,
                               st: Counter, focus_from_reasoning: bool = False
                               ) -> tuple[str, list[str], list[str] | None]:
    """On the ORIGINAL instruction: every view_range result (and, with --web, browser_find
    result) becomes a slot with the harness prompt; existing slots keep their synthesised
    purpose under the new head. ``prompts`` None = the caller rebuilds them from the calls.
    ``focus_from_reasoning``: a slot left without a purpose gets one from the issuing turn's
    reasoning (see slot_prompts)."""
    msgs = json.loads(ins_raw)
    if isinstance(msgs, dict):
        msgs = [msgs]
    new_ctx: list[str] = []
    new_pr: list[str] = []
    k = 0
    pending: list[dict] = []
    reasoning = ""
    changed = False
    for m in msgs:
        if not isinstance(m, dict):
            return ins_raw, ctx, prompts
        if m.get("role") == "assistant":
            pending = list(m.get("tool_calls") or [])
            reasoning = turn_reasoning(m) if focus_from_reasoning else ""
            continue
        if m.get("role") != "tool":
            continue
        call = pending.pop(0) if pending else {}
        fn = call.get("function", call) if isinstance(call, dict) else {}
        name, a = str(fn.get("name", "")), as_args(fn.get("arguments", {}))
        content = str(m.get("content", ""))
        is_slot = SPLICE_SENTINEL in content
        text = ctx[k] if is_slot else content
        old_prompt = (prompts[k] if prompts is not None else "") if is_slot else ""
        vr = a.get("view_range")
        head = None
        if (name in ("str_replace_editor", "str_replace_based_edit_tool")
                and str(a.get("command", "view")) == "view" and isinstance(vr, list)
                and len(vr) == 2):
            try:
                head = zoom_prompt(str(a.get("path", "")).strip(), int(vr[0]), int(vr[1]))
            except (TypeError, ValueError):
                head = None
            ok = is_slot or text.startswith(CAT_N)
            kind = "zoom"
        elif WEB and name in ("browser_search", "browser_open", "browser_find"):
            head = _web_head(name, a, text)
            ok = is_slot or (name == "browser_find" and "No `find` results" not in text
                             and not text.lstrip().lower().startswith("error"))
            kind = "find" if name == "browser_find" else "web"
            if name == "browser_find" and not is_slot and not ok:
                pat = str(a.get("pattern", "")).strip()
                m["content"] = f"{pat!r} not found on the page"
                st["find_not_found_inline"] += 1
                changed = True
        if head is None or not ok:
            if head is not None:
                st[f"{kind}_kept_inline"] += 1
            if is_slot:
                new_ctx.append(text)
                new_pr.append(old_prompt)
                k += 1
            continue
        purpose = _purpose(old_prompt, name, a) if is_slot else ""
        st[f"{kind}_{'slot_reprompted' if is_slot else 'slotted'}"] += 1
        if not is_slot:
            st[f"{kind}_slotted_elided"] += "\n[...]\n" in text
            m["content"] = SPLICE_SENTINEL
            changed = True
        else:
            k += 1
        new_ctx.append(text)
        if purpose:
            new_pr.append(f"{head}: {purpose}")
        else:
            new_pr.append(_with_focus(head, a, reasoning) if reasoning else head)
    if k != len(ctx):
        return ins_raw, ctx, prompts  # sentinel / context mismatch: left to the caller's check
    if not changed and prompts is None:
        return ins_raw, ctx, prompts
    return json.dumps(msgs, ensure_ascii=False), new_ctx, (new_pr if prompts is not None
                                                            else None)


def strip_dropped_slots(ins_raw: str, ctx: list[str], prompts: list[str] | None
                        ) -> tuple[str, list[str], list[str] | None]:
    """--ext-harnesses: the result of a no-op call (drop_set) that the builder made a slot (a
    large ``todowrite`` list, a ``submit`` diff) loses its slot here, context and prompt with it,
    so ``remap_messages`` removing the result keeps sentinels and contexts aligned. Unchanged
    when no such slot exists or the sentinel count does not match the contexts."""
    msgs = json.loads(ins_raw)
    if isinstance(msgs, dict):
        msgs = [msgs]
    if not isinstance(msgs, list):
        return ins_raw, ctx, prompts
    drops = drop_set()
    pending: list[str] = []
    k = 0
    gone: set[int] = set()
    for m in msgs:
        if not isinstance(m, dict):
            return ins_raw, ctx, prompts
        if m.get("role") == "assistant":
            pending = [str((c.get("function", c) if isinstance(c, dict) else {}).get("name", ""))
                       for c in m.get("tool_calls") or []]
            continue
        if m.get("role") != "tool":
            continue
        name = pending.pop(0) if pending else ""
        if SPLICE_SENTINEL not in str(m.get("content", "")):
            continue
        if name in drops:
            gone.add(k)
            m["content"] = "(no-op result)"
        k += 1
    if not gone or k != len(ctx):
        return ins_raw, ctx, prompts
    keep = [i for i in range(len(ctx)) if i not in gone]
    return (json.dumps(msgs, ensure_ascii=False), [ctx[i] for i in keep],
            [prompts[i] for i in keep] if prompts is not None else None)


def remap_messages(raw: str, keep_reasoning: bool = False) -> str | None:
    """Rewrite every tool call in a JSON message list. Calls with no equivalent
    in our harness (think / task_tracker / finish) are folded into the
    assistant's text and their tool-result message is removed with them, so a
    no-op in the HISTORY no longer costs the whole row (it cost 22,419 of
    40,005 rows in the first pass). None when a real call cannot be mapped.

    ``keep_reasoning`` (--keep-reasoning): every assistant message's reasoning
    (fields + think blocks, ``bgkit2.data.reasoning.move_reasoning``) goes to
    ``reasoning_content`` and a ``think`` call's ``thought`` is appended there
    instead of to the content. A turn left with reasoning but no content or
    calls (an OpenHands think-only turn) is dropped and its reasoning carried
    to the next assistant turn of the same list (reset by a user message)."""
    try:
        msgs = json.loads(raw)
    except Exception:
        return None
    if isinstance(msgs, dict):
        msgs = [msgs]
    if not isinstance(msgs, list):
        return None
    out: list[dict] = []
    skip_results = 0
    # --ext-harnesses: a dropped call's result is removed BY POSITION (the legacy counter drops
    # the next N results, which removes the wrong one when a no-op call follows a real call in
    # the same turn -- OpenCode's parallel read + todowrite / task).
    keep_flags: list[bool] = []
    carry: list[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            return None
        m = dict(m)
        if EXT and m.get("role") == "tool":
            if not (keep_flags.pop(0) if keep_flags else True):
                continue
        elif m.get("role") == "tool" and skip_results:
            skip_results -= 1
            continue
        if keep_reasoning and m.get("role") == "user":
            carry = []
        if keep_reasoning and m.get("role") == "assistant":
            m = move_reasoning(m)
            if carry:
                m["reasoning_content"] = join_reasoning([*carry, m.get("reasoning_content", "")])
                carry = []
        calls = m.get("tool_calls") or []
        if m.get("role") == "assistant":
            keep_flags = []
        if calls:
            new_calls = []
            extra_text: list[str] = []
            thoughts: list[str] = []
            for c in calls:
                fn = c.get("function", c)
                name, a = str(fn.get("name", "")), as_args(fn.get("arguments", {}))
                got = remap(name, a)
                if got is None:
                    if name not in drop_set():
                        return None
                    skip_results += 1  # its tool result goes too
                    keep_flags.append(False)
                    txt = fold_text(name, a)
                    if txt and keep_reasoning and name == "think":
                        thoughts.append(txt)
                    elif txt:
                        extra_text.append(txt)
                    continue
                keep_flags.append(True)
                new_calls.append(
                    {"type": "function", "function": {"name": got[0], "arguments": got[1]}}
                )
            if extra_text:
                m["content"] = "\n".join([str(m.get("content", "")).strip(), *extra_text]).strip()
            if thoughts:
                m["reasoning_content"] = join_reasoning([m.get("reasoning_content", ""), *thoughts])
            if new_calls:
                m["tool_calls"] = new_calls
            else:
                m.pop("tool_calls", None)
                if not str(m.get("content", "")).strip():
                    if keep_reasoning and m.get("reasoning_content"):
                        carry.append(m["reasoning_content"])
                    continue  # nothing left of this turn
        out.append(m)
    if not out:
        return None
    return json.dumps(out, ensure_ascii=False)


# ---------------------------------------------------------------------------
# --compact-history: the harness's slot policy + compaction (2026-09-26)
# ---------------------------------------------------------------------------

# A tool error (the harness returns those inline, never as a slot): our ToolRegistry's
# "error: ..." and the source harnesses' read / edit failures.
_ERROR_RE = re.compile(
    r"^\s*(error\b|the path \S+ does not exist|file not found|no such file|"
    r"invalid `view_range`)",
    re.IGNORECASE,
)


def compressible_tools(web: bool = False) -> set[str]:
    """The tools whose results the harness returns as a compressible SLOT (ToolRegistry specs'
    ``compressible`` flag; + web_search / web_open / web_find with --web). There is no size
    threshold: every successful result of such a tool is a slot, whatever its length."""
    reg = ToolRegistry("/tmp")
    out = {name for name, (spec, _fn) in reg._tools.items() if spec.compressible}
    if web:
        out |= {"web_search", "web_open", "web_find"}
    return out


def harness_turns(
    msgs: list[dict],
    ctx: list[str],
    prompts: list[str] | None,
    compressible: set[str],
    focus_from_reasoning: bool = False,
    inline_max_chars: int = 600,
    st: Counter | None = None,
):
    """(system, opening user messages, replay turns) of a remapped instruction under the
    harness's slot policy: the result of a call to a compressible tool is a slot unless it is
    an error or empty (the harness's own ``ToolResult.compressible``); every other result
    (write / edit acknowledgements, errors) is inline text, head+tail capped at
    ``inline_max_chars`` like the builder's inline results. A slot keeps its prompt from
    ``prompts`` when that is non-empty (synthesised purposes, --keep-prompts), else gets the
    rebuilt one (``call_prompt``, focus from the issuing turn's reasoning). None when the
    instruction has no assistant turn or its sentinels do not match ``ctx``."""
    from bgkit2.agent.context import ToolCall
    from bgkit2.agent.replay import Turn

    st = st if st is not None else Counter()
    system = ""
    opening: list[str] = []
    turns: list[Turn] = []
    pending: list[dict] = []
    reasoning = ""
    k = 0
    started = False
    for i, m in enumerate(msgs):
        role = m.get("role")
        content = str(m.get("content") or "")
        if not started and role != "assistant":
            if role == "system" and i == 0:
                system = content
            elif role in ("user", "system"):
                opening.append(content)
            else:
                return None
            continue
        started = True
        if role == "assistant":
            calls = list(m.get("tool_calls") or [])
            pending = list(calls)
            reasoning = turn_reasoning(m) if focus_from_reasoning else ""
            turns.append(Turn(
                "assistant", content,
                tuple(ToolCall.from_message(c) for c in calls),
                m.get("reasoning_content") or None,
            ))
        elif role == "tool":
            call = pending.pop(0) if pending else None
            is_slot = SPLICE_SENTINEL in content
            if is_slot:
                if k >= len(ctx):
                    return None
                text, prompt = ctx[k], (prompts[k] if prompts is not None else "")
                k += 1
            else:
                text, prompt = content, ""
            fn = (call.get("function", call) if isinstance(call, dict) else {}) if call else {}
            name = str(fn.get("name", ""))
            slot = (name in compressible and bool(text.strip())
                    and not (name != "bash" and _ERROR_RE.match(text)))
            if slot:
                if not prompt:
                    prompt = call_prompt(call, reasoning)
                    st["compact_prompt_rebuilt"] += 1
                else:
                    st["compact_prompt_kept"] += 1
                st["compact_slotted" if not is_slot else "compact_slot"] += 1
                turns.append(Turn("tool", text, slot=True, prompt=prompt, tool=name))
            else:
                st["compact_inlined" if is_slot else "compact_inline"] += 1
                if len(text) > inline_max_chars:
                    h = inline_max_chars // 2
                    text = text[:h] + "\n[...]\n" + text[-h:]
                turns.append(Turn("tool", text or "(no output)", tool=name))
        elif role in ("user", "system"):
            turns.append(Turn("user", content))
        else:
            return None
    if k != len(ctx) or not started:
        return None
    return system, opening, turns


def compact_row(
    r: dict,
    ins: str,
    tgt: str,
    ctx: list[str],
    prompts: list[str] | None,
    specs: list[dict],
    renderer,
    *,
    compressible: set[str],
    budget: int,
    keep_recent: int = 1,
    compaction_mode: str = "cons",
    focus_from_reasoning: bool = False,
    inline_max_chars: int = 600,
    st: Counter | None = None,
) -> dict | None:
    """The row as the harness would have prompted the target turn: the remapped history
    replayed through ``bgkit2.agent.context.Conversation`` (``bgkit2.agent.replay``) with the
    harness's slot policy (``harness_turns``) and ``compact(budget, keep_recent)`` before every
    assistant turn -- older rounds (their calls, results and the visible text; a thinking
    decoder's reasoning leaves the prompt with its round, as ``_Msg.plain`` folds it) become
    the cons-cell compaction spine (``compaction_payload``; prompt ``compaction: <task[:200]>``).
    Slots are priced by the harness (``RatioPolicy`` = ``context.DEFAULT_RATIOS``) and the
    prices ride in ``meta.slot_reps`` (``slot_reps_mode: harness`` reads them). None when the
    instruction does not replay or no slot is left."""
    from bgkit2.agent.context import COMPACTION_TOOL, Conversation, RatioPolicy
    from bgkit2.agent.replay import replay_history, spine_meta

    st = st if st is not None else Counter()
    got = harness_turns(json.loads(ins), ctx, prompts, compressible, focus_from_reasoning,
                        inline_max_chars, st)
    if got is None:
        st["compact_unreplayable"] += 1
        return None
    system, opening, turns = got
    task = next((u for u in opening if u.strip()), "")
    conv = Conversation(system, specs, renderer, RatioPolicy(), " ".join(task.split())[:200],
                        compaction_mode=compaction_mode)
    for u in opening:
        conv.add_user(u)
    replay_history(conv, turns, budget=budget, keep_recent=keep_recent)
    slots = conv.slots
    if not slots:
        st["compact_no_slot"] += 1
        return None
    comp = [x for x in slots if x.tool == COMPACTION_TOOL]
    meta = json.loads(r["meta"]) if isinstance(r["meta"], str) else dict(r["meta"] or {})
    meta.update({
        "compacted": bool(comp),
        "n_compactions": conv.n_compactions,
        "compaction_slot_tokens": comp[0].n_tokens if comp else 0,
        **spine_meta(comp[0].text if comp else "", renderer),
        "prefix_tokens_compressed": conv.prefix_tokens(compressed=True),
        "slot_tokens": [x.n_tokens for x in slots],
        "slot_reps": [x.n_reps for x in slots],
        "slot_tools": [x.tool for x in slots],
        "context_budget": budget,
        "compaction_mode": compaction_mode,
    })
    for key in ("n_slots", "slot_chars", "tools"):
        meta.pop(key, None)
    row = conv.to_task_row(target=json.loads(tgt), family=r["family"], split=r["split"],
                           meta=meta)
    row["meta"].pop("digests", None)
    st["compact_rows_compacted"] += bool(comp)
    return row


def _traj_key(r: dict) -> str:
    meta = json.loads(r["meta"]) if isinstance(r["meta"], str) else (r["meta"] or {})
    return f"{meta.get('source', '')}:{meta.get('trajectory_id', '')}"


def sample_spread(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """At most ``n`` rows spread over trajectories: trajectories in a seeded random order, one
    row of each per round (rows of a trajectory also in seeded random order) until ``n`` are
    taken. The kept rows stay in their input order."""
    rng = random.Random(seed)
    by: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by.setdefault(_traj_key(r), []).append(i)
    keys = sorted(by)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(by[k])
    take: list[int] = []
    rnd = 0
    while len(take) < n:
        added = False
        for k in keys:
            if rnd < len(by[k]):
                take.append(by[k][rnd])
                added = True
                if len(take) == n:
                    break
        if not added:
            break
        rnd += 1
    return [rows[i] for i in sorted(take)]


def iter_rows(path: str):
    """Stream the store (pyarrow batches, never the whole table in memory)."""
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=256):
        yield from b.to_pylist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="/home/werg/bgkit-data-nvme/bgkit2/tasks/tool_slots.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", default="tool_slots")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--max-prefix-tokens",
        type=int,
        default=4096,
        help="drop rows whose rendered prefix exceeds the trainer's cap, so the store's "
        "size is honest instead of the loader silently hopping over two thirds of it",
    )
    ap.add_argument("--tokenizer", default="LiquidAI/LFM2.5-350M")
    ap.add_argument(
        "--keep-prompts",
        action="store_true",
        help="keep the store's existing per-slot prompts instead of rebuilding them from the "
        "calls — for rows whose prompts are SYNTHESISED purposes (tool_slots_synth_*), which "
        "are strictly better than '<tool> <arg>' and must not be overwritten",
    )
    ap.add_argument("--web", action="store_true",
                    help="keep web-research rows: browser_search/open/find -> web_search/open/find")
    ap.add_argument(
        "--keep-reasoning",
        action="store_true",
        help="move each assistant turn's reasoning (fields, <think>/<thinking> blocks, `think` "
        "calls) into `reasoning_content` instead of deleting / folding it into the content",
    )
    ap.add_argument(
        "--focus-from-reasoning",
        action="store_true",
        help="a rebuilt slot prompt for a focus-less read-type call gets ': <focus>' from the "
        "last sentences of the issuing turn's reasoning that say what is being looked for",
    )
    ap.add_argument(
        "--ext-harnesses",
        action="store_true",
        help="also map the SWE-agent (submit) and OpenCode (read/grep/glob/list/edit/write/bash/"
        "todowrite/task) tool sets; `create` -> write(path, content)",
    )
    ap.add_argument(
        "--max-target-tokens",
        type=int,
        default=0,
        help="drop rows whose rendered target exceeds this many tokens (the trainer cuts "
        "targets at task_max_target_tokens: a cut reasoning turn loses its call). 0 = keep",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="sample at most this many rows, spread over trajectories (round-robin: one row "
        "per trajectory per round, trajectories shuffled with --sample-seed). 0 = keep all",
    )
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument(
        "--compact-history",
        action="store_true",
        help="(input: bgkit2_build_tasks.py tool_slots --compact-history) replay each row's "
        "whole history through the harness's Conversation: every result of a compressible tool "
        "is a slot (no size threshold), others inline; compact(--context-budget) before every "
        "assistant turn folds the oldest rounds into the compaction spine; per-slot harness "
        "prices in meta.slot_reps. With --keep-prompts, empty prompts are rebuilt",
    )
    ap.add_argument("--context-budget", type=int, default=6144,
                    help="--compact-history: the harness's per-turn prefix cap (compressed)")
    ap.add_argument("--keep-recent", type=int, default=1,
                    help="--compact-history: Conversation.compact keep_recent")
    ap.add_argument("--compaction-mode", default="cons", choices=("flat", "cons"))
    ap.add_argument("--inline-max-chars", type=int, default=600,
                    help="--compact-history: head+tail cap of an inline (non-slot) tool result")
    args = ap.parse_args()
    global WEB, EXT
    WEB = bool(args.web)
    EXT = bool(args.ext_harnesses)
    if Path(args.out).exists() and not args.overwrite:
        sys.exit(f"{args.out} exists; pass --overwrite")
    specs = ToolRegistry("/tmp").specs()
    if WEB:
        from bgkit2.agent.web import web_tool_specs

        specs = specs + web_tool_specs()
    renderer = None
    fast = None
    if args.max_prefix_tokens or args.compact_history:
        from transformers import AutoTokenizer

        from bgkit2.data.tasks import ChatRenderer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        renderer = ChatRenderer.from_tokenizer(tok) if args.max_prefix_tokens else None
        if args.compact_history:
            from bgkit2.agent.replay import FastRenderer

            fast = FastRenderer(tok)
    compressible = compressible_tools(WEB)
    st: Counter = Counter()
    rows: list[dict] = []
    for r in iter_rows(args.store):
        st["in"] += 1
        ins = remap_messages(r["instruction"], args.keep_reasoning)
        if ins is None:
            st["unmappable_instruction"] += 1
            continue
        tgt = remap_messages(r["target"], args.keep_reasoning)
        if tgt is None:
            st["unmappable_target"] += 1
            continue
        ctx = json.loads(r["context"]) if r["context"] else []
        existing = None
        if args.keep_prompts:
            existing = json.loads(r["prompt"]) if r["prompt"] else []
            if len(existing) != len(ctx) or (
                not args.compact_history and any(not p for p in existing)
            ):
                st["prompt_missing"] += 1
                continue
        if EXT:
            raw1, ctx2, existing2 = strip_dropped_slots(r["instruction"], ctx, existing)
            if raw1 is not r["instruction"]:
                st["noop_slots_dropped"] += len(ctx) - len(ctx2)
                r["instruction"], ctx, existing = raw1, ctx2, existing2
                r["context"] = json.dumps(ctx, ensure_ascii=False)
                if not ctx:
                    st["no_slot_left"] += 1
                    continue
                ins = remap_messages(r["instruction"], args.keep_reasoning)
                if ins is None:
                    st["unmappable_instruction"] += 1
                    continue
        # exact-quote results -> encoded slots, on the ORIGINAL messages (then remapped again)
        raw2, ctx, existing = encode_exact_quote_results(
            r["instruction"], ctx, existing, st, args.focus_from_reasoning
        )
        if raw2 is not r["instruction"]:
            ins = remap_messages(raw2, args.keep_reasoning)
            if ins is None:
                st["unmappable_instruction"] += 1
                continue
            r["context"] = json.dumps(ctx, ensure_ascii=False)
        if ins.count(SPLICE_SENTINEL) != len(ctx):
            st["sentinel_mismatch"] += 1
            continue
        if args.compact_history:
            row = compact_row(
                r, ins, tgt, ctx, existing, specs, fast,
                compressible=compressible, budget=args.context_budget,
                keep_recent=args.keep_recent, compaction_mode=args.compaction_mode,
                focus_from_reasoning=args.focus_from_reasoning,
                inline_max_chars=args.inline_max_chars, st=st,
            )
            if row is None:
                continue
            row["meta"]["remapped_from"] = "openhands"
            r = row
            ins = row["instruction"]
            meta = row["meta"]
            if meta["compacted"]:
                st["compacted"] += 1
        elif args.keep_prompts:
            r["prompt"] = json.dumps(existing, ensure_ascii=False)
        else:
            prompts = slot_prompts(json.loads(ins), args.focus_from_reasoning)
            if len(prompts) != len(ctx) or any(not p for p in prompts):
                st["prompt_rebuild_failed"] += 1
                continue
            r["prompt"] = json.dumps(prompts)
        if not args.compact_history:
            meta = json.loads(r["meta"]) if isinstance(r["meta"], str) else (r["meta"] or {})
            meta["tools"] = specs
            meta["remapped_from"] = "openhands"
        r["instruction"], r["target"], r["meta"], r["family"] = ins, tgt, meta, args.family
        if renderer is not None:
            think = has_reasoning(json.loads(ins) + json.loads(tgt))
            try:
                segs = renderer.render_slots(ins, specs, think)
            except Exception:
                st["unrenderable"] += 1
                continue
            if sum(int(x.shape[0]) for x in segs) > args.max_prefix_tokens:
                st["prefix_too_long"] += 1
                continue
            if args.max_target_tokens:
                try:
                    ids, _mask = renderer.render_chunk(json.loads(ins), json.loads(tgt), None,
                                                       specs, think)
                except Exception:
                    st["unrenderable_target"] += 1
                    continue
                if int(ids.shape[0]) > args.max_target_tokens:
                    st["target_too_long"] += 1
                    continue
        rows.append(r)
        st["kept"] += 1
    if args.max_rows and len(rows) > args.max_rows:
        rows = sample_spread(rows, args.max_rows, args.sample_seed)
        st["sampled"] = len(rows)
    if Path(args.out).exists():
        Path(args.out).unlink()
    n = write_task_store(args.out, rows)
    print(json.dumps({"out": args.out, "rows": n, "stats": st}, indent=1))


if __name__ == "__main__":
    main()
