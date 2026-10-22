"""scripts/bgkit2_rerender_agent_zoom.py: inline zoom results -> encoded zoom slots, current specs
and system prompt, cached slots untouched, chat rows get the tool list."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

from bgkit2.agent.taskset import QUESTION_SYSTEM
from bgkit2.data.templates import SPLICE_SENTINEL

_p = Path(__file__).resolve().parents[4] / "scripts" / "bgkit2_rerender_agent_zoom.py"
_spec = importlib.util.spec_from_file_location("rerender_agent_zoom", _p)
Z = importlib.util.module_from_spec(_spec)
sys.modules["rerender_agent_zoom"] = Z
_spec.loader.exec_module(Z)

OLD_SYS = QUESTION_SYSTEM.replace(Z.NEW_ZOOM_SENTENCE, Z.OLD_ZOOM_SENTENCE)
OV = '{"repo_overview": "o/r@abc", "n_reps": 8}'


def _call(name, **a):
    return {"type": "function", "function": {"name": name, "arguments": a}}


def _row(msgs, ctx, prompts, framing="tool_slots", target=None):
    old_zoom = {"type": "function", "function": {"name": "zoom", "description": "verbatim",
                                                 "parameters": {}}}
    return {"family": "agent_tool_slots", "framing": framing,
            "context": json.dumps(ctx) if ctx else "", "prompt": json.dumps(prompts) if ctx else "",
            "instruction": json.dumps(msgs), "tool_name": "", "tool_args": "{}",
            "target": json.dumps(target or [{"role": "assistant", "content": "ANSWER: x"}]),
            "split": "train",
            "meta": json.dumps({"tools": [old_zoom], "n_slots": len(ctx),
                                "digests": [None] * len(ctx)})}


def test_inline_zoom_becomes_a_slot_after_the_cached_overview():
    msgs = [
        {"role": "system", "content": OLD_SYS},
        {"role": "user", "content": f"q\n{SPLICE_SENTINEL}"},  # the always-on overview
        {"role": "assistant", "content": "", "tool_calls": [_call("zoom", path="a.py", start=3,
                                                                   end=90)]},
        {"role": "tool", "content": "3: x = 1\n4: y = 2"},
        {"role": "assistant", "content": "", "tool_calls": [_call("read", path="b.py")]},
        {"role": "tool", "content": SPLICE_SENTINEL},
    ]
    st = Counter()
    r = Z.rerender_row(_row(msgs, [OV, "b body"], ["contents of .", "read b.py"]),
                       Z.current_specs(), st)
    ctx, pr = json.loads(r["context"]), json.loads(r["prompt"])
    assert ctx == [OV, "3: x = 1\n4: y = 2", "b body"]
    assert pr == ["contents of .", "zoom a.py lines 3-4", "read b.py"]
    meta = json.loads(r["meta"])
    assert meta["n_slots"] == 3 and meta["digests"] == [None, None, None]
    assert "exact" not in json.dumps(meta["tools"]) and "whole file" in json.dumps(meta["tools"])
    m = json.loads(r["instruction"])
    assert m[0]["content"] == QUESTION_SYSTEM and r["instruction"].count(SPLICE_SENTINEL) == 3
    assert st["zoom_slotted"] == 1 and st["system_zoom_sentence_replaced"] == 1


def test_chat_row_gets_tool_list_and_error_zoom_stays_inline():
    msgs = [
        {"role": "system", "content": OLD_SYS},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [_call("zoom", path="a.py", start=900,
                                                                   end=950)]},
        {"role": "tool", "content": "(file has 4 lines; nothing in 900..950)"},
    ]
    st = Counter()
    tgt = [{"role": "assistant", "content": "", "tool_calls": [
        _call("zoom", path="a.py", start=1, end=4, exact=True)]}]
    r = Z.rerender_row(_row(msgs, [], [], framing="chat", target=tgt), Z.current_specs(), st)
    assert r["framing"] == "chat" and r["context"] == ""
    assert "List of tools: [" in json.loads(r["instruction"])[0]["content"]
    assert st["zoom_kept_inline"] == 1 and st["zoom_exact_dropped"] == 1
    assert "exact" not in r["target"]
