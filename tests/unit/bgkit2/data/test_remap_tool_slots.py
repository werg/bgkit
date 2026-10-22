"""scripts/bgkit2_remap_tool_slots.py: every view_range read -> an ENCODED zoom slot (also right
before an edit: no `exact` in training rows), browser_find -> an encoded web_find slot (--web)."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

from bgkit2.data.templates import SPLICE_SENTINEL

_p = Path(__file__).resolve().parents[4] / "scripts" / "bgkit2_remap_tool_slots.py"
_spec = importlib.util.spec_from_file_location("remap_tool_slots", _p)
R = importlib.util.module_from_spec(_spec)
sys.modules["remap_tool_slots"] = R
_spec.loader.exec_module(R)

CAT = "Here's the result of running `cat -n` on /w/a.py:\n"


def _call(name: str, **args) -> dict:
    return {"type": "function", "function": {"name": name, "arguments": args}}


def _view(path: str, lo: int, hi: int) -> dict:
    return {"role": "assistant", "content": "look",
            "tool_calls": [_call("str_replace_editor", command="view", path=path,
                                 view_range=[lo, hi])]}


def test_every_view_range_becomes_an_encoded_zoom_slot():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "fix it"},
        _view("/w/a.py", 5, 9),
        {"role": "tool", "content": CAT + "     5\tx = 1\n     6\ty = 2"},  # inline -> slot
        _view("/w/a.py", 1, 3),
        {"role": "tool", "content": SPLICE_SENTINEL},  # already a slot: purpose kept
        _view("/w/a.py", 80, 100),
        {"role": "tool", "content": "ERROR:\nInvalid `view_range` parameter: [80, 100]."},
        {"role": "assistant", "content": "think",
         "tool_calls": [_call("think", thought="the bug is on line 2")]},
        {"role": "tool", "content": "Your thought has been logged."},
    ]
    ctx = [CAT + "     1\tdef f():\n     2\t    return a - b"]
    prompts = ["read /w/a.py: find the bug"]
    st: Counter = Counter()
    raw, ctx2, pr2 = R.encode_exact_quote_results(json.dumps(msgs), ctx, prompts, st)
    assert pr2 == ["zoom /w/a.py lines 5-9", "zoom /w/a.py lines 1-3: find the bug"]
    assert ctx2 == [CAT + "     5\tx = 1\n     6\ty = 2", ctx[0]]
    assert st["zoom_slotted"] == 1 and st["zoom_slot_reprompted"] == 1
    assert st["zoom_kept_inline"] == 1  # the error result stays inline
    ins = R.remap_messages(raw)
    assert ins.count(SPLICE_SENTINEL) == len(ctx2) == 2
    out = json.loads(ins)
    zooms = [c["function"] for m in out for c in m.get("tool_calls") or []
             if c["function"]["name"] == "zoom"]
    assert [z["arguments"] for z in zooms] == [{"path": "/w/a.py", "start": 5, "end": 9},
                                               {"path": "/w/a.py", "start": 1, "end": 3},
                                               {"path": "/w/a.py", "start": 80, "end": 100}]
    assert "Invalid `view_range`" in json.dumps(out)  # inline error kept
    # the edit that follows in the TARGET does not turn any view into an exact zoom
    tgt = R.remap_messages(json.dumps([{"role": "assistant", "content": "fix", "tool_calls": [
        _call("str_replace_editor", command="str_replace", path="/w/a.py", old_str="a - b",
              new_str="a + b")]}]))
    assert json.loads(tgt)[0]["tool_calls"][0]["function"]["name"] == "edit"
    assert "exact" not in ins and "exact" not in tgt


def test_prompts_rebuilt_from_calls_name_the_zoom_range():
    msgs = [_view("/w/a.py", 5, 9), {"role": "tool", "content": SPLICE_SENTINEL}]
    ins = R.remap_messages(json.dumps(msgs))
    assert R.slot_prompts(json.loads(ins)) == ["zoom /w/a.py lines 5-9"]


def test_browser_find_becomes_an_encoded_web_find_slot_with_web():
    find = ("[3] Find results for text: `1889` in `Eiffel Tower` (u/find?pattern=1889)\n"
            "**viewing lines [0 - 1] of 1**\n\nL0: # 【0†match at L40】\nL1: It opened in 1889.")
    none = ("[4] Find results for text: `zzz` in `Eiffel Tower` (u/find?pattern=zzz)\n"
            "L0: No `find` results for pattern: `zzz`")
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [
            _call("browser_find", cursor=1, pattern="1889")]},
        {"role": "tool", "content": find},
        {"role": "assistant", "content": "", "tool_calls": [
            _call("browser_find", cursor=1, pattern="zzz")]},
        {"role": "tool", "content": none},
        {"role": "assistant", "content": "", "tool_calls": [
            _call("browser_search", query="eiffel opened")]},
        {"role": "tool", "content": SPLICE_SENTINEL},
    ]
    st: Counter = Counter()
    old = R.WEB
    try:
        R.WEB = True
        raw, ctx, pr = R.encode_exact_quote_results(
            json.dumps(msgs), ["[1] hits"], ['bash browser_search {"query": "e"}: when it opened'],
            st)
        ins = R.remap_messages(raw)
    finally:
        R.WEB = old
    assert pr == ["find 1889 in Eiffel Tower", "web search eiffel opened: when it opened"]
    assert ctx == [find, "[1] hits"] and ins.count(SPLICE_SENTINEL) == 2
    assert "'zzz' not found on the page" in ins and st["find_not_found_inline"] == 1
