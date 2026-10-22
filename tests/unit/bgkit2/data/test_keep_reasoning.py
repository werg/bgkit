"""--keep-reasoning / --focus-from-reasoning in the agent-trajectory builders
(scripts/bgkit2_build_tasks.py, scripts/bgkit2_remap_tool_slots.py) and bgkit2.data.reasoning."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from bgkit2.data.reasoning import focus_from_reasoning, move_reasoning, split_reasoning
from bgkit2.data.templates import SPLICE_SENTINEL

_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load("build_tasks_kr", "bgkit2_build_tasks.py")
R = _load("remap_tool_slots_kr", "bgkit2_remap_tool_slots.py")


def _call(name: str, **args) -> dict:
    return {"type": "function", "function": {"name": name, "arguments": args}}


ASSISTANT = {
    "role": "assistant",
    "reasoning_content": "Field reasoning.",
    "content": "<think>Tag one.</think>Visible text. <thinking>Tag two.</thinking>",
    "tool_calls": [{"function": {"name": "execute_bash", "arguments": '{"command": "ls"}'}}],
}

ROW = [
    {"role": "system", "content": "s"},
    {"role": "user", "content": "fix the bug"},
    {
        "role": "assistant",
        "content": "<think>I should look.</think>Checking.",
        "tool_calls": [_call("think", thought="The parser is suspect.")],
    },
    {"role": "tool", "content": "Your thought has been logged."},
    {
        "role": "assistant",
        "content": "Reading.",
        "tool_calls": [_call("str_replace_editor", command="view", path="/w/a.py")],
    },
    {"role": "tool", "content": SPLICE_SENTINEL},
    {"role": "assistant", "content": "", "tool_calls": [_call("finish", message="Done.")]},
    {"role": "tool", "content": "ok"},
]


# ---------------------------------------------------------------- default off: unchanged
def test_build_normalise_default_off_is_unchanged():
    out = B._normalise_for_template(dict(ASSISTANT))
    assert out == {
        "role": "assistant",
        # only <think> is stripped by the legacy path; nothing else moves
        "content": "Visible text. <thinking>Tag two.</thinking>",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "execute_bash", "arguments": {"command": "ls"}},
            }
        ],
    }
    assert B._normalise_for_template(dict(ASSISTANT), False) == out


def test_remap_default_off_is_unchanged():
    raw = json.dumps(ROW)
    out = json.loads(R.remap_messages(raw))
    assert R.remap_messages(raw, keep_reasoning=False) == R.remap_messages(raw)
    assert out[2] == {
        "role": "assistant",
        "content": "<think>I should look.</think>Checking.\nThe parser is suspect.",
    }
    assert out[3]["tool_calls"][0]["function"] == {"name": "read", "arguments": {"path": "/w/a.py"}}
    assert out[5] == {"role": "assistant", "content": "Done."}
    assert all("reasoning_content" not in m for m in out)
    assert R.slot_prompts(out) == R.slot_prompts(out, False) == ["read /w/a.py"]


# ---------------------------------------------------------------- keep-reasoning forms
def test_split_reasoning_every_form_in_order():
    r, c = split_reasoning(ASSISTANT)
    assert r == "Field reasoning.\n\nTag one.\n\nTag two."
    assert c == "Visible text."
    for field in ("reasoning", "thinking"):
        r, c = split_reasoning({"role": "assistant", field: " x ", "content": " y "})
        assert (r, c) == ("x", "y")
    # a source that stripped the opener: "...</think>visible"
    assert split_reasoning({"content": "hidden</think>\nshown"}) == ("hidden", "shown")
    assert split_reasoning({"content": "plain"}) == ("", "plain")


def test_move_reasoning_leaves_other_roles_and_drops_empty():
    u = {"role": "user", "content": "<think>x</think>q"}
    assert move_reasoning(u) is u
    m = move_reasoning({"role": "assistant", "content": " a ", "reasoning": ""})
    assert m == {"role": "assistant", "content": "a"}


def test_build_normalise_keep_reasoning():
    out = B._normalise_for_template(dict(ASSISTANT), keep_reasoning=True)
    assert out["content"] == "Visible text."
    assert out["reasoning_content"] == "Field reasoning.\n\nTag one.\n\nTag two."
    assert out["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
    capped = B._cap_msg(dict(out, reasoning_content="r" * 50), 10)
    assert capped["reasoning_content"] == "rrrrr\n[...]\nrrrrr"


def test_remap_keep_reasoning_merges_think_call():
    out = json.loads(R.remap_messages(json.dumps(ROW), keep_reasoning=True))
    assert out[2] == {
        "role": "assistant",
        "content": "Checking.",
        "reasoning_content": "I should look.\n\nThe parser is suspect.",
    }
    assert out[5] == {"role": "assistant", "content": "Done."}  # finish still folds to text


def test_remap_keep_reasoning_carries_think_only_turn():
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [_call("think", thought="Plan A.")]},
        {"role": "tool", "content": "logged"},
        {
            "role": "assistant",
            "content": "Now.",
            "reasoning_content": "Then B.",
            "tool_calls": [_call("execute_bash", command="ls")],
        },
    ]
    out = json.loads(R.remap_messages(json.dumps(msgs), keep_reasoning=True))
    assert [m["role"] for m in out] == ["user", "assistant"]
    assert out[1]["reasoning_content"] == "Plan A.\n\nThen B."
    # off: the think-only turn becomes a text turn
    off = json.loads(R.remap_messages(json.dumps(msgs)))
    assert off[1] == {"role": "assistant", "content": "Plan A."}


# ---------------------------------------------------------------- focus
def test_focus_from_reasoning_picks_last_looking_sentences():
    r = (
        "The test fails on empty input. I need to find where the tokenizer handles empty "
        "strings. It is probably in utils. Let me check the split function."
    )
    assert focus_from_reasoning(r) == (
        "I need to find where the tokenizer handles empty strings. Let me check the split function."
    )
    assert focus_from_reasoning("") == ""
    assert focus_from_reasoning("Plain statement.") == "Plain statement."
    long = "Let me find " + "very " * 80 + "long thing."
    f = focus_from_reasoning(long)
    assert len(f) <= 200 and f.startswith("Let me find very") and not f.endswith(" ")


def test_slot_prompts_focus_from_reasoning():
    msgs = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I want to see how parse works.",
            "tool_calls": [_call("read", path="a.py")],
        },
        {"role": "tool", "content": SPLICE_SENTINEL},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Check the loop.",
            "tool_calls": [_call("read", path="b.py", focus="the loop only")],
        },
        {"role": "tool", "content": SPLICE_SENTINEL},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Let me look at output.",
            "tool_calls": [_call("bash", cmd="make")],
        },
        {"role": "tool", "content": SPLICE_SENTINEL},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("zoom", path="c.py", start=1, end=9)],
        },
        {"role": "tool", "content": SPLICE_SENTINEL},
    ]
    assert R.slot_prompts(msgs, focus_from_reasoning=True) == [
        "read a.py: I want to see how parse works.",
        "read b.py",  # the call has its own focus
        "bash make",  # not a read-type tool
        "zoom c.py lines 1-9",  # no reasoning -> default
    ]
    assert R.slot_prompts(msgs) == ["read a.py", "read b.py", "bash make", "zoom c.py lines 1-9"]


def test_encode_exact_quote_focus_from_reasoning():
    msgs = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "<think>Find where x is set.</think>ok",
            "tool_calls": [
                _call("str_replace_editor", command="view", path="/w/a.py", view_range=[2, 5])
            ],
        },
        {"role": "tool", "content": "Here's the result of running `cat -n` on /w/a.py:\n 2\tx"},
    ]
    raw = json.dumps(msgs)
    _, _, off = R.encode_exact_quote_results(raw, [], [], R.Counter())
    _, _, on = R.encode_exact_quote_results(raw, [], [], R.Counter(), focus_from_reasoning=True)
    assert off == ["zoom /w/a.py lines 2-5"]
    assert on == ["zoom /w/a.py lines 2-5: Find where x is set."]
