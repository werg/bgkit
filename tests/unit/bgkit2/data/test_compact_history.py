"""tool_slots rows that match the harness (2026-09-26): scripts/bgkit2_build_tasks.py tool_slots
--slots-max 0 / --compact-history / --slots-from-store and scripts/bgkit2_remap_tool_slots.py
--compact-history (every compressible result a slot, older rounds folded into the cons-cell
compaction spine by the harness's own Conversation.compact at the same budget)."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

from bgkit2.agent.context import (
    COMPACTION_HEADER,
    COMPACTION_TOOL,
    PlainRenderer,
    RatioPolicy,
    ToolCall,
    compaction_leaves,
    compaction_payload,
)
from bgkit2.agent.runner import AgentTask, AssistantTurn, TrajectoryLog, run_episode
from bgkit2.agent.tools import ToolRegistry
from bgkit2.data.tasks import write_task_store
from bgkit2.data.templates import SPLICE_SENTINEL

_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load("build_tasks_compact", "bgkit2_build_tasks.py")
R = _load("remap_tool_slots_compact", "bgkit2_remap_tool_slots.py")


def _fn(name: str, **args) -> dict:
    return {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


BIG = "x = 1\n" * 120  # 720 chars: >= --slots-min-chars 500


def _trajectory(n_rounds: int = 12) -> list[dict]:
    """SWE-agent-shaped: bash rounds whose results alternate big / small, one str_replace edit."""
    msgs = [
        {"role": "system", "content": "S" * 3000},
        {"role": "user", "content": "fix the bug"},
    ]
    for i in range(n_rounds):
        if i == 5:
            call = _fn("str_replace_editor", command="str_replace", path="/w/a.py",
                       old_str="x = 1", new_str="x = 2")
            res = "The file /w/a.py has been edited."
        else:
            call = _fn("bash", command=f"cat f{i}.py")
            res = f"OBSERVATION:\n{BIG}# {i}\n" if i % 2 == 0 else f"ok {i}"
        msgs.append({"role": "assistant", "content": f"step {i} " + "w" * 2000,
                     "reasoning_content": f"think {i}", "tool_calls": [call]})
        msgs.append({"role": "tool", "content": res})
    msgs.append({"role": "assistant", "content": "done"})
    return msgs


def _write_source(tmp_path: Path, trajs: dict[str, list[dict]]) -> Path:
    f = tmp_path / "src.jsonl.gz"
    with gzip.open(f, "wt") as fh:
        for tid, msgs in trajs.items():
            fh.write(json.dumps({"trajectory_id": tid, "messages": msgs}) + "\n")
    return f


def _args(src: Path, **kw) -> argparse.Namespace:
    d = dict(
        slot_sources=[f"sw={src}"], seed=23, slots_max_files=None, slots_min_chars=500,
        slots_read_always=False, slots_window=10, slots_per_traj=3, slots_max=4,
        keep_reasoning=True, max_msg_chars=1500, max_ctx_chars=24000, max_tool_chars=600,
        eval_frac=0.0, slots_per_source=10_000, slots_verify=0, tokenizer="unused",
        compact_history=False, slots_from_store=None,
    )
    d.update(kw)
    return argparse.Namespace(**d)


# ------------------------------------------------------------------ builder
def test_slots_max_zero_is_unlimited(tmp_path):
    src = _write_source(tmp_path, {"t0": _trajectory(20)})
    capped = B.build_tool_slots(_args(src, slots_per_traj=100))
    free = B.build_tool_slots(_args(src, slots_per_traj=100, slots_max=0))
    assert len(capped) == len(free)
    assert max(len(json.loads(r["context"])) for r in capped) == 4
    # a turn late in the trajectory has the window's 5 big results, all of them slots
    last = max(free, key=lambda r: r["meta"]["msg_index"])
    assert len(json.loads(last["context"])) == 5
    assert "[Earlier turns of this session omitted]" in last["instruction"]


def test_default_build_unchanged_by_the_new_options(tmp_path):
    src = _write_source(tmp_path, {"t0": _trajectory(14), "t1": _trajectory(9)})
    a = B.build_tool_slots(_args(src))
    b = B.build_tool_slots(_args(src, compact_history=False, slots_from_store=None))
    assert a == b and a
    assert all("compact_history" not in r["meta"] for r in a)


def test_compact_history_rows_carry_the_whole_uncapped_history(tmp_path):
    traj = _trajectory(16)
    src = _write_source(tmp_path, {"t0": traj})
    rows = B.build_tool_slots(_args(src, compact_history=True, slots_per_traj=100))
    assert rows
    for r in rows:
        msgs = json.loads(r["instruction"])
        t_idx = r["meta"]["msg_index"]
        assert r["meta"]["compact_history"] is True
        assert len(msgs) == t_idx  # everything before the target, nothing omitted
        assert "[Earlier turns of this session omitted]" not in r["instruction"]
        assert len(msgs[0]["content"]) < 1600  # the anchor stays capped (source system prompt)
        # every non-empty tool result is a slot, small ones included; history uncapped
        n_tool = sum(1 for m in traj[:t_idx] if m["role"] == "tool")
        assert len(json.loads(r["context"])) == n_tool
        assert all(m["content"] == SPLICE_SENTINEL for m in msgs if m["role"] == "tool")
        asst = [m for m in msgs if m["role"] == "assistant"]
        assert all(len(m["content"]) > 2000 for m in asst)
        assert all(m["reasoning_content"].startswith("think") for m in asst)


def test_slots_from_store_builds_its_turns_with_its_prompts(tmp_path):
    traj = _trajectory(12)
    src = _write_source(tmp_path, {"t0": traj, "t1": _trajectory(10)})
    first = B.build_tool_slots(_args(src))  # the "old" store: windowed rows
    old = [r for r in first if r["meta"]["trajectory_id"] == "t0"][:2]
    for r in old:
        r["prompt"] = json.dumps([f"read x: purpose {j}" for j in
                                  range(len(json.loads(r["context"])))])
        r["split"] = "eval"
    store = tmp_path / "old.parquet"
    write_task_store(str(store), old)
    rows = B.build_tool_slots(_args(src, compact_history=True, slots_from_store=str(store)))
    assert sorted(r["meta"]["msg_index"] for r in rows) == sorted(
        r["meta"]["msg_index"] for r in old
    )
    assert all(r["meta"]["trajectory_id"] == "t0" and r["split"] == "eval" for r in rows)
    for r, o in zip(sorted(rows, key=lambda r: r["meta"]["msg_index"]),
                    sorted(old, key=lambda r: r["meta"]["msg_index"]), strict=True):
        by_text = dict(zip(json.loads(o["context"]), json.loads(o["prompt"]), strict=True))
        got = dict(zip(json.loads(r["context"]), json.loads(r["prompt"]), strict=True))
        assert all(got[t] == p for t, p in by_text.items())  # the store's prompts kept
        assert all(p == "" for t, p in got.items() if t not in by_text)  # the rest: rebuilt later
        assert r["meta"]["prompts_from_store"] == len(by_text)


# ------------------------------------------------------------------ remap: slot policy
def test_harness_slot_policy_no_size_threshold():
    comp = R.compressible_tools()
    assert {"read", "search", "bash", "zoom", "ls"} <= comp
    assert not {"write", "edit"} & comp
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "reasoning_content": "I look for the parser",
         "tool_calls": [
             {"type": "function", "function": {"name": "bash", "arguments": {"cmd": "ls"}}},
             {"type": "function", "function": {"name": "edit", "arguments":
                                               {"path": "a", "old": "1", "new": "2"}}},
             {"type": "function", "function": {"name": "read", "arguments": {"path": "b"}}},
             {"type": "function", "function": {"name": "read", "arguments": {"path": "c"}}},
         ]},
        {"role": "tool", "content": SPLICE_SENTINEL},
        {"role": "tool", "content": SPLICE_SENTINEL},
        {"role": "tool", "content": SPLICE_SENTINEL},
        {"role": "tool", "content": "short but real file"},
    ]
    ctx = ["a.py", "The file a has been edited. " + "e" * 900, "ERROR: file not found"]
    st: Counter = Counter()
    system, opening, turns = R.harness_turns(msgs, ctx, ["kept prompt", "", ""], comp, True,
                                             600, st)
    assert system == "sys" and opening == ["task"]
    tools = [t for t in turns if t.role == "tool"]
    assert [t.slot for t in tools] == [True, False, False, True]
    assert tools[0].prompt == "kept prompt" and tools[0].content == "a.py"  # tiny bash: a slot
    assert len(tools[1].content) < 620 and "[...]" in tools[1].content  # inline, capped
    assert tools[3].prompt.startswith("read c")  # rebuilt; an inline read result became a slot
    assert st["compact_inlined"] == 2 and st["compact_slotted"] == 1


# ------------------------------------------------------------------ remap: = the harness
class _Scripted:
    compressed = True
    name = "scripted"

    def __init__(self, turns: list[AssistantTurn]) -> None:
        self.renderer = PlainRenderer()
        self.turns = list(turns)

    def generate(self, conv):
        return self.turns.pop(0)


def _turn(calls: list[ToolCall], reasoning: str | None = None, content: str = "") -> AssistantTurn:
    return AssistantTurn(content, calls, "", 5, reasoning=reasoning)


def _episode(repo: Path, budget: int, log_path: Path) -> list[dict]:
    turns = [
        _turn([ToolCall("read", {"path": "a.py"})], "open a"),
        _turn([ToolCall("bash", {"cmd": "cat b.py"})], "and b"),
        _turn([ToolCall("search", {"query": "def "})]),
        _turn([ToolCall("ls", {})], "layout"),
        _turn([ToolCall("edit", {"path": "b.py", "old": "y = 0", "new": "y = 1"})]),
        _turn([ToolCall("read", {"path": "missing.py"})]),
        _turn([ToolCall("zoom", {"path": "a.py", "start": 1, "end": 3})], "closer"),
        _turn([ToolCall("bash", {"cmd": "echo ok"})]),
        _turn([], "all good", content="done"),
    ]
    tools = ToolRegistry(str(repo))
    log = TrajectoryLog(log_path)
    run_episode(_Scripted(turns), tools, AgentTask("t", "fix the bug in b.py", str(repo)),
                max_turns=20, context_budget=budget, log=log, compaction_mode="cons",
                renderer=PlainRenderer())
    log.close()
    return [json.loads(x) for x in log_path.read_text().splitlines()
            if json.loads(x)["kind"] == "row"]


def test_compact_row_equals_the_harness_row(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("".join(f"def f{i}():\n    return {i}\n" for i in range(80)))
    (repo / "b.py").write_text("y = 0\n" * 60)
    # the uncompacted episode = a raw row of our own schema (every compressible result a slot)
    raw_rows = _episode(repo, 10**9, tmp_path / "raw.jsonl")
    harness_rows = _episode(repo, 1400, tmp_path / "small.jsonl")
    specs = ToolRegistry(str(repo)).specs()
    assert len(raw_rows) == len(harness_rows) == 9
    raw, want = raw_rows[-1], harness_rows[-1]
    sp = compaction_payload(json.loads(want["context"])[0])
    assert sp is not None and len(compaction_leaves(sp)) >= 2  # several compaction events
    st: Counter = Counter()
    got = R.compact_row(
        {**raw, "meta": {"source": "x"}}, raw["instruction"], raw["target"],
        json.loads(raw["context"]), json.loads(raw["prompt"]), specs, PlainRenderer(),
        compressible=R.compressible_tools(), budget=1400, st=st,
    )
    for k in ("instruction", "context", "prompt", "target", "framing"):
        assert got[k] == want[k], k
    m = got["meta"]
    assert m["compacted"] and m["compaction_mode"] == "cons" and m["spine_leaves"] >= 2
    assert m["prefix_tokens_compressed"] <= 1400
    assert m["slot_tools"][0] == COMPACTION_TOOL
    ratios = RatioPolicy()
    assert m["slot_reps"][1:] == [ratios.reps(n, t) for n, t in
                                  zip(m["slot_tokens"][1:], m["slot_tools"][1:], strict=True)]
    msgs = json.loads(got["instruction"])
    assert msgs[2] == {"role": "user", "content": f"{COMPACTION_HEADER}\n{SPLICE_SENTINEL}"}
    assert json.loads(got["prompt"])[0] == "compaction: fix the bug in b.py"
    assert m["tools"] == specs


def test_compact_row_without_pressure_keeps_every_turn(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n" * 30)
    (repo / "b.py").write_text("y = 0\n")
    raw = _episode(repo, 10**9, tmp_path / "raw.jsonl")[-1]
    got = R.compact_row(
        {**raw, "meta": {}}, raw["instruction"], raw["target"], json.loads(raw["context"]),
        None, ToolRegistry(str(repo)).specs(), PlainRenderer(),
        compressible=R.compressible_tools(), budget=10**9,
    )
    assert not got["meta"]["compacted"] and got["meta"]["n_compactions"] == 0
    assert got["instruction"] == raw["instruction"] and got["context"] == raw["context"]
    # prompts rebuilt the remap's way where none were given (read a.py, bash <cmd>, ...)
    assert json.loads(got["prompt"])[0] == "read a.py"
