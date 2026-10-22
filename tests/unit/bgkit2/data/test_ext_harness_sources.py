"""The JSONL source adapter of scripts/bgkit2_build_tasks.py (filtered Open-SWE-Traces /
Nemotron-SFT-SWE dumps: SWE-agent, OpenHands, mini-SWE-agent, OpenCode harnesses),
--slots-read-always, and scripts/bgkit2_remap_tool_slots.py --ext-harnesses / --max-rows."""

from __future__ import annotations

import gzip
import importlib.util
import json
import random
import sys
from pathlib import Path

from bgkit2.data.templates import SPLICE_SENTINEL

_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load("build_tasks_ext", "bgkit2_build_tasks.py")
R = _load("remap_tool_slots_ext", "bgkit2_remap_tool_slots.py")


def _fn(name: str, **args) -> dict:
    return {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


# ------------------------------------------------------------------ tool-result adapter
def test_sweagent_observation_prefix_removed():
    cat = "Here's the result of running `cat -n` on /t/a.ts:\n     1\tx"
    assert B._adapt_ext_tool_result("OBSERVATION:\n" + cat) == cat
    assert B._adapt_ext_tool_result("Observation: done") == "done"
    assert B._adapt_ext_tool_result("Your command ran successfully.") == (
        "Your command ran successfully."
    )


def test_minisweagent_json_result_unwrapped_like_our_bash():
    ok = json.dumps({"returncode": 0, "output": "a\nb\n"})
    assert B._adapt_ext_tool_result(ok) == "a\nb\n"
    bad = json.dumps({"returncode": 2, "output": "boom"})
    assert B._adapt_ext_tool_result(bad) == "boom\n(exit code 2)"
    empty = json.dumps({"returncode": 0, "output": ""})
    assert B._adapt_ext_tool_result(empty) == "(no output)"
    long = json.dumps({"returncode": 0, "output_head": "H", "output_tail": "T",
                       "elided_chars": 99, "warning": "long"})
    assert B._adapt_ext_tool_result(long) == "H\n[... 99 characters elided ...]\nT"
    exc = json.dumps({"returncode": 1, "output": "", "exception_info": "Timeout"})
    assert B._adapt_ext_tool_result(exc) == "Timeout\n(exit code 1)"
    # JSON that is not a mini-SWE-agent result, and plain text, pass through
    other = json.dumps({"returncode_like": 1})
    assert B._adapt_ext_tool_result(other) == other
    assert B._adapt_ext_tool_result('{"returncode": 0, "x": 1}') == '{"returncode": 0, "x": 1}'


def test_is_read_result():
    view = {"command": "view", "path": "/t/a.ts"}
    assert B._is_read_result("str_replace_editor", view, "Here's the result")
    assert B._is_read_result("str_replace_editor", json.dumps(view), "x")
    assert not B._is_read_result("str_replace_editor", {"command": "str_replace"}, "The file")
    assert not B._is_read_result("str_replace_editor", view, "ERROR:\nInvalid `view_range`")
    assert not B._is_read_result("str_replace_editor", view, "The path /t/x does not exist.")
    assert B._is_read_result("read", {"filePath": "/w/a.go"}, "<path>/w/a.go</path>")
    assert B._is_read_result("grep", {"pattern": "x"}, "Found 1 matches")
    assert B._is_read_result("glob", {"pattern": "*.ts"}, "No files found")
    assert not B._is_read_result("bash", {"command": "cat a"}, "text")
    assert not B._is_read_result("edit", {}, "Edit applied successfully.")
    assert not B._is_read_result("read", {"filePath": "/x"}, "   ")


# ------------------------------------------------------------------ JSONL iterator
def _record(i: int, instance_id: object = "default") -> dict:
    return {
        "trajectory_id": f"t{i}",
        "instance_id": f"org__repo-{i}" if instance_id == "default" else instance_id,
        "dominant_ext": ".ts",
        "dominant_ext_family": "ts",
        "source": "nvidia/Open-SWE-Traces:v1.0:data/sweagent/minimax_m25/swe-rebench-v2",
        "messages": [
            {"role": "system", "content": "s", "think": "True"},
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": "", "reasoning_content": "look",
             "tool_calls": [_fn("bash", command="ls")]},
            {"role": "tool", "content": "OBSERVATION:\na.ts"},
        ],
    }


def test_jsonl_iterator_adapts_results_and_keys_split_by_instance(tmp_path):
    f = tmp_path / "x.jsonl.gz"
    with gzip.open(f, "wt") as fh:
        for i in range(3):
            fh.write(json.dumps(_record(i, instance_id="same__task-1" if i < 2 else None)) + "\n")
    got = list(B._iter_source_trajectories("sw", str(f), random.Random(0), None))
    assert [g[0] for g in got] == ["t0", "t1", "t2"]
    _tid, name, traj, extra = got[0]
    assert name == "sw" and traj[3]["content"] == "a.ts"
    assert traj[2]["reasoning_content"] == "look"
    # every harness's trajectory of one task shares the split key
    assert got[0][3]["split_key"] == got[1][3]["split_key"] == "same__task-1"
    assert got[2][3]["split_key"] == "sw:t2"  # no instance id: by trajectory
    assert extra["meta"]["dominant_ext"] == ".ts"
    assert extra["meta"]["dataset"].startswith("nvidia/Open-SWE-Traces")
    # a directory of JSONL files goes through the same adapter
    assert len(list(B._iter_source_trajectories("sw", str(tmp_path), random.Random(0), None))) == 3


def test_jsonl_iterator_without_instance_id_keys_by_trajectory(tmp_path):
    f = tmp_path / "n.jsonl"
    r = _record(0)
    del r["instance_id"], r["trajectory_id"]
    r["uuid"] = "u-1"
    f.write_text(json.dumps(r) + "\n")
    (tid, _n, _t, extra), = B._iter_source_trajectories("nemo", str(f), random.Random(0), None)
    assert tid == "u-1" and extra["split_key"] == "nemo:u-1"


# ------------------------------------------------------------------ remap --ext-harnesses
def _ext(on: bool) -> None:
    R.EXT = on


def test_remap_default_unchanged_for_ext_names():
    _ext(False)
    try:
        assert R.remap("submit", {}) is None
        assert "submit" not in R.drop_set()
        assert R.remap("grep", {"pattern": "x"}) is None
        assert R.remap("read", {"filePath": "/a"}) is None
        assert R.remap("str_replace_editor", {"command": "create", "path": "/a",
                                              "file_text": "t"}) == ("write", {"path": "/a",
                                                                               "text": "t"})
        assert R.remap("bash", {"command": "ls", "workdir": "/w"}) == ("bash", {"cmd": "ls"})
    finally:
        _ext(False)


def test_remap_opencode_and_sweagent_calls():
    _ext(True)
    try:
        assert R.remap("read", {"filePath": "/w/a.go"}) == ("read", {"path": "/w/a.go"})
        assert R.remap("read", {"filePath": "/w/a.go", "offset": 60, "limit": 40}) == (
            "zoom", {"path": "/w/a.go", "start": 60, "end": 99})
        assert R.remap("read", {"filePath": "/w/a.go", "limit": 10}) == (
            "zoom", {"path": "/w/a.go", "start": 1, "end": 10})
        assert R.remap("grep", {"pattern": "Foo", "include": "*.ts"}) == (
            "search", {"query": "Foo"})
        assert R.remap("glob", {"pattern": "**/*.json", "path": "/w/c"}) == (
            "bash", {"cmd": 'rg --files /w/c -g "**/*.json"'})
        assert R.remap("glob", {"pattern": "a$b"}) == ("bash", {"cmd": "rg --files . -g 'a$b'"})
        assert R.remap("list", {"path": "/w"}) == ("ls", {"dir": "/w"})
        assert R.remap("edit", {"filePath": "/w/a", "oldString": "x", "newString": "y"}) == (
            "edit", {"path": "/w/a", "old": "x", "new": "y"})
        assert R.remap("write", {"filePath": "/w/a", "content": "c"}) == (
            "write", {"path": "/w/a", "content": "c"})
        assert R.remap("str_replace_editor", {"command": "create", "path": "/a",
                                              "file_text": "t"}) == ("write", {"path": "/a",
                                                                               "content": "t"})
        assert R.remap("bash", {"command": "go test", "workdir": "/w/x y"}) == (
            "bash", {"cmd": 'cd "/w/x y" && go test'})
        assert R.remap("bash", {"command": "ls", "description": "d"}) == ("bash", {"cmd": "ls"})
        for n in ("submit", "todowrite", "todoread", "task"):
            assert R.remap(n, {}) is None and n in R.drop_set()
        assert R.remap("webfetch", {"url": "u"}) is None and "webfetch" not in R.drop_set()
    finally:
        _ext(False)


def test_remap_messages_drops_todowrite_with_its_result_and_keeps_reasoning():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "fix"},
        {"role": "assistant", "content": "", "reasoning_content": "plan first",
         "tool_calls": [_fn("todowrite", todos=[{"content": "a"}])]},
        {"role": "tool", "content": "[{...}]"},
        {"role": "assistant", "content": "", "reasoning_content": "read the file",
         "tool_calls": [_fn("read", filePath="/w/a.ts"), _fn("task", prompt="p")]},
        {"role": "tool", "content": SPLICE_SENTINEL},
        {"role": "tool", "content": "task_id: x\n<task_result>\n\n</task_result>"},
    ]
    _ext(True)
    try:
        out = json.loads(R.remap_messages(json.dumps(msgs), keep_reasoning=True))
    finally:
        _ext(False)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
    assert out[2]["reasoning_content"] == "plan first\n\nread the file"
    assert out[2]["tool_calls"] == [
        {"type": "function", "function": {"name": "read", "arguments": {"path": "/w/a.ts"}}}
    ]
    assert R.slot_prompts(out, True) == ["read /w/a.ts: read the file"]
    assert R.remap_messages(json.dumps(msgs)) is None  # default: OpenCode names unmappable


def test_slot_prompt_for_ls_names_the_dir():
    msgs = [
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function", "function": {"name": "ls",
                                                          "arguments": {"dir": "/w/src"}}}]},
        {"role": "tool", "content": SPLICE_SENTINEL},
    ]
    assert R.slot_prompts(msgs) == ["ls /w/src"]


# ------------------------------------------------------------------ --max-rows sampling
def test_sample_spread_takes_one_row_per_trajectory_per_round():
    rows = [{"meta": json.dumps({"source": "s", "trajectory_id": f"t{i // 3}"}), "i": i}
            for i in range(30)]  # 10 trajectories x 3 rows
    got = R.sample_spread(rows, 12, seed=1)
    assert len(got) == 12
    per = {}
    for r in got:
        per[json.loads(r["meta"])["trajectory_id"]] = per.get(
            json.loads(r["meta"])["trajectory_id"], 0) + 1
    assert len(per) == 10 and sorted(per.values()) == [1] * 8 + [2] * 2
    assert [r["i"] for r in got] == sorted(r["i"] for r in got)  # input order kept
    assert R.sample_spread(rows, 12, seed=1) == got  # deterministic
    assert len(R.sample_spread(rows, 100)) == 30
