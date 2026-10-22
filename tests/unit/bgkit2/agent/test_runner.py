from __future__ import annotations

import json
from pathlib import Path

from bgkit2.agent.context import PlainRenderer, RatioPolicy, ToolCall, render_tool_calls
from bgkit2.agent.runner import (
    AgentTask,
    CompressedModel,
    FullTextModel,
    SlotRendering,
    TrajectoryLog,
    run_episode,
)
from bgkit2.agent.tools import ToolRegistry
from bgkit2.data.templates import SPLICE_SENTINEL

SCRIPT = [
    "Let me read the file."
    + render_tool_calls([ToolCall("read", {"path": "pkg/calc.py", "focus": "add"})]),
    render_tool_calls(
        [ToolCall("edit", {"path": "pkg/calc.py", "old": "a - b  # BUG", "new": "a + b"})]
    ),
    "Fixed the subtraction bug in add.<|im_end|>",
]


class Scripted:
    def __init__(self):
        self.seen: list = []

    def __call__(self, prompt):
        self.seen.append(prompt)
        return SCRIPT[len(self.seen) - 1]


def _task(repo: Path) -> AgentTask:
    return AgentTask("t1", "Fix the add function in pkg/calc.py so that it adds.", str(repo))


def test_full_text_episode_three_turns_with_accounting(repo: Path, tmp_path: Path):
    stub, renderer = Scripted(), PlainRenderer()
    model = FullTextModel(stub, renderer)
    log_path = tmp_path / "traj.jsonl"
    with TrajectoryLog(log_path) as log:
        res = run_episode(model, ToolRegistry(repo), _task(repo), max_turns=5, log=log)
    assert res.turns == 3 and res.stop_reason == "final" and not res.compressed
    assert res.final_message == "Fixed the subtraction bug in add."
    assert "a + b" in (repo / "pkg/calc.py").read_text()
    # the second prompt carried the full file text (no sentinels in the full-text arm)
    assert "return a - b" in stub.seen[1] and SPLICE_SENTINEL not in stub.seen[1]
    r0, r1, _r2 = res.records
    assert r0.tool_calls == [{"name": "read", "args": {"path": "pkg/calc.py", "focus": "add"}}]
    assert r0.prefix_tokens_consumed == r0.prefix_tokens_full and r0.tokens_saved > 0
    assert (
        r1.prefix_tokens_full > r0.prefix_tokens_full and r1.tokens_saved == 0
    )  # edit is verbatim
    assert res.decoder_tokens == sum(
        r.prefix_tokens_consumed + r.generated_tokens for r in res.records
    )
    assert res.tool_result_tokens == r0.tool_result_tokens + r1.tool_result_tokens
    assert res.tokens_saved == r0.tokens_saved and res.n_compactions == 0
    lines = [json.loads(x) for x in log_path.read_text().splitlines()]
    kinds = [x["kind"] for x in lines]
    assert [k for k in kinds if k != "row"] == ["turn", "turn", "turn", "episode"]
    assert kinds.count("row") == 3  # one tool_slots row per assistant turn
    assert lines[-1]["task_id"] == "t1" and lines[-1]["decoder_tokens"] == res.decoder_tokens
    assert res.task_row["framing"] == "tool_slots" and len(json.loads(res.task_row["context"])) == 1


def test_compressed_episode_sees_slots_and_charges_reps(repo: Path):
    stub, renderer = Scripted(), PlainRenderer()
    seen: list[SlotRendering] = []

    def gen(sr: SlotRendering) -> str:
        seen.append(sr)
        return stub(sr.text)

    ratios = RatioPolicy({"read": 8})
    n = renderer.count((repo / "pkg/calc.py").read_text())  # before the episode edits it
    res = run_episode(
        CompressedModel(gen, renderer), ToolRegistry(repo), _task(repo), max_turns=5, ratios=ratios
    )
    assert res.turns == 3 and res.compressed and res.stop_reason == "final"
    assert (
        seen[0].contexts == [] and seen[1].contexts and seen[1].prompts == ["read pkg/calc.py: add"]
    )
    assert seen[1].text.count(SPLICE_SENTINEL) == 1 and "return a - b" not in seen[1].text
    assert seen[1].messages[-1]["content"] == SPLICE_SENTINEL
    r1 = res.records[1]
    assert r1.prefix_tokens_consumed == r1.prefix_tokens_compressed < r1.prefix_tokens_full
    gap = r1.prefix_tokens_full - r1.prefix_tokens_compressed
    assert abs(gap - res.records[0].tokens_saved) <= 1
    assert res.records[0].tokens_saved == n - (-(-n // 8))


def test_budget_and_parse_error_paths(repo: Path):
    renderer = PlainRenderer()
    res = run_episode(
        FullTextModel(Scripted(), renderer), ToolRegistry(repo), _task(repo), decoder_token_budget=1
    )
    assert res.turns == 0 and res.stop_reason == "budget"

    bad = FullTextModel(lambda _p: "<|tool_call_start|>[read(path=]<|tool_call_end|>", renderer)
    res = run_episode(bad, ToolRegistry(repo), _task(repo), max_turns=2)
    assert res.stop_reason == "max_turns" and all(r.parse_error for r in res.records)
    assert "error: could not parse" in res.task_row["instruction"]


def test_compaction_triggers_inside_the_loop(repo: Path):
    renderer = PlainRenderer()
    # distinct focuses: identical repeats would trip the loop detector
    calls = iter(
        [
            render_tool_calls([ToolCall("read", {"path": "README.md", "focus": f"part {i}"})])
            for i in range(6)
        ]
        + ["done"]
    )
    res = run_episode(
        FullTextModel(lambda _p: next(calls), renderer),
        ToolRegistry(repo),
        _task(repo),
        max_turns=10,
        context_budget=120,
    )
    assert res.stop_reason == "final" and res.n_compactions >= 1
    assert any(r.compacted for r in res.records)
    prompts = json.loads(res.task_row["prompt"])
    assert prompts[0].startswith("compaction: Fix the add function")


def test_identical_repeated_calls_are_not_rerun_and_abort_the_episode(repo: Path, tmp_path: Path):
    renderer = PlainRenderer()
    same = render_tool_calls([ToolCall("read", {"path": "pkg/calc.py"})])
    seen: list[str] = []

    def llm(p):
        seen.append(p)
        return same

    with TrajectoryLog(tmp_path / "t.jsonl") as log:
        res = run_episode(
            FullTextModel(llm, renderer), ToolRegistry(repo), _task(repo), max_turns=8, log=log
        )
    # turn 0 runs the call; turns 1-2 repeat -> notice; turn 3 -> answer demand; turn 4 -> abort
    assert res.stop_reason == "loop" and res.turns == 5
    assert "identical to one you already made (turn 0)" in seen[2] and "identical" not in seen[1]
    assert "Stop calling tools" in seen[4] and "Stop calling tools" not in seen[3]
    assert res.records[1].tool_result_tokens == 0  # nothing executed on the repeat
    lines = [json.loads(x) for x in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert sum(x["kind"] == "row" for x in lines) == 1  # only the first call is a row


def test_empty_replies_are_nudged_then_abort(repo: Path, tmp_path: Path):
    renderer = PlainRenderer()
    seen: list[str] = []

    def llm(p):
        seen.append(p)
        return ""

    with TrajectoryLog(tmp_path / "t.jsonl") as log:
        res = run_episode(
            FullTextModel(llm, renderer), ToolRegistry(repo), _task(repo), max_turns=8, log=log
        )
    assert res.stop_reason == "empty" and res.turns == 2 and res.final_message == ""
    assert "reply was empty" in seen[1]
    lines = [json.loads(x) for x in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert not any(x["kind"] == "row" for x in lines)


def test_interleaved_repeats_count_across_the_episode(repo: Path):
    renderer = PlainRenderer()
    a = render_tool_calls([ToolCall("read", {"path": "pkg/calc.py"})])
    b = render_tool_calls([ToolCall("read", {"path": "pkg/calc.py", "focus": "add"})])
    script = iter([a, b, a, b, a, "done<|im_end|>"])
    res = run_episode(
        FullTextModel(lambda _p: next(script), renderer),
        ToolRegistry(repo),
        _task(repo),
        max_turns=8,
    )
    # a, b run; a (repeat 1), b (repeat 2) get notices; a (repeat 3) -> demand; "done" -> final
    assert res.stop_reason == "final" and res.turns == 6 and res.final_message == "done"
    assert [r.tool_result_tokens > 0 for r in res.records] == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_seed_reads_preloads_the_gold_file_before_the_model_speaks(repo: Path):
    """The oracle arm's contract: the model's FIRST prompt already carries the
    seeded file, so the arms differ on every episode instead of only on the
    ones where the model retrieved something itself (P26 pause 3: the arms
    returned an identical answer on 59 of 60 localisation episodes)."""
    stub, renderer = Scripted(), PlainRenderer()
    res = run_episode(
        FullTextModel(stub, renderer),
        ToolRegistry(repo),
        _task(repo),
        max_turns=5,
        seed_reads=["pkg/calc.py"],
    )
    assert "return a - b" in stub.seen[0]  # seeded before turn 0
    assert res.tool_result_tokens > 0
    # the seed is a real tool round, so the model's own read is now a repeat
    assert res.records[0].prefix_tokens_full > 0


def test_seed_reads_compressed_arm_saves_tokens_on_the_seeded_result(repo: Path):
    renderer = PlainRenderer()
    res = run_episode(
        CompressedModel(lambda sr: SCRIPT[0], renderer),
        ToolRegistry(repo),
        _task(repo),
        max_turns=1,
        ratios=RatioPolicy(),
        seed_reads=["pkg/calc.py"],
    )
    assert res.tokens_saved > 0  # the seeded read went into a slot


def test_seed_reads_missing_path_does_not_crash_the_episode(repo: Path):
    stub, renderer = Scripted(), PlainRenderer()
    res = run_episode(
        FullTextModel(stub, renderer),
        ToolRegistry(repo),
        _task(repo),
        max_turns=5,
        seed_reads=["does/not/exist.py"],
    )
    assert res.turns >= 1  # the error result is seeded, the episode still runs


def test_browse_walks_the_cached_tree_and_rows_carry_node_markers(repo: Path, tmp_path: Path):
    """overview -> browse(pkg) -> read: `browse` exists only with an all-nodes
    cache, its slot names the NODE (nothing to encode), the full-text view is
    the directory listing, and a folded/unknown directory is a model-facing error."""
    import torch
    from safetensors.torch import save_file

    from bgkit2.agent.overview import overview_lookup
    from bgkit2.data.overview import parse_node, parse_overview

    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = ["README.md", "pkg/__init__.py", "pkg/calc.py", "tests/test_calc.py"]
    store = str(tmp_path / "store.parquet")
    pq.write_table(pa.table({"repo": ["own/demo"], "paths": [paths]}), store + ".repos.parquet")
    cache = str(tmp_path / "nodes.safetensors")
    save_file({"own/demo///reps": torch.ones(4, 8), "own/demo//pkg/reps": torch.ones(3, 8),
               "own/demo//tests/reps": torch.ones(2, 8)}, cache, metadata={"max_depth": "6"})
    fn = overview_lookup(cache, store)
    task = AgentTask("t1", "Which file defines `add`?", str(repo), repo="own/demo")
    script = [
        render_tool_calls([ToolCall("browse", {"dir": "pkg/"})]),
        render_tool_calls([ToolCall("browse", {"dir": "nope"})]),
        render_tool_calls([ToolCall("read", {"path": "pkg/calc.py", "focus": "add"})]),
        "ANSWER: pkg/calc.py<|im_end|>",
    ]
    seen: list = []

    def llm(prompt):
        seen.append(prompt)
        return script[len(seen) - 1]

    tools = ToolRegistry(str(repo))
    assert "browse" not in tools.names
    log = TrajectoryLog(tmp_path / "traj.jsonl")
    res = run_episode(FullTextModel(llm, PlainRenderer()), tools, task, overview=fn, log=log)
    log.close()
    assert res.stop_reason == "final" and "browse" in tools.names
    assert "README.md\npkg/\ntests/" in seen[0] and "pkg/calc.py" not in seen[0]  # root only
    assert "__init__.py\ncalc.py" in seen[1]                 # full-text view: the listing
    assert "not a directory of this repository" in seen[2]
    rows = [json.loads(ln) for ln in (tmp_path / "traj.jsonl").open()]
    rows = [r for r in rows if r["kind"] == "row"]
    ctxs = json.loads(rows[-1]["context"])
    assert parse_overview(ctxs[0]) == ("own/demo", 4)
    assert parse_node(ctxs[1]) == ("own/demo", "pkg", 3) and parse_overview(ctxs[1]) is None
    assert parse_node(ctxs[2]) is None                       # the read result stays text
