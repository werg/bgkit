"""SWE-bench Lite localisation harness: parsing, export, scripted episodes, summary."""

from __future__ import annotations

import json
from pathlib import Path

from bgkit2.agent.context import PlainRenderer, ToolCall, render_tool_calls
from bgkit2.agent.runner import CompressedModel, FullTextModel, SlotRendering
from bgkit2.agent.swebench_lite import Instance
from bgkit2.agent.swebench_loc import (
    answered_files,
    gold_files,
    load_done,
    load_records,
    loc_task,
    norm_path,
    run_instance,
    summarize,
)

GOLD_PATCH = (
    "diff --git a/pkg/calc.py b/pkg/calc.py\n"
    "--- a/pkg/calc.py\n+++ b/pkg/calc.py\n@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n-    return a - b  # BUG\n+    return a + b\n"
)


def _instance(base: str) -> Instance:
    return Instance(
        instance_id="demo__demo-1",
        repo="demo/demo",
        base_commit=base,
        problem_statement="add(1, 2) returns -1 instead of 3.",
        patch=GOLD_PATCH,
        test_patch="diff --git a/tests/test_calc.py b/tests/test_calc.py\n",
        hints_text="SECRET_HINT look at calc",
    )


def _mirror(tmp_path: Path, git_repo) -> Path:
    """repos_dir with the fixture repo under ``demo__demo`` (the mirror layout)."""
    repo, _ = git_repo
    repos_dir = tmp_path / "mirrors"
    repos_dir.mkdir()
    repo.rename(repos_dir / "demo__demo")
    return repos_dir


def test_gold_and_answer_parsing():
    two = GOLD_PATCH + "diff --git a/pkg/other.py b/pkg/other.py\n"
    assert gold_files(two) == ["pkg/calc.py", "pkg/other.py"]
    assert norm_path("`./pkg/calc.py::add`") == "pkg/calc.py"
    assert norm_path("/repo/pkg/calc.py:12") == "pkg/calc.py"
    got, form = answered_files("I think it is calc.\nFILES: `pkg/calc.py`, pkg/a.py::f.")
    assert got == ["pkg/calc.py", "pkg/a.py"] and form == "files"
    assert answered_files("ANSWER: pkg/calc.py::add") == (["pkg/calc.py"], "answer")
    assert answered_files("ANSWER: FILES: pkg/calc.py") == (["pkg/calc.py"], "files")
    assert answered_files("no idea") == ([], "none")


def test_task_never_carries_gold(git_repo):
    t = loc_task(_instance(git_repo[1]), "/x")
    text = t.system_prompt + t.instruction
    assert "add(1, 2)" in t.instruction and "FILES: path1, path2" in t.instruction
    assert "FILES:" in t.system_prompt and "ANSWER:" not in t.system_prompt
    for leak in ("a + b", "diff --git", "SECRET_HINT", "test_calc"):
        assert leak not in text


def _read_then_answer(answer: str) -> list[str]:
    return [
        render_tool_calls([ToolCall("read", {"path": "./pkg/calc.py", "focus": "add"})]),
        f"The bug is in add.\nFILES: {answer}<|im_end|>",
    ]


def test_run_instance_both_arms_resume_and_summary(tmp_path: Path, git_repo):
    repos_dir = _mirror(tmp_path, git_repo)
    inst = _instance(git_repo[1])
    work, out = tmp_path / "work", tmp_path / "out"
    seen: dict[str, list] = {"compressed": [], "full_text": []}
    scripts = {
        "compressed": _read_then_answer("pkg/calc.py"),
        "full_text": _read_then_answer("pkg/calc.py, tests/test_calc.py"),
    }

    def make_model(arm):
        def step(x):
            seen[arm].append(x)
            return scripts[arm][len(seen[arm]) - 1]

        if arm == "compressed":
            return CompressedModel(lambda sr: step(sr), PlainRenderer())
        return FullTextModel(step, PlainRenderer())

    recs = run_instance(
        inst, ["compressed", "full_text"], make_model, repos_dir, work, out, max_turns=5
    )
    assert [r["arm"] for r in recs] == ["compressed", "full_text"]
    c, f = recs
    assert c["gold_files"] == ["pkg/calc.py"] and c["read_files"] == ["pkg/calc.py"]
    assert c["read_gold_any"] and c["answer_hit"] and c["answer_exact"]
    assert f["answer_hit"] and not f["answer_exact"]
    assert c["stop"] == "final" and c["turns"] == 2 and c["decoder_tokens"] > 0
    assert c["tool_result_tokens"] > 0
    assert c["edit_calls"] == 0 and c["edit_old_verbatim"] is None  # EditAudit fields present
    # the checkout is the BASE commit (the bug is there), history-free
    sr = seen["compressed"][1]
    assert isinstance(sr, SlotRendering) and any("a - b" in ctx for ctx in sr.contexts)
    assert "a - b" in seen["full_text"][1]
    for arm_seen in seen.values():
        for x in arm_seen:
            blob = (
                json.dumps(x.messages) + "".join(x.contexts) if isinstance(x, SlotRendering) else x
            )
            assert "a + b" not in blob and "SECRET_HINT" not in blob
    assert not (work / inst.instance_id).exists()  # export cleaned up

    # resume: both arms done -> nothing re-run
    done = {a: load_done(out / a / "episodes.jsonl") for a in ("compressed", "full_text")}
    assert done == {"compressed": {inst.instance_id}, "full_text": {inst.instance_id}}
    assert (
        run_instance(inst, ["compressed", "full_text"], make_model, repos_dir, work, out, done=done)
        == []
    )

    s = summarize(list(load_records(out / "compressed" / "episodes.jsonl").values()))
    assert s["n"] == 1 and s["read_gold_any"] == 1.0 and s["answer_exact"] == 1.0
    assert s["loops"] == 0 and s["stop"] == {"final": 1} and s["decoder_tokens"] > 0
    assert s["edit_calls"] == 0 and s["edit_old_verbatim"] is None
    s_full = summarize(list(load_records(out / "full_text" / "episodes.jsonl").values()))
    assert s_full["answer_hit"] == 1.0 and s_full["answer_exact"] == 0.0


def test_missing_mirror_or_commit_is_skipped_and_retried(tmp_path: Path, git_repo):
    repos_dir = _mirror(tmp_path, git_repo)
    inst = _instance("0" * 40)  # not in the mirror (still cloning)
    out = tmp_path / "out"
    recs = run_instance(inst, ["full_text"], lambda arm: None, repos_dir, tmp_path / "w", out)
    assert recs[0]["stop"] == "skipped" and "lacks" in recs[0]["error"]
    assert load_done(out / "full_text" / "episodes.jsonl") == set()
    s = summarize(list(load_records(out / "full_text" / "episodes.jsonl").values()))
    assert s["n"] == 0 and s["skipped"] == 1 and s["answer_hit"] is None
    other = Instance("x__y-1", "x/y", git_repo[1], "issue")
    assert (
        run_instance(other, ["full_text"], lambda a: None, repos_dir, tmp_path / "w", out)[0][
            "stop"
        ]
        == "skipped"
    )


def test_loop_counts(tmp_path: Path):
    recs = [
        {
            "instance_id": str(i),
            "repo": "r",
            "stop": s,
            "read_gold_any": False,
            "answer_hit": False,
            "answer_exact": False,
            "answered_files": [],
            "answer_form": "none",
            "turns": 3,
            "decoder_tokens": 10,
            "tool_result_tokens": 1,
            "tokens_saved": 0,
            "wall_s": 1.0,
        }
        for i, s in enumerate(["loop", "loop", "max_turns"])
    ] + [{"instance_id": "e", "repo": "r", "stop": "error"}]
    s = summarize(recs)
    assert s["loops"] == 2 and s["n"] == 3 and s["errors"] == 1 and s["answered"] == 0.0


def test_edit_exactness_is_pooled_over_edits():
    from bgkit2.agent.swebench_loc import _edit_summary

    recs = [{"edit_calls": 2, "edits": [{"old_verbatim": True, "old_unique": True,
                                         "old_ws_match": True},
                                        {"old_verbatim": False, "old_unique": False,
                                         "old_ws_match": True}]},
            {"edit_calls": 0, "edits": []}, {}]
    s = _edit_summary(recs)
    assert s == {"edit_calls": 2, "episodes_with_edit": 1, "edit_old_verbatim": 0.5,
                 "edit_old_unique": 0.5, "edit_old_ws_match": 1.0}
