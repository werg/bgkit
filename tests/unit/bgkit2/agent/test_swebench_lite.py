from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from bgkit2.agent.swebench_lite import (
    DATASET,
    Instance,
    extract_patch,
    load_instances,
    prediction_record,
    prepare_repo,
    repo_source,
    save_instances,
    score,
    task_from_instance,
    write_predictions,
)

RECORD = {
    "instance_id": "demo__pkg-1",
    "repo": "demo/pkg",
    "base_commit": "",
    "problem_statement": "add() subtracts.\n\nIt should add.",
    "patch": "",
    "test_patch": "",
    "FAIL_TO_PASS": '["tests/test_calc.py::test_add"]',
    "PASS_TO_PASS": "[]",
    "version": "0.1",
    "environment_setup_commit": "",
}


def test_load_instances_from_local_jsonl_and_roundtrip(tmp_path: Path):
    p = tmp_path / "lite.jsonl"
    save_instances(
        [Instance.from_record(RECORD), Instance.from_record({**RECORD, "instance_id": "x-2"})], p
    )
    inst = load_instances(p)
    assert [i.instance_id for i in inst] == ["demo__pkg-1", "x-2"]
    assert inst[0].fail_to_pass == ("tests/test_calc.py::test_add",) and inst[0].pass_to_pass == ()
    assert load_instances(p, limit=1)[0].instance_id == "demo__pkg-1"
    assert [i.instance_id for i in load_instances(p, instance_ids={"x-2"})] == ["x-2"]
    assert Instance.from_record(inst[0].to_record()) == inst[0]
    with pytest.raises(RuntimeError, match="local JSONL"):
        load_instances(tmp_path / "missing-dataset-name")


def test_prepare_repo_checks_out_base_commit(git_repo: tuple[Path, str], tmp_path: Path):
    src, base = git_repo
    inst = Instance.from_record({**RECORD, "base_commit": base})
    mirrors = tmp_path / "mirrors"
    (mirrors / "demo").mkdir(parents=True)
    (mirrors / "demo" / "pkg").symlink_to(src)
    assert repo_source(inst, mirrors) == str(mirrors / "demo" / "pkg")
    assert repo_source(inst) == "https://github.com/demo/pkg.git"

    work = prepare_repo(inst, tmp_path / "work", repos_dir=mirrors)
    assert work == tmp_path / "work" / inst.instance_id
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=work, capture_output=True, text=True
    ).stdout
    assert head.strip() == base and "a - b" in (work / "pkg/calc.py").read_text()

    (work / "pkg/calc.py").write_text("def add(a, b):\n    return a + b\n")
    (work / "pkg/new.py").write_text("y = 2\n")
    patch = extract_patch(work)
    assert "-    return a - b  # BUG" in patch and "+    return a + b" in patch
    assert "b/pkg/new.py" in patch and "+y = 2" in patch
    # re-preparing an existing checkout resets it
    again = prepare_repo(inst, tmp_path / "work", repos_dir=mirrors)
    assert again == work and "a - b" in (work / "pkg/calc.py").read_text()
    assert not (work / "pkg/new.py").exists()

    task = task_from_instance(inst, work)
    assert task.task_id == inst.instance_id and "It should add." in task.instruction
    assert task.repo_dir == str(work) and task.summary.startswith("add() subtracts.")


def test_score_builds_the_harness_command_without_running(tmp_path: Path):
    inst = Instance.from_record(RECORD)
    p = write_predictions(
        [prediction_record(inst, "diff --git a b", "bgkit2")], tmp_path / "p.jsonl"
    )
    cmd = score(p, run_id="r1", instance_ids=[inst.instance_id])
    assert cmd[:3] == ["python", "-m", "swebench.harness.run_evaluation"]
    assert DATASET in cmd and str(p) in cmd and cmd[-1] == inst.instance_id
    assert p.read_text().count("\n") == 1
