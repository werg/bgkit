"""Per-instance repo trees for SWE-style tasks: keys, file sets, the overview provider, the
localisation arms with the overview + browse, and the LFM generator's dry run."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from bgkit2.agent.context import PlainRenderer, ToolCall, render_tool_calls
from bgkit2.agent.overview import OVERVIEW_SYSTEM_NOTE, instance_overview_lookup
from bgkit2.agent.runner import CompressedModel, FullTextModel, SlotRendering
from bgkit2.agent.swe_trees import (
    commit_python_files,
    commit_python_sources,
    jobs_for,
    read_manifest,
    tier_of,
    tree_key,
    write_manifest,
)
from bgkit2.agent.swebench_lite import Instance
from bgkit2.agent.swebench_loc import load_records, run_instance, summarize
from bgkit2.data.overview import OVERVIEW_HEADER, parse_node, parse_overview
from bgkit2.models.multilevel import node_children

ROOT = Path(__file__).resolve().parents[4]
GOLD_PATCH = (
    "diff --git a/pkg/calc.py b/pkg/calc.py\n--- a/pkg/calc.py\n+++ b/pkg/calc.py\n"
    "@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b  # BUG\n+    return a + b\n"
)
PATHS = ["pkg/__init__.py", "pkg/calc.py", "tests/test_calc.py"]


def _inst(base: str, iid: str = "demo__demo-1") -> Instance:
    return Instance(iid, "demo/demo", base, "add(1, 2) returns -1 instead of 3.", patch=GOLD_PATCH)


def _mirror(tmp_path: Path, git_repo) -> Path:
    repos = tmp_path / "mirrors"
    repos.mkdir()
    git_repo[0].rename(repos / "demo__demo")
    return repos


def _fake_cache(d: Path, key: str, instances: list[str], max_depth: int = 6) -> Path:
    """An all-nodes shard dir as scripts/bgkit2_cache_instance_trees.py writes it."""
    d.mkdir(parents=True)
    nodes = sorted(node_children(PATHS, max_depth))
    save_file({f"{key}//{n}/reps": torch.full((3 + i, 4), float(i)) for i, n in enumerate(nodes)},
              str(d / "part-0000.safetensors"))
    (d / "index.jsonl").write_text(json.dumps(
        {"repo": key, "nodes": nodes, "variants": [], "shard": "part-0000.safetensors"}) + "\n")
    (d / "meta.json").write_text(json.dumps({"max_depth": max_depth, "ratio_leaf": 0.125}))
    (d / "trees.jsonl").write_text(json.dumps(
        {"key": key, "repo": "demo/demo", "commit": key.split("@")[1], "instances": instances,
         "n_files": len(PATHS), "tokens": 0, "paths": PATHS}) + "\n"
        # a tree whose shard never landed (interrupted build) is ignored
        + json.dumps({"key": "x/y@000000000000", "repo": "x/y", "commit": "0" * 40,
                      "instances": ["x__y-1"], "paths": ["a.py", "b.py"]}) + "\n")
    return d


def test_commit_file_set_keys_and_manifest(tmp_path, git_repo):
    repo, base = git_repo
    assert [p for p, _ in commit_python_files(repo, base)] == PATHS
    src = {p: t for p, _o, t in commit_python_sources(repo, base)}
    assert "a - b" in src["pkg/calc.py"]  # the BASE commit, not the fix
    assert commit_python_files(repo, base, max_files=2) is None
    assert tree_key("o/n", base) == f"o/n@{base[:12]}"
    assert [tier_of(n) for n in (2, 64, 65, 400, 401, 2000, 2001)] == [
        "live", "live", "normal", "normal", "large", "large", "unsupported"]
    repos = _mirror(tmp_path, git_repo)
    jobs = jobs_for([_inst(base), _inst(base, "demo__demo-2"),
                     Instance("x__y-1", "x/y", base, "no mirror")], repos)
    assert len(jobs) == 1 and jobs[0].instances == ["demo__demo-1", "demo__demo-2"]
    write_manifest(tmp_path / "m.jsonl", jobs)
    assert read_manifest(tmp_path / "m.jsonl")[0].key == tree_key("demo/demo", base)


def test_instance_overview_lookup(tmp_path, git_repo):
    base = git_repo[1]
    key = tree_key("demo/demo", base)
    fn = instance_overview_lookup(str(_fake_cache(tmp_path / "c", key, ["demo__demo-1"])))
    from bgkit2.agent.swebench_loc import loc_task

    got = fn(loc_task(_inst(base), "/x"))
    assert got is not None and got[0] == key and got[1].splitlines() == PATHS
    assert got[2] == fn.n_reps[key] and fn.by_instance == {"demo__demo-1": key}
    view = fn.tree(loc_task(_inst(base), "/x"))
    assert view.listing("") == "pkg/\ntests/" and view.repo == key
    assert fn(loc_task(_inst(base, "other-1"), "/x")) is None  # no tree: no overview
    assert fn(loc_task(Instance("x__y-1", "x/y", base, "q"), "/x")) is None  # never cached


def _browse_read_answer() -> list[str]:
    return [
        render_tool_calls([ToolCall("browse", {"dir": "pkg"})]),
        render_tool_calls([ToolCall("read", {"path": "pkg/calc.py", "focus": "add"})]),
        "The bug is in add.\nFILES: pkg/calc.py<|im_end|>",
    ]


def test_ov_arms_open_with_overview_and_browse(tmp_path, git_repo):
    base = git_repo[1]
    repos = _mirror(tmp_path, git_repo)
    key = tree_key("demo/demo", base)
    fn = instance_overview_lookup(str(_fake_cache(tmp_path / "c", key, ["demo__demo-1"])))
    seen: dict[str, list] = {a: [] for a in ("compressed_ov", "full_text_ov", "full_text")}

    def make_model(arm):
        script = (_browse_read_answer() if arm.endswith("_ov")
                  else ["FILES: pkg/calc.py<|im_end|>"])

        def step(x):
            seen[arm].append(x)
            return script[len(seen[arm]) - 1]

        if arm.startswith("compressed"):
            return CompressedModel(step, PlainRenderer())
        return FullTextModel(step, PlainRenderer())

    out = tmp_path / "out"
    recs = run_instance(_inst(base), list(seen), make_model, repos, tmp_path / "w", out,
                        overview=fn, max_turns=5)
    by = {r["arm"]: r for r in recs}
    # compressed: the FIRST slot is the cached root (marker naming the per-commit key)
    sr0 = seen["compressed_ov"][0]
    assert isinstance(sr0, SlotRendering)
    assert parse_overview(sr0.contexts[0]) == (key, fn.n_reps[key])
    user0 = next(m for m in sr0.messages if m["role"] == "user")
    assert user0["content"].startswith(OVERVIEW_HEADER) and "add(1, 2)" in user0["content"]
    assert OVERVIEW_SYSTEM_NOTE in sr0.messages[0]["content"]
    assert "browse" in {t["function"]["name"] for t in sr0.tools}
    # after browse(pkg): a directory node of the same cached tree
    assert parse_node(seen["compressed_ov"][1].contexts[1]) == (key, "pkg", 4)
    # full text: the root LISTING in the overview's place, the dir listing as the browse result
    t0, t1 = seen["full_text_ov"][0], seen["full_text_ov"][1]
    assert OVERVIEW_HEADER + "pkg/\ntests/" in t0 and "browse" in t0
    assert "__init__.py\ncalc.py" in t1
    for arm in ("compressed_ov", "full_text_ov"):
        r = by[arm]
        assert r["overview"] == key and r["browse_calls"] == 1 and r["browsed_gold_dir"]
        assert r["answer_hit"] and r["stop"] == "final" and r["turns"] == 3
    # the plain arm is unchanged: no overview, no browse tool
    assert "overview" not in by["full_text"] and "Repository overview" not in seen["full_text"][0]
    assert "browse(" not in seen["full_text"][0]
    s = summarize(list(load_records(out / "compressed_ov" / "episodes.jsonl").values()))
    assert s["with_overview"] == 1.0 and s["browsed_gold_dir"] == 1.0


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generator_dry_run_rows_carry_the_overview_slot(tmp_path, git_repo):
    base = git_repo[1]
    repos = tmp_path / "repos"
    (repos / "demo").mkdir(parents=True)
    git_repo[0].rename(repos / "demo" / "demo")  # the crawler's owner/name layout
    key = tree_key("demo/demo", base)
    tasks = [_inst(base, "demo__demo-1"), _inst(base, "demo__demo-2"), _inst(base, "nocache-3")]
    fn = instance_overview_lookup(str(_fake_cache(tmp_path / "c", key,
                                                  ["demo__demo-1", "demo__demo-2"])))
    def llm(prompt: str) -> str:  # a scripted teacher: browse, read, answer
        script = _browse_read_answer()
        if "return a - b" in prompt:  # the read result is in context
            return script[2]
        return script[1] if "__init__.py\ncalc.py" in prompt else script[0]

    gen = _load_script("bgkit2_gen_lfm_agent_trajectories")
    res = gen.generate(tasks, llm, PlainRenderer(), tmp_path / "run", str(repos),
                       str(tmp_path / "work"), samples=2, concurrency=2, overview=fn,
                       max_turns=5, decoder_token_budget=10**9, context_budget=6144)
    assert res["ran"] == 4 and res["no_tree"] == 2 and res["errors"] == 0
    scores = [json.loads(x) for x in (tmp_path / "run" / "scores.jsonl").read_text().splitlines()]
    ok = [s for s in scores if s.get("stop") == "final"]
    assert len(ok) == 4 and all(s["answer_hit"] and s["overview"] == key for s in ok)
    assert {s["arm"] for s in ok} == {"s0", "s1"}
    assert sum(s.get("stop") == "skipped" for s in scores) == 2
    lines = [json.loads(x) for x in (tmp_path / "run" / "trajectories.jsonl").read_text()
             .splitlines()]
    rows = [x for x in lines if x["kind"] == "row"]
    assert len(rows) == 12  # 4 episodes x 3 assistant turns
    for r in rows:
        ctxs = json.loads(r["context"])
        assert parse_overview(ctxs[0]) == (key, fn.n_reps[key])  # the SLOT, not the listing
        assert r["family"] == "agent_swe_lfm" and r["meta"]["repo"] == key
        assert r["meta"]["model"] in ("teacher_lfm25_8b:s0", "teacher_lfm25_8b:s1")
        msgs = json.loads(r["instruction"])
        assert OVERVIEW_SYSTEM_NOTE in msgs[0]["content"]
    assert not (tmp_path / "work" / "demo__demo-1").exists()  # export removed after both samples

    # rejection sampling -> the S2 store
    store = _load_script("bgkit2_build_swe_lfm_store")
    kept, st = store.build([tmp_path / "run"], require_overview=True)
    assert st["kept_episodes"] == 4 and len(kept) == 12 and st["rows_with_overview"] == 12
    assert st["rows_with_browse_slot"] == 8  # turns after the browse carry the node slot
    from bgkit2.data.tasks import write_task_store

    assert write_task_store(str(tmp_path / "s.parquet"), kept) == 12
