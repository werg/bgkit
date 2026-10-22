from __future__ import annotations

from pathlib import Path

from bgkit2.agent.context import DEFAULT_RATIOS, RatioPolicy
from bgkit2.agent.tools import BM25, EditAudit, ToolRegistry, tokenize


def test_specs_list_all_tools_in_template_shape(repo: Path):
    reg = ToolRegistry(repo)
    specs = reg.specs()
    assert [s["function"]["name"] for s in specs] == [
        "repo_tree",
        "overview",
        "ls",
        "read",
        "search",
        "docs",
        "zoom",
        "write",
        "edit",
        "bash",
    ]
    assert all(s["type"] == "function" and "parameters" in s["function"] for s in specs)
    assert reg.spec("zoom").compressible is True and reg.spec("read").compressible is True
    # the verbatim switch is NOT in the default spec the trained model sees
    zoom_props = reg.spec("zoom").parameters["properties"]
    assert "exact" not in zoom_props and "focus" in zoom_props
    assert "exact" not in reg.spec("zoom").description
    ex = ToolRegistry(repo, zoom_exact=True).spec("zoom")
    assert "exact" in ex.parameters["properties"] and "VERBATIM" in ex.description


def test_read_search_zoom(repo: Path):
    reg = ToolRegistry(repo)
    r = reg.call("read", {"path": "pkg/calc.py", "focus": "the add function"})
    assert r.ok and r.compressible and "return a - b" in r.text
    assert r.prompt == "read pkg/calc.py: the add function"

    s = reg.call("search", {"query": r"def \w+\(", "focus": "definitions"})
    assert s.ok and s.compressible
    assert s.text.splitlines() == [
        "pkg/calc.py:1: def add(a, b):",
        "pkg/calc.py:5: def mul(a, b):",
        "tests/test_calc.py:4: def test_add():",
    ]
    assert s.prompt == "search def \\w+\\(: definitions"
    assert "no matches" in reg.call("search", {"query": "zzz_nothing"}).text

    # zoom (default): the WHOLE file, numbered, as a slot whose prompt names the range
    z = reg.call("zoom", {"path": "pkg/calc.py", "start": 1, "end": 2, "focus": "the bug"})
    assert z.ok and z.compressible and z.tool == "zoom"
    assert z.prompt == "zoom pkg/calc.py lines 1-2: the bug"
    assert z.text.startswith("1: def add(a, b):\n2:     return a - b  # BUG\n")
    assert "5: def mul(a, b):" in z.text  # beyond the range: the whole file
    assert reg.call("zoom", {"path": "pkg/calc.py", "start": 1, "end": 2}).prompt == \
        "zoom pkg/calc.py lines 1-2"
    past = reg.call("zoom", {"path": "pkg/calc.py", "start": 900, "end": 950})
    assert past.ok and not past.compressible and "nothing in 900..950" in past.text
    # exact is off unless the registry enables it
    off = reg.call("zoom", {"path": "pkg/calc.py", "start": 1, "end": 2, "exact": True})
    assert not off.ok and "exact" in off.text
    ex = ToolRegistry(repo, zoom_exact=True)
    z = ex.call("zoom", {"path": "pkg/calc.py", "start": 1, "end": 2, "exact": "true"})
    assert z.text == "1: def add(a, b):\n2:     return a - b  # BUG"
    assert z.compressible is False and z.prompt == ""
    assert ex.call("zoom", {"path": "pkg/calc.py", "start": 1, "end": 2}).compressible


def test_ratio_defaults_for_exact_quote_tools():
    rp = RatioPolicy()
    assert DEFAULT_RATIOS["zoom"] == 4 and rp.ratio("web_find") == 4
    assert rp.ratio("web_search") == 4 and rp.ratio("web_open") == 16
    assert rp.reps(40, "zoom") == 10


def test_edit_audit_records_verbatim_old(repo: Path):
    reg = EditAudit(ToolRegistry(repo))
    assert "edit" in reg.names  # delegates the registry
    reg.call("edit", {"path": "pkg/calc.py", "old": "return a  -  b", "new": "x"})  # spacing off
    reg.call("edit", {"path": "pkg/calc.py", "old": "a - b  # BUG", "new": "a + b"})  # copied
    reg.call("edit", {"path": "pkg/calc.py", "old": "a - b  # BUG", "new": "a + b"})  # gone now
    reg.call("edit", {"path": "pkg/none.py", "old": "x", "new": "y"})
    e = reg.edits
    assert [x["old_verbatim"] for x in e] == [False, True, False, False]
    assert e[0]["old_ws_match"] and not e[0]["ok"]
    assert e[1]["old_unique"] and e[1]["ok"] and not e[3]["file_exists"]
    s = reg.summary()
    assert s["edit_calls"] == 4 and s["edit_old_verbatim"] == 0.25 and s["edit_old_ws_match"] == 0.5


def test_edit_write_bash(repo: Path):
    reg = ToolRegistry(repo, bash_timeout=10)
    e = reg.call("edit", {"path": "pkg/calc.py", "old": "a - b  # BUG", "new": "a + b"})
    assert e.ok and not e.compressible and e.text == "edited pkg/calc.py"
    assert "a + b" in (repo / "pkg/calc.py").read_text()
    bad = reg.call("edit", {"path": "pkg/calc.py", "old": "nope", "new": "x"})
    assert not bad.ok and "not found" in bad.text
    dup = reg.call("edit", {"path": "pkg/calc.py", "old": "a", "new": "x"})
    assert not dup.ok and "occurs" in dup.text

    w = reg.call("write", {"path": "new/mod.py", "content": "x = 1\n"})
    assert w.ok and (repo / "new/mod.py").read_text() == "x = 1\n"

    b = reg.call("bash", {"cmd": "python3 -c 'from pkg.calc import add; print(add(1, 2))'"})
    assert b.ok and b.compressible and b.text.strip() == "3" and b.prompt.startswith("bash: ")
    fail = reg.call("bash", {"cmd": "exit 3"})
    assert not fail.ok and "(exit code 3)" in fail.text
    slow = ToolRegistry(repo, bash_timeout=0.2).call("bash", {"cmd": "sleep 2"})
    assert not slow.ok and "timed out" in slow.text


def test_errors_are_results_not_exceptions(repo: Path):
    reg = ToolRegistry(repo)
    u = reg.call("fly", {"x": 1})
    assert not u.ok and "unknown tool" in u.text and not u.compressible
    esc = reg.call("read", {"path": "../outside.txt"})
    assert not esc.ok and "escapes" in esc.text
    missing = reg.call("read", {"path": "pkg/none.py"})
    assert not missing.ok and "not a file" in missing.text
    extra = reg.call("read", {"path": "pkg/calc.py", "bogus": 1})
    assert not extra.ok and "bogus" in extra.text


def test_docs_bm25_and_stub(repo: Path, tmp_path: Path):
    assert not ToolRegistry(repo).call("docs", {"query": "x"}).ok
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "requests.md").write_text("requests.get sends an HTTP GET request with params")
    (docs / "numpy.md").write_text("numpy.zeros allocates an array of zeros")
    (docs / "torch.txt").write_text("torch.cat concatenates tensors along a dim")
    reg = ToolRegistry(repo, docs_dir=docs)
    r = reg.call("docs", {"query": "http get request", "focus": "params"})
    assert r.ok and r.compressible and r.text.startswith("# requests.md")
    assert r.prompt == "docs http get request: params"
    top1 = reg.call("docs", {"query": "concatenate tensors", "k": 1})
    assert "# torch.txt" in top1.text and "numpy" not in top1.text

    bm = BM25([tokenize("a b c"), tokenize("a a d")])
    s = bm.scores(["a"])
    assert s[1] > s[0] > 0 and bm.scores(["zzz"]) == [0.0, 0.0]


def test_ls_lists_files_and_skips_noise(repo: Path):
    (repo / "__pycache__").mkdir(exist_ok=True)
    (repo / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"\x00")
    out = ToolRegistry(repo).call("ls")
    assert out.ok and "pkg/calc.py" in out.text
    assert "__pycache__" not in out.text and ".pyc" not in out.text


def test_read_error_names_the_recovery(repo: Path):
    """The student's dominant failure is a guessed basename followed by
    repeating it; the error has to point at `ls`/`search`, not just refuse."""
    out = ToolRegistry(repo).call("read", {"path": "calc.py"})
    assert not out.ok
    assert "ls" in out.text and "search" in out.text and "do not guess" in out.text


def test_overview_maps_modules_to_their_definitions(repo: Path):
    """The map has to name both the module and what it defines — a bare path
    list is what P26 showed does not help (tool use 17% -> 10%)."""
    out = ToolRegistry(repo).call("overview")
    assert out.ok and out.compressible  # compressible => it becomes an encoder slot
    assert "pkg/calc.py" in out.text and "add" in out.text


def test_repo_tree_tool_emits_a_tree_payload(repo: Path):
    import json

    from bgkit2.agent.student import tree_payload

    out = ToolRegistry(repo).call("repo_tree")
    assert out.ok and out.compressible and out.tool == "repo_tree"
    d = tree_payload(out.text)
    assert d is not None and d["repo_tree"] == 1
    assert any(f["path"] == "pkg/calc.py" and "def add" in f["source"] for f in d["files"])
    assert tree_payload("plain text result") is None
    assert tree_payload(json.dumps({"repo_tree": 1})) is None  # no files -> not a tree
