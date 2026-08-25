"""Shared bare-repo iterator: minified / generated source is rejected
(2026-08-23: whole-file 'lines' in fileneedle/grepset gold answers)."""

from __future__ import annotations

import pytest

from bgkit.data.repo_files import iter_repo_files, looks_generated_name, looks_minified

NORMAL = "\n".join(f"def fn_{i}(x):\n    return x + {i}\n" for i in range(40))
MINIFIED_ONE_LINE = "!function(e,t){" + "a=b;" * 600 + "}(this);\n"
DENSE = "\n".join("x" * 300 for _ in range(20))  # no single huge line, avg 300


def test_looks_minified():
    assert not looks_minified(NORMAL)
    assert looks_minified(MINIFIED_ONE_LINE)  # one 2400-char line
    assert looks_minified(DENSE)  # average line length above threshold
    assert looks_minified("")
    assert not looks_minified(DENSE, max_avg_line_chars=400)


def test_looks_generated_name():
    assert looks_generated_name("dist/app.min.js")
    assert looks_generated_name("static/vendor.bundle.js")
    assert not looks_generated_name("src/app.js")
    assert not looks_generated_name("src/minimal.py")


def _bare_repo_with(tmp_path, files: dict[str, str]):
    pygit2 = pytest.importorskip("pygit2")
    repo = pygit2.init_repository(str(tmp_path / "repo.git"), bare=True)
    builder = repo.TreeBuilder()
    for name, text in files.items():
        oid = repo.create_blob(text.encode())
        builder.insert(name, oid, pygit2.GIT_FILEMODE_BLOB)
    tree = builder.write()
    sig = pygit2.Signature("t", "t@example.com")
    repo.create_commit("HEAD", sig, sig, "init", tree, [])
    return tmp_path / "repo.git"


def test_iter_repo_files_skips_minified_and_generated(tmp_path):
    repo = _bare_repo_with(
        tmp_path,
        {
            "good.py": NORMAL,
            "bundle.js": MINIFIED_ONE_LINE * 2,
            "app.min.js": NORMAL,  # human-scale text but a build-artifact name
            "notes.txt": NORMAL,  # extension not in the allowlist
        },
    )
    got = list(iter_repo_files(repo, min_bytes=10, max_bytes=10_000_000, max_files=10))
    assert [p for p, _, _ in got] == ["good.py"]
    assert got[0][2] == NORMAL
    # Skipped blobs do not consume the max_files budget.
    got1 = list(iter_repo_files(repo, min_bytes=10, max_bytes=10_000_000, max_files=1))
    assert [p for p, _, _ in got1] == ["good.py"]
