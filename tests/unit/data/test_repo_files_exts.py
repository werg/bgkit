"""``iter_repo_files`` gained parameters; its DEFAULT must not have moved.

Four shipped generators (fileneedle, grepset, reconstruct, xref) call this
with no ``exts`` and no ``skip_minified``, and their corpora were floor-tested
against what it yielded then. A widened default would silently pull data files
and minified bundles into families whose answers are lines of source -- the
2026-08-23 failure where fileneedle gold answers reached 116K chars because
minified blobs were in scope.
"""

from __future__ import annotations

from bgkit.data.repo_files import DATA_EXTS, EXTS, looks_generated_name


def test_source_default_is_unchanged() -> None:
    assert {
        ".py", ".js", ".ts", ".tsx", ".go", ".rs", ".java", ".c", ".h", ".cpp", ".rb",
    } == EXTS


def test_data_exts_are_disjoint_from_source() -> None:
    """A data file must never arrive through a source-only call."""
    assert not (EXTS & DATA_EXTS)


def test_lock_files_are_generated() -> None:
    """Lock files are machine-written and near-identical across repos: a
    lookup over one is a lookup over a template, not over the repo."""
    for name in ("package-lock.json", "yarn.lock", "Cargo.lock", "go.sum"):
        assert looks_generated_name(f"a/b/{name}"), name
    assert not looks_generated_name("src/package.json")


def test_default_signature_keeps_minified_rejection_on() -> None:
    import inspect

    from bgkit.data.repo_files import iter_repo_files

    sig = inspect.signature(iter_repo_files)
    assert sig.parameters["skip_minified"].default is True
    assert sig.parameters["exts"].default is None
