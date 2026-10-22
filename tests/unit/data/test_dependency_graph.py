"""Tests for the dependency graph builder."""

from __future__ import annotations

from bgkit.data.structural.dependency_graph import (
    DependencyGraph,
    build_dependency_graph,
)
from bgkit.data.structural.parser import FileSkeleton, ImportDef


def _make_skeleton(
    path: str,
    language: str = "Python",
    imports: list[ImportDef] | None = None,
) -> FileSkeleton:
    """Helper to create a minimal FileSkeleton for testing."""
    return FileSkeleton(
        path=path,
        language=language,
        tier="A",
        imports=imports or [],
    )


# ---------------------------------------------------------------------------
# Python import resolution
# ---------------------------------------------------------------------------

def test_python_absolute_import():
    """Absolute Python imports should resolve to repo-local files."""
    skeletons = [
        _make_skeleton(
            "src/main.py",
            imports=[ImportDef(module="utils", names=["helper"], is_relative=False)],
        ),
        _make_skeleton("src/utils.py"),
    ]
    file_paths = ["src/main.py", "src/utils.py"]

    graph = build_dependency_graph(skeletons, file_paths)

    assert len(graph.edges) == 1
    assert graph.edges[0].source_file == "src/main.py"
    assert graph.edges[0].target_file == "src/utils.py"
    assert "helper" in graph.edges[0].imported_names


def test_python_relative_import():
    """Relative Python imports should resolve against directory structure."""
    skeletons = [
        _make_skeleton(
            "src/pkg/module.py",
            imports=[
                ImportDef(module=".utils", names=["foo"], is_relative=True),
            ],
        ),
        _make_skeleton("src/pkg/utils.py"),
    ]
    file_paths = ["src/pkg/module.py", "src/pkg/utils.py"]

    graph = build_dependency_graph(skeletons, file_paths)

    assert len(graph.edges) == 1
    assert graph.edges[0].target_file == "src/pkg/utils.py"


def test_python_relative_import_parent():
    """Relative imports with .. should go up one directory."""
    skeletons = [
        _make_skeleton(
            "src/pkg/sub/module.py",
            imports=[
                ImportDef(module="..base", names=["Base"], is_relative=True),
            ],
        ),
        _make_skeleton("src/pkg/base.py"),
    ]
    file_paths = ["src/pkg/sub/module.py", "src/pkg/base.py"]

    graph = build_dependency_graph(skeletons, file_paths)

    assert len(graph.edges) == 1
    assert graph.edges[0].target_file == "src/pkg/base.py"


def test_python_init_import():
    """Import of a package should resolve to __init__.py."""
    skeletons = [
        _make_skeleton(
            "src/main.py",
            imports=[ImportDef(module="pkg", names=["Foo"], is_relative=False)],
        ),
        _make_skeleton("src/pkg/__init__.py"),
    ]
    file_paths = ["src/main.py", "src/pkg/__init__.py"]

    graph = build_dependency_graph(skeletons, file_paths)

    assert len(graph.edges) == 1
    assert graph.edges[0].target_file == "src/pkg/__init__.py"


# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------

def test_third_party_imports_dropped():
    """Imports that don't match any repo file should be dropped."""
    skeletons = [
        _make_skeleton(
            "main.py",
            imports=[
                ImportDef(module="numpy", names=["array"], is_relative=False),
                ImportDef(module="requests", names=["get"], is_relative=False),
            ],
        ),
    ]
    file_paths = ["main.py"]

    graph = build_dependency_graph(skeletons, file_paths)

    assert len(graph.edges) == 0


# ---------------------------------------------------------------------------
# JavaScript import resolution
# ---------------------------------------------------------------------------

def test_js_relative_import():
    """JS relative imports should resolve with extension probing."""
    skeletons = [
        _make_skeleton(
            "src/app.js",
            language="JavaScript",
            imports=[ImportDef(module="./utils", names=["helper"], is_relative=True)],
        ),
        _make_skeleton("src/utils.js", language="JavaScript"),
    ]
    file_paths = ["src/app.js", "src/utils.js"]

    graph = build_dependency_graph(skeletons, file_paths)

    assert len(graph.edges) == 1
    assert graph.edges[0].target_file == "src/utils.js"


def test_js_index_import():
    """JS imports of a directory should resolve to index file."""
    skeletons = [
        _make_skeleton(
            "src/app.js",
            language="JavaScript",
            imports=[ImportDef(module="./components", names=["Button"], is_relative=True)],
        ),
        _make_skeleton("src/components/index.js", language="JavaScript"),
    ]
    file_paths = ["src/app.js", "src/components/index.js"]

    graph = build_dependency_graph(skeletons, file_paths)

    assert len(graph.edges) == 1
    assert graph.edges[0].target_file == "src/components/index.js"


# ---------------------------------------------------------------------------
# Go import resolution
# ---------------------------------------------------------------------------

def test_go_local_import():
    """Go imports matching repo directories should resolve."""
    skeletons = [
        _make_skeleton(
            "cmd/main.go",
            language="Go",
            imports=[
                ImportDef(module="github.com/user/repo/pkg/utils", names=["utils"]),
            ],
        ),
        _make_skeleton("pkg/utils/helpers.go", language="Go"),
    ]
    file_paths = ["cmd/main.go", "pkg/utils/helpers.go"]

    graph = build_dependency_graph(skeletons, file_paths)

    assert len(graph.edges) == 1
    assert graph.edges[0].target_file == "pkg/utils/helpers.go"


# ---------------------------------------------------------------------------
# C/C++ include resolution
# ---------------------------------------------------------------------------

def test_c_relative_include():
    """C includes with quotes should resolve relative to source file."""
    skeletons = [
        _make_skeleton(
            "src/main.c",
            language="C",
            imports=[ImportDef(module="utils.h", names=["utils"])],
        ),
        _make_skeleton("src/utils.h", language="C"),
    ]
    file_paths = ["src/main.c", "src/utils.h"]

    graph = build_dependency_graph(skeletons, file_paths)

    assert len(graph.edges) == 1
    assert graph.edges[0].target_file == "src/utils.h"


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------

def test_adjacency_list():
    """Adjacency list should contain unique sorted targets."""
    skeletons = [
        _make_skeleton(
            "a.py",
            imports=[
                ImportDef(module="b", names=["x"], is_relative=False),
                ImportDef(module="c", names=["y"], is_relative=False),
            ],
        ),
        _make_skeleton("b.py"),
        _make_skeleton("c.py"),
    ]
    file_paths = ["a.py", "b.py", "c.py"]

    graph = build_dependency_graph(skeletons, file_paths)

    assert "a.py" in graph.adjacency
    targets = graph.adjacency["a.py"]
    assert targets == sorted(set(targets))
    assert "b.py" in targets
    assert "c.py" in targets


def test_empty_graph():
    """No skeletons should produce an empty graph."""
    graph = build_dependency_graph([], [])
    assert graph.edges == []
    assert dict(graph.adjacency) == {}


def test_rust_crate_import():
    """Rust crate:: imports should resolve to src/ files."""
    skeletons = [
        _make_skeleton(
            "src/main.rs",
            language="Rust",
            imports=[
                ImportDef(
                    module="crate::utils",
                    names=["helper"],
                    is_relative=True,
                ),
            ],
        ),
        _make_skeleton("src/utils.rs", language="Rust"),
    ]
    file_paths = ["src/main.rs", "src/utils.rs"]

    graph = build_dependency_graph(skeletons, file_paths)

    assert len(graph.edges) == 1
    assert graph.edges[0].target_file == "src/utils.rs"
