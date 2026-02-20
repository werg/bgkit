"""Tests for the structural data serializer."""

from __future__ import annotations

from bgkit.data.structural.dependency_graph import DependencyEdge, DependencyGraph
from bgkit.data.structural.parser import (
    ClassDef,
    FileSkeleton,
    FunctionDef,
    ImportDef,
)
from bgkit.data.structural.serializer import (
    serialize_dependencies,
    serialize_module_summary,
    serialize_skeleton,
)


def _sample_skeleton() -> FileSkeleton:
    """Create a sample FileSkeleton for testing."""
    return FileSkeleton(
        path="src/foo.py",
        language="Python",
        tier="A",
        functions=[
            FunctionDef(
                name="standalone_func",
                signature="def standalone_func(x, y):",
                start_line=20,
                end_line=22,
            ),
        ],
        classes=[
            ClassDef(
                name="MyClass",
                bases=["BaseClass"],
                start_line=5,
                end_line=15,
                methods=[
                    FunctionDef(
                        name="method_one",
                        signature="def method_one(self, arg: int) -> str:",
                        start_line=6,
                        end_line=8,
                        is_method=True,
                        parent_class="MyClass",
                    ),
                    FunctionDef(
                        name="method_two",
                        signature="def method_two(self) -> None:",
                        start_line=10,
                        end_line=12,
                        is_method=True,
                        parent_class="MyClass",
                    ),
                ],
            ),
        ],
        imports=[
            ImportDef(module="os", names=["os"]),
            ImportDef(module="pathlib", names=["Path"]),
            ImportDef(module=".utils", names=["helper"], is_relative=True),
        ],
        constants=["CONSTANT_NAME", "MAX_SIZE"],
    )


# ---------------------------------------------------------------------------
# Skeleton serialization
# ---------------------------------------------------------------------------

def test_serialize_skeleton_format():
    skeleton = _sample_skeleton()
    result = serialize_skeleton(skeleton)

    # Check XML tags
    assert result.startswith('<skeleton path="src/foo.py" language="Python">')
    assert result.endswith("</skeleton>")

    # Check class and methods
    assert "class MyClass(BaseClass):" in result
    assert "def method_one" in result
    assert "def method_two" in result

    # Check standalone function
    assert "def standalone_func" in result

    # Check constants
    assert "CONSTANT_NAME = ..." in result
    assert "MAX_SIZE = ..." in result


def test_serialize_skeleton_imports():
    skeleton = _sample_skeleton()
    result = serialize_skeleton(skeleton)

    # Check imports appear
    assert "import os" in result
    assert "from .utils import helper" in result


def test_serialize_skeleton_empty():
    skeleton = FileSkeleton(
        path="empty.py",
        language="Python",
        tier="A",
    )
    result = serialize_skeleton(skeleton)

    assert '<skeleton path="empty.py"' in result
    assert "</skeleton>" in result


def test_serialize_skeleton_class_no_bases():
    skeleton = FileSkeleton(
        path="test.py",
        language="Python",
        tier="A",
        classes=[
            ClassDef(name="Simple", bases=[], start_line=1, end_line=3),
        ],
    )
    result = serialize_skeleton(skeleton)

    assert "class Simple:" in result
    assert "..." in result  # Empty class body placeholder


# ---------------------------------------------------------------------------
# Dependency serialization
# ---------------------------------------------------------------------------

def test_serialize_dependencies_format():
    skeleton = _sample_skeleton()
    graph = DependencyGraph(
        edges=[
            DependencyEdge(
                source_file="src/foo.py",
                target_file="src/bar.py",
                imported_names=["Bar", "baz"],
            ),
            DependencyEdge(
                source_file="src/main.py",
                target_file="src/foo.py",
                imported_names=["MyClass"],
            ),
        ],
    )

    result = serialize_dependencies(skeleton, graph)

    assert result.startswith('<dependencies path="src/foo.py">')
    assert result.endswith("</dependencies>")

    # Outbound
    assert "imports:" in result
    assert "src/bar.py (Bar, baz)" in result

    # Inbound
    assert "imported_by:" in result
    assert "src/main.py (MyClass)" in result


def test_serialize_dependencies_no_edges():
    skeleton = FileSkeleton(
        path="isolated.py",
        language="Python",
        tier="A",
    )
    graph = DependencyGraph()

    result = serialize_dependencies(skeleton, graph)

    assert "(no dependencies)" in result


def test_serialize_dependencies_only_outbound():
    skeleton = FileSkeleton(
        path="consumer.py",
        language="Python",
        tier="A",
    )
    graph = DependencyGraph(
        edges=[
            DependencyEdge(
                source_file="consumer.py",
                target_file="provider.py",
                imported_names=["func"],
            ),
        ],
    )

    result = serialize_dependencies(skeleton, graph)

    assert "imports:" in result
    assert "provider.py (func)" in result
    assert "imported_by:" not in result


# ---------------------------------------------------------------------------
# Module summary serialization
# ---------------------------------------------------------------------------

def test_serialize_module_summary_format():
    skeletons = [
        FileSkeleton(
            path="src/pkg/parser.py",
            language="Python",
            tier="A",
            functions=[
                FunctionDef(
                    name="parse_file",
                    signature="def parse_file(path, content, language):",
                    start_line=1,
                    end_line=10,
                ),
                FunctionDef(
                    name="_private_func",
                    signature="def _private_func():",
                    start_line=12,
                    end_line=15,
                ),
            ],
            classes=[
                ClassDef(
                    name="FileSkeleton",
                    bases=[],
                    start_line=20,
                    end_line=30,
                ),
            ],
            constants=["MAX_SIZE"],
        ),
        FileSkeleton(
            path="src/pkg/graph.py",
            language="Python",
            tier="A",
            functions=[
                FunctionDef(
                    name="build_graph",
                    signature="def build_graph(skeletons):",
                    start_line=1,
                    end_line=20,
                ),
            ],
            classes=[
                ClassDef(
                    name="DependencyGraph",
                    bases=[],
                    start_line=25,
                    end_line=35,
                ),
            ],
        ),
        # File in a different directory -- should NOT appear
        FileSkeleton(
            path="src/other/module.py",
            language="Python",
            tier="A",
            functions=[
                FunctionDef(
                    name="other_func",
                    signature="def other_func():",
                    start_line=1,
                    end_line=5,
                ),
            ],
        ),
    ]

    result = serialize_module_summary(skeletons, "src/pkg")

    assert result.startswith('<module path="src/pkg">')
    assert result.endswith("</module>")

    # Public symbols should appear
    assert "class FileSkeleton" in result
    assert "def parse_file" in result
    assert "class DependencyGraph" in result
    assert "def build_graph" in result
    assert "MAX_SIZE = ..." in result

    # Private symbols should NOT appear
    assert "_private_func" not in result

    # File from other directory should NOT appear
    assert "other_func" not in result


def test_serialize_module_summary_empty():
    result = serialize_module_summary([], "src/nonexistent")

    assert "(empty module)" in result


def test_serialize_module_summary_trailing_slash():
    """Module path with trailing slash should still work."""
    skeletons = [
        FileSkeleton(
            path="pkg/mod.py",
            language="Python",
            tier="A",
            functions=[
                FunctionDef(
                    name="func",
                    signature="def func():",
                    start_line=1,
                    end_line=2,
                ),
            ],
        ),
    ]

    result = serialize_module_summary(skeletons, "pkg/")

    assert "def func" in result
