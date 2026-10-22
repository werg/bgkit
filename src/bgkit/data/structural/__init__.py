"""Structural code extraction using tree-sitter.

Parses source files into ASTs, extracts symbol definitions (functions, classes,
imports, constants), resolves imports into a dependency graph, and serializes
as structured text.
"""

from __future__ import annotations

from bgkit.data.structural.dependency_graph import (
    DependencyEdge,
    DependencyGraph,
    build_dependency_graph,
)
from bgkit.data.structural.parser import (
    ClassDef,
    FileSkeleton,
    FunctionDef,
    ImportDef,
    parse_file,
    parse_repo,
)
from bgkit.data.structural.serializer import (
    serialize_dependencies,
    serialize_module_summary,
    serialize_skeleton,
)

__all__ = [
    "ClassDef",
    "DependencyEdge",
    "DependencyGraph",
    "FileSkeleton",
    "FunctionDef",
    "ImportDef",
    "build_dependency_graph",
    "parse_file",
    "parse_repo",
    "serialize_dependencies",
    "serialize_module_summary",
    "serialize_skeleton",
]
