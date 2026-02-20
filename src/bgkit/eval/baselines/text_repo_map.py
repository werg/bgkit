"""Text-based repo map baseline using tree-sitter structural parsing.

Parses source files into ASTs, extracts function/class signatures, builds
a dependency graph, and produces a compact structural overview similar to
aider's repo map. Falls back to a simple file listing if tree-sitter
dependencies are not installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bgkit.data.structural.dependency_graph import DependencyGraph
    from bgkit.data.structural.parser import FileSkeleton

logger = logging.getLogger(__name__)


def generate_repo_map(files: dict[str, str], max_tokens: int = 8000) -> str:
    """Generate a tree-sitter-based repo map for comparison.

    Parses source files, extracts function/class signatures, and builds
    a compact structural overview similar to aider's repo map.

    Args:
        files: Dict mapping file paths to contents.
        max_tokens: Approximate token budget (uses ~4 chars per token heuristic).

    Returns:
        Text representation with file skeletons and dependency info.
    """
    try:
        return _generate_structural_map(files, max_tokens)
    except ImportError:
        logger.warning(
            "tree-sitter-language-pack not installed, falling back to simple file listing"
        )
        return _generate_simple_map(files)


def _generate_simple_map(files: dict[str, str]) -> str:
    """Fallback: simple file listing without structural info."""
    lines = ["# Repository Structure", ""]
    for path in sorted(files.keys()):
        lines.append(f"  {path}")
    return "\n".join(lines)


def _generate_structural_map(files: dict[str, str], max_tokens: int) -> str:
    """Full tree-sitter-based structural map generation."""
    from bgkit.data.repo_processing import detect_language
    from bgkit.data.structural.dependency_graph import build_dependency_graph
    from bgkit.data.structural.parser import parse_file

    # 1. Parse all files
    skeletons: list[FileSkeleton] = []
    for path, content in files.items():
        language = detect_language(path)
        if language is None:
            continue
        try:
            skeleton = parse_file(path, content, language)
        except Exception:
            logger.debug("Failed to parse %s", path, exc_info=True)
            continue
        if skeleton is not None:
            skeletons.append(skeleton)

    # 2. Build dependency graph
    graph = build_dependency_graph(skeletons, list(files.keys()))

    # 3. Rank files by importance (number of times imported by others)
    import_counts: dict[str, int] = {}
    for edge in graph.edges:
        import_counts[edge.target_file] = import_counts.get(edge.target_file, 0) + 1

    # 4. Sort skeletons: most-imported files first, then alphabetically
    skeletons.sort(key=lambda s: (-import_counts.get(s.path, 0), s.path))

    # 5. Build output within budget
    char_budget = max_tokens * 4  # ~4 chars per token
    lines: list[str] = ["# Repository Structure", ""]

    # File tree first
    lines.append("## Files")
    for path in sorted(files.keys()):
        lines.append(f"  {path}")
    lines.append("")

    # Then skeletons for most important files
    lines.append("## Code Structure")
    lines.append("")

    current_chars = sum(len(line) + 1 for line in lines)

    for skeleton in skeletons:
        skeleton_text = _format_skeleton(skeleton)
        if current_chars + len(skeleton_text) > char_budget:
            break
        lines.append(skeleton_text)
        current_chars += len(skeleton_text)

    # Add dependency info if budget allows
    if graph.edges and current_chars < char_budget * 0.9:
        dep_text = _format_dependencies(graph, int(char_budget - current_chars))
        if dep_text:
            lines.append("")
            lines.append("## Dependencies")
            lines.append(dep_text)

    return "\n".join(lines)


def _format_skeleton(skeleton: FileSkeleton) -> str:
    """Format a file skeleton as compact text."""
    lines = [f"### {skeleton.path} ({skeleton.language})"]

    for cls in skeleton.classes:
        bases = f"({', '.join(cls.bases)})" if cls.bases else ""
        lines.append(f"  class {cls.name}{bases}:")
        for method in cls.methods:
            lines.append(f"    {method.signature}")
        if not cls.methods:
            lines.append("    ...")

    for func in skeleton.functions:
        if not func.is_method:
            lines.append(f"  {func.signature}")

    if skeleton.constants:
        for const in skeleton.constants:
            lines.append(f"  {const} = ...")

    lines.append("")
    return "\n".join(lines)


def _format_dependencies(graph: DependencyGraph, char_budget: int) -> str:
    """Format dependency edges as compact text."""
    lines: list[str] = []
    current = 0
    for edge in graph.edges:
        names = ", ".join(edge.imported_names) if edge.imported_names else "*"
        line = f"  {edge.source_file} -> {edge.target_file}: {names}"
        if current + len(line) + 1 > char_budget:
            break
        lines.append(line)
        current += len(line) + 1
    return "\n".join(lines)
