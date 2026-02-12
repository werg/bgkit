"""Text-based repo map baseline.

Simple text representation of repository structure for comparison.
"""

from __future__ import annotations


def generate_repo_map(files: dict[str, str], max_tokens: int = 8000) -> str:
    """Generate a text-based repo map for comparison.

    Args:
        files: Dict mapping file paths to contents.
        max_tokens: Maximum token budget for the map.

    Returns:
        Text representation of the repository structure.
    """
    # TODO: Implement tree-sitter based repo map (like aider's repo map)
    lines = ["Repository structure:", ""]
    for path in sorted(files.keys()):
        lines.append(f"  {path}")
    return "\n".join(lines)
