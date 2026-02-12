"""Reconstruction metrics: loss and parse success rate."""

from __future__ import annotations


def parse_success_rate(generated_code: list[str], language: str = "python") -> float:
    """Check if generated code parses successfully.

    Args:
        generated_code: List of generated code strings.
        language: Programming language for parsing.

    Returns:
        Fraction of examples that parse successfully.
    """
    import ast

    if language != "python":
        # TODO: Support other languages via tree-sitter
        raise NotImplementedError(f"Parsing not implemented for {language}")

    successes = 0
    for code in generated_code:
        try:
            ast.parse(code)
            successes += 1
        except SyntaxError:
            pass
    return successes / max(len(generated_code), 1)
