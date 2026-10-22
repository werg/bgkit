"""Reconstruction metrics: loss and parse success rate."""

from __future__ import annotations

import re

import structlog

logger = structlog.get_logger()

# Mapping from dataset language names to tree-sitter grammar names.
# Python uses ast.parse() instead of tree-sitter for stricter checking.
_LANGUAGE_TO_TS_GRAMMAR: dict[str, str] = {
    "JavaScript": "javascript",
    "TypeScript": "typescript",
    "Java": "java",
    "Go": "go",
    "Rust": "rust",
    "C": "c",
    "C++": "cpp",
    "C#": "c_sharp",
    "Ruby": "ruby",
    "PHP": "php",
    "Swift": "swift",
    "Kotlin": "kotlin",
    "Scala": "scala",
    "Shell": "bash",
    "Lua": "lua",
    "R": "r",
    "Julia": "julia",
    "Haskell": "haskell",
    "OCaml": "ocaml",
    "Elixir": "elixir",
    "Erlang": "erlang",
    "Clojure": "clojure",
    "F#": "c_sharp",  # No dedicated F# grammar; best-effort via C#
    "Zig": "zig",
    "Perl": "perl",
    "SQL": "sql",
    "HTML": "html",
    "CSS": "css",
    "Markdown": "markdown",
    "YAML": "yaml",
    "TOML": "toml",
    "JSON": "json",
    "XML": "xml",
    "Protocol Buffers": "proto",
    "Terraform": "hcl",
    "Dockerfile": "dockerfile",
    "CMake": "cmake",
    "Nix": "nix",
    "Dart": "dart",
}

# Languages we know we can't parse (no grammar available).
# These are excluded from the metric denominator.
_UNPARSEABLE_LANGUAGES: set[str] = {
    "Nim", "D", "V", "Crystal", "Emacs Lisp", "Vim script",
    "reStructuredText", "Gradle", "Makefile", "Just",
}


# Case-insensitive reverse lookup: lowercase -> canonical title-cased name.
_LANGUAGE_LOWERCASE: dict[str, str] = {
    k.lower(): k
    for k in list(_LANGUAGE_TO_TS_GRAMMAR.keys()) + list(_UNPARSEABLE_LANGUAGES) + ["Python"]
}


def _normalize_language(lang: str) -> str:
    """Normalize a language name to the canonical title-cased form."""
    return _LANGUAGE_LOWERCASE.get(lang.lower(), lang)


def extract_code_from_chat_response(text: str) -> str:
    """Extract code content from a chat-formatted decoder response.

    The decoder produces chat-template output with a response prefix, think
    block, and markdown code fence. This function extracts just the code
    inside the **last** fence, which is what should be evaluated for parse success.

    Uses last-fence matching because the generated content is always the final
    fence before ``<|im_end|>`` or end of string. This correctly handles files
    containing inner triple backticks (e.g., markdown/README files).

    Best-effort for third-party/external text only — our own generation pipeline
    uses structural ``GenerationOutput`` (token-level boundaries) and never calls
    this function.

    Falls back to returning the full text if no code fence is found
    (e.g., raw output without chat template wrapping).
    """
    matches = re.findall(r"```\w*\n(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text


def _parse_python(code: str) -> bool:
    """Check if code parses as valid Python using ast.parse()."""
    import ast

    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _get_ts_parser(grammar: str):
    """Get a cached tree-sitter parser for the given grammar name.

    Returns None if tree-sitter-language-pack is not installed or the
    grammar is not available.
    """
    cache = _get_ts_parser.__dict__.setdefault("_cache", {})
    if grammar in cache:
        return cache[grammar]

    try:
        from tree_sitter_language_pack import get_parser

        parser = get_parser(grammar)
        cache[grammar] = parser
        return parser
    except ImportError:
        cache[grammar] = None
        return None
    except Exception:
        cache[grammar] = None
        return None


def _parse_with_tree_sitter(code: str, grammar: str) -> bool | None:
    """Parse code using tree-sitter and check for errors.

    Returns True if parsing succeeds without errors, False if there are
    parse errors, or None if tree-sitter is unavailable for this grammar.
    """
    parser = _get_ts_parser(grammar)
    if parser is None:
        return None
    tree = parser.parse(code.encode("utf-8"))
    return not tree.root_node.has_error


def parse_success_rate(
    generated_code: list[str],
    language: str = "Python",
    languages: list[str] | None = None,
    chat_formatted: bool = False,
) -> float:
    """Check if generated code parses successfully.

    By default, expects raw code strings (as returned by
    ``GenerationOutput.content_text``).  Set ``chat_formatted=True`` to
    auto-extract code from markdown fences before parsing.

    Supports Python (via ``ast.parse``) and 30+ other languages (via
    ``tree-sitter-language-pack``). Languages without a parser are excluded
    from the denominator and logged.

    Args:
        generated_code: List of generated code strings.
        language: Default language for all samples (used when ``languages`` is None).
        languages: Per-sample language labels. When provided, each sample is
            parsed according to its own language.
        chat_formatted: If True, extract code from markdown fences before parsing.
            Use this only for raw chat-template output, not for content already
            extracted via ``GenerationOutput``.

    Returns:
        Fraction of parseable-language examples that parse successfully.
    """
    if languages is not None and len(languages) != len(generated_code):
        raise ValueError(
            f"languages length ({len(languages)}) != generated_code length ({len(generated_code)})"
        )

    successes = 0
    evaluated = 0
    skipped_langs: set[str] = set()

    for i, code in enumerate(generated_code):
        lang = languages[i] if languages is not None else language

        if chat_formatted:
            code = extract_code_from_chat_response(code)

        # Normalize language casing for lookup
        lang_key = _normalize_language(lang)

        # Python: use ast.parse (stricter than tree-sitter)
        if lang_key == "Python":
            evaluated += 1
            if _parse_python(code):
                successes += 1
            continue

        # Known unparseable languages: skip
        if lang_key in _UNPARSEABLE_LANGUAGES:
            skipped_langs.add(lang)
            continue

        # Try tree-sitter
        grammar = _LANGUAGE_TO_TS_GRAMMAR.get(lang_key)
        if grammar is None:
            skipped_langs.add(lang)
            continue

        result = _parse_with_tree_sitter(code, grammar)
        if result is None:
            # tree-sitter-language-pack not installed; skip non-Python
            skipped_langs.add(lang)
            continue

        evaluated += 1
        if result:
            successes += 1

    if skipped_langs:
        logger.debug("parse_success_skipped_languages", languages=sorted(skipped_langs))

    return successes / max(evaluated, 1)
