"""Core tree-sitter parsing engine for structural code extraction.

Parses source files into ASTs with tree-sitter, extracts symbol definitions
(functions, classes, imports, constants), and produces FileSkeleton objects.

Two-tier language support:
  - Tier A: Custom .scm queries for top languages (Python, JS, TS, Go, Java,
    Rust, C, C++, Ruby, PHP, Shell). Full extraction.
  - Tier B: Generic node-type queries for all other tree-sitter-supported
    languages. Returns partial skeletons.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import structlog
import tree_sitter
from tree_sitter_language_pack import get_language, get_parser

from bgkit.data.repo_processing import RepoSnapshot

logger = structlog.get_logger()

# Directory containing .scm query files
_QUERIES_DIR = Path(__file__).parent / "queries"

# Map from detect_language() names to tree-sitter language names
LANGUAGE_MAP: dict[str, str] = {
    "Python": "python",
    "JavaScript": "javascript",
    "TypeScript": "typescript",
    "Go": "go",
    "Java": "java",
    "Rust": "rust",
    "C": "c",
    "C++": "cpp",
    "C#": "c_sharp",
    "Ruby": "ruby",
    "PHP": "php",
    "Shell": "bash",
    "Kotlin": "kotlin",
    "Scala": "scala",
    "Swift": "swift",
    "Lua": "lua",
    "R": "r",
    "Julia": "julia",
    "Haskell": "haskell",
    "OCaml": "ocaml",
    "Elixir": "elixir",
    "Erlang": "erlang",
    "Clojure": "clojure",
    "Dart": "dart",
    "Zig": "zig",
    "Nim": "nim",
    "Perl": "perl",
    "HTML": "html",
    "CSS": "css",
    "YAML": "yaml",
    "TOML": "toml",
    "JSON": "json",
    "SQL": "sql",
    "Markdown": "markdown",
    "Dockerfile": "dockerfile",
    "Makefile": "make",
    "CMake": "cmake",
    "Nix": "nix",
}

# Tier A languages with custom .scm query files
TIER_A_LANGUAGES: frozenset[str] = frozenset({
    "python", "javascript", "typescript", "go", "java",
    "rust", "c", "cpp", "ruby", "php", "bash",
})

# Cache for loaded query strings
_query_cache: dict[str, str] = {}

# Cache for compiled Query objects (keyed by ts_lang_name)
_compiled_query_cache: dict[str, tree_sitter.Query | None] = {}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FunctionDef:
    """A function or method definition."""

    name: str
    signature: str  # Full signature line(s)
    start_line: int
    end_line: int
    is_method: bool = False
    parent_class: str | None = None


@dataclass
class ClassDef:
    """A class, struct, interface, or type definition."""

    name: str
    bases: list[str]  # Inheritance chain
    start_line: int
    end_line: int
    methods: list[FunctionDef] = field(default_factory=list)


@dataclass
class ImportDef:
    """An import statement."""

    module: str  # What's being imported from
    names: list[str]  # What names are imported
    is_relative: bool = False


@dataclass
class FileSkeleton:
    """Structural skeleton of a single source file."""

    path: str
    language: str
    tier: str  # "A" or "B"
    functions: list[FunctionDef] = field(default_factory=list)
    classes: list[ClassDef] = field(default_factory=list)
    imports: list[ImportDef] = field(default_factory=list)
    constants: list[str] = field(default_factory=list)  # Top-level constant names


# ---------------------------------------------------------------------------
# Query loading
# ---------------------------------------------------------------------------

def _load_query_string(ts_lang_name: str) -> str | None:
    """Load a .scm query file for the given tree-sitter language name.

    Returns None if no query file exists (Tier B / unsupported).
    """
    if ts_lang_name in _query_cache:
        return _query_cache[ts_lang_name]

    scm_path = _QUERIES_DIR / f"{ts_lang_name}.scm"
    if not scm_path.exists():
        return None

    query_str = scm_path.read_text(encoding="utf-8")
    _query_cache[ts_lang_name] = query_str
    return query_str


def _get_compiled_query(ts_lang_name: str) -> tree_sitter.Query | None:
    """Get or compile a tree-sitter Query for the given language.

    Returns None if the query fails to compile or no .scm file exists.
    """
    if ts_lang_name in _compiled_query_cache:
        return _compiled_query_cache[ts_lang_name]

    query_str = _load_query_string(ts_lang_name)
    if query_str is None:
        _compiled_query_cache[ts_lang_name] = None
        return None

    try:
        lang = get_language(ts_lang_name)
        query = tree_sitter.Query(lang, query_str)
        _compiled_query_cache[ts_lang_name] = query
        return query
    except Exception:
        logger.warning("query_compile_failed", language=ts_lang_name)
        _compiled_query_cache[ts_lang_name] = None
        return None


# ---------------------------------------------------------------------------
# Signature extraction helpers
# ---------------------------------------------------------------------------

def _extract_signature(node: tree_sitter.Node, max_lines: int = 3) -> str:
    """Extract the signature portion of a function/class node.

    Takes at most `max_lines` lines from the start of the node text.
    For function definitions, this typically captures the def/func line.
    """
    text = node.text.decode("utf-8", errors="replace")
    lines = text.split("\n")
    sig_lines = lines[:max_lines]
    return "\n".join(sig_lines).rstrip()


def _first_line(node: tree_sitter.Node) -> str:
    """Get the first line of a node's text."""
    text = node.text.decode("utf-8", errors="replace")
    return text.split("\n", 1)[0].rstrip()


# ---------------------------------------------------------------------------
# Tier A extraction (custom .scm queries)
# ---------------------------------------------------------------------------

def _extract_tier_a(
    root: tree_sitter.Node,
    ts_lang_name: str,
    query: tree_sitter.Query,
) -> FileSkeleton:
    """Extract structural info using a custom Tier A query."""
    skeleton = FileSkeleton(path="", language=ts_lang_name, tier="A")

    cursor = tree_sitter.QueryCursor(query)
    captures = cursor.captures(root)

    # Dispatch to language-specific extractors
    if ts_lang_name == "python":
        _extract_python(skeleton, root, captures)
    elif ts_lang_name in ("javascript", "typescript"):
        _extract_js_ts(skeleton, root, captures)
    elif ts_lang_name == "go":
        _extract_go(skeleton, root, captures)
    elif ts_lang_name == "java":
        _extract_java(skeleton, root, captures)
    elif ts_lang_name == "rust":
        _extract_rust(skeleton, root, captures)
    elif ts_lang_name in ("c", "cpp"):
        _extract_c_cpp(skeleton, root, captures)
    elif ts_lang_name == "ruby":
        _extract_ruby(skeleton, root, captures)
    elif ts_lang_name == "php":
        _extract_php(skeleton, root, captures)
    elif ts_lang_name == "bash":
        _extract_bash(skeleton, root, captures)

    return skeleton


def _extract_python(
    skeleton: FileSkeleton,
    root: tree_sitter.Node,
    captures: dict[str, list[tree_sitter.Node]],
) -> None:
    """Extract Python structural elements from query captures."""
    # Build class map (name -> ClassDef)
    class_map: dict[str, ClassDef] = {}

    # Process classes
    cls_defs = captures.get("cls_def", [])
    cls_names = captures.get("cls_name", [])
    cls_bases = captures.get("cls_base", [])

    # Build a set of base names per class node id
    cls_node_bases: dict[int, list[str]] = {}
    for base_node in cls_bases:
        # Walk up to find the parent class_definition
        parent = base_node.parent
        while parent is not None and parent.type != "class_definition":
            parent = parent.parent
        if parent is not None:
            cls_node_bases.setdefault(parent.id, []).append(
                base_node.text.decode("utf-8", errors="replace")
            )

    cls_pairs = _pair_defs_and_names(cls_defs, cls_names)
    for cls_node, name in cls_pairs:
        bases = cls_node_bases.get(cls_node.id, [])
        cls = ClassDef(
            name=name,
            bases=bases,
            start_line=cls_node.start_point.row + 1,
            end_line=cls_node.end_point.row + 1,
        )
        class_map[name] = cls
        skeleton.classes.append(cls)

    # Process functions
    fn_pairs = _pair_defs_and_names(
        captures.get("fn_def", []),
        captures.get("fn_name", []),
    )

    for fn_node, name in fn_pairs:
        sig = _first_line(fn_node)

        # Check if this function is inside a class
        parent_cls = None
        is_method = False
        parent = fn_node.parent
        while parent is not None:
            if parent.type == "class_definition":
                # Find the class name
                for child in parent.children:
                    if child.type == "identifier":
                        parent_cls = child.text.decode("utf-8", errors="replace")
                        break
                is_method = True
                break
            parent = parent.parent

        func = FunctionDef(
            name=name,
            signature=sig,
            start_line=fn_node.start_point.row + 1,
            end_line=fn_node.end_point.row + 1,
            is_method=is_method,
            parent_class=parent_cls,
        )

        if is_method and parent_cls and parent_cls in class_map:
            class_map[parent_cls].methods.append(func)
        else:
            skeleton.functions.append(func)

    # Process imports
    for node in captures.get("imp_def", []):
        text = node.text.decode("utf-8", errors="replace")
        # "import os" or "import os.path"
        parts = text.split()
        if len(parts) >= 2:
            module = parts[1]
            skeleton.imports.append(ImportDef(module=module, names=[module.split(".")[-1]]))

    for node in captures.get("imp_from_def", []):
        text = node.text.decode("utf-8", errors="replace")
        _parse_python_from_import(skeleton, text)

    # Process constants
    for node in captures.get("const_name", []):
        name = node.text.decode("utf-8", errors="replace")
        # Only include UPPER_CASE or Title_Case names as constants
        if re.match(r"^[A-Z_][A-Z0-9_]*$", name) or re.match(r"^[A-Z][a-zA-Z0-9]*$", name):
            skeleton.constants.append(name)


def _parse_python_from_import(skeleton: FileSkeleton, text: str) -> None:
    """Parse a Python 'from X import Y, Z' statement."""
    # from . import utils
    # from ..base import BaseClass
    # from pathlib import Path, PurePosixPath
    match = re.match(r"from\s+([\w.]+|\.+[\w.]*)\s+import\s+(.+)", text)
    if not match:
        return
    module = match.group(1)
    names_str = match.group(2)
    is_relative = module.startswith(".")
    names = [n.strip().split(" as ")[0].strip() for n in names_str.split(",")]
    names = [n for n in names if n and n != "("]
    skeleton.imports.append(ImportDef(module=module, names=names, is_relative=is_relative))


def _extract_js_ts(
    skeleton: FileSkeleton,
    root: tree_sitter.Node,
    captures: dict[str, list[tree_sitter.Node]],
) -> None:
    """Extract JavaScript/TypeScript structural elements."""
    class_map: dict[str, ClassDef] = {}

    # Classes
    for cls_node in captures.get("cls_def", []):
        name = ""
        bases: list[str] = []
        for child in cls_node.children:
            if child.type in ("identifier", "type_identifier"):
                name = child.text.decode("utf-8", errors="replace")
            if child.type == "class_heritage":
                for hc in child.children:
                    if hc.type in ("identifier", "type_identifier"):
                        bases.append(hc.text.decode("utf-8", errors="replace"))

        if name:
            cls = ClassDef(
                name=name,
                bases=bases,
                start_line=cls_node.start_point.row + 1,
                end_line=cls_node.end_point.row + 1,
            )
            class_map[name] = cls
            skeleton.classes.append(cls)

    # Functions
    for fn_node in captures.get("fn_def", []):
        name = ""
        for name_node in captures.get("fn_name", []):
            if _node_is_child_of(name_node, fn_node):
                name = name_node.text.decode("utf-8", errors="replace")
                break
        if name:
            skeleton.functions.append(FunctionDef(
                name=name,
                signature=_first_line(fn_node),
                start_line=fn_node.start_point.row + 1,
                end_line=fn_node.end_point.row + 1,
            ))

    # Methods
    for method_node in captures.get("method_def", []):
        name = ""
        for name_node in captures.get("method_name", []):
            if _node_is_child_of(name_node, method_node):
                name = name_node.text.decode("utf-8", errors="replace")
                break

        parent_cls = _find_parent_class_name(method_node)
        if name:
            func = FunctionDef(
                name=name,
                signature=_first_line(method_node),
                start_line=method_node.start_point.row + 1,
                end_line=method_node.end_point.row + 1,
                is_method=True,
                parent_class=parent_cls,
            )
            if parent_cls and parent_cls in class_map:
                class_map[parent_cls].methods.append(func)
            else:
                skeleton.functions.append(func)

    # Imports
    for imp_node in captures.get("imp_def", []):
        text = imp_node.text.decode("utf-8", errors="replace")
        _parse_js_import(skeleton, text)

    # Constants
    for node in captures.get("const_name", []):
        name = node.text.decode("utf-8", errors="replace")
        skeleton.constants.append(name)


def _parse_js_import(skeleton: FileSkeleton, text: str) -> None:
    """Parse a JS/TS import statement."""
    # import { foo, bar } from 'module';
    # import baz from 'module';
    # import * as ns from 'module';
    source_match = re.search(r"""from\s+['"]([^'"]+)['"]""", text)
    if not source_match:
        # import 'module'; (side-effect import)
        source_match = re.search(r"""import\s+['"]([^'"]+)['"]""", text)
    if not source_match:
        return

    module = source_match.group(1)
    is_relative = module.startswith(".")

    names: list[str] = []
    # Named imports: { foo, bar }
    named_match = re.search(r"\{([^}]+)\}", text)
    if named_match:
        names = [
            n.strip().split(" as ")[0].strip()
            for n in named_match.group(1).split(",")
            if n.strip()
        ]
    # Default import: import foo from 'bar'
    default_match = re.match(r"import\s+(\w+)\s+from", text)
    if default_match:
        names.insert(0, default_match.group(1))

    if not names:
        names = [module.split("/")[-1]]

    skeleton.imports.append(ImportDef(module=module, names=names, is_relative=is_relative))


def _extract_go(
    skeleton: FileSkeleton,
    root: tree_sitter.Node,
    captures: dict[str, list[tree_sitter.Node]],
) -> None:
    """Extract Go structural elements."""
    # Type declarations (structs / interfaces)
    for cls_node in captures.get("cls_def", []):
        name = ""
        for name_node in captures.get("cls_name", []):
            if _node_is_child_of(name_node, cls_node):
                name = name_node.text.decode("utf-8", errors="replace")
                break
        if name:
            skeleton.classes.append(ClassDef(
                name=name,
                bases=[],
                start_line=cls_node.start_point.row + 1,
                end_line=cls_node.end_point.row + 1,
            ))

    # Functions and methods
    fn_pairs = _pair_defs_and_names(
        captures.get("fn_def", []),
        captures.get("fn_name", []),
    )

    for fn_node, name in fn_pairs:
        is_method = fn_node.type == "method_declaration"
        parent_cls = None
        if is_method:
            # Extract receiver type
            for child in fn_node.children:
                if child.type == "parameter_list":
                    receiver_text = child.text.decode("utf-8", errors="replace")
                    # (*Animal) or (a Animal) or (a *Animal)
                    type_match = re.search(r"\*?(\w+)\s*\)", receiver_text)
                    if type_match:
                        parent_cls = type_match.group(1)
                    break

        func = FunctionDef(
            name=name,
            signature=_first_line(fn_node),
            start_line=fn_node.start_point.row + 1,
            end_line=fn_node.end_point.row + 1,
            is_method=is_method,
            parent_class=parent_cls,
        )

        # Attach method to class if possible
        if is_method and parent_cls:
            attached = False
            for cls in skeleton.classes:
                if cls.name == parent_cls:
                    cls.methods.append(func)
                    attached = True
                    break
            if not attached:
                skeleton.functions.append(func)
        else:
            skeleton.functions.append(func)

    # Imports
    for imp_node in captures.get("imp_def", []):
        text = imp_node.text.decode("utf-8", errors="replace")
        # Extract quoted import paths
        paths = re.findall(r'"([^"]+)"', text)
        for p in paths:
            skeleton.imports.append(ImportDef(
                module=p,
                names=[p.split("/")[-1]],
            ))

    # Constants
    for node in captures.get("const_name", []):
        name = node.text.decode("utf-8", errors="replace")
        skeleton.constants.append(name)


def _extract_java(
    skeleton: FileSkeleton,
    root: tree_sitter.Node,
    captures: dict[str, list[tree_sitter.Node]],
) -> None:
    """Extract Java structural elements."""
    class_map: dict[str, ClassDef] = {}

    # Classes / interfaces / enums
    for cls_node in captures.get("cls_def", []):
        name = ""
        bases: list[str] = []
        for child in cls_node.children:
            if child.type == "identifier":
                name = child.text.decode("utf-8", errors="replace")
            if child.type == "superclass":
                for sc in child.children:
                    if sc.type == "type_identifier":
                        bases.append(sc.text.decode("utf-8", errors="replace"))
            if child.type == "super_interfaces":
                for si in child.children:
                    if si.type == "type_list":
                        for t in si.children:
                            if t.type == "type_identifier":
                                bases.append(t.text.decode("utf-8", errors="replace"))

        if name:
            cls = ClassDef(
                name=name,
                bases=bases,
                start_line=cls_node.start_point.row + 1,
                end_line=cls_node.end_point.row + 1,
            )
            class_map[name] = cls
            skeleton.classes.append(cls)

    # Methods / constructors
    fn_pairs = _pair_defs_and_names(
        captures.get("fn_def", []),
        captures.get("fn_name", []),
    )

    for fn_node, name in fn_pairs:
        parent_cls = _find_parent_class_name_java(fn_node)

        func = FunctionDef(
            name=name,
            signature=_first_line(fn_node),
            start_line=fn_node.start_point.row + 1,
            end_line=fn_node.end_point.row + 1,
            is_method=parent_cls is not None,
            parent_class=parent_cls,
        )
        if parent_cls and parent_cls in class_map:
            class_map[parent_cls].methods.append(func)
        else:
            skeleton.functions.append(func)

    # Imports
    for imp_node in captures.get("imp_def", []):
        text = imp_node.text.decode("utf-8", errors="replace")
        # import java.util.List;
        match = re.match(r"import\s+(?:static\s+)?([\w.]+);?", text)
        if match:
            module = match.group(1)
            skeleton.imports.append(ImportDef(
                module=module,
                names=[module.split(".")[-1]],
            ))

    # Constants (static final fields)
    for node in captures.get("const_name", []):
        name = node.text.decode("utf-8", errors="replace")
        # Check if the modifier sibling contains 'static' and 'final'
        parent = node.parent
        if parent and parent.parent:
            field_decl = parent.parent
            mods_node = field_decl.child_by_field_name("modifiers") if hasattr(
                field_decl, "child_by_field_name"
            ) else None
            if mods_node is None:
                for child in field_decl.children:
                    if child.type == "modifiers":
                        mods_node = child
                        break
            if mods_node:
                mods_text = mods_node.text.decode("utf-8", errors="replace")
                if "static" in mods_text and "final" in mods_text:
                    skeleton.constants.append(name)


def _extract_rust(
    skeleton: FileSkeleton,
    root: tree_sitter.Node,
    captures: dict[str, list[tree_sitter.Node]],
) -> None:
    """Extract Rust structural elements."""
    class_map: dict[str, ClassDef] = {}

    # Structs / enums / traits
    for cls_node in captures.get("cls_def", []):
        name = ""
        for name_node in captures.get("cls_name", []):
            if _node_is_child_of(name_node, cls_node):
                name = name_node.text.decode("utf-8", errors="replace")
                break
        if name:
            cls = ClassDef(
                name=name,
                bases=[],
                start_line=cls_node.start_point.row + 1,
                end_line=cls_node.end_point.row + 1,
            )
            class_map[name] = cls
            skeleton.classes.append(cls)

    # Impl blocks: attach methods to the implemented type
    for impl_node in captures.get("impl_def", []):
        impl_type = None
        for child in impl_node.children:
            if child.type == "type_identifier":
                impl_type = child.text.decode("utf-8", errors="replace")
                break
            if child.type == "generic_type":
                for gc in child.children:
                    if gc.type == "type_identifier":
                        impl_type = gc.text.decode("utf-8", errors="replace")
                        break
                if impl_type:
                    break

        # Find function items inside the impl block
        if impl_type:
            _extract_rust_impl_methods(skeleton, impl_node, impl_type, class_map)

    # Top-level functions (not inside impl)
    fn_pairs = _pair_defs_and_names(
        captures.get("fn_def", []),
        captures.get("fn_name", []),
    )

    for fn_node, name in fn_pairs:
        # Skip functions inside impl blocks (already extracted)
        if _is_inside_impl(fn_node):
            continue
        skeleton.functions.append(FunctionDef(
            name=name,
            signature=_first_line(fn_node),
            start_line=fn_node.start_point.row + 1,
            end_line=fn_node.end_point.row + 1,
        ))

    # Use declarations (imports)
    for imp_node in captures.get("imp_def", []):
        text = imp_node.text.decode("utf-8", errors="replace")
        # use std::collections::HashMap;
        match = re.match(r"use\s+(.+);?", text)
        if match:
            path = match.group(1).rstrip(";").strip()
            is_relative = path.startswith("crate::") or path.startswith("super::")
            names_part = path.split("::")[-1] if "::" in path else path
            # Handle {Name1, Name2} syntax
            if "{" in names_part:
                inner = re.search(r"\{([^}]+)\}", names_part)
                names = [n.strip() for n in inner.group(1).split(",")] if inner else [names_part]
            else:
                names = [names_part]
            skeleton.imports.append(ImportDef(
                module=path,
                names=names,
                is_relative=is_relative,
            ))

    # Constants
    for node in captures.get("const_name", []):
        name = node.text.decode("utf-8", errors="replace")
        skeleton.constants.append(name)


def _extract_rust_impl_methods(
    skeleton: FileSkeleton,
    impl_node: tree_sitter.Node,
    impl_type: str,
    class_map: dict[str, ClassDef],
) -> None:
    """Extract methods from a Rust impl block."""
    for child in impl_node.children:
        if child.type == "declaration_list":
            for item in child.children:
                if item.type == "function_item":
                    name = ""
                    for ic in item.children:
                        if ic.type == "identifier":
                            name = ic.text.decode("utf-8", errors="replace")
                            break
                    if name:
                        func = FunctionDef(
                            name=name,
                            signature=_first_line(item),
                            start_line=item.start_point.row + 1,
                            end_line=item.end_point.row + 1,
                            is_method=True,
                            parent_class=impl_type,
                        )
                        if impl_type in class_map:
                            class_map[impl_type].methods.append(func)
                        else:
                            skeleton.functions.append(func)


def _is_inside_impl(node: tree_sitter.Node) -> bool:
    """Check if a node is inside an impl_item."""
    parent = node.parent
    while parent is not None:
        if parent.type == "impl_item":
            return True
        parent = parent.parent
    return False


def _extract_c_cpp(
    skeleton: FileSkeleton,
    root: tree_sitter.Node,
    captures: dict[str, list[tree_sitter.Node]],
) -> None:
    """Extract C/C++ structural elements."""
    # Functions
    fn_pairs = _pair_defs_and_names(
        captures.get("fn_def", []),
        captures.get("fn_name", []),
    )

    for fn_node, name in fn_pairs:
        skeleton.functions.append(FunctionDef(
            name=name,
            signature=_first_line(fn_node),
            start_line=fn_node.start_point.row + 1,
            end_line=fn_node.end_point.row + 1,
        ))

    # Classes / structs
    for cls_node in captures.get("cls_def", []):
        name = ""
        bases: list[str] = []
        for name_node in captures.get("cls_name", []):
            if _node_is_child_of(name_node, cls_node):
                name = name_node.text.decode("utf-8", errors="replace")
                break

        # C++ base classes
        for child in cls_node.children:
            if child.type == "base_class_clause":
                for bc in child.children:
                    if bc.type == "type_identifier":
                        bases.append(bc.text.decode("utf-8", errors="replace"))

        if name:
            skeleton.classes.append(ClassDef(
                name=name,
                bases=bases,
                start_line=cls_node.start_point.row + 1,
                end_line=cls_node.end_point.row + 1,
            ))

    # Includes (imports)
    for imp_node in captures.get("imp_def", []):
        text = imp_node.text.decode("utf-8", errors="replace")
        # #include <stdio.h> or #include "myheader.h"
        match = re.search(r'#include\s+[<"]([^>"]+)[>"]', text)
        if match:
            module = match.group(1)
            skeleton.imports.append(ImportDef(
                module=module,
                names=[module.split("/")[-1].split(".")[0]],
            ))

    # Namespace definitions (C++ only)
    for ns_node in captures.get("ns_def", []):
        for name_node in captures.get("ns_name", []):
            if _node_is_child_of(name_node, ns_node):
                break

    # Constants (preprocessor defines)
    for node in captures.get("const_name", []):
        name = node.text.decode("utf-8", errors="replace")
        skeleton.constants.append(name)


def _extract_ruby(
    skeleton: FileSkeleton,
    root: tree_sitter.Node,
    captures: dict[str, list[tree_sitter.Node]],
) -> None:
    """Extract Ruby structural elements."""
    class_map: dict[str, ClassDef] = {}

    # Classes / modules
    for cls_node in captures.get("cls_def", []):
        name = ""
        bases: list[str] = []
        for name_node in captures.get("cls_name", []):
            if _node_is_child_of(name_node, cls_node):
                name = name_node.text.decode("utf-8", errors="replace")
                break

        # Superclass: class Foo < Bar
        for child in cls_node.children:
            if child.type == "superclass":
                for sc in child.children:
                    if sc.type in ("constant", "scope_resolution"):
                        bases.append(sc.text.decode("utf-8", errors="replace"))

        if name:
            cls = ClassDef(
                name=name,
                bases=bases,
                start_line=cls_node.start_point.row + 1,
                end_line=cls_node.end_point.row + 1,
            )
            class_map[name] = cls
            skeleton.classes.append(cls)

    # Methods
    fn_pairs = _pair_defs_and_names(
        captures.get("fn_def", []),
        captures.get("fn_name", []),
    )

    for fn_node, name in fn_pairs:
        parent_cls = None
        is_method = False

        parent = fn_node.parent
        while parent is not None:
            if parent.type in ("class", "module"):
                for child in parent.children:
                    if child.type in ("constant", "scope_resolution"):
                        parent_cls = child.text.decode("utf-8", errors="replace")
                        break
                is_method = True
                break
            parent = parent.parent

        func = FunctionDef(
            name=name,
            signature=_first_line(fn_node),
            start_line=fn_node.start_point.row + 1,
            end_line=fn_node.end_point.row + 1,
            is_method=is_method,
            parent_class=parent_cls,
        )
        if is_method and parent_cls and parent_cls in class_map:
            class_map[parent_cls].methods.append(func)
        else:
            skeleton.functions.append(func)

    # Imports (require calls)
    for node in captures.get("imp_source", []):
        text = node.text.decode("utf-8", errors="replace").strip("'\"")
        is_relative = text.startswith("./") or text.startswith("../")
        skeleton.imports.append(ImportDef(
            module=text,
            names=[text.split("/")[-1]],
            is_relative=is_relative,
        ))

    # Constants
    for node in captures.get("const_name", []):
        name = node.text.decode("utf-8", errors="replace")
        skeleton.constants.append(name)


def _extract_php(
    skeleton: FileSkeleton,
    root: tree_sitter.Node,
    captures: dict[str, list[tree_sitter.Node]],
) -> None:
    """Extract PHP structural elements."""
    class_map: dict[str, ClassDef] = {}

    # Classes / interfaces / traits
    for cls_node in captures.get("cls_def", []):
        name = ""
        bases: list[str] = []
        for name_node in captures.get("cls_name", []):
            if _node_is_child_of(name_node, cls_node):
                name = name_node.text.decode("utf-8", errors="replace")
                break

        for child in cls_node.children:
            if child.type == "base_clause":
                for bc in child.children:
                    if bc.type in ("name", "qualified_name"):
                        bases.append(bc.text.decode("utf-8", errors="replace"))
            if child.type == "class_interface_clause":
                for ic in child.children:
                    if ic.type in ("name", "qualified_name"):
                        bases.append(ic.text.decode("utf-8", errors="replace"))

        if name:
            cls = ClassDef(
                name=name,
                bases=bases,
                start_line=cls_node.start_point.row + 1,
                end_line=cls_node.end_point.row + 1,
            )
            class_map[name] = cls
            skeleton.classes.append(cls)

    # Functions / methods
    fn_pairs = _pair_defs_and_names(
        captures.get("fn_def", []),
        captures.get("fn_name", []),
    )

    for fn_node, name in fn_pairs:
        is_method = fn_node.type == "method_declaration"
        parent_cls = None

        if is_method:
            parent = fn_node.parent
            while parent is not None:
                if parent.type in ("class_declaration", "interface_declaration",
                                   "trait_declaration"):
                    for child in parent.children:
                        if child.type == "name":
                            parent_cls = child.text.decode("utf-8", errors="replace")
                            break
                    break
                parent = parent.parent

        func = FunctionDef(
            name=name,
            signature=_first_line(fn_node),
            start_line=fn_node.start_point.row + 1,
            end_line=fn_node.end_point.row + 1,
            is_method=is_method,
            parent_class=parent_cls,
        )
        if is_method and parent_cls and parent_cls in class_map:
            class_map[parent_cls].methods.append(func)
        else:
            skeleton.functions.append(func)

    # Imports (use declarations)
    for imp_node in captures.get("imp_def", []):
        text = imp_node.text.decode("utf-8", errors="replace")
        # use App\Base\Model;
        match = re.match(r"use\s+([\w\\]+)", text)
        if match:
            module = match.group(1)
            skeleton.imports.append(ImportDef(
                module=module,
                names=[module.split("\\")[-1]],
            ))

    # Constants
    for node in captures.get("const_name", []):
        name = node.text.decode("utf-8", errors="replace")
        skeleton.constants.append(name)


def _extract_bash(
    skeleton: FileSkeleton,
    root: tree_sitter.Node,
    captures: dict[str, list[tree_sitter.Node]],
) -> None:
    """Extract Bash structural elements."""
    # Functions
    fn_pairs = _pair_defs_and_names(
        captures.get("fn_def", []),
        captures.get("fn_name", []),
    )

    for fn_node, name in fn_pairs:
        skeleton.functions.append(FunctionDef(
            name=name,
            signature=_first_line(fn_node),
            start_line=fn_node.start_point.row + 1,
            end_line=fn_node.end_point.row + 1,
        ))

    # Source imports
    for node in captures.get("imp_source", []):
        path = node.text.decode("utf-8", errors="replace")
        is_relative = path.startswith("./") or path.startswith("../")
        skeleton.imports.append(ImportDef(
            module=path,
            names=[path.split("/")[-1]],
            is_relative=is_relative,
        ))

    # Constants (variable assignments)
    for node in captures.get("const_name", []):
        name = node.text.decode("utf-8", errors="replace")
        if re.match(r"^[A-Z_][A-Z0-9_]*$", name):
            skeleton.constants.append(name)


# ---------------------------------------------------------------------------
# Tier B extraction (generic, best-effort)
# ---------------------------------------------------------------------------

# Common node types across tree-sitter grammars
_TIER_B_FUNCTION_TYPES: frozenset[str] = frozenset({
    "function_definition", "function_declaration", "method_definition",
    "method_declaration", "function_item", "func_literal",
})

_TIER_B_CLASS_TYPES: frozenset[str] = frozenset({
    "class_definition", "class_declaration", "struct_item",
    "struct_specifier", "interface_declaration", "trait_item",
    "type_declaration", "enum_declaration", "enum_item",
})

_TIER_B_IMPORT_TYPES: frozenset[str] = frozenset({
    "import_statement", "import_declaration", "import_from_statement",
    "use_declaration", "preproc_include", "namespace_use_declaration",
})


def _extract_tier_b(root: tree_sitter.Node, ts_lang_name: str) -> FileSkeleton:
    """Best-effort extraction using generic node types."""
    skeleton = FileSkeleton(path="", language=ts_lang_name, tier="B")

    for child in root.children:
        _walk_tier_b(child, skeleton)

    return skeleton


def _walk_tier_b(node: tree_sitter.Node, skeleton: FileSkeleton) -> None:
    """Recursively walk nodes looking for known structural types."""
    if node.type in _TIER_B_FUNCTION_TYPES:
        name = _get_name_child(node)
        if name:
            skeleton.functions.append(FunctionDef(
                name=name,
                signature=_first_line(node),
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
            ))
        return  # Don't recurse into function bodies

    if node.type in _TIER_B_CLASS_TYPES:
        name = _get_name_child(node)
        if name:
            skeleton.classes.append(ClassDef(
                name=name,
                bases=[],
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
            ))
        # Recurse to find methods inside class body
        for child in node.children:
            _walk_tier_b(child, skeleton)
        return

    if node.type in _TIER_B_IMPORT_TYPES:
        text = node.text.decode("utf-8", errors="replace")
        skeleton.imports.append(ImportDef(
            module=text,
            names=[],
        ))
        return

    # Recurse into other nodes
    for child in node.children:
        _walk_tier_b(child, skeleton)


def _get_name_child(node: tree_sitter.Node) -> str | None:
    """Try to find a name identifier among a node's children."""
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "field_identifier",
                          "name", "property_identifier", "constant"):
            return child.text.decode("utf-8", errors="replace")
    return None


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _node_is_child_of(child: tree_sitter.Node, parent: tree_sitter.Node) -> bool:
    """Check if child is a descendant of parent by byte range containment."""
    return (
        child.start_byte >= parent.start_byte
        and child.end_byte <= parent.end_byte
    )


def _find_name_for_def(
    def_node: tree_sitter.Node,
    name_nodes: list[tree_sitter.Node],
) -> str | None:
    """Find the name node that belongs to a definition node by containment."""
    for name_node in name_nodes:
        if _node_is_child_of(name_node, def_node):
            return name_node.text.decode("utf-8", errors="replace")
    return None


def _pair_defs_and_names(
    def_nodes: list[tree_sitter.Node],
    name_nodes: list[tree_sitter.Node],
) -> list[tuple[tree_sitter.Node, str]]:
    """Pair definition nodes with their name strings using containment matching.

    Returns list of (def_node, name_string) pairs. Definitions without
    a matching name are skipped.
    """
    pairs: list[tuple[tree_sitter.Node, str]] = []
    for def_node in def_nodes:
        name = _find_name_for_def(def_node, name_nodes)
        if name is not None:
            pairs.append((def_node, name))
    return pairs


def _find_parent_class_name(node: tree_sitter.Node) -> str | None:
    """Walk up the tree to find an enclosing class name (JS/TS)."""
    parent = node.parent
    while parent is not None:
        if parent.type == "class_declaration":
            for child in parent.children:
                if child.type in ("identifier", "type_identifier"):
                    return child.text.decode("utf-8", errors="replace")
        if parent.type == "class_body":
            parent = parent.parent
            continue
        parent = parent.parent
    return None


def _find_parent_class_name_java(node: tree_sitter.Node) -> str | None:
    """Walk up the tree to find an enclosing class/interface name (Java)."""
    parent = node.parent
    while parent is not None:
        if parent.type in ("class_declaration", "interface_declaration",
                           "enum_declaration", "class_body"):
            if parent.type == "class_body":
                parent = parent.parent
                continue
            for child in parent.children:
                if child.type == "identifier":
                    return child.text.decode("utf-8", errors="replace")
        parent = parent.parent
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_file(
    path: str,
    content: str,
    language: str | None,
) -> FileSkeleton | None:
    """Parse a single source file and extract its structural skeleton.

    Args:
        path: File path (relative to repo root).
        content: File content as text.
        language: Language name as returned by detect_language() (e.g. "Python").
            If None, returns None.

    Returns:
        FileSkeleton with extracted symbols, or None if the language is unsupported
        or parsing fails.
    """
    if language is None:
        return None

    ts_lang_name = LANGUAGE_MAP.get(language)
    if ts_lang_name is None:
        return None

    # Try to get parser for this language
    try:
        parser = get_parser(ts_lang_name)
    except LookupError:
        logger.debug("unsupported_language", language=language, ts_name=ts_lang_name)
        return None

    # Parse the file
    try:
        tree = parser.parse(content.encode("utf-8"))
    except Exception:
        logger.debug("parse_failed", path=path, language=language)
        return None

    root = tree.root_node
    if root.has_error:
        # Still try to extract what we can -- partial parses are common
        pass

    # Check if we have a Tier A query
    query = _get_compiled_query(ts_lang_name)

    if query is not None:
        tier = "A"
        skeleton = _extract_tier_a(root, ts_lang_name, query)
        logger.debug("parsed_file", path=path, language=language, tier=tier)
    else:
        tier = "B"
        skeleton = _extract_tier_b(root, ts_lang_name)
        logger.debug("parsed_file", path=path, language=language, tier=tier)

    skeleton.path = path
    skeleton.language = language
    return skeleton


def parse_repo(snapshot: RepoSnapshot) -> list[FileSkeleton]:
    """Parse all files in a repo snapshot, extracting structural skeletons.

    Args:
        snapshot: RepoSnapshot containing files to parse.

    Returns:
        List of FileSkeleton objects (one per successfully parsed file).
    """
    skeletons: list[FileSkeleton] = []
    for file_record in snapshot.files:
        skeleton = parse_file(
            path=file_record.path,
            content=file_record.content,
            language=file_record.language,
        )
        if skeleton is not None:
            skeletons.append(skeleton)

    logger.info(
        "parsed_repo",
        repo=snapshot.repo_path,
        total_files=len(snapshot.files),
        parsed=len(skeletons),
    )
    return skeletons
