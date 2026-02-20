"""Import resolution and dependency graph construction.

Builds a file-level dependency graph from parsed FileSkeleton objects by
resolving imports against the actual file paths in the repository.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import structlog

from bgkit.data.structural.parser import FileSkeleton

logger = structlog.get_logger()


@dataclass
class DependencyEdge:
    """A directed edge from source_file to target_file."""

    source_file: str
    target_file: str
    imported_names: list[str]


@dataclass
class DependencyGraph:
    """File-level dependency graph for a repository."""

    edges: list[DependencyEdge] = field(default_factory=list)
    adjacency: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))


def build_dependency_graph(
    skeletons: list[FileSkeleton],
    file_paths: list[str],
) -> DependencyGraph:
    """Build dependency graph from parsed skeletons.

    Resolution rules:
    - Relative imports resolve against directory structure
    - Absolute imports match repo-local modules
    - Unresolved (third-party) imports are dropped

    Args:
        skeletons: List of parsed file skeletons.
        file_paths: List of all file paths in the repository (for resolution).

    Returns:
        DependencyGraph with resolved edges.
    """
    graph = DependencyGraph()

    # Build lookup indices for fast resolution
    path_set = frozenset(file_paths)
    # Module name -> file path mapping (for Python-style imports)
    module_index = _build_module_index(file_paths)

    for skeleton in skeletons:
        source_file = skeleton.path
        language = skeleton.language

        for imp in skeleton.imports:
            resolved = _resolve_import(
                source_file=source_file,
                module=imp.module,
                names=imp.names,
                is_relative=imp.is_relative,
                language=language,
                path_set=path_set,
                module_index=module_index,
            )
            if resolved is not None:
                target_file, imported_names = resolved
                edge = DependencyEdge(
                    source_file=source_file,
                    target_file=target_file,
                    imported_names=imported_names,
                )
                graph.edges.append(edge)
                graph.adjacency[source_file].append(target_file)

    # Deduplicate adjacency lists
    for src in graph.adjacency:
        graph.adjacency[src] = sorted(set(graph.adjacency[src]))

    logger.debug(
        "dependency_graph_built",
        files=len(skeletons),
        edges=len(graph.edges),
    )
    return graph


def _build_module_index(file_paths: list[str]) -> dict[str, str]:
    """Build a mapping from dotted module names to file paths.

    For example:
        "src/bgkit/data/parser.py" -> "bgkit.data.parser"
        "utils/__init__.py" -> "utils"

    This supports Python-style absolute import resolution.
    """
    index: dict[str, str] = {}
    for path in file_paths:
        p = PurePosixPath(path)

        # Python modules
        if p.suffix == ".py":
            # Convert path to module name
            parts = list(p.parts)
            # Remove .py extension
            parts[-1] = p.stem

            # Handle __init__.py -> module is the directory
            if parts[-1] == "__init__":
                parts = parts[:-1]

            if not parts:
                continue

            # Try different prefix lengths to find the module
            # e.g., "src/bgkit/data/parser.py" matches "bgkit.data.parser" or "data.parser"
            for start in range(len(parts)):
                module_name = ".".join(parts[start:])
                if module_name and module_name not in index:
                    index[module_name] = path

        # JavaScript/TypeScript modules
        elif p.suffix in (".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"):
            # Without extension
            no_ext = str(p.with_suffix(""))
            index[no_ext] = path
            # Also index with extension
            index[str(p)] = path
            # Handle index files
            if p.stem == "index":
                parent = str(p.parent)
                if parent and parent not in index:
                    index[parent] = path

        # Go packages (by directory)
        elif p.suffix == ".go":
            parent = str(p.parent)
            if parent and parent not in index:
                index[parent] = path

    return index


def _resolve_import(
    source_file: str,
    module: str,
    names: list[str],
    is_relative: bool,
    language: str,
    path_set: frozenset[str],
    module_index: dict[str, str],
) -> tuple[str, list[str]] | None:
    """Resolve a single import to a target file path.

    Returns (target_file, imported_names) or None if unresolved.
    """
    if language == "Python":
        return _resolve_python_import(
            source_file, module, names, is_relative, path_set, module_index
        )
    elif language in ("JavaScript", "TypeScript"):
        return _resolve_js_import(source_file, module, names, path_set, module_index)
    elif language == "Go":
        return _resolve_go_import(module, path_set, module_index)
    elif language in ("C", "C++"):
        return _resolve_c_import(source_file, module, path_set)
    elif language == "Ruby":
        return _resolve_ruby_import(source_file, module, names, path_set, module_index)
    elif language == "PHP":
        return _resolve_php_import(module, names, path_set, module_index)
    elif language == "Rust":
        return _resolve_rust_import(source_file, module, names, path_set, module_index)
    elif language == "Shell":
        return _resolve_shell_import(source_file, module, path_set)
    elif language == "Java":
        return _resolve_java_import(module, names, path_set, module_index)
    else:
        return None


def _resolve_python_import(
    source_file: str,
    module: str,
    names: list[str],
    is_relative: bool,
    path_set: frozenset[str],
    module_index: dict[str, str],
) -> tuple[str, list[str]] | None:
    """Resolve a Python import."""
    if is_relative:
        # Count leading dots
        dots = len(module) - len(module.lstrip("."))
        relative_module = module.lstrip(".")

        source_dir = PurePosixPath(source_file).parent
        # Go up `dots - 1` directories (one dot = current package)
        for _ in range(dots - 1):
            source_dir = source_dir.parent

        if relative_module:
            parts = relative_module.split(".")
            target_dir = source_dir / "/".join(parts)
        else:
            target_dir = source_dir

        # Try as a module file
        candidates = [
            str(target_dir) + ".py",
            str(target_dir / "__init__.py"),
        ]
        for candidate in candidates:
            if candidate in path_set:
                return (candidate, names)
        return None

    # Absolute import
    target = module_index.get(module)
    if target is not None:
        return (target, names)

    # Try partial matches (e.g., "bgkit.data" might match "src/bgkit/data/__init__.py")
    # Already handled by module_index with multiple prefix lengths
    return None


def _resolve_js_import(
    source_file: str,
    module: str,
    names: list[str],
    path_set: frozenset[str],
    module_index: dict[str, str],
) -> tuple[str, list[str]] | None:
    """Resolve a JS/TS import."""
    if module.startswith("."):
        # Relative import
        source_dir = PurePosixPath(source_file).parent
        resolved = source_dir / module

        # Try various extensions
        candidates = [
            str(resolved),
            str(resolved) + ".js",
            str(resolved) + ".ts",
            str(resolved) + ".tsx",
            str(resolved) + ".jsx",
            str(resolved / "index.js"),
            str(resolved / "index.ts"),
            str(resolved / "index.tsx"),
        ]
        for candidate in candidates:
            # Normalize the path (resolve ..)
            normalized = str(PurePosixPath(candidate))
            if normalized in path_set:
                return (normalized, names)
        return None

    # Non-relative: check module_index (could be a repo-local package)
    target = module_index.get(module)
    if target is not None:
        return (target, names)

    return None


def _resolve_go_import(
    module: str,
    path_set: frozenset[str],
    module_index: dict[str, str],
) -> tuple[str, list[str]] | None:
    """Resolve a Go import path to a repo-local file."""
    # Go imports are full paths like "github.com/user/repo/pkg"
    # We only resolve repo-local imports
    # Try the last segment(s) as directory paths
    parts = module.split("/")
    for start in range(len(parts)):
        candidate_dir = "/".join(parts[start:])
        target = module_index.get(candidate_dir)
        if target is not None:
            return (target, [parts[-1]])
    return None


def _resolve_c_import(
    source_file: str,
    module: str,
    path_set: frozenset[str],
) -> tuple[str, list[str]] | None:
    """Resolve a C/C++ include to a repo-local file."""
    # Try relative to source file
    source_dir = PurePosixPath(source_file).parent
    candidate = str(source_dir / module)
    if candidate in path_set:
        return (candidate, [module.split("/")[-1].split(".")[0]])

    # Try from repo root
    if module in path_set:
        return (module, [module.split("/")[-1].split(".")[0]])

    # Try common include directories
    for prefix in ("include", "src", "lib"):
        candidate = f"{prefix}/{module}"
        if candidate in path_set:
            return (candidate, [module.split("/")[-1].split(".")[0]])

    return None


def _resolve_ruby_import(
    source_file: str,
    module: str,
    names: list[str],
    path_set: frozenset[str],
    module_index: dict[str, str],
) -> tuple[str, list[str]] | None:
    """Resolve a Ruby require/require_relative."""
    if module.startswith("./") or module.startswith("../"):
        # Relative require
        source_dir = PurePosixPath(source_file).parent
        resolved = source_dir / module
        candidates = [str(resolved), str(resolved) + ".rb"]
        for candidate in candidates:
            normalized = str(PurePosixPath(candidate))
            if normalized in path_set:
                return (normalized, names)
        return None

    # Absolute require
    candidates = [module + ".rb", f"lib/{module}.rb"]
    for candidate in candidates:
        if candidate in path_set:
            return (candidate, names)

    return None


def _resolve_php_import(
    module: str,
    names: list[str],
    path_set: frozenset[str],
    module_index: dict[str, str],
) -> tuple[str, list[str]] | None:
    """Resolve a PHP use statement."""
    # Convert namespace separators to path
    path = module.replace("\\", "/") + ".php"
    if path in path_set:
        return (path, names)
    # Try src/ prefix
    candidate = f"src/{path}"
    if candidate in path_set:
        return (candidate, names)
    # Try lowercase
    path_lower = path.lower()
    for fp in path_set:
        if fp.lower() == path_lower:
            return (fp, names)
    return None


def _resolve_rust_import(
    source_file: str,
    module: str,
    names: list[str],
    path_set: frozenset[str],
    module_index: dict[str, str],
) -> tuple[str, list[str]] | None:
    """Resolve a Rust use statement."""
    if not (module.startswith("crate::") or module.startswith("super::")):
        return None  # External crate

    if module.startswith("crate::"):
        # crate:: means from the crate root (typically src/)
        remainder = module[len("crate::"):]
        parts = remainder.split("::")
        # Try as src/parts.../last.rs or src/parts.../mod.rs
        path_base = "src/" + "/".join(parts)
        candidates = [path_base + ".rs", path_base + "/mod.rs"]
    else:
        # super:: means parent module
        source_dir = PurePosixPath(source_file).parent
        remainder = module[len("super::"):]
        parts = remainder.split("::") if remainder else []
        base = source_dir.parent
        path_base = str(base / "/".join(parts)) if parts else str(base)
        candidates = [path_base + ".rs", path_base + "/mod.rs"]

    for candidate in candidates:
        if candidate in path_set:
            return (candidate, names)

    return None


def _resolve_shell_import(
    source_file: str,
    module: str,
    path_set: frozenset[str],
) -> tuple[str, list[str]] | None:
    """Resolve a bash source/dot command."""
    if module.startswith("./") or module.startswith("../"):
        source_dir = PurePosixPath(source_file).parent
        resolved = str(source_dir / module)
        normalized = str(PurePosixPath(resolved))
        if normalized in path_set:
            return (normalized, [module.split("/")[-1]])
    elif module in path_set:
        return (module, [module.split("/")[-1]])
    return None


def _resolve_java_import(
    module: str,
    names: list[str],
    path_set: frozenset[str],
    module_index: dict[str, str],
) -> tuple[str, list[str]] | None:
    """Resolve a Java import to a repo-local file."""
    # java.util.List -> java/util/List.java
    path = module.replace(".", "/") + ".java"
    if path in path_set:
        return (path, names)
    # Try src/main/java prefix
    candidate = f"src/main/java/{path}"
    if candidate in path_set:
        return (candidate, names)
    # Try src/ prefix
    candidate = f"src/{path}"
    if candidate in path_set:
        return (candidate, names)
    return None
