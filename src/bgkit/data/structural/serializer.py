"""Serialize structural data to text for model consumption.

Three serialization modes:
  1. Skeleton view: function/class signatures without bodies (Aider-style)
  2. Dependency listing: import graph edges touching a file
  3. Module summary: exported API surface of a directory
"""

from __future__ import annotations

from pathlib import PurePosixPath

from bgkit.data.structural.dependency_graph import DependencyGraph
from bgkit.data.structural.parser import ClassDef, FileSkeleton, FunctionDef


def serialize_skeleton(skeleton: FileSkeleton) -> str:
    """Serialize a FileSkeleton into an XML-tagged skeleton view.

    Format:
        <skeleton path="src/foo.py" language="Python">
        class MyClass:
            def method_one(self, arg: int) -> str: ...
            def method_two(self) -> None: ...
        def standalone_func(x, y): ...
        CONSTANT_NAME = ...
        </skeleton>
    """
    lines: list[str] = []
    lines.append(f'<skeleton path="{skeleton.path}" language="{skeleton.language}">')

    # Imports (compact form)
    for imp in skeleton.imports:
        if imp.is_relative:
            lines.append(f"from {imp.module} import {', '.join(imp.names)}")
        elif imp.names and imp.names != [imp.module.split(".")[-1]]:
            lines.append(f"import {imp.module}  # {', '.join(imp.names)}")
        else:
            lines.append(f"import {imp.module}")

    if skeleton.imports and (skeleton.classes or skeleton.functions or skeleton.constants):
        lines.append("")

    # Classes with methods
    for cls in skeleton.classes:
        lines.append(_serialize_class(cls, skeleton.language))

    # Top-level functions
    for func in skeleton.functions:
        lines.append(_serialize_function(func, indent=0))

    # Constants
    for const in skeleton.constants:
        lines.append(f"{const} = ...")

    lines.append("</skeleton>")
    return "\n".join(lines)


def _serialize_class(cls: ClassDef, language: str) -> str:
    """Serialize a class definition with its methods."""
    lines: list[str] = []

    # Class header
    if cls.bases:
        bases_str = ", ".join(cls.bases)
        if language in ("Python", "Ruby"):
            lines.append(f"class {cls.name}({bases_str}):")
        elif language in ("Java", "C#"):
            lines.append(f"class {cls.name} extends {bases_str}:")
        elif language in ("C++",):
            lines.append(f"class {cls.name} : {bases_str}:")
        else:
            lines.append(f"class {cls.name}({bases_str}):")
    else:
        lines.append(f"class {cls.name}:")

    if cls.methods:
        for method in cls.methods:
            lines.append(_serialize_function(method, indent=4))
    else:
        lines.append("    ...")

    return "\n".join(lines)


def _serialize_function(func: FunctionDef, indent: int = 0) -> str:
    """Serialize a function/method signature."""
    prefix = " " * indent
    sig = func.signature.strip()

    # Truncate the signature at the body start
    # For Python: strip everything after the colon at end of def line
    # For other languages: strip everything after the opening brace
    if ":" in sig and not sig.endswith(":"):
        # Python-style: "def foo(x): return x" -> "def foo(x): ..."
        colon_idx = sig.rindex(":")
        sig = sig[: colon_idx + 1] + " ..."
    elif "{" in sig:
        # C-style: "int foo(int x) {" -> "int foo(int x): ..."
        brace_idx = sig.index("{")
        sig = sig[:brace_idx].rstrip() + ": ..."
    else:
        sig = sig + ": ..."

    return f"{prefix}{sig}"


def serialize_dependencies(
    skeleton: FileSkeleton,
    graph: DependencyGraph,
) -> str:
    """Serialize the dependency edges touching a given file.

    Shows both inbound (files that import this file) and outbound
    (files this file imports) edges.

    Format:
        <dependencies path="src/foo.py">
        imports:
          src/bar.py (Bar, baz)
          src/utils.py (helper)
        imported_by:
          src/main.py (Foo)
        </dependencies>
    """
    path = skeleton.path
    lines: list[str] = []
    lines.append(f'<dependencies path="{path}">')

    # Outbound edges (this file imports)
    outbound: list[str] = []
    for edge in graph.edges:
        if edge.source_file == path:
            names_str = ", ".join(edge.imported_names) if edge.imported_names else "*"
            outbound.append(f"  {edge.target_file} ({names_str})")

    if outbound:
        lines.append("imports:")
        lines.extend(outbound)

    # Inbound edges (other files import this file)
    inbound: list[str] = []
    for edge in graph.edges:
        if edge.target_file == path:
            names_str = ", ".join(edge.imported_names) if edge.imported_names else "*"
            inbound.append(f"  {edge.source_file} ({names_str})")

    if inbound:
        lines.append("imported_by:")
        lines.extend(inbound)

    if not outbound and not inbound:
        lines.append("  (no dependencies)")

    lines.append("</dependencies>")
    return "\n".join(lines)


def serialize_module_summary(
    skeletons: list[FileSkeleton],
    module_path: str,
) -> str:
    """Serialize the exported API surface of a directory/module.

    Lists all public symbols from files in the given directory.

    Args:
        skeletons: All file skeletons in the repository.
        module_path: Directory path to summarize (e.g. "src/bgkit/data").

    Format:
        <module path="src/bgkit/data">
        src/bgkit/data/parser.py:
          class FileSkeleton
          def parse_file(path, content, language)
          def parse_repo(snapshot)
        src/bgkit/data/graph.py:
          class DependencyGraph
          def build_dependency_graph(skeletons, file_paths)
        </module>
    """
    # Normalize module_path (remove trailing slash)
    module_path = module_path.rstrip("/")

    # Filter skeletons to those in the module directory
    module_skeletons = [
        s for s in skeletons
        if _is_in_directory(s.path, module_path)
    ]

    # Sort by path for consistent output
    module_skeletons.sort(key=lambda s: s.path)

    lines: list[str] = []
    lines.append(f'<module path="{module_path}">')

    for skeleton in module_skeletons:
        file_symbols: list[str] = []

        # Classes (public only -- skip underscore-prefixed)
        for cls in skeleton.classes:
            if not cls.name.startswith("_"):
                bases_str = f"({', '.join(cls.bases)})" if cls.bases else ""
                file_symbols.append(f"  class {cls.name}{bases_str}")
                for method in cls.methods:
                    if not method.name.startswith("_") or method.name.startswith("__"):
                        file_symbols.append(f"    def {method.name}(...)")

        # Top-level functions (public only)
        for func in skeleton.functions:
            if not func.name.startswith("_"):
                # Extract parameter names from signature
                params = _extract_param_names(func.signature)
                file_symbols.append(f"  def {func.name}({params})")

        # Constants
        for const in skeleton.constants:
            if not const.startswith("_"):
                file_symbols.append(f"  {const} = ...")

        if file_symbols:
            lines.append(f"{skeleton.path}:")
            lines.extend(file_symbols)

    if len(lines) == 1:
        lines.append("  (empty module)")

    lines.append("</module>")
    return "\n".join(lines)


def _is_in_directory(file_path: str, dir_path: str) -> bool:
    """Check if file_path is directly inside dir_path (not in subdirectories)."""
    parent = str(PurePosixPath(file_path).parent)
    return parent == dir_path


def _extract_param_names(signature: str) -> str:
    """Extract parameter names from a function signature string.

    Given "def foo(self, x: int, y: str) -> bool:", returns "self, x, y".
    Given "func greet(name string) string {", returns "name".
    """
    # Find parenthesized content
    paren_start = signature.find("(")
    if paren_start == -1:
        return "..."

    paren_end = signature.find(")", paren_start)
    if paren_end == -1:
        return "..."

    params_str = signature[paren_start + 1 : paren_end].strip()
    if not params_str:
        return ""

    # Split by comma and take first word of each param
    params: list[str] = []
    for param in params_str.split(","):
        param = param.strip()
        if not param:
            continue
        # Take the parameter name (first identifier)
        # Handle "x: int", "int x", "self", "*args", "**kwargs"
        parts = param.split()
        if parts:
            name = parts[0].rstrip(":").lstrip("*")
            if name:
                params.append(name)

    return ", ".join(params)
