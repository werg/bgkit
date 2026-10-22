"""The local repository corpus as seen by the bgkit2 task generators.

Repos live under ``$DATA_DIR/repos/<owner>/<name>`` as BARE clones (full
history, no working tree); the crawler's SQLite state at
``$DATA_DIR/crawl_state.db`` carries the GitHub language + star count. Both
generators (``scripts/bgkit2_gen_repo_qa.py`` and
``scripts/bgkit2_gen_localisation.py``) pick candidates here so they agree on
what "a pure-Python repo with tests" means, and share the helpers that turn a
bare-repo blob into a training context and a Python AST into qualified defs.

A single-file context cannot reveal its own path, and several question forms
answer WITH a path ("which file defines X"), so every context starts with a
one-line ``# file: <relative path>`` header (see :func:`file_header`).
"""

from __future__ import annotations

import ast
import hashlib
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

SOURCE_EXTS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".rb",
    ".cs",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".m",
    ".cc",
}
TEST_DIR_NAMES = {"test", "tests", "testing", "__tests__"}


def split_of(key: str, eval_frac: float) -> str:
    """Deterministic train/eval split by hashing ``key`` (a repo name)."""
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if h < eval_frac else "train"


def file_header(path: str) -> str:
    return f"# file: {path}\n"


def is_test_path(path: str) -> bool:
    parts = Path(path).parts
    if any(p.lower() in TEST_DIR_NAMES for p in parts[:-1]):
        return True
    name = parts[-1].lower()
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


@dataclass(frozen=True)
class RepoCandidate:
    full_name: str
    stars: int
    path: str


def python_repo_candidates(data_dir: str | None = None) -> list[RepoCandidate]:
    """Downloaded repos the crawler tagged ``Python``, most-starred first,
    restricted to those whose bare clone exists on disk."""
    data_dir = data_dir or os.environ["DATA_DIR"]
    db = Path(data_dir, "crawl_state.db")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT full_name, COALESCE(stars, 0) FROM repositories "
            "WHERE language = 'Python' AND status = 'downloaded' "
            "ORDER BY stars DESC, full_name"
        ).fetchall()
    finally:
        con.close()
    out = []
    for full_name, stars in rows:
        p = Path(data_dir, "repos", full_name)
        if (p / "HEAD").exists():
            out.append(RepoCandidate(full_name, int(stars), str(p)))
    return out


@dataclass
class TreeScan:
    head_sha: str
    n_py: int = 0
    n_source: int = 0
    has_tests: bool = False
    truncated: bool = False

    @property
    def py_frac(self) -> float:
        return self.n_py / self.n_source if self.n_source else 0.0


def scan_head_tree(repo, *, max_entries: int = 50_000) -> TreeScan:
    """Count source files by extension at HEAD and look for a tests directory."""
    head = repo.revparse_single("HEAD")
    scan = TreeScan(head_sha=str(head.id))
    stack = [(head.tree, "")]
    seen = 0
    while stack:
        tree, prefix = stack.pop()
        for entry in tree:
            seen += 1
            if seen > max_entries:
                scan.truncated = True
                return scan
            if entry.type_str == "tree":
                if entry.name.lower() in TEST_DIR_NAMES:
                    scan.has_tests = True
                stack.append((repo[entry.id], f"{prefix}{entry.name}/"))
                continue
            ext = Path(entry.name).suffix.lower()
            if ext in SOURCE_EXTS:
                scan.n_source += 1
                if ext == ".py":
                    scan.n_py += 1
    return scan


def repo_eligible(
    scan: TreeScan,
    *,
    min_py: int = 5,
    min_py_frac: float = 0.8,
    require_tests: bool = True,
) -> str | None:
    """None when the repo qualifies, else the reason it is skipped.

    ``require_tests`` exists for the FIX task generator, which has to run a
    test to verify its edit. Repo-QA questions are read off the AST and
    localisation answers come from git history, so neither needs a tests
    directory; demanding one there only shrinks the pool (of the 853 repos not
    already used for training, "no tests directory" rejected about half).
    """
    if scan.n_py < min_py:
        return f"too few python files ({scan.n_py})"
    if scan.py_frac < min_py_frac:
        return f"not pure python (py fraction {scan.py_frac:.2f})"
    if require_tests and not scan.has_tests:
        return "no tests directory"
    return None


def module_map_line(path: str, source: str, *, max_defs: int = 12) -> str | None:
    """One line of a repository map: ``path: Qualified, Names, In.Source.Order``.

    Shared by the `repo_overview` store builder and the agent's `overview` tool
    so the model meets the SAME surface form in training and at inference. They
    diverged when first written (the tool used a bare-name ``ast.walk``, giving
    ``Foo, top, bar, inner`` where the store gave ``Foo, Foo.bar, Foo.bar.inner,
    top``) — different names and different order for the same file. Qualified
    names are also what the localisation task asks for (`path::Class.method`).
    Returns None when the file does not parse.
    """
    try:
        defs = qualified_defs(ast.parse(source))
    except (SyntaxError, ValueError, RecursionError):
        return None
    names = [d.qualname for d in defs][:max_defs]
    return f"{path}: {', '.join(names)}" if names else f"{path}:"


def blob_text(repo, tree, path: str, *, max_bytes: int) -> str | None:
    """UTF-8 text of ``path`` in ``tree`` or None (missing, binary, too big)."""
    try:
        entry = tree[path]
    except KeyError:
        return None
    if entry.type_str != "blob":
        return None
    blob = repo[entry.id]
    if blob.size > max_bytes:
        return None
    data = blob.data
    if b"\x00" in data[:4096]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def tree_python_paths(
    repo, tree, *, include_tests: bool = True, max_files: int | None = None
) -> list[tuple[str, object]] | None:
    """(path, blob id) of every ``.py`` file under ``tree``, sorted by path --
    the WHOLE-REPO file set of the tree stores (``repo_tree_v4``/``v6``):
    directories starting with ``.`` and ``__pycache__`` are skipped, tests are
    included unless ``include_tests`` is False. None when there are more than
    ``max_files`` of them (the stores' "degenerate" repos). Shared by
    ``scripts/bgkit2_build_repo_tree.py`` (HEAD) and the per-commit SWE trees
    (``bgkit2.agent.swe_trees``) so the two file sets cannot drift."""
    out: list[tuple[str, object]] = []
    stack = [("", tree)]
    while stack:
        prefix, node = stack.pop()
        for e in node:
            p = f"{prefix}{e.name}"
            if e.type_str == "tree":
                if not e.name.startswith(".") and e.name != "__pycache__":
                    stack.append((f"{p}/", repo[e.id]))
            elif p.endswith(".py") and (include_tests or not is_test_path(p)):
                out.append((p, e.id))
                if max_files is not None and len(out) > max_files:
                    return None
    return sorted(out, key=lambda x: x[0])


def parseable_python(text: str | None) -> bool:
    """The stores keep a file only when it is non-empty text that ``ast`` parses."""
    if not text:
        return False
    try:
        ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    return True


# ----------------------------------------------------------------------------
# Qualified definitions


@dataclass
class DefInfo:
    qualname: str  # Class.method / outer.inner / name
    name: str
    kind: str  # "function" | "class"
    lineno: int
    end_lineno: int
    depth: int  # 1 = module level, 2 = directly inside a class / function
    node: ast.AST = field(repr=False)

    def contains(self, line: int) -> bool:
        return self.lineno <= line <= self.end_lineno


def qualified_defs(module: ast.Module) -> list[DefInfo]:
    """Every function / class definition with a dotted qualname, in source order."""
    out: list[DefInfo] = []

    def visit(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = ".".join([*stack, child.name])
                kind = "class" if isinstance(child, ast.ClassDef) else "function"
                out.append(
                    DefInfo(
                        q,
                        child.name,
                        kind,
                        child.lineno,
                        child.end_lineno or child.lineno,
                        len(stack) + 1,
                        child,
                    )
                )
                visit(child, [*stack, child.name])
            else:
                visit(child, stack)

    visit(module, [])
    return out


def innermost_function(defs: list[DefInfo], line: int) -> DefInfo | None:
    """The deepest function definition whose span contains ``line``."""
    best: DefInfo | None = None
    for d in defs:
        if d.kind == "function" and d.contains(line) and (best is None or d.depth > best.depth):
            best = d
    return best


def normalised_def_line(node: ast.AST) -> str:
    """The definition line rebuilt from the AST: decorators dropped, argument
    spelling and spacing canonicalised by ``ast.unparse``."""
    if isinstance(node, ast.ClassDef):
        heads = [ast.unparse(b) for b in node.bases] + [
            f"{k.arg}={ast.unparse(k.value)}" for k in node.keywords
        ]
        return f"class {node.name}({', '.join(heads)}):" if heads else f"class {node.name}:"
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){ret}:"
