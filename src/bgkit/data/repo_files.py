"""Shared bare-repo file iteration for Family-B data generators."""

from __future__ import annotations

from pathlib import Path

EXTS = {".py", ".js", ".ts", ".tsx", ".go", ".rs", ".java", ".c", ".h", ".cpp", ".rb"}

#: Structured data files (config, manifests, tables). NOT in :data:`EXTS`,
#: which is deliberately source-only -- pass this explicitly via ``exts``.
#: These are the bulky, machine-shaped blobs an agent actually has to compact
#: (a tool result, a manifest, a query dump), and their line structure carries
#: no meaning, so ``skip_minified`` should be off when reading them.
DATA_EXTS = {".json", ".csv", ".tsv", ".yaml", ".yml", ".toml", ".ini"}

#: Dependency lock files are machine-generated, enormous, and near-identical
#: across repos -- a lookup over one is a lookup over a template.
_LOCK_NAMES = {
    "package-lock.json", "yarn.lock", "composer.lock", "cargo.lock",
    "poetry.lock", "pnpm-lock.yaml", "gemfile.lock", "go.sum",
}

#: Minified / generated source (webpack bundles, ``*.min.js``, UMD builds) has
#: no human-scale lines. A "quote the line" needle on such a file is the whole
#: file (2026-08-23: fileneedle gold answers up to 116K chars; 3% of samples,
#: dominating the token-weighted loss), so these files are rejected at the
#: shared iterator, for every generator built on it.
MAX_LINE_CHARS = 1000
MAX_AVG_LINE_CHARS = 200
_GENERATED_NAME_MARKERS = (".min.", ".bundle.", "-bundle.", ".pack.", ".umd.")


def looks_minified(
    text: str,
    *,
    max_line_chars: int = MAX_LINE_CHARS,
    max_avg_line_chars: int = MAX_AVG_LINE_CHARS,
) -> bool:
    """True for minified / generated text: any line longer than
    ``max_line_chars`` or an average line longer than ``max_avg_line_chars``."""
    lines = text.splitlines()
    if not lines:
        return True
    longest = max(len(ln) for ln in lines)
    if longest > max_line_chars:
        return True
    return len(text) / len(lines) > max_avg_line_chars


def looks_generated_name(path: str) -> bool:
    """Filename markers of build artifacts (``app.min.js``, ``vendor.bundle.js``)
    and dependency lock files."""
    name = Path(path).name.lower()
    if name in _LOCK_NAMES:
        return True
    return any(marker in name for marker in _GENERATED_NAME_MARKERS)


def iter_repo_files(
    repo_path: Path,
    *,
    min_bytes: int,
    max_bytes: int,
    max_files: int,
    max_line_chars: int = MAX_LINE_CHARS,
    max_avg_line_chars: int = MAX_AVG_LINE_CHARS,
    exts: set[str] | None = None,
    skip_minified: bool = True,
):
    """Yield (path, blob_sha8, text) for human-written text source files at
    HEAD of a bare repo. Binary, undecodable, generated-by-name and minified
    blobs (see :func:`looks_minified`) are skipped and do not count toward
    ``max_files``.

    ``exts`` defaults to the source-only :data:`EXTS`; pass :data:`DATA_EXTS`
    to read structured data files instead. ``skip_minified`` should be turned
    off for those: the line-length heuristics encode "a human wrote these
    lines", which is not a property serialized JSON has or needs.
    """
    allowed = EXTS if exts is None else exts
    import pygit2

    try:
        repo = pygit2.Repository(str(repo_path))
        tree = repo.revparse_single("HEAD").tree
    except Exception:
        return
    stack: list[tuple] = [(tree, "")]
    n = 0
    while stack and n < max_files:
        tree, prefix = stack.pop()
        for entry in tree:
            if n >= max_files:
                break
            path = f"{prefix}{entry.name}"
            try:
                obj = repo[entry.id]
            except Exception:
                continue
            if entry.type_str == "tree":
                stack.append((obj, path + "/"))
                continue
            if not isinstance(obj, pygit2.Blob):
                continue  # submodule commits etc.
            if Path(path).suffix.lower() not in allowed or looks_generated_name(path):
                continue
            if not (min_bytes <= obj.size <= max_bytes):
                continue
            data = obj.data
            if b"\x00" in data[:4096]:
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if skip_minified and looks_minified(
                text, max_line_chars=max_line_chars, max_avg_line_chars=max_avg_line_chars
            ):
                continue
            n += 1
            yield path, str(entry.id)[:8], text
