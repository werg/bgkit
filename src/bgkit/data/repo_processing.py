"""Git repo loading and file extraction.

Processes git repos on disk: extracts file trees at specific commits,
reads file contents, detects languages, and produces structured file records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath

import pygit2

# Extensions to always skip (binary, generated, vendored, large data)
SKIP_EXTENSIONS: frozenset[str] = frozenset({
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".tiff", ".tif", ".psd", ".ai",
    # Fonts
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    # Audio/Video
    ".mp3", ".mp4", ".wav", ".ogg", ".flac", ".avi", ".mov", ".webm",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    # Compiled / object
    ".pyc", ".pyo", ".class", ".o", ".a", ".so", ".dylib", ".dll",
    ".exe", ".bin", ".wasm",
    # Data / model weights
    ".npy", ".npz", ".h5", ".hdf5", ".parquet", ".arrow", ".feather",
    ".pkl", ".pickle", ".pt", ".pth", ".onnx", ".safetensors",
    ".db", ".sqlite", ".sqlite3",
    # PDF / docs
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    # Misc
    ".min.js", ".min.css", ".map",
    # Test snapshots
    ".snap",
    # Data files
    ".csv", ".tsv", ".srt", ".ipset",
    # 3D / firmware / binary-ish
    ".stl", ".uf2", ".dex",
    # Resource / project files (generated)
    ".resx", ".pbxproj", ".sln",
})

# Filenames to always skip
SKIP_FILENAMES: frozenset[str] = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "go.sum", "Gemfile.lock", "composer.lock",
    "poetry.lock", "uv.lock", "Pipfile.lock",
    ".DS_Store", "Thumbs.db",
    "shrinkwrap.json",
    "Gopkg.lock",
})

# Directories to skip entirely
SKIP_DIRS: frozenset[str] = frozenset({
    "node_modules", ".git", "__pycache__", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "venv", ".venv", "env",
    "vendor", "third_party", "3rdparty",
    "dist", "build", "_build", ".next", ".nuxt",
    ".gradle", ".idea", ".vscode", ".settings",
    "__snapshots__", "coverage", ".cache",
    "target", "Godeps", "site", "external",
})

# Extension -> language mapping (most common)
EXTENSION_LANGUAGES: dict[str, str] = {
    ".py": "Python", ".pyi": "Python",
    ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".jsx": "JavaScript",
    ".java": "Java",
    ".go": "Go",
    ".rs": "Rust",
    ".c": "C", ".h": "C",
    ".cpp": "C++", ".cc": "C++", ".cxx": "C++", ".hpp": "C++", ".hxx": "C++",
    ".cs": "C#",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".kt": "Kotlin", ".kts": "Kotlin",
    ".scala": "Scala", ".sc": "Scala",
    ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell",
    ".lua": "Lua",
    ".r": "R", ".R": "R",
    ".jl": "Julia",
    ".hs": "Haskell",
    ".ml": "OCaml", ".mli": "OCaml",
    ".ex": "Elixir", ".exs": "Elixir",
    ".erl": "Erlang", ".hrl": "Erlang",
    ".clj": "Clojure", ".cljs": "Clojure", ".cljc": "Clojure",
    ".fs": "F#", ".fsx": "F#",
    ".zig": "Zig",
    ".nim": "Nim",
    ".d": "D",
    ".dart": "Dart",
    ".v": "V",
    ".cr": "Crystal",
    ".pl": "Perl", ".pm": "Perl",
    ".el": "Emacs Lisp",
    ".vim": "Vim script",
    ".sql": "SQL",
    ".html": "HTML", ".htm": "HTML",
    ".css": "CSS", ".scss": "CSS", ".sass": "CSS", ".less": "CSS",
    ".md": "Markdown", ".mdx": "Markdown", ".rst": "reStructuredText",
    ".yaml": "YAML", ".yml": "YAML",
    ".toml": "TOML",
    ".json": "JSON", ".jsonl": "JSON",
    ".xml": "XML",
    ".proto": "Protocol Buffers",
    ".tf": "Terraform", ".tfvars": "Terraform",
    ".dockerfile": "Dockerfile",
    ".cmake": "CMake",
    ".gradle": "Gradle",
    ".nix": "Nix",
}

# Maximum file size to read (256 KB) — larger files are typically data, not code
MAX_FILE_SIZE: int = 256 * 1024

# Per-extension size limits (bytes) for types that are often data/generated when large
EXTENSION_SIZE_CAPS: dict[str, int] = {
    ".html": 20 * 1024,
    ".htm": 20 * 1024,
    ".xml": 20 * 1024,
    ".json": 10 * 1024,
    ".txt": 10 * 1024,
    ".css": 20 * 1024,
    ".ipynb": 20 * 1024,
    "": 5 * 1024,  # No extension: usually data/binary
}

# In public/ dirs, only keep actual code files (not static assets)
_PUBLIC_KEEP_EXTS: frozenset[str] = frozenset({
    ".ts", ".tsx", ".js", ".jsx", ".vue", ".svelte",
    ".py", ".rb", ".go", ".rs", ".java", ".php",
})


@dataclass
class FileRecord:
    """A single extracted file with metadata."""

    path: str
    content: str
    size_bytes: int
    language: str | None = None


@dataclass
class RepoSnapshot:
    """All extracted files from a repo at a specific commit."""

    repo_path: str
    commit_sha: str
    files: list[FileRecord] = field(default_factory=list)
    skipped_binary: int = 0
    skipped_large: int = 0
    skipped_pattern: int = 0

    @property
    def total_files(self) -> int:
        return len(self.files)

    @property
    def total_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)


def detect_language(path: str) -> str | None:
    """Detect programming language from file path."""
    name = PurePosixPath(path).name.lower()

    # Special filenames
    if name == "dockerfile" or name.startswith("dockerfile."):
        return "Dockerfile"
    if name == "makefile" or name == "gnumakefile":
        return "Makefile"
    if name == "cmakelists.txt":
        return "CMake"
    if name == "rakefile":
        return "Ruby"
    if name == "gemfile":
        return "Ruby"
    if name == "justfile":
        return "Just"
    if name == "vagrantfile":
        return "Ruby"

    suffix = PurePosixPath(path).suffix.lower()
    return EXTENSION_LANGUAGES.get(suffix)


def _should_skip_path(path: str) -> bool:
    """Check if a file path should be skipped."""
    parts = PurePosixPath(path).parts
    name = parts[-1] if parts else ""

    # Skip by directory
    if any(part in SKIP_DIRS for part in parts[:-1]):
        return True

    # Skip by filename
    if name in SKIP_FILENAMES:
        return True

    # Skip by extension
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in SKIP_EXTENSIONS:
        return True

    # Skip protobuf generated bindings (compound extensions — .suffix only returns last part)
    if name.endswith((".pb.go", ".pb.rs", "_pb2.py")):
        return True

    # Skip files with generated-code suffixes
    if name.endswith((
        "_generated.go", "_generated.ts", "_generated.js",
        ".generated.cs", ".designer.cs",
    )):
        return True
    if "swagger_doc_generated" in name:
        return True

    # In public/ dirs, only keep actual code files (not static assets like HTML/CSS)
    return "public" in parts[:-1] and suffix not in _PUBLIC_KEEP_EXTS


def _walk_tree(
    repo: pygit2.Repository,
    tree: pygit2.Tree,
    prefix: str = "",
) -> list[tuple[str, pygit2.Oid, int]]:
    """Recursively walk a git tree, returning (path, blob_oid, size) tuples.

    Skips directories in SKIP_DIRS early to avoid descending into them.
    """
    entries = []
    for entry in tree:
        path = f"{prefix}/{entry.name}" if prefix else entry.name

        if entry.type_str == "tree":
            if entry.name in SKIP_DIRS:
                continue
            subtree = repo.get(entry.id)
            entries.extend(_walk_tree(repo, subtree, path))
        elif entry.type_str == "blob":
            blob = repo.get(entry.id)
            entries.append((path, entry.id, blob.size))

    return entries


def load_repo_files(
    repo_path: str,
    commit_sha: str | None = None,
    max_file_size: int = MAX_FILE_SIZE,
) -> dict[str, str]:
    """Load all text files from a git repo at a specific commit.

    Args:
        repo_path: Path to the git repo (working dir with .git/).
        commit_sha: Git commit SHA to read from. If None, uses HEAD.

    Returns:
        Dict mapping file paths to file contents (text files only).
    """
    snapshot = extract_repo_snapshot(repo_path, commit_sha, max_file_size)
    return {f.path: f.content for f in snapshot.files}


def extract_repo_snapshot(
    repo_path: str,
    commit_sha: str | None = None,
    max_file_size: int = MAX_FILE_SIZE,
) -> RepoSnapshot:
    """Extract all text files from a repo at a specific commit, with metadata.

    Args:
        repo_path: Path to the git repo.
        commit_sha: Git commit SHA. If None, uses HEAD.
        max_file_size: Skip files larger than this (bytes).

    Returns:
        RepoSnapshot with file records and skip counts.
    """
    repo = pygit2.Repository(repo_path)

    if commit_sha is None:
        commit = repo.head.peel(pygit2.Commit)
    else:
        commit = repo.revparse_single(commit_sha).peel(pygit2.Commit)

    snapshot = RepoSnapshot(
        repo_path=repo_path,
        commit_sha=str(commit.id),
    )

    entries = _walk_tree(repo, commit.tree)

    for path, blob_oid, size in entries:
        # Skip by path pattern
        if _should_skip_path(path):
            snapshot.skipped_pattern += 1
            continue

        # Skip large files
        if size > max_file_size:
            snapshot.skipped_large += 1
            continue

        # Per-extension size cap (e.g., large HTML/JSON are usually generated/data)
        ext = PurePosixPath(path).suffix.lower()
        ext_cap = EXTENSION_SIZE_CAPS.get(ext)
        if ext_cap is not None and size > ext_cap:
            snapshot.skipped_large += 1
            continue

        blob = repo.get(blob_oid)

        # Skip binary files
        if blob.is_binary:
            snapshot.skipped_binary += 1
            continue

        # Decode content
        try:
            content = blob.data.decode("utf-8")
        except UnicodeDecodeError:
            snapshot.skipped_binary += 1
            continue

        snapshot.files.append(FileRecord(
            path=path,
            content=content,
            size_bytes=size,
            language=detect_language(path),
        ))

    return snapshot
