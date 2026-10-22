"""Versioned git-history knowledge-base contract.

Schema v2 walks the coherent HEAD first-parent chain, preserves real commit and
parent SHAs, complete unified patches, and rename lineages, and admits a file
state as a target only after exact structured-hunk replay. Periodic base
snapshots make every window independently and boundedly reconstructable.

The tree is a balanced opaque-ID forest, roughly
``root → repo → window → chunk16 → chunk4 → commit → evidence``.
Repository splits and generation fingerprints bind the raw, tree, mmap, and
trajectory artifacts so mixed builds fail before training.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pygit2

from bgkit.data.bgkit_tool_template import TrajectoryTurn
from bgkit.data.browse_tree import BrowseNode, BrowseTree
from bgkit.data.drilldown import (
    DEFAULT_MODE_WEIGHTS,
    DEFAULT_TRUNCATION_MAX_DEPTH,
    DEFAULT_TRUNCATION_MIN_DEPTH,
    DrillTarget,
    build_drilldown_trajectory,
)
from bgkit.data.opaque_ids import bip39_id

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_NAME = "git_commit_repro"
GIT_REPRO_SCHEMA_VERSION = 2
ID_SCHEME_VERSION = 2
DEFAULT_ID_SALT = "bgkit/git_commit_repro/v2"
DEFAULT_RECONSTRUCTION_ANCHOR_INTERVAL = 8
SUPPORTED_CHANGE_TYPES = frozenset({
    "added", "copied", "deleted", "modified", "renamed", "typechange",
})

# File-state-reconstruction prompt templates. The decoder is given an exact
# commit key as well as the human-readable message.  Schema-v2 evidence stores
# the same key + message in the commit's leaves, making navigation a real join
# rather than a semantic guess or a repository-memorisation shortcut.
QUERY_TEMPLATES = (
    "give me the state of {filename} at commit {sha} ({message})",
    "show the full contents of {filename} after commit {sha}; message: {message}",
    "reconstruct {filename} at git commit {sha}, described as: {message}",
    "what does {filename} look like after {sha}? Commit message: {message}",
)

# --- Drill-down redesign (2026-07-03) defaults ---
# 0..this distractor branches (wrong siblings, loss=False) sprinkled per sample.
MAX_DISTRACTORS = 4

# Head-node task query — the sample-specific compression prompt that orients the
# drill-down (distinct from the reconstruction question). Query-navigation phrasing.
HEAD_QUERY_TEMPLATES = (
    "find the reconstruction chain for {filename} through commit {sha} ({message})",
    "locate every required change to {filename} through git commit {sha}: {message}",
)

# Target average commit count per repo. The global token budget B is derived
# from this so that mean(N) ≈ TARGET_AVG_COMMITS across the corpus.
TARGET_AVG_COMMITS = 64

# Positional chunking fan-out (~4-ary). chunk4 groups ~4 commits; chunk16
# groups ~4 chunk4 nodes (so ~16 commits).
CHUNK_FANOUT = 4

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FileChange:
    """One changed file within a commit — the L0-encodable diff leaf.

    For file-state RECONSTRUCTION the leaf still carries the per-(commit, file)
    ``diff_text`` (the retrieved evidence), but a valid TARGET also carries the
    file's full blob content AT THIS COMMIT in ``blob_text`` (the gold the
    decoder reconstructs). ``is_target`` is True only when the blob is decodable
    and within the gold-token cap; the diff leaf exists regardless (so the file
    can still be a navigation / drill step in some OTHER file's history).
    """

    file_idx: int
    path: str
    diff_text: str  # complete unified patch, including hunk coordinates
    n_tokens: int = 0  # diff tokens
    blob_text: str = ""  # full file content at this commit (only stored for targets)
    n_blob_tokens: int = 0  # gold blob token count (0 == deleted / no blob)
    is_target: bool = False  # blob decodable AND ≤ gold cap
    old_path: str = ""
    change_type: str = "modified"
    lineage_id: str = ""
    # Structured hunks are the machine-verifiable form of ``diff_text``.  The
    # text is encoded into the KB; these records let the artifact builder prove
    # that applying the evidence reproduces ``blob_text`` exactly.
    hunks: list[dict[str, Any]] = field(default_factory=list)
    base_blob_text: str = ""
    base_blob_available: bool = False
    is_anchor: bool = False
    reconstruction_valid: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReproCommit:
    """One commit's reproduction record.

    ``window_idx`` tiles a repo's full history into contiguous token-budget-B
    windows (window 0 = oldest B-budget commits). ``ordinal`` is the position
    WITHIN that window (0 == oldest commit of the window).
    """

    repo: str  # "owner/repo"
    sha: str
    parent_sha: str
    ordinal: int  # 0 == oldest commit within this window
    message: str
    timestamp: int
    window_idx: int = 0
    file_changes: list[FileChange] = field(default_factory=list)
    n_diff_tokens: int = 0  # sum over file_changes
    schema_version: int = GIT_REPRO_SCHEMA_VERSION

    # -- id helpers (stable keys shared by mmap / tree / trajectory) --
    #
    # Model-facing ids are opaque hashes.  Positional paths and filenames were a
    # copy shortcut: the decoder could manufacture a child/article id without
    # reading the parent representation.  The sidecar remains the authoritative
    # join between independently-built tree/mmap/trajectory artifacts.

    @property
    def commit_key(self) -> str:
        """Positional ``(repo, window, ordinal)`` LOOKUP key.

        This is *not* a model-facing id — it keys the ``commit_node_ids`` sidecar
        (``commit_key -> path node id``) so the trajectory / mmap builders, which
        know a commit by ``(repo, window, ordinal)``, can find the path id the
        tree builder assigned it.
        """
        return commit_key(self.repo, self.window_idx, self.ordinal)

    def _duplicated_paths(self) -> set[str]:
        """Paths that appear on more than one file-change in this commit.

        Practically always empty (git emits one delta per path per commit), but
        we disambiguate defensively so leaf ids stay unique per commit.
        """
        seen: set[str] = set()
        dup: set[str] = set()
        for fc in self.file_changes:
            if fc.path in seen:
                dup.add(fc.path)
            seen.add(fc.path)
        return dup

    def to_dict(self) -> dict:
        return asdict(self)


def commit_key(repo: str, window_idx: int, ordinal: int) -> str:
    """Positional sidecar lookup key (see :attr:`ReproCommit.commit_key`)."""
    return f"{repo}@w{window_idx:03d}:{ordinal:04d}"


def _opaque_id(namespace: str, key: str, *, salt: str = DEFAULT_ID_SALT) -> str:
    """Return a short, non-positional model-facing id.

    Four BIP-39 words provide a 44-bit space.  Builders additionally reject any
    collision across the concrete artifact, so a collision can never silently
    alias two nodes.
    """
    return bip39_id(
        f"{ID_SCHEME_VERSION}\x00{salt}\x00{namespace}\x00{key}",
        n_words=4,
    )


def sha_for_record(rec: dict) -> str:
    """Return the real git SHA, rejecting legacy surrogate-only records."""
    real = str(rec.get("sha", "")).strip()
    if real:
        return real
    raise ValueError(
        "git_commit_repro record has no real commit SHA; re-extract schema-v2 "
        "artifacts instead of manufacturing an identity from row position"
    )


# ---------------------------------------------------------------------------
# Opaque node-id construction (shared by tree / mmap / trajectory builders)
# ---------------------------------------------------------------------------


def window_node_id(
    repo: str, window_idx: int, *, id_salt: str = DEFAULT_ID_SALT,
) -> str:
    """Opaque id for a repository history window."""
    return _opaque_id("window", f"{repo}\x00{window_idx}", salt=id_salt)


def commit_node_id(commit: ReproCommit, *, id_salt: str = DEFAULT_ID_SALT) -> str:
    """Opaque id for a commit; never depends on its displayed position."""
    return _opaque_id(
        "commit", f"{commit.repo}\x00{commit.sha}", salt=id_salt,
    )


def file_change_leaf_id(
    commit_node_id: str,
    path: str,
    file_idx: int,
    *,
    duplicated: bool = False,
    id_salt: str = DEFAULT_ID_SALT,
) -> str:
    """Opaque file-evidence id scoped to its opaque commit id."""
    discriminator = file_idx if duplicated else 0
    return _opaque_id(
        "file", f"{commit_node_id}\x00{path}\x00{discriminator}", salt=id_salt,
    )


# ---------------------------------------------------------------------------
# Extraction (pygit2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkedFileChange:
    """A lossless textual delta emitted by the git walker."""

    path: str
    old_path: str
    change_type: str
    lineage_id: str
    diff_text: str
    hunks: list[dict[str, Any]]
    base_blob_text: str | None
    blob_text: str | None


def _delta_type(delta: pygit2.DiffDelta) -> str:
    names = {
        pygit2.GIT_DELTA_ADDED: "added",
        pygit2.GIT_DELTA_DELETED: "deleted",
        pygit2.GIT_DELTA_MODIFIED: "modified",
        pygit2.GIT_DELTA_RENAMED: "renamed",
        pygit2.GIT_DELTA_COPIED: "copied",
        pygit2.GIT_DELTA_TYPECHANGE: "typechange",
    }
    return names.get(int(delta.status), f"delta-{int(delta.status)}")


def _structured_hunks(patch: pygit2.Patch) -> list[dict[str, Any]]:
    """Serialize every hunk coordinate and line needed for exact replay."""
    return [
        {
            "old_start": int(h.old_start),
            "old_lines": int(h.old_lines),
            "new_start": int(h.new_start),
            "new_lines": int(h.new_lines),
            "lines": [
                {"origin": str(line.origin), "content": str(line.content)}
                for line in h.lines
                if str(line.origin) in {" ", "+", "-"}
            ],
        }
        for h in patch.hunks
    ]


def _per_file_diff_text(patch: pygit2.Patch) -> str:
    """Return the complete unified patch, including headers and coordinates."""
    try:
        return str(patch.text)
    except (AttributeError, UnicodeDecodeError):
        delta = patch.delta
        parts = [
            f"diff --git a/{delta.old_file.path} b/{delta.new_file.path}",
            f"--- a/{delta.old_file.path}",
            f"+++ b/{delta.new_file.path}",
        ]
        for hunk in patch.hunks:
            parts.append(
                f"@@ -{hunk.old_start},{hunk.old_lines} "
                f"+{hunk.new_start},{hunk.new_lines} @@"
            )
            parts.append("".join(f"{ln.origin}{ln.content}" for ln in hunk.lines))
        return "\n".join(parts)


def _blob_text_at_commit(
    commit: pygit2.Commit | None, path: str,
) -> str | None:
    """Return the file's full decoded content at ``commit``, or ``None``.

    ``None`` means the path does not exist at this commit (a deletion — no gold
    reconstructable) OR the content is binary / undecodable. Either way the
    file-change has no valid reconstruction target, though its diff leaf may
    still exist for navigation in another file's history.
    """
    if commit is None:
        return ""
    try:
        data = commit.tree[path].data
    except (KeyError, AttributeError):
        return None  # deleted at this commit
    try:
        return data.decode("utf-8")
    except (UnicodeDecodeError, AttributeError):
        return None  # binary / undecodable


def apply_structured_patch(base_text: str, hunks: list[dict[str, Any]]) -> str:
    """Apply serialized textual hunks, validating coordinates and context."""
    base = base_text.splitlines(keepends=True)
    output: list[str] = []
    cursor = 0
    for hunk in hunks:
        old_start = int(hunk["old_start"])
        start = 0 if old_start == 0 else old_start - 1
        if start < cursor or start > len(base):
            raise ValueError(
                f"invalid/overlapping hunk start {old_start} for {len(base)} base lines"
            )
        output.extend(base[cursor:start])
        cursor = start
        consumed = 0
        produced = 0
        for line in hunk.get("lines", []):
            origin = str(line["origin"])
            content = str(line["content"])
            if origin in {" ", "-"}:
                if cursor >= len(base) or base[cursor] != content:
                    actual = "<eof>" if cursor >= len(base) else repr(base[cursor])
                    raise ValueError(
                        f"patch context mismatch at line {cursor + 1}: "
                        f"expected {content!r}, got {actual}"
                    )
                if origin == " ":
                    output.append(content)
                    produced += 1
                cursor += 1
                consumed += 1
            elif origin == "+":
                output.append(content)
                produced += 1
            else:
                raise ValueError(f"unsupported patch line origin {origin!r}")
        if consumed != int(hunk["old_lines"]):
            raise ValueError(
                f"hunk consumed {consumed} old lines, expected {hunk['old_lines']}"
            )
        if produced != int(hunk["new_lines"]):
            raise ValueError(
                f"hunk produced {produced} new lines, expected {hunk['new_lines']}"
            )
    output.extend(base[cursor:])
    return "".join(output)


def walk_repo_commits_oldest_first(
    repo_path: str,
    repo_id: str,
    *,
    max_walked: int | None = None,
):
    """Yield a coherent HEAD first-parent history, oldest first.

    ``sha`` is the full commit hex id (the real git sha — used to build the
    opaque, unique commit node id; historically dropped, which collapsed every
    commit of a repo onto one id). ``message`` is the FULL commit message (used
    verbatim in the reconstruction prompt). ``blob_text`` is the file's full
    content at this commit (the gold for file-state reconstruction) or ``None``
    for deletions/binary. Token counting / budgeting / gating is done by the
    caller so this stays tokenizer-free (cheap to reuse in calibration).
    """
    if max_walked is not None and max_walked < 1:
        raise ValueError("max_walked must be >= 1 or None")
    repo = pygit2.Repository(repo_path)
    try:
        head = repo.head.target
    except pygit2.GitError:
        return
    chain: list[pygit2.Commit] = []
    commit = repo[head]
    while isinstance(commit, pygit2.Commit):
        chain.append(commit)
        # Walk from HEAD so a cap keeps the most recent history and also bounds
        # traversal work. The old implementation traversed the entire repo and
        # then retained the *oldest* N commits after reversing the list.
        if max_walked is not None and len(chain) >= max_walked:
            break
        if not commit.parents:
            break
        commit = commit.parents[0]
    chain.reverse()

    lineage_by_path: dict[str, str] = {}
    for commit in chain:
        parent = commit.parents[0] if commit.parents else None
        try:
            diff = (
                repo.diff(parent, commit)
                if parent is not None
                else commit.tree.diff_to_tree(swap=True)
            )
            diff.find_similar()
        except (pygit2.GitError, ValueError, KeyError):
            continue
        files: list[WalkedFileChange] = []
        for patch in diff:
            try:
                if patch.delta.is_binary:
                    continue
                delta = patch.delta
                old_path = str(delta.old_file.path or "")
                path = str(delta.new_file.path or old_path)
                change_type = _delta_type(delta)
                if change_type not in SUPPORTED_CHANGE_TYPES:
                    continue
                if change_type == "deleted":
                    path = old_path

                if change_type == "renamed":
                    lineage_id = lineage_by_path.pop(old_path, "")
                    if not lineage_id:
                        lineage_id = hashlib.sha256(
                            f"{repo_id}\x00{commit.id}\x00{old_path}".encode()
                        ).hexdigest()
                    lineage_by_path[path] = lineage_id
                elif change_type == "copied":
                    lineage_id = hashlib.sha256(
                        f"{repo_id}\x00{commit.id}\x00copy\x00{path}".encode()
                    ).hexdigest()
                    lineage_by_path[path] = lineage_id
                else:
                    lineage_id = lineage_by_path.get(path, "")
                    if not lineage_id:
                        lineage_id = hashlib.sha256(
                            f"{repo_id}\x00{commit.id}\x00{path}".encode()
                        ).hexdigest()
                    if change_type == "deleted":
                        lineage_by_path.pop(path, None)
                    else:
                        lineage_by_path[path] = lineage_id

                diff_text = _per_file_diff_text(patch)
                hunks = _structured_hunks(patch)
            except (pygit2.GitError, ValueError):
                continue
            files.append(WalkedFileChange(
                path=path,
                old_path=old_path,
                change_type=change_type,
                lineage_id=lineage_id,
                diff_text=diff_text,
                hunks=hunks,
                base_blob_text=_blob_text_at_commit(parent, old_path),
                blob_text=(
                    None if change_type == "deleted"
                    else _blob_text_at_commit(commit, path)
                ),
            ))
        if not files:
            continue
        yield (
            str(commit.id),
            str(parent.id) if parent is not None else "",
            commit.message.strip(),
            int(commit.commit_time),
            files,
        )


def require_record_schema(rec: dict[str, Any]) -> None:
    """Reject legacy git-repro records before they contaminate new artifacts."""
    version = int(rec.get("schema_version", 0) or 0)
    if version != GIT_REPRO_SCHEMA_VERSION:
        raise ValueError(
            "git_commit_repro artifact schema mismatch: "
            f"expected {GIT_REPRO_SCHEMA_VERSION}, got {version}. "
            "Re-run scripts/extract_commit_repro.py; schema-v1 records omit "
            "reconstruction anchors, parent SHAs, and lossless hunks."
        )
    sha = str(rec.get("sha", "")).strip()
    if not sha:
        raise ValueError("schema-v2 git_commit_repro record has no real commit SHA")
    if len(sha) not in {40, 64} or any(c not in "0123456789abcdef" for c in sha.lower()):
        raise ValueError(f"schema-v2 git_commit_repro record has invalid SHA {sha!r}")
    if "parent_sha" not in rec:
        raise ValueError("schema-v2 git_commit_repro record has no parent_sha")
    parent_sha = str(rec.get("parent_sha", "")).strip()
    if parent_sha and (
        len(parent_sha) not in {40, 64}
        or any(c not in "0123456789abcdef" for c in parent_sha.lower())
    ):
        raise ValueError(f"schema-v2 record has invalid parent SHA {parent_sha!r}")
    if not str(rec.get("repo", "")).strip():
        raise ValueError("schema-v2 git_commit_repro record has no repository id")
    if int(rec.get("window_idx", -1)) < 0 or int(rec.get("ordinal", -1)) < 0:
        raise ValueError("schema-v2 git_commit_repro window/ordinal must be non-negative")
    file_changes = rec.get("file_changes", [])
    if not isinstance(file_changes, list) or not file_changes:
        raise ValueError("schema-v2 git_commit_repro record has no file changes")
    file_indices: set[int] = set()
    for fc in file_changes:
        required = {
            "file_idx", "path", "old_path", "change_type", "lineage_id",
            "diff_text", "hunks", "base_blob_text", "base_blob_available",
            "blob_text", "is_target", "is_anchor", "reconstruction_valid",
        }
        missing = required - set(fc)
        if missing:
            raise ValueError(
                f"schema-v2 file change is missing required fields: {sorted(missing)}"
            )
        file_idx = int(fc.get("file_idx", -1))
        if file_idx < 0 or file_idx in file_indices:
            raise ValueError("schema-v2 file_idx values must be unique and non-negative")
        file_indices.add(file_idx)
        if not str(fc.get("path", "")) or not str(fc.get("lineage_id", "")):
            raise ValueError("schema-v2 file change has an empty path or lineage")
        if str(fc.get("change_type")) not in SUPPORTED_CHANGE_TYPES:
            raise ValueError("schema-v2 file change has an invalid change_type")
        if not isinstance(fc.get("diff_text"), str):
            raise ValueError("schema-v2 file-change diff_text must be text")
        if not isinstance(fc.get("base_blob_text"), str):
            raise ValueError("schema-v2 file-change base_blob_text must be text")
        if not isinstance(fc.get("blob_text"), str):
            raise ValueError("schema-v2 file-change blob_text must be text")
        if not isinstance(fc.get("hunks"), list):
            raise ValueError("schema-v2 file-change hunks must be a list")
        if bool(fc.get("is_target")) != bool(fc.get("reconstruction_valid")):
            raise ValueError(
                "schema-v2 target and reconstruction_valid flags must agree"
            )
        if bool(fc.get("is_anchor")) and not (
            bool(fc.get("reconstruction_valid"))
            and bool(fc.get("base_blob_available"))
        ):
            raise ValueError("schema-v2 reconstruction anchor has no valid base")


def file_sha256(path: str | Path) -> str:
    """Stream a file fingerprint used to bind independently-built artifacts."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_commit_node_sidecar(
    path: str | Path,
    *,
    expected_source_sha256: str | None = None,
) -> tuple[dict[str, str], str, dict[str, Any]]:
    """Load and validate the versioned tree/mmap/trajectory ID join."""
    payload = json.loads(Path(path).read_text())
    if int(payload.get("schema_version", 0)) != GIT_REPRO_SCHEMA_VERSION:
        raise ValueError(
            "legacy or invalid commit-node sidecar; rebuild the browse tree "
            "with scripts/build_commit_repro_tree.py"
        )
    if int(payload.get("id_scheme_version", 0)) != ID_SCHEME_VERSION:
        raise ValueError("commit-node sidecar uses an unsupported id scheme")
    source_sha = str(payload.get("source_sha256", ""))
    if not source_sha or not str(payload.get("tree_sha256", "")):
        raise ValueError("commit-node sidecar has incomplete provenance hashes")
    id_salt = str(payload.get("id_salt", ""))
    if not id_salt:
        raise ValueError("commit-node sidecar has no ID salt")
    if expected_source_sha256 is not None and source_sha != expected_source_sha256:
        raise ValueError(
            "commit-node sidecar was built from a different raw JSONL; rebuild "
            "tree, mmap, and trajectories together"
        )
    mapping = payload.get("commit_node_ids")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("commit-node sidecar has no commit_node_ids mapping")
    if len(mapping) != len(set(str(value) for value in mapping.values())):
        raise ValueError("commit-node sidecar aliases multiple commits to one node")
    return (
        {str(key): str(value) for key, value in mapping.items()},
        id_salt,
        payload,
    )


def stable_repo_split(
    repo: str,
    *,
    eval_fraction: float = 0.05,
    seed: int = 42,
) -> str:
    """Assign a repository—not a row—to a deterministic train/eval split."""
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError(f"eval_fraction must be in (0, 1), got {eval_fraction}")
    digest = hashlib.sha256(f"{seed}\x00{repo}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "eval" if bucket < eval_fraction else "train"


def prepare_reconstruction_chains(
    commits: list[ReproCommit],
    *,
    anchor_interval: int = DEFAULT_RECONSTRUCTION_ANCHOR_INTERVAL,
    state: dict[str, tuple[str, int]] | None = None,
) -> dict[str, int]:
    """Validate every target and install periodic exact base snapshots.

    A target is retained only when applying its structured patch to the actual
    parent blob exactly reproduces the gold blob.  Within each file lineage, a
    base snapshot is retained at the first target, after any discontinuity
    caused by filtering/deletion/binary data, and at least every
    ``anchor_interval`` changes.  Consequently every target has a bounded,
    deterministic evidence chain and later windows are self-contained.
    """
    if anchor_interval < 1:
        raise ValueError("anchor_interval must be >= 1")

    stats = {
        "targets_checked": 0,
        "targets_valid": 0,
        "targets_rejected": 0,
        "anchors": 0,
    }
    chain_state = state if state is not None else {}
    for commit in sorted(commits, key=lambda c: c.ordinal):
        for fc in commit.file_changes:
            fc.is_anchor = False
            fc.reconstruction_valid = False
            lineage = fc.lineage_id or fc.path
            if not fc.is_target:
                # An unobserved textual state breaks the patch chain.  The next
                # eligible target re-anchors against its real git parent blob.
                chain_state.pop(lineage, None)
                continue

            stats["targets_checked"] += 1
            if not fc.base_blob_available:
                fc.is_target = False
                fc.blob_text = ""
                stats["targets_rejected"] += 1
                chain_state.pop(lineage, None)
                continue
            try:
                reconstructed = apply_structured_patch(fc.base_blob_text, fc.hunks)
            except ValueError:
                reconstructed = None
            if reconstructed is None or reconstructed != fc.blob_text:
                fc.is_target = False
                fc.blob_text = ""
                fc.base_blob_text = ""
                stats["targets_rejected"] += 1
                chain_state.pop(lineage, None)
                continue

            prior = chain_state.get(lineage)
            continuous = prior is not None and prior[0] == fc.base_blob_text
            distance = prior[1] + 1 if continuous else anchor_interval
            if not continuous or distance >= anchor_interval:
                fc.is_anchor = True
                distance = 0
                stats["anchors"] += 1
            else:
                # The previous validated target supplies this transition's
                # base, so the raw artifact does not need another full copy.
                fc.base_blob_text = ""
            fc.reconstruction_valid = True
            chain_state[lineage] = (fc.blob_text, distance)
            stats["targets_valid"] += 1
    return stats


def reconstruction_chain(
    history: list[tuple[int, FileChange]], target_pos: int,
) -> list[tuple[int, FileChange]]:
    """Return the complete validated anchor→target evidence chain."""
    if target_pos < 0 or target_pos >= len(history):
        raise IndexError(target_pos)
    target = history[target_pos][1]
    if not target.is_target or not target.reconstruction_valid:
        raise ValueError("target is not exactly reconstructable")
    anchor_pos = target_pos
    while anchor_pos >= 0 and not history[anchor_pos][1].is_anchor:
        anchor_pos -= 1
    if anchor_pos < 0:
        raise ValueError("reconstructable target has no base-snapshot anchor")
    chain = history[anchor_pos:target_pos + 1]
    if not all(fc.reconstruction_valid for _ordinal, fc in chain):
        raise ValueError("reconstruction chain contains an invalid state transition")
    state = chain[0][1].base_blob_text
    if not chain[0][1].base_blob_available:
        raise ValueError("reconstruction chain anchor has no base snapshot")
    for ordinal, fc in chain:
        try:
            state = apply_structured_patch(state, fc.hunks)
        except ValueError as exc:
            raise ValueError(
                f"reconstruction chain patch failed at ordinal {ordinal}: {exc}"
            ) from exc
        if state != fc.blob_text:
            raise ValueError(
                f"reconstruction chain does not reproduce ordinal {ordinal}"
            )
    return chain


def render_file_change_evidence(commit: ReproCommit, fc: FileChange) -> str:
    """Render the exact, query-addressable evidence encoded into the KB."""
    message = commit.message.strip()
    fields = [
        f"bgkit-git-evidence-v{GIT_REPRO_SCHEMA_VERSION}",
        f"commit-sha: {commit.sha}",
        f"parent-sha: {commit.parent_sha or '<root>'}",
        f"commit-message: {message}",
        f"timestamp: {commit.timestamp}",
        f"change-type: {fc.change_type}",
        f"old-path: {fc.old_path or fc.path}",
        f"path: {fc.path}",
        f"lineage: {fc.lineage_id}",
        f"base-snapshot: {'present' if fc.is_anchor else 'previous-state'}",
    ]
    if fc.is_anchor:
        fields.extend(("--- BEGIN BASE SNAPSHOT ---", fc.base_blob_text,
                       "--- END BASE SNAPSHOT ---"))
    fields.extend(("--- BEGIN UNIFIED PATCH ---", fc.diff_text,
                   "--- END UNIFIED PATCH ---"))
    return "\n".join(fields)


# ---------------------------------------------------------------------------
# Budget calibration
# ---------------------------------------------------------------------------


def diff_token_budget_from_median(median_per_commit_tokens: float) -> int:
    """B = TARGET_AVG_COMMITS * median(per-commit diff tokens).

    The median is the robust per-commit diff size measured on a calibration
    sample. Using the *global* median (rather than per-repo) means a repo with
    larger-than-typical commits keeps fewer than 64 and a repo with small
    commits keeps more, so N varies per repo while the cross-repo average
    lands near ``TARGET_AVG_COMMITS``. The exact B should be re-tuned by
    bisecting on observed mean(N) over a few-hundred-repo calibration set;
    this closed form is the starting point.
    """
    return max(1, round(TARGET_AVG_COMMITS * median_per_commit_tokens))


# ---------------------------------------------------------------------------
# Tree construction (forest of per-repo balanced ~4-ary subtrees)
# ---------------------------------------------------------------------------


def _chunk(seq: list, size: int) -> list[list]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def build_window_subtree_nodes(
    commits: list[ReproCommit],
    *,
    id_salt: str = DEFAULT_ID_SALT,
) -> tuple[list[BrowseNode], dict[int, str], str]:
    """Build the BrowseNodes for ONE ``(repo, window)`` (excluding root + repo).

    Returns ``(nodes, ord_to_node_id, window_node_id)`` where ``nodes`` covers
    the window node, any chunk16 / chunk4 grouping nodes, and one leaf-tag node
    per commit (``articles`` = file-change ids); ``ord_to_node_id`` maps each
    commit's in-window ``ordinal`` to its POSITIONAL PATH leaf-tag node id
    (``{repo}/w000/c16.00/c4.01/cm.03``). The window node's ``parent`` is the
    repo node id. Every id is the node's root-relative chunk path: each level's
    id = its parent's id + one small within-parent index segment, so a drill turn
    emits the (known, context-copied) parent path + one new index — learnable
    positional navigation, no opaque token to copy. Sized proportionally to the
    window's commit count (flat / one chunk level / two chunk levels), so a short
    final window gets a shallower subtree; the path DEPTH therefore varies with
    ``n`` and can only be assigned here (not from ``(repo, sha)`` alone) — hence
    the sidecar carries these ids to the mmap / trajectory builders.
    """
    assert commits, "window must be non-empty"
    repo_id = commits[0].repo
    window_idx = commits[0].window_idx
    window_id = window_node_id(repo_id, window_idx, id_salt=id_salt)
    repo_node = _opaque_id("repo", repo_id, salt=id_salt)
    nodes: list[BrowseNode] = []
    ord_to_node: dict[int, str] = {}

    def commit_leaf(c: ReproCommit, parent: str) -> BrowseNode:
        node_id = commit_node_id(c, id_salt=id_salt)
        ord_to_node[c.ordinal] = node_id
        dup = c._duplicated_paths()
        article_ids = tuple(
            file_change_leaf_id(
                node_id, fc.path, fc.file_idx, duplicated=fc.path in dup,
                id_salt=id_salt,
            )
            for fc in c.file_changes
        )
        return BrowseNode(
            id=node_id, parent=parent, kind="sub-tag",
            size=len(article_ids), children=(), articles=article_ids,
        )

    def size_of(cs: list[ReproCommit]) -> int:
        return sum(len(c.file_changes) for c in cs)

    n = len(commits)
    window_size = size_of(commits)

    def window_node(children: tuple[str, ...]) -> BrowseNode:
        return BrowseNode(
            id=window_id, parent=repo_node, kind="sub-tag", size=window_size,
            children=children, articles=(),
        )

    if n < CHUNK_FANOUT:
        child_ids = [commit_node_id(c, id_salt=id_salt) for c in commits]
        for c in commits:
            nodes.append(commit_leaf(c, window_id))
        nodes.append(window_node(tuple(child_ids)))
        return nodes, ord_to_node, window_id

    # chunk4: group commits into ~4 (consecutive ordinal order). For N > ~16 we
    # add a second level grouping chunk4 nodes into chunk16 nodes.
    c4_groups = _chunk(commits, CHUNK_FANOUT)
    two_levels = len(c4_groups) > CHUNK_FANOUT

    if not two_levels:
        # window → c4.{g} → cm.{i}
        c4_ids: list[str] = []
        for group in c4_groups:
            c4_id = _opaque_id(
                "chunk4",
                f"{repo_id}\x00{window_idx}\x00" + "\x00".join(c.sha for c in group),
                salt=id_salt,
            )
            c4_ids.append(c4_id)
            child_ids = [commit_node_id(c, id_salt=id_salt) for c in group]
            for c in group:
                nodes.append(commit_leaf(c, c4_id))
            nodes.append(BrowseNode(
                id=c4_id, parent=window_id, kind="sub-tag", size=size_of(group),
                children=tuple(child_ids), articles=(),
            ))
        nodes.append(window_node(tuple(c4_ids)))
        return nodes, ord_to_node, window_id

    # window → c16.{a} → c4.{b} → cm.{i}
    c16_groups = _chunk(c4_groups, CHUNK_FANOUT)
    c16_ids: list[str] = []
    for c16_group in c16_groups:
        c16_commits = [c for group in c16_group for c in group]
        c16_id = _opaque_id(
            "chunk16",
            f"{repo_id}\x00{window_idx}\x00"
            + "\x00".join(c.sha for c in c16_commits),
            salt=id_salt,
        )
        c16_ids.append(c16_id)
        c4_ids = []
        for group in c16_group:
            c4_id = _opaque_id(
                "chunk4",
                f"{repo_id}\x00{window_idx}\x00" + "\x00".join(c.sha for c in group),
                salt=id_salt,
            )
            c4_ids.append(c4_id)
            child_ids = [commit_node_id(c, id_salt=id_salt) for c in group]
            for c in group:
                nodes.append(commit_leaf(c, c4_id))
            nodes.append(BrowseNode(
                id=c4_id, parent=c16_id, kind="sub-tag", size=size_of(group),
                children=tuple(child_ids), articles=(),
            ))
        nodes.append(BrowseNode(
            id=c16_id, parent=window_id, kind="sub-tag",
            size=sum(size_of(g) for g in c16_group),
            children=tuple(c4_ids), articles=(),
        ))
    nodes.append(window_node(tuple(c16_ids)))
    return nodes, ord_to_node, window_id


def build_forest(
    repo_commits: dict[str, list[ReproCommit]],
    *,
    id_salt: str = DEFAULT_ID_SALT,
) -> tuple[BrowseTree, dict[str, str]]:
    """Assemble the full ``git_commit_repro`` browse forest under one root.

    Layout: ``root → repo → repo/wK (window) → c16 → c4 → commit``. Each
    ``(repo, window)`` is a self-contained subtree under the repo node.

    ``repo_commits`` maps ``repo_id`` → that repo's commits across ALL windows
    (each carries its ``window_idx`` + in-window ``ordinal``).

    Returns ``(tree, commit_node_ids)`` where ``commit_node_ids`` maps
    ``commit_key(repo, window, ordinal)`` → the commit's leaf-tag node id.
    """
    all_nodes: list[BrowseNode] = []
    repo_ids: list[str] = []
    commit_node_ids: dict[str, str] = {}
    for repo_id, commits in repo_commits.items():
        if not commits:
            continue
        # Group this repo's commits by window, preserving window order.
        windows: dict[int, list[ReproCommit]] = {}
        for c in commits:
            windows.setdefault(c.window_idx, []).append(c)
        window_node_ids: list[str] = []
        repo_size = 0
        for window_idx in sorted(windows):
            wcommits = sorted(windows[window_idx], key=lambda c: c.ordinal)
            sub, ord_to_node, window_id = build_window_subtree_nodes(
                wcommits, id_salt=id_salt,
            )
            all_nodes.extend(sub)
            window_node_ids.append(window_id)
            repo_size += sum(len(c.file_changes) for c in wcommits)
            for ordinal, node_id in ord_to_node.items():
                commit_node_ids[commit_key(repo_id, window_idx, ordinal)] = node_id
        repo_node = _opaque_id("repo", repo_id, salt=id_salt)
        all_nodes.append(BrowseNode(
            id=repo_node, parent="root", kind="sub-tag", size=repo_size,
            children=tuple(window_node_ids), articles=(),
        ))
        repo_ids.append(repo_node)
    total = sum(
        sum(len(c.file_changes) for c in commits)
        for commits in repo_commits.values()
    )
    root = BrowseNode(
        id="root", parent=None, kind="sub-tag", size=total,
        children=tuple(repo_ids), articles=(),
    )
    all_nodes.append(root)
    model_ids = [node.id for node in all_nodes]
    model_ids.extend(article for node in all_nodes for article in node.articles)
    if len(model_ids) != len(set(model_ids)):
        raise ValueError(
            "opaque id collision while building git_commit_repro; choose a new "
            "--id-salt and rebuild all artifacts"
        )
    return BrowseTree.from_nodes(DATASET_NAME, all_nodes), commit_node_ids


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_query(
    filename: str, message: str, sha: str = "", template_idx: int = 0,
) -> str:
    """File-state prompt with an unambiguous key also present in the evidence."""
    tmpl = QUERY_TEMPLATES[template_idx % len(QUERY_TEMPLATES)]
    return tmpl.format(filename=filename, message=message.strip(), sha=sha)


# ---------------------------------------------------------------------------
# Per-file commit index
# ---------------------------------------------------------------------------


def build_per_file_index(
    commits: list[ReproCommit],
) -> dict[str, list[tuple[int, FileChange]]]:
    """Within one window, map stable lineage → oldest-first changes.
    ``(commit_ordinal, FileChange)`` for every commit that touched the file.

    Commits must already be the window's commits in ascending ordinal order.
    """
    index: dict[str, list[tuple[int, FileChange]]] = {}
    for c in sorted(commits, key=lambda x: x.ordinal):
        for fc in c.file_changes:
            index.setdefault(fc.lineage_id or fc.path, []).append((c.ordinal, fc))
    return index


# ---------------------------------------------------------------------------
# Trajectory construction — file-state reconstruction over a shared tree
# ---------------------------------------------------------------------------


def build_head_query(
    filename: str, message: str, sha: str = "", template_idx: int = 0,
) -> str:
    """Head-node task query — the navigation-oriented compression prompt that
    orients the drill-down (distinct from the reconstruction question)."""
    tmpl = HEAD_QUERY_TEMPLATES[template_idx % len(HEAD_QUERY_TEMPLATES)]
    return tmpl.format(filename=filename, message=message.strip(), sha=sha)


def build_file_drilldown_trajectory(
    tree: BrowseTree,
    commit_node_ids: dict[str, str],
    repo: str,
    window_idx: int,
    file_path: str,
    touching: list[tuple[int, FileChange]],
    head_message: str,
    gold_blob: str,
    *,
    ord_to_commit: dict[int, ReproCommit],
    id_salt: str = DEFAULT_ID_SALT,
    head_template_idx: int = 0,
    n_distractors: int = MAX_DISTRACTORS,
    mode_weights: tuple[float, float, float] = DEFAULT_MODE_WEIGHTS,
    truncation_min_depth: int = DEFAULT_TRUNCATION_MIN_DEPTH,
    truncation_max_depth: int = DEFAULT_TRUNCATION_MAX_DEPTH,
    rng=None,
) -> list[TrajectoryTurn] | None:
    """Pure recursive drill-down trajectory for one ``(file, target)`` reconstruction.

    ``touching`` is the chronological ``[(ordinal, FileChange)]`` for the commits
    that touched ``file_path`` up to (and including) the target. Each becomes a
    drill target that retrieves that
    commit's file-change diff. ``commit_node_ids`` (the sidecar) maps each
    commit_key to its PATH node id; the retrieve id is
    ``file_change_leaf_id(path_node_id, file_name, ...)`` — identical string to
    the tree ``articles`` and the mmap ``document_id`` (``ord_to_commit`` supplies
    the per-commit file list for same-path disambiguation). The head is the
    window node, encoded live with the task query. Returns ``None`` if any
    required node id is missing.

    ``mode_weights`` ``(full, no_drill, truncated)`` selects the per-sample drill
    shape (see :func:`bgkit.data.drilldown.build_drilldown_trajectory`);
    ``truncation_{min,max}_depth`` bound the truncated-mode branch depths. The
    git-repro tree levels map to truncation depths as
    ``window(1) → c16(2) → c4(3) → cm(4)``; the file-diff leaf sits below cm and
    is never reached in ``no_drill`` / ``truncated`` mode.
    """
    head_id = window_node_id(repo, window_idx, id_salt=id_salt)
    if head_id not in tree:
        return None
    targets: list[DrillTarget] = []
    for ord_i, fc_i in touching:
        node_id = commit_node_ids.get(commit_key(repo, window_idx, ord_i))
        commit = ord_to_commit.get(ord_i)
        if node_id is None or node_id not in tree or commit is None:
            return None
        # Retrieve id = the commit's PATH id (from the sidecar) + the file name —
        # identical string to the tree article id + the mmap document_id.
        fcid = file_change_leaf_id(
            node_id, fc_i.path, fc_i.file_idx,
            duplicated=fc_i.path in commit._duplicated_paths(),
            id_salt=id_salt,
        )
        targets.append(DrillTarget(leaf_node_id=node_id, retrieve_ids=(fcid,)))
    if not targets:
        return None
    target_commit = ord_to_commit.get(touching[-1][0])
    target_sha = target_commit.sha if target_commit is not None else ""
    task_query = build_head_query(
        file_path, head_message, target_sha, head_template_idx,
    )
    return build_drilldown_trajectory(
        tree, head_id, targets, task_query, gold_blob,
        n_distractors=n_distractors,
        mode_weights=mode_weights,
        truncation_min_depth=truncation_min_depth,
        truncation_max_depth=truncation_max_depth,
        rng=rng,
    )


def summarize_distribution(counts: list[int], label: str = "N") -> dict:
    """Summary stats (mean / median / min / max / quartiles / total / p90)."""
    if not counts:
        return {"n": 0}
    cs = sorted(counts)
    n = len(cs)
    return {
        "n": n,
        f"mean_{label}": round(statistics.mean(cs), 2),
        f"median_{label}": statistics.median(cs),
        f"min_{label}": cs[0],
        f"max_{label}": cs[-1],
        f"p25_{label}": cs[n // 4],
        f"p75_{label}": cs[(3 * n) // 4],
        f"p90_{label}": cs[min(n - 1, (9 * n) // 10)],
        "total": sum(cs),
    }
