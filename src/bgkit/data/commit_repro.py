"""Git-commit-reproduction Phase 2 dataset (``git_commit_repro``).

This module is the shared library behind the four ``scripts/*_commit_repro*``
build stages. It replaces the flat ``git_qa`` / ``git_history`` task with a
*multi-level commit-diff reproduction* task:

- **Per repo** we walk the OLDEST commits first (chronological ascending from
  the root commit) and take commits until the cumulative diff-token count
  reaches a per-repo token budget ``B``. ``B`` is chosen globally so the
  AVERAGE commit count across repos ≈ 64 (commits vary in size, so the actual
  count ``N`` varies per repo).

- **Leaf content** is the *per-file diff patch* — one mmap document per changed
  file per commit, L0-encodable exactly like a Phase 2 article.

- **Tree** (one balanced ~4-ary subtree per repo, all under a shared synthetic
  ``root`` — a forest): ``root → repo → chunk16 → chunk4 → commit``. ``commit``
  is a BrowseTree *leaf-tag* whose ``articles`` list holds that commit's
  file-change document ids. The chunk levels are positional (commit order),
  not semantic. Proportionally sized to ``N``:

  * ``N < 4``    → flat: ``root → repo → commit``
  * ``4 ≤ N ≤ 16`` → one chunk level: ``root → repo → chunk4 → commit``
  * ``N > 16``   → two chunk levels: ``root → repo → chunk16 → chunk4 → commit``

- **Task / trajectory**: reproduce the commit's full diff. The *query* is the
  commit message wrapped in a prompt (the message is NOT stored in the tree).
  ``gold_answer`` is the full serialized diff. The trajectory walks
  ``browse(repo) → browse(chunk16) → browse(chunk4) → bgkit([commit], query)
  → answer(diff)`` over the SHARED per-repo tree; each commit gets its own
  trajectory keyed by its message-query.

The shared tree means: encode the repo's tree once, reproduce each commit by
its own query.
"""

from __future__ import annotations

import hashlib
import statistics
from dataclasses import asdict, dataclass, field

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_NAME = "git_commit_repro"

# File-state-reconstruction prompt templates. The decoder is given a filename
# + the FULL target-commit message and must emit the file's full content as it
# stands AFTER that commit, reconstructed from the file's diff history (and
# imputed where history is partial). The commit messages / filenames are the
# query — never part of the encodable tree content (the tree holds only diffs).
QUERY_TEMPLATES = (
    "give me the state of {filename} after applying the commit with this "
    "message: {message}",
    "show the full contents of {filename} as of the commit described by: "
    "{message}",
    "reconstruct {filename} as it stands after the change with commit message: "
    "{message}",
    "what does {filename} look like after the commit whose message is: "
    "{message}",
)

# Reconstruction-walk defaults (tunable via scripts/args).
MAX_PRECEDING_COMMITS = 6  # K — preceding file-touching commits to walk (legacy browse walk)
DRILL_PROB = 0.5  # P(bgkit-drill the target file's diff at a PRECEDING commit) (legacy)
GOLD_TOKEN_CAP = 8192  # a (file, commit) is a target only if its blob ≤ this

# --- Drill-down redesign (2026-07-03) defaults ---
# Cap on the number of touching-diff drill targets per sample (a file is touched
# by many commits; the drill-down finds up to this many, most-recent first).
MAX_TOUCHING_DIFFS = 8
# 0..this distractor branches (wrong siblings, loss=False) sprinkled per sample.
MAX_DISTRACTORS = 4

# Head-node task query — the sample-specific compression prompt that orients the
# drill-down (distinct from the reconstruction question). Query-navigation phrasing.
HEAD_QUERY_TEMPLATES = (
    "find the diffs that touched {filename} up to the commit with this message: "
    "{message}",
    "locate every change to {filename} through the commit described by: {message}",
)

# Target average commit count per repo. The global token budget B is derived
# from this so that mean(N) ≈ TARGET_AVG_COMMITS across the corpus.
TARGET_AVG_COMMITS = 64

# Positional chunking fan-out (~4-ary). chunk4 groups ~4 commits; chunk16
# groups ~4 chunk4 nodes (so ~16 commits).
CHUNK_FANOUT = 4

# pygit2 sort: oldest-ancestor first.
try:  # pygit2 >= 1.15 exposes enums
    from pygit2.enums import SortMode as _SortMode

    _SORT_OLDEST_FIRST = _SortMode.TOPOLOGICAL | _SortMode.REVERSE
except Exception:  # pragma: no cover - very old pygit2
    _SORT_OLDEST_FIRST = pygit2.GIT_SORT_TOPOLOGICAL | pygit2.GIT_SORT_REVERSE


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
    diff_text: str  # "--- {path}\n{hunks...}"
    n_tokens: int = 0  # diff tokens
    blob_text: str = ""  # full file content at this commit (only stored for targets)
    n_blob_tokens: int = 0  # gold blob token count (0 == deleted / no blob)
    is_target: bool = False  # blob decodable AND ≤ gold cap

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
    ordinal: int  # 0 == oldest commit within this window
    message: str
    timestamp: int
    window_idx: int = 0
    file_changes: list[FileChange] = field(default_factory=list)
    n_diff_tokens: int = 0  # sum over file_changes

    # -- id helpers (stable keys shared by mmap / tree / trajectory) --
    #
    # POSITIONAL-PATH ids (2026-07): interior tree node ids are the node's
    # root-relative chunk path (``{repo}/w000/c16.00/c4.01/cm.03``), so a drill
    # turn emits the known parent path + one new small-index segment (learnable
    # navigation) instead of copying an arbitrary opaque token. Leaf (file-change)
    # ids are the file PATH scoped by the commit PATH id. The path components
    # depend on the commit's position WITHIN its window's chunking, which only the
    # tree builder knows — so the commit/leaf ids are produced by
    # :func:`build_window_subtree_nodes` and carried to the mmap / trajectory
    # builders via the ``commit_node_ids`` sidecar (``commit_key -> path id``).
    # There is no pure ``(repo, sha) -> id`` function any more.

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


def surrogate_sha(
    repo: str, window_idx: int, ordinal: int, message: str, timestamp: int,
) -> str:
    """Stable UNIQUE per-commit hash used as the commit's ``sha`` for id
    derivation.

    The upstream extractor never recorded the real git sha (``sha=""`` hardcoded
    in ``extract_commit_repro.py``), so ``commit_node_id = bip39('cm|'+sha)``
    collapsed EVERY commit of a repo onto one id — 99.3% of repos had all commits
    map to a single node, destroying commit-level navigation. This surrogate
    restores the intended property: a stable, unique, content-linked hash per
    commit (the ``(repo, window, ordinal)`` positional key guarantees
    uniqueness; the message + timestamp fold in commit content). ``bip39_id``
    then maps it to opaque, tokenizer-friendly words exactly as designed —
    unguessable from the ordinal because the model cannot invert sha256.

    NOTE: this is not the real git sha (recovering that needs a full
    re-extraction / re-tokenization). It is functionally equivalent for the id
    scheme: unique + stable + opaque. A later real-sha re-extraction and/or
    per-epoch id re-salting (anti-memorization) can supersede it.
    """
    payload = f"{repo}\x00{window_idx}\x00{ordinal}\x00{message}\x00{timestamp}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha_for_record(rec: dict) -> str:
    """Compute :func:`surrogate_sha` from a raw JSONL commit record.

    Single source of truth so the tree / mmap / trajectory build stages derive
    BIT-IDENTICAL commit shas (and therefore identical node ids) from the same
    record — a divergence here would silently break the mmap document_id ↔
    trajectory file_change_id join.

    Prefers a REAL git sha when the record carries one (a future re-extraction
    that populates ``sha``); falls back to :func:`surrogate_sha` for the current
    ``sha=""`` corpus.
    """
    real = str(rec.get("sha", "")).strip()
    if real:
        return real
    return surrogate_sha(
        repo=str(rec["repo"]),
        window_idx=int(rec.get("window_idx", 0)),
        ordinal=int(rec["ordinal"]),
        message=str(rec.get("message", "")),
        timestamp=int(rec.get("timestamp", 0)),
    )


# ---------------------------------------------------------------------------
# Opaque node-id construction (shared by tree / mmap / trajectory builders)
# ---------------------------------------------------------------------------


def window_node_id(repo: str, window_idx: int) -> str:
    """Positional-path id for a ``(repo, window)`` node: ``f"{repo}/w{window:03d}"``.

    The drill-down head, and the root of every commit's chunk path. Path-based
    (was opaque BIP-39): the id IS the node's root-relative address, so a drill
    turn emits the parent path + one new segment — learnable positional
    navigation. This is a pure function of ``(repo, window)``, so the tree
    builder and the trajectory head compute the identical string; the deeper
    (c16 / c4 / commit) path ids need the window's chunking and are assigned by
    :func:`build_window_subtree_nodes` (carried via the sidecar).
    """
    return f"{repo}/w{window_idx:03d}"


def file_change_leaf_id(
    commit_node_id: str, path: str, file_idx: int, *, duplicated: bool = False,
) -> str:
    """Leaf (file-change) id: the file ``path`` scoped by the commit PATH id.

    ``commit_node_id`` is now the commit's positional chunk path, so the leaf id
    is ``{repo}/w000/c16.00/c4.01/cm.03/{path}`` — a globally-unique address whose
    final component is the file name the query already names. ``duplicated`` (a
    commit touching two identical paths) appends a minimal ``#f{file_idx:03d}``
    discriminator; otherwise the bare path is used.
    """
    base = f"{commit_node_id}/{path}"
    return f"{base}#f{file_idx:03d}" if duplicated else base


# ---------------------------------------------------------------------------
# Extraction (pygit2)
# ---------------------------------------------------------------------------


def _per_file_diff_text(patch: pygit2.Patch) -> tuple[str, str]:
    """Return ``(path, "--- {path}\\n{hunks}")`` for one file patch."""
    path = patch.delta.new_file.path
    parts = [f"--- {path}"]
    for hunk in patch.hunks:
        parts.append("".join(f"{ln.origin}{ln.content}" for ln in hunk.lines))
    return path, "\n".join(parts)


def _blob_text_at_commit(
    commit: pygit2.Commit, path: str,
) -> str | None:
    """Return the file's full decoded content at ``commit``, or ``None``.

    ``None`` means the path does not exist at this commit (a deletion — no gold
    reconstructable) OR the content is binary / undecodable. Either way the
    file-change has no valid reconstruction target, though its diff leaf may
    still exist for navigation in another file's history.
    """
    try:
        data = commit.tree[path].data
    except (KeyError, AttributeError):
        return None  # deleted at this commit
    try:
        return data.decode("utf-8")
    except (UnicodeDecodeError, AttributeError):
        return None  # binary / undecodable


def walk_repo_commits_oldest_first(
    repo_path: str,
    repo_id: str,
    *,
    exclude_merges: bool = True,
    max_walked: int | None = None,
):
    """Yield ``(sha, message, timestamp, [(path, diff_text, blob_text)])`` oldest-first.

    ``sha`` is the full commit hex id (the real git sha — used to build the
    opaque, unique commit node id; historically dropped, which collapsed every
    commit of a repo onto one id). ``message`` is the FULL commit message (used
    verbatim in the reconstruction prompt). ``blob_text`` is the file's full
    content at this commit (the gold for file-state reconstruction) or ``None``
    for deletions/binary. Token counting / budgeting / gating is done by the
    caller so this stays tokenizer-free (cheap to reuse in calibration).
    """
    repo = pygit2.Repository(repo_path)
    try:
        head = repo.head.target
    except pygit2.GitError:
        return
    walked = 0
    for commit in repo.walk(head, _SORT_OLDEST_FIRST):
        walked += 1
        if max_walked is not None and walked > max_walked:
            break
        if exclude_merges and len(commit.parents) > 1:
            continue
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
        files: list[tuple[str, str, str | None]] = []
        for patch in diff:
            try:
                # Skip binary files (pygit2 yields no text hunks) and
                # content-less changes (pure rename / mode-only): both would
                # otherwise become trivial header-only "--- path" leaves that
                # pollute the reproduction targets with unreproducible noise.
                if patch.delta.is_binary or not patch.hunks:
                    continue
                path, diff_text = _per_file_diff_text(patch)
            except (pygit2.GitError, ValueError):
                continue
            blob_text = _blob_text_at_commit(commit, path)
            files.append((path, diff_text, blob_text))
        if not files:
            continue
        yield str(commit.id), commit.message.strip(), int(commit.commit_time), files


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
    window_id = window_node_id(repo_id, window_idx)
    nodes: list[BrowseNode] = []
    ord_to_node: dict[int, str] = {}

    def commit_leaf(c: ReproCommit, parent: str, node_id: str) -> BrowseNode:
        ord_to_node[c.ordinal] = node_id
        dup = c._duplicated_paths()
        article_ids = tuple(
            file_change_leaf_id(
                node_id, fc.path, fc.file_idx, duplicated=fc.path in dup,
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
            id=window_id, parent=repo_id, kind="sub-tag", size=window_size,
            children=children, articles=(),
        )

    if n < CHUNK_FANOUT:
        # Flat: commits directly under the window node → {window}/cm.{i:02d}.
        child_ids = [f"{window_id}/cm.{i:02d}" for i in range(n)]
        for i, c in enumerate(commits):
            nodes.append(commit_leaf(c, window_id, child_ids[i]))
        nodes.append(window_node(tuple(child_ids)))
        return nodes, ord_to_node, window_id

    # chunk4: group commits into ~4 (consecutive ordinal order). For N > ~16 we
    # add a second level grouping chunk4 nodes into chunk16 nodes.
    c4_groups = _chunk(commits, CHUNK_FANOUT)
    two_levels = len(c4_groups) > CHUNK_FANOUT

    if not two_levels:
        # window → c4.{g} → cm.{i}
        c4_ids: list[str] = []
        for g, group in enumerate(c4_groups):
            c4_id = f"{window_id}/c4.{g:02d}"
            c4_ids.append(c4_id)
            child_ids = [f"{c4_id}/cm.{i:02d}" for i in range(len(group))]
            for i, c in enumerate(group):
                nodes.append(commit_leaf(c, c4_id, child_ids[i]))
            nodes.append(BrowseNode(
                id=c4_id, parent=window_id, kind="sub-tag", size=size_of(group),
                children=tuple(child_ids), articles=(),
            ))
        nodes.append(window_node(tuple(c4_ids)))
        return nodes, ord_to_node, window_id

    # window → c16.{a} → c4.{b} → cm.{i}
    c16_groups = _chunk(c4_groups, CHUNK_FANOUT)
    c16_ids: list[str] = []
    for a, c16_group in enumerate(c16_groups):
        c16_id = f"{window_id}/c16.{a:02d}"
        c16_ids.append(c16_id)
        c4_ids = []
        for b, group in enumerate(c16_group):
            c4_id = f"{c16_id}/c4.{b:02d}"
            c4_ids.append(c4_id)
            child_ids = [f"{c4_id}/cm.{i:02d}" for i in range(len(group))]
            for i, c in enumerate(group):
                nodes.append(commit_leaf(c, c4_id, child_ids[i]))
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
            sub, ord_to_node, window_id = build_window_subtree_nodes(wcommits)
            all_nodes.extend(sub)
            window_node_ids.append(window_id)
            repo_size += sum(len(c.file_changes) for c in wcommits)
            for ordinal, node_id in ord_to_node.items():
                commit_node_ids[commit_key(repo_id, window_idx, ordinal)] = node_id
        all_nodes.append(BrowseNode(
            id=repo_id, parent="root", kind="sub-tag", size=repo_size,
            children=tuple(window_node_ids), articles=(),
        ))
        repo_ids.append(repo_id)
    total = sum(
        sum(len(c.file_changes) for c in commits)
        for commits in repo_commits.values()
    )
    root = BrowseNode(
        id="root", parent=None, kind="sub-tag", size=total,
        children=tuple(repo_ids), articles=(),
    )
    all_nodes.append(root)
    return BrowseTree.from_nodes(DATASET_NAME, all_nodes), commit_node_ids


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_query(filename: str, message: str, template_idx: int = 0) -> str:
    """File-state-reconstruction prompt: full filename + FULL commit message."""
    tmpl = QUERY_TEMPLATES[template_idx % len(QUERY_TEMPLATES)]
    return tmpl.format(filename=filename, message=message.strip())


# ---------------------------------------------------------------------------
# Per-file commit index
# ---------------------------------------------------------------------------


def build_per_file_index(
    commits: list[ReproCommit],
) -> dict[str, list[tuple[int, FileChange]]]:
    """Within ONE (repo, window), map ``file_path`` → oldest-first list of
    ``(commit_ordinal, FileChange)`` for every commit that touched the file.

    Commits must already be the window's commits in ascending ordinal order.
    """
    index: dict[str, list[tuple[int, FileChange]]] = {}
    for c in sorted(commits, key=lambda x: x.ordinal):
        for fc in c.file_changes:
            index.setdefault(fc.path, []).append((c.ordinal, fc))
    return index


# ---------------------------------------------------------------------------
# Trajectory construction — file-state reconstruction over a shared tree
# ---------------------------------------------------------------------------


def build_head_query(filename: str, message: str, template_idx: int = 0) -> str:
    """Head-node task query — the navigation-oriented compression prompt that
    orients the drill-down (distinct from the reconstruction question)."""
    tmpl = HEAD_QUERY_TEMPLATES[template_idx % len(HEAD_QUERY_TEMPLATES)]
    return tmpl.format(filename=filename, message=message.strip())


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
    head_template_idx: int = 0,
    n_distractors: int = MAX_DISTRACTORS,
    mode_weights: tuple[float, float, float] = DEFAULT_MODE_WEIGHTS,
    truncation_min_depth: int = DEFAULT_TRUNCATION_MIN_DEPTH,
    truncation_max_depth: int = DEFAULT_TRUNCATION_MAX_DEPTH,
    rng=None,
) -> list[TrajectoryTurn] | None:
    """Pure recursive drill-down trajectory for one ``(file, target)`` reconstruction.

    ``touching`` is the chronological ``[(ordinal, FileChange)]`` for the commits
    that touched ``file_path`` up to (and including) the target, already capped to
    :data:`MAX_TOUCHING_DIFFS`. Each becomes a drill target that retrieves that
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
    head_id = window_node_id(repo, window_idx)
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
        )
        targets.append(DrillTarget(leaf_node_id=node_id, retrieve_ids=(fcid,)))
    if not targets:
        return None
    task_query = build_head_query(file_path, head_message, head_template_idx)
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
