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

import statistics
from dataclasses import asdict, dataclass, field

import pygit2

from bgkit.data.bgkit_tool_template import TrajectoryTurn
from bgkit.data.browse_tree import BrowseNode, BrowseTree

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
MAX_PRECEDING_COMMITS = 6  # K — preceding file-touching commits to walk
DRILL_PROB = 0.5  # P(bgkit-drill the target file's diff at a PRECEDING commit)
GOLD_TOKEN_CAP = 8192  # a (file, commit) is a target only if its blob ≤ this

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

    @property
    def commit_key(self) -> str:
        """Stable ``(repo, window, ordinal)`` key used by the node-id sidecar."""
        return f"{self.repo}@w{self.window_idx:03d}:{self.ordinal:04d}"

    def file_change_id(self, file_idx: int) -> str:
        """mmap document_id for one file-change leaf."""
        return f"{self.commit_key}#f{file_idx:03d}"

    def to_dict(self) -> dict:
        return asdict(self)


def commit_key(repo: str, window_idx: int, ordinal: int) -> str:
    return f"{repo}@w{window_idx:03d}:{ordinal:04d}"


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
    """Yield ``(message, timestamp, [(path, diff_text, blob_text)])`` oldest-first.

    ``message`` is the FULL commit message (used verbatim in the reconstruction
    prompt). ``blob_text`` is the file's full content at this commit (the gold
    for file-state reconstruction) or ``None`` for deletions/binary. Token
    counting / budgeting / gating is done by the caller so this stays
    tokenizer-free (cheap to reuse in calibration).
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
        yield commit.message.strip(), int(commit.commit_time), files


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
    commit's in-window ``ordinal`` to its leaf-tag node id (the id embeds the
    full chunk path, so callers can't reconstruct it). The window node's
    ``parent`` is the repo node id. Sized proportionally to the window's commit
    count (flat / one chunk level / two chunk levels), so a short final window
    gets a shallower subtree.
    """
    assert commits, "window must be non-empty"
    repo_id = commits[0].repo
    window_idx = commits[0].window_idx
    window_id = f"{repo_id}/w{window_idx:03d}"
    nodes: list[BrowseNode] = []
    ord_to_node: dict[int, str] = {}

    def commit_leaf(c: ReproCommit, parent: str, path_id: str) -> BrowseNode:
        ord_to_node[c.ordinal] = path_id
        article_ids = tuple(c.file_change_id(fc.file_idx) for fc in c.file_changes)
        return BrowseNode(
            id=path_id, parent=parent, kind="sub-tag",
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
        # Flat: commits directly under the window node.
        child_ids = [f"{window_id}/cm_{c.ordinal:04d}" for c in commits]
        for c, cid in zip(commits, child_ids, strict=True):
            nodes.append(commit_leaf(c, window_id, cid))
        nodes.append(window_node(tuple(child_ids)))
        return nodes, ord_to_node, window_id

    # chunk4: group commits into ~4. For N > CHUNK_FANOUT**2 (~16) we add a
    # second level grouping chunk4 nodes into chunk16 nodes.
    c4_groups = _chunk(commits, CHUNK_FANOUT)
    two_levels = len(c4_groups) > CHUNK_FANOUT
    c4_specs = [
        (f"{window_id}/c4_{gi:03d}", group) for gi, group in enumerate(c4_groups)
    ]

    if not two_levels:
        # window → c4 → commit
        for c4_id, group in c4_specs:
            child_ids = [f"{c4_id}/cm_{c.ordinal:04d}" for c in group]
            for c, cid in zip(group, child_ids, strict=True):
                nodes.append(commit_leaf(c, c4_id, cid))
            nodes.append(BrowseNode(
                id=c4_id, parent=window_id, kind="sub-tag", size=size_of(group),
                children=tuple(child_ids), articles=(),
            ))
        nodes.append(window_node(tuple(cid for cid, _ in c4_specs)))
        return nodes, ord_to_node, window_id

    # window → c16 → c4 → commit
    c16_groups = _chunk(c4_specs, CHUNK_FANOUT)
    c16_ids: list[str] = []
    for hi, c16_group in enumerate(c16_groups):
        c16_id = f"{window_id}/c16_{hi:03d}"
        c16_ids.append(c16_id)
        for c4_id, group in c16_group:
            child_ids = [f"{c4_id}/cm_{c.ordinal:04d}" for c in group]
            for c, cid in zip(group, child_ids, strict=True):
                nodes.append(commit_leaf(c, c4_id, cid))
            nodes.append(BrowseNode(
                id=c4_id, parent=c16_id, kind="sub-tag", size=size_of(group),
                children=tuple(child_ids), articles=(),
            ))
        nodes.append(BrowseNode(
            id=c16_id, parent=window_id, kind="sub-tag",
            size=sum(size_of(g) for _, g in c16_group),
            children=tuple(cid for cid, _ in c16_group), articles=(),
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


@dataclass
class WalkStep:
    """One commit in a file's reconstruction walk (oldest→target)."""

    commit_node_id: str  # browse-tree leaf-tag id for the commit
    file_change_id: str  # mmap doc id of the target file's diff at this commit
    is_target: bool  # the final commit being reconstructed
    drill: bool  # whether to bgkit-drill the file's diff at this commit


def build_file_reconstruction_trajectory(
    tree: BrowseTree,
    walk: list[WalkStep],
    query: str,
    gold_blob: str,
) -> list[TrajectoryTurn]:
    """Browse-navigate oldest→target across a file's commit history, drilling
    the target-file diff at each ``drill`` step, then answer the file's blob.

    Navigation is deduplicated: shared ancestors (window / c16 / c4) are
    browsed once; each commit node is browsed exactly once (commit nodes never
    repeat). A PRECEDING commit with ``drill=False`` is still browsed (a
    no-retrieval "stop at the commit" step → random drill-down depth); the
    TARGET commit always drills. The final ``answer`` turn carries the file's
    full blob at the target commit (the reconstruction gold).
    """
    turns: list[TrajectoryTurn] = []
    visited: set[str] = set()
    for step in walk:
        path = tree.path_to(step.commit_node_id)  # [root, repo, window, ..., commit]
        # Drop root + repo; browse only not-yet-seen ancestors, then the commit.
        for node_id in path[2:]:
            if node_id in visited:
                continue
            visited.add(node_id)
            turns.append(TrajectoryTurn(
                kind="browse",
                args={"id": node_id},
                response=tree.render_browse_response(node_id),
                loss=True,
            ))
        if step.drill:
            turns.append(TrajectoryTurn(
                kind="bgkit",
                args={"ids": [step.file_change_id], "query": query},
                response="",
                loss=True,
            ))
    turns.append(TrajectoryTurn(
        kind="answer", args={}, response=gold_blob, loss=True,
    ))
    return turns


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
