"""Per-dataset browse tree: load, query, render.

A browse tree is a pre-filtered hierarchy of tags → sub-tags → articles that
the decoder drills through via ``bgkit`` tool calls on the IDs surfaced at each
level. Leaves are pre-capped to at most ``leaf_cap`` articles at build time
(see :mod:`bgkit.data.tagging`); intermediate nodes are pre-capped to at most
``fanout_cap`` children.

The on-disk format is a single parquet per dataset with these columns:

- ``id`` (string): full tag ID (includes path, unique within dataset)
- ``parent`` (string | None): parent tag ID, None for root
- ``kind`` (string): ``"sub-tag"`` or ``"article"`` (leaf article, pinned to a tag)
- ``size`` (int64): article count under this node
- ``children`` (list<string>): child IDs in canonical order
- ``articles`` (list<string>): article IDs in this leaf (only when this is a leaf tag)

Article nodes with ``kind == "article"`` carry the leaf article's own ID in
``id`` and nothing in ``children``/``articles``. Tag leaves (pre-capped
≤100 articles) use ``kind == "sub-tag"`` with a non-empty ``articles`` list
and an empty ``children`` list.

A synthetic ``root`` node is always present.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class BrowseNode:
    id: str
    parent: str | None
    kind: str  # "sub-tag" or "article"
    size: int
    children: tuple[str, ...]
    articles: tuple[str, ...]

    @property
    def is_leaf_tag(self) -> bool:
        """True if this is a tag leaf — a sub-tag whose articles are its direct
        member list (no further sub-tags below)."""
        return self.kind == "sub-tag" and len(self.articles) > 0

    @property
    def is_article(self) -> bool:
        return self.kind == "article"


class BrowseTree:
    """In-memory query interface over a per-dataset browse tree parquet."""

    def __init__(self, dataset: str, nodes: dict[str, BrowseNode]):
        self.dataset = dataset
        self._nodes = nodes
        self._article_to_leaf_tag: dict[str, str] | None = None  # lazy O(1) index
        if "root" not in nodes:
            raise ValueError(f"Browse tree for {dataset!r} must contain a 'root' node")

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path, dataset: str | None = None) -> BrowseTree:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Browse tree not found: {path}")
        table = pq.read_table(path)
        rows = table.to_pylist()
        nodes: dict[str, BrowseNode] = {}
        for row in rows:
            nid = str(row["id"])
            node = BrowseNode(
                id=nid,
                parent=(None if row["parent"] is None else str(row["parent"])),
                kind=str(row["kind"]),
                size=int(row["size"]),
                children=tuple(row.get("children", []) or ()),
                articles=tuple(row.get("articles", []) or ()),
            )
            nodes[nid] = node
        return cls(dataset=dataset or path.stem, nodes=nodes)

    @classmethod
    def from_nodes(
        cls,
        dataset: str,
        nodes: Iterable[BrowseNode],
    ) -> BrowseTree:
        return cls(dataset, {n.id: n for n in nodes})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for node in self._nodes.values():
            rows.append({
                "id": node.id,
                "parent": node.parent,
                "kind": node.kind,
                "size": node.size,
                "children": list(node.children),
                "articles": list(node.articles),
            })
        table = pa.Table.from_pylist(rows, schema=pa.schema([
            ("id", pa.string()),
            ("parent", pa.string()),
            ("kind", pa.string()),
            ("size", pa.int64()),
            ("children", pa.list_(pa.string())),
            ("articles", pa.list_(pa.string())),
        ]))
        pq.write_table(table, path)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def get(self, node_id: str) -> BrowseNode:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"Browse node {node_id!r} not in {self.dataset!r}") from exc

    def children(self, node_id: str) -> list[BrowseNode]:
        return [self._nodes[c] for c in self.get(node_id).children if c in self._nodes]

    def articles(self, node_id: str) -> list[str]:
        node = self.get(node_id)
        if node.is_leaf_tag:
            return list(node.articles)
        # For non-leaf nodes, recurse through children.
        result: list[str] = []
        stack = [node_id]
        while stack:
            current = self._nodes[stack.pop()]
            if current.is_article:
                result.append(current.id)
            elif current.is_leaf_tag:
                result.extend(current.articles)
            else:
                stack.extend(reversed(current.children))
        return result

    def path_to(self, target: str) -> list[str]:
        """Return the list of ancestors from root → target (inclusive).

        Used by :mod:`bgkit.data.teacher_trajectories` to derive browse
        trajectories from dataset provenance.
        """
        if target not in self._nodes:
            raise KeyError(f"path_to: {target!r} not in browse tree {self.dataset!r}")
        path: list[str] = []
        current: str | None = target
        while current is not None:
            path.append(current)
            current = self._nodes[current].parent
        return list(reversed(path))

    def leaf_tag_for_article(self, article_id: str) -> str | None:
        """Find the leaf tag containing ``article_id``.

        Returns None if the article is not in the tree. Used by trajectory
        generation when provenance gives a bare article ID rather than a
        tag path.
        """
        node = self._nodes.get(article_id)
        if node is not None and node.is_article and node.parent is not None:
            return node.parent
        # O(1) fallback via a lazily-built article -> leaf-tag index. The
        # naive per-call linear scan is O(nodes) — catastrophic on large trees
        # (git-repro: ~79K nodes x millions of coverage checks = a >1h setup
        # hang). Build the index once, then look up in O(1).
        if self._article_to_leaf_tag is None:
            idx: dict[str, str] = {}
            for n in self._nodes.values():
                if n.is_leaf_tag:
                    for aid in n.articles:
                        idx[aid] = n.id
            self._article_to_leaf_tag = idx
        return self._article_to_leaf_tag.get(article_id)

    def top_level_topic_list(self) -> list[str]:
        """Return the direct children of ``root`` (e.g. Wikipedia top categories).

        Used by the topic-list system prompt template to seed the LLM with
        the list of entrypoints.
        """
        return list(self.get("root").children)

    def is_flat(self) -> bool:
        """True if this tree has no meaningful navigation hierarchy.

        A tree is flat when every gold-article path from the root passes
        only through nodes that offer the decoder no real navigational
        choice. Concretely: walking down from root, at each step we
        skip single-child chains (one child = no decision) and
        synthetic auto-bucketed chains (children whose local name
        starts with ``~``, the marker
        :class:`bgkit.data.tagging.BrowseTreeBuilder` uses for
        alphabet/hash sub-divisions). If we reach a leaf-tag or
        article without ever encountering a fork between semantically
        named children, the tree is flat.

        Hierarchical trees (KILT via DBpedia categories, PubMedQA via
        MeSH) have at least one semantically named branching level and
        exercise the browse machinery. Flat trees (NewsQA, MS MARCO,
        SearchQA, git history with no repo grouping, memory) skip
        browse turns in teacher trajectories — the decoder calls
        ``bgkit([gold_article_id], query)`` directly on the gold
        article instead of pretending to navigate.
        """
        current = self.get("root")
        # Walk down, skipping single-child chains and synthetic forks.
        max_depth = 16  # safety bound, real trees never come close
        for _ in range(max_depth):
            if current.is_article or current.is_leaf_tag:
                return True
            if not current.children:
                return True
            named_children = [
                self._nodes[c]
                for c in current.children
                if c in self._nodes
                and not c.rsplit("/", 1)[-1].startswith("~")
            ]
            if len(named_children) >= 2:
                # Real semantic fork: more than one non-synthetic child.
                return False
            if len(named_children) == 1:
                # Single-child chain — keep descending.
                current = named_children[0]
                continue
            # No named children at all (all synthetic ~ buckets) → flat.
            return True
        return False
