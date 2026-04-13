"""Build per-dataset browse trees from raw metadata.

Input: a flat list of ``(article_id, [tag_path])`` pairs, where ``tag_path``
is an ordered list of ancestor tag names from root → leaf. The builder:

1. Materializes tag nodes for every prefix of every article's path.
2. Attaches each article to its most specific tag (the deepest one in its path).
3. Sub-divides any leaf tag with more than ``leaf_cap`` articles into
   alphabetical buckets ≤ ``leaf_cap`` each.
4. Sub-divides any intermediate node with more than ``fanout_cap`` direct
   children the same way.
5. Drops nodes that cannot be sub-divided further (flat article lists where
   every article name collides on every bucketing key) and logs them.

The output is a :class:`bgkit.data.browse_tree.BrowseTree` ready to save.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import structlog

from bgkit.data.browse_tree import BrowseNode, BrowseTree

logger = structlog.get_logger()


@dataclass
class TaggingConfig:
    dataset: str
    leaf_cap: int = 100
    fanout_cap: int = 100
    # Max number of automatic sub-division iterations before giving up.
    max_bucket_rounds: int = 8


def _bucket_key(name: str, round_idx: int) -> str:
    """Return a deterministic bucket key for ``name`` at the given round.

    Round 0: first alphanumeric character (A-Z / 0-9 / '#').
    Round 1: first two characters.
    Round 2: first three.
    Round ≥3: crude hash-mod-32.
    """
    normalized = re.sub(r"[^0-9A-Za-z]", "", name).upper() or "#"
    if round_idx == 0:
        return normalized[:1]
    if round_idx == 1:
        return normalized[:2]
    if round_idx == 2:
        return normalized[:3]
    # Fallback: hash mod 32 for nodes that collide on all prefixes.
    return f"bucket_{hash(normalized) % 32:02d}"


class _BuildNode:
    __slots__ = ("articles", "children", "id", "is_article", "parent")

    def __init__(self, node_id: str, parent: str | None, is_article: bool = False):
        self.id = node_id
        self.parent = parent
        self.children: list[str] = []
        self.articles: list[str] = []
        self.is_article = is_article


class BrowseTreeBuilder:
    """Build a :class:`BrowseTree` from ``(article_id, tag_path)`` pairs."""

    def __init__(self, cfg: TaggingConfig):
        self.cfg = cfg
        self._nodes: dict[str, _BuildNode] = {}
        self._nodes["root"] = _BuildNode("root", None)
        self._dropped: list[str] = []

    def add_article(self, article_id: str, tag_path: list[str]) -> None:
        """Attach ``article_id`` to the deepest tag in ``tag_path``.

        Missing ancestor tags are created. The article becomes a member of
        the deepest tag's ``articles`` list. No article-as-child-node is
        created yet — that happens later in ``_materialize_articles`` when
        tree sizes are stable.
        """
        if not tag_path:
            tag_path = ["misc"]
        parent = "root"
        current_id = "root"
        for i, raw_name in enumerate(tag_path):
            name = raw_name.strip() or f"_unnamed_{i}"
            current_id = f"{parent}/{name}" if parent != "root" else name
            node = self._nodes.get(current_id)
            if node is None:
                node = _BuildNode(current_id, parent)
                self._nodes[current_id] = node
                self._nodes[parent].children.append(current_id)
            parent = current_id
        # Attach article to deepest node's article list.
        leaf = self._nodes[current_id]
        leaf.articles.append(article_id)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> BrowseTree:
        self._subdivide_oversized_leaves()
        self._subdivide_oversized_fanouts()
        self._materialize_article_nodes()
        sizes = self._recompute_sizes()

        nodes: dict[str, BrowseNode] = {}
        for nid, bn in self._nodes.items():
            if bn.is_article:
                nodes[nid] = BrowseNode(
                    id=nid,
                    parent=bn.parent,
                    kind="article",
                    size=1,
                    children=(),
                    articles=(),
                )
            else:
                is_leaf_tag = len(bn.articles) > 0 and not any(
                    not self._nodes[c].is_article for c in bn.children
                )
                nodes[nid] = BrowseNode(
                    id=nid,
                    parent=bn.parent,
                    kind="sub-tag",
                    size=int(sizes.get(nid, 0)),
                    children=tuple(bn.children),
                    articles=tuple(bn.articles) if is_leaf_tag else (),
                )
        if self._dropped:
            logger.warning(
                "browse_tree_dropped_nodes",
                dataset=self.cfg.dataset,
                count=len(self._dropped),
                examples=self._dropped[:5],
            )
        return BrowseTree(self.cfg.dataset, nodes)

    # ------------------------------------------------------------------
    # Sub-division primitives
    # ------------------------------------------------------------------

    def _subdivide_oversized_leaves(self) -> None:
        cap = self.cfg.leaf_cap
        # Only leaves (nodes with no sub-tag children) can host articles.
        for nid in list(self._nodes.keys()):
            bn = self._nodes.get(nid)
            if bn is None or bn.is_article:
                continue
            if len(bn.articles) <= cap:
                continue
            self._subdivide_leaf(nid)

    def _subdivide_leaf(self, nid: str) -> None:
        cap = self.cfg.leaf_cap
        bn = self._nodes[nid]
        articles = list(bn.articles)
        bn.articles = []

        round_idx = 0
        while round_idx < self.cfg.max_bucket_rounds:
            buckets: dict[str, list[str]] = defaultdict(list)
            for a in articles:
                buckets[_bucket_key(a, round_idx)].append(a)
            if len(buckets) > 1:
                break
            round_idx += 1
        else:
            # No bucketing worked — keep first ``cap`` and drop the rest with a warning.
            self._dropped.append(nid)
            bn.articles = articles[:cap]
            return

        for key, group in sorted(buckets.items()):
            sub_id = f"{nid}/~{key}"
            sub = _BuildNode(sub_id, nid)
            self._nodes[sub_id] = sub
            bn.children.append(sub_id)
            sub.articles = group
            if len(group) > cap:
                self._subdivide_leaf(sub_id)

    def _subdivide_oversized_fanouts(self) -> None:
        cap = self.cfg.fanout_cap
        changed = True
        while changed:
            changed = False
            for nid in list(self._nodes.keys()):
                bn = self._nodes.get(nid)
                if bn is None or bn.is_article:
                    continue
                if len(bn.children) <= cap:
                    continue
                self._subdivide_fanout(nid)
                changed = True

    def _subdivide_fanout(self, nid: str) -> None:
        cap = self.cfg.fanout_cap
        bn = self._nodes[nid]
        children = list(bn.children)
        bn.children = []
        round_idx = 0
        while round_idx < self.cfg.max_bucket_rounds:
            buckets: dict[str, list[str]] = defaultdict(list)
            for c in children:
                local = c.rsplit("/", 1)[-1]
                buckets[_bucket_key(local, round_idx)].append(c)
            if len(buckets) > 1:
                break
            round_idx += 1
        else:
            # Couldn't split — keep first ``cap`` children and drop rest.
            self._dropped.append(nid)
            bn.children = children[:cap]
            for c in children[cap:]:
                self._nodes.pop(c, None)
            return

        for key, group in sorted(buckets.items()):
            sub_id = f"{nid}/~{key}"
            sub = _BuildNode(sub_id, nid)
            self._nodes[sub_id] = sub
            bn.children.append(sub_id)
            for c in group:
                self._nodes[c].parent = sub_id
                sub.children.append(c)
            if len(sub.children) > cap:
                self._subdivide_fanout(sub_id)

    def _materialize_article_nodes(self) -> None:
        """Create an article node per article in every leaf tag.

        This lets the decoder call ``bgkit(ids=["article_x"], ...)`` on a
        single article without needing extra pathway logic.
        """
        to_add: list[_BuildNode] = []
        for nid, bn in list(self._nodes.items()):
            if bn.is_article or not bn.articles:
                continue
            for a in bn.articles:
                if a in self._nodes:
                    # Name collision — skip (already materialized).
                    continue
                node = _BuildNode(a, nid, is_article=True)
                to_add.append(node)
        for node in to_add:
            self._nodes[node.id] = node

    def _recompute_sizes(self) -> dict[str, int]:
        """Post-order size computation over the built tree."""
        sizes: dict[str, int] = {}

        def visit(nid: str) -> int:
            bn = self._nodes[nid]
            if bn.is_article:
                sizes[nid] = 1
                return 1
            # size = number of articles in subtree
            n_articles = len(bn.articles)
            for c in bn.children:
                if self._nodes[c].is_article:
                    # Direct article-child (leaf tag with materialized articles).
                    # Already counted via bn.articles above, skip.
                    continue
                n_articles += visit(c)
            sizes[nid] = n_articles
            return n_articles

        visit("root")
        return sizes
