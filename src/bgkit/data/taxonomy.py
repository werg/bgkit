"""Hierarchical tag taxonomy for Phase 2 topic embeddings."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TagNode:
    """Single tag entry in the taxonomy."""

    name: str
    parent: str | None
    frequency: int


class TagTaxonomy:
    """Stores hierarchical tags and expands tags through ancestor chains."""

    def __init__(self, nodes: dict[str, TagNode], separator: str = "/"):
        self._nodes = dict(nodes)
        self.separator = separator

    @classmethod
    def from_tag_counts(
        cls,
        counts: dict[str, int] | Counter[str],
        *,
        min_frequency: int = 1,
        separator: str = "/",
    ) -> TagTaxonomy:
        nodes: dict[str, TagNode] = {}
        for tag, freq in counts.items():
            if freq < min_frequency:
                continue
            parent = tag.rpartition(separator)[0] or None
            nodes[tag] = TagNode(name=tag, parent=parent, frequency=int(freq))
            current = parent
            while current:
                nodes.setdefault(
                    current,
                    TagNode(
                        name=current,
                        parent=current.rpartition(separator)[0] or None,
                        frequency=0,
                    ),
                )
                current = current.rpartition(separator)[0] or None
        return cls(nodes, separator=separator)

    @classmethod
    def build(
        cls,
        tag_lists: list[list[str]],
        *,
        min_frequency: int = 1,
        separator: str = "/",
    ) -> TagTaxonomy:
        counter: Counter[str] = Counter()
        for tags in tag_lists:
            counter.update(map(str, tags))
        return cls.from_tag_counts(counter, min_frequency=min_frequency, separator=separator)

    @classmethod
    def from_browse_tree(
        cls,
        tree: object,
        *,
        frequency_from_size: bool = True,
    ) -> TagTaxonomy:
        """Build a taxonomy from a :class:`bgkit.data.browse_tree.BrowseTree`.

        Every non-article node in the tree becomes a tag. The parent
        relationship is inherited from the browse tree's ``parent`` field.
        When ``frequency_from_size`` is True (default) each tag's
        ``frequency`` is set to the number of articles underneath it,
        which gives ``TopicEmbeddingModule``'s LR scaler something to
        work with even without real usage counts.

        Tag IDs coming from a browse tree already use ``/`` as a
        hierarchical separator (e.g. ``Physics/Quantum_mechanics``), so
        we preserve that as the canonical separator.
        """
        nodes: dict[str, TagNode] = {}
        for node_id in getattr(tree, "_nodes", {}):
            node = tree.get(node_id)
            if node.is_article:
                continue
            parent = node.parent if node.parent != "root" else None
            freq = int(node.size) if frequency_from_size else 0
            nodes[node_id] = TagNode(
                name=node_id, parent=parent, frequency=max(freq, 0),
            )
        if "root" not in nodes:
            # Ensure a root node exists for ancestors() to terminate cleanly.
            nodes["root"] = TagNode(name="root", parent=None, frequency=0)
        return cls(nodes, separator="/")

    @classmethod
    def load(cls, path: str | Path) -> TagTaxonomy:
        payload = json.loads(Path(path).read_text())
        nodes = {
            name: TagNode(name=name, parent=node.get("parent"), frequency=int(node["frequency"]))
            for name, node in payload["nodes"].items()
        }
        return cls(nodes, separator=payload.get("separator", "/"))

    def save(self, path: str | Path) -> None:
        payload = {
            "separator": self.separator,
            "nodes": {
                name: {"parent": node.parent, "frequency": node.frequency}
                for name, node in self._nodes.items()
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    def __contains__(self, tag: str) -> bool:
        return tag in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def with_frequencies(
        self,
        counts: dict[str, int] | Counter[str],
    ) -> TagTaxonomy:
        """Return a new taxonomy with frequencies replaced from ``counts``.

        Tags not present in ``counts`` get frequency 0. Parent/child
        structure is preserved. Used by the KB-scale trainer to replace
        the (misleading) tree-size-derived frequencies from
        :meth:`from_browse_tree` with actual per-tag occurrence counts
        computed over the loaded trajectory dataset — which is what the
        optimizer's sqrt-frequency LR scaling really wants to see.
        """
        new_nodes: dict[str, TagNode] = {}
        for name, node in self._nodes.items():
            new_nodes[name] = TagNode(
                name=node.name,
                parent=node.parent,
                frequency=int(counts.get(name, 0)),
            )
        return TagTaxonomy(new_nodes, separator=self.separator)

    @property
    def tags(self) -> list[str]:
        return sorted(self._nodes)

    def frequency(self, tag: str) -> int:
        return self._nodes.get(tag, TagNode(tag, None, 0)).frequency

    def parent(self, tag: str) -> str | None:
        node = self._nodes.get(tag)
        if node is not None:
            return node.parent
        parent = tag.rpartition(self.separator)[0]
        return parent or None

    def ancestors(self, tag: str, *, include_self: bool = True) -> list[str]:
        current = tag if include_self else self.parent(tag)
        result: list[str] = []
        while current:
            result.append(current)
            current = self.parent(current)
        return result

    def expand_tags(self, tags: list[str]) -> list[str]:
        seen: set[str] = set()
        expanded: list[str] = []
        for tag in tags:
            for ancestor in reversed(self.ancestors(str(tag), include_self=True)):
                if ancestor not in seen:
                    seen.add(ancestor)
                    expanded.append(ancestor)
        return expanded

    def lookup_tags(self, metadata: dict[str, object]) -> list[str]:
        """Extract and expand tags from common metadata shapes."""
        raw_tags: list[str] = []
        for key in (
            "tags",
            "wikipedia_categories",
            "mesh_terms",
            "memory_types",
            "dependencies",
        ):
            value = metadata.get(key)
            if isinstance(value, list):
                raw_tags.extend(str(item) for item in value)
        language = metadata.get("language")
        if language:
            raw_tags.append(f"language/{language}")
        return self.expand_tags(raw_tags)
