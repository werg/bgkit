"""Tests for the topic taxonomy and embeddings."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.taxonomy import TagTaxonomy
from bgkit.models.topic_embeddings import TopicEmbeddingModule


def test_taxonomy_expands_ancestors():
    taxonomy = TagTaxonomy.build([["lang/python/asyncio", "dep/numpy"]])
    assert taxonomy.expand_tags(["lang/python/asyncio"]) == [
        "lang",
        "lang/python",
        "lang/python/asyncio",
    ]


def test_topic_embeddings_return_padded_blocks():
    taxonomy = TagTaxonomy.build([["lang/python/asyncio"], ["dep/numpy"]])
    module = TopicEmbeddingModule(taxonomy, positions_per_tag=2, hidden_dim=4)
    embeddings, mask = module([["lang/python/asyncio"], ["dep/numpy"]])
    assert embeddings.shape == (2, 6, 4)
    assert mask[0].sum().item() == 6
    assert mask[1].sum().item() == 4
