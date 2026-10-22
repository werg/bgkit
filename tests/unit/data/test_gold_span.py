"""Tests for gold-span token offsets (v5 span-level relevance supervision)."""

from __future__ import annotations

import pytest

from bgkit.data.gold_span import answer_token_span, char_span_to_token_span, span_to_json


def test_char_to_token_span_overlap_rule():
    offsets = [(0, 3), (3, 6), (6, 10), (10, 14)]
    assert char_span_to_token_span(offsets, 3, 6) == (1, 2)
    assert char_span_to_token_span(offsets, 4, 8) == (1, 3)  # partial overlaps included
    assert char_span_to_token_span(offsets, 20, 25) is None
    assert char_span_to_token_span([(0, 0), (0, 3)], 0, 2) == (1, 2)  # skip zero-width


def test_span_json_roundtrip():
    assert span_to_json(None) is None
    assert span_to_json((3, 7)) == "[3, 7]"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    except Exception:
        pytest.skip("Qwen3.5 tokenizer not available locally")


def test_answer_span_consistent_with_plain_encoding(tokenizer):
    """The span must index into tokenizer.encode(article) — the same ids the
    mmap store holds — and decode back to text containing the answer."""
    article = (
        "081109 INFO dfs.DataNode: ok\n"
        "081109 ERROR dfs.FSNamesystem: addStoredBlock failed for blk_71283\n"
        "081109 INFO done\n"
    )
    answer = "081109 ERROR dfs.FSNamesystem: addStoredBlock failed for blk_71283"
    span = answer_token_span(tokenizer, article, answer)
    assert span is not None
    ids = tokenizer.encode(article, add_special_tokens=False)
    a, b = span
    assert 0 <= a < b <= len(ids)
    decoded = tokenizer.decode(ids[a:b])
    assert answer in decoded or decoded.strip() in answer or answer.strip() in decoded
    assert answer_token_span(tokenizer, article, "not in there") is None
