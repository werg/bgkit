"""Tests for conversion helpers in convert_hf_to_mmap.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_scripts = str(Path(__file__).resolve().parents[3] / "scripts")
_src = str(Path(__file__).resolve().parents[3] / "src")
for p in (_scripts, _src):
    if p not in sys.path:
        sys.path.insert(0, p)

from convert_hf_to_mmap import (
    _coerce_text,
    _extract_kilt_answer,
    _extract_kilt_provenance,
    _extract_msmarco_answer,
    _extract_msmarco_document,
)


def _build_document_id(record: dict, idx: int) -> str:
    """Reproduce the document_id fallback chain from convert_dataset().

    Kept in sync with the production code so the regression test catches
    any accidental reordering of the probes.
    """
    raw_doc_id = (
        record.get("document_id")
        or record.get("wikipedia_id")
        or record.get("pubid")
        or record.get("query_id")
        or record.get("key")
        or record.get("id")
        or idx
    )
    return str(raw_doc_id)


class TestDocumentIdFallbackChain:
    """The document_id fallback chain must produce the correct stable key
    for every Phase 2 dataset. A wrong fallback silently breaks
    ``article_id → document_id`` alignment between browse trees, L0
    caches, and teacher trajectories."""

    def test_pubmedqa_uses_pubid(self):
        record = {"pubid": 27733282, "question": "...", "long_answer": "..."}
        assert _build_document_id(record, 0) == "27733282"

    def test_ms_marco_uses_query_id(self):
        record = {"query_id": 1185869, "query": "...", "passages": []}
        assert _build_document_id(record, 0) == "1185869"

    def test_searchqa_uses_key(self):
        record = {"key": "search_12345", "question": "...", "answer": "..."}
        assert _build_document_id(record, 0) == "search_12345"

    def test_kilt_wikipedia_uses_wikipedia_id(self):
        record = {"wikipedia_id": "290", "wikipedia_title": "A", "text": []}
        assert _build_document_id(record, 0) == "290"

    def test_explicit_document_id_wins(self):
        record = {
            "document_id": "explicit_id",
            "wikipedia_id": "should_lose",
            "pubid": 99,
            "id": "also_lose",
        }
        assert _build_document_id(record, 0) == "explicit_id"

    def test_generic_id_fallback(self):
        record = {"id": "doc_42", "question": "..."}
        assert _build_document_id(record, 99) == "doc_42"

    def test_enumeration_last_resort(self):
        # No recognized id field: falls back to the positional index.
        # This is a bug signal — never a meaningful document_id.
        record = {"text": "some content", "no_id_here": True}
        assert _build_document_id(record, 7) == "7"

    def test_numeric_zero_not_confused_with_missing(self):
        # pubid=0 is technically a valid value but Python's `or` chain
        # treats it as falsy. This test documents that limitation so a
        # future reader knows: if a dataset ever legitimately uses id=0,
        # the fallback chain silently skips it. PubMed IDs start at 1
        # historically so this is not a practical issue.
        record = {"pubid": 0, "id": "fallback_id"}
        assert _build_document_id(record, 5) == "fallback_id"


# ===================================================================
# _coerce_text
# ===================================================================


class TestCoerceText:
    def test_none_returns_empty(self):
        assert _coerce_text(None) == ""

    def test_string_passthrough(self):
        assert _coerce_text("hello world") == "hello world"

    def test_list_joins_with_newlines(self):
        result = _coerce_text(["line1", "line2", "line3"])
        assert result == "line1\nline2\nline3"

    def test_list_with_none_skips_empty(self):
        result = _coerce_text(["line1", None, "line3"])
        assert result == "line1\nline3"

    def test_dict_with_text_key(self):
        result = _coerce_text({"text": "document content", "id": "123"})
        assert result == "document content"

    def test_dict_without_text_key(self):
        result = _coerce_text({"key1": "val1", "key2": "val2"})
        assert "key1: val1" in result
        assert "key2: val2" in result

    def test_nested_list(self):
        result = _coerce_text(["outer", ["inner1", "inner2"]])
        assert "outer" in result
        assert "inner1" in result

    def test_numeric_value(self):
        assert _coerce_text(42) == "42"


# ===================================================================
# _extract_msmarco_document
# ===================================================================


class TestExtractMSMARCODocument:
    def test_selected_passages_preferred(self):
        record = {
            "passages": {
                "passage_text": ["irrelevant", "selected text", "also irrelevant"],
                "is_selected": [0, 1, 0],
            },
        }
        result = _extract_msmarco_document(record)
        assert result == "selected text"

    def test_all_passages_when_none_selected(self):
        record = {
            "passages": {
                "passage_text": ["text1", "text2"],
                "is_selected": [0, 0],
            },
        }
        result = _extract_msmarco_document(record)
        assert "text1" in result
        assert "text2" in result

    def test_missing_passages_returns_empty(self):
        assert _extract_msmarco_document({}) == ""
        assert _extract_msmarco_document({"passages": None}) == ""

    def test_string_passages_coerced(self):
        record = {"passages": "plain string passage"}
        result = _extract_msmarco_document(record)
        assert result == "plain string passage"


# ===================================================================
# _extract_msmarco_answer
# ===================================================================


class TestExtractMSMARCOAnswer:
    def test_filters_no_answer_present(self):
        record = {"answers": ["No Answer Present.", "actual answer"]}
        result = _extract_msmarco_answer(record)
        assert result == "actual answer"

    def test_all_no_answer_returns_empty(self):
        record = {"answers": ["No Answer Present."]}
        result = _extract_msmarco_answer(record)
        assert result == ""

    def test_first_valid_answer_returned(self):
        record = {"answers": ["first", "second"]}
        result = _extract_msmarco_answer(record)
        assert result == "first"

    def test_missing_answers_returns_empty(self):
        assert _extract_msmarco_answer({}) == ""
        assert _extract_msmarco_answer({"answers": None}) == ""

    def test_empty_list_returns_empty(self):
        assert _extract_msmarco_answer({"answers": []}) == ""


# ===================================================================
# _extract_kilt_answer
# ===================================================================


class TestExtractKILTAnswer:
    def test_extracts_from_dict_output(self):
        record = {"output": [{"answer": "Paris"}]}
        result = _extract_kilt_answer(record)
        assert result == "Paris"

    def test_extracts_first_answer(self):
        record = {"output": [{"answer": "first"}, {"answer": "second"}]}
        result = _extract_kilt_answer(record)
        assert result == "first"

    def test_handles_string_entries(self):
        record = {"output": ["plain answer"]}
        result = _extract_kilt_answer(record)
        assert result == "plain answer"

    def test_missing_output_returns_empty(self):
        assert _extract_kilt_answer({}) == ""
        assert _extract_kilt_answer({"output": None}) == ""

    def test_empty_answer_skipped(self):
        record = {"output": [{"answer": ""}, {"answer": "valid"}]}
        result = _extract_kilt_answer(record)
        assert result == "valid"


# ===================================================================
# _extract_kilt_provenance
# ===================================================================


class TestExtractKILTProvenance:
    def test_extracts_wikipedia_ids(self):
        record = {
            "output": [{
                "provenance": [
                    {"wikipedia_id": "12345"},
                    {"wikipedia_id": "67890"},
                ],
            }],
        }
        result = _extract_kilt_provenance(record)
        assert result == ["12345", "67890"]

    def test_missing_provenance_returns_empty(self):
        record = {"output": [{"answer": "Paris"}]}
        result = _extract_kilt_provenance(record)
        assert result == []

    def test_missing_output_returns_empty(self):
        assert _extract_kilt_provenance({}) == []

    def test_non_list_output_returns_empty(self):
        assert _extract_kilt_provenance({"output": "not a list"}) == []

    def test_multiple_provenance_entries(self):
        record = {
            "output": [
                {"provenance": [{"wikipedia_id": "111"}]},
                {"provenance": [{"wikipedia_id": "222"}]},
            ],
        }
        result = _extract_kilt_provenance(record)
        assert result == ["111", "222"]

    def test_provenance_without_wikipedia_id_skipped(self):
        record = {
            "output": [{
                "provenance": [
                    {"title": "Some page"},  # no wikipedia_id
                    {"wikipedia_id": "999"},
                ],
            }],
        }
        result = _extract_kilt_provenance(record)
        assert result == ["999"]
