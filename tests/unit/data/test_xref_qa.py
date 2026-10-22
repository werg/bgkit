"""The whole point of this family is that its answers are unguessable.

Every existing wide-net family fails on some axis (measured 2026-09-03):
lognav's top-20 answers cover 52.5% of eval, fileneedle's questions echo
0.227 against their own answers, reconstruct's 124-char answers proved
unlearnable. These tests pin the properties that make xref different.
"""

from __future__ import annotations

import random

from bgkit.data.xref_qa import filter_candidates, generate_xref_samples

CODE = "\n".join(
    [
        "import os",
        "from typing import Any",
        "",
        "def parse_config(path: str) -> dict:",
        "    raw_payload_marker = open(path).read()",
        "    return {}",
        "",
        "def validate_schema(cfg: dict) -> bool:",
        "    unique_sentinel_token = cfg.get('schema')",
        "    return bool(unique_sentinel_token)",
        "",
        "class ResourceCache:",
        "    def __init__(self, capacity: int) -> None:",
        "        self._capacity = capacity",
        "",
        "def flush_pending_writes(cache: Any) -> None:",
        "    distinctive_flush_id = 42",
        "    del distinctive_flush_id",
        "",
        "def rebuild_index_tree(root: str) -> None:",
        "    another_rare_marker = root",
        "    return None",
    ]
)


def _samples(seed: int = 0, **kw):
    return generate_xref_samples(
        CODE, source_label="repo:pkg/cfg.py", rng=random.Random(seed), **kw,
    )


def test_the_question_never_contains_its_answer():
    """THE property. fileneedle's questions name the symbol their answers
    contain, which is why echoing the prompt scores 0.227 there -- and why
    its no-document arm scored 0.241, i.e. it was echoing."""
    for seed in range(40):
        for s in _samples(seed):
            assert s.answer not in s.question, (seed, s.qtype, s.answer)


def test_answers_are_short_enough_to_survive_compression():
    """reconstruct's 124-char answers were unlearnable at 5% retention;
    a symbol name is ~15 characters."""
    for seed in range(20):
        for s in _samples(seed):
            assert len(s.answer) <= 60


def test_every_answer_occurs_in_the_document():
    for seed in range(20):
        for s in _samples(seed):
            assert s.answer in CODE
            assert s.span_text == s.answer


def test_next_def_names_the_definition_that_follows_its_anchor():
    seen = False
    for seed in range(40):
        for s in _samples(seed):
            if s.qtype != "next_def":
                continue
            seen = True
            anchor = s.question.split("`")[1]
            assert CODE.index(anchor) < CODE.index(f" {s.answer}")
    assert seen, "no next_def samples generated"


def test_enclosing_def_anchor_occurs_exactly_once_and_is_not_a_definition():
    seen = False
    for seed in range(40):
        for s in _samples(seed):
            if s.qtype != "enclosing_def":
                continue
            seen = True
            tok = s.question.split("`")[1]
            assert CODE.count(tok) == 1
            assert f"def {tok}" not in CODE and f"class {tok}" not in CODE
    assert seen, "no enclosing_def samples generated"


def test_a_file_with_no_definitions_yields_nothing():
    assert generate_xref_samples(
        "x = 1\ny = 2\n", source_label="r:t.py", rng=random.Random(0),
    ) == []


def test_corpus_common_answers_are_dropped():
    """A generator seeing one file cannot know that __init__ answers a
    thousand others. Measured: corpus-common answers put enclosing_def's
    top-20 at 66.4%."""
    rows = [{"answer": "__init__", "doc_id": f"d{i}"} for i in range(100)]
    rows += [{"answer": "rare_specific_name", "doc_id": "d0"}]
    kept = filter_candidates(rows)
    assert [r["answer"] for r in kept] == ["rare_specific_name"]


def test_repeats_within_one_document_are_capped():
    """One large function contains many rare tokens, so a single file
    otherwise emits a dozen rows all answering with its name -- which kept
    the top-20 at 68% even after the corpus filter."""
    # Enough other documents that "handler" is NOT corpus-common -- otherwise
    # the corpus filter removes it first and this tests nothing.
    rows = [{"answer": "handler", "doc_id": "d0"} for _ in range(12)]
    rows += [{"answer": f"symbol_{i}", "doc_id": f"d{i}"} for i in range(1, 200)]
    kept = filter_candidates(rows)
    assert sum(1 for r in kept if r["doc_id"] == "d0") == 1
    assert len(kept) == 200


def test_the_filters_keep_a_genuinely_varied_corpus():
    rows = [{"answer": f"symbol_{i}", "doc_id": f"d{i}"} for i in range(200)]
    assert len(filter_candidates(rows)) == 200


def test_generation_is_deterministic_for_a_seed():
    for seed in range(5):
        assert [s.answer for s in _samples(seed)] == [s.answer for s in _samples(seed)]


def test_the_header_carries_the_query_like_the_other_families():
    for s in _samples(1):
        assert 'query="' in s.blob_header
