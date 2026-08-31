"""Tests for file-read needle QA generation (Family B)."""

from __future__ import annotations

import random

from bgkit.data.file_needle_qa import (
    constant_assignments,
    defined_symbols,
    generate_file_samples,
    unique_identifier_lines,
)

CODE = """\
MAX_RETRIES = 5
TIMEOUT_SECS: 30

def fetch_page(url, retries=MAX_RETRIES):
    return _do_fetch(url, retries)

def _do_fetch(url, retries):
    session_token_9x = make_token()
    return url

class PageCache:
    def get(self, key):
        return self._store.get(key)
"""


def test_defined_symbols_unique_defs():
    defs = defined_symbols(CODE)
    assert "fetch_page" in defs and defs["fetch_page"].startswith("def fetch_page")
    assert "PageCache" in defs
    # `get` is a def too — defined once here
    assert "get" in defs


def test_constant_assignments():
    consts = constant_assignments(CODE)
    assert consts["MAX_RETRIES"] == "5"
    assert consts["TIMEOUT_SECS"] == "30"


def test_unique_identifier_lines():
    uniq = unique_identifier_lines(CODE)
    assert "session_token_9x" in uniq
    assert "retries" not in uniq  # appears on several lines


def test_generate_file_samples_types_and_presence():
    rng = random.Random(3)
    samples = generate_file_samples(
        CODE,
        source_label="repo:pkg/fetch.py",
        rng=rng,
        absent_symbols=["totally_absent_fn"],
    )
    types = {s.qtype for s in samples}
    assert {"signature", "assignment", "needle_token"} <= types
    assert types & {"presence_absent", "presence_present"}
    for s in samples:
        assert 'query="' in s.blob_header
        if s.qtype == "signature":
            sym = s.question.split("`")[1]
            assert sym in s.answer
        if s.qtype == "presence_absent":
            assert "not defined" in s.answer


def test_presence_question_form_is_identical_across_classes():
    """The whole point of the balanced rewrite: if the two classes were
    distinguishable from the question text, the negative would still be
    answerable with zero retrieval -- which is what made 19.9% of fileneedle
    rep-blind and inflated the family's rep-dependence number."""
    forms: dict[str, set[str]] = {}
    for seed in range(60):
        for s in generate_file_samples(
            CODE,
            source_label="repo:pkg/fetch.py",
            rng=random.Random(seed),
            absent_symbols=["totally_absent_fn", "another_missing_fn"],
        ):
            if s.qtype.startswith("presence_"):
                sym = s.question.split("`")[1]
                forms.setdefault(s.qtype, set()).add(s.question.replace(sym, "SYM"))
    assert set(forms) == {"presence_absent", "presence_present"}, forms
    # One question template, both classes -- the symbol is the only variable.
    assert forms["presence_absent"] == forms["presence_present"]


def test_presence_classes_are_both_drawn():
    """Both classes must actually appear; a pool that is empty in practice
    would silently restore the negatives-only dataset."""
    counts = {"presence_absent": 0, "presence_present": 0}
    for seed in range(60):
        for s in generate_file_samples(
            CODE,
            source_label="repo:pkg/fetch.py",
            rng=random.Random(seed),
            absent_symbols=["totally_absent_fn", "another_missing_fn"],
        ):
            if s.qtype in counts:
                counts[s.qtype] += 1
    assert counts["presence_absent"] > 10
    assert counts["presence_present"] > 10


def test_presence_present_answer_is_a_real_line_with_a_span():
    """The positive answer must be quotable from the file, or the gold span
    for it cannot exist and the class becomes unanswerable-by-reading."""
    for seed in range(60):
        for s in generate_file_samples(
            CODE,
            source_label="repo:pkg/fetch.py",
            rng=random.Random(seed),
            absent_symbols=["totally_absent_fn"],
        ):
            if s.qtype == "presence_present":
                assert s.span_text == s.answer
                assert s.answer in CODE
            if s.qtype == "presence_absent":
                # Explicitly NOT in the text; the builder keys span extraction
                # off this so it never searches for a string that is absent.
                assert s.span_text is None
                assert s.answer not in CODE


def test_every_quoting_answer_carries_its_own_span_text():
    for s in generate_file_samples(
        CODE, source_label="repo:pkg/fetch.py", rng=random.Random(1),
        absent_symbols=["totally_absent_fn"],
    ):
        if s.qtype != "presence_absent":
            assert s.span_text == s.answer


def test_presence_present_does_not_repeat_a_signature_symbol():
    """Same symbol, same answer line, two questions = a duplicate, and the
    presence class would then be answerable by recalling the earlier row."""
    for seed in range(60):
        samples = generate_file_samples(
            CODE,
            source_label="repo:pkg/fetch.py",
            rng=random.Random(seed),
            absent_symbols=["totally_absent_fn"],
        )
        sig = {s.question.split("`")[1] for s in samples if s.qtype == "signature"}
        pres = {
            s.question.split("`")[1] for s in samples if s.qtype == "presence_present"
        }
        assert not (sig & pres), (seed, sig, pres)


def test_generate_file_samples_skips_overlong_answer_lines():
    """A needle is one human-scale line: candidates longer than
    ``max_answer_chars`` (data tables, embedded blobs) are never asked."""
    blob = "PAYLOAD_TABLE = [" + ", ".join(f"'{i}_entry_value_xyz'" for i in range(200)) + "]"
    code = CODE + "\n" + blob + "\ndef tiny_helper_fn():\n    return 1\n"
    assert len(blob) > 400
    samples = generate_file_samples(code, source_label="r:f.py", rng=random.Random(0))
    assert samples
    for s in samples:
        assert len(s.answer) <= 400
        assert "PAYLOAD_TABLE" not in s.question
    # The cap is a parameter; loosening it re-admits the long line.
    loose = generate_file_samples(
        code, source_label="r:f.py", rng=random.Random(0), max_answer_chars=10_000
    )
    assert any(len(s.answer) > 400 for s in loose)
