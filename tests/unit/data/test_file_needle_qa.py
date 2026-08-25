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


def test_generate_file_samples_types_and_absent():
    rng = random.Random(3)
    samples = generate_file_samples(
        CODE,
        source_label="repo:pkg/fetch.py",
        rng=rng,
        absent_symbols=["totally_absent_fn"],
    )
    types = {s.qtype for s in samples}
    assert {"signature", "assignment", "needle_token", "absent"} <= types
    for s in samples:
        assert 'query="' in s.blob_header
        if s.qtype == "signature":
            sym = s.question.split("`")[1]
            assert sym in s.answer
        if s.qtype == "absent":
            assert "not defined" in s.answer


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
