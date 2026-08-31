"""The reconstruction family only works if the prompt never contains the target.

A previous reconstruction attempt (``git_commit_repro``) sat at recon_gap ~0
for thousands of steps because a browse plaintext-copy path let the decoder
copy the answer instead of reading the reps. That is the one property this
family cannot be allowed to lose, so it is tested first and hardest.
"""

from __future__ import annotations

import random

import pytest

from bgkit.data.reconstruct_qa import generate_reconstruct_samples

CODE = "\n".join(
    [
        "import os",
        "import sys",
        "from typing import Any",
        "",
        "MAX_RETRIES = 5",
        "TIMEOUT_SECS = 30",
        "",
        "def fetch_resource(url: str, retries: int = MAX_RETRIES) -> Any:",
        "    session_token_9x = os.environ['TOKEN']",
        "    for attempt in range(retries):",
        "        try:",
        "            return _do_fetch(url, session_token_9x)",
        "        except TimeoutError:",
        "            continue",
        "    raise RuntimeError('exhausted retries')",
        "",
        "def _do_fetch(url: str, token: str) -> Any:",
        "    headers = {'Authorization': f'Bearer {token}'}",
        "    return _client.get(url, headers=headers, timeout=TIMEOUT_SECS)",
        "",
        "class ResourceCache:",
        "    def __init__(self, capacity: int) -> None:",
        "        self._capacity = capacity",
        "        self._entries: dict[str, Any] = {}",
        "",
        "    def put(self, key: str, value: Any) -> None:",
        "        if len(self._entries) >= self._capacity:",
        "            self._entries.pop(next(iter(self._entries)))",
        "        self._entries[key] = value",
        "",
        "    def get(self, key: str) -> Any:",
        "        return self._entries.get(key)",
    ]
)


def _samples(seed: int = 0, **kw):
    return generate_reconstruct_samples(
        CODE, source_label="repo:pkg/fetch.py", rng=random.Random(seed), **kw,
    )


def test_the_prompt_never_contains_the_answer():
    """THE property. If the question carries the target, the decoder can copy
    it and the reps stay decorative -- which is exactly how the previous
    reconstruction attempt failed."""
    for seed in range(40):
        for s in _samples(seed):
            assert s.answer not in s.question, (seed, s.qtype)


def test_the_tail_question_names_no_content_at_all():
    for seed in range(40):
        for s in _samples(seed):
            if s.qtype == "tail":
                assert "`" not in s.question


def test_there_is_no_head_variant():
    """The first lines of a source file are its most predictable region --
    shebang, licence, imports -- so a correct answer there is weak evidence of
    having read anything. Spot checks on real repos produced Apache headers
    and bundler preambles, which a model reproduces from its prior."""
    for seed in range(40):
        assert all(s.qtype != "head" for s in _samples(seed))


def test_licence_headers_are_rejected_wherever_they_land():
    licensed = "\n".join(
        ["/*"]
        + [" * Copyright 2011 Example Inc."]
        + [" * Licensed under the Apache License, Version 2.0"]
        + [f" * line {i} of the notice text here" for i in range(20)]
        + [" */", ""]
        + [f"def real_function_{i}(argument):" for i in range(20)]
    )
    for seed in range(30):
        for s in generate_reconstruct_samples(
            licensed, source_label="r:l.py", rng=random.Random(seed),
        ):
            assert "Copyright" not in s.answer
            assert "Licensed under" not in s.answer


def test_comment_blocks_are_rejected():
    """A span that is mostly prose commentary is closer to a language prior
    than to the document's content."""
    commented = "\n".join(
        [f"# explanatory note number {i} about the module" for i in range(30)]
        + [f"value_{i} = compute_something(i, {i})" for i in range(30)]
    )
    for seed in range(30):
        for s in generate_reconstruct_samples(
            commented, source_label="r:c.py", rng=random.Random(seed),
        ):
            lines = [ln for ln in s.answer.split("\n") if ln.strip()]
            assert sum(1 for ln in lines if ln.lstrip().startswith("#")) <= len(lines) * 0.5


def test_the_anchor_is_revealed_but_the_target_is_not():
    seen = False
    for seed in range(40):
        for s in _samples(seed):
            if s.qtype != "after_anchor":
                continue
            seen = True
            anchor = s.question.split("`")[1]
            assert anchor in CODE
            # The anchor must not be part of what is asked for.
            assert anchor not in s.answer
    assert seen, "no anchored samples generated"


def test_every_answer_is_a_verbatim_span_of_the_file():
    for seed in range(20):
        for s in _samples(seed):
            assert s.answer in CODE
            assert s.span_text == s.answer


def test_spans_within_one_document_do_not_overlap():
    """The union over a document's questions has to approach the whole file --
    that is what makes a fixed subset of positions unable to satisfy them all,
    and query-blind selection was measured to be OPTIMAL under the old
    families (own-vs-foreign survivor Jaccard 0.967)."""
    for seed in range(20):
        spans = _samples(seed)
        for i, a in enumerate(spans):
            for b in spans[i + 1:]:
                assert a.answer not in b.answer
                assert b.answer not in a.answer


def test_spans_vary_across_documents_of_the_same_shape():
    """A fixed span per document would let the encoder keep just that."""
    starts = {tuple(s.answer[:40] for s in _samples(seed)) for seed in range(20)}
    assert len(starts) > 5


def test_answers_are_long_enough_not_to_be_guessable():
    for seed in range(20):
        for s in _samples(seed):
            assert len(s.answer) >= 120


def test_answers_are_capped_so_one_family_cannot_own_the_loss():
    for seed in range(20):
        for s in _samples(seed, max_chars=400):
            assert len(s.answer) <= 400


def test_blank_and_punctuation_only_spans_are_rejected():
    """A run of blank lines is reproducible without having read anything."""
    blank_heavy = "def f():\n    pass\n" + "\n" * 40 + "def g():\n    pass\n"
    for seed in range(20):
        for s in generate_reconstruct_samples(
            blank_heavy, source_label="r:b.py", rng=random.Random(seed),
        ):
            assert s.answer.strip()
            assert any(c.isalnum() for c in s.answer)


def test_a_file_too_short_to_span_yields_nothing():
    assert generate_reconstruct_samples(
        "x = 1\ny = 2\n", source_label="r:t.py", rng=random.Random(0),
    ) == []


def test_the_sample_count_is_respected():
    for seed in range(10):
        assert len(_samples(seed, n_samples=3)) <= 3


def test_the_header_carries_the_query_like_the_other_families():
    for s in _samples(1):
        assert 'query="' in s.blob_header


@pytest.mark.parametrize("seed", range(5))
def test_generation_is_deterministic_for_a_seed(seed):
    a = [s.answer for s in _samples(seed)]
    b = [s.answer for s in _samples(seed)]
    assert a == b
