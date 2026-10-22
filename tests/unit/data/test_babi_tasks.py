"""bAbI parsing + BABILong-style haystack construction (2026-08-25)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from bgkit.data.babi_tasks import (
    build_haystack,
    format_answer,
    parse_babi,
)

BABI = """1 Mary moved to the bathroom.
2 John went to the hallway.
3 Where is Mary? \tbathroom\t1
4 Daniel went back to the hallway.
5 Where is Daniel? \thallway\t4
1 Sandra journeyed to the bedroom.
2 Sandra got the apple there.
3 Sandra travelled to the hallway.
4 Where is the apple? \thallway\t3 2
"""


def test_parse_babi_restarts_stories_and_resolves_supporting_lines():
    samples = parse_babi(BABI, "qa1")
    assert len(samples) == 3
    first, second, third = samples
    # Question lines are NOT facts, so the supporting line numbers must map
    # through to fact indices rather than being used directly.
    assert first.facts == ["Mary moved to the bathroom.", "John went to the hallway."]
    assert first.supporting == [0] and first.answer == "bathroom"
    assert second.facts[-1] == "Daniel went back to the hallway."
    assert second.supporting == [2]  # line 4 is the THIRD fact (line 3 was a question)
    # Line 1 restarts the story: the third sample must not see Mary/John.
    assert third.facts == [
        "Sandra journeyed to the bedroom.",
        "Sandra got the apple there.",
        "Sandra travelled to the hallway.",
    ]
    assert third.supporting == [2, 1] and third.answer == "hallway"


def test_format_answer_matches_the_post_prompt_sentence():
    assert (
        format_answer("qa1", "Where is John?", "hallway")
        == "The most recent location of John is hallway."
    )
    assert format_answer("qa2", "Where is the milk?", "kitchen") == "The milk is in kitchen."
    with pytest.raises(ValueError):
        format_answer("qa3", "q", "a")


def test_build_haystack_spans_recover_the_inserted_runs_exactly():
    """The span-relevance signal is only as good as these offsets: every
    reported span must slice back to exactly the run that was inserted."""
    noise = np.arange(100, 200, dtype=np.int32)
    allowed = np.zeros(256, dtype=bool)
    allowed[100:200:5] = True  # every 5th noise token starts a word
    runs = [np.array([7, 8, 9], dtype=np.int32), np.array([11, 12], dtype=np.int32)]
    hay, spans = build_haystack(noise, runs, allowed, random.Random(0))

    assert hay.size == noise.size + sum(r.size for r in runs)
    for run, (a, b) in zip(runs, spans, strict=True):
        assert hay[a:b].tolist() == run.tolist()
    # Noise order is preserved: dropping the inserted spans restores it.
    keep = np.ones(hay.size, dtype=bool)
    for a, b in spans:
        keep[a:b] = False
    assert hay[keep].tolist() == noise.tolist()


def test_build_haystack_refuses_when_no_word_start_positions():
    noise = np.arange(10, dtype=np.int32)
    allowed = np.zeros(256, dtype=bool)
    with pytest.raises(ValueError, match="word-start positions"):
        build_haystack(noise, [np.array([1], dtype=np.int32)], allowed, random.Random(0))
