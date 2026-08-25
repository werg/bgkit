"""bAbI task parsing + BABILong-style haystack construction (2026-08-25).

Source of the *signal* for the ``babilong_style`` training family: bAbI's
**train** split, read straight out of the ``tasks_1-20_v1-2.zip`` that ships
with the babilong repo. The BABILong benchmark draws its samples from the
bAbI **test** split embedded in PG19 book text, so training here touches
neither the benchmark samples nor its noise distribution — see
``scripts/build_babilong_style_phase2.py`` for the full contamination
boundary.

bAbI line format, per story (line numbering restarts at 1)::

    1 Mary moved to the bathroom.
    2 John went to the hallway.
    3 Where is Mary? \tbathroom\t1

A question line is any line containing a tab; its trailing fields are the
answer and the 1-based line numbers of the *supporting* facts.
"""

from __future__ import annotations

import random
import zipfile
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BabiSample:
    """One question with the facts stated before it in its story."""

    task: str
    facts: list[str]
    question: str
    answer: str
    # Indices INTO ``facts`` of the supporting statements (may be empty when
    # a supporting line was itself a question line, which bAbI never emits).
    supporting: list[int]


def parse_babi(text: str, task: str) -> list[BabiSample]:
    """Parse one bAbI task file into per-question samples."""
    samples: list[BabiSample] = []
    facts: list[str] = []
    line_to_fact: dict[int, int] = {}
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        number, _, rest = raw.partition(" ")
        try:
            line_no = int(number)
        except ValueError:
            continue
        if line_no == 1:
            facts = []
            line_to_fact = {}
        if "\t" in rest:
            question, answer, support = (part.strip() for part in rest.split("\t")[:3])
            supporting = [
                line_to_fact[int(tok)]
                for tok in support.replace(",", " ").split()
                if tok.isdigit() and int(tok) in line_to_fact
            ]
            samples.append(
                BabiSample(
                    task=task,
                    facts=list(facts),
                    question=question,
                    answer=answer,
                    supporting=supporting,
                )
            )
        else:
            line_to_fact[line_no] = len(facts)
            facts.append(rest)
    return samples


def read_babi_zip(zip_path: str, task: str, split: str = "train") -> list[BabiSample]:
    """Read ``tasks_1-20_v1-2.zip``'s ``en-10k`` file for ``task``."""
    with zipfile.ZipFile(zip_path) as zf:
        matches = [
            name
            for name in zf.namelist()
            if f"/en-10k/{task}_" in name and name.endswith(f"_{split}.txt")
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one en-10k {split} file for {task}, got {matches}")
        return parse_babi(zf.read(matches[0]).decode("utf-8"), task)


# --- answer templating -----------------------------------------------------
#
# BABILong's post_prompt demands a fixed answer sentence per task, and the
# few-shot examples demonstrate it. Training on the bare bAbI answer while
# evaluating under that instruction would be a train/eval prompt mismatch, so
# the training target is rendered in the demanded format.


def _strip_question(question: str, prefixes: tuple[str, ...]) -> str:
    q = question.strip().rstrip("?").strip()
    low = q.lower()
    for prefix in prefixes:
        if low.startswith(prefix):
            return q[len(prefix) :].strip()
    return q


def format_answer(task: str, question: str, answer: str) -> str:
    """Render the bAbI answer in the sentence BABILong's post_prompt demands."""
    if task == "qa1":
        # "Where is John?" -> "The most recent location of John is hallway."
        person = _strip_question(question, ("where is ",))
        return f"The most recent location of {person} is {answer}."
    if task == "qa2":
        # "Where is the milk?" -> "The milk is in kitchen."
        item = _strip_question(question, ("where is the ", "where is "))
        return f"The {item} is in {answer}."
    raise ValueError(f"no answer template for task {task!r}")


SUPPORTED_TASKS = ("qa1", "qa2")


def word_start_mask(tokenizer) -> np.ndarray:
    """Per-vocab-id flag: does this token begin a new word?

    Insertions are only allowed before such tokens so a fact sentence never
    lands mid-word in the noise.
    """
    vocab_size = len(tokenizer)
    tokens = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    return np.asarray(
        [bool(t) and (t[0] in "ĠĊ") for t in tokens],
        dtype=bool,
    )


def build_haystack(
    noise: np.ndarray,
    fact_token_runs: list[np.ndarray],
    allowed: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Insert each fact run into ``noise`` at a distinct word-start position.

    Returns the haystack and, per fact, its ``[start, end)`` span in it.
    """
    candidates = np.flatnonzero(allowed[noise])
    candidates = candidates[(candidates > 0) & (candidates < noise.size)]
    k = len(fact_token_runs)
    if candidates.size < k:
        raise ValueError(f"only {candidates.size} word-start positions for {k} facts")
    picks = sorted(rng.sample(list(candidates.tolist()), k))

    pieces: list[np.ndarray] = []
    spans: list[tuple[int, int]] = []
    out_len = 0
    prev = 0
    for pos, run in zip(picks, fact_token_runs, strict=True):
        chunk = noise[prev:pos]
        pieces.append(chunk)
        out_len += int(chunk.size)
        spans.append((out_len, out_len + int(run.size)))
        pieces.append(run)
        out_len += int(run.size)
        prev = pos
    pieces.append(noise[prev:])
    return np.concatenate(pieces), spans
