"""Guessability floors for a candidate task family, before any GPU time.

WHY THIS IS A SCRIPT AND NOT A NOTEBOOK. Five separate Phase-2 metrics have
read near zero for reasons unrelated to rep-dependence, and two candidate
formulations (lognav, fileneedle) shipped into training before anyone
measured what a model could score WITHOUT the document. fileneedle's zeroed
arm scored 0.241 against a question-echo floor of 0.227: the no-document
model was echoing the prompt, and the family's headline number was measuring
that. A gate set below the governing floor is not a gate.

Two floors, and THE HIGHER ONE GOVERNS:

``echo``       score the QUESTION against its own answer. An answer whose
               tokens appear in its own question is reachable with zero
               retrieval.
``cross_doc``  score answers of OTHER documents against this one. Catches a
               closed answer vocabulary (lognav's 9-element severity enum).

Plus the distributional pair that explains a floor once it is high:
``unique`` (distinct answers / rows) and ``top20`` (share of rows covered by
the 20 most common answers).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from bgkit.eval.metrics.qa_metrics import token_f1


def audit(rows: list[dict], *, seed: int = 0, max_pairs: int = 4000) -> dict:
    """``rows`` are ``{"question": str, "answer": str}`` (``doc_id`` optional)."""
    if not rows:
        raise ValueError("no rows to audit")
    answers = [r["answer"] for r in rows]
    counts = Counter(answers)
    n = len(rows)

    echo = sum(token_f1(r["question"], [r["answer"]]) for r in rows) / n

    # Cross-document: another row's answer, scored against this one. Sampled
    # against a fixed seed so the number is reproducible across candidates.
    rng = random.Random(seed)
    idx = list(range(n))
    pairs = min(max_pairs, n)
    cross = 0.0
    for i in rng.sample(idx, pairs):
        j = rng.randrange(n)
        while j == i and n > 1:
            j = rng.randrange(n)
        cross += token_f1(answers[j], [answers[i]])
    cross /= pairs

    # Most-common-answer baseline: always emit the single modal answer.
    modal, modal_count = counts.most_common(1)[0]
    modal_f1 = sum(token_f1(modal, [a]) for a in answers) / n

    # token_f1 has a FORMAT floor that exact-match does not: every semver
    # string shares "^", "." and digits, every filename shares its extension.
    # Measured on the tabular candidates, the modal answer scored 0.208 by
    # token_f1 and 0.006 by exact match -- the first number is describing the
    # notation, not a prior over the content. Report the floor for the metric
    # the family will actually be scored on.
    modal_em = modal_count / n

    return {
        "n_rows": n,
        "unique_frac": len(counts) / n,
        "top20_coverage": sum(c for _, c in counts.most_common(20)) / n,
        "echo_f1": echo,
        "cross_doc_f1": cross,
        "modal_answer_f1": modal_f1,
        "modal_answer_em": modal_em,
        "gate": max(echo, cross, modal_f1),
        "gate_em": modal_em,
        "mean_answer_chars": sum(len(a) for a in answers) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path, help="rows with question/answer keys")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    rows = [json.loads(x) for x in a.jsonl.read_text().splitlines() if x.strip()]
    res = audit(rows)
    res["label"] = a.label or a.jsonl.stem
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
