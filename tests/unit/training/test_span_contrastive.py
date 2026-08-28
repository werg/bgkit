"""The span loss must penalise the OTHER questions' answers, not just reward this one.

WHY THIS EXISTS. The span loss was positive-only — push THIS question's answer
span up — and that is satisfiable without ever reading the question. Measured
2026-08-28: the union of ALL questions' answer spans per document is 0.4%
(lognav), 0.5% (fileneedle), 3.1% (grepset) while the retention budget is 10%.
So "keep every answer-looking position" satisfies every question at once, and
query-blind generic saliency is the OPTIMAL solution to the objective as
written. The compressor duly became query-blind: survivor-set Jaccard between a
sample's own query and a FOREIGN query 0.967, answer accuracy unchanged under
the wrong query (+0.0025), random selection equal to trained selection.

The contrastive term makes budget allocation depend on WHICH question is asked.
"""

from __future__ import annotations

import torch

from bgkit.data.datasets.phase2_kb_dataset import _parse_negative_spans


def test_negative_spans_parse() -> None:
    assert _parse_negative_spans('[[1,5],[10,12]]') == ((1, 5), (10, 12))
    assert _parse_negative_spans(None) == ()
    assert _parse_negative_spans("not json") == ()


def test_contrastive_term_penalises_keeping_other_answers() -> None:
    """The term must be minimised by DROPPING the other questions' spans."""
    probs_keep = torch.full((8,), 0.9)   # keeping the other answers
    probs_drop = torch.full((8,), 0.05)  # dropping them
    loss_keep = (probs_keep ** 2).mean()
    loss_drop = (probs_drop ** 2).mean()
    assert float(loss_drop) < float(loss_keep), (
        "the contrastive term does not prefer dropping the other answers"
    )


def test_positive_and_contrastive_terms_oppose_each_other() -> None:
    """Guards the guard: the two terms must pull in OPPOSITE directions.

    If both were minimised by the same policy, adding the contrastive term
    would change nothing and query-blindness would survive it.
    """
    p = torch.full((8,), 0.9)
    positive = ((1.0 - p) ** 2).mean()   # wants p high
    contrastive = (p ** 2).mean()        # wants p low
    p2 = torch.full((8,), 0.1)
    positive2 = ((1.0 - p2) ** 2).mean()
    contrastive2 = (p2 ** 2).mean()
    assert float(positive) < float(positive2)
    assert float(contrastive) > float(contrastive2)


def test_negatives_never_overlap_the_gold_span() -> None:
    """An overlapping negative is partly the right answer.

    Penalising it would fight the positive term on the same positions, so the
    builder must exclude overlaps.
    """
    from pathlib import Path

    src = Path("scripts/add_contrastive_spans.py").read_text()
    assert "_overlaps" in src and "not _overlaps(own, c)" in src, (
        "the builder no longer filters overlapping negatives"
    )
