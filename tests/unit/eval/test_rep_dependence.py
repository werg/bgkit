"""The gap statistic must distinguish the two cases a pooled mean cannot.

Pooled CE has been actively misleading in this project: on the Phase-2 replay
task, CE fell 2.42 -> 2.17 while the rep gap SHRANK 0.039 -> 0.025. A mean near
zero is consistent with BOTH "reps are useless" and "reps are load-bearing for
a minority of content tokens", and those demand opposite conclusions. These
tests pin that the summary separates them.
"""

from __future__ import annotations

import torch

from bgkit.eval.rep_dependence import (
    per_token_ce,
    split_by_source_overlap,
    summarize_gap,
)


def test_useless_reps_and_minority_dependence_have_the_same_mean() -> None:
    """The case that motivates the whole module."""
    n = 1000
    # Both arms are constructed to have EXACTLY the same pooled mean (0.10),
    # which is the point: the mean cannot tell them apart.
    # (a) reps do nothing, but a uniform 0.10 of noise-level help everywhere.
    useless = summarize_gap(torch.ones(n), torch.ones(n) + 0.10)
    # (b) reps carry 2% of tokens by 5 nats each, nothing elsewhere. 0.02*5 = 0.10.
    gap_b = torch.zeros(n)
    gap_b[:20] = 5.0
    minority = summarize_gap(torch.ones(n), torch.ones(n) + gap_b)
    assert abs(float(gap_b.mean()) - 0.10) < 1e-6

    assert abs(useless.gap_mean - minority.gap_mean) < 0.01   # SAME pooled mean
    # ...and the distribution tells them apart decisively.
    assert useless.gap_p99 < 0.2
    assert minority.gap_p99 > 4.0
    assert minority.frac_gap_over_2p0 > 0.015
    assert useless.frac_gap_over_2p0 == 0.0


def test_top_decile_mean_surfaces_a_thin_tail() -> None:
    """1% of tokens at 10 nats is invisible in the mean, obvious in the tail."""
    n = 1000
    gap = torch.zeros(n)
    gap[:10] = 10.0
    s = summarize_gap(torch.ones(n), torch.ones(n) + gap)
    assert s.gap_mean < 0.11
    assert s.gap_top_decile_mean > 0.9


def test_negative_gap_is_reported_not_clipped() -> None:
    """Reps actively HURTING is a real outcome (v8 measured a negative
    exact-match rep_gain) and must not be hidden."""
    n = 100
    s = summarize_gap(torch.ones(n), torch.ones(n) - 0.5)
    assert s.gap_mean < 0
    assert s.frac_gap_negative == 1.0


def test_empty_input_is_safe() -> None:
    s = summarize_gap(torch.zeros(0), torch.zeros(0))
    assert s.n_tokens == 0


def test_per_token_ce_matches_manual_cross_entropy() -> None:
    """Chunked logsumexp must equal the textbook value — this replaces the
    (S, V) logits materialisation, so it has to be exactly right."""
    torch.manual_seed(0)
    n_pos, dim, vocab = 12, 8, 40
    h = torch.randn(n_pos, dim)
    lm = torch.nn.Linear(dim, vocab, bias=False)
    tok = torch.randint(0, vocab, (n_pos,))
    mask = torch.zeros(n_pos, dtype=torch.bool)
    mask[5:10] = True

    ce, pos = per_token_ce(h, tok, mask, lm, chunk=2)

    logits = lm(h[:-1]).float()
    ref = torch.nn.functional.cross_entropy(
        logits, tok[1:], reduction="none",
    )[mask[1:]]
    assert torch.allclose(ce, ref, atol=1e-5)
    assert int(pos.numel()) == int(mask[1:].sum())


def test_per_token_ce_respects_the_shift() -> None:
    """hidden at t predicts token t+1. An off-by-one here would silently
    score the wrong token and produce a plausible-looking wrong answer."""
    torch.manual_seed(0)
    n_pos, dim, vocab = 6, 4, 10
    h = torch.randn(n_pos, dim)
    lm = torch.nn.Linear(dim, vocab, bias=False)
    tok = torch.arange(n_pos) % vocab
    mask = torch.zeros(n_pos, dtype=torch.bool)
    mask[3] = True
    _ce, pos = per_token_ce(h, tok, mask, lm)
    assert pos.tolist() == [2]          # index into the shifted array


def test_source_overlap_excludes_common_ids() -> None:
    """Without a stopword exclusion the split is swamped by function words."""
    tgt = torch.tensor([1, 2, 3, 4])
    src = torch.tensor([1, 2, 9])
    assert split_by_source_overlap(tgt, src).tolist() == [True, True, False, False]
    assert split_by_source_overlap(tgt, src, common_ids={1}).tolist() == [
        False, True, False, False,
    ]
