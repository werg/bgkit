"""Text-agreement metrics on token id sequences (no tokenizer needed)."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


def exact_match(pred: Sequence[int], gold: Sequence[int]) -> float:
    return 1.0 if list(pred) == list(gold) else 0.0


def token_f1(pred: Sequence[int], gold: Sequence[int]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    common = Counter(pred) & Counter(gold)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(pred)
    r = n / len(gold)
    return 2 * p * r / (p + r)


def _ngrams(seq: Sequence[int], n: int) -> Counter:
    return Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))


def corpus_bleu(preds: list[Sequence[int]], golds: list[Sequence[int]], max_n: int = 4) -> float:
    """Corpus BLEU-4 with add-one smoothing on higher orders (Lin & Och 2004)."""
    if len(preds) != len(golds) or not preds:
        return 0.0
    match = [0] * max_n
    total = [0] * max_n
    hyp_len = ref_len = 0
    for p, g in zip(preds, golds, strict=True):
        p, g = list(p), list(g)
        hyp_len += len(p)
        ref_len += len(g)
        for n in range(1, max_n + 1):
            pn, gn = _ngrams(p, n), _ngrams(g, n)
            match[n - 1] += sum((pn & gn).values())
            total[n - 1] += max(len(p) - n + 1, 0)
    log_prec = 0.0
    for n in range(max_n):
        m, t = match[n], total[n]
        if n > 0:
            m, t = m + 1, t + 1
        if t == 0 or m == 0:
            return 0.0
        log_prec += math.log(m / t)
    log_prec /= max_n
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return bp * math.exp(log_prec)


def prefix_accuracy(pred: Sequence[int], gold: Sequence[int]) -> float:
    """Fraction of gold tokens reproduced before the first divergence."""
    if not gold:
        return 1.0
    n = 0
    for a, b in zip(pred, gold, strict=False):
        if a != b:
            break
        n += 1
    return n / len(gold)
