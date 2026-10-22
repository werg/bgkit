from __future__ import annotations

import torch

from bgkit2.training.config import DataCfg
from bgkit2.training.core import BgKIT2Core


class _Out:
    def __init__(self, reps, mask):
        self.reps, self.rep_mask = reps, mask


class _Core(BgKIT2Core):
    """Only encode_extra is exercised; the encoder records what it was asked for."""

    def __init__(self, budget: int):
        self.calls: list[tuple[int, int]] = []
        cfg = type("C", (), {})()
        cfg.data = DataCfg()
        cfg.data.extra_encode_token_sq_budget = budget
        self.cfg2 = cfg

    def encoder(self, ids, mask, ratio=None, selection=None, prompt_ids=None, prompt_mask=None):
        self.calls.append((int(ids.shape[0]), int(ids.shape[1])))
        lens = mask.sum(-1)
        reps = torch.stack([torch.full((4, 8), float(lens[i])) for i in range(ids.shape[0])])
        return _Out(reps, torch.ones(ids.shape[0], 4))

    def finish_reps(self, ids, out):
        return out


def _batch(lens: list[int], owners: list[int], width: int = 8192):
    b = type("B", (), {})()
    b.extra_input_ids = torch.zeros(len(lens), width, dtype=torch.long)
    b.extra_attention_mask = torch.zeros(len(lens), width, dtype=torch.long)
    for i, n in enumerate(lens):
        b.extra_attention_mask[i, :n] = 1
    b.extra_prompt_ids = b.extra_prompt_mask = None
    b.extra_owner = torch.tensor(owners)
    b.tasks = ["a"] * (max(owners) + 1)
    return b


def test_one_long_slot_does_not_pad_the_short_ones():
    lens = [8000, 200, 180, 300, 7900, 150]
    core = _Core(budget=200_000_000)
    core.encode_extra(_batch(lens, [0, 0, 0, 1, 1, 1]), 0.05)
    assert all(rows * w * w <= 200_000_000 for rows, w in core.calls)
    assert max(w for _, w in core.calls) <= max(lens)
    assert len(core.calls) > 1


def test_slots_reach_splice_in_their_original_order():
    lens = [8000, 200, 180, 300, 7900, 150]
    core = _Core(budget=200_000_000)
    per = core.encode_extra(_batch(lens, [0, 0, 0, 1, 1, 1]), 0.05)
    got = [[int(r[0, 0].item()) for r, _ in sample] for sample in per]
    assert got == [lens[0:3], lens[3:6]]


def test_a_tiny_budget_still_makes_progress_one_row_at_a_time():
    core = _Core(budget=1)
    per = core.encode_extra(_batch([4000, 4000, 4000], [0, 0, 0]), 0.05)
    assert [rows for rows, _ in core.calls] == [1, 1, 1]
    assert len(per[0]) == 3


def test_no_extra_slots_returns_none():
    b = type("B", (), {})()
    b.extra_input_ids = None
    assert _Core(budget=10).encode_extra(b, 0.05) is None
