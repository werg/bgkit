from __future__ import annotations

import torch

from bgkit2.training.config import KDCfg
from bgkit2.training.core import BgKIT2Core


class _Core(BgKIT2Core):
    def __init__(self, cap: int):
        cfg = type("C", (), {})()
        cfg.kd = KDCfg()
        cfg.kd.selection_max_ctx_tokens = cap
        self.cfg2 = cfg


def _batch(lens: list[int], owners: list[int]):
    b = type("B", (), {})()
    b.attention_mask = torch.zeros(len(lens), max(lens), dtype=torch.long)
    for i, n in enumerate(lens):
        b.attention_mask[i, :n] = 1
    b.tasks = ["t"] * len(lens)
    b.extra_owner = torch.tensor(owners) if owners else torch.zeros(0, dtype=torch.long)
    return b


def test_none_when_every_row_is_eligible():
    assert _Core(4096).selection_rows(_batch([100, 200, 300], [])) is None


def test_long_context_rows_are_dropped():
    assert _Core(4096).selection_rows(_batch([100, 5000, 300], [])) == [0, 2]


def test_multi_slot_rows_are_dropped_whatever_their_context_length():
    # the eager teacher pass is (chunk, H, T, T) over the DECODER sequence, and a
    # multi-slot row's T is prefix + every slot; selection_loss also only ever
    # targeted slot 0, so these rows never had a usable signal
    assert _Core(4096).selection_rows(_batch([100, 200, 300], [0, 0, 2])) == [1]


def test_multi_slot_rows_are_dropped_even_with_the_cap_off():
    assert _Core(0).selection_rows(_batch([100, 200, 300], [1])) == [0, 2]
