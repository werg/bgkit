"""Length-bucketing batch sampler for padded batches.

Each dataset index is already an i.i.d. draw, so the natural order is random;
padding to the longest sample in a batch of log-uniform lengths (64-1024)
wastes a large share of the decoder's compute. This sampler takes windows of
``window_batches * batch_size`` consecutive indices, sorts each window by the
datasets' cheap ``length_hint`` (no tokens loaded), cuts it into batches of
similar length and emits those batches in a seeded random order (so short and
long batches interleave). Deterministic per (seed, epoch): resume-by-skip
works. The old lineage's ``PackedTokenBudgetSampler`` (sum L^2 budget,
variable batch size) is the packed-attention analogue; here the batch size
stays fixed so the padded shapes stay bounded.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        n: int,
        batch_size: int,
        length_hint,
        window_batches: int = 32,
        seed: int = 0,
        drop_last: bool = True,
        max_tokens: int | None = None,
    ) -> None:
        """``max_tokens``: padded token budget per batch (``len(batch) *
        longest hint``); a batch closes when adding the next row would
        exceed it, so long-context rows (8k needle documents, long agentic
        histories) come in small batches and short rows in full ones."""
        if batch_size < 1 or window_batches < 1:
            raise ValueError("batch_size and window_batches must be >= 1")
        self.n, self.batch_size = int(n), int(batch_size)
        self.length_hint = length_hint
        self.window = int(window_batches) * self.batch_size
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.max_tokens = int(max_tokens) if max_tokens else None
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        full = self.n // self.batch_size
        return full if self.drop_last else -(-self.n // self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng([self.seed, self.epoch])
        for start in range(0, self.n, self.window):
            idx = np.arange(start, min(start + self.window, self.n))
            lengths = np.fromiter((self.length_hint(int(i)) for i in idx), dtype=np.int64)
            srt = np.argsort(lengths, kind="stable")
            order, lens = idx[srt], lengths[srt]
            if self.max_tokens is None:
                batches = [
                    order[b : b + self.batch_size].tolist()
                    for b in range(0, order.shape[0], self.batch_size)
                ]
                if self.drop_last and batches and len(batches[-1]) < self.batch_size:
                    batches.pop()
            else:
                batches, cur, cur_max = [], [], 0
                for i, ln in zip(order.tolist(), lens.tolist(), strict=True):
                    m = max(cur_max, int(ln))
                    if cur and (
                        len(cur) >= self.batch_size or (len(cur) + 1) * m > self.max_tokens
                    ):
                        batches.append(cur)
                        cur, cur_max = [], 0
                        m = int(ln)
                    cur.append(i)
                    cur_max = m
                if cur:
                    batches.append(cur)
            for b in rng.permutation(len(batches)):
                yield batches[int(b)]
