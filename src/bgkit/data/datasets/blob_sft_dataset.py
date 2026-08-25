"""Dataset for faithful blob-format compaction SFT (Family A).

Bridges SWE-Zero-style trajectory shards to trainer-ready samples in the
release blob format (`plans/capability_packaging_2026_08_20.md` §3-§4):

    sample = {
        "token_ids":       (L,) long — chat-template-rendered prefix+target,
        "loss_mask":       (L,) bool — target turn only,
        "sentinel_spans":  [(start, end), ...] — one per blob, to be replaced
                           by projected survivor embeddings at forward time,
        "blob_content_ids": [(N_i,) long, ...] — the compacted span's raw
                           tokens per blob (encoder input),
        "meta": {...}
    }

Deterministic per-index sampling: index -> (row, draw) so epochs revisit
the same trajectory with the same compaction draw (rng seeded by idx).
Rows are read lazily from parquet shards via row-group paging.

The final decoder sequence length after splicing is computable ahead of the
forward under exact-ratio selection:
    L_final = L - sum(span_len_i) + sum(ceil(r_l0*N_i) survivors -> ceil(...))
so token-budget batching can pack correctly before any GPU work.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from bgkit.data.blob_tokenize import tokenize_blob_sample
from bgkit.data.compaction_sampler import CompactionSample, sample_trajectory


def render_span_text(messages: list[dict]) -> str:
    """Plain-text rendering of a message span (encoder input)."""
    parts = []
    for m in messages:
        role = m.get("role", "?")
        content = str(m.get("content") or "")
        tc = m.get("tool_calls") or []
        call_txt = ""
        if tc:
            names = ",".join(str((c.get("function") or {}).get("name", "")) for c in tc)
            call_txt = f" [tool_calls: {names}]"
        parts.append(f"<{role}>{call_txt}\n{content}")
    return "\n".join(parts)


def _parse_span_ref(source_ref: str) -> tuple[int, int]:
    """``traj:{tid}:msgs:{a}-{b}`` -> (a, b)."""
    span = source_ref.rsplit(":", 1)[-1]
    a, b = span.split("-")
    return int(a), int(b)


class BlobSFTDataset(Dataset):
    """Compaction SFT samples drawn from trajectory parquet shards."""

    def __init__(
        self,
        shard_paths: list[str | Path],
        tokenizer,
        *,
        draws_per_trajectory: int = 4,
        max_rows_per_shard: int | None = None,
        max_blob_content_tokens: int = 45_000,
        min_blob_content_tokens: int = 64,
        seed: int = 17,
        repo_holdout_fraction: float = 0.0,
        split: str = "train",
    ) -> None:
        import pyarrow.parquet as pq

        self._tokenizer = tokenizer
        self._draws = draws_per_trajectory
        self._max_blob_tokens = max_blob_content_tokens
        self._min_blob_tokens = min_blob_content_tokens
        self._seed = seed
        # REPO-level train/eval split (2026-08-24): a deterministic crc32
        # bucket of the row's ``repo`` string decides membership, so the same
        # shard list yields disjoint train/eval datasets with NO repo overlap
        # (shard-level holdout leaked repos across shards). ``split="eval"``
        # keeps only held-out repos; ``"train"`` keeps the rest.
        if split not in ("train", "eval"):
            raise ValueError(f"split must be 'train' or 'eval'; got {split!r}")
        if split == "eval" and not repo_holdout_fraction > 0.0:
            raise ValueError(
                "split='eval' requires repo_holdout_fraction > 0 — otherwise "
                "the eval dataset would be silently empty"
            )
        self._holdout = float(repo_holdout_fraction)
        self._split = split
        self._files = [Path(p) for p in shard_paths]
        # index: (file_i, row_group, row_in_group) per trajectory row
        self._index: list[tuple[int, int, int]] = []
        self._pq = pq
        for fi, f in enumerate(self._files):
            pf = pq.ParquetFile(f)
            taken = 0
            for rg in range(pf.num_row_groups):
                n = pf.metadata.row_group(rg).num_rows
                repos: list | None = None
                if self._holdout > 0.0:
                    repos = (
                        pf.read_row_group(rg, columns=["repo"])
                        .column("repo")
                        .to_pylist()
                    )
                for r in range(n):
                    if max_rows_per_shard is not None and taken >= max_rows_per_shard:
                        break
                    if repos is not None and (
                        self._repo_is_eval(str(repos[r] or ""))
                        != (split == "eval")
                    ):
                        continue
                    self._index.append((fi, rg, r))
                    taken += 1
                if max_rows_per_shard is not None and taken >= max_rows_per_shard:
                    break

        self._pf_cache: dict[int, pq.ParquetFile] = {}
        self._rg_cache: tuple[tuple[int, int], list[dict]] | None = None

    def _repo_is_eval(self, repo: str) -> bool:
        import zlib

        return (zlib.crc32(repo.encode("utf-8")) % 1000) < int(self._holdout * 1000)

    def __len__(self) -> int:
        return len(self._index) * self._draws

    def _row(self, fi: int, rg: int, r: int) -> dict:
        key = (fi, rg)
        if self._rg_cache is None or self._rg_cache[0] != key:
            pf = self._pf_cache.get(fi)
            if pf is None:
                pf = self._pq.ParquetFile(self._files[fi])
                self._pf_cache[fi] = pf
            self._rg_cache = (key, pf.read_row_group(rg).to_pylist())
        return self._rg_cache[1][r]

    def __getitem__(self, idx: int) -> dict | None:
        row_idx, draw = divmod(idx, self._draws)
        fi, rg, r = self._index[row_idx]
        row = self._row(fi, rg, r)
        msgs = row["trajectory"]
        msgs = json.loads(msgs) if isinstance(msgs, str) else msgs
        tid = str(row.get("trajectory_id") or f"{fi}:{rg}:{r}")
        rng = random.Random(f"{self._seed}:{tid}:{draw}")
        samples = sample_trajectory(msgs, trajectory_id=tid, rng=rng, samples_per_trajectory=1)
        if not samples:
            return None
        # sample_trajectory APPENDS the recall-probe variant after the base
        # continuation sample when its probe roll fires; taking [0]
        # unconditionally dropped every probe (found 2026-08-25: the
        # blob_sft_v1 step-250 eval was 128/128 continuation — the designed
        # probe gate was silently absent). Prefer the probe when present:
        # other draws of the same trajectory supply the continuations.
        sample: CompactionSample = samples[-1]
        try:
            rendered = tokenize_blob_sample(self._tokenizer, sample)
        except ValueError:
            return None

        blob_content: list[torch.Tensor] = []
        for blob in sample.blobs:
            a, b = _parse_span_ref(blob.source_ref)
            text = render_span_text(msgs[a:b])
            ids = self._tokenizer.encode(text, add_special_tokens=False)
            if not (self._min_blob_tokens <= len(ids) <= self._max_blob_tokens):
                return None
            blob_content.append(torch.tensor(ids, dtype=torch.long))

        return {
            "token_ids": rendered.token_ids,
            "loss_mask": rendered.loss_mask,
            "sentinel_spans": rendered.blob_sentinel_spans,
            "blob_content_ids": blob_content,
            "meta": {
                "trajectory_id": tid,
                "repo": str(row.get("repo") or ""),
                "qtype": sample.qtype,
                "mode": sample.mode,
                "n_blobs": len(sample.blobs),
            },
        }


def spliced_length(sample: dict, *, l0_ratio: float, l1_ratio: float) -> int:
    """Final decoder sequence length after replacing sentinels with survivors.

    Exact under exact-topk selection: per blob, k = ceil(l1_ratio *
    ceil(l0_ratio * N)). Used by token-budget batching before any GPU work.
    """
    import math

    total = int(sample["token_ids"].shape[0])
    for (a, b), content in zip(sample["sentinel_spans"], sample["blob_content_ids"], strict=True):
        survivors = math.ceil(l1_ratio * math.ceil(l0_ratio * int(content.shape[0])))
        total += survivors - (b - a)
    return total
