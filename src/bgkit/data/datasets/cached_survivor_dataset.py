"""Cached survivor dataset for Falcon dense-seed projection training.

Reads the output of ``scripts/build_dense_seed_cache.py`` (a static cache of
encoder-l0 survivor embeddings for a subsample of chunks) via mmap and yields
per-chunk dicts compatible with a projection-only training loop. The cache is
deterministic w.r.t. (source encoder checkpoint, source companion mmap).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedSurvivorDataset(Dataset):
    """Indexable mmap'd view of cached pre-projection survivor embeddings."""

    def __init__(self, cache_dir: str | Path):
        self._cache_path = Path(cache_dir)
        manifest_path = self._cache_path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Falcon dense-seed cache manifest not found: {manifest_path}. "
                "Build with scripts/build_dense_seed_cache.py first."
            )
        self.manifest = json.loads(manifest_path.read_text())
        self.n_chunks = int(self.manifest["n_chunks"])
        self.hidden_dim = int(self.manifest["hidden_dim"])
        if self.manifest.get("embedding_dtype") != "bfloat16":
            raise ValueError(
                "Cache was built with non-bfloat16 embeddings; rebuild required."
            )

        self._survivor_offsets = np.load(
            self._cache_path / "survivor_offsets.npy", mmap_mode="r"
        )
        if self._survivor_offsets.shape[0] != self.n_chunks + 1:
            raise ValueError(
                f"survivor_offsets length {self._survivor_offsets.shape[0]} "
                f"!= n_chunks+1 ({self.n_chunks + 1})"
            )
        total_survivors = int(self._survivor_offsets[-1])
        if total_survivors != int(self.manifest.get("total_survivors", 0)):
            raise ValueError(
                "survivor_offsets total does not match manifest total_survivors"
            )

        # bf16 bytes mmapped as uint16 (no native numpy bf16); converted to
        # torch.bfloat16 via tensor view on access.
        self._embeddings_u16 = np.memmap(
            self._cache_path / "survivor_embeddings.bin",
            mode="r",
            dtype=np.uint16,
            shape=(total_survivors, self.hidden_dim),
        )
        self._target_pair_ids = np.load(
            self._cache_path / "target_pair_ids.npy", mmap_mode="r"
        )
        self._pair_loss_mask = np.load(
            self._cache_path / "pair_loss_mask.npy", mmap_mode="r"
        )
        self._alignment_scores = np.load(
            self._cache_path / "alignment_scores.npy", mmap_mode="r"
        )

        # Per-chunk survivor count = offsets diff. Used by samplers.
        self._lengths = (
            self._survivor_offsets[1:] - self._survivor_offsets[:-1]
        ).astype(np.int64)

    @property
    def lengths(self) -> np.ndarray:
        """Per-chunk survivor counts (the natural token-budget cost unit)."""
        return self._lengths

    def __len__(self) -> int:
        return self.n_chunks

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_embeddings_u16"] = None
        state["_survivor_offsets"] = None
        state["_target_pair_ids"] = None
        state["_pair_loss_mask"] = None
        state["_alignment_scores"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-mmap in worker processes after fork.
        self._survivor_offsets = np.load(
            self._cache_path / "survivor_offsets.npy", mmap_mode="r"
        )
        total_survivors = int(self._survivor_offsets[-1])
        self._embeddings_u16 = np.memmap(
            self._cache_path / "survivor_embeddings.bin",
            mode="r",
            dtype=np.uint16,
            shape=(total_survivors, self.hidden_dim),
        )
        self._target_pair_ids = np.load(
            self._cache_path / "target_pair_ids.npy", mmap_mode="r"
        )
        self._pair_loss_mask = np.load(
            self._cache_path / "pair_loss_mask.npy", mmap_mode="r"
        )
        self._alignment_scores = np.load(
            self._cache_path / "alignment_scores.npy", mmap_mode="r"
        )

    def __getitem__(self, idx: int) -> dict:
        if not 0 <= idx < self.n_chunks:
            raise IndexError(idx)
        lo = int(self._survivor_offsets[idx])
        hi = int(self._survivor_offsets[idx + 1])
        # Copy the slice (mmap → owned) so DataLoader workers can return it.
        emb_u16 = np.array(self._embeddings_u16[lo:hi], dtype=np.uint16)
        # Reinterpret the same bytes as bfloat16 via a torch tensor view.
        emb_tensor = torch.from_numpy(emb_u16).view(torch.bfloat16)
        # pair_ids / pair_loss_mask are stored as (N_survivors, 2) in the cache,
        # matching projection_block.output_split_factor=2: each surviving Qwen
        # position decodes into 2 Falcon-vocab tokens. The collator flattens.
        pair_ids = torch.from_numpy(
            np.array(self._target_pair_ids[lo:hi], dtype=np.int64)
        )
        pair_mask = torch.from_numpy(
            np.array(self._pair_loss_mask[lo:hi], dtype=bool)
        )
        align = torch.from_numpy(
            np.array(self._alignment_scores[lo:hi], dtype=np.float32)
        )
        return {
            "survivor_embeddings": emb_tensor,
            "target_pair_ids": pair_ids,
            "pair_loss_mask": pair_mask,
            "alignment_scores": align,
            "survivor_count": int(hi - lo),
        }


def collate_cached_dense_seed(batch: list[dict]) -> dict:
    """Pack a list of cached per-chunk samples into a flat batch.

    Returns a dict with:
      - ``survivor_embeddings``: ``(N_total, hidden_dim)`` bf16
      - ``cu_seqlens``: ``(B+1,)`` int32 — per-chunk boundaries
      - ``target_pair_ids``: ``(2*N_total,)`` int64 (Falcon vocab ids)
      - ``pair_loss_mask``: ``(2*N_total,)`` bool
      - ``alignment_scores``: ``(N_total,)`` float32
      - ``survivor_counts``: ``(B,)`` int64
    """
    if not batch:
        raise ValueError("collate_cached_dense_seed received an empty batch")

    emb = torch.cat([s["survivor_embeddings"] for s in batch], dim=0)
    # (N_total, 2) → (2*N_total,) so it aligns 1:1 with the projection block's
    # split=2 output (each survivor → 2 falcon-dim output rows).
    pair_ids = torch.cat(
        [s["target_pair_ids"] for s in batch], dim=0
    ).reshape(-1)
    pair_mask = torch.cat(
        [s["pair_loss_mask"] for s in batch], dim=0
    ).reshape(-1)
    align = torch.cat([s["alignment_scores"] for s in batch], dim=0)

    counts = torch.tensor(
        [s["survivor_count"] for s in batch], dtype=torch.int64
    )
    cu = torch.zeros(len(batch) + 1, dtype=torch.int32)
    torch.cumsum(counts.to(torch.int32), 0, out=cu[1:])

    return {
        "survivor_embeddings": emb,
        "cu_seqlens": cu,
        "target_pair_ids": pair_ids,
        "pair_loss_mask": pair_mask,
        "alignment_scores": align,
        "survivor_counts": counts,
    }
