"""Dataset for commit reproduction training: token IDs from memory-mapped numpy arrays.

Replaces the parquet-based CommitReproDataset. Workers share token data via OS
page cache instead of each building independent Arrow table caches.

Important: Commits are NOT chunked. One sample = one commit, truncated to
max_seq_len. This is a raw-content dataset — for Phase 1 Step 2, it will need
a chat-template wrapper similar to ChatReproDataset.
"""

from __future__ import annotations

from bgkit.data.datasets.base_mmap_dataset import BaseMmapDataset


class CommitReproDataset(BaseMmapDataset):
    """Dataset yielding token ID sequences for commit reproduction training.

    Loads pre-converted commit data (tokens.npy, offsets.npy, manifest.json)
    for memory-efficient random access. Each sample is one serialized commit,
    truncated to ``max_seq_len``.

    Workers share the mmap'd token array via OS page cache. Pickle excludes
    the mmap; workers re-open from the same path.
    """

    CONVERT_HINT = (
        "Convert with: python scripts/convert_commits_to_npy.py "
        "--input-dir <data_dir>"
    )

    def __init__(self, data_dir: str, max_seq_len: int = 4096):
        super().__init__(data_dir, max_seq_len=max_seq_len)
