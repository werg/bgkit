"""Load article tokens by ``(dataset, document_id)`` from the canonical
Phase 2 mmap layout.

The Phase 2 mmap layout written by ``scripts/convert_hf_to_mmap.py`` and
``scripts/convert_memory_datasets.py`` stores one row per article/document
under::

    {mmap_root}/{dataset}/
        tokens.npy            # int32, flat concatenation of all document tokens
        offsets.npy           # int64, CSR offsets (N+1 long)
        metadata.parquet      # one row per document, columns include document_id
        manifest.json

The ``metadata.parquet`` ``document_id`` column is the stable per-article
key that the rest of the pipeline uses:

- ``PrecomputedL0Cache`` keys on ``document_id``.
- :class:`bgkit.data.l0_cache.L0Cache` keys on ``(dataset, article_id)``
  where ``article_id`` is this same ``document_id``.
- The KB-scale browse trees and teacher trajectories reference the same IDs.

This module exposes :class:`ArticleTokenStore`, a small read-only adapter
that loads ``metadata.parquet`` once per dataset to build a
``document_id → row_index`` map, then slices ``tokens.npy`` via ``offsets.npy``
on demand to return the token IDs for any article by ID.

It is the canonical way to fetch raw article tokens for live-L0 encoding
in the Phase 2 KB-scale trainer and for the offline L0 pre-compute script.
No sidecar ``{dataset}_tokens.parquet`` is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


class _DatasetView:
    """Lazy view over one phase2 mmap dataset."""

    __slots__ = ("_id_to_row", "_mmap_dir", "_offsets", "_tokens")

    def __init__(self, mmap_dir: Path) -> None:
        self._mmap_dir = mmap_dir
        required = ["tokens.npy", "offsets.npy", "metadata.parquet"]
        missing = [f for f in required if not (mmap_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"ArticleTokenStore: {mmap_dir} is missing required files "
                f"{missing}. Run scripts/convert_hf_to_mmap.py (or "
                f"convert_memory_datasets.py) for this dataset first."
            )
        self._tokens: np.ndarray = np.load(mmap_dir / "tokens.npy", mmap_mode="r")
        self._offsets: np.ndarray = np.load(mmap_dir / "offsets.npy")

        # Build document_id → row_index map. metadata.parquet is small
        # enough (one row per document, tens of millions tops for
        # Wikipedia-scale corpora) to load in full.
        table = pq.read_table(mmap_dir / "metadata.parquet", columns=["document_id"])
        ids = table.column("document_id").to_pylist()
        self._id_to_row: dict[str, int] = {
            str(doc_id): i for i, doc_id in enumerate(ids)
        }

    def __len__(self) -> int:
        return len(self._id_to_row)

    def __contains__(self, document_id: str) -> bool:
        return document_id in self._id_to_row

    def get(self, document_id: str) -> torch.Tensor:
        try:
            row = self._id_to_row[document_id]
        except KeyError as exc:
            raise KeyError(
                f"document_id={document_id!r} not found in {self._mmap_dir}"
            ) from exc
        start = int(self._offsets[row])
        end = int(self._offsets[row + 1])
        return torch.from_numpy(self._tokens[start:end].astype(np.int64))

    def length(self, document_id: str) -> int:
        """Token count for one article via the offsets CSR — no token load."""
        row = self._id_to_row[document_id]
        return int(self._offsets[row + 1]) - int(self._offsets[row])

    def document_ids(self) -> list[str]:
        return list(self._id_to_row.keys())


class ArticleTokenStore:
    """Multi-dataset token store keyed by ``(dataset, document_id)``.

    Lazy: each dataset's mmap arrays are loaded on first access. Loading is
    cheap because ``tokens.npy`` is mmap'd, not read — only
    ``metadata.parquet`` (the ID map) is materialized in memory, and that's
    a few MB even at Wikipedia scale.

    The root directory is expected to contain one subdirectory per dataset,
    each with the Phase 2 mmap layout. This matches what
    ``scripts/convert_hf_to_mmap.py`` writes under ``$DATA_DIR/mmap/phase2/``.
    """

    def __init__(self, mmap_root: str | Path) -> None:
        self._root = Path(mmap_root)
        if not self._root.exists():
            raise FileNotFoundError(f"ArticleTokenStore root missing: {self._root}")
        self._views: dict[str, _DatasetView] = {}

    def _view(self, dataset: str) -> _DatasetView:
        view = self._views.get(dataset)
        if view is None:
            view = _DatasetView(self._root / dataset)
            self._views[dataset] = view
        return view

    def has(self, dataset: str, document_id: str) -> bool:
        try:
            return document_id in self._view(dataset)
        except FileNotFoundError:
            return False

    def __contains__(self, key: tuple[str, str]) -> bool:
        dataset, document_id = key
        return self.has(dataset, document_id)

    def get(self, dataset: str, document_id: str) -> torch.Tensor:
        """Return a (L,) int64 tensor of token IDs for the given article."""
        return self._view(dataset).get(document_id)

    def length(self, dataset: str, document_id: str) -> int:
        """Token count for one article without materializing its tokens."""
        return self._view(dataset).length(document_id)

    def get_batch(
        self,
        dataset: str,
        document_ids: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a padded ``(B, max_L)`` token tensor and boolean mask.

        Missing IDs raise KeyError — the caller is responsible for pre-filtering
        with :meth:`has` if some may be absent.
        """
        view = self._view(dataset)
        seqs = [view.get(d) for d in document_ids]
        if not seqs:
            raise ValueError("get_batch called with empty document_id list")
        max_len = max(int(s.size(0)) for s in seqs)
        out = torch.zeros((len(seqs), max_len), dtype=torch.long)
        mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
        for i, s in enumerate(seqs):
            n = int(s.size(0))
            out[i, :n] = s
            mask[i, :n] = True
        return out, mask

    def document_ids(self, dataset: str) -> list[str]:
        return self._view(dataset).document_ids()
