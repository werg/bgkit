"""Namespaced L0 survivor cache for the Phase 2 KB-scale pipeline.

Unlike :class:`bgkit.data.datasets.precomputed_l0_cache.PrecomputedL0Cache`
(which stores one dataset per directory and keys on ``document_id`` alone),
this cache is built to hold many datasets side-by-side and key on
``(dataset, article_id)`` so that a Wikipedia article and a git commit with
accidentally-colliding IDs never trample each other.

Layout under ``cache_dir``::

    cache_dir/
      dataset_a/
        shard_0000/
          survivors.npy    # (N_total, D) float16 concatenation of all rows
          offsets.npy      # (N_rows + 1,) int64 row boundaries
        shard_0001/...
        index.parquet      # article_id → (shard_id, row_index)
      dataset_b/...

Multiple shards per dataset keep individual files small enough for random
access; the trainer only loads a shard when it needs an article from it.
Shards are immutable after write, so extending the cache across stages is
just "add more shards + rewrite index.parquet".
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


@dataclass(frozen=True)
class _Entry:
    shard_id: str
    row_index: int


class L0Cache:
    """Read-only view over a multi-dataset L0 survivor cache."""

    def __init__(self, cache_dir: str | Path):
        self._cache_dir = Path(cache_dir)
        if not self._cache_dir.exists():
            raise FileNotFoundError(f"L0 cache dir missing: {self._cache_dir}")
        self._indices: dict[str, dict[str, _Entry]] = {}
        self._shards: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Dataset registration
    # ------------------------------------------------------------------

    def load_dataset(self, dataset: str) -> None:
        """Load the parquet index for ``dataset`` into memory."""
        if dataset in self._indices:
            return
        idx_path = self._cache_dir / dataset / "index.parquet"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"L0 cache index missing for dataset {dataset!r}: {idx_path}"
            )
        table = pq.read_table(idx_path).to_pylist()
        d: dict[str, _Entry] = {}
        for row in table:
            d[str(row["article_id"])] = _Entry(
                shard_id=str(row["shard_id"]),
                row_index=int(row["row_index"]),
            )
        self._indices[dataset] = d

    def has(self, dataset: str, article_id: str) -> bool:
        if dataset not in self._indices:
            try:
                self.load_dataset(dataset)
            except FileNotFoundError:
                return False
        return article_id in self._indices[dataset]

    def __contains__(self, key: tuple[str, str]) -> bool:
        dataset, article_id = key
        return self.has(dataset, article_id)

    def __len__(self) -> int:
        total = 0
        for d in self._indices.values():
            total += len(d)
        return total

    # ------------------------------------------------------------------
    # Shard loading
    # ------------------------------------------------------------------

    def _load_shard(self, dataset: str, shard_id: str) -> dict[str, np.ndarray]:
        key = (dataset, shard_id)
        cached = self._shards.get(key)
        if cached is not None:
            return cached
        shard_dir = self._cache_dir / dataset / shard_id
        arrays = {
            "survivors": np.load(shard_dir / "survivors.npy", mmap_mode="r"),
            "offsets": np.load(shard_dir / "offsets.npy"),
        }
        self._shards[key] = arrays
        return arrays

    def _lookup(self, dataset: str, article_id: str) -> tuple[dict[str, np.ndarray], int]:
        if dataset not in self._indices:
            self.load_dataset(dataset)
        try:
            entry = self._indices[dataset][article_id]
        except KeyError as exc:
            raise KeyError(
                f"L0 cache miss: dataset={dataset!r} article_id={article_id!r}"
            ) from exc
        return self._load_shard(dataset, entry.shard_id), entry.row_index

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, dataset: str, article_id: str) -> torch.Tensor:
        """Return the survivor rows for one article as a (K, D) tensor."""
        arrays, row_index = self._lookup(dataset, article_id)
        offsets = arrays["offsets"]
        start = int(offsets[row_index])
        end = int(offsets[row_index + 1])
        rows = np.asarray(arrays["survivors"][start:end])
        return torch.from_numpy(np.array(rows))

    def get_batch(
        self,
        dataset: str,
        article_ids: Iterable[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a padded (B, max_K, D) tensor and (B, max_K) mask.

        Missing articles raise KeyError — the caller is expected to filter
        first via :meth:`has`.
        """
        rows: list[torch.Tensor] = [self.get(dataset, aid) for aid in article_ids]
        if not rows:
            raise ValueError("get_batch called with empty article list")
        max_k = max(r.size(0) for r in rows)
        hidden = rows[0].size(-1)
        batch = torch.zeros((len(rows), max_k, hidden), dtype=rows[0].dtype)
        mask = torch.zeros((len(rows), max_k), dtype=torch.bool)
        for i, r in enumerate(rows):
            k = r.size(0)
            batch[i, :k] = r
            mask[i, :k] = True
        return batch, mask


class L0CacheWriter:
    """Builder for a new shard inside an :class:`L0Cache`.

    Usage::

        writer = L0CacheWriter(cache_dir, dataset="kilt_wikipedia", shard_id="shard_0000")
        for aid, survivors in iter_articles():
            writer.add(aid, survivors)
        writer.finalize()
    """

    def __init__(
        self,
        cache_dir: str | Path,
        dataset: str,
        shard_id: str,
    ) -> None:
        self._root = Path(cache_dir) / dataset / shard_id
        self._root.mkdir(parents=True, exist_ok=True)
        self._article_ids: list[str] = []
        self._rows: list[np.ndarray] = []

    def add(self, article_id: str, survivors: np.ndarray | torch.Tensor) -> None:
        if isinstance(survivors, torch.Tensor):
            survivors = survivors.detach().cpu().float().numpy()
        if survivors.ndim != 2:
            raise ValueError(
                f"add: expected (K, D), got shape {survivors.shape}"
            )
        self._article_ids.append(article_id)
        self._rows.append(survivors.astype(np.float16))

    def finalize(self) -> tuple[int, list[tuple[str, int]]]:
        """Write this shard to disk, return (num_articles, index_rows).

        The index rows are ``(article_id, row_index)`` pairs that the caller
        must pass to :func:`update_dataset_index` together with the
        ``shard_id``.
        """
        if not self._rows:
            return 0, []
        offsets = np.zeros(len(self._rows) + 1, dtype=np.int64)
        for i, r in enumerate(self._rows):
            offsets[i + 1] = offsets[i] + r.shape[0]
        hidden = self._rows[0].shape[1]
        total = int(offsets[-1])
        flat = np.zeros((total, hidden), dtype=np.float16)
        for i, r in enumerate(self._rows):
            flat[int(offsets[i]):int(offsets[i + 1])] = r
        np.save(self._root / "survivors.npy", flat)
        np.save(self._root / "offsets.npy", offsets)
        index_rows = [(aid, i) for i, aid in enumerate(self._article_ids)]
        return len(self._rows), index_rows


def update_dataset_index(
    cache_dir: str | Path,
    dataset: str,
    shard_id: str,
    index_rows: list[tuple[str, int]],
) -> None:
    """Append or refresh a shard's contribution to ``dataset/index.parquet``.

    Ensures entries for this shard_id are replaced rather than duplicated,
    so that re-running :class:`L0CacheWriter` on the same shard is idempotent.
    """
    idx_path = Path(cache_dir) / dataset / "index.parquet"
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if idx_path.exists():
        existing = pq.read_table(idx_path).to_pylist()
        existing = [r for r in existing if r["shard_id"] != shard_id]

    for aid, row_idx in index_rows:
        existing.append({
            "article_id": aid,
            "shard_id": shard_id,
            "row_index": int(row_idx),
        })

    table = pa.Table.from_pylist(existing, schema=pa.schema([
        ("article_id", pa.string()),
        ("shard_id", pa.string()),
        ("row_index", pa.int64()),
    ]))
    pq.write_table(table, idx_path)


# ---------------------------------------------------------------------------
# Cache manifest — provenance for the encoder weights that built the cache
# ---------------------------------------------------------------------------


_MANIFEST_NAME = "cache_manifest.json"


def file_sha256(path: Path) -> str:
    """Cheap content fingerprint of a checkpoint file.

    Used as the source-of-truth identifier for "which encoder produced
    these L0 survivors". Reads in 1 MB chunks so it works on multi-GB
    checkpoint files without loading them into memory.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_cache_manifest(
    cache_dir: str | Path,
    dataset: str,
    *,
    phase1_checkpoint: Path | None,
    stage_a_checkpoint: Path | None,
    lora_rank: int,
    lora_alpha: float | None,
    retention: float,
    extra: dict | None = None,
) -> Path:
    """Write a JSON manifest recording the provenance of a dataset's L0 cache.

    The manifest captures everything that determines whether the cached
    survivors are compatible with a given trainer run:

    - ``phase1_sha`` / ``phase1_path`` — base encoder weights.
    - ``stage_a_sha`` / ``stage_a_path`` — optional LoRA adapter weights.
    - ``lora_rank`` / ``lora_alpha`` — adapter shape.
    - ``retention`` — the L0 retention ratio used at encode time.

    Re-running :func:`write_cache_manifest` on a dataset that already has
    a manifest *appends* an entry to ``manifest_history`` and updates the
    top-level fields. The trainer's ``KRKBTrainer.setup`` reads the
    top-level fields and fails loudly if they don't match the
    configured ``training.phase1_checkpoint`` / ``training.
    stage_a_checkpoint`` / ``training.lora.*`` fields.
    """
    out_path = Path(cache_dir) / dataset / _MANIFEST_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)

    record: dict = {
        "dataset": dataset,
        "phase1_path": str(phase1_checkpoint) if phase1_checkpoint else None,
        "phase1_sha": (
            file_sha256(phase1_checkpoint)
            if phase1_checkpoint and phase1_checkpoint.is_file()
            else None
        ),
        "stage_a_path": str(stage_a_checkpoint) if stage_a_checkpoint else None,
        "stage_a_sha": (
            file_sha256(stage_a_checkpoint)
            if stage_a_checkpoint and stage_a_checkpoint.is_file()
            else None
        ),
        "lora_rank": int(lora_rank),
        "lora_alpha": float(lora_alpha) if lora_alpha is not None else None,
        "retention": float(retention),
    }
    if extra:
        record.update(dict(extra))

    history: list[dict] = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            history = list(existing.get("manifest_history", []))
            existing.pop("manifest_history", None)
            history.append(existing)
        except (json.JSONDecodeError, OSError):
            pass

    record["manifest_history"] = history
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True))
    return out_path


def read_cache_manifest(
    cache_dir: str | Path, dataset: str,
) -> dict | None:
    """Return the manifest dict for ``dataset``, or ``None`` if missing."""
    path = Path(cache_dir) / dataset / _MANIFEST_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


class L0CacheManifestMismatch(RuntimeError):  # noqa: N818  # used as exception
    """Raised when an L0 cache's recorded provenance doesn't match the
    encoder/checkpoint config the trainer is about to use.

    Catching this loudly at trainer setup prevents Stage B/C from
    silently training on a cache that was built with a different L0
    LoRA — historically the most insidious failure mode of the
    Stage A → Stage B handoff.
    """


def assert_cache_manifest_matches(
    cache_dir: str | Path,
    dataset: str,
    *,
    phase1_checkpoint: Path | None,
    stage_a_checkpoint: Path | None,
    lora_rank: int,
    lora_alpha: float | None = None,
    retention: float | None = None,
) -> None:
    """Assert the cache manifest is compatible with the trainer config.

    A missing manifest is permitted (legacy caches and bootstrap runs
    don't have one) but logged as a warning by the caller. A manifest
    that exists must match every load-bearing field; otherwise raise
    :class:`L0CacheManifestMismatch`.
    """
    manifest = read_cache_manifest(cache_dir, dataset)
    if manifest is None:
        return  # Caller decides whether a missing manifest is fatal.

    def _norm_sha(p: Path | None) -> str | None:
        return file_sha256(p) if p and p.is_file() else None

    expected_phase1_sha = _norm_sha(phase1_checkpoint)
    expected_stage_a_sha = _norm_sha(stage_a_checkpoint)

    mismatches: list[str] = []
    if (
        expected_phase1_sha is not None
        and manifest.get("phase1_sha") is not None
        and manifest["phase1_sha"] != expected_phase1_sha
    ):
        mismatches.append(
            f"phase1_sha: cache={manifest['phase1_sha'][:12]}, "
            f"trainer={expected_phase1_sha[:12]}"
        )
    if (
        expected_stage_a_sha is not None
        and manifest.get("stage_a_sha") is not None
        and manifest["stage_a_sha"] != expected_stage_a_sha
    ):
        mismatches.append(
            f"stage_a_sha: cache={manifest['stage_a_sha'][:12]}, "
            f"trainer={expected_stage_a_sha[:12]}"
        )
    # The trainer expects a Stage A LoRA but the cache was built without one.
    if expected_stage_a_sha is not None and manifest.get("stage_a_sha") is None:
        mismatches.append(
            "stage_a_sha: cache has no Stage A LoRA but trainer is "
            "configured to use one — re-run precompute_l0_subset.py "
            "with --stage-a-checkpoint."
        )
    if manifest.get("lora_rank") is not None and int(manifest["lora_rank"]) != int(lora_rank):
        mismatches.append(
            f"lora_rank: cache={manifest['lora_rank']}, trainer={lora_rank}"
        )
    if (
        lora_alpha is not None
        and manifest.get("lora_alpha") is not None
        and float(manifest["lora_alpha"]) != float(lora_alpha)
    ):
        mismatches.append(
            f"lora_alpha: cache={manifest['lora_alpha']}, trainer={lora_alpha}"
        )
    if (
        retention is not None
        and manifest.get("retention") is not None
        and abs(float(manifest["retention"]) - float(retention)) > 1e-6
    ):
        mismatches.append(
            f"retention: cache={manifest['retention']}, trainer={retention}"
        )

    if mismatches:
        raise L0CacheManifestMismatch(
            f"L0 cache for dataset {dataset!r} at {cache_dir} was built "
            f"with a different encoder configuration than the trainer is "
            f"about to use:\n  - " + "\n  - ".join(mismatches) + "\n"
            "Re-run scripts/precompute_l0_subset.py with the trainer's "
            "checkpoints, or point training.l0_cache_dir at the matching "
            "cache."
        )
