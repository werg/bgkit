"""Phase 2 KB-scale trajectory dataset.

Each sample is a ``(scope_context, question, trajectory, gold_answer)`` tuple
produced offline by :mod:`bgkit.data.teacher_trajectories` and cached in
``{DATA_DIR}/trajectories/{dataset}.parquet``.

At training time the trainer reads the parquet row, re-constructs the
:class:`TrajectoryTurn` list, and passes it to the tokenizer + L1 splicer.
Heavy work (template building, L1 encoding, loss mask construction) happens
inside the trainer so the dataset can stay lightweight and picklable across
DataLoader workers.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from bgkit.data.bgkit_tool_template import TrajectoryTurn, trajectory_from_json
from bgkit.data.commit_repro import GIT_REPRO_SCHEMA_VERSION, file_sha256

MAX_SAFE_PARQUET_FALLBACK_BYTES = 512 * 1024 * 1024


@dataclass
class KBSample:
    dataset_name: str  # upstream corpus: "kilt_wikipedia", "pubmedqa", ...
    scope_template: str  # "topic_list" or "pre_scoped"
    scope_description: str  # pre_scoped: "Wikipedia" / "git:foo/bar" / ...
    topic_list: list[str]  # topic_list: top-level children of root
    question: str
    gold_answer: str
    trajectory: list[TrajectoryTurn]
    group_id: str = ""
    repo_id: str = ""
    split: str = ""
    target_sha: str = ""
    target_path: str = ""
    drill_mode: str = ""
    structural_depth: int = 0
    artifact_schema_version: int = 0
    id_scheme_version: int = 0
    source_sha256: str = ""
    gold_span: tuple[int, int] | None = None
    """[tok_start, tok_end) of the answer inside the gold article's tokens
    (v5 span-level relevance supervision; from ``gold_span_json``)."""


def _parse_gold_span(raw) -> tuple[int, int] | None:
    if not raw:
        return None
    try:
        a, b = json.loads(raw)
    except (ValueError, TypeError):
        return None
    a, b = int(a), int(b)
    return (a, b) if b > a >= 0 else None


class KBTrajectoryDataset(Dataset):
    """Loads one-dataset trajectory parquet into a PyTorch Dataset.

    Parquet schema:

    - ``dataset_name`` (string)
    - ``scope_template`` (string)  "topic_list" | "pre_scoped"
    - ``scope_description`` (string | null)
    - ``topic_list_json`` (string | null)
    - ``question`` (string)
    - ``gold_answer`` (string)
    - ``trajectory_json`` (string)  Serialized list[TrajectoryTurn]
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        # Prefer an UNCOMPRESSED Arrow IPC sibling (``<name>.arrow``, written by
        # scripts/convert_trajectory_to_feather.py) accessed PAGED via mmap.
        #
        # Why not parquet: ``pq.read_table(..., memory_map=True)`` maps the
        # COMPRESSED file but still DECOMPRESSES every column into anonymous
        # heap buffers — ~28 GB resident, NON-reclaimable, for the
        # git_commit_repro trajectory parquet (the host-margin killer on the
        # unified-memory DGX). The uncompressed IPC is ``pa.memory_map``-ed and
        # indexed per-record-batch WITHOUT materialization, so column data pages
        # from the mmap on demand and the touched pages live in RECLAIMABLE page
        # cache. We never call ``read_all()``/``to_pandas()`` (that would
        # re-materialize and defeat the purpose).
        #
        # Fallback: if the IPC file is absent, use the parquet-lazy path
        # (compact Arrow table + per-row slice) — still better than the old
        # eager ``to_pylist()`` list-of-dicts (~51 GB), just not paged.
        ipc_path = self._path.with_suffix(".arrow")
        self._mode: str
        self._reader = None
        self._table = None
        if ipc_path.exists():
            self._mode = "ipc"
            self._ipc_path = ipc_path
            # mmap handle + IPC file reader (in-process; num_workers=0 so no
            # cross-process pickling of these handles).
            self._src = pa.memory_map(str(ipc_path), "r")
            self._reader = pa.ipc.open_file(self._src)
            # Cumulative row offset at the start of each record batch, so a
            # global idx maps to (batch_i, row_in_batch) by binary search.
            # get_batch(i).num_rows reads only the batch's metadata — cheap, no
            # column data paged in.
            self._batch_starts: list[int] = [0]
            for i in range(self._reader.num_record_batches):
                self._batch_starts.append(
                    self._batch_starts[-1] + self._reader.get_batch(i).num_rows,
                )
            self._n = self._batch_starts[-1]
            self._schema = self._reader.schema
        elif self._path.exists():
            if (
                self._path.stem == "git_commit_repro"
                and self._path.stat().st_size > MAX_SAFE_PARQUET_FALLBACK_BYTES
            ):
                raise ValueError(
                    "large git_commit_repro parquet has no paged Arrow sibling; "
                    "run scripts/convert_trajectory_to_feather.py before setup"
                )
            self._mode = "parquet"
            self._table = pq.read_table(self._path, memory_map=True)
            self._n = self._table.num_rows
            self._schema = self._table.schema
        else:
            raise FileNotFoundError(
                f"Trajectory data missing: neither {ipc_path} nor {self._path}",
            )
        self.artifact_manifest: dict = {}
        if self._path.stem == "git_commit_repro":
            required = {
                "group_id", "repo_id", "split", "target_sha", "target_path",
                "drill_mode", "structural_depth", "artifact_schema_version",
                "id_scheme_version", "source_sha256",
            }
            missing = required - set(self._schema.names)
            if missing:
                raise ValueError(
                    "legacy git_commit_repro trajectories are unsafe to train: "
                    f"missing schema-v2 columns {sorted(missing)}. Rebuild the "
                    "raw/tree/mmap/trajectory artifact set."
                )
            manifest_path = self._path.with_suffix(".manifest.json")
            if not manifest_path.exists():
                raise ValueError(
                    f"git_commit_repro trajectory manifest missing: {manifest_path}"
                )
            manifest = json.loads(manifest_path.read_text())
            if int(manifest.get("schema_version", 0)) != GIT_REPRO_SCHEMA_VERSION:
                raise ValueError("git_commit_repro trajectory manifest schema mismatch")
            if int(manifest.get("row_count", -1)) != self._n:
                raise ValueError(
                    "git_commit_repro trajectory row count disagrees with manifest"
                )
            if int(manifest.get("id_scheme_version", 0)) <= 0:
                raise ValueError("git_commit_repro manifest has no id scheme version")
            for key in ("source_sha256", "tree_sha256", "trajectory_sha256", "id_salt"):
                if not str(manifest.get(key, "")):
                    raise ValueError(f"git_commit_repro manifest is missing {key}")
            if not self._path.exists():
                raise ValueError(
                    "git_commit_repro Arrow IPC has no source parquet to verify"
                )
            actual_trajectory_sha = file_sha256(self._path)
            if actual_trajectory_sha != str(manifest["trajectory_sha256"]):
                raise ValueError(
                    "git_commit_repro trajectory parquet does not match its "
                    "manifest; rebuild the trajectory and Arrow artifacts"
                )
            versions = self._column_unique_values("artifact_schema_version")
            if versions != {GIT_REPRO_SCHEMA_VERSION}:
                raise ValueError(
                    "git_commit_repro rows contain mixed artifact schema versions"
                )
            id_versions = self._column_unique_values("id_scheme_version")
            if id_versions != {int(manifest["id_scheme_version"])}:
                raise ValueError(
                    "git_commit_repro rows and manifest use different ID schemes"
                )
            sources = self._column_unique_values("source_sha256")
            if sources != {str(manifest["source_sha256"])}:
                raise ValueError(
                    "git_commit_repro rows and manifest have different source hashes"
                )
            modes = {str(value) for value in self._column_unique_values("drill_mode")}
            manifest_modes = {
                str(mode)
                for mode, count in dict(manifest.get("drill_mode_counts", {})).items()
                if int(count) > 0
            }
            if not modes or modes != manifest_modes:
                raise ValueError(
                    "git_commit_repro rows and manifest have different drill modes"
                )
            if self._mode == "ipc":
                metadata = self._schema.metadata or {}
                ipc_source = metadata.get(b"bgkit_source_parquet_sha256", b"").decode()
                if ipc_source != str(manifest["trajectory_sha256"]):
                    raise ValueError(
                        "git_commit_repro Arrow IPC is stale; rerun "
                        "scripts/convert_trajectory_to_feather.py --force"
                    )
            self.artifact_manifest = manifest

    def __len__(self) -> int:
        return self._n

    def _row_at(self, idx: int) -> dict:
        """Materialize exactly ONE row's dict — equivalent to the old
        ``table.to_pylist()[idx]`` — paging from the mmap (IPC) or slicing the
        compact table (parquet fallback)."""
        if self._mode == "ipc":
            # idx -> (batch, row_in_batch). bisect_right - 1 finds the batch
            # whose start offset is the greatest <= idx.
            bi = bisect.bisect_right(self._batch_starts, idx) - 1
            row_in_batch = idx - self._batch_starts[bi]
            batch = self._reader.get_batch(bi)
            return batch.slice(row_in_batch, 1).to_pylist()[0]
        return self._table.slice(idx, 1).to_pylist()[0]

    def __getitem__(self, idx: int) -> KBSample:
        row = self._row_at(idx)
        topic_list_raw = row.get("topic_list_json")
        topic_list: list[str] = []
        if topic_list_raw:
            topic_list = list(json.loads(topic_list_raw))
        return KBSample(
            dataset_name=str(row["dataset_name"]),
            scope_template=str(row["scope_template"]),
            scope_description=str(row.get("scope_description") or ""),
            topic_list=topic_list,
            question=str(row["question"]),
            gold_answer=str(row["gold_answer"]),
            trajectory=trajectory_from_json(str(row["trajectory_json"])),
            group_id=str(row.get("group_id") or ""),
            repo_id=str(row.get("repo_id") or ""),
            split=str(row.get("split") or ""),
            target_sha=str(row.get("target_sha") or ""),
            target_path=str(row.get("target_path") or ""),
            drill_mode=str(row.get("drill_mode") or ""),
            structural_depth=int(row.get("structural_depth") or 0),
            artifact_schema_version=int(row.get("artifact_schema_version") or 0),
            id_scheme_version=int(row.get("id_scheme_version") or 0),
            source_sha256=str(row.get("source_sha256") or ""),
            gold_span=_parse_gold_span(row.get("gold_span_json")),
        )

    def _column_unique_values(self, name: str) -> set:
        """Read distinct values from a small provenance column batch-wise."""
        if name not in self._schema.names:
            return set()
        if self._mode == "ipc":
            field_index = self._reader.schema.get_field_index(name)
            values: set = set()
            for batch_index in range(self._reader.num_record_batches):
                column = self._reader.get_batch(batch_index).column(field_index)
                values.update(column.unique().to_pylist())
                if len(values) > 8:
                    break
            return values
        return set(self._table.column(name).unique().to_pylist())

    def _column_values(self, name: str) -> list:
        """Read one lightweight column without materializing gold blobs."""
        if name not in self._schema.names:
            return []
        if self._mode == "ipc":
            field_index = self._reader.schema.get_field_index(name)
            values: list = []
            for batch_index in range(self._reader.num_record_batches):
                values.extend(
                    self._reader.get_batch(batch_index).column(field_index).to_pylist()
                )
            return values
        return self._table.column(name).to_pylist()

    def split_labels(self) -> list[str]:
        """Explicit artifact split labels, or an empty list for legacy datasets."""
        return [str(value or "") for value in self._column_values("split")]

    def repo_ids(self) -> list[str]:
        return [str(value or "") for value in self._column_values("repo_id")]

    def group_keys(self) -> list[str]:
        """Bulk-extract each sample's ``is_head`` window-node id (the per-repo
        shared-subtree group key) reading ONLY the ``trajectory_json`` column —
        never paging the (large) gold blob. Batch-level iteration with a single
        column materialization per batch makes this ~30x faster than
        ``[_repo_group_key(self[i]) for i in range(len(self))]`` (which pages
        every row's full record incl. the gold answer — an >1h hang at 1.87M
        samples). Same key semantics as ``KRKBTrainer._repo_group_key``: the
        first ``is_head`` bgkit turn's ``ids[0]`` (``""`` if none)."""
        explicit = self._column_values("group_id")
        if explicit:
            return [str(value or "") for value in explicit]

        def _head_key(tj: str) -> str:
            for turn in trajectory_from_json(tj):
                if turn.kind == "bgkit" and bool(turn.args.get("is_head")):
                    ids = turn.args.get("ids", [])
                    return str(ids[0]) if ids else ""
            return ""

        keys: list[str] = []
        if self._mode == "ipc":
            fi = self._reader.schema.get_field_index("trajectory_json")
            for bi in range(self._reader.num_record_batches):
                col = self._reader.get_batch(bi).column(fi).to_pylist()
                keys.extend(_head_key(str(tj)) for tj in col)
        else:
            col = self._table.column("trajectory_json").to_pylist()
            keys.extend(_head_key(str(tj)) for tj in col)
        return keys
