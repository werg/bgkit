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


@dataclass
class KBSample:
    dataset_name: str  # upstream corpus: "kilt_wikipedia", "pubmedqa", ...
    scope_template: str  # "topic_list" or "pre_scoped"
    scope_description: str  # pre_scoped: "Wikipedia" / "git:foo/bar" / ...
    topic_list: list[str]  # topic_list: top-level children of root
    question: str
    gold_answer: str
    trajectory: list[TrajectoryTurn]


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
        elif self._path.exists():
            self._mode = "parquet"
            self._table = pq.read_table(self._path, memory_map=True)
            self._n = self._table.num_rows
        else:
            raise FileNotFoundError(
                f"Trajectory data missing: neither {ipc_path} nor {self._path}",
            )

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
        )
