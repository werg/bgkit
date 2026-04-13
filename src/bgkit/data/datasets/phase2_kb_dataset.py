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

from dataclasses import dataclass
from pathlib import Path

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
        if not self._path.exists():
            raise FileNotFoundError(f"Trajectory parquet missing: {self._path}")
        table = pq.read_table(self._path)
        self._rows = table.to_pylist()

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> KBSample:
        row = self._rows[idx]
        topic_list_raw = row.get("topic_list_json")
        topic_list: list[str] = []
        if topic_list_raw:
            import json

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
