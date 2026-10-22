"""Generic mmap dataset for Phase 2 document-question-answer tuples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from bgkit.data.datasets.base_mmap_dataset import (
    check_required_files,
    load_and_validate_manifest,
    validate_manifest_counts,
)
from bgkit.data.datasets.qa_sample import QASample


class _LazyArrowRows:
    """Sequence view over a pyarrow ``Table`` that materializes one row dict
    on access instead of eagerly building a full list-of-dicts.

    Indexing, iteration, and ``len`` produce output byte-identical to
    ``table.to_pylist()`` — ``self[i]`` equals ``table.to_pylist()[i]`` and
    iterating yields the same dicts in the same order. This keeps the whole
    per-row metadata (potentially millions of dicts) out of RAM; only the
    underlying Arrow table is retained (a column-projected subset already
    held by the dataset).
    """

    __slots__ = ("_table",)

    def __init__(self, table):
        self._table = table

    def __len__(self) -> int:
        return self._table.num_rows

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(self._table.num_rows))]
        if idx < 0:
            idx += self._table.num_rows
        return self._table.slice(idx, 1).to_pylist()[0]

    def __iter__(self):
        for batch in self._table.to_batches():
            yield from batch.to_pylist()


def merge_metadata_columns(
    data_dir: str,
    extra: tuple[str, ...],
    explicit: list[str] | None,
) -> list[str] | None:
    """Merge *extra* columns into the inferred or explicit column set.

    When ``explicit`` is provided, merges extras into it.
    When ``explicit`` is None, infers the base set from the schema's
    preferred columns (same as ``Phase2QADataset._infer_metadata_columns``)
    and adds extras on top.

    Filters against the parquet schema so missing columns don't crash.
    """
    schema = pq.read_schema(Path(data_dir) / "metadata.parquet")
    available = set(schema.names)

    if explicit is not None:
        columns = list(explicit)
    else:
        # Replicate the base class's preferred-column inference so we
        # don't lose id/document_id/tags when only extras are requested.
        preferred = [
            "id", "document_id", "file_path", "language",
            "dataset_name", "answer_type", "task_name",
            "tags", "tag_list_json",
        ]
        columns = [c for c in preferred if c in available]
        if not columns:
            columns = list(available)

    for col in extra:
        if col not in columns:
            columns.append(col)

    columns = [c for c in columns if c in available]
    return columns or None


class Phase2QADataset(Dataset):
    """Loads the Phase 2 mmap schema introduced for knowledge retrieval.

    Required files:
    - ``tokens.npy`` / ``offsets.npy`` for source document tokens
    - ``question_tokens.npy`` / ``question_offsets.npy`` for questions
    - ``answer_tokens.npy`` / ``answer_offsets.npy`` for answers
    - ``metadata.parquet`` containing per-row metadata
    - ``manifest.json`` with row_count and tokenizer metadata
    """

    CONVERT_HINT = (
        "Convert with: python scripts/convert_hf_to_mmap.py "
        "--dataset-id <hf_dataset> --output-dir <phase2_dataset_dir>"
    )

    def __init__(
        self,
        data_dir: str,
        *,
        dataset_name: str | None = None,
        max_document_len: int = 8192,
        max_question_len: int = 512,
        max_answer_len: int = 512,
        tokenizer=None,
        variant: dict[str, str] | None = None,
        metadata_columns: list[str] | None = None,
    ):
        self._data_path = Path(data_dir)
        required = [
            "tokens.npy",
            "offsets.npy",
            "question_tokens.npy",
            "question_offsets.npy",
            "answer_tokens.npy",
            "answer_offsets.npy",
            "manifest.json",
            "metadata.parquet",
        ]
        check_required_files(self._data_path, required, self.CONVERT_HINT)
        manifest = load_and_validate_manifest(self._data_path)

        self.dataset_name = dataset_name or manifest.get("dataset_name") or self._data_path.name
        self._max_document_len = max_document_len
        self._max_question_len = max_question_len
        self._max_answer_len = max_answer_len
        self._tokenizer = tokenizer
        self._variant = variant or {}

        self._tokens = np.load(self._data_path / "tokens.npy", mmap_mode="r")
        self._offsets = np.load(self._data_path / "offsets.npy")
        self._question_tokens = np.load(self._data_path / "question_tokens.npy", mmap_mode="r")
        self._question_offsets = np.load(self._data_path / "question_offsets.npy")
        self._answer_tokens = np.load(self._data_path / "answer_tokens.npy", mmap_mode="r")
        self._answer_offsets = np.load(self._data_path / "answer_offsets.npy")

        validate_manifest_counts(manifest, self._offsets, self._tokens)
        self._validate_parallel_counts(manifest)

        doc_lengths = (self._offsets[1:] - self._offsets[:-1]).astype(np.int64)
        question_lengths = (self._question_offsets[1:] - self._question_offsets[:-1]).astype(
            np.int64,
        )
        answer_lengths = (self._answer_offsets[1:] - self._answer_offsets[:-1]).astype(np.int64)
        valid = (doc_lengths > 0) & (question_lengths > 0) & (answer_lengths > 0)

        self._valid_indices = np.where(valid)[0].astype(np.int64)
        self._doc_lengths = np.minimum(doc_lengths[valid], max_document_len).astype(np.int32)
        self._question_lengths = np.minimum(question_lengths[valid], max_question_len).astype(
            np.int32,
        )
        self._answer_lengths = np.minimum(answer_lengths[valid], max_answer_len).astype(
            np.int32,
        )
        self._lengths = (
            self._doc_lengths + self._question_lengths + self._answer_lengths
        ).astype(np.int32)

        metadata_columns = metadata_columns or self._infer_metadata_columns()
        self._metadata = pq.read_table(
            self._data_path / "metadata.parquet",
            columns=metadata_columns,
        )
        # Lazy per-row view over the Arrow table — avoids eagerly
        # materializing a full list-of-dicts (~one dict per corpus row) into
        # RAM at construction. Indexing/iteration are byte-identical to the
        # former ``self._metadata.to_pylist()``.
        self._metadata_rows = _LazyArrowRows(self._metadata)

    def _validate_parallel_counts(self, manifest: dict) -> None:
        row_count = len(self._offsets) - 1
        if len(self._question_offsets) - 1 != row_count:
            raise ValueError("question_offsets.npy row count does not match offsets.npy")
        if len(self._answer_offsets) - 1 != row_count:
            raise ValueError("answer_offsets.npy row count does not match offsets.npy")
        expected = manifest.get("row_count")
        if expected is not None and expected != row_count:
            raise ValueError(
                f"manifest row_count {expected} does not match mmap row count {row_count}",
            )

    def _infer_metadata_columns(self) -> list[str]:
        schema = pq.read_schema(self._data_path / "metadata.parquet")
        preferred = [
            "id",
            "document_id",
            "file_path",
            "language",
            "dataset_name",
            "answer_type",
            "task_name",
            "tags",
            "tag_list_json",
        ]
        return [name for name in preferred if name in schema.names] or schema.names

    @property
    def lengths(self) -> np.ndarray:
        return self._lengths

    @property
    def metadata(self) -> list[dict[str, object]]:
        return self._metadata.to_pylist()

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        for field in (
            "_tokens",
            "_question_tokens",
            "_answer_tokens",
        ):
            state[field] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tokens = np.load(self._data_path / "tokens.npy", mmap_mode="r")
        self._question_tokens = np.load(self._data_path / "question_tokens.npy", mmap_mode="r")
        self._answer_tokens = np.load(self._data_path / "answer_tokens.npy", mmap_mode="r")

    def _slice(
        self,
        arr: np.ndarray,
        offsets: np.ndarray,
        orig_idx: int,
        max_len: int,
    ) -> torch.Tensor:
        start = int(offsets[orig_idx])
        end = min(int(offsets[orig_idx + 1]), start + max_len)
        return torch.from_numpy(arr[start:end].astype(np.int64))

    @staticmethod
    def _extract_tags(metadata: dict[str, object]) -> list[str]:
        if "tags" in metadata and isinstance(metadata["tags"], list):
            return [str(tag) for tag in metadata["tags"]]
        if metadata.get("tag_list_json"):
            try:
                value = json.loads(str(metadata["tag_list_json"]))
            except json.JSONDecodeError:
                value = []
            if isinstance(value, list):
                return [str(tag) for tag in value]
        return []

    def _build_target(
        self,
        question_ids: torch.Tensor,
        answer_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_ids = torch.cat([question_ids, answer_ids], dim=0)
        loss_mask = torch.cat([
            torch.zeros_like(question_ids, dtype=torch.bool),
            torch.ones_like(answer_ids, dtype=torch.bool),
        ])
        return {
            "token_ids": target_ids,
            "loss_mask": loss_mask,
            "prefix_ids": question_ids.clone(),
            "compression_prompt_ids": question_ids.clone(),
            "bgkit_splice_start": torch.tensor(question_ids.size(0), dtype=torch.long),
            "bgkit_splice_len": torch.tensor(0, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> QASample:
        orig_idx = int(self._valid_indices[idx])
        document_ids = self._slice(self._tokens, self._offsets, orig_idx, self._max_document_len)
        question_ids = self._slice(
            self._question_tokens,
            self._question_offsets,
            orig_idx,
            self._max_question_len,
        )
        answer_ids = self._slice(
            self._answer_tokens,
            self._answer_offsets,
            orig_idx,
            self._max_answer_len,
        )

        metadata = dict(self._metadata_rows[orig_idx])
        tags = self._extract_tags(metadata)
        target = self._build_target(question_ids, answer_ids)
        if not isinstance(target["token_ids"], torch.Tensor):
            target["token_ids"] = torch.as_tensor(target["token_ids"], dtype=torch.long)
        if not isinstance(target["loss_mask"], torch.Tensor):
            target["loss_mask"] = torch.as_tensor(target["loss_mask"], dtype=torch.bool)
        if not isinstance(target["prefix_ids"], torch.Tensor):
            target["prefix_ids"] = torch.as_tensor(target["prefix_ids"], dtype=torch.long)
        if not isinstance(target["compression_prompt_ids"], torch.Tensor):
            target["compression_prompt_ids"] = torch.as_tensor(
                target["compression_prompt_ids"], dtype=torch.long,
            )

        content_mask = torch.ones(document_ids.size(0), dtype=torch.bool)
        target_attention = torch.ones(target["token_ids"].size(0), dtype=torch.bool)

        return QASample(
            objective="phase2_qa",
            content_token_ids=document_ids,
            content_attention_mask=content_mask,
            compression_ratio=0.0,
            compression_level=0,
            target_token_ids=target["token_ids"],
            target_attention_mask=target_attention,
            target_loss_mask=target["loss_mask"].to(dtype=torch.bool),
            prefix_ids=target["prefix_ids"],
            compression_prompt_ids=target["compression_prompt_ids"],
            bgkit_splice_start=int(target["bgkit_splice_start"].item()),
            bgkit_splice_len=int(target["bgkit_splice_len"].item()),
            question_token_ids=question_ids,
            answer_token_ids=answer_ids,
            sample_id=str(metadata.get("id", f"{self.dataset_name}:{orig_idx}")),
            dataset_name=str(metadata.get("dataset_name", self.dataset_name)),
            document_id=str(metadata.get("document_id", metadata.get("id", orig_idx))),
            tags=tags,
            metadata=metadata,
        )
