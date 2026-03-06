"""Mmap dataset for query-conditioned compression targets.

Loads pre-tokenized QA pairs (question + answer) and joins them to source
file tokens via (repo_path, file_path, commit_sha) composite key.

Two classes:
- MmapQAConditionedDataset: raw mmap access layer (answer + question tokens)
- QAConditionedSubset: training wrapper for CompressionDataset (joins to source files)
"""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from bgkit.data.chat_template import (
    TOOL_CONFIGS,
    build_messages,
    build_tools,
    tokenize_with_sentinel,
)
from bgkit.data.datasets.base_mmap_dataset import BaseMmapDataset


class MmapQAConditionedDataset(BaseMmapDataset):
    """Dataset yielding QA pair tokens from mmap'd numpy arrays.

    Index-based (not key-based) — multiple QA rows per file.
    Each row exposes repo_path, file_path, commit_sha for join resolution.

    Loads:
    - tokens.npy / offsets.npy — answer token sequences (decoder targets)
    - question_tokens.npy / question_offsets.npy — question token sequences
    - metadata.parquet — join keys + provenance

    Args:
        data_dir: Directory containing mmap artifacts.
        max_seq_len: Maximum answer token length.
    """

    CONVERT_HINT = (
        "Convert with: python scripts/convert_qa_pairs_to_npy.py "
        "--input-dir <qa_pairs_jsonl_dir> "
        "--output-dir <output_dir>"
    )

    def __init__(self, data_dir: str, max_seq_len: int = 2048):
        super().__init__(
            data_dir, max_seq_len=max_seq_len,
            extra_required_files=[
                "metadata.parquet",
                "question_tokens.npy",
                "question_offsets.npy",
            ],
        )

        # Load question token arrays
        self._question_tokens = np.load(
            self._data_path / "question_tokens.npy", mmap_mode="r"
        )
        self._question_offsets = np.load(
            self._data_path / "question_offsets.npy"
        )

        # Load metadata for join keys
        meta = pq.read_table(
            self._data_path / "metadata.parquet",
            columns=["repo_path", "file_path", "commit_sha"],
        )
        self._repo_paths = meta.column("repo_path").to_pylist()
        self._file_paths = meta.column("file_path").to_pylist()
        self._commit_shas = meta.column("commit_sha").to_pylist()

    def file_key(self, idx: int) -> tuple[str, str, str] | None:
        """Return (repo_path, file_path, commit_sha) for a valid index."""
        orig_idx = int(self._valid_indices[idx])
        rp = self._repo_paths[orig_idx]
        fp = self._file_paths[orig_idx]
        cs = self._commit_shas[orig_idx]
        if not rp or not fp:
            return None
        return (rp, fp, cs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        result = super().__getitem__(idx)  # answer token_ids

        # Get question tokens for this row
        orig_idx = int(self._valid_indices[idx])
        q_start = int(self._question_offsets[orig_idx])
        q_end = int(self._question_offsets[orig_idx + 1])
        q_tokens = self._question_tokens[q_start:q_end].astype(np.int64)
        result["question_token_ids"] = torch.from_numpy(q_tokens)
        result["answer_token_ids"] = result.pop("token_ids")

        return result

    def _get_mmap_fields(self) -> list[str]:
        return ["_tokens", "_question_tokens"]

    def _reopen_mmaps(self) -> None:
        super()._reopen_mmaps()
        self._question_tokens = np.load(
            self._data_path / "question_tokens.npy", mmap_mode="r"
        )


_FILE_READ_QUERY_CONFIG = TOOL_CONFIGS["file_read_query"]


class QAConditionedSubset(Dataset):
    """Sub-dataset for query-conditioned compression (Objective 5).

    Iterates QA rows and joins each to its source file tokens via
    (repo_path, file_path, commit_sha). Returns FileCompressionSample.

    Join direction: QA->file (iterate QA rows, look up source file).
    Only joins to single-chunk files for correctness.
    """

    def __init__(self, token_dataset, qa_dataset, tokenizer, seed=42):
        self._token_ds = token_dataset
        self._qa_ds = qa_dataset
        self._tokenizer = tokenizer
        self._base_seed = seed
        self._epoch_seed = seed

        # Build reverse index: file key -> (first_chunk_idx, n_chunks)
        file_key_to_chunk_info: dict[tuple[str, str, str], tuple[int, int]] = {}
        for tok_idx in range(len(token_dataset)):
            key = token_dataset.file_key(tok_idx)
            if key is None:
                continue
            if key not in file_key_to_chunk_info:
                file_key_to_chunk_info[key] = (tok_idx, 1)
            else:
                first, count = file_key_to_chunk_info[key]
                file_key_to_chunk_info[key] = (first, count + 1)

        # Keep only single-chunk files
        single_chunk_keys = {
            k: first for k, (first, count) in file_key_to_chunk_info.items()
            if count == 1
        }

        # Join QA rows to source files
        self._joined_indices: list[tuple[int, int]] = []  # (token_idx, qa_idx)
        for qa_idx in range(len(qa_dataset)):
            key = qa_dataset.file_key(qa_idx)
            if key is None:
                continue
            tok_idx = single_chunk_keys.get(key)
            if tok_idx is not None:
                self._joined_indices.append((tok_idx, qa_idx))

        # Precompute overhead for length estimates
        tools = build_tools(_FILE_READ_QUERY_CONFIG)
        dummy_variant = {
            "system_prompt": "You are an AI coding assistant.",
            "user_prompt": "Analyze {file_path}",
            "compression_prompt": "Answer the question about the file",
            "response_prefix": "Here is the analysis of {file_path}:",
        }
        messages = build_messages(
            dummy_variant, _FILE_READ_QUERY_CONFIG, "x.py", "python", "X",
        )
        overhead_str = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools,
        ).replace("X", "", 1)
        self._max_overhead = len(
            self._tokenizer.encode(overhead_str, add_special_tokens=False)
        )

        # Cache lengths: source file tokens + answer tokens + overhead
        source_lens = np.array([
            token_dataset.lengths[tok_idx] for tok_idx, _ in self._joined_indices
        ], dtype=np.int32)
        answer_lens = np.array([
            qa_dataset.lengths[qa_idx] for _, qa_idx in self._joined_indices
        ], dtype=np.int32)
        self._lengths = source_lens + answer_lens + self._max_overhead

    @property
    def lengths(self) -> np.ndarray:
        return self._lengths

    def set_epoch(self, epoch: int) -> None:
        self._epoch_seed = self._base_seed + epoch

    def __len__(self) -> int:
        return len(self._joined_indices)

    def __getitem__(self, idx: int):
        from bgkit.data.datasets.compression_dataset import FileCompressionSample

        tok_idx, qa_idx = self._joined_indices[idx]
        source = self._token_ds[tok_idx]
        qa = self._qa_ds[qa_idx]

        # Build a variant dict using the question as compression_prompt
        question_ids = qa["question_token_ids"]
        question_text = self._tokenizer.decode(question_ids, skip_special_tokens=True)

        variant = {
            "system_prompt": "You are an AI coding assistant with access to the "
                             "bgkit_read_file tool for reading and analyzing file "
                             "contents from compressed context.",
            "user_prompt": f"Analyze `{{file_path}}`: {question_text}",
            "compression_prompt": question_text,
            "response_prefix": "Here is the analysis of `{file_path}`:",
        }

        file_path = source.get("file_path", "unknown")
        language = source.get("language", "")

        result = tokenize_with_sentinel(
            self._tokenizer,
            variant,
            _FILE_READ_QUERY_CONFIG,
            file_path,
            language,
            qa["answer_token_ids"],
        )

        content_ids = source["token_ids"]
        attn_mask = torch.ones(content_ids.size(0), dtype=torch.long)
        target_attn = torch.ones(result["token_ids"].size(0), dtype=torch.long)

        return FileCompressionSample(
            objective="query_conditioned",
            content_token_ids=content_ids,
            content_attention_mask=attn_mask,
            compression_ratio=0.0,
            compression_level=0,
            target_token_ids=result["token_ids"],
            target_attention_mask=target_attn,
            target_loss_mask=result["loss_mask"],
            prefix_ids=result["prefix_ids"],
            compression_prompt_ids=result["compression_prompt_ids"],
        )
