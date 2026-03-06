"""Chat-formatted QA dataset for decoder init training.

Like ChatReproDataset but uses question-conditioned prompts and answer targets
instead of verbatim file content. Uses file_read_query template config
(content_in_code_fence=False).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from bgkit.data.chat_template import (
    TOOL_CONFIGS,
    build_messages,
    build_tools,
    compute_suffix_ids,
    tokenize_with_sentinel,
)
from bgkit.data.datasets.mmap_token_dataset import MmapTokenDataset
from bgkit.data.datasets.qa_conditioned_dataset import MmapQAConditionedDataset

_FILE_READ_QUERY_CONFIG = TOOL_CONFIGS["file_read_query"]


class QAChatReproDataset(Dataset):
    """Chat-formatted QA wrapper for decoder init training.

    Joins QA pairs to source file tokens and wraps in chat template
    with question-conditioned prompts. Uses file_read_query config.

    Args:
        qa_dataset: MmapQAConditionedDataset providing QA pairs.
        file_token_dataset: MmapTokenDataset providing source file tokens.
        tokenizer: Qwen3 tokenizer.
        seed: Base seed for reproducibility.
    """

    def __init__(
        self,
        qa_dataset: MmapQAConditionedDataset,
        file_token_dataset: MmapTokenDataset,
        tokenizer,
        seed: int = 42,
    ):
        self._qa_ds = qa_dataset
        self._file_ds = file_token_dataset
        self._tokenizer = tokenizer
        self._base_seed = seed
        self._epoch_seed = seed

        # Build join: QA row -> source file token index (single-chunk only)
        file_key_to_chunk: dict[tuple[str, str, str], tuple[int, int]] = {}
        for tok_idx in range(len(file_token_dataset)):
            key = file_token_dataset.file_key(tok_idx)
            if key is None:
                continue
            if key not in file_key_to_chunk:
                file_key_to_chunk[key] = (tok_idx, 1)
            else:
                first, count = file_key_to_chunk[key]
                file_key_to_chunk[key] = (first, count + 1)

        single_chunk_keys = {
            k: first for k, (first, count) in file_key_to_chunk.items()
            if count == 1
        }

        self._joined_indices: list[tuple[int, int]] = []
        for qa_idx in range(len(qa_dataset)):
            key = qa_dataset.file_key(qa_idx)
            if key is None:
                continue
            tok_idx = single_chunk_keys.get(key)
            if tok_idx is not None:
                self._joined_indices.append((tok_idx, qa_idx))

        # Build stub variants for suffix computation
        stub_variants = [self._build_variant_stub()]
        self._suffix_ids = compute_suffix_ids(
            tokenizer, stub_variants, _FILE_READ_QUERY_CONFIG,
        )

        # Compute overhead and cache lengths
        tools = build_tools(_FILE_READ_QUERY_CONFIG)
        messages = build_messages(
            stub_variants[0], _FILE_READ_QUERY_CONFIG, "x.py", "python", "X",
        )
        overhead_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools,
        ).replace("X", "", 1)
        self._max_overhead = len(
            tokenizer.encode(overhead_str, add_special_tokens=False)
        )

        # Lengths = answer tokens + overhead
        answer_lens = np.array([
            qa_dataset.lengths[qa_idx] for _, qa_idx in self._joined_indices
        ], dtype=np.int32)
        self._lengths = answer_lens + self._max_overhead

    @staticmethod
    def _build_variant_stub() -> dict[str, str]:
        return {
            "system_prompt": "You are an AI coding assistant with access to the "
                             "bgkit_read_file tool for reading and analyzing file "
                             "contents from compressed context.",
            "user_prompt": "Analyze `{file_path}`: What is this code about?",
            "compression_prompt": "What is this code about?",
            "response_prefix": "Here is the analysis of `{file_path}`:",
        }

    @property
    def suffix_ids(self) -> torch.Tensor:
        return self._suffix_ids

    @property
    def lengths(self) -> np.ndarray:
        return self._lengths

    def set_epoch(self, epoch: int) -> None:
        self._epoch_seed = self._base_seed + epoch

    def __len__(self) -> int:
        return len(self._joined_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tok_idx, qa_idx = self._joined_indices[idx]
        qa = self._qa_ds[qa_idx]

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

        source = self._file_ds[tok_idx]
        file_path = source.get("file_path", "unknown")
        language = source.get("language", "")

        result = tokenize_with_sentinel(
            self._tokenizer,
            variant,
            _FILE_READ_QUERY_CONFIG,
            file_path,
            language,
            qa["answer_token_ids"],  # decoder target: predict answer tokens
        )
        # Override content_token_ids with SOURCE file tokens for BgKIT encoder.
        # tokenize_with_sentinel sets content_token_ids = answer tokens (used for
        # building the decoder sequence), but _compute_survivors feeds
        # content_token_ids into the encoder, which must compress the source file.
        result["content_token_ids"] = source["token_ids"]
        result["language"] = language
        return result
