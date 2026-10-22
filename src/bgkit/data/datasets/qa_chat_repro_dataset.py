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
    BGKIT_TOOL_RESPONSE_SENTINEL,
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

        # Build join: QA row -> source file token index. The two sides
        # store ``repo_path`` differently — QA records use relative
        # ``owner/repo`` (set at QA-generation time), while the file
        # dataset stores the absolute disk path. Normalize both to
        # ``owner/repo`` + file_path (commit_sha dropped — see
        # ``_normalize_key``) before keying.
        #
        # When the same (owner/repo, file_path) appears multiple times in
        # the file_token_dataset (different commits, same code path), we
        # take the FIRST occurrence. Empirically chunking-into-multiple-
        # rows isn't happening in our token mmap (max single-row length
        # is 248k tokens; conversion does not split files), so duplicates
        # come from re-scanning at different commits, not from chunking.
        file_key_to_first_idx: dict[tuple[str, str], int] = {}
        for tok_idx in range(len(file_token_dataset)):
            key = file_token_dataset.file_key(tok_idx)
            if key is None:
                continue
            key = self._normalize_key(key)
            if key not in file_key_to_first_idx:
                file_key_to_first_idx[key] = tok_idx

        self._joined_indices: list[tuple[int, int]] = []
        for qa_idx in range(len(qa_dataset)):
            key = qa_dataset.file_key(qa_idx)
            if key is None:
                continue
            key = self._normalize_key(key)
            tok_idx = file_key_to_first_idx.get(key)
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
            stub_variants[0],
            _FILE_READ_QUERY_CONFIG,
            "x.py",
            "python",
            "X",
            tool_response_content=BGKIT_TOOL_RESPONSE_SENTINEL,
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
    def _normalize_key(key: tuple[str, str, str]) -> tuple[str, str]:
        """Reduce repo_path to ``owner/repo`` and drop ``commit_sha`` so
        absolute/relative forms match AND files match across commits.

        The QA generation scan and the file-token conversion scan ran at
        different times; same file paths but different commit_sha values
        for ~43% of samples. Dropping commit_sha from the join key
        recovers those samples — bgkit trains on whatever version of the
        file is currently in the file-token dataset, not the exact
        version the QA was generated against. For most QA pairs (which
        ask high-level structural questions) this is fine; for a small
        subset where the file content meaningfully changed between the
        two scans, the encoder sees slightly different code than the QA
        expects. Net effect: more data, slight noise in QA grounding,
        broader context exposure for bgkit.
        """
        repo_path, file_path, _commit_sha = key
        # Strip trailing slash, take last two components.
        parts = repo_path.rstrip("/").split("/")
        norm_repo = "/".join(parts[-2:]) if len(parts) >= 2 else repo_path
        return (norm_repo, file_path)

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
        source_tokens = source["token_ids"]
        result["content_token_ids"] = source_tokens
        result["language"] = language

        # Build per-content-token answer-position mask for the survivorship
        # head's QA position loss. Indices in qa["answer_position_indices"]
        # are token positions inside the *original* source file at filter
        # time; clip against the in-batch source length in case the dataset
        # truncated the file shorter than the filter saw.
        #
        # For samples where the underlying QA dataset has no attribution
        # (the un-filtered ``qa_conditioned/`` mmap was generated before
        # answer-position attribution; only ``qa_conditioned_filtered/``
        # has positions), we still emit an all-False mask so the collator
        # can pack mixed batches. The loss then trains the head toward
        # ``qa_non_answer_target`` at every position for those samples — a
        # weak sparsity prior, harmless to the attributed samples in the
        # same batch and lets us use the full ~115k QA pool instead of
        # the filtered 86k.
        pos_indices = qa.get("answer_position_indices")
        n_src = int(source_tokens.size(0))
        mask = torch.zeros(n_src, dtype=torch.bool)
        if pos_indices is not None and pos_indices.numel() > 0:
            in_range = pos_indices[(pos_indices >= 0) & (pos_indices < n_src)]
            if in_range.numel() > 0:
                mask[in_range] = True
        result["answer_position_mask"] = mask
        return result
