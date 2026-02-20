"""Chat-formatted reproduction dataset wrapping MmapTokenDataset.

Wraps raw file token chunks in Qwen3's native chat template with tool-call
format, producing in-distribution agentic conversation for decoder training.
Loss is masked to only the file content tokens inside the markdown code fence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from bgkit.data.chat_template import (
    CONTENT_SENTINEL,
    TOOL_CONFIGS,
    build_messages,
    compute_suffix_ids,
    make_tool_definition,
    select_variant,
    tokenize_with_sentinel,
)
from bgkit.data.datasets.mmap_token_dataset import MmapTokenDataset

# Re-export for backward compatibility — other modules import these from here.
__all__ = ["CONTENT_SENTINEL", "ChatReproDataset", "_build_messages"]

_FILE_READ_CONFIG = TOOL_CONFIGS["file_read_repro"]

# Re-export the tool definition string for backward compatibility
TOOL_DEFINITION = make_tool_definition(_FILE_READ_CONFIG)


def _build_messages(
    variant: dict[str, str],
    file_path: str,
    language: str,
    file_content_placeholder: str,
) -> list[dict[str, str]]:
    """Build the chat messages list from a variant and sample metadata.

    Backward-compatible wrapper around the shared build_messages function.
    The file_content_placeholder is inserted where the actual file content
    goes — either real content for tokenization or CONTENT_SENTINEL for
    boundary detection.
    """
    return build_messages(
        variant, _FILE_READ_CONFIG, file_path, language, file_content_placeholder,
    )


class ChatReproDataset(Dataset):
    """Chat-formatted wrapper around MmapTokenDataset.

    For each sample, selects a prompt variant (deterministic per sample+epoch),
    wraps the file content in Qwen3's chat template with tool-call format, and
    returns tokenized IDs with a loss mask covering only the file content.

    Args:
        inner_dataset: MmapTokenDataset providing raw file token chunks.
        tokenizer: Qwen3 tokenizer (must support apply_chat_template).
        variant_bank_path: Path to JSON file with prompt variants.
        seed: Base seed for variant selection.
    """

    def __init__(
        self,
        inner_dataset: MmapTokenDataset,
        tokenizer,
        variant_bank_path: str | Path,
        seed: int = 42,
    ):
        self._inner = inner_dataset
        self._tokenizer = tokenizer
        self._base_seed = seed
        self._epoch_seed = seed

        # Load variant bank
        with open(variant_bank_path) as f:
            self._variants: list[dict[str, str]] = json.load(f)
        if not self._variants:
            raise ValueError(f"Empty variant bank at {variant_bank_path}")

        # Precompute max template overhead for conservative length estimates.
        self._max_template_overhead = self._compute_max_template_overhead()

        # Cache lengths array (content + overhead) — avoids allocation per access
        self._lengths = self._inner.lengths + self._max_template_overhead

        # Precompute suffix_ids — structurally constant across all variants.
        self._suffix_ids = compute_suffix_ids(
            self._tokenizer, self._variants, _FILE_READ_CONFIG,
        )

    def _compute_max_template_overhead(self) -> int:
        """Compute the maximum template token overhead across all variants.

        Uses a dummy file_path and language to measure scaffolding size.
        """
        max_overhead = 0
        dummy_path = "src/example/placeholder_file.py"
        test_languages = ["python", "typescript", "javascript", ""]

        for variant in self._variants:
            for lang in test_languages:
                messages = build_messages(
                    variant, _FILE_READ_CONFIG, dummy_path, lang, "X",
                )
                template_str = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
                overhead_str = template_str.replace("X", "", 1)
                overhead_tokens = len(
                    self._tokenizer.encode(overhead_str, add_special_tokens=False)
                )
                max_overhead = max(max_overhead, overhead_tokens)

        return max_overhead

    @property
    def suffix_ids(self) -> torch.Tensor:
        """Constant 1D suffix token IDs (closing fence + end-of-turn).

        Not batched — accessed directly by the trainer for generation.
        """
        return self._suffix_ids

    @property
    def lengths(self) -> np.ndarray:
        """Conservative upper-bound lengths for token budget batching.

        Returns content_length + max_template_overhead for each sample.
        """
        return self._lengths

    @property
    def content_lengths(self) -> np.ndarray:
        """Raw content token lengths (for understanding BgKIT input sizes)."""
        return self._inner.lengths

    def set_epoch(self, epoch: int) -> None:
        """Update epoch seed for variant selection diversity across epochs."""
        self._epoch_seed = self._base_seed + epoch

    def _select_variant(self, idx: int) -> dict[str, str]:
        """Select a variant deterministically per (epoch, idx)."""
        return select_variant(self._variants, idx, self._epoch_seed)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        inner_sample = self._inner[idx]
        content_token_ids = inner_sample["token_ids"]
        file_path = inner_sample["file_path"]
        language = inner_sample["language"]

        variant = self._select_variant(idx)

        result = tokenize_with_sentinel(
            self._tokenizer,
            variant,
            _FILE_READ_CONFIG,
            file_path,
            language,
            content_token_ids,
        )
        result["language"] = language
        return result
