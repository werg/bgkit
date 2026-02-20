"""Chat-formatted reproduction dataset wrapping MmapTokenDataset.

Wraps raw file token chunks in Qwen3's native chat template with tool-call
format, producing in-distribution agentic conversation for decoder training.
Loss is masked to only the file content tokens inside the markdown code fence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from bgkit.data.datasets.mmap_token_dataset import MmapTokenDataset

# Sentinel used to locate exact content boundaries within the template.
# Long random suffix makes accidental collision near-impossible.
CONTENT_SENTINEL = "<<<BGKIT_CONTENT_a7f3b2e1>>>"

# Tool definition JSON (structural, never reformulated)
TOOL_DEFINITION = json.dumps(
    {
        "type": "function",
        "function": {
            "name": "bgkit_read_file",
            "description": "Read the contents of a file from BgKIT compressed context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Instructions for how to process/return the file contents"
                        ),
                    },
                },
                "required": ["file_path", "prompt"],
            },
        },
    },
    ensure_ascii=False,
)


def _build_messages(
    variant: dict[str, str],
    file_path: str,
    language: str,
    file_content_placeholder: str,
) -> list[dict[str, str]]:
    """Build the chat messages list from a variant and sample metadata.

    The file_content_placeholder is inserted where the actual file content
    goes — either real content for tokenization or CONTENT_SENTINEL for
    boundary detection.
    """
    system_prompt = variant["system_prompt"]
    user_prompt = variant["user_prompt"].replace("{file_path}", file_path)
    compression_prompt = variant["compression_prompt"]
    response_prefix = variant["response_prefix"].replace("{file_path}", file_path)

    # Build tool call JSON
    tool_call_json = json.dumps(
        {
            "name": "bgkit_read_file",
            "arguments": {
                "file_path": file_path,
                "prompt": compression_prompt,
            },
        },
        ensure_ascii=False,
    )

    # System message with tool definition
    system_text = (
        f"{system_prompt}\n\n# Tools\n\n"
        f"You may call one or more functions to assist with the user query.\n\n"
        f"You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{TOOL_DEFINITION}\n</tools>"
    )

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_prompt},
        {
            "role": "assistant",
            "content": f"<tool_call>\n{tool_call_json}\n</tool_call>",
        },
        {
            "role": "user",
            "content": (
                "<tool_response>\n"
                "File contents provided as BgKIT compressed context.\n"
                "</tool_response>"
            ),
        },
        {
            "role": "assistant",
            "content": (
                f"<think>\n\n</think>\n\n"
                f"{response_prefix}\n\n"
                f"```{language}\n"
                f"{file_content_placeholder}\n"
                f"```"
            ),
        },
    ]
    return messages


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
        # Tokenize the longest variant's scaffolding (with sentinel as content)
        # to get the maximum number of non-content tokens.
        self._max_template_overhead = self._compute_max_template_overhead()

        # Cache lengths array (content + overhead) — avoids allocation per access
        self._lengths = self._inner.lengths + self._max_template_overhead

        # Precompute suffix_ids — structurally constant across all variants.
        # The suffix is always `\n``` ` + `<|im_end|>` (closing fence + end-of-turn).
        self._suffix_ids = self._compute_suffix_ids()

    def _compute_max_template_overhead(self) -> int:
        """Compute the maximum template token overhead across all variants.

        Uses a dummy file_path and language to measure scaffolding size.
        """
        max_overhead = 0
        # Use representative file path and longest common language name
        dummy_path = "src/example/placeholder_file.py"
        # Check a few language names that might be longest
        test_languages = ["python", "typescript", "javascript", ""]

        for variant in self._variants:
            for lang in test_languages:
                messages = _build_messages(variant, dummy_path, lang, "X")
                template_str = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
                # Remove the single-char "X" placeholder to get pure overhead
                overhead_str = template_str.replace("X", "", 1)
                overhead_tokens = len(
                    self._tokenizer.encode(overhead_str, add_special_tokens=False)
                )
                max_overhead = max(max_overhead, overhead_tokens)

        return max_overhead

    def _compute_suffix_ids(self) -> torch.Tensor:
        """Compute the constant suffix token IDs (closing fence + end-of-turn).

        The suffix is structurally identical across all variants — only the
        text *before* the sentinel differs per variant. Verify across a few
        variants as a sanity check.
        """
        suffix_ids = None
        for variant in self._variants:
            messages = _build_messages(variant, "test.py", "python", CONTENT_SENTINEL)
            template_str = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            _, suffix_str = template_str.split(CONTENT_SENTINEL)
            ids = self._tokenizer.encode(suffix_str, add_special_tokens=False)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            if suffix_ids is None:
                suffix_ids = ids_tensor
            elif not suffix_ids.equal(ids_tensor):
                raise ValueError(
                    "Suffix token IDs differ across variants — expected constant suffix. "
                    f"Got {suffix_ids.tolist()} vs {ids_tensor.tolist()}"
                )
        return suffix_ids

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
        """Select a variant deterministically per (epoch, idx).

        Uses hashlib.md5 for stability across DataLoader workers
        (Python's hash() is randomized by PYTHONHASHSEED per process).
        """
        key = f"{self._epoch_seed}:{idx}".encode()
        h = int.from_bytes(hashlib.md5(key).digest()[:8], "little")
        return self._variants[h % len(self._variants)]

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        inner_sample = self._inner[idx]
        content_token_ids = inner_sample["token_ids"]
        file_path = inner_sample["file_path"]
        language = inner_sample["language"]

        variant = self._select_variant(idx)

        # --- Build template with sentinel for boundary detection ---
        messages_with_sentinel = _build_messages(
            variant, file_path, language, CONTENT_SENTINEL,
        )
        template_str = self._tokenizer.apply_chat_template(
            messages_with_sentinel, tokenize=False, add_generation_prompt=False,
        )

        # Validate sentinel uniqueness
        sentinel_count = template_str.count(CONTENT_SENTINEL)
        if sentinel_count != 1:
            raise ValueError(
                f"Expected exactly 1 sentinel in template, found {sentinel_count}. "
                f"Variant text may accidentally contain the sentinel string."
            )

        # Split on sentinel to get prefix and suffix strings
        prefix_str, suffix_str = template_str.split(CONTENT_SENTINEL)

        # Tokenize each piece separately (no special tokens per piece)
        prefix_ids = self._tokenizer.encode(prefix_str, add_special_tokens=False)
        suffix_ids = self._tokenizer.encode(suffix_str, add_special_tokens=False)

        # Content token IDs from the inner dataset (already tokenized)
        content_ids = content_token_ids.tolist()

        # Concatenate: prefix + content + suffix
        full_ids = prefix_ids + content_ids + suffix_ids
        token_ids = torch.tensor(full_ids, dtype=torch.long)

        # Build loss mask: 1 only for content tokens
        loss_mask = torch.zeros(len(full_ids), dtype=torch.long)
        content_start = len(prefix_ids)
        content_end = content_start + len(content_ids)
        loss_mask[content_start:content_end] = 1

        # Tokenize compression prompt for BgKIT conditioning
        compression_prompt = variant["compression_prompt"]
        compression_prompt_ids = torch.tensor(
            self._tokenizer.encode(compression_prompt, add_special_tokens=False),
            dtype=torch.long,
        )

        return {
            "token_ids": token_ids,
            "loss_mask": loss_mask,
            "content_token_ids": content_token_ids,
            "compression_prompt_ids": compression_prompt_ids,
            "prefix_ids": torch.tensor(prefix_ids, dtype=torch.long),
            "language": language,
        }
