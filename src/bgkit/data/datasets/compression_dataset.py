"""Dataset for Phase 1 compression training.

Concatenates four sub-datasets (one per objective) into a single index space.
Sub-dataset sizes are proportional to objective weights. Supports L0->L1
curriculum transition with dataloader rebuild.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from torch.utils.data import Dataset

from bgkit.data.chat_template import (
    TOOL_CONFIGS,
    ChatTemplateConfig,
    build_messages,
    build_tools,
    select_variant,
    tokenize_with_sentinel,
)

logger = logging.getLogger(__name__)

# Maximum number of variants to probe for overhead estimation (ISSUE 7)
_MAX_OVERHEAD_PROBE_VARIANTS = 8

# Default L1 hard caps (BUG 4)
DEFAULT_MAX_FILES_PER_REPO = 32
DEFAULT_MAX_TOKENS_PER_FILE = 8192


def _compute_max_overhead(
    tokenizer,
    variants: list[dict[str, str]],
    config: ChatTemplateConfig,
    dummy_path: str = "src/example/placeholder.py",
    dummy_lang: str = "python",
) -> int:
    """Compute maximum template token overhead across a sample of variants.

    Probes up to _MAX_OVERHEAD_PROBE_VARIANTS evenly spaced variants to avoid
    O(N) tokenizer calls when variant banks are large (ISSUE 7).
    """
    max_overhead = 0
    n = len(variants)
    # Sample evenly spaced indices
    if n <= _MAX_OVERHEAD_PROBE_VARIANTS:
        probe_indices = range(n)
    else:
        step = n / _MAX_OVERHEAD_PROBE_VARIANTS
        probe_indices = [int(i * step) for i in range(_MAX_OVERHEAD_PROBE_VARIANTS)]

    tools = build_tools(config)
    for vi in probe_indices:
        variant = variants[vi]
        messages = build_messages(variant, config, dummy_path, dummy_lang, "X")
        template_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            tools=tools,
        )
        overhead_str = template_str.replace("X", "", 1)
        overhead_tokens = len(
            tokenizer.encode(overhead_str, add_special_tokens=False)
        )
        max_overhead = max(max_overhead, overhead_tokens)
    return max_overhead


@dataclass
class FileCompressionSample:
    """Single-file sample for L0 objectives."""

    objective: str
    content_token_ids: torch.Tensor
    content_attention_mask: torch.Tensor
    compression_ratio: float
    compression_level: int  # always 0
    target_token_ids: torch.Tensor
    target_attention_mask: torch.Tensor
    target_loss_mask: torch.Tensor
    prefix_ids: torch.Tensor
    compression_prompt_ids: torch.Tensor


@dataclass
class RepoCompressionSample:
    """Repo-grouped sample for L1 objectives."""

    objective: str
    file_token_ids: list[torch.Tensor]
    file_attention_masks: list[torch.Tensor]
    compression_ratio: float
    compression_level: int  # always 1
    target_token_ids: torch.Tensor
    target_attention_mask: torch.Tensor
    target_loss_mask: torch.Tensor
    prefix_ids: torch.Tensor
    compression_prompt_ids: torch.Tensor
    direct_l1: bool = False  # If True, skip L0 and compress raw tokens at L1


def _gather_l1_files(
    repo_indices: list[int],
    joined_indices: list[tuple[int, int]],
    token_ds,
    max_files: int = DEFAULT_MAX_FILES_PER_REPO,
    max_tokens: int = DEFAULT_MAX_TOKENS_PER_FILE,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Gather and cap repo files for L1 samples (BUG 4).

    Caps to max_files files and max_tokens tokens per file to prevent OOM.
    Returns (file_ids_list, file_masks_list).
    """
    # Cap number of files
    capped_indices = repo_indices[:max_files]

    file_ids_list: list[torch.Tensor] = []
    file_masks_list: list[torch.Tensor] = []

    for ri in capped_indices:
        rtok_idx, _ = joined_indices[ri]
        rinner = token_ds[rtok_idx]
        rids = rinner["token_ids"]
        # Cap token length per file
        if rids.size(0) > max_tokens:
            rids = rids[:max_tokens]
        file_ids_list.append(rids)
        file_masks_list.append(torch.ones(rids.size(0), dtype=torch.bool))

    return file_ids_list, file_masks_list


class DataReconstructionSubset(Dataset):
    """Sub-dataset for Objective 1: data reconstruction.

    Wraps MmapTokenDataset + optional MmapICEDataset + chat template.
    Always returns FileCompressionSample (L0 only).
    """

    def __init__(
        self,
        token_dataset,
        tokenizer,
        variant_bank: list[dict[str, str]],
        config: ChatTemplateConfig,
        seed: int = 42,
    ):
        self._token_ds = token_dataset
        self._tokenizer = tokenizer
        self._variants = variant_bank
        self._config = config
        self._base_seed = seed
        self._epoch_seed = seed

        # Precompute max template overhead for length estimates
        self._max_overhead = _compute_max_overhead(tokenizer, variant_bank, config)
        self._cached_lengths = self._token_ds.lengths + self._max_overhead

    def set_epoch(self, epoch: int) -> None:
        self._epoch_seed = self._base_seed + epoch

    def token_length(self, idx: int) -> int:
        return int(self._cached_lengths[idx])

    def __len__(self) -> int:
        return len(self._token_ds)

    def __getitem__(self, idx: int) -> FileCompressionSample:
        inner = self._token_ds[idx]
        content_ids = inner["token_ids"]
        file_path = inner.get("file_path", "unknown.txt")
        language = inner.get("language", "")

        variant = select_variant(self._variants, idx, self._epoch_seed)
        result = tokenize_with_sentinel(
            self._tokenizer, variant, self._config,
            file_path, language, content_ids,
        )

        content_mask = torch.ones(content_ids.size(0), dtype=torch.bool)

        return FileCompressionSample(
            objective="data_reconstruction",
            content_token_ids=content_ids,
            content_attention_mask=content_mask,
            compression_ratio=0.0,  # placeholder; actual ratio sampled by trainer
            compression_level=0,
            target_token_ids=result["token_ids"],
            target_attention_mask=torch.ones(result["token_ids"].size(0), dtype=torch.bool),
            target_loss_mask=result["loss_mask"],
            prefix_ids=result["prefix_ids"],
            compression_prompt_ids=result["compression_prompt_ids"],
        )


class DescriptionSubset(Dataset):
    """Sub-dataset for Objective 2: description generation.

    L0 phase: returns FileCompressionSample (single file -> description).
    L1 phase: returns RepoCompressionSample (repo files -> description).
    """

    def __init__(
        self,
        token_dataset,
        description_dataset,
        tokenizer,
        variant_bank: list[dict[str, str]],
        config: ChatTemplateConfig,
        seed: int = 42,
        max_files_per_repo: int = DEFAULT_MAX_FILES_PER_REPO,
        max_tokens_per_file: int = DEFAULT_MAX_TOKENS_PER_FILE,
        repo_description_dataset=None,
    ):
        self._token_ds = token_dataset
        self._desc_ds = description_dataset
        self._repo_desc_ds = repo_description_dataset
        self._tokenizer = tokenizer
        self._variants = variant_bank
        self._config = config
        self._base_seed = seed
        self._epoch_seed = seed
        self._l1_enabled = False
        self._max_files_per_repo = max_files_per_repo
        self._max_tokens_per_file = max_tokens_per_file

        # Build index of files that have both source tokens and descriptions.
        # Key-based join via (repo_path, file_path, commit_sha).
        self._joined_indices: list[tuple[int, int]] = []  # (token_idx, desc_valid_idx)
        self._build_joined_index()

        # Precompute max template overhead
        self._max_overhead = _compute_max_overhead(tokenizer, variant_bank, config)

        # Cache lengths: source content length + overhead for L0
        self._cached_lengths: np.ndarray = np.empty(len(self._joined_indices), dtype=np.int32)
        for i, (tok_idx, _) in enumerate(self._joined_indices):
            self._cached_lengths[i] = int(self._token_ds.lengths[tok_idx]) + self._max_overhead

        # For L1: group indices by (repo, commit) + cached metadata pylists
        self._repo_commit_groups: dict[tuple[str, str], list[int]] | None = None
        self._cached_repo_paths: list[str] | None = None
        self._cached_commit_shas: list[str] | None = None

    def _build_joined_index(self) -> None:
        """Build index of (token_idx, desc_idx) pairs joined by key."""
        self._joined_indices = []

        # Use public API (ISSUE 5)
        chunk_file_idx = self._token_ds.chunk_file_indices
        if chunk_file_idx is None:
            return

        meta = self._token_ds.get_metadata_table()
        if "repo_path" not in meta.column_names or "commit_sha" not in meta.column_names:
            return

        repo_paths = meta.column("repo_path").to_pylist()
        file_paths = meta.column("file_path").to_pylist()
        commit_shas = meta.column("commit_sha").to_pylist()

        for tok_idx in range(len(self._token_ds)):
            file_idx = int(chunk_file_idx[tok_idx])
            rp = repo_paths[file_idx]
            fp = file_paths[file_idx]
            cs = commit_shas[file_idx]

            desc_vi = self._desc_ds.lookup_index(rp, fp, cs)
            if desc_vi is not None:
                self._joined_indices.append((tok_idx, desc_vi))

    def enable_l1(self) -> None:
        """Switch from FileCompressionSample to RepoCompressionSample."""
        self._l1_enabled = True
        # Build repo-commit groups + cache metadata pylists
        if self._repo_commit_groups is None:
            self._repo_commit_groups = {}
            chunk_file_idx = self._token_ds.chunk_file_indices
            meta = self._token_ds.get_metadata_table()
            repo_paths = meta.column("repo_path").to_pylist()
            commit_shas = meta.column("commit_sha").to_pylist()
            self._cached_repo_paths = repo_paths
            self._cached_commit_shas = commit_shas
            for i, (tok_idx, _desc_idx) in enumerate(self._joined_indices):
                file_idx = int(chunk_file_idx[tok_idx])
                key = (repo_paths[file_idx], commit_shas[file_idx])
                if key not in self._repo_commit_groups:
                    self._repo_commit_groups[key] = []
                self._repo_commit_groups[key].append(i)

            # Recompute cached lengths for L1 repo-grouped samples
            for i, (tok_idx, _) in enumerate(self._joined_indices):
                file_idx = int(chunk_file_idx[tok_idx])
                key = (repo_paths[file_idx], commit_shas[file_idx])
                group = self._repo_commit_groups[key]
                n_files = min(len(group), self._max_files_per_repo)
                total_tokens = 0
                for gi in group[:n_files]:
                    gtok_idx, _ = self._joined_indices[gi]
                    total_tokens += min(
                        int(self._token_ds.lengths[gtok_idx]),
                        self._max_tokens_per_file,
                    )
                self._cached_lengths[i] = total_tokens + self._max_overhead

    def set_epoch(self, epoch: int) -> None:
        self._epoch_seed = self._base_seed + epoch

    def token_length(self, idx: int) -> int:
        return int(self._cached_lengths[idx])

    def __len__(self) -> int:
        return len(self._joined_indices)

    def __getitem__(self, idx: int) -> FileCompressionSample | RepoCompressionSample:
        tok_idx, desc_vi = self._joined_indices[idx]
        inner = self._token_ds[tok_idx]
        content_ids = inner["token_ids"]
        file_path = inner.get("file_path", "unknown.txt")
        language = inner.get("language", "")

        variant = select_variant(self._variants, idx, self._epoch_seed)

        if not self._l1_enabled:
            # L0: use file-level description target
            desc_sample = self._desc_ds[desc_vi]
            desc_ids = desc_sample["token_ids"]

            result = tokenize_with_sentinel(
                self._tokenizer, variant, self._config,
                file_path, language, desc_ids,
            )

            content_mask = torch.ones(content_ids.size(0), dtype=torch.bool)

            return FileCompressionSample(
                objective="description_generation",
                content_token_ids=content_ids,
                content_attention_mask=content_mask,
                compression_ratio=0.0,
                compression_level=0,
                target_token_ids=result["token_ids"],
                target_attention_mask=torch.ones(
                    result["token_ids"].size(0), dtype=torch.bool
                ),
                target_loss_mask=result["loss_mask"],
                prefix_ids=result["prefix_ids"],
                compression_prompt_ids=result["compression_prompt_ids"],
            )

        # L1: gather repo files grouped by (repo_path, commit_sha)
        chunk_file_idx = self._token_ds.chunk_file_indices
        file_idx = int(chunk_file_idx[tok_idx])
        rp = self._cached_repo_paths[file_idx]
        cs = self._cached_commit_shas[file_idx]
        group_key = (rp, cs)
        repo_indices = (
            self._repo_commit_groups.get(group_key, [idx])
            if self._repo_commit_groups else [idx]
        )

        file_ids_list, file_masks_list = _gather_l1_files(
            repo_indices, self._joined_indices, self._token_ds,
            max_files=self._max_files_per_repo,
            max_tokens=self._max_tokens_per_file,
        )

        # L1 target: prefer repo-level description, fall back to file-level
        repo_desc_sample = (
            self._repo_desc_ds.get_by_key(rp, cs)
            if self._repo_desc_ds is not None else None
        )
        if repo_desc_sample is not None:
            # Repo-level description: use repo_path as context, no language
            desc_ids = repo_desc_sample["token_ids"]
            result = tokenize_with_sentinel(
                self._tokenizer, variant, self._config,
                rp, "", desc_ids,
            )
        else:
            # Fallback to file-level description
            desc_sample = self._desc_ds[desc_vi]
            desc_ids = desc_sample["token_ids"]
            result = tokenize_with_sentinel(
                self._tokenizer, variant, self._config,
                file_path, language, desc_ids,
            )

        return RepoCompressionSample(
            objective="description_generation",
            file_token_ids=file_ids_list,
            file_attention_masks=file_masks_list,
            compression_ratio=0.0,
            compression_level=1,
            target_token_ids=result["token_ids"],
            target_attention_mask=torch.ones(
                result["token_ids"].size(0), dtype=torch.bool
            ),
            target_loss_mask=result["loss_mask"],
            prefix_ids=result["prefix_ids"],
            compression_prompt_ids=result["compression_prompt_ids"],
        )


class StructuralSubset(Dataset):
    """Sub-dataset for Objective 3: structural reconstruction.

    Same L0/L1 pattern as DescriptionSubset but uses MmapStructuralDataset.
    """

    def __init__(
        self,
        token_dataset,
        structural_dataset,
        tokenizer,
        variant_bank: list[dict[str, str]],
        config: ChatTemplateConfig,
        seed: int = 42,
        max_files_per_repo: int = DEFAULT_MAX_FILES_PER_REPO,
        max_tokens_per_file: int = DEFAULT_MAX_TOKENS_PER_FILE,
    ):
        self._token_ds = token_dataset
        self._struct_ds = structural_dataset
        self._tokenizer = tokenizer
        self._variants = variant_bank
        self._config = config
        self._base_seed = seed
        self._epoch_seed = seed
        self._l1_enabled = False
        self._max_files_per_repo = max_files_per_repo
        self._max_tokens_per_file = max_tokens_per_file

        # Build joined index
        self._joined_indices: list[tuple[int, int]] = []
        self._joined_keys: list[tuple[str, str, str]] = []  # (rp, fp, cs) per entry
        self._build_joined_index()

        # Precompute max template overhead
        self._max_overhead = _compute_max_overhead(tokenizer, variant_bank, config)

        # Cache lengths
        self._cached_lengths: np.ndarray = np.empty(len(self._joined_indices), dtype=np.int32)
        for i, (tok_idx, _) in enumerate(self._joined_indices):
            self._cached_lengths[i] = int(self._token_ds.lengths[tok_idx]) + self._max_overhead

        self._repo_commit_groups: dict[tuple[str, str], list[int]] | None = None
        self._cached_repo_paths: list[str] | None = None
        self._cached_commit_shas: list[str] | None = None

    def _build_joined_index(self) -> None:
        """Build index of (token_idx, struct_valid_idx) pairs joined by key."""
        self._joined_indices = []
        self._joined_keys = []

        # Use public API (ISSUE 5)
        chunk_file_idx = self._token_ds.chunk_file_indices
        if chunk_file_idx is None:
            return

        meta = self._token_ds.get_metadata_table()
        if "repo_path" not in meta.column_names or "commit_sha" not in meta.column_names:
            return

        repo_paths = meta.column("repo_path").to_pylist()
        file_paths = meta.column("file_path").to_pylist()
        commit_shas = meta.column("commit_sha").to_pylist()

        for tok_idx in range(len(self._token_ds)):
            file_idx = int(chunk_file_idx[tok_idx])
            rp = repo_paths[file_idx]
            fp = file_paths[file_idx]
            cs = commit_shas[file_idx]

            struct_vi = self._struct_ds.lookup_index(rp, fp, cs)
            if struct_vi is not None:
                self._joined_indices.append((tok_idx, struct_vi))
                self._joined_keys.append((rp, fp, cs))

    def enable_l1(self) -> None:
        """Switch from FileCompressionSample to RepoCompressionSample."""
        self._l1_enabled = True
        # Build repo-commit groups + cache metadata pylists
        if self._repo_commit_groups is None:
            self._repo_commit_groups = {}
            chunk_file_idx = self._token_ds.chunk_file_indices
            meta = self._token_ds.get_metadata_table()
            repo_paths = meta.column("repo_path").to_pylist()
            commit_shas = meta.column("commit_sha").to_pylist()
            self._cached_repo_paths = repo_paths
            self._cached_commit_shas = commit_shas
            for i, (tok_idx, _) in enumerate(self._joined_indices):
                file_idx = int(chunk_file_idx[tok_idx])
                key = (repo_paths[file_idx], commit_shas[file_idx])
                if key not in self._repo_commit_groups:
                    self._repo_commit_groups[key] = []
                self._repo_commit_groups[key].append(i)

            # Recompute cached lengths for L1 repo-grouped samples
            for i, (tok_idx, _) in enumerate(self._joined_indices):
                file_idx = int(chunk_file_idx[tok_idx])
                key = (repo_paths[file_idx], commit_shas[file_idx])
                group = self._repo_commit_groups[key]
                n_files = min(len(group), self._max_files_per_repo)
                total_tokens = 0
                for gi in group[:n_files]:
                    gtok_idx, _ = self._joined_indices[gi]
                    total_tokens += min(
                        int(self._token_ds.lengths[gtok_idx]),
                        self._max_tokens_per_file,
                    )
                self._cached_lengths[i] = total_tokens + self._max_overhead

    def set_epoch(self, epoch: int) -> None:
        self._epoch_seed = self._base_seed + epoch

    def token_length(self, idx: int) -> int:
        return int(self._cached_lengths[idx])

    def __len__(self) -> int:
        return len(self._joined_indices)

    def __getitem__(self, idx: int) -> FileCompressionSample | RepoCompressionSample:
        tok_idx, _struct_vi = self._joined_indices[idx]
        inner = self._token_ds[tok_idx]
        content_ids = inner["token_ids"]
        file_path = inner.get("file_path", "unknown.txt")
        language = inner.get("language", "")

        variant = select_variant(self._variants, idx, self._epoch_seed)

        # Get structural target with random type selection across
        # skeleton/dependency/module_summary views
        rp, fp, cs = self._joined_keys[idx]
        struct_sample = self._struct_ds.get_by_key(rp, fp, cs)
        struct_ids = struct_sample["token_ids"]

        result = tokenize_with_sentinel(
            self._tokenizer, variant, self._config,
            file_path, language, struct_ids,
        )

        if not self._l1_enabled:
            content_mask = torch.ones(content_ids.size(0), dtype=torch.bool)

            return FileCompressionSample(
                objective="structural_relational",
                content_token_ids=content_ids,
                content_attention_mask=content_mask,
                compression_ratio=0.0,
                compression_level=0,
                target_token_ids=result["token_ids"],
                target_attention_mask=torch.ones(
                    result["token_ids"].size(0), dtype=torch.bool
                ),
                target_loss_mask=result["loss_mask"],
                prefix_ids=result["prefix_ids"],
                compression_prompt_ids=result["compression_prompt_ids"],
            )

        # L1: gather repo files grouped by (repo_path, commit_sha)
        chunk_file_idx = self._token_ds.chunk_file_indices
        file_idx = int(chunk_file_idx[tok_idx])
        rp = self._cached_repo_paths[file_idx]
        cs = self._cached_commit_shas[file_idx]
        group_key = (rp, cs)
        repo_indices = (
            self._repo_commit_groups.get(group_key, [idx])
            if self._repo_commit_groups else [idx]
        )

        file_ids_list, file_masks_list = _gather_l1_files(
            repo_indices, self._joined_indices, self._token_ds,
            max_files=self._max_files_per_repo,
            max_tokens=self._max_tokens_per_file,
        )

        return RepoCompressionSample(
            objective="structural_relational",
            file_token_ids=file_ids_list,
            file_attention_masks=file_masks_list,
            compression_ratio=0.0,
            compression_level=1,
            target_token_ids=result["token_ids"],
            target_attention_mask=torch.ones(
                result["token_ids"].size(0), dtype=torch.bool
            ),
            target_loss_mask=result["loss_mask"],
            prefix_ids=result["prefix_ids"],
            compression_prompt_ids=result["compression_prompt_ids"],
        )


class CommitReproSubset(Dataset):
    """Sub-dataset for Objective 4: commit reproduction.

    Always returns RepoCompressionSample (commit tokens -> L1 directly).
    Wraps CommitReproDataset with chat template formatting.
    """

    def __init__(
        self,
        commit_dataset,
        tokenizer,
        variant_bank: list[dict[str, str]],
        config: ChatTemplateConfig,
        seed: int = 42,
    ):
        self._commit_ds = commit_dataset
        self._tokenizer = tokenizer
        self._variants = variant_bank
        self._config = config
        self._base_seed = seed
        self._epoch_seed = seed

        # Precompute max template overhead
        self._max_overhead = _compute_max_overhead(
            tokenizer, variant_bank, config, dummy_path="repo/commit", dummy_lang="",
        )
        self._cached_lengths = self._commit_ds.lengths + self._max_overhead

    def set_epoch(self, epoch: int) -> None:
        self._epoch_seed = self._base_seed + epoch

    def token_length(self, idx: int) -> int:
        return int(self._cached_lengths[idx])

    def __len__(self) -> int:
        return len(self._commit_ds)

    def __getitem__(self, idx: int) -> RepoCompressionSample:
        commit_sample = self._commit_ds[idx]
        commit_ids = commit_sample["token_ids"]

        variant = select_variant(self._variants, idx, self._epoch_seed)

        result = tokenize_with_sentinel(
            self._tokenizer, variant, self._config,
            "commit", "", commit_ids,
        )

        # Commit reproduction uses the commit tokens as a single "file"
        # in the repo-level sample format, with direct L1 (skip L0)
        return RepoCompressionSample(
            objective="commit_reproduction",
            file_token_ids=[commit_ids],
            file_attention_masks=[torch.ones(commit_ids.size(0), dtype=torch.bool)],
            compression_ratio=0.0,
            compression_level=1,
            target_token_ids=result["token_ids"],
            target_attention_mask=torch.ones(
                result["token_ids"].size(0), dtype=torch.bool
            ),
            target_loss_mask=result["loss_mask"],
            prefix_ids=result["prefix_ids"],
            compression_prompt_ids=result["compression_prompt_ids"],
            direct_l1=True,
        )


class CompressionDataset(Dataset):
    """Concatenated multi-objective dataset for compression training.

    Fixed index ranges: indices 0-N1 = data_reconstruction, N1-N2 = description,
    N2-N3 = structural, N3-N4 = commit reproduction.

    Sub-dataset sizes are proportional to objective weights.
    """

    # Canonical objective ordering
    OBJECTIVE_ORDER: ClassVar[list[str]] = [
        "data_reconstruction",
        "description_generation",
        "structural_relational",
        "commit_reproduction",
        "query_conditioned",
    ]

    DEFAULT_WEIGHTS: ClassVar[dict[str, float]] = {
        "data_reconstruction": 0.35,
        "description_generation": 0.15,
        "structural_relational": 0.12,
        "commit_reproduction": 0.20,
        "query_conditioned": 0.18,
    }

    def __init__(
        self,
        subsets: dict[str, Dataset],
        weights: dict[str, float] | None = None,
        seed: int = 42,
    ):
        self._subsets = subsets
        self._weights = weights or dict(self.DEFAULT_WEIGHTS)
        self._seed = seed

        # Curriculum state
        self._l1_enabled = False

        # Build offset table
        self._build_index()

    def _build_index(self) -> None:
        """Compute sub-dataset sizes proportional to weights and build offset table."""
        total_natural = sum(len(self._subsets[k]) for k in self._subsets)
        if total_natural == 0:
            self._offsets = [0]
            self._objective_order = []
            self._effective_sizes = []
            return

        # Normalize weights for present objectives only (skip empty subsets)
        present_keys = [k for k in self.OBJECTIVE_ORDER
                        if k in self._subsets and len(self._subsets[k]) > 0]
        weight_sum = sum(self._weights.get(k, 0.0) for k in present_keys)
        if weight_sum == 0:
            weight_sum = 1.0

        # Target total size = total natural size (maintain overall dataset size)
        target_total = total_natural
        target_sizes: dict[str, int] = {}
        for k in present_keys:
            w = self._weights.get(k, 0.0) / weight_sum
            target_sizes[k] = max(1, round(w * target_total))

        # Build ordered lists
        self._objective_order = present_keys
        self._effective_sizes = [target_sizes[k] for k in present_keys]

        # Build cumulative offsets: [0, N1, N1+N2, ...]
        self._offsets = [0]
        for sz in self._effective_sizes:
            self._offsets.append(self._offsets[-1] + sz)

    @classmethod
    def from_config(
        cls,
        cfg,
        tokenizer,
        seed: int = 42,
        objective_weights: dict[str, float] | None = None,
    ) -> CompressionDataset:
        """Factory: build from config, loading all sub-datasets.

        Args:
            cfg: The ``data:`` section of the training config (contains
                file_tokens_path, etc.).
            tokenizer: HuggingFace tokenizer for chat template encoding.
            seed: Random seed.
            objective_weights: Objective weight dict (from ``training.objectives``
                in the YAML). Falls back to DEFAULT_WEIGHTS if None.
        """
        from bgkit.data.datasets.commit_repro_dataset import CommitReproDataset
        from bgkit.data.datasets.description_dataset import (
            MmapDescriptionDataset,
            MmapRepoDescriptionDataset,
        )
        from bgkit.data.datasets.mmap_token_dataset import MmapTokenDataset
        from bgkit.data.datasets.structural_dataset import MmapStructuralDataset

        subsets: dict[str, Dataset] = {}
        weights = objective_weights or cls.DEFAULT_WEIGHTS

        # Load variant banks (per-objective when available, shared fallback)
        fallback_bank = [{
            "system_prompt": "You are a helpful assistant.",
            "user_prompt": "Read {file_path}",
            "compression_prompt": "Reproduce the file contents exactly.",
            "response_prefix": "Here is {file_path}:",
        }]

        variant_banks: dict[str, list] = {}
        variant_dir = getattr(cfg, "prompt_variants_dir", None)
        if variant_dir and Path(variant_dir).is_dir():
            p = Path(variant_dir)
            # Load shared fallback
            shared_bank = None
            if (p / "file_read_repro.json").exists():
                with open(p / "file_read_repro.json") as f:
                    shared_bank = json.load(f)
            # Load per-objective banks, falling back to shared
            for obj_key in [
                "file_read_repro", "description_gen", "structural_repro", "commit_repro",
            ]:
                bank_path = p / f"{obj_key}.json"
                if bank_path.exists():
                    with open(bank_path) as f:
                        bank = json.load(f)
                    if bank:  # guard against empty JSON arrays
                        variant_banks[obj_key] = bank
                        continue
                if shared_bank:
                    variant_banks[obj_key] = shared_bank
        elif variant_dir and Path(variant_dir).is_file():
            with open(variant_dir) as f:
                shared = json.load(f)
            if shared:
                for obj_key in [
                    "file_read_repro", "description_gen", "structural_repro", "commit_repro",
                ]:
                    variant_banks[obj_key] = shared

        max_seq_len = getattr(cfg, "max_seq_len", 8192)

        # Load shared datasets (BUG 2: use correct config field names)
        token_ds = None

        file_tokens_path = getattr(cfg, "file_tokens_path", None)
        if file_tokens_path and Path(file_tokens_path).exists():
            try:
                token_ds = MmapTokenDataset(
                    file_tokens_path,
                    max_seq_len=max_seq_len,
                    include_metadata=True,
                    require_commit_sha=True,
                )
            except (FileNotFoundError, ValueError) as e:
                logger.warning("Could not load token dataset: %s", e)

        # Warn if token dataset has empty commit_sha values (backfilled by
        # convert_tokens_to_npy when source parquet lacked the column).
        # Empty strings won't match description/structural keys, silently
        # disabling those objectives.
        if token_ds is not None:
            meta = token_ds.get_metadata_table()
            if "commit_sha" in meta.column_names:
                sha_col = meta.column("commit_sha")
                n_empty = sum(
                    1 for i in range(len(sha_col))
                    if not sha_col[i].as_py()
                )
                if n_empty > 0:
                    logger.warning(
                        "%d / %d samples have empty commit_sha — they won't "
                        "join with description/structural targets. Re-run "
                        "convert_tokens_to_npy.py with commit_sha-aware "
                        "parquet shards to fix.",
                        n_empty, len(sha_col),
                    )

        # Objective 1: Data reconstruction
        if token_ds is not None:
            config = TOOL_CONFIGS["file_read_repro"]
            subsets["data_reconstruction"] = DataReconstructionSubset(
                token_ds, tokenizer,
                variant_banks.get("file_read_repro", fallback_bank),
                config, seed=seed,
            )

        # Objective 2: Description generation
        # Converter writes scope-specific subdirs (file/, module/, repo/).
        # Config may point to parent dir or directly to file/ subdir.
        desc_path = getattr(cfg, "description_tokens_path", None)
        if desc_path:
            desc_p = Path(desc_path)
            if desc_p.is_dir() and (desc_p / "file").is_dir():
                desc_p = desc_p / "file"
            desc_path = str(desc_p)
        if token_ds is not None and desc_path and Path(desc_path).exists():
            try:
                desc_ds = MmapDescriptionDataset(desc_path, max_seq_len=2048)

                # Try loading repo-scope descriptions from sibling repo/ dir
                repo_desc_ds = None
                repo_desc_dir = Path(desc_path).parent / "repo"
                if repo_desc_dir.is_dir():
                    try:
                        repo_desc_ds = MmapRepoDescriptionDataset(
                            str(repo_desc_dir), max_seq_len=2048,
                        )
                    except (FileNotFoundError, ValueError) as e:
                        logger.warning("Could not load repo description dataset: %s", e)

                config = TOOL_CONFIGS["description_gen"]
                subsets["description_generation"] = DescriptionSubset(
                    token_ds, desc_ds, tokenizer,
                    variant_banks.get("description_gen", fallback_bank),
                    config, seed=seed,
                    repo_description_dataset=repo_desc_ds,
                )
            except (FileNotFoundError, ValueError) as e:
                logger.warning("Could not load description dataset: %s", e)

        # Objective 3: Structural reconstruction
        struct_path = getattr(cfg, "structural_tokens_path", None)
        if token_ds is not None and struct_path and Path(struct_path).exists():
            try:
                struct_ds = MmapStructuralDataset(struct_path, max_seq_len=2048)
                config = TOOL_CONFIGS["structural_repro"]
                subsets["structural_relational"] = StructuralSubset(
                    token_ds, struct_ds, tokenizer,
                    variant_banks.get("structural_repro", fallback_bank),
                    config, seed=seed,
                )
            except (FileNotFoundError, ValueError) as e:
                logger.warning("Could not load structural dataset: %s", e)

        # Objective 4: Commit reproduction
        commit_path = getattr(cfg, "commit_tokens_path", None)
        if commit_path and Path(commit_path).exists():
            try:
                commit_ds = CommitReproDataset(commit_path, max_seq_len=4096)
                config = TOOL_CONFIGS["commit_repro"]
                subsets["commit_reproduction"] = CommitReproSubset(
                    commit_ds, tokenizer,
                    variant_banks.get("commit_repro", fallback_bank),
                    config, seed=seed,
                )
            except (FileNotFoundError, ValueError) as e:
                logger.warning("Could not load commit dataset: %s", e)

        # Objective 5: Query-conditioned compression
        qa_path = getattr(cfg, "qa_data_dir", None)
        if token_ds is not None and qa_path and Path(qa_path).exists():
            try:
                from bgkit.data.datasets.qa_conditioned_dataset import (
                    MmapQAConditionedDataset,
                    QAConditionedSubset,
                )

                qa_ds = MmapQAConditionedDataset(qa_path, max_seq_len=2048)
                subsets["query_conditioned"] = QAConditionedSubset(
                    token_ds, qa_ds, tokenizer, seed=seed,
                )
            except (FileNotFoundError, ValueError) as e:
                logger.warning("Could not load QA conditioned dataset: %s", e)

        return cls(subsets=subsets, weights=weights, seed=seed)

    def set_curriculum_state(
        self,
        step: int,
        l1_enabled: bool,
    ) -> bool:
        """Update curriculum state. Returns True if L1 was newly enabled (needs rebuild)."""
        newly_enabled = l1_enabled and not self._l1_enabled
        if newly_enabled:
            self._l1_enabled = True
            self.rebuild_for_l1()
        return newly_enabled

    def set_epoch(self, epoch: int) -> None:
        """Propagate epoch to all sub-datasets."""
        for key in self._objective_order:
            subset = self._subsets[key]
            if hasattr(subset, "set_epoch"):
                subset.set_epoch(epoch)

    def token_length(self, idx: int) -> int:
        """Deterministic token length for the given index."""
        obj_name, local_idx = self._map_index(idx)
        subset = self._subsets[obj_name]
        if hasattr(subset, "token_length"):
            return subset.token_length(local_idx)
        # Fallback: use lengths array
        if hasattr(subset, "lengths"):
            return int(subset.lengths[local_idx])
        return 512  # conservative default

    def objective_for_idx(self, idx: int) -> str:
        """Return which objective an index belongs to."""
        obj_name, _ = self._map_index(idx)
        return obj_name

    def _map_index(self, idx: int) -> tuple[str, int]:
        """Map a global index to (objective_name, local_index)."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        for i, key in enumerate(self._objective_order):
            if idx < self._offsets[i + 1]:
                local_idx = idx - self._offsets[i]
                # Handle upsampling: wrap around if local_idx >= natural size
                natural_size = len(self._subsets[key])
                if natural_size > 0:
                    local_idx = local_idx % natural_size
                return key, local_idx

        # Should not reach here
        raise IndexError(f"Index {idx} could not be mapped")

    def __len__(self) -> int:
        if not self._offsets:
            return 0
        return self._offsets[-1]

    def __getitem__(self, idx: int) -> FileCompressionSample | RepoCompressionSample:
        obj_name, local_idx = self._map_index(idx)
        return self._subsets[obj_name][local_idx]

    def rebuild_for_l1(self) -> None:
        """Rebuild sub-datasets for L1 phase.

        - Enable L1 on description and structural subsets
        - Recompute sizes (upsample repo-level via index repetition)
        - Rebuild index mapping
        """
        for key in self._objective_order:
            subset = self._subsets[key]
            if hasattr(subset, "enable_l1"):
                subset.enable_l1()

        # Rebuild index with potentially different effective sizes
        self._build_index()
