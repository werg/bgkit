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
    BGKIT_TOOL_RESPONSE_SENTINEL,
    TOOL_CONFIGS,
    ChatTemplateConfig,
    build_messages,
    build_tools,
    select_variant,
    tokenize_with_sentinel,
)
from bgkit.data.datasets.tokenizer_compat import assert_subset_tokenizers_compatible

logger = logging.getLogger(__name__)

# Maximum number of variants to probe for overhead estimation (ISSUE 7)
_MAX_OVERHEAD_PROBE_VARIANTS = 8

# Default L1 hard caps (BUG 4)
DEFAULT_MAX_FILES_PER_REPO = 32
DEFAULT_MAX_TOKENS_PER_FILE = 8192

_DEFAULT_VARIANT_BANKS: dict[str, list[dict[str, str]]] = {
    "file_read_repro": [{
        "system_prompt": (
            "You are an AI coding assistant with access to the "
            "bgkit_read_file tool for reading file contents."
        ),
        "user_prompt": "Read the file `{file_path}`",
        "compression_prompt": "Return the file contents verbatim",
        "response_prefix": "Here are the contents of `{file_path}`:",
    }],
    "description_gen": [{
        "system_prompt": (
            "You are an AI coding assistant with access to the "
            "bgkit_describe tool for generating descriptions from "
            "compressed code context."
        ),
        "user_prompt": "Describe `{file_path}`",
        "compression_prompt": "Generate an accurate description of the target",
        "response_prefix": "Here is a description of `{file_path}`:",
    }],
    "structural_repro": [{
        "system_prompt": (
            "You are an AI coding assistant with access to the "
            "bgkit_extract_structure tool for extracting structure from "
            "compressed code context."
        ),
        "user_prompt": "Extract the structure of `{file_path}`",
        "compression_prompt": "Extract the structural information faithfully",
        "response_prefix": "Here is the structure of `{file_path}`:",
    }],
    "commit_repro": [{
        "system_prompt": (
            "You are an AI coding assistant with access to the "
            "bgkit_reproduce_commit tool for reconstructing commits from "
            "compressed repository context."
        ),
        "user_prompt": "Reproduce the commit from repository {file_path}",
        "compression_prompt": "Reproduce the complete commit from compressed context",
        "response_prefix": "Here is the reconstructed commit:",
    }],
}


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
        messages = build_messages(
            variant,
            config,
            dummy_path,
            dummy_lang,
            "X",
            tool_response_content=BGKIT_TOOL_RESPONSE_SENTINEL,
        )
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


def _load_variant_bank(path: Path) -> list[dict[str, str]] | None:
    """Load a variant bank JSON file, returning None when absent or empty."""
    if not path.exists():
        return None
    with open(path) as f:
        bank = json.load(f)
    return bank or None


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
    bgkit_splice_start: int = -1
    bgkit_splice_len: int = 0
    # Optional per-position bool mask aligned 1:1 with content_token_ids.
    # When non-None and ``survivorship.l0.forced_survivor_bce_weight > 0``,
    # ``compute_survivorship_losses`` adds a BCE loss on the head's
    # ``base_raw`` against this mask. Populated by DataReconstructionSubset
    # from the Falcon companion's ``forced_survivor_indices`` so during
    # Phase D's compression ramp the head's natural selection stays aligned
    # with the (well-aligned-tokenizer-pair) positions the projection
    # learned in Phase A/B/C. ``None`` when the underlying token dataset
    # has no companion or the chunk has no forced indices.
    forced_survivor_mask: torch.Tensor | None = None
    # Per-forced-position Falcon pair token ids ``(N_forced, 2)``, sourced
    # from the companion's ``target_falcon_pair_ids``. Used by Phase C's
    # "ablation eval" (perfect-projection baseline): the decoder receives
    # ``embed_tokens(target_falcon_pair_ids)`` in place of the projection
    # output at survivor positions, isolating projection-quality cost vs
    # decoder LM cost. None when no companion is loaded.
    target_falcon_pair_ids: torch.Tensor | None = None


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
    bgkit_splice_start: int = -1
    bgkit_splice_len: int = 0


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

    Wraps MmapTokenDataset + chat template.
    Always returns FileCompressionSample (L0 only).
    """

    def __init__(
        self,
        token_dataset,
        tokenizer,
        variant_bank: list[dict[str, str]],
        config: ChatTemplateConfig,
        seed: int = 42,
        encoder_tokenizer=None,
        max_chunks_per_repo: int | None = None,
    ):
        self._token_ds = token_dataset
        self._tokenizer = tokenizer
        self._encoder_tokenizer = encoder_tokenizer or tokenizer
        self._variants = variant_bank
        self._config = config
        self._base_seed = seed
        self._epoch_seed = seed

        # When a Falcon companion is loaded, drop chunks that the converter
        # marked pathological / alignment-skipped — they remain
        # ``__getitem__``-addressable on MmapTokenDataset with zero falcon
        # tokens or zero forced survivors, which would feed an empty target
        # into the decoder. ``companion_valid_indices`` returns None when no
        # companion is loaded, in which case every chunk is valid.
        valid_indices = getattr(token_dataset, "companion_valid_indices", None)
        self._valid_indices = (
            np.asarray(valid_indices, dtype=np.int64)
            if valid_indices is not None
            else None
        )

        # Per-repo cap (heavy-tail rebalancing). Mirrors the cap that
        # forced_adapt's ``_ValidCompanionSubset`` introduced: the raw
        # corpus is power-law skewed (top 1% of repos = ~17% of chunks,
        # top repo has 198x the median's chunk count). Capping each
        # repo to ``max_chunks_per_repo`` brings the top-1% share down
        # to ~3% and gives every repo roughly comparable exposure
        # across an epoch. None disables (default).
        if max_chunks_per_repo is not None and int(max_chunks_per_repo) > 0:
            cap = int(max_chunks_per_repo)
            self._valid_indices = self._apply_per_repo_cap(
                token_dataset,
                self._valid_indices,
                cap=cap,
                seed=seed,
            )

        # Precompute max template overhead for length estimates
        self._max_overhead = _compute_max_overhead(tokenizer, variant_bank, config)
        content_lengths = np.asarray(self._token_ds.lengths, dtype=np.int64)
        decoder_lengths = np.asarray(
            getattr(self._token_ds, "decoder_lengths", content_lengths),
            dtype=np.int64,
        )
        if self._valid_indices is not None:
            content_lengths = content_lengths[self._valid_indices]
            decoder_lengths = decoder_lengths[self._valid_indices]
        self._cached_content_lengths = content_lengths
        self._cached_decoder_lengths = decoder_lengths + self._max_overhead
        self._cached_lengths = np.maximum(content_lengths, decoder_lengths) + (
            self._max_overhead
        )

    @staticmethod
    def _apply_per_repo_cap(
        token_dataset,
        valid_indices: np.ndarray | None,
        *,
        cap: int,
        seed: int,
    ) -> np.ndarray:
        """Subsample chunk indices so no repo contributes more than ``cap``.

        Operates on chunk indices into ``token_dataset`` (length = n_chunks).
        Returns a sorted ndarray of selected chunk indices. If
        ``valid_indices`` is provided, only those are eligible.
        """
        meta = token_dataset.get_metadata_table()
        repo_col = meta.column("repo_path").to_numpy(zero_copy_only=False)
        chunk_file_idx = token_dataset.chunk_file_indices
        if chunk_file_idx is None:
            raise ValueError(
                "max_chunks_per_repo requires MmapTokenDataset(include_metadata=True)"
            )
        if valid_indices is None:
            eligible = np.arange(len(token_dataset), dtype=np.int64)
        else:
            eligible = np.asarray(valid_indices, dtype=np.int64)
        repo_for_chunk = repo_col[chunk_file_idx[eligible]]
        rng = np.random.default_rng(int(seed))
        unique_repos, inverse = np.unique(repo_for_chunk, return_inverse=True)
        keep_local: list[int] = []
        for repo_id in range(unique_repos.size):
            in_repo = np.flatnonzero(inverse == repo_id)
            if in_repo.size <= cap:
                keep_local.extend(in_repo.tolist())
            else:
                chosen = rng.choice(in_repo, size=cap, replace=False)
                keep_local.extend(chosen.tolist())
        keep_local_arr = np.array(sorted(keep_local), dtype=np.int64)
        result = eligible[keep_local_arr]
        logger.info(
            "compression_dataset_per_repo_cap_applied",
            cap=cap,
            unique_repos=int(unique_repos.size),
            kept_chunks=int(result.size),
            dropped_chunks=int(eligible.size - result.size),
        )
        return result

    def _inner_index(self, idx: int) -> int:
        if self._valid_indices is None:
            return int(idx)
        return int(self._valid_indices[idx])

    def set_epoch(self, epoch: int) -> None:
        self._epoch_seed = self._base_seed + epoch

    def token_length(self, idx: int) -> int:
        return int(self._cached_lengths[idx])

    def content_token_length(self, idx: int) -> int:
        return int(self._cached_content_lengths[idx])

    def decoder_token_length(self, idx: int) -> int:
        return int(self._cached_decoder_lengths[idx])

    def __len__(self) -> int:
        if self._valid_indices is not None:
            return int(self._valid_indices.size)
        return len(self._token_ds)

    def __getitem__(self, idx: int) -> FileCompressionSample:
        inner = self._token_ds[self._inner_index(idx)]
        content_ids = inner["token_ids"]
        decoder_content_ids = inner.get("decoder_content_token_ids", content_ids)
        file_path = inner.get("file_path", "unknown.txt")
        language = inner.get("language", "")

        variant = select_variant(self._variants, idx, self._epoch_seed)
        result = tokenize_with_sentinel(
            self._tokenizer, variant, self._config,
            file_path, language, decoder_content_ids,
            encoder_tokenizer=self._encoder_tokenizer,
        )

        content_mask = torch.ones(content_ids.size(0), dtype=torch.bool)

        # Build per-position forced-survivor bool mask if the underlying
        # MmapTokenDataset has a Falcon companion. Used downstream by
        # ``compute_survivorship_losses`` to BCE-supervise the head against
        # the forced selection (Phase D head re-alignment).
        forced_indices = inner.get("forced_survivor_indices", None)
        forced_mask: torch.Tensor | None = None
        target_pair_ids: torch.Tensor | None = None
        if forced_indices is not None and forced_indices.numel() > 0:
            forced_mask = torch.zeros(content_ids.size(0), dtype=torch.bool)
            # Indices are chunk-local Qwen positions; clamp defensively in
            # case a stale companion ever drifts past the chunk length.
            valid = (forced_indices >= 0) & (forced_indices < content_ids.size(0))
            valid_positions = forced_indices[valid].to(torch.long)
            forced_mask[valid_positions] = True
            # Per-forced-position Falcon pair IDs aligned with the mask.
            # forced_survivor_indices may not be sorted, but downstream
            # ``content[forced_mask]`` enumerates in content-position order,
            # so we sort pair_ids by chunk-local position too — that way
            # the i-th pair_id always matches the i-th survivor.
            pair_ids_full = inner.get("target_falcon_pair_ids", None)
            if pair_ids_full is not None and pair_ids_full.numel() > 0:
                sort_order = valid_positions.argsort()
                target_pair_ids = pair_ids_full[valid][sort_order].to(torch.long)

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
            bgkit_splice_start=int(result["bgkit_splice_start"].item()),
            bgkit_splice_len=int(result["bgkit_splice_len"].item()),
            forced_survivor_mask=forced_mask,
            target_falcon_pair_ids=target_pair_ids,
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
        encoder_tokenizer=None,
    ):
        self._token_ds = token_dataset
        self._desc_ds = description_dataset
        self._repo_desc_ds = repo_description_dataset
        self._tokenizer = tokenizer
        self._encoder_tokenizer = encoder_tokenizer or tokenizer
        assert_subset_tokenizers_compatible(
            subset_name="DescriptionSubset",
            encoder_tokenizer=self._encoder_tokenizer,
            decoder_tokenizer=self._tokenizer,
            objective="description_generation",
        )
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
                encoder_tokenizer=self._encoder_tokenizer,
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
                bgkit_splice_start=int(result["bgkit_splice_start"].item()),
                bgkit_splice_len=int(result["bgkit_splice_len"].item()),
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
                encoder_tokenizer=self._encoder_tokenizer,
            )
        else:
            # Fallback to file-level description
            desc_sample = self._desc_ds[desc_vi]
            desc_ids = desc_sample["token_ids"]
            result = tokenize_with_sentinel(
                self._tokenizer, variant, self._config,
                file_path, language, desc_ids,
                encoder_tokenizer=self._encoder_tokenizer,
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
            bgkit_splice_start=int(result["bgkit_splice_start"].item()),
            bgkit_splice_len=int(result["bgkit_splice_len"].item()),
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
        encoder_tokenizer=None,
    ):
        self._token_ds = token_dataset
        self._struct_ds = structural_dataset
        self._tokenizer = tokenizer
        self._encoder_tokenizer = encoder_tokenizer or tokenizer
        assert_subset_tokenizers_compatible(
            subset_name="StructuralSubset",
            encoder_tokenizer=self._encoder_tokenizer,
            decoder_tokenizer=self._tokenizer,
            objective="structural_relational",
        )
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
            encoder_tokenizer=self._encoder_tokenizer,
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
                bgkit_splice_start=int(result["bgkit_splice_start"].item()),
                bgkit_splice_len=int(result["bgkit_splice_len"].item()),
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
            bgkit_splice_start=int(result["bgkit_splice_start"].item()),
            bgkit_splice_len=int(result["bgkit_splice_len"].item()),
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
        encoder_tokenizer=None,
    ):
        self._commit_ds = commit_dataset
        self._tokenizer = tokenizer
        self._encoder_tokenizer = encoder_tokenizer or tokenizer
        assert_subset_tokenizers_compatible(
            subset_name="CommitReproSubset",
            encoder_tokenizer=self._encoder_tokenizer,
            decoder_tokenizer=self._tokenizer,
            objective="commit_reproduction",
        )
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
            encoder_tokenizer=self._encoder_tokenizer,
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
            bgkit_splice_start=int(result["bgkit_splice_start"].item()),
            bgkit_splice_len=int(result["bgkit_splice_len"].item()),
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
        total_available = sum(len(self._subsets[k]) for k in self._subsets)
        if total_available == 0:
            self._offsets = [0]
            self._objective_order = []
            self._effective_sizes = []
            return

        # Normalize weights for present objectives only. Explicitly zero or
        # omitted objectives are skipped; they must not leak a minimum sample
        # into decoder-family-specific runs such as Falcon data reconstruction.
        present_keys = [k for k in self.OBJECTIVE_ORDER
                        if (
                            k in self._subsets
                            and len(self._subsets[k]) > 0
                            and self._weights.get(k, 0.0) > 0.0
                        )]
        weight_sum = sum(self._weights.get(k, 0.0) for k in present_keys)
        if weight_sum == 0:
            weight_sum = 1.0

        # Target total size = included natural size. Zero-weight objectives
        # should not affect sampling composition or epoch length.
        target_total = sum(len(self._subsets[k]) for k in present_keys)
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
        encoder_tokenizer=None,
    ) -> CompressionDataset:
        """Factory: build from config, loading all sub-datasets.

        Args:
            cfg: The ``data:`` section of the training config (contains
                file_tokens_path, etc.).
            tokenizer: HuggingFace tokenizer for decoder-side chat template encoding.
            seed: Random seed.
            objective_weights: Objective weight dict (from ``training.objectives``
                in the YAML). Falls back to DEFAULT_WEIGHTS if None.
            encoder_tokenizer: Optional tokenizer for encoder-side compression
                prompts. Defaults to ``tokenizer`` when encoder and decoder
                token families match.
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
        encoder_tokenizer = encoder_tokenizer or tokenizer

        variant_banks: dict[str, list] = {}
        variant_dir = getattr(cfg, "prompt_variants_dir", None)
        # Per-objective override map: data.variant_bank_overrides:
        #   file_read_repro: document_verbatim_repro
        # tells the file_read_repro objective to load
        # ``document_verbatim_repro.json`` instead of ``file_read_repro.json``.
        # Use this when the corpus is prose (FineWeb-Edu) and the default
        # code-framed file_read prompts ("You are an AI coding assistant
        # with access to bgkit_read_file") would feed wildly off-distribution
        # framing to the decoder. The override file must live in the same
        # ``prompt_variants_dir`` and follow the same per-variant schema
        # (system_prompt, user_prompt, compression_prompt, response_prefix).
        variant_bank_overrides = dict(getattr(cfg, "variant_bank_overrides", {}) or {})
        if variant_dir and Path(variant_dir).is_dir():
            p = Path(variant_dir)
            default_files = {
                "file_read_repro": ["file_read_repro.json"],
                "description_gen": ["description_gen.json"],
                "structural_repro": ["structural_repro.json"],
                # Step 4 stores commit prompts in commit_encoding.json; accept
                # that file as an alias when the step-5-specific name is absent.
                "commit_repro": ["commit_repro.json", "commit_encoding.json"],
            }
            bank_candidates: dict[str, list[str]] = {}
            for obj_key, defaults in default_files.items():
                override = variant_bank_overrides.get(obj_key)
                if override:
                    # Prepend the override file (with .json appended if absent).
                    fn = override if str(override).endswith(".json") else f"{override}.json"
                    bank_candidates[obj_key] = [fn, *defaults]
                else:
                    bank_candidates[obj_key] = defaults
            for obj_key, candidates in bank_candidates.items():
                bank = None
                for candidate in candidates:
                    bank = _load_variant_bank(p / candidate)
                    if bank:
                        break
                if bank is None:
                    bank = _DEFAULT_VARIANT_BANKS[obj_key]
                variant_banks[obj_key] = bank
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
        companion_dir = getattr(cfg, "falcon_companion_dir", None) or getattr(
            cfg,
            "companion_dir",
            None,
        )
        if file_tokens_path and Path(file_tokens_path).exists():
            try:
                token_ds = MmapTokenDataset(
                    file_tokens_path,
                    max_seq_len=max_seq_len,
                    include_metadata=True,
                    require_commit_sha=True,
                    companion_dir=companion_dir,
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
            max_chunks_per_repo = getattr(cfg, "max_chunks_per_repo", None)
            subsets["data_reconstruction"] = DataReconstructionSubset(
                token_ds, tokenizer,
                variant_banks.get("file_read_repro", _DEFAULT_VARIANT_BANKS["file_read_repro"]),
                config, seed=seed,
                encoder_tokenizer=encoder_tokenizer,
                max_chunks_per_repo=max_chunks_per_repo,
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
                    variant_banks.get("description_gen", _DEFAULT_VARIANT_BANKS["description_gen"]),
                    config, seed=seed,
                    repo_description_dataset=repo_desc_ds,
                    encoder_tokenizer=encoder_tokenizer,
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
                    variant_banks.get(
                        "structural_repro",
                        _DEFAULT_VARIANT_BANKS["structural_repro"],
                    ),
                    config, seed=seed,
                    encoder_tokenizer=encoder_tokenizer,
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
                    variant_banks.get("commit_repro", _DEFAULT_VARIANT_BANKS["commit_repro"]),
                    config, seed=seed,
                    encoder_tokenizer=encoder_tokenizer,
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
                    token_ds,
                    qa_ds,
                    tokenizer,
                    seed=seed,
                    encoder_tokenizer=encoder_tokenizer,
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

    def content_token_length(self, idx: int) -> int:
        """Content-only token length (encoder input size, no template overhead).

        Drives the ``min_sample_length`` filter, which cares about how
        much actual content the encoder sees — not the chat-formatted
        decoder input. Every subset follows the same shape:
        ``_cached_lengths = content + _max_overhead``, so we recover the
        content length by subtracting the subset's per-template overhead.
        """
        obj_name, local_idx = self._map_index(idx)
        subset = self._subsets[obj_name]
        if hasattr(subset, "content_token_length"):
            return int(subset.content_token_length(local_idx))
        overhead = getattr(subset, "_max_overhead", 0)
        return max(0, self.token_length(idx) - int(overhead))

    def decoder_token_length(self, idx: int) -> int:
        """Decoder-side target length estimate for padded decoder batching.

        ``token_length`` intentionally uses the larger of encoder-content and
        decoder-token lengths so packed encoder/decoder phases stay under a
        single conservative budget. Falcon-H1 is different: its decoder
        attention path pads each microbatch, so batching should be driven by
        the decoder sequence length that controls ``B * max_len^2`` work.
        """
        obj_name, local_idx = self._map_index(idx)
        subset = self._subsets[obj_name]
        if hasattr(subset, "decoder_token_length"):
            return int(subset.decoder_token_length(local_idx))
        return self.token_length(idx)

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
