"""Dataset for commit encoding training phase.

Backed by parallel mmap arrays with shared index. Does NOT extend
BaseMmapDataset (which filters zero-length rows and remaps indices,
breaking cross-array alignment). Instead implements the same operational
features directly: mmap-backed storage, worker-safe pickling, lazy
metadata loading.

Asymmetric encoder/decoder prompting:
  - Encoder gets commit message as prompt ("Encode these changes for
    the following commit message: {message}")
  - Decoder must reconstruct the full commit (including the message)
    purely from compressed survivors
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from bgkit.data.chat_template import (
    TOOL_CONFIGS,
    build_encoder_prefix_ids,
    select_variant,
    tokenize_with_sentinel,
)
from bgkit.data.datasets.compression_dataset import RepoCompressionSample

logger = logging.getLogger(__name__)


class CommitEncodingDataset(Dataset):
    """Dataset for two-level commit encoding training.

    Reads parallel mmap arrays produced by convert_commit_encoding_to_npy.py.
    Each sample contains per-file diffs (for L0→L1 compression) and a full
    serialized commit (decoder target).

    Encoder prompt is asymmetric: contains the commit message.
    Decoder template does NOT contain the commit message.
    """

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer,
        variant_bank: list[dict[str, str]],
        config=None,
        max_diff_tokens_per_file: int = 4096,
        max_files_per_commit: int = 16,
        max_message_tokens: int = 256,
        seed: int = 42,
    ):
        self._data_path = Path(data_dir)
        self._tokenizer = tokenizer
        self._variants = variant_bank
        self._config = config or TOOL_CONFIGS["commit_encoding"]
        self._max_diff_per_file = max_diff_tokens_per_file
        self._max_files = max_files_per_commit
        self._max_message_tokens = max_message_tokens
        self._base_seed = seed
        self._epoch_seed = seed

        # Validate manifest
        manifest_path = self._data_path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {data_dir}")
        with open(manifest_path) as f:
            self._manifest = json.load(f)

        expected_rows = self._manifest["row_count"]

        # Load offset arrays (small, kept in pickle state for batch sampler)
        self._target_offsets = np.load(self._data_path / "target_offsets.npy")
        self._diff_offsets = np.load(self._data_path / "diff_offsets.npy")
        self._file_boundary_offsets = np.load(
            self._data_path / "file_boundary_offsets.npy"
        )

        # Validate row counts match
        for name, offsets in [
            ("target_offsets", self._target_offsets),
            ("diff_offsets", self._diff_offsets),
            ("file_boundary_offsets", self._file_boundary_offsets),
        ]:
            actual = len(offsets) - 1
            if actual != expected_rows:
                raise ValueError(
                    f"{name} has {actual} rows but manifest says {expected_rows}"
                )

        # Validate metadata row count (read metadata header only)
        meta_path = self._data_path / "metadata.parquet"
        if meta_path.exists():
            meta_metadata = pq.read_metadata(meta_path)
            if meta_metadata.num_rows != expected_rows:
                raise ValueError(
                    f"metadata.parquet has {meta_metadata.num_rows} rows "
                    f"but manifest says {expected_rows}"
                )

        self._row_count = expected_rows

        # Precompute lengths for batch sampler (target token count + overhead estimate)
        target_lengths = (
            self._target_offsets[1:] - self._target_offsets[:-1]
        ).astype(np.int32)
        # Rough overhead: template tokens + encoder prompt
        overhead = 200
        self._lengths = target_lengths + overhead

        # Open mmap arrays
        self._target_tokens: np.ndarray | None = None
        self._diff_tokens: np.ndarray | None = None
        self._file_boundaries: np.ndarray | None = None
        self._reopen_mmaps()

        # Metadata is lazy-loaded per worker
        self._metadata = None

    @property
    def lengths(self) -> np.ndarray:
        """Token lengths per sample (for batch sampler)."""
        return self._lengths

    def set_epoch(self, epoch: int) -> None:
        self._epoch_seed = self._base_seed + epoch

    def token_length(self, idx: int) -> int:
        return int(self._lengths[idx])

    def __len__(self) -> int:
        return self._row_count

    def __getitem__(self, idx: int) -> RepoCompressionSample:
        # --- Load per-file diffs ---
        d_start = int(self._diff_offsets[idx])
        d_end = int(self._diff_offsets[idx + 1])
        all_diffs = self._diff_tokens[d_start:d_end]

        fb_start = int(self._file_boundary_offsets[idx])
        fb_end = int(self._file_boundary_offsets[idx + 1])
        bounds = self._file_boundaries[fb_start:fb_end]

        # Split diffs by file boundaries
        file_token_ids: list[torch.Tensor] = []
        file_attention_masks: list[torch.Tensor] = []
        n_bounds = len(bounds)
        for i in range(n_bounds - 1):
            if len(file_token_ids) >= self._max_files:
                break
            start = int(bounds[i])
            end = int(bounds[i + 1])
            if end <= start:
                continue
            file_ids = all_diffs[start:end]
            # Truncate per-file
            if len(file_ids) > self._max_diff_per_file:
                file_ids = file_ids[: self._max_diff_per_file]
            t = torch.from_numpy(file_ids.astype(np.int64))
            file_token_ids.append(t)
            file_attention_masks.append(torch.ones(t.size(0), dtype=torch.bool))

        # Must have at least 1 file (data guarantees >= 2, but be safe)
        if not file_token_ids:
            dummy = torch.zeros(1, dtype=torch.long)
            file_token_ids = [dummy]
            file_attention_masks = [torch.ones(1, dtype=torch.bool)]

        # --- Load decoder target ---
        t_start = int(self._target_offsets[idx])
        t_end = int(self._target_offsets[idx + 1])
        target_raw = torch.from_numpy(
            self._target_tokens[t_start:t_end].astype(np.int64)
        )

        # --- Build encoder prompt (asymmetric: contains commit message) ---
        meta = self._get_metadata()
        message = meta.column("message")[idx].as_py()
        msg_tokens = self._tokenizer.encode(message, add_special_tokens=False)
        if len(msg_tokens) > self._max_message_tokens:
            msg_tokens = msg_tokens[: self._max_message_tokens]
        msg_text = self._tokenizer.decode(msg_tokens)
        encoder_prompt = (
            f"Encode these changes for the following commit message: {msg_text}"
        )
        compression_prompt_ids = build_encoder_prefix_ids(
            self._tokenizer, encoder_prompt
        )

        # --- Build decoder template (does NOT contain commit message) ---
        variant = select_variant(self._variants, idx, self._epoch_seed)

        repo_id = meta.column("repo_path")[idx].as_py()

        result = tokenize_with_sentinel(
            self._tokenizer,
            variant,
            self._config,
            repo_id,
            "",  # no language for commits
            target_raw,
        )

        return RepoCompressionSample(
            objective="commit_encoding",
            file_token_ids=file_token_ids,
            file_attention_masks=file_attention_masks,
            compression_ratio=0.0,
            compression_level=1,
            target_token_ids=result["token_ids"],
            target_attention_mask=torch.ones(
                result["token_ids"].size(0), dtype=torch.bool
            ),
            target_loss_mask=result["loss_mask"],
            prefix_ids=result["prefix_ids"],
            compression_prompt_ids=compression_prompt_ids,
            bgkit_splice_start=int(result["bgkit_splice_start"].item()),
            bgkit_splice_len=int(result["bgkit_splice_len"].item()),
        )

    # --- Worker-safe pickling (mirrors MmapTokenDataset pattern) ---

    def __getstate__(self):
        """Exclude mmap arrays + metadata from pickle — workers re-open."""
        state = self.__dict__.copy()
        state["_target_tokens"] = None
        state["_diff_tokens"] = None
        state["_file_boundaries"] = None
        state["_metadata"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reopen_mmaps()

    def _reopen_mmaps(self):
        """Re-open mmap arrays from disk after unpickling in worker."""
        self._target_tokens = np.load(
            self._data_path / "target_tokens.npy", mmap_mode="r"
        )
        self._diff_tokens = np.load(
            self._data_path / "diff_tokens.npy", mmap_mode="r"
        )
        self._file_boundaries = np.load(
            self._data_path / "file_boundaries.npy", mmap_mode="r"
        )

    def _get_metadata(self):
        """Lazy-load metadata with column projection."""
        if self._metadata is None:
            self._metadata = pq.read_table(
                self._data_path / "metadata.parquet",
                columns=["message", "file_paths", "repo_path"],
            )
        return self._metadata
