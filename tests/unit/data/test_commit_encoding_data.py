"""Tests for commit encoding dataset and data pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from bgkit.data.chat_template import TOOL_CONFIGS


class _MockTokenizer:
    """Module-level mock tokenizer (must be picklable)."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 256 for c in text[:50]]

    def decode(self, ids):
        return "decoded text"

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        return "".join(parts)


def _write_test_commit_encoding_data(
    data_dir: Path,
    n_rows: int = 10,
    n_files_per_commit: int = 3,
    tokens_per_file: int = 50,
    target_tokens: int = 200,
) -> None:
    """Write minimal test commit encoding mmap data."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # Target tokens
    all_target = []
    target_lengths = []
    for _ in range(n_rows):
        t = np.random.randint(100, 10000, size=target_tokens, dtype=np.int32)
        all_target.append(t)
        target_lengths.append(len(t))
    target_flat = np.concatenate(all_target)
    target_offsets = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(target_lengths, out=target_offsets[1:])

    # Diff tokens (per-file diffs concatenated per commit)
    all_diffs = []
    diff_lengths = []
    all_boundaries = []
    boundary_lengths = []
    for _ in range(n_rows):
        commit_diffs = []
        bounds = [0]
        for _f in range(n_files_per_commit):
            file_toks = np.random.randint(100, 10000, size=tokens_per_file, dtype=np.int32)
            commit_diffs.append(file_toks)
            bounds.append(bounds[-1] + len(file_toks))
        flat_diffs = np.concatenate(commit_diffs)
        all_diffs.append(flat_diffs)
        diff_lengths.append(len(flat_diffs))
        all_boundaries.append(np.array(bounds, dtype=np.int32))
        boundary_lengths.append(len(bounds))

    diff_flat = np.concatenate(all_diffs)
    diff_offsets = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(diff_lengths, out=diff_offsets[1:])

    boundaries_flat = np.concatenate(all_boundaries)
    boundary_offsets = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(boundary_lengths, out=boundary_offsets[1:])

    # Save arrays
    np.save(data_dir / "target_tokens.npy", target_flat)
    np.save(data_dir / "target_offsets.npy", target_offsets)
    np.save(data_dir / "diff_tokens.npy", diff_flat)
    np.save(data_dir / "diff_offsets.npy", diff_offsets)
    np.save(data_dir / "file_boundaries.npy", boundaries_flat)
    np.save(data_dir / "file_boundary_offsets.npy", boundary_offsets)

    # Metadata parquet
    messages = [f"Fix issue #{i}" for i in range(n_rows)]
    file_paths = [[f"src/file{j}.py" for j in range(n_files_per_commit)] for _ in range(n_rows)]
    repo_paths = [f"owner/repo{i}" for i in range(n_rows)]
    shas = [f"abc{i:04d}" for i in range(n_rows)]

    meta_table = pa.table({
        "message": pa.array(messages),
        "file_paths": pa.array(file_paths, type=pa.list_(pa.string())),
        "repo_path": pa.array(repo_paths),
        "sha": pa.array(shas),
    })
    pq.write_table(meta_table, data_dir / "metadata.parquet")

    # Manifest
    manifest = {
        "schema_version": 1,
        "row_count": n_rows,
        "total_target_tokens": int(target_offsets[-1]),
        "total_diff_tokens": int(diff_offsets[-1]),
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest))


class TestCommitEncodingDataset:
    @pytest.fixture
    def data_dir(self, tmp_path):
        d = tmp_path / "commit_encoding"
        _write_test_commit_encoding_data(d, n_rows=5)
        return d

    @pytest.fixture
    def tokenizer(self):
        """Minimal mock tokenizer."""
        return _MockTokenizer()

    def test_creation_and_length(self, data_dir, tokenizer):
        from bgkit.data.datasets.commit_encoding_dataset import CommitEncodingDataset

        ds = CommitEncodingDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            variant_bank=[{
                "system_prompt": "Test",
                "user_prompt": "Reproduce commit from {file_path}",
                "compression_prompt": "Reproduce the commit",
                "response_prefix": "Here is the commit:",
            }],
            config=TOOL_CONFIGS["commit_encoding"],
        )
        assert len(ds) == 5

    def test_getitem_returns_repo_compression_sample(self, data_dir, tokenizer):
        from bgkit.data.datasets.commit_encoding_dataset import CommitEncodingDataset
        from bgkit.data.datasets.compression_dataset import RepoCompressionSample

        ds = CommitEncodingDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            variant_bank=[{
                "system_prompt": "Test",
                "user_prompt": "Reproduce commit from {file_path}",
                "compression_prompt": "Reproduce the commit",
                "response_prefix": "Here is the commit:",
            }],
            config=TOOL_CONFIGS["commit_encoding"],
        )
        sample = ds[0]
        assert isinstance(sample, RepoCompressionSample)
        assert sample.objective == "commit_encoding"
        assert len(sample.file_token_ids) >= 1
        assert sample.target_token_ids.ndim == 1
        assert sample.compression_prompt_ids.ndim == 1

    def test_per_file_structure(self, data_dir, tokenizer):
        """Each sample should have multiple file diffs (cross-file commits)."""
        from bgkit.data.datasets.commit_encoding_dataset import CommitEncodingDataset

        ds = CommitEncodingDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            variant_bank=[{
                "system_prompt": "Test",
                "user_prompt": "Reproduce commit from {file_path}",
                "compression_prompt": "Reproduce the commit",
                "response_prefix": "Here is the commit:",
            }],
            config=TOOL_CONFIGS["commit_encoding"],
        )
        sample = ds[0]
        # Test data has 3 files per commit
        assert len(sample.file_token_ids) == 3
        assert len(sample.file_attention_masks) == 3
        for t in sample.file_token_ids:
            assert isinstance(t, torch.Tensor)
            assert t.dtype == torch.int64

    def test_max_files_truncation(self, data_dir, tokenizer):
        """max_files_per_commit should limit number of files."""
        from bgkit.data.datasets.commit_encoding_dataset import CommitEncodingDataset

        ds = CommitEncodingDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            variant_bank=[{
                "system_prompt": "Test",
                "user_prompt": "Reproduce commit from {file_path}",
                "compression_prompt": "Reproduce the commit",
                "response_prefix": "Here is the commit:",
            }],
            config=TOOL_CONFIGS["commit_encoding"],
            max_files_per_commit=2,
        )
        sample = ds[0]
        assert len(sample.file_token_ids) <= 2

    def test_pickle_roundtrip(self, data_dir, tokenizer):
        """Dataset should survive pickle/unpickle (for DataLoader workers)."""
        import pickle

        from bgkit.data.datasets.commit_encoding_dataset import CommitEncodingDataset

        ds = CommitEncodingDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            variant_bank=[{
                "system_prompt": "Test",
                "user_prompt": "Reproduce commit from {file_path}",
                "compression_prompt": "Reproduce the commit",
                "response_prefix": "Here is the commit:",
            }],
            config=TOOL_CONFIGS["commit_encoding"],
        )

        pickled = pickle.dumps(ds)
        ds2 = pickle.loads(pickled)
        assert len(ds2) == len(ds)
        # Verify we can still access items after unpickle
        sample = ds2[0]
        assert sample.objective == "commit_encoding"

    def test_manifest_row_count_mismatch(self, tmp_path, tokenizer):
        """Should raise if manifest row_count doesn't match offset arrays."""
        from bgkit.data.datasets.commit_encoding_dataset import CommitEncodingDataset

        data_dir = tmp_path / "bad_data"
        _write_test_commit_encoding_data(data_dir, n_rows=5)
        # Corrupt manifest
        manifest = json.loads((data_dir / "manifest.json").read_text())
        manifest["row_count"] = 999
        (data_dir / "manifest.json").write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="rows but manifest says 999"):
            CommitEncodingDataset(
                data_dir=data_dir,
                tokenizer=tokenizer,
                variant_bank=[{
                    "system_prompt": "Test",
                    "user_prompt": "Reproduce commit from {file_path}",
                    "compression_prompt": "Reproduce the commit",
                    "response_prefix": "Here is the commit:",
                }],
                config=TOOL_CONFIGS["commit_encoding"],
            )

    def test_token_length_returns_positive(self, data_dir, tokenizer):
        from bgkit.data.datasets.commit_encoding_dataset import CommitEncodingDataset

        ds = CommitEncodingDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            variant_bank=[{
                "system_prompt": "Test",
                "user_prompt": "Reproduce commit from {file_path}",
                "compression_prompt": "Reproduce the commit",
                "response_prefix": "Here is the commit:",
            }],
            config=TOOL_CONFIGS["commit_encoding"],
        )
        for i in range(len(ds)):
            assert ds.token_length(i) > 0
