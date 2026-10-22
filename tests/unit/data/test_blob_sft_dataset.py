"""Tests for BlobSFTDataset (Family A blob-format compaction SFT)."""

from __future__ import annotations

import json

import pytest

from tests.unit.data.test_compaction_sampler import make_trajectory


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    except Exception:
        pytest.skip("Qwen3.5 tokenizer not available locally")


@pytest.fixture()
def shard(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for i in range(6):
        rows.append(
            {
                "trajectory_id": f"t{i}",
                "repo": f"owner/repo{i % 2}",
                "trajectory": json.dumps(make_trajectory(n_turns=8 + i)),
            }
        )
    path = tmp_path / "shard.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_dataset_yields_wellformed_samples(tokenizer, shard):
    from bgkit.data.datasets.blob_sft_dataset import BlobSFTDataset, spliced_length

    ds = BlobSFTDataset([shard], tokenizer, draws_per_trajectory=2, min_blob_content_tokens=16)
    assert len(ds) == 12
    got = 0
    for i in range(len(ds)):
        s = ds[i]
        if s is None:
            continue
        got += 1
        assert s["token_ids"].shape == s["loss_mask"].shape
        assert s["loss_mask"].any()
        assert len(s["sentinel_spans"]) == len(s["blob_content_ids"]) == s["meta"]["n_blobs"]
        for (a, b), content in zip(s["sentinel_spans"], s["blob_content_ids"]):
            assert 0 < a < b <= s["token_ids"].shape[0]
            assert content.shape[0] >= 16
        L = spliced_length(s, l0_ratio=0.35, l1_ratio=0.5)
        assert L > 0
    assert got >= 6  # most draws should produce a sample


def test_recall_probes_appear_across_draws(tokenizer, shard):
    """The probe variant is APPENDED after the base sample by
    sample_trajectory; taking [0] unconditionally dropped every probe
    (2026-08-25: blob_sft_v1's step-250 eval was 128/128 continuation).
    The dataset must surface probes for a meaningful fraction of draws."""
    from bgkit.data.datasets.blob_sft_dataset import BlobSFTDataset

    ds = BlobSFTDataset(
        [shard], tokenizer, draws_per_trajectory=8, min_blob_content_tokens=16,
    )
    qtypes = {
        s["meta"]["qtype"] for s in (ds[i] for i in range(len(ds))) if s is not None
    }
    assert "recall_probe" in qtypes
    assert "continuation" in qtypes


def test_repo_holdout_split_is_repo_disjoint_and_complete(tokenizer, shard):
    import zlib

    from bgkit.data.datasets.blob_sft_dataset import BlobSFTDataset

    kw = dict(draws_per_trajectory=1, min_blob_content_tokens=16)
    full = BlobSFTDataset([shard], tokenizer, **kw)
    train = BlobSFTDataset(
        [shard], tokenizer, **kw, repo_holdout_fraction=0.5, split="train",
    )
    ev = BlobSFTDataset(
        [shard], tokenizer, **kw, repo_holdout_fraction=0.5, split="eval",
    )
    assert len(train) + len(ev) == len(full)

    def repos(ds):
        return {
            str(ds._row(fi, rg, r).get("repo")) for fi, rg, r in ds._index
        }

    train_repos, eval_repos = repos(train), repos(ev)
    assert not (train_repos & eval_repos)
    # Membership follows the deterministic crc32 rule exactly.
    for repo in eval_repos:
        assert (zlib.crc32(repo.encode()) % 1000) < 500
    for repo in train_repos:
        assert (zlib.crc32(repo.encode()) % 1000) >= 500


def test_eval_split_without_holdout_raises(tokenizer, shard):
    from bgkit.data.datasets.blob_sft_dataset import BlobSFTDataset

    with pytest.raises(ValueError, match="repo_holdout_fraction"):
        BlobSFTDataset([shard], tokenizer, split="eval")


def test_deterministic_per_index(tokenizer, shard):
    from bgkit.data.datasets.blob_sft_dataset import BlobSFTDataset

    a = BlobSFTDataset([shard], tokenizer, draws_per_trajectory=2, min_blob_content_tokens=16)
    b = BlobSFTDataset([shard], tokenizer, draws_per_trajectory=2, min_blob_content_tokens=16)
    for i in (0, 3, 7):
        sa, sb = a[i], b[i]
        if sa is None or sb is None:
            assert sa is None and sb is None
            continue
        assert (sa["token_ids"] == sb["token_ids"]).all()
        assert sa["sentinel_spans"] == sb["sentinel_spans"]
