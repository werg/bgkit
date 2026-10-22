"""Tests for L0Cache manifest provenance + mismatch detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bgkit.data.l0_cache import (
    L0CacheManifestMismatch,
    assert_cache_manifest_matches,
    checkpoint_fingerprint,
    file_sha256,
    read_cache_manifest,
    write_cache_manifest,
)


def _ckpt(path: Path, content: bytes = b"phase1-weights") -> Path:
    path.write_bytes(content)
    return path


def test_write_and_read_manifest_roundtrip(tmp_path: Path):
    phase1 = _ckpt(tmp_path / "phase1.pt", b"phase1")
    out = write_cache_manifest(
        tmp_path / "cache",
        "kilt_wikipedia",
        phase1_checkpoint=phase1,
        stage_a_checkpoint=None,
        lora_rank=32,
        lora_alpha=64.0,
        retention=0.05,
        extra={"shard_count": 3},
    )
    assert out.exists()
    manifest = read_cache_manifest(tmp_path / "cache", "kilt_wikipedia")
    assert manifest is not None
    assert manifest["lora_rank"] == 32
    assert manifest["retention"] == 0.05
    assert manifest["phase1_sha"] == file_sha256(phase1)
    assert manifest["stage_a_sha"] is None
    assert manifest["shard_count"] == 3
    # Top-level history is empty on first write.
    assert manifest["manifest_history"] == []


def test_manifest_history_grows_on_rewrite(tmp_path: Path):
    phase1 = _ckpt(tmp_path / "phase1.pt", b"v1")
    write_cache_manifest(
        tmp_path / "cache",
        "kilt",
        phase1_checkpoint=phase1,
        stage_a_checkpoint=None,
        lora_rank=32, lora_alpha=64.0, retention=0.05,
    )
    phase1.write_bytes(b"v2")
    write_cache_manifest(
        tmp_path / "cache",
        "kilt",
        phase1_checkpoint=phase1,
        stage_a_checkpoint=None,
        lora_rank=32, lora_alpha=64.0, retention=0.05,
    )
    manifest = read_cache_manifest(tmp_path / "cache", "kilt")
    assert len(manifest["manifest_history"]) == 1
    # The current sha reflects v2; the historical record carries v1.
    assert manifest["phase1_sha"] == file_sha256(phase1)


def test_checkpoint_directory_fingerprint_is_recorded_and_validated(tmp_path: Path):
    phase1 = tmp_path / "phase1"
    stage_a = tmp_path / "stage_a"
    phase1.mkdir()
    stage_a.mkdir()
    (phase1 / "metadata.json").write_text('{"phase":"phase1"}')
    (phase1 / "encoder.pt").write_bytes(b"base")
    (stage_a / "metadata.json").write_text('{"phase":"phase2_kb"}')
    (stage_a / "model.pt").write_bytes(b"stage-a")

    write_cache_manifest(
        tmp_path / "cache",
        "ds",
        phase1_checkpoint=phase1,
        stage_a_checkpoint=stage_a,
        lora_rank=0,
        lora_alpha=None,
        retention=0.1,
    )
    manifest = read_cache_manifest(tmp_path / "cache", "ds")
    assert manifest["phase1_sha"] == checkpoint_fingerprint(phase1)
    assert manifest["stage_a_sha"] == checkpoint_fingerprint(stage_a)
    assert_cache_manifest_matches(
        tmp_path / "cache",
        "ds",
        phase1_checkpoint=phase1,
        stage_a_checkpoint=stage_a,
        lora_rank=0,
        retention=0.1,
    )


def test_assert_passes_on_match(tmp_path: Path):
    phase1 = _ckpt(tmp_path / "p1.pt", b"phase1-weights")
    write_cache_manifest(
        tmp_path / "cache",
        "ds",
        phase1_checkpoint=phase1,
        stage_a_checkpoint=None,
        lora_rank=16, lora_alpha=32.0, retention=0.10,
    )
    # Should not raise.
    assert_cache_manifest_matches(
        tmp_path / "cache", "ds",
        phase1_checkpoint=phase1,
        stage_a_checkpoint=None,
        lora_rank=16, lora_alpha=32.0, retention=0.10,
    )


def test_assert_raises_on_phase1_sha_mismatch(tmp_path: Path):
    phase1_a = _ckpt(tmp_path / "p1a.pt", b"weights-A")
    phase1_b = _ckpt(tmp_path / "p1b.pt", b"weights-B")
    write_cache_manifest(
        tmp_path / "cache",
        "ds",
        phase1_checkpoint=phase1_a,
        stage_a_checkpoint=None,
        lora_rank=32, lora_alpha=64.0, retention=0.10,
    )
    with pytest.raises(L0CacheManifestMismatch, match="phase1_sha"):
        assert_cache_manifest_matches(
            tmp_path / "cache", "ds",
            phase1_checkpoint=phase1_b,
            stage_a_checkpoint=None,
            lora_rank=32, lora_alpha=64.0, retention=0.10,
        )


def test_assert_raises_when_trainer_expects_stage_a_but_cache_lacks_it(tmp_path: Path):
    phase1 = _ckpt(tmp_path / "p1.pt", b"weights")
    stage_a = _ckpt(tmp_path / "stage_a.pt", b"stage-a-weights")
    write_cache_manifest(
        tmp_path / "cache",
        "ds",
        phase1_checkpoint=phase1,
        stage_a_checkpoint=None,  # bootstrap pre-compute, no Stage A LoRA
        lora_rank=32, lora_alpha=64.0, retention=0.10,
    )
    with pytest.raises(L0CacheManifestMismatch, match="stage_a"):
        assert_cache_manifest_matches(
            tmp_path / "cache", "ds",
            phase1_checkpoint=phase1,
            stage_a_checkpoint=stage_a,
            lora_rank=32, lora_alpha=64.0, retention=0.10,
        )


def test_assert_raises_on_lora_rank_mismatch(tmp_path: Path):
    phase1 = _ckpt(tmp_path / "p1.pt", b"weights")
    write_cache_manifest(
        tmp_path / "cache",
        "ds",
        phase1_checkpoint=phase1,
        stage_a_checkpoint=None,
        lora_rank=16, lora_alpha=32.0, retention=0.10,
    )
    with pytest.raises(L0CacheManifestMismatch, match="lora_rank"):
        assert_cache_manifest_matches(
            tmp_path / "cache", "ds",
            phase1_checkpoint=phase1,
            stage_a_checkpoint=None,
            lora_rank=32, lora_alpha=64.0, retention=0.10,
        )


def test_assert_raises_on_retention_mismatch(tmp_path: Path):
    phase1 = _ckpt(tmp_path / "p1.pt", b"weights")
    write_cache_manifest(
        tmp_path / "cache",
        "ds",
        phase1_checkpoint=phase1,
        stage_a_checkpoint=None,
        lora_rank=32, lora_alpha=64.0, retention=0.05,
    )
    with pytest.raises(L0CacheManifestMismatch, match="retention"):
        assert_cache_manifest_matches(
            tmp_path / "cache", "ds",
            phase1_checkpoint=phase1,
            stage_a_checkpoint=None,
            lora_rank=32, lora_alpha=64.0, retention=0.20,
        )


def test_assert_silent_when_no_manifest(tmp_path: Path):
    """Missing manifest is permitted (legacy caches)."""
    phase1 = _ckpt(tmp_path / "p1.pt", b"weights")
    # Should not raise — caller decides whether to warn.
    assert_cache_manifest_matches(
        tmp_path / "cache", "absent_dataset",
        phase1_checkpoint=phase1,
        stage_a_checkpoint=None,
        lora_rank=32, lora_alpha=64.0, retention=0.10,
    )
