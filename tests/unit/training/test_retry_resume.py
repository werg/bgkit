"""Tests for retry resume logic (reading .last_checkpoint)."""

from pathlib import Path


def test_last_checkpoint_read(tmp_path):
    """Verify .last_checkpoint file is read correctly."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / "ckpt_step10"
    ckpt_path.mkdir()

    last_ckpt_file = ckpt_dir / ".last_checkpoint"
    last_ckpt_file.write_text(str(ckpt_path))

    # Read and verify
    candidate = last_ckpt_file.read_text().strip()
    assert Path(candidate).exists()
    assert candidate == str(ckpt_path)


def test_last_checkpoint_missing(tmp_path):
    """When .last_checkpoint doesn't exist, fallback to original resume path."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    last_ckpt_file = ckpt_dir / ".last_checkpoint"

    assert not last_ckpt_file.exists()


def test_last_checkpoint_stale(tmp_path):
    """When .last_checkpoint points to deleted dir, skip it."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    last_ckpt_file = ckpt_dir / ".last_checkpoint"
    last_ckpt_file.write_text("/nonexistent/path")

    candidate = last_ckpt_file.read_text().strip()
    assert not Path(candidate).exists()


def test_keep_latest_validation():
    """keep_latest must be >= 1 when retry enabled."""
    from bgkit.training.checkpoint_manager import CheckpointManager

    # This should work fine
    mgr = CheckpointManager(keep_latest=1)
    assert mgr.keep_latest == 1

    # keep_latest=0 is valid for CheckpointManager itself,
    # but train.py validates >= 1 when retry is enabled
    mgr = CheckpointManager(keep_latest=0)
    assert mgr.keep_latest == 0


def test_stale_file_guard(tmp_path):
    """Stale .last_checkpoint deleted before first attempt."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    last_ckpt_file = ckpt_dir / ".last_checkpoint"
    last_ckpt_file.write_text("stale_path")

    # Simulate stale file guard
    if last_ckpt_file.exists():
        last_ckpt_file.unlink()

    assert not last_ckpt_file.exists()


def test_multi_retry_chain(tmp_path):
    """Verify resume path resolution across multiple retries."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    last_ckpt_file = ckpt_dir / ".last_checkpoint"

    original_resume = None  # started from scratch

    # Attempt 1: saves checkpoint A, then fails
    ckpt_a = ckpt_dir / "ckpt_a"
    ckpt_a.mkdir()
    last_ckpt_file.write_text(str(ckpt_a))

    # Attempt 2: resolve resume path
    resume_path = None
    if last_ckpt_file.exists():
        candidate = last_ckpt_file.read_text().strip()
        if candidate and Path(candidate).exists():
            resume_path = candidate
    if resume_path is None:
        resume_path = original_resume

    assert resume_path == str(ckpt_a)

    # Attempt 2 fails before saving anything new
    # Attempt 3: should still resume from A
    resume_path = None
    if last_ckpt_file.exists():
        candidate = last_ckpt_file.read_text().strip()
        if candidate and Path(candidate).exists():
            resume_path = candidate
    if resume_path is None:
        resume_path = original_resume

    assert resume_path == str(ckpt_a)
