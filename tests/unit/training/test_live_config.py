"""Tests for LiveConfig."""

import json
import time

from bgkit.training.live_config import LiveConfig


def test_missing_file(tmp_path):
    lc = LiveConfig(tmp_path / "nonexistent.json")
    assert lc.poll() == {}


def test_none_path():
    lc = LiveConfig(None)
    assert lc.poll() == {}


def test_initial_read(tmp_path):
    f = tmp_path / "control.json"
    f.write_text(json.dumps({"lr": 1e-4}))
    lc = LiveConfig(f)
    changes = lc.poll()
    assert changes == {"lr": 1e-4}


def test_unchanged_returns_empty(tmp_path):
    f = tmp_path / "control.json"
    f.write_text(json.dumps({"lr": 1e-4}))
    lc = LiveConfig(f)
    lc.poll()
    assert lc.poll() == {}


def test_changed_value(tmp_path):
    f = tmp_path / "control.json"
    f.write_text(json.dumps({"lr": 1e-4}))
    lc = LiveConfig(f)
    lc.poll()

    # Need to ensure mtime differs
    time.sleep(0.05)
    f.write_text(json.dumps({"lr": 5e-5}))
    changes = lc.poll()
    assert changes == {"lr": 5e-5}


def test_new_key(tmp_path):
    f = tmp_path / "control.json"
    f.write_text(json.dumps({"lr": 1e-4}))
    lc = LiveConfig(f)
    lc.poll()

    time.sleep(0.05)
    f.write_text(json.dumps({"lr": 1e-4, "w_repro": 0.5}))
    changes = lc.poll()
    assert changes == {"w_repro": 0.5}


def test_bad_json(tmp_path):
    f = tmp_path / "control.json"
    f.write_text("not json {{{")
    lc = LiveConfig(f)
    assert lc.poll() == {}


def test_file_deleted_after_read(tmp_path):
    f = tmp_path / "control.json"
    f.write_text(json.dumps({"lr": 1e-4}))
    lc = LiveConfig(f)
    lc.poll()

    f.unlink()
    assert lc.poll() == {}


def test_not_dict(tmp_path):
    f = tmp_path / "control.json"
    f.write_text(json.dumps([1, 2, 3]))
    lc = LiveConfig(f)
    assert lc.poll() == {}


def test_bad_json_then_fixed_same_mtime(tmp_path):
    """After a parse error, correcting the file without changing mtime is re-read."""
    import os

    f = tmp_path / "control.json"
    f.write_text("not json {{{")
    lc = LiveConfig(f)
    assert lc.poll() == {}

    # Fix the file content but preserve the same mtime
    mtime = f.stat().st_mtime
    f.write_text(json.dumps({"lr": 1e-4}))
    os.utime(f, (mtime, mtime))

    # Should still pick up the corrected content because _last_mtime
    # was not updated on the failed parse
    changes = lc.poll()
    assert changes == {"lr": 1e-4}
