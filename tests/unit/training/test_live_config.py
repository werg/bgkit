"""Tests for LiveConfig and BaseTrainer.apply_live_config."""

import json
import time

from bgkit.training.base_trainer import BaseTrainer
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


# --- Tests for BaseTrainer.apply_live_config ---


class _StubTrainer(BaseTrainer):
    """Minimal concrete trainer for testing live config."""

    LIVE_CONFIG_FIELDS = {"weight_a": "w_a", "weight_b": "w_b"}

    def __init__(self):
        # Skip BaseTrainer.__init__ — we just need the attributes
        self.w_a = 1.0
        self.w_b = 0.5

    def setup(self):
        pass

    def _forward_backward(self, batch):
        return {}

    def evaluate(self):
        return {}

    def trainable_parameters(self):
        return []


class _ChildTrainer(_StubTrainer):
    """Subclass adding more fields via MRO merge."""

    LIVE_CONFIG_FIELDS = {"weight_c": "w_c"}

    def __init__(self):
        super().__init__()
        self.w_c = 0.1


def test_apply_live_config_updates_field():
    t = _StubTrainer()
    t.apply_live_config({"weight_a": 2.0})
    assert t.w_a == 2.0
    assert t.w_b == 0.5  # unchanged


def test_apply_live_config_ignores_unknown_keys():
    t = _StubTrainer()
    t.apply_live_config({"unknown_key": 42})
    assert t.w_a == 1.0


def test_apply_live_config_rejects_non_numeric():
    t = _StubTrainer()
    t.apply_live_config({"weight_a": "bad"})
    assert t.w_a == 1.0  # unchanged


def test_apply_live_config_mro_merge():
    t = _ChildTrainer()
    t.apply_live_config({"weight_a": 3.0, "weight_c": 0.9})
    assert t.w_a == 3.0
    assert t.w_c == 0.9


def test_apply_live_config_preserves_type():
    t = _StubTrainer()
    t.w_a = 1  # int
    t.apply_live_config({"weight_a": 2.5})
    assert t.w_a == 2  # int(2.5) = 2
    assert isinstance(t.w_a, int)
