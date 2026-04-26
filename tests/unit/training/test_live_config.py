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


# --- Tests for the shared sampling-window-above live handler ---


class _SamplerStubTrainer(BaseTrainer):
    """Trainer with a single ratio sampler config attribute."""

    LIVE_CONFIG_HANDLERS = {
        "target_ratio_sampling_window_above": "_handle_ratio_sampling_window_above",
    }

    def __init__(self):
        from bgkit.training.ratio_sampling import build_ratio_sampler_config

        self._target_ratio_sampler_cfg = build_ratio_sampler_config(
            {"enabled": True, "mode": "window", "window_above": 0.10},
            anchor_grid=(0.10, 0.50),
            default_ratio=0.10,
            enabled_default=False,
            mode_default="window",
        )

    def setup(self):
        pass

    def _forward_backward(self, batch):
        return {}

    def evaluate(self):
        return {}

    def trainable_parameters(self):
        return []


class _MultiSamplerStubTrainer(_SamplerStubTrainer):
    """Trainer that exposes two sampler configs (e.g. L0 + L1)."""

    RATIO_SAMPLER_CFG_ATTRS = ("_l0_ratio_sampler_cfg", "_l1_ratio_sampler_cfg")

    def __init__(self):
        from bgkit.training.ratio_sampling import build_ratio_sampler_config

        self._l0_ratio_sampler_cfg = build_ratio_sampler_config(
            {"enabled": True, "mode": "window", "window_above": 0.10},
            anchor_grid=(0.10,),
            default_ratio=0.10,
            enabled_default=False,
            mode_default="window",
        )
        self._l1_ratio_sampler_cfg = build_ratio_sampler_config(
            {"enabled": True, "mode": "window", "window_above": 0.20},
            anchor_grid=(0.15,),
            default_ratio=0.15,
            enabled_default=False,
            mode_default="window",
        )


def test_window_above_live_update_replaces_frozen_cfg():
    t = _SamplerStubTrainer()
    old_cfg = t._target_ratio_sampler_cfg
    t.apply_live_config({"target_ratio_sampling_window_above": 0.05})
    assert t._target_ratio_sampler_cfg.window_above == 0.05
    # Other fields preserved
    assert t._target_ratio_sampler_cfg.enabled is old_cfg.enabled
    assert t._target_ratio_sampler_cfg.anchor_grid == old_cfg.anchor_grid


def test_window_above_live_update_rejects_negative():
    t = _SamplerStubTrainer()
    t.apply_live_config({"target_ratio_sampling_window_above": -0.1})
    assert t._target_ratio_sampler_cfg.window_above == 0.10  # unchanged


def test_window_above_live_update_rebuilds_all_configured_attrs():
    t = _MultiSamplerStubTrainer()
    t.apply_live_config({"target_ratio_sampling_window_above": 0.07})
    assert t._l0_ratio_sampler_cfg.window_above == 0.07
    assert t._l1_ratio_sampler_cfg.window_above == 0.07


# --- Tests for the shared sampling-enabled live handler ---


class _SamplingEnabledStub(_SamplerStubTrainer):
    LIVE_CONFIG_HANDLERS = {
        "sample_target_ratio_during_training": "_handle_ratio_sampling_enabled",
    }


def test_sampling_enabled_live_update_flips_enabled():
    t = _SamplingEnabledStub()
    assert t._target_ratio_sampler_cfg.enabled is True
    t.apply_live_config({"sample_target_ratio_during_training": False})
    assert t._target_ratio_sampler_cfg.enabled is False
    # Other fields preserved.
    assert t._target_ratio_sampler_cfg.window_above == 0.10
    t.apply_live_config({"sample_target_ratio_during_training": True})
    assert t._target_ratio_sampler_cfg.enabled is True


def test_sampling_enabled_live_update_rejects_non_bool():
    t = _SamplingEnabledStub()
    t.apply_live_config({"sample_target_ratio_during_training": "yes"})
    assert t._target_ratio_sampler_cfg.enabled is True  # unchanged


# --- Tests for the shared anchor-sampling-prob live handler ---


class _AnchorProbStub(_SamplerStubTrainer):
    LIVE_CONFIG_HANDLERS = {
        "target_ratio_anchor_sampling_prob": "_handle_ratio_sampling_anchor_prob",
    }


def test_anchor_prob_live_update_changes_field():
    t = _AnchorProbStub()
    assert t._target_ratio_sampler_cfg.anchor_sampling_prob == 0.30
    t.apply_live_config({"target_ratio_anchor_sampling_prob": 0.5})
    assert t._target_ratio_sampler_cfg.anchor_sampling_prob == 0.5
    t.apply_live_config({"target_ratio_anchor_sampling_prob": 0.0})
    assert t._target_ratio_sampler_cfg.anchor_sampling_prob == 0.0


def test_anchor_prob_live_update_rejects_out_of_range():
    t = _AnchorProbStub()
    t.apply_live_config({"target_ratio_anchor_sampling_prob": 1.5})
    assert t._target_ratio_sampler_cfg.anchor_sampling_prob == 0.30  # unchanged
    t.apply_live_config({"target_ratio_anchor_sampling_prob": -0.1})
    assert t._target_ratio_sampler_cfg.anchor_sampling_prob == 0.30  # unchanged
    t.apply_live_config({"target_ratio_anchor_sampling_prob": "half"})
    assert t._target_ratio_sampler_cfg.anchor_sampling_prob == 0.30  # unchanged
