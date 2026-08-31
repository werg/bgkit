"""Auto-resume must not silently continue under a changed model config.

A graceful-shutdown save writes a checkpoint under the RUN's name, and
auto-resume is scoped by run name -- so relaunching that run with a changed
model config resumes weights trained under the old one. On 2026-08-31 a
step-74 emergency save from an ``interface_affine: true`` launch was sitting
in the fast dir when the same run was relaunched with the affine pinned; it
was caught by hand, minutes before it would have made an isolated-variable
arm not isolated.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from bgkit.training.checkpointing import (
    CheckpointMetadata,
    fingerprint_mismatch,
    load_checkpoint_metadata,
    model_fingerprint,
)


class _Cfg(dict):
    """A dict that also answers ``.get`` on nested access, like OmegaConf."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _cfg(**encoder):
    return _Cfg(
        model=_Cfg(encoder=_Cfg(**encoder)),
        training=_Cfg(selection_mode=_Cfg(l0="exact_topk", l1="exact_topk")),
    )


def test_fingerprint_captures_the_keys_that_change_what_a_weight_means():
    fp = model_fingerprint(_cfg(interface_norm=True, interface_affine=False))
    assert fp["model.encoder.interface_norm"] is True
    assert fp["model.encoder.interface_affine"] is False
    assert fp["training.selection_mode.l0"] == "exact_topk"


def test_absent_keys_are_recorded_so_adding_one_reads_as_a_change():
    """"The key did not exist before" is exactly the silent-continuation case
    being guarded against, so it must not read as a match."""
    before = model_fingerprint(_cfg())
    after = model_fingerprint(_cfg(interface_norm=True))
    assert before["model.encoder.interface_norm"] is None
    assert fingerprint_mismatch(before, after) == {
        "model.encoder.interface_norm": (None, True),
    }


def test_the_exact_case_that_nearly_slipped_through():
    saved = model_fingerprint(_cfg(interface_norm=True, interface_affine=True))
    now = model_fingerprint(_cfg(interface_norm=True, interface_affine=False))
    drift = fingerprint_mismatch(saved, now)
    assert drift == {"model.encoder.interface_affine": (True, False)}


def test_an_unchanged_config_is_not_a_mismatch():
    fp = model_fingerprint(_cfg(interface_norm=True, interface_affine=False))
    assert fingerprint_mismatch(fp, dict(fp)) == {}


def test_a_checkpoint_without_a_fingerprint_cannot_disagree():
    """Checkpoints predating the field must still auto-resume; refusing them
    would break every older run for no evidence of a change."""
    assert fingerprint_mismatch(None, model_fingerprint(_cfg())) == {}
    assert fingerprint_mismatch({}, model_fingerprint(_cfg())) == {}


def test_the_fingerprint_is_short_on_purpose():
    """A fingerprint including schedule or logging knobs would refuse to
    resume its own run after any edit, which trains people to bypass it."""
    keys = set(model_fingerprint(_cfg()))
    assert len(keys) <= 8
    assert not any("lr" in k or "eval" in k or "steps" in k for k in keys)


def test_metadata_round_trips_through_the_checkpoint_file(tmp_path):
    meta = CheckpointMetadata(
        phase="phase2_kb", step=74, epoch=0, parent_checkpoint=None,
        run_name="phase2_kb_widenet_v9_interface",
        model_fingerprint=model_fingerprint(
            _cfg(interface_norm=True, interface_affine=True),
        ),
    )
    (tmp_path / "metadata.json").write_text(json.dumps(asdict(meta)))
    loaded = load_checkpoint_metadata(tmp_path)
    assert loaded.model_fingerprint == meta.model_fingerprint
    assert loaded.run_name == meta.run_name


def test_metadata_only_load_does_not_need_the_tensors(tmp_path):
    """The check runs before deciding to load at all, and the state dicts are
    ~15 GB on spinning storage."""
    (tmp_path / "metadata.json").write_text(
        json.dumps({"phase": "p", "step": 1, "epoch": 0, "parent_checkpoint": None}),
    )
    assert load_checkpoint_metadata(tmp_path).step == 1
    assert not list(tmp_path.glob("*.pt"))


def test_unknown_metadata_keys_are_tolerated(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({
            "phase": "p", "step": 1, "epoch": 0, "parent_checkpoint": None,
            "a_field_from_a_newer_version": 1,
        }),
    )
    assert load_checkpoint_metadata(tmp_path).phase == "p"


@pytest.mark.parametrize("value", [True, False, 0.1, "exact_topk", None])
def test_fingerprint_values_survive_json(value, tmp_path):
    fp = model_fingerprint(_cfg(interface_affine=value))
    assert json.loads(json.dumps(fp)) == fp
