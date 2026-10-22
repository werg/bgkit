"""Both on-disk checkpoint layouts must be readable by the same consumers.

Phase-2 saves one flat ``model.pt``; Phase-1 saves ``encoder.pt`` +
``decoder_qwen.pt`` + ``decoder_falcon.pt`` with unprefixed keys. Every tool
that wrote ``state["model"]`` therefore raised KeyError on Phase-1 checkpoints.

On 2026-08-28 that turned every lineage comparison against the Phase-1 base
into a ``SKIPPED (KeyError)`` row, so the measurement that distinguishes
"Phase-2 training broke the projection" from "it was never aligned" silently
never ran. The adapter is shared rather than per-script precisely because the
assumption had already spread to three scripts and two modules.
"""

from __future__ import annotations

import pytest
import torch

from bgkit.training.checkpointing import normalize_model_state


def _phase1_state() -> dict:
    return {
        "encoder": {"l0.backbone.w": torch.zeros(2), "projection_blocks.qwen35.b": torch.ones(2)},
        "decoder_qwen": {"backbone.lm_head.weight": torch.zeros(3)},
        "decoder_falcon": {"backbone.lm_head.weight": torch.ones(3)},
        "optimizer_state_by_name": {"irrelevant": 1},
    }


def test_phase1_split_layout_becomes_the_flat_layout() -> None:
    out = normalize_model_state(_phase1_state())
    m = out["model"]
    assert "encoder.l0.backbone.w" in m
    assert "encoder.projection_blocks.qwen35.b" in m
    assert "decoders.qwen35.backbone.lm_head.weight" in m
    assert "decoders.falcon_h1.backbone.lm_head.weight" in m


def test_phase2_flat_layout_passes_through_unchanged() -> None:
    """Idempotence matters: consumers call this unconditionally."""
    state = {"model": {"encoder.x": torch.zeros(1)}, "optimizer_state_by_name": {}}
    assert normalize_model_state(state) is state


def test_tensors_are_not_copied() -> None:
    """A checkpoint is multi-GB; the adapter must re-key, not duplicate."""
    st = _phase1_state()
    out = normalize_model_state(st)
    assert out["model"]["encoder.l0.backbone.w"] is st["encoder"]["l0.backbone.w"]


def test_sibling_substates_are_preserved() -> None:
    """Callers still read optimizer state off the same dict."""
    out = normalize_model_state(_phase1_state())
    assert out["optimizer_state_by_name"] == {"irrelevant": 1}


def test_unknown_layout_raises_with_the_keys_it_saw() -> None:
    """Fail loudly and name the keys. The original failure mode was a bare
    KeyError swallowed into 'SKIPPED (KeyError)', which said nothing about
    what the checkpoint actually contained."""
    with pytest.raises(KeyError) as exc:
        normalize_model_state({"something_else": {}})
    assert "something_else" in str(exc.value)
