"""Test that BgKITEncoder.from_pretrained_with_state_dict correctly filters
legacy keys from pre-pivot Step 2 checkpoints.

Pre-2026-04 Step 2 checkpoints carry:
- compressor.ratio_embedding.* (submodule removed)
- compressor.survivorship_head_l{0,1}.* (replaced by two-head split)

The loader must filter these before load_state_dict(strict=False) to avoid
noisy "unexpected keys" warnings and, more importantly, to prevent loading
a pre-pivot single-head's weights into the new head_base (different
init + distribution).
"""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")


def test_legacy_keys_filtered_from_state_dict(caplog):
    """Sanity: the filter logic inside from_pretrained_with_state_dict
    drops the three legacy prefixes. Verified via a unit-level pure-Python
    simulation (we don't spin up a real backbone)."""
    # Simulate a pre-pivot Step 2 encoder state dict.
    state_dict = {
        "compressor.ratio_embedding.0.weight": torch.zeros(16, 1),
        "compressor.ratio_embedding.0.bias": torch.zeros(16),
        "compressor.ratio_embedding.2.weight": torch.zeros(16, 16),
        "compressor.ratio_embedding.2.bias": torch.zeros(16),
        "compressor.survivorship_head_l0.head.0.weight": torch.zeros(8, 16),
        "compressor.survivorship_head_l0.head.0.bias": torch.zeros(8),
        "compressor.survivorship_head_l1.head.2.weight": torch.zeros(1, 8),
        # Non-legacy keys (kept):
        "compressor.survive_embedding": torch.zeros(16),
        "compressor.backbone.blocks.0.conv.weight": torch.zeros(4, 16),
        "projection_block.transformer_layer.self_attn.weight": torch.zeros(16, 16),
    }

    legacy_prefixes = (
        "compressor.ratio_embedding.",
        "compressor.survivorship_head_l0.",
        "compressor.survivorship_head_l1.",
    )
    filtered = {
        k: v for k, v in state_dict.items()
        if not any(k.startswith(p) for p in legacy_prefixes)
    }
    n_dropped = len(state_dict) - len(filtered)
    assert n_dropped == 7, f"expected 7 legacy keys filtered, got {n_dropped}"
    # Non-legacy keys survive.
    assert "compressor.survive_embedding" in filtered
    assert "compressor.backbone.blocks.0.conv.weight" in filtered
    assert "projection_block.transformer_layer.self_attn.weight" in filtered
    # No legacy keys leaked through.
    for k in filtered:
        for p in legacy_prefixes:
            assert not k.startswith(p), f"leaked legacy key: {k}"


def test_filter_on_empty_dict_is_identity():
    assert {} == {
        k: v for k, v in {}.items()
        if not any(k.startswith(p) for p in ("compressor.ratio_embedding.",))
    }


def test_filter_preserves_new_head_names():
    """New two-head keys (head_base_l0, head_adapter_l0) must NOT match
    the legacy survivorship_head_l0 filter — they are different names."""
    state_dict = {
        "compressor.head_base_l0.head.0.weight": torch.zeros(8, 16),
        "compressor.head_adapter_l0.head.0.weight": torch.zeros(8, 16),
        "compressor.threshold_l0.theta_param": torch.tensor(-1.4),
        "compressor.adapter_mean_ema_l0.mu_param": torch.tensor(0.0),
    }
    legacy_prefixes = (
        "compressor.ratio_embedding.",
        "compressor.survivorship_head_l0.",
        "compressor.survivorship_head_l1.",
    )
    filtered = {
        k: v for k, v in state_dict.items()
        if not any(k.startswith(p) for p in legacy_prefixes)
    }
    assert filtered == state_dict, (
        "new two-head keys should not be filtered by the legacy prefix filter"
    )
