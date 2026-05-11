from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_scripts = str(Path(__file__).resolve().parents[3] / "scripts")
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

from convert_tokens_to_falcon_mmap import (
    PathologyConfig,
    _falcon_pathology_reason,
    _linear_offsets,
    _text_pathology_reason,
    fast_monotone_dense_pair_alignment,
    monotone_dense_pair_alignment,
)


def test_monotone_dp_selects_expected_qwen_positions_and_odd_mask():
    qwen_offsets = [(0, 1), (1, 2), (2, 3)]
    falcon_ids = [10, 11, 12]
    falcon_offsets = [(0, 1), (1, 2), (2, 3)]

    out = monotone_dense_pair_alignment(
        qwen_offsets,
        falcon_ids,
        falcon_offsets,
        falcon_pad_id=0,
    )

    assert not out.skipped
    assert out.forced_survivor_indices.tolist() == [0, 2]
    assert out.target_falcon_pair_ids.tolist() == [[10, 11], [12, 0]]
    assert out.target_pair_loss_mask.tolist() == [[True, True], [True, False]]
    assert np.all(out.alignment_scores >= 1.0)


def test_monotone_dp_skips_when_falcon_pairs_exceed_qwen_tokens():
    out = monotone_dense_pair_alignment(
        qwen_offsets=[(0, 1)],
        falcon_ids=[10, 11, 12],
        falcon_offsets=[(0, 1), (1, 2), (2, 3)],
        falcon_pad_id=0,
    )

    assert out.skipped
    assert out.skip_reason == "ceil_falcon_pairs_gt_qwen_tokens"
    assert out.forced_survivor_indices.shape == (0,)
    assert out.target_falcon_pair_ids.shape == (0, 2)


def test_fast_alignment_uses_linear_offsets_with_expected_shapes():
    qwen_offsets = _linear_offsets(100, 10)
    falcon_offsets = _linear_offsets(100, 8)

    out = fast_monotone_dense_pair_alignment(
        qwen_offsets=qwen_offsets,
        falcon_ids=list(range(8)),
        falcon_offsets=falcon_offsets,
        falcon_pad_id=0,
    )

    assert not out.skipped
    assert out.forced_survivor_indices.shape == (4,)
    assert np.all(np.diff(out.forced_survivor_indices) > 0)
    assert out.target_falcon_pair_ids.shape == (4, 2)
    assert out.target_pair_loss_mask.shape == (4, 2)
    assert out.alignment_scores.shape == (4,)


def test_text_pathology_filter_catches_oversized_and_long_line_chunks():
    assert (
        _text_pathology_reason(
            "abcdef",
            PathologyConfig(max_decoded_chars=5),
        )
        == "decoded_chars_gt_limit"
    )
    assert (
        _text_pathology_reason(
            "a" * 9,
            PathologyConfig(max_line_chars=8),
        )
        == "line_chars_gt_limit"
    )


def test_text_pathology_filter_catches_binaryish_and_base64ish_chunks():
    assert (
        _text_pathology_reason(
            "\x00" * 10,
            PathologyConfig(min_printable_ratio=0.80),
        )
        == "printable_ratio_lt_limit"
    )
    assert (
        _text_pathology_reason(
            "QUJDREVGR0g=",
            PathologyConfig(
                min_base64ish_chars=8,
                max_base64ish_ratio=0.90,
                max_line_chars=100,
            ),
        )
        == "base64ish_ratio_gt_limit"
    )


def test_falcon_pathology_filter_catches_expansion_and_can_be_disabled():
    assert (
        _falcon_pathology_reason(
            qwen_tokens=10,
            falcon_tokens=41,
            cfg=PathologyConfig(max_falcon_expansion=4.0),
        )
        == "falcon_expansion_gt_limit"
    )
    assert (
        _falcon_pathology_reason(
            qwen_tokens=10,
            falcon_tokens=100_000,
            cfg=PathologyConfig(enabled=False),
        )
        is None
    )
