"""Tests for the blob tokenization bridge (compaction sample -> tokens)."""

from __future__ import annotations

import random

import pytest

from bgkit.data.blob_tokenize import tokenize_blob_sample
from bgkit.data.compaction_sampler import build_sample
from tests.unit.data.test_compaction_sampler import make_trajectory


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    except Exception:
        pytest.skip("Qwen3.5 tokenizer not available locally")


def _sample(n_blobs: int, mode: str = "segments"):
    msgs = make_trajectory(n_turns=8)
    s = build_sample(
        msgs,
        trajectory_id="t1",
        target_index=len(msgs) - 1,
        live_window_turns=2,
        n_blobs=n_blobs,
        mode=mode,
    )
    assert s is not None
    return s


def test_render_loss_mask_and_sentinels(tokenizer):
    s = _sample(n_blobs=3)
    r = tokenize_blob_sample(tokenizer, s)
    assert len(r.blob_sentinel_spans) == 3
    # sentinel spans are inside the prefix and carry no loss
    for a, b in r.blob_sentinel_spans:
        assert not r.loss_mask[a:b].any()
    # loss covers the target turn only — decode masked region, check content
    target_text = tokenizer.decode(r.token_ids[r.loss_mask])
    assert "All done." in target_text
    # nothing before the last sentinel carries loss
    last_sentinel_end = r.blob_sentinel_spans[-1][1]
    assert not r.loss_mask[:last_sentinel_end].any()


def test_merged_mode_single_sentinel(tokenizer):
    s = _sample(n_blobs=4, mode="merged")
    r = tokenize_blob_sample(tokenizer, s)
    assert len(r.blob_sentinel_spans) == 1


def test_probe_sample_tokenizes(tokenizer):
    from bgkit.data.compaction_sampler import build_probe_sample

    msgs = make_trajectory(n_turns=8)
    base = build_sample(
        msgs, trajectory_id="t1", target_index=len(msgs) - 1, live_window_turns=1, n_blobs=2
    )
    probe = build_probe_sample(base, msgs, random.Random(0))
    assert probe is not None
    r = tokenize_blob_sample(tokenizer, probe)
    answer_text = tokenizer.decode(r.token_ids[r.loss_mask])
    assert probe.probe_fact[1] in answer_text
