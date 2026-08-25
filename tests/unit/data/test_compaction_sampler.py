"""Tests for compaction sampling over chat trajectories (Family A)."""

from __future__ import annotations

import random

from bgkit.data.blob_format import parse_blobs
from bgkit.data.compaction_sampler import (
    build_probe_sample,
    build_sample,
    detect_anchor_len,
    extract_probe_facts,
    sample_trajectory,
    turn_boundaries,
)


def make_trajectory(n_turns: int = 8) -> list[dict]:
    msgs: list[dict] = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Fix the bug in the repo."},
    ]
    for i in range(n_turns):
        msgs.append(
            {
                "role": "assistant",
                "content": f"Turn {i}: editing src/pkg/module_{i}.py now.",
                "tool_calls": [{"function": {"name": "edit", "arguments": "{\"path\": \"x\"}"}}],
            }
        )
        msgs.append(
            {
                "role": "tool",
                "content": f"def handler_{i}(x):\n    return x  # in src/pkg/module_{i}.py",
            }
        )
    msgs.append({"role": "assistant", "content": "All done."})
    return msgs


def test_anchor_and_turn_boundaries():
    msgs = make_trajectory()
    anchor = detect_anchor_len(msgs)
    assert anchor == 2
    bounds = turn_boundaries(msgs, anchor)
    assert all(msgs[b]["role"] == "assistant" for b in bounds)
    assert bounds[0] == 2


def test_build_sample_segments_mode():
    msgs = make_trajectory()
    target = len(msgs) - 1  # final assistant message
    s = build_sample(
        msgs,
        trajectory_id="t1",
        target_index=target,
        live_window_turns=2,
        n_blobs=3,
        mode="segments",
    )
    assert s is not None
    assert s.target_message["content"] == "All done."
    # anchor preserved verbatim
    assert s.prefix_messages[0]["role"] == "system"
    assert s.prefix_messages[1]["content"] == "Fix the bug in the repo."
    # one history-replacement message with 3 blob blocks
    history = s.prefix_messages[2]
    assert history["role"] == "user"
    blobs = parse_blobs(history["content"])
    assert len(blobs) == 3 and all(k == "compaction" for k, _ in blobs)
    # compacted turns are gone from the prefix; live window kept raw
    prefix_text = "\n".join(str(m.get("content")) for m in s.prefix_messages)
    assert "Turn 0" not in prefix_text
    assert f"Turn {6}" in prefix_text  # live window (last 2 turns before target)
    # blob source_refs cover the span contiguously
    spans = [b.source_ref.rsplit(":", 1)[-1] for b in s.blobs]
    starts = [int(sp.split("-")[0]) for sp in spans]
    assert starts[0] == s.compacted_span[0]


def test_build_sample_merged_mode_single_blob():
    msgs = make_trajectory()
    s = build_sample(
        msgs,
        trajectory_id="t1",
        target_index=len(msgs) - 1,
        live_window_turns=1,
        n_blobs=4,
        mode="merged",
    )
    assert s is not None and s.mode == "merged" and len(s.blobs) == 1


def test_too_short_trajectory_returns_none():
    msgs = make_trajectory(n_turns=1)
    s = build_sample(
        msgs,
        trajectory_id="t1",
        target_index=len(msgs) - 1,
        live_window_turns=2,
        n_blobs=2,
    )
    assert s is None


def test_probe_sample_targets_span_only_fact():
    msgs = make_trajectory()
    base = build_sample(
        msgs,
        trajectory_id="t1",
        target_index=len(msgs) - 1,
        live_window_turns=1,
        n_blobs=2,
    )
    assert base is not None
    probe = build_probe_sample(base, msgs, random.Random(0))
    assert probe is not None
    assert probe.qtype == "recall_probe"
    kind, fact = probe.probe_fact
    # the fact must come from the compacted span and not appear after it
    a, b = probe.compacted_span
    span_text = "\n".join(str(m.get("content")) for m in msgs[a:b])
    rest_text = "\n".join(str(m.get("content")) for m in msgs[b:])
    assert fact in span_text and fact not in rest_text
    assert probe.target_message["content"] == fact


def test_extract_probe_facts_paths_and_symbols():
    facts = extract_probe_facts(make_trajectory())
    kinds = {k for k, _ in facts}
    assert "path" in kinds and "symbol" in kinds
    assert ("path", "src/pkg/module_0.py") in facts


def test_sample_trajectory_yields_mixed_samples():
    msgs = make_trajectory(n_turns=12)
    samples = sample_trajectory(
        msgs, trajectory_id="t9", rng=random.Random(7), samples_per_trajectory=8
    )
    assert samples
    qtypes = {s.qtype for s in samples}
    assert "continuation" in qtypes
    modes = {s.mode for s in samples}
    assert modes <= {"segments", "merged"}
    for s in samples:
        assert s.prefix_messages[0]["role"] == "system"
        assert s.target_message["role"] == "assistant"
