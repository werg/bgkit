"""Tests for the unified BgKIT blob format (capability-packaging plan §3)."""

from __future__ import annotations

import pytest

from bgkit.data.blob_format import (
    BLOB_KINDS,
    BLOB_SPLICE_SENTINEL,
    BgkitBlob,
    blob_sentinel_positions,
    parse_blobs,
    render_header,
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    except Exception:
        pytest.skip("Qwen3.5 tokenizer not available locally")


def test_render_and_parse_roundtrip_all_kinds():
    blobs = [
        BgkitBlob(
            kind="compaction",
            header=render_header(
                "compaction", span="turns 3-17", stats="2 file edits, 4 test runs"
            ),
            source_ref="traj:42:turns:3-17",
        ),
        BgkitBlob(
            kind="tool",
            header=render_header(
                "tool", source="build.log", stats="48,210 lines", query="first failing test"
            ),
            source_ref="log:build:full",
            query="first failing test",
        ),
        BgkitBlob(
            kind="memory",
            header=render_header("memory", source="session", timestamp="2026-07-14"),
            source_ref="session:2026-07-14",
        ),
        BgkitBlob(
            kind="ambient",
            header=render_header("ambient", source="fastapi @ a1b2c3", stats="1,412 files"),
            source_ref="repo:fastapi:a1b2c3",
        ),
    ]
    text = "intro\n" + "\nmiddle\n".join(b.render() for b in blobs) + "\ntail"
    parsed = parse_blobs(text)
    assert [k for k, _ in parsed] == list(BLOB_KINDS)
    assert parsed[1][1] == 'compressed: build.log (48,210 lines) query="first failing test"'


def test_header_must_be_single_line():
    with pytest.raises(ValueError):
        BgkitBlob(kind="tool", header="two\nlines", source_ref="x").render()


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        BgkitBlob(kind="banana", header="h", source_ref="x").render()  # type: ignore[arg-type]


def test_sentinel_position_search():
    sent = [7, 8, 9]
    ids = [1, 7, 8, 9, 2, 7, 8, 9, 3]
    assert blob_sentinel_positions(ids, sent) == [1, 5]
    assert blob_sentinel_positions([7, 8], sent) == []


def test_sentinel_tokenizes_stably_inside_rendered_blob(tokenizer):
    """BPE caution (encode is not concat-distributive): the sentinel's token
    run must appear verbatim when the *full* surrounding blob text is
    tokenized — the newline framing guarantees no cross-boundary merges."""
    sentinel_ids = tokenizer.encode(BLOB_SPLICE_SENTINEL, add_special_tokens=False)
    blob = BgkitBlob(
        kind="tool",
        header=render_header(
            "tool", source="results.json", stats="912 rows", query="rows with status=failed"
        ),
        source_ref="db:rows",
        query="rows with status=failed",
    )
    text = "Some assistant text before.\n" + blob.render() + "\nAnd text after the blob."
    full_ids = tokenizer.encode(text, add_special_tokens=False)
    positions = blob_sentinel_positions(full_ids, sentinel_ids)
    assert len(positions) == 1

    # Two blobs back-to-back must yield two distinct, non-overlapping hits.
    text2 = blob.render() + "\n" + blob.render()
    ids2 = tokenizer.encode(text2, add_special_tokens=False)
    assert len(blob_sentinel_positions(ids2, sentinel_ids)) == 2


def test_blob_sentinel_distinct_from_phase2_sentinel():
    from bgkit.data.bgkit_tool_template import BGKIT_SENTINEL

    assert BLOB_SPLICE_SENTINEL != BGKIT_SENTINEL
    assert BGKIT_SENTINEL not in BgkitBlob(kind="memory", header="m", source_ref="x").render()
