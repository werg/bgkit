"""Tests for log-needle QA generation (Family B wide-net tool results)."""

from __future__ import annotations

import random

from bgkit.data.lognav_qa import (
    generate_from_file,
    generate_window_samples,
    iter_windows,
    line_severity_is_error,
    materialize_window,
    rare_id_tokens,
)

LOG = """\
081109 203615 148 INFO dfs.DataNode: PacketResponder 1 for block blk_38865049064139660 terminating
081109 203807 222 INFO dfs.DataNode: PacketResponder 0 for block blk_69528719 terminating
081109 204005 35 ERROR dfs.FSNamesystem: BLOCK* addStoredBlock failed for blk_7128370237687728475
081109 204106 329 INFO dfs.DataNode: Received block blk_38865049064139660 again
081109 204132 26 INFO dfs.NameSystem: allocate blk_904791815409399662 to /user/rbo/out
"""


def test_severity_detection():
    lines = LOG.splitlines()
    assert not line_severity_is_error(lines[0])
    assert line_severity_is_error(lines[2])


def test_rare_id_tokens_unique_only():
    lines = LOG.splitlines()
    toks = rare_id_tokens(lines)
    # blk_38865049064139660 appears twice -> excluded; unique ids retained.
    assert "blk_38865049064139660" not in toks
    assert "blk_7128370237687728475" in toks


def test_iter_windows_covers_all_lines():
    lines = LOG.splitlines()
    spans = iter_windows(lines, window_chars=200)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(lines)


def test_window_samples_first_error_and_needle():
    lines = LOG.splitlines()
    samples = generate_window_samples(
        lines, (0, len(lines)), dataset="hdfs", path="mem.log", rng=random.Random(0)
    )
    by_type = {s.qtype: s for s in samples}
    assert "first_error" in by_type
    assert "addStoredBlock failed" in by_type["first_error"].answer
    needle = [s for s in samples if s.qtype == "needle_token"]
    assert needle, "expected at least one needle sample"
    for s in needle:
        tok = s.question.split()[-1].rstrip(".")
        assert tok in s.answer
    assert all('query="' in s.blob_header for s in samples)


def test_error_absent_negative():
    lines = [ln for ln in LOG.splitlines() if "ERROR" not in ln]
    samples = generate_window_samples(
        lines, (0, len(lines)), dataset="hdfs", path="mem.log", rng=random.Random(0)
    )
    assert samples[0].qtype == "error_absent"


def test_generate_from_file_and_materialize(tmp_path):
    f = tmp_path / "toy.log"
    f.write_text(LOG)
    rows = generate_from_file(f, dataset="hdfs", window_chars=10_000)
    assert rows and rows[0]["path"] == str(f)
    window = materialize_window(rows[0])
    assert window.splitlines()[0].startswith("081109 203615")


def test_iter_error_windows_contain_errors():
    from bgkit.data.lognav_qa import iter_error_windows

    lines = ["081109 INFO ok"] * 50 + ["081109 ERROR boom"] + ["081109 INFO ok"] * 50
    spans = iter_error_windows(lines, window_chars=400, max_windows=3, rng=random.Random(0))
    assert spans
    for a, b in spans:
        assert any("ERROR" in ln for ln in lines[a:b])
