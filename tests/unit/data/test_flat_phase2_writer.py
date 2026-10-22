"""Flat Phase-2 writer: the trajectory-side half of the leaf-drill contract."""

from __future__ import annotations

import json

from bgkit.data.flat_phase2_writer import TRAJ_COLUMNS, flat_trajectory_row


def test_flat_row_emits_plain_leaf_drill():
    """The bgkit turn must be the PLAIN leaf form ``{ids, query}``: any
    git-repro drill tag (``is_head`` above all) reroutes the turn away from the
    live L0→L1 path — ``is_head`` sent every widenet v1→v4 splice to the
    shared-tree head drill, which has no infrastructure for flat datasets and
    returned a ZERO survivor (2026-08-22 diagnosis)."""
    row = flat_trajectory_row(
        dataset_name="lognav",
        doc_id="log:x:y:0-10",
        question="Is there an error line?",
        answer="No.",
        scope_description="a log window",
        split="train",
        group_id="x",
        gold_span_json="[3, 7]",
    )
    assert set(row) == set(TRAJ_COLUMNS)
    traj = json.loads(row["trajectory_json"])
    assert [t["kind"] for t in traj] == ["bgkit", "answer"]
    bgkit = traj[0]
    assert bgkit["args"] == {"ids": ["log:x:y:0-10"], "query": "Is there an error line?"}
    assert "is_head" not in bgkit["args"]
    assert bgkit["loss"] is True and traj[1]["response"] == "No."
    assert row["gold_span_json"] == "[3, 7]"
    # The prompt says "start with the entrypoint ID named in the knowledge-base
    # description" — so the description must name the call's id.
    from bgkit.eval.kb_trajectory_eval import scope_entrypoint_ids

    assert row["scope_description"] == "a log window; entrypoint id: `log:x:y:0-10`"
    assert scope_entrypoint_ids(row["scope_description"]) == bgkit["args"]["ids"]
    spaced = flat_trajectory_row(
        dataset_name="fileneedle", doc_id="file:o/r:Google Chrome/x.js@8cba9029",
        question="q", answer="a", scope_description="source file Google Chrome/x.js",
        split="train", group_id="o/r",
    )
    assert scope_entrypoint_ids(spaced["scope_description"]) == [
        "file:o/r:Google Chrome/x.js@8cba9029"
    ]
    import pytest

    with pytest.raises(ValueError):
        flat_trajectory_row(
            dataset_name="x", doc_id="bad`id", question="q", answer="a",
            scope_description="s", split="train", group_id="g",
        )


def test_query_override_separates_selection_query_from_user_turn():
    """L0 selection is query-conditioned, so prompt scaffolding in the user
    turn must be keepable OUT of the retrieval query — BABILong's few-shot
    examples name every candidate answer label and would bias survivors."""
    row = flat_trajectory_row(
        dataset_name="babilong_qa1_16k",
        doc_id="babilong:qa1:16k:0000",
        question="<example>...hallway...kitchen...</example>\n\nWhere is John?",
        answer="hallway",
        scope_description="a long story transcript",
        split="eval",
        group_id="babilong:qa1:16k:0000",
        query="Where is John?",
    )
    traj = json.loads(row["trajectory_json"])
    assert traj[0]["args"]["query"] == "Where is John?"
    assert "hallway" in row["question"]  # user turn keeps the full prompt
