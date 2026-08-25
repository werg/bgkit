"""Tests for the KB-scale trajectory eval metrics.

Uses a toy trainer stub (same pattern as ``tests/unit/training/
test_kr_kb_trainer_pieces.py``) wired to a fake decoder that returns
deterministic logits, so each metric can be verified on synthetic
inputs without touching real Qwen / Phase 1 checkpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.bgkit_tool_template import TrajectoryTurn
from bgkit.data.browse_tree import BrowseNode, BrowseTree
from bgkit.data.datasets.phase2_kb_dataset import KBSample
from bgkit.eval.kb_trajectory_eval import (
    answer_token_f1,
    evaluate_free_running_sample,
    evaluate_sample,
    parse_bgkit_call,
    parse_bgkit_call_ids,
    tool_call_id_accuracy,
    trajectory_step_accuracy,
)
from bgkit.training.phase2.kr_kb_trainer import _KBDecodeTrace

# ---------------------------------------------------------------------------
# Toy fixtures
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer stub.

    Each character maps 1:1 to a token id (``ord(c)``), so ``decode``
    just reverses the mapping. That's enough to exercise the substring
    match in ``tool_call_id_accuracy`` and the F1 path in
    ``answer_token_f1`` without depending on a real vocab.
    """

    pad_token_id = 0
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(chr(int(i)) for i in ids if int(i) > 0)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


@dataclass
class _FakeOutput:
    """Stand-in for ``InterleavedForwardOutput``.

    The test writes the loss mask and token IDs explicitly, and returns
    a canned ``argmax_predictions`` tensor — this lets each test drive
    the decoder predictions directly without forwarding through a real
    model.
    """

    token_ids: torch.Tensor
    loss_mask: torch.Tensor
    loss: torch.Tensor
    _argmax: torch.Tensor

    def argmax_predictions(self) -> torch.Tensor:
        return self._argmax


class _FakeDecoder:
    hidden_dim = 4

    def __init__(self, output: _FakeOutput) -> None:
        self._output = output

    def forward_interleaved_with_loss(
        self, segments, *, return_hidden_states: bool = False,
    ):
        return self._output


class _ToyTrainer:
    """Trainer stub exposing exactly the surface our metrics need.

    - ``tokenizer``: for decoding predicted ids
    - ``decoder``: stub with ``forward_interleaved_with_loss``
    - ``_build_decoder_segments_with_trace(sample)``: returns ``([],
      trace)`` where ``trace`` is a ``_KBDecodeTrace`` with concat-
      coordinate spans the test crafts by hand
    """

    def __init__(
        self,
        *,
        tokenizer: _FakeTokenizer,
        output: _FakeOutput,
        trace: _KBDecodeTrace,
    ) -> None:
        self.tokenizer = tokenizer
        self.decoder = _FakeDecoder(output)
        self._trace = trace

    def _build_decoder_segments_with_trace(self, _sample):
        # Segments aren't touched by metrics — they only read from the
        # decoder's return value — so an empty list is fine.
        return [], self._trace


def _build_sample(
    *,
    trajectory: list[TrajectoryTurn],
    gold_answer: str = "forty two",
) -> KBSample:
    return KBSample(
        dataset_name="toy",
        scope_template="topic_list",
        scope_description="",
        topic_list=["Physics"],
        question="q?",
        gold_answer=gold_answer,
        trajectory=trajectory,
    )


def _str_to_ids(text: str) -> list[int]:
    return [ord(c) for c in text]


def _call_text(*ids: str) -> str:
    return json.dumps({"ids": list(ids), "query": "q"}, separators=(",", ":"))


def test_tool_call_parser_is_structured_and_exact():
    assert parse_bgkit_call_ids('<tool_call>' + _call_text("opaque-id") + '</tool_call>') == [
        "opaque-id"
    ]
    assert parse_bgkit_call_ids('{"name":"bgkit","arguments":"{\\"ids\\":[\\"x\\"]}"}') == ["x"]
    assert parse_bgkit_call_ids("the id is opaque-id") is None
    assert parse_bgkit_call_ids('{"ids":["opaque-id"') is None
    assert parse_bgkit_call_ids('{"answer":{"ids":["opaque-id"]}}') is None
    assert parse_bgkit_call_ids('{"ids":["opaque-id"],"offline_depth":3}') is None
    assert parse_bgkit_call_ids(
        '<tool_call>' + _call_text("opaque-id") + "</tool_call> trailing",
    ) is None


class _FreeRunningTrainer:
    def __init__(self, outputs: list[str], tree: BrowseTree):
        self.outputs = iter(outputs)
        self._trees = {"toy": tree}
        self.histories: list[list[TrajectoryTurn]] = []

    def generate_kb_turn(self, _sample, history, *, max_new_tokens):
        assert max_new_tokens > 0
        self.histories.append(list(history))
        return next(self.outputs)


def _free_running_fixture(outputs: list[str]):
    tree = BrowseTree.from_nodes("toy", [
        BrowseNode(
            id="root", parent=None, kind="sub-tag", size=1,
            children=("head",), articles=(),
        ),
        BrowseNode(
            id="head", parent="root", kind="sub-tag", size=1,
            children=("commit",), articles=(),
        ),
        BrowseNode(
            id="commit", parent="head", kind="sub-tag", size=1,
            children=(), articles=("evidence",),
        ),
    ])
    trajectory = [
        TrajectoryTurn(
            kind="bgkit",
            args={"ids": ["head"], "query": "q", "is_head": True},
        ),
        TrajectoryTurn(kind="bgkit", args={"ids": ["commit"], "query": ""}),
        TrajectoryTurn(
            kind="bgkit",
            args={"ids": ["evidence"], "query": "q", "is_retrieval": True},
        ),
        TrajectoryTurn(kind="answer", response="gold state"),
    ]
    sample = _build_sample(trajectory=trajectory, gold_answer="gold state")
    sample.group_id = "head"
    return _FreeRunningTrainer(outputs, tree), sample


_XML_CALL = (
    "<tool_call>\n<function=bgkit>\n<parameter=ids>\n[\"head\"]\n</parameter>\n"
    "<parameter=query>\nwhere?\n</parameter>\n</function>\n</tool_call>"
)


def test_tool_call_parser_accepts_qwen_xml_rendering():
    """Qwen3.5's template renders tool calls as XML — the exact text the
    decoder is trained on. 2026-08-23: the parser knew only JSON, so every
    well-formed Qwen call scored ``malformed_tool_call``."""
    assert parse_bgkit_call(_XML_CALL) == {"ids": ["head"], "query": "where?"}
    # ids only, no query
    assert parse_bgkit_call_ids(
        '<tool_call>\n<function=bgkit>\n<parameter=ids>\n["a","b"]\n</parameter>\n'
        "</function>\n</tool_call>"
    ) == ["a", "b"]
    # wrong function, extra parameter, stray text, bad ids JSON: all fail closed
    assert parse_bgkit_call(_XML_CALL.replace("function=bgkit", "function=other")) is None
    assert parse_bgkit_call(
        _XML_CALL.replace("</function>", "<parameter=depth>\n3\n</parameter>\n</function>")
    ) is None
    assert parse_bgkit_call(
        _XML_CALL.replace("</parameter>\n</function>", "</parameter>\nhi\n</function>")
    ) is None
    assert parse_bgkit_call(_XML_CALL.replace('["head"]', "head")) is None
    assert parse_bgkit_call(_XML_CALL + " trailing prose") is None


def test_free_running_eval_accepts_qwen_xml_calls():
    trainer, sample = _free_running_fixture([
        _XML_CALL,
        _XML_CALL.replace('["head"]', '["commit"]'),
        _XML_CALL.replace('["head"]', '["evidence"]'),
        "gold state",
    ])
    result = evaluate_free_running_sample(trainer, sample)
    assert result["route_exact"] == 1.0
    assert result["answer_exact_match"] == 1.0
    assert result["raw_text"] == "gold state"


def test_scope_entrypoint_ids():
    from bgkit.eval.kb_trajectory_eval import scope_entrypoint_ids

    assert scope_entrypoint_ids("source file a.py; entrypoint id: file:o/r:a.py@1234abcd") == [
        "file:o/r:a.py@1234abcd"
    ]
    # Quoted ids survive spaces and commas in file paths (16 fileneedle rows).
    assert scope_entrypoint_ids(
        "source file Google Chrome/x.js; entrypoint id: `file:o/r:Google Chrome/x.js@8cba9029`"
    ) == ["file:o/r:Google Chrome/x.js@8cba9029"]
    assert scope_entrypoint_ids("repo; entrypoint ids: `a, b`, `c`") == ["a, b", "c"]
    assert scope_entrypoint_ids("repository; entrypoint ids: n1, n2") == ["n1", "n2"]
    assert scope_entrypoint_ids("a log window") == []


def test_free_running_eval_accepts_ids_named_in_scope():
    """Flat single-document rows: group_id is the repo label (split grouping),
    the document id is named in the scope description — that is what the
    prompt exposes, so it is available from the first turn. The flat browse
    tree indexes the document itself as an ARTICLE node (2026-08-23: that made
    `identifier in tree` classify the call as a head drill and trip the
    trainer's flat-dataset guard)."""
    trainer, sample = _free_running_fixture([_call_text("evidence"), "gold state"])
    trainer._trees["toy"] = BrowseTree.from_nodes("toy", [
        BrowseNode(id="root", parent=None, kind="sub-tag", size=1,
                   children=(), articles=("evidence",)),
        BrowseNode(id="evidence", parent="root", kind="article", size=1,
                   children=(), articles=()),
    ])
    sample.group_id = "owner/repo"
    sample.scope_description = "source file a.py; entrypoint id: `evidence`"
    sample.trajectory = [
        TrajectoryTurn(kind="bgkit", args={"ids": ["evidence"], "query": "q"}),
        TrajectoryTurn(kind="answer", response="gold state"),
    ]
    result = evaluate_free_running_sample(trainer, sample)
    assert result["invalid_reason"] == ""
    assert result["route_exact"] == 1.0
    assert result["answer_exact_match"] == 1.0
    # The recorded retrieval call is the PLAIN leaf form the trainer routes to
    # the live L0→L1 path — no head/structural tags (those raise on flat data).
    assert trainer.histories[-1][0].args == {"ids": ["evidence"], "query": "q"}


def test_free_running_eval_tags_tree_node_calls_as_head_drills():
    trainer, sample = _free_running_fixture([
        _call_text("head"), _call_text("commit"), _call_text("evidence"), "gold state",
    ])
    evaluate_free_running_sample(trainer, sample)
    head, node, leaf = trainer.histories[-1]
    assert head.args["is_head"] is True and head.args["is_retrieval"] is False
    assert node.args["is_head"] is False and node.args["structural_depth"] == 2
    assert leaf.args == {"ids": ["evidence"], "query": "q"}


def test_free_running_eval_executes_only_surfaced_ids():
    trainer, sample = _free_running_fixture([
        _call_text("head"),
        _call_text("commit"),
        _call_text("evidence"),
        "gold state",
    ])
    result = evaluate_free_running_sample(trainer, sample)
    assert result["route_exact"] == 1.0
    assert result["valid_navigation"] == 1.0
    assert result["evidence_recall"] == 1.0
    assert result["answer_token_f1"] == 1.0
    assert result["answer_exact_match"] == 1.0
    assert [len(history) for history in trainer.histories] == [0, 1, 2, 3]


def test_free_running_eval_rejects_unsurfaced_id():
    trainer, sample = _free_running_fixture([_call_text("evidence")])
    result = evaluate_free_running_sample(trainer, sample)
    assert result["valid_navigation"] == 0.0
    assert result["route_exact"] == 0.0
    assert result["invalid_reason"] == "unsurfaced_id:evidence"


def test_free_running_eval_rejects_malformed_explicit_tool_call():
    trainer, sample = _free_running_fixture([
        '<tool_call>{"ids":["head"]}',
    ])
    result = evaluate_free_running_sample(trainer, sample)
    assert result["invalid_reason"] == "malformed_tool_call"
    assert result["valid_navigation"] == 0.0
    assert result["pred_answer"] == ""


def test_free_running_eval_can_exactly_reconstruct_an_empty_file():
    trainer, sample = _free_running_fixture([
        _call_text("head"),
        _call_text("commit"),
        _call_text("evidence"),
        "",
    ])
    sample.gold_answer = ""
    result = evaluate_free_running_sample(trainer, sample)
    assert result["route_exact"] == 1.0
    assert result["valid_navigation"] == 1.0
    assert result["answer_exact_match"] == 1.0


# ---------------------------------------------------------------------------
# trajectory_step_accuracy
# ---------------------------------------------------------------------------


def test_trajectory_step_accuracy_all_correct():
    """Greedy argmax matches teacher exactly → 1.0."""
    # Concat layout (length 6): [b, g, b, g, b, g] where g = gold, b = non-gold
    # loss-bearing. Shifted view is length 5; positions 0..4 predict 1..5.
    token_ids = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long)
    loss_mask = torch.tensor([[False, True, True, True, True, True]])
    # argmax_predictions is shape (1, 5) — positions that predict token[i+1]
    argmax = torch.tensor([[20, 30, 40, 50, 60]], dtype=torch.long)

    output = _FakeOutput(
        token_ids=token_ids, loss_mask=loss_mask,
        loss=torch.tensor(0.0), _argmax=argmax,
    )
    trace = _KBDecodeTrace(
        answer_span=(5, 6),
        bgkit_turns=[], bgkit_call_spans=[],
    )
    trainer = _ToyTrainer(
        tokenizer=_FakeTokenizer(), output=output, trace=trace,
    )
    sample = _build_sample(trajectory=[
        TrajectoryTurn(kind="answer", response="x", loss=True),
    ])

    assert trajectory_step_accuracy(trainer, sample) == pytest.approx(1.0)


def test_trajectory_step_accuracy_partial():
    """Half correct → 0.5 (even when some positions are mask=False)."""
    token_ids = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long)
    # Loss at positions 1, 2, 3, 4 — that's 4 shifted positions (0..3
    # predict 1..4).
    loss_mask = torch.tensor([[False, True, True, True, True, False]])
    # predicts: wrong, right, wrong, right, _
    argmax = torch.tensor([[99, 30, 99, 50, 99]], dtype=torch.long)

    output = _FakeOutput(
        token_ids=token_ids, loss_mask=loss_mask,
        loss=torch.tensor(0.0), _argmax=argmax,
    )
    trace = _KBDecodeTrace(
        answer_span=None,
        bgkit_turns=[], bgkit_call_spans=[],
    )
    trainer = _ToyTrainer(
        tokenizer=_FakeTokenizer(), output=output, trace=trace,
    )
    sample = _build_sample(trajectory=[])
    # 2 correct out of 4 loss-bearing shifted positions
    assert trajectory_step_accuracy(trainer, sample) == pytest.approx(0.5)


def test_trajectory_step_accuracy_no_loss_positions():
    """Zero loss-bearing positions → 0.0 (doesn't crash)."""
    token_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
    loss_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    argmax = torch.tensor([[20, 30]], dtype=torch.long)

    output = _FakeOutput(
        token_ids=token_ids, loss_mask=loss_mask,
        loss=torch.tensor(0.0), _argmax=argmax,
    )
    trace = _KBDecodeTrace(
        answer_span=None,
        bgkit_turns=[], bgkit_call_spans=[],
    )
    trainer = _ToyTrainer(
        tokenizer=_FakeTokenizer(), output=output, trace=trace,
    )
    sample = _build_sample(trajectory=[])
    assert trajectory_step_accuracy(trainer, sample) == 0.0


# ---------------------------------------------------------------------------
# tool_call_id_accuracy
# ---------------------------------------------------------------------------


def test_tool_call_id_accuracy_all_correct():
    """Teacher bgkit(ids=['A/b']) — decoded greedy predictions contain
    the ID as a substring → 1.0 on every sub-metric."""
    tokenizer = _FakeTokenizer()
    # Concat layout:
    #   [header, <tool resp>, <bgkit call = "A/b q">]
    bgkit_text = _call_text("A/b")
    header = "H"
    resp = "R" * 3

    full_text = header + resp + bgkit_text
    full_ids = _str_to_ids(full_text)

    # Concat len: len(full_text). loss_mask is True only on the
    # bgkit_call span.
    n = len(full_ids)
    loss_mask = [False] * n
    bgkit_start = len(header) + len(resp)
    bgkit_end = bgkit_start + len(bgkit_text)
    for i in range(bgkit_start, bgkit_end):
        loss_mask[i] = True

    # argmax (shifted length n-1): the model predicts position i+1 from
    # hidden state i. We want the decoded tool-call tokens to perfectly
    # match the teacher so substring check passes.
    argmax_ids = full_ids[1:]  # predicts perfect

    token_ids = torch.tensor([full_ids], dtype=torch.long)
    loss_mask_t = torch.tensor([loss_mask], dtype=torch.bool)
    argmax = torch.tensor([argmax_ids], dtype=torch.long)

    output = _FakeOutput(
        token_ids=token_ids, loss_mask=loss_mask_t,
        loss=torch.tensor(0.0), _argmax=argmax,
    )
    trace = _KBDecodeTrace(
        answer_span=None,
        bgkit_turns=[
            TrajectoryTurn(
                kind="bgkit", args={"ids": ["A/b"], "query": "q"}, response="",
            ),
        ],
        bgkit_call_spans=[(bgkit_start, bgkit_end)],
    )
    trainer = _ToyTrainer(tokenizer=tokenizer, output=output, trace=trace)
    sample = _build_sample(trajectory=[])

    result = tool_call_id_accuracy(trainer, sample)
    assert result["bgkit"] == pytest.approx(1.0)
    assert result["overall"] == pytest.approx(1.0)
    assert result["n_bgkit"] == 1


def test_tool_call_id_accuracy_wrong_ids():
    """Greedy predicts the wrong tag — substring check fails on all calls."""
    tokenizer = _FakeTokenizer()
    # Teacher: bgkit(ids=["A/b"])
    # Model predicts "zzzzz" in that slot → no substring match.
    header = "H"
    resp = "RRR"
    bgkit_text = _call_text("A/b")
    full_text = header + resp + bgkit_text
    full_ids = _str_to_ids(full_text)

    n = len(full_ids)
    loss_mask = [False] * n
    bgkit_start = len(header) + len(resp)
    bgkit_end = bgkit_start + len(bgkit_text)
    for i in range(bgkit_start, bgkit_end):
        loss_mask[i] = True

    # Model outputs all wrong at the tool-call positions.
    argmax_ids = list(full_ids[1:])
    for i in range(bgkit_start - 1, bgkit_end - 1):
        argmax_ids[i] = ord("z")

    token_ids = torch.tensor([full_ids], dtype=torch.long)
    loss_mask_t = torch.tensor([loss_mask], dtype=torch.bool)
    argmax = torch.tensor([argmax_ids], dtype=torch.long)

    output = _FakeOutput(
        token_ids=token_ids, loss_mask=loss_mask_t,
        loss=torch.tensor(0.0), _argmax=argmax,
    )
    trace = _KBDecodeTrace(
        answer_span=None,
        bgkit_turns=[
            TrajectoryTurn(
                kind="bgkit", args={"ids": ["A/b"], "query": "q"}, response="",
            ),
        ],
        bgkit_call_spans=[(bgkit_start, bgkit_end)],
    )
    trainer = _ToyTrainer(tokenizer=tokenizer, output=output, trace=trace)
    sample = _build_sample(trajectory=[])

    result = tool_call_id_accuracy(trainer, sample)
    assert result["bgkit"] == 0.0
    assert result["overall"] == 0.0
    assert result["n_bgkit"] == 1


def test_tool_call_id_accuracy_no_calls():
    """Trace with no tool calls → all zeros (including counts)."""
    tokenizer = _FakeTokenizer()
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    loss_mask = torch.tensor([[False, True, True]])
    argmax = torch.tensor([[2, 3]], dtype=torch.long)

    output = _FakeOutput(
        token_ids=token_ids, loss_mask=loss_mask,
        loss=torch.tensor(0.0), _argmax=argmax,
    )
    trace = _KBDecodeTrace(
        answer_span=None,
        bgkit_turns=[], bgkit_call_spans=[],
    )
    trainer = _ToyTrainer(tokenizer=tokenizer, output=output, trace=trace)
    sample = _build_sample(trajectory=[])

    result = tool_call_id_accuracy(trainer, sample)
    assert result["bgkit"] == 0.0
    assert result["overall"] == 0.0
    assert result["n_bgkit"] == 0


# ---------------------------------------------------------------------------
# answer_token_f1
# ---------------------------------------------------------------------------


def test_answer_token_f1_perfect_match():
    """Greedy decode exactly reproduces the gold answer → F1 == 1.0."""
    tokenizer = _FakeTokenizer()
    gold = "forty two"
    # Concat: [H] + gold. Answer span is (1, 1 + len(gold)).
    full_text = "H" + gold
    full_ids = _str_to_ids(full_text)
    loss_mask = [False] + [True] * len(gold)
    # Model predicts gold perfectly
    argmax_ids = full_ids[1:]

    output = _FakeOutput(
        token_ids=torch.tensor([full_ids], dtype=torch.long),
        loss_mask=torch.tensor([loss_mask], dtype=torch.bool),
        loss=torch.tensor(0.0),
        _argmax=torch.tensor([argmax_ids], dtype=torch.long),
    )
    trace = _KBDecodeTrace(
        answer_span=(1, 1 + len(gold)),
        bgkit_turns=[], bgkit_call_spans=[],
    )
    trainer = _ToyTrainer(tokenizer=tokenizer, output=output, trace=trace)
    sample = _build_sample(
        trajectory=[TrajectoryTurn(kind="answer", response=gold, loss=True)],
        gold_answer=gold,
    )

    assert answer_token_f1(trainer, sample) == pytest.approx(1.0)


def test_answer_token_f1_partial_overlap():
    """Token F1 is between 0 and 1 when prediction partially overlaps."""
    tokenizer = _FakeTokenizer()
    gold = "forty two"
    pred_text = "forty three"  # overlap: "forty", miss "two" vs "three"
    # Concat: [H] + pred.  Teacher span = (1, 1+len(pred))
    header = "H"
    full_text = header + pred_text
    full_ids = _str_to_ids(full_text)
    loss_mask = [False] + [True] * len(pred_text)
    argmax_ids = full_ids[1:]  # predicts text exactly

    output = _FakeOutput(
        token_ids=torch.tensor([full_ids], dtype=torch.long),
        loss_mask=torch.tensor([loss_mask], dtype=torch.bool),
        loss=torch.tensor(0.0),
        _argmax=torch.tensor([argmax_ids], dtype=torch.long),
    )
    trace = _KBDecodeTrace(
        answer_span=(1, 1 + len(pred_text)),
        bgkit_turns=[], bgkit_call_spans=[],
    )
    trainer = _ToyTrainer(tokenizer=tokenizer, output=output, trace=trace)
    sample = _build_sample(trajectory=[], gold_answer=gold)

    f1 = answer_token_f1(trainer, sample)
    assert 0.0 < f1 < 1.0


def test_answer_token_f1_empty_prediction():
    """No answer span → 0.0 without crashing."""
    tokenizer = _FakeTokenizer()
    output = _FakeOutput(
        token_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        loss_mask=torch.tensor([[False, True, True]]),
        loss=torch.tensor(0.0),
        _argmax=torch.tensor([[2, 3]], dtype=torch.long),
    )
    # Empty span → empty pred → 0.0
    trace = _KBDecodeTrace(
        answer_span=None,
        bgkit_turns=[], bgkit_call_spans=[],
    )
    trainer = _ToyTrainer(tokenizer=tokenizer, output=output, trace=trace)
    sample = _build_sample(trajectory=[], gold_answer="forty two")
    assert answer_token_f1(trainer, sample) == 0.0


def test_answer_token_f1_zero_length_span():
    """Answer span exists but collapses to zero shifted length → 0.0."""
    tokenizer = _FakeTokenizer()
    output = _FakeOutput(
        token_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        loss_mask=torch.tensor([[False, False, False]]),
        loss=torch.tensor(0.0),
        _argmax=torch.tensor([[2, 3]], dtype=torch.long),
    )
    trace = _KBDecodeTrace(
        answer_span=(2, 2),  # zero-length
        bgkit_turns=[], bgkit_call_spans=[],
    )
    trainer = _ToyTrainer(tokenizer=tokenizer, output=output, trace=trace)
    sample = _build_sample(trajectory=[], gold_answer="forty two")
    assert answer_token_f1(trainer, sample) == 0.0


# ---------------------------------------------------------------------------
# evaluate_sample — one-shot bundle
# ---------------------------------------------------------------------------


def test_evaluate_sample_bundles_all_metrics():
    """One forward call produces trajectory accuracy + tool-call + F1."""
    tokenizer = _FakeTokenizer()
    # Layout: H | A/b q (bgkit tool call) | RRR (tool resp) | forty two (answer)
    header = "H"
    bgkit_text = _call_text("A/b")
    resp = "RRR"
    gold = "forty two"
    full_text = header + bgkit_text + resp + gold
    full_ids = _str_to_ids(full_text)
    n = len(full_ids)

    loss_mask = [False] * n
    bgkit_start = len(header)
    bgkit_end = bgkit_start + len(bgkit_text)
    answer_start = bgkit_end + len(resp)
    answer_end = answer_start + len(gold)
    for i in range(bgkit_start, bgkit_end):
        loss_mask[i] = True
    for i in range(answer_start, answer_end):
        loss_mask[i] = True

    # Perfect prediction
    argmax_ids = full_ids[1:]

    output = _FakeOutput(
        token_ids=torch.tensor([full_ids], dtype=torch.long),
        loss_mask=torch.tensor([loss_mask], dtype=torch.bool),
        loss=torch.tensor(0.0),
        _argmax=torch.tensor([argmax_ids], dtype=torch.long),
    )
    trace = _KBDecodeTrace(
        answer_span=(answer_start, answer_end),
        bgkit_turns=[
            TrajectoryTurn(
                kind="bgkit", args={"ids": ["A/b"], "query": "q"}, response="",
            ),
        ],
        bgkit_call_spans=[(bgkit_start, bgkit_end)],
    )
    trainer = _ToyTrainer(tokenizer=tokenizer, output=output, trace=trace)
    sample = _build_sample(trajectory=[], gold_answer=gold)

    result = evaluate_sample(trainer, sample)
    assert result["trajectory_step_accuracy"] == pytest.approx(1.0)
    assert result["trajectory_total_tokens"] == len(bgkit_text) + len(gold)
    assert result["trajectory_correct_tokens"] == len(bgkit_text) + len(gold)
    tc = result["tool_call_id_accuracy"]
    assert tc["bgkit"] == 1.0
    assert tc["overall"] == 1.0
    assert tc["n_bgkit"] == 1
    assert result["answer_token_f1"] == pytest.approx(1.0)
    assert result["pred_answer"] == gold
    assert result["gold_answer"] == gold


def test_evaluate_sample_empty_when_no_loss_tokens():
    """Zero loss positions → returns the EMPTY_RESULT shape."""
    from bgkit.eval.kb_trajectory_eval import EMPTY_RESULT

    tokenizer = _FakeTokenizer()
    output = _FakeOutput(
        token_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        loss_mask=torch.zeros(1, 3, dtype=torch.bool),
        loss=torch.tensor(0.0),
        _argmax=torch.tensor([[2, 3]], dtype=torch.long),
    )
    trace = _KBDecodeTrace(
        answer_span=None,
        bgkit_turns=[], bgkit_call_spans=[],
    )
    trainer = _ToyTrainer(tokenizer=tokenizer, output=output, trace=trace)
    sample = _build_sample(trajectory=[], gold_answer="x")

    result = evaluate_sample(trainer, sample)
    # Shapes match the empty sentinel
    assert set(result.keys()) == set(EMPTY_RESULT.keys())
    assert result["trajectory_step_accuracy"] == 0.0
    assert result["answer_token_f1"] == 0.0
    assert result["tool_call_id_accuracy"]["overall"] == 0.0


def test_tool_call_id_accuracy_excludes_head_and_distractors():
    """Only supervised, non-head drills count: the head (is_head) and
    distractors (loss=False) must be EXCLUDED, so a poisoned metric that
    scored them (head unpredictable, distractor unmasked→~0) is fixed."""
    tokenizer = _FakeTokenizer()
    header = "H"
    resp = "RR"
    # three bgkit calls, all decoded perfectly (argmax=perfect)
    head_txt = _call_text("a/hd")
    dist_txt = _call_text("a/ds")
    good_txt = _call_text("a/gd")
    full_text = header + resp + head_txt + dist_txt + good_txt
    full_ids = _str_to_ids(full_text)
    n = len(full_ids)
    loss_mask = [False] * n
    s0 = len(header) + len(resp)
    s1 = s0 + len(head_txt)
    s2 = s1 + len(dist_txt)
    s3 = s2 + len(good_txt)
    # loss-bearing on head span (is_head) + good span; NOT on distractor span
    for i in range(s0, s1):  # head — loss-bearing but is_head -> excluded
        loss_mask[i] = True
    for i in range(s2, s3):  # good on-path drill -> the ONLY one that should count
        loss_mask[i] = True
    argmax = full_ids[1:]  # perfect prediction everywhere
    output = _FakeOutput(
        token_ids=torch.tensor([full_ids], dtype=torch.long),
        loss_mask=torch.tensor([loss_mask], dtype=torch.bool),
        loss=torch.tensor(0.0), _argmax=torch.tensor([argmax], dtype=torch.long),
    )
    trace = _KBDecodeTrace(
        answer_span=None,
        bgkit_turns=[
            TrajectoryTurn(
                kind="bgkit",
                args={"ids": ["a/hd"], "is_head": True},
                response="",
                loss=True,
            ),
            TrajectoryTurn(kind="bgkit", args={"ids": ["a/ds"]}, response="", loss=False),
            TrajectoryTurn(kind="bgkit", args={"ids": ["a/gd"]}, response="", loss=True),
        ],
        bgkit_call_spans=[(s0, s1), (s1, s2), (s2, s3)],
    )
    trainer = _ToyTrainer(tokenizer=tokenizer, output=output, trace=trace)
    sample = _build_sample(trajectory=[])
    result = tool_call_id_accuracy(trainer, sample)
    # Only the on-path non-head drill is scored (n=1), and it's correct.
    assert result["n_bgkit"] == 1
    assert result["bgkit"] == pytest.approx(1.0)
