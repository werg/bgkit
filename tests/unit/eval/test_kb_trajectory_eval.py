"""Tests for the KB-scale trajectory eval metrics.

Uses a toy trainer stub (same pattern as ``tests/unit/training/
test_kr_kb_trainer_pieces.py``) wired to a fake decoder that returns
deterministic logits, so each metric can be verified on synthetic
inputs without touching real Qwen / Phase 1 checkpoints.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dataclasses import dataclass

from bgkit.data.bgkit_tool_template import TrajectoryTurn
from bgkit.data.datasets.phase2_kb_dataset import KBSample
from bgkit.eval.kb_trajectory_eval import (
    answer_token_f1,
    evaluate_sample,
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
    bgkit_text = "A/b q"
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
    bgkit_text = "A/b q"
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
    bgkit_text = "A/b q"
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
