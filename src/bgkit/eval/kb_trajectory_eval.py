"""KB-scale trajectory evaluation metrics.

Pure-function metrics that take a configured :class:`bgkit.training.phase2.
kr_kb_trainer.KRKBTrainer` instance and a :class:`bgkit.data.datasets.
phase2_kb_dataset.KBSample` and report what we actually care about during
Phase 2 KB-scale evaluation:

1. :func:`trajectory_step_accuracy` — strictly per-token greedy-argmax
   match over every loss-bearing position in the trajectory.
2. :func:`tool_call_id_accuracy` — per bgkit call, does the greedy
   decode of the assistant tool-call tokens contain the teacher's
   ``ids`` argument as a substring of the decoded text?
3. :func:`answer_token_f1` — token-level F1 between the decoded
   predicted answer and ``sample.gold_answer``.

All functions run under :func:`torch.no_grad` context: the caller is
responsible for wrapping if they need to disable autograd globally, but
each function also wraps its own forward pass for safety.

Design notes
------------
The trainer already owns the machinery to render a trajectory into
interleaved decoder segments (:meth:`KRKBTrainer._build_decoder_segments_
with_trace`) and the trace it returns carries concat-coordinate spans
for the answer turn and every tool-call emission. That's everything we
need — we just run one decoder forward, greedy argmax the logits, and
slice / compare against the teacher tensor.

Pure-function / no state: these helpers do not mutate the trainer or
wandb, they simply return floats / dicts. Aggregation across a dataset
is the caller's responsibility (see :mod:`scripts.eval_phase2_kb`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from bgkit.eval.metrics.qa_metrics import token_f1

if TYPE_CHECKING:
    from bgkit.data.datasets.phase2_kb_dataset import KBSample
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer


__all__ = [
    "EMPTY_RESULT",
    "answer_token_f1",
    "evaluate_sample",
    "tool_call_id_accuracy",
    "trajectory_step_accuracy",
]


# Returned for samples with no loss-bearing positions so callers can
# aggregate without sprinkling ``None`` checks.
EMPTY_RESULT: dict = {
    "trajectory_step_accuracy": 0.0,
    "trajectory_total_tokens": 0,
    "trajectory_correct_tokens": 0,
    "tool_call_id_accuracy": {
        "bgkit": 0.0,
        "overall": 0.0,
        "n_bgkit": 0,
    },
    "answer_token_f1": 0.0,
    "pred_answer": "",
    "gold_answer": "",
}


# ---------------------------------------------------------------------------
# Core forward helper
# ---------------------------------------------------------------------------


def _run_forward(
    trainer: KRKBTrainer, sample: KBSample,
) -> tuple[object, object] | None:
    """Build segments + trace, run the decoder forward with hidden states.

    Returns ``(output, trace)`` or ``None`` if the sample has zero
    loss-bearing positions (shouldn't normally happen for a real
    trajectory, but we guard against it for toy inputs).
    """
    segments, trace = trainer._build_decoder_segments_with_trace(sample)
    output = trainer.decoder.forward_interleaved_with_loss(
        segments, return_hidden_states=True,
    )
    return output, trace


def _greedy_shift_view(output) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(preds, shift_targets, shift_mask)`` under next-token shift.

    ``preds[i]`` is the model's greedy pick at concat position ``i+1``
    given hidden state at ``i``. ``shift_targets[i]`` is the gold token
    at position ``i+1``. ``shift_mask[i]`` is the loss-bearing flag at
    position ``i+1``. All tensors are ``(B, S-1)``.
    """
    token_ids_full = output.token_ids
    loss_mask_full = output.loss_mask
    shift_t = token_ids_full[:, 1:]
    shift_m = loss_mask_full[:, 1:]
    preds = output.argmax_predictions()
    return preds, shift_t, shift_m


# ---------------------------------------------------------------------------
# 1. Trajectory step accuracy
# ---------------------------------------------------------------------------


@torch.no_grad()
def trajectory_step_accuracy(
    trainer: KRKBTrainer, sample: KBSample,
) -> float:
    """Fraction of loss-bearing positions where greedy argmax matches gold.

    Counts every position in the trajectory whose ``loss_mask`` is True
    after the shift. Returns 0.0 for samples with no loss-bearing tokens.
    """
    fwd = _run_forward(trainer, sample)
    if fwd is None:
        return 0.0
    output, _trace = fwd
    preds, shift_t, shift_m = _greedy_shift_view(output)
    total = int(shift_m.sum().item())
    if total == 0:
        return 0.0
    correct = int(((preds == shift_t) & shift_m).sum().item())
    return correct / total


# ---------------------------------------------------------------------------
# 2. Tool-call ID accuracy
# ---------------------------------------------------------------------------


def _tool_call_gold_ids(turn) -> list[str]:
    """Normalize a trajectory turn's ``id`` / ``ids`` argument into a list."""
    args = getattr(turn, "args", {}) or {}
    if turn.kind == "bgkit":
        raw = args.get("ids", [])
        if isinstance(raw, (list, tuple)):
            return [str(x) for x in raw]
        return [str(raw)] if raw else []
    return []


def _score_call_span(
    preds: torch.Tensor,
    shift_m: torch.Tensor,
    concat_span: tuple[int, int],
    gold_ids: list[str],
    tokenizer,
) -> float:
    """Decode the greedy prediction for a single tool call span and score.

    ``concat_span`` is the concat-coordinate ``[start, end)`` of the
    assistant tool-call tokens (loss-bearing in the teacher mask). We
    translate to shifted coordinates (``i -> i-1``), decode the predicted
    ids, and return ``1.0`` iff every gold tag/article ID in
    ``gold_ids`` appears as a substring of the decoded text. Empty gold
    lists score 0.0 (treated as a no-op turn that can't be correct).

    Rationale: tool-call JSON is serialized many ways by different
    templates (e.g. ``"ids": ["Physics/sub1"]`` vs ``<tool>ids=
    Physics/sub1</tool>``). Substring matching is robust across formats
    and catches the failure mode we actually care about: the decoder
    emits a different tag ID than the teacher's. Exact-string match
    against the raw JSON would be brittle against whitespace and quote
    differences.
    """
    if not gold_ids:
        return 0.0
    c_start, c_end = concat_span
    shift_start = max(0, c_start - 1)
    shift_end = max(0, c_end - 1)
    if shift_end <= shift_start:
        return 0.0
    pred_slice = preds[0, shift_start:shift_end]
    mask_slice = shift_m[0, shift_start:shift_end]
    if mask_slice.any():
        pred_slice = pred_slice[mask_slice]
    if pred_slice.numel() == 0:
        return 0.0
    pred_text = tokenizer.decode(
        pred_slice.cpu().tolist(), skip_special_tokens=True,
    )
    return float(all(gid in pred_text for gid in gold_ids))


@torch.no_grad()
def tool_call_id_accuracy(
    trainer: KRKBTrainer, sample: KBSample,
) -> dict[str, float]:
    """Per bgkit call, score whether greedy decode matches gold IDs.

    Returns a dict with keys:

    - ``bgkit``: mean bgkit-call accuracy (1.0 if all tool-call tokens
      under greedy decode contain every gold ID in ``ids``, else 0.0),
      averaged across bgkit calls. ``0.0`` when no bgkit calls.
    - ``overall``: micro-averaged across all tool calls. Equal to
      ``bgkit`` now that browse is gone. ``0.0`` when no tool calls.
    - ``n_bgkit``: count used for the average, so the caller can
      aggregate across a dataset with the correct weighting.
    """
    fwd = _run_forward(trainer, sample)
    if fwd is None:
        return {"bgkit": 0.0, "overall": 0.0, "n_bgkit": 0}
    output, trace = fwd
    preds, _shift_t, shift_m = _greedy_shift_view(output)
    tokenizer = trainer.tokenizer

    bgkit_scores: list[float] = []
    for turn, span in zip(
        trace.bgkit_turns, trace.bgkit_call_spans, strict=True,
    ):
        gold_ids = _tool_call_gold_ids(turn)
        bgkit_scores.append(
            _score_call_span(preds, shift_m, span, gold_ids, tokenizer)
        )

    n_bgkit = len(bgkit_scores)
    bgkit_mean = sum(bgkit_scores) / n_bgkit if n_bgkit else 0.0
    return {
        "bgkit": bgkit_mean,
        "overall": bgkit_mean,
        "n_bgkit": n_bgkit,
    }


# ---------------------------------------------------------------------------
# 3. Answer token F1
# ---------------------------------------------------------------------------


def _decode_answer_prediction(
    output, trace, tokenizer,
) -> tuple[str, str]:
    """Return ``(pred_text, gold_text)`` for the answer turn.

    Uses the concat-coordinate answer span from the trace, translates to
    shifted coordinates, and decodes greedy prediction vs. teacher
    tokens. Both outputs are empty strings when the sample has no
    answer turn.
    """
    if trace.answer_span is None:
        return "", ""
    preds, shift_t, shift_m = _greedy_shift_view(output)
    a_start, a_end = trace.answer_span
    shift_a_start = max(0, a_start - 1)
    shift_a_end = max(0, a_end - 1)
    if shift_a_end <= shift_a_start:
        return "", ""
    pred_ids = preds[0, shift_a_start:shift_a_end]
    gold_ids = shift_t[0, shift_a_start:shift_a_end]
    mask_ans = shift_m[0, shift_a_start:shift_a_end]
    if mask_ans.any():
        pred_ids = pred_ids[mask_ans]
        gold_ids = gold_ids[mask_ans]
    if pred_ids.numel() == 0:
        return "", ""
    pred_text = tokenizer.decode(pred_ids.cpu().tolist(), skip_special_tokens=True)
    gold_text = tokenizer.decode(gold_ids.cpu().tolist(), skip_special_tokens=True)
    return pred_text, gold_text


@torch.no_grad()
def answer_token_f1(
    trainer: KRKBTrainer, sample: KBSample,
) -> float:
    """Token-level F1 between greedy-decoded answer and ``sample.gold_answer``.

    Uses the trainer-resolved answer span to slice the predicted tokens
    out of the full trajectory decode. Compares the decoded string to
    ``sample.gold_answer`` via :func:`bgkit.eval.metrics.qa_metrics.
    token_f1`.

    Returns 0.0 for:

    - Samples with no answer turn (trace.answer_span is None).
    - Empty predicted answers (greedy decoder emitted zero tokens in
      the answer range after mask filtering) — the underlying
      ``token_f1`` already returns 0.0 for empty predictions but we
      short-circuit here to keep the semantics explicit.
    """
    fwd = _run_forward(trainer, sample)
    if fwd is None:
        return 0.0
    output, trace = fwd
    pred_text, _gold_decoded = _decode_answer_prediction(
        output, trace, trainer.tokenizer,
    )
    if not pred_text:
        return 0.0
    gold_answer = str(getattr(sample, "gold_answer", "") or "")
    if not gold_answer:
        return 0.0
    return token_f1(pred_text, [gold_answer])


# ---------------------------------------------------------------------------
# One-shot bundle — runs a single forward pass and computes all metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_sample(
    trainer: KRKBTrainer, sample: KBSample,
) -> dict:
    """Run one decoder forward and compute all KB trajectory metrics.

    Equivalent to calling :func:`trajectory_step_accuracy`,
    :func:`tool_call_id_accuracy`, and :func:`answer_token_f1`
    individually, but amortizes the segment build + decoder forward
    across the three metrics — which matters for real trajectories
    where each forward can be seconds long.

    Returns a dict with the shape of :data:`EMPTY_RESULT`, plus the
    decoded predicted and gold answer strings under ``pred_answer`` /
    ``gold_answer`` (useful for per-sample error inspection).
    """
    fwd = _run_forward(trainer, sample)
    if fwd is None:
        return dict(EMPTY_RESULT)
    output, trace = fwd
    preds, shift_t, shift_m = _greedy_shift_view(output)
    tokenizer = trainer.tokenizer

    total_tokens = int(shift_m.sum().item())
    if total_tokens == 0:
        return dict(EMPTY_RESULT)
    correct_tokens = int(((preds == shift_t) & shift_m).sum().item())
    step_acc = correct_tokens / total_tokens

    bgkit_scores: list[float] = []
    for turn, span in zip(
        trace.bgkit_turns, trace.bgkit_call_spans, strict=True,
    ):
        gold_ids = _tool_call_gold_ids(turn)
        bgkit_scores.append(
            _score_call_span(preds, shift_m, span, gold_ids, tokenizer)
        )
    n_bgkit = len(bgkit_scores)
    bgkit_mean = sum(bgkit_scores) / n_bgkit if n_bgkit else 0.0
    tool_call_metrics = {
        "bgkit": bgkit_mean,
        "overall": bgkit_mean,
        "n_bgkit": n_bgkit,
    }

    pred_text, _gold_decoded = _decode_answer_prediction(
        output, trace, tokenizer,
    )
    gold_answer = str(getattr(sample, "gold_answer", "") or "")
    f1 = token_f1(pred_text, [gold_answer]) if pred_text and gold_answer else 0.0

    return {
        "trajectory_step_accuracy": step_acc,
        "trajectory_total_tokens": total_tokens,
        "trajectory_correct_tokens": correct_tokens,
        "tool_call_id_accuracy": tool_call_metrics,
        "answer_token_f1": f1,
        "pred_answer": pred_text,
        "gold_answer": gold_answer,
    }
