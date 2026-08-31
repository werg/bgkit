"""KB-scale trajectory evaluation metrics.

Pure-function metrics that take a configured :class:`bgkit.training.phase2.
kr_kb_trainer.KRKBTrainer` instance and a :class:`bgkit.data.datasets.
phase2_kb_dataset.KBSample` and report what we actually care about during
Phase 2 KB-scale evaluation has two deliberately separate layers:

1. :func:`evaluate_free_running_sample` is the capability metric. It generates
   one turn at a time, validates that called IDs were actually surfaced by an
   executed node, and never renders future teacher calls or the gold answer.
2. The trajectory/token helpers are teacher-forced diagnostics for training
   health. Tool calls are parsed as exact structured JSON; substring matches do
   not count.

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

The free-running helper necessarily advances the trainer's per-sample shared
tree cache while executing calls, but does not update model weights or logging.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import torch

from bgkit.data.bgkit_tool_template import TrajectoryTurn
from bgkit.eval.metrics.qa_metrics import exact_match, token_f1

if TYPE_CHECKING:
    from bgkit.data.datasets.phase2_kb_dataset import KBSample
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer


__all__ = [
    "EMPTY_RESULT",
    "answer_token_f1",
    "evaluate_free_running_sample",
    "evaluate_sample",
    "parse_bgkit_call",
    "parse_bgkit_call_ids",
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
    "answer_exact_match": 0.0,
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
    ensure_tree = getattr(trainer, "_ensure_eval_shared_tree", None)
    if callable(ensure_tree):
        ensure_tree(sample)
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


def _is_scored_bgkit_turn(turn) -> bool:
    """Whether a call is supervised and depends on a previously-read rep."""
    return bool(
        turn.kind == "bgkit"
        and getattr(turn, "loss", True)
        and not turn.args.get("is_head")
    )


def _validated_bgkit_args(value) -> dict | None:
    """Validate the public argument object without scanning arbitrary JSON."""
    if not isinstance(value, dict) or "ids" not in value:
        return None
    if not set(value).issubset({"ids", "query"}):
        return None
    raw = value["ids"]
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        return None
    query = value.get("query", "")
    if not isinstance(query, str):
        return None
    return {"ids": list(raw), "query": query}


def _call_from_json_value(value) -> dict | None:
    """Validate the exact argument or standard function-call JSON shape."""
    direct = _validated_bgkit_args(value)
    if direct is not None:
        return direct
    if not isinstance(value, dict):
        return None

    function = value.get("function")
    if function is not None:
        if not isinstance(function, dict):
            return None
        return _call_from_json_value(function)

    if "arguments" not in value or value.get("name") != "bgkit":
        return None
    arguments = value["arguments"]
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    return _validated_bgkit_args(arguments)


_SCOPE_ENTRYPOINT_RE = re.compile(
    r"entrypoint ids?:\s*((?:`[^`]*`(?:\s*,\s*)?)+|[^\s;,]+(?:\s*,\s*[^\s;,]+)*)"
)


def scope_entrypoint_ids(scope_description: str) -> list[str]:
    """IDs a pre-scoped system prompt names as entrypoints.

    Backtick-quoted ids are taken verbatim (file paths may contain spaces or
    commas: ``"source file a.py; entrypoint id: `file:o/r:a b.py@1234abcd`"``);
    unquoted ids are whitespace/comma/semicolon-delimited tokens.
    """
    out: list[str] = []
    for match in _SCOPE_ENTRYPOINT_RE.finditer(scope_description):
        group = match.group(1)
        quoted = re.findall(r"`([^`]*)`", group)
        if quoted:
            out.extend(q for q in quoted if q)
        else:
            out.extend(part.strip() for part in group.split(",") if part.strip())
    return out


_XML_FUNCTION_RE = re.compile(
    r"\A<function=(?P<name>[A-Za-z0-9_.-]+)>\s*(?P<body>.*?)\s*</function>\Z", re.S
)
_XML_PARAMETER_RE = re.compile(
    r"<parameter=(?P<key>[A-Za-z0-9_]+)>\n?(?P<value>.*?)\n?</parameter>", re.S
)


def _call_from_xml(inner: str) -> dict | None:
    """Parse the XML function-call body Qwen3.5's chat template renders::

        <function=bgkit>
        <parameter=ids>
        ["doc-1"]
        </parameter>
        <parameter=query>
        why?
        </parameter>
        </function>

    This is the exact format the decoder is TRAINED on (the template renders
    ``tool_calls`` this way, see ``bgkit_tool_template.tokenize_trajectory``);
    Falcon-H1's template renders JSON instead, handled by
    :func:`_call_from_json_value`. Fails closed on anything else — 2026-08-23:
    this parser only knew JSON, so every well-formed Qwen call was scored
    ``malformed_tool_call`` and no answer was ever generated.
    """
    match = _XML_FUNCTION_RE.match(inner.strip())
    if match is None or match.group("name") != "bgkit":
        return None
    body = match.group("body")
    params: dict[str, str] = {}
    end = 0
    for pm in _XML_PARAMETER_RE.finditer(body):
        if body[end:pm.start()].strip():
            return None  # stray text between parameters
        if pm.group("key") in params:
            return None
        params[pm.group("key")] = pm.group("value")
        end = pm.end()
    if body[end:].strip():
        return None
    if "ids" not in params or not set(params).issubset({"ids", "query"}):
        return None
    try:
        ids = json.loads(params["ids"])
    except json.JSONDecodeError:
        return None
    return _validated_bgkit_args({**params, "ids": ids})


def parse_bgkit_call(text: str) -> dict | None:
    """Parse a structured bgkit call; malformed/prose-only output fails closed.

    Accepts exactly the two renderings the decoders are trained on: the
    JSON body (Falcon-H1's template, also bare JSON) and the XML
    ``<function=bgkit>`` body (Qwen3.5's template), optionally wrapped in
    ``<tool_call>`` tags that must span the whole turn.
    """
    decoder = json.JSONDecoder()
    stripped = text.strip()
    candidates = [stripped]
    open_tag = "<tool_call>"
    close_tag = "</tool_call>"
    if open_tag in stripped or close_tag in stripped:
        # An explicitly tagged call must consume the entire assistant turn.
        # Otherwise prose after valid inner JSON would be silently ignored.
        if not (stripped.startswith(open_tag) and stripped.endswith(close_tag)):
            return None
        inner = stripped[len(open_tag):-len(close_tag)]
        if open_tag in inner or close_tag in inner:
            return None
        candidates = [inner.strip()]
    for candidate in candidates:
        if candidate.startswith("<function="):
            return _call_from_xml(candidate)
        if not candidate or candidate[0] not in "[{":
            continue
        try:
            value, end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if candidate[end:].strip():
            continue
        found = _call_from_json_value(value)
        if found is not None:
            return found
    return None


def parse_bgkit_call_ids(text: str) -> list[str] | None:
    """Parse exact IDs from rendered tool-call JSON; malformed text fails closed."""
    call = parse_bgkit_call(text)
    return list(call["ids"]) if call is not None else None


@torch.no_grad()
def evaluate_free_running_sample(
    trainer: KRKBTrainer,
    sample: KBSample,
    *,
    max_tool_calls: int = 16,
    max_new_tokens: int = 8192,
    force_first_call: bool = False,
) -> dict:
    """Evaluate autonomous navigation without rendering future teacher turns.

    Each generated call is accepted only if every ID was initially exposed in
    scope or surfaced by an already executed tree node. Accepted calls are
    appended to the observed history and re-encoded before generation resumes.
    The first non-call output is the model's final answer.

    ``force_first_call`` seeds the history with the sample's first gold
    retrieval turn instead of making the model produce it. Use it when
    retrieval is NOT what is under test — a benchmark arm compared against
    baselines that are simply *handed* their context (BABILong's ``full`` /
    ``truncate``) must not additionally be charged for finding it. It also
    keeps an out-of-distribution ID format from masking the capability: the
    2026-08-25 BABILong arm scored invalid on 100% of samples because the
    wide net, trained on ``file:``/``log:`` ids, hallucinated one rather than
    copying the ``babilong:`` entrypoint out of the scope line.
    """
    tree = trainer._trees[sample.dataset_name]
    entrypoint = str(getattr(sample, "group_id", "") or "")
    available: set[str] = {entrypoint} if entrypoint else set(sample.topic_list)
    # IDs the system prompt names explicitly ("...; entrypoint id: X") are in
    # scope from the first turn — flat single-document tasks expose their
    # document this way (2026-08-23: before the writer named it, the trained
    # call carried an id the prompt never showed, and this evaluator rejected
    # every flat call as unsurfaced).
    available.update(scope_entrypoint_ids(str(getattr(sample, "scope_description", "") or "")))
    if not available and "root" in tree:
        available.add("root")

    expected_calls = [
        list(turn.args.get("ids", []))
        for turn in sample.trajectory
        if turn.kind == "bgkit" and getattr(turn, "loss", True)
    ]
    required_evidence = {
        str(identifier)
        for turn in sample.trajectory
        if turn.kind == "bgkit"
        and getattr(turn, "loss", True)
        and turn.args.get("is_retrieval")
        for identifier in turn.args.get("ids", [])
    }

    history: list[TrajectoryTurn] = []
    generated_calls: list[list[str]] = []
    called_ids: set[str] = set()
    if force_first_call:
        gold_first = next(
            (t for t in sample.trajectory if t.kind == "bgkit" and getattr(t, "loss", True)),
            None,
        )
        if gold_first is not None:
            ids = [str(i) for i in gold_first.args.get("ids", [])]
            history.append(
                TrajectoryTurn(
                    kind="bgkit",
                    args=dict(gold_first.args),
                    response="",
                    loss=True,
                )
            )
            generated_calls.append(ids)
            called_ids.update(ids)
            available.update(ids)
    invalid_reason = ""
    pred_answer = ""
    final_answer_emitted = False
    last_text = ""
    for call_index in range(max_tool_calls + 1):
        text = trainer.generate_kb_turn(
            sample,
            history,
            max_new_tokens=max_new_tokens,
        )
        last_text = text
        call = parse_bgkit_call(text)
        if call is None:
            if "<tool_call" in text or "</tool_call>" in text:
                invalid_reason = "malformed_tool_call"
                break
            # Keep exact bytes for code-state exact match. Token-F1 performs
            # its own normalization, but the capability metric must penalize
            # added/removed leading or trailing whitespace and final newlines.
            pred_answer = text
            final_answer_emitted = True
            break
        if call_index >= max_tool_calls:
            invalid_reason = "tool_call_limit"
            break
        ids = [str(identifier) for identifier in call["ids"]]
        if not ids:
            invalid_reason = "empty_ids"
            break
        unavailable = [identifier for identifier in ids if identifier not in available]
        if unavailable:
            invalid_reason = f"unsurfaced_id:{unavailable[0]}"
            break

        is_retrieval = True
        structural_depth = 0
        for identifier in ids:
            called_ids.add(identifier)
            # Flat browse trees index ARTICLES as nodes too (`is_article`);
            # only a non-article node is a structural (head / drill) call.
            node = tree.get(identifier) if identifier in tree else None
            if node is not None and not node.is_article:
                is_retrieval = False
                available.update(node.children)
                available.update(node.articles)
                structural_depth = max(
                    structural_depth,
                    max(0, len(tree.path_to(identifier)) - 1),
                )
            elif node is None and tree.leaf_tag_for_article(identifier) is None:
                invalid_reason = f"unknown_article:{identifier}"
                break
        if invalid_reason:
            break
        generated_calls.append(ids)
        # The recorded turn must match the DATA contract the trainer routes
        # on: a retrieval call (article ids, the flat datasets' only form) is a
        # plain leaf drill ``{ids, query}``; only a tree-NODE call is a head /
        # structural drill. 2026-08-23: tagging every first call ``is_head``
        # sent the first valid fileneedle call into the head-drill guard and
        # the exception killed the training run (227 steps lost).
        args: dict = {"ids": ids, "query": str(call.get("query", ""))}
        if not is_retrieval:
            args.update({
                "is_head": call_index == 0,
                "is_retrieval": False,
                "structural_depth": structural_depth,
                "drill_mode": "free_running",
            })
        history.append(TrajectoryTurn(kind="bgkit", args=args, response="", loss=True))
    else:  # pragma: no cover - loop always exits through its explicit bounds
        invalid_reason = "tool_call_limit"

    gold_answer = str(getattr(sample, "gold_answer", "") or "")
    answer_f1 = token_f1(pred_answer, [gold_answer]) if pred_answer and gold_answer else 0.0
    answer_exact = float(final_answer_emitted and pred_answer == gold_answer)
    matched_prefix = 0
    for predicted, expected in zip(generated_calls, expected_calls, strict=False):
        if predicted != expected:
            break
        matched_prefix += 1
    evidence_recall = (
        len(required_evidence & called_ids) / len(required_evidence)
        if required_evidence else 1.0
    )
    route_exact = bool(
        not invalid_reason
        and final_answer_emitted
        and generated_calls == expected_calls
    )
    return {
        "route_exact": float(route_exact),
        "valid_navigation": float(not invalid_reason and final_answer_emitted),
        "tool_calls": len(generated_calls),
        "expected_tool_calls": len(expected_calls),
        "matched_call_prefix": matched_prefix,
        "evidence_recall": evidence_recall,
        "answer_token_f1": answer_f1,
        "answer_exact_match": answer_exact,
        "pred_answer": pred_answer,
        "gold_answer": gold_answer,
        "invalid_reason": invalid_reason,
        # The last raw generation (truncated) so an invalid verdict can be
        # inspected without re-running the model — the 2026-08-23 parser /
        # stop-token defects were only visible in the raw text.
        "raw_text": last_text[:300],
    }


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
    ids, parse the predicted tool-call JSON, and return ``1.0`` only when
    the exact ordered ``ids`` list equals ``gold_ids``. Empty gold lists,
    malformed JSON, prose substrings, and additional IDs all score zero.
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
    return float(parse_bgkit_call_ids(pred_text) == gold_ids)


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
        # Score ONLY the supervised, rep-dependent nav drills. Skip:
        #  - the head drill (is_head): its opaque id has no prior rep to read
        #    from (its own rep is spliced AFTER the id) — pure memorization, not
        #    a nav test.
        #  - distractors (loss=False): never trained, and _score_call_span would
        #    score the model's UNMASKED prediction over their span (no loss
        #    tokens) → guaranteed ~0, pinning the micro-average near zero
        #    regardless of what the real nav drills learn.
        if not _is_scored_bgkit_turn(turn):
            continue
        gold_ids = _tool_call_gold_ids(turn)
        if not gold_ids:
            continue
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
        if not _is_scored_bgkit_turn(turn):
            continue
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
    scorable = bool(pred_text and gold_answer)
    f1 = token_f1(pred_text, [gold_answer]) if scorable else 0.0
    # Exact match beside F1. F1 is generous to a prediction that overlaps the
    # gold without being it, and set-valued answers (grepset's path lists)
    # score partial credit for naming one member -- so a family can look like
    # it is reading while getting the answer wrong. The free-running path has
    # reported both since it was written; teacher forcing reported only F1,
    # which is why every rep-dependence comparison had to be assembled by hand.
    em = exact_match(pred_text, [gold_answer]) if scorable else 0.0

    return {
        "trajectory_step_accuracy": step_acc,
        "trajectory_total_tokens": total_tokens,
        "trajectory_correct_tokens": correct_tokens,
        "tool_call_id_accuracy": tool_call_metrics,
        "answer_token_f1": f1,
        "answer_exact_match": em,
        "pred_answer": pred_text,
        "gold_answer": gold_answer,
    }
