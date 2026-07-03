"""Chat templating for the Phase 2 KB-scale ``bgkit`` tool.

Contains:

- Tool schemas used with ``tokenizer.apply_chat_template(..., tools=...)``.
- System prompt templates: ``topic_list`` (for Wikipedia / PubMedQA where the
  LLM picks the top-level topic) and ``pre_scoped`` (for single-book, single-repo,
  single-user corpora where the caller has already narrowed scope).
- A multi-sentinel rendering helper that turns an annotated trajectory
  (bgkit turns + answer) into (token_ids, loss_mask, bgkit_injection_points)
  with per-turn loss masking controlled by the trajectory's ``loss`` flags.

Design notes
------------
The decoder consumes a single long multi-turn sequence. Each ``bgkit`` tool
response is a sentinel string in the text; during forward pass we splice in
live-computed L1 survivor embeddings at each sentinel position. Because each
bgkit call produces a variable number of survivor positions, splicing must
happen *in embedding space* — we replace the single sentinel token with a
run of survivor vectors and shift everything that follows.

Loss mask semantics
-------------------
Each assistant-role tool call and final assistant text carries a ``loss``
flag from the trajectory. Sibling exploration tool calls have ``loss=False``
— the decoder is not trained to emit them, but the encoder gradient still
flows through the survivor embeddings they produced (training-time
augmentation that stops L1 from only working on perfect targets).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import torch

BGKIT_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "bgkit",
        "description": (
            "Retrieve query-conditioned compressed knowledge. Runs the BgKIT "
            "encoder over the articles in the given tag(s) with the query as "
            "prompt and returns query-focused survivors that summarize relevant "
            "information. The response includes IDs of related articles or "
            "sub-tags you can drill into with further bgkit calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tag or article IDs to retrieve.",
                },
                "query": {
                    "type": "string",
                    "description": "Query to condition retrieval on.",
                },
            },
            "required": ["ids", "query"],
        },
    },
}

BGKIT_TOPIC_KNOWLEDGE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "bgkit_topic_knowledge",
        "description": (
            "Load dense learned embeddings for a set of taxonomy tags. "
            "Provides domain-level prior knowledge (e.g. 'python/webdev/flask') "
            "that complements document-specific compressed context from the "
            "bgkit tool. Called automatically at the start of a session based "
            "on the sample's tags — the decoder is NOT trained to emit this "
            "call itself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Taxonomy tags whose learned embeddings to load.",
                },
            },
            "required": ["tags"],
        },
    },
}


SYSTEM_TOPIC_LIST = (
    "You have access to a knowledge base through one tool:\n"
    "- bgkit(ids, query): retrieve query-focused compressed content for a tag or article\n"
    "\n"
    "Topics available: {topic_list}\n"
    "\n"
    "To answer a question, identify a relevant topic, then use bgkit to "
    "retrieve compressed content focused on the question. Drill into specific "
    "articles or sub-tags via additional bgkit calls on the IDs the response "
    "surfaces. Write the final answer when you have enough context."
)


SYSTEM_PRE_SCOPED = (
    "You have access to a knowledge base through one tool:\n"
    "- bgkit(ids, query): retrieve query-focused compressed content\n"
    "\n"
    "Knowledge base: {scope_description}\n"
    "\n"
    'Start by calling bgkit(ids=["root"], query=...), then drill down via '
    "further bgkit calls on the IDs the response surfaces to answer the question."
)


SYSTEM_FLAT = (
    "You have access to a knowledge base through one tool:\n"
    "- bgkit(ids, query): retrieve query-focused compressed content for a "
    "specific article or set of articles\n"
    "\n"
    "Knowledge base: {scope_description}\n"
    "\n"
    "This corpus has no navigable hierarchy. Call bgkit directly with the "
    "article ID(s) you need and the question, then write the final answer."
)


# Unique, long-random sentinel strings so tokenization collisions are effectively zero.
BGKIT_SENTINEL = "<<<BGKIT_L1_SURVIVORS_7f31a4c2>>>"
BGKIT_TOPIC_SENTINEL = "<<<BGKIT_TOPIC_KNOWLEDGE_3f82a1e0>>>"


def make_system_prompt(
    template: Literal["topic_list", "pre_scoped", "flat"],
    *,
    topic_list: list[str] | None = None,
    scope_description: str | None = None,
) -> str:
    if template == "topic_list":
        if not topic_list:
            raise ValueError("topic_list template requires non-empty topic_list")
        return SYSTEM_TOPIC_LIST.format(topic_list=", ".join(topic_list))
    if template == "pre_scoped":
        if not scope_description:
            raise ValueError("pre_scoped template requires scope_description")
        return SYSTEM_PRE_SCOPED.format(scope_description=scope_description)
    if template == "flat":
        if not scope_description:
            raise ValueError("flat template requires scope_description")
        return SYSTEM_FLAT.format(scope_description=scope_description)
    raise ValueError(f"Unknown template: {template!r}")


# ---------------------------------------------------------------------------
# Trajectory types
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryTurn:
    """One step in a teacher trajectory.

    Kinds and the meaning of ``response`` for each:

    - ``bgkit``: ``args = {"ids": [...], "query": "..."}``. ``response`` is
      unused and should be the empty string — at training time the tool
      response is always just :data:`BGKIT_SENTINEL`, and the trainer
      splices L1 survivor embeddings in at that position.
    - ``answer``: final assistant text. ``args`` is empty; ``response``
      is the gold answer string the decoder is trained to emit.
    """

    kind: Literal["bgkit", "answer"]
    args: dict = field(default_factory=dict)
    response: str = ""
    loss: bool = True


@dataclass
class RenderedTrajectory:
    token_ids: torch.Tensor  # (L,) long
    loss_mask: torch.Tensor  # (L,) bool — True where decoder CE should apply
    bgkit_sentinel_positions: list[int]  # index per bgkit turn, aligned with bgkit_turns
    bgkit_turns: list[TrajectoryTurn]     # the ordered list of bgkit calls
    answer_span: tuple[int, int] | None = None
    """Absolute ``[start, end)`` token range of the final ``answer`` turn in
    ``token_ids``, or ``None`` if the trajectory has no answer turn.

    Used by the KB trainer's eval path to compute EM/F1 only over the
    answer (excluding bgkit calls and tool responses).
    """
    bgkit_call_spans: list[tuple[int, int]] = field(default_factory=list)
    """Per ``bgkit_turns[i]``, the absolute ``[start, end)`` token range of
    the *assistant tool-call emission* (the loss-bearing tokens that encode
    the tool call JSON — ``ids`` and ``query``). Excludes the sentinel and
    tool-response payload. Aligned by index with :attr:`bgkit_turns`.
    """
    topic_sentinel_position: int | None = None
    """Absolute position of the ``BGKIT_TOPIC_SENTINEL`` token in
    ``token_ids`` when topic knowledge is enabled for this sample, else
    ``None``. The trainer splices a single topic embedding block at this
    sentinel (replacing the sentinel tokens) just as it does for bgkit
    survivors. The topic turn itself is loss-masked — the decoder is not
    trained to emit the ``bgkit_topic_knowledge`` tool call, only to
    condition on the resulting embedding block.
    """
    topic_tags: list[str] = field(default_factory=list)
    """The list of tags the topic knowledge tool call references, for the
    trainer's side-channel (and for eval harnesses that want to log
    which tags were active per sample).
    """


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------


def _tool_call_assistant_message(name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args,
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# Tokenization with per-turn boundaries and loss masking
# ---------------------------------------------------------------------------


def _render_turn_boundary(
    tokenizer,
    prior_messages: list[dict],
    turn_messages: list[dict],
    tools: list[dict],
) -> tuple[str, str]:
    """Return (prefix, with_turn) both fully rendered via apply_chat_template.

    ``with_turn`` is the template for ``prior + turn``. ``prefix`` is the
    template for ``prior`` alone. The caller subtracts one from the other to
    isolate exactly this turn's rendered text.
    """
    prefix = tokenizer.apply_chat_template(
        prior_messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
    )
    with_turn = tokenizer.apply_chat_template(
        prior_messages + turn_messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
    )
    return prefix, with_turn


def tokenize_trajectory(
    tokenizer,
    system_prompt: str,
    question: str,
    trajectory: list[TrajectoryTurn],
    *,
    topic_knowledge_tags: list[str] | None = None,
) -> RenderedTrajectory:
    """Tokenize a multi-turn trajectory with per-turn loss masking.

    For each turn in the trajectory we render the template before and after
    including that turn and diff the strings to obtain the turn's rendered
    text. Tokens produced from assistant-role turns (bgkit tool calls or the
    final answer) are loss-masked according to ``turn.loss``; everything else
    (system, user, tool responses) is masked out.

    Sentinels for bgkit tool responses are detected and their token positions
    are returned so the trainer can splice survivor embeddings there.

    Args:
        tokenizer: HF tokenizer. Must support ``apply_chat_template(tools=...)``.
        system_prompt: system-role content.
        question: user-role content.
        trajectory: ordered list of trajectory turns (bgkit/answer).
        topic_knowledge_tags: optional list of taxonomy tags. When supplied
            and non-empty, the renderer injects a ``bgkit_topic_knowledge``
            tool-call pair right after the user question and before the
            first trajectory turn. The assistant tool-call message is
            always loss-masked (the decoder is not trained to emit it),
            and the tool-response body carries ``BGKIT_TOPIC_SENTINEL``.
            The trainer splices the topic embedding block at the sentinel
            position exactly the way bgkit survivors are spliced.
    """
    tools = [BGKIT_TOOL, BGKIT_TOPIC_KNOWLEDGE_TOOL]

    # Base: system + user. The loss on these tokens is False.
    base_messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    base_str = tokenizer.apply_chat_template(
        base_messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
    )
    token_ids = tokenizer.encode(base_str, add_special_tokens=False)
    loss_mask = [False] * len(token_ids)

    messages: list[dict] = list(base_messages)
    prior_str = base_str

    bgkit_turns: list[TrajectoryTurn] = []
    bgkit_sentinel_positions: list[int] = []
    bgkit_call_spans: list[tuple[int, int]] = []
    answer_span: tuple[int, int] | None = None
    topic_sentinel_position: int | None = None
    resolved_topic_tags: list[str] = list(topic_knowledge_tags or [])

    # Inject the bgkit_topic_knowledge tool-call pair before the main
    # trajectory loop, if topic tags are supplied. The assistant tool
    # call and the tool response are BOTH loss-masked — the decoder is
    # not trained to emit either side; the whole point is to condition
    # on the topic embedding block that will be spliced at the
    # ``BGKIT_TOPIC_SENTINEL`` inside the tool response body.
    if resolved_topic_tags:
        topic_sub = [
            _tool_call_assistant_message(
                "bgkit_topic_knowledge",
                {"tags": list(resolved_topic_tags)},
            ),
            {
                "role": "tool",
                "name": "bgkit_topic_knowledge",
                "content": BGKIT_TOPIC_SENTINEL,
            },
        ]
        new_str = tokenizer.apply_chat_template(
            messages + topic_sub,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )
        if not new_str.startswith(prior_str):
            raise RuntimeError(
                "apply_chat_template did not produce a prefix-extension "
                "for the topic_knowledge injection",
            )
        turn_text = new_str[len(prior_str):]
        sentinel_idx = turn_text.find(BGKIT_TOPIC_SENTINEL)
        if sentinel_idx < 0:
            raise RuntimeError(
                "BGKIT_TOPIC_SENTINEL missing from rendered topic_knowledge "
                "turn — template may have dropped the tool content",
            )
        before = turn_text[:sentinel_idx]
        after = turn_text[sentinel_idx + len(BGKIT_TOPIC_SENTINEL):]
        before_ids = tokenizer.encode(before, add_special_tokens=False)
        sentinel_ids = tokenizer.encode(
            BGKIT_TOPIC_SENTINEL, add_special_tokens=False,
        )
        after_ids = tokenizer.encode(after, add_special_tokens=False)
        # All topic turn tokens get loss=False: the decoder is never
        # trained to emit the tool call or its response.
        topic_sentinel_position = len(token_ids) + len(before_ids)
        token_ids.extend(before_ids)
        loss_mask.extend([False] * len(before_ids))
        token_ids.extend(sentinel_ids)
        loss_mask.extend([False] * len(sentinel_ids))
        token_ids.extend(after_ids)
        loss_mask.extend([False] * len(after_ids))
        messages.extend(topic_sub)
        prior_str = new_str

    for turn in trajectory:
        # Build the sub-messages this turn adds.
        if turn.kind == "bgkit":
            # The bgkit tool response is JUST the sentinel. There is no
            # text side-channel — drill-down relies entirely on ID
            # pinning carrying article IDs through the L1 encoder so
            # the decoder can read them out of the spliced survivor
            # embeddings. ``turn.response`` is ignored for bgkit turns.
            sub = [
                _tool_call_assistant_message("bgkit", dict(turn.args)),
                {"role": "tool", "name": "bgkit", "content": BGKIT_SENTINEL},
            ]
        elif turn.kind == "answer":
            sub = [{"role": "assistant", "content": turn.response}]
        else:
            raise ValueError(f"Unknown turn kind: {turn.kind!r}")

        new_str = tokenizer.apply_chat_template(
            messages + sub,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )
        if not new_str.startswith(prior_str):
            raise RuntimeError(
                "apply_chat_template did not produce a prefix-extension — "
                "template is not idempotent over message append."
            )
        turn_text = new_str[len(prior_str):]

        # The assistant-role message (tool call or answer) is the part that
        # should receive loss on tool-call turns. The subsequent tool-response
        # text should NOT (it's model context, not model output). For the
        # answer turn there is no tool response.
        # Heuristic: split the rendered turn text on the start of the tool
        # response. When sub contains a tool message, we split where the
        # content begins. For ``bgkit`` turns the sentinel is easy to find.

        tool_turn_tokens: list[int]
        assistant_tokens: list[int]
        tool_turn_tokens_start = -1

        if turn.kind == "answer":
            assistant_tokens = tokenizer.encode(turn_text, add_special_tokens=False)
            tool_turn_tokens = []
        elif turn.kind == "bgkit":
            # Split at the sentinel string. The part before the sentinel is
            # the assistant tool call + the template's opening tool wrapper
            # (treated as assistant-region tokens for loss purposes); the
            # sentinel + the trailing tool-response text are the "tool turn"
            # tokens (loss always False).
            sentinel_idx = turn_text.find(BGKIT_SENTINEL)
            if sentinel_idx < 0:
                raise RuntimeError(
                    "Sentinel missing from rendered bgkit turn — template "
                    "may have dropped the tool content"
                )
            before = turn_text[:sentinel_idx]
            after = turn_text[sentinel_idx + len(BGKIT_SENTINEL):]
            before_ids = tokenizer.encode(before, add_special_tokens=False)
            sentinel_ids = tokenizer.encode(BGKIT_SENTINEL, add_special_tokens=False)
            after_ids = tokenizer.encode(after, add_special_tokens=False)
            assistant_tokens = before_ids
            tool_turn_tokens = sentinel_ids + after_ids
            # Sentinel lives at offset 0 within tool_turn_tokens.
            tool_turn_tokens_start = 0
        else:
            raise ValueError(f"Unknown turn kind: {turn.kind!r}")

        # For answer turns, capture the absolute [start, end) token range so
        # the trainer's eval path can compute EM/F1 over only the answer
        # text (not the bgkit tool call emissions that also bear loss). The
        # last answer turn wins if a trajectory has multiple.
        answer_start_abs = len(token_ids) if turn.kind == "answer" else -1
        # Absolute token offset of the assistant tool-call emission (before
        # we extend ``token_ids``). Used below to record per-call spans so
        # downstream eval harnesses can score tool-call ID accuracy over
        # exactly the tokens that encode the tool arguments.
        assistant_start_abs = len(token_ids)

        token_ids.extend(assistant_tokens)
        loss_mask.extend([bool(turn.loss)] * len(assistant_tokens))
        if turn.kind == "bgkit" and assistant_tokens:
            bgkit_call_spans.append(
                (assistant_start_abs, assistant_start_abs + len(assistant_tokens))
            )
        if tool_turn_tokens:
            if turn.kind == "bgkit":
                # Capture sentinel position for splice-time replacement.
                abs_start = len(token_ids) + tool_turn_tokens_start
                bgkit_turns.append(turn)
                bgkit_sentinel_positions.append(abs_start)
            token_ids.extend(tool_turn_tokens)
            loss_mask.extend([False] * len(tool_turn_tokens))

        if turn.kind == "answer" and turn.loss:
            answer_span = (answer_start_abs, answer_start_abs + len(assistant_tokens))

        messages.extend(sub)
        prior_str = new_str

    return RenderedTrajectory(
        token_ids=torch.tensor(token_ids, dtype=torch.long),
        loss_mask=torch.tensor(loss_mask, dtype=torch.bool),
        bgkit_sentinel_positions=bgkit_sentinel_positions,
        bgkit_turns=bgkit_turns,
        answer_span=answer_span,
        bgkit_call_spans=bgkit_call_spans,
        topic_sentinel_position=topic_sentinel_position,
        topic_tags=resolved_topic_tags,
    )


# ---------------------------------------------------------------------------
# Trajectory serialization for offline builders
# ---------------------------------------------------------------------------


def trajectory_to_json(trajectory: list[TrajectoryTurn]) -> str:
    return json.dumps([
        {
            "kind": t.kind,
            "args": t.args,
            "response": t.response,
            "loss": t.loss,
        }
        for t in trajectory
    ])


def trajectory_from_json(blob: str) -> list[TrajectoryTurn]:
    data = json.loads(blob)
    return [
        TrajectoryTurn(
            kind=row["kind"],
            args=dict(row.get("args", {})),
            response=row.get("response", ""),
            loss=bool(row.get("loss", True)),
        )
        for row in data
    ]


def articles_referenced_by_trajectory(
    trajectory: Iterable[TrajectoryTurn],
) -> list[str]:
    """Return the flat list of all tag/article IDs referenced by bgkit turns."""
    out: list[str] = []
    for t in trajectory:
        if t.kind == "bgkit":
            ids = t.args.get("ids", [])
            if isinstance(ids, list):
                out.extend(str(x) for x in ids)
    return out
