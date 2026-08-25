"""Tests for the bgkit tool schema and multi-sentinel chat rendering."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.bgkit_tool_template import (
    BGKIT_SENTINEL,
    BGKIT_TOOL,
    TrajectoryTurn,
    assistant_generation_prompt_ids,
    make_system_prompt,
    tokenize_trajectory,
)


@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    except Exception:
        pytest.skip("Qwen3.5 tokenizer not available locally")


def test_system_prompt_topic_list():
    prompt = make_system_prompt("topic_list", topic_list=["Physics", "Biology"])
    assert "bgkit(ids, query)" in prompt
    assert "browse(" not in prompt
    assert "Physics" in prompt and "Biology" in prompt


def test_system_prompt_pre_scoped():
    prompt = make_system_prompt("pre_scoped", scope_description="git:foo/bar")
    assert "git:foo/bar" in prompt
    assert "browse(" not in prompt
    assert "bgkit(ids" in prompt


def test_tool_schemas_present():
    assert BGKIT_TOOL["function"]["name"] == "bgkit"
    assert "ids" in BGKIT_TOOL["function"]["parameters"]["properties"]
    assert "query" in BGKIT_TOOL["function"]["parameters"]["properties"]


def test_qwen_apply_chat_template_renders_bgkit_tool(tokenizer):
    """Verify Qwen's native apply_chat_template accepts BGKIT_TOOL and emits
    its schema in the rendered system context.

    If Qwen's template drops or garbles the tool, training data rendered
    via ``tokenize_trajectory`` wouldn't include the tool schema the
    decoder needs to learn to use it — silent training bug. This test
    catches that before it happens.
    """
    messages = [
        {"role": "system", "content": "You are a helper."},
        {"role": "user", "content": "test"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=[BGKIT_TOOL],
        tokenize=False,
        add_generation_prompt=False,
    )
    # Tool name and key parameters must appear in the rendered string.
    assert "bgkit" in rendered
    assert '"ids"' in rendered or "ids" in rendered
    assert '"query"' in rendered or "query" in rendered
    # The system prompt the user supplied must be preserved.
    assert "You are a helper" in rendered


def test_qwen_apply_chat_template_accepts_tool_call_roundtrip(tokenizer):
    """Verify that a tool-call-formatted assistant message with BGKIT_TOOL
    in ``tools=`` survives rendering AND tokenization without corruption.

    This exercises the full path the trainer takes — if Qwen's chat
    template can't handle tool_calls with our specific BGKIT_TOOL schema
    (e.g. because the schema uses an 'array of string' where Qwen expects
    'string'), rendering fails here loud instead of during training.
    """
    messages = [
        {"role": "system", "content": "You are a helper."},
        {"role": "user", "content": "test"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "bgkit",
                        "arguments": {
                            "ids": ["Physics/sub_1"],
                            "query": "what?",
                        },
                    },
                },
            ],
        },
        {
            "role": "tool",
            "name": "bgkit",
            "content": BGKIT_SENTINEL,
        },
        {"role": "assistant", "content": "answer"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=[BGKIT_TOOL],
        tokenize=False,
        add_generation_prompt=False,
    )
    # Bgkit tool call appears with its id list
    assert "bgkit" in rendered
    assert "Physics/sub_1" in rendered
    # The sentinel survived the chat template rendering intact so the
    # trainer's position detection can find it.
    assert BGKIT_SENTINEL in rendered
    # Final answer is in the output
    assert "answer" in rendered
    # Sanity: the tokenizer can actually tokenize the rendered string
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    assert len(ids) > 0


def test_tokenize_trajectory_loss_mask_and_sentinel(tokenizer):
    trajectory = [
        TrajectoryTurn(
            kind="bgkit", args={"ids": ["Physics/sub_1"], "query": "why?"},
            loss=True,
        ),
        TrajectoryTurn(
            kind="bgkit", args={"ids": ["Physics/sub_2"], "query": "why?"},
            loss=False,
        ),  # sibling exploration
        TrajectoryTurn(
            kind="answer", response="42", loss=True,
        ),
    ]
    system = make_system_prompt("topic_list", topic_list=["Physics"])
    rendered = tokenize_trajectory(tokenizer, system, "what is life?", trajectory)

    # Two bgkit calls → two sentinel positions
    assert len(rendered.bgkit_sentinel_positions) == 2
    assert len(rendered.bgkit_turns) == 2
    # The exploration turn must be the second one (index 1)
    assert rendered.bgkit_turns[0].loss is True
    assert rendered.bgkit_turns[1].loss is False

    # Loss mask is True somewhere (the answer at least) and False on tool responses
    assert rendered.loss_mask.any().item()
    assert not rendered.loss_mask.all().item()

    # Sentinel positions must fall in loss-masked-False region (tool response)
    for pos in rendered.bgkit_sentinel_positions:
        assert rendered.loss_mask[pos].item() is False


@pytest.fixture(scope="module", params=["Qwen/Qwen3.5-0.8B", "tiiuae/Falcon-H1-Tiny-90M-Instruct"])
def any_tokenizer(request):
    """Both decoder families' templates (Qwen3.5 renders XML tool calls and an
    empty ``<think>`` scaffold; Falcon-H1 renders JSON and closes turns with
    ``<|im_end|>`` while its eos is ``<|end_of_text|>``)."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from bgkit.data.chat_template import patch_falcon_h1_chat_template

    try:
        tok = AutoTokenizer.from_pretrained(request.param, trust_remote_code=True)
    except Exception:
        pytest.skip(f"{request.param} tokenizer not available locally")
    patch_falcon_h1_chat_template(tok)
    return tok


def test_loss_and_spans_cover_content_not_scaffold(any_tokenizer):
    """Loss applies to the emitted content + the end-of-turn token; the
    assistant scaffold (header, Qwen's empty think block) and template glue
    are masked. ``answer_span`` / ``bgkit_call_spans`` decode to exactly the
    content, and the decoded call body round-trips through the eval parser
    (ties the parser to what the decoder is trained to emit)."""
    from bgkit.data.bgkit_tool_template import assistant_turn_scaffold
    from bgkit.data.chat_template import end_of_turn_token_ids
    from bgkit.eval.kb_trajectory_eval import parse_bgkit_call

    tok = any_tokenizer
    trajectory = [
        TrajectoryTurn(kind="bgkit", args={"ids": ["doc-1"], "query": "why?"}, loss=True),
        TrajectoryTurn(kind="answer", response="def fetch_page(url):", loss=True),
    ]
    system = make_system_prompt("pre_scoped", scope_description="source file a.py")
    r = tokenize_trajectory(tok, system, "Quote the line.", trajectory)
    ids = r.token_ids.tolist()
    mask = r.loss_mask.tolist()

    def end_of_turn_end(start: int) -> int:
        """Index after the first end-of-turn marker at/after ``start``."""
        eot = end_of_turn_token_ids(tok)
        return next(i for i in range(start, len(ids)) if ids[i] in eot) + 1

    a, b = r.answer_span
    assert tok.decode(ids[a:b]) == "def fetch_page(url):"
    # Targets: content + end-of-turn emission (glue + marker); nothing after.
    e_ans = end_of_turn_end(b)
    assert e_ans - b <= 2 and all(mask[a:e_ans])
    assert all(not m for m in mask[e_ans:])
    # Nothing before the answer content inside the answer turn bears loss.
    history = [{"role": "system", "content": system}, {"role": "user", "content": "Quote the line."}]
    pre, _post = assistant_turn_scaffold(tok, history, [BGKIT_TOOL])
    n_pre = len(tok.encode(pre, add_special_tokens=False))
    assert not any(mask[a - n_pre : a])
    assert "assistant" in pre and "<think>" not in tok.decode(ids[a:b])

    (c0, c1), = r.bgkit_call_spans
    body = tok.decode(ids[c0:c1])
    assert body.startswith("<tool_call>") and body.endswith("</tool_call>")
    assert parse_bgkit_call(body) == {"ids": ["doc-1"], "query": "why?"}
    e = end_of_turn_end(c1)
    assert e - c1 <= 2 and all(mask[c0:e])
    assert not any(mask[c0 - n_pre : c0])
    # Tool responses never bear loss.
    for pos in r.bgkit_sentinel_positions:
        assert not mask[pos]
    # Generation-time prompt == the masked scaffold, so the decoder never
    # has to emit it; the glue it emits before the stop token is known.
    from bgkit.data.bgkit_tool_template import assistant_turn_end_glue

    gen = assistant_generation_prompt_ids(tok, system, "Quote the line.", trajectory[:1])
    assert tok.decode(gen) == pre
    glue = assistant_turn_end_glue(tok, history, [BGKIT_TOOL])
    assert glue in ("", "\n")
    assert tok.decode(ids[b : e_ans - 1]) == glue


def test_loss_false_turn_has_no_targets(any_tokenizer):
    trajectory = [
        TrajectoryTurn(kind="bgkit", args={"ids": ["doc-1"], "query": "q"}, loss=False),
        TrajectoryTurn(kind="answer", response="x", loss=True),
    ]
    r = tokenize_trajectory(
        any_tokenizer, make_system_prompt("topic_list", topic_list=["T"]), "q", trajectory
    )
    (_c0, c1), = r.bgkit_call_spans
    assert not any(r.loss_mask.tolist()[: c1 + 2])  # incl. its end-of-turn emission
    assert r.loss_mask[r.answer_span[0]].item()


def test_generation_prompt_appends_after_observed_tool_history(tokenizer):
    history = [TrajectoryTurn(
        kind="bgkit",
        args={"ids": ["opaque entrypoint"], "query": "where?", "is_head": True},
    )]
    system = make_system_prompt(
        "pre_scoped",
        scope_description="repository; entrypoint id: opaque entrypoint",
    )
    prompt = assistant_generation_prompt_ids(
        tokenizer,
        system,
        "question",
        history,
    )
    assert prompt.dtype == torch.long
    assert prompt.numel() > 0
