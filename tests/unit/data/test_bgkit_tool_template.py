"""Tests for the bgkit tool schema and multi-sentinel chat rendering."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.bgkit_tool_template import (
    BGKIT_SENTINEL,
    BGKIT_TOOL,
    TrajectoryTurn,
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
