"""Third-party trajectory adapters -> the role/content/tool_calls shape."""

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "conv", Path(__file__).resolve().parents[3] / "scripts" / "bgkit2_convert_trajectories.py"
)
conv = importlib.util.module_from_spec(spec)
sys.modules["conv"] = conv
spec.loader.exec_module(conv)


def _h(role, text, channel=None, recipient=None):
    return {"role": role, "channel": channel, "recipient": recipient, "content": [{"text": text}]}


def test_openresearcher_harmony_to_openai():
    msgs = [
        _h("system", "You are ChatGPT."),
        _h("developer", "You are a deep research agent."),
        _h("user", "Question: who?"),
        _h("assistant", "We need to search.", channel="analysis"),
        _h(
            "assistant",
            '{"query": "who", "topn": 10}',
            channel="analysis",
            recipient="browser.search",
        ),
        _h("tool", "[0] result lines", channel="analysis", recipient="assistant"),
        _h("assistant", "Found it.", channel="analysis"),
        _h("assistant", "\\boxed{Bob}", channel="final"),
    ]
    out = conv.convert_openresearcher(msgs)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "assistant"]
    assert out[0]["content"] == "You are a deep research agent."
    call = out[2]
    assert call["content"] == "We need to search."
    assert call["tool_calls"][0]["function"] == {
        "name": "browser_search",
        "arguments": {"query": "who", "topn": 10},
    }
    assert out[3] == {"role": "tool", "content": "[0] result lines"}
    assert out[4]["content"] == "Found it.\n\\boxed{Bob}"


def test_sharegpt_roles():
    out = conv.convert_sharegpt(
        [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "do it"},
            {"from": "gpt", "value": "<tool>ls</tool>"},
            {"from": "observation", "value": "a b c"},
            {"from": "weird", "value": "x"},
            {"from": "gpt", "value": {"k": 1}},
        ]
    )
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "assistant"]
    assert out[-1]["content"] == '{"k": 1}'
