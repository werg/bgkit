"""scripts/bgkit2_build_web_sft_store.py --keep-reasoning: each assistant turn's reasoning goes
to ``reasoning_content`` (target and history) instead of into ``content`` as untagged prose."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from bgkit2.data.templates import SPLICE_SENTINEL

_p = Path(__file__).resolve().parents[4] / "scripts" / "bgkit2_build_web_sft_store.py"
_spec = importlib.util.spec_from_file_location("web_sft_kr", _p)
W = importlib.util.module_from_spec(_spec)
sys.modules["web_sft_kr"] = W
_spec.loader.exec_module(W)

LFM8B = Path("/home/werg/bgkit-data-nvme/models_hf/LFM2.5-8B-A1B")


def _rows(ep, **kw):
    from bgkit2.agent.context import PlainRenderer

    opts = dict(budget=10_000, max_rows=0, base_meta={})
    opts.update(kw)
    return W.replay_rows(ep, "web_x", PlainRenderer(), **opts)


def _r1_episode():
    q = "who wrote the hobbit?"
    rec = {
        "messages": [
            {"role": "system", "content": "You are a helpful and harmless assistant."},
            {"role": "user", "content": f"Answer the given question ... Question: {q}"},
            {
                "role": "assistant",
                "content": "<thinking>\nI should search.\n</thinking>\n"
                "<search>author of the hobbit</search>",
            },
            {
                "role": "user",
                "content": '<information>[Document 1] (ID: 42, Score: 0.9)\n"The Hobbit"\n'
                "The Hobbit The Hobbit is a novel by J. R. R. Tolkien.</information>",
            },
            {
                "role": "assistant",
                "content": "<thinking>It is Tolkien.</thinking>\n<answer>J. R. R. Tolkien</answer>",
            },
        ]
    }
    return W.parse_search_r1(rec, "t1")


def test_default_merges_reasoning_into_content():
    ep = _r1_episode()
    msgs, _ = W.episode_messages(ep)
    assert msgs[2]["content"] == "I should search." and "reasoning_content" not in msgs[2]
    assert msgs[4] == {"role": "assistant", "content": "It is Tolkien.\n\nANSWER: J. R. R. Tolkien"}
    for r in _rows(ep):
        assert "reasoning_content" not in r["target"] + r["instruction"]


def test_keep_reasoning_moves_it_to_reasoning_content():
    ep = _r1_episode()
    msgs, slots = W.episode_messages(ep, keep_reasoning=True)
    assert msgs[2]["content"] == ""
    assert msgs[2]["reasoning_content"] == "I should search."
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "web_search"
    assert msgs[3] == {"role": "tool", "content": SPLICE_SENTINEL}
    assert msgs[4] == {
        "role": "assistant",
        "content": "ANSWER: J. R. R. Tolkien",
        "reasoning_content": "It is Tolkien.",
    }
    # slots / prompts are the default build's
    assert slots == W.episode_messages(ep)[1]
    rows = _rows(ep, keep_reasoning=True)
    base = _rows(ep)
    assert [r["framing"] for r in rows] == [r["framing"] for r in base] == ["chat", "tool_slots"]
    for r, b in zip(rows, base, strict=True):
        assert (r["context"], r["prompt"]) == (b["context"], b["prompt"])
    (t0,) = json.loads(rows[0]["target"])
    assert t0["reasoning_content"] == "I should search." and t0["content"] == ""
    (t1,) = json.loads(rows[1]["target"])
    assert t1 == {
        "role": "assistant",
        "content": "ANSWER: J. R. R. Tolkien",
        "reasoning_content": "It is Tolkien.",
    }
    # the history keeps the earlier turn's reasoning, as the harness does for a thinking decoder
    hist = [m for m in json.loads(rows[1]["instruction"]) if m["role"] == "assistant"]
    assert [m.get("reasoning_content") for m in hist] == ["I should search."]
    assert all("<think" not in m.get("content", "") for m in hist)


def test_sds_think_block_closes_in_the_answering_generation():
    page = {"id": 1, "title": "Winnie", "url": "https://w/pooh", "context": "Shepard drew it."}
    rec = {
        "id": 2,
        "question": "who illustrated winnie the pooh",
        "answer": "E. H. Shepard",
        "output": [
            {"gen": "Let me search. <|begin_search_query|>pooh illustrator<|end_search_query|>"},
            {"search_query": "pooh illustrator"},
            {"doc": "**Web Page 1:**\n" + json.dumps(page, indent=2)},
            {"gen": "It is Shepard.\n</think>\n\nShepard.\n\n\\boxed{E. H. Shepard}"},
        ],
    }
    ep = W.parse_sds(rec, "s")
    merged, _ = W.episode_messages(ep)
    assert merged[-1]["content"] == "It is Shepard.\n\nShepard.\n\nANSWER: E. H. Shepard"
    msgs, _ = W.episode_messages(ep, keep_reasoning=True)
    assert (msgs[2]["reasoning_content"], msgs[2]["content"]) == ("Let me search.", "")
    assert msgs[-1]["reasoning_content"] == "It is Shepard."
    assert msgs[-1]["content"] == "Shepard.\n\nANSWER: E. H. Shepard"


def _call(cid, name, args):
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_openseeker_reasoning_content_and_call_less_carry():
    q = "Who founded the fort?"
    rec = {
        "question": q,
        "answer": "Anthony Wayne",
        "source": "openseeker",
        "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": "Plan first.", "reasoning_content": "Think a."},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Search it.",
                "tool_calls": [_call("c1", "search", {"query": "fort founder"})],
            },
            {
                "role": "tool",
                "content": "### A Google search for 'fort founder' found 1 results:\n\n"
                "1. [Fort X](https://f/x)\nFounded by Wayne.",
                "tool_call_id": "c1",
            },
            {
                "role": "assistant",
                "content": "Done. <answer>General Anthony Wayne</answer>",
                "reasoning_content": "Wayne.",
                "tool_calls": None,
            },
        ],
    }
    ep = W.parse_mix(rec, "os")
    merged, _ = W.episode_messages(ep)
    assert merged[2]["content"] == "Think a.\nPlan first.\n\nSearch it."
    msgs, _ = W.episode_messages(ep, keep_reasoning=True)
    # the call-less turn's words carry into the next turn, each half into its own field
    assert msgs[2]["reasoning_content"] == "Think a.\n\nSearch it."
    assert msgs[2]["content"] == "Plan first."
    assert msgs[-1] == {
        "role": "assistant",
        "content": "Done.\n\nANSWER: General Anthony Wayne",
        "reasoning_content": "Wayne.",
    }


def test_turn_without_reasoning_has_no_field():
    ep = _r1_episode()
    ep.final_reasoning = ""
    msgs, _ = W.episode_messages(ep, keep_reasoning=True)
    assert msgs[-1] == {"role": "assistant", "content": "ANSWER: J. R. R. Tolkien"}


@pytest.mark.skipif(not LFM8B.exists(), reason="LFM2.5-8B-A1B tokenizer not on disk")
def test_lfm8b_stock_template_renders_think_blocks_prefix_consistently():
    from transformers import AutoTokenizer

    from bgkit2.agent.web import web_tool_specs
    from bgkit2.data.tasks import ChatRenderer

    tok = AutoTokenizer.from_pretrained(str(LFM8B))
    rd = ChatRenderer.from_tokenizer(tok)
    r = _rows(_r1_episode(), keep_reasoning=True)[1]
    prefix = json.loads(r["instruction"])
    ids, mask = rd.render_chunk(prefix, json.loads(r["target"]), tools=web_tool_specs())
    text = tok.decode(ids)
    assert "<think>" in text and "It is Tolkien." in text and "</think>" in text
    assert text.index("</think>") < text.index("ANSWER: J. R. R. Tolkien")
    assert bool(mask.all())
    # the history's reasoning renders too (a turn after the last user message)
    pre = tok.decode(rd._apply(prefix, web_tool_specs()))
    assert "I should search." in pre and "<think>" in pre
