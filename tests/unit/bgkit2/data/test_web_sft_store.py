"""scripts/bgkit2_build_web_sft_store.py: every source's calls -> web_search / web_open / web_find,
results -> slots with the harness prompt, the eval's system / instruction framing, ANSWER turn."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from bgkit2.agent.tools import ToolRegistry
from bgkit2.agent.web import Page as WebPage
from bgkit2.agent.web import WebCorpus, snippet, web_tool_specs
from bgkit2.agent.web_eval import WEB_INSTRUCTION, WEB_SYSTEM
from bgkit2.data.templates import SPLICE_SENTINEL

_p = Path(__file__).resolve().parents[4] / "scripts" / "bgkit2_build_web_sft_store.py"
_spec = importlib.util.spec_from_file_location("web_sft", _p)
W = importlib.util.module_from_spec(_spec)
sys.modules["web_sft"] = W
_spec.loader.exec_module(W)


def _head(msgs: list[dict], q: str) -> None:
    assert msgs[0] == {"role": "system", "content": WEB_SYSTEM}
    assert msgs[1] == {"role": "user", "content": f"Question: {q}\n\n{WEB_INSTRUCTION}"}


def _call(m: dict) -> tuple[str, dict]:
    (c,) = m["tool_calls"]
    return c["function"]["name"], c["function"]["arguments"]


def _rows(ep, **kw):
    from bgkit2.agent.context import PlainRenderer

    opts = dict(budget=10_000, max_rows=0, base_meta={})
    opts.update(kw)
    return W.replay_rows(ep, "web_x", PlainRenderer(), **opts)


def _check_row(r: dict) -> None:
    ctx = json.loads(r["context"]) if r["context"] else []
    prm = json.loads(r["prompt"]) if r["prompt"] else []
    assert r["instruction"].count(SPLICE_SENTINEL) == len(ctx) == len(prm)
    assert r["framing"] == ("tool_slots" if ctx else "chat")
    (t,) = json.loads(r["target"])
    assert t["role"] == "assistant"


# ---------------------------------------------------------------------------


def test_search_hits_are_100_word_passages(tmp_path: Path):
    words = (
        " ".join(f"w{i}" for i in range(400))
        + " eiffel tower completed 1889 "
        + " ".join(f"x{i}" for i in range(400))
    )
    reg = ToolRegistry(str(tmp_path))
    reg.attach_web(WebCorpus([WebPage("p1", "Eiffel", "u", words)]))
    body = reg.call("web_search", {"query": "eiffel tower completed"}).text.split("\n", 1)[1]
    assert len(body.split()) == 100 and "1889" in body
    reg.attach_web(WebCorpus([WebPage("p1", "Eiffel", "u", words)]), snippet_words=40)
    body = reg.call("web_search", {"query": "eiffel tower completed"}).text.split("\n", 1)[1]
    assert len(body.split()) == 40 and "1889" in body


def test_search_r1_layout_and_answer_turn():
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
                "The Hobbit The Hobbit is a novel by J. R. R. Tolkien.\n\n"
                '[Document 2] (ID: 7, Score: 0.8)\n"Tolkien"\nHe wrote it.'
                "</information>",
            },
            {
                "role": "assistant",
                "content": "<thinking>It is Tolkien.</thinking>\n<answer>J. R. R. Tolkien</answer>",
            },
        ]
    }
    ep = W.parse_search_r1(rec, "t1")
    msgs, slots = W.episode_messages(ep)
    _head(msgs, q)
    assert msgs[2]["content"] == "I should search."
    assert _call(msgs[2]) == ("web_search", {"query": "author of the hobbit", "focus": q})
    assert msgs[3] == {"role": "tool", "content": SPLICE_SENTINEL}
    text, prompt = slots[3]
    assert (
        text == "[42] The Hobbit\nThe Hobbit is a novel by J. R. R. Tolkien.\n\n"
        "[7] Tolkien\nHe wrote it."
    )
    assert prompt == f"web search author of the hobbit: {q}"
    assert msgs[4] == {"role": "assistant", "content": "It is Tolkien.\n\nANSWER: J. R. R. Tolkien"}
    rows = _rows(ep)
    assert [r["framing"] for r in rows] == ["chat", "tool_slots"]
    for r in rows:
        _check_row(r)
    # the no-slot row carries the tool list inside its system message
    assert json.loads(rows[0]["instruction"])[0]["content"] == W.system_with_tools(web_tool_specs())
    assert json.loads(rows[1]["context"]) == [text]
    assert W.answer_matches("Tolkien, J. R. R. Tolkien", ["J. R. R. Tolkien"]) == "gold_in_pred"


def test_sds_hits_use_the_harness_snippet():
    page = {
        "id": 1,
        "title": "Winnie",
        "url": "https://w/pooh",
        "snippet": "s",
        "context": "Winnie-the-Pooh is a 1926 book illustrated by E. H. Shepard.",
    }
    rec = {
        "id": 2,
        "question": "who illustrated winnie the pooh",
        "answer": "E. H. Shepard",
        "output": [
            {"gen": "Let me search. <|begin_search_query|>pooh illustrator<|end_search_query|>"},
            {"search_query": "pooh illustrator"},
            {"doc": "**Web Page 1:**\n" + json.dumps(page, indent=2)},
            {"doc_gen": "summary the slot replaces"},
            {"gen": "It is Shepard.\n</think>\n\n\\boxed{E. H. Shepard}"},
        ],
    }
    ep = W.parse_sds(rec, "s")
    msgs, slots = W.episode_messages(ep)
    _head(msgs, rec["question"])
    assert msgs[2]["content"] == "Let me search."
    assert slots[3][0] == f"[https://w/pooh] Winnie\n{snippet(page['context'], 'pooh illustrator')}"
    assert msgs[-1]["content"] == "It is Shepard.\n\nANSWER: E. H. Shepard"
    assert ep.gold == ["E. H. Shepard"] and W.answer_matches(ep.pred, ep.gold) == "em"


def _h(role, text, channel=None, recipient=None):
    return {"role": role, "channel": channel, "recipient": recipient, "content": [{"text": text}]}


OR_SEARCH = (
    "[0] eiffel completed (web-search://ts=1)\n**viewing lines [0 - 6] of 6**\n\nL0: \n"
    "L1: URL: web-search://ts=1\nL2: # Search Results\nL3: \n"
    "L4:   * 【0†Doc 55†en.wikipedia.org】 Eiffel Tower The tower was completed in\n"
    "L5:  1889 for the World's Fair.\n"
    "L6:   * 【1†Doc 9†example.com】 Another page."
)
OR_OPEN = (
    "[1] Doc 55 (https://en.wikipedia.org/wiki/Eiffel_Tower)\n**viewing lines [0 - 3] of 90**\n\n"
    "L0: \nL1: URL: https://en.wikipedia.org/wiki/Eiffel_Tower\nL2: The Eiffel Tower is a tower.\n"
    "L3: See 【4†Paris†en.wikipedia.org】 for more."
)
OR_SCROLL = (
    "[2] Doc 55 (https://en.wikipedia.org/wiki/Eiffel_Tower)\n"
    "**viewing lines [40 - 40] of 90**\n\nL40: It opened in 1889."
)
OR_FIND = (
    "[3] Find results for text: `1889` in `Doc 55` (u/find?pattern=1889)\n"
    "**viewing lines [0 - 1] of 1**\n\nL0: # 【0†match at L40】\nL1: It opened in 1889."
)


def test_openresearcher_browser_calls_scroll_merge_and_find():
    q = "When was the Eiffel Tower completed?"
    rec = {
        "qid": 1,
        "question": q,
        "answer": "\\boxed{1889}",
        "messages": [
            _h("system", "You are ChatGPT."),
            _h("developer", "deep research agent"),
            _h("user", f"Question: {q}"),
            _h("assistant", "Search first 【9†L1】.", "analysis"),
            _h(
                "assistant",
                json.dumps({"query": "eiffel completed", "topn": 10}),
                "analysis",
                "browser.search",
            ),
            _h("tool", OR_SEARCH, "analysis", "assistant"),
            _h("assistant", json.dumps({"cursor": 0, "id": 0}), "analysis", "browser.open"),
            _h("tool", OR_OPEN, "analysis", "assistant"),
            _h("assistant", "Scroll down.", "analysis"),
            _h("assistant", json.dumps({"cursor": 1, "loc": 40}), "analysis", "browser.open"),
            _h("tool", OR_SCROLL, "analysis", "assistant"),
            _h(
                "assistant",
                json.dumps({"cursor": 1, "pattern": "1889"}),
                "analysis",
                "browser.find",
            ),
            _h("tool", OR_FIND, "analysis", "assistant"),
            _h("assistant", "Done.", "analysis"),
            _h(
                "assistant",
                "Explanation: ...\n\n**Exact Answer:** 1889\n\nConfidence: 90%",
                "final",
            ),
        ],
    }
    ep = W.parse_openresearcher(rec, "or1")
    msgs, slots = W.episode_messages(ep)
    _head(msgs, q)
    assert msgs[2]["content"] == "Search first ."
    assert _call(msgs[2])[0] == "web_search"
    assert slots[3][0] == (
        "[55] en.wikipedia.org\nEiffel Tower The tower was completed in 1889 for "
        "the World's Fair.\n\n[9] example.com\nAnother page."
    )
    assert _call(msgs[4]) == ("web_open", {"id": "55", "focus": q})
    page, prompt = slots[5]
    # the scroll is merged into the ONE page slot (our web_open returns the whole page)
    assert page == (
        "# Eiffel Tower\nhttps://en.wikipedia.org/wiki/Eiffel_Tower\n\n"
        "The Eiffel Tower is a tower.\nSee Paris for more.\n...\nIt opened in 1889."
    )
    assert prompt == f"page Eiffel Tower: {q}"
    # the scroll turn's words move onto the next call: web_find, now a slot over the WHOLE page
    # (the page opened last: no id), prompted as the harness's encoded web_find
    assert msgs[6]["content"] == "Scroll down." and _call(msgs[6]) == (
        "web_find",
        {"pattern": "1889", "focus": q},
    )
    assert msgs[7] == {"role": "tool", "content": SPLICE_SENTINEL}
    assert slots[7] == (page, f"find 1889 in Eiffel Tower: {q}")
    assert msgs[8]["content"] == "Done.\n\nANSWER: 1889"
    assert ep.gold == ["1889"]
    for r in _rows(ep):
        _check_row(r)


def _or_rec(q: str, calls: list[tuple[str, dict, str]]) -> dict:
    msgs = [_h("system", "You are ChatGPT."), _h("user", f"Question: {q}")]
    for rcpt, args, res in calls:
        msgs += [_h("assistant", json.dumps(args), "analysis", rcpt),
                 _h("tool", res, "analysis", "assistant")]
    msgs.append(_h("assistant", "**Exact Answer:** 1889", "final"))
    return {"qid": 2, "question": q, "answer": "1889", "messages": msgs}


def test_openresearcher_find_after_open_is_a_whole_page_slot():
    q = "When did the Eiffel Tower open?"
    other = (
        "[2] Doc 9 (https://example.com/other)\n**viewing lines [0 - 1] of 5**\n\n"
        "L0: \nL1: Another page."
    )
    # the find (cursor 1 = Doc 55) shows page lines the agent never viewed (L40-L41)
    find = (
        "[3] Find results for text: `opened` in `Doc 55` (u/find?pattern=opened)\n"
        "**viewing lines [0 - 3] of 3**\n\nL0: # 【0†match at L40】\nL1: It opened in 1889 to\n"
        "L2: the public.\nL3: "
    )
    unresolved = (
        "[4] Find results for text: `tower` in `Doc 777` (v/find?pattern=tower)\n"
        "**viewing lines [0 - 1] of 1**\n\nL0: # 【0†match at L3】\nL1: A tower line."
    )
    missing = (
        "[5] Find results for text: `zzz` in `Doc 55` (u/find?pattern=zzz)\n"
        "**viewing lines [0 - 0] of 0**\n\nL0: No `find` results for pattern: `zzz`"
    )
    rec = _or_rec(q, [
        ("browser.search", {"query": "eiffel"}, OR_SEARCH),
        ("browser.open", {"cursor": 0, "id": 0}, OR_OPEN),
        ("browser.open", {"cursor": 0, "id": 1}, other),
        ("browser.find", {"cursor": 1, "pattern": "opened"}, find),
        ("browser.find", {"cursor": 99, "pattern": "tower"}, unresolved),
        ("browser.find", {"cursor": 1, "pattern": "zzz"}, missing),
    ])
    ep = W.parse_openresearcher(rec, "or2")
    finds = [c for st in ep.steps for c in st.calls if c.tool == "web_find"]
    assert [c.note for c in finds] == ["slot", "inline_unresolved", "not_found"]
    msgs, slots = W.episode_messages(ep)
    i = next(k for k, m in enumerate(msgs) if m.get("tool_calls")
             and _call(m)[0] == "web_find")
    # Doc 9 was opened last -> the find names its page by id
    assert _call(msgs[i]) == ("web_find", {"pattern": "opened", "id": "55", "focus": q})
    text, prompt = slots[i + 1]
    assert prompt == f"find opened in Eiffel Tower: {q}"
    assert text.startswith("# Eiffel Tower\nhttps://en.wikipedia.org/wiki/Eiffel_Tower\n\n")
    assert "It opened in 1889 to\nthe public." in text  # the find's page lines, merged
    # the web_open slot of the same page is the same whole page
    j = next(k for k, m in enumerate(msgs) if m.get("tool_calls") and _call(m)[0] == "web_open")
    assert slots[j + 1][0] == text
    # unresolved -> the verbatim result stays inline; no match -> the harness's "not found"
    assert msgs[i + 3] == {"role": "tool", "content": "A tower line."} and slots[i + 3] is None
    assert msgs[i + 5]["content"] == "'zzz' not found on the page" and slots[i + 5] is None
    for r in _rows(ep):
        _check_row(r)


def test_webshaper_find_formats():
    st = W.BrowserState("When did it open?")
    st.search({"query": "eiffel"}, OR_SEARCH)
    st.open({"cursor": 0, "id": 0}, OR_OPEN)  # page display [1] = Doc 55
    # WebShaper: 'Found N match(es) for "p" in [k]:' + the PAGE's own line numbers
    c = st.find({"cursor": 1, "pattern": "opened"},
                'Found 1 match(es) for "opened" in [1]:\nL40: It opened in 1889.')
    assert c.note == "slot" and c.is_slot() and c.args == {"pattern": "opened"}
    assert c.page.lines[40] == "It opened in 1889." and "It opened in 1889." in c.result()
    assert c.prompt() == "find opened in Eiffel Tower: When did it open?"
    # a find over the SEARCH listing has no page text -> inline verbatim
    u = st.find({"cursor": 0, "pattern": "tower"},
                'Found 1 match(es) for "tower" in [0]:\nL4: Eiffel Tower The tower')
    assert u.note == "inline_unresolved" and not u.is_slot() and "Eiffel Tower" in u.result()
    n = st.find({"cursor": 1, "pattern": "zzz"}, 'No matches for "zzz" in [1].')
    assert n.note == "not_found" and not n.is_slot() and n.result() == "'zzz' not found on the page"


def test_webshaper_harmony_and_cursor_links():
    q = "Which city hosts the tower?"
    search = (
        '[1]\nL1: Searched for "tower city"\nL2: \nL3: 【1†Eiffel Tower - Wikipedia】\n'
        "L4: URL: https://en.wikipedia.org/wiki/Eiffel_Tower\nL5: The tower is in Paris."
    )
    conv = [
        {
            "role": "assistant",
            "content": "<|channel|>analysis<|message|>Search it.<|end|><|start|>"
            "assistant<|channel|>commentary to=browser.search code<|message|>{}",
            "tool_call": str(
                {"tool": "browser.search", "args": {"query": "tower city", "topn": 10}}
            ),
        },
        {"role": "tool", "tool": "browser.search", "content": search},
        {
            "role": "assistant",
            "content": "<|channel|>analysis<|message|>Open it.",
            "tool_call": str({"tool": "browser.open", "args": {"cursor": 1, "id": 1}}),
        },
        {"role": "tool", "tool": "browser.open", "content": "[2]\nL1: Paris is the city."},
        {
            "role": "assistant",
            "content": "<|channel|>analysis<|message|>It is Paris.<|end|>",
            "final_answer": "Paris \\boxed{Paris}",
        },
    ]
    rec = {
        "qid": "a",
        "traj_idx": 0,
        "question": q,
        "conversation": json.dumps(conv),
        "boxed_answer": "Paris",
        "reference_answer": "Paris",
        "correct": True,
    }
    ep = W.parse_webshaper(rec, "ws")
    msgs, slots = W.episode_messages(ep)
    _head(msgs, q)
    assert msgs[2]["content"] == "Search it."
    assert (
        slots[3][0] == "[https://en.wikipedia.org/wiki/Eiffel_Tower] Eiffel Tower - Wikipedia\n"
        "The tower is in Paris."
    )
    assert _call(msgs[4]) == (
        "web_open",
        {"id": "https://en.wikipedia.org/wiki/Eiffel_Tower", "focus": q},
    )
    assert slots[5] == (
        "# Eiffel Tower - Wikipedia\nhttps://en.wikipedia.org/wiki/Eiffel_Tower\n\n"
        "Paris is the city.",
        f"page Eiffel Tower - Wikipedia: {q}",
    )
    assert msgs[6]["content"] == "It is Paris.\n\nANSWER: Paris"
    assert ep.judged_correct is True


def test_openseeker_batched_search_and_visit_goal():
    q = "Who founded the fort?"
    google = (
        "### A Google search for 'fort founder' found 1 results:\n\n"
        "1. [Fort X](https://f/x)\nDate published: 2020\n\nFounded by Wayne.\n\n---\n"
        "### A Google search for 'fort x history' found 0 results:"
    )
    visit = (
        "The useful information in https://f/x for user goal founder as follows: \n\n"
        "Evidence in page: \nWayne built it.\n\nSummary: \nWayne."
    )
    rec = {
        "question": q,
        "answer": "Anthony Wayne",
        "source": "openseeker",
        "messages": [
            {"role": "system", "content": "sys", "reasoning_content": None, "tool_calls": None},
            {"role": "user", "content": q, "reasoning_content": None, "tool_calls": None},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Search two ways.",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"query": ["fort founder", "fort x history"]}),
                        },
                    }
                ],
            },
            {"role": "tool", "content": google, "tool_call_id": "c1"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Visit it.",
                "tool_calls": [
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {
                            "name": "visit",
                            "arguments": json.dumps({"url": "https://f/x", "goal": "founder"}),
                        },
                    }
                ],
            },
            {"role": "tool", "content": visit, "tool_call_id": "c2"},
            {
                "role": "assistant",
                "content": "<answer>General Anthony Wayne</answer>",
                "reasoning_content": "Wayne.",
                "tool_calls": None,
            },
        ],
    }
    ep = W.parse_mix(rec, "os")
    msgs, slots = W.episode_messages(ep)
    _head(msgs, q)
    calls = msgs[2]["tool_calls"]
    assert [c["function"]["arguments"]["query"] for c in calls] == [
        "fort founder",
        "fort x history",
    ]
    assert slots[3][0] == "[https://f/x] Fort X\nFounded by Wayne."
    # a search with no hits is a (compressible) result, as the harness returns it
    assert msgs[4] == {"role": "tool", "content": SPLICE_SENTINEL}
    assert slots[4][0] == "no results for 'fort x history'"
    assert _call(msgs[5]) == ("web_open", {"id": "https://f/x", "focus": "founder"})
    assert slots[6] == (f"# x\nhttps://f/x\n\n{visit}", "page x: founder")
    assert msgs[7]["content"] == "Wayne.\n\nANSWER: General Anthony Wayne"
    assert W.answer_matches(ep.pred, ep.gold, lenient=True) == "gold_in_pred"


def test_miroverse_serper_json_and_echoed_goal():
    q = "Where is Tombras headquartered?"
    serper = {
        "searchParameters": {"q": "tombras hq"},
        "answerBox": {"title": "HQ", "answer": "Knoxville"},
        "organic": [{"title": "Tombras", "link": "https://t/", "snippet": "Knoxville, TN."}],
    }
    page = {"text": "Offices: Knoxville.", "metadata": {"title": "Contact - Tombras"}}
    thought = "Let me verify on the official website."
    rec = {
        "question": q,
        "answer": "Knoxville",
        "source": "MiroVerse-MuSiQue",
        "messages": [
            {"role": "user", "content": q},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Search.",
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"query": ["tombras hq"]}),
                        },
                    }
                ],
            },
            {"role": "tool", "content": json.dumps(serper), "tool_call_id": "a"},
            {
                "role": "assistant",
                "content": thought,
                "reasoning_content": None,
                "tool_calls": [
                    {
                        "id": "b",
                        "type": "function",
                        "function": {
                            "name": "visit",
                            "arguments": json.dumps({"url": "https://t/contact", "goal": thought}),
                        },
                    }
                ],
            },
            {"role": "tool", "content": json.dumps(page), "tool_call_id": "b"},
            {
                "role": "assistant",
                "content": "<answer>Knoxville</answer>",
                "reasoning_content": None,
                "tool_calls": None,
            },
        ],
    }
    ep = W.parse_mix(rec, "mv")
    msgs, slots = W.episode_messages(ep)
    assert (
        slots[3][0] == "[google-answer-box] HQ\nKnoxville\n\n[https://t/] Tombras\nKnoxville, TN."
    )
    # the goal repeats the turn's narration -> the question is the focus
    assert _call(msgs[4]) == ("web_open", {"id": "https://t/contact", "focus": q})
    assert slots[5][0] == "# Contact - Tombras\nhttps://t/contact\n\nOffices: Knoxville."
    assert msgs[-1]["content"] == "ANSWER: Knoxville"


def _long_episode(n: int = 8):
    steps = [
        W.Step(
            f"t{i}",
            [
                W.Call(
                    "web_search", {"query": f"q{i}"}, "f", text=f"[{i}] T\n" + f"passage {i} " * 60
                )
            ],
        )
        for i in range(n)
    ]
    return W.Episode("traj", "question?", "A", steps, "done")


def test_replay_compacts_the_oldest_rounds_like_the_harness():
    """The replay IS the harness's Conversation: under a small budget the oldest rounds fold
    verbatim into the one compaction slot (after the task message), the recent rounds stay
    their own slots, every turn is a row and the final row is the ANSWER turn."""
    from bgkit2.agent.context import COMPACTION_HEADER, PlainRenderer

    ep = _long_episode()
    rows = _rows(ep, budget=900)
    assert len(rows) == 9 and json.loads(rows[-1]["target"])[0]["content"] == "done\n\nANSWER: A"
    last = rows[-1]
    _check_row(last)
    assert last["meta"]["compacted"] and last["meta"]["n_compactions"] >= 1
    ins = json.loads(last["instruction"])
    assert ins[2] == {"role": "user", "content": f"{COMPACTION_HEADER}\n{SPLICE_SENTINEL}"}
    ctx, prm = json.loads(last["context"]), json.loads(last["prompt"])
    assert prm[0] == "compaction: question?"
    # the compaction slot holds the oldest rounds verbatim, in order
    folded = [i for i in range(8) if f"passage {i} " in ctx[0]]
    assert folded == list(range(len(folded))) and folded
    n_recent = 8 - len(folded)
    assert [c.split("\n")[0] for c in ctx[1:]] == [f"[{i}] T" for i in range(8 - n_recent, 8)]
    assert "assistant: t0" in ctx[0] and "tool: [0] T" in ctx[0]
    assert last["meta"]["prefix_tokens_compressed"] <= 900
    # recent slots keep the harness prompt; nothing compacts under a large budget
    assert prm[-1] == "web search q7: f"
    assert not any(r["meta"]["compacted"] for r in _rows(ep, budget=100_000))
    # depth-stratified sampling keeps the final turn
    assert json.loads(_rows(ep, max_rows=3)[-1]["target"])[0]["content"].endswith("ANSWER: A")
    assert len(_rows(ep, max_rows=3)) == 3
    assert PlainRenderer  # the replay takes any harness Renderer


def test_benchmark_excluder_and_lenient_match():
    ex = W.Excluder({W.norm_q("Who designed the Eiffel Tower in Paris?"): "hotpot"})
    assert ex.hit("who designed the eiffel tower in paris") == "hotpot"
    assert ex.hit("Q: Who designed the Eiffel Tower in Paris? Answer briefly.") == "hotpot"
    assert ex.hit("designed the Eiffel Tower in Paris") == "hotpot"  # contained in a benchmark q
    assert ex.hit("Who built the Tower Bridge in London?") is None
    gold = ['Neil Young performed the song "This Note\'s for You."']
    assert W.answer_matches("Neil Young", gold) is None
    assert W.answer_matches("Neil\u202fYoung", gold, lenient=True) == "pred_in_gold"


def test_fast_renderer_counts_match_the_trainer_renderer():
    from transformers import AutoTokenizer

    from bgkit2.data.tasks import ChatRenderer

    cache = Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots"
    snaps = sorted(cache.glob("*/chat_template.jinja")) if cache.exists() else []
    if not snaps:
        pytest.skip("LFM2.5-350M tokenizer not cached")
    tok = AutoTokenizer.from_pretrained(str(snaps[0].parent))
    ep = _long_episode()
    ep.steps.append(W.Step("", [W.Call("web_find", {"pattern": "y"}, text="x y")]))
    fast = W.FastRenderer(tok)
    rows = W.replay_rows(ep, "web_x", fast, budget=900, base_meta={})
    rend = ChatRenderer.from_tokenizer(tok)
    assert any(r["meta"]["compacted"] for r in rows)
    for r in rows:
        if r["framing"] == "chat":
            n = int(rend.render("chat", r["instruction"])[0].shape[0])
        else:
            n = sum(int(s.shape[0]) for s in rend.render_slots(r["instruction"], web_tool_specs()))
        assert n == r["meta"]["prefix_text_tokens"]


if __name__ == "__main__":
    pytest.main([__file__])


def test_source_artefact_filters():
    q = (
        "What is the highest point in the state where WJLD is located? You should follow the "
        "format instruction in the requestion strictly and wrap the final answer in \\boxed{}."
    )
    assert W.clean_question(q) == "What is the highest point in the state where WJLD is located?"
    q2 = (
        "Identify the paternal grandfather of Wladyslaw IV Vasa, following the strict format: "
        "wrap the final answer in \\boxed{}."
    )
    assert W.clean_question(q2) == "Identify the paternal grandfather of Wladyslaw IV Vasa."
    keep = "The final answer is not the name of the feature itself. What is it?"
    assert W.clean_question(keep) == keep
    assert W.pure_math("Solve \\( x + 3/4 = 5/9 \\).")
    assert not W.pure_math("Find the top goal-scorers in the 2007-08 season.")
    assert not W.pure_math("What is 2 + 2 in the Roman numerals used by Caesar?")
    hunt = W.Step("Good. We need a citation for the founding year.", [])
    ok = W.Step("The article was cited by later works.", [])
    ep = W.Episode("t", "q?", "A", [ok, hunt], "Answer now. We can cite the page.\nDone.")
    assert W.citation_hunting(ep) == "need a citation"
    assert W.citation_hunting(W.Episode("t", "q?", "A", [ok], "")) is None
    assert W.strip_citation_sentences(ep.final_thought) == "Answer now.\nDone."


def test_cons_rows_load_as_spine_samples_and_collate(tmp_path):
    """A cons-compacted row -> TaskDataset puts the spine's leaves in
    Sample.spine_slot (nothing encoded as text for that slot); the collator and
    Batch.subset carry it; the recent rounds stay ordinary text slots."""
    from transformers import AutoTokenizer

    from bgkit2.agent.context import PlainRenderer
    from bgkit2.data.autoencode import Collator
    from bgkit2.data.tasks import ChatRenderer, TaskDataset, write_task_store

    cache = Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots"
    snaps = sorted(cache.glob("*/chat_template.jinja")) if cache.exists() else []
    if not snaps:
        pytest.skip("LFM2.5-350M tokenizer not cached")
    tok = AutoTokenizer.from_pretrained(str(snaps[0].parent))
    ep = _long_episode()
    rows = W.replay_rows(ep, "web_x", PlainRenderer(), budget=900, base_meta={},
                         compaction_mode="cons")
    last = rows[-1]
    assert last["meta"]["compacted"] and last["meta"]["n_compactions"] >= 2
    for r in rows:
        r["split"] = "train"
    store = tmp_path / "s.parquet"
    write_task_store(str(store), rows)
    ds = TaskDataset(str(store), ChatRenderer.from_tokenizer(tok), max_prefix_tokens=None,
                     max_ctx_tokens=4096)
    idx = next(i for i in range(ds.n) if ds.row(i)["target"] == last["target"])
    s = ds[idx]
    from bgkit2.agent.context import compaction_leaves, compaction_payload

    leaves = compaction_leaves(compaction_payload(json.loads(last["context"])[0]))
    assert s.spine_slot is not None and len(s.spine_slot) == len(leaves) >= 2
    assert s.ctx_ids.shape[0] == 0 and s.extra_spine is None and len(s.extra_ctx) >= 1
    assert tok.decode(s.spine_slot[0]).strip() == leaves[0].strip()
    from bgkit2.data.templates import Templates

    other = ds[(idx + 1) % ds.n]
    batch = Collator(Templates.from_tokenizer(tok, style="chat"))([other, s])
    assert batch.spine_slot is not None and batch.spine_slot[1] is s.spine_slot
    sub = batch.subset([1])
    assert sub.spine_slot == [s.spine_slot] and sub.extra_input_ids is not None
