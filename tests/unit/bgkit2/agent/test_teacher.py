"""Teacher-side pieces: K2 output parsing, the digest flow through
run_episode (teacher reads digests, student slots keep the raw text), and
the OpenAI-compatible model wrapper with a stubbed transport."""

from __future__ import annotations

import json

from bgkit2.agent.context import PlainRenderer
from bgkit2.agent.runner import AgentTask, ChatAPIModel, run_episode
from bgkit2.agent.teacher import TeacherEngine, digest_prompt, parse_k2_output
from bgkit2.agent.tools import ToolRegistry


def test_parse_k2_output_think_content_and_json_calls():
    raw = (
        "<think>\nlook at the file first\n</think>Let me read it.\n"
        '<ifm|tool_calls>\n<ifm|tool_call>{"name": "read", "arguments": {"path": "a.py", '
        '"focus": "the bug"}}</ifm|tool_call>\n</ifm|tool_calls>'
    )
    p = parse_k2_output(raw)
    assert p.reasoning == "look at the file first"
    assert p.content == "Let me read it."
    assert p.parse_error is None
    assert len(p.tool_calls) == 1
    f = p.tool_calls[0]["function"]
    assert f["name"] == "read"
    assert json.loads(f["arguments"]) == {"path": "a.py", "focus": "the bug"}
    # plain answer, no calls, no thinking
    q = parse_k2_output("The answer is 42.")
    assert q.content == "The answer is 42." and not q.tool_calls and q.reasoning == ""
    # malformed call is reported, not raised
    r = parse_k2_output("<ifm|tool_calls>\n<ifm|tool_call>{oops</ifm|tool_call>\n</ifm|tool_calls>")
    assert r.parse_error and not r.tool_calls


def test_digest_flow_teacher_reads_digest_student_keeps_raw(repo):
    tools = ToolRegistry(repo)
    renderer = PlainRenderer()
    seen_prompts: list[str] = []

    def digest_fn(prompt: str, text: str) -> str:
        seen_prompts.append(prompt)
        return f"DIGEST({len(text)} chars)"

    scripted = iter(
        [
            {
                "content": "reading",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read",
                            "arguments": {"path": "pkg/calc.py", "focus": "add"},
                        }
                    }
                ],
                "reasoning_content": "plan",
            },
            {"content": "done", "tool_calls": [], "reasoning_content": ""},
        ]
    )
    posted: list[dict] = []

    def fake_post(url, body, timeout):
        posted.append(body)
        msg = next(scripted)
        return {"choices": [{"message": msg}], "usage": {"completion_tokens": 3}}

    model = ChatAPIModel("http://x", renderer, post=fake_post)
    task = AgentTask(task_id="t", repo_dir=str(repo), system_prompt="sys", instruction="fix it")
    res = run_episode(model, tools, task, max_turns=4, digest_fn=digest_fn)
    assert res.stop_reason == "final" and res.turns == 2
    assert seen_prompts and "read" in seen_prompts[0]
    # second request carried the DIGEST as the tool message and the reasoning back
    msgs = posted[1]["messages"]
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert tool_msgs and tool_msgs[0]["content"].startswith("DIGEST(")
    asst = next(m for m in msgs if m["role"] == "assistant")
    assert asst.get("reasoning_content") == "plan"


def test_to_task_row_keeps_raw_context_and_digest(repo):
    from bgkit2.agent.context import Conversation, RatioPolicy

    conv = Conversation("sys", [], PlainRenderer(), RatioPolicy(), "task")
    conv.teacher_view = True
    conv.add_user("go")
    conv.add_assistant("reading", [], reasoning="r")
    conv.add_tool_result("RAW TEXT " * 50, True, "read: a.py", "read", digest="short digest")
    row = conv.to_task_row(target="ok")
    assert json.loads(row["context"])[0].startswith("RAW TEXT")
    meta = json.loads(row["meta"]) if isinstance(row["meta"], str) else row["meta"]
    assert meta["digests"] == ["short digest"]
    # the student view never shows the digest
    assert all("short digest" not in json.dumps(m) for m in conv.messages_with_slots())
    assert any(m.get("content") == "short digest" for m in conv.messages_full())


def test_teacher_engine_renders_and_parses_with_a_fake_model():
    class Tok:
        eos_token = "<|im_end|>"

        def apply_chat_template(
            self,
            msgs,
            tools=None,
            add_generation_prompt=True,
            return_tensors=None,
            return_dict=False,
            **kw,
        ):
            import torch

            assert msgs[-1]["role"] == "user"
            assert all("reasoning_content" in m for m in msgs if m["role"] == "assistant")
            n = 5
            return {
                "input_ids": torch.zeros(1, n, dtype=torch.long),
                "attention_mask": torch.ones(1, n, dtype=torch.long),
            }

        def decode(self, ids, skip_special_tokens=False):
            return (
                'Hi<ifm|tool_calls>\n<ifm|tool_call>{"name": "search", "arguments": {"q": "x"}}'
                "</ifm|tool_call>\n</ifm|tool_calls><|im_end|>"
            )

    class Model:
        def generate(self, input_ids, attention_mask, **kw):
            import torch

            return torch.cat([input_ids, torch.ones(1, 7, dtype=torch.long)], dim=1)

    eng = TeacherEngine(Model(), Tok(), device="cpu")
    out = eng.chat(
        [
            {
                "role": "assistant",
                "content": "prev",
                "tool_calls": [{"function": {"name": "a", "arguments": "{}"}}],
            },
            {"role": "user", "content": "hi"},
        ],
        tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}],
    )
    msg = out["choices"][0]["message"]
    assert msg["content"] == "Hi" and msg["tool_calls"][0]["function"]["name"] == "search"
    assert out["usage"]["completion_tokens"] == 7
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    dp = digest_prompt("read: a.py", "x" * 10)
    assert dp[0]["role"] == "system" and "read: a.py" in dp[1]["content"]


def test_parse_k2_output_ifm_think_tags_and_stray_closing_tag():
    p = parse_k2_output("<ifm|think_faster>\nquick\n</ifm|think_faster>Answer.")
    assert p.reasoning == "quick" and p.content == "Answer."
    q = parse_k2_output("</ifm|think_faster>")
    assert q.content == "" and not q.tool_calls and q.parse_error is None
    r = parse_k2_output("<ifm|think>\ndeep\n</ifm|think>Done")
    assert r.reasoning == "deep" and r.content == "Done"


def test_chat_api_model_reports_malformed_tool_call_arguments(repo):
    renderer = PlainRenderer()

    def fake_post(url, body, timeout):
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "read", "arguments": '{"path": "a.py'}}
                        ],
                    }
                }
            ],
            "usage": {"completion_tokens": 3},
        }

    from bgkit2.agent.context import Conversation, RatioPolicy

    model = ChatAPIModel("http://x", renderer, post=fake_post)
    conv = Conversation("sys", [], renderer, RatioPolicy(), "task")
    conv.add_user("go")
    out = model.generate(conv)
    assert out.parse_error and not out.tool_calls


def test_compact_can_measure_the_full_view():
    from bgkit2.agent.context import Conversation, RatioPolicy

    conv = Conversation("sys", [], PlainRenderer(), RatioPolicy(), "task")
    conv.add_user("go")
    for i in range(4):
        conv.add_assistant(f"step {i}", [])
        conv.add_tool_result("x " * 400, True, f"read: f{i}", "read")
    full_before = conv.prefix_tokens(compressed=False)
    assert conv.compact(full_before // 2, compressed=False)
    assert conv.prefix_tokens(compressed=False) < full_before
