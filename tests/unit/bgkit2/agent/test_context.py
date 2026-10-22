from __future__ import annotations

import json

from bgkit2.agent.context import (
    COMPACTION_HEADER,
    Conversation,
    PlainRenderer,
    RatioPolicy,
    TemplateRenderer,
    ToolCall,
    parse_tool_calls,
    render_tool_calls,
)
from bgkit2.data.templates import SPLICE_SENTINEL

from .conftest import real_tok

TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {}}}]


def _conv(renderer=None, system="sys"):
    c = Conversation(system, TOOLS, renderer or PlainRenderer(), RatioPolicy(), "fix add")
    c.add_user("Fix add in pkg/calc.py")
    return c


def test_render_tool_calls_matches_the_real_template():
    tok = real_tok()
    calls = [
        ToolCall("zoom", {"path": "a b.py", "start": 3, "end": 10}),
        ToolCall("bash", {"cmd": "pytest -x 'tests/x.py'\necho done", "flag": True}),
    ]
    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "Let me look.",
            "tool_calls": [c.as_message() for c in calls],
        },
    ]
    real = tok.apply_chat_template(msgs, tokenize=False)
    assert "Let me look." + render_tool_calls(calls) + "<|im_end|>" in real


def test_parse_tool_calls_roundtrip_and_errors():
    calls = [
        ToolCall("search", {"query": "def foo", "focus": "it's here\n"}),
        ToolCall("zoom", {"path": "a.py", "start": 3, "end": 10}),
    ]
    raw = "Looking." + render_tool_calls(calls) + "<|im_end|>"
    content, parsed, err = parse_tool_calls(raw)
    assert err is None and content == "Looking." and parsed == calls
    # cut-off generation (no end token) still parses
    assert parse_tool_calls("<|tool_call_start|>[read(path='x')]")[1] == [
        ToolCall("read", {"path": "x"})
    ]
    # plain answer: no calls
    assert parse_tool_calls("Done.<|im_end|>") == ("Done.", [], None)
    # broken syntax and positional args are reported, not raised
    _, none, e1 = parse_tool_calls("<|tool_call_start|>[read(path=]<|tool_call_end|>")
    _, none2, e2 = parse_tool_calls("<|tool_call_start|>[read('x')]<|tool_call_end|>")
    assert none == [] and none2 == [] and e1 and "positional" in e2


def test_one_sentinel_per_compressible_result_and_reassembly():
    c = _conv()
    c.add_assistant("", [ToolCall("read", {"path": "a.py"})])
    s1 = c.add_tool_result("def add(a, b):\n    return a - b\n" * 4, True, "read a.py", "read")
    c.add_assistant("", [ToolCall("zoom", {"path": "a.py", "start": 1, "end": 2})])
    z = c.add_tool_result("1: def add(a, b):", False, "", "zoom")
    c.add_assistant("", [ToolCall("search", {"query": "add"})])
    s2 = c.add_tool_result("a.py:1: def add(a, b):", True, "search add", "search")
    assert z is None and [s.tool for s in c.slots] == ["read", "search"]
    slotted = c.render_slots()
    assert slotted.count(SPLICE_SENTINEL) == 2
    assert c.contexts() == [s1.text, s2.text] and c.prompts() == ["read a.py", "search add"]
    # replacing each sentinel by its slot text gives the plain rendering
    back = slotted
    for t in c.contexts():
        back = back.replace(SPLICE_SENTINEL, t, 1)
    assert back == c.render_full()
    # accounting: compressed prefix = full minus (tokens - reps) per slot
    assert s1.n_reps == -(-s1.n_tokens // 8) and s2.n_reps == 1
    saved = sum(s.n_tokens - s.n_reps for s in c.slots)
    gap = c.prefix_tokens(compressed=False) - c.prefix_tokens(compressed=True)
    assert abs(gap - saved) <= len(c.slots)
    row = c.to_task_row(target=[{"role": "assistant", "content": "fixed"}])
    assert row["framing"] == "tool_slots" and json.loads(row["context"]) == c.contexts()
    assert json.loads(row["prompt"]) == c.prompts()
    msgs = json.loads(row["instruction"])
    assert sum(SPLICE_SENTINEL in m["content"] for m in msgs) == 2
    assert msgs[2]["role"] == "assistant" and msgs[2]["tool_calls"][0]["function"]["name"] == "read"
    assert row["meta"]["n_slots"] == 2 and _conv().to_task_row()["framing"] == "chat"


def test_slots_render_through_the_real_template():
    tok = real_tok()
    from bgkit2.data.tasks import ChatRenderer

    c = _conv(TemplateRenderer(tok))
    c.add_assistant("", [ToolCall("read", {"path": "a.py", "focus": "add"})])
    c.add_tool_result("def add(a, b):\n    return a - b\n", True, "read a.py: add", "read")
    c.add_assistant("", [ToolCall("search", {"query": "add"})])
    c.add_tool_result("a.py:1: def add(a, b):", True, "search add", "search")
    row = c.to_task_row()
    segs = ChatRenderer.from_tokenizer(tok).render_slots(row["instruction"])
    assert len(segs) == 3  # prefix, one middle, the tail
    # the ChatRenderer path (no tools= kwarg) and ours agree once tools are dropped
    c_notools = Conversation("sys", [], TemplateRenderer(tok))
    assert c.prefix_tokens(True) > c_notools.prefix_tokens(True)


def test_compaction_folds_oldest_rounds_into_one_slot():
    c = _conv()
    for i in range(5):
        c.add_assistant("", [ToolCall("read", {"path": f"f{i}.py"})])
        c.add_tool_result(f"# file {i}\n" + f"line {i}\n" * 40, True, f"read f{i}.py", "read")
    n_full = c.prefix_tokens(compressed=True)
    assert c.compact(n_full + 1) is False  # fits: nothing folded
    assert c.compact(n_full - 1) is True  # one round over: fold exactly the oldest
    msgs = c.messages_with_slots()
    assert msgs[0]["role"] == "system" and msgs[1]["content"].startswith("Fix add")
    assert (
        msgs[2]["role"] == "user"
        and msgs[2]["content"] == f"{COMPACTION_HEADER}\n{SPLICE_SENTINEL}"
    )
    comp = c.slots[0]
    assert comp.tool == "compaction" and comp.prompt == "compaction: fix add"
    assert "# file 0" in comp.text and "read(path='f0.py')" in comp.text
    assert "# file 1" not in comp.text and len(c.slots) == 5 and c.n_compactions == 1
    assert comp.n_reps == -(-comp.n_tokens // 32)
    # a second compaction extends the SAME slot and always keeps the newest round
    assert c.compact(1) is True
    assert len([s for s in c.slots if s.tool == "compaction"]) == 1 and c.n_compactions == 2
    comp = c.slots[0]
    assert all(f"# file {i}" in comp.text for i in range(4)) and "# file 4" not in comp.text
    assert len(c.slots) == 2 and c.slots[-1].text.startswith("# file 4")
    assert c.compact(1) is False  # nothing left to fold
    # contexts / prompts still line up with the sentinels
    assert c.render_slots().count(SPLICE_SENTINEL) == len(c.slots) == 2


def test_always_on_repo_overview_shares_the_opening_turn_and_survives_compaction():
    """The overview slot: the student's context NAMES the repo (its cached
    root reps fill the slot), the teacher / full-text view reads the path
    listing in the same place, the task instruction follows in the same user
    turn, rows carry meta.repo, and compaction never folds it."""
    from bgkit2.agent.context import OVERVIEW_HEADER, OVERVIEW_TOOL, overview_marker, overview_repo

    c = Conversation("sys", TOOLS, PlainRenderer(), RatioPolicy(), "find it")
    listing = "pkg/a.py\npkg/b.py\ntests/test_a.py"
    slot = c.add_user_with_overview("Which file defines `foo`?", "own/repo", listing, 319)
    assert slot.tool == OVERVIEW_TOOL and slot.n_reps == 319
    assert overview_repo(c.contexts()[0]) == "own/repo" == overview_repo(overview_marker("own/repo"))
    assert overview_repo("# a file\nprint(1)") is None
    # student view: header + ONE sentinel + the task, in the opening user turn
    m = c.messages_with_slots()[1]
    assert m["role"] == "user"
    assert m["content"] == f"{OVERVIEW_HEADER}{SPLICE_SENTINEL}\n\nWhich file defines `foo`?"
    # teacher / full-text view: the listing in the same position
    c.teacher_view = True
    full = c.messages_full()[1]["content"]
    assert full == f"{OVERVIEW_HEADER}{listing}\n\nWhich file defines `foo`?"
    # a first-move row is now a SLOTTED row and names its repo
    row = c.to_task_row(target=[{"role": "assistant", "content": "x"}])
    assert row["framing"] == "tool_slots" and row["meta"]["repo"] == "own/repo"
    assert json.loads(row["context"]) == [overview_marker("own/repo", 319)]
    assert row["instruction"].count(SPLICE_SENTINEL) == 1
    # compaction folds tool rounds, never the overview (it sits in the head)
    for i in range(5):
        c.add_assistant("", [ToolCall("read", {"path": f"f{i}.py"})])
        c.add_tool_result(f"# file {i}\n" + f"line {i}\n" * 40, True, f"read f{i}.py", "read")
    assert c.compact(1) is True
    tools_in_order = [s.tool for s in c.slots]
    assert tools_in_order[0] == OVERVIEW_TOOL and tools_in_order[1] == "compaction"
    assert c.render_slots().count(SPLICE_SENTINEL) == len(c.slots) == len(c.contexts())
    assert c.prefix_tokens(compressed=True) >= 319


def _five_reads(mode: str) -> Conversation:
    c = Conversation("sys", TOOLS, PlainRenderer(), RatioPolicy(), "fix add", compaction_mode=mode)
    c.add_user("Fix add.")
    return c


def _round(c: Conversation, i: int) -> None:
    c.add_assistant("", [ToolCall("read", {"path": f"f{i}.py"})])
    c.add_tool_result(f"# file {i}\n" + f"line {i}\n" * 40, True, f"read f{i}.py", "read")


def test_cons_compaction_builds_a_spine_of_folded_rounds():
    """Three compaction events -> a depth-3 spine cons(cons(cons(None, r0), r1), r2) whose
    leaves are the folded rounds verbatim and in order; the recent rounds stay their own slots;
    the slot is priced at the top cell's reps."""
    from bgkit2.agent.context import compaction_leaves, compaction_payload

    c = _five_reads("cons")
    plain = {}
    for i in range(6):
        _round(c, i)
        plain[i] = "\n".join(m.plain() for m in c._msgs[-2:])
    for _ in range(3):  # each call folds exactly the oldest remaining round
        c.compact(c.prefix_tokens(compressed=True) - 1)
    assert c.n_compactions == 3
    comp = c.slots[0]
    assert comp.tool == "compaction" and comp.prompt == "compaction: fix add"
    p = compaction_payload(comp.text)
    depth, node = 0, p
    while node is not None:
        depth, node = depth + 1, node["child"]
    assert depth == 3
    assert compaction_leaves(p) == [plain[0], plain[1], plain[2]]
    assert [s.text.split("\n")[0] for s in c.slots[1:]] == ["# file 3", "# file 4", "# file 5"]
    child = compaction_payload(json.dumps(p["child"]))
    # priced as the encoder will build it: the top cell's reps of the wide-cell layout
    from bgkit2.data.spine import spine_top_reps

    counts = [PlainRenderer().count(x) for x in compaction_leaves(p)]
    assert comp.n_tokens == sum(counts) and comp.n_reps == spine_top_reps(counts)
    assert child is not None and c.render_slots().count(SPLICE_SENTINEL) == len(c.slots) == 4
    # one call that folds several rounds adds ONE node with those rounds as its leaf
    c.compact(1)
    p2 = compaction_payload(c.slots[0].text)
    assert c.n_compactions == 4 and compaction_leaves(p2)[-1] == plain[3] + "\n" + plain[4]


def test_flat_compaction_is_the_default_and_unchanged():
    a, b = (
        _five_reads("flat"),
        Conversation("sys", TOOLS, PlainRenderer(), RatioPolicy(), "fix add"),
    )
    b.add_user("Fix add.")
    folded = []
    for i in range(5):
        _round(a, i)
        _round(b, i)
    for c in (a, b):
        c.compact(c.prefix_tokens(compressed=True) - 1)
        c.compact(1)
    assert a.messages_with_slots() == b.messages_with_slots() and a.contexts() == b.contexts()
    folded = [
        f"assistant: {render_tool_calls([ToolCall('read', {'path': f'f{i}.py'})])}\n"
        f"tool: # file {i}\n" + f"line {i}\n" * 40
        for i in range(4)
    ]
    assert a.contexts()[0] == "\n".join(folded)


def test_cons_fold_above_the_leaf_cap_becomes_several_nodes():
    from bgkit2.agent.context import compaction_leaves, compaction_payload

    c = Conversation("sys", TOOLS, PlainRenderer(), RatioPolicy(), "fix add",
                     compaction_mode="cons", compaction_leaf_max_tokens=60)
    c.add_user("Fix add.")
    for i in range(4):
        _round(c, i)
    c.compact(1)  # one event folds three rounds of ~100 tokens each
    leaves = compaction_leaves(compaction_payload(c.slots[0].text))
    assert len(leaves) > 3 and all(PlainRenderer().count(x) <= 60 for x in leaves)
    assert "".join(leaves).replace("\n", "") == "".join(
        f"assistant: {render_tool_calls([ToolCall('read', {'path': f'f{i}.py'})])}"
        f"tool: # file {i}" + f"line {i}" * 40 for i in range(3)
    ).replace("\n", "")
    # the full-text view still reads the folded history as plain text
    assert "tool: # file 0" in c.messages_full()[2]["content"]


def test_over_long_result_is_split_into_consecutive_slots():
    c = Conversation("sys", TOOLS, PlainRenderer(), RatioPolicy(), "q", slot_max_tokens=50)
    c.add_user("Read it.")
    c.add_assistant("", [ToolCall("read", {"path": "big.py"})])
    text = "".join(f"line {i:03d} of the file\n" for i in range(40))
    c.add_tool_result(text, True, "read big.py", "read")
    parts = c.contexts()
    assert len(parts) > 1 and "".join(parts) == text
    assert all(PlainRenderer().count(p) <= 50 for p in parts)
    assert c.prompts() == [f"read big.py (part {i + 1}/{len(parts)})" for i in range(len(parts))]
    msgs = c.messages_with_slots()
    assert msgs[-1]["content"] == "\n".join([SPLICE_SENTINEL] * len(parts))
    assert c.messages_full()[-1]["content"] == text  # the full-text arm reads it whole
    row = c.to_task_row(target=[{"role": "assistant", "content": "ok"}])
    assert row["instruction"].count(SPLICE_SENTINEL) == len(json.loads(row["context"]))


def test_chat_rows_carry_the_tool_list_in_the_system_message():
    from bgkit2.agent.context import system_with_tool_list

    c = Conversation("sys", TOOLS, PlainRenderer(), RatioPolicy(), "q")
    c.add_user("hi")
    row = c.to_task_row(target=[{"role": "assistant", "content": "ok"}])
    assert row["framing"] == "chat"
    ins = json.loads(row["instruction"])
    assert ins[0]["content"] == system_with_tool_list("sys", TOOLS)
    assert ins[0]["content"].startswith("sys\nList of tools: [")
    tok = real_tok()
    from bgkit2.data.tasks import ChatRenderer

    rend = ChatRenderer.from_tokenizer(tok)
    assert rend._apply(ins) == rend._apply(c.messages_with_slots(), TOOLS)


def test_full_view_compaction_is_bounded_for_text_models():
    """A text model's view of the compaction slot is the TAIL of the folded history, so compact()
    can bound a text model's prompt (it could not: the full view rendered every folded round)."""
    from bgkit2.agent.context import COMPACTION_TOOL, Conversation

    class R:
        def count(self, text: str) -> int:
            return len(text.split())

        def render(self, msgs, tools, add_generation_prompt=True) -> str:
            return "\n".join(str(m.get("content", "")) for m in msgs)

    conv = Conversation("sys", [], R(), task_summary="t", full_view_compaction_tokens=50)
    conv.add_user("task")
    for i in range(20):
        conv.add_assistant(f"step {i} " + "word " * 30)
        conv.add_user(f"result {i} " + "data " * 30)
    conv.compact(200, compressed=False)
    msgs = conv.messages_full()
    comp = [m for m in msgs if m["content"].startswith(Conversation.TRUNCATED_HEAD)]
    assert comp, "compaction slot should be tail-truncated in the full view"
    assert R().count(comp[0]["content"]) <= 50 + 4
    assert conv.prefix_tokens(compressed=False) <= 200
    # the student's slot still holds every folded round
    s = [x for x in conv.slots if x.tool == COMPACTION_TOOL]
    assert s and "step 0" in s[0].text
