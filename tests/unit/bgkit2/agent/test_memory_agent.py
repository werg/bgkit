"""Two-stage agentic memory: tools + slot prompts, the always-on index, a scripted episode, the
training-row builder and the ``agent_memory`` bench arm (stub models; no GPU, no real data)."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bgkit2.agent.context import Conversation, PlainRenderer, ToolCall, render_tool_calls
from bgkit2.agent.memory import MemoryStore, add_memory_index, index_entries
from bgkit2.agent.memory_eval import memory_ratios, run_memory_episode
from bgkit2.agent.runner import CompressedModel, TrajectoryLog
from bgkit2.agent.tools import ToolRegistry
from bgkit2.data.memory_framing import (
    ABSTAIN,
    MEMORY_AGENT_SYSTEM,
    MEMORY_INDEX_HEADER,
    MEMORY_INDEX_MAX_REPS,
    MEMORY_INDEX_MIN_REPS,
    MEMORY_INDEX_PROMPT,
    MEMORY_INDEX_TOOL,
    MEMORY_TOOL_NAMES,
    index_reps,
    read_prompt,
    session_date,
)
from bgkit2.data.templates import SPLICE_SENTINEL

from .conftest import real_tok

ROOT = Path(__file__).resolve().parents[4]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SESSIONS = [
    "[session on 2023/05/01 (Mon) 10:00]\nuser: I adopted a cat named Miso.\nassistant: Lovely!",
    "[session on 2023/05/09 (Tue) 18:30]\nuser: We talked about the weather and a rainy trip.\n"
    "assistant: Bring an umbrella.",
    "[session on 2023/06/02 (Fri) 09:15]\nuser: My sister Lena moved to Lisbon for work.\n"
    "assistant: Exciting news for her.",
]
LOCOMO = ["[session 1, 1:56 pm on 8 May, 2023]\nCaroline: Hi Mel!",
          "[session 2, 1:14 pm on 25 May, 2023]\nMelanie: I ran a charity race."]


def reg_for(tmp_path, store) -> ToolRegistry:
    reg = ToolRegistry(str(tmp_path))
    reg.attach_memory(store, k=2)
    reg.keep_only(MEMORY_TOOL_NAMES)
    return reg


# ---------------------------------------------------------------------------
# tools + prompts
# ---------------------------------------------------------------------------


def test_session_dates_and_prompts():
    assert session_date(SESSIONS[0]) == "2023/05/01 (Mon) 10:00"
    assert session_date(LOCOMO[1]) == "1:14 pm on 25 May, 2023"
    assert session_date("no header\ntext") == ""
    assert read_prompt("2023/05/01 (Mon) 10:00", "the cat's name") == \
        "session 2023/05/01 (Mon) 10:00: the cat's name"
    assert read_prompt("", "") == "session"
    assert index_reps(10) == MEMORY_INDEX_MIN_REPS and index_reps(10**6) == MEMORY_INDEX_MAX_REPS
    assert index_reps(640) == 10


def test_memory_tools(tmp_path):
    store = MemoryStore.from_texts(SESSIONS)
    assert [s.id for s in store.sessions] == ["S1", "S2", "S3"]
    for alias in ("S3", "s3", "3", "[S3]", "session 3"):
        assert store.resolve(alias).id == "S3"
    reg = reg_for(tmp_path, store)
    assert reg.names == ["memory_search", "memory_read"]
    assert reg.spec("memory_search").compressible and reg.spec("memory_read").compressible
    res = reg.call("memory_search", {"query": "sister Lisbon", "focus": "where she moved"})
    # a compressible slot like web_search, with the ids + dates inside the slot text
    assert res.ok and res.compressible and res.tool == "memory_search"
    assert res.prompt == "memory search sister Lisbon: where she moved"
    assert memory_ratios().ratio("memory_search") == 4
    assert res.text.startswith("[S3] 2023/06/02 (Fri) 09:15\n") and "Lisbon" in res.text
    rd = reg.call("memory_read", {"session": "S1", "focus": "the cat's name"})
    assert rd.ok and rd.compressible and rd.tool == "memory_read"
    assert rd.text == SESSIONS[0]
    assert rd.prompt == "session 2023/05/01 (Mon) 10:00: the cat's name"
    bad = reg.call("memory_read", {"session": "S9"})
    assert not bad.ok and "S1..S3" in bad.text
    assert "no sessions match" in reg.call("memory_search", {"query": "zzzz"}).text


def test_index_turn_layout():
    store = MemoryStore.from_texts(SESSIONS)
    conv = Conversation(MEMORY_AGENT_SYSTEM, [], PlainRenderer(), memory_ratios())
    slots = add_memory_index(conv, store, "What is my cat called?")
    assert len(slots) == 3 and conv.slots == slots
    assert all(s.prompt == MEMORY_INDEX_PROMPT and s.tool == MEMORY_INDEX_TOOL for s in slots)
    assert all(MEMORY_INDEX_MIN_REPS <= s.n_reps <= MEMORY_INDEX_MAX_REPS for s in slots)
    user = conv.messages_with_slots()[1]["content"]
    assert user == (f"{MEMORY_INDEX_HEADER}\n[S1] 2023/05/01 (Mon) 10:00\n{SPLICE_SENTINEL}\n"
                    f"[S2] 2023/05/09 (Tue) 18:30\n{SPLICE_SENTINEL}\n"
                    f"[S3] 2023/06/02 (Fri) 09:15\n{SPLICE_SENTINEL}\n\n"
                    "Question: What is my cat called?")
    full = conv.messages_full()[1]["content"]
    assert SPLICE_SENTINEL not in full and SESSIONS[2] in full and "[S2] 2023/05/09" in full
    assert conv.contexts() == SESSIONS and conv.prompts() == [MEMORY_INDEX_PROMPT] * 3
    # the index is the head: compaction never folds it
    assert not conv.compact(budget=1)
    # an over-long session splits into labelled parts, no silent cut
    long = MemoryStore.from_texts(["[session on 2023/01/01 (Sun) 01:00]\n" + "word " * 400])
    ents = index_entries(long, PlainRenderer().count, slot_max_tokens=200)
    assert len(ents) > 1 and ents[0][0].endswith(f"(part 1/{len(ents)})")
    assert "".join(s.text for _, s in ents) == long.sessions[0].text


# ---------------------------------------------------------------------------
# scripted episode (search -> read gold -> answer) through run_episode
# ---------------------------------------------------------------------------


def scripted(turns: list[str]):
    it = iter(turns)
    seen: list = []

    def fn(sr) -> str:
        seen.append(sr)
        return next(it)

    return fn, seen


def test_scripted_episode(tmp_path):
    item = {"id": "m:1", "bench": "longmemeval", "subset": "single-session-user",
            "segments": SESSIONS, "question": "What is my cat called?", "answers": ["Miso"]}
    fn, seen = scripted([
        render_tool_calls([ToolCall("memory_search", {"query": "cat", "focus": "cat name"})]),
        render_tool_calls([ToolCall("memory_read", {"session": "S1", "focus": "cat name"})]),
        "Miso",
    ])
    model = CompressedModel(fn, PlainRenderer())
    with TrajectoryLog(tmp_path / "log.jsonl") as log:
        rec = run_memory_episode(item, model, tmp_path / "w", log=log)
    assert rec["pred"] == "Miso" and rec["stop"] == "final" and rec["turns"] == 3
    assert rec["sessions_read"] == ["S1"] and rec["n_read"] == 1 and rec["n_search"] == 1
    assert rec["index_slots"] == 3 and rec["read_tokens"] > 0 and rec["search_tokens"] > 0
    assert rec["encoder_tokens_nominal"] == (rec["index_tokens"] + rec["read_tokens"]
                                             + rec["search_tokens"])
    assert rec["decoder_tokens"] > 0
    # turn 1: index only; turn 2: + the search slot; turn 3: + the read under its focused prompt
    assert len(seen[0].contexts) == 3 and seen[0].prompts == [MEMORY_INDEX_PROMPT] * 3
    assert len(seen[1].contexts) == 4 and seen[1].prompts[-1] == "memory search cat: cat name"
    assert seen[1].contexts[-1].startswith("[S1] 2023/05/01 (Mon) 10:00\n")
    assert seen[1].n_reps[-1] == memory_ratios().reps(seen[1].n_tokens[-1], "memory_search")
    assert len(seen[2].contexts) == 5
    assert seen[2].prompts[-1] == "session 2023/05/01 (Mon) 10:00: cat name"
    assert seen[0].messages[0]["content"] == MEMORY_AGENT_SYSTEM
    assert {t["function"]["name"] for t in seen[0].tools} == set(MEMORY_TOOL_NAMES)
    rows = [json.loads(ln) for ln in (tmp_path / "log.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["kind"] == "row"]
    assert len(rows) == 3 and all(r["framing"] == "tool_slots" for r in rows)


# ---------------------------------------------------------------------------
# training rows
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def builder():
    return _load("bgkit2_build_memory_agent_store")


def test_identify_gold(builder):
    sess = ["[session on 2023/01/01 (Sun) 10:00]\nuser: I adopted a cat named Miso.",
            "[session on 2023/01/05 (Thu) 10:00]\nuser: I work at the bakery now; it goes well.",
            "[session on 2023/01/09 (Mon) 10:00]\nuser: I quit the bakery; now I work at a bank.",
            "[session on 2023/01/10 (Tue) 10:00]\nuser: Unrelated distractor about the cat Miso."]
    g = builder.identify_gold
    assert g("single_session", "What is my cat's name?", "Miso", sess, [0, 1, 2]) == [0]
    assert g("single_session", "What is my cat's name?", "Rex", sess, [0, 1, 2]) is None
    assert g("abstention", "q", ABSTAIN, sess, [0]) == []
    assert g("single_session", "q", "Miso", sess, []) is None  # no episode sessions
    # the new state (answer) + the old state (question overlap)
    assert g("knowledge_update", "Where do I work now?", "At a bank", sess, [0, 1, 2]) == [1, 2]
    assert g("knowledge_update", "Which bank?", "a bank", sess, [0, 1, 2]) == [2]
    assert g("multi_session", "Name my cat and my employer.", "Miso and the bank", sess,
             [0, 1, 2]) == [0, 2]
    assert g("temporal", "(asked on 2023/02/01 (Wed) 10:00) When did I adopt my cat?",
             "2023/01/01", sess, [0, 1, 2]) == [0]
    assert builder.STAMP.sub("", "(asked on 2026/02/18 (Wed) 05:00) What?") == "What?"


def test_episode_rows_layout(builder, tmp_path):
    q = "(asked on 2023/07/01 (Sat) 12:00) Where did my sister move?"
    meta = {"episode": 7, "qtype": "single_session"}
    rows, route = builder.episode_rows(SESSIONS, q, "Lisbon", "single_session", [2],
                                       PlainRenderer(), random.Random(0), "train", meta,
                                       str(tmp_path), p_direct=0.0)
    assert route == "search_read" and len(rows) == 3
    for r in rows:
        assert r["family"] == "memory_agent_v1" and r["framing"] == "tool_slots"
        ctx, prompts = json.loads(r["context"]), json.loads(r["prompt"])
        n_sent = r["instruction"].count(SPLICE_SENTINEL)
        assert n_sent == len(ctx) == len(prompts) == r["meta"]["n_slots"]
        assert len(r["meta"]["slot_reps"]) == len(ctx) and r["meta"]["gold_sessions"] == ["S3"]
        assert {t["function"]["name"] for t in r["meta"]["tools"]} == set(MEMORY_TOOL_NAMES)
    t0, t1 = json.loads(rows[0]["target"]), json.loads(rows[1]["target"])
    call0 = t0[0]["tool_calls"][0]["function"]
    assert call0["name"] == "memory_search"
    assert call0["arguments"]["query"] == "Where did my sister move?"
    call1 = t1[0]["tool_calls"][0]["function"]
    assert call1 == {"name": "memory_read", "arguments": {
        "session": "S3", "focus": "Where did my sister move?"}}
    assert rows[2]["target"] == "Lisbon"
    last = rows[2]
    assert json.loads(last["prompt"])[-1] == ("session 2023/06/02 (Fri) 09:15: "
                                              "Where did my sister move?")
    assert last["meta"]["slot_tools"] == [MEMORY_INDEX_TOOL] * 3 + ["memory_search",
                                                                     "memory_read"]
    assert json.loads(rows[1]["prompt"])[-1] == ("memory search Where did my sister move?: "
                                                 "Where did my sister move?")
    assert json.loads(rows[1]["context"])[-1].startswith("[S3] 2023/06/02 (Fri) 09:15\n")
    assert json.loads(last["context"])[-1] == SESSIONS[2]
    # direct read route + abstention reads the BM25 best candidate, then abstains
    rows_d, route_d = builder.episode_rows(SESSIONS, q, "Lisbon", "single_session", [2],
                                           PlainRenderer(), random.Random(0), "eval", meta,
                                           str(tmp_path), p_direct=1.0)
    assert route_d == "read" and len(rows_d) == 2 and rows_d[0]["split"] == "eval"
    rows_a, route_a = builder.episode_rows(SESSIONS, "Where does Lena live?", ABSTAIN,
                                           "abstention", [], PlainRenderer(), random.Random(0),
                                           "train", meta, str(tmp_path), p_abstain_search=0.0)
    assert route_a == "read_abstain" and rows_a[-1]["target"] == ABSTAIN
    read = json.loads(rows_a[0]["target"])[0]["tool_calls"][0]["function"]["arguments"]
    assert read["session"] == "S3"


def test_rows_render_through_trainer_template(builder, tmp_path):
    tok = real_tok()
    from bgkit2.agent.context import TemplateRenderer
    from bgkit2.data.tasks import ChatRenderer

    rows, _ = builder.episode_rows(SESSIONS, "Where did my sister move?", "Lisbon",
                                   "single_session", [2], TemplateRenderer(tok),
                                   random.Random(0), "train", {}, str(tmp_path), p_direct=0.0)
    chat = ChatRenderer.from_tokenizer(tok)
    for r in rows:
        segs = chat.render_slots(r["instruction"], r["meta"]["tools"])
        assert len(segs) - 1 == len(json.loads(r["context"]))


# ---------------------------------------------------------------------------
# bench arm (2-item fake memory bench) + judge rows
# ---------------------------------------------------------------------------


class StubStudent:
    """A 'student' whose policy reads the conversation: first turn -> memory_read(S2), after a
    tool result -> the answer. Counts the slots it had to encode."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        self.encoded: list[str] = []
        self.last: list[dict] = []
        stub = self

        class R:
            eos_id = 7

            def render_slots(self, instruction, tools, enable_thinking=None):
                stub.last = json.loads(instruction)
                k = sum(str(m.get("content") or "").count(SPLICE_SENTINEL) for m in stub.last)
                return [torch.zeros(3, dtype=torch.long) for _ in range(k + 1)]

        class D:
            def build_batch(self, prefix, reps, mask, suffix, **kw):
                n = int(reps.shape[1]) + sum(int(r.shape[0]) for r, _ in kw["extra"][0])
                return SimpleNamespace(inputs_embeds=torch.zeros(1, n + 6, 2))

            def generate(self, batch, **kw):
                return torch.tensor([[1 if stub.last[-1]["role"] == "tool" else 0]])

        self.renderer, self.decoder = R(), D()
        self.core = SimpleNamespace(autocast=contextlib.nullcontext)

    def _encode_slots(self, contexts, prompts, n_reps):
        self.encoded.extend(contexts)
        r = max(int(x) for x in n_reps)
        reps = torch.zeros(len(contexts), r, 2)
        mask = torch.zeros(len(contexts), r, dtype=torch.bool)
        for j, n in enumerate(n_reps):
            mask[j, : int(n)] = True
        return reps, mask, [len(c) // 4 for c in contexts]

    def _think_arg(self):
        return None

    def _budget(self, n):
        return n

    def _decode_turn(self, ids):
        return self._decode(ids)

    def _decode(self, ids):
        if ids[0] == 0:  # read S2 with the question as the focus
            q = str(self.last[1]["content"]).split("Question: ")[-1]
            return render_tool_calls([ToolCall("memory_read", {"session": "S2", "focus": q})])
        return "a charity race"


def test_bench_agent_memory_arm(tmp_path, monkeypatch):
    bench = _load("bgkit2_bench")
    student = StubStudent()
    student._agent_renderer = PlainRenderer()
    cache = bench.SlotRepCache(student, enc_batch=2)
    items = [{"id": f"locomo:c1:{i}", "bench": "locomo", "subset": "single_hop",
              "segments": LOCOMO, "question": q, "answers": ["a charity race"],
              "scorer": "em_f1", "options": None}
             for i, q in enumerate(["What did Melanie run?", "What race did Melanie run?"])]
    args = SimpleNamespace(out=str(tmp_path / "locomo.json"), agent_max_turns=4,
                           agent_max_new_tokens=32)
    agg, rows = {}, []
    recs = []
    for it in items:
        rec = bench.run_agent_memory_safe(student, cache, it, args)
        recs.append(rec)
        bench.agent_row(agg, rows, it, rec)
    a, b = recs
    assert a["pred"] == "a charity race" and a["turns"] == 2 and a["sessions_read"] == ["S2"]
    # item 1 encodes the 2 index slots + its read (turn 2 re-uses both from the cache);
    # item 2 asks the SAME conversation: its index is a cache hit, only its read is new
    assert a["encoder_tokens_computed"] > b["encoder_tokens_computed"] > 0
    assert len(student.encoded) == 2 + 1 + 1
    assert a["cache_hits"] == 2 and b["cache_hits"] == 4  # index on both turns of item 2
    assert a["encoder_tokens_nominal"] == a["index_tokens"] + a["read_tokens"]
    assert [r["arm"] for r in rows] == ["agent_memory@agent"] * 2
    assert all(r["em"] == 1.0 for r in rows)
    s = agg["agent_memory@agent"]
    assert s["turns"] == [2.0, 2.0] and s["n_read"] == [1.0, 1.0]
    # the judge runs on these rows unchanged
    judge = _load("bgkit2_bench_judge")
    (tmp_path / "locomo.rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    summary = {k: {m: sum(v) / len(v) for m, v in d.items()} for k, d in agg.items()}
    (tmp_path / "locomo.json").write_text(json.dumps({"summary": summary}))
    items_path = tmp_path / "items.jsonl"
    items_path.write_text("\n".join(json.dumps(it) for it in items))
    monkeypatch.setattr(judge, "ask", lambda url, prompt: "yes")
    monkeypatch.setattr(sys, "argv", ["judge", "--url", "x", "--run-dir", str(tmp_path),
                                      "--bench", "locomo", "--items", str(items_path)])
    judge.main()
    out = json.loads((tmp_path / "locomo.json").read_text())
    assert out["summary"]["agent_memory@agent"]["judge_acc"] == 1.0


def test_slot_cache_evicts_but_keeps_current_call():
    bench = _load("bgkit2_bench")
    student = StubStudent()
    cache = bench.SlotRepCache(student, enc_batch=4, max_reps=5)
    out, n = cache.encode(["a" * 40, "b" * 40], ["p", "p"], [4, 4])
    assert len(out) == 2 and n == 20 and cache.n_reps == 8  # both kept: they are this call's
    cache.encode(["c" * 40], ["p"], [4])
    assert cache.n_reps <= 5 + 4 and len(cache.store) == 1
