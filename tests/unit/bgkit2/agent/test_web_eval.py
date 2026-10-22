"""Offline web-research eval: answer parsing, arm budgets, a scripted episode, the disk index and
the benchmark prep on a tiny fake corpus (no GPU, no real data)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from bgkit2.agent.context import PlainRenderer, RatioPolicy, ToolCall, render_tool_calls
from bgkit2.agent.runner import CompressedModel, FullTextModel, SlotRendering
from bgkit2.agent.web import (
    DiskWebCorpus,
    Page,
    WebCorpus,
    build_disk_index,
    parse_front_matter,
)
from bgkit2.agent.web_eval import (
    judge_rows,
    load_done,
    load_records,
    parse_answer,
    run_item,
    score_answer,
    summarize,
    truncate_tokens,
    web_task,
)

ROOT = Path(__file__).resolve().parents[4]
PREP = ROOT / "scripts/bgkit2_web_bench_prep.py"
_spec = importlib.util.spec_from_file_location("web_prep", PREP)
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)

LONG = " ".join(f"Filler sentence number {i} about unrelated towers." for i in range(200))
PAGES = [
    Page("p1", "Eiffel Tower", "https://x/eiffel",
         "The Eiffel Tower is in Paris. It was completed in 1889. " + LONG),
    Page("p2", "Tower Bridge", "https://x/bridge", "Tower Bridge crosses the Thames in London."),
    Page("p3", "Gustave Eiffel", "", "Gustave Eiffel was a French civil engineer and architect."),
]
ITEM = {"id": "t:1", "bench": "web_test", "subset": "s", "corpus": "c", "segments": [],
        "question": "In which year was the Eiffel Tower completed?", "answers": ["1889"],
        "scorer": "em_f1", "options": None, "meta": {"gold_ids": ["p1"],
                                                     "gold_titles": ["Eiffel Tower"]}}


def test_parse_and_score():
    assert parse_answer("I found it.\nANSWER: 1889.") == ("1889", "answer")
    assert parse_answer("answer: x\nmore\nANSWER: **The Eiffel Tower**") == (
        "The Eiffel Tower", "answer")
    assert parse_answer("It is Paris") == ("It is Paris", "bare")
    assert parse_answer("") == ("", "none") and parse_answer("ANSWER:  ") == ("", "none")
    assert score_answer("the Eiffel Tower!", ["Eiffel tower"]) == {"em": 1.0, "f1": 1.0}
    s = score_answer("Paris France", ["Paris", "Lyon"])
    assert s["em"] == 0.0 and abs(s["f1"] - 2 / 3) < 1e-9


def test_task_never_carries_gold():
    t = web_task(ITEM, "/x")
    blob = t.system_prompt + t.instruction
    assert ITEM["question"] in t.instruction and "ANSWER:" in t.system_prompt
    assert "1889" not in blob and "p1" not in blob


def test_disk_index_matches_in_memory(tmp_path: Path):
    pages = [*PAGES, *[Page(f"q{i}", f"Page {i}", "", f"river {i} bridge " + "city " * (i + 1))
                       for i in range(30)]]
    meta = build_disk_index(iter(pages), tmp_path / "idx", chunk_docs=7)  # several spills
    d, m = DiskWebCorpus(tmp_path / "idx"), WebCorpus(pages)
    assert meta["n_docs"] == len(pages) == len(d)
    for q in ("eiffel tower completed", "bridge thames", "river 7 city", "nothing-matches-zzz"):
        a = [(p.id, round(s, 4)) for p, s in m.search(q, 5)]
        b = [(p.id, round(s, 4)) for p, s in d.search(q, 5)]
        assert [x[0] for x in a] == [x[0] for x in b] and a == pytest.approx(b, rel=1e-4)
    assert d.get("p2").text == PAGES[1].text and d.get("nope") is None
    assert not (tmp_path / "idx.tmp").exists()


def _script(arm_calls: list[str]):
    seen: list = []

    def step(x):
        seen.append(x)
        return arm_calls[len(seen) - 1]

    return step, seen


SEARCH = render_tool_calls([ToolCall("web_search", {"query": "Eiffel Tower completed",
                                                     "focus": "completion year"})])
OPEN = render_tool_calls([ToolCall("web_open", {"id": "p1", "focus": "completion year"})])


def test_scripted_episode_both_arms_resume_summary(tmp_path: Path):
    corpus = WebCorpus(PAGES)
    seen: dict[str, list] = {}

    def make_model(arm):
        step, seen[arm] = _script([SEARCH, OPEN, "Finished in 1889.\nANSWER: 1889<|im_end|>"])
        if arm == "bgkit":
            return CompressedModel(step, PlainRenderer())
        return FullTextModel(step, PlainRenderer())

    out = tmp_path / "out"

    def write(arm, rec):
        with (out / f"{arm}.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    out.mkdir()
    ratios = RatioPolicy()
    recs = run_item(ITEM, ["bgkit", "full_text"], make_model, corpus, tmp_path / "wd",
                    write=write, ratios=ratios, max_turns=6, decoder_token_budget=100_000)
    b, f = recs
    for r in (b, f):
        assert r["stop"] == "final" and r["turns"] == 3 and r["em"] == 1.0 and r["answered"]
        assert r["gold_opened_any"] and r["gold_opened_all"] and r["gold_searched_any"]
        assert r["gold_title_opened_any"] and r["n_search"] == 1 and r["n_open"] == 1
    assert b["encoder_tokens"] > 0 and b["encoder_tokens_unique"] > 0
    assert b["web_find_mode"] == f["web_find_mode"] == "encoded"
    assert f["encoder_tokens"] == 0 and f["truncated_results"] >= 1
    # only the web tools are offered
    sr = seen["bgkit"][-1]
    assert isinstance(sr, SlotRendering)
    assert sorted(t["function"]["name"] for t in sr.tools) == ["web_find", "web_open", "web_search"]
    assert any("completed in 1889" in c for c in sr.contexts)  # the page is a slot
    assert "completed in 1889" in seen["full_text"][-1]  # the head survives the cut
    assert "Filler sentence number 199" not in seen["full_text"][-1]  # the tail does not
    # resume: nothing re-run
    done = {a: load_done(out / f"{a}.jsonl") for a in ("bgkit", "full_text")}
    assert run_item(ITEM, ["bgkit", "full_text"], make_model, corpus, tmp_path / "wd",
                    done=done, write=write) == []
    recs_by = {a: load_records(out / f"{a}.jsonl") for a in ("bgkit", "full_text")}
    rows = judge_rows(recs_by)
    assert {(r["id"], r["arm"], r["pred"]) for r in rows} == {("t:1", "bgkit", "1889"),
                                                              ("t:1", "full_text", "1889")}
    s = summarize(list(recs_by["bgkit"].values()))
    assert s["n"] == 1 and s["em"] == 1.0 and s["gold_opened_any"] == 1.0 and s["turns"] == 3.0


def test_arm_budgets(tmp_path: Path):
    r = PlainRenderer()
    text = "word " * 400  # 500 tokens at 4 chars/token
    cut = truncate_tokens(text, 32, r)
    assert r.count(cut) <= 32 and cut.endswith("[... truncated]") and text.startswith(cut[:40])
    assert truncate_tokens("short", 32, r) == "short"
    # full_text "none": the page arrives verbatim
    step, seen = _script([OPEN, "ANSWER: 1889"])
    rec = run_item(ITEM, ["full_text"], lambda a: FullTextModel(step, r), WebCorpus(PAGES),
                   tmp_path, full_text_results="none", max_turns=4)[0]
    assert rec["truncated_results"] == 0 and "Filler sentence number 199" in seen[-1]
    assert rec["em"] == 1.0 and not rec["gold_searched_any"]
    # the episode decoder budget stops the loop before the model is called again
    step, seen = _script([OPEN, "ANSWER: 1889"])
    rec = run_item(ITEM, ["bgkit"], lambda a: CompressedModel(step, r), WebCorpus(PAGES),
                   tmp_path, max_turns=4, decoder_token_budget=900)[0]
    assert rec["stop"] == "budget" and rec["em"] == 0.0 and rec["pred"] == ""
    # an arm run on the wrong kind of model is an error record, not a silent mis-measurement
    rec = run_item(ITEM, ["bgkit"], lambda a: FullTextModel(step, r), WebCorpus(PAGES), tmp_path)[0]
    assert rec["stop"] == "error" and "compressed" in rec["error"]
    assert summarize([rec])["errors"] == 1


def test_prep_on_fake_data(tmp_path: Path):
    hp = [{"id": "h1", "question": "Q1", "answer": "A1", "type": "bridge", "level": "hard",
           "supporting_facts": {"title": ["T1", "T2", "T1"], "sent_id": [0, 0, 1]},
           "context": {"title": ["T1", "T2", "T3"],
                       "sentences": [["One.", " Two."], ["Three."], ["Shared."]]}}]
    mq = [{"id": "2hop__1_2", "question": "Q2", "answer": "A2", "answer_aliases": [],
           "paragraphs": [{"idx": 0, "title": "T3", "paragraph_text": "Shared.",
                           "is_supporting": True},
                          {"idx": 1, "title": "T4", "paragraph_text": "Four.",
                           "is_supporting": False}]}]
    pages, q2ids = prep.build_pool(hp, mq)
    assert [p.id for p in pages] == ["p0", "p1", "p2", "p3"]  # T3 "Shared." deduped
    assert pages[0].text == "One. Two." and q2ids["musique:2hop__1_2"] == ["p2", "p3"]
    rows = {"hotpot:h1": hp[0], "musique:2hop__1_2": mq[0]}
    sc_h = [{"id": "hotpot:h1", "question": "Q1", "answers": ["A1"]}]
    sc_m = [{"id": "musique:2hop__1_2", "question": "Q2", "answers": ["A2"]}]
    h = prep.web_items(sc_h, rows, q2ids, "hotpot")[0]
    m = prep.web_items(sc_m, rows, q2ids, "musique")[0]
    assert h["segments"] == [] and h["corpus"] == "pool" and h["bench"] == "web_hotpot"
    assert h["meta"] == {"gold_ids": ["p0", "p1"], "gold_titles": ["T1", "T2"]}
    assert m["meta"]["gold_ids"] == ["p2"] and m["subset"] == "2hop" and m["answers"] == ["A2"]

    doc = lambda i: {"docid": str(i), "url": f"https://x/{i}",  # noqa: E731
                     "text": f"---\ntitle: Doc {i}\ndate: 2020\n---\nBody of doc {i}."}
    qs = tmp_path / "queries.jsonl"
    qs.write_text("\n".join(json.dumps({"query_id": str(k), "query": f"q{k}", "answer": f"a{k}",
                                        "gold_docs": [doc(k)], "evidence_docs": [doc(k), doc(9)],
                                        "negative_docs": [doc(8)]}) for k in range(5)))
    items, oracle = prep.bcp_items(qs, 3, seed=1)
    assert len(items) == len(oracle) == 3 and items[0]["corpus"] == "browsecomp_plus"
    o = oracle[0]
    k = o["id"].split(":")[1]
    assert o["meta"] == {"gold_ids": [k], "evidence_ids": [k, "9"]}
    assert o["segments"] == [f"Doc {k}\nBody of doc {k}.", "Doc 9\nBody of doc 9."]
    assert set(o) >= {"id", "bench", "subset", "segments", "question", "answers", "scorer",
                      "options"}
    # the candidate-set bench holds the SAME seeded questions, with gold+evidence+negative pages
    ns = prep.bcp_candtree(qs, 3, 1, tmp_path / "cand.jsonl")
    cand = [json.loads(ln) for ln in (tmp_path / "cand.jsonl").open()]
    assert [c["id"] for c in cand] == [i["id"] for i in items] and ns == [3, 3, 3]
    assert cand[0]["bench"] == "browsecomp_plus_candtree"
    assert [p["id"] for p in cand[0]["pages"]] == [k, "9", "8"]
    assert cand[0]["meta"]["negative_ids"] == ["8"]
    assert parse_front_matter(doc(3)["text"]) == ("Doc 3", "Body of doc 3.")
    assert parse_front_matter("plain") == ("", "plain")

    import pyarrow as pa
    import pyarrow.parquet as pq

    (tmp_path / "corpus").mkdir()
    pq.write_table(pa.Table.from_pylist([doc(i) for i in range(4)]),
                   tmp_path / "corpus" / "train-0.parquet")
    got = list(prep.iter_bcp_pages(tmp_path / "corpus"))
    assert [(p.id, p.title, p.text) for p in got][1] == ("1", "Doc 1", "Body of doc 1.")
    build_disk_index(iter(got), tmp_path / "idx")
    assert DiskWebCorpus(tmp_path / "idx").search("doc 2 body", 1)[0][0].id == "2"


def test_web_find_mode_framing(tmp_path: Path):
    from bgkit2.agent.web_eval import WEB_SYSTEM, WEB_SYSTEM_VERBATIM_FIND, web_system

    assert web_system("encoded") == WEB_SYSTEM and "read again" in WEB_SYSTEM
    assert "verbatim" not in WEB_SYSTEM
    assert "`web_find` is verbatim" in WEB_SYSTEM_VERBATIM_FIND
    assert web_task(ITEM, tmp_path).system_prompt == WEB_SYSTEM
    assert web_task(ITEM, tmp_path, "verbatim").system_prompt == WEB_SYSTEM_VERBATIM_FIND
