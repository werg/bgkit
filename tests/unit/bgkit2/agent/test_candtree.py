"""BrowseComp-Plus candidate-set tree (bgkit2.agent.candtree): deterministic structure, the
question prompt at every node, the focus on web_open's re-encode, the equal-budget full_text
arm -- on fake candidate sets and a tiny CPU encoder (no real data, no GPU)."""

from __future__ import annotations

import random

import pytest
import torch

from bgkit2.agent.candtree import (
    CANDTREE_ROLE,
    FANOUT,
    CandidateCorpus,
    CandidateTreeView,
    MemoryNodeCache,
    TreeEncoding,
    build_candidate_tree,
    candtree_item,
    encode_candidate_tree,
    item_pages,
    leaf_name,
    run_candtree_item,
    tree_encoder_tokens,
    tree_prompt,
)
from bgkit2.agent.context import PlainRenderer, RatioPolicy, ToolCall, render_tool_calls
from bgkit2.agent.runner import CompressedModel, FullTextModel
from bgkit2.agent.web import Page
from bgkit2.models.multilevel import node_children

from ..test_data_and_training import _tiny_trainer

DOMAINS = ["a.org", "b.com", "www.c.net", "d.io", "e.edu"]
FILLER = " ".join(f"Unrelated filler sentence {i} about rivers and towers." for i in range(120))


def _pages(n: int, seed: int = 0) -> list[Page]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        dom = DOMAINS[rng.randrange(len(DOMAINS))] if i % 3 else f"solo{i}.example"
        body = f"Page {i} text. The secret number of page {i} is {1000 + i}. " + (
            FILLER if i % 4 == 0 else "")
        out.append(Page(str(5000 + i), f"Title {i} / part", f"https://{dom}/p{i}", body))
    return out


def _query(n: int) -> dict:
    pages = _pages(n)
    doc = lambda p: {"docid": p.id, "url": p.url,  # noqa: E731
                     "text": f"---\ntitle: {p.title}\n---\n{p.text}"}
    return {"query_id": "q1", "query": "What is the secret number of page 3?", "answer": "1003",
            "gold_docs": [doc(pages[3])], "evidence_docs": [doc(pages[3]), doc(pages[4])],
            "negative_docs": [doc(p) for p in pages]}


# ---------------------------------------------------------------------------- structure


@pytest.mark.parametrize("n,levels", [(10, 0), (16, 0), (85, 1), (256, 1), (300, 2)])
def test_tree_is_deterministic_bounded_and_complete(n, levels):
    pages = _pages(n)
    t1 = build_candidate_tree(pages, "k")
    shuffled = pages[:]
    random.Random(5).shuffle(shuffled)
    t2 = build_candidate_tree(shuffled + pages[:3], "k")  # order + duplicates do not matter
    assert t1.paths == t2.paths and [p.id for p in t1.pages] == [p.id for p in t2.pages]
    assert t1.max_depth == levels + 1
    assert sorted(p.id for p in t1.pages) == sorted(p.id for p in pages)  # every page one leaf
    assert all(len(p.split("/")) == levels + 1 for p in t1.paths)
    kids = node_children(t1.paths, t1.max_depth)
    assert all(len(c) <= FANOUT for c in kids.values()), {k: len(c) for k, c in kids.items()}
    # folders are contiguous in (domain, title, id) order: one site's pages sit together
    if levels:
        by_folder = {}
        for p, path in zip(t1.pages, t1.paths, strict=True):
            by_folder.setdefault(path.split("/")[0], []).append(p)
        sizes = [len(v) for v in by_folder.values()]
        assert max(sizes) - min(sizes) <= FANOUT  # balanced
    # leaves carry title and id; no path separator leaks out of a title
    assert t1.paths[0].endswith(f"[{t1.pages[0].id}]") and "Title" in t1.paths[0]
    assert leaf_name(pages[0]).count("/") == 0


def test_candidate_item_and_lenient_page_lookup():
    item = candtree_item(_query(20))
    assert item["meta"]["gold_ids"] == ["5003"] and item["answers"] == ["1003"]
    pages = item_pages(item)
    assert len(pages) == 20 and len({p.id for p in pages}) == 20  # gold/evidence deduped
    assert pages[0].id == "5003" and pages[0].title == "Title 3 / part"  # gold first, parsed
    c = CandidateCorpus(pages)
    p = pages[5]
    assert c.get(p.id) is p and c.get(f"[{p.id}]") is p and c.get(leaf_name(p)) is p
    tree = build_candidate_tree(pages, "k")
    assert c.get(tree.paths[0]) is tree.pages[0]  # a full tree path
    assert c.get(p.title.upper()) is p and c.get("nope") is None
    # web_search in the full_text arm sees ONLY the candidates
    assert {h.id for h, _ in c.search("secret number page", 50)} <= {q.id for q in pages}


def test_cost_estimate_counts_windows_prompts_and_capped_leaves():
    c = tree_encoder_tokens([100, 10_000], [5, 5], 20, leaf_window=4096, ratio_leaf=0.125,
                            leaf_max_reps=512, n_nodes=2)
    assert c["tree_leaf_windows"] == 1 + 3
    assert c["tree_leaf_tokens"] == 100 + 25 + 10_000 + 3 * 25
    assert c["tree_node_tokens"] == round(0.125 * 105) + 512 + 2 * 20


# ---------------------------------------------------------------------------- encoding


def _tok(s: str) -> list[int]:
    return [3 + (ord(ch) % 40) for ch in s]


def _core(tmp_path, **over):
    tr = _tiny_trainer(tmp_path, **{"optim.encoder_mode": "adapter", "model.repro_head": "true",
                                    "multilevel.bridge": "identity",
                                    "multilevel.ratio_leaf": 0.25, "multilevel.ratio_node": 0.5,
                                    "multilevel.leaf_window": 64,
                                    "multilevel.leaf_max_reps": 24,
                                    "multilevel.node_max_tokens": 64,
                                    "data.repo_tree_file_tokens": 100_000, **over})
    tr.encoder.eval()
    return tr


def test_query_prompt_reaches_every_node_of_the_tree(tmp_path):
    tr = _core(tmp_path)
    pages = _pages(40)
    tree = build_candidate_tree(pages, "k", fanout=4)  # 40 pages -> 2 folder levels at fan-out 4
    assert tree.max_depth == 3
    q = "What is the secret number of page 3?"
    calls: list[tuple] = []
    orig = tr.encoder.forward

    def spy(*a, **kw):
        calls.append((kw.get("prompt_ids"), tr.encoder.active_role))
        return orig(*a, **kw)

    tr.encoder.forward = spy
    te = encode_candidate_tree(tr, tree, q, _tok)
    want = torch.tensor(_tok(tree_prompt(q)))
    assert len(calls) >= 3
    for p, role in calls:  # leaves, windows, folders, virtual folders, root: ALL prompted
        assert p is not None and all(torch.equal(row, want) for row in p)
        assert role == CANDTREE_ROLE
    assert te.stats["tree_leaf_windows"] > len(pages)  # long pages were windowed
    assert te.stats["n_virtual"] > 0  # small node budget: virtual folders, prompted too
    assert set(te.nodes) == set(node_children(tree.paths, tree.max_depth))  # every folder encoded
    tr.encoder.forward = orig
    other = encode_candidate_tree(tr, tree, "Which page mentions towers?", _tok)
    assert not torch.allclose(te.nodes[""][0].float(), other.nodes[""][0].float())


# ---------------------------------------------------------------------------- episodes


def _fake_encode(tree, question):
    nodes = {n: (torch.full((3 + len(n) % 4, 8), float(len(n))), torch.ones(3 + len(n) % 4,
                                                                                dtype=torch.bool))
             for n in node_children(tree.paths, tree.max_depth)}
    return TreeEncoding(nodes=nodes, stats={"tree_encoder_tokens": 1234, "root_reps": 3})


def _script(replies):
    seen = []

    def step(x):
        seen.append(x)
        return replies[len(seen) - 1]

    return step, seen


def test_tree_arm_browses_and_web_open_reencodes_with_the_focus(tmp_path):
    item = candtree_item(_query(40))
    tree = build_candidate_tree(item_pages(item), "x")
    folder = tree.paths[tree.pages.index(next(p for p in tree.pages if p.id == "5003"))]
    folder = folder.split("/")[0]
    focus = "the secret number of page 3"
    replies = [render_tool_calls([ToolCall("browse", {"dir": folder})]),
               render_tool_calls([ToolCall("web_open", {"id": leaf_name(tree.page_of(
                   next(p for p in tree.paths if p.endswith("[5003]")))), "focus": focus})]),
               "It is 1003.\nANSWER: 1003"]
    step, seen = _script(replies)
    cache = MemoryNodeCache()
    recs = run_candtree_item(item, ["bgkit_tree"],
                             lambda arm: CompressedModel(step, PlainRenderer()), tmp_path / "wd",
                             encode=_fake_encode, cache=cache, ratios=RatioPolicy(),
                             max_turns=6, decoder_token_budget=100_000)
    (r,) = recs
    assert r["stop"] == "final" and r["em"] == 1.0 and r["n_browse"] == 1
    assert r["opened_ids"] == ["5003"] and r["gold_opened_any"]  # alias resolved to the id
    assert r["tree"]["tree_encoder_tokens"] == 1234
    assert r["encoder_tokens_total"] == r["encoder_tokens"] + 1234
    last = seen[-1]
    assert last.contexts[0].startswith('{"repo_overview"')  # root overview slot
    assert f'"node": "{folder}"' in last.contexts[1]  # the browsed folder's node reps
    assert "secret number of page 3 is 1003" in last.contexts[2]  # the page, raw, re-encoded
    assert focus in last.prompts[2] and last.prompts[2].startswith("page ")
    assert "browse" in {t["function"]["name"] for t in last.tools}
    assert "web_search" not in {t["function"]["name"] for t in last.tools}
    assert len(cache) == 0  # the per-query tree is dropped after the episode


def test_student_serves_tree_nodes_and_encodes_the_open_page_under_the_focus(tmp_path):
    """The GPU half on CPU: overview / browse slots come from the per-query node cache, the
    web_open slot goes through the encoder with the FOCUS prompt."""
    from bgkit2.agent.student import StudentModel

    tr = _core(tmp_path)
    cache = MemoryNodeCache()
    d = tr.encoder.backbone.hidden_size
    cache.put("candtree/q", {"": (torch.randn(5, d), torch.ones(5, dtype=torch.bool)),
                             "01 a.org": (torch.randn(7, d), torch.tensor([1] * 6 + [0]).bool())})

    class R:
        def encode_ctx(self, s):  # encoder-side text (one tokenizer here)
            return torch.tensor(_tok(s), dtype=torch.long)

    student = StudentModel.__new__(StudentModel)
    student.core, student.encoder, student.device = tr, tr.encoder, torch.device("cpu")
    student.renderer, student.max_ctx_tokens = R(), 8192
    student.tree_root_cache = cache
    seen = []
    orig = tr.encoder.forward

    def spy(*a, **kw):
        seen.append(kw.get("prompt_ids"))
        return orig(*a, **kw)

    tr.encoder.forward = spy
    prompt = "page Title 3: the secret number of page 3"
    contexts = ['{"repo_overview": "candtree/q", "n_reps": 5}',
                '{"repo_overview": "candtree/q", "n_reps": 6, "node": "01 a.org"}',
                "# Title 3\nhttps://a.org/p3\n\nThe secret number of page 3 is 1003."]
    with torch.no_grad():
        reps, mask, _n = student._encode_slots(contexts, ["", "", prompt], [5, 6, 4])
    assert mask[0].sum() == 5 and mask[1].sum() == 6 and mask[2].sum() >= 1
    assert torch.allclose(reps[0, :5], cache.get("candtree/q", "")[0])
    (p,) = seen  # ONE encoder call: the page, prompted by the focus
    assert torch.equal(p[0][p[0] != 0], torch.tensor(_tok(prompt)))


def test_full_text_arm_searches_only_candidates_at_equal_budget(tmp_path):
    item = candtree_item(_query(30))
    focus = "secret number"
    replies = [render_tool_calls([ToolCall("web_search", {"query": "secret number page 0",
                                                          "focus": focus})]),
               render_tool_calls([ToolCall("web_open", {"id": "5000", "focus": focus})]),
               "ANSWER: 1000"]
    prompts: list[str] = []

    def step(x):
        prompts.append(x)
        return replies[len(prompts) - 1]

    ratios = RatioPolicy()
    ratios.ratios["web_open"] = 16
    kw = dict(ratios=ratios, max_turns=6, decoder_token_budget=100_000)
    (r,) = run_candtree_item(item, ["full_text"], lambda arm: FullTextModel(step, PlainRenderer()),
                             tmp_path / "wd", **kw)
    assert r["arm"] == "full_text" and r["stop"] == "final" and r["n_search"] == 1
    assert r["truncated_results"] >= 1  # page 5000 carries the filler: cut to the reps budget
    assert r["encoder_tokens_total"] == 0 and r["n_candidates"] == 30
    rend = PlainRenderer()
    page = next(p for p in item_pages(item) if p.id == "5000")
    full = f"# {page.title}\n{page.url}\n\n{page.text}"
    cap = ratios.reps(rend.count(full), "web_open")
    assert FILLER[-40:] not in prompts[-1] and "[... truncated]" in prompts[-1]
    assert cap < rend.count(full)
    # the tree arm with the same kwargs gets the same decoder budget (one kwargs set)
    step2, _ = _script(["ANSWER: 1000"])
    (b,) = run_candtree_item(item, ["bgkit_tree"],
                             lambda arm: CompressedModel(step2, PlainRenderer()), tmp_path / "wd",
                             encode=_fake_encode, cache=MemoryNodeCache(), **kw)
    assert b["stop"] == "final" and b["decoder_tokens"] <= kw["decoder_token_budget"]


def test_view_listing_names_folders_and_pages(tmp_path):
    pages = _pages(40)
    tree = build_candidate_tree(pages, "k")
    cache = MemoryNodeCache()
    te = _fake_encode(tree, "q")
    cache.put("k", te.nodes)
    v = CandidateTreeView(tree, cache)
    root = v.listing("")
    assert all(line.split(" (")[0].endswith("/") for line in root.splitlines())
    first = next(iter(sorted(v.children[""])))[:-1]
    sub = v.listing(first)
    assert all(line.startswith("[") for line in sub.splitlines())
    assert v.n_reps("") == te.nodes[""][0].shape[0]
