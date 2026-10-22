"""BrowseComp-Plus CANDIDATE-SET TREE (user design, 2026-09-22).

The BrowseComp-Plus corpus is flat (100k pages over 34k domains, 25k of them
contributing one page), so a whole-corpus tree has no structure to walk. But
every query comes with a curated CANDIDATE SET -- its gold, evidence and hard
negative pages (~85 pages, ~600k tokens) -- and that set is small enough to
encode as ONE tree per query. Because the tree is per query anyway, it is
encoded with the QUESTION as the encoder prompt at every node (leaves,
windows, directories, root: ``encode_tree(prompt_ids=...)``), not the frozen
query-independent node cache the repo trees use. The agent reads the root as
its opening overview, ``browse``s a folder (that node's reps), and when it
opens a page with ``web_open`` the page is re-encoded FLAT under the agent's
own ``focus`` (the ordinary compressible-slot path: the slot prompt is
``page <title>: <focus>``) -- question-conditioned tree first, then the
agent's refined intent on the page it chose (the looped-transformer effect
the user pointed out).

Tree structure over flat pages (``build_candidate_tree``), simple and
deterministic: pages sorted by (domain, title, id), then cut into balanced
CONTIGUOUS groups of at most ``fanout`` (16) -- so pages of one site sit
together and a folder name says which sites it holds. ~85 pages = one folder
level (6 folders of ~14); above ``fanout**2`` pages a second folder level is
added (never needed for BrowseComp-Plus). Leaves are the pages, named
``<title> [<id>]`` so the path header every leaf is encoded under carries both
the title and the id ``web_open`` takes; long pages are windowed and capped
exactly as the run's live trees are (``core.tree_encode_kwargs``).

Arms (same weights, same loop, same ``decoder_token_budget``):

  bgkit_tree  root overview (reps) + ``browse`` (folder reps) + ``web_open``
              (page re-encoded under the focus) + ``web_find`` (page re-encoded
              with the pattern + focus as prompt; ``web_find_mode="verbatim"``
              = the control arm's verbatim lines)
  full_text   ``web_search`` over ONLY the candidate set + ``web_open`` +
              ``web_find``, every compressible result cut to the decoder
              positions the compressed arm would spend on it (equal budget)

The encoder role is ``slots`` (``CANDTREE_ROLE``): the role the trainer uses
for LIVE trees, so a checkpoint with the encoder adapter encodes the
candidate tree through it (without an adapter ``slots`` IS the base encoder).
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .context import Conversation, RatioPolicy, Renderer
from .runner import AgentTask, Model, TrajectoryLog, run_episode
from .tools import ToolRegistry
from .web import SNIPPET_WORDS, WEB_TOOL_NAMES, Page, WebCorpus, parse_front_matter
from .web_eval import (
    WEB_COMPACTION_MODE,
    WEB_INSTRUCTION,
    WEB_SLOT_MAX_TOKENS,
    EncoderCounter,
    EpisodeTools,
    episode_record,
    error_record,
    web_system,
)

CANDTREE_BENCH = "browsecomp_plus_candtree"
CANDTREE_ARMS = ("bgkit_tree", "full_text")
CANDTREE_ROLE = "slots"  # the trainer's LIVE-tree role (BgKIT2Core.tree_role for a live tree)
FANOUT = 16
MAX_FOLDER_LEVELS = 2  # + the page level = encode_tree max_depth 3
PROMPT_MAX_TOKENS = 256  # as BgKIT2Core.prompted_node
TREE_TOOLS = ("browse", "web_open", "web_find")

_CANDTREE_HEAD = (
    "You are a research assistant answering a question from a fixed set of candidate web "
    "pages. You do not know the answer in advance and must find it in the pages. Before the "
    "question you are given a compressed overview of ALL candidate pages, grouped into folders "
    "by website. `browse` with a folder name opens that folder: its pages, compressed. "
    "`web_open` with a page id (the number in brackets after a page title) reads that page, and "
)
_CANDTREE_FIND = {
    "encoded": (
        "`web_find` looks for a phrase on a page (the page opened last, or `id`): the whole page "
        "is read again for you with the phrase as what to keep. Give a `focus` with `web_open` "
        "and `web_find` saying what you are looking for; their results may be summarised for "
        "you. "
    ),
    "verbatim": (
        "`web_find` gives the exact lines of the page opened last that contain a phrase. Give a "
        "`focus` with `web_open` saying what you are looking for; its result may be summarised "
        "for you, `web_find` is verbatim. "
    ),
}
_CANDTREE_TAIL = (
    "A question may need several pages: open each page that can hold a missing fact. Never "
    "repeat a call you already made. When you know the answer, reply with ONLY the answer on "
    "the last line in the form `ANSWER: <answer>` (a short name, number, date or phrase, no "
    "explanation) and no tool call."
)


def candtree_system(find_mode: str = "encoded") -> str:
    return _CANDTREE_HEAD + _CANDTREE_FIND[find_mode] + _CANDTREE_TAIL


CANDTREE_SYSTEM = candtree_system("encoded")


# ---------------------------------------------------------------------------
# Candidate sets
# ---------------------------------------------------------------------------


def bcp_page(d: dict) -> Page:
    """A BrowseComp-Plus doc (``docid`` / ``url`` / front-matter ``text``) as a Page (the same
    parse as scripts/bgkit2_web_bench_prep.py's ``_doc_page``)."""
    title, body = parse_front_matter(d.get("text") or "")
    return Page(str(d["docid"]), title or d.get("url") or str(d["docid"]), d.get("url") or "",
                body.strip())


def candidate_pages(q: dict) -> list[Page]:
    """The query's candidate set: gold + evidence + negative docs, first occurrence kept."""
    seen: dict[str, Page] = {}
    for key in ("gold_docs", "evidence_docs", "negative_docs"):
        for d in q.get(key) or []:
            seen.setdefault(str(d["docid"]), bcp_page(d))
    return list(seen.values())


def candtree_item(q: dict) -> dict:
    """One bench item (the web eval's item schema plus the candidate ``pages``)."""
    ids = lambda k: [str(d["docid"]) for d in q.get(k) or []]  # noqa: E731
    pages = candidate_pages(q)
    return {
        "id": f"browsecomp_plus:{q['query_id']}", "bench": CANDTREE_BENCH,
        "subset": "browsecomp_plus", "question": q["query"], "answers": [str(q["answer"])],
        "scorer": "em_f1", "options": None,
        "meta": {"gold_ids": ids("gold_docs"), "evidence_ids": ids("evidence_docs"),
                 "negative_ids": ids("negative_docs")},
        "pages": [{"id": p.id, "title": p.title, "url": p.url, "text": p.text} for p in pages],
    }


def item_pages(item: dict) -> list[Page]:
    return [Page(str(p["id"]), p["title"], p["url"], p["text"]) for p in item["pages"]]


# ---------------------------------------------------------------------------
# Tree structure
# ---------------------------------------------------------------------------


def page_domain(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        host = ""
    host = host.removeprefix("www.")
    return host or "no-domain"


def _clean(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", str(s).replace("/", "-").replace("[", "(").replace("]", ")")).strip()
    return s[:n].rstrip() or "untitled"


def leaf_name(p: Page) -> str:
    """``<title> [<id>]``: the path component a page is encoded (and listed) under."""
    return f"{_clean(p.title, 60)} [{_clean(p.id, 40)}]"


def _balanced(n: int, k: int) -> list[int]:
    base, extra = divmod(n, k)
    return [base + (i < extra) for i in range(k)]


def _folder_name(idx: int, pages: list[Page]) -> str:
    doms = list(dict.fromkeys(page_domain(p.url) for p in pages))
    label = doms[0] if len(doms) == 1 else f"{doms[0]} .. {doms[-1]}"
    return f"{idx:02d} {_clean(label, 48)}"


@dataclass
class CandidateTree:
    key: str  # the overview / node-cache key (one per query)
    pages: list[Page]  # in tree order
    paths: list[str]  # aligned with pages
    max_depth: int  # encode_tree / build_tree depth (path components)
    folders: dict[str, list[str]] = field(default_factory=dict)  # folder path -> domains

    def page_of(self, path: str) -> Page:
        return self.pages[self.paths.index(path)]


def build_candidate_tree(pages: list[Page], key: str, fanout: int = FANOUT) -> CandidateTree:
    """Deterministic folder hierarchy over a flat candidate set (see the module docstring).
    Independent of the input order; duplicate ids keep their first page."""
    if fanout < 2:
        raise ValueError("fanout must be >= 2")
    uniq: dict[str, Page] = {}
    for p in pages:
        uniq.setdefault(p.id, p)
    order = sorted(uniq.values(),
                   key=lambda p: (page_domain(p.url), " ".join(p.title.lower().split()), p.id))
    n = len(order)
    if n == 0:
        raise ValueError("empty candidate set")
    levels = 0
    while fanout ** (levels + 1) < n and levels < MAX_FOLDER_LEVELS:
        levels += 1
    paths: list[str] = []
    folders: dict[str, list[str]] = {}

    def split(items: list[Page], prefix: str, remaining: int) -> None:
        if remaining == 0:
            paths.extend(f"{prefix}{leaf_name(p)}" for p in items)
            return
        k = min(len(items), max(1, math.ceil(len(items) / fanout ** remaining)))
        start = 0
        for gi, size in enumerate(_balanced(len(items), k)):
            chunk = items[start : start + size]
            start += size
            name = _folder_name(gi + 1, chunk)
            folders[f"{prefix}{name}"] = list(dict.fromkeys(page_domain(p.url) for p in chunk))
            split(chunk, f"{prefix}{name}/", remaining - 1)

    split(order, "", levels)
    if len(set(paths)) != len(paths):
        raise AssertionError("candidate tree paths collide")
    return CandidateTree(key=key, pages=order, paths=paths, max_depth=levels + 1,
                         folders=folders)


# ---------------------------------------------------------------------------
# The harness side: page lookup, node store, browse view, overview callable
# ---------------------------------------------------------------------------


_BRACKET_ID = re.compile(r"\[([^\[\]]+)\]\s*$")


class CandidateCorpus(WebCorpus):
    """BM25 over the candidate pages only (the full_text arm's ``web_search``), and a lenient
    ``get``: the id, ``[id]``, a leaf name / tree path (``<title> [<id>]``) or a unique title --
    what an agent reading a folder listing (or its reps) may pass to ``web_open``."""

    def __init__(self, pages: list[Page]) -> None:
        super().__init__(pages)
        self._by_title: dict[str, list[Page]] = {}
        for p in pages:
            self._by_title.setdefault(" ".join(p.title.lower().split()), []).append(p)

    def get(self, page_id: str) -> Page | None:
        s = str(page_id).strip()
        hit = super().get(s)
        if hit is not None:
            return hit
        m = _BRACKET_ID.search(s)
        if m:
            hit = super().get(m.group(1).strip())
            if hit is not None:
                return hit
        t = self._by_title.get(" ".join(s.lower().split()))
        return t[0] if t and len(t) == 1 else None


class MemoryNodeCache:
    """The subset of ``bgkit2.data.node_cache.NodeCache`` the student and ``browse`` read,
    in memory: {(key, node): valid reps (K, D)} for per-query trees."""

    def __init__(self) -> None:
        self._d: dict[tuple[str, str], object] = {}
        self.meta: dict = {}

    @property
    def repos(self) -> set[str]:
        return {k for k, _ in self._d}

    def put(self, key: str, nodes: dict) -> None:
        """``nodes``: {node path: (reps (K, D), mask (K,))} as ``TreeOutput.nodes``."""
        for node, (r, m) in nodes.items():
            self._d[(key, node)] = r[m.bool()] if m is not None else r

    def drop(self, key: str) -> None:
        for k in [k for k in self._d if k[0] == key]:
            del self._d[k]

    def __contains__(self, kn) -> bool:
        return tuple(kn) in self._d

    def __len__(self) -> int:
        return len(self._d)

    def get(self, key: str, node: str = ""):
        import torch

        r = self._d[(key, node)]  # KeyError names the missing node
        return r, torch.ones(r.shape[0], dtype=torch.bool, device=r.device)

    def n_reps(self, key: str, node: str = "") -> int:
        return int(self._d[(key, node)].shape[0])


class CandidateTreeView:
    """``RepoTreeView``'s interface (``repo`` / ``paths`` / ``children`` / ``n_reps`` /
    ``listing``) over a candidate tree, for ``ToolRegistry.attach_tree`` + ``browse``. The text
    listing (the teacher's / full-text view of a browse result) names folders with their sites
    and pages as ``[id] title``."""

    def __init__(self, tree: CandidateTree, cache) -> None:
        from ..models.multilevel import node_children

        self.tree, self.repo, self.paths, self._cache = tree, tree.key, list(tree.paths), cache
        self.max_depth = tree.max_depth
        self.children = {n: sorted(c) for n, c in node_children(self.paths,
                                                                  self.max_depth).items()
                          if (self.repo, n) in cache}

    def n_reps(self, node: str = "") -> int:
        return self._cache.n_reps(self.repo, node)

    def listing(self, node: str = "") -> str:
        lines = []
        pre = f"{node}/" if node else ""
        for c in self.children[node]:
            if c.endswith("/"):
                full = pre + c[:-1]
                n = sum(p.startswith(full + "/") for p in self.paths)
                doms = self.tree.folders.get(full, [])
                shown = ", ".join(doms[:4]) + (f", +{len(doms) - 4} more" if len(doms) > 4 else "")
                lines.append(f"{c} ({n} pages: {shown})")
            else:
                p = self.tree.page_of(pre + c)
                lines.append(f"[{p.id}] {p.title}")
        return "\n".join(lines)


def candtree_overview(view: CandidateTreeView) -> Callable:
    """The ``run_episode(overview=...)`` callable of ONE query's tree."""

    def fn(task):
        return view.repo, view.listing(""), view.n_reps("")

    fn.tree = lambda task: view  # type: ignore[attr-defined]
    return fn


# ---------------------------------------------------------------------------
# Encoding (the GPU half; CPU-testable with a tiny encoder)
# ---------------------------------------------------------------------------


def page_source(p: Page) -> str:
    """A page as a leaf reads it -- the same text ``web_open`` returns."""
    return f"# {p.title}\n{p.url}\n\n{p.text}"


def tree_prompt(question: str) -> str:
    """The encoder prompt of every node of a query's tree: the question itself, as
    ``BgKIT2Core.prompted_node`` prompts its question-conditioned read-out."""
    return " ".join(str(question).split())


@dataclass
class TreeEncoding:
    nodes: dict  # {node path: (reps, mask)}
    stats: dict


def encode_candidate_tree(core, tree: CandidateTree, question: str, tok_encode,
                          prompt: str | None = None) -> TreeEncoding:
    """Encode ``tree`` for ``question``: the run's own live-tree settings
    (``core.tree_file_tokens`` / ``core.tree_encode_kwargs``), the live-tree encoder role, and
    the question as the encoder prompt of EVERY node. No gradient."""
    import torch

    from ..models.multilevel import encode_tree

    cap = core.tree_file_tokens()
    raw = [tok_encode(page_source(p)) for p in tree.pages]
    ids = [torch.tensor(x[:cap], dtype=torch.long) for x in raw]
    p_text = tree_prompt(question) if prompt is None else prompt
    p_ids = torch.tensor(tok_encode(p_text)[:PROMPT_MAX_TOKENS], dtype=torch.long)
    kw = core.tree_encode_kwargs(grad_levels=0)
    kw["max_depth"] = tree.max_depth
    t0 = time.perf_counter()
    with torch.no_grad(), core.encoder.role(CANDTREE_ROLE), core.autocast():
        out = encode_tree(core.encoder, ids, tree.paths, tok_encode=tok_encode,
                          prompt_ids=p_ids, **kw)
    cost = tree_encoder_tokens(
        [int(x.shape[0]) for x in ids],
        [len(tok_encode(f"# {p}\n")) for p in tree.paths],
        int(p_ids.shape[0]), leaf_window=int(kw["leaf_window"]),
        ratio_leaf=float(kw["ratio_leaf"]), leaf_max_reps=int(kw["leaf_max_reps"]),
        n_nodes=out.n_nodes + out.n_virtual)
    stats = {
        "n_pages": len(tree.pages), "tree_depth": tree.max_depth,
        "page_tokens": sum(len(x) for x in raw),
        "pages_cut": sum(len(x) > cap for x in raw),
        "n_nodes": out.n_nodes, "n_virtual": out.n_virtual,
        "root_reps": int(out.root.rep_mask[0].sum()),
        "tree_encode_s": round(time.perf_counter() - t0, 3), **cost,
    }
    return TreeEncoding(nodes=dict(out.nodes or {}), stats=stats)


def tree_encoder_tokens(file_lens: list[int], header_lens: list[int], prompt_len: int, *,
                        leaf_window: int, ratio_leaf: float, leaf_max_reps: int,
                        n_nodes: int | None = None) -> dict:
    """Encoder positions one prompted tree costs, from token counts alone (the cost estimate;
    no model). Leaves: every window is ``[prompt | header | window]``; nodes: the survivors of
    the leaves (the dominant part of every directory input) plus a prompt per node."""
    leaf_tok = leaf_rows = leaf_reps = 0
    for n, h in zip(file_lens, header_lens, strict=True):
        w = 1 if leaf_window <= 0 or n <= leaf_window else math.ceil(n / leaf_window)
        leaf_tok += n + w * (h + prompt_len)
        leaf_rows += w
        r = ratio_leaf
        if leaf_max_reps > 0 and n > 0:
            r = min(r, leaf_max_reps / (n + w * h))
        leaf_reps += max(1, round(r * (n + w * h)))
    nodes = n_nodes if n_nodes is not None else max(1, math.ceil(len(file_lens) / FANOUT) + 1)
    node_tok = leaf_reps + nodes * prompt_len
    return {"tree_encoder_tokens": leaf_tok + node_tok, "tree_leaf_tokens": leaf_tok,
            "tree_leaf_windows": leaf_rows, "tree_node_tokens": node_tok}


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


class TreeEncoderCounter(EncoderCounter):
    """``EncoderCounter`` without the cached tree slots: an overview / browse slot holds node
    reps encoded ONCE per query (``tree_encoder_tokens``), not re-read by the encoder per turn."""

    def generate(self, conv: Conversation):
        from ..data.overview import parse_node

        self.calls += 1
        if self.compressed:
            self.encoder_tokens += sum(s.n_tokens for s in conv.slots if parse_node(s.text) is None)
        return self.model.generate(conv)


def candtree_task(item: dict, work_dir: str | Path, arm: str,
                  web_find_mode: str = "encoded") -> AgentTask:
    q = " ".join(str(item["question"]).split())
    return AgentTask(
        task_id=item["id"],
        instruction=f"Question: {item['question'].strip()}\n\n{WEB_INSTRUCTION}",
        repo_dir=str(work_dir),
        system_prompt=(candtree_system(web_find_mode) if arm == "bgkit_tree"
                       else web_system(web_find_mode)),
        summary=q[:200],
    )


def run_candtree_arm(
    item: dict,
    arm: str,
    model: Model,
    work_dir: str | Path,
    *,
    tree: CandidateTree | None = None,
    view: CandidateTreeView | None = None,
    tree_stats: dict | None = None,
    ratios: RatioPolicy | None = None,
    renderer: Renderer | None = None,
    full_text_results: str = "match",
    web_k: int = 5,
    max_page_chars: int = 40_000,
    snippet_words: int = SNIPPET_WORDS,
    tree_search: bool = False,
    compaction_mode: str = WEB_COMPACTION_MODE,
    slot_max_tokens: int | None = WEB_SLOT_MAX_TOKENS,
    log: TrajectoryLog | None = None,
    web_find_mode: str = "encoded",
    **episode_kwargs,
) -> dict:
    """One (arm, item) episode over the item's candidate set -> record. ``bgkit_tree`` needs
    the query's encoded ``view`` (its node reps already in the student's node cache).
    ``web_find_mode="verbatim"`` = the verbatim-find control arm (tool + system prompt)."""
    if arm not in CANDTREE_ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {CANDTREE_ARMS}")
    if (arm == "bgkit_tree") != bool(model.compressed):
        raise ValueError(f"arm {arm} needs a {'compressed' if arm == 'bgkit_tree' else 'text'} "
                         "model")
    if arm == "bgkit_tree" and view is None:
        raise ValueError("bgkit_tree needs the query's encoded CandidateTreeView")
    ratios = ratios or RatioPolicy()
    renderer = renderer or model.renderer
    pages = tree.pages if tree is not None else item_pages(item)
    corpus = CandidateCorpus(pages)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    reg = ToolRegistry(str(work_dir), web_find_mode=web_find_mode)
    reg.attach_web(corpus, k=web_k, max_page_chars=max_page_chars, snippet_words=snippet_words)
    if arm == "bgkit_tree":
        # `browse` joins in run_episode (attach_tree), after this filter
        reg.keep_only({"web_open", "web_find", *(("web_search",) if tree_search else ())})
    else:
        reg.keep_only(WEB_TOOL_NAMES)
    tools = EpisodeTools(reg, corpus, renderer, ratios,
                         truncate_to_reps=arm == "full_text" and full_text_results == "match")
    enc = TreeEncoderCounter(model)
    task = candtree_task(item, work_dir, arm, web_find_mode)
    res = run_episode(enc, tools, task, ratios=ratios,
                      renderer=renderer, log=log, compaction_mode=compaction_mode,
                      slot_max_tokens=slot_max_tokens,
                      overview=candtree_overview(view) if arm == "bgkit_tree" else None,
                      **episode_kwargs)
    rec = episode_record(item, arm, res, tools, enc)
    rec["web_find_mode"] = web_find_mode
    ids = {p.id for p in pages}
    rec["n_browse"] = tools.n_calls.get("browse", 0)
    rec["n_candidates"] = len(pages)
    rec["gold_in_candidates"] = bool(set(rec["gold_ids"]) & ids)
    if arm == "bgkit_tree":
        st = dict(tree_stats or {})
        rec["tree"] = st
        rec["encoder_tokens_total"] = rec["encoder_tokens"] + int(st.get("tree_encoder_tokens", 0))
    else:
        rec["encoder_tokens_total"] = 0
    return rec


def run_candtree_item(
    item: dict,
    arms: list[str],
    make_model: Callable[[str], Model],
    work_dir: str | Path,
    *,
    encode: Callable[[CandidateTree, str], TreeEncoding] | None = None,
    cache: MemoryNodeCache | None = None,
    fanout: int = FANOUT,
    done: dict[str, set[str]] | None = None,
    write: Callable[[str, dict], None] | None = None,
    logs: dict[str, TrajectoryLog] | None = None,
    **arm_kwargs,
) -> list[dict]:
    """Every not-yet-done arm of one item. ``bgkit_tree`` builds the query's tree, encodes it
    with ``encode(tree, question)`` into ``cache`` (the student's node cache) and drops it after
    the episode. An exception becomes an ``error`` record (retried on resume)."""
    done = done or {}
    tree = build_candidate_tree(item_pages(item), key=f"candtree/{item['id']}", fanout=fanout)
    recs = []
    for arm in arms:
        if item["id"] in done.get(arm, set()):
            continue
        view = stats = None
        try:
            if arm == "bgkit_tree":
                if encode is None or cache is None:
                    raise ValueError("bgkit_tree needs encode= and cache=")
                te = encode(tree, item["question"])
                cache.put(tree.key, te.nodes)
                view, stats = CandidateTreeView(tree, cache), te.stats
            rec = run_candtree_arm(item, arm, make_model(arm), work_dir, tree=tree, view=view,
                                   tree_stats=stats, log=(logs or {}).get(arm), **arm_kwargs)
        except Exception as exc:  # server hiccup / OOM guard etc.
            rec = error_record(item, arm, exc)
        finally:
            if cache is not None:
                cache.drop(tree.key)
        recs.append(rec)
        if write is not None:
            write(arm, rec)
    return recs


def candtree_summary_extra(records: list[dict]) -> dict:
    """Candidate-tree metrics beside ``web_eval.summarize``: browse use and tree cost."""
    ok = [r for r in records if r.get("stop") != "error"]
    if not ok:
        return {}

    def mean(xs):
        xs = [float(x) for x in xs]
        return round(sum(xs) / len(xs), 4) if xs else None

    out = {"n_browse": mean(r.get("n_browse", 0) for r in ok),
           "n_candidates": mean(r.get("n_candidates", 0) for r in ok),
           "gold_in_candidates": mean(r.get("gold_in_candidates", False) for r in ok),
           "encoder_tokens_total": mean(r.get("encoder_tokens_total", 0) for r in ok)}
    trees = [r["tree"] for r in ok if r.get("tree")]
    if trees:
        for k in ("tree_encoder_tokens", "page_tokens", "root_reps", "n_nodes", "n_virtual",
                  "pages_cut", "tree_encode_s"):
            out[f"tree_{k}" if not k.startswith("tree_") else k] = mean(t.get(k, 0)
                                                                       for t in trees)
    out["browse_turns"] = dict(Counter(min(int(r.get("n_browse", 0)), 5) for r in ok))
    return out
