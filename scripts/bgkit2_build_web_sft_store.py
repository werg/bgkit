#!/usr/bin/env python
"""Public WEB-RESEARCH SFT trajectories -> bgkit2 ``tool_slots`` rows over OUR offline web tools
(2026-09-22). CPU-only, host venv; big files are streamed (ijson / pyarrow row groups).

Every source's search / open-page / find call is rewritten onto ``web_search`` / ``web_open`` /
``web_find`` (bgkit2.agent.web) and every row is framed exactly as the offline web eval runs an
episode (bgkit2.agent.web_eval): system ``WEB_SYSTEM``, user ``"Question: <q>\\n\\n" +
WEB_INSTRUCTION``, tool list = ``web_tool_specs()`` only, and the final turn ``<thinking>\\n\\n
ANSWER: <answer>`` with no tool call. ``web_search`` / ``web_open`` results are COMPRESSED SLOTS
(the sentinel in the message, the text in ``context``) whose encoder prompt is built the way the
harness builds it (``ToolRegistry._prompt``: ``"web search <query>: <focus>"``, ``"page <title>:
<focus>"``). ``web_find`` (2026-09-22, user: exact-quote tools go through bgkit too) is a slot
over the WHOLE page it searched -- the page as the trajectory shows it (every window the agent
viewed, plus the page lines the find itself displayed) -- prompted ``"find <pattern> in <title>:
<focus>"`` like the harness's encoded ``web_find``; the call names ``id`` when that page is not
the one opened last. A find whose page cannot be resolved keeps its verbatim result as inline
text (counted: ``find:inline_unresolved``), a find with no match is the harness's inline "not
found". Errors and empty results stay inline text. The trajectory's own tool results carry the
text (the ids are replay labels: the source's doc id or url).

Sources (``--source``; family = store name):

  search_r1      ThornZ/Search-R1-SFT (qwen3-4b + qwen2.5-7b files, deduped by question,
                 qwen3 preferred). Gold recovered from nq_open + HotpotQA by question.
  sds            RUC-AIBOX/0.8k-data-SimpleDeepSearcher (search only; each hit shown as the
                 harness would: ``snippet(page, query)``).
  webshaper      rl-rag/webshaper-gpt-oss-120b-260222 (gpt-oss browser.* calls; the judge's
                 verdict and the dataset's contamination flag are recorded in meta).
  openseeker / miroverse_nc
                 Zephyr271828/openseeker-miroverse-mix-full split by its ``source`` column:
                 OpenSeeker rows -> web_openseeker, MiroVerse rows (cc-by-nc-4.0) ->
                 web_miroverse_nc. search(query=[...]) -> one web_search per query,
                 visit(url, goal) -> web_open per url with the goal as focus.
  openresearcher OpenResearcher/OpenResearcher-Dataset (gpt-oss browser.* over its own corpus;
                 questions compiled from MiroVerse -> flagged NC-derived).

  (aidenjhwu/ASearcher_en_no-math_Qwen3-8B-reject-sample has NO trajectories -- an RL prompt
  set of question + ground truth -- so there is nothing to convert.)

Row builder (``replay_rows``, 2026-09-22 v3): every trajectory is replayed WHOLE through the
harness's own ``bgkit2.agent.context.Conversation`` exactly as ``run_episode`` runs a web-eval
episode: tool results become slots / inline text as the harness returns them, and before every
assistant turn ``compact(--context-budget, --keep-recent, compressed=True)`` folds the oldest
rounds into the compaction slot (prompt ``compaction: <question[:200]>``), so long histories are
handled by the SAME compaction at train and eval time (meta.compacted / n_compactions /
compaction_slot_tokens). No correctness / per-question / length filtering: correctness is
recorded (meta.answer_match, gold, judged_correct; flags pure_math, citation_hunting,
contaminated). The source's answer-FORMAT instructions are stripped from the question (framing).
Benchmark exclusion (showcase / web / BrowseComp-Plus questions plus the full HotpotQA-distractor
validation and MuSiQue dev sets, which the web eval's paragraph pool is built from) stays;
split = question hash (1% eval). ``--max-rows-per-traj`` (0 = every turn) samples turns by depth.

Reasoning (``--keep-reasoning``, 2026-09-26: the LFM2.5-8B-A1B decoder runs in its native
always-thinking mode). By default each turn's reasoning -- Search-R1 ``<thinking>``, the
SimpleDeepSearcher think block, the gpt-oss analysis channel (WebShaper, OpenResearcher),
``reasoning_content`` (OpenSeeker) -- is merged into ``content`` as untagged prose. With the flag
it goes to the assistant message's ``reasoning_content`` (stripped; the chat template renders
``<think>...</think>`` before the content) in the target AND the history, as the harness keeps a
thinking decoder's reasoning; ``content`` keeps only what the model said plus ``ANSWER:``.
Slots, prompts, focus and the question are unchanged; compaction folds the visible text only.
The prefix count then includes every turn's reasoning (8B template semantics), so the check
renders with ``preserve_thinking`` (``verify_thinking_row``).

    .venv/bin/python scripts/bgkit2_build_web_sft_store.py --source search_r1
"""

from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import random
import re
import string
import sys
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bgkit2.agent.replay import FastRenderer, _Counted, spine_meta  # noqa: F401 - re-exported
from bgkit2.agent.tools import ToolRegistry
from bgkit2.agent.web import SNIPPET_WORDS, snippet, web_find_prompt, web_tool_specs
from bgkit2.agent.web_eval import WEB_INSTRUCTION, WEB_SYSTEM
from bgkit2.data.templates import SPLICE_SENTINEL

WEB_SFT = "/mnt/external/bgkit-data/web_sft"
TASKS = "/home/werg/bgkit-data-nvme/bgkit2/tasks"
BENCH = "/home/werg/bgkit-data-nvme/bgkit2/benchmarks"
HF = str(Path.home() / ".cache/huggingface/hub")
MAX_PAGE_CHARS = 40_000  # ToolRegistry.attach_web / web_eval default
HIT_CHARS = 7 * SNIPPET_WORDS  # a search hit's passage, the width web_search shows at eval
FOCUS_CHARS = 200

SOURCES = {
    # name: (family, hf repo, license note)
    "search_r1": (
        "web_search_r1",
        "ThornZ/Search-R1-SFT",
        "unspecified on the card (Search-R1 NQ/HotpotQA train questions)",
    ),
    "sds": ("web_sds", "RUC-AIBOX/0.8k-data-SimpleDeepSearcher", "unspecified on the card"),
    "webshaper": (
        "web_webshaper",
        "rl-rag/webshaper-gpt-oss-120b-260222",
        "unspecified on the card (questions: Alibaba-NLP/WebShaper; gpt-oss-120b outputs)",
    ),
    "openseeker": (
        "web_openseeker",
        "Zephyr271828/openseeker-miroverse-mix-full",
        "card: 'other' (OpenSeeker half: Zephyr271828/openseeker_v1_sft, unspecified)",
    ),
    "miroverse_nc": (
        "web_miroverse_nc",
        "Zephyr271828/openseeker-miroverse-mix-full",
        "NON-COMMERCIAL: derived from miromind-ai/MiroVerse-v0.1 (cc-by-nc-4.0)",
    ),
    "openresearcher": (
        "web_openresearcher",
        "OpenResearcher/OpenResearcher-Dataset",
        "mit on the card, BUT questions compiled from MiroVerse-v0.1 (cc-by-nc-4.0) -> NC-derived",
    ),
}


# ---------------------------------------------------------------------------
# Episode: the source-independent shape every parser produces
# ---------------------------------------------------------------------------


@dataclass
class Page:
    """A page the agent viewed; later views of the SAME page (gpt-oss scroll / re-open) merge
    their windows here, because our web_open returns the whole page in one slot."""

    id: str
    title: str
    url: str
    lines: dict[int, str] = field(default_factory=dict)
    text: str = ""  # sources that deliver the page body at once

    def render(self) -> str:
        body = self.text if self.text else join_windows(self.lines)
        return f"# {self.title}\n{self.url}\n\n{body}"[:MAX_PAGE_CHARS]


@dataclass
class Call:
    tool: str  # web_search | web_open | web_find
    args: dict  # without focus
    focus: str = ""
    text: str = ""  # search / find / error text
    page: Page | None = None  # web_open, and a web_find whose page was resolved (its slot)
    ok: bool = True
    note: str = ""  # web_find: slot | inline_unresolved | not_found | error (counted)

    def result(self) -> str:
        return self.page.render() if self.page is not None else self.text

    def is_slot(self) -> bool:
        """Compressible exactly when the harness's result would be: web_search / web_open /
        web_find (encoded, over a resolved page) that did not error (a search with no hits is
        still a compressible "no results" result)."""
        if self.tool not in ("web_search", "web_open", "web_find") or not self.result().strip():
            return False
        if self.tool == "web_find":
            return self.ok and self.page is not None
        return self.ok or self.text.startswith("no results for")

    def prompt(self) -> str:
        if self.tool == "web_search":
            return ToolRegistry._prompt(f"web search {self.args['query']}", self.focus)
        if self.tool == "web_find":
            return web_find_prompt(self.args["pattern"], self.page.title if self.page else "",
                                   self.focus)
        return ToolRegistry._prompt(f"page {self.page.title if self.page else ''}", self.focus)

    def message(self) -> dict:
        a = dict(self.args)
        if self.tool in ("web_search", "web_open", "web_find") and self.focus:
            a["focus"] = self.focus
        return {"type": "function", "function": {"name": self.tool, "arguments": a}}


@dataclass
class Step:
    """One assistant turn of the source. ``thought`` is everything the turn wrote, reasoning and
    visible text merged (the default build's content); ``reasoning`` / ``said`` are the same text
    kept apart (``--keep-reasoning``: reasoning -> ``reasoning_content``, said -> content)."""

    thought: str
    calls: list[Call]
    reasoning: str = ""
    said: str = ""


@dataclass
class Episode:
    traj_id: str
    question: str
    answer: str  # what goes after ANSWER:
    steps: list[Step]
    final_thought: str = ""
    gold: list[str] | None = None
    final_reasoning: str = ""  # final_thought kept apart (see Step)
    final_said: str = ""
    pred: str = ""  # the source's own final answer (for the correctness filter)
    judged_correct: bool | None = None  # a source-provided verdict (webshaper's judge)
    meta: dict = field(default_factory=dict)


class DropError(Exception):
    """A trajectory the converter cannot map faithfully (the reason is counted)."""


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def normalize_answer(s: str) -> str:
    s = "".join(ch for ch in str(s).lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def norm_q(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(s).lower()).split())


def token_f1(p: str, g: str) -> float:
    pw, gw = p.split(), g.split()
    common = sum((Counter(pw) & Counter(gw)).values())
    if not common:
        return 0.0
    pr, rc = common / len(pw), common / len(gw)
    return 2 * pr * rc / (pr + rc)


def answer_matches(pred: str, golds: list[str], lenient: bool = False) -> str | None:
    """The criterion a prediction meets against any gold alias, or None: SQuAD ``em``, or the
    gold contained as whole words in the prediction (``gold_in_pred``). ``lenient`` (sources
    whose gold is a free-form SENTENCE -- MiroVerse-derived: OpenResearcher, the mix) adds the
    prediction contained in the gold (``pred_in_gold``, >= 3 chars) and token F1 >= 0.6."""
    p = normalize_answer(pred)
    if not p:
        return None
    for g in golds:
        gn = normalize_answer(g)
        if not gn:
            continue
        if gn == p:
            return "em"
        if f" {gn} " in f" {p} ":
            return "gold_in_pred"
    if lenient:
        for g in golds:
            gn = normalize_answer(g)
            if gn and len(p) >= 3 and f" {p} " in f" {gn} ":
                return "pred_in_gold"
            if gn and token_f1(p, gn) >= 0.6:
                return "f1"
    return None


def clip_words(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= n:
        return text
    cut = text[:n]
    return cut[: cut.rfind(" ")] if " " in cut else cut


def focus_of(question: str) -> str:
    return clip_words(question, FOCUS_CHARS)


def url_title(url: str) -> str:
    """A readable page title from a url (the sources that give none): last path segment."""
    try:
        u = urlparse(url)
    except ValueError:
        return url
    seg = [s for s in unquote(u.path).split("/") if s]
    if seg:
        t = re.sub(r"\.(html?|php|aspx?|pdf)$", "", seg[-1], flags=re.I)
        t = " ".join(re.sub(r"[_\-+]+", " ", t).split())
        if t and not t.isdigit():
            return t
    return u.netloc or url


_CITE = re.compile(r"【[^】\n]*】")
_THINK_TAGS = re.compile(r"</?(think|thinking)>")


def clean_thought(text: str) -> str:
    """Model reasoning as decoder content: no think tags (the LFM template treats ``</think>``
    specially), no gpt-oss cursor citations (they point at cursor numbers we do not show)."""
    t = _THINK_TAGS.sub("", str(text or ""))
    t = _CITE.sub("", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def join_wrapped(lines: list[str]) -> str:
    """gpt-oss browser displays wrap text at ~80 chars keeping the separating space on one side;
    short lines are real line breaks."""
    out = ""
    prev = None
    for ln in lines:
        if prev is None:
            out = ln
        elif not ln.strip():
            out += "\n"
        elif len(prev.strip()) >= 40 and (prev.endswith(" ") or ln.startswith(" ")):
            out += ln
        else:
            out += "\n" + ln
        prev = ln
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def join_windows(lines: dict[int, str]) -> str:
    """Viewed line windows of one page, in line order, gaps marked."""
    if not lines:
        return ""
    keys = sorted(lines)
    runs: list[list[str]] = [[]]
    for i, k in enumerate(keys):
        if i and k != keys[i - 1] + 1:
            runs.append([])
        runs[-1].append(lines[k])
    return "\n...\n".join(join_wrapped(r) for r in runs)


def search_text(hits: list[tuple[str, str, str]]) -> str:
    """web_search's result format: ``[<id>] <title>\\n<passage>`` per hit, blank-line joined."""
    return "\n\n".join(f"[{i}] {t}\n{clip_words(x, HIT_CHARS)}".rstrip() for i, t, x in hits)


# --- source-artefact filters (2026-09-22 audit) --------------------------------
# (1) the source's answer-FORMAT instructions inside the question ("wrap the final answer in
#     \boxed{}", "Exact Answer:" templates) contradict our ANSWER: framing and would ride into
#     every slot's focus prompt -> stripped (clause first, else the sentence);
# (2) gpt-oss / OpenResearcher agents were told to CITE evidence, which drives citation-hunting
#     tool calls our task never asks for -> a trajectory with any citation-motivated call turn
#     is dropped (before the per-question cap, so another seed's trajectory can take its place);
#     citation sentences in the kept final turns are removed;
# (3) pure-math questions (equations / LaTeX, no named entity) need no web lookup -> dropped.
_FMT = re.compile(
    r"\\boxed|format instruction|\bExact Answer\s*:|\bExplanation\s*:|\bConfidence\s*:|<answer>"
    r"|\bwrap(?:ped|ping)?\b[^.]{0,60}\banswer|\banswer\b[^.]{0,40}\bwrap",
    re.I,
)
_FMT_CLAUSE = re.compile(
    r"\s*[,;:(]\s*(?:and\s+)?(?:following|in|using|provide|ensure|wrap|with|as per|"
    r"suitable|ready|for use|formatted|then|as the)\b[^.]*\.?$",
    re.I,
)


def clean_question(q):
    q = str(q).strip()
    if not _FMT.search(q):
        return q
    parts = re.split(r"(?<=[.!?])\s+|\n+", q)
    out = []
    for s in parts:
        if not _FMT.search(s):
            out.append(s)
            continue
        c = _FMT_CLAUSE.sub("", s).strip()
        if c and not _FMT.search(c) and len(c.split()) >= 4:
            out.append(c.rstrip(",;: ") + ("" if c.rstrip()[-1] in ".?!" else "."))
    cleaned = " ".join(x for x in out if x).strip()
    return cleaned if len(cleaned.split()) >= 3 else q  # never empty a question


_MATHY = re.compile(
    r"\\\(|\\\[|\$|\\(?:frac|sqrt|int|sum|cdot|times|pi|log|delta|alpha)\b|\b[a-z]\s*(?:\^|=)|=\s*[\d(a-z]"
    r"|\b[a-z]\s*[\+\*]\s*\d|\d\s*[\+\*\^]\s*\d"
)
_MATH_WORDS = {
    "Solve",
    "Find",
    "Compute",
    "Calculate",
    "Evaluate",
    "Simplify",
    "What",
    "Let",
    "If",
    "Determine",
    "Given",
    "Express",
    "Write",
    "The",
    "A",
    "An",
    "How",
    "Suppose",
    "Show",
    "Prove",
    "Factor",
    "Expand",
    "Round",
    "Is",
    "Which",
    "For",
    "In",
    "Two",
    "Three",
    "Each",
    "Then",
    "Assume",
    "Consider",
    "Use",
    "Using",
    "Answer",
    "You",
}


def pure_math(q):
    if not _MATHY.search(q):
        return False
    s = re.sub(r"\\\(.*?\\\)|\\\[.*?\\\]|\$[^$]*\$", " ", q, flags=re.S)
    toks = re.findall(r"\b[A-Z]\w+", s)
    return not [t for t in toks if t not in _MATH_WORDS]


CITE_HUNT = re.compile(
    r"\b(?:need|needs|needed|require[sd]?|want|get|obtain|find|gather|collect|provide|include|add"
    r"|use|grab|capture)\b[^.\n]{0,25}\bcitations?\b"
    r"|\bfor (?:the )?citations?\b|\bgood for citations?\b"
    r"|\b(?:we(?:'ll| will| can| should| must| need to| may)|let's|to|must|should) (?:also )?cite\b"
    r"|\bcite (?:the |these |those |both |all |each |a |our )?(?:sources?|lines?|evidence|passages?"
    r"|them|it|both|each)\b"
    r"|\bcitations? (?:for|from|lines?) (?:each|every|all|the)\b|\bsources? for each\b",
    re.I,
)


def citation_hunting(ep) -> str | None:
    """The matched phrase when any tool-call turn's reasoning is about gathering citations."""
    for st in ep.steps:
        m = CITE_HUNT.search(st.thought)
        if m:
            return m.group(0)
    return None


def strip_citation_sentences(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return "\n".join(x for x in parts if x.strip() and not CITE_HUNT.search(x)).strip()


# ---------------------------------------------------------------------------
# Benchmark exclusion
# ---------------------------------------------------------------------------


class Excluder:
    """Normalised benchmark questions: a row is dropped when its question equals, contains, or
    is contained in (>= 20 chars) one of them. N-gram keyed so 100k rows x 14k questions is
    cheap."""

    K = 4

    def __init__(self, questions: dict[str, str]) -> None:
        self.src = {q: s for q, s in questions.items() if len(q) >= 20}
        self.by_head: dict[tuple, list[str]] = {}
        self.by_gram: dict[tuple, set[str]] = {}
        for q in self.src:
            w = q.split()
            self.by_head.setdefault(tuple(w[: self.K]), []).append(q)
            for i in range(max(1, len(w) - self.K + 1)):
                self.by_gram.setdefault(tuple(w[i : i + self.K]), set()).add(q)
        self.lens = sorted({min(len(q.split()), self.K) for q in self.src})

    def hit(self, question: str) -> str | None:
        qn = norm_q(question)
        if qn in self.src:
            return self.src[qn]
        w = qn.split()
        pad = f" {qn} "
        for n in self.lens:
            for i in range(0, max(0, len(w) - n) + 1):
                for b in self.by_head.get(tuple(w[i : i + n]), ()):
                    if len(b) <= len(qn) and f" {b} " in pad:
                        return self.src[b]
        if len(qn) >= 20 and len(w) >= self.K:
            for b in self.by_gram.get(tuple(w[: self.K]), ()):
                if f" {qn} " in f" {b} ":
                    return self.src[b]
        return None


def load_excluder() -> Excluder:
    qs: dict[str, str] = {}

    def jsonl(path: str):
        """Streamed: some bench files carry MB of pages per line (candtree / oracle items);
        loading them whole took the builder to ~4.7 GB before the first trajectory."""
        with open(path) as fh:
            for ln in fh:
                if ln.strip():
                    yield json.loads(ln)

    for f in sorted(glob.glob(f"{BENCH}/showcase/*.jsonl") + glob.glob(f"{BENCH}/web/*.jsonl")):
        for r in jsonl(f):
            qs.setdefault(norm_q(r.get("question", "")), Path(f).stem)
    for r in jsonl(f"{BENCH}/browsecomp_plus/queries.jsonl"):
        qs.setdefault(norm_q(r.get("query", "")), "browsecomp_plus")
    import pyarrow.parquet as pq

    for p in glob.glob(f"{HF}/datasets--hotpotqa--hotpot_qa/snapshots/*/distractor/validation-*"):
        for q in pq.read_table(p, columns=["question"]).column("question").to_pylist():
            qs.setdefault(norm_q(q), "hotpotqa_validation")
    for p in glob.glob(f"{HF}/datasets--dgslibisey--MuSiQue/snapshots/*/musique_ans_v1.0_dev*"):
        for r in jsonl(p):
            qs.setdefault(norm_q(r["question"]), "musique_dev")
    qs.pop("", None)
    return Excluder(qs)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

# --- Search-R1 --------------------------------------------------------------

_R1_DOC = re.compile(
    r"\[Document \d+\]\s*\(ID:\s*([^,)]+)[^)]*\)\s*\n\"?(.*?)\"?\n(.*?)"
    r"(?=\n\s*\[Document \d+\]|\Z)",
    re.S,
)


def r1_documents(info: str) -> list[tuple[str, str, str]]:
    info = info.replace("<information>", "").replace("</information>", "").strip()
    hits = []
    for pid, title, text in _R1_DOC.findall(info):
        title, text = title.strip().strip('"'), " ".join(text.split())
        if title and text.startswith(title + " "):
            text = text[len(title) + 1 :]
        hits.append((pid.strip(), title, text))
    return hits


def _tag(text: str, name: str) -> str | None:
    m = re.search(rf"<{name}>(.*?)</{name}>", text, re.S)
    return m.group(1).strip() if m else None


def parse_search_r1(rec: dict, tid: str) -> Episode:
    msgs = rec["messages"]
    users = [m for m in msgs if m["role"] == "user"]
    if not users or "Question:" not in users[0]["content"]:
        raise DropError("no_question")
    question = users[0]["content"].split("Question:", 1)[1].strip()
    steps: list[Step] = []
    i = msgs.index(users[0]) + 1
    answer = None
    final_thought = ""
    while i < len(msgs):
        m = msgs[i]
        if m["role"] != "assistant":
            raise DropError("unexpected_role")
        text = m["content"]
        thought = _tag(text, "thinking") or _tag(text, "think") or ""
        q = _tag(text, "search")
        a = _tag(text, "answer")
        if a is not None and q is None:
            answer, final_thought = a, thought
            if i != len(msgs) - 1:
                raise DropError("answer_not_last")
            break
        if q is None:
            raise DropError("no_action")
        if i + 1 >= len(msgs) or msgs[i + 1]["role"] != "user":
            raise DropError("search_without_result")
        hits = r1_documents(msgs[i + 1]["content"])
        call = Call(
            "web_search",
            {"query": q},
            focus=focus_of(question),
            text=search_text(hits) if hits else f"no results for {q!r}",
            ok=bool(hits),
        )
        t = clean_thought(thought)  # the <thinking> block is all the turn says besides its call
        steps.append(Step(t, [call], reasoning=t))
        i += 2
    if answer is None:
        raise DropError("no_final_answer")
    t = clean_thought(final_thought)
    return Episode(tid, question, answer, steps, t, pred=answer, final_reasoning=t)


def iter_search_r1(args) -> Iterator[tuple[str, dict]]:
    import ijson

    d = f"{WEB_SFT}/ThornZ__Search-R1-SFT"
    for tag in ("qwen3-4b-instruct-sft", "qwen2.5-7b-instruct-sft"):
        with open(f"{d}/{tag}.json", "rb") as fh:
            for n, rec in enumerate(ijson.items(fh, "item")):
                yield f"search_r1/{tag}/{n}", rec


def load_gold_nq_hotpot() -> dict[str, list[str]]:
    """Search-R1's questions are the NQ + HotpotQA TRAIN sets (FlashRAG); their gold answers
    are recoverable by question from the local HF caches."""
    import pyarrow.parquet as pq

    gold: dict[str, list[str]] = {}
    nq = f"{HF}/datasets--google-research-datasets--nq_open/snapshots/*/nq_open/*.parquet"
    for p in glob.glob(nq):
        t = pq.read_table(p, columns=["question", "answer"]).to_pylist()
        for r in t:
            gold.setdefault(norm_q(r["question"]), []).extend(r["answer"])
    for p in glob.glob(f"{HF}/datasets--hotpotqa--hotpot_qa/snapshots/*/distractor/*.parquet"):
        t = pq.read_table(p, columns=["question", "answer"]).to_pylist()
        for r in t:
            gold.setdefault(norm_q(r["question"]), []).append(r["answer"])
    return gold


# --- SimpleDeepSearcher -----------------------------------------------------

_SDS_PAGE = re.compile(r"\*\*Web Page \d+:\*\*\s*\n")
_SDS_Q = re.compile(r"<\|begin_search_query\|>.*?(<\|end_search_query\|>|$)", re.S)


def sds_pages(doc: str) -> list[dict]:
    out = []
    for block in _SDS_PAGE.split(doc)[1:]:
        block = block.strip()
        try:
            out.append(json.loads(block))
            continue
        except json.JSONDecodeError:
            pass
        d = {}
        for k in ("title", "url", "snippet", "context"):
            m = re.search(rf'"{k}":\s*"((?:[^"\\]|\\.)*)', block, re.S)
            if m:
                try:
                    d[k] = json.loads(f'"{m.group(1)}"')
                except json.JSONDecodeError:
                    d[k] = m.group(1)
        if d.get("url"):
            out.append(d)
    return out


def _boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    k = j
    while k < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        k += 1
    ans = text[j : k - 1] if depth == 0 else text[j:]
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    return ans.strip()


def sds_split(text: str, in_think: bool) -> tuple[str, str, bool]:
    """(reasoning, visible text, still in the think block) of one SimpleDeepSearcher generation:
    the model (QwQ-style) starts inside its think block and closes it once, with ``</think>``,
    in the generation that answers; everything before the close is reasoning."""
    if not in_think:
        return "", clean_thought(text), False
    if "</think>" in text:
        r, said = text.split("</think>", 1)
        return clean_thought(r), clean_thought(said), False
    return clean_thought(text), "", True


def parse_sds(rec: dict, tid: str) -> Episode:
    question = str(rec["question"]).strip()
    out = rec["output"]
    steps: list[Step] = []
    pending_thought: str | None = None
    final = None
    in_think = True  # the generation opens inside the think block (QwQ-style)
    i = 0
    while i < len(out):
        o = out[i]
        if "gen" in o:
            g = o["gen"]
            if "<|begin_search_query|>" in g:
                pending_thought = _SDS_Q.sub("", g)
            else:
                final = g
            i += 1
        elif "search_query" in o:
            q = str(o["search_query"]).strip()
            doc = out[i + 1].get("doc", "") if i + 1 < len(out) else ""
            pages = sds_pages(doc)
            hits = [
                (
                    p.get("url", ""),
                    str(p.get("title", "")).strip(),
                    snippet(str(p.get("context") or p.get("snippet") or ""), q),
                )
                for p in pages
                if p.get("url")
            ]
            call = Call(
                "web_search",
                {"query": q},
                focus=focus_of(question),
                text=search_text(hits) if hits else f"no results for {q!r}",
                ok=bool(hits),
            )
            r, said, in_think = sds_split(pending_thought or "", in_think)
            steps.append(Step(clean_thought(pending_thought or ""), [call], r, said))
            pending_thought = None
            i += 2
            if i < len(out) and "doc_gen" in out[i]:
                i += 1  # the reasoning model's summary of the pages: the slot replaces it
        else:
            i += 1
    if final is None:
        raise DropError("no_final_gen")
    ans = _boxed(final)
    if not ans:
        raise DropError("no_boxed_answer")
    head = final[: final.rfind("\\boxed{")]
    r, said, _ = sds_split(head, in_think)
    gold = [str(rec["answer"])] if rec.get("answer") else None
    return Episode(tid, question, ans, steps, clean_thought(head), gold=gold, pred=ans,
                   final_reasoning=r, final_said=said)


def iter_sds(args) -> Iterator[tuple[str, dict]]:
    import ijson

    path = f"{WEB_SFT}/RUC-AIBOX__0.8k-data-SimpleDeepSearcher/0.8k_train_data.json"
    with open(path, "rb") as fh:
        for rec in ijson.items(fh, "item"):
            yield f"sds/{rec['id']}", rec


# --- gpt-oss browser displays (OpenResearcher, WebShaper) --------------------

_LNUM = re.compile(r"^L(\d+): ?(.*)$")
_LINK = re.compile(r"【(\d+)†([^†】]*)(?:†([^】]*))?】")
_CURSOR = re.compile(r"^\[(\d+)\]\s*(.*?)\s*$")


def display_lines(text: str) -> tuple[str, dict[int, str]]:
    """(header, {line number: text}) of a gpt-oss browser display."""
    header_lines, lines = [], {}
    for raw in text.split("\n"):
        m = _LNUM.match(raw)
        if m:
            lines[int(m.group(1))] = m.group(2)
        elif not lines:
            header_lines.append(raw)
    return "\n".join(header_lines).strip(), lines


def parse_search_display(text: str) -> tuple[int | None, list[dict]]:
    """Cursor + hits (k, id, title, url, text) of a browser.search display. OpenResearcher
    marks hits ``【k†Doc N†domain】 text``, WebShaper ``【k†Title】`` + ``URL: u`` + snippet."""
    header, lines = display_lines(text)
    m = _CURSOR.match(header.split("\n")[0]) if header else None
    cursor = int(m.group(1)) if m else None
    hits: list[dict] = []
    for _n, ln in sorted(lines.items()):
        s = ln.strip()
        lk = _LINK.search(s)
        if lk and s.lstrip("* ").startswith("【"):
            a, b = lk.group(2).strip(), (lk.group(3) or "").strip()
            hits.append(
                {"k": int(lk.group(1)), "a": a, "b": b, "url": "", "lines": [s[lk.end() :].strip()]}
            )
            continue
        if not hits:
            continue
        if s.startswith("URL: ") and not hits[-1]["url"]:
            hits[-1]["url"] = s[5:].strip()
        else:
            hits[-1]["lines"].append(ln)
    out = []
    for h in hits:
        md = re.fullmatch(r"Doc (\S+)", h["a"])
        if md:
            pid, title = md.group(1), h["b"] or h["a"]
        else:
            pid, title = (h["url"] or h["a"]), h["a"]
        body = _LINK.sub(lambda x: x.group(2), join_wrapped(h["lines"]))
        out.append({"k": h["k"], "id": pid, "title": title, "url": h["url"], "text": body})
    return cursor, out


def parse_page_display(text: str) -> tuple[int | None, str, str, dict[int, str]]:
    """(cursor, header title, header url, lines) of a browser.open display."""
    header, lines = display_lines(text)
    first = header.split("\n")[0] if header else ""
    cursor, title, url = None, "", ""
    m = _CURSOR.match(first)
    if m:
        cursor = int(m.group(1))
        rest = m.group(2)
        mu = re.match(r"^(.*?)\s*\((\S+)\)\s*$", rest)
        title, url = (mu.group(1), mu.group(2)) if mu else (rest, "")
    clean = {}
    for k, v in lines.items():
        if v.strip() == f"URL: {url}" and url:
            v = ""  # the header already names the url; blank keeps the window contiguous
        clean[k] = _LINK.sub(lambda x: x.group(2), v)
    return cursor, title.strip(), url.strip(), clean


_FIND_MATCH = re.compile(r"^#\s*【\d+†match at L(\d+)】")
_FIND_HEADER = re.compile(
    r"Find results for text: `.*?` in `(.*?)`(?: \((\S+?)(?:/find\?[^)]*)?\))?"
)
# WebShaper's browser: 'Found N match(es) for "p" in [k]:' + page lines 'L<n>: text'
_FIND_FOUND = re.compile(r'^Found \d+ match\(es\) for ".*" in \[(\d+)\]')
_FIND_NONE = re.compile(r"No `find` results for pattern|^No matches for ")


def find_page_lines(text: str) -> dict[int, str]:
    """The PAGE lines a gpt-oss find display shows: each ``# 【k†match at L<n>】`` block's
    lines are page lines n, n+1, ... (empty lines -- the block separators -- are skipped)."""
    _h, lines = display_lines(text)
    out: dict[int, str] = {}
    at = None
    for _k, v in sorted(lines.items()):
        m = _FIND_MATCH.match(v.strip())
        if m:
            at = int(m.group(1))
            continue
        if at is None:
            continue
        if v.strip():
            out.setdefault(at, _LINK.sub(lambda x: x.group(2), v))
        at += 1
    return out


def find_text(text: str) -> str:
    _h, lines = display_lines(text)
    keep = []
    for _k, v in sorted(lines.items()):
        if re.match(r"^#\s*【\d+†match at L\d+】", v.strip()):
            keep.append("")
            continue
        keep.append(v)
    return join_wrapped(keep)


_ERR = re.compile(r"^(error|unknown tool|no results|search error|failed|invalid)", re.I)


class BrowserState:
    """Cursor bookkeeping of one gpt-oss browser session: which cursor is a search listing
    (link k -> page) and which is a page; pages merged by id."""

    def __init__(self, question: str) -> None:
        self.focus = focus_of(question)
        self.links: dict[int, dict[int, dict]] = {}  # search cursor -> k -> hit
        self.page_of: dict[int, Page] = {}  # page cursor -> page
        self.pages: dict[str, Page] = {}
        self.last_cursor: int | None = None
        self.search_calls: dict[int, tuple[Call, list[dict]]] = {}  # search cursor -> call, hits
        self.last_open_call_page: Page | None = None  # the page of the last EMITTED web_open

    def search(self, args: dict, result: str) -> Call:
        q = str(args.get("query", "")).strip()
        if not q:
            raise DropError("search_without_query")
        if _ERR.match(result.strip()):
            return Call(
                "web_search", {"query": q}, self.focus, text=f"error: {result.strip()}", ok=False
            )
        cursor, hits = parse_search_display(result)
        if cursor is not None:
            self.links[cursor] = {h["k"]: h for h in hits}
            self.last_cursor = cursor
        if not hits:
            return Call(
                "web_search", {"query": q}, self.focus, text=f"no results for {q!r}", ok=False
            )
        call = Call(
            "web_search",
            {"query": q},
            self.focus,
            text=search_text([(h["id"], h["title"], h["text"]) for h in hits]),
        )
        if cursor is not None:
            self.search_calls[cursor] = (call, hits)
        return call

    def _cursor(self, cur_arg) -> int | None:
        """The cursor a call refers to; unknown / default (-1) = the latest display, the
        browser tool's own fallback."""
        try:
            c = int(cur_arg) if cur_arg is not None else None
        except (TypeError, ValueError):
            c = None
        if c is None or (c not in self.links and c not in self.page_of):
            return self.last_cursor
        return c

    def open(self, args: dict, result: str) -> Call | None:
        """A web_open call, or None when this view re-shows an already-open page (its lines are
        merged into that page's slot: our web_open returns the whole page)."""
        res = result.strip()
        cur_arg = args.get("cursor")
        link = args.get("id")
        if _ERR.match(res):
            label = link if isinstance(link, str) and link.strip() else f"cursor {cur_arg}"
            return Call(
                "web_open", {"id": str(label)}, self.focus, text=f"error: {res[:500]}", ok=False
            )
        cursor, htitle, hurl, lines = parse_page_display(res)
        pid = title = url = ""
        md = re.fullmatch(r"Doc (\S+)", htitle or "")
        if md:
            pid, url = md.group(1), hurl
            title = url_title(url) if url else htitle
        elif hurl:
            pid, url, title = hurl, hurl, htitle or url_title(hurl)
        if not pid:
            # no header: resolve through the cursor the call pointed at
            c = self._cursor(cur_arg)
            if link is None and c in self.search_calls:
                # scrolling a search listing: more hits of the SAME search -> merged into it
                call, hits = self.search_calls[c]
                _cur, more = parse_search_display(res)
                seen = {h["id"] for h in hits}
                hits.extend(h for h in more if h["id"] not in seen)
                self.links[c].update({h["k"]: h for h in more})
                call.text = search_text([(h["id"], h["title"], h["text"]) for h in hits])
                return None
            if isinstance(link, str) and not link.lstrip("-").isdigit():
                pid, url, title = link, link, url_title(link)
            elif link is not None and c in self.links:
                h = self.links[c].get(int(link))
                if h is None:
                    raise DropError("unresolvable_open")
                pid, url, title = h["id"], h["url"], h["title"]
            elif link is None and c in self.page_of:
                p = self.page_of[c]
                pid, url, title = p.id, p.url, p.title
            else:
                raise DropError("unresolvable_open")
        if not lines:
            return Call(
                "web_open", {"id": str(pid)}, self.focus, text=f"error: {res[:500]}", ok=False
            )
        page = self.pages.get(pid)
        new = page is None
        if new:
            page = Page(str(pid), title or url_title(url) or str(pid), url or str(pid))
            self.pages[pid] = page
        page.lines.update(lines)
        if cursor is not None:
            self.page_of[cursor] = page
            self.last_cursor = cursor
        if not new:
            return None
        self.last_open_call_page = page
        return Call("web_open", {"id": str(pid)}, self.focus, page=page)

    def _find_page(self, cur_arg, res: str) -> Page | None:
        """The page a find searched. The display header names it (``in `Doc N` (url/find?...)``)
        and is authoritative: its doc id, else its url, among the pages the agent opened (a page
        it never opened -> None: the trajectory does not have its text). Without a header: the
        page display the call's cursor points at (the browser's own -1 = latest fallback)."""
        first = res.split("\n", 1)[0]
        f = _FIND_FOUND.match(first)
        if f:  # WebShaper names the display it searched: a page, or a search listing (None)
            return self.page_of.get(int(f.group(1)))
        m = _FIND_HEADER.search(first)
        if m:
            md = re.fullmatch(r"Doc (\S+)", m.group(1).strip())
            if md:
                return self.pages.get(md.group(1))
            url = (m.group(2) or "").strip()
            hit = [pg for pg in self.pages.values() if url and pg.url == url]
            if hit:
                return hit[0]
            byt = [pg for pg in self.pages.values() if pg.title == m.group(1).strip()]
            return byt[0] if len(byt) == 1 else None
        c = self._cursor(cur_arg)
        return self.page_of.get(c) if c is not None else None

    def find(self, args: dict, result: str) -> Call:
        """Encoded web_find: a slot over the WHOLE page it searched (the find's own displayed
        page lines merged in), prompted with the pattern + the question focus."""
        pat = str(args.get("pattern", "")).strip()
        if not pat:
            raise DropError("find_without_pattern")
        res = result.strip()
        if _ERR.match(res):
            return Call("web_find", {"pattern": pat}, text=f"error: {res[:500]}", ok=False,
                        note="error")
        if _FIND_NONE.search(res):
            return Call("web_find", {"pattern": pat}, self.focus,
                        text=f"{pat!r} not found on the page", note="not_found")
        page = self._find_page(args.get("cursor"), res)
        if page is None:
            body = find_text(res)
            if not body:
                return Call("web_find", {"pattern": pat}, self.focus,
                            text=f"{pat!r} not found on the page", note="not_found")
            return Call("web_find", {"pattern": pat}, text=body, note="inline_unresolved")
        if _FIND_FOUND.match(res):
            shown = {k: _LINK.sub(lambda x: x.group(2), v) for k, v in display_lines(res)[1].items()
                     if v.strip()}
        else:
            shown = find_page_lines(res)
        page.lines.update({k: v for k, v in shown.items() if k not in page.lines})
        a = {"pattern": pat}
        if page is not self.last_open_call_page:
            a["id"] = str(page.id)
        return Call("web_find", a, self.focus, page=page, note="slot")


def exact_answer(final_text: str) -> str | None:
    """gpt-oss's ``Exact Answer: X`` (also bold / multi-line lists, up to ``Confidence``)."""
    i = final_text.rfind("Exact Answer")
    if i < 0:
        return _boxed(final_text)
    a = final_text[i + len("Exact Answer") :].lstrip("*: \t")
    a = re.split(r"\n\s*\**\s*Confidence", a)[0]
    parts = [
        re.sub(r"^\s*([-*\u2022]|\d+[.)])\s+", "", ln).strip().strip("*").strip()
        for ln in a.split("\n")
    ]
    a = "; ".join(x for x in parts if x)
    return (_boxed(a) or a).strip() or None


def _or_text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(str(p.get("text") or "") for p in content or [] if isinstance(p, dict))


def parse_openresearcher(rec: dict, tid: str) -> Episode:
    q = str(rec["question"]).strip()
    if re.search(r"\bNote: A[n]? .{0,40}file '", q) or "file '/" in q:
        raise DropError("file_task")
    st = BrowserState(q)
    steps: list[Step] = []
    thought: list[str] = []
    final = None
    msgs = rec["messages"]
    i = 0
    while i < len(msgs):
        m = msgs[i]
        role, rcpt = m.get("role"), m.get("recipient")
        text = _or_text(m.get("content"))
        if role in ("system", "developer", "user"):
            i += 1
            continue
        if role == "assistant" and not rcpt:
            if m.get("channel") == "final":
                final = text
            else:
                thought.append(text)
            i += 1
            continue
        if role == "assistant" and rcpt:
            if rcpt not in ("browser.search", "browser.open", "browser.find"):
                raise DropError(f"unmappable_call:{rcpt[:30]}")
            try:
                a = json.loads(text)
            except json.JSONDecodeError:
                raise DropError("bad_call_args") from None
            if not isinstance(a, dict):
                raise DropError("bad_call_args")
            res = _or_text(msgs[i + 1].get("content")) if i + 1 < len(msgs) else None
            if res is None or msgs[i + 1].get("role") != "tool":
                raise DropError("call_without_result")
            call = getattr(st, rcpt.split(".")[1])(a, res)
            t = clean_thought("\n".join(thought))  # analysis channel only: all reasoning
            steps.append(Step(t, [call] if call else [], reasoning=t))
            thought = []
            i += 2
            continue
        i += 1
    if final is None:
        raise DropError("no_final_answer")
    ans = exact_answer(final)
    if not ans:
        raise DropError("no_exact_answer")
    gold_raw = str(rec.get("answer") or "")
    gold = [_boxed(gold_raw) or gold_raw] if gold_raw.strip() else None
    t = clean_thought("\n".join(thought))
    return Episode(tid, q, ans, steps, t, gold=gold, pred=ans, final_reasoning=t)


def iter_openresearcher(args) -> Iterator[tuple[str, dict]]:
    import pyarrow.parquet as pq

    root = args.input or f"{WEB_SFT}/OpenResearcher__OpenResearcher-Dataset"
    for f in sorted(glob.glob(f"{root}/seed_*/*.parquet")):
        pf = pq.ParquetFile(f)
        seed = Path(f).parent.name
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(
                rg, columns=["qid", "question", "answer", "messages", "status"]
            ).to_pylist():
                if r.get("status") not in (None, "success"):
                    continue
                yield f"openresearcher/{seed}/{r['qid']}", r


# --- WebShaper (gpt-oss, harmony strings) -----------------------------------

_HARMONY_ANALYSIS = re.compile(
    r"<\|channel\|>analysis<\|message\|>(.*?)(?=<\|end\|>|<\|start\|>|<\|call\|>|<\|channel\|>|$)",
    re.S,
)


def harmony_analysis(content: str) -> str:
    if "<|channel|>" not in content:
        return content
    return "\n".join(x.strip() for x in _HARMONY_ANALYSIS.findall(content))


def parse_webshaper(rec: dict, tid: str) -> Episode:
    q = str(rec["question"]).strip()
    conv = json.loads(rec["conversation"])
    st = BrowserState(q)
    steps: list[Step] = []
    final_thought = None
    for i, m in enumerate(conv):
        if m["role"] != "assistant":
            continue
        if "final_answer" in m:
            final_thought = harmony_analysis(m.get("content") or "")
            break
        tc = m.get("tool_call")
        try:
            call = ast.literal_eval(tc) if isinstance(tc, str) else tc
        except (ValueError, SyntaxError):
            raise DropError("bad_call") from None
        name = str(call.get("tool", ""))
        if name not in ("browser.search", "browser.open", "browser.find"):
            raise DropError(f"unmappable_call:{name[:30]}")
        if i + 1 >= len(conv) or conv[i + 1]["role"] != "tool":
            raise DropError("call_without_result")
        c = getattr(st, name.split(".")[1])(dict(call.get("args") or {}), conv[i + 1]["content"])
        t = clean_thought(harmony_analysis(m.get("content") or ""))  # analysis channel: reasoning
        steps.append(Step(t, [c] if c else [], reasoning=t))
    if final_thought is None:
        raise DropError("no_final_answer")
    ans = str(rec.get("boxed_answer") or "").strip()
    if not ans:
        raise DropError("no_boxed_answer")
    ref = str(rec.get("reference_answer") or "")
    return Episode(
        tid,
        q,
        ans,
        steps,
        clean_thought(final_thought),
        gold=[ref] if ref.strip() else None,
        final_reasoning=clean_thought(final_thought),
        pred=ans,
        judged_correct=bool(rec.get("correct")),
        meta={
            "question_accuracy": rec.get("question_accuracy"),
            # search results contained the WebShaper dataset page itself (the answer leaked)
            "contaminated": bool(rec.get("contaminated")),
        },
    )


def iter_webshaper(args) -> Iterator[tuple[str, dict]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(
        f"{WEB_SFT}/rl-rag__webshaper-gpt-oss-120b-260222/data/train-00000-of-00001.parquet"
    )
    for b in pf.iter_batches(batch_size=64):
        for r in b.to_pylist():
            if r.get("status") != "success":
                continue
            yield f"webshaper/{r['qid']}/{r['traj_idx']}", r


# --- OpenSeeker / MiroVerse mix ---------------------------------------------

_GOOGLE_BLOCK = re.compile(
    r"### A Google search for '(.*?)' found \d+ results?:[ \t]*(?:\n|$)", re.S
)
_GOOGLE_HIT = re.compile(r"^\d+\. \[(.*?)\]\((\S+?)\)\s*$", re.M)


def google_blocks(text: str) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """OpenSeeker search displays (one block per query) -> [(query, hits)]."""
    parts = _GOOGLE_BLOCK.split(text)
    out = []
    for j in range(1, len(parts) - 1, 2):
        q, body = parts[j], parts[j + 1]
        hits = []
        ms = list(_GOOGLE_HIT.finditer(body))
        for n, mh in enumerate(ms):
            end = ms[n + 1].start() if n + 1 < len(ms) else len(body)
            snip = body[mh.end() : end]
            snip = re.sub(r"^(Date published:.*|-{3,}\s*)$", "", snip, flags=re.M)
            hits.append((mh.group(2), mh.group(1).strip(), " ".join(snip.split())))
        out.append((q, hits))
    return out


def serper_hits(obj: dict) -> list[tuple[str, str, str]]:
    hits = []
    ab = obj.get("answerBox") or {}
    if ab:
        txt = " ".join(str(ab.get(k, "")) for k in ("answer", "snippet") if ab.get(k))
        hits.append((ab.get("link") or "google-answer-box", str(ab.get("title", "")).strip(), txt))
    kg = obj.get("knowledgeGraph") or {}
    if kg.get("description"):
        hits.append(
            (
                kg.get("descriptionLink") or kg.get("website") or "google-knowledge-graph",
                str(kg.get("title", "")).strip(),
                str(kg["description"]),
            )
        )
    for h in obj.get("organic") or []:
        if h.get("link"):
            hits.append((h["link"], str(h.get("title", "")).strip(), str(h.get("snippet", ""))))
    return hits


def _mix_args(fn: dict) -> dict:
    a = fn.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except json.JSONDecodeError:
            raise DropError("bad_call_args") from None
    if not isinstance(a, dict):
        raise DropError("bad_call_args")
    return a


def _none(x) -> str:
    return "" if x is None or x == "None" else str(x)


def _answer_tag(text: str) -> str | None:
    m = re.findall(r"<answer>(.*?)(?:</answer>|$)", text, re.S)
    return m[-1].strip() if m else None


def mix_visit_pages(text: str, urls: list[str]) -> list[tuple[str, str]] | None:
    """[(title, body)] per visited url, or None when the result cannot be split per url."""
    t = text.strip()
    if t.startswith("{"):
        try:
            obj = json.loads(t)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and "text" in obj and len(urls) == 1:
            title = str((obj.get("metadata") or {}).get("title") or "").strip()
            return [(title, str(obj.get("text") or ""))]
        if isinstance(obj, list) and len(obj) == len(urls):
            return [
                (str((o.get("metadata") or {}).get("title") or ""), str(o.get("text") or ""))
                for o in obj
                if isinstance(o, dict)
            ]
    if len(urls) == 1:
        return [("", t)]
    parts = re.split(r"(?=The useful information in )", t)
    parts = [p for p in parts if p.strip()]
    if len(parts) == len(urls):
        return [("", p.strip()) for p in parts]
    return None


def parse_mix(rec: dict, tid: str) -> Episode:
    q = str(rec["question"]).strip()
    msgs = list(rec["messages"])
    steps: list[Step] = []
    final = None
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m["role"] != "assistant":
            i += 1
            continue
        thought = clean_thought(
            "\n".join(
                x for x in (_none(m.get("reasoning_content")), _none(m.get("content"))) if x.strip()
            )
        )
        sep = (  # (reasoning, said): --keep-reasoning
            clean_thought(_none(m.get("reasoning_content"))),
            clean_thought(_none(m.get("content"))),
        )
        calls_raw = m.get("tool_calls")
        if isinstance(calls_raw, str):
            calls_raw = None if calls_raw == "None" else ast.literal_eval(calls_raw)
        if not calls_raw:
            last = not any(x["role"] == "assistant" for x in msgs[i + 1 :])
            if last or "<answer>" in _none(m.get("content")):
                final = (m, thought)
                break
            steps.append(Step(thought, [], *sep))  # a call-less turn: its words carry forward
            i += 1
            continue
        results: dict[str, str] = {}
        j = i + 1
        order: list[str] = []
        while j < len(msgs) and msgs[j]["role"] == "tool":
            cid = _none(msgs[j].get("tool_call_id"))
            results[cid] = _none(msgs[j].get("content"))
            order.append(cid)
            j += 1
        calls: list[Call] = []
        for n, c in enumerate(calls_raw):
            fn = c.get("function") or {}
            name = fn.get("name")
            a = _mix_args(fn)
            res = results.get(_none(c.get("id")))
            if res is None:
                res = results.get(order[n]) if n < len(order) else None
            if res is None:
                raise DropError("call_without_result")
            if name == "search":
                qs = a.get("query")
                qs = [qs] if isinstance(qs, str) else list(qs or [])
                qs = [str(x).strip() for x in qs if str(x).strip()]
                if not qs:
                    raise DropError("search_without_query")
                blocks = google_blocks(res)
                if blocks:
                    per = {b[0]: b[1] for b in blocks}
                    parts = [per.get(x) for x in qs]
                    if any(p is None for p in parts):
                        parts = [b[1] for b in blocks] if len(blocks) == len(qs) else None
                    if parts is None:
                        raise DropError("search_split")
                else:
                    try:
                        obj = json.loads(res)
                    except json.JSONDecodeError:
                        obj = None
                    if isinstance(obj, dict) and len(qs) == 1:
                        parts = [serper_hits(obj)]
                    elif isinstance(obj, list) and len(obj) == len(qs):
                        parts = [serper_hits(o) for o in obj]
                    elif len(qs) == 1:
                        parts = [None]
                    else:
                        raise DropError("search_split")
                for x, hits in zip(qs, parts, strict=True):
                    if hits:
                        text, ok = search_text(hits), True
                    elif hits is None:  # not a result listing: the tool's own error text
                        text, ok = f"error: {res.strip()[:300] or 'search failed'}", False
                    else:
                        text, ok = f"no results for {x!r}", False
                    calls.append(Call("web_search", {"query": x}, focus_of(q), text=text, ok=ok))
            elif name == "visit":
                urls = a.get("url")
                urls = [urls] if isinstance(urls, str) else list(urls or [])
                urls = [str(u).strip() for u in urls if str(u).strip()]
                if not urls:
                    raise DropError("visit_without_url")
                goal = " ".join(str(a.get("goal") or "").split())
                # MiroVerse's converter often copied the turn's own text into ``goal``: that
                # is narration, not a focus -> the question, like the other sources
                echo = goal and norm_q(goal)[:80] in norm_q(thought)
                focus = goal if goal and len(goal) <= 300 and not echo else focus_of(q)
                pages = mix_visit_pages(res, urls)
                if pages is None:
                    raise DropError("visit_split")
                for u, (title, body) in zip(urls, pages, strict=True):
                    bad = not body.strip() or re.match(
                        r"^\[?visit\]? ?(failed|error)|^error", body.strip(), re.I
                    )
                    if bad:
                        calls.append(
                            Call(
                                "web_open",
                                {"id": u},
                                focus,
                                text=f"error: {body.strip()[:300] or 'empty page'}",
                                ok=False,
                            )
                        )
                    else:
                        pg = Page(u, title or url_title(u), u, text=body.strip())
                        calls.append(Call("web_open", {"id": u}, focus, page=pg))
            else:
                raise DropError(f"unmappable_call:{str(name)[:30]}")
        steps.append(Step(thought, calls, *sep))
        i = j
    if final is None:
        raise DropError("no_final_answer")
    m, thought = final
    content = _none(m.get("content"))
    pred = _answer_tag(content)
    if not pred:
        raise DropError("no_answer_tag")
    thought = clean_thought(
        "\n".join(
            x
            for x in (_none(m.get("reasoning_content")), content[: content.find("<answer>")])
            if x.strip()
        )
    )
    gold = [str(rec["answer"])] if rec.get("answer") else None
    return Episode(
        tid, q, pred, steps, thought, gold=gold, pred=pred, meta={"mix_source": rec.get("source")},
        final_reasoning=clean_thought(_none(m.get("reasoning_content"))),
        final_said=clean_thought(content[: content.find("<answer>")]),
    )


def iter_mix(args, want: str) -> Iterator[tuple[str, dict]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(f"{WEB_SFT}/Zephyr271828__openseeker-miroverse-mix-full/train.parquet")
    n = 0
    for b in pf.iter_batches(batch_size=64, columns=["messages", "question", "answer", "source"]):
        for r in b.to_pylist():
            n += 1
            is_os = r.get("source") == "openseeker"
            if (want == "openseeker") != is_os:
                continue
            yield f"{want}/{n}", r


# ---------------------------------------------------------------------------
# Row builder: replay through the harness's own Conversation + compaction
# ---------------------------------------------------------------------------

WEB_EVAL_CONTEXT_BUDGET = 6144  # scripts/bgkit2_eval_web.py --context-budget (per-turn prefix cap)
WEB_EVAL_RATIOS = "web_search=4,web_open=16,web_find=4"  # scripts/bgkit2_eval_web.py DEFAULT_RATIOS


# FastRenderer / _Counted live in bgkit2.agent.replay (shared with the tool_slots remap's
# --compact-history, 2026-09-26); imported at the top of this script.


def _join(*parts: str) -> str:
    return "\n\n".join(x for x in parts if x)


def episode_turns(
    ep: Episode, keep_reasoning: bool = False
) -> list[tuple[str, list[Call], str | None]]:
    """(content, calls, reasoning) per assistant turn, the final ANSWER turn last. Steps without
    calls (a view merged into an earlier page) pass their words on to the next turn.

    Default: the turn's reasoning is part of ``content`` (untagged prose) and ``reasoning`` is
    None. ``keep_reasoning`` (2026-09-26, the LFM2.5-8B-A1B decoder runs in its always-thinking
    mode): ``content`` is only what the model SAID (plus ``ANSWER:`` on the final turn) and the
    reasoning is returned apart, stripped, for the message's ``reasoning_content`` (None when the
    turn has none); the chat template renders it as ``<think>...</think>`` before the content."""
    out: list[tuple[str, list[Call], str | None]] = []
    if not keep_reasoning:
        carry = ""
        for st in ep.steps:
            thought = _join(carry, st.thought)
            if not st.calls:
                carry = thought
                continue
            carry = ""
            out.append((thought, st.calls, None))
        final = _join(carry, ep.final_thought)
        out.append(((final + "\n\n" if final else "") + f"ANSWER: {ep.answer}", [], None))
        return out
    carry_r = carry_s = ""
    for st in ep.steps:
        r, said = _join(carry_r, st.reasoning), _join(carry_s, st.said)
        if not st.calls:
            carry_r, carry_s = r, said
            continue
        carry_r = carry_s = ""
        out.append((said, st.calls, r or None))
    r, said = _join(carry_r, ep.final_reasoning), _join(carry_s, ep.final_said)
    out.append(((said + "\n\n" if said else "") + f"ANSWER: {ep.answer}", [], r or None))
    return out


def assistant_message(content: str, calls: list[Call], reasoning: str | None) -> dict:
    m: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    if calls:
        m["tool_calls"] = [c.message() for c in calls]
    return m


def episode_messages(
    ep: Episode, keep_reasoning: bool = False
) -> tuple[list[dict], list[tuple[str, str] | None]]:
    """The uncompacted slotted conversation: (messages, per-message slot (text, prompt))."""
    msgs = [
        {"role": "system", "content": WEB_SYSTEM},
        {"role": "user", "content": f"Question: {ep.question}\n\n{WEB_INSTRUCTION}"},
    ]
    slots: list[tuple[str, str] | None] = [None, None]
    for content, calls, reasoning in episode_turns(ep, keep_reasoning):
        m = assistant_message(content, calls, reasoning)
        msgs.append(m)
        slots.append(None)
        for c in calls:
            if c.is_slot():
                msgs.append({"role": "tool", "content": SPLICE_SENTINEL})
                slots.append((c.result(), c.prompt()))
            else:
                msgs.append({"role": "tool", "content": c.result() or "(empty result)"})
                slots.append(None)
    return msgs, slots


def system_with_tools(tools: list[dict]) -> str:
    """WEB_SYSTEM + the tool list as the LFM chat template appends it for ``tools=``."""
    from bgkit2.agent.context import system_with_tool_list

    return system_with_tool_list(WEB_SYSTEM, tools)


def split_of(question: str, eval_pct: int = 1) -> str:
    h = int(hashlib.sha1(norm_q(question).encode()).hexdigest()[:8], 16)
    return "eval" if h % 100 < eval_pct else "train"


def sample_turns(n: int, k: int, seed: str) -> list[int]:
    """All turns when ``k`` is 0 or >= n; else one uniformly drawn turn from each of k-1 equal
    depth bins over turns 0..n-2 plus the final turn (every depth range, incl. the late turns
    where compaction lives, is covered)."""
    if not k or n <= k:
        return list(range(n))
    rng = random.Random(seed)
    m = n - 1
    picks = {
        rng.randrange(b * m // (k - 1), max(b * m // (k - 1) + 1, (b + 1) * m // (k - 1)))
        for b in range(k - 1)
    }
    return sorted(picks | {n - 1})


def ratio_policy(spec: str = WEB_EVAL_RATIOS):
    from bgkit2.agent.context import RatioPolicy

    rp = RatioPolicy()
    for kv in [x for x in spec.split(",") if x]:
        k, v = kv.split("=")
        rp.ratios[k] = int(v)
    return rp


def replay_rows(
    ep: Episode,
    family: str,
    renderer,
    *,
    budget: int = WEB_EVAL_CONTEXT_BUDGET,
    keep_recent: int = 1,
    ratios=None,
    max_rows: int = 0,
    base_meta: dict | None = None,
    eval_pct: int = 1,
    stats: Counter | None = None,
    compaction_mode: str = "cons",
    slot_max_tokens: int | None = 8192,
    keep_reasoning: bool = False,
) -> list[dict]:
    """Replay the trajectory through ``bgkit2.agent.context.Conversation`` exactly as
    ``run_episode`` builds an episode of the web eval: the same system prompt, tool list, task
    message and compaction summary (``web_eval.web_task``), each tool result added as a slot
    (compressible) or inline text, and ``compact(budget, keep_recent, compressed=True)`` before
    every assistant turn -- the oldest rounds fold into the one compaction slot as at eval.
    A row is ``conv.to_task_row`` at the sampled turns (a ``chat`` row, before the first slot,
    carries the tool list in its system message: the chat framing renders no ``tools=``).
    ``keep_reasoning``: each turn's reasoning rides in ``reasoning_content`` of the target AND
    of the history, as the harness keeps a thinking decoder's reasoning
    (``runner`` -> ``Conversation.add_assistant(..., reasoning=...)``); compaction folds only the
    visible text (``_Msg.plain``), as at eval."""
    from bgkit2.agent.context import Conversation, ToolCall

    stats = stats if stats is not None else Counter()
    tools = web_tool_specs()
    ratios = ratios or ratio_policy()
    summary = " ".join(str(ep.question).split())[:200]
    conv = Conversation(
        WEB_SYSTEM,
        tools,
        renderer,
        ratios,
        summary,
        compaction_mode=compaction_mode,
        slot_max_tokens=slot_max_tokens,
    )
    conv.add_user(f"Question: {ep.question.strip()}\n\n{WEB_INSTRUCTION}")
    turns = episode_turns(ep, keep_reasoning)
    keep = set(sample_turns(len(turns), max_rows, ep.traj_id))
    split = split_of(ep.question, eval_pct)
    rows = []
    for i, (content, calls, reasoning) in enumerate(turns):
        conv.compact(budget, keep_recent=keep_recent, compressed=True)
        target = assistant_message(content, calls, reasoning)
        if i in keep:
            comp = [s for s in conv.slots if s.tool == "compaction"]
            meta = dict(base_meta or {})
            meta.update(ep.meta)
            meta.update(
                {
                    "trajectory_id": ep.traj_id,
                    "turn_index": i,
                    "n_turns": len(turns),
                    "final": i == len(turns) - 1,
                    "question": ep.question,
                    "gold": ep.gold,
                    "compacted": bool(comp),
                    "n_compactions": conv.n_compactions,
                    "compaction_slot_tokens": comp[0].n_tokens if comp else 0,
                    **spine_meta(comp[0].text if comp else "", renderer),
                    "prefix_tokens_compressed": conv.prefix_tokens(compressed=True),
                    "prefix_text_tokens": renderer.count(conv.render_slots()),
                    "slot_tokens": [s.n_tokens for s in conv.slots],
                    "context_budget": budget,
                }
            )
            r = conv.to_task_row(target=[target], family=family, split=split, meta=meta)
            r["meta"].pop("digests", None)
            if r["framing"] == "chat":  # to_task_row wrote the tool list into the system turn
                stats["rows_chat_no_slot"] += 1
            stats["rows_compacted"] += bool(comp)
            rows.append(r)
        conv.add_assistant(
            content,
            [ToolCall(c.tool, c.message()["function"]["arguments"]) for c in calls],
            reasoning=reasoning,
        )
        for c in calls:
            conv.add_tool_result(
                c.result() or "(empty result)",
                c.is_slot(),
                c.prompt() if c.is_slot() else "",
                c.tool,
            )
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

PARSERS = {
    "search_r1": (iter_search_r1, parse_search_r1),
    "sds": (iter_sds, parse_sds),
    "webshaper": (iter_webshaper, parse_webshaper),
    "openseeker": (lambda a: iter_mix(a, "openseeker"), parse_mix),
    "miroverse_nc": (lambda a: iter_mix(a, "miroverse_nc"), parse_mix),
    "openresearcher": (iter_openresearcher, parse_openresearcher),
}


class StoreWriter:
    """``write_task_store``'s row -> string-column conversion, streamed: rows are buffered,
    shuffled within the buffer (so one trajectory's rows do not sit together) and written as
    parquet row groups -- the whole OpenResearcher store never sits in memory."""

    def __init__(self, path: Path, pa, pq, columns: list[str], buf_rows: int = 8192) -> None:
        self.path, self.pa, self.columns, self.buf_rows = path, pa, columns, buf_rows
        self.schema = pa.schema([(c, pa.string()) for c in columns])
        self.w = pq.ParquetWriter(str(path), self.schema, compression="zstd")
        self.buf: list[dict] = []
        self.n = 0
        self.rng = random.Random(0)

    def add(self, r: dict) -> None:
        self.buf.append(r)
        if len(self.buf) >= self.buf_rows:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        self.rng.shuffle(self.buf)
        cols: dict[str, list[str]] = {c: [] for c in self.columns}
        for r in self.buf:
            for c in self.columns:
                v = r.get(c, "")
                if c in ("tool_args", "meta") and not isinstance(v, str):
                    v = json.dumps(v, ensure_ascii=False)
                cols[c].append("" if v is None else str(v))
        t = self.pa.table({c: self.pa.array(v, type=self.pa.string()) for c, v in cols.items()})
        self.w.write_table(t, row_group_size=2048)
        self.n += len(self.buf)
        self.buf = []

    def close(self) -> None:
        self.flush()
        self.w.close()


def source_size(args) -> int:
    """Trajectories a source yields before any parsing (for --sample-trajectories)."""
    import pyarrow.parquet as pq

    if args.source == "openresearcher":
        root = args.input or f"{WEB_SFT}/OpenResearcher__OpenResearcher-Dataset"
        return sum(
            pq.ParquetFile(f).metadata.num_rows for f in glob.glob(f"{root}/seed_*/*.parquet")
        )
    if args.source in ("openseeker", "miroverse_nc"):
        col = (
            pq.read_table(
                f"{WEB_SFT}/Zephyr271828__openseeker-miroverse-mix-full/train.parquet",
                columns=["source"],
            )
            .column("source")
            .to_pylist()
        )
        n_os = sum(x == "openseeker" for x in col)
        return n_os if args.source == "openseeker" else len(col) - n_os
    return sum(1 for _ in PARSERS[args.source][0](args))


def verify_thinking_row(r: dict, tok, tools: list[dict], fast: FastRenderer) -> int:
    """``--keep-reasoning`` rows: the prefix's token count rendered by the real template with
    every assistant turn's reasoning shown (``preserve_thinking``: what LFM2.5-8B-A1B renders for
    turns after the last user message, and what FastRenderer counts), sentinels excluded; also
    checks the slot / sentinel / prompt counts and, for a chat row, the tool-list rendering."""
    msgs = json.loads(r["instruction"])
    kw = {"preserve_thinking": True, "add_generation_prompt": True, "tokenize": True}
    if r["framing"] == "chat":
        want = [{"role": "system", "content": WEB_SYSTEM}, *msgs[1:]]
        got = tok.apply_chat_template(msgs, **kw)
        assert got == tok.apply_chat_template(want, tools=tools, **kw), "tool list rendering"
    else:
        ctx = json.loads(r["context"])
        assert r["instruction"].count(SPLICE_SENTINEL) == len(ctx) == len(json.loads(r["prompt"]))
        got = tok.apply_chat_template(msgs, tools=tools, **kw)
    if not isinstance(got, list):
        got = got["input_ids"]
    return sum(1 for i in got if i != fast.sentinel)


def dist(xs: list[int]) -> dict:
    return {"p50": pct(xs, 0.5), "p95": pct(xs, 0.95), "max": max(xs, default=0)}


def pct(xs: list[int], q: float) -> int:
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(PARSERS))
    ap.add_argument("--input", default="", help="override the source's input location")
    ap.add_argument("--out", default="", help="default: <tasks>/<family>.parquet (+ .arrow)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--max-prefix-tokens",
        type=int,
        default=16384,
        help="drop a row whose decoder text prefix (reps not counted) still exceeds this after "
        "compaction (only possible when one round alone is that long)",
    )
    ap.add_argument(
        "--context-budget",
        type=int,
        default=WEB_EVAL_CONTEXT_BUDGET,
        help="compaction budget: the web eval's --context-budget (compressed decoder positions)",
    )
    ap.add_argument("--keep-recent", type=int, default=1, help="Conversation.compact keep_recent")
    ap.add_argument(
        "--compaction-mode",
        default="cons",
        choices=("flat", "cons"),
        help="Conversation compaction_mode (the web eval's WEB_COMPACTION_MODE)",
    )
    ap.add_argument(
        "--slot-max-tokens",
        type=int,
        default=8192,
        help="split a compressible result into slots of at most this many tokens (the "
        "trainer's data.task_max_ctx_tokens / web eval WEB_SLOT_MAX_TOKENS); 0 = never split",
    )
    ap.add_argument(
        "--sample-trajectories",
        type=int,
        default=0,
        help="keep ~N source trajectories, drawn uniformly (seeded hash) before any parsing; "
        "0 = all. A size budget, not a quality filter",
    )
    ap.add_argument("--sample-seed", default="web-sft-2026-09-22")
    ap.add_argument("--ratios", default=WEB_EVAL_RATIOS, help="RatioPolicy, as the web eval's")
    ap.add_argument(
        "--max-rows-per-traj",
        type=int,
        default=0,
        help="0 = a row for every assistant turn; else depth-stratified sample + the final turn",
    )
    ap.add_argument("--eval-pct", type=int, default=1)
    ap.add_argument("--buffer-rows", type=int, default=8192,
                    help="rows buffered (and shuffled) before a parquet write: the builder's "
                    "memory peak on the long-trajectory sources (~100 KB per OpenResearcher row)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N source trajectories")
    ap.add_argument("--tokenizer", default="LiquidAI/LFM2.5-350M")
    ap.add_argument(
        "--keep-reasoning",
        action="store_true",
        help="each assistant turn's reasoning (Search-R1 <thinking>, SimpleDeepSearcher's think "
        "block, gpt-oss analysis channel, reasoning_content) goes to the message's "
        "reasoning_content -- target and history -- instead of into content as untagged prose "
        "(the LFM2.5-8B-A1B decoder in its always-thinking mode, 2026-09-26)",
    )
    ap.add_argument(
        "--verify-every",
        type=int,
        default=200,
        help="re-render every Nth row through ChatRenderer.render_slots and assert "
        "the prefix count and sentinel/slot counts",
    )
    args = ap.parse_args()
    family, repo, lic = SOURCES[args.source]
    out = Path(args.out or f"{TASKS}/{family}.parquet")
    arrow = out.with_suffix(".arrow")
    for p in (out, arrow):
        if p.exists() and not args.overwrite:
            sys.exit(f"{p} exists; pass --overwrite")

    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from bgkit2.data.tasks import COLUMNS, ChatRenderer, write_task_store_arrow

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    fast = FastRenderer(tok)
    renderer = ChatRenderer.from_tokenizer(tok)
    ratios = ratio_policy(args.ratios)
    tools = web_tool_specs()
    excl = load_excluder()
    gold_table = load_gold_nq_hotpot() if args.source == "search_r1" else None
    it, parse = PARSERS[args.source]
    lenient = args.source in ("openresearcher", "openseeker", "miroverse_nc")
    st: Counter = Counter()
    excluded_by: Counter = Counter()
    cite_phrases: Counter = Counter()
    t0 = time.time()
    writer = StoreWriter(out.with_suffix(".parquet.tmp"), pa, pq, COLUMNS,
                         buf_rows=args.buffer_rows)
    pre_c: list[int] = []
    pre_t: list[int] = []
    comp_tok: list[int] = []
    spine_depth: list[int] = []
    spine_leaves: list[int] = []
    leaf_tok: list[int] = []
    max_slot = args.slot_max_tokens or 1 << 30
    max_leaf = 7680
    nsl: list[int] = []
    tgt_tok: list[int] = []
    n_eval = 0
    base_meta = {
        **({"reasoning": "reasoning_content"} if args.keep_reasoning else {}),
        "source": args.source,
        "dataset": repo,
        "license": lic,
        "converted_by": "scripts/bgkit2_build_web_sft_store.py",
        "snippet_words": SNIPPET_WORDS,
        "max_page_chars": MAX_PAGE_CHARS,
    }
    sample_p = 1.0
    if args.sample_trajectories:
        total = source_size(args)
        sample_p = min(1.0, args.sample_trajectories / max(1, total))
        st["source_trajectories"] = total
    for tid, rec in it(args):
        st["in"] += 1
        if args.limit and st["in"] > args.limit:
            st["in"] -= 1
            break
        if st["in"] % 2000 == 0:
            print(
                f"[{time.time() - t0:.0f}s] in={st['in']} kept_traj={st['kept_traj']} "
                f"rows={writer.n}",
                flush=True,
            )
        if sample_p < 1.0:
            h = int(hashlib.sha1(f"{args.sample_seed}:{tid}".encode()).hexdigest()[:12], 16)
            if h / float(1 << 48) >= sample_p:
                st["not_sampled"] += 1
                continue
        if "_drop" in rec:
            st[f"drop:{rec['_drop']}"] += 1
            continue
        if rec.get("question"):
            cq = clean_question(rec["question"])
            if cq != str(rec["question"]).strip():
                st["question_format_instruction_stripped"] += 1
                rec = {**rec, "question": cq}
        try:
            ep = parse(rec, tid)
        except DropError as d:
            st[f"drop:{d}"] += 1
            continue
        qn = norm_q(ep.question)
        if args.source == "search_r1":
            ep.gold = gold_table.get(qn) or None
            ep.meta["source_file"] = tid.split("/")[1]
        b = excl.hit(ep.question)
        if b is not None:
            st["drop:benchmark_question"] += 1
            excluded_by[b] += 1
            continue
        # correctness is RECORDED (meta.answer_match / gold / judged_correct), never filtered
        if ep.gold:
            st["gold_available"] += 1
            crit = answer_matches(ep.pred, ep.gold, lenient)
            ep.meta["answer_match"] = crit
            st[f"match:{crit}"] += 1
            st["em_or_contained"] += bool(crit)
        else:
            st["no_gold"] += 1
            ep.meta["answer_match"] = None
        if ep.judged_correct is not None:
            ep.meta["judged_correct"] = ep.judged_correct
            st[f"judged_correct:{ep.judged_correct}"] += 1
        # a long free-form final answer -> the gold alias it contains (ANSWER: is a short phrase)
        if len(ep.answer) > 100 and ep.gold:
            g = [x for x in ep.gold if answer_matches(ep.pred, [x])]
            if g:
                ep.answer = g[0]
                ep.meta["answer_from"] = "gold_alias_contained_in_final_answer"
                st["answer_from_gold_alias"] += 1
        ep.answer = " ".join(ep.answer.split())
        if not ep.answer:
            st["drop:empty_answer"] += 1
            continue
        for stp in ep.steps:
            for c in stp.calls:
                if c.tool == "web_find":
                    st[f"find:{c.note or 'inline'}"] += 1
        # flags only (the trainer may weight / filter on them)
        ep.meta["pure_math"] = pure_math(ep.question)
        hunt = citation_hunting(ep)
        ep.meta["citation_hunting"] = hunt
        st["flag:pure_math"] += ep.meta["pure_math"]
        st["flag:citation_hunting"] += bool(hunt)
        if hunt:
            cite_phrases[hunt.lower()] += 1
        new = replay_rows(
            ep,
            family,
            fast,
            budget=args.context_budget,
            keep_recent=args.keep_recent,
            ratios=ratios,
            max_rows=args.max_rows_per_traj,
            base_meta=base_meta,
            eval_pct=args.eval_pct,
            stats=st,
            compaction_mode=args.compaction_mode,
            slot_max_tokens=args.slot_max_tokens or None,
            keep_reasoning=args.keep_reasoning,
        )
        st["kept_traj"] += 1
        for r in new:
            m = r["meta"]
            if m["prefix_text_tokens"] > args.max_prefix_tokens:
                st["drop_row:prefix_over_cap"] += 1
                continue
            k = writer.n + len(writer.buf)
            if args.keep_reasoning:
                tg = json.loads(r["target"])[0]
                st["rows_target_reasoning"] += bool(tg.get("reasoning_content"))
            if args.verify_every and k % args.verify_every == 0 and args.keep_reasoning:
                n = verify_thinking_row(r, tok, tools, fast)
                assert n == m["prefix_text_tokens"], (n, m["prefix_text_tokens"], tid)
                st["verified_rows"] += 1
            elif args.verify_every and k % args.verify_every == 0:
                if r["framing"] == "chat":
                    pre_ids, _ = renderer.render("chat", r["instruction"])
                    want = json.loads(r["instruction"])
                    want[0] = {"role": "system", "content": WEB_SYSTEM}
                    assert renderer._apply(json.loads(r["instruction"])) == renderer._apply(
                        want, tools
                    ), (tid, "tool list rendering")
                    n = int(pre_ids.shape[0])
                else:
                    segs = renderer.render_slots(r["instruction"], tools)
                    n = sum(int(x.shape[0]) for x in segs)
                    ctx = json.loads(r["context"])
                    assert len(segs) - 1 == len(ctx) == len(json.loads(r["prompt"])), m
                    assert r["instruction"].count(SPLICE_SENTINEL) == len(ctx)
                    # slot token counts (incl. the incrementally counted compaction slot)
                    for c, nt in zip(ctx, m["slot_tokens"], strict=True):
                        if not c.startswith('{"compaction"'):  # a spine's n = leaf + child reps
                            st["slot_count_mismatch"] += fast._raw(c) != nt
                assert n == m["prefix_text_tokens"], (n, m["prefix_text_tokens"], tid)
                st["verified_rows"] += 1
            pre_c.append(m["prefix_tokens_compressed"])
            pre_t.append(m["prefix_text_tokens"])
            if m["compacted"]:
                comp_tok.append(m["compaction_slot_tokens"])
                spine_depth.append(m["spine_depth"])
                spine_leaves.append(m["spine_leaves"])
                leaf_tok.extend(m["spine_leaf_tokens"])
            ctx_list = json.loads(r["context"] or "[]")
            for c, nt in zip(ctx_list, m["slot_tokens"], strict=True):
                if not c.startswith('{"compaction"') and nt > max_slot:
                    st["slot_over_max_ctx"] += 1
            if any(x > max_leaf for x in m["spine_leaf_tokens"]):
                st["spine_leaf_over_cap"] += 1
            nsl.append(m["n_slots"])
            n_eval += r["split"] == "eval"
            if k % 25 == 0:
                tgt = json.loads(r["target"])[0]
                txt = tgt["content"] + (tgt.get("reasoning_content") or "")
                tgt_tok.append(len(tok(txt, add_special_tokens=False)["input_ids"]))
            writer.add(r)
    writer.close()
    report = {
        "source": args.source,
        "family": family,
        "dataset": repo,
        "license": lic,
        "out": str(out),
        "rows": writer.n,
        "rows_eval": n_eval,
        "stats": dict(sorted(st.items())),
        "benchmark_exclusions_by_set": dict(excluded_by),
        "citation_hunting_phrases_top": dict(cite_phrases.most_common(15)),
        "correct_rate_among_gold": round(st["em_or_contained"] / st["gold_available"], 4)
        if st["gold_available"]
        else None,
        "prefix_tokens_compressed": dist(pre_c),
        "prefix_text_tokens": dist(pre_t),
        # text tokens folded into the spine (the encoder reads it leaf by leaf, <= 7680 each)
        "compaction_text_tokens": dict(
            dist(comp_tok), rows=len(comp_tok)
        ),
        "slots_per_row": dist(nsl),
        "spine_depth_cells": dist(spine_depth),
        "spine_leaves": dist(spine_leaves),
        "spine_leaf_tokens": dist(leaf_tok),
        # content (+ reasoning_content with --keep-reasoning), in --tokenizer tokens
        "target_content_tokens_sampled": {
            "p50": pct(tgt_tok, 0.5),
            "p95": pct(tgt_tok, 0.95),
            "max": max(tgt_tok, default=0),
            "frac_gt_640": round(sum(x > 640 for x in tgt_tok) / max(1, len(tgt_tok)), 4),
            **(
                {"frac_gt_1024": round(sum(x > 1024 for x in tgt_tok) / max(1, len(tgt_tok)), 4)}
                if args.keep_reasoning
                else {}
            ),
        },
        **({"keep_reasoning": True} if args.keep_reasoning else {}),
        "wall_s": round(time.time() - t0),
    }
    if not writer.n:
        print(json.dumps(report, indent=1))
        sys.exit("no rows")
    out.with_suffix(".parquet.tmp").replace(out)
    # the .arrow copy (memory-mapped by TaskDataset), streamed batch by batch
    tmpa = arrow.with_suffix(".arrow.tmp")
    pf = pq.ParquetFile(str(out))
    with pa.OSFile(str(tmpa), "wb") as sink, pa.ipc.new_file(sink, pf.schema_arrow) as w:
        for batch in pf.iter_batches(batch_size=4096):
            w.write_batch(batch)
    tmpa.replace(arrow)
    assert write_task_store_arrow  # same format as bgkit2.data.tasks.write_task_store_arrow
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
