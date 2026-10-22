"""OFFLINE WEB-RESEARCH eval: answer a question by searching and reading a fixed page corpus.

The CPU-side logic of ``scripts/bgkit2_eval_web.py`` (that script only adds the tokenizer, the
student-server HTTP models and the CLI), kept here so it is unit-testable with a scripted model --
the same split as ``swebench_loc`` / ``scripts/bgkit2_eval_swebench_loc.py``.

Per (arm, item) the existing ``run_episode`` loop runs over a ``ToolRegistry`` that carries ONLY
the web tools (``web_search`` / ``web_open`` / ``web_find``, ``attach_web`` + ``keep_only``) over
the item's corpus (the HotpotQA+MuSiQue paragraph pool, or the BrowseComp-Plus corpus). The task
is the question alone: the answers and the gold pages are only used when the record is written.

Arms (one student's weights, same tools, same loop, same ``decoder_token_budget``):

  bgkit      every compressible result (web_search, web_open) is a slot of encoder reps, its
             ``focus`` the encoder prompt; ``RatioPolicy`` sets the rep count
  full_text  every result as plain text. ``full_text_results="match"`` (default) cuts each
             compressible result to the SAME number of decoder positions the bgkit arm spends on
             it (``RatioPolicy.reps(n_tokens, tool)``) -- the equal-budget text baseline, like
             ``text@B`` in scripts/bgkit2_bench.py; ``"none"`` gives the verbatim result (the
             swebench_loc behaviour: only the episode budget and compaction bound it)

``web_find`` (2026-09-22, user: exact-quote tools go through bgkit too) re-encodes the WHOLE page
as a slot prompted ``find <pattern> in <title>: <focus>`` at ratio 4 (``web_find_mode="encoded"``,
the default and the trained semantics); ``web_find_mode="verbatim"`` is the CONTROL arm that
measures the precision cost: the page's matching lines, verbatim and uncompressed (the pre-change
tool; its system prompt says so -- ``web_system(mode)``).

Scoring (``episode_record`` / ``summarize``): the last ``ANSWER:`` line of the final message (a
bare last line is used when the tag is missing: form ``bare``), SQuAD-normalised EM / F1 against
every gold alias; gold-page retrieval from the ``web_open`` / ``web_search`` calls -- the gold ids
are the supporting paragraphs (HotpotQA supporting facts, MuSiQue ``is_supporting``) or
BrowseComp-Plus ``gold_docs`` (``evidence_docs`` reported separately). Decoder tokens are
``run_episode``'s accounting; encoder tokens are the slot tokens of every compressed generate call
(the encoder re-reads the slots each turn) and, separately, each compressible result once.
"""

from __future__ import annotations

import json
import re
import statistics
import string
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from .context import Conversation, RatioPolicy, Renderer
from .runner import AgentTask, EpisodeResult, Model, TrajectoryLog, run_episode
from .tools import ToolRegistry, ToolResult
from .web import SNIPPET_WORDS, WEB_TOOL_NAMES

ARMS = ("bgkit", "full_text")
FULL_TEXT_RESULTS = ("match", "none")

_WEB_SYSTEM_HEAD = (
    "You are a research assistant answering a question from an offline web corpus. You do not "
    "know the answer in advance and must find it in the pages. Use `web_search` to find pages "
    "(each result starts with its page id in brackets), `web_open` with a page id from the "
    "results to read that page, and "
)
_WEB_SYSTEM_FIND = {
    "encoded": (
        "`web_find` to look for a phrase on a page (the page opened last, or `id`): the whole "
        "page is read again for you with the phrase as what to keep. Give a `focus` with every "
        "call saying what you are looking for; the results may be summarised for you. "
    ),
    "verbatim": (
        "`web_find` to get the exact lines of the page opened last that contain a phrase. Give "
        "a `focus` with `web_search` and `web_open` saying what you are looking for; their "
        "results may be summarised for you, `web_find` is verbatim. "
    ),
}
_WEB_SYSTEM_TAIL = (
    "A question may need several pages: search again for each missing fact, with the names you "
    "found. Never repeat a call you already made. When you know the answer, reply with ONLY the "
    "answer on the last line in the form `ANSWER: <answer>` (a short name, number, date or "
    "phrase, no explanation) and no tool call."
)


def web_system(find_mode: str = "encoded") -> str:
    """The web eval's (and the web SFT converter's) system prompt for a ``web_find`` mode."""
    return _WEB_SYSTEM_HEAD + _WEB_SYSTEM_FIND[find_mode] + _WEB_SYSTEM_TAIL


WEB_SYSTEM = web_system("encoded")
WEB_SYSTEM_VERBATIM_FIND = web_system("verbatim")  # the control arm (and the pre-change prompt)
# Web episodes compact as a SPINE (cons, 2026-09-22) and split over-long results into slots of
# at most the encoder's max_ctx (task_max_ctx_tokens) -- the converter's rows do the same.
WEB_COMPACTION_MODE = "cons"
WEB_SLOT_MAX_TOKENS = 8192
WEB_INSTRUCTION = "Search the web corpus to answer the question. Finish with `ANSWER: <answer>`."

_SEARCH_ID = re.compile(r"^\[([^\]\n]+)\] ", re.MULTILINE)
TRUNC_MARK = "\n[... truncated]"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def normalize_answer(s: str) -> str:
    """SQuAD normalisation (lower, no punctuation, no articles, single spaces) -- the same as
    scripts/bgkit2_bench.py's scorer, so web and reading-only scores are comparable."""
    s = "".join(ch for ch in str(s).lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def answer_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def score_answer(pred: str, golds: list[str]) -> dict[str, float]:
    golds = [str(g) for g in golds if str(g).strip()]
    if not golds:
        return {"em": 0.0, "f1": 0.0}
    return {"em": max(float(normalize_answer(pred) == normalize_answer(g)) for g in golds),
            "f1": max(answer_f1(pred, g) for g in golds)}


def parse_answer(text: str) -> tuple[str, str]:
    """(answer, form) from the final message: the LAST line with ``ANSWER:`` (form ``answer``),
    else the last non-empty line (form ``bare``), else ("", ``none``)."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    hits = [ln for ln in lines if "ANSWER:" in ln.upper()]
    if hits:
        ln = hits[-1]
        ans, form = ln[ln.upper().rindex("ANSWER:") + len("ANSWER:") :], "answer"
    elif lines:
        ans, form = lines[-1], "bare"
    else:
        return "", "none"
    ans = ans.strip().strip("*`").strip().strip("\"'").strip()
    ans = ans[:-1] if ans.endswith(".") and not ans.endswith("..") else ans
    return ans.strip(), form if ans.strip() else "none"


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


def load_items(path: str | Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).open() if ln.strip()]


def web_task(item: dict, work_dir: str | Path, web_find_mode: str = "encoded") -> AgentTask:
    """The agent's task: the question only (never the answers or the gold pages)."""
    q = " ".join(str(item["question"]).split())
    return AgentTask(
        task_id=item["id"],
        instruction=f"Question: {item['question'].strip()}\n\n{WEB_INSTRUCTION}",
        repo_dir=str(work_dir),
        system_prompt=web_system(web_find_mode),
        summary=q[:200],
    )


# ---------------------------------------------------------------------------
# Tools + model wrappers (per-episode accounting)
# ---------------------------------------------------------------------------


def truncate_tokens(text: str, n: int, renderer: Renderer) -> str:
    """The longest head of ``text`` that counts at most ``n`` tokens (with ``TRUNC_MARK`` when
    it is cut and the mark fits)."""
    if renderer.count(text) <= n:
        return text
    mark = TRUNC_MARK if renderer.count(TRUNC_MARK) < n else ""
    room = n - (renderer.count(mark) if mark else 0)
    lo, hi = 0, len(text)
    while lo < hi:  # largest L with count(text[:L]) <= room
        mid = (lo + hi + 1) // 2
        if renderer.count(text[:mid]) <= room:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + mark


class EpisodeTools:
    """A ``ToolRegistry`` for one episode that records what the agent searched / opened and, in
    the full_text arm with ``truncate_to_reps``, cuts each compressible result to the decoder
    positions the bgkit arm would spend on it."""

    def __init__(self, reg: ToolRegistry, corpus, renderer: Renderer, ratios: RatioPolicy,
                 truncate_to_reps: bool = False) -> None:
        self.reg, self.corpus, self.renderer, self.ratios = reg, corpus, renderer, ratios
        self.truncate_to_reps = truncate_to_reps
        self.opened: list[str] = []
        self.opened_titles: list[str] = []
        self.searched: list[str] = []  # page ids shown in search results, first-seen order
        self.n_search = self.n_open = self.n_find = 0
        self.n_calls: Counter = Counter()  # every tool (browse included), ok or not
        self.result_tokens_compressible = 0  # each compressible result once, before any cut
        self.n_truncated = 0

    def specs(self) -> list[dict]:
        return self.reg.specs()

    def __getattr__(self, name):  # anything else run_episode may need
        return getattr(self.reg, name)

    def call(self, name: str, args: dict | None = None) -> ToolResult:
        res = self.reg.call(name, args)
        self.n_calls[name] += 1
        self.n_search += name == "web_search"
        self.n_open += name == "web_open"
        self.n_find += name == "web_find"
        if res.ok and name == "web_search":
            for pid in _SEARCH_ID.findall(res.text):
                if pid not in self.searched:
                    self.searched.append(pid)
        if res.ok and name == "web_open":
            page = self.corpus.get(str((args or {}).get("id")))
            # the RESOLVED id: a corpus may accept aliases (candtree: "<title> [<id>]")
            pid = page.id if page is not None else str((args or {}).get("id"))
            if pid not in self.opened:
                self.opened.append(pid)
                self.opened_titles.append(page.title if page is not None else "")
        if res.ok and res.compressible:
            n = self.renderer.count(res.text)
            self.result_tokens_compressible += n
            if self.truncate_to_reps:
                cap = self.ratios.reps(n, res.tool or name)
                if n > cap:
                    self.n_truncated += 1
                    res = replace(res, text=truncate_tokens(res.text, cap, self.renderer))
        return res


class EncoderCounter:
    """Wraps a ``Model``: sums the slot tokens the encoder reads on every compressed call."""

    def __init__(self, model: Model) -> None:
        self.model = model
        self.name, self.compressed, self.renderer = model.name, model.compressed, model.renderer
        self.encoder_tokens = 0
        self.calls = 0

    def generate(self, conv: Conversation):
        self.calls += 1
        if self.compressed:
            self.encoder_tokens += sum(s.n_tokens for s in conv.slots)
        return self.model.generate(conv)


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


def _gold_title_norm(t: str) -> str:
    return " ".join(str(t).lower().split())


def episode_record(item: dict, arm: str, res: EpisodeResult, tools: EpisodeTools,
                   enc: EncoderCounter) -> dict:
    meta = item.get("meta") or {}
    final = res.final_message if res.stop_reason == "final" else ""
    pred, form = parse_answer(final)
    sc = score_answer(pred, item["answers"]) if pred else {"em": 0.0, "f1": 0.0}
    gold = [str(g) for g in meta.get("gold_ids") or []]
    evid = [str(g) for g in meta.get("evidence_ids") or []]
    gtitles = {_gold_title_norm(t) for t in meta.get("gold_titles") or []}
    opened = set(tools.opened)
    rec = {
        "id": item["id"],
        "bench": item.get("bench", ""),
        "subset": item.get("subset", ""),
        "arm": arm,
        "pred": pred,
        "answer_form": form,
        "gold": list(item["answers"])[:3],
        **sc,
        "answered": form == "answer",
        "turns": res.turns,
        "decoder_tokens": res.decoder_tokens,
        "encoder_tokens": enc.encoder_tokens,
        "encoder_tokens_unique": tools.result_tokens_compressible if enc.compressed else 0,
        "tool_result_tokens": res.tool_result_tokens,
        "tokens_saved": res.tokens_saved,
        "truncated_results": tools.n_truncated,
        "compactions": res.n_compactions,
        "n_search": tools.n_search,
        "n_open": tools.n_open,
        "n_find": tools.n_find,
        "opened_ids": tools.opened[:50],
        "gold_ids": gold,
        "gold_opened_any": bool(opened & set(gold)),
        "gold_opened_all": bool(gold) and set(gold) <= opened,
        "gold_opened_frac": (len(opened & set(gold)) / len(gold)) if gold else 0.0,
        "gold_searched_any": bool(set(tools.searched) & set(gold)),
        "gold_title_opened_any": bool(gtitles & {_gold_title_norm(t) for t in tools.opened_titles}),
        "stop": res.stop_reason,
        "wall_s": round(res.wall_s, 2),
        "final_message": final[-1500:],
    }
    if evid:
        rec["evidence_opened_any"] = bool(opened & set(evid))
    return rec


def error_record(item: dict, arm: str, exc: Exception) -> dict:
    return {"id": item["id"], "bench": item.get("bench", ""), "subset": item.get("subset", ""),
            "arm": arm, "stop": "error", "error": str(exc)[:300]}


def run_arm(
    item: dict,
    arm: str,
    corpus,
    model: Model,
    work_dir: str | Path,
    ratios: RatioPolicy | None = None,
    renderer: Renderer | None = None,
    full_text_results: str = "match",
    web_k: int = 5,
    max_page_chars: int = 40_000,
    snippet_words: int = SNIPPET_WORDS,
    compaction_mode: str = WEB_COMPACTION_MODE,
    slot_max_tokens: int | None = WEB_SLOT_MAX_TOKENS,
    log: TrajectoryLog | None = None,
    web_find_mode: str = "encoded",
    **episode_kwargs,
) -> dict:
    """One (arm, item) episode with only the web tools over ``corpus`` -> record.
    ``web_find_mode="verbatim"`` = the verbatim-find control arm (tool + system prompt)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
    if full_text_results not in FULL_TEXT_RESULTS:
        raise ValueError(f"full_text_results must be one of {FULL_TEXT_RESULTS}")
    if (arm == "bgkit") != bool(model.compressed):
        raise ValueError(f"arm {arm} needs a {'compressed' if arm == 'bgkit' else 'text'} model")
    ratios = ratios or RatioPolicy()
    renderer = renderer or model.renderer
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    reg = ToolRegistry(str(work_dir), web_find_mode=web_find_mode)
    reg.attach_web(corpus, k=web_k, max_page_chars=max_page_chars, snippet_words=snippet_words)
    reg.keep_only(WEB_TOOL_NAMES)
    tools = EpisodeTools(reg, corpus, renderer, ratios,
                         truncate_to_reps=arm == "full_text" and full_text_results == "match")
    enc = EncoderCounter(model)
    res = run_episode(enc, tools, web_task(item, work_dir, web_find_mode), ratios=ratios,
                      renderer=renderer, log=log, compaction_mode=compaction_mode,
                      slot_max_tokens=slot_max_tokens, **episode_kwargs)
    rec = episode_record(item, arm, res, tools, enc)
    rec["web_find_mode"] = web_find_mode
    return rec


def run_item(
    item: dict,
    arms: list[str],
    make_model: Callable[[str], Model],
    corpus,
    work_dir: str | Path,
    done: dict[str, set[str]] | None = None,
    write: Callable[[str, dict], None] | None = None,
    logs: dict[str, TrajectoryLog] | None = None,
    **arm_kwargs,
) -> list[dict]:
    """Every not-yet-done arm of one item; an exception becomes an ``error`` record (retried on
    resume)."""
    done = done or {}
    recs = []
    for arm in arms:
        if item["id"] in done.get(arm, set()):
            continue
        try:
            rec = run_arm(item, arm, corpus, make_model(arm), work_dir,
                          log=(logs or {}).get(arm), **arm_kwargs)
        except Exception as exc:  # server hiccup etc.
            rec = error_record(item, arm, exc)
        recs.append(rec)
        if write is not None:
            write(arm, rec)
    return recs


def load_records(path: str | Path) -> dict[str, dict]:
    """Last record per item id from an episodes jsonl."""
    out: dict[str, dict] = {}
    p = Path(path)
    if p.exists():
        for ln in p.open():
            if ln.strip():
                r = json.loads(ln)
                out[r["id"]] = r
    return out


def load_done(path: str | Path) -> set[str]:
    return {k for k, v in load_records(path).items() if v.get("stop") != "error"}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _mean(xs) -> float | None:
    xs = [float(x) for x in xs]
    return round(statistics.mean(xs), 4) if xs else None


_MEANS = ("em", "f1", "answered", "turns", "decoder_tokens", "encoder_tokens",
          "encoder_tokens_unique", "tool_result_tokens", "tokens_saved", "truncated_results",
          "n_search", "n_open", "n_find", "gold_opened_any", "gold_opened_all",
          "gold_opened_frac", "gold_searched_any", "gold_title_opened_any", "wall_s")


def summarize(records: list[dict]) -> dict:
    """Per-arm metrics over scored episodes (errors are counted, not scored as zeros)."""
    ok = [r for r in records if r.get("stop") != "error"]
    out: dict = {"n": len(ok), "errors": sum(r.get("stop") == "error" for r in records)}
    for k in _MEANS:
        out[k] = _mean(r[k] for r in ok if k in r)
    out["answered_any"] = _mean(bool(r["pred"]) for r in ok)
    ev = [r["evidence_opened_any"] for r in ok if "evidence_opened_any" in r]
    if ev:
        out["evidence_opened_any"] = _mean(ev)
    out["answer_form"] = dict(Counter(r["answer_form"] for r in ok))
    out["stop"] = dict(Counter(r["stop"] for r in ok))
    out["loops"] = sum(r["stop"] == "loop" for r in ok)
    by: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        by[str(r.get("subset", ""))].append(r)
    out["by_subset"] = {s: {"n": len(rs), "em": _mean(r["em"] for r in rs),
                            "f1": _mean(r["f1"] for r in rs),
                            "gold_opened_any": _mean(r["gold_opened_any"] for r in rs)}
                        for s, rs in sorted(by.items())}
    return out


def judge_rows(records_by_arm: dict[str, dict[str, dict]]) -> list[dict]:
    """Rows for scripts/bgkit2_bench_judge.py (``id`` / ``arm`` / ``pred``, plus our scores)."""
    rows = []
    for arm, recs in records_by_arm.items():
        for r in recs.values():
            if r.get("stop") == "error":
                continue
            rows.append({"id": r["id"], "subset": r.get("subset", ""), "arm": arm,
                         "pred": r["pred"], "gold": r.get("gold", []), "em": r["em"],
                         "f1": r["f1"], "decoder_tokens": r["decoder_tokens"],
                         "turns": r["turns"]})
    return rows
