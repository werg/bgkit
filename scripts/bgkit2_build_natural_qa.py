#!/usr/bin/env python
"""Build NATURAL long-document QA task stores for bgkit2 (text parquet, see
bgkit2.data.tasks). Same row schema and conventions as
scripts/bgkit2_build_tasks.py (framing chat_context, prompt
"Compress the text to answer: <question>\\n", meta.doc = document key shared
by sibling rows, meta.gold_window = verbatim supporting sentence(s)).

Why: the long documents in the current mix (multineedle, babi_filler) are
stitched / synthetic and the model started specialising to that texture.
These stores are real documents with several questions each (siblings), so
the prompt-contrast loss keeps working on natural long text.

  squad_article        SQuAD v1.1 regrouped by Wikipedia ARTICLE (windows of
                       whole paragraphs up to --window-tokens LFM tokens);
                       gold_window = the sentence holding the answer span
  quality              QuALITY (tasksource/quality): article + its questions,
                       target = the correct option's TEXT. v1 rendered the
                       options into the instruction, so the decoder copied an
                       option (teacher-forced loss ~0.3 nats WITHOUT context,
                       generative EM 0.30 = chance); v2 (--free-form, default)
                       asks the question alone and keeps options + gold letter
                       in meta for a multiple-choice eval; --drop-guessable
                       removes questions that only make sense against the
                       option list or whose answer needs no reading
  narrativeqa          NarrativeQA (deepmind/narrativeqa): the SUMMARY is the
                       context, ~30 questions per document (capped)
  hotpotqa_distractor  HotpotQA distractor: 10 titled paragraphs in the given
                       order; gold_window = supporting-fact sentences.
                       One question per document (meta.siblings = false)

Examples (host venv, CPU):
  scripts/bgkit2_build_natural_qa.py squad_article --out $B/tasks/qa_squad_article.parquet
  scripts/bgkit2_build_natural_qa.py quality --out $B/tasks/qa_quality.parquet
  scripts/bgkit2_build_natural_qa.py narrativeqa --out $B/tasks/qa_narrativeqa.parquet
  scripts/bgkit2_build_natural_qa.py hotpotqa_distractor \\
      --out $B/tasks/qa_hotpotqa_distractor.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgkit2.data.tasks import write_task_store

LFM_ID = "LiquidAI/LFM2.5-350M"
# Sentence boundary: terminal punctuation + whitespace, but NOT after common
# abbreviations / initials ("U.S. Food", "St. Helena", "J. R. R. Tolkien") —
# v1 split there and produced 198 truncated / 71 lowercase-leading gold windows.
_ABBREV = (
    r"(?<!\b[A-Z])(?<!\bU\.S)(?<!\bU\.K)(?<!\bSt)(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)"
    r"(?<!\bJr)(?<!\bSr)(?<!\bvs)(?<!\bNo)(?<!\bInc)(?<!\bCo)(?<!\betc)(?<!\be\.g)(?<!\bi\.e)"
    r"(?<!\bProf)(?<!\bGen)(?<!\bLt)(?<!\bCol)(?<!\bMt)(?<!\bFt)(?<!\bJan)(?<!\bFeb)(?<!\bAug)"
    r"(?<!\bSept)(?<!\bOct)(?<!\bNov)(?<!\bDec)"
)
_SENT_BOUNDARY = re.compile(_ABBREV + r"(?<=[.!?])\s+(?=[^a-z])")


def _lfm_tokenizer():
    """The LFM2.5 tokenizer from the HF cache (the same loader as
    tests/unit/bgkit2/test_tasks.py::_real_tok), used only to size windows."""
    from transformers import AutoTokenizer

    cache = Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots"
    snaps = sorted(cache.glob("*/tokenizer.json")) if cache.exists() else []
    if snaps:
        return AutoTokenizer.from_pretrained(str(snaps[0].parent))
    return AutoTokenizer.from_pretrained(LFM_ID)


def _split_of(key: str, eval_frac: float) -> str:
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if h < eval_frac else "train"


def _sentence_at(text: str, start: int, end: int) -> str:
    """The sentence(s) of ``text`` covering [start, end) — a verbatim
    substring, extended to whole sentences on both sides."""
    bounds = [0] + [m.end() for m in _SENT_BOUNDARY.finditer(text)] + [len(text)]
    s0 = max(b for b in bounds if b <= start)
    s1 = min(b for b in bounds if b >= end)
    return text[s0:s1].strip()


def _cap(items: list, k: int | None, rng: random.Random) -> list:
    if k is None or len(items) <= k:
        return items
    keep = sorted(rng.sample(range(len(items)), k))
    return [items[i] for i in keep]


# ----------------------------------------------------------------------------
def build_squad_article(args) -> list[dict]:
    """SQuAD v1.1 regrouped by article. Paragraphs are concatenated in their
    original order (blank-line separated) into windows of at most
    ``--window-tokens`` LFM tokens (whole paragraphs; a longer article
    becomes consecutive windows, each its own document). Every question of a
    paragraph in the window is a row; gold_window = the sentence of that
    paragraph containing the answer span (located with answer_start).
    meta.doc = "<title>#<window>". Split by ARTICLE: official validation ->
    eval, train -> train, plus a --holdout-frac slice of train articles when
    validation has fewer than --min-eval-articles articles."""
    from datasets import load_from_disk

    tok = _lfm_tokenizer()
    rng = random.Random(args.seed)
    d = load_from_disk(args.squad_dir)
    rows: list[dict] = []
    stats = Counter()

    # Article -> ordered paragraphs -> questions. The HF flattening keeps the
    # original article / paragraph order (rows are contiguous by both).
    def articles_of(ds) -> dict[str, list[tuple[str, list[dict]]]]:
        out: dict[str, list[tuple[str, list[dict]]]] = {}
        seen: dict[str, dict[str, int]] = {}
        for ex in ds:
            title = ex["title"]
            paras = out.setdefault(title, [])
            idx = seen.setdefault(title, {})
            if ex["context"] not in idx:
                idx[ex["context"]] = len(paras)
                paras.append((ex["context"], []))
            paras[idx[ex["context"]]][1].append(ex)
        return out

    n_val_articles = len(set(d["validation"]["title"]))
    holdout = n_val_articles < args.min_eval_articles
    for split_name, base_split in (("train", "train"), ("validation", "eval")):
        arts = articles_of(d[split_name])
        for title, paras in arts.items():
            split = base_split
            if base_split == "train" and holdout:
                split = _split_of(title, args.holdout_frac)
            # token length per paragraph (windows are sized on the paragraph
            # text alone; the "\n\n" joins add a token each at most)
            lens = [
                len(ids)
                for ids in tok([p for p, _ in paras], add_special_tokens=False)["input_ids"]
            ]
            windows: list[list[int]] = [[]]
            have = 0
            for pi, n in enumerate(lens):
                if windows[-1] and have + n > args.window_tokens:
                    windows.append([])
                    have = 0
                windows[-1].append(pi)
                have += n + 1
            for wi, members in enumerate(windows):
                doc_id = f"{title}#{wi}"
                context = "\n\n".join(paras[pi][0] for pi in members)
                # every question whose paragraph is in the window
                cands: list[tuple[int, dict]] = [(pi, q) for pi in members for q in paras[pi][1]]
                stats["questions_total"] += len(cands)
                cands = _cap(cands, args.max_per_doc, rng)
                for pi, ex in cands:
                    answers = ex["answers"]["text"]
                    if not answers:
                        continue
                    para = paras[pi][0]
                    a0 = int(ex["answers"]["answer_start"][0])
                    a1 = a0 + len(answers[0])
                    if para[a0:a1] != answers[0]:
                        # answer_start does not match: find the answer text
                        j = para.find(answers[0])
                        if j < 0:
                            stats["answer_not_found"] += 1
                            continue
                        a0, a1 = j, j + len(answers[0])
                        stats["answer_start_repaired"] += 1
                    gold = _sentence_at(para, a0, a1)
                    if gold not in context:
                        stats["gold_not_verbatim"] += 1
                        gold = ""
                    rows.append(
                        {
                            "family": "qa_squad_article",
                            "framing": "chat_context",
                            "context": context,
                            "prompt": f"Compress the text to answer: {ex['question']}\n",
                            "instruction": ex["question"],
                            "target": answers[0],
                            "split": split,
                            "meta": {
                                "doc": doc_id,
                                "id": ex["id"],
                                "title": title,
                                "window": wi,
                                "n_windows": len(windows),
                                "paragraph": pi,
                                "answers": answers,
                                "gold_window": gold,
                            },
                        }
                    )
                stats[f"docs_{split}"] += 1
        print(
            f"{split_name}: {len(arts)} articles -> rows so far {len(rows)}",
            flush=True,
        )
    print(json.dumps({"holdout_from_train": holdout, **stats}), flush=True)
    return rows


# ----------------------------------------------------------------------------
def build_quality(args) -> list[dict]:
    """QuALITY from HF ``tasksource/quality`` (chosen over emozilla/quality
    because it carries article_id, title and the official ``difficult``
    flag; the article text is identical). context = the article; each of its
    questions is a row; the instruction renders the four options so the
    answer can be GENERATED as the option's text; target = that text.
    Official train -> train, dev (HF "validation") -> eval."""
    from datasets import load_dataset

    d = load_dataset(args.quality_id)
    tok = _lfm_tokenizer() if args.max_ctx_tokens else None
    rows: list[dict] = []
    stats = Counter()
    letters = "ABCD"
    article_tokens: dict[str, int] = {}
    for split_name, split in (("train", "train"), ("validation", "eval")):
        ds = d[split_name]
        for ex in ds:
            opts = [str(o).strip() for o in ex["options"]]
            gold = int(ex["gold_label"]) - 1  # 1-based in the source
            if not (0 <= gold < len(opts)) or len(opts) != 4:
                stats["bad_options"] += 1
                continue
            q = str(ex["question"]).strip()
            doc = str(ex["article_id"])
            if tok is not None:
                if doc not in article_tokens:
                    ids = tok(ex["article"], add_special_tokens=False)["input_ids"]
                    article_tokens[doc] = len(ids)
                if article_tokens[doc] > args.max_ctx_tokens:
                    stats["dropped_article_over_cap"] += 1
                    continue
            reason = None
            if args.drop_guessable:
                reason = _quality_guessable(q, opts[gold], ex["article"])
            if reason:
                stats[f"dropped_{reason}"] += 1
                continue
            if args.free_form:
                instruction = q
            else:
                rendered = " ".join(f"({letters[i]}) {o}" for i, o in enumerate(opts))
                instruction = f"{q}\nOptions: {rendered}"
            rows.append(
                {
                    "family": "qa_quality",
                    "framing": "chat_context",
                    "context": ex["article"],
                    "prompt": f"Compress the text to answer: {q}\n",
                    "instruction": instruction,
                    "target": opts[gold],
                    "split": split,
                    "meta": {
                        "doc": doc,
                        "id": ex["question_unique_id"],
                        "title": ex["title"],
                        "hard": bool(ex["difficult"]),
                        "gold_label": letters[gold],
                        "options": opts,
                        "free_form": bool(args.free_form),
                        "answer_in_article": opts[gold] in ex["article"],
                    },
                }
            )
        print(f"{split_name}: rows so far {len(rows)}", flush=True)
    print(json.dumps(stats), flush=True)
    return rows


_OPTION_RELATIVE_Q = re.compile(
    r"\b(of the following|the following|of these|these options|which option|which statement|"
    r"statement below|option below|listed|all of the above|none of the above|which of the|"
    r"which one of|best describes|most accurately describes|most likely (to be )?true|"
    r"is (not|NOT) true|is (not|NOT) a reason|is (not|NOT) (a|an) |\bexcept\b|least likely|"
    r"which is (not|NOT)|what is (not|NOT))",
    re.I,
)
_OPTION_RELATIVE_A = re.compile(
    r"^(all|none|neither|both|either|any|some)\b|\b(of the above|of these|of the options)\b",
    re.I,
)
_YES_NO = {"yes", "no", "true", "false", "maybe"}


def _quality_guessable(question: str, answer: str, article: str) -> str | None:
    """Why a QuALITY question cannot be asked free-form (None = keep):
    ``option_relative_question`` (it only makes sense against the option
    list: "which of the following", "EXCEPT", "is NOT a reason"),
    ``option_relative_answer`` ("None of the above", "Both ..."),
    ``yes_no`` / ``answer_in_question`` (no reading needed), and
    ``one_word_not_in_article`` (a single generic word the article never
    states — a vocabulary / inference guess rather than retrieval)."""
    a = answer.strip().rstrip(".")
    if _OPTION_RELATIVE_Q.search(question):
        return "option_relative_question"
    if _OPTION_RELATIVE_A.search(a):
        return "option_relative_answer"
    if a.lower() in _YES_NO or re.match(r"^(yes|no)\b[,.]", a, re.I):
        return "yes_no"
    if a.lower() in question.lower():
        return "answer_in_question"
    if len(a.split()) == 1 and a.lower() not in article.lower():
        return "one_word_not_in_article"
    return None


# ----------------------------------------------------------------------------
def build_narrativeqa(args) -> list[dict]:
    """NarrativeQA (HF deepmind/narrativeqa) with the SUMMARY as the context
    (full stories are far beyond 8k tokens). Rows = the document's questions
    (capped per document, seeded), target = the first human answer.
    Official train -> train, validation -> eval (test unused)."""
    import pyarrow.compute as pc
    from datasets import load_dataset

    rng = random.Random(args.seed)
    d = load_dataset(args.narrativeqa_id)
    rows: list[dict] = []
    stats = Counter()
    for split_name, split in (("train", "train"), ("validation", "eval")):
        # Pull the columns through Arrow so the ~800 KB full text of every
        # row is never materialised (32k rows x 0.8 MB otherwise).
        t = d[split_name].data.table
        doc = t.column("document")
        ids = pc.struct_field(doc, "id").to_pylist()
        kinds = pc.struct_field(doc, "kind").to_pylist()
        summaries = pc.struct_field(pc.struct_field(doc, "summary"), "text").to_pylist()
        titles = pc.struct_field(pc.struct_field(doc, "summary"), "title").to_pylist()
        questions = pc.struct_field(t.column("question"), "text").to_pylist()
        answers = t.column("answers").to_pylist()
        by_doc: dict[str, list[int]] = {}
        for i, did in enumerate(ids):
            by_doc.setdefault(did, []).append(i)
        for did, idxs in by_doc.items():
            summary = str(summaries[idxs[0]]).strip()
            if len(summary) < args.min_summary_chars:
                stats["short_summary_docs"] += 1
                continue
            stats["questions_total"] += len(idxs)
            for i in _cap(idxs, args.max_per_doc, rng):
                ans = [a["text"].strip() for a in answers[i] if a.get("text", "").strip()]
                if not ans:
                    continue
                q = str(questions[i]).strip()
                rows.append(
                    {
                        "family": "qa_narrativeqa",
                        "framing": "chat_context",
                        "context": summary,
                        "prompt": f"Compress the text to answer: {q}\n",
                        "instruction": q,
                        "target": ans[0],
                        "split": split,
                        "meta": {
                            "doc": did,
                            "title": titles[idxs[0]],
                            "kind": kinds[idxs[0]],
                            "answers": ans,
                        },
                    }
                )
            stats[f"docs_{split}"] += 1
        print(f"{split_name}: {len(by_doc)} documents -> rows so far {len(rows)}", flush=True)
    print(json.dumps(stats), flush=True)
    return rows


# ----------------------------------------------------------------------------
def build_hotpotqa_distractor(args) -> list[dict]:
    """HotpotQA distractor: all ten paragraphs, each "<title>\\n<sentences>",
    in the ORDER GIVEN (gold is not moved first), blank-line separated.
    gold_window = the supporting-fact sentences joined by " " when each is a
    verbatim substring of the context (else ""). One question per document:
    meta.doc = the question id and meta.siblings = false — this store is not
    for the sibling contrast. Train is a seeded sample of --max-train rows;
    validation -> eval in full."""
    from datasets import load_from_disk

    rng = random.Random(args.seed)
    d = load_from_disk(f"{args.bench_dir}/hotpotqa_distractor")
    rows: list[dict] = []
    stats = Counter()
    for split_name, split in (("train", "train"), ("validation", "eval")):
        ds = d[split_name]
        order = list(range(len(ds)))
        if split == "train" and args.max_train and len(order) > args.max_train:
            order = sorted(rng.sample(order, args.max_train))
        for i in order:
            ex = ds[i]
            titles = ex["context"]["title"]
            sents = ex["context"]["sentences"]
            paras = [f"{t}\n{''.join(s)}" for t, s in zip(titles, sents, strict=True)]
            ctx = "\n\n".join(paras)
            if len(ctx) > args.max_ctx_chars:
                stats["ctx_truncated"] += 1
                ctx = ctx[: args.max_ctx_chars]
            by_title = dict(zip(titles, sents, strict=True))
            pieces: list[str] = []
            ok = True
            for t, sid in zip(
                ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"], strict=True
            ):
                s = by_title.get(t)
                if s is None or not (0 <= sid < len(s)):
                    ok = False
                    break
                piece = s[sid].strip()
                if not piece or piece not in ctx:
                    ok = False
                    break
                pieces.append(piece)
            gold = " ".join(pieces) if ok and pieces else ""
            stats["gold_window_present" if gold else "gold_window_missing"] += 1
            rows.append(
                {
                    "family": "qa_hotpotqa_distractor",
                    "framing": "chat_context",
                    "context": ctx,
                    "prompt": f"Compress the text to answer: {ex['question']}\n",
                    "instruction": ex["question"],
                    "target": ex["answer"],
                    "split": split,
                    "meta": {
                        "doc": ex["id"],
                        "siblings": False,
                        "type": ex["type"],
                        "level": ex["level"],
                        "n_supporting": len(pieces) if ok else 0,
                        "gold_window": gold,
                    },
                }
            )
        print(f"{split_name}: rows so far {len(rows)}", flush=True)
    print(json.dumps(stats), flush=True)
    return rows


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "family", choices=["squad_article", "quality", "narrativeqa", "hotpotqa_distractor"]
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument(
        "--max-per-doc", type=int, default=12, help="questions kept per document (seeded)"
    )
    ap.add_argument("--squad-dir", default="/home/werg/bgkit-data-nvme/bgkit2/benchmarks/squad")
    ap.add_argument("--bench-dir", default="/home/werg/bgkit-data-nvme/bgkit2/benchmarks")
    ap.add_argument(
        "--window-tokens", type=int, default=7000, help="squad_article: LFM tokens per window"
    )
    ap.add_argument("--min-eval-articles", type=int, default=40)
    ap.add_argument("--holdout-frac", type=float, default=0.05)
    ap.add_argument("--quality-id", default="tasksource/quality")
    ap.add_argument(
        "--free-form",
        type=int,
        default=1,
        help="quality: instruction = the question alone (options stay in meta); 0 = v1 rendering",
    )
    ap.add_argument(
        "--drop-guessable",
        type=int,
        default=1,
        help="quality: drop option-relative / yes-no / answer-in-question / unsupported "
        "one-word rows",
    )
    ap.add_argument(
        "--max-ctx-tokens",
        type=int,
        default=8192,
        help="quality: drop articles longer than this many LFM tokens (the encoder cap); 0 = keep",
    )
    ap.add_argument("--narrativeqa-id", default="deepmind/narrativeqa")
    ap.add_argument("--min-summary-chars", type=int, default=200)
    ap.add_argument(
        "--max-train",
        type=int,
        default=40000,
        help="hotpotqa_distractor: train rows (seeded sample)",
    )
    ap.add_argument("--max-ctx-chars", type=int, default=24000)
    args = ap.parse_args()
    if Path(args.out).exists():
        sys.exit(f"{args.out} exists; refusing to overwrite")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = {
        "squad_article": build_squad_article,
        "quality": build_quality,
        "narrativeqa": build_narrativeqa,
        "hotpotqa_distractor": build_hotpotqa_distractor,
    }[args.family](args)
    n = write_task_store(args.out, rows)
    splits: Counter = Counter(r["split"] for r in rows)
    docs = {sp: len({r["meta"]["doc"] for r in rows if r["split"] == sp}) for sp in splits}
    print(json.dumps({"out": args.out, "rows": n, "splits": dict(splits), "docs": docs}))


if __name__ == "__main__":
    main()
