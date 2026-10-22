#!/usr/bin/env python
"""Long-term MEMORY episodes for the showcase mix (2026-09-21): a multi-session conversation
history in COMPRESSED SLOTS (one slot per dated session, like the LongMemEval / LoCoMo harness),
and a question answered from it. CPU, host venv.

Sources (decoded from the phase-2 mmaps, Qwen3.5 tokenizer): Conversation Chronicles (160k
five-session episodes, role pairs like "Classmates A/B", per-session + whole-history summaries),
MSC and SHARE dialogues (distractor sessions only: their questions are open-ended recall with no
single answer).

--kind recall (no teacher needed): the target episode's sessions, dated, scattered in order among
distractor sessions (other role pairs only -- never the same pair label) up to a sampled history
length; question = "Summarize the conversation between the <pair> on <date>." (4 phrasings) ->
that session's summary (sessions 1-3: the only ones aligned with their summaries in the mmap).

--kind qa (teacher sub-agent, LongMemEval question types): the same haystacks; the sub-agent writes
questions over the target sessions -- single-session fact, multi-session, temporal, knowledge
update, abstention -- with the answer; rows whose answer is not grounded in the sessions (except
abstention) are dropped. Needs --teacher-url.

    .venv/bin/python scripts/bgkit2_build_memory_store.py --kind recall --n-rows 40000 \\
        --out /home/werg/bgkit-data-nvme/bgkit2/tasks/memory_recall_v1.parquet
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bgkit2.data.memory_framing import MEMORY_SYSTEM, memory_user  # noqa: E402
from bgkit2.data.tasks import write_task_store  # noqa: E402
from bgkit2.data.templates import SPLICE_SENTINEL  # noqa: E402

MM = "/mnt/external/bgkit-data/mmap/phase2"
SYSTEM = MEMORY_SYSTEM  # shared with the showcase bench (bgkit2.data.memory_framing)
PAIR_RE = re.compile(r"^([A-Z][\w' -]{1,40}?) [AB]:", re.M)
# history length (tokens of the whole memory) ~ LoCoMo (~25k) .. LongMemEval_s (~115k)
LENGTHS = [(0.30, 4000, 16000), (0.40, 16000, 48000), (0.30, 48000, 110000)]


class Mmap:
    def __init__(self, name: str, tok) -> None:
        p = f"{MM}/{name}/"
        self.off = np.load(p + "offsets.npy")
        self.toks = np.load(p + "tokens.npy", mmap_mode="r")
        self.qo = np.load(p + "question_offsets.npy")
        self.qt = np.load(p + "question_tokens.npy", mmap_mode="r")
        self.ao = np.load(p + "answer_offsets.npy")
        self.at = np.load(p + "answer_tokens.npy", mmap_mode="r")
        self.tok = tok
        self.n = len(self.off) - 1

    def doc(self, i: int) -> str:
        return self.tok.decode(self.toks[self.off[i] : self.off[i + 1]])

    def q(self, i: int) -> str:
        return self.tok.decode(self.qt[self.qo[i] : self.qo[i + 1]])

    def a(self, i: int) -> str:
        return self.tok.decode(self.at[self.ao[i] : self.ao[i + 1]])


def sessions_of(doc: str) -> list[str]:
    return [s.strip() for s in doc.split("[Session break]") if s.strip()]


def pair_label(session: str) -> str | None:
    m = PAIR_RE.search(session)
    return m.group(1).strip() if m else None


def sample_length(rng: random.Random) -> int:
    x, acc = rng.random(), 0.0
    for p, lo, hi in LENGTHS:
        acc += p
        if x <= acc:
            return rng.randint(lo, hi)
    return LENGTHS[-1][2]


def timeline(rng: random.Random, n: int) -> list[dt.datetime]:
    t = dt.datetime(2022, 1, 1) + dt.timedelta(days=rng.randint(0, 900), hours=rng.randint(8, 20))
    out = []
    for _ in range(n):
        out.append(t)
        t += dt.timedelta(hours=rng.choice([3, 20, 30, 50, 80, 120, 200, 400, 900]),
                          minutes=rng.randint(0, 59))
    return out


def fmt_date(t: dt.datetime) -> str:
    return t.strftime("%Y/%m/%d (%a) %H:%M")


_TURN = re.compile(r"^([^\n:]{1,60}):", re.M)


def as_user_assistant(session: str) -> str:
    """LongMemEval framing: the session's first speaker becomes `user`, the second `assistant`
    (any further speakers are folded into the nearest of the two by alternation)."""
    order: list[str] = []
    for m in _TURN.finditer(session):
        if m.group(1) not in order:
            order.append(m.group(1))
    role = {sp: ("user" if i % 2 == 0 else "assistant") for i, sp in enumerate(order)}
    return _TURN.sub(lambda m: f"{role.get(m.group(1), 'user')}:", session)


def first_person_summary(summary: str, pair: str) -> str:
    """Rewrite a Chronicles summary ('Co-worker A ... Co-worker B ...', 'Two co-workers ...') for
    the user/assistant framing, answered by the assistant: A -> you, B -> I, both -> we."""
    stem = pair.lower().rstrip("s")
    s = summary
    s = re.sub(rf"\b(two|the|both) {re.escape(stem)}s?\b", "we", s, flags=re.I)
    s = re.sub(rf"\b{re.escape(stem)}s? a and (?:{re.escape(stem)}s? )?b\b", "you and I", s,
               flags=re.I)
    s = re.sub(rf"\b{re.escape(stem)}s? a's\b", "your", s, flags=re.I)
    s = re.sub(rf"\b{re.escape(stem)}s? b's\b", "my", s, flags=re.I)
    s = re.sub(rf"\b{re.escape(stem)}s? a\b", "you", s, flags=re.I)
    s = re.sub(rf"\b{re.escape(stem)}s? b\b", "I", s, flags=re.I)
    s = re.sub(r"(^|[.!?]\s+)(we|you)\b", lambda m: m.group(1) + m.group(2).capitalize(), s)
    return s


def build_haystack(rng, target: list[str], pools, target_len: int, count_tokens, forbid: str,
                   style: str = "pairs"):
    """target sessions (in order) scattered among distractors -> (slots, dates of target sessions)."""
    distract: list[str] = []
    total = sum(count_tokens(s) for s in target)
    tries = 0
    while total < target_len and tries < 5000:
        tries += 1
        pool = rng.choice(pools)
        s = pool(rng)
        if not s or (forbid and pair_label(s) == forbid):
            continue
        distract.append(s)
        total += count_tokens(s)
    n = len(target) + len(distract)
    pos = sorted(rng.sample(range(n), len(target)))
    dates = timeline(rng, n)
    slots, tdates, di, ti = [], [], 0, 0
    for k in range(n):
        if ti < len(target) and k == pos[ti]:
            s = target[ti]
            tdates.append(dates[k])
            ti += 1
        else:
            s = distract[di]
            di += 1
        if style == "user_assistant":
            s = as_user_assistant(s)
        slots.append(f"[session on {fmt_date(dates[k])}]\n{s}")
    return slots, tdates, total


def make_row(family, slots, question, answer, split, meta) -> dict:
    user = memory_user("\n".join([SPLICE_SENTINEL] * len(slots)), question)
    return {"family": family, "framing": "tool_slots", "context": json.dumps(slots),
            "prompt": json.dumps([question] * len(slots)),
            "instruction": json.dumps([{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": user}]),
            "target": answer.strip(), "split": split, "meta": meta}


QA_SYSTEM = (
    "You write memory questions for testing an assistant's long-term memory of dated conversation "
    "sessions. Reply with ONE JSON object and nothing else."
)
QA_TYPES = {
    "single_session": "a specific detail stated in exactly one session (a name, number, place, "
                      "plan, preference or event)",
    "multi_session": "a question that needs facts from at least two different sessions combined",
    "temporal": "a question about WHEN or in what ORDER something happened, answerable from the "
                "session dates (e.g. which came first, how many days apart, what was discussed "
                "on a date)",
    "knowledge_update": "a fact that CHANGES between sessions (a plan, status, opinion or number); "
                        "the question asks for the latest state",
    "abstention": "a plausible question about these people whose answer is NOT in any session; "
                  "the answer must be exactly: That was not mentioned in our conversations.",
    # S2 (2026-09-22): LongMemEval's remaining types; user_assistant style only
    "single_session_assistant": "a detail the ASSISTANT said in one session (a recommendation, "
                                "fact, number or name it gave); the user asks what the assistant "
                                "told them",
    "preference": "the user asks for a recommendation or advice; the right answer takes into "
                  "account a preference, habit or constraint the USER stated in a session -- "
                  "answer with a short personalised suggestion that names that preference",
    "temporal_relative": "a question RELATIVE to the asked-on date given above (how many days / weeks / "
                         "months ago something happened, or what happened last week / month); "
                         "compute the answer from the session dates",
}
# v1 mix (the types the v1 stores were built with) and the S2 user_assistant mix
QA_MIX_V1 = {"single_session": 0.3, "multi_session": 0.2, "temporal": 0.2,
             "knowledge_update": 0.15, "abstention": 0.15}
QA_MIX_S2 = {"single_session": 0.15, "multi_session": 0.15, "temporal": 0.1,
             "temporal_relative": 0.15, "knowledge_update": 0.15, "abstention": 0.1,
             "single_session_assistant": 0.1, "preference": 0.1}
UA_ONLY = {"single_session_assistant", "preference"}
ABSTAIN = "That was not mentioned in our conversations."


def asked_on_stamp(last: dt.datetime, rng: random.Random) -> str:
    """LongMemEval's question-date prefix, 1-30 days after the last session of the haystack."""
    t = last + dt.timedelta(days=rng.randint(1, 30), minutes=rng.randint(0, 24 * 60 - 1))
    return f"(asked on {fmt_date(t)})"


def qa_prompt(pair: str, sessions: list[str], dates: list, qtype: str,
              style: str = "pairs", asked_on: str = "") -> list[dict]:
    if style == "user_assistant":
        sessions = [as_user_assistant(x) for x in sessions]
    body = "\n\n".join(f"[session on {fmt_date(d)}]\n{s}" for s, d in zip(sessions, dates,
                                                                          strict=True))
    if style == "user_assistant":
        who = ("Write the question in the FIRST PERSON as the user asking the assistant about "
               "their own past conversations (e.g. 'What did I say I ...', 'When did I ...').")
        when = f"The question is {asked_on.strip('()')}.\n" if asked_on else ""
        user = (f"SESSIONS between the user and the assistant:\n{body}\n\n{when}"
                f"Write ONE question of type '{qtype}': {QA_TYPES[qtype]}.\n{who} The answer must "
                "be short (a few words or one sentence) and follow from the sessions alone.\n"
                'Reply as {"question": "...", "answer": "...", "evidence": "<quote from the '
                'sessions>"}')
        return [{"role": "system", "content": QA_SYSTEM}, {"role": "user", "content": user}]
    when = f"The question is {asked_on.strip('()')}.\n" if asked_on else ""
    user = (f"SESSIONS between the {pair}:\n{body}\n\n{when}"
            f"Write ONE question of type '{qtype}': {QA_TYPES[qtype]}.\n"
            "Refer to the people as 'the " + pair + "' (or A / B). The answer must be short "
            "(a few words or one sentence) and follow from the sessions alone.\n"
            'Reply as {"question": "...", "answer": "...", "evidence": "<quote from the sessions>"}')
    return [{"role": "system", "content": QA_SYSTEM}, {"role": "user", "content": user}]


def _norm(x: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", x.lower()).split())


def grounded(answer: str, evidence: str, sessions_text: str) -> bool:
    """Evidence must come from the (dated) sessions: split into fragments (the teacher joins
    quotes with '...' and newlines, drops speaker prefixes, re-punctuates), and at least half of
    the fragments (>= 12 normalised chars) must occur; dates it cites must be session dates. Half
    of the answer's content words must occur in the sessions too."""
    st = _norm(sessions_text)
    body = re.sub(r"\[?session on [^\]:]*[\]:]?", "\n", evidence, flags=re.I)  # headers cited inline
    frags = [_norm(f) for f in re.split(r"\.\.\.|\u2026|\n|\"|'(?=\s)|(?<=\s)'", body)]
    frags = [f for f in frags if len(f) >= 12]
    dates = re.findall(r"\d{4}/\d{2}/\d{2}", evidence)
    if dates and not all(d in sessions_text for d in dates):
        return False
    if frags:
        found = sum(f[:60] in st for f in frags)
        if found * 2 < len(frags):
            return False
    elif not dates:
        return False
    words = [w for w in re.findall(r"[a-z0-9']+", answer.lower()) if len(w) > 3]
    if not words:
        return True
    return sum(w in st for w in words) / len(words) >= 0.5


def main_qa(args, chron, pools, count_tokens, n_ep) -> None:
    """Teacher sub-agent questions in LongMemEval's types over the target sessions, each asked
    against its own haystack (the target sessions' dates are fixed by the haystack first)."""
    import threading
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    rng = random.Random(args.seed + 1000 * args.shard)
    eps = [e for e in random.Random(args.seed + 7).sample(range(n_ep), min(n_ep, args.n_rows))
           if e % args.n_shards == args.shard]
    jobs = []
    for e in eps:
        ss = sessions_of(chron.doc(e * 6))
        pair = pair_label(ss[0]) if ss else None
        if len(ss) < 3 or not pair:
            continue
        slots, tdates, total = build_haystack(rng, ss, pools, sample_length(rng), count_tokens,
                                              pair, args.style)
        mix = QA_MIX_S2 if args.qa_mix == "s2" else QA_MIX_V1
        if args.style != "user_assistant":
            mix = {k: v for k, v in mix.items() if k not in UA_ONLY}
        qtype = rng.choices(list(mix), weights=list(mix.values()))[0]
        stamp = ""
        if args.qa_mix == "s2" and (qtype == "temporal_relative" or rng.random() < 0.8):
            stamp = asked_on_stamp(max(tdates), rng)
        jobs.append((e, pair, ss, tdates, slots, total, qtype, stamp))
    lock = threading.Lock()
    rows, stats = [], {"asked": 0, "kept": 0, "unparsable": 0, "ungrounded": 0, "failed": 0}

    def one(job) -> None:
        e, pair, ss, tdates, slots, total, qtype, stamp = job
        body = {"model": "x", "messages": qa_prompt(pair, ss, tdates, qtype, args.style, stamp),
                "max_tokens": 300,
                "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(f"{args.teacher_url}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                text = json.loads(r.read().decode())["choices"][0]["message"].get("content") or ""
        except Exception:
            with lock:
                stats["failed"] += 1
            return
        m = re.search(r"\{.*\}", text, re.S)
        try:
            d = json.loads(m.group(0)) if m else None
        except Exception:
            d = None
        if not isinstance(d, dict) or not d.get("question") or not d.get("answer"):
            with lock:
                stats["unparsable"] += 1
            return
        q, a = str(d["question"]).strip(), str(d["answer"]).strip()
        if stamp:
            q = f"{stamp} {q}"
        if qtype == "abstention":
            a = ABSTAIN
        # preference answers are suggestions and temporal_relative answers are computed ("3 weeks
        # ago"): their EVIDENCE must be grounded, their answer words need not occur in the sessions
        elif not grounded("" if qtype in ("preference", "temporal_relative") else a,
                          str(d.get("evidence", "")), "\n".join(
                f"[session on {fmt_date(t)}] "
                f"{as_user_assistant(x) if args.style == 'user_assistant' else x}"
                for x, t in zip(ss, tdates, strict=True))):
            with lock:
                stats["ungrounded"] += 1
                if args.rejects:
                    with open(args.rejects, "a") as fh:
                        fh.write(json.dumps({"qtype": qtype, "q": q, "a": a,
                                             "evidence": str(d.get("evidence", ""))}) + "\n")
            return
        h = int(hashlib.sha1(str(e).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        row = make_row("memory_qa" if args.style == "pairs" else "memory_qa_ua", slots, q, a, "eval" if h < args.eval_frac else "train",
                       {"source": "chronicles+teacher", "episode": int(e), "qtype": qtype,
                        **({"asked_on": stamp} if stamp else {}),
                        "history_tokens": total, "n_sessions": len(slots)})
        with lock:
            rows.append(row)
            stats["kept"] += 1

    with ThreadPoolExecutor(args.concurrency) as ex:
        for i, _ in enumerate(ex.map(one, jobs)):
            stats["asked"] = i + 1
            if (i + 1) % 500 == 0:
                print(json.dumps(stats), flush=True)
            if stats["failed"] >= 50 and stats["failed"] > stats["kept"]:
                # the teacher is gone (2026-09-21: the memory guard evacuated it and the chain
                # wrote 14 EMPTY shards in minutes, which the resume logic then treated as done)
                ex.shutdown(wait=False, cancel_futures=True)
                print(json.dumps({"ABORTED_TEACHER_DOWN": True, **stats}), flush=True)
                raise SystemExit(3)
    write_task_store(args.out, rows)
    print(json.dumps({"out": args.out, **stats}), flush=True)


def main() -> None:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="recall", choices=["recall", "qa"])
    ap.add_argument("--n-rows", type=int, default=40000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--eval-frac", type=float, default=0.01)
    ap.add_argument("--teacher-url", default="")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--rejects", default="", help="jsonl of rejected (ungrounded) questions")
    ap.add_argument("--style", default="pairs", choices=["pairs", "user_assistant"],
                    help="user_assistant = LongMemEval framing (first speaker 'user', second "
                         "'assistant', first-person questions)")
    ap.add_argument("--qa-mix", default="v1", choices=["v1", "s2"],
                    help="s2 = all eight LongMemEval-style types + LongMemEval's asked-on date "
                         "prefix on 80%% of questions (always on temporal_relative)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()
    qtok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    def count_tokens(s: str) -> int:
        # history LENGTH only steers how many distractors to add: chars/4 is close enough and
        # tokenising ~100 sessions per row made the builder ~1 s/row
        return len(s) // 4

    chron, msc, share = Mmap("chronicles", qtok), Mmap("msc", qtok), Mmap("share", qtok)
    n_ep = chron.n // 6
    rng = random.Random(args.seed + 1000 * args.shard)

    def chron_session(r):
        e = r.randrange(n_ep)
        ss = sessions_of(chron.doc(e * 6))
        return r.choice(ss) if ss else None

    def msc_session(r):
        return msc.doc(r.randrange(msc.n))

    def share_session(r):
        return share.doc(r.randrange(share.n))

    pools = [chron_session, chron_session, msc_session, share_session]
    if args.kind == "recall" and args.style == "user_assistant":
        # tried 2026-09-21: rewriting the dataset's third-person summaries into first person by
        # regex gives broken English ("You expresses frustration with I") -- training targets must
        # be fluent. The user/assistant framing comes from the teacher QA family instead.
        raise SystemExit("--kind recall --style user_assistant is not supported (see comment)")
    if args.kind == "qa":
        if not args.teacher_url:
            raise SystemExit("--kind qa needs --teacher-url")
        main_qa(args, chron, pools, count_tokens, n_ep)
        return
    rows, stats = [], {"kept": 0, "no_pair": 0, "bad_sessions": 0}
    eps = [e for e in random.Random(args.seed).sample(range(n_ep), min(n_ep, args.n_rows))
           if e % args.n_shards == args.shard]
    for ei, e in enumerate(eps):
        base = e * 6
        ss = sessions_of(chron.doc(base))
        # The phase-2 conversion kept 4 of the 5 sessions (3 breaks per doc); summaries 1-3 align
        # with doc sessions 1-3 (checked by hand 2026-09-21), later ones do not, and the
        # whole-history summary covers a session the doc lacks -> ask about sessions 1-3 only.
        if len(ss) < 3:
            stats["bad_sessions"] += 1
            continue
        pair = pair_label(ss[0])
        if not pair:
            stats["no_pair"] += 1
            continue
        qi = rng.randrange(3)  # session 1-3 summary
        slots, tdates, total = build_haystack(rng, ss, pools, sample_length(rng), count_tokens,
                                              pair, args.style)
        if args.style == "user_assistant":
            day = tdates[qi].strftime('%Y/%m/%d')
            if sum(x.startswith(f"[session on {day}") for x in slots) != 1:
                stats["ambiguous_date"] = stats.get("ambiguous_date", 0) + 1
                continue  # no pair label to disambiguate: the date alone must
            q = rng.choice(["Summarize our conversation on {d}.", "What did we talk about on {d}?",
                            "Can you recap what we discussed on {d}?",
                            "What was our chat on {d} about?"]).format(d=day)
            ans = first_person_summary(chron.a(base + qi), pair)
            if re.search(rf"\b{re.escape(pair.lower().rstrip('s'))}", ans.lower()):
                stats["unrewritable_summary"] = stats.get("unrewritable_summary", 0) + 1
                continue  # still names the role pair: the framing has no such people
            h = int(hashlib.sha1(str(e).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            rows.append(make_row("memory_recall_ua", slots, q, ans,
                                 "eval" if h < args.eval_frac else "train",
                                 {"source": "chronicles", "episode": int(e), "q": qi,
                                  "history_tokens": total, "n_sessions": len(slots),
                                  "style": "user_assistant"}))
            stats["kept"] += 1
            continue
        who = pair if pair.lower().startswith(("the ", "a ")) else f"the {pair}"
        q = rng.choice([
            "Summarize the conversation between {w} on {d}.",
            "What did {w} talk about on {d}?",
            "Recap my conversation with {w} from {d}.",
            "What happened when {w} talked on {d}?",
        ]).format(w=who, d=tdates[qi].strftime('%Y/%m/%d'))
        ans = chron.a(base + qi)
        h = int(hashlib.sha1(str(e).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        rows.append(make_row("memory_recall", slots, q, ans,
                             "eval" if h < args.eval_frac else "train",
                             {"source": "chronicles", "episode": int(e), "q": qi,
                              "history_tokens": total, "n_sessions": len(slots)}))
        stats["kept"] += 1
        if stats["kept"] % 2000 == 0:
            print(json.dumps(stats), flush=True)
    write_task_store(args.out, rows)
    print(json.dumps({"out": args.out, **stats}), flush=True)


if __name__ == "__main__":
    main()
