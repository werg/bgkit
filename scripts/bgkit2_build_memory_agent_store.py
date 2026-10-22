#!/usr/bin/env python
"""Two-stage AGENTIC memory rows (``memory_agent_v1``, 2026-09-22) from the memory QA stores.

Each source row (``memory_qa_v2_{user_assistant,pairs}``: a haystack of dated sessions in slots,
a question, a short answer; meta episode / qtype / asked_on) is replayed as an agent episode
through the harness ``Conversation`` -- the same objects the ``agent_memory`` bench arm runs
(bgkit2.agent.memory / memory_eval), so train == bench:

  turn 1   system MEMORY_AGENT_SYSTEM + memory tools; user = the memory INDEX (one slot per
           session under the generic MEMORY_INDEX_PROMPT, verbatim label ``[S3] <date>``) + the
           question -> the assistant calls ``memory_search(query, focus)`` (when BM25 ranks a
           gold session in its top k, and not on the ``--p-direct-read`` share of those episodes;
           its result is a compressible slot like web_search's, prompt ``memory search <q>: <q>``,
           ratio 4) or directly ``memory_read`` on the GOLD session(s)
  turn 2   (after a search) ``memory_read`` on the gold session(s), several calls in one turn for
           multi-session questions; each read is a slot with the prompt ``session <date>: <q>``
  last     the source row's answer, plain text

One row per assistant turn (``Conversation.to_task_row``). The direct (no-tool) memory rows stay
as they are: this family ADDS the agentic route.

GOLD sessions: the teacher's evidence quote is not kept in the stores (only rejects keep it), so
the gold is recovered from the ANSWER's grounding, restricted to the episode's own sessions (the
Chronicles episode ``meta.episode`` is decoded from the phase-2 mmap and matched to the haystack's
slots; distractors come from other episodes / MSC / SHARE). Per qtype (``identify_gold``): the
session(s) with the highest answer-word coverage (+ question-word overlap), several for
multi-session, the new + old state for knowledge updates, sessions whose date the answer names
for temporal questions. Abstention reads the best BM25 candidate, then abstains. Rows whose gold
cannot be identified are DROPPED and counted per qtype.

Split: the source rows' own split (sha1(episode) < 0.01 -> eval, the builder's formula; checked
row by row), so an eval episode is eval in every memory family.

    .venv/bin/python scripts/bgkit2_build_memory_agent_store.py \\
        --out /home/werg/bgkit-data-nvme/bgkit2/tasks/memory_agent_v1.parquet

CPU, host venv; streams the (memory-mapped) inputs batch by batch and writes the parquet in row
groups, then the .arrow copy batch by batch (~1-2 GB RSS).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bgkit2.agent.context import Conversation, ToolCall
from bgkit2.agent.memory import MemoryStore, add_memory_index
from bgkit2.agent.memory_eval import MEMORY_SLOT_MAX_TOKENS, memory_ratios
from bgkit2.agent.tools import ToolRegistry
from bgkit2.data.memory_framing import (
    ABSTAIN,
    MEMORY_AGENT_SYSTEM,
    MEMORY_TOOL_NAMES,
)
from bgkit2.data.tasks import COLUMNS

TASKS = "/home/werg/bgkit-data-nvme/bgkit2/tasks"
INPUTS = [f"{TASKS}/memory_qa_v2_user_assistant.arrow", f"{TASKS}/memory_qa_v2_pairs.arrow"]
FAMILY = "memory_agent_v1"
STAMP = re.compile(r"^\s*\(asked on [^()]*(?:\([^()]*\)[^()]*)*\)\s*")
_W = re.compile(r"[a-z0-9]+")
STOP = frozenset(
    """
    a an the and or but of to in on at for with from by as is are was were be been being do
    does did done have has had i me my mine we us our ours you your yours he him his she her it
    its they them their this that these those what which who whom whose when where why how many
    much did not no yes about into over after before then than there here also just very so can
    could would should will shall may might must said say says tell told talk talked talking
    ask asked mention mentioned conversation conversations session sessions assistant user one
    two""".split()  # noqa: SIM905 - a word list reads better as text
)
COMPUTED = frozenset({"preference", "temporal_relative"})  # answers not grounded word-for-word


def words(text: str) -> set[str]:
    return {w for w in _W.findall(text.lower()) if w not in STOP and (len(w) >= 3 or w.isdigit())}


def _dates_in(text: str) -> set[str]:
    return set(re.findall(r"\d{4}/\d{2}/\d{2}", text))


def identify_gold(qtype: str, question: str, answer: str, sessions: list[str],
                  targets: list[int]) -> list[int] | None:
    """Gold session indices (chronological) among ``targets`` (the episode's own sessions), or
    None when they cannot be identified. Abstention -> [] (no gold)."""
    if qtype == "abstention":
        return []
    if not targets:
        return None
    a, q = words(answer), words(STAMP.sub("", question))
    wa, wq = (1.0, 1.0) if qtype in COMPUTED else (1.0, 0.5)
    cov, qov, sc = {}, {}, {}
    for j in targets:
        w = words(sessions[j])
        cov[j] = len(a & w) / len(a) if a else 0.0
        qov[j] = len(q & w) / len(q) if q else 0.0
        sc[j] = wa * cov[j] + wq * qov[j]
    order = sorted(targets, key=lambda j: (-sc[j], j))
    best = order[0]
    if sc[best] <= 0:
        return None
    grounded = (cov[best] >= 0.5) if (a and qtype not in COMPUTED) else (qov[best] >= 0.34)
    if qtype == "multi_session":
        # greedy cover of the answer's words over the episode's sessions (<= 3). The teacher's
        # "multi-session" questions are often answerable from ONE session: then one read is the
        # right behaviour, not a reason to drop the row. Yes/no answers (no content words): the
        # sessions sharing most of the question's words.
        if not a:
            top = qov[order[0]] if q else 0.0
            gold = [j for j in targets if qov[j] >= max(0.34, 0.7 * top)]
            return sorted(gold[:3]) if gold else None
        left, gold = set(a), []
        ws = {j: a & words(sessions[j]) for j in targets}
        while left and len(gold) < 3:
            j = max(targets, key=lambda x: (len(ws[x] & left), qov[x], -x))
            if not ws[j] & left:
                break
            gold.append(j)
            left -= ws[j]
        return sorted(gold) if gold and 1 - len(left) / len(a) >= 0.5 else None
    if qtype == "knowledge_update":
        new = max(targets, key=lambda j: (cov[j], qov[j], j))
        if not ((a and cov[new] >= 0.5) or qov[new] >= 0.5):
            return None
        rest = [j for j in order if j != new and qov[j] >= 0.34]
        return sorted([new, *rest[:1]])
    if qtype in ("temporal", "temporal_relative"):
        named = _dates_in(answer) | _dates_in(STAMP.sub("", question))
        dated = [j for j in order if _dates_in(sessions[j].split("\n", 1)[0]) & named]
        if not dated and not grounded:
            return None
        gold = [best] if (grounded or not dated) else []
        gold += [j for j in dated if j not in gold]
        return sorted(gold[:2])
    # single_session, single_session_assistant, preference (+ unknown types)
    if not grounded:
        return None
    tie = [j for j in order[:2] if sc[j] >= sc[best] - 0.05]
    return sorted(tie)


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "mem_builder", ROOT / "scripts/bgkit2_build_memory_store.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class EpisodeSessions:
    """The Chronicles episode's own sessions, rendered as the source store rendered them."""

    def __init__(self, mm_dir: str | None) -> None:
        self.mb = load_builder()
        self.chron = None
        if mm_dir:
            from transformers import AutoTokenizer

            self.mb.MM = mm_dir
            self.chron = self.mb.Mmap("chronicles", AutoTokenizer.from_pretrained(
                "Qwen/Qwen3.5-0.8B"))

    def targets(self, episode: int, style: str, slots: list[str]) -> list[int]:
        if self.chron is None:
            return []
        ss = self.mb.sessions_of(self.chron.doc(int(episode) * 6))
        if style == "user_assistant":
            ss = [self.mb.as_user_assistant(s) for s in ss]
        bodies = {s.strip() for s in ss}
        return [j for j, t in enumerate(slots) if t.split("\n", 1)[-1].strip() in bodies]


def episode_rows(slots: list[str], question: str, answer: str, qtype: str, gold: list[int],
                 renderer, rng: random.Random, split: str, meta: dict, work_dir: str,
                 p_direct: float = 0.4, p_abstain_search: float = 0.6, k: int = 5,
                 family: str = FAMILY) -> tuple[list[dict], str]:
    """The agent episode's rows (one per assistant turn) and its route."""
    store = MemoryStore.from_texts(slots)
    reg = ToolRegistry(work_dir)
    reg.attach_memory(store, k=k)
    reg.keep_only(MEMORY_TOOL_NAMES)
    focus = STAMP.sub("", question).strip()
    conv = Conversation(MEMORY_AGENT_SYSTEM, reg.specs(), renderer, memory_ratios(),
                        task_summary=focus[:200], slot_max_tokens=MEMORY_SLOT_MAX_TOKENS)
    add_memory_index(conv, store, question, MEMORY_SLOT_MAX_TOKENS)
    hits = [store.sessions.index(s) for s, _ in store.search(focus, k=k)]
    if qtype == "abstention":
        reads = hits[:1] or [0]
        search = rng.random() < p_abstain_search
    else:
        reads = list(gold)
        search = bool(set(hits) & set(gold)) and rng.random() >= p_direct
    rows: list[dict] = []
    turn = 0

    def emit(target) -> None:
        slots_now = conv.slots
        m = dict(meta, turn=turn, route=route, gold_sessions=[store.sessions[j].id for j in gold],
                 read_sessions=[store.sessions[j].id for j in reads],
                 slot_tools=[s.tool for s in slots_now], slot_reps=[s.n_reps for s in slots_now])
        rows.append(conv.to_task_row(target=target, family=family, split=split, meta=m))

    def call_turn(calls: list[ToolCall]) -> None:
        nonlocal turn
        msg = {"role": "assistant", "content": "", "tool_calls": [c.as_message() for c in calls]}
        emit([msg])
        conv.add_assistant("", calls)
        for c in calls:
            res = reg.call(c.name, c.args)
            if not res.ok:
                raise ValueError(f"scripted call failed: {res.text}")
            conv.add_tool_result(res.text, res.compressible, res.prompt, res.tool or c.name)
        turn += 1

    route = ("search_read" if search else "read") + ("_abstain" if qtype == "abstention" else "")
    if search:
        call_turn([ToolCall("memory_search", {"query": focus, "focus": focus})])
    call_turn([ToolCall("memory_read", {"session": store.sessions[j].id, "focus": focus})
               for j in reads])
    emit(ABSTAIN if qtype == "abstention" else answer.strip())
    return rows, route


class StreamWriter:
    """Row dicts -> string columns, shuffled within a buffer, parquet row groups."""

    def __init__(self, path: Path, buf_rows: int = 256, seed: int = 0) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.pa = pa
        self.schema = pa.schema([(c, pa.string()) for c in COLUMNS])
        self.w = pq.ParquetWriter(str(path), self.schema, compression="zstd")
        self.buf: list[dict] = []
        self.buf_rows, self.n, self.rng = buf_rows, 0, random.Random(seed)

    def add(self, r: dict) -> None:
        self.buf.append(r)
        if len(self.buf) >= self.buf_rows:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        self.rng.shuffle(self.buf)
        cols: dict[str, list[str]] = {c: [] for c in COLUMNS}
        for r in self.buf:
            for c in COLUMNS:
                v = r.get(c, "")
                if c in ("tool_args", "meta") and not isinstance(v, str):
                    v = json.dumps(v, ensure_ascii=False)
                cols[c].append("" if v is None else str(v))
        t = self.pa.table({c: self.pa.array(v, type=self.pa.string()) for c, v in cols.items()})
        self.w.write_table(t, row_group_size=256)
        self.n += len(self.buf)
        self.buf = []

    def close(self) -> None:
        self.flush()
        self.w.close()


def iter_rows(path: str):
    """Source rows as dicts, batch by batch from the memory-mapped .arrow (or a parquet)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = ["family", "context", "prompt", "target", "split", "meta"]
    if path.endswith(".arrow"):
        r = pa.ipc.open_file(pa.memory_map(path))
        batches = (r.get_batch(i).select(cols) for i in range(r.num_record_batches))
    else:
        batches = pq.ParquetFile(path).iter_batches(batch_size=256, columns=cols)
    for b in batches:
        yield from b.to_pylist()


def pct(xs: list[int], q: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=INPUTS)
    ap.add_argument("--out", default=f"{TASKS}/memory_agent_v1.parquet")
    ap.add_argument("--mmap", default="/mnt/external/bgkit-data/mmap/phase2",
                    help="phase-2 mmaps (the Chronicles episodes); '' = no episode restriction")
    ap.add_argument("--tokenizer", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--p-direct-read", type=float, default=0.4)
    ap.add_argument("--p-abstain-search", type=float, default=0.6)
    ap.add_argument("--search-k", type=int, default=5)
    ap.add_argument("--eval-frac", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0, help="source rows per input (0 = all)")
    ap.add_argument("--check-rows", type=int, default=200,
                    help="render this many rows through the trainer's ChatRenderer")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from bgkit2.agent.context import TemplateRenderer
    from bgkit2.data.tasks import ChatRenderer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    renderer = TemplateRenderer(tok)
    chat = ChatRenderer.from_tokenizer(tok)
    eps = EpisodeSessions(args.mmap or None)
    out = Path(args.out)
    tmp = out.with_suffix(".parquet.tmp")
    writer = StreamWriter(tmp, seed=args.seed)
    rng = random.Random(args.seed)
    st: dict[str, Counter] = defaultdict(Counter)
    routes, splits = Counter(), Counter()
    prefix_tok: list[int] = []
    rows_per_ep: list[int] = []
    t0 = time.time()
    work = tempfile.mkdtemp(prefix="memagent_")
    n_src = 0
    for path in args.inputs:
        for i, r in enumerate(iter_rows(path)):
            if args.limit and i >= args.limit:
                break
            n_src += 1
            meta = json.loads(r["meta"]) if r["meta"] else {}
            qtype = str(meta.get("qtype", "?"))
            st[qtype]["source"] += 1
            e = int(meta["episode"])
            h = int(hashlib.sha1(str(e).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            split = "eval" if h < args.eval_frac else "train"
            if split != r["split"]:
                raise SystemExit(f"{path} row {i}: split {r['split']} != episode hash {split}")
            style = "user_assistant" if r["family"].endswith("_ua") else "pairs"
            slots = json.loads(r["context"])
            question = json.loads(r["prompt"])[0]
            targets = eps.targets(e, style, slots)
            if not targets:
                st[qtype]["no_episode_sessions"] += 1
            gold = identify_gold(qtype, question, r["target"], slots, targets)
            if gold is None:
                st[qtype]["dropped_no_gold"] += 1
                continue
            st[qtype]["identified"] += 1
            st[qtype][f"gold_{len(gold)}"] += 1
            src_meta = {"source": f"{r['family']}+agent", "episode": e, "qtype": qtype,
                        "style": style, "n_sessions": len(slots),
                        **({"asked_on": meta["asked_on"]} if meta.get("asked_on") else {})}
            rows, route = episode_rows(slots, question, r["target"], qtype, gold, renderer, rng,
                                       split, src_meta, work, args.p_direct_read,
                                       args.p_abstain_search, args.search_k)
            routes[route] += 1
            rows_per_ep.append(len(rows))
            for row in rows:
                if len(prefix_tok) < args.check_rows:
                    segs = chat.render_slots(row["instruction"], row["meta"]["tools"])
                    if len(segs) - 1 != len(json.loads(row["context"])):
                        raise SystemExit("sentinel / slot count mismatch")
                    prefix_tok.append(int(sum(int(s.shape[0]) for s in segs)))
                splits[row["split"]] += 1
                writer.add(row)
            if n_src % 1000 == 0:
                print(json.dumps({"src": n_src, "rows": writer.n + len(writer.buf),
                                  "s": round(time.time() - t0)}), flush=True)
    writer.close()
    if not writer.n:
        raise SystemExit("no rows")
    tmp.replace(out)
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrow = out.with_suffix(".arrow")
    tmpa = arrow.with_suffix(".arrow.tmp")
    pf = pq.ParquetFile(str(out))
    with pa.OSFile(str(tmpa), "wb") as sink, pa.ipc.new_file(sink, pf.schema_arrow) as w:
        for batch in pf.iter_batches(batch_size=256):
            w.write_batch(batch)  # = bgkit2.data.tasks.write_task_store_arrow, streamed
    tmpa.replace(arrow)
    per_q = {}
    for q, c in sorted(st.items()):
        per_q[q] = {**dict(c), "identified_rate": round(c["identified"] / max(1, c["source"]), 4)}
    tot = sum(c["source"] for c in st.values())
    ident = sum(c["identified"] for c in st.values())
    report = {"out": str(out), "arrow": str(arrow), "source_rows": tot, "episodes": ident,
              "dropped_no_gold": tot - ident, "identified_rate": round(ident / max(1, tot), 4),
              "rows": writer.n, "rows_by_split": dict(splits), "routes": dict(routes),
              "rows_per_episode": dict(Counter(rows_per_ep)), "per_qtype": per_q,
              "prefix_tokens_sampled": {"p50": pct(prefix_tok, 0.5), "p95": pct(prefix_tok, 0.95),
                                        "max": max(prefix_tok, default=0)},
              "wall_s": round(time.time() - t0)}
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
