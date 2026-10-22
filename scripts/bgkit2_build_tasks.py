#!/usr/bin/env python
"""Build bgkit2 task stores (text parquet; see bgkit2.data.tasks).

  squad          SQuAD v1.1 passage QA (saved by download_m1_data.sh)
  file_overview  repo file -> LLM-written description (mmap/descriptions +
                 processed_v2 content, both Qwen-tokenised -> decoded to text)

Examples:
  scripts/bgkit2_build_tasks.py squad --out $B/tasks/qa_squad.parquet
  scripts/bgkit2_build_tasks.py file_overview --out $B/tasks/file_overview.parquet \
      --shards 0:40 --max-rows 120000
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgkit2.data.reasoning import move_reasoning
from bgkit2.data.tasks import write_task_store

QWEN_ID = "Qwen/Qwen3.5-0.8B"
OVERVIEW_PROMPT = (
    "Compress this file so that its purpose, contents and role in the repository "
    "can be described.\n"
)
OVERVIEW_INSTRUCTIONS = [
    "Give an overview of what this file contains and what it is for.",
    "Describe this file: its purpose, what it defines, and what it depends on.",
    "Summarise the contents and role of this file.",
]


def _split_of(key: str, eval_frac: float) -> str:
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if h < eval_frac else "train"


# ----------------------------------------------------------------------------
def build_squad(args) -> list[dict]:
    from datasets import load_from_disk

    d = load_from_disk(args.squad_dir)
    rows = []
    for split_name, split in (("train", "train"), ("validation", "eval")):
        ds = d[split_name]
        n = len(ds) if args.max_rows is None else min(len(ds), args.max_rows)
        for i in range(n):
            ex = ds[i]
            answers = ex["answers"]["text"]
            if not answers:
                continue
            rows.append(
                {
                    "family": "qa_squad",
                    "framing": "chat_context",
                    "context": ex["context"],
                    "prompt": f"Compress the passage to answer: {ex['question']}\n",
                    "instruction": ex["question"],
                    "target": answers[0],
                    "split": split,
                    "meta": {"id": ex["id"], "title": ex["title"], "answers": answers},
                }
            )
    return rows


# ----------------------------------------------------------------------------
def _repo_key(repo_path: str) -> str:
    parts = repo_path.rstrip("/").split("/")
    return "/".join(parts[-2:])


def build_file_overview(args) -> list[dict]:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    data_dir = os.environ["DATA_DIR"]
    shards = sorted(Path(data_dir, "processed_v2", "tokens").glob("shard_*.parquet"))
    a, b = (int(x) for x in args.shards.split(":"))
    shards = shards[a:b]
    qwen = AutoTokenizer.from_pretrained(QWEN_ID)
    rows: list[dict] = []
    t0 = time.time()
    desc_cache: dict[str, dict[str, str]] = {}

    def descriptions_for(repo_key: str) -> dict[str, str]:
        if repo_key in desc_cache:
            return desc_cache[repo_key]
        f = Path(data_dir, "descriptions", repo_key + ".jsonl")
        out: dict[str, str] = {}
        if f.exists():
            with open(f) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("scope") == "file" and r.get("description"):
                        out[r["file_path"]] = r["description"]
        desc_cache[repo_key] = out
        return out

    for si, shard in enumerate(shards):
        t = pq.read_table(shard, columns=["repo_path", "file_path", "token_ids"])
        rp = t.column("repo_path").to_pylist()
        fp = t.column("file_path").to_pylist()
        tok_col = t.column("token_ids")
        for i in range(t.num_rows):
            key = _repo_key(rp[i])
            descs = descriptions_for(key)
            desc = descs.get(fp[i])
            if not desc:
                continue
            ids = tok_col[i].as_py()
            if not ids or len(ids) < args.min_tokens or len(ids) > args.max_tokens:
                continue
            text = qwen.decode(ids, skip_special_tokens=True)
            k = len(rows)
            framing = "tool_response" if k % 2 == 1 else "chat_context"
            rows.append(
                {
                    "family": "file_overview",
                    "framing": framing,
                    "context": text,
                    "prompt": OVERVIEW_PROMPT,
                    "instruction": OVERVIEW_INSTRUCTIONS[k % len(OVERVIEW_INSTRUCTIONS)]
                    if framing == "chat_context"
                    else f"What does the file {fp[i]} contain and what is it for?",
                    "tool_name": "read_file" if framing == "tool_response" else "",
                    "tool_args": {"path": fp[i]} if framing == "tool_response" else "",
                    "target": desc,
                    "split": _split_of(key, args.eval_frac),
                    "meta": {"repo": key, "path": fp[i], "qwen_tokens": len(ids)},
                }
            )
            if args.max_rows and len(rows) >= args.max_rows:
                break
        print(
            f"[{si + 1}/{len(shards)}] {shard.name}: rows so far {len(rows)}; "
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
        if args.max_rows and len(rows) >= args.max_rows:
            break
    return rows


# ----------------------------------------------------------------------------
def build_hotpotqa(args) -> list[dict]:
    """HotpotQA distractor: all ten paragraphs (titled) are the context."""
    from datasets import load_from_disk

    d = load_from_disk(f"{args.bench_dir}/hotpotqa_distractor")
    rows = []
    for split_name, split in (("train", "train"), ("validation", "eval")):
        ds = d[split_name]
        n = len(ds) if args.max_rows is None else min(len(ds), args.max_rows)
        for i in range(n):
            ex = ds[i]
            paras = [
                f"{t}\n{''.join(sents)}"
                for t, sents in zip(ex["context"]["title"], ex["context"]["sentences"], strict=True)
            ]
            ctx = "\n\n".join(paras)[: args.max_ctx_chars]
            rows.append(
                {
                    "family": "qa_hotpotqa",
                    "framing": "chat_context",
                    "context": ctx,
                    "prompt": f"Compress the passages to answer: {ex['question']}\n",
                    "instruction": ex["question"],
                    "target": ex["answer"],
                    "split": split,
                    "meta": {"id": ex["id"], "type": ex["type"], "level": ex["level"]},
                }
            )
    return rows


def build_triviaqa(args) -> list[dict]:
    """TriviaQA rc: the highest-ranked web search context is the passage."""
    from datasets import load_from_disk

    d = load_from_disk(f"{args.bench_dir}/triviaqa_rc")
    rows = []
    for split_name, split in (("train", "train"), ("validation", "eval")):
        ds = d[split_name]
        n = len(ds) if args.max_rows is None else min(len(ds), args.max_rows)
        for i in range(n):
            ex = ds[i]
            ctxs = ex["search_results"]["search_context"] or ex["entity_pages"]["wiki_context"]
            if not ctxs:
                continue
            ctx = max(ctxs, key=len)[: args.max_ctx_chars]
            ans = ex["answer"]["value"]
            if not ans or ans.lower() not in ctx.lower():
                continue  # keep only passages that actually contain the answer
            rows.append(
                {
                    "family": "qa_triviaqa",
                    "framing": "chat_context",
                    "context": ctx,
                    "prompt": f"Compress the passage to answer: {ex['question']}\n",
                    "instruction": ex["question"],
                    "target": ans,
                    "split": split,
                    "meta": {"id": ex["question_id"], "aliases": ex["answer"]["aliases"][:8]},
                }
            )
    return rows


# ----------------------------------------------------------------------------
def build_multineedle(args) -> list[dict]:
    """Prompt-conditioned needle finding: K passages from the QA stores are
    concatenated into ONE document and each passage's question becomes its
    own row with that question as the encoder prompt. The union of the
    document's answer evidence exceeds the rep budget at high compression,
    so a query-INDEPENDENT selection cannot serve every row — the test the
    single-passage QA stores cannot run (their answer spans fit at 4x).

    Sources: the qa_squad / qa_hotpotqa task stores (context, question,
    answer). Passages are grouped within a split, distinct source ids,
    shuffled order inside the document."""
    import random

    import pyarrow.parquet as pq

    rng = random.Random(23)
    rows: list[dict] = []
    for split in ("train", "eval"):
        pool: list[dict] = []
        for path in args.needle_sources:
            t = pq.read_table(path, columns=["family", "context", "instruction", "target", "split"])
            for r in t.to_pylist():
                if r["split"] != split or len(r["context"]) > args.needle_passage_chars:
                    continue
                pool.append(r)
        rng.shuffle(pool)
        n_docs = 0
        for a in range(0, len(pool) - args.needle_k + 1, args.needle_k):
            group = pool[a : a + args.needle_k]
            order = list(range(len(group)))
            rng.shuffle(order)
            doc = "\n\n".join(group[i]["context"] for i in order)
            if len(doc) > args.max_ctx_chars:
                continue
            n_docs += 1
            doc_id = f"needle:{split}:{n_docs:06d}"
            for pos, i in enumerate(order):
                g = group[i]
                rows.append(
                    {
                        "family": "qa_multineedle",
                        "framing": "chat_context",
                        "context": doc,
                        "prompt": f"Compress the passages to answer: {g['instruction']}\n",
                        "instruction": g["instruction"],
                        "target": g["target"],
                        "split": split,
                        "meta": {"doc": doc_id, "position": pos, "source": g["family"]},
                    }
                )
            if args.max_rows and len(rows) >= args.max_rows:
                break
        print(f"{split}: {n_docs} documents, {len(rows)} rows so far", flush=True)
    return rows


def build_babilong(args) -> list[dict]:
    """BABILong (the 1k-samples release): bAbI facts hidden in PG-19 noise at a
    chosen context length; the question names the entity. An EVAL-ONLY
    external long-context needle benchmark (every row split=eval), one family
    per (task, length): babilong_<task>_<len>."""
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for length in args.babilong_lengths:
        for task in args.babilong_tasks:
            path = Path(args.babilong_dir) / length / f"{task}-00000-of-00001.parquet"
            if not path.exists():
                print(f"skip {path}", flush=True)
                continue
            t = pq.read_table(path).to_pylist()
            n = len(t) if args.max_rows is None else min(len(t), args.max_rows)
            for i in range(n):
                ex = t[i]
                q = str(ex["question"]).strip()
                rows.append(
                    {
                        "family": f"babilong_{task}_{length}",
                        "framing": "chat_context",
                        "context": str(ex["input"])[: args.max_ctx_chars],
                        "prompt": f"Compress the text to answer: {q}\n",
                        "instruction": q,
                        "target": str(ex["target"]).strip(),
                        "split": "eval",
                        "meta": {"task": task, "length": length, "index": i},
                    }
                )
            print(f"{task} {length}: {n} rows", flush=True)
    return rows


def build_babi_filler(args) -> list[dict]:
    """bAbI facts (tasks 1-3, en-10k TRAIN split) hidden in unrelated filler
    prose at 1k-8k tokens — the structure of BABILong, built from the training
    split and our own filler (SQuAD / HotpotQA paragraphs, not PG-19) so the
    eval-only BABILong store stays an external test. Each question keeps the
    story's facts up to that question, in order, scattered through the filler.
    qa1 rows additionally get DERIVED sibling questions ("Where is <other
    person>?", answered from the same facts by tracking last locations), so
    the paired prompt contrast gets same-document pairs on long documents.
    meta carries the supporting facts (gold_window) for oracle-window
    distillation."""
    import random
    import re
    import zipfile

    import pyarrow.parquet as pq

    rng = random.Random(29)
    # ---- filler ----
    # v1 (2026-09-11): a global shuffled pool of SQuAD / HotpotQA sentences —
    # incoherent Wikipedia-flavoured filler. BaBiLong's filler is CONTIGUOUS
    # book text (PG-19), and P22's BaBiLong gen EM slid 0.667 -> 0.552 over
    # 36k-48k while the trained store stayed flat: the run specialised to the
    # v1 texture. v2 (``--filler-jsonl``): filler documents from several text
    # sources (books, web, wiki, plus the QA paragraphs joined per document),
    # and each row takes ONE contiguous window of ONE document, so the facts
    # sit inside coherent prose the way they do in the benchmark.
    filler_docs: list[tuple[str, str]] = []  # (source, text)
    for path in args.filler_sources:
        t = pq.read_table(path, columns=["context", "split"])
        seen = set()
        for ctx, sp in zip(
            t.column("context").to_pylist(), t.column("split").to_pylist(), strict=True
        ):
            if sp != "train" or ctx in seen or len(ctx) < 200:
                continue
            seen.add(ctx)
            filler_docs.append(("qa", ctx))
    for path in args.filler_jsonl:
        src = Path(path).stem
        n0 = len(filler_docs)
        with open(path) as fh:
            for ln in fh:
                d = json.loads(ln)
                txt = d.get("text", "")
                if len(txt) >= 2000:
                    # source = the dump's stem (pg19 / fineweb_edu / wikitext)
                    filler_docs.append((src, txt))
        print(f"filler {src}: {len(filler_docs) - n0} documents", flush=True)
    rng.shuffle(filler_docs)
    by_src: dict[str, list[str]] = {}
    for src, txt in filler_docs:
        by_src.setdefault(src, []).append(txt)
    print(
        "filler documents: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(by_src.items())),
        flush=True,
    )
    sources = sorted(by_src)
    src_cursor: dict[str, int] = dict.fromkeys(sources, 0)
    # Chars per LFM token measured PER SOURCE on a sample (wikitext / pg19 run
    # denser than web text): a fixed 3.8 undershot the buckets by ~12% and a
    # fixed 4.3 pushed 8% of the 8000 bucket past the 8192-token encoder cap.
    cpt: dict[str, float] = {}
    if args.babi_chars_per_token > 0:
        cpt = dict.fromkeys(sources, float(args.babi_chars_per_token))
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        for src in sources:
            sample = by_src[src][:64]
            n_tok = sum(len(x) for x in tok(sample, add_special_tokens=False)["input_ids"])
            cpt[src] = sum(len(x) for x in sample) / max(1, n_tok)
        print("filler chars/token: " + ", ".join(f"{k}={v:.2f}" for k, v in cpt.items()))

    def take_filler(n_tokens: int) -> tuple[str, list[str]]:
        """(source, ~``n_tokens`` of filler as sentences): one contiguous window
        of one document, sources drawn uniformly; short documents are
        concatenated (same source) until the window is filled."""
        src = sources[rng.randrange(len(sources))]
        n_chars = int(n_tokens * cpt[src])
        docs = by_src[src]
        parts: list[str] = []
        have = 0
        while have < n_chars:
            i = src_cursor[src] % len(docs)
            src_cursor[src] += 1
            txt = docs[i]
            if len(txt) > n_chars - have + 400:
                off = rng.randrange(0, len(txt) - (n_chars - have) - 200)
                txt = txt[off:]
                # start at a sentence boundary
                m = re.search(r"(?<=[.!?])\s+", txt[:400])
                txt = txt[m.end() :] if m else txt
            parts.append(txt)
            have += len(txt) + 1
        out: list[str] = []
        used = 0
        for sent in re.split(r"(?<=[.!?])\s+", " ".join(parts)):
            sent = sent.strip()
            if not (3 <= len(sent) <= 600):
                continue
            out.append(sent)
            used += len(sent) + 1
            if used >= n_chars:
                break
        return src, out

    # ---- bAbI stories ----
    verbs = "moved|went|journeyed|travelled|went back|moved back|journeyed back|travelled back"
    move_re = re.compile(r"^(\w+) (?:" + verbs + r") to the (\w+)\.$")
    zf = zipfile.ZipFile(args.babi_zip)
    names = {
        n.split("/")[-1]: n for n in zf.namelist() if "/en-10k/" in n and n.endswith("_train.txt")
    }
    rows: list[dict] = []

    for task in args.babi_tasks:
        fname = next((n for k, n in names.items() if k.startswith(task + "_")), None)
        if fname is None:
            print(f"skip {task}: not in zip", flush=True)
            continue
        lines = zf.read(fname).decode("utf-8").splitlines()
        stories: list[list[tuple[int, str]]] = []
        cur: list[tuple[int, str]] = []
        for ln in lines:
            i, rest = ln.split(" ", 1)
            i = int(i)
            if i == 1 and cur:
                stories.append(cur)
                cur = []
            cur.append((i, rest))
        if cur:
            stories.append(cur)
        n_story = 0
        for si, story in enumerate(stories):
            split = "eval" if si % 20 == 0 else "train"
            facts: list[tuple[int, str]] = []
            loc: dict[str, str] = {}
            q_idx = 0
            for i, rest in story:
                if "\t" in rest:
                    q, ans, sup = rest.split("\t")
                    q, ans = q.strip(), ans.strip()
                    sup_ids = [int(x) for x in sup.split()]
                    if not facts:
                        continue
                    n_story += 1
                    doc_id = f"babi:{task}:{si}:{q_idx}"
                    length = rng.choice(args.babi_lengths)
                    n_fact_tokens = sum(len(f) for _, f in facts) // 4
                    filler_src, fill = take_filler(
                        max(0, int(length * args.babi_token_margin) - n_fact_tokens)
                    )
                    # scatter the facts, in order, among the filler sentences
                    slots = sorted(rng.sample(range(len(fill) + 1), min(len(facts), len(fill) + 1)))
                    pieces, fi = [], 0
                    for pos, sent in enumerate(fill):
                        while fi < len(facts) and slots[fi] == pos:
                            pieces.append(facts[fi][1])
                            fi += 1
                        pieces.append(sent)
                    while fi < len(facts):
                        pieces.append(facts[fi][1])
                        fi += 1
                    doc = " ".join(pieces)
                    # v2 (2026-09-11) cut the assembled document at the generic
                    # --max-ctx-chars (24,000): the 8k bucket was built at
                    # 30,400 chars, so 67% of its documents lost supporting
                    # facts and 44% of its rows had underivable targets. The
                    # length bucket defines the size; nothing is cut here.
                    gold = " ".join(f for j, f in facts if j in sup_ids)
                    pos = 0
                    for _j, f in facts:
                        k = doc.find(f, pos)
                        assert k >= 0, (doc_id, f)
                        pos = k + len(f)
                    questions = [(q, ans)]
                    if task == "qa1":
                        # derived siblings: other people whose location the facts fix
                        person = q.replace("Where is ", "").rstrip("?").strip()
                        for other, where in loc.items():
                            if other != person:
                                questions.append((f"Where is {other}?", where))
                    for qi, (qq, aa) in enumerate(questions[: 1 + args.babi_max_siblings]):
                        rows.append(
                            {
                                "family": "babi_filler",
                                "framing": "chat_context",
                                "context": doc,
                                "prompt": f"Compress the text to answer: {qq}\n",
                                "instruction": qq,
                                "target": aa,
                                "split": split,
                                "meta": {
                                    "doc": doc_id,
                                    "task": task,
                                    "length": length,
                                    "derived": qi > 0,
                                    "gold_window": gold if qi == 0 else "",
                                    "filler_source": filler_src,
                                    "n_facts": len(facts),
                                },
                            }
                        )
                    q_idx += 1
                else:
                    facts.append((i, rest.strip()))
                    m = move_re.match(rest.strip())
                    if m:
                        loc[m.group(1)] = m.group(2)
            if args.max_rows and len(rows) >= args.max_rows:
                break
        print(f"{task}: {n_story} questions, {len(rows)} rows so far", flush=True)
    return rows


def _cap_msg(mm: dict, max_chars: int) -> dict:
    """Keep the head and tail of an over-long message (anchors, tool dumps)."""
    for k in ("content", "reasoning_content"):
        if k in mm and len(mm[k]) > max_chars:
            h = max_chars // 2
            mm[k] = mm[k][:h] + "\n[...]\n" + mm[k][-h:]
    return mm


def _trajectory_files(dirs: list[str], shards: str) -> list[Path]:
    """Parquet shards under each directory (recursively), then the
    ``a:b`` slice of that sorted list."""
    files: list[Path] = []
    for d in dirs:
        files += sorted(Path(d).rglob("*.parquet"))
    a, b = (int(x) for x in shards.split(":"))
    return files[a:b]


def build_trajectory_chunks(args) -> list[dict]:
    """Agentic trajectories -> chunked continuation samples (family
    ``compaction_chunks``, framing ``trajectory``).

    A trajectory is walked in chunks of 1-4 assistant TURNS (an assistant
    message plus the tool results it caused). For each chunk the ENCODER gets
    the whole history between the anchor and a short live window (0-2 turns
    kept as text), and the DECODER predicts the chunk itself: every assistant
    message in it, tool calls included, rendered through the chat template;
    the tool results inside the chunk are masked out of the loss (they are
    the environment's). One trajectory yields up to ``--samples-per-traj``
    samples at distinct positions, so the decoder sees compaction at every
    depth of a run rather than one next turn (compaction_v2's median target
    was 83 characters)."""
    import random

    import pyarrow.parquet as pq

    from bgkit.data.compaction_sampler import detect_anchor_len, turn_boundaries

    files = _trajectory_files(args.swe_dir, args.shards)
    rng = random.Random(19)
    rows: list[dict] = []
    t0 = time.time()
    n_traj = 0
    for fi, f in enumerate(files):
        names = pq.read_schema(f).names
        # SWE-Zero / nebius carry `trajectory`; Open-SWE-Traces carries
        # `messages` — same role/content/tool_calls message shape.
        col = "trajectory" if "trajectory" in names else "messages"
        cols = [c for c in ("trajectory_id", "instance_id", col, "repo") if c in names]
        source = f.parent.name if f.parent.name != "data" else f.parents[1].name
        # Stream row groups: a 2 GB shard materialised with to_pylist() took
        # 31 GB of host RAM beside the live trainer (2026-09-08).
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            _rows = pf.read_row_group(rg, columns=cols).to_pylist()
            for r in _rows:
                traj = r[col]
                r.setdefault("repo", r.get("instance_id", ""))
                r.setdefault("trajectory_id", r.get("instance_id", ""))
                if isinstance(traj, str):
                    traj = json.loads(traj)
                if not isinstance(traj, list) or len(traj) < 6:
                    continue
                anchor_len = detect_anchor_len(traj)
                bounds = turn_boundaries(traj, anchor_len)
                if len(bounds) < 4:
                    continue
                n_traj += 1
                bounds = [*bounds, len(traj)]
                # Chunk starts: every turn boundary from the 2nd turn on (the
                # first chunk has no history to compress); a random subset.
                starts = list(range(1, len(bounds) - 1))
                rng.shuffle(starts)
                for si in sorted(starts[: args.samples_per_traj]):
                    # Grow the chunk turn by turn until it fills the target
                    # budget (so a chunk is 1 long turn or several short ones).
                    k, chars = 1, 0
                    while si + k < len(bounds) - 1:
                        chars += sum(
                            len(_msg_text(m)) for m in traj[bounds[si + k - 1] : bounds[si + k]]
                        )
                        if chars >= args.chunk_target_chars:
                            break
                        k += 1
                    live = rng.choice((0, 1, 2))
                    live = min(live, si)
                    chunk_msgs = traj[bounds[si] : bounds[min(si + k, len(bounds) - 1)]]
                    if not chunk_msgs or chunk_msgs[0].get("role") != "assistant":
                        continue
                    live_start = bounds[si - live]
                    hist = traj[anchor_len:live_start]
                    ctx = "\n\n".join(_msg_text(m) for m in hist)[: args.max_ctx_chars]
                    if len(ctx) < 200:
                        continue
                    prefix = [
                        _cap_msg(
                            _normalise_for_template(m, args.keep_reasoning), args.max_msg_chars
                        )
                        for m in traj[:anchor_len]
                    ]
                    prefix.append(
                        {
                            "role": "user",
                            "content": "[Earlier conversation compacted]\n<|reserved_6|>",
                        }
                    )
                    prefix += [
                        _cap_msg(
                            _normalise_for_template(m, args.keep_reasoning), args.max_msg_chars
                        )
                        for m in traj[live_start : bounds[si]]
                    ]
                    if sum("<|reserved_6|>" in m["content"] for m in prefix) != 1:
                        continue
                    # Tool results inside the chunk are loss-masked; cap them hard
                    # so they do not eat the decoder sequence.
                    chunk = [
                        _cap_msg(
                            _normalise_for_template(m, args.keep_reasoning),
                            args.max_tool_chars if m.get("role") == "tool" else args.max_msg_chars,
                        )
                        for m in chunk_msgs
                    ]
                    # Leading/trailing whitespace in a message merges with the
                    # template's own newlines into different BPE tokens, so the
                    # rendered chunk would not extend the rendered prefix
                    # (feedback_bpe_not_concat_distributive).
                    for m in chunk:
                        m["content"] = m["content"].strip()
                    if not any(
                        (m["content"].strip() or m.get("tool_calls"))
                        for m in chunk
                        if m["role"] == "assistant"
                    ):
                        continue
                    rows.append(
                        {
                            "family": "compaction_chunks",
                            "framing": "trajectory",
                            "context": ctx,
                            "prompt": (
                                "Compact this earlier part of the session so the task can be "
                                "continued.\n"
                            ),
                            "instruction": json.dumps(prefix),
                            "target": json.dumps(chunk),
                            "split": _split_of(str(r["repo"]), args.eval_frac),
                            "meta": {
                                "trajectory_id": str(r["trajectory_id"]),
                                "source": source,
                                "span": [anchor_len, live_start],
                                "chunk": [bounds[si], bounds[min(si + k, len(bounds) - 1)]],
                                "turns": k,
                                "live_turns": live,
                            },
                        }
                    )
                if args.max_rows and len(rows) >= args.max_rows:
                    break
            if args.max_rows and len(rows) >= args.max_rows:
                break
        print(
            f"[{fi + 1}/{len(files)}] {f.name}: trajectories {n_traj} rows {len(rows)}; "
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
        if args.max_rows and len(rows) >= args.max_rows:
            break
    return rows


def _msg_text(m: dict) -> str:
    role = m.get("role", "")
    parts = [str(m.get("content") or "")]
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", {})
        parts.append(f"<tool_call>{fn.get('name')}({fn.get('arguments')})</tool_call>")
    return f"[{role}]\n" + "\n".join(p for p in parts if p)


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _normalise_for_template(m: dict, keep_reasoning: bool = False) -> dict:
    """OpenAI-style message -> what LFM2.5's template accepts (dict args).
    By default ``<think>...</think>`` blocks are removed and reasoning fields
    ignored: the chat template keeps them only on the LAST assistant message,
    so a history containing one renders differently once a later message is
    appended (not prefix-consistent). ``keep_reasoning`` (``--keep-reasoning``,
    for the 8B decoder trained with per-turn reasoning) instead moves the
    ``reasoning_content`` / ``reasoning`` / ``thinking`` field and every
    ``<think>`` / ``<thinking>`` block of an assistant message into
    ``reasoning_content`` (``bgkit2.data.reasoning.split_reasoning``); the
    content keeps only the visible text."""
    content = str(m.get("content") or "")
    reasoning = ""
    if m.get("role") == "assistant":
        if keep_reasoning:
            mv = move_reasoning(m)
            content, reasoning = mv["content"], mv.get("reasoning_content", "")
        else:
            content = _THINK_RE.sub("", content)
    out = {"role": m.get("role", "user"), "content": content}
    if reasoning:
        out["reasoning_content"] = reasoning
    if m.get("tool_calls"):
        calls = []
        for tc in m["tool_calls"]:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"input": args}
            calls.append(
                {
                    "type": "function",
                    "function": {"name": fn.get("name", "tool"), "arguments": args},
                }
            )
        out["tool_calls"] = calls
    return out


def build_compaction(args) -> list[dict]:
    """SWE-Zero trajectories -> compaction samples (old bgkit sampler, new store).

    The compacted span's text (role-tagged) is the ENCODER context; the decoder
    sees the anchor, a history message holding the sentinel, the live window,
    and predicts the next assistant turn (or a recall probe answer)."""
    import random

    import pyarrow.parquet as pq

    from bgkit.data.compaction_sampler import sample_trajectory

    files = _trajectory_files(args.swe_dir, args.shards)
    rng = random.Random(17)
    rows: list[dict] = []
    t0 = time.time()
    for fi, f in enumerate(files):
        t = pq.read_table(f, columns=["trajectory_id", "trajectory", "repo"])
        for r in t.to_pylist():
            traj = r["trajectory"]
            if isinstance(traj, str):
                traj = json.loads(traj)
            if not isinstance(traj, list) or len(traj) < 6:
                continue
            for s in sample_trajectory(traj, trajectory_id=str(r["trajectory_id"]), rng=rng):
                a_, b_ = s.compacted_span
                ctx = "\n\n".join(_msg_text(m) for m in traj[a_:b_])[: args.max_ctx_chars]
                if len(ctx) < 200:
                    continue
                msgs = []
                for m in s.prefix_messages:
                    mm = _normalise_for_template(m, args.keep_reasoning)
                    if mm["content"].startswith("[Earlier conversation compacted]"):
                        mm["content"] = "[Earlier conversation compacted]\n<|reserved_6|>"
                    elif len(mm["content"]) > args.max_msg_chars:
                        # Long anchors (6k-char system prompts) and tool dumps
                        # in the live window blow up the DECODER sequence; keep
                        # the head and the tail of each message.
                        h = args.max_msg_chars // 2
                        mm["content"] = mm["content"][:h] + "\n[...]\n" + mm["content"][-h:]
                    msgs.append(mm)
                if sum("<|reserved_6|>" in m["content"] for m in msgs) != 1:
                    continue
                target = str(s.target_message.get("content") or "")
                if not target.strip():
                    continue
                rows.append(
                    {
                        "family": "compaction_probe" if s.qtype == "recall_probe" else "compaction",
                        "framing": "messages",
                        "context": ctx,
                        "prompt": (
                            "Compact this earlier part of the session so the task can be "
                            "continued.\n"
                        ),
                        "instruction": json.dumps(msgs),
                        "target": target[:4000],
                        "split": _split_of(str(r["repo"]), args.eval_frac),
                        "meta": {
                            "trajectory_id": s.trajectory_id,
                            "span": list(s.compacted_span),
                            "mode": s.mode,
                        },
                    }
                )
            if args.max_rows and len(rows) >= args.max_rows:
                break
        print(
            f"[{fi + 1}/{len(files)}] {f.name}: rows so far {len(rows)}; {time.time() - t0:.0f}s",
            flush=True,
        )
        if args.max_rows and len(rows) >= args.max_rows:
            break
    return rows


# ----------------------------------------------------------------------------
TOOL_SLOT_SOURCES = [
    "nebius=/mnt/external/bgkit-data/trajectories_ext/swe_rebench_openhands",
    "open_swe=/mnt/external/bgkit-data/trajectories_ext/open_swe_traces",
    "swe_zero=/home/werg/bgkit-data-nvme/capability_packaging/trajectories_ext/swe_zero/data",
    "openresearcher=/mnt/external/bgkit-data/trajectories_ext/openresearcher_conv",
]
_BARE_ARG_KEYS = ("command", "cmd", "query", "path", "url", "pattern", "old_str", "new_str")


def _compact_call(name: str, args, max_chars: int = 200) -> str:
    """ "<tool>: <arguments>" for the compression prompt of a tool result —
    the well-known keys (command, path, query, ...) bare and in order, the
    rest as ``k=v``; whitespace collapsed; cut at ``max_chars``."""
    if isinstance(args, str):
        with contextlib.suppress(json.JSONDecodeError):
            args = json.loads(args)
    if isinstance(args, dict):
        parts = [str(args[k]) for k in _BARE_ARG_KEYS if k in args and args[k] not in ("", None)]
        parts += [
            f"{k}={json.dumps(v) if not isinstance(v, str) else v}"
            for k, v in args.items()
            if k not in _BARE_ARG_KEYS and v not in ("", None)
        ]
        text = " ".join(parts)
    else:
        text = str(args or "")
    text = " ".join(text.split())
    out = f"{name}: {text}" if text else f"{name}"
    return out[:max_chars]


def _cap_call_args(mm: dict, max_chars: int) -> dict:
    """Head+tail-cap every string argument of a (normalised) history tool
    call: a ``str_replace`` / ``create`` call carries whole files in its
    arguments (up to 39k chars per turn on nebius), which ``_cap_msg`` does
    not touch."""
    for tc in mm.get("tool_calls") or []:
        args = tc["function"].get("arguments")
        if isinstance(args, dict):
            for k, v in args.items():
                if isinstance(v, str) and len(v) > max_chars:
                    h = max_chars // 2
                    args[k] = v[:h] + "\n[...]\n" + v[-h:]
    return mm


def _tool_calls_of(m: dict) -> list[tuple[str, object]]:
    out = []
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function") or {}
        out.append((fn.get("name") or "tool", fn.get("arguments")))
    return out


def _tool_result_calls(traj: list[dict]) -> dict[int, tuple[str, object]]:
    """message index of every tool result -> (tool name, arguments) of the
    call that produced it: the calls of the preceding assistant turn, in
    order (every source here emits one tool message per call, in order)."""
    out: dict[int, tuple[str, object]] = {}
    pending: list[tuple[str, object]] = []
    for i, m in enumerate(traj):
        role = m.get("role")
        if role == "assistant":
            pending = _tool_calls_of(m)
        elif role == "tool":
            out[i] = pending.pop(0) if pending else (str(m.get("name") or "tool"), {})
    return out


_OBSERVATION_RE = re.compile(r"^(?:OBSERVATION|Observation):[ \t]*\n?")


def _adapt_ext_tool_result(content: object) -> str:
    """A tool result of the filtered JSONL trajectory sources in the text our harness returns.

    * SWE-agent prefixes every observation with ``OBSERVATION:\\n`` -> removed (the
      ``str_replace_editor`` displays then start with "Here's the result of running `cat -n`",
      which the remapper's zoom path recognises).
    * mini-SWE-agent returns JSON ``{"returncode", "output"}`` (long outputs as
      ``output_head`` / ``output_tail`` / ``elided_chars``, failures with ``exception_info``)
      -> the output text, ``STDERR``-free, with ``(exit code N)`` on a non-zero return code,
      the shape ``ToolRegistry.bash`` emits.
    Anything else is returned unchanged."""
    text = str(content or "")
    m = _OBSERVATION_RE.match(text)
    if m:
        return text[m.end() :]
    if text.lstrip().startswith("{") and '"returncode"' in text:
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            return text
        if not isinstance(d, dict) or "returncode" not in d:
            return text
        if "output" in d:
            out = str(d.get("output") or "")
        elif "output_head" in d or "output_tail" in d:
            out = (
                str(d.get("output_head") or "")
                + f"\n[... {d.get('elided_chars', '?')} characters elided ...]\n"
                + str(d.get("output_tail") or "")
            )
        else:
            return text
        if d.get("exception_info"):
            out += ("\n" if out else "") + str(d["exception_info"])
        try:
            rc = int(d.get("returncode") or 0)
        except (TypeError, ValueError):
            rc = 0
        if rc != 0:
            out += ("\n" if out else "") + f"(exit code {rc})"
        return out or "(no output)"
    return text


def _iter_jsonl_trajectories(name: str, root: str, rng, max_files: int | None):
    """(trajectory_id, source, trajectory, extra) over ``*.jsonl.gz`` / ``*.jsonl`` files (a
    file, or a directory searched recursively) -- the filtered Open-SWE-Traces and
    Nemotron-SFT-SWE dumps: one record per line with OpenAI-style ``messages`` (``tool_calls``
    with JSON-string arguments, ``reasoning_content``). Streamed line by line. Tool results go
    through ``_adapt_ext_tool_result``. ``extra``: ``split_key`` (``instance_id`` when the record
    has one, so every harness's trajectory of the same task lands in the same split; else
    ``<source>:<trajectory id>``) and ``meta`` (task / language fields for the store)."""
    import gzip

    p = Path(root)
    files = (
        [p] if p.is_file()
        else sorted([*p.rglob("*.jsonl.gz"), *p.rglob("*.jsonl")])
    )
    rng.shuffle(files)
    if max_files:
        files = files[:max_files]
    for f in files:
        opener = gzip.open if f.name.endswith(".gz") else open
        with opener(f, "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                traj = r.get("messages") or r.get("trajectory")
                if isinstance(traj, str):
                    traj = json.loads(traj)
                if not isinstance(traj, list):
                    continue
                traj = [
                    {**m, "content": _adapt_ext_tool_result(m.get("content"))}
                    if isinstance(m, dict) and m.get("role") == "tool" else m
                    for m in traj
                ]
                tid = str(r.get("trajectory_id") or r.get("uuid") or r.get("instance_id") or "")
                if not tid:
                    tid = hashlib.sha1(json.dumps(traj[:2]).encode()).hexdigest()[:16]
                inst = r.get("instance_id")
                meta = {
                    k: r[k]
                    for k in ("instance_id", "repo", "language", "dominant_ext",
                              "dominant_ext_family", "source", "source_file")
                    if r.get(k) is not None
                }
                if "source" in meta:
                    meta["dataset"] = meta.pop("source")
                yield tid, name, traj, {
                    "split_key": str(inst) if inst else f"{name}:{tid}",
                    "meta": meta,
                }
        print(f"  {name}: finished {f.name}", flush=True)


def _iter_source_trajectories(name: str, root: str, rng, max_files: int | None):
    """(trajectory_id, source, trajectory, extra) over the parquet shards under
    ``root`` (shard order shuffled with ``rng``), streamed by row group; ``extra``
    is empty. A ``*.jsonl[.gz]`` root, or a directory holding only such files,
    goes through ``_iter_jsonl_trajectories``."""
    import pyarrow.parquet as pq

    rp = Path(root)
    if (rp.is_file() and rp.name.endswith((".jsonl", ".jsonl.gz"))) or (
        rp.is_dir()
        and not any(rp.rglob("*.parquet"))
        and any([*rp.rglob("*.jsonl.gz"), *rp.rglob("*.jsonl")])
    ):
        yield from _iter_jsonl_trajectories(name, root, rng, max_files)
        return
    files = sorted(Path(root).rglob("*.parquet"))
    rng.shuffle(files)
    if max_files:
        files = files[:max_files]
    for f in files:
        names = pq.read_schema(f).names
        col = "trajectory" if "trajectory" in names else "messages"
        cols = [c for c in ("trajectory_id", "instance_id", col) if c in names]
        pf = pq.ParquetFile(f)
        rgs = list(range(pf.num_row_groups))
        rng.shuffle(rgs)
        for rg in rgs:
            for r in pf.read_row_group(rg, columns=cols).to_pylist():
                traj = r[col]
                if isinstance(traj, str):
                    traj = json.loads(traj)
                if not isinstance(traj, list):
                    continue
                tid = str(r.get("trajectory_id") or r.get("instance_id") or "")
                if not tid:
                    tid = hashlib.sha1(json.dumps(traj[:2]).encode()).hexdigest()[:16]
                yield tid, name, traj, {}
        print(f"  {name}: finished {f.name}", flush=True)


# Read-type calls of the source harnesses (--slots-read-always): str_replace_editor `view`
# (file or directory), OpenCode read / grep / glob / list, and our own readers.
_READ_CALLS = {"read", "grep", "glob", "list", "zoom", "search", "ls"}
_READ_ERROR_RE = re.compile(
    r"^\s*(error\b|the path \S+ does not exist|file not found|no such file|"
    r"invalid `view_range`)",
    re.IGNORECASE,
)


def _is_read_result(name: str, args, content: str) -> bool:
    """Whether a tool result is a SUCCESSFUL read-type result: our harness returns those
    compressed (read / zoom / search / ls are compressible), errors inline."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    if name in ("str_replace_editor", "str_replace_based_edit_tool"):
        is_read = str(args.get("command", "view")) == "view"
    else:
        is_read = name in _READ_CALLS
    return is_read and bool(content.strip()) and not _READ_ERROR_RE.match(content)


def _text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()


def _slot_targets_from_store(
    path: str, max_ctx_chars: int
) -> tuple[dict[str, dict[str, dict[int, str]]], dict[tuple[str, str, str], str]]:
    """(targets, prompts) of an existing tool_slots store for ``--slots-from-store``:
    ``targets[source][trajectory_id][msg_index] = split`` and
    ``prompts[(source, trajectory_id, sha1(slot text))] = the store's prompt`` (non-empty only).
    The store's slot texts are compared after the builder's own ``--max-ctx-chars`` cap, which
    is how they were written."""
    import pyarrow.parquet as pq

    targets: dict[str, dict[str, dict[int, str]]] = {}
    prompts: dict[tuple[str, str, str], str] = {}
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=512, columns=["context", "prompt", "split", "meta"]):
        for r in b.to_pylist():
            meta = json.loads(r["meta"]) if isinstance(r["meta"], str) else (r["meta"] or {})
            src, tid = str(meta.get("source", "")), str(meta.get("trajectory_id", ""))
            if not src or not tid or meta.get("msg_index") is None:
                continue
            targets.setdefault(src, {}).setdefault(tid, {})[int(meta["msg_index"])] = r["split"]
            ctx = json.loads(r["context"]) if r["context"] else []
            pr = json.loads(r["prompt"]) if r["prompt"] else []
            for text, p in zip(ctx, pr, strict=False):
                if p:
                    text = _cap_msg({"content": text}, max_ctx_chars)["content"]
                    prompts[(src, tid, _text_key(text))] = p
    return targets, prompts


def build_tool_slots(args) -> list[dict]:
    """Agent trajectories -> multi-slot next-turn samples (family / framing
    ``tool_slots``).

    For an assistant turn t the DECODER sees the anchor (system + task user
    message), a note that earlier turns were omitted, and the last
    ``--slots-window`` turns of history — in which the LAST up to
    ``--slots-max`` tool results of >= ``--slots-min-chars`` chars are each
    replaced by the sentinel (one encoder slot per result; the slot's
    compression prompt is the call that produced it: "<tool>: <args>").
    Smaller tool results stay as text (capped like the chunk builder's
    in-chunk tool results); other history messages are capped at
    ``--max-msg-chars`` (content AND string tool-call arguments — an edit
    call carries whole files). The target is turn t itself (content + tool calls),
    a one-message chunk with the loss on the assistant tokens. Up to
    ``--slots-per-traj`` turns per trajectory, ``--slots-per-source`` rows per
    source, and every row is rendered through the real decoder tokenizer
    before it is kept (``--slots-verify``), so no unrenderable row reaches a
    training worker (``TaskDataset._slots_item`` has no skip path).

    ``--slots-max 0``: no per-row cap -- every large result in the window is a slot.

    ``--compact-history`` (2026-09-26, user: the agentic stores must match the harness): the row
    carries the WHOLE history before turn t -- no ``--slots-window``, no "[Earlier turns of this
    session omitted]" note, history messages and call arguments uncapped (the anchor is still
    capped at ``--max-msg-chars``: the source harness's system prompt describes ITS tools) -- and
    EVERY non-empty tool result is a slot (text capped at ``--max-ctx-chars``; ``--slots-max`` /
    ``--slots-min-chars`` do not apply). Which of them the
    harness would encode is only known in OUR tool schema, so the remap decides it and folds the
    history into the compaction spine at the harness's context budget
    (``scripts/bgkit2_remap_tool_slots.py --compact-history``); these rows are its input, not a
    training store. ``--slots-from-store S``: build exactly the target turns of store S (its
    ``meta.source`` / ``trajectory_id`` / ``msg_index``, keeping each row's split) and take S's
    per-slot prompts for the slots whose text S carried (matched on source, trajectory and slot
    text; the other slots get an empty prompt for the remap to rebuild) -- tool_slots_synth
    rebuilt with its hindsight prompts."""
    import random

    from bgkit.data.compaction_sampler import detect_anchor_len, turn_boundaries
    from bgkit2.data.templates import SPLICE_SENTINEL

    renderer = None
    if args.slots_verify:
        from transformers import AutoTokenizer

        from bgkit2.data.tasks import ChatRenderer

        renderer = ChatRenderer.from_tokenizer(AutoTokenizer.from_pretrained(args.tokenizer))

    rows: list[dict] = []
    skipped: dict[str, int] = {}
    t0 = time.time()

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    targets, prompt_of = (
        _slot_targets_from_store(args.slots_from_store, args.max_ctx_chars)
        if getattr(args, "slots_from_store", None)
        else (None, {})
    )
    compact = bool(getattr(args, "compact_history", False))
    for spec in args.slot_sources:
        name, root = spec.split("=", 1)
        rng = random.Random(f"{args.seed}:{name}")
        n_src = 0
        n_traj = 0
        want = None if targets is None else targets.get(name, {})
        if want is not None and not want:
            print(f"{name}: no target turns in --slots-from-store; skipped", flush=True)
            continue
        seen: set[str] = set()
        for tid, source, traj, extra in _iter_source_trajectories(
            name, root, rng, args.slots_max_files
        ):
            if want is not None:
                if len(seen) >= len(want):
                    break  # every target trajectory of this source found
                if tid not in want or tid in seen:
                    continue
                seen.add(tid)
            if len(traj) < 6:
                skip("short_trajectory")
                continue
            anchor_len = detect_anchor_len(traj)
            bounds = turn_boundaries(traj, anchor_len)
            if len(bounds) < 3:
                skip("few_turns")
                continue
            if any(SPLICE_SENTINEL in str(m.get("content") or "") for m in traj):
                skip("sentinel_in_source")
                continue
            calls = _tool_result_calls(traj)
            big = {
                i
                for i, m in enumerate(traj)
                if m.get("role") == "tool"
                and len(str(m.get("content") or "")) >= args.slots_min_chars
            }
            # --slots-read-always: every successful read-type result is a slot
            always = (
                {
                    i
                    for i, m in enumerate(traj)
                    if m.get("role") == "tool"
                    and _is_read_result(*calls.get(i, ("tool", {})), str(m.get("content") or ""))
                }
                if args.slots_read_always
                else set()
            )
            if compact:
                # every non-empty tool result (the remap applies the harness's policy)
                big = {
                    i
                    for i, m in enumerate(traj)
                    if m.get("role") == "tool" and str(m.get("content") or "").strip()
                }
                always = set()
            n_traj += 1
            # Candidate turns: the window of history before t holds a large
            # tool result. Trailing indices get the whole trajectory's tail.
            cands = []
            for ti in range(1, len(bounds)):
                lo = anchor_len if compact else bounds[max(0, ti - args.slots_window)]
                if any(lo <= i < bounds[ti] for i in big | always):
                    cands.append(ti)
            rng.shuffle(cands)
            picked = cands[: args.slots_per_traj]
            if want is not None:  # exactly the store's target turns
                picked = [ti for ti in cands if bounds[ti] in want[tid]]
                for _ in range(len(want[tid]) - len(picked)):
                    skip("target_turn_not_candidate")
            for ti in sorted(picked):
                t_idx = bounds[ti]
                lo = anchor_len if compact else bounds[max(0, ti - args.slots_window)]
                target_msg = _normalise_for_template(traj[t_idx], args.keep_reasoning)
                target_msg["content"] = target_msg["content"].strip()
                if not (target_msg["content"] or target_msg.get("tool_calls")):
                    skip("empty_target")
                    continue
                in_window = sorted(i for i in big - always if lo <= i < t_idx)
                slot_idx = (
                    in_window[-args.slots_max :]
                    if args.slots_max > 0 and not compact
                    else in_window
                )
                slot_idx = sorted({*slot_idx, *(i for i in always if lo <= i < t_idx)})
                msgs = [
                    _cap_msg(_normalise_for_template(m, args.keep_reasoning), args.max_msg_chars)
                    for m in traj[:anchor_len]
                ]
                if lo > anchor_len:
                    msgs.append(
                        {"role": "user", "content": "[Earlier turns of this session omitted]"}
                    )
                ctxs, prompts, slot_chars, tools = [], [], [], []
                for i in range(lo, t_idx):
                    m = traj[i]
                    if i in slot_idx:
                        text = str(m.get("content") or "")
                        slot_chars.append(len(text))
                        text = _cap_msg({"content": text}, args.max_ctx_chars)["content"]
                        tname, targs = calls.get(i, ("tool", {}))
                        ctxs.append(text)
                        if want is not None:
                            prompts.append(prompt_of.get((source, tid, _text_key(text)), ""))
                        else:
                            prompts.append(_compact_call(tname, targs))
                        tools.append(tname)
                        msgs.append({"role": "tool", "content": SPLICE_SENTINEL})
                    elif compact:  # the harness keeps its history verbatim; compaction folds it
                        msgs.append(_normalise_for_template(m, args.keep_reasoning))
                    else:
                        cap = args.max_tool_chars if m.get("role") == "tool" else args.max_msg_chars
                        msgs.append(
                            _cap_call_args(
                                _cap_msg(_normalise_for_template(m, args.keep_reasoning), cap),
                                args.max_msg_chars,
                            )
                        )
                if sum(m["content"] == SPLICE_SENTINEL for m in msgs) != len(ctxs) or not ctxs:
                    skip("sentinel_count")
                    continue
                instruction = json.dumps(msgs)
                target = json.dumps([target_msg])
                if renderer is not None:
                    try:
                        segs = renderer.render_slots(instruction)
                        if len(segs) != len(ctxs) + 1:
                            raise ValueError("segments != slots + 1")
                        _ids, mask = renderer.render_chunk(msgs, [target_msg])
                        if not bool(mask.any()):
                            raise ValueError("empty target")
                    except ValueError:
                        skip("unrenderable")
                        continue
                rows.append(
                    {
                        "family": "tool_slots",
                        "framing": "tool_slots",
                        "context": json.dumps(ctxs),
                        "prompt": json.dumps(prompts),
                        "instruction": instruction,
                        "target": target,
                        "split": (
                            want[tid][t_idx]
                            if want is not None
                            else _split_of(
                                extra.get("split_key", f"{source}:{tid}"), args.eval_frac
                            )
                        ),
                        "meta": {
                            **extra.get("meta", {}),
                            "source": source,
                            "trajectory_id": tid,
                            "turn_index": ti,
                            "msg_index": t_idx,
                            "n_slots": len(ctxs),
                            "slot_chars": slot_chars,
                            "tools": tools,
                            **({"compact_history": True} if compact else {}),
                            **(
                                {"prompts_from_store": sum(bool(p) for p in prompts)}
                                if want is not None
                                else {}
                            ),
                        },
                    }
                )
                n_src += 1
            if want is None and n_src >= args.slots_per_source:
                break
        if want is not None and len(seen) < len(want):
            print(f"{name}: {len(want) - len(seen)} target trajectories not found", flush=True)
            skipped["target_trajectory_missing"] = skipped.get(
                "target_trajectory_missing", 0
            ) + len(want) - len(seen)
        print(
            f"{name}: {n_traj} trajectories -> {n_src} rows (total {len(rows)}); "
            f"skipped {skipped}; {time.time() - t0:.0f}s",
            flush=True,
        )
    return rows


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "family",
        choices=[
            "squad",
            "hotpotqa",
            "triviaqa",
            "file_overview",
            "compaction",
            "trajectory_chunks",
            "multineedle",
            "babilong",
            "babi_filler",
            "tool_slots",
        ],
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--squad-dir", default="/home/werg/bgkit-data-nvme/bgkit2/benchmarks/squad")
    ap.add_argument("--bench-dir", default="/home/werg/bgkit-data-nvme/bgkit2/benchmarks")
    ap.add_argument(
        "--swe-dir",
        nargs="+",
        default=["/home/werg/bgkit-data-nvme/capability_packaging/trajectories_ext/swe_zero/data"],
        help="trajectory parquet dirs, searched recursively (SWE-Zero, nebius, Open-SWE-Traces)",
    )
    ap.add_argument("--max-ctx-chars", type=int, default=24000)
    ap.add_argument(
        "--max-msg-chars",
        type=int,
        default=1500,
        help="compaction: cap each prefix message (anchor + live window) to this many chars",
    )
    ap.add_argument("--shards", default="0:20")
    ap.add_argument("--samples-per-traj", type=int, default=4)
    ap.add_argument(
        "--needle-sources",
        nargs="+",
        default=[
            "/home/werg/bgkit-data-nvme/bgkit2/tasks/qa_squad.parquet",
            "/home/werg/bgkit-data-nvme/bgkit2/tasks/qa_hotpotqa.parquet",
        ],
    )
    ap.add_argument("--needle-k", type=int, default=10, help="passages per document")
    ap.add_argument(
        "--babi-zip",
        default="/home/werg/bgkit-data-nvme/capability_packaging/benchmarks/babilong/babilong-repo/data/tasks_1-20_v1-2.zip",
    )
    ap.add_argument("--babi-tasks", nargs="+", default=["qa1", "qa2", "qa3"])
    ap.add_argument("--babi-lengths", nargs="+", type=int, default=[1000, 2000, 4000, 8000])
    ap.add_argument("--babi-max-siblings", type=int, default=2)
    ap.add_argument(
        "--babi-chars-per-token",
        type=float,
        default=0.0,
        help="babi_filler: fixed chars per length-bucket token; 0 = measure per filler source "
        "with --tokenizer (default)",
    )
    ap.add_argument(
        "--babi-token-margin",
        type=float,
        default=0.92,
        help="babi_filler: target tokens = bucket length x this, so the 8000 bucket's tail "
        "stays under the 8192-token encoder cap",
    )
    ap.add_argument(
        "--filler-sources",
        nargs="+",
        default=[
            "/home/werg/bgkit-data-nvme/bgkit2/tasks/qa_squad.parquet",
            "/home/werg/bgkit-data-nvme/bgkit2/tasks/qa_hotpotqa.parquet",
        ],
    )
    ap.add_argument(
        "--filler-jsonl",
        nargs="*",
        default=[],
        help="babi_filler v2: JSONL text dumps ({text, source}) used as CONTIGUOUS filler "
        "documents beside the QA paragraphs (scripts/bgkit2_fetch_filler.py)",
    )
    ap.add_argument(
        "--babilong-dir",
        default="/home/werg/bgkit-data-nvme/capability_packaging/benchmarks/babilong/babilong-1k-samples",
    )
    ap.add_argument("--babilong-lengths", nargs="+", default=["1k", "2k", "4k", "8k"])
    ap.add_argument("--babilong-tasks", nargs="+", default=["qa1", "qa2", "qa3"])
    ap.add_argument("--needle-passage-chars", type=int, default=900)
    ap.add_argument(
        "--chunk-target-chars",
        type=int,
        default=2400,
        help="trajectory_chunks: grow a chunk by turns until it holds about this much text",
    )
    ap.add_argument(
        "--max-tool-chars",
        type=int,
        default=600,
        help="trajectory_chunks: cap tool results INSIDE the chunk (loss-masked) to this",
    )
    ap.add_argument(
        "--slot-sources",
        nargs="+",
        default=TOOL_SLOT_SOURCES,
        help="tool_slots: name=dir of trajectory parquet shards (searched recursively)",
    )
    ap.add_argument("--slots-per-source", type=int, default=10000)
    ap.add_argument("--slots-per-traj", type=int, default=3)
    ap.add_argument(
        "--slots-max", type=int, default=4, help="tool_slots: slots per row (0 = no cap)"
    )
    ap.add_argument(
        "--compact-history",
        action="store_true",
        help="tool_slots: the WHOLE history before the target turn, uncapped, every non-empty "
        "tool result a slot -- input for bgkit2_remap_tool_slots.py --compact-history, which "
        "applies the harness's slot policy and folds the history into the compaction spine "
        "(default off: byte-identical builds)",
    )
    ap.add_argument(
        "--slots-from-store",
        default=None,
        help="tool_slots: build exactly this store's target turns (meta source / trajectory_id "
        "/ msg_index, its split) and reuse its per-slot prompts where the slot text matches",
    )
    ap.add_argument("--slots-min-chars", type=int, default=500)
    ap.add_argument("--slots-window", type=int, default=10, help="tool_slots: history turns kept")
    ap.add_argument("--slots-max-files", type=int, default=None, help="shards per source")
    ap.add_argument("--slots-verify", type=int, default=1, help="tool_slots: render every row")
    ap.add_argument(
        "--slots-read-always",
        action="store_true",
        help="tool_slots: every successful read-type result in the window (str_replace_editor "
        "view, OpenCode read/grep/glob/list) is a slot whatever its size, beside the last "
        "--slots-max large other results (default off: byte-identical builds)",
    )
    ap.add_argument("--tokenizer", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--min-tokens", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument(
        "--keep-reasoning",
        action="store_true",
        help="trajectory_chunks / compaction / tool_slots: keep each assistant turn's reasoning "
        "(reasoning_content/reasoning/thinking field, <think>/<thinking> blocks) in "
        "`reasoning_content` instead of deleting it (default off: byte-identical builds)",
    )
    args = ap.parse_args()
    if Path(args.out).exists():
        sys.exit(f"{args.out} exists; refusing to overwrite")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = {
        "squad": build_squad,
        "hotpotqa": build_hotpotqa,
        "triviaqa": build_triviaqa,
        "file_overview": build_file_overview,
        "compaction": build_compaction,
        "trajectory_chunks": build_trajectory_chunks,
        "multineedle": build_multineedle,
        "babilong": build_babilong,
        "babi_filler": build_babi_filler,
        "tool_slots": build_tool_slots,
    }[args.family](args)
    n = write_task_store(args.out, rows)
    splits = {}
    for r in rows:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    print(json.dumps({"out": args.out, "rows": n, "splits": splits}))


if __name__ == "__main__":
    main()
