#!/usr/bin/env python
"""Build the OFFLINE web-research benchmarks for scripts/bgkit2_eval_web.py (CPU, host venv).

Writes to --out-dir (default $B/benchmarks/web):

  web_pool_corpus.parquet   the paragraph POOL: every paragraph of every HotpotQA-distractor
                            validation question and every MuSiQue validation question, deduped by
                            (title, text), as pages (id ``p<n>``, title, url "", text) -- an
                            open-retrieval environment: the gold paragraphs of a question are
                            somewhere among ~100k others, including every other question's
                            distractors
  web_hotpot.jsonl          the SAME questions / ids / answers as showcase/hotpot.jsonl, with NO
  web_musique.jsonl         segments (the agent must search the pool); ``meta.gold_ids`` /
                            ``meta.gold_titles`` = the supporting paragraphs (retrieval recall)
  browsecomp_plus.jsonl     a seeded sample (--bcp-n, default 150) of BrowseComp-Plus queries over
                            its own 100k-page corpus; ``meta.gold_ids`` = gold_docs,
                            ``meta.evidence_ids`` = evidence_docs
  browsecomp_plus_oracle.jsonl  the same questions with segments = the gold + evidence pages
                            (reading-only variant, scripts/bgkit2_bench.py item schema)
  browsecomp_plus_candtree.jsonl  the same questions with ``pages`` = the query's CANDIDATE SET
                            (gold + evidence + negative docs, deduped) for the per-query
                            candidate-set tree (bgkit2.agent.candtree,
                            scripts/bgkit2_eval_web_candtree.py); --steps candtree
  index/pool, index/browsecomp_plus   on-disk BM25 indexes (agent.web.build_disk_index), with
                            --steps index_pool / index_bcp

Every item has the scripts/bgkit2_bench.py schema {id, bench, subset, segments, question, answers,
scorer, options} plus ``corpus`` (which index to search) and ``meta``. EVAL-ONLY data: the
HotpotQA / MuSiQue validation splits and BrowseComp-Plus's queries. Prints counts only -- never
question or answer text.

    .venv/bin/python scripts/bgkit2_web_bench_prep.py                       # all steps
    .venv/bin/python scripts/bgkit2_web_bench_prep.py --steps index_bcp     # one step
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import sys
import time
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bgkit2.agent.web import Page, build_disk_index, parse_front_matter

B = "/home/werg/bgkit-data-nvme/bgkit2"
BCP_QUERIES = f"{B}/benchmarks/browsecomp_plus/queries.jsonl"
BCP_CORPUS = "/mnt/external/bgkit-data/browsecomp_plus/browsecomp-plus-corpus/data"
STEPS = ("pool", "items", "bcp", "candtree", "index_pool", "index_bcp")


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KiB on Linux


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(path)
    print(f"{path.name}: {len(rows)} items", flush=True)


# ---------------------------------------------------------------- pool
def hotpot_paragraphs(row: dict) -> list[tuple[str, str]]:
    return [(t, " ".join(x.strip() for x in s).strip())
            for t, s in zip(row["context"]["title"], row["context"]["sentences"], strict=True)]


def musique_paragraphs(row: dict) -> list[tuple[str, str]]:
    return [(p["title"], p["paragraph_text"].strip()) for p in row["paragraphs"]]


def build_pool(hotpot_rows, musique_rows) -> tuple[list[Page], dict[str, list[str]]]:
    """Pages deduped by (title, text) in first-seen order (HotpotQA first), and per question
    (``hotpot:<id>`` / ``musique:<id>``) the page ids of its paragraphs, in context order."""
    key2id: dict[tuple[str, str], str] = {}
    pages: list[Page] = []
    q2ids: dict[str, list[str]] = {}

    def add(qid: str, paras: list[tuple[str, str]]) -> None:
        ids = []
        for title, text in paras:
            k = (title, text)
            pid = key2id.get(k)
            if pid is None:
                pid = key2id[k] = f"p{len(pages)}"
                pages.append(Page(pid, title, "", text))
            ids.append(pid)
        q2ids[qid] = ids

    for r in hotpot_rows:
        add(f"hotpot:{r['id']}", hotpot_paragraphs(r))
    for r in musique_rows:
        add(f"musique:{r['id']}", musique_paragraphs(r))
    return pages, q2ids


def web_items(showcase: list[dict], rows: dict[str, dict], q2ids: dict[str, list[str]],
              kind: str) -> list[dict]:
    """The showcase questions (same ids / questions / answers) as pool-search items."""
    out = []
    for it in showcase:
        r = rows[it["id"]]
        ids = q2ids[it["id"]]
        if kind == "hotpot":
            titles = list(dict.fromkeys(r["supporting_facts"]["title"]))
            paras = hotpot_paragraphs(r)
            gold = [pid for pid, (t, _x) in zip(ids, paras, strict=True) if t in set(titles)]
            subset = f"{r.get('type', '')}/{r.get('level', '')}"
        else:
            sup = [p["is_supporting"] for p in r["paragraphs"]]
            gold = [pid for pid, s in zip(ids, sup, strict=True) if s]
            titles = [p["title"] for p in r["paragraphs"] if p["is_supporting"]]
            subset = it["id"].split(":", 1)[1].split("__", 1)[0]  # 2hop / 3hop1 / 4hop2 ...
        out.append({"id": it["id"], "bench": f"web_{kind}", "subset": subset, "corpus": "pool",
                    "segments": [], "question": it["question"], "answers": it["answers"],
                    "scorer": "em_f1", "options": None,
                    "meta": {"gold_ids": list(dict.fromkeys(gold)), "gold_titles": titles}})
    return out


def load_hf():
    from datasets import load_dataset

    hp = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    mq = load_dataset("dgslibisey/MuSiQue", split="validation")
    return hp, mq


def step_pool_and_items(out: Path, showcase_dir: Path, write_pool: bool, write_items: bool) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    t0 = time.time()
    hp, mq = load_hf()
    hp_rows = list(hp)
    mq_rows = list(mq)
    pages, q2ids = build_pool(hp_rows, mq_rows)
    n_para = sum(len(v) for v in q2ids.values())
    print(f"pool: {len(pages)} pages from {n_para} paragraphs of {len(hp_rows)} hotpot + "
          f"{len(mq_rows)} musique questions ({time.time() - t0:.0f}s)", flush=True)
    if write_pool:
        tbl = pa.table({"id": [p.id for p in pages], "title": [p.title for p in pages],
                        "url": [p.url for p in pages], "text": [p.text for p in pages]})
        pq.write_table(tbl, out / "web_pool_corpus.parquet")
        print(f"web_pool_corpus.parquet: {len(pages)} pages", flush=True)
    if write_items:
        rows = {f"hotpot:{r['id']}": r for r in hp_rows}
        rows.update({f"musique:{r['id']}": r for r in mq_rows})
        for kind in ("hotpot", "musique"):
            sc = [json.loads(ln) for ln in (showcase_dir / f"{kind}.jsonl").open() if ln.strip()]
            items = web_items(sc, rows, q2ids, kind)
            ng = [len(i["meta"]["gold_ids"]) for i in items]
            print(f"web_{kind}: gold pages per question min {min(ng)} max {max(ng)} "
                  f"mean {sum(ng) / len(ng):.2f}", flush=True)
            _write(out / f"web_{kind}.jsonl", items)


# ---------------------------------------------------------------- BrowseComp-Plus
def _doc_page(d: dict) -> Page:
    title, body = parse_front_matter(d.get("text") or "")
    return Page(str(d["docid"]), title or d.get("url") or str(d["docid"]), d.get("url") or "",
                body.strip())


def _bcp_keep(queries_path: Path, n: int, seed: int) -> set[int]:
    """The seeded sample of query line indices -- ONE rule for every BrowseComp-Plus bench file,
    so browsecomp_plus / _oracle / _candtree hold the same questions."""
    with queries_path.open("rb") as fh:
        n_lines = sum(1 for ln in fh if ln.strip())
    return set(random.Random(seed).sample(range(n_lines), min(n, n_lines)))


def bcp_candtree(queries_path: Path, n: int, seed: int, out_path: Path) -> list[int]:
    """Stream the candidate-set items of the seeded sample to ``out_path``; returns the candidate
    counts (the file holds the pages, ~2-4 MB per query, so it is never built in memory)."""
    from bgkit2.agent.candtree import candtree_item

    keep = _bcp_keep(queries_path, n, seed)
    counts = []
    tmp = out_path.with_suffix(".jsonl.tmp")
    with queries_path.open() as fh, tmp.open("w") as out:
        i = -1
        for ln in fh:
            if not ln.strip():
                continue
            i += 1
            if i not in keep:
                continue
            it = candtree_item(json.loads(ln))
            counts.append(len(it["pages"]))
            out.write(json.dumps(it, ensure_ascii=False) + "\n")
    tmp.replace(out_path)
    return counts


def bcp_items(queries_path: Path, n: int, seed: int) -> tuple[list[dict], list[dict]]:
    """(search items, oracle items) for a seeded sample of ``n`` queries (file order kept)."""
    keep = _bcp_keep(queries_path, n, seed)
    items, oracle = [], []
    with queries_path.open() as fh:
        i = -1
        for ln in fh:
            if not ln.strip():
                continue
            i += 1
            if i not in keep:
                continue
            q = json.loads(ln)
            gold = [str(d["docid"]) for d in q.get("gold_docs") or []]
            evid = [str(d["docid"]) for d in q.get("evidence_docs") or []]
            base = {"id": f"browsecomp_plus:{q['query_id']}", "subset": "browsecomp_plus",
                    "question": q["query"], "answers": [str(q["answer"])], "scorer": "em_f1",
                    "options": None, "meta": {"gold_ids": gold, "evidence_ids": evid}}
            items.append({**base, "bench": "browsecomp_plus", "corpus": "browsecomp_plus",
                          "segments": []})
            seen: dict[str, Page] = {}
            for d in [*(q.get("gold_docs") or []), *(q.get("evidence_docs") or [])]:
                seen.setdefault(str(d["docid"]), _doc_page(d))
            segs = [f"{p.title}\n{p.text}" for p in seen.values()]
            oracle.append({**base, "bench": "browsecomp_plus_oracle", "segments": segs})
    return items, oracle


def iter_bcp_pages(corpus_dir: Path, limit: int = 0) -> Iterator[Page]:
    import pyarrow.parquet as pq

    n = 0
    for f in sorted(corpus_dir.glob("*.parquet")):
        pf = pq.ParquetFile(f)
        for g in range(pf.metadata.num_row_groups):
            for d in pf.read_row_group(g, columns=["docid", "url", "text"]).to_pylist():
                yield _doc_page(d)
                n += 1
                if limit and n >= limit:
                    return


def iter_pool_pages(path: Path) -> Iterator[Page]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for g in range(pf.metadata.num_row_groups):
        for d in pf.read_row_group(g).to_pylist():
            yield Page(d["id"], d["title"], d["url"], d["text"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default=f"{B}/benchmarks/web")
    ap.add_argument("--showcase-dir", default=f"{B}/benchmarks/showcase")
    ap.add_argument("--bcp-queries", default=BCP_QUERIES)
    ap.add_argument("--bcp-corpus", default=BCP_CORPUS)
    ap.add_argument("--bcp-n", type=int, default=150)
    ap.add_argument("--bcp-limit", type=int, default=0, help="index only the first N pages")
    ap.add_argument("--index-name", default="", help="index dir name override (index_bcp)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", default=",".join(STEPS))
    args = ap.parse_args()
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")  # cached copies; the network is slow
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    steps = [s for s in args.steps.split(",") if s]
    bad = [s for s in steps if s not in STEPS]
    if bad:
        sys.exit(f"unknown steps {bad}; choose from {STEPS}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if "pool" in steps or "items" in steps:
        step_pool_and_items(out, Path(args.showcase_dir), "pool" in steps, "items" in steps)
    if "bcp" in steps:
        t0 = time.time()
        items, oracle = bcp_items(Path(args.bcp_queries), args.bcp_n, args.seed)
        _write(out / "browsecomp_plus.jsonl", items)
        _write(out / "browsecomp_plus_oracle.jsonl", oracle)
        ns = [len(o["segments"]) for o in oracle]
        print(f"browsecomp_plus: oracle pages per question min {min(ns)} max {max(ns)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    if "candtree" in steps:
        t0 = time.time()
        ns = bcp_candtree(Path(args.bcp_queries), args.bcp_n, args.seed,
                          out / "browsecomp_plus_candtree.jsonl")
        print(f"browsecomp_plus_candtree: {len(ns)} queries, candidates per question min "
              f"{min(ns)} mean {sum(ns) / len(ns):.1f} max {max(ns)} ({time.time() - t0:.0f}s, "
              f"peak_rss {peak_rss_gb():.2f} GB)", flush=True)
    for step, name, pages in (
        ("index_pool", "pool", lambda: iter_pool_pages(out / "web_pool_corpus.parquet")),
        ("index_bcp", args.index_name or "browsecomp_plus",
         lambda: iter_bcp_pages(Path(args.bcp_corpus), args.bcp_limit)),
    ):
        if step not in steps:
            continue
        t0 = time.time()
        meta = build_disk_index(pages(), out / "index" / name)
        print(f"index/{name}: {json.dumps(meta)} wall {time.time() - t0:.0f}s "
              f"peak_rss {peak_rss_gb():.2f} GB", flush=True)
    print(f"done; peak_rss {peak_rss_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
