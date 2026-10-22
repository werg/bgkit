"""Offline WEB tools for the agent (S2, 2026-09-22): search / open / find over a FIXED document
corpus (BrowseComp-Plus's 100k-page corpus, or any list of pages), so research trajectories can be
trained and evaluated reproducibly without network access.

  web_search(query, focus)   top-k pages (id, title, snippet) by BM25 over an inverted index
  web_open(id, focus)        the page's text (compressible: a slot with the focus as its prompt)
  web_find(pattern, id?, focus?)
                             the WHOLE page (the one opened last, or ``id``) re-encoded as a
                             compressible slot whose encoder prompt is ``find <pattern> in
                             <title>: <focus>`` (2026-09-22, user: exact-quote tools go through
                             bgkit too; ratio 4 so names / dates / numbers survive). The old
                             behaviour -- the page's lines containing the pattern, verbatim -- is
                             kept ONLY as an eval control arm (``ToolRegistry(web_find_mode=
                             "verbatim")``, ``--web-find verbatim``) to measure the precision cost.

``WEB_TOOL_NAMES`` / ``web_tool_specs()`` are also what the remap of the OpenResearcher
trajectories (browser_search / browser_open / browser_find) targets.
"""

from __future__ import annotations

import bisect
import json
import math
import re
import shutil
import time
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

WEB_TOOL_NAMES = ("web_search", "web_open", "web_find")
_TOK = re.compile(r"[a-z0-9]+")
_STOP = frozenset(["the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is", "are", "was", "were", "be", "by", "with", "as", "at", "from", "that", "this", "it", "which", "who", "what", "when", "where", "how"])


def _terms(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in _STOP]


@dataclass
class Page:
    id: str
    title: str
    url: str
    text: str


class WebCorpus:
    """BM25 over an inverted index (the ToolRegistry's list-scan BM25 is fine for a repo's docs,
    not for 100k pages)."""

    def __init__(self, pages: list[Page], k1: float = 0.9, b: float = 0.4) -> None:
        self.pages = pages
        self.by_id = {p.id: i for i, p in enumerate(pages)}
        self.k1, self.b = k1, b
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lens: list[int] = []
        for i, p in enumerate(pages):
            tf = Counter(_terms(f"{p.title} {p.text}"))
            self.lens.append(sum(tf.values()))
            for t, c in tf.items():
                self.post[t].append((i, c))
        n = max(len(pages), 1)
        self.avg = sum(self.lens) / n if self.lens else 1.0
        self.idf = {t: math.log(1 + (n - len(ps) + 0.5) / (len(ps) + 0.5))
                    for t, ps in self.post.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[Page, float]]:
        acc: dict[int, float] = defaultdict(float)
        for t in set(_terms(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, f in self.post[t]:
                ln = self.lens[i]
                acc[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
        top = sorted(acc.items(), key=lambda x: -x[1])[:k]
        return [(self.pages[i], s) for i, s in top]

    def get(self, page_id: str) -> Page | None:
        i = self.by_id.get(str(page_id))
        return self.pages[i] if i is not None else None


def parse_front_matter(text: str) -> tuple[str, str]:
    """(title, body) of a page whose text opens with a ``---`` front-matter block
    (BrowseComp-Plus: ``---\\ntitle: ...\\ndate: ...\\n---\\n<body>``); ("", text) without one."""
    if not text.startswith("---\n"):
        return "", text
    end = text.find("\n---", 4)
    if end < 0:
        return "", text
    title = ""
    for ln in text[4:end].splitlines():
        if ln.lower().startswith("title:"):
            title = ln.split(":", 1)[1].strip().strip("\"'")
            break
    body = text[end + 4 :]
    return title, body[1:] if body.startswith("\n") else body


# ---------------------------------------------------------------------------
# On-disk index (2026-09-22): the in-memory ``WebCorpus`` keeps every page's text plus one Python
# tuple per (term, page) posting -- fine for a few thousand pages, ~10+ GB for BrowseComp-Plus
# (100k pages, 3.3 GB of text). ``build_disk_index`` writes the SAME BM25 (same terms, k1, b, idf)
# as flat numpy arrays and ``DiskWebCorpus`` memory-maps them, so a searcher costs a few hundred MB.
#
#   meta.json                  n_docs, avg_len, k1, b, n_terms, n_postings, build stats
#   docs.json                  [[id, title, url], ...]
#   text.bin / text_off.npy    utf-8 page texts concatenated, int64 offsets (n_docs + 1)
#   doclen.npy                 int32 BM25 length (term count) per page
#   vocab.bin / vocab_off.npy  sorted terms (ascii) concatenated, int64 offsets; term id = rank
#   post_off.npy               int64 (n_terms + 1) offsets into post_doc / post_tf
#   post_doc.npy / post_tf.npy int32 page index / uint16 term frequency (clipped), page order
# ---------------------------------------------------------------------------

_MAX_TERM_LEN = 40  # longer [a-z0-9]+ runs are hashes / base64 / minified junk, never a query term
INDEX_FILES = ("meta.json", "docs.json", "text.bin", "text_off.npy", "doclen.npy", "vocab.bin",
               "vocab_off.npy", "post_off.npy", "post_doc.npy", "post_tf.npy")


def build_disk_index(pages: Iterable[Page], out_dir: str | Path, k1: float = 0.9, b: float = 0.4,
                     chunk_docs: int = 4096, log_every: int = 10_000) -> dict:
    """Stream ``pages`` into an on-disk BM25 index at ``out_dir`` (replaced atomically: built in
    ``<out_dir>.tmp`` and renamed). Postings are spilled per chunk of pages and scattered into
    their final term-sorted place at the end, so peak memory is the vocabulary dict plus one
    chunk, never the whole posting list. Returns the meta dict."""
    import numpy as np

    out = Path(out_dir)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "spill").mkdir(parents=True)
    t0 = time.time()
    vocab: dict[str, int] = {}
    df = array("q")
    docs: list[list[str]] = []
    lens = array("i")
    text_off = array("q", [0])
    spills: list[Path] = []
    c_doc, c_term, c_tf = array("i"), array("i"), array("H")
    n_post = 0

    def spill() -> None:
        if not c_term:
            return
        p = tmp / "spill" / f"{len(spills):05d}.npz"
        np.savez(p, doc=np.frombuffer(c_doc, np.int32), term=np.frombuffer(c_term, np.int32),
                 tf=np.frombuffer(c_tf, np.uint16))
        spills.append(p)
        del c_doc[:], c_term[:], c_tf[:]

    with (tmp / "text.bin").open("wb") as tfh:
        for i, p in enumerate(pages):
            tf = Counter(t for t in _terms(f"{p.title} {p.text}") if len(t) <= _MAX_TERM_LEN)
            lens.append(sum(tf.values()))
            for t, c in tf.items():
                j = vocab.get(t)
                if j is None:
                    j = vocab[t] = len(vocab)
                    df.append(0)
                df[j] += 1
                c_term.append(j)
                c_tf.append(min(c, 65535))
            c_doc.extend([i] * len(tf))
            n_post += len(tf)
            raw = p.text.encode("utf-8")
            tfh.write(raw)
            text_off.append(text_off[-1] + len(raw))
            docs.append([str(p.id), p.title, p.url])
            if (i + 1) % chunk_docs == 0:
                spill()
            if log_every and (i + 1) % log_every == 0:
                print(f"  indexed {i + 1} pages, {len(vocab)} terms, {n_post} postings, "
                      f"{time.time() - t0:.0f}s", flush=True)
    spill()
    n_docs = len(docs)
    if len({d[0] for d in docs}) != n_docs:
        raise ValueError("page ids are not unique")
    # term id := rank in sorted order (binary-searchable vocabulary on disk)
    terms = list(vocab)  # provisional-id order (dict insertion order)
    order = sorted(range(len(terms)), key=terms.__getitem__)  # rank -> provisional id
    rank = np.empty(len(terms), np.int32)
    rank[np.asarray(order, dtype=np.int64)] = np.arange(len(terms), dtype=np.int32)
    del vocab
    df_np = np.frombuffer(df, np.int64)[np.asarray(order, dtype=np.int64)] if terms else \
        np.zeros(0, np.int64)
    post_off = np.zeros(len(terms) + 1, np.int64)
    np.cumsum(df_np, out=post_off[1:])
    with (tmp / "vocab.bin").open("wb") as vfh:
        voff = array("q", [0])
        for j in order:
            raw = terms[j].encode("ascii")
            vfh.write(raw)
            voff.append(voff[-1] + len(raw))
    np.save(tmp / "vocab_off.npy", np.frombuffer(voff, np.int64))
    del terms, order
    post_doc = np.lib.format.open_memmap(tmp / "post_doc.npy", "w+", np.int32, (n_post,))
    post_tf = np.lib.format.open_memmap(tmp / "post_tf.npy", "w+", np.uint16, (n_post,))
    cursor = post_off[:-1].copy()
    for sp in spills:  # pages ascend across (and within) chunks: each term's list stays in order
        z = np.load(sp)
        t = rank[z["term"]]
        o = np.argsort(t, kind="stable")
        t = t[o]
        first = np.searchsorted(t, t, side="left")
        pos = cursor[t] + (np.arange(len(t)) - first)
        post_doc[pos] = z["doc"][o]
        post_tf[pos] = z["tf"][o]
        cursor += np.bincount(t, minlength=len(cursor))
        sp.unlink()
    assert np.array_equal(cursor, post_off[1:]), "posting scatter mismatch"
    post_doc.flush()
    post_tf.flush()
    del post_doc, post_tf
    (tmp / "spill").rmdir()
    np.save(tmp / "post_off.npy", post_off)
    np.save(tmp / "text_off.npy", np.frombuffer(text_off, np.int64))
    np.save(tmp / "doclen.npy", np.frombuffer(lens, np.int32))
    (tmp / "docs.json").write_text(json.dumps(docs, ensure_ascii=False))
    meta = {"n_docs": n_docs, "avg_len": (sum(lens) / n_docs) if n_docs else 1.0, "k1": k1,
            "b": b, "n_terms": int(len(post_off) - 1), "n_postings": int(n_post),
            "max_term_len": _MAX_TERM_LEN, "build_s": round(time.time() - t0, 1)}
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return meta


class _Vocab:
    """Sorted on-disk term list as a sequence (for ``bisect``)."""

    def __init__(self, blob, off) -> None:
        self.blob, self.off = blob, off

    def __len__(self) -> int:
        return len(self.off) - 1

    def __getitem__(self, i: int) -> str:
        return bytes(self.blob[self.off[i] : self.off[i + 1]]).decode("ascii")


class DiskWebCorpus:
    """``WebCorpus``'s search/get over an index written by ``build_disk_index`` (memory-mapped;
    thread-safe for concurrent searches)."""

    def __init__(self, index_dir: str | Path) -> None:
        import numpy as np

        d = Path(index_dir)
        missing = [f for f in INDEX_FILES if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(f"incomplete web index at {d}: missing {missing}")
        self.meta = json.loads((d / "meta.json").read_text())
        self.k1, self.b, self.avg = self.meta["k1"], self.meta["b"], self.meta["avg_len"]
        self.docs = json.loads((d / "docs.json").read_text())
        self.by_id = {row[0]: i for i, row in enumerate(self.docs)}
        mm = {"mmap_mode": "r"}
        self._text = np.memmap(d / "text.bin", np.uint8, "r") if (d / "text.bin").stat().st_size \
            else np.zeros(0, np.uint8)
        self._text_off = np.load(d / "text_off.npy", **mm)
        self._lens = np.load(d / "doclen.npy").astype(np.float32)
        vb = d / "vocab.bin"
        self._vocab = _Vocab(np.memmap(vb, np.uint8, "r") if vb.stat().st_size else
                             np.zeros(0, np.uint8), np.load(d / "vocab_off.npy", **mm))
        self._post_off = np.load(d / "post_off.npy", **mm)
        self._post_doc = np.load(d / "post_doc.npy", **mm)
        self._post_tf = np.load(d / "post_tf.npy", **mm)
        self._norm = (self.k1 * (1 - self.b + self.b * self._lens / self.avg)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.docs)

    def _term_id(self, t: str) -> int | None:
        i = bisect.bisect_left(self._vocab, t)
        return i if i < len(self._vocab) and self._vocab[i] == t else None

    def page(self, i: int) -> Page:
        pid, title, url = self.docs[i]
        a, e = int(self._text_off[i]), int(self._text_off[i + 1])
        return Page(pid, title, url, bytes(self._text[a:e]).decode("utf-8"))

    def search(self, query: str, k: int = 5) -> list[tuple[Page, float]]:
        import numpy as np

        n = len(self.docs)
        acc = None
        for t in set(_terms(query)):
            j = self._term_id(t)
            if j is None:
                continue
            a, e = int(self._post_off[j]), int(self._post_off[j + 1])
            idf = math.log(1 + (n - (e - a) + 0.5) / ((e - a) + 0.5))
            docs = np.asarray(self._post_doc[a:e])
            f = np.asarray(self._post_tf[a:e], dtype=np.float32)
            if acc is None:
                acc = np.zeros(n, np.float32)
            acc[docs] += idf * f * (self.k1 + 1) / (f + self._norm[docs])
        if acc is None:
            return []
        hit = np.flatnonzero(acc)
        if len(hit) > k:
            hit = hit[np.argpartition(-acc[hit], k - 1)[:k]]
        hit = sorted(hit.tolist(), key=lambda i: (-float(acc[i]), i))
        return [(self.page(i), float(acc[i])) for i in hit]

    def get(self, page_id: str) -> Page | None:
        i = self.by_id.get(str(page_id))
        return self.page(i) if i is not None else None


SNIPPET_WORDS = 100  # per-hit passage width of web_search (a compressible slot, 2026-09-22)


def snippet(text: str, query: str, words: int = SNIPPET_WORDS, width: int | None = None) -> str:
    """The ``words``-word window of ``text`` with the most query terms (windows step by a fifth
    of the width), cut to ``width`` characters (default ``7 * words``, ~700 chars at 100 words).
    Search results are compressible slots, so the passage is long enough to carry the answer
    rather than a 240-char teaser."""
    words = max(1, int(words))
    width = 7 * words if width is None else int(width)
    step = max(1, words // 5)
    q = set(_terms(query))
    toks = text.split()
    best, best_i = -1, 0
    for i in range(0, max(1, len(toks) - words), step):
        s = sum(w.lower().strip(".,;:()\"'") in q for w in toks[i : i + words])
        if s > best:
            best, best_i = s, i
    return " ".join(toks[best_i : best_i + words])[:width]


WEB_FIND_MODES = ("encoded", "verbatim")
WEB_FIND_ENCODED_DESCRIPTION = (
    "Look for a phrase on a page: the whole page (the one opened last, or `id`) is read again "
    "for you with the phrase and your `focus` as what to keep, so the exact names, dates and "
    "numbers around the phrase survive. Says so when the phrase is not on the page."
)
WEB_FIND_VERBATIM_DESCRIPTION = (
    "Find lines containing a pattern in the page opened last (verbatim).")


def web_find_prompt(pattern: str, title: str, focus: str = "") -> str:
    """The encoder prompt of an encoded ``web_find`` slot (``ToolRegistry._prompt`` shape)."""
    head = f"find {pattern} in {title}".strip()
    return f"{head}: {focus}" if focus else head


def web_tool_specs(find_mode: str = "encoded") -> list[dict]:
    """The three web tools' schemas. ``find_mode`` ``"encoded"`` (default) is the trained
    semantics; ``"verbatim"`` is the eval control arm's (the pre-2026-09-22 spec)."""
    from .tools import _FOCUS, ToolSpec, _schema

    if find_mode not in WEB_FIND_MODES:
        raise ValueError(f"find_mode must be one of {WEB_FIND_MODES}, got {find_mode!r}")
    if find_mode == "encoded":
        find = ToolSpec(
            "web_find", WEB_FIND_ENCODED_DESCRIPTION,
            _schema({"pattern": ("string", "text to find"),
                     "id": ("string", "page id; omit for the page opened last"),
                     "focus": _FOCUS}, ["pattern"]),
            compressible=True)
    else:
        find = ToolSpec("web_find", WEB_FIND_VERBATIM_DESCRIPTION,
                        _schema({"pattern": ("string", "text to find")}, ["pattern"]),
                        compressible=False)
    return [
        ToolSpec("web_search", "Search the web. Returns page ids, titles and snippets.",
                 _schema({"query": ("string", "search query"), "focus": _FOCUS}, ["query"]),
                 compressible=True).as_openai(),
        ToolSpec("web_open", "Open a page by its id from web_search results.",
                 _schema({"id": ("string", "page id"), "focus": _FOCUS}, ["id"]),
                 compressible=True).as_openai(),
        find.as_openai(),
    ]


def page_view(page: Page, max_chars: int) -> str:
    """What ``web_open`` (and an encoded ``web_find``) returns for a page."""
    return f"# {page.title}\n{page.url}\n\n{page.text}"[:max_chars]
