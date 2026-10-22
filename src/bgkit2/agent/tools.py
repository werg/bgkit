"""The agent's tools, each returning a ``ToolResult``.

Every result says whether the encoder may compress it (``compressible``) and
carries the compression ``prompt`` the encoder gets (tool name + focus /
arguments — the tool call IS the compression prompt, plan twist 2). ``write``
and ``edit`` are never compressed (one-line acknowledgements).

``zoom`` is encoded too (2026-09-22): the whole file is the slot, the range + focus
its prompt. Predicting verbatim lines (an ``edit``'s ``old``) from an encoding is
the capability mid-training builds (user), so the verbatim path
``zoom(..., exact=true)`` exists but is OFF and absent from the default spec:
``ToolRegistry(zoom_exact=True)`` enables it for post-training / control evals.

    read(path, focus)         whole file                            compressible
    search(query, focus)      "path:line: text" lines (rg or grep)  compressible
    docs(query, focus)        top-k BM25 pages from a page store    compressible
    bash(cmd)                 subprocess in the repo dir, timeout   compressible
    zoom(path, start, end, focus)
                              the WHOLE file, numbered, prompt      compressible (ratio 4)
                              "zoom <path> lines a-b: focus"
      (+ exact, only with ToolRegistry(zoom_exact=True): exact=true = lines start..end, verbatim)
    write(path, content)      create / overwrite a file             verbatim
    edit(path, old, new)      replace a unique substring            verbatim

    attach_web:     web_search / web_open / web_find (all compressible; web_find re-encodes
                    the whole page with "find <pattern> in <title>: <focus>" as its prompt;
                    ``web_find_mode="verbatim"`` = the eval control arm's verbatim lines)
    attach_memory:  memory_search, memory_read (both compressible)

Exact-quote tools go through bgkit too (user, 2026-09-22): only the opt-in control paths
(``zoom_exact=True`` + ``exact=true``, ``web_find_mode="verbatim"``) and the one-line
acknowledgements stay verbatim.

"Sandboxed" here means: paths are confined to the repo directory and ``bash``
runs with a timeout in that directory. It is NOT an isolation boundary — the
SWE-bench scoring step runs in the official harness's docker images, and a
real evaluation should run the agent's ``bash`` inside the same container.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

DOC_SUFFIXES = (".md", ".txt", ".rst")


@dataclass(frozen=True)
class ToolResult:
    """What a tool returned and how the decoder should receive it."""

    text: str
    compressible: bool
    prompt: str
    tool: str = ""
    args: dict = field(default_factory=dict)
    ok: bool = True
    # A CACHED slot (``browse``): (marker context, n reps). The student fills
    # the slot from the tree-node cache; ``text`` is the teacher's/full-text
    # view of the same node (its directory listing).
    cached: tuple[str, int] | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict
    compressible: bool

    def as_openai(self) -> dict:
        """The ``tools=`` entry the LFM2.5 chat template serialises."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolError(Exception):
    """A tool refused its arguments; reported to the model as an error result."""


def _schema(props: dict[str, tuple[str, str]], required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {k: {"type": t, "description": d} for k, (t, d) in props.items()},
        "required": required,
    }


_FOCUS = ("string", "what to look for in the result (guides compression)")

ZOOM_DESCRIPTION = (
    "Look closely at lines start..end of a file: the whole file is read for you with that "
    "range (and your `focus`) as what to keep."
)
# only with ToolRegistry(zoom_exact=True) (post-training / control evals)
ZOOM_EXACT_SENTENCE = (
    " Set `exact` to true to get lines start..end VERBATIM instead (never compressed), e.g. "
    "right before `edit`, whose `old` must match the file byte for byte."
)


def zoom_spec(exact: bool = False) -> ToolSpec:
    """``zoom``'s schema; ``exact`` adds the verbatim switch (off by default: not in the spec
    the trained model sees)."""
    props = {
        "path": ("string", "path relative to the repo root"),
        "start": ("integer", "first line (1-based, inclusive)"),
        "end": ("integer", "last line (inclusive)"),
        "focus": _FOCUS,
    }
    if exact:
        props["exact"] = ("boolean", "true = lines start..end verbatim, never compressed")
    return ToolSpec("zoom", ZOOM_DESCRIPTION + (ZOOM_EXACT_SENTENCE if exact else ""),
                    _schema(props, ["path", "start", "end"]), compressible=True)


def zoom_prompt(path: str, start: int, end: int, focus: str = "") -> str:
    """The encoder prompt of an encoded ``zoom`` slot (``ToolRegistry._prompt`` shape)."""
    head = f"zoom {path} lines {start}-{end}"
    return f"{head}: {focus}" if focus else head


def as_bool(x) -> bool:
    """A model-emitted boolean argument (``true`` / ``"true"`` / ``1``)."""
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes")
    return bool(x)


class BM25:
    """Plain Okapi BM25 over pre-tokenised documents."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = (sum(self.lens) / len(docs)) if docs else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, ln in zip(self.tfs, self.lens, strict=True):
            s = 0.0
            for t in query:
                if t not in tf:
                    continue
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * ln / max(self.avg, 1e-9))
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            out.append(s)
        return out


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _ws_norm(text: str) -> str:
    return "\n".join(" ".join(ln.split()) for ln in text.strip().splitlines())


class EditAudit:
    """A ``ToolRegistry`` wrapper that records, for every ``edit`` call, whether its ``old``
    occurs VERBATIM in the file at that point -- copying vs guessing, now that ``zoom`` is an
    encoding (the agent has to reproduce ``old`` byte for byte from reps). Per edit:
    ``old_verbatim`` (occurs at least once), ``old_unique`` (exactly once: the edit applies),
    ``old_ws_match`` (occurs up to whitespace: a copy with the spacing wrong), ``ok``."""

    def __init__(self, reg: ToolRegistry) -> None:
        self.reg = reg
        self.edits: list[dict] = []

    def __getattr__(self, name):  # everything else run_episode needs
        return getattr(self.reg, name)

    def call(self, name: str, args: dict | None = None) -> ToolResult:
        if name != "edit":
            return self.reg.call(name, args)
        a = dict(args or {})
        old = a.get("old")
        rec = {"path": str(a.get("path", "")), "old_chars": len(old) if isinstance(old, str) else 0,
               "file_exists": False, "old_occurrences": 0, "old_verbatim": False,
               "old_unique": False, "old_ws_match": False}
        try:
            p = self.reg._resolve(rec["path"])
        except ToolError:
            p = None
        if p is not None and p.is_file() and isinstance(old, str) and old:
            text = p.read_text(errors="replace")
            n = text.count(old)
            rec.update(file_exists=True, old_occurrences=n, old_verbatim=n > 0, old_unique=n == 1,
                       old_ws_match=n > 0 or _ws_norm(old) in _ws_norm(text))
        elif p is not None and p.is_file():
            rec["file_exists"] = True
        res = self.reg.call(name, args)
        rec["ok"] = res.ok
        self.edits.append(rec)
        return res

    def summary(self, max_listed: int = 20) -> dict:
        """The per-episode record fields (``edit_*``)."""
        n = len(self.edits)

        def frac(k: str) -> float | None:
            return round(sum(bool(e[k]) for e in self.edits) / n, 4) if n else None

        return {"edit_calls": n, "edit_old_verbatim": frac("old_verbatim"),
                "edit_old_unique": frac("old_unique"), "edit_old_ws_match": frac("old_ws_match"),
                "edits": self.edits[:max_listed]}


class ToolRegistry:
    """The tool set over one repository checkout.

    ``call(name, args)`` never raises for model-facing mistakes (unknown tool,
    bad arguments, missing file): those come back as ``ok=False`` results so
    the loop can hand the error to the model.
    """

    def __init__(
        self,
        repo_dir: str | Path,
        docs_dir: str | Path | None = None,
        bash_timeout: float = 30.0,
        max_result_chars: int = 40_000,
        max_search_results: int = 200,
        max_zoom_lines: int = 400,
        max_ls_entries: int = 400,
        max_tree_files: int = 96,
        web_find_mode: str = "encoded",
        zoom_exact: bool = False,
    ) -> None:
        from .web import WEB_FIND_MODES

        if web_find_mode not in WEB_FIND_MODES:
            raise ValueError(f"web_find_mode must be one of {WEB_FIND_MODES}")
        self.web_find_mode = web_find_mode
        # the verbatim zoom path (exact=true): off, and absent from the spec, unless enabled
        self.zoom_exact = bool(zoom_exact)
        self.repo_dir = Path(repo_dir).resolve()
        if not self.repo_dir.is_dir():
            raise ValueError(f"repo_dir is not a directory: {self.repo_dir}")
        self.docs_dir = Path(docs_dir).resolve() if docs_dir else None
        self.bash_timeout = bash_timeout
        self.max_result_chars = max_result_chars
        self.max_search_results = max_search_results
        self.max_zoom_lines = max_zoom_lines
        self.max_ls_entries = max_ls_entries
        self.max_tree_files = max_tree_files
        self._tools: dict[str, tuple[ToolSpec, Callable[..., ToolResult]]] = {}
        self._bm25: tuple[list[Path], BM25] | None = None
        self._register_defaults()

    # ------------------------------------------------------------------
    def register(self, spec: ToolSpec, fn: Callable[..., ToolResult]) -> None:
        self._tools[spec.name] = (spec, fn)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def spec(self, name: str) -> ToolSpec:
        return self._tools[name][0]

    def specs(self) -> list[dict]:
        return [s.as_openai() for s, _ in self._tools.values()]

    def call(self, name: str, args: dict | None = None) -> ToolResult:
        args = dict(args or {})
        if name not in self._tools:
            return ToolResult(
                f"error: unknown tool {name!r}; available: {', '.join(self.names)}",
                compressible=False,
                prompt="",
                tool=name,
                args=args,
                ok=False,
            )
        _spec, fn = self._tools[name]
        try:
            return fn(**args)
        except (ToolError, TypeError, ValueError, OSError) as exc:
            return ToolResult(
                f"error: {name}: {exc}",
                compressible=False,
                prompt="",
                tool=name,
                args=args,
                ok=False,
            )

    # ------------------------------------------------------------------
    def _register_defaults(self) -> None:
        self.register(
            ToolSpec(
                "repo_tree",
                "Load the repository's source files as ONE compressed tree encoding "
                "(mirrors the directory structure; the answer to 'which file' is a path). "
                "Call this first on an unfamiliar repository.",
                _schema({"focus": _FOCUS}, []),
                compressible=True,
            ),
            self.repo_tree,
        )
        self.register(
            ToolSpec(
                "overview",
                "Map of the repository: one line per module with the classes and functions "
                "it defines. Call this FIRST to find which file is relevant.",
                _schema({"focus": _FOCUS}, []),
                compressible=True,
            ),
            self.overview,
        )
        self.register(
            ToolSpec(
                "ls",
                "List files in the repository (optionally under a directory). "
                "Use this FIRST when you do not know the layout — `read` needs an exact path.",
                _schema(
                    {
                        "dir": ("string", "directory relative to the repo root; omit for the root"),
                        "focus": _FOCUS,
                    },
                    [],
                ),
                compressible=True,
            ),
            self.ls,
        )
        self.register(
            ToolSpec(
                "read",
                "Read a file from the repository.",
                _schema(
                    {"path": ("string", "path relative to the repo root"), "focus": _FOCUS},
                    ["path"],
                ),
                compressible=True,
            ),
            self.read,
        )
        self.register(
            ToolSpec(
                "search",
                "Search the repository for a regular expression; returns path:line: text.",
                _schema({"query": ("string", "regular expression"), "focus": _FOCUS}, ["query"]),
                compressible=True,
            ),
            self.search,
        )
        self.register(
            ToolSpec(
                "docs",
                "Search library / framework documentation pages.",
                _schema({"query": ("string", "search terms"), "focus": _FOCUS}, ["query"]),
                compressible=True,
            ),
            self.docs,
        )
        self.register(zoom_spec(self.zoom_exact), self.zoom)
        self.register(
            ToolSpec(
                "write",
                "Create or overwrite a file with the given content.",
                _schema(
                    {
                        "path": ("string", "path relative to the repo root"),
                        "content": ("string", ""),
                    },
                    ["path", "content"],
                ),
                compressible=False,
            ),
            self.write,
        )
        self.register(
            ToolSpec(
                "edit",
                "Replace one unique occurrence of `old` in a file with `new`.",
                _schema(
                    {
                        "path": ("string", "path relative to the repo root"),
                        "old": ("string", "exact text to replace (must occur once)"),
                        "new": ("string", "replacement text"),
                    },
                    ["path", "old", "new"],
                ),
                compressible=False,
            ),
            self.edit,
        )
        self.register(
            ToolSpec(
                "bash",
                "Run a shell command in the repository root (with a timeout).",
                _schema({"cmd": ("string", "the command")}, ["cmd"]),
                compressible=True,
            ),
            self.bash,
        )

    # ------------------------------------------------------------------
    def attach_tree(self, view) -> None:
        """Enable ``browse`` over a repo's cached tree (agent.overview.RepoTreeView)."""
        self._tree = view
        self.register(
            ToolSpec(
                "browse",
                "Open one directory of the repository overview: its files and "
                "sub-directories, compressed. `dir` is a directory path from the "
                "repository root ('' = the root).",
                _schema({"dir": ("string", "directory path, e.g. 'src/models'")}, ["dir"]),
                compressible=True,
            ),
            self.browse,
        )

    def attach_web(self, corpus, k: int = 5, max_page_chars: int = 40_000,
                   snippet_words: int | None = None) -> None:
        """Enable the offline web tools over a fixed corpus (agent.web.WebCorpus).
        ``snippet_words``: the per-hit passage width of ``web_search`` (default
        ``web.SNIPPET_WORDS`` = 100 words, ~700 chars; the result is a compressible slot)."""
        from .web import SNIPPET_WORDS, web_tool_specs

        self._web, self._web_k, self._web_page_chars = corpus, int(k), int(max_page_chars)
        self._web_snippet_words = int(snippet_words or SNIPPET_WORDS)
        self._web_last: str = ""
        self._web_last_page = None  # the page opened last (web_find's default target)
        fns = {"web_search": self.web_search, "web_open": self.web_open, "web_find": self.web_find}
        verbatim = self.web_find_mode == "verbatim"
        for spec in web_tool_specs(self.web_find_mode):
            fn = spec["function"]
            self.register(ToolSpec(fn["name"], fn["description"], fn["parameters"],
                                   compressible=not (verbatim and fn["name"] == "web_find")),
                          fns[fn["name"]])

    def attach_memory(self, store, k: int = 5, snippet_words: int | None = None) -> None:
        """Enable the two-stage memory tools over a ``agent.memory.MemoryStore``:
        ``memory_search`` (ids + dates + snippets, a compressible slot like
        ``web_search``, prompt ``memory search <query>: <focus>``) and
        ``memory_read`` (a session as a compressible slot, prompt
        ``session <date>: <focus>``).
        The always-on index is not a tool: ``agent.memory.add_memory_index``."""
        from ..data.memory_framing import MEMORY_SNIPPET_WORDS
        from .memory import memory_tool_specs

        self._memory, self._memory_k = store, int(k)
        self._memory_snippet_words = int(snippet_words or MEMORY_SNIPPET_WORDS)
        fns = {"memory_search": self.memory_search, "memory_read": self.memory_read}
        for spec in memory_tool_specs():
            fn = spec["function"]
            self.register(ToolSpec(fn["name"], fn["description"], fn["parameters"],
                                   compressible=True), fns[fn["name"]])

    def memory_search(self, query: str, focus: str = "") -> ToolResult:
        from ..data.memory_framing import search_prompt
        from .memory import search_text

        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string")
        text = search_text(self._memory, query, str(focus or ""), k=self._memory_k,
                           snippet_words=self._memory_snippet_words)
        return ToolResult(text, compressible=True, prompt=search_prompt(query, str(focus or "")),
                          tool="memory_search", args={"query": query, "focus": focus})

    def memory_read(self, session: str, focus: str = "") -> ToolResult:
        from ..data.memory_framing import read_prompt

        s = self._memory.resolve(session)
        if s is None:
            n = len(self._memory)
            raise ToolError(f"no session {session!r}; the ids are S1..S{n} (see the memory index "
                            "or memory_search)")
        return ToolResult(s.text, compressible=True, prompt=read_prompt(s.date, str(focus or "")),
                          tool="memory_read", args={"session": session, "focus": focus})

    def keep_only(self, names) -> None:
        """Drop every tool not in ``names`` (the web-research eval exposes ONLY the web tools)."""
        keep = set(names)
        self._tools = {k: v for k, v in self._tools.items() if k in keep}

    def web_search(self, query: str, focus: str = "") -> ToolResult:
        from .web import snippet

        hits = self._web.search(str(query), k=self._web_k)
        body = "\n\n".join(f"[{p.id}] {p.title}\n{snippet(p.text, query, self._web_snippet_words)}"
                           for p, _ in hits)
        return ToolResult(body or f"no results for {query!r}", compressible=True,
                          prompt=self._prompt(f"web search {query}", focus), tool="web_search",
                          args={"query": query, "focus": focus})

    def web_open(self, id: str, focus: str = "") -> ToolResult:
        from .web import page_view

        page = self._web.get(str(id))
        if page is None:
            raise ToolError(f"no page with id {id!r} (use an id from web_search)")
        self._web_last, self._web_last_page = page.text, page
        return ToolResult(page_view(page, self._web_page_chars), compressible=True,
                          prompt=self._prompt(f"page {page.title}", focus),
                          tool="web_open", args={"id": id, "focus": focus})

    def web_find(self, pattern: str, focus: str = "", id: str | None = None) -> ToolResult:
        """Encoded (default): the WHOLE page -- ``id``'s, else the page opened last -- as a
        compressible slot (the same text ``web_open`` returns) with the prompt ``find <pattern>
        in <title>: <focus>``; a pattern that is not on the page is an inline "not found".
        ``web_find_mode="verbatim"`` (eval control arm only): the page's lines containing the
        pattern, verbatim."""
        from .web import page_view, web_find_prompt

        if id is not None and str(id).strip():
            page = self._web.get(str(id))
            if page is None:
                raise ToolError(f"no page with id {id!r} (use an id from web_search)")
        else:
            page = self._web_last_page
            if page is None:
                raise ToolError("open a page with web_open first (or pass its id)")
        pat = str(pattern).lower()
        args = {"pattern": pattern, **({"id": id} if id else {}), **({"focus": focus} if focus
                                                                     else {})}
        if self.web_find_mode == "verbatim":
            lines = [ln.strip() for ln in re.split(r"(?<=[.!?])\s+|\n", page.text)
                     if pat and pat in ln.lower()][:20]
            return ToolResult("\n".join(lines) or f"{pattern!r} not found on the page",
                              compressible=False, prompt="", tool="web_find", args=args)
        if not pat.strip() or pat not in f"{page.title}\n{page.text}".lower():
            return ToolResult(f"{pattern!r} not found on the page", compressible=False,
                              prompt="", tool="web_find", args=args)
        return ToolResult(page_view(page, self._web_page_chars), compressible=True,
                          prompt=web_find_prompt(str(pattern), page.title, str(focus or "")),
                          tool="web_find", args=args)

    def browse(self, dir: str = "") -> ToolResult:
        from ..data.overview import overview_marker

        view = getattr(self, "_tree", None)
        if view is None:
            raise ToolError("no repository overview is loaded")
        if not isinstance(dir, str):
            raise ToolError("dir must be a string")
        node = dir.strip().removeprefix("./").strip("/")
        node = "" if node == "." else node
        if node not in view.children:
            anc = node
            while anc and anc not in view.children:
                anc = anc.rpartition("/")[0]
            is_dir = any(p.startswith(node + "/") for p in view.paths)
            if is_dir:
                raise ToolError(
                    f"`{node}` has no entry of its own; its files are listed under "
                    f"`{anc or '.'}` -- browse that, or read the file directly"
                )
            raise ToolError(f"not a directory of this repository: {dir}")
        return ToolResult(
            view.listing(node) or "(empty)",
            compressible=True,
            prompt=f"contents of {node or '.'}",
            tool="browse",
            args={"dir": dir},
            cached=(overview_marker(view.repo, view.n_reps(node), node), view.n_reps(node)),
        )

    def _resolve(self, path: str) -> Path:
        if not isinstance(path, str) or not path:
            raise ToolError("path must be a non-empty string")
        p = (self.repo_dir / path).resolve()
        if p != self.repo_dir and self.repo_dir not in p.parents:
            raise ToolError(f"path escapes the repository: {path}")
        return p

    def _rel(self, p: Path) -> str:
        return str(p.relative_to(self.repo_dir))

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_result_chars:
            return text
        return text[: self.max_result_chars] + f"\n... (truncated at {self.max_result_chars} chars)"

    @staticmethod
    def _prompt(head: str, focus: str) -> str:
        return f"{head}: {focus}" if focus else head

    # ------------------------------------------------------------------
    def repo_tree(self, focus: str = "") -> ToolResult:
        """The repository as a TREE payload for the student server.

        Unlike every other tool, the result is not text the encoder reads
        flat: it is a JSON payload of (path, source) pairs that the server
        hands to ``encode_tree`` — file -> directory -> root through the repro
        bridge — so the slot holds the root encoding of the whole repo
        (plans/multilevel_encoder_2026_09_16.md). ``overview`` (an AST name
        index) stays as the baseline this must beat.
        """
        import json as _json

        pairs: list[dict] = []
        for q in sorted(self.repo_dir.rglob("*.py")):
            if any(part.startswith(".") or part == "__pycache__" for part in q.parts):
                continue
            try:
                src = q.read_text(errors="replace")
            except OSError:
                continue
            if not src.strip():
                continue
            pairs.append({"path": self._rel(q), "source": src[: self.max_result_chars]})
            if len(pairs) >= self.max_tree_files:
                break
        payload = _json.dumps({"repo_tree": 1, "files": pairs})
        return ToolResult(
            payload,
            compressible=True,
            prompt=self._prompt("encode the repository tree for navigation", focus),
            tool="repo_tree",
            args={"focus": focus},
        )

    def overview(self, focus: str = "") -> ToolResult:
        """Structural map of the repository.

        `ls` gives bare paths, which say nothing about what a file does — P26's
        eval showed a flat listing moved tool use 17% -> 10%, i.e. not at all.
        This gives the model something to reason over, and because the result is
        compressible it becomes an encoder slot: finding the file is then a
        compression task, which is what the encoder is good at.
        """
        from ..data.repo_corpus import module_map_line

        lines: list[str] = []
        for q in sorted(self.repo_dir.rglob("*.py")):
            if any(part.startswith(".") or part == "__pycache__" for part in q.parts):
                continue
            try:
                source = q.read_text(errors="replace")
            except OSError:
                continue
            line = module_map_line(self._rel(q), source)
            if line is None:
                continue
            lines.append(line)
            if len(lines) >= self.max_ls_entries:
                break
        body = "\n".join(lines)
        return ToolResult(
            body or "(no python modules)",
            compressible=True,
            prompt=self._prompt("map of the repository's modules and definitions", focus),
            tool="overview",
            args={"focus": focus},
        )

    def ls(self, dir: str = "", focus: str = "") -> ToolResult:
        """File listing. The agent has no other way to learn the layout, and
        without it a wrong basename is its only possible first move (P26 final
        eval: 84% of unseen-repo episodes answered without calling any tool)."""
        root = self._resolve(dir) if dir else self.repo_dir
        if not root.is_dir():
            raise ToolError(f"not a directory: {dir}")
        skip = {"__pycache__", "node_modules", "site-packages", "dist", "build", ".git"}
        names = sorted(
            self._rel(q)
            for q in root.rglob("*")
            if q.is_file()
            and not any(part.startswith(".") or part in skip for part in q.parts)
            and q.suffix not in {".pyc", ".pyo", ".so", ".o", ".a", ".bin", ".pkl", ".png",
                                 ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz", ".whl"}
        )
        shown = names[: self.max_ls_entries]
        body = "\n".join(shown)
        if len(names) > len(shown):
            body += f"\n... and {len(names) - len(shown)} more files"
        return ToolResult(
            body or "(empty)",
            compressible=True,
            prompt=self._prompt(f"list files under {self._rel(root)}", focus),
            tool="ls",
            args={"dir": dir, "focus": focus},
        )

    def read(self, path: str, focus: str = "") -> ToolResult:
        p = self._resolve(path)
        if not p.is_file():
            # Name the recovery explicitly: the student's dominant failure is a
            # bare basename that does not exist, followed by repeating it.
            raise ToolError(
                f"not a file: {path}. Call `ls` to see the real paths, or `search` "
                "for a term from the question; do not guess another filename."
            )
        text = p.read_text(errors="replace")
        return ToolResult(
            self._truncate(text),
            compressible=True,
            prompt=self._prompt(f"read {self._rel(p)}", focus),
            tool="read",
            args={"path": path, "focus": focus},
        )

    def search(self, query: str, focus: str = "") -> ToolResult:
        if not isinstance(query, str) or not query:
            raise ToolError("query must be a non-empty string")
        rg = shutil.which("rg")
        if rg:
            cmd = [rg, "--no-heading", "--line-number", "--no-color", "-e", query, "."]
        else:
            cmd = ["grep", "-rnIE", "--exclude-dir=.git", "-e", query, "."]
        proc = subprocess.run(
            cmd, cwd=self.repo_dir, capture_output=True, text=True, timeout=self.bash_timeout
        )
        if proc.returncode not in (0, 1):
            raise ToolError(proc.stderr.strip() or f"search failed ({proc.returncode})")
        lines = []
        for raw in proc.stdout.splitlines():
            parts = raw.split(":", 2)
            if len(parts) < 3:
                continue
            path, line, text = parts
            path = path[2:] if path.startswith("./") else path
            lines.append(f"{path}:{line}: {text.rstrip()}")
        lines.sort()
        n = len(lines)
        if n > self.max_search_results:
            lines = lines[: self.max_search_results]
            lines.append(f"... ({n - self.max_search_results} more matches omitted)")
        text = "\n".join(lines) if lines else f"no matches for {query!r}"
        return ToolResult(
            self._truncate(text),
            compressible=True,
            prompt=self._prompt(f"search {query}", focus),
            tool="search",
            args={"query": query, "focus": focus},
        )

    def _doc_index(self) -> tuple[list[Path], BM25] | None:
        if self.docs_dir is None or not self.docs_dir.is_dir():
            return None
        if self._bm25 is None:
            pages = sorted(p for p in self.docs_dir.rglob("*") if p.suffix in DOC_SUFFIXES)
            self._bm25 = (pages, BM25([tokenize(p.read_text(errors="replace")) for p in pages]))
        return self._bm25

    def docs(self, query: str, focus: str = "", k: int = 3) -> ToolResult:
        index = self._doc_index()
        if index is None or not index[0]:
            return ToolResult(
                "docs: no documentation store is configured for this run",
                compressible=False,
                prompt="",
                tool="docs",
                args={"query": query, "focus": focus},
                ok=False,
            )
        pages, bm25 = index
        root = pages[0].parent if self.docs_dir is None else self.docs_dir
        scores = bm25.scores(tokenize(query))
        order = sorted(range(len(pages)), key=lambda i: (-scores[i], str(pages[i])))
        hits = [i for i in order[: max(1, int(k))] if scores[i] > 0]
        if not hits:
            text = f"no documentation pages match {query!r}"
        else:
            text = "\n\n".join(
                f"# {pages[i].relative_to(root)}\n{pages[i].read_text(errors='replace')}"
                for i in hits
            )
        return ToolResult(
            self._truncate(text),
            compressible=True,
            prompt=self._prompt(f"docs {query}", focus),
            tool="docs",
            args={"query": query, "focus": focus},
        )

    def zoom(
        self, path: str, start: int, end: int, focus: str = "", exact: bool | str = False
    ) -> ToolResult:
        """The WHOLE file, numbered, as a compressible slot whose prompt names the range
        (``zoom <path> lines <a>-<b>: <focus>``, ratio 4). ``exact=true`` (only when the
        registry was built with ``zoom_exact=True``): lines start..end verbatim, never
        compressed."""
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"not a file: {path}")
        start, end = int(start), int(end)
        if start < 1 or end < start:
            raise ToolError("need 1 <= start <= end")
        exact = as_bool(exact)
        if exact and not self.zoom_exact:
            raise ToolError("zoom has no `exact` argument in this run; drop it")
        lines = p.read_text(errors="replace").splitlines()
        if not exact:
            if start > len(lines):
                return ToolResult(f"(file has {len(lines)} lines; nothing in {start}..{end})",
                                  compressible=False, prompt="", tool="zoom",
                                  args={"path": path, "start": start, "end": end})
            text = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))
            return ToolResult(
                self._truncate(text),
                compressible=True,
                prompt=zoom_prompt(self._rel(p), start, min(end, len(lines)), str(focus or "")),
                tool="zoom",
                args={"path": path, "start": start, "end": end, **({"focus": focus} if focus
                                                                   else {})},
            )
        end = min(end, start + self.max_zoom_lines - 1)
        chunk = lines[start - 1 : end]
        text = "\n".join(f"{start + i}: {line}" for i, line in enumerate(chunk))
        if not chunk:
            text = f"(file has {len(lines)} lines; nothing in {start}..{end})"
        return ToolResult(
            text,
            compressible=False,
            prompt="",
            tool="zoom",
            args={"path": path, "start": start, "end": end, "exact": True},
        )

    def write(self, path: str, content: str) -> ToolResult:
        p = self._resolve(path)
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return ToolResult(
            f"wrote {self._rel(p)} ({len(content.splitlines())} lines)",
            compressible=False,
            prompt="",
            tool="write",
            args={"path": path},
        )

    def edit(self, path: str, old: str, new: str) -> ToolResult:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"not a file: {path}")
        if not isinstance(old, str) or not old:
            raise ToolError("old must be a non-empty string")
        text = p.read_text()
        n = text.count(old)
        if n == 0:
            raise ToolError(f"old text not found in {path}")
        if n > 1:
            raise ToolError(f"old text occurs {n} times in {path}; include more context")
        p.write_text(text.replace(old, new, 1))
        return ToolResult(
            f"edited {self._rel(p)}",
            compressible=False,
            prompt="",
            tool="edit",
            args={"path": path},
        )

    def bash(self, cmd: str) -> ToolResult:
        if not isinstance(cmd, str) or not cmd.strip():
            raise ToolError("cmd must be a non-empty string")
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                timeout=self.bash_timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                f"error: command timed out after {self.bash_timeout:g}s",
                compressible=False,
                prompt="",
                tool="bash",
                args={"cmd": cmd},
                ok=False,
            )
        out = proc.stdout
        if proc.stderr:
            out += ("\n" if out else "") + "STDERR:\n" + proc.stderr
        if proc.returncode != 0:
            out += ("\n" if out else "") + f"(exit code {proc.returncode})"
        return ToolResult(
            self._truncate(out or "(no output)"),
            compressible=True,
            prompt=f"bash: {cmd}",
            tool="bash",
            args={"cmd": cmd},
            ok=proc.returncode == 0,
        )
