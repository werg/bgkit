#!/usr/bin/env python
"""Repo-awareness questions with EXACT answers, read off the Python AST.

For each eligible repo (pure-Python, has a tests directory; see
``bgkit2.data.repo_corpus``) every ``.py`` blob at HEAD is parsed and six
question forms are emitted, each with an exact answer and evidence paths:

    def_path             Which file defines the function/class `X`?   (X unique in repo)
    signature            What is the signature of `X` in `path`?      (normalised def line)
    callers              Which functions in `path` call `Y`?          (Y a unique repo def,
                                                                       called as a bare name or
                                                                       as self./cls.Y)
    return_type          What is the declared return type of `X` in `path`?
    return_literal       What does `X` in `path` return?              (every return = one literal)
    raises               Which exception does `X` in `path` raise?    (exactly one class)
    importers            Which modules import `Z`?                    (1-5 paths)
    constant_readers     Which files read the constant `K`?           (1-5 paths)
    config_key_readers   Which files read the config key / env var `K`? (1-5 paths)

Outputs (never overwritten; ``--suffix`` names a new generation):
    <out_dir>/repo_qa_tasks<suffix>.jsonl   one task per line
    <out_dir>/repo_qa_file<suffix>.parquet  bgkit2 task store, family ``repo_qa_file``

2026-09-14 audit fixes: decorators and default arguments of a ``def`` are no
longer attributed to that function as calls (``@for_app("java") def match``
made ``match`` a caller of ``for_app`` in 45 rows), and ``callers`` only counts
calls that can resolve to the repo's definition — bare names and ``self.`` /
``cls.`` attribute calls — so ``m.group()`` / ``s.upper()`` / ``parser.parse_args()``
no longer count as calls of a repo function that happens to share the name.

The store row's context is the evidence file (``# file: <path>`` header +
text, <= 24,000 chars). Questions whose evidence spans several files
(importers / readers with >1 path) go to the JSONL only unless
``--store-multi-evidence`` is given: a target listing paths the context does
not show would train the decoder to guess.

Run:
    .venv/bin/python scripts/bgkit2_gen_repo_qa.py --workers 4
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pathlib
import random
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bgkit.data.repo_files import iter_repo_files
from bgkit2.data.repo_corpus import (
    DefInfo,
    RepoCandidate,
    file_header,
    is_test_path,
    normalised_def_line,
    python_repo_candidates,
    qualified_defs,
    repo_eligible,
    scan_head_tree,
    split_of,
)
from bgkit2.data.tasks import write_task_store

MAX_CONTEXT_CHARS = 24_000
MAX_FILE_BYTES = 400_000  # parsed for uniqueness; only <= MAX_CONTEXT_CHARS are evidence
MAX_FILES_PER_REPO = 3000
MAX_SET_ANSWER = 5

_UPPER_CONST = re.compile(r"^[A-Z][A-Z0-9_]{3,}$")
_CONFIG_NAMES = {"config", "cfg", "conf", "settings", "options", "opts", "params", "args"}
# Callee names too generic to make a meaningful "who calls Y" question even
# when the repo happens to define one.
_COMMON_CALLEES = {
    "get",
    "set",
    "add",
    "run",
    "main",
    "setup",
    "init",
    "load",
    "save",
    "read",
    "write",
    "open",
    "close",
    "update",
    "create",
    "delete",
    "start",
    "stop",
    "parse",
    "process",
    "call",
    "apply",
    "log",
    "info",
    "debug",
    "warning",
    "error",
    "format",
    "join",
    "append",
    "items",
    "keys",
    "values",
    "copy",
    "len",
    "str",
    "int",
    "float",
    "list",
    "dict",
    "print",
    "isinstance",
    "range",
    "super",
    "type",
    "hasattr",
    "getattr",
    "setattr",
    "sorted",
    "enumerate",
    "zip",
    "map",
    "filter",
    "any",
    "all",
    "min",
    "max",
    "sum",
    "abs",
    "round",
    "next",
    "iter",
    "id",
    "hash",
    "repr",
    "bool",
    "tuple",
    "bytes",
    "exit",
    "forward",
    "predict",
    "fit",
    "transform",
    "step",
    "reset",
    "render",
    "build",
    "execute",
    "handle",
    "check",
    "validate",
    "clean",
    "test",
    "assert_equal",
}
_SKIP_DEF_NAMES = {"main", "setup", "run", "test", "cli", "app", "get", "set"}


@dataclass
class FileFacts:
    path: str
    text: str
    defs: list[DefInfo]
    def_names: Counter
    callers: dict[str, set[str]]  # callee name -> innermost enclosing function qualnames
    imports: set[str]
    constants: set[str]  # module-level UPPER_CASE assignments
    loads: set[str]  # Name loads + Attribute attrs
    config_keys: set[str]
    env_keys: set[str]

    @property
    def store_ok(self) -> bool:
        return len(self.text) + len(file_header(self.path)) <= MAX_CONTEXT_CHARS


def _callee_name(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _is_environ(node: ast.AST) -> bool:
    return (isinstance(node, ast.Name) and node.id == "environ") or (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _str_const(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value:
        return node.value
    return None


def analyse_file(path: str, text: str) -> FileFacts | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # SyntaxWarning on bad escapes in old code
            module = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None
    defs = qualified_defs(module)
    def_by_node = {id(d.node): d for d in defs}
    callers: dict[str, set[str]] = defaultdict(set)
    imports: set[str] = set()
    constants: set[str] = set()
    loads: set[str] = set()
    config_keys: set[str] = set()
    env_keys: set[str] = set()

    for node in module.body:
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Name) and _UPPER_CONST.match(t.id):
                constants.add(t.id)

    def visit(node: ast.AST, fn_stack: list[DefInfo]) -> None:
        for child in ast.iter_child_nodes(node):
            visit_node(child, fn_stack)

    def visit_node(child: ast.AST, stack: list[DefInfo]) -> None:
        d = def_by_node.get(id(child))
        if d is not None and d.kind == "function":
            # Decorators and default values run in the ENCLOSING scope: walk
            # them with the outer stack; only the body is "inside" (nested
            # defs included — each body statement comes back through here).
            for dec in child.decorator_list:
                visit_node(dec, stack)
            visit(child.args, stack)
            inner = [*stack, d]
            for stmt in child.body:
                visit_node(stmt, inner)
            return
        if isinstance(child, ast.Import):
            imports.update(a.name for a in child.names)
        elif isinstance(child, ast.ImportFrom):
            if child.level == 0 and child.module:
                imports.add(child.module)
        elif isinstance(child, ast.Call):
            name = _callee_name(child)
            f = child.func
            resolvable = isinstance(f, ast.Name) or (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name)
                and f.value.id in ("self", "cls")
            )
            if name and stack and resolvable:
                callers[name].add(stack[-1].qualname)
            f = child.func
            if isinstance(f, ast.Attribute) and f.attr in ("get", "getenv") and child.args:
                key = _str_const(child.args[0])
                if key:
                    if _is_environ(f.value) or (
                        f.attr == "getenv" and isinstance(f.value, ast.Name) and f.value.id == "os"
                    ):
                        env_keys.add(key)
                    elif isinstance(f.value, ast.Name) and f.value.id in _CONFIG_NAMES:
                        config_keys.add(key)
            elif isinstance(f, ast.Name) and f.id == "getenv" and child.args:
                key = _str_const(child.args[0])
                if key:
                    env_keys.add(key)
        elif isinstance(child, ast.Subscript):
            key = _str_const(child.slice)
            if key:
                if _is_environ(child.value):
                    env_keys.add(key)
                elif isinstance(child.value, ast.Name) and child.value.id in _CONFIG_NAMES:
                    config_keys.add(key)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            loads.add(child.id)
        elif isinstance(child, ast.Attribute):
            loads.add(child.attr)
        visit(child, stack)

    visit(module, [])
    return FileFacts(
        path=path,
        text=text,
        defs=defs,
        def_names=Counter(d.name for d in defs),
        callers=dict(callers),
        imports=imports,
        constants=constants,
        loads=loads,
        config_keys=config_keys,
        env_keys=env_keys,
    )


def _direct_returns_and_raises(fn: ast.AST) -> tuple[list[ast.Return], list[str | None]]:
    """Return / raise statements of ``fn`` excluding nested defs and lambdas.
    A raise entry is the exception class name, or None when not a plain name."""
    returns: list[ast.Return] = []
    raises: list[str | None] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Return):
                returns.append(child)
            elif isinstance(child, ast.Raise):
                exc = child.exc
                if isinstance(exc, ast.Call):
                    exc = exc.func
                if isinstance(exc, ast.Name):
                    raises.append(exc.id)
                elif isinstance(exc, ast.Attribute):
                    raises.append(exc.attr)
                else:
                    raises.append(None)
            visit(child)

    visit(fn)
    return returns, raises


@dataclass
class Candidate:
    answer_type: str
    question: str
    answer: str
    evidence: list[str]  # evidence[0] is the primary file
    verifier: str


def _lit(value: object) -> str:
    return repr(value)


def candidates_for_repo(facts: dict[str, FileFacts]) -> list[Candidate]:
    global_def_count: Counter = Counter()
    for f in facts.values():
        global_def_count.update(f.def_names)
    const_definers: dict[str, list[str]] = defaultdict(list)
    for f in facts.values():
        for k in f.constants:
            const_definers[k].append(f.path)
    out: list[Candidate] = []

    for path, f in sorted(facts.items()):
        # Test files still count for uniqueness and appear in set answers, but
        # are not worth a question of their own as the evidence file.
        if not f.store_ok or is_test_path(path):
            continue
        local_qual = Counter(d.qualname for d in f.defs)
        for d in f.defs:
            if d.name.startswith("__") or d.name.startswith("test") or len(d.name) < 3:
                continue
            if d.name in _SKIP_DEF_NAMES:
                continue
            if global_def_count[d.name] == 1:
                out.append(
                    Candidate(
                        "def_path",
                        f"Which file defines the {d.kind} `{d.name}`?",
                        path,
                        [path],
                        "path",
                    )
                )
            if d.depth > 2 or local_qual[d.qualname] != 1 or f.def_names[d.name] != 1:
                continue
            ref = f"`{d.qualname}` in `{path}`"
            out.append(
                Candidate(
                    "signature",
                    f"What is the signature of {ref}?",
                    normalised_def_line(d.node),
                    [path],
                    "exact",
                )
            )
            if d.kind != "function":
                continue
            node = d.node
            assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            if node.returns is not None:
                out.append(
                    Candidate(
                        "return_type",
                        f"What is the declared return type of {ref}?",
                        ast.unparse(node.returns),
                        [path],
                        "exact",
                    )
                )
            returns, raises = _direct_returns_and_raises(node)
            if returns and all(
                r.value is not None
                and isinstance(r.value, ast.Constant)
                and not isinstance(r.value.value, (bytes, type(...)))
                for r in returns
            ):
                lits = {_lit(r.value.value) for r in returns}  # type: ignore[union-attr]
                if len(lits) == 1:
                    out.append(
                        Candidate(
                            "return_literal",
                            f"What does {ref} return?",
                            next(iter(lits)),
                            [path],
                            "exact",
                        )
                    )
            if raises and None not in raises and len(set(raises)) == 1:
                out.append(
                    Candidate(
                        "raises",
                        f"Which exception does {ref} raise?",
                        raises[0],
                        [path],
                        "exact",
                    )
                )
        for callee, fns in sorted(f.callers.items()):
            if (
                global_def_count[callee] != 1
                or callee in _COMMON_CALLEES
                or callee.startswith("_")
                or len(callee) < 3
                or not (1 <= len(fns) <= MAX_SET_ANSWER)
            ):
                continue
            out.append(
                Candidate(
                    "callers",
                    f"Which functions in `{path}` call `{callee}`?",
                    ", ".join(sorted(fns)),
                    [path],
                    "set",
                )
            )

    importers: dict[str, list[str]] = defaultdict(list)
    for f in facts.values():
        for m in f.imports:
            importers[m].append(f.path)
    for mod, paths in sorted(importers.items()):
        if not (1 <= len(paths) <= MAX_SET_ANSWER) or ("." not in mod and len(mod) < 4):
            continue
        paths = sorted(paths)
        out.append(
            Candidate("importers", f"Which modules import `{mod}`?", ", ".join(paths), paths, "set")
        )

    for k, definers in sorted(const_definers.items()):
        if len(definers) != 1:
            continue
        readers = sorted(f.path for f in facts.values() if k in f.loads)
        if not (1 <= len(readers) <= MAX_SET_ANSWER):
            continue
        out.append(
            Candidate(
                "constant_readers",
                f"Which files read the constant `{k}`?",
                ", ".join(readers),
                readers,
                "set",
            )
        )

    key_readers: dict[tuple[str, str], list[str]] = defaultdict(list)
    for f in facts.values():
        for k in f.config_keys:
            key_readers[("config key", k)].append(f.path)
        for k in f.env_keys:
            key_readers[("environment variable", k)].append(f.path)
    for (kind, k), paths in sorted(key_readers.items()):
        if not (1 <= len(paths) <= MAX_SET_ANSWER):
            continue
        paths = sorted(paths)
        out.append(
            Candidate(
                "config_key_readers",
                f"Which files read the {kind} `{k}`?",
                ", ".join(paths),
                paths,
                "set",
            )
        )
    return out


def sample_repo(
    cands: list[Candidate],
    facts: dict[str, FileFacts],
    *,
    rng: random.Random,
    max_per_repo: int,
    min_per_file: int,
    max_per_file: int,
    max_multi: int,
) -> tuple[list[Candidate], list[Candidate]]:
    """(store-eligible picks grouped by file with >= min_per_file each,
    JSONL-only multi-evidence picks)."""
    by_file: dict[str, list[Candidate]] = defaultdict(list)
    multi: list[Candidate] = []
    for c in cands:
        if len(c.evidence) == 1 and facts[c.evidence[0]].store_ok:
            by_file[c.evidence[0]].append(c)
        else:
            multi.append(c)
    files = [p for p, cs in by_file.items() if len(cs) >= min_per_file]
    rng.shuffle(files)
    picked: list[Candidate] = []
    for p in files:
        if len(picked) + min_per_file > max_per_repo:
            break
        cs = by_file[p]
        rng.shuffle(cs)
        # round-robin over answer types so a file group is not all one form
        by_type: dict[str, list[Candidate]] = defaultdict(list)
        for c in cs:
            by_type[c.answer_type].append(c)
        take: list[Candidate] = []
        types = sorted(by_type)
        rng.shuffle(types)
        while len(take) < max_per_file and any(by_type.values()):
            for t in types:
                if by_type[t] and len(take) < max_per_file:
                    take.append(by_type[t].pop())
        room = max_per_repo - len(picked)
        if len(take) > room:
            take = take[:room]
        if len(take) >= min_per_file:
            picked.extend(take)
    rng.shuffle(multi)
    room = max_per_repo - len(picked)
    return picked, multi[: max(0, min(max_multi, room))]


def _task_id(repo: str, question: str) -> str:
    return "repo_qa:" + hashlib.sha1(f"{repo}\n{question}".encode()).hexdigest()[:16]


def process_repo(job: tuple[RepoCandidate, dict]) -> dict:
    cand, cfg = job
    import pygit2

    t0 = time.time()
    res: dict = {"repo": cand.full_name, "stars": cand.stars, "skip": None, "tasks": [], "rows": []}
    try:
        repo = pygit2.Repository(cand.path)
        scan = scan_head_tree(repo)
    except Exception as e:  # unreadable clone
        res["skip"] = f"unreadable: {type(e).__name__}"
        return res
    reason = repo_eligible(
        scan,
        min_py_frac=cfg.get("min_py_frac", 0.8),
        require_tests=cfg.get("require_tests", True),
    )
    if reason:
        res["skip"] = reason
        return res
    facts: dict[str, FileFacts] = {}
    n_unparsed = 0
    for path, _sha, text in iter_repo_files(
        Path(cand.path),
        min_bytes=1,
        max_bytes=MAX_FILE_BYTES,
        max_files=MAX_FILES_PER_REPO,
        exts={".py"},
    ):
        f = analyse_file(path, text)
        if f is None:
            n_unparsed += 1
            continue
        facts[path] = f
    if len(facts) < 5:
        res["skip"] = f"fewer than 5 parseable python files ({len(facts)})"
        return res
    cands = candidates_for_repo(facts)
    rng = random.Random(f"{cfg['seed']}:{cand.full_name}")
    picked, multi = sample_repo(
        cands,
        facts,
        rng=rng,
        max_per_repo=cfg["max_per_repo"],
        min_per_file=cfg["min_per_file"],
        max_per_file=cfg["max_per_file"],
        max_multi=cfg["max_multi_per_repo"],
    )
    if not picked:
        res["skip"] = f"no file with >= {cfg['min_per_file']} store-eligible questions"
        return res
    split = split_of(cand.full_name, cfg["eval_frac"])
    for c in picked + multi:
        res["tasks"].append(
            {
                "task_id": _task_id(cand.full_name, c.question),
                "repo": cand.full_name,
                "commit": scan.head_sha,
                "question": c.question,
                "answer": c.answer,
                "answer_type": c.answer_type,
                "evidence_paths": c.evidence,
                "verifier": c.verifier,
            }
        )
    for c in picked + (multi if cfg["store_multi_evidence"] else []):
        primary = c.evidence[0]
        f = facts[primary]
        if not f.store_ok:
            continue
        res["rows"].append(
            {
                "family": "repo_qa_file",
                "framing": "chat_context",
                "context": file_header(primary) + f.text,
                "prompt": f"Compress the file to answer: {c.question}\n",
                "instruction": c.question,
                "target": c.answer,
                "split": split,
                "meta": {
                    "doc": f"{cand.full_name}:{primary}",
                    "repo": cand.full_name,
                    "answer_type": c.answer_type,
                    "task_id": _task_id(cand.full_name, c.question),
                    "n_evidence": len(c.evidence),
                },
            }
        )
    res["stats"] = {
        "files": len(facts),
        "unparsed": n_unparsed,
        "candidates": len(cands),
        "secs": round(time.time() - t0, 1),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", default="/home/werg/bgkit-data-nvme/bgkit2/tasks")
    ap.add_argument("--suffix", default="", help='output name suffix, e.g. "_v2"')
    ap.add_argument("--max-repos", type=int, default=400)
    ap.add_argument("--max-per-repo", type=int, default=40)
    ap.add_argument("--min-per-file", type=int, default=3)
    ap.add_argument("--max-per-file", type=int, default=6)
    ap.add_argument(
        "--max-multi-per-repo",
        type=int,
        default=5,
        help="multi-evidence (JSONL-only) questions per repo",
    )
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--store-multi-evidence", action="store_true")
    ap.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="only consider the first N candidate repos (debug)",
    )
    ap.add_argument("--exclude-repos", default="", help="file of repo full_names to skip")
    ap.add_argument("--min-py-frac", type=float, default=0.8)
    ap.add_argument(
        "--no-require-tests",
        action="store_true",
        help="accept repos without a tests directory (AST/history tasks do not need one)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"repo_qa_tasks{args.suffix}.jsonl"
    store_path = out_dir / f"repo_qa_file{args.suffix}.parquet"
    for p in (jsonl_path, store_path):
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")

    cands = python_repo_candidates(os.environ["DATA_DIR"])
    if args.max_candidates:
        cands = cands[: args.max_candidates]
    if args.exclude_repos:
        # Held-out generation: every repo already used by a TRAINING task file
        # is dropped, so the resulting tasks share no repo with training data.
        # Without this the agent eval scored the student on tasks it had
        # trained on -- 87%% of the held-out repo_qa slice and 89%% of
        # localisation, because the eval's task-id hash picks a slice of the
        # same file the training episodes were generated from (2026-09-16).
        drop = {
            ln.strip()
            for ln in pathlib.Path(args.exclude_repos).read_text().splitlines()
            if ln.strip()
        }
        before = len(cands)
        cands = [c for c in cands if c.full_name not in drop]
        print(f"excluded {before - len(cands)} repos named in {args.exclude_repos}", flush=True)
    print(f"{len(cands)} downloaded Python repos as candidates", flush=True)
    cfg = {
        "min_py_frac": args.min_py_frac,
        "require_tests": not args.no_require_tests,
        "seed": args.seed,
        "max_per_repo": args.max_per_repo,
        "min_per_file": args.min_per_file,
        "max_per_file": args.max_per_file,
        "max_multi_per_repo": args.max_multi_per_repo,
        "eval_frac": args.eval_frac,
        "store_multi_evidence": args.store_multi_evidence,
    }
    skips: Counter = Counter()
    used: list[str] = []
    rows: list[dict] = []
    n_tasks = 0
    type_counts: Counter = Counter()
    t0 = time.time()
    with Pool(args.workers) as pool, open(jsonl_path, "w") as fh:
        for i, res in enumerate(pool.imap(process_repo, ((c, cfg) for c in cands))):
            if res["skip"]:
                skips[res["skip"].split(" (")[0].split(":")[0]] += 1
            else:
                used.append(res["repo"])
                for t in res["tasks"]:
                    fh.write(json.dumps(t, ensure_ascii=False) + "\n")
                    type_counts[t["answer_type"]] += 1
                n_tasks += len(res["tasks"])
                rows.extend(res["rows"])
                print(
                    f"[{len(used)}/{args.max_repos}] {res['repo']} (*{res['stars']}): "
                    f"{len(res['tasks'])} tasks, {len(res['rows'])} rows; {res['stats']}; "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )
            if len(used) >= args.max_repos:
                pool.terminate()
                break
            if (i + 1) % 50 == 0:
                print(f"  scanned {i + 1} candidates, {len(used)} used", flush=True)
    n_rows = write_task_store(str(store_path), rows)
    docs = Counter(
        json.loads(r["meta"])["doc"] if isinstance(r["meta"], str) else r["meta"]["doc"]
        for r in rows
    )
    print(
        f"\nrepos used: {len(used)}; tasks: {n_tasks} -> {jsonl_path}; "
        f"store rows: {n_rows} ({sum(1 for r in rows if r['split'] == 'eval')} eval) "
        f"over {len(docs)} documents -> {store_path}",
        flush=True,
    )
    print("per answer_type:", dict(sorted(type_counts.items())), flush=True)
    print("skips:", dict(skips.most_common()), flush=True)


if __name__ == "__main__":
    main()
