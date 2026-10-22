#!/usr/bin/env python
"""Floor-test candidate task families BEFORE any of them costs GPU time.

Four families are prototyped here. Each targets a structural gap that every
shipped wide-net family shares, not a parametric tweak of one:

``repolookup``  MULTI-DOCUMENT. Every existing family scopes one document, so
                a query-INDEPENDENT survivor policy satisfies every question
                at once (measured: own-vs-foreign query Jaccard 0.967). Put K
                files in scope and the model must locate WHICH one.
``aggregate``   DISTRIBUTED. Every existing family's answer is one span, so
                the head can win by keeping one predictable region. An argmax
                over every definition in the file ("which function takes the
                most parameters") cannot be answered from any single span.
``tabular``     STRUCTURED TOOL OUTPUT, and the sharpest anti-query-blindness
                shape available: each question addresses a DIFFERENT row, so
                the union over queries is the whole table -- far past any
                retention budget. Also the bulkiest thing a real agent has to
                compact, and not code or prose.
``compaction``  The actual product goal: recall a fact from a conversation
                prefix that was replaced by blobs.

Nothing here writes a dataset. It emits rows and their floors, because two
families (lognav, fileneedle) shipped into training before anyone measured
what a model scores WITHOUT the document, and fileneedle's headline number
turned out to be its own question-echo.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

CODE_EXT = {".py", ".js", ".ts", ".go", ".rs", ".java", ".rb", ".c", ".cc", ".cpp", ".h"}
DEF_RE = re.compile(
    r"^[ \t]*(?:export\s+)?(?:async\s+)?"
    r"(?:def|class|func|function|fn|type|struct|interface)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.M,
)

# ---------------------------------------------------------------- repolookup


def repolookup_candidates(files: list[tuple[str, str]], rng, *, max_per_repo=4) -> list[dict]:
    """``files`` = [(path, text)] from ONE repo. Answer = the defining file.

    A symbol defined in more than one of the scoped files has no unique
    answer, and one defined in a file that also merely MENTIONS it elsewhere
    is still fine -- the question asks where it is DEFINED.
    """
    where: dict[str, set[str]] = defaultdict(set)
    for path, text in files:
        for m in DEF_RE.finditer(text):
            where[m.group(1)].add(path)
    unique = [(s, next(iter(p))) for s, p in where.items() if len(p) == 1 and len(s) >= 5]
    rng.shuffle(unique)
    out = []
    for sym, path in unique[:max_per_repo]:
        out.append(
            {
                "question": f"Which file defines `{sym}`?",
                "answer": Path(path).name,
                "symbol": sym,
                "doc_id": path,
                "scope": [p for p, _ in files],
            }
        )
    return out


# ----------------------------------------------------------------- aggregate


def aggregate_candidates(
    path: str, text: str, rng, *, max_per_doc=2, min_margin: dict | None = None
) -> list[dict]:
    """Argmax over every definition in the file -- no single span answers it."""
    if not path.endswith(".py"):
        return []
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return []
    fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(fns) < 6:
        return []  # an argmax over three candidates is nearly a coin flip
    out = []

    def _params(f):
        a = f.args
        return len(a.args) + len(a.posonlyargs) + len(a.kwonlyargs)

    def _lines(f):
        return (getattr(f, "end_lineno", f.lineno) or f.lineno) - f.lineno

    # A MARGIN, not merely the absence of a tie. Measured on 150 repos, the
    # parameter-count argmax beat its runner-up by a median of ONE argument:
    # unguessable (gate 0.033) but a discrimination this fine is the reconstruct
    # trap -- a task no 0.8B decoder resolves even from the full text, which
    # produces a floor-level score that says nothing about the reps. The
    # line-count argmax already leads by a median of 11.
    margins = {"params": 2, "lines": 5} | (min_margin or {})
    for key, tag, question in (
        (_params, "params", "Which function in this file takes the most parameters?"),
        (_lines, "lines", "Which function in this file has the most lines?"),
    ):
        ranked = sorted(fns, key=key, reverse=True)
        if len(ranked) < 2 or key(ranked[0]) - key(ranked[1]) < margins[tag]:
            continue
        if key(ranked[0]) <= 1:
            continue
        # STRUCTURAL floors. An answer-string floor cannot see a policy like
        # "always name the first function": that policy still reads the
        # document, so echo and cross-doc both miss it, but it is exactly the
        # query-INDEPENDENT shortcut this family exists to rule out. Record
        # where the argmax sits in definition order so it can be measured.
        order = sorted(fns, key=lambda f: f.lineno)
        rank = order.index(ranked[0])
        out.append(
            {
                "question": question,
                "answer": ranked[0].name,
                "doc_id": path,
                "n_candidates": len(fns),
                "def_rank": rank,
                "is_first": rank == 0,
                "is_last": rank == len(order) - 1,
                "margin": key(ranked[0]) - key(ranked[1]),
                "name_len_rank": sorted(
                    (len(f.name) for f in fns), reverse=True
                ).index(len(ranked[0].name)),
            }
        )
    rng.shuffle(out)
    return out[:max_per_doc]


# ------------------------------------------------------------------- tabular


def tabular_candidates(path: str, text: str, rng, *, max_per_doc=3) -> list[dict]:
    """Key->value lookup inside a structured file. Each question, a different row."""
    name = Path(path).name
    rows: list[tuple[str, str, str]] = []  # (container, key, value)

    if path.endswith(".json"):
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, ValueError, RecursionError):
            return []
        if not isinstance(obj, dict):
            return []
        for k, v in obj.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, str) and 1 <= len(vv) <= 40:
                        rows.append((k, kk, vv))
    elif path.endswith((".csv", ".tsv")):
        sep = "\t" if path.endswith(".tsv") else ","
        lines = [ln for ln in text.splitlines() if ln.strip()][:5000]
        if len(lines) < 20:
            return []
        header = [h.strip() for h in lines[0].split(sep)]
        if len(header) < 2 or any(not h for h in header):
            return []
        key_col = 0
        for ln in lines[1:]:
            cells = [c.strip() for c in ln.split(sep)]
            if len(cells) != len(header):
                continue
            for ci in range(1, len(header)):
                if cells[key_col] and 1 <= len(cells[ci]) <= 40:
                    rows.append((header[ci], cells[key_col], cells[ci]))
    else:
        return []

    if len(rows) < 10:
        return []  # a table small enough to keep whole is not a compression test
    rng.shuffle(rows)
    out = []
    for container, key, value in rows[:max_per_doc]:
        out.append(
            {
                "question": f"In `{name}`, what is the value of `{key}` under `{container}`?",
                "answer": value,
                "doc_id": path,
                "n_rows_in_doc": len(rows),
            }
        )
    return out


# ----------------------------------------------------------------- patchpoint

_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
_KEYWORDS = {
    "self", "this", "return", "import", "from", "class", "def", "None", "True",
    "False", "else", "elif", "while", "const", "let", "var", "function", "type",
    "func", "struct", "public", "private", "static", "void", "int", "string",
    "bool", "error", "async", "await", "export", "default", "value", "data",
}


def patchpoint_candidates(path: str, text: str, rng, *, max_per_doc=3) -> list[dict]:
    """Quote a snippet with ONE identifier swapped; ask for the original.

    The axis no other family touches. ``aggregate`` breaks the single-span
    assumption but offers only two query types per document, so a
    query-independent policy still covers both at once. Here the altered site
    is uniformly distributed over the file, so the union over the query space
    IS the whole document -- the exact condition the query-blindness root
    cause says is required (union of answer spans 0.4-3.1% of the document
    fitted inside a 10% budget, so one fixed policy answered every question).

    The decoy is drawn from the SAME file, so the alteration cannot be spotted
    as an out-of-place token without locating the original.
    """
    lines = text.splitlines()
    if len(lines) < 40:
        return []
    idents = Counter(
        m.group(0)
        for m in _IDENT_RE.finditer(text)
        if m.group(0) not in _KEYWORDS
    )
    pool = [w for w, c in idents.items() if c >= 2]
    if len(pool) < 8:
        return []

    out = []
    for _ in range(max_per_doc * 6):
        if len(out) >= max_per_doc:
            break
        li = rng.randrange(2, len(lines) - 2)
        line = lines[li]
        hits = [m for m in _IDENT_RE.finditer(line) if m.group(0) in idents
                and m.group(0) not in _KEYWORDS]
        if not hits:
            continue
        m = rng.choice(hits)
        original = m.group(0)
        decoy = rng.choice(pool)
        if decoy == original:
            continue
        window = lines[li - 2 : li + 3]
        altered = line[: m.start()] + decoy + line[m.end() :]
        window[2] = altered
        snippet = "\n".join(window)
        # If the original still appears in the quoted window, the answer is
        # readable off the prompt -- that is the fileneedle failure exactly.
        if re.search(rf"\b{re.escape(original)}\b", snippet):
            continue
        # The query-INDEPENDENT shortcut for this family: "always answer the
        # file's most frequent identifier" reads the document but ignores the
        # question, so neither the echo nor the cross-document floor can see
        # it -- the same blind spot that let a first/last-definition policy
        # hide inside ``aggregate``'s floors until it was measured directly.
        ranked_idents = [w for w, _ in idents.most_common()]
        out.append(
            {
                "question": (
                    "This snippet is from the file, but one identifier has been "
                    f"replaced. What was the original identifier?\n\n{snippet}"
                ),
                "answer": original,
                "doc_id": path,
                "line_frac": li / len(lines),
                "n_lines": len(lines),
                "ident_rank": ranked_idents.index(original),
                "n_idents": len(ranked_idents),
            }
        )
    return out


# ----------------------------------------------------------------- the driver


def collect(repos_dir: Path, *, n_repos: int, seed: int) -> dict[str, list[dict]]:
    from bgkit.data.repo_files import DATA_EXTS, iter_repo_files

    rng = random.Random(seed)
    repos: list[Path] = []
    for owner in sorted(repos_dir.iterdir()):
        if owner.is_dir():
            repos.extend(p for p in sorted(owner.iterdir()) if p.is_dir())
    rng.shuffle(repos)

    fams: dict[str, list[dict]] = {
        "repolookup": [], "aggregate": [], "tabular": [], "patchpoint": [],
    }
    used = 0
    for repo in repos:
        if used >= n_repos:
            break
        try:
            got = list(
                iter_repo_files(repo, min_bytes=1_000, max_bytes=400_000, max_files=40)
            )
        except Exception:
            continue
        if not got:
            continue
        used += 1
        code = [(p, t) for p, _, t in got if Path(p).suffix in CODE_EXT]
        if len(code) >= 4:
            fams["repolookup"].extend(repolookup_candidates(code[:12], rng))
        for p, _, t in got:
            fams["aggregate"].extend(aggregate_candidates(p, t, rng))
            fams["patchpoint"].extend(patchpoint_candidates(p, t, rng))
        # Structured data files are a SEPARATE pass: the shared iterator is
        # source-only by design, and its minified heuristics reject serialized
        # data on a property ("a human wrote these lines") that data does not
        # have and does not need.
        try:
            data_files = list(
                iter_repo_files(
                    repo,
                    min_bytes=2_000,
                    max_bytes=2_000_000,
                    max_files=15,
                    exts=DATA_EXTS,
                    skip_minified=False,
                )
            )
        except Exception:
            data_files = []
        for p, _, t in data_files:
            fams["tabular"].extend(tabular_candidates(p, t, rng))

    # The corpus filters are not optional and cannot run per document: whether
    # `index.js` or `__init__` is a guessable answer is a property of the whole
    # corpus, invisible to a generator holding one file. Same filter xref uses.
    from bgkit.data.xref_qa import filter_candidates

    return {k: filter_candidates(v) for k, v in fams.items()}


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from audit_task_floors import audit

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos-dir", required=True)
    ap.add_argument("--n-repos", type=int, default=120)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()

    fams = collect(Path(a.repos_dir), n_repos=a.n_repos, seed=a.seed)
    print(f"{'family':<12} {'rows':>6} {'uniq':>6} {'top20':>7} {'echo':>6} "
          f"{'xdoc':>6} {'modal':>6} {'GATE':>6} {'chars':>6}")
    for name, rows in fams.items():
        if not rows:
            print(f"{name:<12} {'0':>6}  (no candidates)")
            continue
        r = audit(rows)
        print(
            f"{name:<12} {r['n_rows']:>6} {r['unique_frac']:>6.3f} "
            f"{r['top20_coverage']:>7.3f} {r['echo_f1']:>6.3f} "
            f"{r['cross_doc_f1']:>6.3f} {r['modal_answer_f1']:>6.3f} "
            f"{r['gate']:>6.3f} {r['mean_answer_chars']:>6.1f}"
        )
        if a.out_dir:
            a.out_dir.mkdir(parents=True, exist_ok=True)
            (a.out_dir / f"{name}.jsonl").write_text(
                "\n".join(json.dumps(x) for x in rows) + "\n"
            )


if __name__ == "__main__":
    main()
