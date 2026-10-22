"""SWE-bench Lite LOCALISATION eval: which file(s) must change to fix the issue.

The CPU-side logic of ``scripts/bgkit2_eval_swebench_loc.py`` (that script
only adds the tokenizer, the student-server HTTP models and the CLI), kept
here so it is unit-testable with a scripted model.

Per instance: a HISTORY-FREE export of ``base_commit`` from the local mirror
(``git archive | tar``, no ``.git``: a clone would carry the fix commit in its
future history, and ``bash``/``search`` could find it) is made once under
``work_root/<instance_id>/base`` and copied per arm, so the arms see the same
tree and an arm's edits never reach the next one. Each arm runs the existing
``run_episode`` loop with the default ``ToolRegistry``; the task is the issue
text plus a localisation instruction (``LOC_INSTRUCTION``). The gold patch,
test patch and hints are NEVER put into the agent's context: gold files are
parsed from ``Instance.patch`` only when the episode record is written.

Scoring (``episode_record`` / ``summarize``):

* gold files      ``diff --git a/X b/X`` headers of the gold patch (the a-path,
                  i.e. the file as it exists at ``base_commit``)
* read files      paths of ``read`` / ``zoom`` calls, normalised
* answered files  the last ``FILES:`` line of the final message (falls back to
                  an ``ANSWER:`` line); ``path::func`` / ``path:123`` suffixes
                  are stripped
* read_gold_any   any gold file was read
* answer_hit      answered ∩ gold ≠ ∅
* answer_exact    answered == gold (as sets)
* edit_*          edit exactness (``tools.EditAudit``): for every ``edit`` call,
                  does ``old`` occur verbatim in the file at that point
                  (``edit_old_verbatim``), exactly once (``edit_old_unique``), or
                  up to whitespace (``edit_old_ws_match``)? ``zoom`` is an
                  encoding, so this measures copying from reps vs guessing. The
                  summary pools every edit of the arm.

Overview arms (2026-09-22): an arm named ``<base>_ov`` (``compressed_ov``,
``full_text_ov``) runs the same episode with the ALWAYS-ON repo overview in the
opening user turn and the ``browse`` tool -- the repo tree AT ``base_commit``
from a per-instance tree cache (``bgkit2.agent.swe_trees``,
``overview.instance_overview_lookup``): the student reads the cached root
reps, a text model reads the root directory's listing (``run_episode``). The
system prompt gains the same overview/browse notes the agent rows were trained
with. The plain arms are unchanged, so earlier baselines stay comparable.
Records of an overview arm carry ``overview`` (the tree key, or None when the
instance has no cached tree -- the episode then runs WITHOUT one and says so),
``browse_calls`` and ``browsed_gold_dir`` (a browsed directory is an ancestor
of a gold file).
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import statistics
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from .runner import AgentTask, EpisodeResult, Model, TrajectoryLog, run_episode
from .swebench_lite import Instance, repo_source
from .taskset import QUESTION_SYSTEM
from .tools import EditAudit, ToolRegistry

LOC_INSTRUCTION = (
    "Find the file(s) and function(s) that must change to fix this issue. Do not edit "
    "anything. Finish with a line `FILES: path1, path2` listing the repo-relative paths "
    "of the files that must change."
)

_ANSWER_SENTENCE = "When you know the answer, reply with ONLY"
assert _ANSWER_SENTENCE in QUESTION_SYSTEM, "QUESTION_SYSTEM changed; update LOC_SYSTEM"
# The localisation eval's system prompt = the question system prompt the
# student was trained with, with only the final-answer format swapped.
LOC_SYSTEM = (
    QUESTION_SYSTEM.split(_ANSWER_SENTENCE)[0]
    + "When you know the answer, reply with the file list on the last line in the form "
    "`FILES: path1, path2` and no tool call."
)

READ_TOOLS = ("read", "zoom")
OV_SUFFIX = "_ov"


def base_arm(arm: str) -> str:
    """``compressed_ov`` -> ``compressed``: the model an arm runs."""
    return arm[: -len(OV_SUFFIX)] if arm.endswith(OV_SUFFIX) else arm


def uses_overview(arm: str) -> bool:
    return arm.endswith(OV_SUFFIX)


class MirrorUnavailableError(RuntimeError):
    """No local mirror, or the mirror does not (yet) contain ``base_commit``."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)\s*$", re.MULTILINE)


def gold_files(patch: str) -> list[str]:
    """Files touched by a unified diff, in order, from its ``diff --git`` headers."""
    out: list[str] = []
    for a, _b in _DIFF_HEADER.findall(patch or ""):
        if a not in out:
            out.append(a)
    return out


def norm_path(p: str) -> str:
    """Repo-relative form of a path the model wrote: quotes, ``./``, a leading
    ``/`` or ``/repo/`` and ``::qualname`` / ``:line`` suffixes removed."""
    p = str(p).strip().strip("`'\"()[]{}<>").strip()
    p = p.split("::", 1)[0]
    p = re.sub(r":\d+(?:[-:]\d+)?$", "", p)
    p = p.rstrip(".,;:")
    for pre in ("/repo/", "/testbed/"):
        if p.startswith(pre):
            p = p[len(pre) :]
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def read_paths(records) -> list[str]:
    """Normalised paths of every ``read`` / ``zoom`` call, first-seen order.

    ``records`` are ``TurnRecord`` objects or their dicts."""
    out: list[str] = []
    for r in records:
        calls = r["tool_calls"] if isinstance(r, dict) else r.tool_calls
        for c in calls:
            if c.get("name") in READ_TOOLS and isinstance((c.get("args") or {}).get("path"), str):
                p = norm_path(c["args"]["path"])
                if p and p not in out:
                    out.append(p)
    return out


def answered_files(text: str) -> tuple[list[str], str]:
    """(paths, form) from the final message; form is ``files``, ``answer`` or ``none``.

    The LAST ``FILES:`` line wins (a model may restate it); without one the last
    ``ANSWER:`` line is read the same way — the student's trained convention."""
    lines = (text or "").splitlines()
    form, body = "none", ""
    for tag, name in (("FILES:", "files"), ("ANSWER:", "answer")):
        hits = [ln for ln in lines if tag in ln.upper()]
        if hits:
            ln = hits[-1]
            body = ln[ln.upper().rindex(tag) + len(tag) :]
            form = name
            break
    out: list[str] = []
    for tok in re.split(r"[,\s]+", body):
        p = norm_path(tok)
        if p and ("/" in p or "." in p) and p not in out:
            out.append(p)
    return out, form


# ---------------------------------------------------------------------------
# Checkout
# ---------------------------------------------------------------------------


def _mirror(instance: Instance, repos_dir: str | Path) -> Path:
    src = repo_source(instance, repos_dir)
    if not Path(src).is_dir():
        raise MirrorUnavailableError(f"no local mirror of {instance.repo} under {repos_dir}")
    r = subprocess.run(
        ["git", "-C", src, "cat-file", "-e", f"{instance.base_commit}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        raise MirrorUnavailableError(
            f"mirror {src} lacks {instance.base_commit[:12]} (still cloning / not fetched?)"
        )
    return Path(src)


def export_base(instance: Instance, repos_dir: str | Path, work_root: str | Path) -> Path:
    """``work_root/<instance_id>/base``: a history-free export of ``base_commit``,
    made once and reused (a ``.export_ok`` marker records a complete export)."""
    inst_dir = Path(work_root) / instance.instance_id
    base, ok = inst_dir / "base", inst_dir / ".export_ok"
    if ok.exists() and base.is_dir():
        return base
    src = _mirror(instance, repos_dir)
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    r = subprocess.run(
        f"set -o pipefail; git -C {shlex.quote(str(src))} archive --format=tar "
        f"{shlex.quote(instance.base_commit)} | tar -x -C {shlex.quote(str(base))}",
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        shutil.rmtree(base, ignore_errors=True)
        raise RuntimeError(f"export of {instance.instance_id} failed: {r.stderr.strip()[:300]}")
    ok.touch()
    return base


def arm_checkout(base: Path, arm: str) -> Path:
    """A fresh copy of the export for one arm (edits cannot leak across arms)."""
    dst = base.parent / f"arm_{arm}"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(base, dst, symlinks=True)
    return dst


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


def browse_dirs(records) -> list[str]:
    """Directories the agent opened with ``browse`` (normalised, first-seen order)."""
    out: list[str] = []
    for r in records:
        calls = r["tool_calls"] if isinstance(r, dict) else r.tool_calls
        for c in calls:
            if c.get("name") == "browse":
                d = norm_path(str((c.get("args") or {}).get("dir", ""))).strip("/")
                d = "" if d == "." else d
                if d not in out:
                    out.append(d)
    return out


def loc_task(instance: Instance, repo_dir: str | Path) -> AgentTask:
    """The agent's task. Only the issue text: never the patch, tests or hints."""
    instruction = (
        f"Repository: {instance.repo}\n\nIssue:\n{instance.problem_statement.strip()}"
        f"\n\n{LOC_INSTRUCTION}"
    )
    return AgentTask(
        task_id=instance.instance_id,
        instruction=instruction,
        repo_dir=str(repo_dir),
        system_prompt=LOC_SYSTEM,
        summary=" ".join(instance.problem_statement.split())[:200],
        repo=instance.repo,
    )


def episode_record(
    instance: Instance,
    arm: str,
    res: EpisodeResult,
    overview_key: str | None = None,
    edits: EditAudit | None = None,
) -> dict:
    gold = gold_files(instance.patch)
    read = read_paths(res.records)
    final = res.final_message if res.stop_reason == "final" else ""
    answered, form = answered_files(final)
    gs, rs, ans = set(gold), set(read), set(answered)
    return {
        "instance_id": instance.instance_id,
        "repo": instance.repo,
        "arm": arm,
        "gold_files": gold,
        "read_files": read,
        "answered_files": answered,
        "answer_form": form,
        "read_gold_any": bool(gs & rs),
        "answer_hit": bool(gs & ans),
        "answer_exact": bool(ans) and ans == gs,
        "turns": res.turns,
        "decoder_tokens": res.decoder_tokens,
        "tool_result_tokens": res.tool_result_tokens,
        "tokens_saved": res.tokens_saved,
        "compactions": res.n_compactions,
        "stop": res.stop_reason,
        "wall_s": round(res.wall_s, 2),
        "final_message": final[-2000:],
        **(edits.summary() if edits is not None else {}),
        **(
            _overview_fields(gold, res, overview_key)
            if uses_overview(arm) or overview_key is not None
            else {}
        ),
    }


def _overview_fields(gold: list[str], res: EpisodeResult, key: str | None) -> dict:
    dirs = browse_dirs(res.records)
    return {
        "overview": key,
        "browse_calls": sum(
            c.get("name") == "browse"
            for r in res.records
            for c in (r["tool_calls"] if isinstance(r, dict) else r.tool_calls)
        ),
        "browsed_dirs": dirs,
        "browsed_gold_dir": any(
            d and any(g.startswith(d + "/") for g in gold) for d in dirs
        ),
    }


def error_record(instance: Instance, arm: str, stop: str, exc: Exception) -> dict:
    return {
        "instance_id": instance.instance_id,
        "repo": instance.repo,
        "arm": arm,
        "gold_files": gold_files(instance.patch),
        "stop": stop,
        "error": str(exc)[:300],
    }


def run_arm(
    instance: Instance,
    arm: str,
    base: Path,
    model: Model,
    log: TrajectoryLog | None = None,
    tools_kwargs: dict | None = None,
    keep_checkout: bool = False,
    overview: Callable | None = None,
    **episode_kwargs,
) -> dict:
    """One (arm, instance) episode on a fresh copy of the export -> record.

    ``overview``: an overview provider (``overview.instance_overview_lookup``)
    -- the episode opens with the instance's cached repo tree and gets
    ``browse``. Passed only for the ``_ov`` arms by ``run_instance``."""
    wd = arm_checkout(base, arm)
    try:
        tools = EditAudit(ToolRegistry(str(wd), **(tools_kwargs or {})))
        task = loc_task(instance, wd)
        key = None
        if overview is not None:
            ov = overview(task)
            if ov is not None:
                from .overview import with_overview_notes

                key = ov[0]
                browsable = hasattr(overview, "tree") and overview.tree(task) is not None
                task.system_prompt = with_overview_notes(task.system_prompt, browsable)
        res = run_episode(model, tools, task, log=log,
                          overview=overview if key is not None else None, **episode_kwargs)
        return episode_record(instance, arm, res, overview_key=key, edits=tools)
    finally:
        if not keep_checkout:
            shutil.rmtree(wd, ignore_errors=True)


def load_records(path: Path) -> dict[str, dict]:
    """Last record per instance from an ``episodes.jsonl``."""
    out: dict[str, dict] = {}
    if path.exists():
        for ln in path.open():
            if ln.strip():
                r = json.loads(ln)
                out[r["instance_id"]] = r
    return out


def load_done(path: Path) -> set[str]:
    """Instances with a scored record; errored / skipped ones are NOT done (a
    resume retries them)."""
    return {k for k, v in load_records(path).items() if v.get("stop") not in ("error", "skipped")}


def run_instance(
    instance: Instance,
    arms: list[str],
    make_model: Callable[[str], Model],
    repos_dir: str | Path,
    work_root: str | Path,
    out_dir: str | Path,
    done: dict[str, set[str]] | None = None,
    logs: dict[str, TrajectoryLog] | None = None,
    write: Callable[[str, dict], None] | None = None,
    keep_checkouts: bool = False,
    tools_kwargs: dict | None = None,
    overview: Callable | None = None,
    **episode_kwargs,
) -> list[dict]:
    """Every not-yet-done arm of one instance on ONE shared export.

    ``overview`` (a provider, see ``run_arm``) is handed to the ``_ov`` arms
    only; ``make_model`` receives the full arm name.

    ``write(arm, record)`` appends to that arm's ``episodes.jsonl`` (default:
    a plain append; the CLI passes a locked one). A missing mirror yields
    ``stop="skipped"`` records, which a later resume retries."""
    out = Path(out_dir)
    done = done or {}

    def _write(arm: str, rec: dict) -> None:
        if write is not None:
            write(arm, rec)
            return
        p = out / arm / "episodes.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    todo = [a for a in arms if instance.instance_id not in done.get(a, set())]
    if not todo:
        return []
    recs: list[dict] = []
    try:
        base = export_base(instance, repos_dir, work_root)
    except MirrorUnavailableError as exc:
        for arm in todo:
            recs.append(error_record(instance, arm, "skipped", exc))
            _write(arm, recs[-1])
        return recs
    except Exception as exc:  # one bad export must not end the run
        for arm in todo:
            recs.append(error_record(instance, arm, "error", exc))
            _write(arm, recs[-1])
        return recs
    try:
        for arm in todo:
            try:
                rec = run_arm(
                    instance,
                    arm,
                    base,
                    make_model(arm),
                    log=(logs or {}).get(arm),
                    tools_kwargs=tools_kwargs,
                    keep_checkout=keep_checkouts,
                    overview=overview if uses_overview(arm) else None,
                    **episode_kwargs,
                )
            except Exception as exc:  # server hiccup etc.: retried on resume
                rec = error_record(instance, arm, "error", exc)
            recs.append(rec)
            _write(arm, rec)
    finally:
        if not keep_checkouts:
            shutil.rmtree(Path(work_root) / instance.instance_id, ignore_errors=True)
    return recs


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _mean(xs) -> float | None:
    xs = list(xs)
    return round(statistics.mean(xs), 4) if xs else None


def _edit_summary(ok: list[dict]) -> dict:
    """Edit exactness pooled over every edit of the arm (episodes without edits add none)."""
    es = [e for r in ok for e in r.get("edits") or []]
    n = sum(int(r.get("edit_calls", 0)) for r in ok)

    def frac(k: str) -> float | None:
        return round(sum(bool(e.get(k)) for e in es) / len(es), 4) if es else None

    return {"edit_calls": n, "episodes_with_edit": sum(bool(r.get("edit_calls")) for r in ok),
            "edit_old_verbatim": frac("old_verbatim"), "edit_old_unique": frac("old_unique"),
            "edit_old_ws_match": frac("old_ws_match")}


def summarize(records: list[dict]) -> dict:
    """Per-arm metrics over scored episodes (errors / skips are counted, not
    scored as zeros — they say nothing about the model)."""
    ok = [r for r in records if r.get("stop") not in ("error", "skipped")]
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        by_repo[r["repo"]].append(r)
    return {
        "n": len(ok),
        "errors": sum(r.get("stop") == "error" for r in records),
        "skipped": sum(r.get("stop") == "skipped" for r in records),
        "read_gold_any": _mean(float(r["read_gold_any"]) for r in ok),
        "answer_hit": _mean(float(r["answer_hit"]) for r in ok),
        "answer_exact": _mean(float(r["answer_exact"]) for r in ok),
        "answered": _mean(float(bool(r["answered_files"])) for r in ok),
        "answer_form": dict(Counter(r["answer_form"] for r in ok)),
        "stop": dict(Counter(r["stop"] for r in ok)),
        "loops": sum(r["stop"] == "loop" for r in ok),
        "turns": _mean(r["turns"] for r in ok),
        "decoder_tokens": _mean(r["decoder_tokens"] for r in ok),
        "tool_result_tokens": _mean(r["tool_result_tokens"] for r in ok),
        "tokens_saved": _mean(r["tokens_saved"] for r in ok),
        "wall_s": _mean(r["wall_s"] for r in ok),
        **_edit_summary(ok),
        **(
            {
                "with_overview": _mean(float(r.get("overview") is not None) for r in ov),
                "browse_calls": _mean(r.get("browse_calls", 0) for r in ov),
                "browsed_gold_dir": _mean(float(r.get("browsed_gold_dir", False)) for r in ov),
            }
            if (ov := [r for r in ok if "overview" in r])
            else {}
        ),
        "by_repo": {
            repo: {
                "n": len(rs),
                "read_gold_any": _mean(float(r["read_gold_any"]) for r in rs),
                "answer_hit": _mean(float(r["answer_hit"]) for r in rs),
            }
            for repo, rs in sorted(by_repo.items())
        },
    }
