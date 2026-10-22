"""SWE-bench Lite: instances, repo checkout, patch extraction, scoring stub.

* ``load_instances``: the HF dataset ``princeton-nlp/SWE-bench_Lite`` (needs
  network + ``datasets``) or a local JSONL of the same records (offline);
  ``save_instances`` writes that JSONL.
* ``prepare_repo``: ``git clone`` from a local mirror (``repos_dir``) or GitHub
  and ``git checkout base_commit``.
* ``extract_patch``: the working tree's diff including new files.
* ``score``: builds the official harness command
  (``swebench.harness.run_evaluation``) and does NOT run it — the harness needs
  docker and the per-instance images, i.e. the GPU host's docker setup, which
  this CPU skeleton must not touch.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .runner import DEFAULT_SYSTEM, AgentTask

DATASET = "princeton-nlp/SWE-bench_Lite"


@dataclass(frozen=True)
class Instance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str = ""
    test_patch: str = ""
    fail_to_pass: tuple[str, ...] = ()
    pass_to_pass: tuple[str, ...] = ()
    version: str = ""
    environment_setup_commit: str = ""
    hints_text: str = ""

    @classmethod
    def from_record(cls, d: dict) -> Instance:
        def _list(v) -> tuple[str, ...]:
            if isinstance(v, str):
                v = json.loads(v) if v.strip().startswith("[") else [v]
            return tuple(str(x) for x in (v or []))

        return cls(
            instance_id=str(d["instance_id"]),
            repo=str(d["repo"]),
            base_commit=str(d["base_commit"]),
            problem_statement=str(d.get("problem_statement", "")),
            patch=str(d.get("patch", "")),
            test_patch=str(d.get("test_patch", "")),
            fail_to_pass=_list(d.get("FAIL_TO_PASS", d.get("fail_to_pass"))),
            pass_to_pass=_list(d.get("PASS_TO_PASS", d.get("pass_to_pass"))),
            version=str(d.get("version", "")),
            environment_setup_commit=str(d.get("environment_setup_commit", "")),
            hints_text=str(d.get("hints_text", "")),
        )

    def to_record(self) -> dict:
        d = asdict(self)
        d["FAIL_TO_PASS"] = list(d.pop("fail_to_pass"))
        d["PASS_TO_PASS"] = list(d.pop("pass_to_pass"))
        return d


def load_instances(
    source: str | Path = DATASET,
    split: str = "test",
    limit: int | None = None,
    instance_ids: set[str] | None = None,
) -> list[Instance]:
    """Instances from a local JSONL path (offline) or the HF dataset name."""
    p = Path(source)
    if p.exists():
        with p.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]

            records = list(load_dataset(str(source), split=split))
        except Exception as exc:  # network / missing package
            raise RuntimeError(
                f"could not load {source!r} ({exc}); pass a local JSONL written by "
                "save_instances() instead"
            ) from exc
    out = [Instance.from_record(r) for r in records]
    if instance_ids is not None:
        out = [i for i in out if i.instance_id in instance_ids]
    return out[:limit] if limit is not None else out


def save_instances(instances: list[Instance], path: str | Path) -> None:
    with Path(path).open("w") as f:
        for i in instances:
            f.write(json.dumps(i.to_record()) + "\n")


def repo_source(instance: Instance, repos_dir: str | Path | None = None) -> str:
    """A local mirror (``repos_dir/owner__name`` or ``repos_dir/owner/name``)
    when present, else the GitHub URL."""
    if repos_dir is not None:
        base = Path(repos_dir)
        for cand in (base / instance.repo.replace("/", "__"), base / instance.repo):
            if cand.is_dir():
                return str(cand)
    return f"https://github.com/{instance.repo}.git"


def _git(args: list[str], cwd: Path | None, timeout: float) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def prepare_repo(
    instance: Instance,
    workdir: str | Path,
    repos_dir: str | Path | None = None,
    source: str | None = None,
    timeout: float = 600.0,
) -> Path:
    """Clone ``source`` (default: ``repo_source``) into
    ``workdir/<instance_id>`` and check out ``base_commit``; returns the path.
    An existing checkout is reset to the base commit."""
    src = source or repo_source(instance, repos_dir)
    dest = Path(workdir) / instance.instance_id
    if not (dest / ".git").is_dir():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _git(["clone", "--quiet", src, str(dest)], None, timeout)
    else:
        _git(["reset", "--hard", "--quiet"], dest, timeout)
        _git(["clean", "-fdq"], dest, timeout)
    _git(["checkout", "--quiet", instance.base_commit], dest, timeout)
    return dest


def extract_patch(work_dir: str | Path, timeout: float = 60.0) -> str:
    """``git diff`` of the working tree against HEAD, new files included."""
    wd = Path(work_dir)
    _git(["add", "--intent-to-add", "--all"], wd, timeout)
    return _git(["diff", "--no-color", "--no-ext-diff"], wd, timeout)


def task_from_instance(instance: Instance, work_dir: str | Path) -> AgentTask:
    instruction = (
        f"Repository: {instance.repo}\n\nResolve the following issue by editing the "
        f"repository so that the described behaviour is fixed.\n\n{instance.problem_statement}"
    )
    return AgentTask(
        task_id=instance.instance_id,
        instruction=instruction,
        repo_dir=str(work_dir),
        system_prompt=DEFAULT_SYSTEM,
        summary=" ".join(instance.problem_statement.split())[:200],
    )


def prediction_record(instance: Instance, patch: str, model_name: str) -> dict:
    return {
        "instance_id": instance.instance_id,
        "model_name_or_path": model_name,
        "model_patch": patch,
    }


def write_predictions(records: list[dict], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def score(
    predictions_path: str | Path,
    run_id: str,
    instance_ids: list[str] | None = None,
    dataset: str = DATASET,
    split: str = "test",
    max_workers: int = 4,
) -> list[str]:
    """The official harness command for ``predictions_path`` — RETURNED, not run.

    Running it (``pip install swebench``, docker available, ~1 GB+ of images per
    repo) writes ``logs/run_evaluation/<run_id>/...`` and a
    ``<model>.<run_id>.json`` report with ``resolved_ids``; the resolve rate is
    ``len(resolved_ids) / len(instance_ids)``. Report it beside
    ``EpisodeResult.decoder_tokens`` — the harness claim is resolve rate at an
    equal decoder-token budget.
    """
    cmd = [
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset,
        "--split",
        split,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
    ]
    if instance_ids:
        cmd += ["--instance_ids", *instance_ids]
    return cmd
