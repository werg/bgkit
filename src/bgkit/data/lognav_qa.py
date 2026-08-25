"""Log-needle QA generation over raw log files (Family B, wide-net tool results).

Generates (question, answer, window-span) samples from LogHub-style raw logs
for the capability-packaging plan (`plans/capability_packaging_2026_08_20.md`
§5). Also seeds the self-built long-log QA benchmark from
`docs/05_benchmark_targets.md` §2 — as of mid-2026 no long-context log-QA
benchmark exists.

Question types
--------------
- ``first_error``    — quote the first ERROR/FATAL-severity line in the window.
- ``error_absent``   — windows with no error lines become explicit
  "not present" negatives (the model must trust a null answer from a
  compressed read).
- ``needle_token``   — quote the unique line mentioning a rare identifier
  token (block ids, hex ids, long numbers).
- ``count_keyword``  — how many lines mention a token (aggregation;
  definitionally compression-hostile, emitted only when ``include_counts``
  and flagged so evals can report it as an honest-limits slice).

Samples reference windows by ``(path, line_start, line_end)`` span, not by
inline text — a 1M-token window duplicated per sample would explode storage.
``materialize_window`` resolves a span back to text.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from bgkit.data.blob_format import render_header

ERROR_TOKENS = {"ERROR", "FATAL", "SEVERE", "CRITICAL", "FAIL", "FAILED", "FAILURE"}
_ID_TOKEN_RE = re.compile(r"^[\w.\-:/]*\d[\w.\-:/]*$")


@dataclass(frozen=True)
class LogQASample:
    qtype: str
    question: str
    answer: str
    dataset: str
    path: str
    line_start: int  # inclusive, 0-based
    line_end: int  # exclusive
    blob_header: str
    source_ref: str
    is_aggregation: bool = False


def line_severity_is_error(line: str) -> bool:
    return any(tok in ERROR_TOKENS for tok in line.split())


def rare_id_tokens(lines: list[str], *, min_len: int = 6) -> list[str]:
    """Identifier-shaped tokens (contain a digit) occurring exactly once."""
    counts = Counter(
        tok
        for line in lines
        for tok in line.split()
        if len(tok) >= min_len and _ID_TOKEN_RE.match(tok)
    )
    return [tok for tok, c in counts.items() if c == 1]


def iter_windows(
    lines: list[str], *, window_chars: int, stride_fraction: float = 1.0
) -> list[tuple[int, int]]:
    """Contiguous line spans of ~``window_chars`` characters each."""
    spans: list[tuple[int, int]] = []
    start = 0
    n = len(lines)
    while start < n:
        size = 0
        end = start
        while end < n and size < window_chars:
            size += len(lines[end]) + 1
            end += 1
        if end > start:
            spans.append((start, end))
        if end >= n:
            break
        step = max(1, int((end - start) * stride_fraction))
        start += step
    return spans


def iter_error_windows(
    lines: list[str], *, window_chars: int, max_windows: int, rng: random.Random
) -> list[tuple[int, int]]:
    """Window spans centered on error-severity lines.

    Fixes the first_error scarcity observed in the 2026-08-21 build (most
    log windows carry no error line -> 35 first_error vs 925 error_absent):
    pick error lines uniformly, then cut a window around each so the error
    is at a random position inside it.
    """
    error_idx = [i for i, ln in enumerate(lines) if line_severity_is_error(ln)]
    if not error_idx:
        return []
    rng.shuffle(error_idx)
    spans: list[tuple[int, int]] = []
    used: set[int] = set()
    for ei in error_idx:
        if len(spans) >= max_windows:
            break
        if ei in used:
            continue
        # random offset of the error inside the window
        before_budget = int(window_chars * rng.uniform(0.1, 0.9))
        start = ei
        size = len(lines[ei]) + 1
        while start > 0 and size < before_budget:
            start -= 1
            size += len(lines[start]) + 1
        end = ei + 1
        while end < len(lines) and size < window_chars:
            size += len(lines[end]) + 1
            end += 1
        spans.append((start, end))
        used.update(range(start, end))
    return spans


def generate_window_samples(
    lines: list[str],
    span: tuple[int, int],
    *,
    dataset: str,
    path: str,
    rng: random.Random,
    max_needles: int = 3,
    include_counts: bool = False,
) -> list[LogQASample]:
    start, end = span
    window = lines[start:end]
    n_lines = end - start
    samples: list[LogQASample] = []

    def make(qtype: str, question: str, answer: str, *, aggregation: bool = False) -> LogQASample:
        return LogQASample(
            qtype=qtype,
            question=question,
            answer=answer,
            dataset=dataset,
            path=path,
            line_start=start,
            line_end=end,
            blob_header=render_header(
                "tool", source=Path(path).name, stats=f"{n_lines} lines", query=question
            ),
            source_ref=f"log:{dataset}:{Path(path).name}:{start}-{end}",
            is_aggregation=aggregation,
        )

    error_lines = [ln for ln in window if line_severity_is_error(ln)]
    if error_lines:
        q = "Quote the first error-severity line in this log."
        samples.append(make("first_error", q, error_lines[0].strip()))
    else:
        q = "Is there any error-severity line in this log? If so quote it."
        samples.append(make("error_absent", q, "No error-severity lines are present."))

    needles = rare_id_tokens(window)
    rng.shuffle(needles)
    for tok in needles[:max_needles]:
        matching = [ln for ln in window if tok in ln.split()]
        if len(matching) != 1:
            continue
        q = f"Quote the log line that mentions {tok}."
        samples.append(make("needle_token", q, matching[0].strip()))

    if include_counts:
        countable = [
            (tok, c)
            for tok, c in Counter(t for ln in window for t in ln.split()).items()
            if 2 <= c <= 20 and len(tok) >= 4
        ]
        if countable:
            tok, c = rng.choice(countable)
            q = f"How many lines in this log mention {tok}?"
            samples.append(make("count_keyword", q, str(c), aggregation=True))

    return samples


def generate_from_file(
    log_path: str | Path,
    *,
    dataset: str,
    window_chars: int,
    seed: int = 17,
    max_windows: int | None = None,
    include_counts: bool = False,
) -> list[dict]:
    """Generate samples for one raw log file. Returns JSON-ready dicts."""
    log_path = Path(log_path)
    lines = log_path.read_text(errors="replace").splitlines()
    rng = random.Random(seed)
    spans = iter_windows(lines, window_chars=window_chars)
    if max_windows is not None:
        spans = spans[:max_windows]
    out: list[dict] = []
    for span in spans:
        for s in generate_window_samples(
            lines,
            span,
            dataset=dataset,
            path=str(log_path),
            rng=rng,
            include_counts=include_counts,
        ):
            out.append(asdict(s))
    return out


def materialize_window(sample: dict) -> str:
    """Resolve a sample's span back to the raw window text."""
    lines = Path(sample["path"]).read_text(errors="replace").splitlines()
    return "\n".join(lines[sample["line_start"] : sample["line_end"]])
