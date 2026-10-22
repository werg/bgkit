#!/usr/bin/env python
"""Build the ``tool_digest`` store: query-focused compression supervision taken
from the teacher runs' digest sub-agent.

Every compressible tool result in a teacher episode was digested by a
sub-agent under the teacher's compression prompt (the "purpose"), and the row
logs the triple (purpose, RAW result, digest) per slot (meta.digests aligned
with the context list). Those triples are exactly the supervision the encoder
needs: the raw result goes into the slot as reps, the purpose is the
compression prompt, and the digest is the target the decoder must write from
the reps — the sub-agent's job, done by the encoder (user, 2026-09-14).

Framing ``tool_response``: user asks for the digest of a call, the assistant
"makes" the call (tool_name / tool_args recovered from the purpose), the tool
result is the slot, the assistant's reply is the digest.

Filters: empty digests, leaked think-preambles ("Need digest of ...", cleaned
or dropped), digests
that are the raw text (passthrough), digests longer than 90% of the raw text,
raw results under --min-raw-chars, duplicates by (purpose, raw). Splits by the
episode's task id (2% eval).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bgkit2.data.tasks import write_task_store

PREAMBLE = re.compile(
    r"^(need (a )?digest|provide|raw output (shows|has|contains|is)|so digest|digest:|summary:|"
    r"purpose:|the purpose|we need|let me|i need)",
    re.IGNORECASE,
)
SENT = re.compile(r"(?<=[.!?:])\s+(?=[A-Z\-`./])")


def clean_digest(dg: str) -> str:
    """K2's fast-think preamble sometimes leaks into the digest ("Need digest
    of search results. Provide lines. bottle.py:83: ..."): drop leading
    sentences that talk about the task instead of the content."""
    dg = dg.strip()
    parts = SENT.split(dg, maxsplit=6)
    while len(parts) > 1 and PREAMBLE.match(parts[0]):
        parts = parts[1:]
    out = " ".join(parts).strip() if len(parts) > 1 else parts[0]
    # a lone "So digest: ..." prefix inside the first surviving sentence
    out = re.sub(r"^(so )?digest:\s*", "", out, flags=re.IGNORECASE)
    return out.strip()


def parse_purpose(p: str) -> tuple[str, dict, str]:
    """ToolRegistry._prompt: 'read a/b.py: focus' / 'search q: focus' /
    'docs q: focus' / 'bash: cmd' -> (tool, args, focus)."""
    p = p.strip()
    if p.startswith("bash:"):
        return "bash", {"cmd": p[5:].strip()}, ""
    head, _, rest = p.partition(" ")
    rest = rest.strip()
    focus = ""
    if head in ("read", "search", "docs") and ": " in rest:
        rest, focus = rest.split(": ", 1)
        rest, focus = rest.strip(), focus.strip()
    if head == "read":
        args: dict = {"path": rest}
    elif head in ("search", "docs"):
        args = {"query": rest}
    else:
        head, args = "read", {"path": rest or p}
    if focus:
        args["focus"] = focus
    return head, args, focus


def instruction_for(tool: str, args: dict, focus: str) -> str:
    what = focus or {
        "read": f"the contents of {args.get('path', '')}",
        "search": f"the matches for {args.get('query', '')!r}",
        "docs": f"the documentation matches for {args.get('query', '')!r}",
        "bash": f"the output of `{args.get('cmd', '')}`",
    }.get(tool, "the result")
    return (
        f"Give me a compact digest of {what}: keep every exact identifier, path, "
        "line number and value that matters, drop the rest."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="teacher run dirs (trajectories.jsonl inside)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-raw-chars", type=int, default=200)
    ap.add_argument("--max-raw-chars", type=int, default=40000)
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if Path(args.out).exists() and not args.overwrite:
        sys.exit(f"{args.out} exists; pass --overwrite")
    rows: list[dict] = []
    seen: set[str] = set()
    st: Counter = Counter()
    for run in args.runs:
        path = Path(run) / "trajectories.jsonl"
        for ln in path.open():
            if not ln.startswith('{"kind": "row"'):
                continue
            r = json.loads(ln)
            meta = r.get("meta") or {}
            digests = meta.get("digests") or []
            ctx = json.loads(r["context"]) if r.get("context") else []
            prompts = json.loads(r["prompt"]) if r.get("prompt") else []
            if not (len(ctx) == len(prompts) == len(digests)):
                st["misaligned_rows"] += 1
                continue
            tid = str(meta.get("task_id", ""))
            for raw, purpose, dg in zip(ctx, prompts, digests, strict=True):
                st["slots"] += 1
                dg = (dg or "").strip()
                if not dg:
                    st["no_digest"] += 1
                    continue
                if len(raw) < args.min_raw_chars or len(raw) > args.max_raw_chars:
                    st["raw_size"] += 1
                    continue
                if dg == raw.strip() or dg == raw[:2000].strip():
                    st["passthrough"] += 1
                    continue
                if len(dg) > 0.9 * len(raw):
                    st["digest_too_long"] += 1
                    continue
                if PREAMBLE.match(dg):
                    st["preamble_cleaned"] += 1
                    dg = clean_digest(dg)
                    if len(dg) < 40 or PREAMBLE.match(dg):
                        st["echo_dropped"] += 1
                        continue
                key = hashlib.sha1((purpose + "\n" + raw).encode()).hexdigest()
                if key in seen:
                    st["dup"] += 1
                    continue
                seen.add(key)
                tool, targs, focus = parse_purpose(purpose)
                h = int(hashlib.sha1(tid.encode()).hexdigest(), 16) % 10_000
                rows.append(
                    {
                        "family": "tool_digest",
                        "framing": "tool_response",
                        "context": raw,
                        "prompt": f"Compress the {tool} result to digest: {focus or purpose}\n",
                        "instruction": instruction_for(tool, targs, focus),
                        "tool_name": tool,
                        "tool_args": targs,
                        "target": dg,
                        "split": "eval" if h < args.eval_frac * 10_000 else "train",
                        "meta": {
                            "task_id": tid,
                            "run": Path(run).name,
                            "purpose": purpose,
                            "raw_chars": len(raw),
                            "digest_chars": len(dg),
                        },
                    }
                )
                st[f"kept_{tool}"] += 1
    if Path(args.out).exists():
        Path(args.out).unlink()
    n = write_task_store(args.out, rows)
    print(
        json.dumps(
            {
                "out": args.out,
                "rows": n,
                "eval": sum(r["split"] == "eval" for r in rows),
                "stats": st,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
