#!/usr/bin/env python
"""Convert third-party agentic trajectory dumps into the message-list parquet
shape `scripts/bgkit2_build_tasks.py trajectory_chunks` consumes (the
SWE-Zero shape: one row per trajectory with `trajectory_id`, `repo` (a split
key) and `trajectory` = list of {role, content, tool_calls}).

Sources with that shape already (SWE-Zero, nebius SWE-rebench-openhands,
nvidia Open-SWE-Traces) need no conversion. Adapters here:

  openresearcher   OpenResearcher/OpenResearcher-Dataset — Harmony-format rows
                   (system / developer / user / assistant{channel, recipient} /
                   tool). Assistant "analysis" text becomes the assistant
                   content; an assistant message addressed to `browser.<op>`
                   becomes a tool call with the JSON text as arguments; tool
                   rows become tool messages; the `final` channel is the answer.
  agenttrove       open-thoughts/AgentTrove — terminus-2 ShareGPT rows
                   (`conversations` of {from, value} with tool calls inline).

Usage:
  scripts/bgkit2_convert_trajectories.py openresearcher \
      --input '/mnt/external/bgkit-data/trajectories_ext/openresearcher/*/*.parquet' \
      --output-dir /mnt/external/bgkit-data/trajectories_ext/openresearcher_conv
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _text(content) -> str:
    """Harmony content: a list of parts with `text`, or a plain string."""
    if isinstance(content, str):
        return content
    out = []
    for part in content or []:
        if isinstance(part, dict) and part.get("text"):
            out.append(str(part["text"]))
    return "\n".join(out)


def convert_openresearcher(msgs: list[dict]) -> list[dict]:
    """Harmony messages -> OpenAI-style role/content/tool_calls list."""
    out: list[dict] = []
    pending = ""  # assistant "analysis" text awaiting a tool call or answer
    for m in msgs:
        role, text = m.get("role"), _text(m.get("content"))
        if role == "system":
            continue  # model-identity boilerplate; the developer message is the real system prompt
        if role == "developer":
            out.append({"role": "system", "content": text})
        elif role == "user":
            out.append({"role": "user", "content": text})
        elif role == "assistant":
            rcpt = m.get("recipient") or ""
            if rcpt:
                try:
                    args = json.loads(text)
                except json.JSONDecodeError:
                    args = {"input": text}
                if not isinstance(args, dict):
                    args = {"input": args}
                out.append(
                    {
                        "role": "assistant",
                        "content": pending.strip(),
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": rcpt.replace(".", "_"), "arguments": args},
                            }
                        ],
                    }
                )
                pending = ""
            elif m.get("channel") == "final":
                out.append({"role": "assistant", "content": (pending + "\n" + text).strip()})
                pending = ""
            else:
                pending = (pending + "\n" + text) if pending else text
        elif role == "tool":
            out.append({"role": "tool", "content": text})
    if pending.strip():
        out.append({"role": "assistant", "content": pending.strip()})
    return out


_SHAREGPT_ROLE = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "tool": "tool",
    "observation": "tool",
    "function_call": "assistant",
}


def convert_sharegpt(conv: list[dict]) -> list[dict]:
    """ShareGPT {from, value} turns -> role/content list. Tool calls that the
    source keeps inline in the assistant text stay inline (the decoder learns
    the scaffold's own call syntax); `observation`/`tool` turns become tool
    messages."""
    out: list[dict] = []
    for t in conv:
        role = _SHAREGPT_ROLE.get(str(t.get("from", "")).lower())
        if role is None:
            continue
        val = t.get("value")
        if isinstance(val, dict):
            val = json.dumps(val)
        out.append({"role": role, "content": str(val or "")})
    return out


def rows_openresearcher(path: str):
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        for r in pf.read_row_group(
            rg, columns=["qid", "question", "messages", "status"]
        ).to_pylist():
            if r.get("status") not in (None, "success"):
                continue
            traj = convert_openresearcher(r["messages"])
            if len(traj) < 6:
                continue
            yield {
                "trajectory_id": f"openresearcher/{Path(path).parent.name}/{r['qid']}",
                "repo": f"openresearcher_q{r['qid']}",
                "trajectory": traj,
            }


def rows_agenttrove(path: str):
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    conv_col = "conversations" if "conversations" in names else "messages"
    src_col = next(
        (c for c in ("original_source", "source", "dataset", "origin") if c in names), None
    )
    id_col = next((c for c in ("run_id", "id", "uuid", "conversation_id") if c in names), None)
    for rg in range(pf.num_row_groups):
        cols = [c for c in (conv_col, src_col, id_col) if c]
        for i, r in enumerate(pf.read_row_group(rg, columns=cols).to_pylist()):
            conv = r[conv_col]
            if isinstance(conv, str):
                conv = json.loads(conv)
            if not conv:
                continue
            traj = (
                convert_sharegpt(conv)
                if conv and "from" in conv[0]
                else [{"role": m.get("role"), "content": str(m.get("content") or "")} for m in conv]
            )
            if len(traj) < 6:
                continue
            src = str(r.get(src_col, "agenttrove")) if src_col else "agenttrove"
            tid = (
                str(r.get(id_col, f"{Path(path).stem}:{rg}:{i}"))
                if id_col
                else f"{Path(path).stem}:{rg}:{i}"
            )
            yield {"trajectory_id": f"agenttrove/{tid}", "repo": src, "trajectory": traj}


ADAPTERS = {"openresearcher": rows_openresearcher, "agenttrove": rows_agenttrove}


def write(rows: list[dict], out: Path) -> int:
    if not rows:
        return 0
    table = pa.table(
        {
            "trajectory_id": [r["trajectory_id"] for r in rows],
            "repo": [r["repo"] for r in rows],
            "trajectory": [json.dumps(r["trajectory"]) for r in rows],
        }
    )
    pq.write_table(table, out)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=sorted(ADAPTERS))
    ap.add_argument("--input", required=True, help="glob of source parquet files")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-turns", type=int, default=6)
    args = ap.parse_args()
    files = sorted(glob.glob(args.input))
    if not files:
        raise SystemExit(f"no files match {args.input}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for f in files:
        rows = [r for r in ADAPTERS[args.source](f) if len(r["trajectory"]) >= args.min_turns]
        out = out_dir / (Path(f).parent.name + "_" + Path(f).name)
        n = write(rows, out)
        total += n
        print(f"{f} -> {out}: {n} trajectories", flush=True)
    print(json.dumps({"source": args.source, "files": len(files), "trajectories": total}))


if __name__ == "__main__":
    main()
