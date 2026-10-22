#!/usr/bin/env python
"""Re-render an agent ``tool_slots`` store onto the CURRENT harness semantics (CPU, no teacher).

Built for ``agent_tool_slots_v7`` -> ``agent_tool_slots_v8`` (2026-09-22, user: exact-quote tools go
through bgkit too; mid-training builds the capability of predicting verbatim lines from reps, so
no training row carries zoom's ``exact``):

* every INLINE ``zoom`` result in a row's history becomes an ENCODED zoom slot: the slot text is the
  lines the row has (the numbered range the old verbatim zoom returned -- the whole file is not in
  the row), the prompt ``zoom <path> lines <a>-<b>[: <focus>]`` (``tools.zoom_prompt``; ``b`` = the
  last line the result shows, as the harness clips it; the focus is the call's own, else the row's
  synthesised purpose when its prompt list carries one -- v7 has none), digest None;
* ``zoom`` calls (history and target) lose any ``exact`` argument;
* ``meta.tools`` -> the current harness spec of every tool the row offered (``ToolRegistry``
  defaults + ``browse``); the system prompt's old zoom sentence -> the current ``QUESTION_SYSTEM``
  one; the old ``REPEAT_NOTICE`` wording in tool results -> the current one;
* a ``chat`` row (no slot yet) gets the tool list in its system message, as
  ``Conversation.to_task_row`` writes it now (v7's chat rows had none); a chat row whose history
  gains a zoom slot becomes a ``tool_slots`` row.

Every other slot / payload (cached overview and browse nodes, compaction spines, read / search /
bash slots) is untouched, in place. Every output row is rendered through the trainer's
``ChatRenderer`` and its sentinel / context / prompt / digest counts are asserted. Streams the
store (pyarrow batches) and writes .parquet + .arrow.

    .venv/bin/python scripts/bgkit2_rerender_agent_zoom.py \\
        --store $T/agent_tool_slots_v7.parquet --out $T/agent_tool_slots_v8.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bgkit2.agent.context import system_with_tool_list
from bgkit2.agent.runner import REPEAT_NOTICE
from bgkit2.agent.taskset import QUESTION_SYSTEM
from bgkit2.agent.tools import ToolRegistry, zoom_prompt
from bgkit2.data.templates import SPLICE_SENTINEL

OLD_ZOOM_SENTENCE = ("Results other than `zoom` may be summarised for you; use `zoom` for exact "
                     "lines. ")
NEW_ZOOM_SENTENCE = ("Results may be summarised for you; use `zoom` on a line range to look "
                     "closely. ")
OLD_REPEAT = "Do not repeat calls: use `zoom` for exact lines, "
NEW_REPEAT = "Do not repeat calls: `zoom` on a line range, "
_LINE = re.compile(r"^(\d+): ", re.M)
TOOL_LIST = "\nList of tools: ["

assert NEW_ZOOM_SENTENCE in QUESTION_SYSTEM, "QUESTION_SYSTEM changed; update NEW_ZOOM_SENTENCE"
assert NEW_REPEAT in REPEAT_NOTICE, "REPEAT_NOTICE changed; update NEW_REPEAT"


def current_specs() -> dict[str, dict]:
    reg = ToolRegistry("/tmp")
    reg.attach_tree(None)  # registers the browse spec only
    return {s["function"]["name"]: s for s in reg.specs()}


def _args(call: dict) -> dict:
    fn = call.get("function", call)
    a = fn.get("arguments") or {}
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except json.JSONDecodeError:
            a = {}
    return a if isinstance(a, dict) else {}


def _strip_exact(calls: list[dict], st: Counter) -> None:
    for c in calls:
        fn = c.get("function", c)
        if fn.get("name") == "zoom" and isinstance(fn.get("arguments"), dict) \
                and "exact" in fn["arguments"]:
            fn["arguments"].pop("exact")
            st["zoom_exact_dropped"] += 1


def rerender_row(r: dict, specs: dict[str, dict], st: Counter) -> dict:
    """The row on the current semantics (see the module docstring). Raises ValueError on a row
    whose slot bookkeeping does not add up."""
    msgs = json.loads(r["instruction"])
    ctx = json.loads(r["context"]) if r["context"] else []
    prompts = json.loads(r["prompt"]) if r["prompt"] else []
    raw_meta = r["meta"]
    meta = json.loads(raw_meta) if isinstance(raw_meta, str) and raw_meta else dict(raw_meta or {})
    digests = list(meta.get("digests") or [None] * len(ctx))
    if not (len(ctx) == len(prompts) == len(digests)):
        raise ValueError(f"slot lists disagree: {len(ctx)} / {len(prompts)} / {len(digests)}")
    was_chat = r["framing"] == "chat"
    new_ctx, new_pr, new_dg = [], [], []
    k = 0
    pending: list[dict] = []
    for m in msgs:
        content = m.get("content")
        if m.get("role") == "system" and isinstance(content, str):
            base = content.split(TOOL_LIST, 1)[0]
            if OLD_ZOOM_SENTENCE in base:
                base = base.replace(OLD_ZOOM_SENTENCE, NEW_ZOOM_SENTENCE)
                st["system_zoom_sentence_replaced"] += 1
            m["content"] = base
        if m.get("role") == "assistant":
            pending = list(m.get("tool_calls") or [])
            _strip_exact(pending, st)
        n = str(content or "").count(SPLICE_SENTINEL)
        if m.get("role") == "tool":
            call = pending.pop(0) if pending else {}
            if isinstance(content, str) and OLD_REPEAT in content:
                m["content"] = content = content.replace(OLD_REPEAT, NEW_REPEAT)
                st["repeat_notice_replaced"] += 1
            name = call.get("function", call).get("name") if call else None
            if name == "zoom" and n == 0:
                a = _args(call)
                shown = [int(x) for x in _LINE.findall(content or "")]
                if shown and (content or "").startswith(f"{shown[0]}: "):
                    # an inline result has no prompt entry, so no synthesised purpose: the
                    # call's own focus (v7's zoom calls carry none)
                    focus = str(a.get("focus") or "")
                    new_ctx.append(content)
                    new_pr.append(zoom_prompt(str(a.get("path", "")).strip(),
                                              int(a.get("start", shown[0])), shown[-1], focus))
                    new_dg.append(None)
                    m["content"] = SPLICE_SENTINEL
                    st["zoom_slotted"] += 1
                    continue
                st["zoom_kept_inline"] += 1  # an error / empty-range result
        for _ in range(n):
            if k >= len(ctx):
                raise ValueError("more sentinels than contexts")
            new_ctx.append(ctx[k])
            new_pr.append(prompts[k])
            new_dg.append(digests[k])
            k += 1
    if k != len(ctx):
        raise ValueError(f"{len(ctx)} contexts but {k} sentinels")
    tgt = r["target"]
    if isinstance(tgt, str) and tgt.lstrip().startswith("["):
        tm = json.loads(tgt)
        for m in tm:
            _strip_exact(list(m.get("tool_calls") or []), st)
        tgt = json.dumps(tm)
    old_tools = meta.get("tools") or []
    names = [t["function"]["name"] if isinstance(t, dict) else str(t) for t in old_tools]
    missing = [nm for nm in names if nm not in specs]
    if missing:
        raise ValueError(f"no current spec for {missing}")
    tools = [specs[nm] for nm in names]
    meta["tools"] = tools
    meta["n_slots"] = len(new_ctx)
    meta["digests"] = new_dg if new_ctx else []
    meta["rerendered_by"] = "scripts/bgkit2_rerender_agent_zoom.py"
    if new_ctx:
        framing = "tool_slots"
        st["chat_to_tool_slots"] += was_chat
    else:
        framing = "chat"
        # the chat framing renders no ``tools=``: the list goes into the system message
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = system_with_tool_list(msgs[0]["content"], tools)
        else:
            msgs.insert(0, {"role": "system", "content": system_with_tool_list("", tools)})
        st["chat_tool_list_written"] += 1
    out = dict(r)
    out.update(framing=framing, instruction=json.dumps(msgs), target=tgt,
               context=json.dumps(new_ctx) if new_ctx else "",
               prompt=json.dumps(new_pr) if new_ctx else "", meta=json.dumps(meta))
    return out


def check_row(r: dict, renderer) -> None:
    """Render like the trainer (TaskDataset) and assert the slot bookkeeping."""
    meta = json.loads(r["meta"])
    if r["framing"] == "chat":
        assert SPLICE_SENTINEL not in r["instruction"]
        renderer.render("chat", r["instruction"])
        return
    ctx, pr = json.loads(r["context"]), json.loads(r["prompt"])
    segs = renderer.render_slots(r["instruction"], meta.get("tools") or None)
    assert len(segs) - 1 == len(ctx) == len(pr) == len(meta["digests"]) == meta["n_slots"], (
        len(segs) - 1, len(ctx), len(pr), len(meta["digests"]), meta["n_slots"])
    assert r["instruction"].count(SPLICE_SENTINEL) == len(ctx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="input .parquet")
    ap.add_argument("--out", required=True, help="output .parquet (+ .arrow beside it)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--tokenizer", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--check-every", type=int, default=1, help="render-check every Nth row")
    args = ap.parse_args()
    out = Path(args.out)
    arrow = out.with_suffix(".arrow")
    for p in (out, arrow):
        if p.exists() and not args.overwrite:
            sys.exit(f"{p} exists; pass --overwrite")

    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from bgkit2.data.tasks import ChatRenderer

    renderer = ChatRenderer.from_tokenizer(AutoTokenizer.from_pretrained(args.tokenizer))
    specs = current_specs()
    pf = pq.ParquetFile(args.store)
    tmp = out.with_suffix(".parquet.tmp")
    w = pq.ParquetWriter(str(tmp), pf.schema_arrow, compression="zstd")
    st: Counter = Counter()
    t0 = time.time()
    for b in pf.iter_batches(batch_size=1024):
        rows = []
        for r in b.to_pylist():
            st["in"] += 1
            try:
                nr = rerender_row(r, specs, st)
            except (ValueError, KeyError) as exc:
                st[f"drop:{str(exc)[:40]}"] += 1
                continue
            if args.check_every and st["in"] % args.check_every == 0:
                check_row(nr, renderer)
                st["render_checked"] += 1
            st[f"out_framing:{nr['framing']}"] += 1
            rows.append(nr)
        if rows:
            w.write_table(pa.Table.from_pylist(rows, schema=pf.schema_arrow), row_group_size=2048)
            st["out"] += len(rows)
        if st["in"] % 10240 < 1024:
            print(f"[{time.time() - t0:.0f}s] {dict(st)}", flush=True)
    w.close()
    tmp.replace(out)
    ta = arrow.with_suffix(".arrow.tmp")
    pf2 = pq.ParquetFile(str(out))
    with pa.OSFile(str(ta), "wb") as sink, pa.ipc.new_file(sink, pf2.schema_arrow) as aw:
        for batch in pf2.iter_batches(batch_size=4096):
            aw.write_batch(batch)
    ta.replace(arrow)
    report = {"store": args.store, "out": str(out), "rows": st["out"], "stats": dict(st),
              "wall_s": round(time.time() - t0)}
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
