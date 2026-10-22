#!/usr/bin/env python
"""Semantic conversion of the EXTERNAL tool_slots rows (OpenHands / SWE-Zero /
OpenResearcher / Open-SWE traces): a cheap LLM writes, per compressed tool
result, (a) the compression PURPOSE — what the agent needed from that result,
judged from its next action — and (b) a query-focused DIGEST of the raw
result (user, 2026-09-14: "use a cheap LLM to do semantic conversion of
existing tool slot rows in such a way that a compression prompt is
synthesized").

The store's prompts today are the raw call ("str_replace_editor: view
/workspace/..."), which says nothing about what to keep. Outputs:
  --out-slots   tool_slots rows whose prompts become "<our tool> <arg>: <purpose>"
                (only rows whose every slot was synthesised)
  --out-digest  tool_digest rows (framing tool_response): raw result in the
                slot, purpose as the prompt, digest as the target
  --cache       JSONL of {key, purpose, digest} (resumable; one line per slot)

Runs against the K2 llama-server (:8096) alongside the teacher driver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bgkit2.agent.teacher import DIGEST_STOP, parse_k2_output
from bgkit2.data.tasks import write_task_store
from bgkit2.data.templates import SPLICE_SENTINEL

SYSTEM = (
    "You annotate tool results from a software-agent transcript. You have NO tools and "
    "never emit a tool call. Given the task, the call the agent made, the raw result, and "
    "what the agent did next, reply with ONE JSON object and nothing else:\n"
    '{"purpose": "<one imperative line: what to extract from this result for the agent\'s '
    "next step, e.g. 'find where Input handles cursor position and the file that tests it'>\", "
    '"digest": "<plain-text digest of the raw result, at most 120 words, keeping every exact '
    'identifier, path, line number and value the agent used or would need>"}\n'
    "The digest describes ONLY what the raw result itself shows. Never name a file, symbol or "
    "line number that does not appear in the raw result. Never say what the agent should do "
    "next, never recommend a file to open, never infer a path that is not printed. If the raw "
    "result is unhelpful, say so in one sentence."
)

EXT_TOOL_MAP = {  # external tool -> (our tool, arg mapper) for the digest rows
    "str_replace_editor": None,  # resolved per command below
    "execute_bash": ("bash", lambda a: {"cmd": str(a.get("command", ""))}),
    "bash": ("bash", lambda a: {"cmd": str(a.get("command", a.get("cmd", "")))}),
    "read_file": ("read", lambda a: {"path": str(a.get("path", ""))}),
    "read": ("read", lambda a: {"path": str(a.get("path", ""))}),
    "search": ("search", lambda a: {"query": str(a.get("query", ""))}),
    "grep": ("search", lambda a: {"query": str(a.get("pattern", a.get("query", "")))}),
}


def our_call(name: str, args: dict) -> tuple[str, dict]:
    if name == "str_replace_editor":
        cmd = str(args.get("command", "view"))
        if cmd == "view":
            return "read", {"path": str(args.get("path", ""))}
        return "edit", {"path": str(args.get("path", ""))}
    m = EXT_TOOL_MAP.get(name)
    if m:
        return m[0], m[1](args)
    return "bash", {"cmd": f"{name} {json.dumps(args)[:200]}"}


def slot_contexts(msgs: list[dict]) -> list[tuple[int, dict | None, dict | None]]:
    """For each sentinel tool message (in order): (index, preceding assistant
    message, following assistant message)."""
    out = []
    for i, m in enumerate(msgs):
        if m.get("role") == "tool" and SPLICE_SENTINEL in str(m.get("content", "")):
            prev = next(
                (msgs[j] for j in range(i - 1, -1, -1) if msgs[j]["role"] == "assistant"), None
            )
            nxt = next(
                (msgs[j] for j in range(i + 1, len(msgs)) if msgs[j]["role"] == "assistant"), None
            )
            out.append((i, prev, nxt))
    return out


def call_of(msg: dict | None, k: int = 0) -> tuple[str, dict]:
    if not msg:
        return "", {}
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return "", {}
    fn = tcs[min(k, len(tcs) - 1)].get("function", {})
    a = fn.get("arguments", {})
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {"raw": a}
    return str(fn.get("name", "")), a if isinstance(a, dict) else {}


def describe(msg: dict | None, limit: int) -> str:
    if not msg:
        return "(the transcript ends here)"
    parts = [str(msg.get("content", "")).strip()]
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        parts.append(f"CALL {fn.get('name')} {json.dumps(fn.get('arguments', {}))[:300]}")
    return "\n".join(p for p in parts if p)[:limit]


def build_messages(task: str, call: str, raw: str, nxt: str, raw_limit: int) -> list[dict]:
    if len(raw) > raw_limit:
        h = raw_limit // 2
        raw = raw[:h] + "\n[...]\n" + raw[-h:]
    user = (
        f"TASK the agent is working on:\n{task}\n\nCALL the agent made: {call}\n\n"
        f"RAW RESULT:\n{raw}\n\nWHAT THE AGENT DID NEXT:\n{nxt}\n\nReply with the JSON object."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


HINDSIGHT_SYSTEM = SYSTEM + (
    " You are also shown the REST of the transcript (hindsight) and a list of identifiers from the "
    "raw result that the agent later relied on: the digest MUST contain every one of them verbatim, "
    "with the fact around it (line number, value, signature). Use hindsight only to decide what to "
    "KEEP from the raw result; the digest itself may state only what the raw result shows, and the "
    "purpose line must say what to look for (so a reader given only the purpose and the raw result "
    "could produce the same digest). Never mention the agent or anything it did after this result. "
    "The PURPOSE describes what to extract from THIS raw result (the file/output it actually shows), "
    "never the agent's next goal or another file; the DIGEST describes only this raw result, never "
    "another command or its output."
)

_SENT = re.compile(r"(?<=[.!?])\s+")
_FUTURE = re.compile(r"\b(the agent|agent's|afterwards|subsequently|next step)\b", re.I)


def strip_future(digest: str) -> str:
    """Drop sentences that narrate the agent's later actions: the student sees only (purpose, raw)
    and cannot know them (2026-09-21: 15% of plain and 30% of hindsight digests did this)."""
    return " ".join(x for x in _SENT.split(digest) if not _FUTURE.search(x)).strip()


IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}|[\w./-]+\.\w{1,5}|\b\d{2,}\b")
_STOP = {"CALL", "path", "command", "view", "true", "false", "None", "self", "return", "import",
         "from", "def", "class", "with", "this", "that", "file", "line", "print"}


def used_identifiers(raw: str, later: str) -> list[str]:
    """Identifier-like tokens (path, snake/Camel name, number) the agent used AFTER this result
    that the raw result contains -- the must-keep list of the hindsight prompt (at most 15)."""
    seen, out = set(), []
    for w in IDENT.findall(later):
        if not (any(c in w for c in "_./") or any(c.isdigit() for c in w)
                or (w[0].isupper() and any(c.isupper() for c in w[1:]))
                or (w[0].islower() and any(c.isupper() for c in w))):
            continue
        if len(w) >= 4 and w not in _STOP and w in raw and w not in seen:
            seen.add(w)
            out.append(w)
    return out[:15]


def build_messages_hindsight(task: str, call: str, raw: str, later: str, must_keep: list[str],
                             raw_limit: int) -> list[dict]:
    """Teacher prompt with HINDSIGHT: the whole rest of the episode plus the identifiers the agent
    later used. The digest stays a function of (purpose, raw) -- the student sees only those --
    but the teacher chooses what to keep knowing what mattered."""
    if len(raw) > raw_limit:
        h = raw_limit // 2
        raw = raw[:h] + "\n[...]\n" + raw[-h:]
    keep = ", ".join(f"`{k}`" for k in must_keep[:40]) or "(none detected)"
    user = (
        f"TASK the agent is working on:\n{task}\n\nCALL the agent made: {call}\n\n"
        f"RAW RESULT:\n{raw}\n\nWHAT THE AGENT DID AFTERWARDS (hindsight):\n{later}\n\n"
        f"IDENTIFIERS FROM THE RAW RESULT THAT THE AGENT LATER USED (must appear in the digest): "
        f"{keep}\n\nReply with the JSON object. The digest is at most 120 words of plain prose "
        f"that states facts (never a copy of raw lines, never a bare keyword list)."
    )
    return [{"role": "system", "content": HINDSIGHT_SYSTEM}, {"role": "user", "content": user}]


JSON_RE = re.compile(r"\{.*\}", re.S)


PATH_RE = re.compile(
    r"[\w./-]+\.(?:py|js|ts|tsx|md|txt|json|ya?ml|toml|cfg|ini|rst|c|cpp|h|hpp|go|rs|java|sh)\b"
)


def digest_is_grounded(digest: str, raw: str, call: str) -> bool:
    """Every file path the digest names must appear in the raw result (or in
    the call itself). Measured 2026-09-15: 28% of digests named a path that was
    in neither — either the agent's NEXT step leaking in from the prompt, or an
    invented deeper path — and such a row teaches the encoder to emit content
    its input never carried."""
    for p in set(PATH_RE.findall(digest)):
        if p in raw or p in call or p.split("/")[-1] in raw:
            continue
        return False
    return True


def parse_json(text: str) -> dict | None:
    text = parse_k2_output(text).content.strip()
    m = JSON_RE.search(text)
    d: dict | None = None
    if m:
        try:
            d = json.loads(m.group(0))
        except Exception:
            d = None
    if not isinstance(d, dict):
        # K2 sometimes narrates instead ("... Purpose: ... Digest: ..."): salvage
        pm = re.search(r"purpose:\s*(.+?)(?:\n|\s+digest:)", text, re.IGNORECASE | re.S)
        gm = re.search(r"digest:\s*(.+)$", text, re.IGNORECASE | re.S)
        if not (pm and gm):
            return None
        d = {"purpose": pm.group(1), "digest": gm.group(1)}
    p, g = str(d.get("purpose", "")).strip(), str(d.get("digest", "")).strip()
    if len(p) < 8 or len(g) < 20:
        return None
    g = strip_future(g)
    if len(g) < 20:
        return None
    return {"purpose": " ".join(p.split())[:300], "digest": g[:2000]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="/home/werg/bgkit-data-nvme/bgkit2/tasks/tool_slots.parquet")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-slots", required=True)
    ap.add_argument("--out-digest", required=True)
    ap.add_argument("--url", default="http://localhost:8096")
    ap.add_argument("--max-slots", type=int, default=8000, help="unique slots to synthesise")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--model", default="k2", help="served model name; 'k2' = K2 request options")
    ap.add_argument("--prompt", default="plain", choices=["plain", "hindsight"])
    ap.add_argument("--raw-limit", type=int, default=6000)
    ap.add_argument("--min-raw-chars", type=int, default=200)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument(
        "--write-only", action="store_true", help="skip synthesis; build from the cache"
    )
    args = ap.parse_args()

    t = pq.read_table(args.store).to_pandas()
    cache: dict[str, dict] = {}
    cp = Path(args.cache)
    if cp.exists():
        for ln in cp.open():
            d = json.loads(ln)
            cache[d["key"]] = d
    print(f"{len(t)} rows, {len(cache)} cached slots", flush=True)

    # unique slots: key = sha1(source, trajectory_id, tool-message index)
    jobs: dict[str, dict] = {}
    row_slots: list[list[str]] = []
    for r in t.itertuples(index=False):
        meta = json.loads(r.meta) if isinstance(r.meta, str) else r.meta
        msgs = json.loads(r.instruction)
        ctx = json.loads(r.context)
        keys = []
        sc = slot_contexts(msgs)
        if len(sc) != len(ctx):
            row_slots.append([])
            continue
        task = next((str(m.get("content", "")) for m in msgs if m.get("role") == "user"), "")[:900]
        for j, (idx, prev, nxt) in enumerate(sc):
            key = hashlib.sha1(
                f"{meta.get('source')}|{meta.get('trajectory_id')}|{idx}|{len(ctx[j])}".encode()
            ).hexdigest()
            keys.append(key)
            if key not in jobs and key not in cache and len(ctx[j]) >= args.min_raw_chars:
                name, a = call_of(prev)
                later_msgs = [m for m in msgs[idx + 1 :] if m.get("role") == "assistant"]
                later = "\n---\n".join(describe(m, 500) for m in later_msgs)[:3000]
                jobs[key] = {
                    "task": task,
                    "call": f"{name} {json.dumps(a)[:400]}",
                    "raw": ctx[j],
                    "next": describe(nxt, 600),
                    "later": later,
                    "must_keep": used_identifiers(ctx[j], later),
                    "ext_name": name,
                    "ext_args": a,
                }
        row_slots.append(keys)
    import random

    # Work ROW BY ROW, not slot by slot: a row is only usable when EVERY one of
    # its slots has a purpose, so a random sample across 143k slots leaves
    # almost no complete row (measured 2026-09-14: 444 slots cached, 0 rows
    # emitted). Whole rows first means the output grows from the first batch.
    budget = max(0, args.max_slots - len(cache))
    row_order = list(range(len(row_slots)))
    random.Random(args.seed).shuffle(row_order)
    order: list[str] = []
    picked: set[str] = set()
    for ri in row_order:
        need = [k for k in row_slots[ri] if k in jobs and k not in picked]
        if not need or len(order) + len(need) > budget:
            continue
        order += need
        picked.update(need)
        if len(order) >= budget:
            break
    print(
        f"{len(jobs)} uncached unique slots; synthesising {len(order)} "
        f"covering whole rows (budget {budget})",
        flush=True,
    )

    lock = threading.Lock()
    stats: Counter = Counter()

    def post(body: dict) -> dict:
        req = urllib.request.Request(
            f"{args.url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    return json.loads(r.read().decode())
            except Exception:
                time.sleep(10 * (attempt + 1))
        raise RuntimeError("teacher endpoint failed")

    def one(key: str) -> None:
        if args.write_only:
            return
        j = jobs[key]
        msgs = (
            build_messages_hindsight(j["task"], j["call"], j["raw"], j["later"], j["must_keep"],
                                     args.raw_limit)
            if args.prompt == "hindsight"
            else build_messages(j["task"], j["call"], j["raw"], j["next"], args.raw_limit)
        )
        body = {"model": args.model, "messages": msgs, "max_tokens": 500, "temperature": 0.0}
        if args.model == "k2":
            body["stop"] = DIGEST_STOP
            body["chat_template_kwargs"] = {"reasoning_effort": "low", "tool_call_format": "json"}
        else:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            resp = post(body)
            text = resp["choices"][0]["message"].get("content") or ""
        except Exception as exc:
            with lock:
                stats["request_failed"] += 1
            print(f"request failed: {exc}", flush=True)
            return
        d = parse_json(text)
        rec = {"key": key, "ext_name": j["ext_name"], "ext_args": j["ext_args"],
               "teacher": args.model, "prompt": args.prompt}
        if d is None:
            rec["error"] = text[:300]
            with lock:
                stats["unparsable"] += 1
        else:
            rec.update(d)
            with lock:
                stats["ok"] += 1
        with lock:
            cache[key] = rec
            with cp.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            n = stats["ok"] + stats["unparsable"] + stats["request_failed"]
            if n % 25 == 0:
                print(f"{n} done: {dict(stats)}", flush=True)

    if order:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            list(ex.map(one, order))
    print(f"synthesis: {dict(stats)}; cache {len(cache)}", flush=True)

    # ---- build the outputs from the cache --------------------------------
    good = {k: v for k, v in cache.items() if v.get("purpose") and v.get("digest")}
    n_before = len(good)
    ungrounded: set[str] = set()
    slots_rows: list[dict] = []
    digest_rows: list[dict] = []
    seen_digest: set[str] = set()
    for r, keys in zip(t.itertuples(index=False), row_slots, strict=True):
        if not keys or not all(k in good for k in keys):
            continue
        d = r._asdict()
        ctx = json.loads(r.context)
        # The prompt must name OUR tool, not the external one: these rows are
        # remapped into the harness schema (bgkit2_remap_tool_slots.py), so a
        # prompt reading "str_replace_editor: view X" beside a `read(path=X)`
        # call would describe a tool the conversation no longer contains.
        new_prompts = []
        for k in keys:
            g = good[k]
            tool, targs = our_call(g.get("ext_name", ""), g.get("ext_args") or {})
            head = targs.get("path") or targs.get("query") or targs.get("cmd") or ""
            new_prompts.append(f"{tool} {head}: {g['purpose']}".strip())
        d["prompt"] = json.dumps(new_prompts)
        d["meta"] = json.loads(r.meta) if isinstance(r.meta, str) else r.meta
        d["meta"]["prompts_synth"] = True
        slots_rows.append(d)
        for raw, k in zip(ctx, keys, strict=True):
            if k in seen_digest:
                continue
            seen_digest.add(k)
            g = good[k]
            tool, targs = our_call(g.get("ext_name", ""), g.get("ext_args") or {})
            call_txt = json.dumps(g.get("ext_args") or {})
            # the purpose is the ENCODER PROMPT: a purpose naming a file this result does not show
            # (hindsight drift toward the agent's next goal) misleads it as much as a bad digest
            if not (digest_is_grounded(g["digest"], raw, call_txt)
                    and digest_is_grounded(g["purpose"], raw, call_txt)):
                ungrounded.add(k)
                continue
            digest_rows.append(
                {
                    "family": "tool_digest",
                    "framing": "tool_response",
                    "context": raw,
                    "prompt": f"Compress the {tool} result to: {g['purpose']}\n",
                    "instruction": f"Give me a compact digest of this result for: {g['purpose']}. "
                    "Keep every exact identifier, path, line number and value that matters.",
                    "tool_name": tool,
                    "tool_args": targs,
                    "target": g["digest"],
                    "split": r.split,
                    "meta": {"source": "ext_synth", "key": k, "raw_chars": len(raw)},
                }
            )
    for path in (args.out_slots, args.out_digest):
        if Path(path).exists():
            Path(path).unlink()
    n1 = write_task_store(args.out_slots, slots_rows)
    n2 = write_task_store(args.out_digest, digest_rows)
    print(
        json.dumps(
            {
                "slots_rows": n1,
                "digest_rows": n2,
                "good_slots": n_before,
                "digests_dropped_ungrounded": len(ungrounded),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
