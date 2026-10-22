#!/usr/bin/env python
"""Build the bgkit2 ``chat_replay`` task store: plain chat / agentic
conversations with NO context slot (family ``chat_replay``, framing ``chat``).

Each row is one assistant turn to reproduce: ``instruction`` is the JSON list
of chat messages up to (and excluding) that turn, ``target`` is the turn —
its plain-text content, or (v2, 2026-09-14) a JSON list holding the whole
assistant message when the turn carries tool calls, so the decoder learns the
content AND the call it led to (``ChatRenderer.render_chunk`` renders it; v1
stored only the text preamble and taught the model to END its turn there —
every swe_zero row was such a preamble). Tool calls / tool results earlier in
the history are kept (normalised to the dict-argument shape LFM2.5's chat
template renders); assistant turns that are ONLY tool calls are never targets.
Histories are fitted to a TOKEN budget with the real tokenizer
(``--max-prefix-tokens``; v1 used a 12,000-char budget that put swe_zero
histories at ~2,900 tokens, past the trainer's 1,536 prefix cap), and
identical histories are kept once.

Sources:
  smoltalk   HuggingFaceTB/smoltalk (config ``all``, train) — streamed; the
             split is shuffled, so the first ``--smoltalk-scan`` rows are a
             representative slice. Per-source reservoirs, then quotas
             proportional to the source mix, capped per source. One random
             assistant turn per conversation.
  swe_zero   local SWE-Zero trajectory shards (OpenHands agent runs); up to 2
             assistant turns per trajectory, history messages capped
             head/tail (1500 chars, tool results 600).
  toucan     local Toucan tool-calling SFT sample (``tool_call`` /
             ``tool_response`` pseudo-roles folded into OpenAI-style
             tool_calls + ``tool`` messages).

Every row is structurally validated; a random sample (``--render-check``) is
rendered through the decoder tokenizer's real chat template and token-length
stats (prefix / target) are printed. Rows that fail to render are dropped.

Example:
  scripts/bgkit2_build_replay.py --out $B/tasks/chat_replay.parquet
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import itertools
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Same COLUMNS as bgkit2.data.tasks (written with an equivalent pyarrow writer).
COLUMNS = [
    "family",
    "framing",
    "context",
    "prompt",
    "instruction",
    "tool_name",
    "tool_args",
    "target",
    "split",
    "meta",
]

DECODER_ID = "LiquidAI/LFM2.5-350M"  # configs/training/bgkit2.yaml model.decoder_id
SWE_DIR = "/home/werg/bgkit-data-nvme/capability_packaging/trajectories_ext/swe_zero/data"
TOUCAN = (
    "/home/werg/bgkit-data-nvme/capability_packaging/trajectories_ext/toucan/sample_500_sft.jsonl"
)
OUT = "/home/werg/bgkit-data-nvme/bgkit2/tasks/chat_replay.parquet"

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


# ----------------------------------------------------------------------------
# helpers (mirroring scripts/bgkit2_build_tasks.py conventions)
def _split_of(key: str, eval_frac: float) -> str:
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if h < eval_frac else "train"


def _cap_msg(mm: dict, max_chars: int) -> dict:
    """Keep the head and tail of an over-long message (anchors, tool dumps)."""
    if len(mm["content"]) > max_chars:
        h = max_chars // 2
        mm["content"] = mm["content"][:h] + "\n[...]\n" + mm["content"][-h:]
    return mm


def _trajectory_files(dirs: list[str], shards: str) -> list[Path]:
    files: list[Path] = []
    for d in dirs:
        files += sorted(Path(d).rglob("*.parquet"))
    a, b = (int(x) for x in shards.split(":"))
    return files[a:b]


def _parse_args(args) -> dict:
    """Tool-call arguments -> dict (JSON, then Python literal, else wrapped)."""
    if isinstance(args, dict):
        return args
    if args is None:
        return {}
    s = str(args)
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {"input": v}
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, dict) else {"input": v}
    except (ValueError, SyntaxError):
        return {"input": s}


def _normalise(m: dict) -> dict:
    """OpenAI-style message -> what LFM2.5's template accepts (dict args);
    ``<think>`` blocks stripped from assistant content (the template keeps
    them only on the last assistant message: not prefix-consistent)."""
    content = m.get("content")
    if isinstance(content, list):  # multimodal-style parts
        content = "\n".join(
            str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content
        )
    content = str(content or "")
    role = m.get("role", "user")
    if role == "assistant":
        content = _THINK_RE.sub("", content)
    out = {"role": role, "content": content}
    if m.get("tool_calls"):
        calls = []
        for tc in m["tool_calls"]:
            fn = tc.get("function", tc) if isinstance(tc, dict) else {}
            calls.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(fn.get("name", "tool")),
                        "arguments": _parse_args(fn.get("arguments")),
                    },
                }
            )
        out["tool_calls"] = calls
    return out


def _valid_history(msgs: list[dict]) -> bool:
    """Structural contract for the template: known roles, first non-system
    message is a user, no consecutive same-role text turns except tool."""
    if not msgs:
        return False
    roles = [m["role"] for m in msgs]
    if any(r not in ("system", "user", "assistant", "tool") for r in roles):
        return False
    body = roles[1:] if roles[0] == "system" else roles
    if not body or body[0] != "user":
        return False
    for a, b in itertools.pairwise(body):
        if a == b and a in ("user", "assistant"):
            return False
    for m in msgs:
        if m["role"] != "assistant" and m.get("tool_calls"):
            return False
        if m["role"] == "assistant" and not (m["content"].strip() or m.get("tool_calls")):
            return False
        if m["role"] in ("system", "user", "tool") and not m["content"].strip():
            return False
    return True


def _fit_history(msgs: list[dict], fits) -> list[dict] | None:
    """Drop the earliest post-system turns until ``fits(candidate)`` holds,
    keeping the system message. A trimmed history restarts at a later USER
    turn; when there is none (single-task agent trajectories: one user
    message, then assistant/tool alternation) the first user message is kept
    as the task anchor and turns after it are dropped, resuming at an
    assistant turn. None when nothing fits."""
    sys_msg = [msgs[0]] if msgs and msgs[0]["role"] == "system" else []
    body = msgs[len(sys_msg) :]
    if not body or body[0]["role"] != "user":
        return None

    b = body
    while b:
        if b[0]["role"] == "user" and fits(sys_msg + b):
            return sys_msg + b
        b = b[1:]
        while b and b[0]["role"] != "user":
            b = b[1:]
    anchor = [body[0]]
    b = body[1:]
    while b:
        if b[0]["role"] == "assistant" and fits(sys_msg + anchor + b):
            return sys_msg + anchor + b
        b = b[1:]
    return None


_TOK = None


def _tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained(DECODER_ID)
    return _TOK


def _prefix_tokens(msgs: list[dict]) -> int:
    ids = _tokenizer().apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    return len(ids if isinstance(ids, list) else ids["input_ids"])


def _make_row(
    history: list[dict],
    target: dict,
    source: str,
    key: str,
    args,
    extra_meta: dict | None = None,
) -> dict | None:
    """``target`` is the raw assistant message of the turn. A text-only turn
    is stored as its text; a turn with tool calls is stored as a one-message
    JSON list (content + normalised calls) so the whole turn is the target."""
    turn = _normalise(target)
    turn["content"] = turn["content"].strip()
    text = turn["content"]
    if not (args.min_target <= len(text) <= args.max_target):
        return None
    history = [_normalise(m) for m in history]
    for m in history:
        m["content"] = m["content"].strip()

    def fits(cand: list[dict]) -> bool:
        if len(json.dumps(cand)) > args.max_instruction_chars:
            return False
        return _prefix_tokens(cand) <= args.max_prefix_tokens

    history = _fit_history(history, fits)
    if history is None or not _valid_history(history):
        return None
    if history[-1]["role"] not in ("user", "tool"):
        return None
    meta = {"source": source, "n_turns": len(history)}
    if extra_meta:
        meta.update(extra_meta)
    if turn.get("tool_calls"):
        if len(json.dumps(turn)) > args.max_turn_chars:
            return None
        stored = json.dumps([turn], ensure_ascii=False)
        meta["target_kind"] = "turn"
    else:
        stored = text
        meta["target_kind"] = "text"
    return {
        "family": "chat_replay",
        "framing": "chat",
        "context": "",
        "prompt": "",
        "instruction": json.dumps(history, ensure_ascii=False),
        "tool_name": "",
        "tool_args": "",
        "target": stored,
        "split": _split_of(key, args.eval_frac),
        "meta": json.dumps(meta),
    }


def _text_targets(msgs: list[dict]) -> list[int]:
    """Indices of assistant turns with non-empty plain-text content."""
    out = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        c = m.get("content")
        if isinstance(c, list):
            c = "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in c)
        c = _THINK_RE.sub("", str(c or "")).strip()
        if c:
            out.append(i)
    return out


def _pick_targets(msgs: list[dict], cands: list[int], k: int, rng) -> list[int]:
    """Up to k target turns: pure-text assistant turns first, then turns
    that also carry tool calls (stored as whole turns, see ``_make_row``);
    random within each group."""
    pure = [i for i in cands if not msgs[i].get("tool_calls")]
    mixed = [i for i in cands if msgs[i].get("tool_calls")]
    rng.shuffle(pure)
    rng.shuffle(mixed)
    return sorted((pure + mixed)[:k])


def _target_text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        c = "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in c)
    return _THINK_RE.sub("", str(c or "")).strip()


# ----------------------------------------------------------------------------
def build_smoltalk(args, rng: random.Random) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
    cap = args.smoltalk_source_cap
    seen: collections.Counter = collections.Counter()
    reservoir: dict[str, list[dict]] = collections.defaultdict(list)
    t0 = time.time()
    n_scanned = 0
    for i, r in enumerate(ds):
        if i >= args.smoltalk_scan:
            break
        n_scanned += 1
        src = r["source"]
        seen[src] += 1
        msgs = r["messages"]
        cands = _text_targets(msgs)
        if not cands:
            continue
        ti = rng.choice(cands)
        row = _make_row(
            msgs[:ti],
            msgs[ti],
            f"smoltalk/{src}",
            f"smoltalk:{i}",
            args,
            {"turn_index": ti, "n_messages": len(msgs)},
        )
        if row is None:
            continue
        # per-source reservoir sampling (size cap) over the scanned slice
        res = reservoir[src]
        if len(res) < cap:
            res.append(row)
        else:
            j = rng.randrange(seen[src])
            if j < cap:
                res[j] = row
        if (i + 1) % 20000 == 0:
            print(
                f"smoltalk: scanned {i + 1} rows, reservoirs "
                f"{sum(len(v) for v in reservoir.values())}; {time.time() - t0:.0f}s",
                flush=True,
            )
    # proportional quotas with per-source cap; redistribute leftover
    total = args.smoltalk_rows
    avail = {s: len(v) for s, v in reservoir.items()}
    quota = {s: 0 for s in avail}
    remaining = total
    open_srcs = set(avail)
    while remaining > 0 and open_srcs:
        wsum = sum(seen[s] for s in open_srcs)
        alloc = {s: round(remaining * seen[s] / wsum) for s in open_srcs}
        progressed = False
        for s in list(open_srcs):
            room = min(cap, avail[s]) - quota[s]
            take = min(alloc[s], room)
            if take > 0:
                quota[s] += take
                remaining -= take
                progressed = True
            if quota[s] >= min(cap, avail[s]):
                open_srcs.discard(s)
        if not progressed:
            break
    rows: list[dict] = []
    print(f"smoltalk: scanned {n_scanned} rows in {time.time() - t0:.0f}s", flush=True)
    for s in sorted(quota, key=lambda k: -quota[k]):
        pick = reservoir[s]
        rng.shuffle(pick)
        rows += pick[: quota[s]]
        print(
            f"  {s:28s} seen {seen[s]:7d} ({100 * seen[s] / n_scanned:5.1f}%)"
            f"  reservoir {avail[s]:6d}  quota {quota[s]:6d}",
            flush=True,
        )
    return rows


def build_swe_zero(args, rng: random.Random) -> list[dict]:
    files = _trajectory_files([args.swe_dir], args.swe_shards)
    if not files:
        print("swe_zero: no shards found", flush=True)
        return []
    per_shard = -(-args.swe_rows // len(files))
    rows: list[dict] = []
    t0 = time.time()
    for fi, f in enumerate(files):
        pf = pq.ParquetFile(f)
        names = pf.schema_arrow.names
        col = "trajectory" if "trajectory" in names else "messages"
        cols = [c for c in ("trajectory_id", "instance_id", "repo", col) if c in names]
        n_shard = 0
        n_traj = 0
        for batch in pf.iter_batches(batch_size=32, columns=cols):
            for r in batch.to_pylist():
                traj = r[col]
                if isinstance(traj, str):
                    traj = json.loads(traj)
                if not isinstance(traj, list) or len(traj) < 4:
                    continue
                n_traj += 1
                cands = [i for i in _text_targets(traj) if i >= 2]
                if not cands:
                    continue
                tid = str(r.get("trajectory_id") or r.get("instance_id") or f"{f.stem}:{n_traj}")
                for ti in _pick_targets(traj, cands, args.swe_per_traj, rng):
                    hist = []
                    for m in traj[:ti]:
                        mm = _normalise(m)
                        limit = args.swe_tool_chars if mm["role"] == "tool" else args.swe_msg_chars
                        hist.append(_cap_msg(mm, limit))
                    row = _make_row(
                        hist,
                        traj[ti],
                        "swe_zero",
                        f"swe_zero:{tid}",
                        args,
                        {
                            "trajectory_id": tid,
                            "repo": str(r.get("repo") or ""),
                            "turn_index": ti,
                            "n_messages": len(traj),
                            "target_had_tool_calls": bool(traj[ti].get("tool_calls")),
                        },
                    )
                    if row is not None:
                        rows.append(row)
                        n_shard += 1
                if n_shard >= per_shard:
                    break
            if n_shard >= per_shard:
                break
        print(
            f"swe_zero [{fi + 1}/{len(files)}] {f.name}: trajectories {n_traj}, "
            f"rows {n_shard} (total {len(rows)}); {time.time() - t0:.0f}s",
            flush=True,
        )
        if len(rows) >= args.swe_rows:
            break
    return rows[: args.swe_rows]


def _toucan_messages(raw: list[dict]) -> list[dict] | None:
    """Fold Toucan's ``tool_call`` / ``tool_response`` pseudo-roles into
    OpenAI-style assistant tool_calls + ``tool`` messages."""
    out: list[dict] = []
    for m in raw:
        role = m.get("role")
        if role == "tool_call":
            call = _parse_args(m.get("content"))
            if "name" not in call:
                return None
            tc = {
                "type": "function",
                "function": {"name": call["name"], "arguments": _parse_args(call.get("arguments"))},
            }
            if out and out[-1]["role"] == "assistant" and not out[-1].get("_closed"):
                out[-1].setdefault("tool_calls", []).append(tc)
            else:
                out.append({"role": "assistant", "content": "", "tool_calls": [tc]})
        elif role == "tool_response":
            c = m.get("content")
            out.append({"role": "tool", "content": c if isinstance(c, str) else json.dumps(c)})
            if out and len(out) >= 2:
                for prev in reversed(out[:-1]):
                    if prev["role"] == "assistant":
                        prev["_closed"] = True
                        break
        elif role in ("system", "user", "assistant"):
            out.append({"role": role, "content": m.get("content")})
        else:
            return None
    for m in out:
        m.pop("_closed", None)
    return out


def build_toucan(args, rng: random.Random) -> list[dict]:
    path = Path(args.toucan)
    if not path.exists():
        print(f"toucan: {path} missing, skipped", flush=True)
        return []
    rows: list[dict] = []
    n_conv = n_bad = 0
    with open(path, encoding="utf-8") as fh:
        for li, line in enumerate(fh):
            d = json.loads(line)
            raw = d.get("messages")
            if isinstance(raw, str):
                raw = json.loads(raw)
            if not isinstance(raw, list):
                n_bad += 1
                continue
            msgs = _toucan_messages(raw)
            if msgs is None:
                n_bad += 1
                continue
            n_conv += 1
            cands = _text_targets(msgs)
            uid = str(d.get("uuid") or li)
            for ti in _pick_targets(msgs, cands, args.toucan_per_conv, rng):
                hist = [_cap_msg(_normalise(m), args.swe_msg_chars) for m in msgs[:ti]]
                row = _make_row(
                    hist,
                    msgs[ti],
                    "toucan",
                    f"toucan:{uid}",
                    args,
                    {
                        "uuid": uid,
                        "subset": str(d.get("subset_name", "")),
                        "turn_index": ti,
                        "n_messages": len(msgs),
                        "target_had_tool_calls": bool(msgs[ti].get("tool_calls")),
                    },
                )
                if row is not None:
                    rows.append(row)
    print(f"toucan: conversations {n_conv}, unparsable {n_bad}, rows {len(rows)}", flush=True)
    return rows


# ----------------------------------------------------------------------------
def _pct(xs: list[int], q: float) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def render_check(rows: list[dict], args, rng: random.Random) -> set[int]:
    """Render a random sample through the real chat template; return the
    indices (into rows) that failed. Prints per-source failures and token
    length stats."""
    from bgkit2.data.tasks import ChatRenderer

    tok = _tokenizer()
    renderer = ChatRenderer.from_tokenizer(tok)
    idx = list(range(len(rows)))
    if args.render_check and args.render_check < len(idx):
        idx = rng.sample(idx, args.render_check)
    failed: set[int] = set()
    fails_by_src: collections.Counter = collections.Counter()
    n_by_src: collections.Counter = collections.Counter()
    pre_len: dict[str, list[int]] = collections.defaultdict(list)
    tgt_len: dict[str, list[int]] = collections.defaultdict(list)
    t0 = time.time()
    for k, i in enumerate(idx):
        r = rows[i]
        src = json.loads(r["meta"])["source"].split("/")[0]
        n_by_src[src] += 1
        try:
            msgs = json.loads(r["instruction"])
            ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
            if not isinstance(ids, list):
                ids = ids["input_ids"]
            if r["target"].lstrip().startswith("["):
                tgt_ids, mask = renderer.render_chunk(msgs, json.loads(r["target"]))
                if not bool(mask.all()) or tgt_ids.shape[0] == 0:
                    raise ValueError("turn target did not render as one assistant message")
                n_tgt = int(tgt_ids.shape[0])
            else:
                full = tok.apply_chat_template(
                    [*msgs, {"role": "assistant", "content": r["target"]}],
                    tokenize=True,
                    add_generation_prompt=False,
                )
                if not isinstance(full, list):
                    full = full["input_ids"]
                if len(ids) == 0 or len(full) <= len(ids) or full[: len(ids)] != ids:
                    raise ValueError("target rendering does not extend the prefix")
                n_tgt = len(full) - len(ids)
            if len(ids) > args.max_prefix_tokens + 8:
                raise ValueError(f"prefix {len(ids)} tokens over the budget")
            pre_len[src].append(len(ids))
            tgt_len[src].append(n_tgt)
        except Exception as e:
            failed.add(i)
            fails_by_src[src] += 1
            if fails_by_src[src] <= 3:
                print(f"  render fail [{src}] row {i}: {type(e).__name__}: {str(e)[:160]}")
        if (k + 1) % 2000 == 0:
            print(f"render check: {k + 1}/{len(idx)}; {time.time() - t0:.0f}s", flush=True)
    print(f"render check: {len(idx)} rows, {len(failed)} failed; {time.time() - t0:.0f}s")
    for src in sorted(n_by_src):
        p, t = pre_len[src], tgt_len[src]
        print(
            f"  {src:10s} n {n_by_src[src]:5d} fail {fails_by_src[src]:4d} | prefix tokens "
            f"p50 {_pct(p, 0.5):6d} p90 {_pct(p, 0.9):6d} max {max(p, default=0):6d} | "
            f"target tokens p50 {_pct(t, 0.5):5d} p90 {_pct(t, 0.9):5d} max {max(t, default=0):5d}"
        )
    allp = [x for v in pre_len.values() for x in v]
    allt = [x for v in tgt_len.values() for x in v]
    print(
        f"  {'ALL':10s} n {len(idx):5d} fail {len(failed):4d} | prefix tokens "
        f"p50 {_pct(allp, 0.5):6d} p90 {_pct(allp, 0.9):6d} max {max(allp, default=0):6d} | "
        f"target tokens p50 {_pct(allt, 0.5):5d} p90 {_pct(allt, 0.9):5d} "
        f"max {max(allt, default=0):5d}"
    )
    return failed


def write_store(path: str, rows: list[dict]) -> int:
    cols = {c: [] for c in COLUMNS}
    for r in rows:
        for c in COLUMNS:
            v = r.get(c, "")
            if not isinstance(v, str):
                v = json.dumps(v)
            cols[c].append(v)
    table = pa.table({c: pa.array(v, type=pa.string()) for c, v in cols.items()})
    pq.write_table(table, path, compression="zstd")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--sources", default="smoltalk,swe_zero,toucan")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--max-instruction-chars", type=int, default=12000)
    ap.add_argument(
        "--max-prefix-tokens",
        type=int,
        default=1400,
        help="history budget in decoder tokens (rendered with the real template; the trainer "
        "caps prefixes at 1536)",
    )
    ap.add_argument(
        "--max-turn-chars",
        type=int,
        default=6000,
        help="drop tool-calling turns whose JSON exceeds this (whole-file edits)",
    )
    ap.add_argument("--min-target", type=int, default=20)
    ap.add_argument("--max-target", type=int, default=4000)
    ap.add_argument("--smoltalk-rows", type=int, default=60000)
    ap.add_argument("--smoltalk-source-cap", type=int, default=15000)
    ap.add_argument(
        "--smoltalk-scan", type=int, default=240000, help="rows streamed (shuffled split)"
    )
    ap.add_argument("--swe-dir", default=SWE_DIR)
    ap.add_argument("--swe-shards", default="0:6")
    ap.add_argument("--swe-rows", type=int, default=15000)
    ap.add_argument("--swe-per-traj", type=int, default=2)
    ap.add_argument("--swe-msg-chars", type=int, default=1500)
    ap.add_argument("--swe-tool-chars", type=int, default=600)
    ap.add_argument("--toucan", default=TOUCAN)
    ap.add_argument("--toucan-per-conv", type=int, default=2)
    ap.add_argument("--render-check", type=int, default=500, help="rows to render (0 = all)")
    ap.add_argument(
        "--source-fail-frac",
        type=float,
        default=0.05,
        help="drop a whole source when more than this fraction of its sample fails",
    )
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit(f"refusing to overwrite existing {args.out}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    builders = {"smoltalk": build_smoltalk, "swe_zero": build_swe_zero, "toucan": build_toucan}
    rows: list[dict] = []
    for src in args.sources.split(","):
        t0 = time.time()
        got = builders[src](args, rng)
        print(f"== {src}: {len(got)} rows in {time.time() - t0:.0f}s", flush=True)
        rows += got

    # identical histories (smoltalk repeats "snail in a well" 62x) are kept once
    seen_hist: set[str] = set()
    uniq: list[dict] = []
    for r in rows:
        if r["instruction"] in seen_hist:
            continue
        seen_hist.add(r["instruction"])
        uniq.append(r)
    print(f"dropped {len(rows) - len(uniq)} rows with a duplicate history", flush=True)
    rows = uniq
    failed = render_check(rows, args, rng)
    # per-source failure rate on the sample -> drop whole source if bad
    by_src_n: collections.Counter = collections.Counter()
    by_src_f: collections.Counter = collections.Counter()
    for i in failed:
        by_src_f[json.loads(rows[i]["meta"])["source"].split("/")[0]] += 1
    sample_srcs = collections.Counter()
    # recompute sample sizes per source from the same sample the checker used
    # (the checker prints them; here we only need fail fractions per source)
    for r in rows:
        by_src_n[json.loads(r["meta"])["source"].split("/")[0]] += 1
    drop_sources = set()
    if args.render_check:
        for src, nf in by_src_f.items():
            expected = args.render_check * by_src_n[src] / max(1, len(rows))
            if expected > 0 and nf / expected > args.source_fail_frac:
                drop_sources.add(src)
    del sample_srcs
    kept = [
        r
        for i, r in enumerate(rows)
        if i not in failed and json.loads(r["meta"])["source"].split("/")[0] not in drop_sources
    ]
    if drop_sources:
        print(f"dropping sources with failed renders: {sorted(drop_sources)}")
    n = write_store(args.out, kept)
    per_src = collections.Counter(json.loads(r["meta"])["source"] for r in kept)
    per_split = collections.Counter(r["split"] for r in kept)
    print(f"wrote {n} rows -> {args.out}")
    print(f"splits: {dict(per_split)}")
    for s, c in sorted(per_src.items(), key=lambda kv: -kv[1]):
        print(f"  {s:36s} {c:7d}")
    n_turn = sum(json.loads(r["meta"]).get("target_kind") == "turn" for r in kept)
    print(f"targets stored as whole tool-calling turns: {n_turn}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
    # datasets' streaming reader (aiohttp/fsspec threads) aborts the
    # interpreter with PyGILState_Release at finalisation on this venv
    # (datasets 4.8.4); everything is written and flushed by now.
    os._exit(0)
