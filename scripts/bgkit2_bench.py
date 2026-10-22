#!/usr/bin/env python
"""Showcase benchmark runner: score vs DECODER-TOKEN BUDGET, bgkit vs plain text, one model.

Input: jsonl from scripts/bgkit2_bench_prep.py (segments + question + answers). For every budget B
(decoder positions available for the context):

  bgkit   the context in compressed slots (encoder prompt = the question), rep counts sized so the
          slots total ~B positions: ratio = context_tokens / B (1x when it already fits)
  text    the SAME weights reading plain text, middle-truncated to B tokens
  base    the untouched base instruct decoder (adapter disabled), same truncated text
  text_full / base_full   the WHOLE context as plain text (budget ignored; the plain-reading
          ceiling) -- skipped when the context exceeds --full-max-tokens
  summary / base_summary  a larger model's QUESTION-FOCUSED plain-text summary of the context,
          written to ~the budget (scripts/bgkit2_bench_summaries.py, --summaries <jsonl>): the
          prompted-text-compression competitor. Its cost (summariser tokens + seconds) is
          reported beside the reader's
  agent_memory  memory benches only (LongMemEval, LoCoMo), budget-independent (``@agent``): the
          two-stage memory AGENT (bgkit2.agent.memory_eval) -- every session one index slot under
          the generic MEMORY_INDEX_PROMPT, the question in the opening turn, and the student
          calls memory_search / memory_read (the question as the read's focus) for up to
          --agent-max-turns turns. Index slots are cached by content across questions and turns
          (``SlotRepCache``): LoCoMo asks ~200 questions of each conversation, so its index is
          encoded once per conversation; LongMemEval_s has its own haystack per question, so
          only sessions shared between haystacks hit. Rows carry turns, sessions read, decoder
          tokens and encoder tokens (nominal = index once + reads; computed = cache misses)
Every row records decoder prefix positions, context tokens and wall-clock seconds (encoder and
decoder separately for bgkit), so the report can put score against tokens AND time.

Two-tokenizer checkpoints (the LFM2.5-8B-A1B decoder): slots are cut and priced in ENCODER tokens
(``ctx_tokens``, the compression basis), the text arms truncate in DECODER tokens (their budget is
decoder positions), and the full-text gate counts decoder tokens.

One JSON per run: per arm x budget the score, the mean decoder prefix actually used, and the mean
compression. GPU script: CUDA cap + host-memory guard + progress lines.

    docker compose ... run --rm --no-deps train-bgkit2-p28 python scripts/bgkit2_bench.py \\
        --experiment bgkit2_p34_browse_overview_dropout --checkpoint /workspace/checkpoints/<ckpt> \\
        --items /workspace/bgkit2_data/benchmarks/showcase/hotpot.jsonl --budgets 512,2048 \\
        --out /workspace/bgkit2_data/eval/showcase/<run>/hotpot.json
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bgkit2.data.memory_framing import MEMORY_BENCHES, MEMORY_SYSTEM, memory_user  # noqa: E402
from bgkit2.data.templates import SPLICE_SENTINEL  # noqa: E402

SYSTEM = "You answer questions about the provided context. Be brief: give only the answer."


# ---------------------------------------------------------------- scoring
def _norm(s: str) -> str:
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def _f1(pred: str, gold: str) -> float:
    p, g = _norm(pred).split(), _norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def score(item: dict, pred: str) -> dict[str, float]:
    pred = pred.strip()
    golds = [str(a) for a in item["answers"]]
    kind = item["scorer"]
    if kind == "mcq":
        m = re.search(r"\b([ABCD])\b", pred.upper())
        return {"acc": float(bool(m) and m.group(1) == golds[0])}
    if kind == "contains":
        return {"acc": float(any(_norm(g) in _norm(pred) for g in golds))}
    if kind == "contains_all":
        return {"acc": float(all(_norm(g) in _norm(pred) for g in golds))}
    first = pred.split("\n", 1)[0]
    return {"em": max(float(_norm(first) == _norm(g)) for g in golds),
            "f1": max(_f1(first, g) for g in golds),
            "contains": float(any(_norm(g) and _norm(g) in _norm(pred) for g in golds))}


# ---------------------------------------------------------------- prompts
def question_text(item: dict) -> str:
    q = f"Question: {item['question']}"
    if item.get("options"):
        q += "\n" + "\n".join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(item["options"]))
        q += "\nAnswer with the letter only."
    return q


def frame(item: dict, body: str) -> list[dict]:
    """System + user messages, identical for every arm: memory benchmarks (LongMemEval, LoCoMo)
    use the memory framing the memory task rows are trained on (bgkit2.data.memory_framing),
    everything else the generic QA framing. ``body`` = slot sentinels or plain text."""
    if item.get("bench") in MEMORY_BENCHES:
        q = question_text(item).removeprefix("Question: ")
        return [{"role": "system", "content": MEMORY_SYSTEM},
                {"role": "user", "content": memory_user(body, q)}]
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{body}\n\n{question_text(item)}"}]


def pack_chunks(tok, segments: list[str], slot_tokens: int) -> tuple[list[str], list[int]]:
    """Segments -> slot texts of <= slot_tokens tokens: short neighbours are packed together, long
    segments are split. Returns (texts, token counts)."""
    ids_per = [tok(s, add_special_tokens=False)["input_ids"] for s in segments]
    sep = tok("\n\n", add_special_tokens=False)["input_ids"]
    chunks: list[list[int]] = []
    cur: list[int] = []
    for ids in ids_per:
        if len(ids) > slot_tokens:
            if cur:
                chunks.append(cur)
                cur = []
            chunks.extend(ids[i : i + slot_tokens] for i in range(0, len(ids), slot_tokens))
            continue
        if cur and len(cur) + len(sep) + len(ids) > slot_tokens:
            chunks.append(cur)
            cur = []
        cur = [*cur, *(sep if cur else []), *ids]
    if cur:
        chunks.append(cur)
    return [tok.decode(c) for c in chunks], [len(c) for c in chunks]


def truncate_middle(tok, text: str, budget: int) -> tuple[str, int]:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return text, len(ids)
    h = budget // 2
    return tok.decode(ids[:h]) + "\n[...]\n" + tok.decode(ids[-(budget - h):]), budget


# ---------------------------------------------------------------- arms
@torch.no_grad()
def gen_slots(student, messages, contexts, prompts, n_reps, max_new, enc_batch: int) -> dict:
    """StudentModel.generate_slots with the slots encoded in sub-batches (a 128k context is ~60
    slots; one padded encoder call over all of them is what we never do)."""
    segs = student.renderer.render_slots(json.dumps(messages), None, student._think_arg())
    k = len(segs) - 1
    assert k == len(contexts), (k, len(contexts))
    parts = []
    torch.cuda.synchronize()
    t_enc = time.perf_counter()
    for i in range(0, k, enc_batch):
        r, m, _ = student._encode_slots(contexts[i : i + enc_batch], prompts[i : i + enc_batch],
                                        n_reps[i : i + enc_batch])
        parts.extend((r[j][m[j]], m[j][m[j]]) for j in range(r.shape[0]))
    torch.cuda.synchronize()
    enc_s = time.perf_counter() - t_enc
    t_dec = time.perf_counter()
    reps, mask = parts[0][0][None], parts[0][1][None]
    extra = [[parts[j] for j in range(1, k)]]
    with student.core.autocast():
        batch = student.decoder.build_batch([segs[0]], reps, mask, [segs[-1]], pad_side="left",
                                            loss_on_suffix=False, extra=extra,
                                            mid_ids=[list(segs[1:-1])])
        gen = student.decoder.generate(batch, max_new_tokens=student._budget(max_new),
                                       eos_token_id=student.renderer.eos_id)
    torch.cuda.synchronize()
    return {"text": student.decode_answer(gen[0].tolist()),
            "prefix": int(batch.inputs_embeds.shape[1]),
            "reps": int(sum(p[0].shape[0] for p in parts)),
            "enc_s": enc_s, "dec_s": time.perf_counter() - t_dec}


def _sync(student) -> None:
    """CUDA sync for honest timings -- only when the student lives on a GPU (the CPU stub tests
    must never touch CUDA)."""
    dev = getattr(student, "device", None)
    if dev is not None and getattr(dev, "type", "") == "cuda":
        torch.cuda.synchronize()


class SlotRepCache:
    """Encoded slot reps by CONTENT (prompt, rep count, text) with an LRU bound on the total
    cached reps. The memory index's slots depend only on the session and the generic prompt, so
    every question over the same memory (and every turn of an episode) reuses them; a read's
    slot is reused by the later turns of its episode."""

    def __init__(self, student, enc_batch: int, max_reps: int = 400_000) -> None:
        from collections import OrderedDict

        self.student, self.enc_batch, self.max_reps = student, int(enc_batch), int(max_reps)
        self.store: OrderedDict = OrderedDict()
        self.n_reps = self.hits = self.misses = 0

    @staticmethod
    def key(text: str, prompt: str, n_reps: int) -> str:
        import hashlib

        return hashlib.sha1(f"{n_reps}\x00{prompt}\x00{text}".encode()).hexdigest()

    def encode(self, contexts, prompts, n_reps) -> tuple[list, int]:
        """[(reps, mask)] per slot (trimmed to its real reps) and the slot tokens encoded now."""
        keys = [self.key(c, p, int(r)) for c, p, r in zip(contexts, prompts, n_reps, strict=True)]
        todo = [j for j, k in enumerate(keys) if k not in self.store]
        self.hits += len(keys) - len(todo)
        self.misses += len(todo)
        n_tok = 0
        for i in range(0, len(todo), self.enc_batch):
            js = todo[i : i + self.enc_batch]
            r, m, nt = self.student._encode_slots([contexts[j] for j in js],
                                                  [prompts[j] for j in js],
                                                  [int(n_reps[j]) for j in js])
            n_tok += int(sum(nt))
            for q, j in enumerate(js):
                rep, mk = r[q][m[q]], m[q][m[q]]
                if keys[j] not in self.store:
                    self.store[keys[j]] = (rep, mk)
                    self.n_reps += int(rep.shape[0])
        out = [self.store[k] for k in keys]  # before eviction: this call's slots stay valid
        for k in keys:
            self.store.move_to_end(k)
        while self.n_reps > self.max_reps and len(self.store) > len(set(keys)):
            _k, (rep, _m) = self.store.popitem(last=False)
            self.n_reps -= int(rep.shape[0])
        return out, n_tok


def make_agent_generate(student, cache: SlotRepCache, max_new: int, stats: dict):
    """``runner.CompressedModel``'s generate_fn: the slotted conversation (tool list in the
    system turn, one sentinel per slot) through the student with cached slot reps."""

    @torch.no_grad()
    def generate(sr) -> str:
        segs = student.renderer.render_slots(json.dumps(sr.messages), sr.tools or None,
                                              student._think_arg())
        k = len(segs) - 1
        assert k == len(sr.contexts) and k >= 1, (k, len(sr.contexts))
        _sync(student)
        t0 = time.perf_counter()
        parts, n_tok = cache.encode(sr.contexts, sr.prompts, sr.n_reps)
        _sync(student)
        stats["enc_s"] = stats.get("enc_s", 0.0) + time.perf_counter() - t0
        stats["encoder_tokens_computed"] = stats.get("encoder_tokens_computed", 0) + n_tok
        t1 = time.perf_counter()
        with student.core.autocast():
            batch = student.decoder.build_batch(
                [segs[0]], parts[0][0][None], parts[0][1][None], [segs[-1]], pad_side="left",
                loss_on_suffix=False, extra=[[parts[j] for j in range(1, k)]],
                mid_ids=[list(segs[1:-1])])
            gen = student.decoder.generate(batch, max_new_tokens=student._budget(max_new),
                                           eos_token_id=student.renderer.eos_id)
        _sync(student)
        stats["dec_s"] = stats.get("dec_s", 0.0) + time.perf_counter() - t1
        stats["prefix_max"] = max(stats.get("prefix_max", 0), int(batch.inputs_embeds.shape[1]))
        # keep the tool-call markers (special tokens survive skip_special_tokens: checked)
        return student._decode_turn(gen[0].tolist())

    return generate


def run_agent_memory(student, cache: SlotRepCache, it: dict, args) -> dict:
    """One ``agent_memory`` episode -> the memory_eval record + GPU cost."""
    from bgkit2.agent.context import TemplateRenderer
    from bgkit2.agent.memory_eval import run_memory_episode
    from bgkit2.agent.runner import CompressedModel

    renderer = getattr(student, "_agent_renderer", None)
    if renderer is None:
        # the harness counts (prices) slots in ENCODER tokens, as training priced them; the
        # student re-renders the messages with the decoder's own template and sentinel
        renderer = student._agent_renderer = TemplateRenderer(student.renderer.ctx_tok)
    stats: dict = {}
    hits0 = cache.hits
    model = CompressedModel(make_agent_generate(student, cache, args.agent_max_new_tokens, stats),
                            renderer, name="student_agent_memory")
    rec = run_memory_episode(it, model, Path(args.out).parent / "agent_work",
                             renderer=renderer, max_turns=args.agent_max_turns)
    rec.update(encoder_tokens_computed=int(stats.get("encoder_tokens_computed", 0)),
               cache_hits=cache.hits - hits0, enc_s=float(stats.get("enc_s", 0.0)),
               dec_s=float(stats.get("dec_s", 0.0)), prefix=int(stats.get("prefix_max", 0)))
    return rec


@torch.no_grad()
def gen_text(student, messages, max_new: int) -> dict:
    ids = torch.tensor(student.renderer._apply(messages, None, student._think_arg()),
                       dtype=torch.long)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = student._generate_ids(ids, max_new)
    torch.cuda.synchronize()
    from bgkit2.agent.runner import split_think

    return {"text": split_think(out["text"])[1], "prefix": out["prefix_tokens"], "reps": 0,
            "enc_s": 0.0,
            "dec_s": time.perf_counter() - t0}


AGENT_MEANS = ("turns", "n_read", "n_search", "decoder_tokens", "encoder_tokens_nominal",
               "encoder_tokens_computed", "index_tokens", "index_reps", "read_tokens",
               "search_tokens", "cache_hits", "enc_s", "dec_s", "prefix")


def run_agent_memory_safe(student, cache, it: dict, args) -> dict:
    try:
        return run_agent_memory(student, cache, it, args)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"id": it["id"], "subset": it.get("subset", ""), "pred": "", "stop": "oom",
                "oom": True}


def agent_row(agg: dict, rows: list, it: dict, rec: dict) -> None:
    """An agent_memory record into the per-arm aggregate and the judge's rows
    (``id`` / ``arm`` / ``pred``: scripts/bgkit2_bench_judge.py reads nothing else)."""
    key = "agent_memory@agent"
    sc = score(it, rec.get("pred", ""))
    d = agg.setdefault(key, {})
    for k2, v in {**sc, "oom": float(rec.get("oom", False)),
                  "answered": float(bool(rec.get("pred"))),
                  **{m: rec[m] for m in AGENT_MEANS if m in rec}}.items():
        d.setdefault(k2, []).append(float(v))
    rows.append({"id": it["id"], "subset": it.get("subset", ""), "arm": key,
                 "pred": rec.get("pred", "")[:400], "gold": it["answers"][:2], **sc,
                 "stop": rec.get("stop", ""), "sessions_read": rec.get("sessions_read", []),
                 **{m: rec[m] for m in AGENT_MEANS if m in rec}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", default="512,2048")
    ap.add_argument("--arms", default="bgkit,text,base")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--slot-tokens", type=int, default=2048)
    ap.add_argument("--max-ratio", type=float, default=64.0)
    ap.add_argument("--enc-batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--cuda-fraction", type=float, default=0.5)
    ap.add_argument("--full-max-tokens", type=int, default=32768,
                    help="text_full/base_full only when the whole context fits")
    ap.add_argument("--summaries", default="", help="jsonl from bgkit2_bench_summaries.py")
    ap.add_argument("--agent-max-turns", type=int, default=6, help="agent_memory turn cap")
    ap.add_argument("--agent-max-new-tokens", type=int, default=160)
    ap.add_argument("--thinking", action="store_true",
                    help="always-thinking decoder (StudentModel.thinking): answers are scored "
                         "after the think block")
    ap.add_argument("--agent-cache-reps", type=int, default=400_000,
                    help="agent_memory slot-rep cache bound (reps; ~2 KB each in bf16)")
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir

    from bgkit2.agent.student import StudentModel
    from bgkit_core.host_memory_guard import arm, cap_cuda

    arm(10.0)
    cap_cuda(args.cuda_fraction)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[f"+experiment={args.experiment}"])
    student = StudentModel(cfg, Path(args.checkpoint), torch.device("cuda"))
    student.thinking = bool(args.thinking)
    student.max_ctx_tokens = max(student.max_ctx_tokens, args.slot_tokens)
    # two tokenizers (LFM2.5-8B-A1B decoder): slots are cut and priced in ENCODER tokens, the
    # text arms truncate and count in DECODER tokens; one tokenizer: the same object
    tok, ctx_tok = student.tok, student.renderer.ctx_tok
    split = student.renderer.split_tokenizers
    items = [json.loads(ln) for ln in open(args.items)]
    if args.n:
        items = items[: args.n]
    budgets = [int(b) for b in args.budgets.split(",")]
    summ: dict[tuple[str, int], dict] = {}
    if args.summaries:
        for ln in open(args.summaries):
            d = json.loads(ln)
            summ[(d["id"], int(d["budget"]))] = d
    arms = [a for a in args.arms.split(",") if a]
    agg: dict[str, dict[str, list[float]]] = {}
    agent_cache = SlotRepCache(student, args.enc_batch, args.agent_cache_reps)
    rows = []
    t0 = time.time()
    for n_done, it in enumerate(items):
        chunks, lens = pack_chunks(ctx_tok, it["segments"], args.slot_tokens)
        total = sum(lens)
        full_text = "\n\n".join(it["segments"])
        # the whole context as the DECODER reads it (text_full / base_full gate)
        total_dec = len(tok(full_text, add_special_tokens=False)["input_ids"]) if split else total
        max_new = 4 if it["scorer"] == "mcq" else args.max_new_tokens
        full_done: set[str] = set()
        for b in budgets:
            for a in arms:
                key = f"{a}@{b}"
                if a == "agent_memory":
                    if a in full_done or it.get("bench") not in MEMORY_BENCHES:
                        continue
                    full_done.add(a)
                    agent_row(agg, rows, it, run_agent_memory_safe(student, agent_cache, it, args))
                    continue
                if a in ("text_full", "base_full"):
                    if a in full_done:
                        continue  # budget-independent: run once, report under @full
                    full_done.add(a)
                    key = f"{a}@full"
                    if total_dec > args.full_max_tokens:
                        continue
                if a in ("summary", "base_summary") and (it["id"], b) not in summ:
                    continue
                try:
                    if a == "bgkit":
                        ratio = min(max(total / b, 1.0), args.max_ratio)
                        n_reps = [max(4, round(ln / ratio)) for ln in lens]
                        msgs = frame(it, "\n".join([SPLICE_SENTINEL] * len(chunks)))
                        out = gen_slots(student, msgs, chunks, [it["question"]] * len(chunks),
                                        n_reps, max_new, args.enc_batch)
                    else:
                        if a in ("text_full", "base_full"):
                            text = full_text
                        elif a in ("summary", "base_summary"):
                            text = summ[(it["id"], b)]["summary"]
                        else:
                            text, _n = truncate_middle(tok, full_text, b)
                        msgs = frame(it, text)
                        if a.startswith("base"):
                            with student.decoder.disable_adapter():
                                out = gen_text(student, msgs, max_new)
                        else:
                            out = gen_text(student, msgs, max_new)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    out = {"text": "", "prefix": 0, "reps": 0, "oom": True, "enc_s": 0.0,
                           "dec_s": 0.0}
                sc = score(it, out["text"])
                d = agg.setdefault(key, {})
                extra_cost = {}
                if a in ("summary", "base_summary"):
                    sd = summ[(it["id"], b)]
                    extra_cost = {"summ_in_tokens": sd.get("in_tokens", 0),
                                  "summ_out_tokens": sd.get("out_tokens", 0),
                                  "summ_s": sd.get("seconds", 0.0)}
                for k2, v in {**sc, "prefix": out["prefix"], "ctx_tokens": total,
                              "oom": float(out.get("oom", False)), "enc_s": out["enc_s"],
                              "dec_s": out["dec_s"], **extra_cost}.items():
                    d.setdefault(k2, []).append(float(v))
                rows.append({"id": it["id"], "subset": it["subset"], "arm": key,
                             "pred": out["text"][:200], "gold": it["answers"][:2], **sc,
                             "prefix": out["prefix"], "enc_s": round(out["enc_s"], 4),
                             "dec_s": round(out["dec_s"], 4), **extra_cost})
        if n_done % 10 == 0:
            m0 = next(iter(score(it, "")))
            head = {k: round(sum(v[m0]) / len(v[m0]), 3) for k, v in agg.items() if m0 in v}
            print(f"[{n_done + 1}/{len(items)}] {time.time() - t0:.0f}s {head}", flush=True)
    summary = {k: {m: round(sum(v) / len(v), 4) for m, v in d.items()} for k, d in agg.items()}
    rec = {"experiment": args.experiment, "checkpoint": args.checkpoint, "items": args.items,
           "n": len(items), "budgets": budgets, "slot_tokens": args.slot_tokens,
           "seconds": round(time.time() - t0), "summary": summary}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=1))
    Path(args.out).with_suffix(".rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    print(json.dumps(rec["summary"], indent=1), flush=True)


if __name__ == "__main__":
    main()
