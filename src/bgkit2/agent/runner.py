"""The episode loop with decoder-token accounting and a JSONL trajectory log.

Two model interfaces over the same ``Conversation``:

* ``FullTextModel`` renders every tool result as text and calls an LLM stub
  ``llm(prompt_text) -> generation``.
* ``CompressedModel`` renders with sentinels and calls
  ``generate_fn(SlotRendering) -> generation``; the real bgkit2 call lives
  behind that argument (see ``SlotRendering`` for the recipe). Nothing here
  loads a model.

Accounting (per turn, ``TurnRecord``): decoder tokens consumed = prefix
positions (a slot counts as its rep count, ``ceil(n_tokens / ratio[tool])``)
+ generated tokens; tokens saved = sum over compressible results of
``n_tokens - n_reps``. ``EpisodeResult`` sums them. The claim under test is
effective context per decoder token, so both arms report the same numbers.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from .context import (
    Conversation,
    RatioPolicy,
    Renderer,
    ToolCall,
    parse_tool_calls,
)
from .tools import ToolRegistry

DEFAULT_SYSTEM = (
    "You are a software engineer working in a repository checkout. Use the tools to "
    "understand the code, make the change, and run the tests. `zoom` looks closely at a "
    "line range; results may be summarised for you. Give a `focus` with `read`, `zoom`, "
    "`search` and `docs` to say what you are looking for. When the task is complete, "
    "reply with a short summary and no tool call."
)


@dataclass
class AgentTask:
    task_id: str
    instruction: str
    repo_dir: str
    system_prompt: str = DEFAULT_SYSTEM
    summary: str = ""  # feeds the compaction prompt; defaults to the instruction head
    repo: str = ""  # "<owner>/<name>": keys the cached whole-repo overview

    def compaction_summary(self) -> str:
        return self.summary or " ".join(self.instruction.split())[:200]


@dataclass
class AssistantTurn:
    content: str
    tool_calls: list[ToolCall]
    raw: str
    n_tokens: int
    parse_error: str | None = None
    reasoning: str | None = None


class Model(Protocol):
    name: str
    compressed: bool
    renderer: Renderer

    def generate(self, conv: Conversation) -> AssistantTurn: ...


@dataclass
class SlotRendering:
    """What the compressed decoder gets — the ``tool_slots`` layout.

    To run it through bgkit2 (GPU, not implemented here):
    ``segs = ChatRenderer.render_slots(json.dumps(messages))`` gives K+1 token
    segments; encode ``contexts[j]`` with ``prompts[j]`` to reps ``R_j`` at the
    slot's ratio; ``SpliceDecoder.build_batch([segs[0]], R_0, mask_0,
    [segs[-1]], extra=[[(R_j, m_j) for j >= 1]], mid_ids=[segs[1:-1]],
    pad_side="left")`` then ``generate``. Plan problem 2 (extend a live KV
    cache with new reps instead of re-prefilling) is what makes this cheap.
    """

    messages: list[dict]
    contexts: list[str]
    prompts: list[str]
    tools: list[dict]
    text: str  # the rendering with sentinels, for logging / stubs
    n_reps: list[int] = field(default_factory=list)  # reps granted per slot (RatioPolicy)
    n_tokens: list[int] = field(default_factory=list)  # raw tokens per slot


class FullTextModel:
    compressed = False

    def __init__(self, llm: Callable[[str], str], renderer: Renderer, name: str = "full_text"):
        self.llm = llm
        self.renderer = renderer
        self.name = name

    def generate(self, conv: Conversation) -> AssistantTurn:
        raw = self.llm(conv.render_full())
        content, calls, err = parse_tool_calls(raw)
        return AssistantTurn(content, calls, raw, self.renderer.count(raw), err)


class ChatAPIModel:
    """A teacher behind an OpenAI-compatible chat endpoint (our HF teacher
    server, vLLM, llama-server): sends the conversation as messages + tools,
    receives ``content`` / ``tool_calls`` / ``reasoning_content`` already
    parsed by the server. The conversation's ``teacher_view`` decides whether
    tool results are sent raw or as digests."""

    compressed = False

    def __init__(
        self,
        url: str,
        renderer: Renderer,
        model: str = "teacher",
        name: str = "teacher",
        chat_template_kwargs: dict | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        timeout_s: float = 600.0,
        post: Callable[[str, dict, float], dict] | None = None,
    ):
        self.url = url.rstrip("/")
        self.renderer = renderer
        self.model = model
        self.name = name
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self._post = post or _http_post_json

    def generate(self, conv: Conversation) -> AssistantTurn:
        body = {
            "model": self.model,
            "messages": conv.messages_full(),
            "tools": conv.tools,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "chat_template_kwargs": self.chat_template_kwargs,
        }
        resp = self._post(f"{self.url}/v1/chat/completions", body, self.timeout_s)
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""
        raw = msg.get("raw") or content
        reasoning = msg.get("reasoning_content")
        err = msg.get("parse_error")
        calls = []
        for c in msg.get("tool_calls") or []:
            try:
                calls.append(ToolCall.from_message(c))
            except (ValueError, KeyError, TypeError) as exc:
                # a malformed argument string from the model: report, do not crash
                err = err or f"bad tool call arguments: {exc}"
        if "<ifm|tool_calls>" in content or ("think" in content and "<" in content):
            # A plain llama-server / vLLM-without-parser reply: K2 Horizon
            # markup is still in the content — parse it here.
            from bgkit2.agent.teacher import parse_k2_output

            parsed = parse_k2_output(content)
            content = parsed.content
            if not calls:
                calls = [ToolCall.from_message(c) for c in parsed.tool_calls]
            reasoning = reasoning or parsed.reasoning
            err = err or parsed.parse_error
        n = int(resp.get("usage", {}).get("completion_tokens") or self.renderer.count(raw))
        turn = AssistantTurn(content, calls, raw, n, err)
        turn.reasoning = reasoning
        return turn


def _http_post_json(url: str, body: dict, timeout_s: float) -> dict:
    import json as _json
    import urllib.request

    req = urllib.request.Request(
        url, data=_json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return _json.loads(r.read().decode())


class CompressedModel:
    compressed = True

    def __init__(
        self,
        generate_fn: Callable[[SlotRendering], str],
        renderer: Renderer,
        name: str = "compressed",
    ):
        self.generate_fn = generate_fn
        self.renderer = renderer
        self.name = name

    def generate(self, conv: Conversation) -> AssistantTurn:
        sr = SlotRendering(
            conv.messages_with_slots(),
            conv.contexts(),
            conv.prompts(),
            conv.tools,
            conv.render_slots(),
            n_reps=[s.n_reps for s in conv.slots],
            n_tokens=[s.n_tokens for s in conv.slots],
        )
        raw = self.generate_fn(sr)
        reasoning, visible = split_think(raw)
        content, calls, err = parse_tool_calls(visible)
        turn = AssistantTurn(content, calls, raw, self.renderer.count(raw), err)
        turn.reasoning = reasoning
        return turn


_THINK_BLOCK = re.compile(r"^\s*<think>\n?(.*?)\n?</think>\n?", re.S)


def split_think(raw: str) -> tuple[str | None, str]:
    """(reasoning, visible text) of a thinking student's turn (``StudentModel._decode_turn``
    keeps the reasoning as a leading ``<think>...</think>`` block); no block -> (None, raw).
    The reasoning goes back into the history as ``reasoning_content``."""
    m = _THINK_BLOCK.match(raw)
    if not m:
        return None, raw
    return m.group(1).strip() or None, raw[m.end() :].strip()


# ---------------------------------------------------------------------------
# Accounting + log
# ---------------------------------------------------------------------------


@dataclass
class TurnRecord:
    turn: int
    prefix_tokens_full: int
    prefix_tokens_compressed: int
    prefix_tokens_consumed: int
    generated_tokens: int
    tool_calls: list[dict]
    tool_result_tokens: int
    tokens_saved: int
    compacted: bool
    wall_s: float
    content: str
    parse_error: str | None = None


@dataclass
class EpisodeResult:
    task_id: str
    model: str
    compressed: bool
    turns: int
    decoder_tokens: int
    tool_result_tokens: int
    tokens_saved: int
    n_compactions: int
    stop_reason: str  # final | max_turns | budget | loop | empty
    wall_s: float
    final_message: str
    records: list[TurnRecord] = field(default_factory=list)
    task_row: dict | None = None
    patch: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


class TrajectoryLog:
    """JSONL: one ``{"kind": "turn", ...}`` line per turn and one
    ``{"kind": "episode", ...}`` per finished episode."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a")

    def write_turn(self, task_id: str, model: str, rec: TurnRecord) -> None:
        self._f.write(
            json.dumps({"kind": "turn", "task_id": task_id, "model": model, **asdict(rec)})
        )
        self._f.write("\n")
        self._f.flush()

    def write_row(self, row: dict) -> None:
        """A tool_slots task-store row (one per assistant turn): the
        conversation BEFORE the turn as the prefix, the turn as the target."""
        self._f.write(json.dumps({"kind": "row", **row}) + "\n")
        self._f.flush()

    def write_episode(self, result: EpisodeResult) -> None:
        d = result.to_json()
        d.pop("records", None)  # already logged per turn
        self._f.write(json.dumps({"kind": "episode", **d}) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> TrajectoryLog:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------


REPEAT_NOTICE = (
    "error: this call is identical to one you already made (turn {turn}); its result "
    "is unchanged and is all there is. Do not repeat calls: `zoom` on a line range, "
    "change the query, or give your final answer."
)
FINAL_DEMAND = (
    "You keep repeating calls you already made. Stop calling tools now and reply with "
    "your best final answer in the required form, even if you are not certain."
)
EMPTY_NOTICE = "Your reply was empty. Continue: call a tool, or give the final answer now."


def _call_signature(calls: list[ToolCall]) -> str | None:
    if not calls:
        return None
    return json.dumps([[c.name, c.args] for c in calls], sort_keys=True, default=str)


def run_episode(
    model: Model,
    tools: ToolRegistry,
    task: AgentTask,
    max_turns: int = 20,
    decoder_token_budget: int = 200_000,
    context_budget: int = 8192,
    ratios: RatioPolicy | None = None,
    renderer: Renderer | None = None,
    log: TrajectoryLog | None = None,
    digest_fn: Callable[[str, str], str] | None = None,
    row_family: str = "agent_tool_slots",
    row_split: str = "train",
    loop_abort_repeats: int = 3,
    max_empty_replies: int = 2,
    seed_reads: list[str] | None = None,
    overview: Callable[[AgentTask], tuple[str, str, int] | None] | None = None,
    compaction_mode: str = "flat",
    slot_max_tokens: int | None = None,
    opening: Callable[[Conversation, AgentTask], None] | None = None,
) -> EpisodeResult:
    """Run one task to completion, a turn cap, or budget exhaustion.

    ``context_budget`` is the per-turn prefix cap (compressed positions) above
    which the oldest rounds are compacted; ``decoder_token_budget`` caps the
    whole episode's consumed decoder tokens (the equal-budget comparison).

    Degenerate turns (teacher run 2, 2026-09-14: 33 of 71 episodes hit the cap,
    most by re-issuing one bash command a dozen times; 3 ended on an empty
    reply; run 3 interleaved the same `read` with other calls): a tool-call
    set identical to one ALREADY MADE in the episode is NOT re-executed — the
    model gets ``REPEAT_NOTICE`` (naming the earlier turn) and the episode
    at ``loop_abort_repeats`` repeats in total the model is told to stop and
    answer (``FINAL_DEMAND``, one more turn) and a further tool call aborts with
    ``stop_reason="loop"`` (run 3: 11 of 38 episodes looped at turn 15-29 with
    no answer at all; a best guess scores 0.5 for the right file); an empty
    reply (no content, no call) gets ``EMPTY_NOTICE`` and
    aborts with ``"empty"`` after ``max_empty_replies``. Neither kind of turn
    is written as a training row.

    ``seed_reads`` pre-loads the episode with a read of each named path, as a
    turn-(-1) assistant call plus its tool result, before the model sees the
    conversation. It exists because the arms are otherwise only comparable on
    episodes where the model happens to retrieve something: at P26 pause 3 the
    compressed and full-text arms returned an identical answer on 59 of 60
    localisation episodes and 36 of 60 repo_qa episodes, because the model
    read a bare basename, got an error, and never pulled a compressible result
    into context at all. Seeding the gold file makes the treatment apply to
    every episode and gives the task a real ceiling.

    ``opening(conv, task)`` builds the opening user turn itself instead of
    ``add_user(task.instruction)`` (the two-stage memory agent: the always-on
    memory INDEX slots + the question, ``bgkit2.agent.memory.add_memory_index``).
    """
    renderer = renderer or model.renderer
    # ``overview(task)`` -> (repo, path listing, n root reps) puts the ALWAYS-ON
    # repo overview in front of the task (None = this repo has no cached root).
    ov = overview(task) if overview is not None else None
    view = overview.tree(task) if ov is not None and hasattr(overview, "tree") else None
    if view is not None:
        tools.attach_tree(view)  # before the Conversation: `browse` joins the tool specs
        # With a browsable tree the text view of the opening slot is the ROOT
        # directory's listing, not every path of the repo: a teacher that reads
        # all paths up front never browses, and the student's root reps do not
        # carry whole paths (generated path exact 2.3%, 2026-09-19) -- the walk
        # is the behaviour to distil, and the full-text arm is its fair ceiling.
        ov = (ov[0], view.listing(""), ov[2])
    conv = Conversation(
        task.system_prompt, tools.specs(), renderer, ratios, task.compaction_summary(),
        compaction_mode=compaction_mode, slot_max_tokens=slot_max_tokens,
        # a TEXT model's compacted history is the tail of the folded rounds (half the context
        # budget); the student reads the encoded spine instead
        full_view_compaction_tokens=None if model.compressed else max(1, context_budget // 2),
    )
    if digest_fn is not None:
        # Teacher generation (user, 2026-09-14): the teacher never sees a full
        # tool result — a digest sub-agent compresses it under the call's
        # compression prompt and the teacher reads THAT; the raw text stays
        # in the slot for the student.
        conv.teacher_view = True
    if opening is not None:
        opening(conv, task)
    elif ov is not None:
        conv.add_user_with_overview(task.instruction, *ov)
    else:
        conv.add_user(task.instruction)
    records: list[TurnRecord] = []
    decoder_tokens = tool_tokens = saved = 0
    seeded_saved = seeded_tool_tokens = 0
    for path in seed_reads or []:
        call = ToolCall("read", {"path": path, "focus": task.instruction})
        res = tools.call(call.name, call.args)
        conv.add_assistant("", [call])
        slot = conv.add_tool_result(res.text, res.compressible, res.prompt, res.tool or call.name)
        seeded_tool_tokens += slot.n_tokens if slot else renderer.count(res.text)
        if slot is not None:
            seeded_saved += slot.n_tokens - slot.n_reps
    tool_tokens += seeded_tool_tokens
    saved += seeded_saved
    stop = "max_turns"
    final = ""
    seen_sigs: dict[str, int] = {}
    repeats = empties = 0
    demanded = False
    abort: str | None = None
    t_start = time.perf_counter()
    for turn in range(max_turns):
        t0 = time.perf_counter()
        compacted = conv.compact(context_budget, compressed=model.compressed)
        pre_full = conv.prefix_tokens(compressed=False)
        pre_c = conv.prefix_tokens(compressed=True)
        consumed = pre_c if model.compressed else pre_full
        if decoder_tokens + consumed > decoder_token_budget:
            stop = "budget"
            break
        out = model.generate(conv)
        decoder_tokens += consumed + out.n_tokens
        sig = _call_signature(out.tool_calls)
        repeated = sig is not None and sig in seen_sigs
        if sig is not None and not repeated:
            seen_sigs[sig] = turn
        empty = not out.tool_calls and not out.parse_error and not out.content.strip()
        if log is not None and not out.parse_error and not repeated and not empty:
            target = {"role": "assistant", "content": out.content}
            if out.tool_calls:
                target["tool_calls"] = [c.as_message() for c in out.tool_calls]
            log.write_row(
                conv.to_task_row(
                    target=[target],
                    family=row_family,
                    split=row_split,
                    meta={"task_id": task.task_id, "turn": turn, "model": model.name},
                )
            )
        conv.add_assistant(out.content, out.tool_calls, reasoning=out.reasoning)
        turn_tool_tokens = turn_saved = 0
        if out.parse_error:
            conv.add_tool_result(f"error: {out.parse_error}", compressible=False, prompt="")
        if repeated or (demanded and out.tool_calls):
            repeats += 1
            if demanded:
                abort = "loop"  # told to answer, called a tool anyway
            elif repeats >= loop_abort_repeats:
                demanded = True
                for _call in out.tool_calls:
                    conv.add_tool_result(FINAL_DEMAND, compressible=False, prompt="")
            else:
                notice = REPEAT_NOTICE.format(turn=seen_sigs[sig])
                for _call in out.tool_calls:  # one result per call keeps the template valid
                    conv.add_tool_result(notice, compressible=False, prompt="")
        elif empty:
            empties += 1
            if empties >= max_empty_replies:
                abort = "empty"
            else:
                conv.add_user(EMPTY_NOTICE)
        for call in [] if (repeated or abort) else out.tool_calls:
            res = tools.call(call.name, call.args)
            digest = None
            if res.cached is not None:
                # ``browse``: a cached tree node; its listing is already short
                # text, so the teacher reads it as is (no digest sub-agent).
                slot = conv.add_cached_tool_result(
                    res.text, res.cached[0], res.cached[1], res.prompt, res.tool or call.name
                )
            else:
                if digest_fn is not None and res.compressible:
                    digest = digest_fn(res.prompt, res.text)
                slot = conv.add_tool_result(
                    res.text, res.compressible, res.prompt, res.tool or call.name, digest=digest
                )
            n = slot.n_tokens if slot else renderer.count(res.text)
            turn_tool_tokens += n
            if slot is not None:
                turn_saved += slot.n_tokens - slot.n_reps
        tool_tokens += turn_tool_tokens
        saved += turn_saved
        rec = TurnRecord(
            turn=turn,
            prefix_tokens_full=pre_full,
            prefix_tokens_compressed=pre_c,
            prefix_tokens_consumed=consumed,
            generated_tokens=out.n_tokens,
            tool_calls=[{"name": c.name, "args": c.args} for c in out.tool_calls],
            tool_result_tokens=turn_tool_tokens,
            tokens_saved=turn_saved,
            compacted=compacted,
            wall_s=time.perf_counter() - t0,
            content=out.content,
            parse_error=out.parse_error,
        )
        records.append(rec)
        if log is not None:
            log.write_turn(task.task_id, model.name, rec)
        if abort is not None:
            stop = abort
            break
        if not out.tool_calls and not out.parse_error and not empty:
            stop = "final"
            final = out.content
            break
    result = EpisodeResult(
        task_id=task.task_id,
        model=model.name,
        compressed=model.compressed,
        turns=len(records),
        decoder_tokens=decoder_tokens,
        tool_result_tokens=tool_tokens,
        tokens_saved=saved,
        n_compactions=conv.n_compactions,
        stop_reason=stop,
        wall_s=time.perf_counter() - t_start,
        final_message=final,
        records=records,
        task_row=conv.to_task_row(meta={"task_id": task.task_id, "model": model.name}),
    )
    if log is not None:
        log.write_episode(result)
    return result
