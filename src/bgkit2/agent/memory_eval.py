"""The ``agent_memory`` bench arm's CPU logic (scripts/bgkit2_bench.py adds the GPU model and the
index cache): a memory benchmark item (LongMemEval / LoCoMo: dated session segments + a question)
becomes a ``MemoryStore``; the student runs the ordinary ``run_episode`` loop with ONLY the
memory tools, the opening turn being the always-on memory index + the question
(``memory.add_memory_index``) -- the same framing ``scripts/bgkit2_build_memory_agent_store.py``
writes as training rows.

Per episode the record carries the answer, turns, sessions read, decoder tokens (``run_episode``'s
accounting: every turn's prefix with slots at their rep counts + generated tokens) and the
encoder tokens NOMINALLY needed: the index once (its slots are question-independent, so a cache
pays them once per memory, not per question or turn) plus every read / search result once (both
are compressible slots, like every tool result but the opt-in verbatim controls).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from bgkit2.data.memory_framing import MEMORY_AGENT_SYSTEM, MEMORY_RATIOS, MEMORY_TOOL_NAMES

from .context import DEFAULT_RATIOS, Conversation, RatioPolicy, Renderer
from .memory import MemoryStore, memory_opening
from .runner import AgentTask, Model, TrajectoryLog, run_episode
from .tools import ToolRegistry, ToolResult

AGENT_MEMORY_ARM = "agent_memory"
MEMORY_SLOT_MAX_TOKENS = 8192  # = the encoder's max_ctx (task_max_ctx_tokens)


def memory_ratios() -> RatioPolicy:
    return RatioPolicy({**DEFAULT_RATIOS, **MEMORY_RATIOS})


def memory_task(item: dict, work_dir: str | Path) -> AgentTask:
    q = str(item["question"]).strip()
    return AgentTask(task_id=str(item["id"]), instruction=q, repo_dir=str(work_dir),
                     system_prompt=MEMORY_AGENT_SYSTEM, summary=" ".join(q.split())[:200])


class MemoryEpisodeTools:
    """The memory tools of one episode, recording what the agent searched and read."""

    def __init__(self, reg: ToolRegistry, store: MemoryStore, renderer: Renderer) -> None:
        self.reg, self.store, self.renderer = reg, store, renderer
        self.read: list[str] = []  # session ids, first-read order
        self.n_search = self.n_read = self.n_errors = 0
        self.read_tokens = self.search_tokens = 0  # each compressible result once

    def specs(self) -> list[dict]:
        return self.reg.specs()

    def __getattr__(self, name):
        return getattr(self.reg, name)

    def call(self, name: str, args: dict | None = None) -> ToolResult:
        res = self.reg.call(name, args)
        self.n_search += name == "memory_search"
        self.n_read += name == "memory_read"
        self.n_errors += not res.ok
        if res.ok and name == "memory_read":
            s = self.store.resolve((args or {}).get("session"))
            if s is not None and s.id not in self.read:
                self.read.append(s.id)
        if res.ok and res.compressible:
            n = self.renderer.count(res.text)
            if name == "memory_read":
                self.read_tokens += n
            else:
                self.search_tokens += n
        return res


def build_memory_tools(store: MemoryStore, work_dir: str | Path, renderer: Renderer,
                       k: int = 5) -> MemoryEpisodeTools:
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    reg = ToolRegistry(str(work_dir))
    reg.attach_memory(store, k=k)
    reg.keep_only(MEMORY_TOOL_NAMES)
    return MemoryEpisodeTools(reg, store, renderer)


def run_memory_episode(
    item: dict,
    model: Model,
    work_dir: str | Path,
    ratios: RatioPolicy | None = None,
    renderer: Renderer | None = None,
    max_turns: int = 6,
    k: int = 5,
    slot_max_tokens: int | None = MEMORY_SLOT_MAX_TOKENS,
    log: TrajectoryLog | None = None,
    index_hook: Callable[[Conversation], None] | None = None,
    **episode_kwargs,
) -> dict:
    """One ``agent_memory`` episode over the item's sessions -> record (no scores: the caller
    scores ``pred`` with the bench's own scorer / the LLM judge)."""
    renderer = renderer or model.renderer
    ratios = ratios or memory_ratios()
    store = MemoryStore.from_texts(list(item["segments"]))
    tools = build_memory_tools(store, work_dir, renderer, k=k)
    opening = memory_opening(store, str(item["question"]).strip(), slot_max_tokens)
    index: dict = {}

    def open_and_record(conv: Conversation, task: AgentTask) -> None:
        opening(conv, task)
        slots = conv.slots
        index.update(n_slots=len(slots), tokens=sum(s.n_tokens for s in slots),
                     reps=sum(s.n_reps for s in slots))
        if index_hook is not None:
            index_hook(conv)

    res = run_episode(model, tools, memory_task(item, work_dir), max_turns=max_turns,
                      ratios=ratios, renderer=renderer, log=log, slot_max_tokens=slot_max_tokens,
                      opening=open_and_record, **episode_kwargs)
    final = res.final_message.strip() if res.stop_reason == "final" else ""
    return {
        "id": item["id"],
        "bench": item.get("bench", ""),
        "subset": item.get("subset", ""),
        "arm": AGENT_MEMORY_ARM,
        "pred": final,
        "gold": list(item.get("answers") or [])[:3],
        "turns": res.turns,
        "stop": res.stop_reason,
        "n_sessions": len(store),
        "sessions_read": list(tools.read),
        "n_read": tools.n_read,
        "n_search": tools.n_search,
        "tool_errors": tools.n_errors,
        "index_slots": index.get("n_slots", 0),
        "index_tokens": index.get("tokens", 0),
        "index_reps": index.get("reps", 0),
        "read_tokens": tools.read_tokens,
        "search_tokens": tools.search_tokens,
        # the index once (cacheable, question-independent) + each read / search result once
        "encoder_tokens_nominal": index.get("tokens", 0) + tools.read_tokens + tools.search_tokens,
        "decoder_tokens": res.decoder_tokens,
        "compactions": res.n_compactions,
        "wall_s": round(res.wall_s, 2),
        "final_message": final[-1500:],
    }
