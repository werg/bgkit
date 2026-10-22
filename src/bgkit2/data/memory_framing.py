"""The ONE framing of conversational-memory tasks, shared by the memory task builder
(scripts/bgkit2_build_memory_store.py) and the showcase benchmark harness (scripts/bgkit2_bench.py)
so training rows and LongMemEval / LoCoMo prompts are the same text.

Why a shared module (2026-09-22): the bench framed LongMemEval with the generic QA system prompt and
"Context:", the memory rows with a memory system prompt and "Memory:"; P34 (no memory data) answered
74% of LongMemEval items in the bgkit arm with a hallucinated TOOL CALL instead of text -- dated
chat sessions in slots read as an agent log -- so its judge score measured the framing, not memory.
"""

from __future__ import annotations

MEMORY_SYSTEM = (
    "You are an assistant with a long-term memory of past conversations, given as dated "
    "sessions. Answer from the memory. If it does not contain the answer, say so."
)
MEMORY_HEADER = "Memory:"
MEMORY_BENCHES = frozenset({"longmemeval", "locomo"})


def memory_user(body: str, question: str) -> str:
    """The user turn: the memory body (slot sentinels or plain session text) then the question."""
    return f"{MEMORY_HEADER}\n{body}\n\nQuestion: {question}"


# ---------------------------------------------------------------------------
# Two-stage AGENTIC memory (2026-09-22): every session is encoded ONCE under a generic prompt
# (the always-on INDEX, cacheable: "encode once, ask many"), and the agent re-reads the sessions
# it needs with ``memory_read(session, focus)`` -- the question as the encoder prompt, which is
# where bgkit's gains come from (prompt-conditioned encoding, 1.3-2.0 nats). Shared by the
# training-row builder (scripts/bgkit2_build_memory_agent_store.py), the harness tools
# (bgkit2.agent.memory / ToolRegistry.attach_memory) and the bench arm (agent_memory in
# scripts/bgkit2_bench.py), so train == bench.
# ---------------------------------------------------------------------------

# The ONE generic encoder prompt of every index slot. Never question-dependent: an index slot's
# encoding is a function of (session text, this prompt, rep count) only, so it is cached across
# every question asked of the same memory.
MEMORY_INDEX_PROMPT = (
    "Index this conversation session for later lookup: who talked, the topics, and the facts, "
    "plans, preferences, numbers and events mentioned."
)
MEMORY_INDEX_HEADER = "Memory index (one compressed entry per past session):"
MEMORY_INDEX_TOOL = "memory_index"  # the slots' ``tool`` (prices them; never a callable tool)
MEMORY_TOOL_NAMES = ("memory_search", "memory_read")
MEMORY_AGENT_SYSTEM = (
    "You are an assistant with a long-term memory of past conversations. The memory index shows "
    "every past session with its id and date, in compressed form. To answer, read the sessions "
    "you need with `memory_read` (a session id such as S3, several calls for several sessions), "
    "or find them with `memory_search`. Give a `focus` saying what you are looking for; search "
    "results and read sessions may be summarised for you. Answer from the memory. If it does "
    "not contain the answer, say so. When you know the answer, reply with only the answer and "
    "no tool call."
)
# Index slots are paid for on EVERY turn of every question, reads once: the index is coarse,
# a read is fine. Index reps = ceil(tokens / MEMORY_INDEX_RATIO) clamped to [MIN, MAX] (a
# LongMemEval_s session of ~2.4k tokens gets 24, a ~400-token chat session 7); reads and searches
# go through the harness RatioPolicy at MEMORY_RATIOS. memory_search is a COMPRESSIBLE slot like
# web_search, at the same gentle ratio (4): the agent copies session ids out of its reps as it
# copies page ids out of web_search's (user, 2026-09-22). zoom and web_find are encoded too (their
# verbatim paths are opt-in eval controls).
MEMORY_INDEX_RATIO = 64
MEMORY_INDEX_MIN_REPS = 4
MEMORY_INDEX_MAX_REPS = 24
MEMORY_RATIOS = {"memory_read": 8, "memory_search": 4}
MEMORY_SNIPPET_WORDS = 40  # per-hit snippet of memory_search (after its "[S3] <date>" line)
ABSTAIN = "That was not mentioned in our conversations."


def index_reps(n_tokens: int) -> int:
    import math

    return max(MEMORY_INDEX_MIN_REPS,
               min(MEMORY_INDEX_MAX_REPS, math.ceil(max(0, n_tokens) / MEMORY_INDEX_RATIO)))


def session_date(text: str) -> str:
    """The date of a session from its header line: ``[session on 2023/05/20 (Sat) 08:23]`` (our
    stores, LongMemEval) or ``[session 3, 1:56 pm on 8 May, 2023]`` (LoCoMo); "" without one."""
    head = text.lstrip().split("\n", 1)[0].strip()
    if not (head.startswith("[") and head.endswith("]")):
        return ""
    head = head[1:-1].strip()
    if head.lower().startswith("session on "):
        return head[len("session on "):].strip()
    if head.lower().startswith("session"):
        rest = head[len("session"):].lstrip()
        return rest.split(",", 1)[1].strip() if "," in rest else rest
    return head


def index_label(session_id: str, date: str) -> str:
    """The verbatim line in front of an index slot: the id the agent passes to ``memory_read``."""
    return f"[{session_id}] {date}".rstrip() if date else f"[{session_id}]"


def read_prompt(date: str, focus: str) -> str:
    """The encoder prompt of a ``memory_read`` result: ``session <date>: <focus>``."""
    head = f"session {date}".rstrip() if date else "session"
    return f"{head}: {focus}" if focus else head


def search_prompt(query: str, focus: str) -> str:
    """The encoder prompt of a ``memory_search`` result (the harness's ``<head>: <focus>``)."""
    head = f"memory search {query}".rstrip()
    return f"{head}: {focus}" if focus else head


def memory_agent_user(question: str) -> tuple[str, str]:
    """(text before the index slots, text after them) of the opening user turn."""
    return f"{MEMORY_INDEX_HEADER}\n", f"\n\nQuestion: {question}"
