"""Evaluation harness for the compressed-context coding agent (bgkit2).

CPU-testable skeleton of plan ``plans/agentic_stage_plan_2026_09_13.md``
problem 4: the same tools once returning full text and once returning
encoder-compressed results (a *slot* per compressible tool result), run over
SWE-bench Lite at an equal decoder-token budget.

    tools.py          tool registry (read / search / docs / zoom / write / edit / bash)
    context.py        conversation builder: slots, tool_slots rendering, compaction,
                      the LFM2.5 tool-call syntax and its parser
    runner.py         episode loop + decoder-token accounting + JSONL trajectory log
    swebench_lite.py  SWE-bench Lite instances, repo checkout, patch, harness stub
    swebench_loc.py   SWE-bench Lite localisation eval (history-free export, scoring)

Nothing here imports torch or loads a model: the bgkit2 model call sits behind
``CompressedModel(generate_fn=...)``.
"""

from .context import Conversation, RatioPolicy, Slot, ToolCall, parse_tool_calls
from .runner import (
    AgentTask,
    AssistantTurn,
    CompressedModel,
    EpisodeResult,
    FullTextModel,
    SlotRendering,
    TrajectoryLog,
    run_episode,
)
from .tools import ToolRegistry, ToolResult, ToolSpec

__all__ = [
    "AgentTask",
    "AssistantTurn",
    "CompressedModel",
    "Conversation",
    "EpisodeResult",
    "FullTextModel",
    "RatioPolicy",
    "Slot",
    "SlotRendering",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "TrajectoryLog",
    "parse_tool_calls",
    "run_episode",
]
