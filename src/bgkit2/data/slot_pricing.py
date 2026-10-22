"""Per-slot rep pricing for ``tool_slots`` rows: train == eval (2026-09-22).

At eval the harness prices every compressible slot itself -- ``RatioPolicy.reps(n_tokens, tool)``
(read 1/8, web_open 1/16, web_search / memory_search 1/4, ...) and the memory index at
``memory_framing.index_reps`` (4-24 per session). The trainer used to hand every slot of a row the
row's ONE ratio (``ratio.family_rep_budget`` / the step's band, computed from the PRIMARY slot's
length), so a memory row trained its index, search and read slots at the same compression and a
web row its search hits at the page ratio. ``ratio.slot_reps_mode`` (config) makes the trainer
price each slot like the harness does:

  row      today's behaviour (one ratio per row)
  meta     rows whose ``meta.slot_reps`` (the harness's own n_reps, written by
           ``Conversation.to_task_row`` callers that record them) has one entry per slot
  harness  ``meta`` where present, else the slot's tool inferred from its compression PROMPT
           (the harness's ``<head>: <focus>`` forms, ``prompt_tool``) priced with the harness
           ratios -- for stores built before slot_reps existed

Slots with no price (-1: unknown prompt, cached tree nodes / overview, compaction spines) keep the
row ratio. The trainer applies a log-uniform jitter band (``ratio.slot_reps_jitter``) around the
price so the model sees a range around the eval pricing.
"""

from __future__ import annotations

import math

from .memory_framing import MEMORY_INDEX_PROMPT, MEMORY_INDEX_TOOL, MEMORY_RATIOS, index_reps

SLOT_REPS_MODES = ("row", "meta", "harness")

# (prompt head, tool) in match order: longer / more specific heads first. The heads are the
# ``ToolRegistry`` / ``Conversation`` prompt forms ("<head>" or "<head>: <focus>").
_HEADS: tuple[tuple[str, str], ...] = (
    ("memory search ", "memory_search"),
    ("web search ", "web_search"),
    ("find ", "web_find"),   # "find <pattern> in <title>" (web.web_find_prompt)
    ("page ", "web_open"),
    ("session ", "memory_read"),
    ("search ", "search"),
    ("read ", "read"),
    ("docs ", "docs"),
    ("bash: ", "bash"),
    ("bash ", "bash"),   # tool_slots_synth_*_mapped writes "bash <cmd>"
    ("compaction: ", "compaction"),   # a FLAT compaction slot (spines are unpriced by content)
    ("zoom ", "zoom"),   # "zoom <path> lines a-b" (tools.zoom_prompt)
    ("list files under", "ls"),
    ("map of the repository", "overview"),
)
# never priced here: tree payloads and cached tree nodes (browse). A compaction SPINE prices itself
# (its layout); the caller marks spine slots unpriced by content.
_UNPRICED_HEADS = ("encode the repository tree", "contents of")


def harness_ratios() -> dict[str, int]:
    """The ratios the eval harness prices with: ``context.DEFAULT_RATIOS`` + the memory tools.
    web_search / web_open default to scripts/bgkit2_eval_web.py's ``--ratios`` (4 / 16) where
    DEFAULT_RATIOS does not (yet) list them."""
    from ..agent.context import DEFAULT_RATIOS

    return {"web_search": 4, "web_open": 16, **DEFAULT_RATIOS, **MEMORY_RATIOS}


def prompt_tool(prompt: str) -> str | None:
    """The tool whose result carries this compression prompt, or None when unknown / unpriced."""
    p = (prompt or "").strip()
    if not p:
        return None
    if p == MEMORY_INDEX_PROMPT:
        return MEMORY_INDEX_TOOL
    if p.startswith(_UNPRICED_HEADS):
        return None
    for head, tool in _HEADS:
        if p.startswith(head) or p == head.strip():
            return tool
    return None


def price(tool: str | None, n_tokens: int, ratios: dict[str, int] | None = None,
          default: int = 16) -> int:
    """The harness's rep count for a slot (-1 = not priced)."""
    if tool is None or n_tokens <= 0:
        return -1
    if tool == MEMORY_INDEX_TOOL:
        return index_reps(n_tokens)
    r = (ratios or harness_ratios()).get(tool, default)
    return max(1, math.ceil(n_tokens / int(r)))


def row_slot_reps(meta: dict, prompts: list[str], n_tokens: list[int],
                  unpriced: list[bool] | None = None) -> tuple[list[int], str]:
    """Per-slot prices of one ``tool_slots`` row and their source: ``meta.slot_reps`` when it has
    one entry per slot ("meta"), else the prompt-inferred harness prices ("prompt"; -1 where
    unknown). ``unpriced[j]`` forces -1 (cached / spine slots)."""
    unpriced = unpriced or [False] * len(prompts)
    sr = meta.get("slot_reps") if isinstance(meta, dict) else None
    if isinstance(sr, list) and len(sr) == len(prompts):
        return [(-1 if u else int(x)) for x, u in zip(sr, unpriced, strict=True)], "meta"
    ratios = harness_ratios()
    out = [(-1 if u else price(prompt_tool(p), n, ratios))
           for p, n, u in zip(prompts, n_tokens, unpriced, strict=True)]
    return out, "prompt"
