"""Rendering-time phrasing diversity for task stores.

Every task store bakes ONE phrasing per family into the parquet: the decoder
framing ("Context:\\n<reps>\\n{question}"), the encoder prompt ("Compress the
passage to answer: {q}") and, for the tool framing, one tool name. A decoder
adapter trained on a single wording learns the wording (P19d: the no-context
floor fell 1.7 nats). This module varies all three at render time so no data
rebuild is needed and eval can score HELD-OUT phrasings the model never saw:

    canonical   the stored phrasing (index 0 of each bank; what P19a-P19e saw)
    train       sampled from the train slice of each bank
    heldout     sampled from the held-out slice (never used in training)

Draws are deterministic per (dataset seed, row index) — see ``TaskDataset`` —
so an eval arm renders the same phrasing for the same row every time.
BenchPress's own templates are NOT in any bank: they stay the external test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .templates import SPLICE_SENTINEL

# {ctx} = where the reps go; {q} = the instruction. Index 0 is the stored form.
CHAT_WRAPPERS: tuple[str, ...] = (
    "Context:\n{ctx}\n{q}",
    "{ctx}\n\n{q}",
    "{q}\n\n{ctx}",
    "Document:\n{ctx}\n\nQuestion: {q}",
    "Use the material below to respond.\n\n{ctx}\n\n{q}",
    "Here is some source material:\n{ctx}\n\nBased on it: {q}",
    "{q}\n\nRefer to the following:\n{ctx}",
    "Passage:\n{ctx}\n\n{q}",
    "<document>\n{ctx}\n</document>\n\n{q}",
    "Read this first.\n{ctx}\n\nNow: {q}",
    "Excerpt:\n{ctx}\n\nTask: {q}",
    "{q}\nMaterial:\n{ctx}",
    # ---- held out (never trained on) ----
    "Notes:\n{ctx}\n\nQ: {q}",
    "Consider the text below.\n\n{ctx}\n\nThen answer: {q}",
    "[context]\n{ctx}\n[/context]\n{q}",
    "{q}\n\nSee:\n{ctx}",
)
CHAT_HELDOUT_FROM = 12

SYSTEM_MESSAGES: tuple[str, ...] = (
    "You are a helpful assistant.",
    "Answer using the provided material.",
    "You are a concise assistant. Ground your answer in the given text.",
    "You are a coding assistant working inside a repository.",
    # ---- held out ----
    "Respond briefly and only from the supplied context.",
    "You help the user with questions about documents they share.",
)
SYSTEM_HELDOUT_FROM = 4

# (tool name, argument key) for the tool_response framing. Index 0 = stored.
TOOL_NAMES: tuple[tuple[str, str], ...] = (
    ("read_file", "path"),
    ("cat", "path"),
    ("open_file", "file"),
    ("view", "path"),
    ("read", "filename"),
    # ---- held out ----
    ("show_file", "path"),
    ("get_file_contents", "file_path"),
)
TOOL_HELDOUT_FROM = 5

# Encoder compression prompts, keyed by family prefix. {q} = the instruction.
# Index 0 of each = the stored form (scripts/bgkit2_build_tasks.py).
PROMPTS: dict[str, tuple[str, ...]] = {
    "qa": (
        "Compress the passage to answer: {q}\n",
        "Keep what is needed to answer: {q}\n",
        "Question: {q}\nRetain the facts that answer it.\n",
        "Compress for this question: {q}\n",
        "{q}\n",
        "Preserve the parts of the text relevant to: {q}\n",
        # ---- held out ----
        "Condense the text, keeping whatever answers: {q}\n",
        "Focus the compression on: {q}\n",
    ),
    "file_overview": (
        "Compress this file so that its purpose, contents and role in the repository "
        "can be described.\n",
        "Keep what is needed to describe what this file does and defines.\n",
        "Compress for a summary of this file's purpose and structure.\n",
        "Retain the parts that show what this file is for and what it depends on.\n",
        # ---- held out ----
        "Condense this file so its role in the codebase can be explained.\n",
    ),
    "compaction": (
        "Compact this earlier part of the session so the task can be continued.\n",
        "Keep what a continuation of this session would need.\n",
        "Compress the session history, retaining decisions, state and open items.\n",
        # ---- held out ----
        "Condense this history so the work can carry on from it.\n",
    ),
}
PROMPT_HELDOUT_FROM: dict[str, int] = {"qa": 6, "file_overview": 4, "compaction": 3}

SPLITS = ("canonical", "train", "heldout")


def _slice(bank: tuple, heldout_from: int, split: str) -> tuple:
    if split == "canonical":
        return bank[:1]
    if split == "train":
        return bank[:heldout_from]
    if split == "heldout":
        return bank[heldout_from:]
    raise ValueError(f"unknown phrasing split {split!r}; expected one of {SPLITS}")


def prompt_key(family: str) -> str | None:
    if family.startswith("qa_"):
        return "qa"
    for k in PROMPTS:
        if family == k or family.startswith(k + "_"):
            return k
    return None


@dataclass(frozen=True)
class FramingBank:
    """Draws a phrasing for each of the three surfaces from one split.

    ``p_system``: probability of prepending a system message (train/heldout
    only). ``p_no_prompt``: probability of handing the encoder NO prompt —
    the within-model measurement of what prompt conditioning buys."""

    split: str = "train"
    p_system: float = 0.3
    p_no_prompt: float = 0.1

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"unknown phrasing split {self.split!r}; expected one of {SPLITS}")

    def _pick(self, rng: np.random.Generator, bank: tuple, heldout_from: int):
        cands = _slice(bank, heldout_from, self.split)
        return cands[int(rng.integers(len(cands)))]

    def chat_messages(self, rng: np.random.Generator, instruction: str) -> list[dict]:
        wrap = self._pick(rng, CHAT_WRAPPERS, CHAT_HELDOUT_FROM)
        msgs = [{"role": "user", "content": wrap.format(ctx=SPLICE_SENTINEL, q=instruction)}]
        if self.split != "canonical" and rng.random() < self.p_system:
            sys_msg = self._pick(rng, SYSTEM_MESSAGES, SYSTEM_HELDOUT_FROM)
            msgs.insert(0, {"role": "system", "content": sys_msg})
        return msgs

    def tool_messages(
        self, rng: np.random.Generator, instruction: str, tool_name: str, tool_args: dict | None
    ) -> list[dict]:
        name, key = self._pick(rng, TOOL_NAMES, TOOL_HELDOUT_FROM)
        args = dict(tool_args or {})
        if self.split != "canonical" and "path" in args and key != "path":
            args = {key: args.pop("path"), **args}
        if self.split == "canonical":
            name = tool_name or name
        return [
            {"role": "user", "content": instruction},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}],
            },
            {"role": "tool", "content": SPLICE_SENTINEL},
        ]

    def prompt(self, rng: np.random.Generator, family: str, stored: str, instruction: str) -> str:
        """The encoder prompt: a bank phrasing for known families, the stored
        one otherwise; empty with ``p_no_prompt`` (train/heldout)."""
        if self.split != "canonical" and rng.random() < self.p_no_prompt:
            return ""
        key = prompt_key(family)
        if key is None or self.split == "canonical":
            return stored
        return self._pick(rng, PROMPTS[key], PROMPT_HELDOUT_FROM[key]).format(q=instruction)
