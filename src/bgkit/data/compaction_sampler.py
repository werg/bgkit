"""Compaction sampling over chat trajectories (Family A, trajectory SFT).

Turns a long chat trajectory (OpenAI-style ``role``/``content``/``tool_calls``
message list — e.g. SWE-Zero OpenHands rows or Toucan SFT rows) into training
samples where a prefix of the history is replaced by bgkit compaction blobs
and the model is supervised on a later assistant turn
(`plans/capability_packaging_2026_08_20.md` §4).

Structure of every sample::

    [anchor: system + initial user task]
    [ONE history-replacement user message containing 1..n_blobs
     <bgkit_blob kind=compaction> blocks]          <- compressed at forward time
    [live window: recent messages kept as raw text]
    -> target assistant message (loss)

Design points:

- Compaction boundaries land on *turn* boundaries only (an assistant message
  and its following tool responses stay together).
- The anchor (system prompt + first user message) is never compacted —
  matching how real harnesses compact.
- Per-segment mode renders one blob per segment; merged mode renders a
  single blob covering the whole compacted span. Both are emitted from the
  same spec so inference can use either.
- ``source_ref`` spans refer to message indices in the *original*
  trajectory; the trainer resolves them to text and encodes them. This
  module never duplicates the compacted text into the sample.
- Recall-probe samples append an explicit user question about a fact
  (file path / function name) that only occurs inside the compacted span,
  with the fact string as the target answer — forcing load-bearing detail
  through the bottleneck rather than gist.

Tokenization and loss masking are downstream concerns (the chat-template
machinery in :mod:`bgkit.data.bgkit_tool_template` and the trainer); this
module is pure message-list surgery.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from bgkit.data.blob_format import BgkitBlob, render_header

Message = dict


_PATH_RE = re.compile(r"(?<![\w/])(/?[\w.\-]+(?:/[\w.\-]+)+\.[A-Za-z]{1,6})(?![\w/])")
_DEF_RE = re.compile(r"\b(?:def|class)\s+([A-Za-z_]\w+)")


def turn_boundaries(messages: list[Message], anchor_len: int) -> list[int]:
    """Indices where a new assistant turn starts, after the anchor.

    A "turn" is an assistant message plus its following non-assistant
    responses (tool/user). Compaction may only cut at these indices, so an
    assistant message is never separated from the tool results it caused.
    """
    return [i for i in range(anchor_len, len(messages)) if messages[i].get("role") == "assistant"]


def detect_anchor_len(messages: list[Message]) -> int:
    """Anchor = leading system message(s) + the first user message."""
    n = 0
    while n < len(messages) and messages[n].get("role") == "system":
        n += 1
    if n < len(messages) and messages[n].get("role") == "user":
        n += 1
    return n


def _span_stats(messages: list[Message]) -> str:
    n_tool = sum(1 for m in messages if m.get("role") == "tool")
    n_asst = sum(1 for m in messages if m.get("role") == "assistant")
    return f"{n_asst} assistant turns, {n_tool} tool results"


def extract_probe_facts(messages: list[Message]) -> list[tuple[str, str]]:
    """(kind, fact) pairs recoverable only by reading the span content."""
    text = "\n".join(str(m.get("content") or "") for m in messages)
    facts: list[tuple[str, str]] = []
    facts.extend(("path", p) for p in dict.fromkeys(_PATH_RE.findall(text)))
    facts.extend(("symbol", s) for s in dict.fromkeys(_DEF_RE.findall(text)))
    return facts


@dataclass(frozen=True)
class CompactionSample:
    """One SFT sample. ``prefix_messages`` ends right before the target."""

    prefix_messages: list[Message]
    target_message: Message
    blobs: list[BgkitBlob]  # encode specs, aligned with blob blocks in prefix
    trajectory_id: str
    compacted_span: tuple[int, int]  # original message-index span [a, b)
    mode: str  # "segments" | "merged"
    qtype: str = "continuation"  # or "recall_probe"
    probe_fact: tuple[str, str] | None = None
    meta: dict = field(default_factory=dict)


def build_sample(
    messages: list[Message],
    *,
    trajectory_id: str,
    target_index: int,
    live_window_turns: int,
    n_blobs: int,
    mode: str = "segments",
) -> CompactionSample | None:
    """Compact everything between the anchor and the live window.

    ``target_index`` must point at an assistant message; ``live_window_turns``
    assistant turns immediately before it stay raw. Returns None when the
    trajectory is too short to leave a non-empty compacted span.
    """
    if messages[target_index].get("role") != "assistant":
        raise ValueError("target_index must point at an assistant message")
    anchor_len = detect_anchor_len(messages)
    bounds = [b for b in turn_boundaries(messages, anchor_len) if b < target_index]
    if len(bounds) <= live_window_turns:
        return None
    live_start = bounds[-live_window_turns] if live_window_turns > 0 else target_index
    span = (anchor_len, live_start)
    if span[1] - span[0] < 1:
        return None

    seg_bounds = [b for b in bounds if span[0] <= b < span[1]]
    if mode == "merged" or n_blobs <= 1 or len(seg_bounds) < 2:
        segments = [span]
        mode = "merged" if mode == "merged" or n_blobs <= 1 else "segments"
    else:
        cuts = [span[0], *sorted(
            seg_bounds[i * len(seg_bounds) // n_blobs] for i in range(1, n_blobs)
        )]
        cuts = [*sorted(set(cuts)), span[1]]
        segments = list(zip(cuts[:-1], cuts[1:], strict=False))

    blobs: list[BgkitBlob] = []
    for a, b in segments:
        seg_msgs = messages[a:b]
        blobs.append(
            BgkitBlob(
                kind="compaction",
                header=render_header(
                    "compaction",
                    span=f"messages {a}-{b - 1}",
                    stats=_span_stats(seg_msgs),
                ),
                source_ref=f"traj:{trajectory_id}:msgs:{a}-{b}",
            )
        )

    history_msg: Message = {
        "role": "user",
        "content": "[Earlier conversation compacted]\n" + "\n".join(b.render() for b in blobs),
    }
    prefix = list(messages[:anchor_len]) + [history_msg] + list(messages[live_start:target_index])
    return CompactionSample(
        prefix_messages=prefix,
        target_message=messages[target_index],
        blobs=blobs,
        trajectory_id=trajectory_id,
        compacted_span=span,
        mode=mode,
    )


def build_probe_sample(
    base: CompactionSample,
    messages: list[Message],
    rng: random.Random,
) -> CompactionSample | None:
    """Recall probe: ask for a fact that exists only inside the compacted span."""
    a, b = base.compacted_span
    span_facts = extract_probe_facts(messages[a:b])
    outside = {f for _, f in extract_probe_facts(messages[:a] + messages[b : len(messages)])}
    candidates = [(k, f) for k, f in span_facts if f not in outside]
    if not candidates:
        return None
    kind, fact = rng.choice(candidates)
    if kind == "path":
        q = (
            "Earlier in this session (now compacted) you worked with a file "
            f"whose name ends with '{fact.rsplit('/', 1)[-1]}'. Quote its full path."
        )
    else:
        q = (
            "Earlier in this session (now compacted) a function or class "
            f"definition starting with '{fact[:3]}' was shown. State its exact name."
        )
    probe_prefix = base.prefix_messages + [{"role": "user", "content": q}]
    return CompactionSample(
        prefix_messages=probe_prefix,
        target_message={"role": "assistant", "content": fact},
        blobs=base.blobs,
        trajectory_id=base.trajectory_id,
        compacted_span=base.compacted_span,
        mode=base.mode,
        qtype="recall_probe",
        probe_fact=(kind, fact),
    )


def sample_trajectory(
    messages: list[Message],
    *,
    trajectory_id: str,
    rng: random.Random,
    samples_per_trajectory: int = 4,
    live_window_turns_choices: tuple[int, ...] = (1, 2, 4),
    n_blobs_choices: tuple[int, ...] = (1, 2, 4),
    probe_fraction: float = 0.25,
) -> list[CompactionSample]:
    """Draw several compaction samples (and probes) from one trajectory."""
    anchor_len = detect_anchor_len(messages)
    targets = turn_boundaries(messages, anchor_len)
    if len(targets) < 3:
        return []
    out: list[CompactionSample] = []
    for _ in range(samples_per_trajectory):
        target_index = rng.choice(targets[2:])
        sample = build_sample(
            messages,
            trajectory_id=trajectory_id,
            target_index=target_index,
            live_window_turns=rng.choice(live_window_turns_choices),
            n_blobs=rng.choice(n_blobs_choices),
            mode="merged" if rng.random() < 0.5 else "segments",
        )
        if sample is None:
            continue
        out.append(sample)
        if rng.random() < probe_fraction:
            probe = build_probe_sample(sample, messages, rng)
            if probe is not None:
                out.append(probe)
    return out
