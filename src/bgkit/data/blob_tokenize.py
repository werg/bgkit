"""Tokenization bridge for blob samples (compaction sampler -> trainer).

Renders a :class:`~bgkit.data.compaction_sampler.CompactionSample` through the
decoder's chat template and returns flat token ids, a loss mask covering only
the target assistant turn, and the token spans of every blob splice sentinel
(to be replaced by projected survivor embeddings at forward time — same
embedding-space splice primitive as Phase 2's ``BGKIT_SENTINEL``).

Follows the diff-rendering approach of
:func:`bgkit.data.bgkit_tool_template.tokenize_trajectory`: the target turn's
rendered text is isolated by rendering the template with and without the turn
and diffing the strings, so template framing stays exactly what the decoder
family expects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from bgkit.data.blob_format import BLOB_SPLICE_SENTINEL, blob_sentinel_positions
from bgkit.data.compaction_sampler import CompactionSample


@dataclass
class RenderedBlobSample:
    token_ids: torch.Tensor  # (L,) long
    loss_mask: torch.Tensor  # (L,) bool — True on the target turn's tokens
    blob_sentinel_spans: list[tuple[int, int]]  # [start, end) token span per blob
    sample: CompactionSample


def _normalize_tool_calls(tool_calls: list) -> list[dict]:
    """Normalize to the structure the Qwen chat template accepts.

    Trajectory corpora (SWE-Zero, Toucan) carry OpenAI-style tool calls whose
    ``function.arguments`` is a JSON *string*; the template iterates arguments
    as a mapping. Parse strings to dicts and force the ``type: function``
    wrapper (mirrors ``bgkit_tool_template._tool_call_assistant_message``).
    """
    out = []
    for tc in tool_calls:
        fn = dict(tc.get("function") or {})
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"_raw": args}
        fn["arguments"] = args if isinstance(args, dict) else {}
        fn.setdefault("name", "")
        out.append(
            {"type": "function", "function": {"name": fn["name"], "arguments": fn["arguments"]}}
        )
    return out


def _strip_role_dicts(messages: list[dict]) -> list[dict]:
    """Chat-template-safe copies: role/content (+normalized tool_calls)."""
    out = []
    for m in messages:
        d = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("tool_calls"):
            d["tool_calls"] = _normalize_tool_calls(m["tool_calls"])
        out.append(d)
    return out


def tokenize_blob_sample(
    tokenizer,
    sample: CompactionSample,
) -> RenderedBlobSample:
    prefix = _strip_role_dicts(sample.prefix_messages)
    full = prefix + _strip_role_dicts([sample.target_message])

    prefix_text = tokenizer.apply_chat_template(prefix, tokenize=False, add_generation_prompt=False)
    full_text = tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
    if not full_text.startswith(prefix_text):
        raise ValueError("chat template is not prefix-stable for this message list")

    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if full_ids[: len(prefix_ids)] != prefix_ids:
        # BPE boundary merge at the seam (encode is not concat-distributive):
        # fall back to masking from the first divergence point.
        div = 0
        for a, b in zip(prefix_ids, full_ids, strict=False):
            if a != b:
                break
            div += 1
        loss_start = div
    else:
        loss_start = len(prefix_ids)

    token_ids = torch.tensor(full_ids, dtype=torch.long)
    loss_mask = torch.zeros(len(full_ids), dtype=torch.bool)
    loss_mask[loss_start:] = True

    sentinel_ids = tokenizer.encode(BLOB_SPLICE_SENTINEL, add_special_tokens=False)
    starts = blob_sentinel_positions(full_ids, sentinel_ids)
    if len(starts) != len(sample.blobs):
        raise ValueError(
            f"expected {len(sample.blobs)} blob sentinels in rendered sample, found {len(starts)}"
        )
    spans = [(s, s + len(sentinel_ids)) for s in starts]
    # sentinels live in the prefix — they must never carry loss
    for s, e in spans:
        loss_mask[s:e] = False

    return RenderedBlobSample(
        token_ids=token_ids,
        loss_mask=loss_mask,
        blob_sentinel_spans=spans,
        sample=sample,
    )
