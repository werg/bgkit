"""Teacher-side pieces that do not need a GPU in the loop.

``parse_k2_output`` turns a K2 Horizon completion into (reasoning, content,
tool_calls): the model emits an optional ``<think>...</think>`` block, then
content, then one ``<ifm|tool_calls>`` block holding ``<ifm|tool_call>``
entries — JSON objects when the chat template is rendered with
``tool_call_format="json"`` (what our server requests). The XML formats are
not parsed here.

``TeacherEngine`` is the model-agnostic core of ``scripts/bgkit2_teacher_server.py``
(load a HF causal LM with its own chat template, render messages + tools,
generate, parse); the FastAPI layer around it is thin. ``digest_prompt``
builds the digest sub-agent's request: the tool result compressed under the
call's compression prompt.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# K2 Horizon's thinking tags depend on reasoning_effort: <ifm|think> (high),
# <ifm|think_fast> (medium), <ifm|think_faster> (low); plain <think> for other models.
_THINK_TAG = r"(ifm\|think(?:_fast|_faster)?|think(?:_fast|_faster)?)"
_THINK_RE = re.compile(r"<" + _THINK_TAG + r">\n?(.*?)</\1>\s*", re.S)
_STRAY_THINK_RE = re.compile(r"</?" + _THINK_TAG + r">\s*")
_CALLS_RE = re.compile(r"<ifm\|tool_calls>(.*?)</ifm\|tool_calls>", re.S)
_CALL_RE = re.compile(r"<ifm\|tool_call>(.*?)</ifm\|tool_call>", re.S)


@dataclass
class ParsedOutput:
    reasoning: str
    content: str
    tool_calls: list[dict] = field(default_factory=list)  # OpenAI function-call dicts
    parse_error: str | None = None


def parse_k2_output(text: str) -> ParsedOutput:
    reasoning = ""
    m = _THINK_RE.search(text)
    if m:
        reasoning = m.group(2).strip()
        text = text[: m.start()] + text[m.end() :]
    # A server that already split the reasoning off can leave the closing tag
    # (llama-server + the K2 fork returns content "</ifm|think_faster>").
    text = _STRAY_THINK_RE.sub("", text)
    calls: list[dict] = []
    err = None
    mc = _CALLS_RE.search(text)
    if mc:
        for raw in _CALL_RE.findall(mc.group(1)):
            raw = raw.strip()
            try:
                d = json.loads(raw)
                name = str(d["name"])
                args = d.get("arguments") or {}
                if not isinstance(args, dict):
                    raise ValueError("arguments is not an object")
            except (ValueError, KeyError, TypeError) as exc:
                err = f"bad tool call {raw[:80]!r}: {exc}"
                continue
            calls.append(
                {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            )
        text = text[: mc.start()] + text[mc.end() :]
    elif "<ifm|tool_calls>" in text:
        err = "unterminated tool_calls block"
    return ParsedOutput(reasoning, text.strip(), calls, err)


DIGEST_SYSTEM = (
    "You are a summariser, not an agent: you have NO tools and must never emit a tool "
    "call. You are given the purpose of a tool call and its raw output. Reply with a "
    "compact PLAIN-TEXT digest of the raw output: keep every detail the purpose needs "
    "(exact identifiers, file paths, line numbers, values, error messages, the code "
    "lines that matter, verbatim where exact text matters) and drop everything else. "
    "Keep compact formats as they are (e.g. one `path:line: text` per line). "
    "When the output is source code, name the enclosing function or class of the "
    "lines shown and the path:line range; copy identifiers verbatim, never paraphrase "
    "them ('the download function' loses the name the reader needs). "
    "No JSON, no markdown fences, no headings, no commentary, no invention. "
    "Aim for at most 150 words unless the purpose genuinely needs more."
)

# Stop sequences for digest generation: an agentic digest model must not start a call.
DIGEST_STOP = ["<|tool_call_start|>", "<ifm|tool_calls>"]


def digest_prompt(compression_prompt: str, text: str, max_chars: int = 60_000) -> list[dict]:
    """Messages for the digest sub-agent."""
    if len(text) > max_chars:
        h = max_chars // 2
        text = text[:h] + "\n[...]\n" + text[-h:]
    return [
        {"role": "system", "content": DIGEST_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Purpose of the call: {compression_prompt}\n\nRaw tool output:\n{text}\n\n"
                "Write the digest now (plain text only)."
            ),
        },
    ]


class TeacherEngine:
    """HF causal LM + its chat template, OpenAI-shaped in and out.

    ``model``/``tokenizer`` are any objects with ``generate`` and
    ``apply_chat_template``; ``parse`` maps raw text to ``ParsedOutput``
    (``parse_k2_output`` for K2, a passthrough for models without tool
    markup)."""

    def __init__(self, model: Any, tokenizer: Any, parse=parse_k2_output, device: str = "cuda"):
        self.model = model
        self.tok = tokenizer
        self.parse = parse
        self.device = device

    @staticmethod
    def _normalise_messages(messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            m = dict(m)
            if m.get("role") == "assistant":
                for c in m.get("tool_calls") or []:
                    f = c.get("function", c)
                    if isinstance(f.get("arguments"), str):
                        f["arguments"] = json.loads(f["arguments"] or "{}")
                # K2's template requires a thinking field on every assistant turn
                m.setdefault("reasoning_content", "")
            out.append(m)
        return out

    def render(self, messages: list[dict], tools: list[dict] | None, kwargs: dict) -> Any:
        msgs = self._normalise_messages(messages)
        return self.tok.apply_chat_template(
            msgs,
            tools=tools or None,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **kwargs,
        )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        chat_template_kwargs: dict | None = None,
    ) -> dict:
        import torch

        enc = self.render(messages, tools, chat_template_kwargs or {})
        ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        gen_kwargs: dict = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            gen_kwargs.update(temperature=temperature, top_p=top_p)
        with torch.no_grad():
            out = self.model.generate(input_ids=ids, attention_mask=attn, **gen_kwargs)
        new = out[0, ids.shape[1] :]
        raw = self.tok.decode(new, skip_special_tokens=False)
        # strip the end-of-turn / eos markers the template closes with
        for stop in (self.tok.eos_token or "", "<|im_end|>", "<|endoftext|>"):
            if stop and stop in raw:
                raw = raw.split(stop)[0]
        parsed = self.parse(raw)
        return {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if parsed.tool_calls else "stop",
                    "message": {
                        "role": "assistant",
                        "content": parsed.content,
                        "reasoning_content": parsed.reasoning,
                        "tool_calls": parsed.tool_calls,
                        "raw": raw,
                        "parse_error": parsed.parse_error,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": int(ids.shape[1]),
                "completion_tokens": int(new.shape[0]),
            },
        }
