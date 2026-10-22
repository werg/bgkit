"""Prompt pieces for the M0 tasks, tokenised once. No chat template anywhere.

    reconstruct:  [bos] Context:\\n  <reps>  \\nRepeat the context verbatim:\\n  <target> [eos]
    continue:     [bos] Context:\\n  <reps>  \\nContinue the text:\\n          <target> [eos]

The instruction after the reps is part of the suffix but carries no loss
(``suffix_label_start``); the target and the closing eos do.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

TASKS = ("reconstruct", "continue")

_PREFIX = "Context:\n"
_INSTR = {
    "reconstruct": "\nRepeat the context verbatim:\n",
    "continue": "\nContinue the text:\n",
}
_CHAT_INSTR = {
    "reconstruct": "\nRepeat the context verbatim.",
    "continue": "\nContinue the text.",
}
SPLICE_SENTINEL = "<|reserved_6|>"
# SPLICE_SENTINEL is the CANONICAL slot marker: stores, builders and the harness write it into
# message text. A decoder whose vocabulary lacks it (LFM2.5-8B-A1B, 2026-09-25) gets it mapped to a
# single special token it does have, at template time only (``decoder_sentinel`` / ``with_sentinel``).
SENTINEL_CANDIDATES = (SPLICE_SENTINEL, "<|img_row_1_col_1|>")


def decoder_sentinel(tok) -> tuple[str, int]:
    """(token string, id) of the slot sentinel in ``tok``: the first candidate that is ONE
    token of the vocabulary."""
    for s in SENTINEL_CANDIDATES:
        ids = tok(s, add_special_tokens=False)["input_ids"]
        if len(ids) == 1 and ids[0] != getattr(tok, "unk_token_id", None):
            return s, int(ids[0])
    raise ValueError(f"tokenizer has none of the sentinel candidates {SENTINEL_CANDIDATES}")


def with_sentinel(msgs: list[dict], sentinel: str) -> list[dict]:
    """``msgs`` with the canonical SPLICE_SENTINEL replaced by the decoder's ``sentinel``
    (a no-op, returning ``msgs`` itself, when they are the same)."""
    if sentinel == SPLICE_SENTINEL:
        return msgs
    out = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, str) and SPLICE_SENTINEL in c:
            m = {**m, "content": c.replace(SPLICE_SENTINEL, sentinel)}
        out.append(m)
    return out


# LFM2.5-8B-A1B is a reasoning model: its assistant turns open with a think block, and its template
# has no non-thinking switch. The empty block spelled "<think>\n</think>\n" makes it answer
# directly (measured 2026-09-26 on the base model: no block -> "<think>\nOkay, so the question
# is..."; "<think></think>" -> thinks anyway; "<think>\n\n</think>\n\n" -> reasons in the open;
# "<think>\n</think>\n" -> "The authorship of *Hamlet*..."). ``set_no_think_template`` renders
# it in the generation prompt and on EVERY assistant turn (the stock template keeps reasoning
# only after the last user message; a block that disappears when a later user message arrives
# makes a multi-user trajectory chunk prefix-inconsistent -- 60% of compaction_chunks rows were
# unrenderable, found by the pre-launch validator 2026-09-26).
NO_THINK_BLOCK = "<think>\n</think>\n"
# the stock template's snippets (Jinja source: "\n" inside its string literals is backslash-n)
_THINK_TURN = """{%- if thinking and (preserve_thinking or loop.index0 > ns.last_user_index) -%}
            {{- "<think>" + thinking + "</think>" -}}
        {%- endif -%}"""
_GEN_PROMPT = """{%- if add_generation_prompt -%}
    {{- "<|im_start|>assistant\\n" -}}
{%- endif -%}"""
_BLOCK_SRC = "<think>\\n</think>\\n"  # NO_THINK_BLOCK as a Jinja string literal


def set_no_think_template(tok) -> None:
    """Switch ``tok``'s chat template to non-thinking mode (see NO_THINK_BLOCK); idempotent.
    Raises when the template is not the one this was written against, so a changed template
    cannot silently render turns the model answers with a think block."""
    t = tok.chat_template or ""
    if _BLOCK_SRC in t:
        return
    if t.count(_THINK_TURN) != 1 or t.count(_GEN_PROMPT) != 1:
        raise ValueError("decoder_no_think: unrecognised chat template (think block / "
                         "generation prompt not found exactly once)")
    # enable_thinking=True (apply_chat_template kwarg) keeps the model's own thinking format:
    # the reasoning-replay rows (framing "chat_think") are rendered that way
    turn = _THINK_TURN.replace(
        "        {%- endif -%}",
        "        {%- elif not enable_thinking -%}\n"
        '            {{- "' + _BLOCK_SRC + '" -}}\n'
        "        {%- endif -%}",
    )
    gen = _GEN_PROMPT.replace(
        '{{- "<|im_start|>assistant\\n" -}}',
        '{{- "<|im_start|>assistant\\n" -}}\n'
        '    {%- if not enable_thinking -%}{{- "' + _BLOCK_SRC + '" -}}{%- endif -%}',
    )
    tok.chat_template = t.replace(_THINK_TURN, turn).replace(_GEN_PROMPT, gen)


# Compression prompts handed to the ENCODER for the stage-0 tasks.
COMPRESSION_PROMPTS = {
    "reconstruct": "Compress the following text so that it can be reproduced verbatim.\n",
    "continue": "Compress the following text so that it can be continued.\n",
}
STYLES = ("plain", "chat")


@dataclass
class Templates:
    prefix: torch.Tensor  # [bos] + "Context:\n"
    instr: dict[str, torch.Tensor]
    eos_id: int
    bos_id: int
    pad_id: int

    @classmethod
    def from_tokenizer(cls, tok, style: str = "plain") -> Templates:
        if style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}; got {style!r}")
        bos = tok.bos_token_id
        eos = tok.eos_token_id
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        if bos is None or eos is None:
            raise ValueError("tokenizer must define bos and eos")

        def enc(s: str) -> torch.Tensor:
            return torch.tensor(tok(s, add_special_tokens=False)["input_ids"], dtype=torch.long)

        if style == "plain":
            prefix = torch.cat([torch.tensor([bos]), enc(_PREFIX)])
            instr = {t: enc(s) for t, s in _INSTR.items()}
            return cls(
                prefix=prefix, instr=instr, eos_id=int(eos), bos_id=int(bos), pad_id=int(pad)
            )

        sent_str, sentinel = decoder_sentinel(tok)
        prefix = None
        instr = {}
        for t, text in _CHAT_INSTR.items():
            msgs = [{"role": "user", "content": _PREFIX + sent_str + text}]
            ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
            if not isinstance(ids, list):
                ids = ids["input_ids"]
            ids = [int(i) for i in ids]
            hits = [i for i, x in enumerate(ids) if x == sentinel]
            if len(hits) != 1:
                raise ValueError(
                    f"chat template must contain the sentinel exactly once; got {hits}"
                )
            cut = hits[0]
            this_prefix = torch.tensor(ids[:cut], dtype=torch.long)
            if prefix is None:
                prefix = this_prefix
            elif not torch.equal(prefix, this_prefix):
                raise ValueError(
                    "chat prefix differs between tasks; the template is not splittable"
                )
            instr[t] = torch.tensor(ids[cut + 1 :], dtype=torch.long)
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        eos_id = int(im_end) if im_end is not None and im_end != tok.unk_token_id else int(eos)
        return cls(prefix=prefix, instr=instr, eos_id=eos_id, bos_id=int(bos), pad_id=int(pad))

    @staticmethod
    def compression_prompt_ids(tok) -> dict[str, torch.Tensor]:
        return {
            t: torch.tensor(tok(s, add_special_tokens=False)["input_ids"], dtype=torch.long)
            for t, s in COMPRESSION_PROMPTS.items()
        }

    def suffix(self, task: str, target: torch.Tensor) -> tuple[torch.Tensor, int]:
        """(suffix ids, index where loss starts)."""
        ins = self.instr[task]
        s = torch.cat([ins, target.to(torch.long), torch.tensor([self.eos_id])])
        return s, int(ins.shape[0])
