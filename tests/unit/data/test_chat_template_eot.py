"""End-of-turn token discovery for generation stop (2026-08-23: Falcon-H1's
eos is <|end_of_text|> but its template closes turns with <|im_end|>)."""

from __future__ import annotations

from typing import ClassVar

import pytest

from bgkit.data.chat_template import end_of_turn_token_ids


class _Tok:
    """Minimal tokenizer double with a ChatML-style template."""

    eos_token_id = 11
    all_special_ids: ClassVar[list[int]] = [11, 7, 8]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)

    def encode(self, text, add_special_tokens=False):
        out = []
        for piece in text.replace("<|im_end|>", " \x00im_end ").split():
            out.append(8 if piece == "\x00im_end" else 1)
        return out


def test_end_of_turn_adds_template_marker_to_eos():
    assert end_of_turn_token_ids(_Tok()) == [11, 8]


def test_end_of_turn_dedups_when_eos_is_the_marker():
    tok = _Tok()
    tok.eos_token_id = 8
    assert end_of_turn_token_ids(tok) == [8]


def test_end_of_turn_without_template_falls_back_to_eos():
    class NoTemplate(_Tok):
        def apply_chat_template(self, *a, **k):
            raise ValueError("no chat template")

    assert end_of_turn_token_ids(NoTemplate()) == [11]


@pytest.mark.parametrize(
    ("name", "expect_distinct_marker"),
    [("Qwen/Qwen3.5-0.8B", False), ("tiiuae/Falcon-H1-Tiny-90M-Instruct", True)],
)
def test_end_of_turn_real_templates(name, expect_distinct_marker):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    except Exception:
        pytest.skip(f"{name} not available locally")
    ids = end_of_turn_token_ids(tok)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert im_end in ids
    assert (ids[0] != im_end) == expect_distinct_marker
