"""Two-tokenizer runs (2026-09-25): the LFM2.5-8B-A1B decoder shares no token ids with the encoder's
tokenizer. Everything the encoder reads must be encoder ids, everything the decoder reads decoder ids,
and the slot sentinel must be one decoder token. Runs on the real tokenizers when they are on disk."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest
import torch

from bgkit2.data.autoencode import Collator, Sample
from bgkit2.data.tasks import ChatRenderer
from bgkit2.data.templates import SPLICE_SENTINEL, Templates, decoder_sentinel

M8 = "/mnt/external/bgkit-data/models/hf/LFM2.5-8B-A1B"
S350 = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots/*"))

pytestmark = pytest.mark.skipif(not (Path(M8).is_dir() and S350), reason="tokenizers not on disk")


@pytest.fixture(scope="module")
def toks():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(M8), AutoTokenizer.from_pretrained(S350[0])


def test_sentinel_is_one_decoder_token_on_both(toks):
    t8, t3 = toks
    s8, i8 = decoder_sentinel(t8)
    s3, i3 = decoder_sentinel(t3)
    assert s3 == SPLICE_SENTINEL and s8 != SPLICE_SENTINEL
    assert t8(s8, add_special_tokens=False)["input_ids"] == [i8]


def test_renderer_splits_on_the_mapped_sentinel_and_encodes_ctx_with_the_encoder(toks):
    t8, t3 = toks
    r = ChatRenderer.from_tokenizer(t8, enc_tok=t3)
    msgs = [{"role": "user", "content": f"Context:\n{SPLICE_SENTINEL}\nWhat is the capital?"}]
    prefix, instr = r.render("messages", json.dumps(msgs))
    assert r.sentinel_id not in prefix.tolist() and r.sentinel_id not in instr.tolist()
    assert max(prefix.tolist() + instr.tolist()) < len(t8)
    text = "The capital of France is Paris. def f(x): return x"
    ctx = r.encode_ctx(text)
    assert ctx.tolist() == t3(text, add_special_tokens=False)["input_ids"]
    dec = r.dec_view(ctx)
    assert t8.decode(dec.tolist()) == text
    assert r.enc_pad_id < 65536 and r.pad_id >= 65536  # the 8B pad is not an encoder id


def test_single_tokenizer_renderer_is_unchanged(toks):
    _, t3 = toks
    r = ChatRenderer.from_tokenizer(t3)
    assert not r.split_tokenizers and r.dec_view(torch.tensor([1, 2])) is None
    assert r.sentinel_str == SPLICE_SENTINEL


def test_collator_pads_encoder_side_with_the_encoder_pad(toks):
    t8, t3 = toks
    r = ChatRenderer.from_tokenizer(t8, enc_tok=t3)
    tpl = Templates.from_tokenizer(t8, style="chat")
    col = Collator(tpl, enc_pad_id=r.enc_pad_id)
    a = r.encode_ctx("short")
    b = r.encode_ctx("a somewhat longer context about something")
    samples = [
        Sample(ctx_ids=x, target_ids=torch.tensor([5]), task="reconstruct", store=-1, doc=i,
               prompt_ids=r.encode_ctx("Compress:"), dec_ctx_ids=r.dec_view(x))
        for i, x in enumerate((a, b))
    ]
    batch = col(samples)
    assert int(batch.input_ids.max()) < 65536
    assert int(batch.input_ids[0, -1]) == r.enc_pad_id
    assert [t.tolist() for t in batch.dec_ctx] == [r.dec_view(a).tolist(), r.dec_view(b).tolist()]
    sub = batch.subset([1])
    assert sub.dec_ctx[0].tolist() == r.dec_view(b).tolist()


def test_no_think_template_renders_the_empty_block_where_the_model_would_think(toks):
    from transformers import AutoTokenizer

    from bgkit2.data.templates import NO_THINK_BLOCK, set_no_think_template

    t8 = AutoTokenizer.from_pretrained(M8)  # a fresh copy: the fixture's stays stock
    set_no_think_template(t8)
    set_no_think_template(t8)  # idempotent
    gen = t8.apply_chat_template(
        [{"role": "user", "content": "Q?"}], tokenize=False, add_generation_prompt=True
    )
    assert gen.endswith("<|im_start|>assistant\n" + NO_THINK_BLOCK)
    hist = t8.apply_chat_template(
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
         {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"},
         {"role": "tool", "content": "e"}, {"role": "assistant", "content": "f"}],
        tokenize=False,
    )
    # every assistant turn carries the block (a later user message must not remove it)
    assert "assistant\n" + NO_THINK_BLOCK + "b<|im_end|>" in hist
    assert hist.count(NO_THINK_BLOCK) == 3
    # task prompts end in the block; trajectory chunks stay prefix-consistent and the first
    # block is prompt, not target
    tpl = Templates.from_tokenizer(t8, style="chat")
    assert all(t8.decode(v).endswith(NO_THINK_BLOCK) for v in tpl.instr.values())
    r = ChatRenderer.from_tokenizer(t8, enc_tok=toks[1])
    ids, mask = r.render_chunk(
        [{"role": "user", "content": "q"}],
        [{"role": "assistant", "content": "x"}, {"role": "tool", "content": "y"},
         {"role": "assistant", "content": "z"}],
    )
    # a chunk with a later user message stays prefix-consistent (compaction_chunks rows)
    r.render_chunk(
        [{"role": "user", "content": "q"}],
        [{"role": "assistant", "content": "x"}, {"role": "user", "content": "more"},
         {"role": "assistant", "content": "z"}],
    )
    text = t8.decode(ids)
    assert text.startswith("x") and text.count(NO_THINK_BLOCK) == 1
    with pytest.raises(ValueError, match="unrecognised"):
        set_no_think_template(toks[1])  # the 350M template has no think block


def test_rows_with_reasoning_render_in_the_native_thinking_format(toks):
    from transformers import AutoTokenizer

    from bgkit2.agent.runner import split_think
    from bgkit2.data.tasks import row_has_reasoning
    from bgkit2.data.templates import NO_THINK_BLOCK, set_no_think_template

    t8 = AutoTokenizer.from_pretrained(M8)
    set_no_think_template(t8)
    r = ChatRenderer.from_tokenizer(t8, enc_tok=toks[1])
    user = [{"role": "user", "content": "fix the bug"}]
    chunk = [
        {"role": "assistant", "content": "", "reasoning_content": "look at io.ts first",
         "tool_calls": [{"type": "function", "function": {"name": "read",
                                                          "arguments": {"path": "io.ts"}}}]},
        {"role": "tool", "content": "export const x = 1;"},
        {"role": "assistant", "content": "Fixed.", "reasoning_content": "x is wrong"},
    ]
    # the whole row decides: the prompt carries no empty block, the target is native
    ids, mask = r.render_chunk(user, chunk)
    text = t8.decode(ids)
    assert NO_THINK_BLOCK not in t8.decode(r._apply(user, enable_thinking=True))
    assert text.startswith("<think>\nlook at io.ts first\n</think>\n")
    assert "<think>\nx is wrong\n</think>\nFixed." in text
    think_id = t8.convert_tokens_to_ids("<think>")
    assert all(bool(mask[i]) for i, x in enumerate(ids.tolist()) if x == think_id)
    # a row without reasoning keeps the non-thinking prompt
    plain = r.render_chunk(user, [{"role": "assistant", "content": "Done."}])[0]
    assert t8.decode(plain).startswith("Done.")
    assert t8.decode(r._apply(user)).endswith(NO_THINK_BLOCK)
    # history reasoning alone switches a slotted prompt to thinking (auto)
    hist = user + chunk[:2]
    assert not t8.decode(r._apply(hist)).endswith(NO_THINK_BLOCK)
    row = {"instruction": json.dumps(user), "target": json.dumps(chunk)}
    assert row_has_reasoning(row)
    assert not row_has_reasoning({"instruction": json.dumps(user), "target": "Done."})
    # the harness gets the reasoning back out of a thinking student's turn
    assert split_think("<think>\nplan\n</think>\nFILES: a.ts") == ("plan", "FILES: a.ts")
    assert split_think("FILES: a.ts") == (None, "FILES: a.ts")
