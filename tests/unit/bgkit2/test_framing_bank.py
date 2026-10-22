"""Rendering-time phrasing diversity: canonical == stored phrasing; train and
held-out slices are disjoint; draws are deterministic per (seed, index)."""

import numpy as np
import pytest

from bgkit2.data import framing_bank as fb
from bgkit2.data.framing_bank import FramingBank
from bgkit2.data.templates import SPLICE_SENTINEL


def test_slices_disjoint_and_nonempty():
    for bank, cut in (
        (fb.CHAT_WRAPPERS, fb.CHAT_HELDOUT_FROM),
        (fb.SYSTEM_MESSAGES, fb.SYSTEM_HELDOUT_FROM),
        (fb.TOOL_NAMES, fb.TOOL_HELDOUT_FROM),
        *((fb.PROMPTS[k], fb.PROMPT_HELDOUT_FROM[k]) for k in fb.PROMPTS),
    ):
        train, held = fb._slice(bank, cut, "train"), fb._slice(bank, cut, "heldout")
        assert train and held and not set(train) & set(held)
        assert len(set(bank)) == len(bank)


def test_every_wrapper_places_sentinel_once():
    for w in fb.CHAT_WRAPPERS:
        assert w.count("{ctx}") == 1 and w.count("{q}") == 1
        assert w.format(ctx=SPLICE_SENTINEL, q="why?").count(SPLICE_SENTINEL) == 1
    for k, bank in fb.PROMPTS.items():
        for p in bank:
            assert p.endswith("\n")
            assert ("{q}" in p) == (k == "qa")


def test_canonical_is_the_stored_phrasing():
    bank = FramingBank("canonical", p_system=1.0, p_no_prompt=1.0)
    rng = np.random.default_rng(0)
    msgs = bank.chat_messages(rng, "what?")
    assert msgs == [{"role": "user", "content": f"Context:\n{SPLICE_SENTINEL}\nwhat?"}]
    tool = bank.tool_messages(rng, "read it", "read_file", {"path": "a/b.py"})
    assert tool[1]["tool_calls"][0]["function"] == {
        "name": "read_file",
        "arguments": {"path": "a/b.py"},
    }
    stored = "Compress the passage to answer: what?\n"
    assert bank.prompt(rng, "qa_squad", stored, "what?") == stored
    assert bank.prompt(rng, "file_overview", "x\n", "") == "x\n"


def test_train_draws_deterministic_and_varied():
    bank = FramingBank("train", p_system=0.5, p_no_prompt=0.1)
    a = [bank.chat_messages(np.random.default_rng([3, i]), "q?") for i in range(64)]
    b = [bank.chat_messages(np.random.default_rng([3, i]), "q?") for i in range(64)]
    assert a == b
    assert len({m[-1]["content"] for m in a}) > 4
    assert any(m[0]["role"] == "system" for m in a) and any(m[0]["role"] == "user" for m in a)
    for m in a:
        assert sum(SPLICE_SENTINEL in x["content"] for x in m) == 1
        assert "q?" in m[-1]["content"]


def test_heldout_never_uses_train_phrasings():
    held = FramingBank("heldout", p_system=1.0, p_no_prompt=0.0)
    train_wraps = {
        w.format(ctx=SPLICE_SENTINEL, q="q?") for w in fb.CHAT_WRAPPERS[: fb.CHAT_HELDOUT_FROM]
    }
    train_sys = set(fb.SYSTEM_MESSAGES[: fb.SYSTEM_HELDOUT_FROM])
    for i in range(32):
        m = held.chat_messages(np.random.default_rng(i), "q?")
        assert m[-1]["content"] not in train_wraps
        assert m[0]["content"] not in train_sys
        p = held.prompt(np.random.default_rng(i), "qa_x", "stored\n", "q?")
        assert p not in {t.format(q="q?") for t in fb.PROMPTS["qa"][: fb.PROMPT_HELDOUT_FROM["qa"]]}


def test_no_prompt_fraction_and_unknown_family():
    bank = FramingBank("train", p_system=0.0, p_no_prompt=0.5)
    ps = [bank.prompt(np.random.default_rng(i), "qa_x", "s\n", "q?") for i in range(200)]
    empty = sum(p == "" for p in ps)
    assert 60 < empty < 140
    assert all(p == "" or p.endswith("\n") for p in ps)
    assert bank.prompt(np.random.default_rng(1), "weird_family", "s\n", "q?") in ("", "s\n")


def test_tool_arg_key_follows_tool_name():
    bank = FramingBank("train", p_system=0.0, p_no_prompt=0.0)
    seen = set()
    for i in range(64):
        m = bank.tool_messages(np.random.default_rng(i), "read", "read_file", {"path": "p.py"})
        fn = m[1]["tool_calls"][0]["function"]
        seen.add(fn["name"])
        assert list(fn["arguments"].values()) == ["p.py"]
        assert (fn["name"], next(iter(fn["arguments"]))) in fb.TOOL_NAMES[: fb.TOOL_HELDOUT_FROM]
    assert len(seen) > 2


def test_bad_split_rejected():
    with pytest.raises(ValueError):
        FramingBank("test")


def test_collator_mixes_prompted_and_unprompted_samples():
    import torch

    from bgkit2.data.autoencode import Collator, Sample

    class _T:  # minimal Templates stand-in
        pad_id = 0
        eos_id = 2

    a = Sample(
        ctx_ids=torch.tensor([5, 6, 7]),
        target_ids=torch.tensor([8]),
        task="x",
        store=0,
        doc=0,
        prompt_ids=torch.tensor([11, 12]),
        prefix_ids=torch.tensor([1]),
        instr_ids=torch.tensor([3]),
    )
    b = Sample(
        ctx_ids=torch.tensor([5, 6]),
        target_ids=torch.tensor([8]),
        task="x",
        store=0,
        doc=1,
        prompt_ids=None,
        prefix_ids=torch.tensor([1]),
        instr_ids=torch.tensor([3]),
    )
    batch = Collator(_T())([a, b])
    assert batch.prompt_ids.shape == (2, 2)
    assert batch.prompt_mask.tolist() == [[1, 1], [0, 0]]
    assert batch.prompt_ids[1].tolist() == [0, 0]
