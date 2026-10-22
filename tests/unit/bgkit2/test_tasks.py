from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from bgkit2.data.autoencode import Collator
from bgkit2.data.tasks import COLUMNS, ChatRenderer, TaskDataset, write_task_store
from bgkit2.data.templates import SPLICE_SENTINEL, Templates


def _real_tok():
    from transformers import AutoTokenizer

    cache = Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots"
    snaps = sorted(cache.glob("*/chat_template.jinja")) if cache.exists() else []
    if not snaps:
        pytest.skip("LFM2.5-350M tokenizer not cached")
    return AutoTokenizer.from_pretrained(str(snaps[0].parent))


def test_write_and_read_task_store_renders_both_framings(tmp_path):
    tok = _real_tok()
    rows = [
        {
            "family": "qa_squad",
            "framing": "chat_context",
            "context": "Paris is the capital of France.",
            "prompt": "Compress the passage to answer: capital?\n",
            "instruction": "What is the capital of France?",
            "target": "Paris",
            "split": "train",
            "meta": {"id": "a"},
        },
        {
            "family": "file_overview",
            "framing": "tool_response",
            "context": "def create_app():\n    return App()\n",
            "prompt": "Compress this file.\n",
            "instruction": "What does src/app.py contain?",
            "tool_name": "read_file",
            "tool_args": {"path": "src/app.py"},
            "target": "It defines create_app().",
            "split": "train",
            "meta": {},
        },
        {
            "family": "qa_squad",
            "framing": "chat_context",
            "context": "x",
            "prompt": "",
            "instruction": "q?",
            "target": "a",
            "split": "eval",
            "meta": {},
        },
    ]
    p = tmp_path / "t.parquet"
    assert write_task_store(str(p), rows) == 3
    r = ChatRenderer.from_tokenizer(tok)
    ds = TaskDataset(str(p), r, split="train")
    assert len(ds) == 2 and ds.families == ["file_overview", "qa_squad"]
    s0, s1 = ds[0], ds[1]
    sid = r.sentinel_id
    for s in (s0, s1):
        assert sid not in s.prefix_ids.tolist() and sid not in s.instr_ids.tolist()
        text = tok.decode(torch.cat([s.prefix_ids, torch.tensor([sid]), s.instr_ids]))
        assert text.endswith("<|im_end|>\n<|im_start|>assistant\n")
        assert SPLICE_SENTINEL in text
    assert "<|tool_call_start|>" in tok.decode(s1.prefix_ids)
    assert "read_file" in tok.decode(s1.prefix_ids) and "src/app.py" in tok.decode(s1.prefix_ids)
    assert "<|im_start|>tool" in tok.decode(s1.prefix_ids)
    # the collator honours the pre-rendered framing and puts loss on target + eos
    t = Templates.from_tokenizer(tok, style="chat")
    batch = Collator(t)([s0, s1])
    for i, s in enumerate((s0, s1)):
        st = batch.suffix_label_start[i]
        assert st == s.instr_ids.shape[0]
        assert torch.equal(batch.suffix_ids[i][st:-1], s.target_ids)
        assert int(batch.suffix_ids[i][-1]) == t.eos_id
        assert torch.equal(batch.prefix_ids[i], s.prefix_ids)
    assert batch.prompt_ids is not None and int(batch.prompt_mask.sum()) > 0
    ev = TaskDataset(str(p), r, split="eval")
    assert len(ev) == 1
    with pytest.raises(ValueError):
        TaskDataset(str(p), r, split="nope")
    assert set(COLUMNS) >= {"family", "framing", "context", "target"}
    j = next(i for i in range(ds.n) if ds.row(i)["framing"] == "tool_response")
    assert json.loads(ds.row(j)["tool_args"]) == {"path": "src/app.py"}


def test_messages_framing_renders_a_compaction_history():
    tok = _real_tok()
    r = ChatRenderer.from_tokenizer(tok)
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the bug in foo.py."},
        {"role": "user", "content": f"[Earlier conversation compacted]\n{SPLICE_SENTINEL}"},
        {"role": "assistant", "content": "I will run the tests."},
    ]
    prefix, instr = r.render("messages", json.dumps(msgs))
    text = tok.decode(torch.cat([prefix, torch.tensor([r.sentinel_id]), instr]))
    assert "You are a coding agent." in tok.decode(prefix)
    assert "[Earlier conversation compacted]" in tok.decode(prefix)
    assert "I will run the tests." in tok.decode(instr)
    assert text.endswith("<|im_end|>\n<|im_start|>assistant\n")
    with pytest.raises(ValueError):
        r.render("messages", json.dumps([{"role": "user", "content": "no sentinel"}]))


def test_benchpress_template_split_and_arms():
    import importlib.util
    from pathlib import Path as PathT

    bp = PathT("/home/werg/bgkit-data-nvme/capability_packaging/benchmarks/benchpress/src")
    if not bp.exists():
        pytest.skip("benchpress not checked out")
    import sys

    sys.path.insert(0, str(bp))
    spec = importlib.util.spec_from_file_location(
        "bp_eval", PathT(__file__).resolve().parents[3] / "scripts" / "bgkit2_eval_benchpress.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    t = "Passage: {<<C>>}\nQuestion: {<<Q>>}\nAnswer: {<<A>>}"
    pre, post = mod.split_template(t, "Why?")
    assert pre == "Passage: " and post == "\nQuestion: Why?\nAnswer: "
    assert mod.user_content("full", t, "CTX", "Why?") == "Passage: CTX\nQuestion: Why?\nAnswer: "
    assert mod.user_content("noctx", t, "CTX", "Why?") == "Passage: \nQuestion: Why?\nAnswer: "
    assert SPLICE_SENTINEL in mod.user_content("bgkit", t, "CTX", "Why?")
    assert mod.user_content("fullslot", t, "CTX", "Why?") == mod.user_content(
        "bgkit", t, "CTX", "Why?"
    )


def test_task_dataset_skips_overlong_prefixes(tmp_path):
    tok = _real_tok()
    rows = [
        {
            "family": "f",
            "framing": "chat_context",
            "context": "x",
            "prompt": "",
            "instruction": "long " * 400,
            "target": "a",
            "split": "train",
            "meta": {},
        },
        {
            "family": "f",
            "framing": "chat_context",
            "context": "y",
            "prompt": "",
            "instruction": "short?",
            "target": "b",
            "split": "train",
            "meta": {},
        },
    ]
    p = tmp_path / "t.parquet"
    write_task_store(str(p), rows)
    ds = TaskDataset(str(p), ChatRenderer.from_tokenizer(tok), split="train", max_prefix_tokens=64)
    s = ds[0]
    assert tok.decode(s.target_ids) == "b" and ds.skipped_long_prefix == 1


def test_sibling_prompts_come_from_the_same_document(tmp_path):
    tok = _real_tok()
    rows = []
    for d in range(2):
        for q in range(3):
            rows.append(
                {
                    "family": "qa_multineedle",
                    "framing": "chat_context",
                    "context": f"doc {d}",
                    "prompt": f"Compress to answer: question {d}-{q}\n",
                    "instruction": f"question {d}-{q}",
                    "target": f"answer {d}-{q}",
                    "split": "train",
                    "meta": {"doc": f"needle:{d}", "position": q},
                }
            )
    rows.append(
        {
            "family": "qa_squad",
            "framing": "chat_context",
            "context": "single",
            "prompt": "Compress: lone\n",
            "instruction": "lone?",
            "target": "x",
            "split": "train",
            "meta": {"id": "solo"},
        }
    )
    p = tmp_path / "n.parquet"
    write_task_store(str(p), rows)
    r = ChatRenderer.from_tokenizer(tok)
    ds = TaskDataset(str(p), r, split="train", sibling_prompts=True)
    for i in range(6):
        s = ds[i]
        own = tok.decode(s.prompt_ids)
        alt = tok.decode(s.alt_prompt_ids)
        d = i // 3
        assert f"question {d}-" in alt and alt != own, (own, alt)
    lone = ds[6]
    assert lone.alt_prompt_ids is None
    assert torch.equal(ds[1].alt_prompt_ids, ds[1].alt_prompt_ids)  # deterministic per index
    sib = ds[0].sibling
    assert sib is not None and torch.equal(sib.ctx_ids, ds[0].ctx_ids)
    assert torch.equal(sib.prompt_ids, ds[0].alt_prompt_ids)
    assert tok.decode(sib.target_ids) != tok.decode(ds[0].target_ids)
    assert tok.decode(sib.target_ids).startswith("answer 0-")
    assert tok.decode(sib.instr_ids) != tok.decode(ds[0].instr_ids)  # its own framing
    batch = Collator(Templates.from_tokenizer(tok, style="chat"))([ds[0], ds[6], ds[4]])
    assert batch.alt_present.tolist() == [True, False, True]
    assert batch.sibling is not None and batch.sibling.tasks == ["qa_multineedle"] * 2
    assert torch.equal(batch.sibling.input_ids, batch.input_ids[[0, 2]])
    assert torch.equal(batch.sibling.prompt_ids, batch.alt_prompt_ids[[0, 2]])
    assert torch.equal(batch.sibling.target_ids[1], ds[4].sibling.target_ids)
    assert int(batch.alt_prompt_mask[1].sum()) == 0
    sub = batch.subset([0, 2]).with_alt_prompts()
    assert sub.alt_present.tolist() == [True, True]
    assert torch.equal(sub.prompt_ids, batch.alt_prompt_ids[[0, 2]])
    plain = TaskDataset(str(p), r, split="train")
    assert plain[0].alt_prompt_ids is None
    with pytest.raises(ValueError):
        TaskDataset(str(p), r, split="train", sibling_prompts=True, epoch_size=1).sibling  # noqa: B018
        raise ValueError  # the store above has siblings; a store without must raise at init


def test_span_mask_uses_character_offsets():
    tok = _real_tok()
    r = ChatRenderer.from_tokenizer(tok)
    text = "Some filler sentence here. Mary moved to the bathroom. More filler follows here."
    m = r.span_mask(text, ["Mary moved to the bathroom."])
    ids = r.encode(text)
    assert m is not None and m.shape[0] == ids.shape[0]
    assert tok.decode(ids[m]).strip() == "Mary moved to the bathroom."
    assert r.span_mask(text, ["not in the text"]) is None


def test_tool_slots_framing_renders_one_segment_per_sentinel(tmp_path):
    """framing "tool_slots": an agent turn whose tool results are each a
    compressed slot — the rendered conversation is cut at every sentinel."""
    import json

    from bgkit2.data.tasks import SPLICE_SENTINEL

    tok = _real_tok()
    msgs = [
        {"role": "user", "content": "Find where the cache is invalidated."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "search", "arguments": {"q": "cache"}}}
            ],
        },
        {"role": "tool", "content": SPLICE_SENTINEL},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "read", "arguments": {"path": "a.py"}}}
            ],
        },
        {"role": "tool", "content": SPLICE_SENTINEL},
    ]
    target = [{"role": "assistant", "content": "It is invalidated in a.py."}]
    rows = [
        {
            "family": "tool_slots",
            "framing": "tool_slots",
            "context": json.dumps(
                ["src/a.py: def invalidate(): ...", "def invalidate():\n    cache.clear()\n"]
            ),
            "prompt": json.dumps(["search: cache", "read a.py"]),
            "instruction": json.dumps(msgs),
            "target": json.dumps(target),
            "split": "train",
            "meta": {},
        }
    ]
    p = tmp_path / "slots.parquet"
    write_task_store(str(p), rows)
    r = ChatRenderer.from_tokenizer(tok)
    ds = TaskDataset(str(p), r, split="train", max_ctx_tokens=64, max_target_tokens=32)
    s = ds[0]
    assert s.task == "tool_slots" and len(s.extra_ctx) == 1 and len(s.mid_ids) == 1
    assert s.prefix_ids.shape[0] > 0 and s.mid_ids[0].shape[0] > 0 and s.instr_ids.shape[0] > 0
    assert s.prompt_ids is not None and s.extra_prompt[0] is not None
    assert s.target_loss_mask is not None and bool(s.target_loss_mask.any())
    # the rendered prefix + middle + tail reassemble the template minus the sentinels
    full = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    full = full if isinstance(full, list) else full["input_ids"]
    joined = s.prefix_ids.tolist() + s.mid_ids[0].tolist() + s.instr_ids.tolist()
    assert joined == [t for t in full if t != r.sentinel_id]


def test_chat_framing_renders_a_json_turn_target_with_its_tool_call(tmp_path):
    """A replay row whose target is a JSON list with the assistant message
    (content + tool_calls) renders the whole turn — content, the template's
    tool-call block — with the loss mask on every token, instead of a text
    preamble cut before the call."""
    tok = _real_tok()
    history = [{"role": "user", "content": "List the files in /tmp."}]
    turn = {
        "role": "assistant",
        "content": "Let me look at the directory:",
        "tool_calls": [
            {"type": "function", "function": {"name": "bash", "arguments": {"command": "ls /tmp"}}}
        ],
    }
    rows = [
        {
            "family": "chat_replay",
            "framing": "chat",
            "context": "",
            "prompt": "",
            "instruction": json.dumps(history),
            "target": json.dumps([turn]),
            "split": "train",
            "meta": {"source": "t", "target_kind": "turn"},
        },
        {
            "family": "chat_replay",
            "framing": "chat",
            "context": "",
            "prompt": "",
            "instruction": json.dumps(history),
            "target": "Sure, here they are.",
            "split": "train",
            "meta": {"source": "t", "target_kind": "text"},
        },
        {
            "family": "chat_replay",
            "framing": "chat",
            "context": "",
            "prompt": "",
            "instruction": json.dumps(history),
            "target": "[Note] a text answer that merely starts with a bracket",
            "split": "train",
            "meta": {"source": "t", "target_kind": "text"},
        },
    ]
    path = tmp_path / "replay.parquet"
    write_task_store(str(path), rows)
    ds = TaskDataset(str(path), ChatRenderer.from_tokenizer(tok), split="train")
    s_turn, s_text = ds[0], ds[1]
    rendered = tok.decode(s_turn.target_ids)
    assert rendered.startswith("Let me look at the directory:")
    assert "<|tool_call_start|>[bash(command='ls /tmp')]<|tool_call_end|>" in rendered
    assert "<|im_end|>" not in rendered  # the collator appends the end-of-turn
    assert s_turn.target_loss_mask is not None and bool(s_turn.target_loss_mask.all())
    assert s_turn.target_loss_mask.shape[0] == s_turn.target_ids.shape[0]
    assert s_text.target_loss_mask is None
    assert tok.decode(s_text.target_ids) == "Sure, here they are."
    s_bracket = ds[2]
    assert s_bracket.target_loss_mask is None
    assert tok.decode(s_bracket.target_ids).startswith("[Note] a text answer")


def test_tool_slots_over_long_prefix_hops_to_the_next_row(tmp_path):
    """task_max_prefix_tokens is enforced for the agentic framing too (2026-09-14
    audit: it was only counted, so 2.6k-3.4k-token prefixes passed a 1536 cap)."""
    import json

    from bgkit2.data.tasks import SPLICE_SENTINEL

    tok = _real_tok()

    def row(n_words: int) -> dict:
        msgs = [
            {"role": "user", "content": "word " * n_words},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "read", "arguments": {"path": "a"}}}
                ],
            },
            {"role": "tool", "content": SPLICE_SENTINEL},
        ]
        return {
            "family": "tool_slots",
            "framing": "tool_slots",
            "context": json.dumps(["ctx"]),
            "prompt": json.dumps(["read a"]),
            "instruction": json.dumps(msgs),
            "target": "ok",
            "split": "train",
            "meta": {"n": n_words},
        }

    p = tmp_path / "s.parquet"
    write_task_store(str(p), [row(400), row(3)])
    r = ChatRenderer.from_tokenizer(tok)
    ds = TaskDataset(
        str(p), r, split="train", max_ctx_tokens=64, max_target_tokens=16, max_prefix_tokens=120
    )
    s = ds[0]
    assert ds.skipped_long_prefix == 1
    assert s.prefix_ids.shape[0] < 120  # the short row was served instead
    ds2 = TaskDataset(
        str(p), r, split="train", max_ctx_tokens=64, max_target_tokens=16, max_prefix_tokens=None
    )
    assert ds2[0].prefix_ids.shape[0] > 120


def test_tool_slots_rows_render_their_tool_list_like_the_harness(tmp_path):
    """meta.tools (what the agent harness passes to the template at inference)
    must render into the training prefix too, or train/infer prompts differ."""
    import json

    from bgkit2.data.tasks import SPLICE_SENTINEL

    tok = _real_tok()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "grep the repo",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]
    msgs = [
        {"role": "user", "content": "find foo"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "search", "arguments": {"query": "foo"}}}
            ],
        },
        {"role": "tool", "content": SPLICE_SENTINEL},
    ]
    row = {
        "family": "agent_tool_slots",
        "framing": "tool_slots",
        "context": json.dumps(["a.py:1: foo"]),
        "prompt": json.dumps(["search: foo"]),
        "instruction": json.dumps(msgs),
        "target": json.dumps([{"role": "assistant", "content": "Found it."}]),
        "split": "train",
        "meta": {"tools": tools},
    }
    p = tmp_path / "t.parquet"
    write_task_store(str(p), [row])
    r = ChatRenderer.from_tokenizer(tok)
    ds = TaskDataset(str(p), r, split="train", max_ctx_tokens=64, max_target_tokens=32)
    s = ds[0]
    prefix = tok.decode(s.prefix_ids)
    assert "grep the repo" in prefix and '"name": "search"' in prefix
    assert tok.decode(s.target_ids).startswith("Found it.")


def test_overview_row_from_the_harness_becomes_a_cached_slot_sample(tmp_path):
    """Conversation -> task row -> TaskDataset: the always-on repo overview is
    the first slot; nothing is tokenised for it (its reps come from the root
    cache), the tool result after it is an ordinary extra text slot."""
    from bgkit2.agent.context import Conversation, PlainRenderer, RatioPolicy, ToolCall

    tok = _real_tok()
    tools = [{"type": "function", "function": {"name": "read", "parameters": {}}}]
    c = Conversation("sys", tools, PlainRenderer(), RatioPolicy(), "find it")
    c.add_user_with_overview("Which file defines `foo`?", "own/repo", "pkg/a.py\npkg/b.py", 7)
    first = c.to_task_row(target=[{"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "read", "arguments": {"path": "pkg/a.py"}}}]}],
        split="train")
    c.add_assistant("", [ToolCall("read", {"path": "pkg/a.py"})])
    c.add_tool_result("def foo():\n    return 1\n" * 8, True, "read pkg/a.py", "read")
    second = c.to_task_row(target=[{"role": "assistant", "content": "ANSWER: pkg/a.py"}],
                           split="train")
    p = tmp_path / "ov.parquet"
    write_task_store(str(p), [first, second])
    ds = TaskDataset(str(p), ChatRenderer.from_tokenizer(tok), split="train")
    s0, s1 = ds[0], ds[1]
    assert s0.cached_slot == ("own/repo", 7) and s0.ctx_ids.numel() == 0 and not s0.extra_ctx
    assert s1.cached_slot == ("own/repo", 7) and s1.ctx_ids.numel() == 0
    assert len(s1.extra_ctx) == 1 and s1.extra_ctx[0].numel() > 10  # the read result: text


def test_browse_row_from_the_harness_becomes_a_cached_extra_slot(tmp_path):
    """overview -> browse -> read, through the task store: the browse slot is
    an EXTRA slot with nothing tokenised and ``extra_cached`` naming the node;
    the read after it stays text. A browse without the overview is rejected."""
    from bgkit2.agent.context import Conversation, PlainRenderer, RatioPolicy, ToolCall
    from bgkit2.data.overview import overview_marker

    tok = _real_tok()
    tools = [{"type": "function", "function": {"name": "browse", "parameters": {}}}]
    c = Conversation("sys", tools, PlainRenderer(), RatioPolicy(), "find it")
    c.add_user_with_overview("Which file defines `foo`?", "own/repo", "pkg/a.py\npkg/b.py", 7)
    c.add_assistant("", [ToolCall("browse", {"dir": "pkg"})])
    c.add_cached_tool_result("a.py\nb.py", overview_marker("own/repo", 3, "pkg"), 3,
                             "contents of pkg", "browse")
    c.add_assistant("", [ToolCall("read", {"path": "pkg/a.py"})])
    c.add_tool_result("def foo():\n    return 1\n" * 8, True, "read pkg/a.py", "read")
    row = c.to_task_row(target=[{"role": "assistant", "content": "ANSWER: pkg/a.py"}],
                        split="train")
    p = tmp_path / "br.parquet"
    write_task_store(str(p), [row])
    s = TaskDataset(str(p), ChatRenderer.from_tokenizer(tok), split="train")[0]
    assert s.cached_slot == ("own/repo", 7)
    assert s.extra_cached == [("own/repo", "pkg"), None]
    assert s.extra_ctx[0].numel() == 0 and s.extra_ctx[1].numel() > 10
    assert len(s.mid_ids) == 2
