"""Trajectory-chunk framing: the target is a rendered chunk of messages with
the tool results masked out of the loss, end to end through the collator
and the decoder's label construction."""

import json

import pytest
import torch
from transformers import Lfm2ForCausalLM

from bgkit2.data.autoencode import Collator, Sample
from bgkit2.data.tasks import COLUMNS, ChatRenderer, TaskDataset, write_task_store
from bgkit2.data.templates import SPLICE_SENTINEL, Templates
from bgkit2.models.decoder import SpliceDecoder

from .conftest import tiny_config


def _tok():
    from bgkit2.models.loading import load_tokenizer

    try:
        return load_tokenizer("LiquidAI/LFM2.5-350M")
    except Exception:  # pragma: no cover - offline
        pytest.skip("LFM2.5-350M tokenizer not cached")


PREFIX = [
    {"role": "system", "content": "You fix bugs."},
    {"role": "user", "content": "Fix the failing test."},
    {"role": "user", "content": f"[Earlier conversation compacted]\n{SPLICE_SENTINEL}"},
    {"role": "assistant", "content": "I will look at the file."},
]
CHUNK = [
    {
        "role": "assistant",
        "content": "Reading it.",
        "tool_calls": [
            {"type": "function", "function": {"name": "read_file", "arguments": {"path": "a.py"}}}
        ],
    },
    {"role": "tool", "content": "def f():\n    return 1\n"},
    {"role": "assistant", "content": "The bug is in f. Patching."},
]


def test_render_chunk_masks_tool_results_and_keeps_assistant_turns():
    r = ChatRenderer.from_tokenizer(_tok())
    ids, mask = r.render_chunk(PREFIX, CHUNK)
    assert ids.shape == mask.shape and ids.shape[0] > 10
    kept = r.tok.decode(ids[mask].tolist())
    dropped = r.tok.decode(ids[~mask].tolist())
    assert "Reading it." in kept and "read_file" in kept and "Patching." in kept
    assert "return 1" in dropped and "return 1" not in kept
    assert ids[-1].item() != r.eos_id  # the collator appends the end-of-turn
    # The rendering is the template's own: prefix(+gen prompt) + target == full.
    full = r._apply(PREFIX + CHUNK)  # includes a trailing generation prompt
    pre = r._apply(PREFIX)
    assert full[: len(pre)] == pre and full[len(pre) : len(pre) + ids.shape[0]] == ids.tolist()


def test_render_chunk_rejects_non_assistant_start():
    r = ChatRenderer.from_tokenizer(_tok())
    with pytest.raises(ValueError):
        r.render_chunk(PREFIX, CHUNK[1:])


def test_trajectory_store_round_trips_with_loss_mask(tmp_path):
    tok = _tok()
    row = {
        "family": "compaction_chunks",
        "framing": "trajectory",
        "context": "[assistant]\nearlier work " * 20,
        "prompt": "Compact this.\n",
        "instruction": json.dumps(PREFIX),
        "target": json.dumps(CHUNK),
        "split": "train",
        "meta": {},
    }
    p = tmp_path / "chunks.parquet"
    write_task_store(str(p), [row, row])
    ds = TaskDataset(str(p), ChatRenderer.from_tokenizer(tok), split="train", max_target_tokens=64)
    s = ds[0]
    assert s.target_loss_mask is not None and s.target_loss_mask.shape == s.target_ids.shape
    assert s.target_ids.shape[0] <= 64
    assert bool(s.target_loss_mask.any()) and not bool(s.target_loss_mask.all())
    assert set(COLUMNS) >= {"family", "framing", "target"}

    t = Templates.from_tokenizer(tok, style="chat")
    batch = Collator(t)([s, s])
    assert batch.suffix_loss_mask is not None and len(batch.suffix_loss_mask) == 2
    m = batch.suffix_loss_mask[0]
    assert m.shape[0] == batch.suffix_ids[0].shape[0]
    assert not m[: batch.suffix_label_start[0]].any()
    assert bool(m[-1])  # end-of-turn stays in the loss
    sub = batch.subset([1])
    assert torch.equal(sub.suffix_loss_mask[0], m)


def test_build_batch_applies_suffix_loss_mask():
    dec = SpliceDecoder(Lfm2ForCausalLM(tiny_config()))
    prefix = [torch.tensor([5, 6, 7])]
    suffix = [torch.tensor([11, 12, 13, 14, 15])]
    mask = [torch.tensor([True, True, False, False, True])]
    b = dec.build_batch(prefix, None, None, suffix, suffix_label_start=[1], suffix_loss_mask=mask)
    lab = b.labels[0, 3:].tolist()
    assert lab == [-100, 12, -100, -100, 15]
    b2 = dec.build_batch(prefix, None, None, suffix, suffix_label_start=[1])
    assert b2.labels[0, 3:].tolist() == [-100, 12, 13, 14, 15]
    with pytest.raises(ValueError):
        dec.build_batch(prefix, None, None, suffix, suffix_loss_mask=[torch.tensor([True])])


def test_sample_mask_length_checked():
    tok = _tok()
    t = Templates.from_tokenizer(tok, style="chat")
    s = Sample(
        ctx_ids=torch.tensor([1, 2, 3]),
        target_ids=torch.tensor([4, 5]),
        task="x",
        store=0,
        doc=0,
        prefix_ids=torch.tensor([9]),
        instr_ids=torch.tensor([8]),
        target_loss_mask=torch.tensor([True]),
    )
    with pytest.raises(ValueError):
        Collator(t)([s])


def test_arrow_task_store_round_trip(tmp_path):
    import pyarrow.parquet as pq

    from bgkit2.data.tasks import write_task_store_arrow

    tok = _tok()
    row = {
        "family": "compaction_chunks",
        "framing": "trajectory",
        "context": "history " * 50,
        "prompt": "Compact this.\n",
        "instruction": json.dumps(PREFIX),
        "target": json.dumps(CHUNK),
        "split": "train",
        "meta": {},
    }
    p = tmp_path / "chunks.parquet"
    write_task_store(str(p), [row, dict(row, split="eval")])
    a = tmp_path / "chunks.arrow"
    write_task_store_arrow(str(a), pq.read_table(p))
    ds = TaskDataset(str(a), ChatRenderer.from_tokenizer(tok), split="train")
    assert ds.n == 1 and ds.families == ["compaction_chunks"]
    s = ds[0]
    assert s.target_loss_mask is not None and ds.length_hint(0) > 0
    assert TaskDataset(str(a), ChatRenderer.from_tokenizer(tok), split="eval").n == 1
