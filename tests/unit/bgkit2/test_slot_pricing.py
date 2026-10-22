"""Per-slot rep pricing (ratio.slot_reps_mode): every slot of a tool_slots row is encoded at the
rep count the eval harness prices it at (meta.slot_reps, or the prompt-inferred tool), within the
jitter band; rows without prices and the "row" mode are unchanged."""

from __future__ import annotations

import json
import math
import random

import torch

from bgkit2.data.autoencode import Batch, Sample
from bgkit2.data.memory_framing import MEMORY_INDEX_PROMPT, index_reps
from bgkit2.data.slot_pricing import price, prompt_tool, row_slot_reps
from bgkit2.training.config import Config
from bgkit2.training.core import BgKIT2Core

from .test_data_and_training import _tiny_trainer


def test_prompt_tool_and_prices():
    assert prompt_tool("web search eiffel: when built") == "web_search"
    assert prompt_tool("page Eiffel Tower: year") == "web_open"
    assert prompt_tool("memory search cat: name") == "memory_search"
    assert prompt_tool("session 2023/05/01 (Mon) 10:00: cat") == "memory_read"
    assert prompt_tool(MEMORY_INDEX_PROMPT) == "memory_index"
    assert prompt_tool("read src/a.py: the bug") == "read"
    assert prompt_tool("bash cd /workspace/x && ls") == "bash"
    assert prompt_tool("compaction: fix the bug") == "compaction"
    assert prompt_tool("contents of src") is None  # a cached tree node
    assert prompt_tool("What is my cat called?") is None  # a question: not a harness slot
    assert prompt_tool("") is None
    assert price("web_search", 100) == 25 and price("web_open", 160) == 10
    assert price("read", 81) == 11 and price("memory_index", 640) == index_reps(640)
    assert price(None, 100) == -1
    # meta wins when it has one entry per slot; unpriced slots are forced to -1
    sr, src = row_slot_reps({"slot_reps": [5, 7, 9]}, ["a", "b", "c"], [10, 10, 10],
                            [False, True, False])
    assert (sr, src) == ([5, -1, 9], "meta")
    sr, src = row_slot_reps({}, ["web search x", "page y", "???"], [40, 160, 50])
    assert (sr, src) == ([10, 10, -1], "prompt")


def _core(mode: str, jitter=(1.0, 1.0)) -> BgKIT2Core:
    core = BgKIT2Core()
    core.cfg2 = Config()
    core.cfg2.ratio.slot_reps_mode = mode
    core.cfg2.ratio.slot_reps_jitter = list(jitter)
    core.device = torch.device("cpu")
    core.select_rng = random.Random(0)
    return core


def _batch(src=("meta", None, "prompt")) -> Batch:
    """3 rows: row 0 (meta) primary 100 tokens + 2 extras (40, 80); row 1 unpriced;
    row 2 (prompt-inferred) primary 50 tokens + 1 extra (64)."""
    mask = torch.zeros(3, 100, dtype=torch.long)
    for i, n in enumerate((100, 60, 50)):
        mask[i, :n] = 1
    emask = torch.zeros(3, 80, dtype=torch.long)
    for r, n in enumerate((40, 80, 64)):
        emask[r, :n] = 1
    return Batch(
        input_ids=torch.zeros(3, 100, dtype=torch.long), attention_mask=mask,
        prefix_ids=[torch.tensor([1])] * 3, suffix_ids=[torch.tensor([2])] * 3,
        suffix_label_start=[0] * 3, target_ids=[torch.tensor([2])] * 3,
        instr_ids=[torch.tensor([0])] * 3, tasks=["memory_agent_v1", "qa", "web_x"],
        doc_keys=[(0, 0), (0, 1), (0, 2)],
        extra_input_ids=torch.zeros(3, 80, dtype=torch.long), extra_attention_mask=emask,
        extra_owner=torch.tensor([0, 0, 2]),
        mid_ids=[[torch.tensor([3]), torch.tensor([3])], [], [torch.tensor([3])]],
        slot_reps=torch.tensor([6.0, -1.0, 12.0]),
        extra_slot_reps=torch.tensor([20.0, -1.0, 16.0]),
        slot_reps_src=list(src),
    )


def test_slot_ratios_follow_the_harness_prices():
    b = _batch()
    fam, extra = _core("row").slot_ratios(b, 0.25)
    assert fam == 0.25 and extra is None  # the old behaviour
    vec, extra = _core("harness").slot_ratios(b, 0.25)
    assert torch.allclose(vec, torch.tensor([6 / 100, 0.25, 12 / 50]))
    # extra slot 1 is unpriced -> its row's ratio; slot 0 = 20 reps of 40, row 2's 16 of 64
    assert torch.allclose(extra, torch.tensor([20 / 40, 0.25, 16 / 64]))
    # "meta" accepts only meta-priced rows: row 2 (prompt-inferred) keeps the row ratio
    vec, extra = _core("meta").slot_ratios(b, 0.25)
    assert torch.allclose(vec, torch.tensor([6 / 100, 0.25, 0.25]))
    assert torch.allclose(extra, torch.tensor([0.5, 0.25, 0.25]))
    # a family budget still sets the row ratio of the unpriced slots
    c = _core("meta")
    c.cfg2.ratio.family_rep_budget = {"qa": [30, 30]}
    vec, extra = c.slot_ratios(b, 0.25)
    assert abs(float(vec[1]) - 30 / 60) < 1e-6


def test_slot_ratio_jitter_stays_in_band():
    b = _batch()
    core = _core("meta", jitter=(0.5, 2.0))
    seen = []
    for _ in range(200):
        vec, extra = core.slot_ratios(b, 0.25)
        k = float(vec[0]) * 100  # reps granted to the 6-rep primary
        assert 3 - 1e-4 <= k <= 12 + 1e-4
        seen.append(k)
        k2 = float(extra[0]) * 40  # the 20-rep extra slot (clamped at 1/2 = 20 reps)
        assert 10 - 1e-4 <= k2 <= 20 + 1e-4
    assert min(seen) < 4.5 and max(seen) > 8  # the band is actually used
    assert math.isclose(sum(math.log(x / 6) for x in seen) / len(seen), 0.0, abs_tol=0.15)


def _priced_sample(doc: int, reps: list[int] | None) -> Sample:
    return Sample(
        ctx_ids=torch.arange(10, 30),  # 20 tokens
        target_ids=torch.tensor([7, 8, 9]), task="memory_agent_v1", store=-1, doc=doc,
        prompt_ids=torch.tensor([23, 24]), prefix_ids=torch.tensor([3, 4, 5]),
        instr_ids=torch.tensor([6]),
        extra_ctx=[torch.arange(40, 54), torch.arange(60, 74)],  # 14 tokens each
        extra_prompt=[torch.tensor([28]), torch.tensor([29])],
        mid_ids=[torch.tensor([30, 31]), torch.tensor([32])],
        slot_reps=reps, slot_reps_src="meta" if reps else None,
    )


def test_collate_subset_and_training_step_price_each_slot(tmp_path):
    tr = _tiny_trainer(tmp_path)
    b = tr.collate([_priced_sample(0, [2, 3, 7]), _priced_sample(1, None)])
    assert b.slot_reps.tolist() == [2.0, -1.0]
    assert b.extra_slot_reps.tolist() == [3.0, 7.0, -1.0, -1.0]
    assert b.slot_reps_src == ["meta", None]
    sub = b.subset([1, 0])
    assert sub.slot_reps.tolist() == [-1.0, 2.0] and sub.slot_reps_src == [None, "meta"]
    assert sub.extra_slot_reps.tolist() == [3.0, 7.0, -1.0, -1.0]  # extra rows keep their order
    # no priced sample -> no price tensors at all (old batches unchanged)
    b0 = tr.collate([_priced_sample(0, None)])
    assert b0.slot_reps is None and b0.extra_slot_reps is None
    # the training step hands the encoder each slot's own ratio
    tr.cfg2.ratio.slot_reps_mode = "meta"
    tr.cfg2.ratio.slot_reps_jitter = [1.0, 1.0]
    tr.cfg2.ratio.p_random_selection = 0.0
    batch = b.to(tr.device)
    seen = []
    orig = tr.encoder.forward

    def spy(ids, mask, **kw):
        r = kw.get("ratio")
        seen.append((int(ids.shape[0]), r.tolist() if torch.is_tensor(r) else r))
        return orig(ids, mask, **kw)

    tr.encoder.forward = spy
    total, stats = tr.training_loss(batch)
    assert torch.isfinite(total)
    prim = seen[0][1]
    assert abs(prim[0] - 2 / 20) < 1e-6  # priced primary; row 1 keeps the step ratio
    extras = sorted(x for _, r in seen[1:] for x in (r if isinstance(r, list) else [r]))
    assert any(abs(x - 3 / 14) < 1e-6 for x in extras)
    assert any(abs(x - 7 / 14) < 1e-6 for x in extras)
    assert stats["ratio/priced_slots"] == 3.0
    # the same row in "row" mode: every extra slot follows its row
    tr.cfg2.ratio.slot_reps_mode = "row"
    seen.clear()
    tr.training_loss(tr.collate([_priced_sample(0, [2, 3, 7])]).to(tr.device))
    r_prim = seen[0][1]
    r_ext = seen[1][1]
    r_prim = r_prim[0] if isinstance(r_prim, list) else r_prim
    assert all(abs(x - r_prim) < 1e-6 for x in (r_ext if isinstance(r_ext, list) else [r_ext]))


def test_task_rows_carry_their_slot_prices(tmp_path):
    from bgkit2.data.tasks import ChatRenderer, TaskDataset, write_task_store
    from bgkit2.data.templates import SPLICE_SENTINEL

    from .test_tasks import _real_tok

    tok = _real_tok()
    msgs = [{"role": "user", "content": f"Memory:\n{SPLICE_SENTINEL}\n{SPLICE_SENTINEL}\nQ?"}]
    base = {"family": "memory_agent_v1", "framing": "tool_slots",
            "context": json.dumps(["one two three four", "web page text here"]),
            "instruction": json.dumps(msgs), "target": "A", "split": "train"}
    rows = [dict(base, prompt=json.dumps([MEMORY_INDEX_PROMPT, "page P: q"]),
                 meta={"slot_reps": [5, 9]}),
            dict(base, prompt=json.dumps([MEMORY_INDEX_PROMPT, "page P: q"]), meta={})]
    p = tmp_path / "s.parquet"
    write_task_store(str(p), rows)
    ds = TaskDataset(str(p), ChatRenderer.from_tokenizer(tok), split="train",
                     max_ctx_tokens=64, max_target_tokens=16)
    s0, s1 = ds._slots_item(0, ds.row(0)), ds._slots_item(1, ds.row(1))
    assert s0.slot_reps == [5, 9] and s0.slot_reps_src == "meta"
    n1 = int(s1.extra_ctx[0].shape[0])
    assert s1.slot_reps_src == "prompt" and s1.slot_reps == [index_reps(1), price("web_open", n1)]
