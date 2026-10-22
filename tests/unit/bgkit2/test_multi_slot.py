"""Multi-slot samples: an agent turn with several compressed tool results.
The decoder input is the ordinary rendered conversation with each tool
result's text replaced by that result's reps (a "slot" = the span those
reps occupy); the batch builder, collator, subset and training step must
handle any number of slots per sample, and single-slot batches are unchanged."""

from __future__ import annotations

import torch

from bgkit2.data.autoencode import Batch, Sample

from .test_data_and_training import _tiny_trainer


def _slots_sample(doc: int, n_extra: int = 2) -> Sample:
    return Sample(
        ctx_ids=torch.tensor([13, 14, 15, 16, 17, 18]),
        target_ids=torch.tensor([7, 8, 9]),
        task="tool_slots",
        store=-1,
        doc=doc,
        prompt_ids=torch.tensor([23, 24]),
        prefix_ids=torch.tensor([3, 4, 5]),
        instr_ids=torch.tensor([6]),
        extra_ctx=[torch.tensor([19, 20, 21, 22, 25, 26, 27]) for _ in range(n_extra)],
        extra_prompt=[torch.tensor([28]) for _ in range(n_extra)],
        mid_ids=[torch.tensor([30, 31]) for _ in range(n_extra)],
    )


def _plain_sample(doc: int) -> Sample:
    return Sample(
        ctx_ids=torch.tensor([13, 14, 15, 16]),
        target_ids=torch.tensor([7, 8]),
        task="qa",
        store=-1,
        doc=doc,
        prompt_ids=torch.tensor([23]),
        prefix_ids=torch.tensor([3, 4]),
        instr_ids=torch.tensor([5]),
    )


def test_build_batch_layout_with_extra_slots(tmp_path):
    tr = _tiny_trainer(tmp_path)
    dec = tr.decoder
    d = dec.embed_tokens.weight.shape[1]
    prefix = [torch.tensor([3, 4, 5])]
    suffix = [torch.tensor([6, 7, 8])]
    reps = torch.randn(1, 4, d)
    mask = torch.tensor([[True, True, True, False]])
    extra = [
        [
            (torch.randn(2, d), torch.tensor([True, True])),
            (torch.randn(3, d), torch.tensor([True, False, True])),
        ]
    ]
    mid = [[torch.tensor([30, 31]), torch.tensor([32])]]
    sb = dec.build_batch(prefix, reps, mask, suffix, extra=extra, mid_ids=mid)
    # [3 prefix][3 reps][2 mid][2 extra0][1 mid][2 extra1][3 suffix] = 16
    assert sb.inputs_embeds.shape[1] == 16
    assert sb.slots == [[(3, 3), (8, 2), (11, 2)]]
    assert int(sb.slot_start[0]) == 3 and int(sb.slot_len[0]) == 3
    assert (sb.labels[0, :13] == -100).all() and (sb.labels[0, 13:] != -100).all()
    # the extra reps land where the slots say
    assert torch.allclose(sb.inputs_embeds[0, 8:10].float(), extra[0][0][0].float(), atol=1e-4)
    # left padding shifts every slot by the pad
    sb2 = dec.build_batch(
        [*prefix, torch.tensor([3])],
        torch.cat([reps, reps]),
        torch.cat([mask, mask]),
        [*suffix, torch.tensor([6])],
        extra=[*extra, []],
        mid_ids=[*mid, []],
        pad_side="left",
    )
    assert sb2.slots[0] == [(3, 3), (8, 2), (11, 2)]
    pad = sb2.inputs_embeds.shape[1] - 5
    assert sb2.slots[1] == [(pad + 1, 3)]
    # no extra anywhere -> single-slot batch, slots None
    sb3 = dec.build_batch(prefix, reps, mask, suffix)
    assert sb3.slots is None


def test_collator_packs_and_subset_keeps_extra_slots(tmp_path):
    tr = _tiny_trainer(tmp_path)
    b: Batch = tr.collate([_plain_sample(0), _slots_sample(1, 2), _slots_sample(2, 1)])
    assert b.extra_input_ids.shape[0] == 3 and b.extra_owner.tolist() == [1, 1, 2]
    assert b.n_extra == [0, 2, 1]
    assert b.extra_prompt_ids.shape[0] == 3
    sub = b.subset([2, 0])
    assert sub.extra_owner.tolist() == [0] and sub.n_extra == [1, 0]
    sub2 = b.subset([0])
    assert sub2.extra_input_ids is None and sub2.mid_ids is None
    b1 = tr.collate([_plain_sample(0), _plain_sample(1)])
    assert b1.extra_input_ids is None and b1.n_extra == [0, 0]


def test_training_step_and_eval_arms_with_multi_slot_rows(tmp_path):
    tr = _tiny_trainer(tmp_path)
    batch = tr.collate([_plain_sample(0), _slots_sample(1, 2)]).to(tr.device)
    calls = []
    orig = tr.encoder.forward

    def spy(ids, mask, **kw):
        calls.append(tuple(ids.shape))
        return orig(ids, mask, **kw)

    tr.encoder.forward = spy
    total, _stats = tr.training_loss(batch)
    assert torch.isfinite(total)
    # primary slots (2 rows) + extra slots (2 rows) = two encoder calls
    assert len(calls) == 2 and calls[0][0] == 2 and calls[1][0] == 2
    total.backward()
    grads = [p.grad for p in tr.encoder.parameters() if p.grad is not None]
    assert any(float(g.abs().sum()) > 0 for g in grads)
    # eval arms: no-context, full text, reps all run on the same batch
    tr.encoder.forward = orig
    for ratio in (None, "full", 0.5):
        summary, _ = tr._arm([batch], ratio)
        assert "tool_slots" in summary and summary["tool_slots"][0] >= 0
    summary, _ = tr._arm([batch], 0.5, zero_reps=True)
    assert "tool_slots" in summary


def _noslot_sample(doc: int) -> Sample:
    """An agent's FIRST move (framing "chat"): no tool result yet, so no
    context slot -- the whole conversation is the prefix."""
    return Sample(
        ctx_ids=torch.zeros(0, dtype=torch.long),
        target_ids=torch.tensor([7, 8, 9]),
        task="tool_slots",
        store=-1,
        doc=doc,
        prompt_ids=None,
        prefix_ids=torch.tensor([3, 4, 5, 6]),
        instr_ids=torch.zeros(0, dtype=torch.long),
    )


class _MixedStore(torch.utils.data.Dataset):
    """A slot family whose store mixes slotted rows with first-move rows,
    the shape of agent_tool_slots_v4."""

    families = ["tool_slots"]

    def __init__(self):
        self.items = [_slots_sample(0, 1), _noslot_sample(1), _slots_sample(2, 1), _noslot_sample(3)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def row(self, i):
        return {"framing": "chat" if self.items[i].ctx_ids.numel() == 0 else "tool_slots"}


def test_task_eval_keeps_noslot_rows_out_of_the_rep_arms(tmp_path):
    """P28 died at its step-12000 eval: a batch made only of first-move rows
    handed the encoder 0-length documents. No-slot rows are scored on the
    floor arm only; the rep arms see the slotted rows."""
    tr = _tiny_trainer(tmp_path)
    tr.cfg2.data.task_eval_factors = [4]
    tr.cfg2.data.task_gen_samples = 0
    m = tr.task_eval_metrics(_MixedStore())
    assert m["noslot/rows"] == 2.0 and "noslot/loss" in m
    assert "x4/rep_gain_nats" in m and "noctx/loss" in m
    with __import__("pytest").raises(ValueError, match="empty documents"):
        tr.encoder(torch.zeros(2, 0, dtype=torch.long), torch.zeros(2, 0, dtype=torch.long), ratio=0.5)


def test_training_on_a_microbatch_of_only_noslot_rows(tmp_path):
    """S2 died at step ~830 (2026-09-23) on a training microbatch made only of
    first-move rows: a 0-width primary reached the encoder. Such rows train on
    their decoder CE with no primary reps (the floor arm is their only arm), and
    a mixed batch still routes its slotted rows through the encoder."""
    tr = _tiny_trainer(tmp_path)
    calls = []
    orig = tr.encoder.forward

    def spy(ids, mask, **kw):
        calls.append(tuple(ids.shape))
        return orig(ids, mask, **kw)

    tr.encoder.forward = spy
    total, stats = tr.training_loss(tr.collate([_noslot_sample(0), _noslot_sample(1)]))
    assert torch.isfinite(total) and float(total) > 0 and calls == []
    assert stats["noslot_rows"] == 2.0 and "loss/tool_slots_noslot" in stats
    total, stats = tr.training_loss(tr.collate([_slots_sample(0, 1), _noslot_sample(1)]))
    assert torch.isfinite(total) and stats["noslot_rows"] == 1.0 and calls


def test_empty_extra_slots_get_no_reps_and_never_reach_the_encoder(tmp_path):
    """S2 died at step ~2500 (2026-09-24): an empty tool result as an extra
    slot formed a chunk of its own in encode_extra, reached the encoder as an
    all-masked document and left the projection block 0 survivors. An empty
    slot has nothing to encode: it gets 0 reps, and the row still trains."""
    tr = _tiny_trainer(tmp_path)
    s = _slots_sample(0, 2)
    s.extra_ctx[1] = torch.zeros(0, dtype=torch.long)
    calls = []
    orig = tr.encoder.forward

    def spy(ids, mask, **kw):
        calls.append(int(mask.sum()))
        return orig(ids, mask, **kw)

    tr.encoder.forward = spy
    batch = tr.collate([s]).to(tr.device)
    extra = tr.encode_extra(batch, 0.5)
    assert [int(m.sum()) for _, m in extra[0]][1] == 0 and extra[0][1][0].shape[0] == 0
    assert 0 not in calls
    total, _ = tr.training_loss(tr.collate([s]))
    assert torch.isfinite(total)
    # an overview row whose only extra slot is empty (the S2 crash's shape)
    from safetensors.torch import save_file

    from bgkit2.data.node_cache import NodeCache

    d = tr.decoder.embed_tokens.weight.shape[1]
    cache_file = str(tmp_path / "roots.safetensors")
    save_file({"own/a/reps": torch.randn(5, d)}, cache_file)
    tr.tree_root_cache = tr.tree_node_cache = NodeCache(cache_file)
    o = _overview_sample(0, "own/a", n_extra=1)
    o.extra_ctx[0] = torch.zeros(0, dtype=torch.long)
    total, stats = tr.training_loss(tr.collate([o]))
    assert torch.isfinite(total) and stats["overview_rows"] == 1.0


def _overview_sample(doc: int, repo: str, n_extra: int = 0) -> Sample:
    """An agent row whose PRIMARY slot is the always-on repo overview."""
    return Sample(
        ctx_ids=torch.zeros(0, dtype=torch.long),
        target_ids=torch.tensor([7, 8, 9]),
        task="tool_slots",
        store=-1,
        doc=doc,
        prompt_ids=None,
        prefix_ids=torch.tensor([3, 4, 5]),
        instr_ids=torch.tensor([6]),
        extra_ctx=[torch.tensor([19, 20, 21, 22, 25, 26, 27]) for _ in range(n_extra)] or None,
        extra_prompt=[torch.tensor([28]) for _ in range(n_extra)] or None,
        mid_ids=[torch.tensor([30, 31]) for _ in range(n_extra)] or None,
        cached_slot=(repo, 5),
    )


def test_overview_rows_train_and_eval_from_the_root_cache(tmp_path):
    """Rows whose primary slot is a cached repo overview: the encoder never
    sees the overview (only the rows' text tool-result slots), a mixed batch
    splits cleanly, and the eval isolates the overview's contribution."""
    tr = _tiny_trainer(tmp_path)
    d = tr.decoder.embed_tokens.weight.shape[1]
    from safetensors.torch import save_file

    from bgkit2.data.node_cache import NodeCache

    cache_file = str(tmp_path / "roots.safetensors")
    save_file({"own/a/reps": torch.randn(5, d), "own/b/reps": torch.randn(3, d)}, cache_file)
    tr.tree_root_cache = tr.tree_node_cache = NodeCache(cache_file)
    calls = []
    orig = tr.encoder.forward

    def spy(ids, mask, **kw):
        calls.append(tuple(ids.shape))
        return orig(ids, mask, **kw)

    tr.encoder.forward = spy
    # overview-only rows: no encoder call at all
    batch = tr.collate([_overview_sample(0, "own/a"), _overview_sample(1, "own/b")])
    assert batch.cached_slot == ["own/a", "own/b"]
    total, stats = tr.training_loss(batch)
    assert torch.isfinite(total) and stats["overview_rows"] == 2.0 and calls == []
    assert "loss/tool_slots_overview" in stats and stats["overview_reps"] == 4.0
    # (this tiny trainer's decoder is frozen: an overview-only batch has no
    # trainable input at all; Stage D trains the decoder adapter)
    # overview + a text tool result: exactly the extra slot goes through the encoder
    batch = tr.collate([_overview_sample(0, "own/a", n_extra=1)])
    total, stats = tr.training_loss(batch)
    assert torch.isfinite(total) and len(calls) == 1 and calls[0][0] == 1
    total.backward()  # gradients reach the encoder through the text slot only
    # mixed with an ordinary text-slot row: both paths run, one loss
    calls.clear()
    batch = tr.collate([_plain_sample(0), _overview_sample(1, "own/b")])
    total, stats = tr.training_loss(batch)
    assert torch.isfinite(total) and stats["overview_rows"] == 1.0 and "loss/qa" in stats
    sub = batch.subset([1])
    assert sub.cached_slot == ["own/b"]
    # eval: with the overview / overview zeroed / no slot at all
    tr.encoder.forward = orig
    m = tr.overview_eval_metrics([tr.collate([_overview_sample(0, "own/a", n_extra=1)])])
    assert m["overview/rows"] == 1.0 and "overview/gain_vs_zeroed_nats" in m
    # a row naming a repo the cache does not hold is an error, not a silent no-slot row
    with __import__("pytest").raises(KeyError):
        tr.training_loss(tr.collate([_overview_sample(0, "own/missing")]))


def test_browse_slots_are_filled_from_the_node_cache_not_encoded(tmp_path):
    """A ``browse`` result is an EXTRA slot holding a cached directory node:
    the encoder never sees it, a text tool result beside it still does, and
    the node's reps reach the decoder in slot order."""
    tr = _tiny_trainer(tmp_path)
    d = tr.decoder.embed_tokens.weight.shape[1]
    from safetensors.torch import save_file

    from bgkit2.data.node_cache import NodeCache

    f = str(tmp_path / "nodes.safetensors")
    save_file({"own/a///reps": torch.randn(5, d), "own/a//pkg/reps": torch.randn(6, d)}, f)
    tr.tree_root_cache = tr.tree_node_cache = NodeCache(f)
    calls = []
    orig = tr.encoder.forward

    def spy(ids, mask, **kw):
        calls.append(tuple(ids.shape))
        return orig(ids, mask, **kw)

    tr.encoder.forward = spy
    s = _overview_sample(0, "own/a", n_extra=2)
    s.extra_ctx[0] = torch.zeros(0, dtype=torch.long)  # slot 1 = browse("pkg")
    s.extra_cached = [("own/a", "pkg"), None]
    batch = tr.collate([s])
    assert batch.extra_cached == [("own/a", "pkg"), None]
    extra = tr.encode_extra(batch.to(tr.device), 0.5)
    assert len(calls) == 1 and calls[0][0] == 1          # only the text slot was encoded
    assert extra[0][0][0].shape[0] == 6 and bool(extra[0][0][1].all())  # node reps, in order
    total, stats = tr.training_loss(batch)
    assert torch.isfinite(total) and stats["overview_rows"] == 1.0
    assert batch.subset([0]).extra_cached == [("own/a", "pkg"), None]
    # browse-only extras: no encoder call at all
    calls.clear()
    s2 = _overview_sample(1, "own/a", n_extra=1)
    s2.extra_ctx[0] = torch.zeros(0, dtype=torch.long)
    s2.extra_cached = [("own/a", "pkg")]
    total, _ = tr.training_loss(tr.collate([s2]))
    assert torch.isfinite(total) and calls == []


def test_prompted_readout_re_encodes_the_cached_node_input_with_the_question(tmp_path):
    """multilevel.prompted_readout: tree_root runs ONE encoder pass over the
    node's cached INPUT with the question as the prompt; the prompt changes the
    reps, the no-prompt control does not see it, and gradients reach the encoder."""
    tr = _tiny_trainer(tmp_path)
    from safetensors.torch import save_file
    from transformers import AutoTokenizer  # noqa: F401 - the tiny trainer's tokenizer is a stub

    from bgkit2.data.node_cache import NodeCache

    h = tr.encoder.backbone.config.hidden_size if hasattr(tr.encoder, "backbone") else 32
    d = tr.decoder.embed_tokens.weight.shape[1]
    f = str(tmp_path / "nodes.safetensors")
    save_file({"own/a//pkg/reps": torch.randn(4, d), "own/a//pkg/input": torch.randn(24, h)}, f)
    tr.tree_root_cache = tr.tree_node_cache = NodeCache(f)
    tr.cfg2.multilevel.prompted_readout = True
    tr.enc_tok = lambda s, add_special_tokens=False: {"input_ids": [5 + (ord(c) % 20) for c in s]}
    item = type("T", (), {"repo": "own/a", "node": "pkg", "question": "where is foo defined?"})()
    tr.encoder.eval()
    with torch.no_grad():
        r_q, m_q = tr.tree_root(item)
        tr.readout_prompt_mode = "none"
        r_n, _ = tr.tree_root(item)
        tr.readout_prompt_mode = "question"
    assert r_q.shape[1] == d and bool(m_q.all()) and r_q.shape[0] >= 1
    assert r_q.shape != r_n.shape or not torch.allclose(r_q, r_n)   # the question matters
    with __import__("pytest").raises(KeyError):
        tr.tree_node_cache.get_input("own/a", "missing")


def _spine_sample(doc: int, depth: int = 3, n_extra: int = 1) -> Sample:
    """A cons-compacted agent turn: the primary slot is a compaction SPINE."""
    s = _slots_sample(doc, n_extra)
    s.ctx_ids = torch.zeros(0, dtype=torch.long)
    s.spine_slot = [torch.tensor([13 + i, 14, 15, 16, 17, 18, 19, 20]) for i in range(depth)]
    return s


def test_spine_rows_train_through_the_spine_adapter_and_evaluate(tmp_path):
    """Rows whose primary slot is a compaction spine: the spine is encoded node
    by node (adapter active), the other slots as text; a mixed batch splits;
    gradients reach the spine adapter; the eval isolates the spine."""
    tr = _tiny_trainer(tmp_path, **{"compaction_spine.enabled": "true",
                                    "multilevel.bridge": "identity"})
    assert tr.encoder.has_spine_adapter
    assert any(g["name"] == "spine" for g in tr.param_groups())
    batch = tr.collate([_plain_sample(0), _spine_sample(1)])
    assert batch.spine_slot[0] is None and len(batch.spine_slot[1]) == 3
    total, stats = tr.training_loss(batch)
    assert torch.isfinite(total) and stats["spine_rows"] == 1.0 and "loss/qa" in stats
    total.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in tr.encoder.spine_parameters())
    m = tr.spine_eval_metrics([tr.collate([_spine_sample(0)])])
    assert m["spine/rows"] == 1.0 and "spine/gain_vs_zeroed_nats" in m
    # a spine as an EXTRA slot (e.g. behind an overview) goes through encode_extra
    s = _slots_sample(2, 2)
    s.extra_spine = [None, [torch.tensor([21, 22, 23, 24]), torch.tensor([25, 26, 27])]]
    s.extra_ctx[1] = torch.zeros(0, dtype=torch.long)
    b2 = tr.collate([s])
    total, _ = tr.training_loss(b2)
    assert torch.isfinite(total)
    # with the feature off, a spine row is an error, not a silent text slot
    tr.cfg2.compaction_spine.enabled = False
    with __import__("pytest").raises(ValueError, match="compaction_spine"):
        tr.training_loss(tr.collate([_spine_sample(0)]))


def test_compaction_spine_auto_enables_exactly_when_a_store_needs_it(tmp_path):
    """compaction_spine.enabled defaults to AUTO: a run whose task stores carry
    spine rows gets the spine adapter without anyone remembering to switch it
    on; a run without any keeps its encoder state (and resumes) unchanged."""
    import json

    from bgkit2.data.tasks import store_has_spines, write_task_store

    spine = json.dumps({"compaction": 1, "child": None, "leaf": "assistant: x"})
    row = {"family": "web_x", "framing": "tool_slots", "context": json.dumps([spine, "r"]),
           "prompt": json.dumps(["compaction: q", "p"]), "instruction": "[]", "target": "t",
           "split": "train"}
    plain = dict(row, context=json.dumps(["just text", "r"]))
    with_sp, without = str(tmp_path / "sp.parquet"), str(tmp_path / "plain.parquet")
    write_task_store(with_sp, [plain, row])
    write_task_store(without, [plain])
    assert store_has_spines(with_sp) and not store_has_spines(without)
    tr = _tiny_trainer(tmp_path)  # no task stores configured -> auto resolves off
    assert tr.cfg2.compaction_spine.enabled is False and not tr.encoder.has_spine_adapter
    tr.cfg2.compaction_spine.enabled = None
    assert tr.resolve_spine_enabled([without]) is False
    tr.cfg2.compaction_spine.enabled = None
    assert tr.resolve_spine_enabled([without, with_sp]) is True
    tr.cfg2.compaction_spine.enabled = None
    assert tr.resolve_spine_enabled([], checkpoint_state={"backbone.x.lora_A.spine.weight": 0})


def test_kd_teacher_forward_is_row_chunked_and_exact(tmp_path):
    """S2 OOMed twice at step ~17630 (2026-09-25) in the KD teacher's single no-grad forward
    over a long full-text batch. Row chunks give the same hidden states with a bounded peak."""
    tr = _tiny_trainer(tmp_path)
    dec = tr.decoder
    d = dec.embed_tokens.weight.shape[1]
    emb = torch.randn(5, 9, d)
    mask = torch.ones(5, 9, dtype=torch.long)
    mask[1, 6:] = 0
    mask[3, 4:] = 0
    with torch.no_grad():
        full = dec._hidden(inputs_embeds=emb, attention_mask=mask)
        dec.nograd_chunk_tokens = 18  # 2 rows per chunk
        chunked = dec._hidden_nograd_chunked(emb, mask)
    assert chunked.shape == full.shape
    m = mask.bool()
    assert torch.allclose(chunked[m], full[m], atol=1e-5)
