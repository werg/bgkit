"""Cons-mode compaction spines: ONE encoder function for trainer and eval server,
bounded gradients, an adapter that never touches ordinary slots."""

from __future__ import annotations

import json

import pytest
import torch

from bgkit2.agent.context import compaction_leaves, compaction_payload
from bgkit2.models.encoder import BgKIT2Encoder
from bgkit2.models.lfm2_bidirectional import Lfm2BidirectionalBackbone
from bgkit2.models.multilevel import encode_spines


def _enc(tiny_lfm2_model, **kw):
    torch.manual_seed(0)
    bb = Lfm2BidirectionalBackbone(tiny_lfm2_model)
    return BgKIT2Encoder(bb, head_layer=2, head_inner_dim=8, repro_head=True, **kw).eval()


def _tok(s: str) -> list[int]:
    return [3 + (ord(ch) % 40) for ch in s]


def _leaves(n: int, length: int = 40, seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(3, 50, (length + 3 * i,), generator=g) for i in range(n)]


def _perturb_spine_adapter(enc: BgKIT2Encoder) -> None:
    with torch.no_grad():
        for name, p in enc.backbone.named_parameters():
            if "lora_B.spine" in name:
                p.normal_(0, 0.2)


R = {"leaf_ratio": 0.25, "cell_ratio": 0.5}


def test_wide_cell_layout_keeps_the_chain_shallow():
    from bgkit2.data.spine import spine_cells, spine_top_reps

    assert spine_cells([100] * 3) == [[0], [1], [2]]  # short history: the plain cons chain
    assert spine_cells([100] * 10) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    long = spine_cells([1000] * 138)
    assert len(long) == 4 and sorted(j for c in long for j in c) == list(range(138))
    # a cell whose input would exceed node_max_tokens closes early (deeper, never unbounded)
    assert len(spine_cells([8000] * 40, node_max_tokens=2000)) > 4
    # leaves at 1/16, cell-to-cell at 1/4: 3 cells of one 1600-token leaf
    assert spine_top_reps([1600] * 3) == 33  # 100 -> 25; 25 + 100 -> 32; 32 + 100 -> 33


def test_spine_batched_single_and_cached_agree(tiny_lfm2_model):
    enc = _enc(tiny_lfm2_model)
    a, b = _leaves(6), _leaves(2, seed=1)
    prompt = torch.tensor(_tok("compaction: q"))
    with torch.no_grad():
        both = encode_spines(
            enc, [a, b], tok_encode=_tok, prompts=[prompt, None], grad_levels=None, **R
        )
        one = encode_spines(enc, [a], tok_encode=_tok, prompts=[prompt], **R)
        # the eval server's cache: every leaf and the cells below the top are given
        cells = one.cells[0]
        top = len(cells) - 1
        resumed = encode_spines(
            enc,
            [a],
            tok_encode=_tok,
            prompts=[prompt],
            **R,
            leaf_given=[dict(one.leaf_bridged[0])],
            cell_given=[{c: one.cell_bridged[0][c] for c in range(top)}],
        )
    assert cells == [[0, 1], [2, 3], [4, 5]] and both.cells[1] == [[0], [1]]
    assert torch.allclose(both.reps[0], one.reps[0], atol=1e-5)
    assert resumed.leaf_bridged[0] == {} and list(resumed.cell_bridged[0]) == [top]
    assert torch.allclose(resumed.reps[0], one.reps[0], atol=1e-5)
    # the chain really feeds the top: a different OLDEST leaf changes it
    other = encode_spines(
        enc, [[_leaves(1, seed=9)[0], *a[1:]]], tok_encode=_tok, prompts=[prompt], **R
    )
    assert not torch.allclose(other.reps[0], one.reps[0])


def test_spine_gradient_reaches_the_top_cells_and_their_leaves(tiny_lfm2_model):
    enc = _enc(tiny_lfm2_model).train()
    enc.inject_spine_adapter(4, 8)
    out = encode_spines(enc, [_leaves(8)], tok_encode=_tok, grad_levels=2, **R)
    cells = out.cells[0]
    assert cells == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert [out.cell_bridged[0][c].requires_grad for c in range(4)] == [False, False, True, True]
    assert [out.leaf_bridged[0][j].requires_grad for j in range(8)] == [False] * 4 + [True] * 4
    assert out.reps[0].requires_grad
    out.reps[0].sum().backward()
    assert any(p.grad is not None for p in enc.spine_parameters())


def test_spine_adapter_leaves_ordinary_slots_bit_identical(tiny_lfm2_model):
    enc = _enc(tiny_lfm2_model)
    ids = torch.randint(0, 50, (2, 32))
    mask = torch.ones(2, 32, dtype=torch.bool)
    with torch.no_grad():
        before = enc(ids, mask, ratio=0.25).reps
        spine_before = encode_spines(enc, [_leaves(2)], tok_encode=_tok, **R).reps[0]
    grads = {k: p.requires_grad for k, p in enc.named_parameters()}
    n = enc.inject_spine_adapter(4, 8)
    assert n > 0 and enc.has_spine_adapter
    _perturb_spine_adapter(enc)
    with torch.no_grad():
        after = enc(ids, mask, ratio=0.25).reps
        spine_after = encode_spines(enc, [_leaves(2)], tok_encode=_tok, **R).reps[0]
    assert torch.equal(before, after)  # ordinary slot: base encoder, bit-identical
    assert not torch.allclose(spine_before, spine_after)  # the adapter is live for spines
    # the spine tensors train; nothing else's requires_grad changed
    for k, p in enc.named_parameters():
        if ".spine." in k or k.startswith("spine_repro"):
            assert p.requires_grad, k
        else:
            assert p.requires_grad == grads[k.replace(".base_layer", "")], k


def test_spine_adapter_checkpoint_round_trip(tiny_lfm2_model):
    from bgkit2.training.core import BgKIT2Core

    plain = _enc(tiny_lfm2_model)
    old_state = {k: v.clone() for k, v in plain.state_dict().items()}
    enc = _enc(tiny_lfm2_model)
    enc.inject_spine_adapter(4, 8)
    # an old checkpoint (no adapter) loads into an encoder with the adapter
    tolerated = BgKIT2Core.load_encoder_state(enc, old_state)
    assert any(".spine." in k for k in tolerated)
    _perturb_spine_adapter(enc)
    saved = {k: v.clone() for k, v in enc.state_dict().items()}
    enc2 = _enc(tiny_lfm2_model)
    enc2.inject_spine_adapter(4, 8)
    assert BgKIT2Core.load_encoder_state(enc2, saved) == []
    for k, v in enc2.state_dict().items():
        assert torch.equal(v, saved[k]), k
    # the reverse (adapter checkpoint into a plain encoder) drops the spine tensors, loudly
    with pytest.warns(UserWarning, match="compaction-spine"):
        BgKIT2Core.load_encoder_state(_enc(tiny_lfm2_model), saved)


class _Tok:
    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": _tok(s)}


class _Renderer:
    def encode(self, text, max_tokens=None):
        ids = torch.tensor(_tok(text), dtype=torch.long)
        return ids if max_tokens is None else ids[:max_tokens]

    encode_ctx = encode  # one tokenizer: encoder-side text is the same ids


def _core(enc):
    from bgkit2.training.config import Config
    from bgkit2.training.core import BgKIT2Core

    core = BgKIT2Core()
    core.cfg2 = Config()
    core.cfg2.compaction_spine.enabled = True
    core.cfg2.compaction_spine.leaf_ratio = 0.25
    core.cfg2.compaction_spine.cell_ratio = 0.5
    core.device = torch.device("cpu")
    core.amp_dtype = torch.float32
    core.tok = _Tok()
    core.encoder = enc
    core.inject_spine_adapter()
    _perturb_spine_adapter(enc)
    return core


def test_student_spine_slot_matches_the_trainer_and_uses_its_cache(tiny_lfm2_model):
    """The eval server's incremental, cached encoding equals the trainer's
    from-scratch forward on the same spine (same function, same weights), and a
    fold re-encodes only its new leaf (plus the cells above it)."""
    from bgkit2.agent.student import StudentModel

    core = _core(_enc(tiny_lfm2_model))
    student = StudentModel.__new__(StudentModel)
    student.core, student.renderer = core, _Renderer()
    leaves = [f"assistant: step {i}\ntool: result number {i} " + "x" * (10 + i) for i in range(7)]

    def payload(k: int) -> dict:
        node = None
        for leaf in leaves[:k]:
            node = {"compaction": 1, "child": node, "leaf": leaf}
        return node

    prompt = "compaction: the question"
    for k in range(1, 8):  # the episode folds one round at a time
        r_eval, m_eval, _n = student._encode_spine_slot(payload(k), prompt)
        ids = [_Renderer().encode(x) for x in leaves[:k]]
        with torch.no_grad():
            ((r_train, _m),) = core.encode_spine_slots([ids], [_Renderer().encode(prompt)])
        assert bool(m_eval.all()) and r_eval.shape == r_train.shape, k
        assert float((r_eval - r_train).abs().max()) < 1e-4, (
            k,
            float((r_eval - r_train).abs().max()),
        )
        new_leaves, _new_cells = student.last_spine_encoded
        assert new_leaves == (k if k == 1 else 1), k  # only the new leaf is encoded
    assert compaction_leaves(payload(3)) == leaves[:3]
    assert compaction_payload(json.dumps(payload(3))) == payload(3)
