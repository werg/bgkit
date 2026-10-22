"""Unit tests for the SHARED tree-node encode primitive
``bgkit.models.recursive_l1.encode_tree_node``.

Every tree-encode path (live full-backprop, cached-QA drill-down, offline
cache build) funnels through this one function, so these tests pin the
mandated child-ID injection contract with a fully stubbed encoder (no
backbone): the primitive must interleave ``[id_emb | survivors]`` per child,
IN ORDER, and pass a ``pinned`` mask marking EXACTLY the id rows.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from bgkit.models.recursive_l1 import encode_tree_node


class _StubTok:
    """One deterministic id token per child: ``"a" -> [0]``, ``"b" -> [1]`` …"""

    def encode(self, text, add_special_tokens=False):
        return [ord(text.strip()) - ord("a")]


def _stub_encoder(d: int = 4):
    """Encoder stub capturing the ``run_l1_and_project`` kwargs.

    ``embed`` rows are known + distinct so the interleaving order is
    verifiable; ``run_l1_and_project`` records its args and echoes the content
    back as the survivors so a caller can inspect it.
    """
    embed = torch.nn.Embedding(4, d)
    embed.weight.data = torch.tensor(
        [[1.0, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 0], [0, 0, 0, 0]],
    )
    captured: dict = {}

    def run_l1_and_project(**kwargs):
        captured.update(kwargs)
        content = kwargs["l1_input_embeddings"]
        l1_out = types.SimpleNamespace(
            survivor_embeddings=content,
            survivor_cu_seqlens=torch.tensor([0, content.shape[0]], dtype=torch.int32),
        )
        proj_out = types.SimpleNamespace(
            projected_embeddings=content.sum(0, keepdim=True),
        )
        return l1_out, proj_out, torch.tensor([0, 1], dtype=torch.int32)

    enc = types.SimpleNamespace(
        l0=types.SimpleNamespace(
            backbone=types.SimpleNamespace(get_input_embeddings=lambda: embed),
        ),
        run_l1_and_project=run_l1_and_project,
    )
    return enc, embed, captured


def test_interleaves_id_then_survivors_in_order_with_pinned_id_rows():
    enc, embed, captured = _stub_encoder(d=4)
    tok = _StubTok()

    surv_a = torch.tensor([[10.0, 10, 10, 10], [11, 11, 11, 11]])  # 2 rows
    surv_b = torch.tensor([[20.0, 20, 20, 20]])                    # 1 row

    proj, l1_out = encode_tree_node(
        enc, tok,
        children_ids=["a", "b"],
        children_survivors_l1in=[surv_a, surv_b],
        query_emb=None,
        ratio=0.5,
    )

    content = captured["l1_input_embeddings"]
    # [id_a | surv_a(2) | id_b | surv_b(1)] — id rows are the embed rows.
    expected = torch.stack([
        embed.weight[0],            # id "a"
        surv_a[0], surv_a[1],       # child a survivors
        embed.weight[1],            # id "b"
        surv_b[0],                  # child b survivors
    ])
    assert content.shape == (5, 4)
    assert torch.allclose(content, expected), content

    # pinned marks EXACTLY the id rows (rows 0 and 3), nothing else.
    pinned = captured["pinned_positions_l1"]
    assert pinned.dtype == torch.bool
    assert pinned.tolist() == [True, False, False, True, False]

    # One L1 segment spanning the whole node; ratio + project plumbed through.
    assert captured["l1_input_cu_seqlens"].tolist() == [0, 5]
    assert float(captured["target_ratio_l1"]) == 0.5
    assert proj is not None
    assert l1_out.survivor_embeddings.shape == (5, 4)


def test_id_embeddings_are_unbridged_and_carry_gradient():
    enc, embed, captured = _stub_encoder(d=4)
    proj, _ = encode_tree_node(
        enc, _StubTok(),
        children_ids=["a"],
        children_survivors_l1in=[torch.zeros(2, 4)],
        query_emb=None,
        ratio=0.3,
    )
    # id rows equal embed rows EXACTLY -> the primitive did NOT bridge them.
    assert torch.allclose(captured["l1_input_embeddings"][0], embed.weight[0])
    # gradient flows into embed_tokens (the model learns id token embeddings).
    proj.sum().backward()
    assert embed.weight.grad is not None
    assert torch.count_nonzero(embed.weight.grad[0]) > 0


def test_query_emb_becomes_l1_prompt():
    enc, _embed, captured = _stub_encoder(d=4)
    q = torch.randn(3, 4)
    encode_tree_node(
        enc, _StubTok(),
        children_ids=["a"],
        children_survivors_l1in=[torch.zeros(2, 4)],
        query_emb=q,
        ratio=0.5,
    )
    assert captured["prompt_embeddings_l1"] is not None
    assert captured["prompt_embeddings_l1"].shape == (3, 4)
    assert captured["prompt_cu_seqlens_l1"].tolist() == [0, 3]


def test_project_false_returns_none_proj_but_still_l1_out():
    enc, _embed, _captured = _stub_encoder(d=4)
    proj, l1_out = encode_tree_node(
        enc, _StubTok(),
        children_ids=["a"],
        children_survivors_l1in=[torch.zeros(2, 4)],
        query_emb=None,
        ratio=0.5,
        project=False,
    )
    assert proj is None
    assert l1_out.survivor_embeddings is not None


def test_rejects_length_mismatch_and_empty():
    enc, _embed, _captured = _stub_encoder(d=4)
    with pytest.raises(ValueError, match="length"):
        encode_tree_node(
            enc, _StubTok(),
            children_ids=["a", "b"],
            children_survivors_l1in=[torch.zeros(2, 4)],
            query_emb=None,
            ratio=0.5,
        )
    with pytest.raises(ValueError, match="no children"):
        encode_tree_node(
            enc, _StubTok(),
            children_ids=[],
            children_survivors_l1in=[],
            query_emb=None,
            ratio=0.5,
        )
