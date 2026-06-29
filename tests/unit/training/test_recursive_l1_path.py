"""Unit tests for the recursive-L1 (Phase 3) path-selective browse encode.

Covers, with NO GPU and a fully stubbed encoder:

1. Browse-node sentinel template wiring + back-compat (flag OFF → byte-for-byte
   identical to the legacy text-only browse path).
2. Path-selective gradient scope: live search path carries gradient; off-path
   siblings are read from the L1-tree cache DETACHED (no grad).
3. Browse sentinel splice: each browse turn injects a dense node-rep
   EmbeddingSegment without disturbing the surrounding token stream.

These mirror the stubbing pattern in
``tests/unit/training/test_kr_kb_trainer_pieces.py`` (build the trainer via
``__new__`` and fill only the attributes the path under test reads).
"""

from __future__ import annotations

import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.bgkit_tool_template import (
    BGKIT_BROWSE_SENTINEL,
    TrajectoryTurn,
    tokenize_trajectory,
)
from bgkit.models.decoder import EmbeddingSegment, TokenSegment

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Append-idempotent fake chat template + char-level encoder.

    Identical contract to the integration test's tokenizer: rendered message
    lists form a prefix chain so ``tokenize_trajectory``'s diff logic works.
    """

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size

    def _format_message(self, m: dict) -> str:
        role = m.get("role", "user")
        if role == "assistant" and m.get("tool_calls"):
            call = m["tool_calls"][0]["function"]
            import json

            args_str = json.dumps(call.get("arguments", {}), sort_keys=True)
            return f"<assistant_call name={call['name']} args={args_str}>"
        if role == "tool":
            return f"<tool name={m.get('name', '')}>{m.get('content', '')}</tool>"
        content = m.get("content", "")
        return f"<{role}>{content}</{role}>"

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, tools=None,
    ) -> str:
        del tokenize, add_generation_prompt, tools
        return "".join(self._format_message(m) for m in messages)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if not text:
            return []
        return [((ord(c) * 131 + 17) % (self.vocab_size - 1)) + 1 for c in text]


class _StubNode:
    def __init__(self, node_id: str, children: list[str]):
        self.id = node_id
        self.children = tuple(children)


class _StubTree:
    def __init__(self, nodes: dict[str, _StubNode]):
        self._nodes = nodes

    def __contains__(self, nid: str) -> bool:
        return nid in self._nodes

    def get(self, nid: str) -> _StubNode:
        return self._nodes[nid]


class _StubTreeCache:
    """Minimal SurvivorBlockCache surface: has/get over an in-memory dict."""

    def __init__(self, data: dict[tuple[str, str], torch.Tensor]):
        self._data = data
        self.gets: list[tuple[str, str]] = []

    def has(self, dataset: str, node_id: str) -> bool:
        return (dataset, node_id) in self._data

    def get(self, dataset: str, node_id: str) -> torch.Tensor:
        self.gets.append((dataset, node_id))
        return self._data[(dataset, node_id)]


class _RecorderEncoder(torch.nn.Module):
    """Stub encoder recording every ``l1_auto_reproduce`` input's grad status.

    ``l1_auto_reproduce`` / ``run_l1_and_project`` pass inputs through trainable
    Linears so live outputs ``requires_grad``; ``run_l1_and_project`` returns
    survivor + projected namespaces matching the real encoder's surface.
    """

    def __init__(self, dim: int, dec_dim: int, vocab: int = 512):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        self.W_bridge = torch.nn.Linear(dim, dim, bias=False)
        self.W_l1 = torch.nn.Linear(dim, dim, bias=False)
        self.W_proj = torch.nn.Linear(dim, dec_dim, bias=False)
        # LIVE-L0 weight: per-repo tests route leaf L0 survivors through this
        # so the shared tree's L0 leaves carry gradient (vs the frozen-cache
        # stub used by the per-sample full-backprop tests).
        self.W_l0 = torch.nn.Linear(dim, dim, bias=False)
        self.l0 = SimpleNamespace(
            backbone=SimpleNamespace(get_input_embeddings=lambda: self.embed),
            auto_reproduce=lambda x: x,  # L0-out -> L1-in identity (base leaf)
        )
        self.bridge_input_grad: list[bool] = []
        # Recorded bridge-input tensors (retain_grad'd when they require grad)
        # so tests can assert WHICH inputs receive gradient after backward.
        self.bridge_inputs: list[torch.Tensor] = []
        self.training = True

    def l1_auto_reproduce(self, x: torch.Tensor) -> torch.Tensor:
        self.bridge_input_grad.append(bool(x.requires_grad))
        if x.requires_grad:
            x.retain_grad()
        self.bridge_inputs.append(x)
        return self.W_bridge(x)

    def run_l1_and_project(
        self,
        *,
        l1_input_embeddings: torch.Tensor,
        l1_input_cu_seqlens: torch.Tensor,
        target_ratio_l1: float,
        **kwargs,
    ):
        n = int(l1_input_embeddings.shape[0])
        keep = l1_input_embeddings[: max(1, n // 2)]
        # Both survivor (L1-output) and projected reps pass through trainable
        # Linears, so live encodes carry gradient back to ``keep``'s producers.
        surv = self.W_l1(keep)
        l1_out = SimpleNamespace(
            survivor_embeddings=surv,
            survivor_cu_seqlens=torch.tensor(
                [0, surv.shape[0]], dtype=torch.int32,
            ),
        )
        proj_out = SimpleNamespace(projected_embeddings=self.W_proj(keep))
        proj_cu = torch.tensor([0, surv.shape[0]], dtype=torch.int32)
        return l1_out, proj_out, proj_cu


def _make_trainer(encoder, tokenizer, tree, tree_cache, dim, dec_dim):
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    t.device = torch.device("cpu")
    t.encoder = encoder
    t.encoder_tokenizer = tokenizer
    t.tokenizer = tokenizer
    t._trees = {"toy": tree}
    t._l1_tree_cache = tree_cache
    t._recursive_l1 = True
    t._recursive_l1_retention = 0.5
    t.lora_router = None
    t.decoder = SimpleNamespace(hidden_dim=dec_dim)
    t._ablation_mode = None
    return t


def _sample(trajectory):
    return SimpleNamespace(
        dataset_name="toy", question="what is X?", trajectory=trajectory,
    )


# ---------------------------------------------------------------------------
# 1. Template: sentinel emission + back-compat
# ---------------------------------------------------------------------------


def test_browse_sentinel_off_by_default():
    tok = _FakeTokenizer()
    traj = [
        TrajectoryTurn(kind="browse", args={"id": "root"}, response="root listing"),
        TrajectoryTurn(kind="bgkit", args={"ids": ["a"], "query": "q"}),
        TrajectoryTurn(kind="answer", response="the answer"),
    ]
    rendered = tokenize_trajectory(tok, "sys", "q?", traj)
    assert rendered.browse_sentinel_positions == []
    # No browse sentinel tokens leaked into the stream.
    sent_ids = tok.encode(BGKIT_BROWSE_SENTINEL)
    ids = rendered.token_ids.tolist()
    assert not _contains_subseq(ids, sent_ids)


def test_browse_sentinel_emitted_when_enabled():
    tok = _FakeTokenizer()
    traj = [
        TrajectoryTurn(kind="browse", args={"id": "root"}, response="root listing"),
        TrajectoryTurn(kind="browse", args={"id": "mid"}, response="mid listing"),
        TrajectoryTurn(kind="bgkit", args={"ids": ["a"], "query": "q"}),
        TrajectoryTurn(kind="answer", response="the answer"),
    ]
    rendered = tokenize_trajectory(tok, "sys", "q?", traj, browse_node_sentinel=True)
    # One sentinel per browse turn, aligned with browse_turns.
    assert len(rendered.browse_turns) == 2
    assert len(rendered.browse_sentinel_positions) == 2
    sent_ids = tok.encode(BGKIT_BROWSE_SENTINEL)
    ids = rendered.token_ids.tolist()
    for pos in rendered.browse_sentinel_positions:
        assert ids[pos : pos + len(sent_ids)] == sent_ids
    # Sentinel positions are loss-masked (tool-response region, never trained).
    lm = rendered.loss_mask.tolist()
    for pos in rendered.browse_sentinel_positions:
        assert not any(lm[pos : pos + len(sent_ids)])


def _contains_subseq(seq: list[int], sub: list[int]) -> bool:
    if not sub:
        return False
    return any(
        seq[i : i + len(sub)] == sub
        for i in range(len(seq) - len(sub) + 1)
    )


# ---------------------------------------------------------------------------
# 2. Path-selective gradient scope
# ---------------------------------------------------------------------------


def test_path_selective_gradient_scope():
    dim, dec_dim = 8, 8
    tok = _FakeTokenizer()
    enc = _RecorderEncoder(dim, dec_dim)

    # Tree: root -> {mid, sib_root}; mid -> {leaf_gold, sib_mid}.
    tree = _StubTree({
        "root": _StubNode("root", ["mid", "sib_root"]),
        "mid": _StubNode("mid", ["leaf_gold", "sib_mid"]),
        "sib_root": _StubNode("sib_root", []),
        "sib_mid": _StubNode("sib_mid", []),
        "leaf_gold": _StubNode("leaf_gold", []),
    })
    cache = _StubTreeCache({
        ("toy", "sib_root"): torch.randn(3, dim),
        ("toy", "sib_mid"): torch.randn(2, dim),
    })
    t = _make_trainer(enc, tok, tree, cache, dim, dec_dim)

    # Base case: live leaf l1out (requires grad — it is the live search path).
    live_leaf = torch.randn(4, dim, requires_grad=True)
    t._encode_path_leaf_l1out = types.MethodType(
        lambda self, dataset, node, q_emb: live_leaf, t,
    )

    traj = [
        TrajectoryTurn(kind="browse", args={"id": "root"}, response="r"),
        TrajectoryTurn(kind="browse", args={"id": "mid"}, response="m"),
        TrajectoryTurn(kind="bgkit", args={"ids": ["leaf_gold"], "query": "q"}),
        TrajectoryTurn(kind="answer", response="a"),
    ]
    rendered = tokenize_trajectory(tok, "sys", "q?", traj, browse_node_sentinel=True)
    reps = t._recursive_browse_node_reps(_sample(traj), rendered)

    # One projected rep per browse turn, all carrying gradient (live path).
    assert len(reps) == 2
    for rep in reps:
        assert rep is not None
        assert rep.requires_grad, "live-path node-rep must carry gradient"
        assert rep.shape[-1] == dec_dim

    # Gradient scope: off-path cache reads are DETACHED (requires_grad False);
    # the live leaf + live on-path child carry gradient. Exactly one off-path
    # sibling is read (sib_root, at the interior root node). The deepest node
    # (mid) uses the live base-case leaf and does NOT read its children from
    # cache (no double-counting), so sib_mid is never fetched.
    assert ("toy", "sib_root") in cache.gets
    assert ("toy", "sib_mid") not in cache.gets
    n_detached = sum(1 for g in enc.bridge_input_grad if not g)
    n_live = sum(1 for g in enc.bridge_input_grad if g)
    assert n_detached == 1, f"expected 1 detached off-path read, got {n_detached}"
    assert n_live >= 2, "live leaf + live on-path child must carry gradient"

    # Backprop reaches the live path (encoder bridge/proj weights get grad)
    # but never the detached cache tensors.
    loss = sum(r.float().pow(2).sum() for r in reps)
    loss.backward()
    assert enc.W_bridge.weight.grad is not None
    assert cache._data[("toy", "sib_root")].grad is None


# ---------------------------------------------------------------------------
# 3. Browse sentinel splice into decoder segments
# ---------------------------------------------------------------------------


def test_browse_node_rep_spliced_into_segments():
    dim, dec_dim = 8, 8
    tok = _FakeTokenizer()
    enc = _RecorderEncoder(dim, dec_dim)
    tree = _StubTree({
        "root": _StubNode("root", ["mid"]),
        "mid": _StubNode("mid", ["leaf_gold"]),
        "leaf_gold": _StubNode("leaf_gold", []),
    })
    cache = _StubTreeCache({})
    t = _make_trainer(enc, tok, tree, cache, dim, dec_dim)
    t._encode_path_leaf_l1out = types.MethodType(
        lambda self, dataset, node, q_emb: torch.randn(4, dim, requires_grad=True),
        t,
    )

    traj = [
        TrajectoryTurn(kind="browse", args={"id": "root"}, response="r"),
        TrajectoryTurn(kind="browse", args={"id": "mid"}, response="m"),
        TrajectoryTurn(kind="answer", response="a"),
    ]
    rendered = tokenize_trajectory(tok, "sys", "q?", traj, browse_node_sentinel=True)
    reps = t._recursive_browse_node_reps(_sample(traj), rendered)

    sent_ids = tok.encode(BGKIT_BROWSE_SENTINEL)
    prep = {
        "rendered": rendered,
        "token_ids": rendered.token_ids,
        "loss_mask": rendered.loss_mask,
        "sentinel_len": 1,
        "topic_sentinel_len": 1,
        "topic_block": None,
        "browse_node_reps": reps,
        "browse_sentinel_len": len(sent_ids),
    }
    segments, _trace = t._assemble_sample_segments(prep, [])

    emb_segs = [s for s in segments if isinstance(s, EmbeddingSegment)]
    tok_segs = [s for s in segments if isinstance(s, TokenSegment)]
    # One EmbeddingSegment per browse turn; no sentinel tokens survive in the
    # token segments.
    assert len(emb_segs) == 2
    for seg in tok_segs:
        ids = seg.token_ids.squeeze(0).tolist()
        assert not _contains_subseq(ids, sent_ids)


def test_recursive_path_noop_when_flag_off():
    """With the flag OFF, no browse reps are computed and no sentinel emitted —
    the legacy text-only browse path is preserved."""
    dim, dec_dim = 8, 8
    tok = _FakeTokenizer()
    enc = _RecorderEncoder(dim, dec_dim)
    tree = _StubTree({"root": _StubNode("root", [])})
    t = _make_trainer(enc, tok, tree, _StubTreeCache({}), dim, dec_dim)
    t._recursive_l1 = False

    traj = [
        TrajectoryTurn(kind="browse", args={"id": "root"}, response="listing"),
        TrajectoryTurn(kind="answer", response="a"),
    ]
    # Flag-off tokenization → no sentinels.
    rendered = tokenize_trajectory(tok, "sys", "q?", traj, browse_node_sentinel=False)
    assert rendered.browse_sentinel_positions == []
    # No encoder bridge calls should occur for browse in the flag-off path
    # (we don't even call the recursive method when positions are empty).
    assert enc.bridge_input_grad == []


# ---------------------------------------------------------------------------
# 4. Full-backprop mode: gradient reaches OFF-PATH siblings
# ---------------------------------------------------------------------------


def _stub_l0_for_full_backprop(trainer, dim: int):
    """Stub L0 leaf access so full-backprop leaves resolve to frozen
    (non-grad) cached L0 survivors — the precomputed input."""
    trainer._resolve_article_ids = types.MethodType(
        lambda self, dataset, ids: [str(i) for i in ids], trainer,
    )

    def fake_l0(self, dataset, article_ids, query_emb=None):
        # Frozen/cached L0 survivors: a non-grad leaf tensor. cu is one
        # segment spanning all rows (single node).
        flat = torch.randn(2 * max(1, len(article_ids)), dim)
        cu = torch.tensor([0, flat.shape[0]], dtype=torch.int32)
        return flat, cu

    trainer._l0_for_articles = types.MethodType(fake_l0, trainer)


def test_full_backprop_reaches_off_path_sibling():
    """In full_backprop mode the WHOLE subtree is re-encoded live, so gradient
    reaches OFF-PATH siblings — contrast the detached path-selective mode."""
    dim, dec_dim = 8, 8
    tok = _FakeTokenizer()
    enc = _RecorderEncoder(dim, dec_dim)
    # root browses to two leaf children: one on-path, one off-path. Both are
    # re-encoded live in full_backprop mode.
    tree = _StubTree({
        "root": _StubNode("root", ["onpath_leaf", "offpath_leaf"]),
        "onpath_leaf": _StubNode("onpath_leaf", []),
        "offpath_leaf": _StubNode("offpath_leaf", []),
    })
    t = _make_trainer(enc, tok, tree, None, dim, dec_dim)
    t._recursive_l1_full_backprop = True
    _stub_l0_for_full_backprop(t, dim)

    traj = [
        TrajectoryTurn(kind="browse", args={"id": "root"}, response="r"),
        TrajectoryTurn(kind="bgkit", args={"ids": ["onpath_leaf"], "query": "q"}),
        TrajectoryTurn(kind="answer", response="a"),
    ]
    rendered = tokenize_trajectory(tok, "sys", "q?", traj, browse_node_sentinel=True)
    reps = t._recursive_browse_node_reps(_sample(traj), rendered)

    # One projected rep for the single browse turn (root), carrying gradient.
    assert len(reps) == 1
    assert reps[0] is not None and reps[0].requires_grad

    # ALL children are re-encoded LIVE — no detached off-path reads. Both
    # of root's children (on-path + off-path) appear as live bridge inputs.
    assert len(enc.bridge_input_grad) == 2
    assert all(enc.bridge_input_grad), (
        "full_backprop must re-encode every child LIVE (no detach)"
    )

    # Backprop reaches the OFF-PATH sibling's L1 encode. root.children order
    # is [onpath_leaf, offpath_leaf] → bridge_inputs[1] is the off-path one.
    reps[0].float().pow(2).sum().backward()
    offpath_input = enc.bridge_inputs[1]
    assert offpath_input.grad is not None, (
        "off-path sibling subtree must receive gradient in full_backprop mode"
    )
    # The shared L1 weight (used by every node's encode) gets gradient too.
    assert enc.W_l1.weight.grad is not None


def test_full_backprop_vs_path_selective_off_path_grad_contrast():
    """Same tree, two modes: full_backprop gives the off-path sibling gradient;
    path-selective detaches it (no gradient)."""
    dim, dec_dim = 8, 8

    def build(full_backprop: bool):
        tok = _FakeTokenizer()
        enc = _RecorderEncoder(dim, dec_dim)
        tree = _StubTree({
            "root": _StubNode("root", ["mid", "sib_root"]),
            "mid": _StubNode("mid", ["leaf"]),
            "leaf": _StubNode("leaf", []),
            "sib_root": _StubNode("sib_root", []),
        })
        cache = _StubTreeCache({("toy", "sib_root"): torch.randn(2, dim)})
        t = _make_trainer(enc, tok, tree, cache if not full_backprop else None,
                          dim, dec_dim)
        t._recursive_l1_full_backprop = full_backprop
        _stub_l0_for_full_backprop(t, dim)
        traj = [
            TrajectoryTurn(kind="browse", args={"id": "root"}, response="r"),
            TrajectoryTurn(kind="browse", args={"id": "mid"}, response="m"),
            TrajectoryTurn(kind="bgkit", args={"ids": ["leaf"], "query": "q"}),
            TrajectoryTurn(kind="answer", response="a"),
        ]
        rendered = tokenize_trajectory(
            tok, "sys", "q?", traj, browse_node_sentinel=True,
        )
        t._recursive_browse_node_reps(_sample(traj), rendered)
        return enc

    enc_full = build(full_backprop=True)
    enc_sel = build(full_backprop=False)

    # Full-backprop: no detached bridge inputs (sib_root re-encoded live).
    assert all(enc_full.bridge_input_grad)
    # Path-selective: at least one detached bridge input (sib_root from cache).
    assert not all(enc_sel.bridge_input_grad)
    assert any(not g for g in enc_sel.bridge_input_grad)


# ---------------------------------------------------------------------------
# 5. PER-REPO full-backprop: shared window-0 tree encoded ONCE per repo
# ---------------------------------------------------------------------------


def _stub_live_l0_per_repo(trainer, enc, dim: int):
    """Stub leaf L0 access so the shared tree's diff leaves are LIVE — each
    leaf's survivors pass through ``enc.W_l0`` (trainable) so the tree carries
    gradient back to L0. Records the per-call ``query_emb`` so tests can assert
    the tree is encoded with the GENERAL prompt."""
    trainer._resolve_article_ids = types.MethodType(
        lambda self, dataset, ids: [str(i) for i in ids], trainer,
    )
    enc.l0_query_embs: list[torch.Tensor] = []

    def live_l0(self, dataset, article_ids, query_emb=None):
        enc.l0_query_embs.append(query_emb)
        n = 2 * max(1, len(article_ids))
        ids = torch.arange(n, dtype=torch.long) % enc.embed.num_embeddings
        flat = enc.W_l0(enc.embed(ids))  # (n, dim), requires grad
        cu = torch.tensor([0, flat.shape[0]], dtype=torch.int32)
        return flat, cu

    trainer._l0_for_articles = types.MethodType(live_l0, trainer)


def _make_per_repo_trainer(encoder, tokenizer, tree, dim, dec_dim, *, step=0,
                           l0_cfg=None, l1_cfg=None):
    t = _make_trainer(encoder, tokenizer, tree, None, dim, dec_dim)
    t.step_cfg = {}
    t.global_step = step
    t._recursive_l1_full_backprop = True
    t._per_repo_full_backprop = True
    t._recursive_l0_retention_cfg = l0_cfg or {"start": 0.15, "end": 0.05, "ramp_steps": 4000}
    t._recursive_l1_retention_cfg = l1_cfg or {"start": 0.30, "end": 0.05, "ramp_steps": 4000}
    t._recursive_l0_override = None
    t._shared_tree_memo = None
    t._per_repo_shared_tree_active = False
    t._recursive_general_prompt = "REPO HISTORY GENERAL PROMPT"
    return t


def _commit_repro_tree():
    # window -> [c4] -> [commitA, commitB]; commits are leaf-tag nodes.
    return _StubTree({
        "repo/w000": _StubNode("repo/w000", ["repo/w000/c4_0"]),
        "repo/w000/c4_0": _StubNode("repo/w000/c4_0", ["commitA", "commitB"]),
        "commitA": _StubNode("commitA", []),
        "commitB": _StubNode("commitB", []),
    })


def test_per_repo_shared_tree_encoded_once_not_per_file_sample():
    dim, dec_dim = 8, 8
    tok = _FakeTokenizer()
    enc = _RecorderEncoder(dim, dec_dim)
    tree = _commit_repro_tree()
    t = _make_per_repo_trainer(enc, tok, tree, dim, dec_dim)

    _stub_live_l0_per_repo(t, enc, dim)

    # Encode the shared window-0 subtree ONCE.
    memo, stats = t._compute_shared_repo_tree("toy", "repo/w000")
    # Every node (2 commits + c4 + window) gets a non-None rep.
    assert stats["nodes"] == 4
    for nid in ("repo/w000", "repo/w000/c4_0", "commitA", "commitB"):
        assert nid in memo and memo[nid][0] is not None
    bridge_calls_after_tree = len(enc.bridge_input_grad)
    assert bridge_calls_after_tree > 0

    # Now THREE file-samples in this repo each look up the shared reps for
    # their browse turns — NO re-encode (bridge-call count must not grow).
    t._shared_tree_memo = memo
    t._per_repo_shared_tree_active = True
    for _ in range(3):
        traj = [
            TrajectoryTurn(kind="browse", args={"id": "repo/w000"}, response="w"),
            TrajectoryTurn(kind="browse", args={"id": "repo/w000/c4_0"}, response="c"),
            TrajectoryTurn(kind="bgkit", args={"ids": ["commitA"], "query": "q"}),
            TrajectoryTurn(kind="answer", response="blob"),
        ]
        rendered = tokenize_trajectory(tok, "sys", "q?", traj, browse_node_sentinel=True)
        reps = t._recursive_browse_node_reps(_sample(traj), rendered)
        assert len(reps) == 2
        # Browse reps are the SHARED memo tensors (same object identity).
        assert reps[0] is memo["repo/w000"][0]
        assert reps[1] is memo["repo/w000/c4_0"][0]
    assert len(enc.bridge_input_grad) == bridge_calls_after_tree, (
        "browse lookups must NOT re-encode the shared tree per file-sample"
    )


def test_per_repo_shared_tree_l0_receives_gradient():
    dim, dec_dim = 8, 8
    tok = _FakeTokenizer()
    enc = _RecorderEncoder(dim, dec_dim)
    t = _make_per_repo_trainer(enc, tok, _commit_repro_tree(), dim, dec_dim)
    _stub_live_l0_per_repo(t, enc, dim)

    memo, _stats = t._compute_shared_repo_tree("toy", "repo/w000")
    memo["repo/w000"][0].float().pow(2).sum().backward()
    # LIVE L0 weight gets gradient (the tree is differentiable to L0).
    assert enc.W_l0.weight.grad is not None
    # And L1 / bridge too.
    assert enc.W_l1.weight.grad is not None
    assert enc.W_bridge.weight.grad is not None


def test_per_repo_tree_uses_general_prompt():
    dim, dec_dim = 8, 8
    tok = _FakeTokenizer()
    enc = _RecorderEncoder(dim, dec_dim)
    t = _make_per_repo_trainer(enc, tok, _commit_repro_tree(), dim, dec_dim)
    _stub_live_l0_per_repo(t, enc, dim)

    t._compute_shared_repo_tree("toy", "repo/w000")
    general = t._recursive_general_prompt_emb()
    # Every leaf L0 encode in the shared tree received the GENERAL prompt
    # (NOT a per-file-sample question).
    assert enc.l0_query_embs, "tree must encode at least one L0 leaf"
    for q in enc.l0_query_embs:
        assert q is not None
        assert torch.equal(q, general)
    # The general prompt embedding differs from a specific drill query — so the
    # drill-down (which uses the turn query) is genuinely a different prompt.
    specific = enc.embed(
        torch.tensor(tok.encode("fix bug in foo.py"), dtype=torch.long),
    )
    assert general.shape != specific.shape or not torch.equal(
        general, specific[: general.shape[0]],
    )


def test_per_repo_retention_ramps_interpolate_by_step():
    dim, dec_dim = 8, 8
    tok = _FakeTokenizer()
    enc = _RecorderEncoder(dim, dec_dim)
    tree = _commit_repro_tree()

    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    cfg = {"start": 0.15, "end": 0.05, "ramp_steps": 4000}
    # Static helper: endpoints + midpoint + clamp.
    assert KRKBTrainer._interp_ratio_ramp(cfg, 0) == pytest.approx(0.15)
    assert KRKBTrainer._interp_ratio_ramp(cfg, 2000) == pytest.approx(0.10)
    assert KRKBTrainer._interp_ratio_ramp(cfg, 4000) == pytest.approx(0.05)
    assert KRKBTrainer._interp_ratio_ramp(cfg, 8000) == pytest.approx(0.05)
    # Scalar passes through.
    assert KRKBTrainer._interp_ratio_ramp(0.25, 1234) == pytest.approx(0.25)

    # Instance helpers interpolate by global_step.
    l1cfg = {"start": 0.30, "end": 0.05, "ramp_steps": 4000}
    t0 = _make_per_repo_trainer(enc, tok, tree, dim, dec_dim, step=0,
                                l0_cfg=cfg, l1_cfg=l1cfg)
    assert t0._recursive_l0_retention_now() == pytest.approx(0.15)
    assert t0._recursive_l1_retention_now() == pytest.approx(0.30)
    t_mid = _make_per_repo_trainer(enc, tok, tree, dim, dec_dim, step=2000,
                                   l0_cfg=cfg, l1_cfg=l1cfg)
    assert t_mid._recursive_l0_retention_now() == pytest.approx(0.10)
    assert t_mid._recursive_l1_retention_now() == pytest.approx(0.175)
    t_end = _make_per_repo_trainer(enc, tok, tree, dim, dec_dim, step=4000,
                                   l0_cfg=cfg, l1_cfg=l1cfg)
    assert t_end._recursive_l0_retention_now() == pytest.approx(0.05)
    assert t_end._recursive_l1_retention_now() == pytest.approx(0.05)


def test_l0_retention_for_honors_recursive_override():
    """While the shared tree is encoded, _l0_retention_for returns the
    recursive override (so the tree's L0 leaves use the recursive ramp)."""
    dim, dec_dim = 8, 8
    tok = _FakeTokenizer()
    enc = _RecorderEncoder(dim, dec_dim)
    t = _make_per_repo_trainer(enc, tok, _commit_repro_tree(), dim, dec_dim)
    t._l0_retention = {"git_commit_repro": 0.42}
    # No override -> per-dataset rate.
    assert t._l0_retention_for("git_commit_repro") == pytest.approx(0.42)
    # Override active -> recursive ramp value, regardless of dataset rate.
    t._recursive_l0_override = 0.11
    assert t._l0_retention_for("git_commit_repro") == pytest.approx(0.11)
    t._recursive_l0_override = None
    assert t._l0_retention_for("git_commit_repro") == pytest.approx(0.42)


def test_per_repo_gradient_accumulates_across_file_samples():
    """retain_graph across the repo's file-samples lets multiple decoder
    losses backprop through the SHARED tree, accumulating gradient. Mirrors
    _forward_backward_per_repo's backward pattern without the real decoder."""
    dim, dec_dim = 8, 8
    tok = _FakeTokenizer()
    enc = _RecorderEncoder(dim, dec_dim)
    t = _make_per_repo_trainer(enc, tok, _commit_repro_tree(), dim, dec_dim)
    _stub_live_l0_per_repo(t, enc, dim)
    t._accum_steps = 1

    memo, _stats = t._compute_shared_repo_tree("toy", "repo/w000")
    root_rep = memo["repo/w000"][0]  # shared, gradient-carrying

    # Three "file-samples", each a distinct linear readout of the shared rep.
    n_files = 3
    readouts = [torch.randn(dec_dim) for _ in range(n_files)]
    single_grads = []
    # First measure a SINGLE file-sample's grad contribution (fresh graph each).
    for w in readouts:
        enc.W_l0.weight.grad = None
        m_solo, _ = t._compute_shared_repo_tree("toy", "repo/w000")
        (m_solo["repo/w000"][0] @ w).sum().backward()
        single_grads.append(enc.W_l0.weight.grad.clone())

    # Now the per-repo pattern: ONE shared tree, retain_graph across samples.
    enc.W_l0.weight.grad = None
    for w in readouts:
        loss = (root_rep @ w).sum() / (n_files * t._accum_steps)
        loss.backward(retain_graph=True)  # retain so the next sample can reuse
    accumulated = enc.W_l0.weight.grad.clone()

    # The accumulated grad == mean of the per-sample grads (within 1/n scaling).
    expected = sum(single_grads) / n_files
    assert torch.allclose(accumulated, expected, atol=1e-5), (
        "per-repo retain_graph must accumulate each file-sample's gradient "
        "through the shared tree"
    )


def test_per_repo_second_backward_requires_retain_graph():
    """Sanity: WITHOUT retain_graph the second file-sample's backward through
    the shared tree raises — proving retain_graph is load-bearing."""
    dim, dec_dim = 8, 8
    tok = _FakeTokenizer()
    enc = _RecorderEncoder(dim, dec_dim)
    t = _make_per_repo_trainer(enc, tok, _commit_repro_tree(), dim, dec_dim)
    _stub_live_l0_per_repo(t, enc, dim)

    memo, _ = t._compute_shared_repo_tree("toy", "repo/w000")
    root_rep = memo["repo/w000"][0]
    (root_rep.sum()).backward(retain_graph=False)  # frees the graph
    with pytest.raises(RuntimeError):
        (root_rep.sum()).backward()  # second pass: graph already freed


# ---------------------------------------------------------------------------
# 6. survivorship_aux flag: drop ALL aux + _pending, keep θ decoupled
# ---------------------------------------------------------------------------


def _fake_enc_out(controllable: int, organic: int):
    """Minimal enc_out exposing the zero-dim scalar counts ``accumulate``
    reads (controllable_count / organic_count / valid_count)."""
    return SimpleNamespace(
        controllable_count=torch.tensor(controllable, dtype=torch.long),
        organic_count=torch.tensor(organic, dtype=torch.long),
        valid_count=torch.tensor(controllable, dtype=torch.long),
    )


def _aux_trainer():
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    t.device = torch.device("cpu")
    return t


def test_aux_off_compute_survivorship_returns_zero_without_touching_pending():
    t = _aux_trainer()
    t._survivorship_aux = False
    # _pending set to a poison value: if the method iterated it, it would blow
    # up. The early return must avoid that.
    t._pending_l0_outputs = ["POISON"]
    t._pending_l1_outputs = ["POISON"]
    loss, metrics = t._compute_survivorship_aux_losses()
    assert float(loss) == 0.0
    assert not loss.requires_grad
    assert metrics == {}


def test_aux_off_utility_grad_bce_is_skipped():
    t = _aux_trainer()
    t._survivorship_aux = False
    t._pending_l0_outputs = ["POISON"]
    t._pending_l1_outputs = ["POISON"]
    assert t._apply_utility_grad_bce_phase2() == {}


def test_aux_on_does_not_early_return():
    """Flag-ON path is NOT short-circuited — it proceeds to consume _pending
    (other phases depend on this). Poison pending proves it iterates rather
    than early-returning like the flag-off path."""
    t = _aux_trainer()
    t._survivorship_aux = True
    t._pending_l0_outputs = ["POISON"]  # not a dict -> iteration will raise
    t._pending_l1_outputs = ["POISON"]
    with pytest.raises((AttributeError, TypeError, KeyError)):
        t._compute_survivorship_aux_losses()


def test_aux_off_theta_accumulates_decoupled_from_pending():
    """θ control still accumulates the per-microbatch keep-rate at encode time
    (into _surv_state), independent of the _pending collection."""
    from bgkit.training.survivorship_helpers import init_state

    t = _aux_trainer()
    t._survivorship_aux = False
    t._surv_state_l0 = init_state()
    t._surv_state_l1 = init_state()

    # Two "microbatches" of L0 at the recursive ramp ratio.
    t._accumulate_theta_state("l0", _fake_enc_out(controllable=100, organic=15), 0.15)
    t._accumulate_theta_state("l0", _fake_enc_out(controllable=100, organic=10), 0.10)
    cc = int(t._surv_state_l0.controllable_count_sum)
    org = int(t._surv_state_l0.organic_count_sum)
    mass = float(t._surv_state_l0.target_ratio_mass_sum)
    assert cc == 200
    assert org == 25
    # Aggregate target = mass / controllable = (0.15*100 + 0.10*100)/200 = 0.125
    assert mass / cc == pytest.approx(0.125)
    # L1 untouched.
    assert int(t._surv_state_l1.controllable_count_sum) == 0


def test_per_repo_drill_uses_recursive_l1_ramp():
    """The per-repo drill-down (_encode_sample_turns) requests the SAME
    recursive L1 ramp ratio as the shared tree (one curriculum), NOT the static
    top-level l1_retention."""
    dim, dec_dim = 8, 8
    tok = _FakeTokenizer()
    enc = _RecorderEncoder(dim, dec_dim)
    enc.active_projection_output_dim = dec_dim
    t = _make_per_repo_trainer(
        enc, tok, _commit_repro_tree(), dim, dec_dim, step=2000,
        l1_cfg={"start": 0.30, "end": 0.05, "ramp_steps": 4000},
    )
    captured: dict = {}

    def fake_run_l1_batch(self, prepared, target_ratio=None):
        captured["target_ratio"] = target_ratio
        return [torch.zeros(1, dec_dim) for _ in prepared]

    t._run_l1_batch = types.MethodType(fake_run_l1_batch, t)
    t._encode_sample_turns({"prepared_turns": [{"dummy": True}]})
    # 0.30 -> 0.05 over 4000 steps, at step 2000 -> 0.175 (the ramp, not 0.10).
    assert captured["target_ratio"] == pytest.approx(0.175)


def test_aux_off_post_step_runs_dual_ascent_without_pending():
    """_post_optimizer_step aux-off path drives θ via the accumulated state and
    must NOT consume _pending (target_ratios=None -> accumulated mass drives
    the target, matching the ramp)."""
    t = _aux_trainer()
    t._survivorship_aux = False
    t._live_l0 = True
    t._ice_teacher = None
    t._max_warmup_step = 0
    # Poison pending: the aux-off path must never iterate it.
    t._pending_l0_outputs = ["POISON"]
    t._pending_l1_outputs = ["POISON"]

    calls = []
    t._run_dual_ascent = types.MethodType(
        lambda self, step, *, target_ratios=None, skip_levels=(): calls.append(
            (step, target_ratios, skip_levels),
        ),
        t,
    )
    t._post_optimizer_step(42)
    assert len(calls) == 1
    step, target_ratios, skip_levels = calls[0]
    assert step == 42
    assert target_ratios is None  # accumulated per-microbatch mass drives target
    assert skip_levels == ()      # live_l0 -> L0 controller also updates


# ---------------------------------------------------------------------------
# 7. B1: l1l1_bridge MUST train (unfrozen + in an optimizer group + gets grad)
# ---------------------------------------------------------------------------


class _FakeEncoderForLora(torch.nn.Module):
    """Minimal encoder surface for _install_lora / _build_optimizer_groups:
    l1l1_bridge is a TOP-LEVEL sibling of l1 (mirrors BgKITEncoder), so
    l1.requires_grad_ / l1.parameters() do NOT cover it."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.l0 = torch.nn.Linear(dim, dim)
        self.l1 = torch.nn.Linear(dim, dim)
        self.l1l1_bridge = torch.nn.Linear(dim, dim)
        self.projection_blocks = torch.nn.Linear(dim, dim)


def test_b1_l1l1_bridge_trains_and_is_optimized():
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    enc = _FakeEncoderForLora(dim=8)
    t = KRKBTrainer.__new__(KRKBTrainer)
    t.encoder = enc
    t.step_cfg = {
        "lora": {"enabled": False, "train_l1_direct": True},
        "l1_lr": 2.0e-4,
        "lr": 1.0e-4,
    }
    t._round_robin = False
    t._live_l0 = False
    t.topic_embeddings = None
    t.decoder = SimpleNamespace(parameters=lambda: iter([]))
    t._decoders_by_family = {}

    t._install_lora()

    # Unfrozen: every bridge param requires grad.
    bridge_params = list(enc.l1l1_bridge.parameters())
    assert bridge_params and all(p.requires_grad for p in bridge_params)

    # In an optimizer group (its own group at the L1 LR), and NOT double-counted
    # inside the l1 group (bridge is a sibling, not under l1).
    groups = t._build_optimizer_groups()
    bridge_ids = {id(p) for p in bridge_params}
    grouped_ids = {id(p) for g in groups for p in g["params"]}
    assert bridge_ids <= grouped_ids, "l1l1_bridge params must be optimized"
    l1_ids = {id(p) for p in enc.l1.parameters()}
    assert not (bridge_ids & l1_ids), "bridge must not be inside l1's param set"
    bridge_group = next(g for g in groups if bridge_ids & {id(p) for p in g["params"]})
    assert bridge_group["lr"] == pytest.approx(2.0e-4)  # same LR as L1 backbone

    # Backward populates the bridge's gradient.
    x = torch.randn(3, 8)
    enc.l1l1_bridge(x).sum().backward()
    assert enc.l1l1_bridge.weight.grad is not None


# ---------------------------------------------------------------------------
# 8. m1: per-repo gradient normalised by the CONTRIBUTING count, not len(batch)
# ---------------------------------------------------------------------------


def test_per_repo_normalizes_by_contributing_count():
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    w = torch.nn.Parameter(torch.zeros(1))

    def fake_decode(segments):
        mult = sum(
            int(s.loss_mask.sum().item())
            for s in segments
            if isinstance(s, TokenSegment) and s.loss_mask is not None
        )
        return w.sum() * float(mult)

    t = KRKBTrainer.__new__(KRKBTrainer)
    t.device = torch.device("cpu")
    t._round_robin = False
    t._accum_steps = 1
    t.topic_embeddings = None
    t._survivorship_aux = False
    t._live_l0 = True
    t._max_file_samples_per_repo = None
    t.step_cfg = {}
    t.global_step = 0
    t._recursive_l0_retention_cfg = 0.1
    t._recursive_l1_retention_cfg = 0.1
    t.decoder = SimpleNamespace(forward_interleaved_with_loss=fake_decode)

    # 3 samples: two contributors (mult 1 and 2) + one non-contributor (empty).
    def make(seg):
        return SimpleNamespace(dataset_name="toy", _seg=seg)

    s_c1 = make([TokenSegment(token_ids=torch.ones(1, dtype=torch.long),
                              loss_mask=torch.ones(1))])
    s_empty = make([])  # no segments -> dropped, must not dilute the average
    s_c2 = make([TokenSegment(token_ids=torch.ones(2, dtype=torch.long),
                              loss_mask=torch.ones(2))])
    batch = [s_c1, s_empty, s_c2]

    t._repo_group_key = types.MethodType(lambda self, s: "root", t)
    t._compute_shared_repo_tree = types.MethodType(
        lambda self, ds, root: ({}, {"nodes": 0}), t,
    )
    t._prepare_sample_for_decode = types.MethodType(
        lambda self, s: {"prepared_turns": [], "_seg": s._seg}, t,
    )
    t._encode_sample_turns = types.MethodType(lambda self, prep: [], t)
    t._assemble_sample_segments = types.MethodType(
        lambda self, prep, per_turn: (prep["_seg"], None), t,
    )

    metrics = t._forward_backward_per_repo(batch)

    # Two contributors -> grad_w = (mult1 + mult2) / (n_contrib * accum)
    #                            = (1 + 2) / (2 * 1) = 1.5
    # The buggy len(batch)=3 normaliser would give 1.0.
    assert w.grad is not None
    assert float(w.grad) == pytest.approx(1.5)
    assert metrics["per_repo_file_samples"] == pytest.approx(3.0)
    assert metrics["per_repo_contributing_samples"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 9. max_file_samples_per_repo is LIVE-TUNABLE via apply_live_config
# ---------------------------------------------------------------------------


def test_max_file_samples_per_repo_live_tunable():
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    t._max_file_samples_per_repo = None  # config default: unlimited

    # The handler is registered (merged across the MRO with BaseTrainer's).
    handlers: dict = {}
    for cls in reversed(type(t).__mro__):
        handlers.update(getattr(cls, "LIVE_CONFIG_HANDLERS", {}))
    assert handlers.get("max_file_samples_per_repo") == "_handle_max_file_samples_per_repo"
    # ...and BaseTrainer's handlers are preserved (not clobbered).
    assert "max_batch_tokens" in handlers

    # control.json write: clamp DOWN to 4.
    t.apply_live_config({"max_file_samples_per_repo": 4})
    assert t._max_file_samples_per_repo == 4
    # Open back UP to null (unlimited).
    t.apply_live_config({"max_file_samples_per_repo": None})
    assert t._max_file_samples_per_repo is None
    # Raise to a larger cap.
    t.apply_live_config({"max_file_samples_per_repo": 32})
    assert t._max_file_samples_per_repo == 32
    # <=0 collapses to unlimited (None).
    t.apply_live_config({"max_file_samples_per_repo": 0})
    assert t._max_file_samples_per_repo is None


def test_subsample_respects_live_value():
    """_subsample_repo_batch reads the live attribute fresh, so a mid-run
    control.json update changes the cap on the very next repo-batch."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    t.cfg = {"seed": 7}
    t.epoch = 0
    t._max_file_samples_per_repo = None

    batch = [SimpleNamespace(idx=i) for i in range(10)]

    # Unlimited: returns the whole repo-batch unchanged.
    assert t._subsample_repo_batch(batch, "repo/w000") is batch

    # Live clamp to 3 → exactly 3 sampled.
    t.apply_live_config({"max_file_samples_per_repo": 3})
    sub = t._subsample_repo_batch(batch, "repo/w000")
    assert len(sub) == 3
    assert all(s in batch for s in sub)

    # Re-open to null → unlimited again, same call site, no restart.
    t.apply_live_config({"max_file_samples_per_repo": None})
    assert t._subsample_repo_batch(batch, "repo/w000") is batch


# ---------------------------------------------------------------------------
# 10. dynamic-ckpt MODE-FLIP is wired: managed models registered
# ---------------------------------------------------------------------------


def test_dynamic_ckpt_managed_models_registered():
    """KRKBTrainer registers L0 + L1 backbones + EVERY decoder backbone so the
    scheduler's GC mode-flip engages (not only the adaptive cache-flush)."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    l0_bb = torch.nn.Linear(4, 4)
    l1_bb = torch.nn.Linear(4, 4)
    qwen_bb = torch.nn.Linear(4, 4)
    falcon_bb = torch.nn.Linear(4, 4)

    t = KRKBTrainer.__new__(KRKBTrainer)
    t.encoder = SimpleNamespace(
        l0=SimpleNamespace(backbone=l0_bb),
        l1=SimpleNamespace(backbone=l1_bb),
    )
    # Round-robin: BOTH decoders registered (whichever is active is managed).
    t._decoders_by_family = {
        "qwen35": SimpleNamespace(backbone=qwen_bb),
        "falcon_h1": SimpleNamespace(backbone=falcon_bb),
    }
    t.decoder = t._decoders_by_family["qwen35"]

    models = t._dynamic_ckpt_managed_models()
    labels = {label for label, _ in models}
    mods = {id(m) for _, m in models}
    assert labels == {"encoder.l0", "encoder.l1", "decoder_qwen35", "decoder_falcon_h1"}
    assert {id(l0_bb), id(l1_bb), id(qwen_bb), id(falcon_bb)} == mods
    # NOT the BaseTrainer default empty list (which would disable mode-flip).
    assert models != []


def test_dynamic_ckpt_managed_models_single_decoder():
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    l0_bb, l1_bb, dec_bb = (torch.nn.Linear(4, 4) for _ in range(3))
    t = KRKBTrainer.__new__(KRKBTrainer)
    t.encoder = SimpleNamespace(
        l0=SimpleNamespace(backbone=l0_bb),
        l1=SimpleNamespace(backbone=l1_bb),
    )
    t._decoders_by_family = None
    t.decoder = SimpleNamespace(backbone=dec_bb)
    labels = {label for label, _ in t._dynamic_ckpt_managed_models()}
    assert labels == {"encoder.l0", "encoder.l1", "decoder"}
