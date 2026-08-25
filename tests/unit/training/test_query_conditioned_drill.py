"""Unit tests for the QUERY-CONDITIONED drill-node mode (2026-07-31 redesign).

Covers, with NO GPU and stubbed encoders (the ``__new__`` stubbing pattern of
``test_kr_kb_trainer_pieces`` / ``test_recursive_l1_path``):

  1. Turn tagging (``_prepare_sample_for_decode``): with the flag ON, the head,
     the on-path interior node turn AND the wrong-sibling DISTRACTOR node turn
     all carry the per-sample task query; the retrieve leaf receives the
     ``drill_leaf_retention.l0`` ratio.
  2. Flag-OFF parity: node dicts and ``_prepare_l1_turn`` calls are exactly the
     legacy form — no ``query`` key, no ``l0_ratio`` kwarg (legacy-signature
     monkeypatched doubles must keep working).
  3. ``_resolve_special_survivor`` routing: query-tagged node turns (on-path +
     distractor) and the head route through the generalized
     ``_shared_tree_head_survivor`` at ``drill_node_retention`` with
     ``force_checkpoint=True``; flag off → static splice lookup / legacy head
     call with no extra kwargs.
  4. Ratio threading: ``_encode_tree_node_live(ratio=…)`` reaches
     ``run_l1_and_project``'s ``target_ratio_l1``; default falls back to the
     recursive ramp.
  5. ``force_checkpoint=True`` wraps the node forward in torch.utils.checkpoint
     even when the per-tree size gates are OFF (recompute-on-backward observed,
     gradients exact) and is skipped in eval mode.
  6. Gradient reaccumulation: an interior node drill consumes DETACHED child
     l1out leaves — the drill backward lands in
     ``_shared_tree_child_l1_reps[c].grad`` (NOT the shared tree), and the
     deferred tree backward carries those grads to the shared tree's producers.
  7. Retrieve-leaf L1 resolution order: ``drill_leaf_retention.l1`` wins over
     the recursive ramp in the drivers' helper and over the sampled fallback in
     ``_run_l1_batch``'s None-default (the eval/training alignment fix); unset
     → exactly legacy.
"""

from __future__ import annotations

import contextlib
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

TASK_QUERY = "reconstruct file F at commit X"


# ---------------------------------------------------------------------------
# Stubs (mirroring test_recursive_l1_path)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Append-idempotent fake chat template + char-level encoder."""

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size

    def _format_message(self, m: dict) -> str:
        role = m.get("role", "user")
        if role == "assistant" and m.get("tool_calls"):
            call = m["tool_calls"][0]["function"]
            import json

            args_str = json.dumps(call.get("arguments", {}), sort_keys=True)
            # Real templates wrap tool calls in the same assistant scaffold as
            # content turns (the body differs, the header/footer do not).
            return f"<assistant>call name={call['name']} args={args_str}</assistant>"
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


def _drill_tree() -> _StubTree:
    # window head -> c4 interior (on-path) + wrongSib (distractor pool);
    # c4 -> two commit leaf-tags.
    return _StubTree({
        "repo/w000": _StubNode("repo/w000", ["repo/w000/c4_0", "wrongSib"]),
        "repo/w000/c4_0": _StubNode("repo/w000/c4_0", ["commitA", "commitB"]),
        "wrongSib": _StubNode("wrongSib", ["commitC"]),
        "commitA": _StubNode("commitA", []),
        "commitB": _StubNode("commitB", []),
        "commitC": _StubNode("commitC", []),
    })


def _turn(kind: str, args=None, response: str = "", loss: bool = True):
    from bgkit.data.bgkit_tool_template import TrajectoryTurn

    return TrajectoryTurn(kind=kind, args=args or {}, response=response, loss=loss)


def _drill_sample():
    """head (is_head, task query) → on-path node → DISTRACTOR node (loss=False)
    → retrieve leaf (id not in tree) → answer."""
    return SimpleNamespace(
        dataset_name="toy",
        question="q?",
        trajectory=[
            _turn("bgkit", {"ids": ["repo/w000"], "query": TASK_QUERY,
                            "is_head": True}),
            _turn("bgkit", {"ids": ["repo/w000/c4_0"], "query": ""}),
            _turn("bgkit", {"ids": ["wrongSib"], "query": ""}, loss=False),
            _turn("bgkit", {"ids": ["diff_1"], "query": ""}),
            _turn("answer", response="file contents"),
        ],
    )


def _prepare_trainer(*, flag_on: bool, leaf_l0_cfg):
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    t.device = torch.device("cpu")
    t.topic_embeddings = None
    t._ablation_mode = None
    t.decoder = SimpleNamespace(hidden_dim=4)
    t.tokenizer = _FakeTokenizer()
    t._system_prompt_for = types.MethodType(lambda self, s: "SYS", t)
    t._trees = {"toy": _drill_tree()}
    t._recursive_l1_full_backprop = True
    # The double models the per-repo shared-tree path: head drills are only
    # legal with head-drill infrastructure (the trainer raises otherwise).
    t._per_repo_full_backprop = True
    t._query_conditioned_drill_nodes = flag_on
    t._drill_leaf_l0_retention_cfg = leaf_l0_cfg
    t.global_step = 0
    return t


# ---------------------------------------------------------------------------
# 1. + 2. Turn tagging (flag on) / flag-off legacy parity
# ---------------------------------------------------------------------------


def test_flag_on_tags_all_node_turns_with_task_query_and_leaf_l0_ratio():
    t = _prepare_trainer(flag_on=True, leaf_l0_cfg=0.63)

    calls: list[dict] = []

    def fake_prepare(self, dataset, ids, query, l0_selection_mode="threshold",
                     l0_ratio=None):
        calls.append({
            "dataset": dataset, "ids": tuple(ids), "query": query,
            "mode": l0_selection_mode, "l0_ratio": l0_ratio,
        })
        return {"content": torch.zeros(2, 4)}

    t._prepare_l1_turn = types.MethodType(fake_prepare, t)

    prep = t._prepare_sample_for_decode(_drill_sample())
    turns = prep["prepared_turns"]
    assert len(turns) == 4

    head, on_path, distractor, leaf = turns
    assert head["mode"] == "head" and head["query"] == TASK_QUERY
    # On-path interior node: tagged with the TASK query.
    assert on_path["mode"] == "node"
    assert on_path["node_id"] == "repo/w000/c4_0"
    assert on_path["query"] == TASK_QUERY
    # DISTRACTOR node drill (loss=False in the trajectory): ALSO tagged —
    # content-driven rejection requires the wrong sibling encoded under the
    # same query.
    assert distractor["mode"] == "node"
    assert distractor["node_id"] == "wrongSib"
    assert distractor["query"] == TASK_QUERY
    # Retrieve leaf: task-query conditioned, exact_topk L0, drill L0 ratio.
    assert isinstance(leaf, dict) and "content" in leaf and "mode" not in leaf
    assert calls == [{
        "dataset": "toy", "ids": ("diff_1",), "query": TASK_QUERY,
        "mode": "exact_topk", "l0_ratio": pytest.approx(0.63),
    }]


def test_flag_off_is_exact_legacy_form():
    """Flag OFF: node dicts carry NO query key and _prepare_l1_turn is called
    WITHOUT l0_ratio — a legacy-signature monkeypatched double must work."""
    t = _prepare_trainer(flag_on=False, leaf_l0_cfg=None)

    calls: list[dict] = []

    # LEGACY signature — no l0_ratio parameter. A TypeError here would mean
    # the flag-off path changed the call surface.
    def fake_prepare_legacy(self, dataset, ids, query, l0_selection_mode="threshold"):
        calls.append({
            "dataset": dataset, "ids": tuple(ids), "query": query,
            "mode": l0_selection_mode,
        })
        return {"content": torch.zeros(2, 4)}

    t._prepare_l1_turn = types.MethodType(fake_prepare_legacy, t)

    prep = t._prepare_sample_for_decode(_drill_sample())
    head, on_path, distractor, _leaf = prep["prepared_turns"]

    assert head == {
        "mode": "head", "node_id": "repo/w000", "query": TASK_QUERY,
        "dataset": "toy",
    }
    # EXACT legacy node dicts — no "query" key.
    assert on_path == {
        "mode": "node", "node_id": "repo/w000/c4_0", "dataset": "toy",
    }
    assert distractor == {"mode": "node", "node_id": "wrongSib", "dataset": "toy"}
    # Leaf keeps the 2026-07-30 behavior (task-query + exact_topk), no ratio.
    assert calls == [{
        "dataset": "toy", "ids": ("diff_1",), "query": TASK_QUERY,
        "mode": "exact_topk",
    }]


def test_flag_on_without_task_query_keeps_legacy_node_dicts():
    """Non-git-repro trajectories have no is_head turn → no task query → node
    turns stay static even with the flag on (guard against empty-query
    conditioning)."""
    t = _prepare_trainer(flag_on=True, leaf_l0_cfg=None)
    t._prepare_l1_turn = types.MethodType(
        lambda self, dataset, ids, query, l0_selection_mode="threshold",
        l0_ratio=None: {"content": torch.zeros(2, 4)},
        t,
    )
    sample = SimpleNamespace(
        dataset_name="toy",
        question="q?",
        trajectory=[
            _turn("bgkit", {"ids": ["repo/w000/c4_0"], "query": ""}),
            _turn("answer", response="a"),
        ],
    )
    prep = t._prepare_sample_for_decode(sample)
    (node,) = prep["prepared_turns"]
    assert node == {
        "mode": "node", "node_id": "repo/w000/c4_0", "dataset": "toy",
    }


def _flat_sample(args: dict):
    return SimpleNamespace(
        dataset_name="toy",
        question="q?",
        trajectory=[_turn("bgkit", args), _turn("answer", response="a")],
    )


def test_head_turn_without_head_infra_raises():
    """CONTRACT (2026-08-22): a flat trainer (no per-repo shared tree, no L1
    tree cache) must REJECT ``is_head`` turns at sample-prep time — otherwise
    every splice silently resolves to a zero survivor and the encoder never
    runs (the widenet v1→v4 failure)."""
    t = _prepare_trainer(flag_on=False, leaf_l0_cfg=None)
    t._per_repo_full_backprop = False
    t._l1_tree_cache = None
    t._prepare_l1_turn = types.MethodType(
        lambda self, dataset, ids, query, l0_selection_mode="threshold",
        l0_ratio=None: {"content": torch.zeros(2, 4)},
        t,
    )
    with pytest.raises(ValueError, match=r"is_head=True .* no head-drill infrastructure"):
        t._prepare_sample_for_decode(
            _flat_sample({"ids": ["repo/w000"], "query": "find the needle", "is_head": True})
        )
    # The plain leaf form (what flat_phase2_writer emits) routes to the live
    # _prepare_l1_turn path — one real leaf dict, no mode tag.
    prep = t._prepare_sample_for_decode(
        _flat_sample({"ids": ["repo/w000"], "query": "find the needle"})
    )
    (leaf,) = prep["prepared_turns"]
    assert "content" in leaf and "mode" not in leaf


def test_ablation_mode_flags_every_decoder_for_the_splice_guard():
    """``_ablation_mode`` is a property: degenerate-rep modes (zeroed / noise /
    topics_only / neither) set ``_rep_norm_guard_expect_degenerate`` on BOTH
    round-robin decoders; None / no_topics clear it."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    t._ablation_mode = None  # no decoders yet: must not fail
    qwen, falcon = SimpleNamespace(), SimpleNamespace()
    t.decoder = qwen
    t._decoders_by_family = {"qwen35": qwen, "falcon_h1": falcon}
    for mode in ("zeroed", "noise", "topics_only", "neither"):
        t._ablation_mode = mode
        assert t._ablation_mode == mode
        assert qwen._rep_norm_guard_expect_degenerate is True
        assert falcon._rep_norm_guard_expect_degenerate is True
    t.set_ablation_mode("no_topics")
    assert qwen._rep_norm_guard_expect_degenerate is False
    t.set_ablation_mode(None)
    assert t._ablation_mode is None
    assert falcon._rep_norm_guard_expect_degenerate is False


def test_flat_leaf_follows_configured_l0_selection_mode():
    """``training.selection_mode.l0: exact_topk`` routes the FLAT retrieval
    leaf's live L0 to exact_topk without the recursive full-backprop flag
    (2026-08-22: threshold mode kept 0.0-0.3% on the base, then ~100%)."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = _prepare_trainer(flag_on=False, leaf_l0_cfg=None)
    t._per_repo_full_backprop = False
    t._recursive_l1_full_backprop = False
    t._selection_mode_l0 = "exact_topk"
    calls: list[str] = []
    t._prepare_l1_turn = types.MethodType(
        lambda self, dataset, ids, query, l0_selection_mode="threshold",
        l0_ratio=None: calls.append(l0_selection_mode) or {"content": torch.zeros(2, 4)},
        t,
    )
    t._prepare_sample_for_decode(_flat_sample({"ids": ["x"], "query": "q"}))
    assert calls == ["exact_topk"]
    # θ dual ascent is skipped for an exact_topk level (θ is irrelevant).
    t._live_l0 = True
    t._selection_mode_l1 = "exact_topk"
    assert KRKBTrainer._theta_skip_levels(t) == ("l0", "l1")
    t._selection_mode_l0 = "threshold"
    t._selection_mode_l1 = "threshold"
    assert KRKBTrainer._theta_skip_levels(t) == ()
    calls.clear()
    t._prepare_sample_for_decode(_flat_sample({"ids": ["x"], "query": "q"}))
    assert calls == ["threshold"]


def test_anchor_sampling_disabled_under_exact_topk():
    """θ-anchor probing of the full grid is pointless under exact_topk (θ is
    never consulted) and splices up to ~37K reps; the trainer zeroes
    ``anchor_sampling_prob`` for an exact_topk level and keeps it otherwise."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
    from bgkit.training.ratio_sampling import RatioSamplerConfig

    cfg = RatioSamplerConfig(
        enabled=True, mode="jitter", anchor_grid=(0.02, 0.16, 0.95),
        anchor_sampling_prob=0.3, window_above=0.05, jitter_abs=0.02,
        jitter_rel=0.2, lower_bound=0.01, upper_bound=0.95,
    )
    t = KRKBTrainer.__new__(KRKBTrainer)
    t._selection_mode_l0 = "exact_topk"
    t._selection_mode_l1 = "threshold"
    out0 = t._anchor_free_if_topk("l0", cfg)
    assert out0.anchor_sampling_prob == 0.0 and out0.mode == "jitter" and out0.enabled
    assert t._anchor_free_if_topk("l1", cfg) is cfg


def test_head_turn_with_l1_tree_cache_is_legal():
    t = _prepare_trainer(flag_on=False, leaf_l0_cfg=None)
    t._per_repo_full_backprop = False
    t._l1_tree_cache = object()  # offline L1-tree cache = head-drill infra
    prep = t._prepare_sample_for_decode(
        _flat_sample({"ids": ["repo/w000"], "query": TASK_QUERY, "is_head": True})
    )
    (head,) = prep["prepared_turns"]
    assert head["mode"] == "head" and head["node_id"] == "repo/w000"


# ---------------------------------------------------------------------------
# 3. _resolve_special_survivor routing
# ---------------------------------------------------------------------------


def _routing_trainer(*, flag_on: bool):
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    t._query_conditioned_drill_nodes = flag_on
    t._drill_node_retention_cfg = 0.4
    t.global_step = 0
    head_calls: list[tuple] = []
    node_calls: list[tuple] = []

    def fake_head(self, node_id, query, dataset, ratio=None, force_checkpoint=False):
        head_calls.append((node_id, query, dataset, ratio, force_checkpoint))
        return torch.zeros(1, 4)

    def fake_node(self, node_id, dataset=""):
        node_calls.append((node_id, dataset))
        return torch.zeros(1, 4)

    t._shared_tree_head_survivor = types.MethodType(fake_head, t)
    t._shared_tree_node_survivor = types.MethodType(fake_node, t)
    return t, head_calls, node_calls


def test_resolve_routes_query_tagged_nodes_through_head_machinery():
    t, head_calls, node_calls = _routing_trainer(flag_on=True)

    t._resolve_special_survivor(
        {"mode": "node", "node_id": "c4", "dataset": "toy", "query": TASK_QUERY},
    )
    t._resolve_special_survivor(
        {"mode": "node", "node_id": "wrongSib", "dataset": "toy",
         "query": TASK_QUERY},
    )
    t._resolve_special_survivor(
        {"mode": "head", "node_id": "repo/w000", "dataset": "toy",
         "query": TASK_QUERY},
    )
    # All three (on-path, distractor, head) → generalized head machinery at
    # drill_node_retention with the forced per-node checkpoint.
    assert head_calls == [
        ("c4", TASK_QUERY, "toy", pytest.approx(0.4), True),
        ("wrongSib", TASK_QUERY, "toy", pytest.approx(0.4), True),
        ("repo/w000", TASK_QUERY, "toy", pytest.approx(0.4), True),
    ]
    assert node_calls == []

    # An untagged node entry (no query) stays static even with the flag on.
    t._resolve_special_survivor({"mode": "node", "node_id": "n", "dataset": "toy"})
    assert node_calls == [("n", "toy")]


def test_resolve_flag_off_is_legacy():
    t, head_calls, node_calls = _routing_trainer(flag_on=False)

    # Even a query-tagged node entry resolves statically when the flag is off.
    t._resolve_special_survivor(
        {"mode": "node", "node_id": "c4", "dataset": "toy", "query": TASK_QUERY},
    )
    assert node_calls == [("c4", "toy")]
    # Head: legacy call — default ratio (None) and NO forced checkpoint.
    t._resolve_special_survivor(
        {"mode": "head", "node_id": "repo/w000", "dataset": "toy",
         "query": TASK_QUERY},
    )
    assert head_calls == [("repo/w000", TASK_QUERY, "toy", None, False)]


def test_drill_node_retention_defaults_to_recursive_ramp():
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    t.global_step = 0
    t._recursive_l1_retention_cfg = 0.3
    t._drill_node_retention_cfg = None
    assert t._drill_node_retention_now() == pytest.approx(0.3)
    t._drill_node_retention_cfg = 0.4
    assert t._drill_node_retention_now() == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# 4. + 5. ratio threading + force_checkpoint in _encode_tree_node_live
# ---------------------------------------------------------------------------


def _node_live_trainer(lin: torch.nn.Linear, ratio_log: list, call_count: list,
                       id_emb: torch.nn.Embedding | None = None):
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
    from bgkit.training.survivorship_helpers import init_state

    d = lin.in_features
    # Share ``id_emb`` across builds when comparing runs — the shared encode
    # primitive injects child-ID embeddings, so grad parity needs identical
    # ID embeddings on both sides.
    id_emb = id_emb if id_emb is not None else torch.nn.Embedding(4, d)

    def fake_run_l1(*, l1_input_embeddings, l1_input_cu_seqlens, target_ratio_l1,
                    **kw):
        call_count.append(1)
        ratio_log.append(float(target_ratio_l1))
        surv = lin(l1_input_embeddings)
        proj = lin(l1_input_embeddings)
        l1_out = SimpleNamespace(
            survivor_embeddings=surv,
            survivor_cu_seqlens=torch.tensor([0, surv.shape[0]], dtype=torch.int32),
            organic_count=torch.tensor(3),
            controllable_count=torch.tensor(10),
            valid_count=torch.tensor(10),
        )
        proj_out = SimpleNamespace(projected_embeddings=proj)
        return l1_out, proj_out, None

    t = KRKBTrainer.__new__(KRKBTrainer)
    t.device = torch.device("cpu")
    t.encoder = SimpleNamespace(
        training=True,
        run_l1_and_project=fake_run_l1,
        l0=SimpleNamespace(backbone=SimpleNamespace(
            get_input_embeddings=lambda: id_emb,
        )),
    )
    t.encoder_tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens=False: [1],
    )
    t._l1_adapter_context = types.MethodType(
        lambda self: contextlib.nullcontext(), t,
    )
    t._recursive_l1_retention_cfg = 0.2
    t.global_step = 0
    t._survivorship_aux = False
    t._surv_state_l0 = init_state()
    t._surv_state_l1 = init_state()
    t._checkpoint_tree_encode = False  # size gates OFF — only force applies
    return t


def test_ratio_threads_into_node_forward_and_defaults_to_ramp():
    d = 8
    lin = torch.nn.Linear(d, d)
    ratios: list[float] = []
    t = _node_live_trainer(lin, ratios, [])
    x = torch.randn(5, d, requires_grad=True)
    q = torch.randn(2, d)

    t._encode_tree_node_live(["c0"], [x], q, ratio=0.4)
    t._encode_tree_node_live(["c0"], [x], q)  # legacy default
    assert ratios == [pytest.approx(0.4), pytest.approx(0.2)]


def test_force_checkpoint_recomputes_on_backward_with_exact_grads():
    """force_checkpoint=True must checkpoint the node forward even though the
    per-tree size gates are OFF: the forward runs once, the backward reruns it
    (recompute — proof the activations were freed), and gradients are exact."""
    d = 8
    lin = torch.nn.Linear(d, d)
    shared_id_emb = torch.nn.Embedding(4, d)
    children = torch.randn(5, d)
    q = torch.randn(2, d)

    def run(force: bool):
        lin.weight.grad = None
        calls: list[int] = []
        t = _node_live_trainer(lin, [], calls, id_emb=shared_id_emb)
        x = children.clone().requires_grad_(True)
        proj, _l1 = t._encode_tree_node_live(
            ["c0"], [x], q, ratio=0.4, force_checkpoint=force,
        )
        n_fwd = len(calls)
        proj.sum().backward()
        return n_fwd, len(calls), lin.weight.grad.detach().clone()

    fwd_ref, total_ref, grad_ref = run(force=False)
    fwd_ck, total_ck, grad_ck = run(force=True)

    assert (fwd_ref, total_ref) == (1, 1)  # no checkpoint → no recompute
    assert (fwd_ck, total_ck) == (1, 2)    # checkpointed → recompute on bwd
    assert torch.allclose(grad_ck, grad_ref, atol=1e-6, rtol=1e-5)

    # θ accumulated exactly once despite the recompute.
    t = _node_live_trainer(lin, [], [], id_emb=shared_id_emb)
    x = children.clone().requires_grad_(True)
    proj, _ = t._encode_tree_node_live(["c0"], [x], q, ratio=0.4,
                                       force_checkpoint=True)
    proj.sum().backward()
    assert int(t._surv_state_l1.controllable_count_sum) == 10


def test_force_checkpoint_skipped_in_eval_mode():
    d = 8
    lin = torch.nn.Linear(d, d)
    calls: list[int] = []
    t = _node_live_trainer(lin, [], calls)
    t.encoder.training = False
    with torch.no_grad():
        x = torch.randn(5, d)
        t._encode_tree_node_live(["c0"], [x], torch.randn(2, d),
                                 ratio=0.4, force_checkpoint=True)
    assert len(calls) == 1  # plain forward, no checkpoint machinery


# ---------------------------------------------------------------------------
# 6. Detach-reaccumulate contract for interior node drills
# ---------------------------------------------------------------------------


def test_interior_node_drill_reaccumulates_into_deferred_tree_backward():
    """A query-conditioned interior-node drill must (a) consume DETACHED child
    l1out leaves (the drill backward reaches _shared_tree_child_l1_reps, NOT
    the shared tree), and (b) the deferred tree backward must carry those
    accumulated grads to the shared tree's producers."""
    from bgkit.training.survivorship_helpers import init_state

    d = 8
    lin = torch.nn.Linear(d, d)
    w_child = torch.nn.Linear(d, d)  # the shared tree's producer stand-in
    t = _node_live_trainer(lin, [], [])
    t._surv_state_l1 = init_state()
    t.encoder.l1_auto_reproduce = lambda x: x  # bridge identity

    tree = _drill_tree()
    t._trees = {"toy": tree}
    t._l1_tree_cache = None
    # Shared-tree memo: child l1outs produced by w_child (live graph).
    memo = {
        "commitA": (None, w_child(torch.ones(3, d))),
        "commitB": (None, w_child(torch.full((2, d), 2.0))),
    }
    t._shared_tree_memo = memo
    t._shared_tree_child_l1_reps = {}
    t._shared_tree_child_l1_used = set()
    t._query_conditioned_drill_nodes = True
    t._drill_node_retention_cfg = 0.4

    proj = t._resolve_special_survivor(
        {"mode": "node", "node_id": "repo/w000/c4_0", "dataset": "toy",
         "query": TASK_QUERY},
    )
    proj.sum().backward()

    # (a) Both children consumed via detached requires_grad leaves; the drill
    # backward filled the LEAF grads but did NOT touch the shared tree.
    assert t._shared_tree_child_l1_used == {"commitA", "commitB"}
    reps = t._shared_tree_child_l1_reps
    for c in ("commitA", "commitB"):
        assert reps[c].grad is not None and reps[c].grad.abs().sum() > 0
    assert w_child.weight.grad is None, (
        "drill backward must stop at the detached child leaves"
    )

    # (b) Deferred tree backward (the per-repo final backward) feeds the
    # accumulated child grads into the live memo l1outs → shared tree producer.
    torch.autograd.backward(
        tensors=[memo["commitA"][1], memo["commitB"][1]],
        grad_tensors=[reps["commitA"].grad, reps["commitB"].grad],
    )
    assert w_child.weight.grad is not None
    assert w_child.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 7. Retrieve-leaf L1 resolution order (drivers + eval None-default)
# ---------------------------------------------------------------------------


def test_drill_leaf_l1_helper_prefers_override_then_ramp():
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    t.global_step = 0
    t._recursive_l1_retention_cfg = 0.3
    t._drill_leaf_l1_retention_cfg = 0.63
    assert t._drill_leaf_l1_retention_override() == pytest.approx(0.63)
    assert t._drill_leaf_l1_retention_now() == pytest.approx(0.63)
    t._drill_leaf_l1_retention_cfg = None
    assert t._drill_leaf_l1_retention_override() is None
    assert t._drill_leaf_l1_retention_now() == pytest.approx(0.3)


def _run_l1_batch_trainer(ratio_log: list):
    """Minimal stub surface for the REAL _run_l1_batch (eval-mode path)."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    d = 4

    def fake_run_l1(*, l1_input_embeddings, l1_input_cu_seqlens, target_ratio_l1,
                    **kw):
        ratio_log.append(float(target_ratio_l1))
        keep = l1_input_embeddings[:1]
        l1_out = SimpleNamespace(
            survivor_embeddings=keep,
            survivor_cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
        )
        proj_out = SimpleNamespace(
            projected_embeddings=keep,
            survivor_cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
        )
        return l1_out, proj_out, torch.tensor([0, 1], dtype=torch.int32)

    t = KRKBTrainer.__new__(KRKBTrainer)
    t.device = torch.device("cpu")
    t.global_step = 0
    t.encoder = SimpleNamespace(
        training=False,  # the single-sample EVAL path
        active_projection_output_dim=d,
        run_l1_and_project=fake_run_l1,
        l0=SimpleNamespace(auto_reproduce=lambda x: x),
    )
    t._l1_adapter_context = types.MethodType(
        lambda self: contextlib.nullcontext(), t,
    )
    t._selection_mode_l1 = "exact_topk"
    # Marker: if the sampled fallback is used, 0.99 shows up in the log.
    t._sample_l1_retention = types.MethodType(lambda self: 0.99, t)
    return t


def _leaf_turn(d: int = 4) -> dict:
    n_rows = 3
    return {
        "content": torch.randn(n_rows, d),
        "pinned": torch.zeros(n_rows, dtype=torch.bool),
        "relevance_mask": torch.ones(n_rows, dtype=torch.bool),
        "survivor_mask": torch.zeros(n_rows, dtype=torch.bool),
        "query_emb": torch.randn(2, d),
    }


def test_run_l1_batch_none_default_uses_leaf_override_for_eval_alignment():
    ratios: list[float] = []
    t = _run_l1_batch_trainer(ratios)
    t._drill_leaf_l1_retention_cfg = 0.63
    t._run_l1_batch([_leaf_turn()])  # target_ratio=None — the eval path
    assert ratios == [pytest.approx(0.63)]


def test_run_l1_batch_none_default_falls_back_to_sampled_when_unset():
    ratios: list[float] = []
    t = _run_l1_batch_trainer(ratios)
    t._drill_leaf_l1_retention_cfg = None
    t._run_l1_batch([_leaf_turn()])
    assert ratios == [pytest.approx(0.99)]  # legacy sampled fallback


def test_run_l1_batch_explicit_target_ratio_wins():
    ratios: list[float] = []
    t = _run_l1_batch_trainer(ratios)
    t._drill_leaf_l1_retention_cfg = 0.63
    t._run_l1_batch([_leaf_turn()], target_ratio=0.31)  # driver-pinned
    assert ratios == [pytest.approx(0.31)]


def test_pre_train_loop_recalibrates_t_for_exact_topk_on_resume():
    """Under exact_topk, T is the loss-score normalizer and must track the
    LOADED head: _pre_train_loop re-probes the exact_topk levels on resume
    (global_step > 0) and leaves threshold levels / cold starts alone."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    calls: list = []
    t._calibrate_head_tanh_temperatures = lambda levels=("l0", "l1"), **kw: calls.append(levels)
    t._selection_mode_l0 = "exact_topk"
    t._selection_mode_l1 = "threshold"
    t.global_step = 0
    KRKBTrainer._pre_train_loop(t)
    assert calls == []  # cold start: the setup-time probe already ran
    t.global_step = 422
    KRKBTrainer._pre_train_loop(t)
    assert calls == [("l0",)]
    t._selection_mode_l1 = "exact_topk"
    KRKBTrainer._pre_train_loop(t)
    assert calls[-1] == ("l0", "l1")
