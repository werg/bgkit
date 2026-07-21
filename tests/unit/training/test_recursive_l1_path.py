"""Unit tests for the recursive-L1 (Phase 3) per-repo shared-tree drill path.

Covers, with NO GPU and a fully stubbed encoder, the per-repo full-backprop
shared-tree encode + the detach-and-reaccumulate / option-A / inner-loop
gradient contracts of ``_forward_backward_per_repo``.

These mirror the stubbing pattern in
``tests/unit/training/test_kr_kb_trainer_pieces.py`` (build the trainer via
``__new__`` and fill only the attributes the path under test reads).
"""

from __future__ import annotations

import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from bgkit.models.decoder import TokenSegment

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

    def live_l0(self, dataset, article_ids, query_emb=None, selection_mode="threshold"):
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
    # their drill-down ``node`` turns — NO re-encode (bridge-call count must
    # not grow). The lookup goes through the live ``_shared_tree_node_survivor``
    # (the surviving per-repo splice/memo lookup).
    t._shared_tree_memo = memo
    t._per_repo_shared_tree_active = True
    for _ in range(3):
        rep0 = t._shared_tree_node_survivor("repo/w000")
        rep1 = t._shared_tree_node_survivor("repo/w000/c4_0")
        # Node reps are the SHARED memo tensors (same object identity).
        assert rep0 is memo["repo/w000"][0]
        assert rep1 is memo["repo/w000/c4_0"][0]
    assert len(enc.bridge_input_grad) == bridge_calls_after_tree, (
        "shared-tree lookups must NOT re-encode the shared tree per file-sample"
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
    """The per-repo drill-down (now via _encode_decode_group) requests the SAME
    recursive L1 ramp ratio as the shared tree (one curriculum), NOT the static
    top-level l1_retention. The ramp is threaded as l1_target_ratio to
    _run_l1_batch."""
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
    t._prepare_sample_for_decode = types.MethodType(
        lambda self, s: {"prepared_turns": [{"content": torch.zeros(4, dim)}]}, t,
    )
    t._assemble_sample_segments = types.MethodType(
        lambda self, prep, per_turn: ([], None), t,
    )
    # Drive the shared encode→decode core with the recursive ramp (as PASS 2
    # does): l1_target_ratio must reach _run_l1_batch.
    t._encode_decode_group(
        [SimpleNamespace(dataset_name="toy")],
        l1_target_ratio=t._recursive_l1_retention_now(),
    )
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
    # PASS 1: cheap contributing count (render-only stub).
    t._sample_contributing_token_count = types.MethodType(
        lambda self, s: sum(int(seg.loss_mask.sum()) for seg in s._seg), t,
    )
    # PASS 2: prep + assemble + decode (group-batched via _encode_decode_group;
    # empty prepared_turns -> no L1 bucketing, so this stays a pure assemble+
    # decode path). Default group size folds all 3 samples into one group.
    t._prepare_sample_for_decode = types.MethodType(
        lambda self, s: {"prepared_turns": [], "_seg": s._seg}, t,
    )
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
    assert metrics["per_repo_backwarded_samples"] == pytest.approx(2.0)


def test_per_repo_pass2_group_batched_bounded_by_g():
    """PASS 2 processes the contributing samples in groups of G: one
    _encode_decode_group call (→ one group backward) per group of G, NOT one
    per sample. The number of groups (and thus backward launches) is
    ceil(n/G) — bounded by G, never all-N staging. Pass 1 still counts WITHOUT
    encoding."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    def run(group_size, n=5):
        t = KRKBTrainer.__new__(KRKBTrainer)
        t.device = torch.device("cpu")
        t._round_robin = False
        t._accum_steps = 1
        t.topic_embeddings = None
        t._survivorship_aux = False
        t._live_l0 = True
        t._max_file_samples_per_repo = None
        t._per_repo_sample_group_size = group_size
        t.step_cfg = {}
        t.global_step = 0
        t._recursive_l0_retention_cfg = 0.1
        t._recursive_l1_retention_cfg = 0.1
        w = torch.nn.Parameter(torch.zeros(1))
        t.decoder = SimpleNamespace(
            forward_interleaved_with_loss=lambda segments: w.sum(),
        )
        batch = [SimpleNamespace(dataset_name="toy", idx=i) for i in range(n)]

        count_calls: list[int] = []
        t._sample_contributing_token_count = types.MethodType(
            lambda self, s: count_calls.append(s.idx) or 1, t,
        )
        t._repo_group_key = types.MethodType(lambda self, s: "root", t)
        t._compute_shared_repo_tree = types.MethodType(
            lambda self, ds, root: ({}, {"nodes": 0}), t,
        )
        t._prepare_sample_for_decode = types.MethodType(
            lambda self, s: {"prepared_turns": []}, t,
        )
        t._assemble_sample_segments = types.MethodType(
            lambda self, prep, per_turn: (
                [TokenSegment(token_ids=torch.ones(1, dtype=torch.long),
                              loss_mask=torch.ones(1))],
                None,
            ),
            t,
        )
        # Count group invocations (= group backwards).
        n_group_calls = {"n": 0}
        orig = t._encode_decode_group

        def counting_group(samples, *a, **k):
            n_group_calls["n"] += 1
            # group size must never exceed G.
            assert len(samples) <= group_size
            return orig(samples, *a, **k)

        t._encode_decode_group = counting_group
        t._forward_backward_per_repo(batch)
        return count_calls, n_group_calls["n"]

    import math as _m
    # Pass 1 counts all without encoding; PASS 2 makes ceil(n/G) group calls.
    for g in (1, 2, 5, 8):
        count_calls, n_groups = run(g, n=5)
        assert count_calls == [0, 1, 2, 3, 4]
        assert n_groups == _m.ceil(5 / g), f"G={g}: expected {_m.ceil(5/g)} groups"


def test_sample_contributing_token_count_is_render_only():
    """The Pass-1 count derives from the rendered loss_mask alone — no encode,
    no drill encode (_prepare_l1_turn), no θ side-effect."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    rendered = SimpleNamespace(loss_mask=torch.tensor([0, 1, 1, 0, 1]))
    t._render_sample = types.MethodType(
        lambda self, s: (rendered, None, []), t,
    )
    # Tripwire: the drill encode must NOT be called by the count.
    def _boom(self, *a, **k):
        raise AssertionError("count path must not encode")
    t._prepare_l1_turn = types.MethodType(_boom, t)

    assert t._sample_contributing_token_count(SimpleNamespace()) == 3


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


# ---------------------------------------------------------------------------
# 11. Per-repo SIZE FILTER: over-threshold repos dropped at grouping
# ---------------------------------------------------------------------------


def test_repo_leaf_token_count_metric():
    """_repo_leaf_token_count sums L0-encoded leaf-diff tokens over the
    window-0 subtree (the shared-tree-encode memory driver)."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    tree = _StubTree({
        "repo/w000": _StubNode("repo/w000", ["c4_0"]),
        "c4_0": _StubNode("c4_0", ["commitA", "commitB"]),
        "commitA": _StubNode("commitA", []),
        "commitB": _StubNode("commitB", []),
    })
    # BrowseTree.articles recurses leaves; stub it directly.
    tree.articles = lambda nid: ["a1", "a2", "a3"] if nid == "repo/w000" else []

    class _Store:
        def length(self, dataset, doc_id):
            return {"a1": 100, "a2": 250, "a3": 50}[doc_id]

    t = KRKBTrainer.__new__(KRKBTrainer)
    t._trees = {"toy": tree}
    t._token_store = _Store()
    t._article_ids_to_document_ids = types.MethodType(
        lambda self, ds, ids: list(ids), t,
    )
    assert t._repo_leaf_token_count("toy", "repo/w000") == 400
    # Missing root -> 0; no token store -> 0.
    assert t._repo_leaf_token_count("toy", "absent") == 0
    t._token_store = None
    assert t._repo_leaf_token_count("toy", "repo/w000") == 0


def test_per_repo_size_filter_drops_over_threshold_at_grouping():
    """Simulate the _build_dataloaders filter logic: over-threshold repos (by
    file-sample count) are dropped so their shared tree never encodes."""
    from collections import Counter

    from bgkit.data.samplers import RepoGroupedBatchSampler

    # r_small x2, r_ok x3, r_monster x6
    group_keys = (
        ["r_small"] * 2 + ["r_ok"] * 3 + ["r_monster"] * 6
    )
    group_sizes = Counter(group_keys)
    max_file_samples = 5
    dropped = {k for k, sz in group_sizes.items() if sz > max_file_samples}
    assert dropped == {"r_monster"}

    s = RepoGroupedBatchSampler(group_keys, shuffle=False, drop_keys=dropped)
    assert len(s) == 2  # r_small, r_ok kept; r_monster dropped
    assert s.dropped_samples == 6
    yielded = sorted(i for b in s for i in b)
    assert yielded == [0, 1, 2, 3, 4]  # the 6 r_monster indices (5..10) gone


# ---------------------------------------------------------------------------
# 12. GATE: detach-and-reaccumulate — gradient match + once-forward
# ---------------------------------------------------------------------------


def _make_detach_reaccum_trainer(tree_w, base, decode, group_size=1 << 30):
    """Per-repo trainer wired for detach-and-reaccumulate, using a trainable
    Linear ``W`` as the shared-tree producer (memo[nid] = W(base[nid])) and a
    fake decoder ``decode(segments)``. Exercises the REAL
    _shared_tree_node_survivor splice + the REAL _forward_backward_per_repo
    reaccumulate + group-batched PASS 2 (``group_size``)."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    t.device = torch.device("cpu")
    t._round_robin = False
    t._accum_steps = 1
    t.topic_embeddings = None
    t._survivorship_aux = False
    t._live_l0 = True
    t._max_file_samples_per_repo = None
    t._per_repo_sample_group_size = group_size
    t.step_cfg = {}
    t.global_step = 0
    t._recursive_l0_retention_cfg = 0.1
    t._recursive_l1_retention_cfg = 0.1
    t._shared_tree_forward_count = 0
    t._trees = {"toy": object()}  # non-None; per-repo splice path ignores it
    t.decoder = SimpleNamespace(forward_interleaved_with_loss=decode)

    def fake_tree(self, ds, root):
        self._shared_tree_forward_count += 1
        memo = {nid: (tree_w(base[nid]), None) for nid in base}
        return memo, {"nodes": len(base)}

    t._compute_shared_repo_tree = types.MethodType(fake_tree, t)
    t._repo_group_key = types.MethodType(lambda self, s: "repo/w000", t)
    t._sample_contributing_token_count = types.MethodType(lambda self, s: 1, t)

    def fake_prep(self, sample):
        rep = self._shared_tree_node_survivor(sample._node)  # REAL splice
        return {"prepared_turns": [], "_rep": rep, "_v": sample._v}

    t._prepare_sample_for_decode = types.MethodType(fake_prep, t)

    def fake_assemble(self, prep, per_turn):
        segs = [
            TokenSegment(token_ids=torch.ones(1, dtype=torch.long),
                         loss_mask=torch.ones(1)),
            SimpleNamespace(_rep=prep["_rep"], _v=prep["_v"]),
        ]
        return segs, None

    t._assemble_sample_segments = types.MethodType(fake_assemble, t)
    return t


def test_gate_detach_reaccumulate_gradient_match():
    """GATE (a): group-batched detach-and-reaccumulate gradient on the
    shared-tree producer params == the ground-truth Σ_i dLi/dparams (single
    combined backward), within tight tol — and INVARIANT to the group size G
    (test G=1, G=4, G=all). Proves batching doesn't change the gradient."""
    d = 8
    # 6 samples: node A spliced 3x, B 3x -> tests R_d accumulation + spans
    # multiple groups at G=1 and G=4.
    n = 6

    def decode(segments):
        loss = torch.zeros(())
        for s in segments:
            if getattr(s, "_rep", None) is not None:
                loss = loss + (s._rep * s._v).sum()
        return loss

    def build_samples():
        nodes = ["A", "B", "A", "B", "A", "B"]
        return [
            SimpleNamespace(dataset_name="toy", _node=nodes[i], _v=torch.randn(d))
            for i in range(n)
        ]

    # Fixed inputs shared across all G + the reference.
    torch.manual_seed(0)
    base = {"A": torch.randn(d), "B": torch.randn(d)}
    samples = build_samples()
    ref_w0 = torch.nn.Linear(d, d)
    ref_state = {k: v.clone() for k, v in ref_w0.state_dict().items()}

    # --- ground truth: ONE combined backward through the live reps ---
    tree_ref = torch.nn.Linear(d, d)
    tree_ref.load_state_dict(ref_state)
    reps_live = {nid: tree_ref(base[nid]) for nid in base}
    loss_total = sum(
        (reps_live[s._node] * s._v).sum() for s in samples
    ) / (n * 1)
    loss_total.backward()
    ref = tree_ref.weight.grad.detach().clone()
    ref_b = tree_ref.bias.grad.detach().clone()

    results = {}
    for g in (1, 4, n):  # G=1, G=4, G=all
        tree_w = torch.nn.Linear(d, d)
        tree_w.load_state_dict(ref_state)
        t = _make_detach_reaccum_trainer(tree_w, base, decode, group_size=g)
        batch = [
            SimpleNamespace(dataset_name="toy", _node=s._node, _v=s._v)
            for s in samples
        ]
        t._forward_backward_per_repo(batch)
        got = tree_w.weight.grad.detach().clone()
        got_b = tree_w.bias.grad.detach().clone()
        max_abs = (got - ref).abs().max().item()
        max_rel = max_abs / (ref.abs().max().item() or 1.0)
        results[g] = (max_abs, max_rel)
        assert torch.allclose(got, ref, atol=1e-6, rtol=1e-5), (
            f"G={g}: weight grad mismatch max_abs={max_abs:.2e} max_rel={max_rel:.2e}"
        )
        assert torch.allclose(got_b, ref_b, atol=1e-6, rtol=1e-5), f"G={g}: bias grad"
    print("[gradient-match] G -> (max_abs_diff, max_rel_diff):", {
        ("all" if g == n else g): (f"{a:.2e}", f"{r:.2e}")
        for g, (a, r) in results.items()
    })


def test_gate_shared_tree_forward_runs_once():
    """GATE (b): the shared-tree forward runs EXACTLY ONCE per
    _forward_backward_per_repo, regardless of the number of file-samples (no
    per-sample re-run / retain_graph chain)."""
    torch.manual_seed(1)
    d = 8
    base = {"A": torch.randn(d), "B": torch.randn(d)}
    samples_spec = [("A", torch.randn(d)) for _ in range(5)] + [("B", torch.randn(d))]

    def decode(segments):
        loss = torch.zeros(())
        for s in segments:
            if getattr(s, "_rep", None) is not None:
                loss = loss + (s._rep * s._v).sum()
        return loss

    tree_w = torch.nn.Linear(d, d)
    t = _make_detach_reaccum_trainer(tree_w, base, decode)
    batch = [SimpleNamespace(dataset_name="toy", _node=n, _v=v) for n, v in samples_spec]
    t._forward_backward_per_repo(batch)

    # 6 file-samples, but the shared-tree forward fired ONCE.
    assert t._shared_tree_forward_count == 1
    # Splice state torn down after the step.
    assert t._shared_tree_splice_reps is None
    assert t._shared_tree_used_nodes is None


# ---------------------------------------------------------------------------
# 13. profile_timing: per-component timing behind the flag
# ---------------------------------------------------------------------------


def test_timed_helper_off_is_noop():
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    t._profile_timing = False
    store: dict = {}
    with t._timed(store, "x", gpu=True):
        pass
    assert store == {}  # no overhead, no key recorded


def test_timed_helper_on_accumulates():
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    t._profile_timing = True
    store: dict = {}
    for _ in range(2):
        with t._timed(store, "x"):  # gpu=False -> no cuda sync needed on CPU
            sum(range(1000))
    assert "x" in store and store["x"] >= 0.0
    # Two calls accumulate into the same key.
    one: dict = {}
    with t._timed(one, "x"):
        pass
    assert store["x"] >= one["x"]


def test_per_repo_timing_emitted_under_flag(monkeypatch):
    """profile_timing=True makes _forward_backward_per_repo emit a
    `per_repo_timing` event with the component keys; off emits nothing."""
    import bgkit.training.phase2.kr_kb_trainer as M

    d = 8
    base = {"A": torch.randn(d), "B": torch.randn(d)}
    samples_spec = [("A", torch.randn(d)), ("B", torch.randn(d))]

    def decode(segments):
        loss = torch.zeros(())
        for s in segments:
            if getattr(s, "_rep", None) is not None:
                loss = loss + (s._rep * s._v).sum()
        return loss

    events: list[tuple[str, dict]] = []

    class _Rec:
        def info(self, event, **kw):
            events.append((event, kw))
        def warning(self, *a, **k):
            pass
        def error(self, *a, **k):
            pass

    monkeypatch.setattr(M, "logger", _Rec())

    # Flag ON.
    tree_w = torch.nn.Linear(d, d)
    t = _make_detach_reaccum_trainer(tree_w, base, decode)
    t._profile_timing = True
    batch = [SimpleNamespace(dataset_name="toy", _node=nid, _v=v)
             for nid, v in samples_spec]
    t._forward_backward_per_repo(batch)

    timing = [kw for ev, kw in events if ev == "per_repo_timing"]
    assert len(timing) == 1, "exactly one per_repo_timing event per repo"
    rec = timing[0]
    for key in (
        "repo", "n_file_samples", "n_contrib", "n_groups", "group_size",
        "repo_wall_s", "gpu_op_s", "cpu_op_s", "shared_tree_encode_s",
        "pass1_count_s", "prep_s", "drill_encode_s", "assemble_s",
        "decode_fwd_s", "group_backward_s", "final_tree_backward_s",
    ):
        assert key in rec, f"missing timing key {key!r}"
    assert rec["n_file_samples"] == 2 and rec["n_contrib"] == 2
    assert rec["n_groups"] >= 1
    assert rec["repo_wall_s"] >= 0.0

    # Flag OFF -> no per_repo_timing event.
    events.clear()
    tree_w2 = torch.nn.Linear(d, d)
    t2 = _make_detach_reaccum_trainer(tree_w2, base, decode)
    t2._profile_timing = False
    batch2 = [SimpleNamespace(dataset_name="toy", _node=nid, _v=v)
              for nid, v in samples_spec]
    t2._forward_backward_per_repo(batch2)
    assert not any(ev == "per_repo_timing" for ev, _ in events)


# ---------------------------------------------------------------------------
# 14. Node-count size filter (retained-graph memory bound)
# ---------------------------------------------------------------------------


def test_repo_tree_node_count_metric():
    """_repo_tree_node_count counts ALL nodes in the window-0 subtree (interior
    + leaf-tag) — the retained-graph memory driver."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    # window -> [c4a, c4b]; c4a -> [cA, cB]; c4b -> [cC]  => 6 nodes total.
    tree = _StubTree({
        "repo/w000": _StubNode("repo/w000", ["c4a", "c4b"]),
        "c4a": _StubNode("c4a", ["cA", "cB"]),
        "c4b": _StubNode("c4b", ["cC"]),
        "cA": _StubNode("cA", []),
        "cB": _StubNode("cB", []),
        "cC": _StubNode("cC", []),
    })
    t = KRKBTrainer.__new__(KRKBTrainer)
    t._trees = {"toy": tree}
    assert t._repo_tree_node_count("toy", "repo/w000") == 6
    assert t._repo_tree_node_count("toy", "absent") == 0


# ---------------------------------------------------------------------------
# 15. Inner-loop COMPUTE primitive gates (driver integration pending)
# ---------------------------------------------------------------------------


def test_gate_inner_loop_first_step_matches_one_step():
    """GATE (a): the FIRST inner step (subset = ALL files, fresh tree, zero
    staleness) backproping through the LIVE retained tree gives the SAME
    encoder grad as the exact one-step detach-reaccumulate path, within tight
    tol."""
    d, n = 8, 6
    torch.manual_seed(0)
    base = {"A": torch.randn(d), "B": torch.randn(d)}
    nodes = ["A", "B", "A", "B", "A", "B"]
    vs = [torch.randn(d) for _ in range(n)]

    def decode(segments):
        loss = torch.zeros(())
        for s in segments:
            if getattr(s, "_rep", None) is not None:
                loss = loss + (s._rep * s._v).sum()
        return loss

    ref_state = {k: v.clone() for k, v in torch.nn.Linear(d, d).state_dict().items()}

    def batch():
        return [SimpleNamespace(dataset_name="toy", _node=nodes[i], _v=vs[i])
                for i in range(n)]

    # one-step detach-reaccumulate path.
    tree_ref = torch.nn.Linear(d, d)
    tree_ref.load_state_dict(ref_state)
    t_ref = _make_detach_reaccum_trainer(tree_ref, base, decode, group_size=n)
    t_ref._forward_backward_per_repo(batch())
    ref = tree_ref.weight.grad.detach().clone()

    # inner-loop first step: encode tree once (live splice) + ONE subset = all.
    tree_il = torch.nn.Linear(d, d)
    tree_il.load_state_dict(ref_state)
    t_il = _make_detach_reaccum_trainer(tree_il, base, decode, group_size=n)
    t_il._encode_repo_tree_for_inner_loop("toy", "repo/w000")
    t_il._inner_loop_subset_backward(
        batch(), l1_target_ratio=t_il._recursive_l1_retention_now(), normalizer=n,
    )
    got = tree_il.weight.grad.detach().clone()

    max_abs = (got - ref).abs().max().item()
    max_rel = max_abs / (ref.abs().max().item() or 1.0)
    assert torch.allclose(got, ref, atol=1e-6, rtol=1e-5), (
        f"inner-loop first-step grad mismatch: max_abs={max_abs:.2e} max_rel={max_rel:.2e}"
    )
    print(f"[inner-loop gate a] first-step max_abs_diff={max_abs:.2e} max_rel_diff={max_rel:.2e}")


def test_gate_inner_loop_k_steps_reuse_tree_once():
    """GATE (b): K inner steps reuse the tree encoded ONCE; the encoder gets a
    fresh grad each inner step (retain_graph lets the live tree be reused); the
    retained graph is freed after the repo."""
    d, n = 8, 6
    torch.manual_seed(1)
    base = {"A": torch.randn(d), "B": torch.randn(d)}
    nodes = ["A", "B"] * 3
    vs = [torch.randn(d) for _ in range(n)]

    def decode(segments):
        loss = torch.zeros(())
        for s in segments:
            if getattr(s, "_rep", None) is not None:
                loss = loss + (s._rep * s._v).sum()
        return loss

    tree_w = torch.nn.Linear(d, d)
    t = _make_detach_reaccum_trainer(tree_w, base, decode, group_size=4)
    t._per_repo_inner_subset_size = 2
    t._per_repo_max_inner_steps = 12
    batch = [SimpleNamespace(dataset_name="toy", _node=nodes[i], _v=vs[i])
             for i in range(n)]

    t._encode_repo_tree_for_inner_loop("toy", "repo/w000")
    assert t._shared_tree_forward_count == 1  # tree encoded ONCE

    subsets = t._partition_inner_subsets(batch)
    assert len(subsets) == 3 and all(len(s) == 2 for s in subsets)

    n_steps = 0
    for subset in subsets:
        tree_w.weight.grad = None  # simulate optimizer.zero_grad before each step
        _loss_v, _tok, done, _turns = t._inner_loop_subset_backward(
            subset, l1_target_ratio=0.2, normalizer=len(subset),
        )
        assert done == 2
        # Encoder got a fresh gradient THIS inner step (backprop through the
        # reused live tree succeeded — retain_graph kept it alive).
        assert tree_w.weight.grad is not None
        n_steps += 1

    assert n_steps == 3
    assert t._shared_tree_forward_count == 1  # NOT re-encoded across inner steps

    t._free_inner_loop_tree()
    assert t._shared_tree_memo is None
    assert t._shared_tree_splice_reps is None
    assert t._shared_tree_used_nodes is None


def test_gate_inner_loop_subset_bounded_by_g():
    """GATE (c): within an inner step, the subset is processed in groups of G
    (≤ G samples per _encode_decode_group), so per-step memory is bounded by G
    + the retained tree, not the whole subset/repo."""
    d = 8
    base = {"A": torch.randn(d)}

    def decode(segments):
        loss = torch.zeros(())
        for s in segments:
            if getattr(s, "_rep", None) is not None:
                loss = loss + (s._rep * s._v).sum()
        return loss

    tree_w = torch.nn.Linear(d, d)
    t = _make_detach_reaccum_trainer(tree_w, base, decode, group_size=2)
    t._encode_repo_tree_for_inner_loop("toy", "repo/w000")
    subset = [SimpleNamespace(dataset_name="toy", _node="A", _v=torch.randn(d))
              for _ in range(5)]
    sizes = []
    orig = t._encode_decode_group
    t._encode_decode_group = lambda samples, *a, **k: (
        sizes.append(len(samples)), orig(samples, *a, **k))[1]
    t._inner_loop_subset_backward(subset, l1_target_ratio=0.2, normalizer=5)
    # 5 samples / G=2 -> groups of [2,2,1], each <= G.
    assert sizes == [2, 2, 1]
    assert all(s <= 2 for s in sizes)


# ---------------------------------------------------------------------------
# 16. WIRED Model-B inner-loop driver gates (a/b/c/d)
# ---------------------------------------------------------------------------


def _wire_inner_loop_trainer(t):
    """Flip a _make_detach_reaccum_trainer into the wired inner-loop path."""
    t._per_repo_full_backprop = True
    t._per_repo_inner_loop = True
    t._inner_loop_active = True
    t._inner_loop_repo_key = None
    t._inner_loop_repo_steps = 0
    t._per_repo_inner_subset_size = 2
    t._per_repo_max_inner_steps = 12
    t.topic_embeddings = None
    return t


def test_gate_wired_inner_loop_first_step_matches_one_step():
    """GATE (a) [WIRED]: the first inner step (subset = ALL files, fresh tree)
    via _forward_backward_inner_loop bit-matches the exact one-step path's
    encoder grad."""
    d, n = 8, 6
    torch.manual_seed(0)
    base = {"A": torch.randn(d), "B": torch.randn(d)}
    nodes = ["A", "B"] * 3
    vs = [torch.randn(d) for _ in range(n)]

    def decode(segments):
        loss = torch.zeros(())
        for s in segments:
            if getattr(s, "_rep", None) is not None:
                loss = loss + (s._rep * s._v).sum()
        return loss

    ref_state = {k: v.clone() for k, v in torch.nn.Linear(d, d).state_dict().items()}

    def batch():
        return [SimpleNamespace(dataset_name="toy", _node=nodes[i], _v=vs[i])
                for i in range(n)]

    # one-step path.
    tree_ref = torch.nn.Linear(d, d)
    tree_ref.load_state_dict(ref_state)
    t_ref = _make_detach_reaccum_trainer(tree_ref, base, decode, group_size=1)
    t_ref._forward_backward_per_repo(batch())
    ref = tree_ref.weight.grad.detach().clone()

    # WIRED inner-loop, ONE subset = all n files (S=n -> K=1).
    tree_il = torch.nn.Linear(d, d)
    tree_il.load_state_dict(ref_state)
    t_il = _make_detach_reaccum_trainer(tree_il, base, decode, group_size=1)
    _wire_inner_loop_trainer(t_il)
    t_il._per_repo_inner_subset_size = n  # one subset = all files
    t_il._forward_backward_inner_loop(batch())  # first (only) subset
    got = tree_il.weight.grad.detach().clone()

    max_abs = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, atol=1e-6, rtol=1e-5), f"max_abs={max_abs:.2e}"
    print(f"[wired inner-loop gate a] first-step max_abs_diff={max_abs:.2e}")


def test_gate_wired_inner_loop_structure_and_mid_repo_eval():
    """GATE (b) [WIRED]: K inner steps reuse a tree encoded ONCE per repo;
    encoder grad each step; a mid-repo eval does NOT free/corrupt the cached
    tree; the tree is freed on repo-change."""
    d = 8
    torch.manual_seed(1)
    base = {"A": torch.randn(d), "B": torch.randn(d)}

    def decode(segments):
        loss = torch.zeros(())
        for s in segments:
            if getattr(s, "_rep", None) is not None:
                loss = loss + (s._rep * s._v).sum()
        return loss

    tree_w = torch.nn.Linear(d, d)
    t = _make_detach_reaccum_trainer(tree_w, base, decode, group_size=1)
    _wire_inner_loop_trainer(t)
    t._repo_group_key = types.MethodType(lambda self, s: s._repo, t)

    # Repo R1: 3 subsets of 2 files (sampler would emit these consecutively).
    def mk(repo, node):
        return SimpleNamespace(dataset_name="toy", _repo=repo, _node=node,
                               _v=torch.randn(d))
    r1_subsets = [[mk("R1", "A"), mk("R1", "B")] for _ in range(3)]

    n_steps = 0
    for i, subset in enumerate(r1_subsets):
        tree_w.weight.grad = None  # base loop optimizer.zero_grad
        m = t._forward_backward_inner_loop(subset)
        assert m["inner_loop"] == 1.0
        assert tree_w.weight.grad is not None  # encoder updated this inner step
        assert t._shared_tree_forward_count == 1  # tree encoded ONCE for R1
        assert t._inner_loop_repo_key == "R1"
        n_steps += 1
        # Simulate a mid-repo eval after the 2nd subset: zero grads (as
        # _release_training_transients does) — must NOT free/stale the tree.
        if i == 1:
            t.optimizer = SimpleNamespace(zero_grad=lambda **k: None)
            t._release_training_transients() if hasattr(t, "_release_training_transients") else None
            assert t._shared_tree_memo is not None  # tree survives eval
            assert t._inner_loop_repo_key == "R1"
    assert n_steps == 3
    assert t._shared_tree_forward_count == 1

    # Repo change R1 -> R2: the R1 tree is freed and R2's encoded fresh (once).
    r2_subset = [mk("R2", "A"), mk("R2", "B")]
    t._forward_backward_inner_loop(r2_subset)
    assert t._inner_loop_repo_key == "R2"
    assert t._shared_tree_forward_count == 2  # exactly one re-encode on change


def test_gate_wired_inner_loop_memory_g1_one_subset():
    """GATE (c) [WIRED]: at G=1 the subset is processed ONE sample at a time
    (≤1 sample per _encode_decode_group call) — no multi-sample concurrency."""
    d = 8
    base = {"A": torch.randn(d)}

    def decode(segments):
        loss = torch.zeros(())
        for s in segments:
            if getattr(s, "_rep", None) is not None:
                loss = loss + (s._rep * s._v).sum()
        return loss

    tree_w = torch.nn.Linear(d, d)
    t = _make_detach_reaccum_trainer(tree_w, base, decode, group_size=1)
    _wire_inner_loop_trainer(t)
    t._repo_group_key = types.MethodType(lambda self, s: "R", t)
    sizes = []
    orig = t._encode_decode_group
    t._encode_decode_group = lambda samples, *a, **k: (
        sizes.append(len(samples)), orig(samples, *a, **k))[1]
    subset = [SimpleNamespace(dataset_name="toy", _repo="R", _node="A",
                              _v=torch.randn(d)) for _ in range(4)]
    t._forward_backward_inner_loop(subset)
    # G=1 -> 4 calls of exactly 1 sample each (one graph at a time).
    assert sizes == [1, 1, 1, 1]


def test_gate_wired_warmup_switch_rebuilds_sampler():
    """GATE (d): the warmup→inner-loop switch in _pre_step_hook flips
    _inner_loop_active and rebuilds the loader into a valid subset-emitting
    sampler at the boundary; before the boundary it stays one-step."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    t._per_repo_full_backprop = True
    t._per_repo_inner_loop = True
    t._inner_loop_warmup_steps = 300
    t._inner_loop_active = False
    t._per_repo_inner_subset_size = 2
    t._per_repo_max_inner_steps = 12
    t._repo_group_keys = ["r1", "r1", "r1", "r2", "r2"]
    t._repo_dropped_keys = set()
    t._repo_num_workers = 0
    t.cfg = {"seed": 7}
    t.epoch = 0
    t.train_dataset = list(range(5))
    t._dataloader_invalidated = False
    # _free_inner_loop_tree needs these.
    t._shared_tree_memo = None
    t._shared_tree_splice_reps = None
    t._shared_tree_used_nodes = None
    t._per_repo_shared_tree_active = False

    import bgkit.training.phase2.kr_kb_trainer as M
    orig_dl = M.DataLoader
    M.DataLoader = lambda *a, **k: ("DL", k.get("batch_sampler"))
    try:
        # Before warmup: no switch.
        t.global_step = 0
        t._pre_step_hook()
        assert t._inner_loop_active is False
        assert t._dataloader_invalidated is False
        # At the boundary: switch fires.
        t.global_step = 300
        t._pre_step_hook()
        assert t._inner_loop_active is True
        assert t._dataloader_invalidated is True
        sampler = t._train_batch_sampler
        assert sampler._inner_loop is True
        # Valid subset-emitting state: r1(3)->[2,1], r2(2)->[2] = 3 batches.
        assert len(sampler) == 3
        # Idempotent: a later step does not re-switch.
        t._dataloader_invalidated = False
        t.global_step = 350
        t._pre_step_hook()
        assert t._dataloader_invalidated is False
    finally:
        M.DataLoader = orig_dl


# ---------------------------------------------------------------------------
# 17. Single-forward activation-peak bounds (the 4th-OOM fix)
# ---------------------------------------------------------------------------


def test_max_decode_tokens_truncates_long_gold_keeps_all():
    """A long-gold sample is NOT skipped — its gold OUTPUT is hard-truncated to
    the first max_decode_tokens gold tokens (no short-file bias). BOTH a short
    and a long sample decode; the long one's gold is capped."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    w = torch.nn.Parameter(torch.zeros(1))
    decoded_gold: list[int] = []

    def fake_decode(segments):
        # record the gold (loss-masked) token count actually decoded.
        g = sum(
            int(s.loss_mask.sum()) for s in segments
            if isinstance(s, TokenSegment) and s.loss_mask is not None
        )
        decoded_gold.append(g)
        return w.sum()

    t = KRKBTrainer.__new__(KRKBTrainer)
    t.device = torch.device("cpu")
    t._max_decode_tokens = 100
    t._per_repo_sample_group_size = 1
    t._recursive_l1_retention_cfg = 0.1
    t.global_step = 0
    t.decoder = SimpleNamespace(forward_interleaved_with_loss=fake_decode)
    t.encoder = SimpleNamespace(active_projection_output_dim=8)

    # The assemble returns a gold segment of the sample's gold length.
    gold_len = {"S": 50, "L": 5000}
    t._prepare_sample_for_decode = types.MethodType(
        lambda self, s: {"prepared_turns": [], "_g": gold_len[s]}, t,
    )
    t._assemble_sample_segments = types.MethodType(
        lambda self, prep, per_turn: (
            [TokenSegment(token_ids=torch.ones(prep["_g"], dtype=torch.long),
                          loss_mask=torch.ones(prep["_g"]))],
            None,
        ),
        t,
    )

    _loss, _tk, n_done, _nt, _ = t._encode_decode_group(["S", "L"])
    # BOTH samples decoded (no skip); the long one's gold capped at 100.
    assert n_done == 2
    assert sorted(decoded_gold) == [50, 100]


def test_max_decode_tokens_off_decodes_all():
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    w = torch.nn.Parameter(torch.zeros(1))
    decoded: list[int] = []
    t = KRKBTrainer.__new__(KRKBTrainer)
    t.device = torch.device("cpu")
    t._max_decode_tokens = 0  # off
    t._per_repo_sample_group_size = 1
    t._recursive_l1_retention_cfg = 0.1
    t.global_step = 0
    t.decoder = SimpleNamespace(
        forward_interleaved_with_loss=lambda segs: decoded.append(1) or w.sum())
    t.encoder = SimpleNamespace(active_projection_output_dim=8)
    long_ = {"prepared_turns": [], "token_ids": torch.ones(5000, dtype=torch.long)}
    t._prepare_sample_for_decode = types.MethodType(lambda self, s: long_, t)
    t._assemble_sample_segments = types.MethodType(
        lambda self, prep, per_turn: (
            [TokenSegment(token_ids=torch.ones(1, dtype=torch.long),
                          loss_mask=torch.ones(1))], None), t)
    _l, _tk, n_done, _nt, _ = t._encode_decode_group(["x"])
    assert n_done == 1 and len(decoded) == 1  # cap off -> decodes


def test_max_l0_encode_tokens_truncates_buffer():
    """_live_l0_encode truncates the per-leaf token buffer to the cap (bounds
    the window-0 initial-import commit's single L0 forward + retained tree)."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    captured = {}

    class _Store:
        def get(self, dataset, aid):
            # two "files": 30 + 90 tokens = 120 total.
            return torch.ones(30 if aid == "a" else 90, dtype=torch.long)

    class _Embed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(8))
        def forward(self, ids):
            captured["n_tokens"] = int(ids.shape[0])  # the buffer fed to L0
            return torch.zeros(int(ids.shape[0]), 8)

    embed = _Embed()
    t = KRKBTrainer.__new__(KRKBTrainer)
    t.device = torch.device("cpu")
    t._token_store = _Store()
    t._max_l0_encode_tokens = 64  # < 120 -> truncate
    t.encoder = SimpleNamespace(
        l0=SimpleNamespace(backbone=SimpleNamespace(get_input_embeddings=lambda: embed)),
    )
    # Stub the heavy bits after the buffer is built.
    t._sample_l0_retention_for = types.MethodType(lambda self, ds: 0.1, t)
    t._surv_l0 = None
    t._l0_prompt_tokens = 0
    t._checkpointed_level = types.MethodType(
        lambda self, level, **kw: SimpleNamespace(
            survivor_embeddings=torch.zeros(1, 8),
            survivor_cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
        ), t,
    )
    from bgkit.training.survivorship_helpers import LevelLossCfg
    t._surv_l0 = LevelLossCfg()
    t.encoder.training = True

    t._live_l0_encode("git_commit_repro", ["a", "b"])
    # The L0 input buffer was truncated from 120 -> 64 tokens.
    assert captured["n_tokens"] == 64


# ---------------------------------------------------------------------------
# 18. FIX 2: tree-encode checkpointing — gradient parity + θ-once
# ---------------------------------------------------------------------------


def test_fix2_tree_encode_checkpoint_gradient_parity():
    """Checkpointing the per-node L1 forward gives the SAME gradient as the
    non-checkpointed path (checkpoint recompute is exact), AND θ is accumulated
    exactly ONCE (hoisted outside the checkpoint — the recompute must not
    double-count)."""
    import contextlib as _ctx

    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
    from bgkit.training.survivorship_helpers import init_state

    d = 8
    lin = torch.nn.Linear(d, d)
    # Shared (module-scope) embedding + tokenizer so the shared encode
    # primitive's child-ID injection is IDENTICAL across the two runs — the
    # grad-parity assertion compares lin.weight.grad between checkpoint on/off.
    id_emb = torch.nn.Embedding(4, d)

    def make_run_l1():
        def fake(*, l1_input_embeddings, l1_input_cu_seqlens, target_ratio_l1, **kw):
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
        return fake

    def build():
        t = KRKBTrainer.__new__(KRKBTrainer)
        t.device = torch.device("cpu")
        l0 = SimpleNamespace(
            backbone=SimpleNamespace(get_input_embeddings=lambda: id_emb),
        )
        t.encoder = SimpleNamespace(
            training=True, run_l1_and_project=make_run_l1(), l0=l0,
        )
        t.encoder_tokenizer = SimpleNamespace(
            encode=lambda text, add_special_tokens=False: [1],
        )
        t._l1_adapter_context = types.MethodType(lambda self: _ctx.nullcontext(), t)
        t._recursive_l1_retention_cfg = 0.2
        t.global_step = 0
        t._survivorship_aux = False
        t._surv_state_l0 = init_state()
        t._surv_state_l1 = init_state()
        return t

    children = torch.randn(5, d)
    q = torch.randn(2, d)

    def run(checkpoint: bool):
        lin.weight.grad = None
        t = build()
        t._checkpoint_tree_encode = checkpoint
        x = children.clone().requires_grad_(True)
        proj, _l1 = t._encode_tree_node_live(["c0"], [x], q)
        proj.sum().backward()
        return (
            lin.weight.grad.detach().clone(),
            int(t._surv_state_l1.controllable_count_sum),
        )

    ref_grad, ref_ctrl = run(checkpoint=False)
    got_grad, got_ctrl = run(checkpoint=True)

    max_abs = (got_grad - ref_grad).abs().max().item()
    assert torch.allclose(got_grad, ref_grad, atol=1e-6, rtol=1e-5), (
        f"checkpointed tree-encode grad mismatch: max_abs={max_abs:.2e}"
    )
    # θ accumulated ONCE in both (controllable=10), not doubled (20) by the
    # checkpoint recompute.
    assert ref_ctrl == 10 and got_ctrl == 10, (ref_ctrl, got_ctrl)
    print(f"[fix2 tree-encode] grad max_abs_diff={max_abs:.2e}  theta_ctrl={got_ctrl}")


# ---------------------------------------------------------------------------
# 19. FIX 2b: L0 leaf-encode checkpointing — gradient parity + θ-once
# ---------------------------------------------------------------------------


def test_fix2b_l0_leaf_checkpoint_gradient_parity():
    """Checkpointing the L0 leaf encode gives the SAME gradient as the
    non-checkpointed path (exact recompute), AND L0 θ is accumulated exactly
    ONCE (hoisted outside the checkpoint — no double-count on recompute)."""
    import contextlib as _ctx

    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
    from bgkit.training.survivorship_helpers import init_state

    d = 8
    lin = torch.nn.Linear(d, d)  # the (trainable) L0 stand-in

    class _L0(SimpleNamespace):
        def auto_reproduce(self, x):
            return lin(x)  # L0-out -> L1-in, trainable

    def make_live_l0():
        # Deterministic given ratio (no sampling inside). Survivors from lin.
        def fake(dataset, article_ids, query_emb=None, ratio=None):
            base = torch.ones(4, d)  # fixed "tokens" surrogate
            out = SimpleNamespace(
                survivor_embeddings=lin(base),
                survivor_cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
                organic_count=torch.tensor(2),
                controllable_count=torch.tensor(8),
                valid_count=torch.tensor(8),
            )
            return out, torch.tensor([0, 4], dtype=torch.int32), float(ratio or 0.1)
        return fake

    def build():
        t = KRKBTrainer.__new__(KRKBTrainer)
        t.device = torch.device("cpu")
        t.encoder = SimpleNamespace(training=True, l0=_L0())
        t._survivorship_aux = False
        t._surv_state_l0 = init_state()
        t._surv_state_l1 = init_state()
        t._live_l0_encode = types.MethodType(
            lambda self, ds, aids, query_emb=None, ratio=None, selection_mode="threshold":
                make_live_l0()(ds, aids, query_emb, ratio), t,
        )
        # _encode_tree_node_live stub: identity-ish over the per-child survivor
        # list, returns (proj, l1out). (The leaf now splits L0 survivors per
        # article and injects each article's id via the shared primitive.)
        t._encode_tree_node_live = types.MethodType(
            lambda self, children_ids, children_survivors_l1in, q: (
                torch.cat(list(children_survivors_l1in), 0),
                torch.cat(list(children_survivors_l1in), 0),
            ),
            t,
        )
        t._sample_l0_retention_for = types.MethodType(lambda self, ds: 0.15, t)

        # Non-checkpoint reference: mirror _l0_for_articles' aux-off branch
        # (live encode + θ accumulated inside, return survivors).
        def _l0_for_articles(self, ds, aids, query_emb=None, selection_mode="threshold"):
            out, cu, ratio = self._live_l0_encode(
                ds, aids, query_emb=query_emb, selection_mode=selection_mode
            )
            self._accumulate_theta_from_counts(
                "l0",
                torch.tensor([
                    float(out.organic_count), float(out.controllable_count),
                    float(out.valid_count),
                ]),
                ratio,
            )
            return out.survivor_embeddings, cu
        t._l0_for_articles = types.MethodType(_l0_for_articles, t)
        return t

    node = SimpleNamespace(id="commitA")
    q = torch.randn(2, d)

    def run(checkpoint: bool):
        lin.weight.grad = None
        t = build()
        t._checkpoint_tree_encode = checkpoint
        t._resolve_article_ids = types.MethodType(
            lambda self, ds, ids: ["a", "b"], t,
        )
        proj, _l1 = t._encode_leaf_subtree("toy", node, q)
        proj.sum().backward()
        return (
            lin.weight.grad.detach().clone(),
            int(t._surv_state_l0.controllable_count_sum),
        )

    ref_grad, ref_ctrl = run(checkpoint=False)
    got_grad, got_ctrl = run(checkpoint=True)
    max_abs = (got_grad - ref_grad).abs().max().item()
    assert torch.allclose(got_grad, ref_grad, atol=1e-6, rtol=1e-5), (
        f"L0-leaf checkpoint grad mismatch: max_abs={max_abs:.2e}"
    )
    # L0 θ accumulated ONCE (controllable=8), not doubled (16) on recompute.
    assert ref_ctrl == 8 and got_ctrl == 8, (ref_ctrl, got_ctrl)
    print(f"[fix2b L0-leaf] grad max_abs_diff={max_abs:.2e}  theta_ctrl={got_ctrl}")


def test_conditional_encode_checkpoint_gated_by_tree_size():
    """SPEED: the encode checkpoint fires ONLY when _tree_encode_ckpt_active is
    True (set per-tree from the node count). With _checkpoint_tree_encode on but
    the per-tree flag off (small tree), _encode_tree_node_live runs the forward
    DIRECTLY (no torch.utils.checkpoint call) → no recompute → fast."""
    import contextlib as _ctx

    import torch.utils.checkpoint as _ckpt_mod

    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
    from bgkit.training.survivorship_helpers import init_state

    d = 8
    lin = torch.nn.Linear(d, d)

    def fake_run_l1(*, l1_input_embeddings, l1_input_cu_seqlens, target_ratio_l1, **kw):
        surv = lin(l1_input_embeddings)
        proj = lin(l1_input_embeddings)
        l1_out = SimpleNamespace(
            survivor_embeddings=surv,
            survivor_cu_seqlens=torch.tensor([0, surv.shape[0]], dtype=torch.int32),
            organic_count=torch.tensor(3),
            controllable_count=torch.tensor(10),
            valid_count=torch.tensor(10),
        )
        return l1_out, SimpleNamespace(projected_embeddings=proj), None

    id_emb = torch.nn.Embedding(4, d)

    def build(active: bool):
        t = KRKBTrainer.__new__(KRKBTrainer)
        t.device = torch.device("cpu")
        l0 = SimpleNamespace(
            backbone=SimpleNamespace(get_input_embeddings=lambda: id_emb),
        )
        t.encoder = SimpleNamespace(
            training=True, run_l1_and_project=fake_run_l1, l0=l0,
        )
        t.encoder_tokenizer = SimpleNamespace(
            encode=lambda text, add_special_tokens=False: [1],
        )
        t._l1_adapter_context = types.MethodType(lambda self: _ctx.nullcontext(), t)
        t._recursive_l1_retention_cfg = 0.2
        t.global_step = 0
        t._survivorship_aux = False
        t._surv_state_l0 = init_state()
        t._surv_state_l1 = init_state()
        t._checkpoint_tree_encode = True
        t._tree_encode_ckpt_active = active
        return t

    q = torch.randn(2, d)

    def run(active: bool, monkeypatch_calls: list):
        t = build(active)
        x = torch.randn(5, d, requires_grad=True)
        orig = _ckpt_mod.checkpoint

        def spy(*a, **kw):
            monkeypatch_calls.append(1)
            return orig(*a, **kw)

        _ckpt_mod.checkpoint = spy
        try:
            proj, _l1 = t._encode_tree_node_live(["c0"], [x], q)
        finally:
            _ckpt_mod.checkpoint = orig
        proj.sum().backward()
        return len(monkeypatch_calls)

    on_calls = run(active=True, monkeypatch_calls=[])
    off_calls = run(active=False, monkeypatch_calls=[])
    assert on_calls == 1, f"expected 1 checkpoint call when active, got {on_calls}"
    assert off_calls == 0, f"expected 0 checkpoint calls when inactive, got {off_calls}"


# ---------------------------------------------------------------------------
# Option A — crash-free amortized per-repo (decoder K-step / encoder 1-step)
# ---------------------------------------------------------------------------


def _option_a_trainer(dec_params, enc_params, *, dec_lr=0.0, enc_lr=0.0):
    """A KRKBTrainer shell with a real single optimizer over a decoder group
    and an encoder group, and the decoder param-id set installed — enough to
    exercise the selective-step / group-classification helpers."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    t.device = torch.device("cpu")
    t.optimizer = torch.optim.SGD(
        [
            {"params": list(dec_params), "lr": dec_lr},
            {"params": list(enc_params), "lr": enc_lr},
        ],
        lr=0.0,
    )
    t._option_a_decoder_param_ids = frozenset(id(p) for p in dec_params)
    return t


def test_option_a_group_classification_by_identity():
    """_option_a_group_indices classifies each param-group as decoder vs
    encoder by param identity (robust to Muon-style group splitting)."""
    dec = torch.nn.Parameter(torch.randn(3, 3))
    enc_a = torch.nn.Parameter(torch.randn(3, 3))
    enc_b = torch.nn.Parameter(torch.randn(3))
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    t.device = torch.device("cpu")
    # Three groups: [decoder], [encoder 2D], [encoder 1D] (mimics a Muon split
    # that put the encoder's 2D and 1D params in separate groups).
    t.optimizer = torch.optim.SGD(
        [
            {"params": [dec], "lr": 0.1},
            {"params": [enc_a], "lr": 0.1},
            {"params": [enc_b], "lr": 0.1},
        ],
        lr=0.0,
    )
    t._option_a_decoder_param_ids = frozenset({id(dec)})
    dec_idx, enc_idx = t._option_a_group_indices()
    assert dec_idx == [0]
    assert enc_idx == [1, 2]
    assert t._option_a_params_for_groups(dec_idx) == [dec]
    assert t._option_a_params_for_groups(enc_idx) == [enc_a, enc_b]


def test_option_a_selective_step_moves_only_chosen_group():
    """_option_a_step_groups steps ONLY the selected param-group; the other
    group's weights are untouched even when it has a grad."""
    dec = torch.nn.Parameter(torch.zeros(2))
    enc = torch.nn.Parameter(torch.zeros(2))
    t = _option_a_trainer([dec], [enc], dec_lr=1.0, enc_lr=1.0)
    dec_idx, _enc_idx = t._option_a_group_indices()
    dec.grad = torch.ones(2)
    enc.grad = torch.ones(2)
    t._option_a_step_groups(dec_idx)  # SGD lr=1 → dec -= 1
    assert torch.allclose(dec, torch.full((2,), -1.0))
    assert torch.allclose(enc, torch.zeros(2)), "encoder must NOT move"
    # param_groups restored after the partial step.
    assert len(t.optimizer.param_groups) == 2


def test_option_a_decoder_grad_matches_standalone_subset():
    """GATE (a): a subset's DECODER grad (reading the DETACHED tree_rd) is bit-exact
    vs a standalone forward+backward of that subset with the same decoder
    weights + same tree_rd (detachment isolates the decoder graph)."""
    torch.manual_seed(0)
    d = 4
    enc_w = torch.nn.Parameter(torch.randn(d, d))
    x = torch.randn(d)
    tree_r = enc_w @ x
    tree_rd = tree_r.detach().requires_grad_(True)

    dec_w = torch.nn.Parameter(torch.randn(d, d))
    # Option-A-style subset decode reading tree_rd.
    loss = (dec_w @ tree_rd).sum()
    loss.backward()
    got = dec_w.grad.clone()

    # Standalone: same weights, same tree_rd, fresh graph.
    dec_w2 = torch.nn.Parameter(dec_w.detach().clone())
    tree_rd2 = tree_r.detach().requires_grad_(True)
    (dec_w2 @ tree_rd2).sum().backward()
    ref = dec_w2.grad

    assert torch.equal(got, ref), (got - ref).abs().max().item()


def test_option_a_encoder_tree_grad_bitexact_sum_of_rd_grads():
    """GATE (b): the ENCODER grad from ONE final tree-backward fed the
    ACCUMULATED tree_rd.grad is bit-exact vs feeding the explicit SUM of the
    per-file tree_rd.grad through one tree-backward. Validates that (1) repeated
    ``loss.backward()`` accumulation sums correctly into ``tree_rd.grad`` and (2)
    one tree-backward of that accumulated grad equals one tree-backward of the
    explicit sum (same input tensor → bit-identical encoder grad)."""
    torch.manual_seed(1)
    d = 4
    x = torch.randn(d)

    # Decoder + K "subset" decodes reading the DETACHED tree_rd → accumulate
    # tree_rd.grad via repeated backward (the production path).
    enc1 = torch.nn.Parameter(torch.randn(d, d))
    tree_r1 = (enc1 ** 2) @ x
    tree_rd1 = tree_r1.detach().requires_grad_(True)
    dec = torch.randn(d)
    for k in range(3):
        ((dec * (k + 1)) * tree_rd1).sum().backward()  # leaf tree_rd1 → no retain needed
    accum = tree_rd1.grad.clone()
    # The accumulation IS the sum of per-file dL/dR_d (= Σ_k dec*(k+1) = 6·dec).
    assert torch.allclose(accum, dec * 6.0, atol=1e-6)

    # One tree-backward of the accumulated grad.
    torch.autograd.backward([tree_r1], [accum])
    enc_grad_accum = enc1.grad.clone()

    # Reference: feed the explicit sum (same tensor) through ONE tree-backward
    # on a fresh identical graph → bit-exact.
    enc2 = torch.nn.Parameter(enc1.detach().clone())
    tree_r2 = (enc2 ** 2) @ x
    torch.autograd.backward([tree_r2], [accum.clone()])
    enc_grad_ref = enc2.grad.clone()

    assert torch.equal(enc_grad_accum, enc_grad_ref), (
        (enc_grad_accum - enc_grad_ref).abs().max().item()
    )


def test_option_a_no_inplace_crash_decoder_steps_then_encoder_backward():
    """GATE (c): stepping the DECODER between subset backwards does NOT
    invalidate tree_r's retained graph — the final encoder tree-backward succeeds.
    Mirrors Option A's exact op sequence (detached tree_rd, drill touches encoder,
    decoder-only step per subset, deferred single encoder backward)."""
    torch.manual_seed(2)
    d = 4
    enc_w = torch.nn.Parameter(torch.randn(d, d))
    dec_w = torch.nn.Parameter(torch.randn(d))
    x = torch.randn(d)

    tree_r = (enc_w ** 2) @ x                # retained tree output (enc_w value-bearing)
    tree_rd = tree_r.detach().requires_grad_(True)
    t = _option_a_trainer([dec_w], [enc_w], dec_lr=0.5, enc_lr=0.5)
    dec_idx, enc_idx = t._option_a_group_indices()

    dec_before = dec_w.detach().clone()
    n_subsets = 3
    for k in range(n_subsets):
        t._option_a_zero_grads([dec_w])  # null ONLY decoder grad
        # subset decode reads tree_rd (detached) AND a live "drill" touching enc_w.
        loss = (dec_w * tree_rd).sum() + (enc_w * (k + 1)).sum()
        loss.backward()                  # dec grad + enc drill grad(+=) + tree_rd.grad(+=)
        assert dec_w.grad is not None
        t._option_a_step_groups(dec_idx)  # step ONLY decoder; enc_w untouched
    # decoder moved (K updates); encoder NOT yet stepped.
    assert not torch.equal(dec_w.detach(), dec_before)

    # accumulated drill grad on enc_w = 1+2+3 = 6 per element (before tree).
    enc_drill_grad = enc_w.grad.clone()
    assert torch.allclose(enc_drill_grad, torch.full((d, d), 6.0))

    # FINAL: null decoder grad, ONE tree-backward through the still-valid tree_r.
    t._option_a_zero_grads([dec_w])
    enc_before = enc_w.detach().clone()
    # This is the line that would raise "modified by an inplace operation" if
    # the encoder had been stepped mid-loop.
    torch.autograd.backward([tree_r], [tree_rd.grad])  # MUST NOT raise
    # enc_w.grad now = drill (6) + tree contribution (d(tree_r)/d(enc_w) · tree_rd.grad).
    assert not torch.equal(enc_w.grad, enc_drill_grad)
    t._option_a_step_groups(enc_idx)          # encoder stepped ONCE
    assert not torch.equal(enc_w.detach(), enc_before)


def test_option_a_negative_control_stepping_encoder_midloop_crashes():
    """Negative control proving the crux is load-bearing: if the ENCODER is
    stepped mid-loop (mutating a weight tree_r's retained graph depends on), the
    later tree-backward RAISES — exactly the original inner-loop crash."""
    torch.manual_seed(3)
    d = 4
    enc_w = torch.nn.Parameter(torch.randn(d, d))
    x = torch.randn(d)
    # (enc_w ** 2) @ x makes enc_w's VALUE load-bearing in tree_r's backward (the
    # pow saves enc_w), so an in-place step on enc_w trips the version guard.
    tree_r = (enc_w ** 2) @ x
    tree_rd = tree_r.detach().requires_grad_(True)
    t = _option_a_trainer([], [enc_w], enc_lr=0.5)
    _dec_idx, enc_idx = t._option_a_group_indices()

    # Populate an encoder grad and STEP it (the wrong thing to do mid-repo).
    enc_w.grad = torch.ones(d, d)
    t._option_a_step_groups(enc_idx)  # mutates enc_w in place
    # Now the retained tree_r graph references the pre-step enc_w version.
    (tree_rd.sum()).backward()  # fills tree_rd.grad
    with pytest.raises(RuntimeError):
        torch.autograd.backward([tree_r], [tree_rd.grad])



# ---------------------------------------------------------------------------
# Full-tree Option A: drill-checkpoint, gold-output truncation, token gate
# ---------------------------------------------------------------------------


def test_partition_option_a_uncapped_vs_inner_capped():
    """Option A partitions into ALL ceil(n/S) subsets (no K-cap → no file
    dropped); the legacy inner-loop partition caps at per_repo_max_inner_steps."""
    import math as _m

    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    t._per_repo_inner_subset_size = 16
    t._per_repo_max_inner_steps = 4
    t._option_a_max_subsets = 0  # unlimited
    samples = list(range(100))  # 100 files, S=16 → 7 subsets
    opt_a = t._partition_option_a_subsets(samples)
    assert len(opt_a) == _m.ceil(100 / 16)            # all 7 — no file dropped
    assert sum(len(s) for s in opt_a) == 100
    # Legacy inner-loop drops the remainder beyond the K-cap.
    inner = t._partition_inner_subsets(samples)
    assert len(inner) == 4 and sum(len(s) for s in inner) == 64


def test_truncate_segments_to_gold_budget():
    """The gold OUTPUT is hard-cut to the first N gold tokens (prefix kept,
    tail + subsequent segments dropped), the sample is NOT skipped."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)

    def tok_seg(n, gold):
        return TokenSegment(
            token_ids=torch.arange(n, dtype=torch.long),
            loss_mask=(torch.ones(n) if gold else torch.zeros(n)),
        )

    # prefix (10 non-gold) + gold(20) + trailing (5 non-gold "suffix/end").
    segs = [tok_seg(10, False), tok_seg(20, True), tok_seg(5, False)]

    # N=8 < 20 → cut inside the gold segment after the 8th gold token; drop the
    # gold tail + the trailing segment.
    out = t._truncate_segments_to_gold_budget(segs, 8)
    assert len(out) == 2  # prefix + truncated gold (trailing dropped)
    assert int(out[0].loss_mask.sum()) == 0
    assert int(out[1].loss_mask.sum()) == 8       # first-8 gold only
    assert out[1].token_ids.shape[0] == 8

    # N >= total gold → unchanged (no truncation, sample kept whole).
    out2 = t._truncate_segments_to_gold_budget(segs, 50)
    assert out2 is segs or len(out2) == 3
    assert sum(int(s.loss_mask.sum()) for s in out2) == 20

    # N=0 → no-op.
    assert t._truncate_segments_to_gold_budget(segs, 0) is segs


def test_drill_checkpoint_gradient_parity_and_theta_once():
    """GATE (a): the drill (run_l1_and_project) checkpointed gives a bit-exact
    encoder gradient vs un-checkpointed, AND θ is accumulated exactly ONCE
    (hoisted outside the checkpoint — recompute must not double-count)."""
    import contextlib as _ctx

    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
    from bgkit.training.survivorship_helpers import LevelLossCfg, init_state

    d = 8
    lin = torch.nn.Linear(d, d)

    def make_enc():
        def run_l1(*, l1_input_embeddings, l1_input_cu_seqlens, target_ratio_l1,
                   **kw):
            surv = lin(l1_input_embeddings)
            out = SimpleNamespace(
                survivor_embeddings=surv,
                organic_count=torch.tensor(3),
                controllable_count=torch.tensor(10),
                valid_count=torch.tensor(10),
            )
            proj_out = SimpleNamespace(
                projected_embeddings=surv, survivor_cu_seqlens=None,
            )
            return out, proj_out, l1_input_cu_seqlens
        return SimpleNamespace(
            training=True,
            active_projection_output_dim=d,
            l0=SimpleNamespace(auto_reproduce=lambda x: x),
            run_l1_and_project=run_l1,
        )

    def build(threshold):
        t = KRKBTrainer.__new__(KRKBTrainer)
        t.device = torch.device("cpu")
        t.encoder = make_enc()
        t._l1_adapter_context = types.MethodType(lambda self: _ctx.nullcontext(), t)
        t._survivorship_aux = False
        t._surv_l1 = LevelLossCfg()
        t._surv_state_l0 = init_state()
        t._surv_state_l1 = init_state()
        t._drill_checkpoint_min_seqlen = threshold
        return t

    # content length 6 (> threshold 2 when checkpointing; > 0 always requires_grad).
    def turn():
        return {
            "content": torch.randn(6, d, requires_grad=True),
            "query_emb": torch.randn(2, d),
            "pinned": torch.zeros(6, dtype=torch.bool),
            "relevance_mask": torch.zeros(6, dtype=torch.bool),
            "survivor_mask": torch.zeros(6, dtype=torch.bool),
        }

    base_turn = turn()

    def run(threshold):
        lin.weight.grad = None
        t = build(threshold)
        tn = {k: (v.clone().detach().requires_grad_(True)
                  if k == "content" else v) for k, v in base_turn.items()}
        out = t._run_l1_batch([tn], target_ratio=0.2)
        out[0].sum().backward()
        return lin.weight.grad.detach().clone(), int(
            t._surv_state_l1.controllable_count_sum
        )

    ref_grad, ref_ctrl = run(0)     # checkpoint OFF
    got_grad, got_ctrl = run(2)     # checkpoint ON (seqlen 6 > 2)
    max_abs = (got_grad - ref_grad).abs().max().item()
    assert torch.allclose(got_grad, ref_grad, atol=1e-6, rtol=1e-5), (
        f"drill-checkpoint grad mismatch: max_abs={max_abs:.2e}"
    )
    assert ref_ctrl == 10 and got_ctrl == 10, (ref_ctrl, got_ctrl)
    print(f"[drill-ckpt] grad max_abs_diff={max_abs:.2e}  theta_ctrl={got_ctrl}")


def test_tree_checkpoint_token_gate_fires_on_low_node_big_leaf():
    """GATE (e, basis): the token gate activates the tree-encode checkpoint for
    a LOW-node repo with a big initial-import leaf (full-tree case), where the
    node gate alone would skip it."""
    dim, dec_dim = 8, 8
    tok = _FakeTokenizer()
    enc = _RecorderEncoder(dim, dec_dim)
    t = _make_per_repo_trainer(enc, tok, _commit_repro_tree(), dim, dec_dim)
    _stub_live_l0_per_repo(t, enc, dim)
    t._checkpoint_tree_encode = True
    # Low-node tree but a "big leaf": stub the counts so the NODE gate would
    # skip (4 <= 64) while the TOKEN gate fires (90k > 16384).
    t._repo_tree_node_count = types.MethodType(lambda self, ds, r: 4, t)
    t._repo_leaf_token_count = types.MethodType(lambda self, ds, r: 90_000, t)
    t._tree_checkpoint_min_nodes = 64
    t._tree_checkpoint_min_tokens = 16384

    captured = {}

    # Capture the per-tree decision mid-encode (set before _encode_subtree,
    # restored in finally). No-op encode → isolates the gate logic.
    def capturing_encode(self, ds, tree, root, q, memo, stats):
        captured["active"] = self._tree_encode_ckpt_active

    t._encode_subtree = types.MethodType(capturing_encode, t)
    t._compute_shared_repo_tree("toy", "repo/w000")

    assert captured["active"] is True, (
        "token gate must activate the per-node checkpoint for the big-leaf "
        "low-node (full-tree) repo"
    )


def test_ensure_eval_shared_tree_installs_reps_and_memoizes():
    """EVAL fix: _ensure_eval_shared_tree encodes the shared repo tree so drill
    reps resolve to REAL survivors (not _drilldown_zero_survivor). Verifies the
    memo/splice install, the forward-count is not inflated, memoization by root,
    the per_repo_full_backprop gate, and _clear teardown."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    t.device = torch.device("cpu")
    t._per_repo_full_backprop = True
    t._shared_tree_forward_count = 5
    t._eval_shared_tree_root = None
    t._shared_tree_memo = None
    t._shared_tree_splice_reps = None
    t._shared_tree_used_nodes = None
    t._shared_tree_child_l1_reps = {"stale": 1}
    t._shared_tree_child_l1_used = {"stale"}
    t._per_repo_shared_tree_active = False

    calls = {"n": 0}

    def fake_tree(self, ds, root):
        calls["n"] += 1
        proj = torch.randn(3, 8)
        l1 = torch.randn(3, 8)
        return {f"{root}/n1": (proj, l1), f"{root}/n2": (proj, l1)}, {"nodes": 2}

    t._repo_group_key = types.MethodType(lambda self, s: s.root, t)
    t._compute_shared_repo_tree = types.MethodType(fake_tree, t)

    s_a = SimpleNamespace(dataset_name="git_commit_repro", root="repo/w000")
    t._ensure_eval_shared_tree(s_a)

    # Installed real reps for both nodes; forward-count NOT inflated (stays 5).
    assert set(t._shared_tree_splice_reps.keys()) == {"repo/w000/n1", "repo/w000/n2"}
    assert t._shared_tree_memo is not None
    assert t._per_repo_shared_tree_active is True
    assert t._eval_shared_tree_root == "repo/w000"
    assert t._shared_tree_forward_count == 5
    # reaccumulate dicts nulled so the head reads memo[c][1] directly (eval path)
    assert t._shared_tree_child_l1_reps is None
    assert calls["n"] == 1

    # Same root -> memoized, no re-encode.
    t._ensure_eval_shared_tree(SimpleNamespace(dataset_name="git_commit_repro", root="repo/w000"))
    assert calls["n"] == 1

    # Different root -> re-encode.
    t._ensure_eval_shared_tree(SimpleNamespace(dataset_name="git_commit_repro", root="repo/w001"))
    assert calls["n"] == 2
    assert t._eval_shared_tree_root == "repo/w001"

    # Teardown clears everything.
    t._clear_eval_shared_tree()
    assert t._shared_tree_memo is None
    assert t._shared_tree_splice_reps is None
    assert t._per_repo_shared_tree_active is False
    assert t._eval_shared_tree_root is None


def test_ensure_eval_shared_tree_gated_off_when_not_per_repo():
    """Flat QA datasets (not per-repo full-backprop) must not touch shared-tree
    state — _run_l1_batch owns their splice reps."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    t.device = torch.device("cpu")
    t._per_repo_full_backprop = False
    t._eval_shared_tree_root = None
    t._shared_tree_memo = None
    t._shared_tree_splice_reps = None

    called = {"n": 0}
    t._repo_group_key = types.MethodType(lambda self, s: "root", t)
    t._compute_shared_repo_tree = types.MethodType(
        lambda self, ds, root: (called.update(n=called["n"] + 1) or ({}, {}), {}), t,
    )
    t._ensure_eval_shared_tree(SimpleNamespace(dataset_name="pubmedqa"))
    assert called["n"] == 0
    assert t._shared_tree_memo is None
    assert t._shared_tree_splice_reps is None
