"""optim.encoder_mode adapter (2026-09-22): the encoder trains again through ONE
adapter on every non-tree slot path, while the cached-tree path keeps running
the frozen base encoder bit-identically (so cached tree roots stay valid)."""

from __future__ import annotations

import pytest
import torch

from bgkit2.models.encoder import BgKIT2Encoder
from bgkit2.models.lfm2_bidirectional import Lfm2BidirectionalBackbone
from bgkit2.models.multilevel import encode_tree

from .test_data_and_training import _tiny_trainer
from .test_multi_slot import _plain_sample, _slots_sample


def _enc(tiny_lfm2_model, **kw) -> BgKIT2Encoder:
    torch.manual_seed(0)
    bb = Lfm2BidirectionalBackbone(tiny_lfm2_model)
    return BgKIT2Encoder(bb, head_layer=2, head_inner_dim=8, repro_head=True, **kw).eval()


def _perturb(enc: BgKIT2Encoder) -> None:
    """Move the adapter away from its zero delta: LoRA B and every part copy."""
    with torch.no_grad():
        for name, p in enc.named_parameters():
            if "lora_B.slots" in name or name.startswith("slot_parts."):
                p.add_(torch.randn_like(p) * 0.2)


def _tok(s: str) -> list[int]:
    return [3 + (ord(ch) % 40) for ch in s]


TREE = dict(tok_encode=_tok, ratio_leaf=0.5, ratio_node=0.5, max_depth=2, bridge=lambda x: x)


def _tree():
    g = torch.Generator().manual_seed(3)
    paths = ["a.py", "pkg/b.py", "pkg/c.py", "d/e.py"]
    return [torch.randint(3, 50, (20 + 4 * i,), generator=g) for i in range(4)], paths


def test_adapter_off_is_bit_identical_to_the_base_encoder(tiny_lfm2_model):
    """Role base after injecting AND perturbing the adapter = the encoder before
    the adapter existed, for slots and for whole trees (what the caches hold)."""
    enc = _enc(tiny_lfm2_model, projection_block=True, prompt_separator=True)
    ids = torch.randint(3, 50, (2, 24))
    mask = torch.ones(2, 24, dtype=torch.bool)
    p_ids = torch.randint(3, 50, (2, 5))
    p_mask = torch.ones(2, 5, dtype=torch.long)
    files, paths = _tree()
    with torch.no_grad():
        before = enc(ids, mask, ratio=0.25, prompt_ids=p_ids, prompt_mask=p_mask).reps
        tree_before = encode_tree(enc, files, paths, **TREE)
    n = enc.inject_slot_adapter(4, 8, layers=[3, 4, 5])
    assert n > 0 and enc.has_slot_adapter and enc.default_role == "slots"
    _perturb(enc)
    with torch.no_grad():
        with enc.role("base"):
            base = enc(ids, mask, ratio=0.25, prompt_ids=p_ids, prompt_mask=p_mask).reps
            tree_base = encode_tree(enc, files, paths, **TREE)
        slots = enc(ids, mask, ratio=0.25, prompt_ids=p_ids, prompt_mask=p_mask).reps
    assert torch.equal(before, base)
    assert torch.equal(tree_before.root.reps, tree_base.root.reps)
    for node, (r, _m) in tree_before.nodes.items():
        assert torch.equal(r, tree_base.nodes[node][0]), node
    assert not torch.allclose(before, slots)  # the default role runs the adapter
    # a tree on an adapter encoder must say which encoder it wants
    with pytest.raises(RuntimeError, match="explicit role"):
        encode_tree(enc, files, paths, **TREE)


def test_adapter_on_trains_and_the_base_takes_no_gradient(tmp_path):
    tr = _tiny_trainer(tmp_path, **{"optim.encoder_mode": "adapter",
                                    "model.encoder_adapter.rank": 4,
                                    "model.encoder_adapter.alpha": 8})
    names = [g["name"] for g in tr.param_groups()]
    assert "encoder_adapter" in names and "encoder_adapter_head" in names
    assert "encoder_adapter_projection" in names and "backbone" not in names
    # upper layers only by default (tiny backbone: 6 layers -> the last 4)
    assert tr.encoder_adapter_layers() == [2, 3, 4, 5]
    wrapped = {n.split(".layers.")[1].split(".")[0] for n, _ in tr.encoder.named_parameters()
               if "lora_A.slots" in n}
    assert wrapped == {"2", "3", "4", "5"}
    _perturb(tr.encoder)
    batch = tr.collate([_slots_sample(0), _plain_sample(1)])
    total, stats = tr.training_loss(batch)
    total.backward()
    trained = {n for n, p in tr.encoder.named_parameters()
               if p.grad is not None and float(p.grad.abs().sum()) > 0}
    assert any(".slots." in n for n in trained)
    assert any(n.startswith("slot_parts.head") for n in trained)
    assert any(n.startswith("slot_parts.projection") for n in trained)
    bad = [n for n in trained if not (n.startswith("slot_parts.") or ".slots." in n)]
    assert bad == []  # nothing the base role (tree cache path) reads moves
    assert "rep/eff_rank_within" in stats and "rep/shared_frac" in stats  # the guard runs


def test_freeze_encoder_alias_keeps_the_old_param_groups(tmp_path):
    from bgkit2.training.config import Config, _validate, encoder_mode

    # (the CPU host's decoder has no adapter: the spine adapter is what trains,
    # exactly the pre-change frozen-mode groups)
    tr = _tiny_trainer(tmp_path, **{"optim.freeze_encoder": "true",
                                    "compaction_spine.enabled": "true"})
    assert encoder_mode(tr.cfg2) == "frozen"
    assert [g["name"] for g in tr.param_groups()] == ["spine"]
    assert not tr.encoder.has_slot_adapter and tr.encoder.has_spine_adapter
    spine = {id(p) for p in tr.encoder.spine_parameters()}
    assert all(not p.requires_grad for p in tr.encoder.parameters() if id(p) not in spine)
    with pytest.raises(ValueError, match="nothing to train"):
        _tiny_trainer(tmp_path / "b", **{"optim.freeze_encoder": "true"})
    c = Config()
    assert encoder_mode(c) == "backbone"
    c.optim.freeze_encoder = True
    c.optim.encoder_mode = "adapter"
    with pytest.raises(ValueError, match="contradicts"):
        _validate(c)


def test_tree_cache_loads_in_adapter_mode_but_not_under_an_open_backbone(tmp_path):
    from safetensors.torch import save_file

    d = 32
    f = str(tmp_path / "roots.safetensors")
    save_file({"own/a/reps": torch.randn(3, d), "own/a/mask": torch.ones(3, dtype=torch.bool)}, f)
    tr = _tiny_trainer(tmp_path, **{"optim.encoder_mode": "adapter"})
    tr.load_tree_root_cache(f)
    item = type("T", (), {"repo": "own/a", "node": "", "paths": ["x.py"], "file_ids": []})()
    r, m = tr.tree_root(item)
    assert r.shape == (3, d) and bool(m.all())
    tr2 = _tiny_trainer(tmp_path / "b", **{"optim.encoder_mode": "backbone"})
    with pytest.raises(ValueError, match="frozen"):
        tr2.load_tree_root_cache(f)


def test_live_small_trees_train_large_ones_take_the_base_path(tmp_path):
    tr = _tiny_trainer(tmp_path, **{"optim.encoder_mode": "adapter", "model.repro_head": "true",
                                    "multilevel.live_tree_max_files": 3,
                                    "multilevel.grad_levels": 1, "multilevel.bridge": "identity",
                                    "multilevel.ratio_leaf": 0.5, "multilevel.ratio_node": 0.5,
                                    "multilevel.max_depth": 2})
    tr.enc_tok = lambda s, add_special_tokens=False: {"input_ids": _tok(s)}
    files, paths = _tree()
    small = type("T", (), {"repo": "r", "node": "", "paths": paths[:3], "file_ids": files[:3]})()
    large = type("T", (), {"repo": "r", "node": "", "paths": paths, "file_ids": files})()
    assert tr.tree_role(3) == "slots" and tr.tree_role(4) == "base"
    tr.encoder.train()
    r, _ = tr.tree_root(small)
    assert r.requires_grad  # upper levels keep a gradient
    r.float().sum().backward()
    assert any(p.grad is not None for p in tr.encoder.slot_adapter_parameters())
    r_large, _ = tr.tree_root(large)
    assert not r_large.requires_grad  # frozen base path, no graph
    with torch.no_grad(), tr.encoder.role("base"):
        ref = encode_tree(tr.encoder, files, paths, **{**TREE, "max_depth": 2})
    assert torch.allclose(r_large.float(), ref.root.reps[0].float())


def test_student_scopes_the_adapter_like_the_trainer(tmp_path):
    """The eval server asks the trainer's rule: small trees through the adapter,
    large ones on the base path; text slots through the adapter."""
    from bgkit2.agent.student import StudentModel

    tr = _tiny_trainer(tmp_path, **{"optim.encoder_mode": "adapter", "model.repro_head": "true",
                                    "multilevel.live_tree_max_files": 3,
                                    "multilevel.bridge": "identity",
                                    "multilevel.ratio_leaf": 0.5, "multilevel.ratio_node": 0.5,
                                    "multilevel.max_depth": 2})
    _perturb(tr.encoder)
    tr.encoder.eval()
    tr.encoder.repro = torch.nn.Identity()  # (the tiny host builds none; the bridge is identity)
    tr.enc_tok = lambda s, add_special_tokens=False: {"input_ids": _tok(s)}
    student = StudentModel.__new__(StudentModel)
    student.core, student.encoder, student.device = tr, tr.encoder, torch.device("cpu")
    student._enc_tok = tr.enc_tok
    for n in (3, 4):
        srcs = [f"def f():\n    return {i}\n" for i in range(n)]
        paths = ["a.py", "pkg/b.py", "pkg/c.py", "d/e.py"][:n]
        payload = {"files": [{"path": p, "source": s} for p, s in zip(paths, srcs, strict=True)]}
        r_eval, _m, _n = student._encode_tree_slot(payload, 8)
        item = type("T", (), {"repo": "r", "node": "", "paths": paths,
                              "file_ids": [torch.tensor(_tok(s)) for s in srcs]})()
        with torch.no_grad():
            r_train, _ = tr.tree_root(item)
        assert torch.allclose(r_eval.float(), r_train.float(), atol=1e-5), n
        with torch.no_grad(), tr.encoder.role("base"):
            base = encode_tree(tr.encoder, item.file_ids, paths, **{**TREE, "max_depth": 2})
        same_as_base = torch.allclose(r_eval.float(), base.root.reps[0].float(), atol=1e-5)
        assert same_as_base == (n == 4), n  # only the large tree runs the frozen base


def test_adapter_checkpoint_round_trip_and_resume_from_a_frozen_checkpoint(tiny_lfm2_model):
    """A frozen-run checkpoint (no adapter) loads into an adapter encoder with
    the base weights in place (peft's base_layer rename) and the part copies
    seeded from the loaded base; the adapter's own tensors round-trip."""
    from bgkit2.training.core import BgKIT2Core

    plain = _enc(tiny_lfm2_model)
    old = {k: v.clone() for k, v in plain.state_dict().items()}
    ids = torch.randint(3, 50, (1, 16))
    mask = torch.ones(1, 16, dtype=torch.bool)
    with torch.no_grad():
        ref = plain(ids, mask, ratio=0.5).reps
    enc = _enc(tiny_lfm2_model)
    with torch.no_grad():
        for p in enc.parameters():
            p.add_(0.5)  # a different starting point: the load must overwrite it
    enc.inject_slot_adapter(4, 8)
    tolerated = BgKIT2Core.load_encoder_state(enc, old)
    assert any(".slots." in k for k in tolerated)
    with torch.no_grad():
        assert torch.equal(enc.slot_parts.projection[1].weight, plain.projection[1].weight)
        assert torch.allclose(enc(ids, mask, ratio=0.5).reps, ref)  # zero LoRA delta
        with enc.role("base"):
            assert torch.equal(enc(ids, mask, ratio=0.5).reps, ref)
    _perturb(enc)
    saved = {k: v.clone() for k, v in enc.state_dict().items()}
    enc2 = _enc(tiny_lfm2_model)
    enc2.inject_slot_adapter(4, 8)
    assert BgKIT2Core.load_encoder_state(enc2, saved) == []
    for k, v in enc2.state_dict().items():
        assert torch.equal(v, saved[k]), k
    with pytest.warns(UserWarning, match="encoder-adapter"):
        BgKIT2Core.load_encoder_state(_enc(tiny_lfm2_model), saved)


def test_gradient_budget_samples_the_gradient_not_the_encoding(tmp_path):
    """optim.encoder_grad_token_budget: primary slots take gradient first, the
    extra slots share the rest (most recent first); the others are encoded
    under no_grad -- every slot is still encoded -- and the split is logged."""
    tr = _tiny_trainer(tmp_path, **{"optim.encoder_mode": "adapter",
                                    "optim.encoder_grad_token_budget": 40})
    tr.encoder.train()
    batch = tr.collate([_slots_sample(0, 3), _slots_sample(1, 3)]).to(tr.device)
    tr._enc_grad_reset()
    tr._account_primary(batch)  # 2 rows x (6 ctx + 2 prompt) = 16 tokens
    plan = tr.plan_extra_grad(batch)
    # each extra slot is 7 + 1 prompt = 8 tokens: 24 left -> 3 slots, the most
    # recent of each row first, then the next most recent of the first row drawn
    owners = batch.extra_owner.tolist()
    assert len(plan) == 3
    last = {o: max(i for i, w in enumerate(owners) if w == o) for o in set(owners)}
    assert set(last.values()) <= plan
    per = tr.encode_extra(batch, 0.5)
    grads = [r.requires_grad for row in per for r, _m in row]
    assert sum(grads) == 3 and len(grads) == 6
    total, stats = tr.training_loss(tr.collate([_slots_sample(0, 3), _slots_sample(1, 3)]))
    assert stats["enc_grad/slots"] == 8.0 and stats["enc_grad/grad_slots"] == 5.0
    assert stats["enc_grad/grad_tokens"] <= 40 < stats["enc_grad/slot_tokens"]
    total.backward()
    with torch.no_grad():  # eval: no plan, everything encodes as before
        assert tr.plan_extra_grad(batch) is None


def test_adapter_composes_with_encoder_gradient_checkpointing(tmp_path):
    tr = _tiny_trainer(tmp_path, **{"optim.encoder_mode": "adapter",
                                    "optim.encoder_grad_token_budget": 24})
    tr.encoder.backbone.gradient_checkpointing_enable()
    tr.encoder.train()
    _perturb(tr.encoder)
    total, stats = tr.training_loss(tr.collate([_slots_sample(0, 3), _slots_sample(1, 3)]))
    total.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in tr.encoder.slot_adapter_parameters())
    assert stats["enc_grad/grad_slots"] < stats["enc_grad/slots"]
    # a role switch with gradients on under checkpointing is refused, not silently wrong
    with pytest.raises(RuntimeError, match="gradient checkpointing"), tr.encoder.role("base"):
        pass


def test_student_tree_slot_uses_the_trainers_leaf_settings(tmp_path, monkeypatch):
    """2026-09-22 audit: the student cut every file to 512 tokens and dropped the leaf window /
    leaf cap / chunk budgets. It must hand encode_tree exactly what the trainer does, from the
    run's own config, and cut files where the trainer's tree dataset does."""
    import bgkit2.models.multilevel as ml
    from bgkit2.agent.student import StudentModel

    tr = _tiny_trainer(tmp_path, **{"optim.encoder_mode": "adapter", "model.repro_head": "true",
                                    "multilevel.bridge": "identity",
                                    "multilevel.ratio_leaf": 0.5, "multilevel.ratio_node": 0.5,
                                    "multilevel.max_depth": 2, "multilevel.leaf_window": 256,
                                    "multilevel.leaf_max_reps": 40, "multilevel.leaf_chunk": 5,
                                    "multilevel.leaf_chunk_tokens": 4096,
                                    "multilevel.node_max_tokens": 777,
                                    "data.repo_tree_file_tokens": 1500})
    tr.encoder.eval()
    tr.encoder.repro = torch.nn.Identity()  # (the tiny host builds none; the bridge is identity)
    tr.enc_tok = lambda s, add_special_tokens=False: {"input_ids": _tok(s)}
    seen: list[tuple[list[int], dict]] = []
    real = ml.encode_tree

    def spy(enc, files, paths, **kw):
        seen.append(([int(f.shape[0]) for f in files], dict(kw)))
        return real(enc, files, paths, **kw)

    monkeypatch.setattr(ml, "encode_tree", spy)
    student = StudentModel.__new__(StudentModel)
    student.core, student.encoder, student.device = tr, tr.encoder, torch.device("cpu")
    student._enc_tok = tr.enc_tok
    srcs = ["x = 1\n" * 400, "def f():\n    return 2\n"]  # 2400 tokens: past 512, past 1500
    paths = ["a.py", "pkg/b.py"]
    student._encode_tree_slot(
        {"files": [{"path": p, "source": s} for p, s in zip(paths, srcs, strict=True)]}, 8)
    cap = tr.cfg2.data.repo_tree_file_tokens
    item = type("T", (), {"repo": "r", "node": "", "paths": paths,
                          "file_ids": [torch.tensor(_tok(s)[:cap]) for s in srcs]})()
    with torch.no_grad():
        tr.tree_root(item)
    (s_lens, s_kw), (t_lens, t_kw) = seen
    assert s_lens == t_lens == [1500, len(_tok(srcs[1]))]  # the run's cap, not 512
    drop = {"grad_levels", "bridge", "tok_encode"}  # no_grad either way; fresh lambdas per call
    assert {k: v for k, v in s_kw.items() if k not in drop} == {
        k: v for k, v in t_kw.items() if k not in drop}
    assert s_kw["leaf_window"] == 256 and s_kw["leaf_max_reps"] == 40
    assert s_kw["node_max_tokens"] == 777 and s_kw["leaf_chunk"] == 5
