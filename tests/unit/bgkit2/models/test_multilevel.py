"""The multi-level (repo-tree) encoder path.

A level consumes the level below through `reproduce` -> `forward(doc_embeds=)`.
These tests pin the CONTRACT, not the weights: the bridge is untrained until the
joint-block distillation runs, so only shapes, spaces and wiring are asserted.
"""

from __future__ import annotations

import pytest
import torch

from bgkit2.models.encoder import BgKIT2Encoder
from bgkit2.models.lfm2_bidirectional import Lfm2BidirectionalBackbone

HIDDEN = 32


def _enc(tiny_lfm2_model, **kw):
    bb = Lfm2BidirectionalBackbone(tiny_lfm2_model)
    return BgKIT2Encoder(bb, head_layer=2, head_inner_dim=8, **kw).eval()


def test_survivors_are_exposed_before_the_decoder_projection(tiny_lfm2_model):
    enc = _enc(tiny_lfm2_model)
    ids = torch.randint(0, 50, (2, 32))
    out = enc(ids, torch.ones(2, 32, dtype=torch.bool), ratio=0.25)
    assert out.survivors is not None
    # encoder hidden space, not decoder space: reps went through projection
    assert out.survivors.shape == out.reps.shape
    assert not torch.allclose(out.survivors, out.reps)


def test_reproduce_requires_the_head(tiny_lfm2_model):
    enc = _enc(tiny_lfm2_model)
    with pytest.raises(RuntimeError, match="repro_head"):
        enc.reproduce(torch.zeros(1, 2, HIDDEN))


def test_a_level_consumes_the_level_below(tiny_lfm2_model):
    """file -> survivors -> bridge -> directory encode, the repo-tree step."""
    enc = _enc(tiny_lfm2_model, repro_head=True)
    ids = torch.randint(0, 50, (2, 64))
    mask = torch.ones(2, 64, dtype=torch.bool)
    with torch.no_grad():
        l0 = enc(ids, mask, ratio=0.25)
        bridged = enc.reproduce(l0.survivors)
        k = bridged.shape[1]
        l1 = enc(
            None,
            l0.rep_mask.reshape(1, 2 * k),
            ratio=0.25,
            doc_embeds=bridged.reshape(1, 2 * k, bridged.shape[-1]),
        )
    assert l1.survivors.shape[1] < l0.survivors.shape[1] * 2  # compressed again
    assert l1.reps.shape[-1] == HIDDEN


def test_doc_embeds_must_match_the_mask(tiny_lfm2_model):
    enc = _enc(tiny_lfm2_model, repro_head=True)
    with pytest.raises(ValueError, match="does not match"):
        enc(None, torch.ones(1, 8, dtype=torch.bool), ratio=0.5,
            doc_embeds=torch.zeros(1, 9, HIDDEN))


def test_forward_still_needs_one_of_ids_or_embeds(tiny_lfm2_model):
    enc = _enc(tiny_lfm2_model)
    with pytest.raises(ValueError, match="input_ids or doc_embeds"):
        enc(None, torch.ones(1, 4, dtype=torch.bool), ratio=0.5)


def test_repro_loss_is_off_without_the_head_and_trains_it_when_on(tiny_lfm2_model):
    """Stage A objective: the repro head must move toward the ENCODER's input
    embeddings, which is a different target space from the decoder anchor."""
    import torch.nn.functional as F

    enc = _enc(tiny_lfm2_model, repro_head=True)
    ids = torch.randint(0, 50, (2, 32))
    out = enc(ids, torch.ones(2, 32, dtype=torch.bool), ratio=0.5)
    target = tiny_lfm2_model.embed_tokens(ids.gather(1, out.positions)).detach()
    opt = torch.optim.SGD(enc.repro.parameters(), lr=0.5)
    first = last = None
    for _ in range(20):
        pred = enc.reproduce(out.survivors.detach())
        loss = F.mse_loss(pred[out.rep_mask].float(), target[out.rep_mask].float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss)
        last = float(loss)
    assert last < first  # the head is learnable against the encoder-input target


def test_forced_keep_overrides_selection(tiny_lfm2_model):
    """Distillation compares a student and a frozen teacher position by
    position, which is only meaningful if both selected the same tokens."""
    enc = _enc(tiny_lfm2_model)
    ids = torch.randint(0, 50, (1, 16))
    mask = torch.ones(1, 16, dtype=torch.bool)
    forced = torch.zeros(1, 16, dtype=torch.bool)
    forced[0, [2, 5, 11]] = True
    out = enc(ids, mask, ratio=0.9, forced_keep=forced)
    assert out.k.tolist() == [3]
    assert sorted(out.positions[0].tolist()) == [2, 5, 11]


def test_forced_keep_is_intersected_with_eligibility(tiny_lfm2_model):
    """A forced position that is padding must not be resurrected."""
    enc = _enc(tiny_lfm2_model)
    ids = torch.randint(0, 50, (1, 16))
    mask = torch.zeros(1, 16, dtype=torch.bool)
    mask[0, :8] = True                      # second half is padding
    forced = torch.zeros(1, 16, dtype=torch.bool)
    forced[0, [1, 12]] = True               # 12 is padding
    out = enc(ids, mask, ratio=0.9, forced_keep=forced)
    assert out.k.tolist() == [1] and out.positions[0, 0].item() == 1


# ----------------------------------------------------------------------------
# The reusable two-level path (models/multilevel.py)
# ----------------------------------------------------------------------------


def test_encode_two_level_groups_items_into_one_document(tiny_lfm2_model):
    from bgkit2.models.multilevel import encode_two_level

    enc = _enc(tiny_lfm2_model, repro_head=True)
    ids = torch.randint(0, 50, (4, 32))
    mask = torch.ones(4, 32, dtype=torch.bool)
    with torch.no_grad():
        out = encode_two_level(enc, ids, mask, ratio_lower=0.5, ratio_upper=0.5,
                               group_sizes=[2, 2])
    assert out.lower.reps.shape[0] == 4      # four "files"
    assert out.top.reps.shape[0] == 2        # two "directories"
    assert out.top.k.sum() < out.lower.k.sum()  # compressed again


def test_group_sizes_must_cover_every_item(tiny_lfm2_model):
    from bgkit2.models.multilevel import encode_two_level

    enc = _enc(tiny_lfm2_model, repro_head=True)
    with pytest.raises(ValueError, match="group_sizes"):
        encode_two_level(enc, torch.randint(0, 50, (3, 8)),
                         torch.ones(3, 8, dtype=torch.bool),
                         ratio_lower=0.5, ratio_upper=0.5, group_sizes=[2])


def test_forced_upper_reproduces_the_teachers_survivor_set(tiny_lfm2_model):
    """The distillation contract: with the lower level keeping the teacher's
    survivors plus extras, the upper level forced from the teacher mask must
    end up with exactly the teacher's tokens."""
    from bgkit2.models.multilevel import encode_two_level, teacher_forced_lower_keep

    enc = _enc(tiny_lfm2_model, repro_head=True)
    ids = torch.randint(0, 50, (2, 32))
    mask = torch.ones(2, 32, dtype=torch.bool)
    teacher = torch.zeros(2, 32, dtype=torch.bool)
    teacher[0, [1, 4, 9]] = True
    teacher[1, [0, 7]] = True
    lower_keep = teacher_forced_lower_keep(teacher, mask, frac_extras=0.5)
    assert (lower_keep & teacher).sum() == teacher.sum()      # never drops one
    assert lower_keep.sum() > teacher.sum()                   # and adds extras
    with torch.no_grad():
        out = encode_two_level(
            enc, ids, mask, ratio_lower=1.0, ratio_upper=1.0,
            group_sizes=[2], forced_lower_keep=lower_keep, forced_upper_from=teacher,
        )
    assert int(out.top.k.sum()) == int(teacher.sum())


def test_distill_loss_is_zero_at_agreement_and_positive_otherwise(tiny_lfm2_model):
    from bgkit2.models.multilevel import distill_loss

    reps = torch.randn(2, 5, 8)
    m = torch.ones(2, 5, dtype=torch.bool)
    same, stats = distill_loss(reps, reps.clone(), m)
    assert float(same) < 1e-6 and stats["cos"] > 0.999
    diff, stats2 = distill_loss(reps, torch.randn(2, 5, 8), m)
    assert float(diff) > float(same) and stats2["cos"] < stats["cos"]


def test_distill_loss_ignores_padded_rows(tiny_lfm2_model):
    from bgkit2.models.multilevel import distill_loss

    s = torch.randn(1, 4, 8)
    t = s.clone()
    t[0, 3] = 99.0  # a padded row disagreeing wildly
    m = torch.tensor([[True, True, True, False]])
    loss, _ = distill_loss(s, t, m)
    assert float(loss) < 1e-6


# ----------------------------------------------------------------------------
# Recursive tree encoding (build_tree / encode_tree)
# ----------------------------------------------------------------------------


def test_build_tree_mirrors_directories_and_caps_depth():
    from bgkit2.models.multilevel import build_tree

    paths = ["a.py", "pkg/b.py", "pkg/sub/c.py", "pkg/sub/deep/d.py"]
    t3 = build_tree(paths, max_depth=3)          # root -> pkg -> sub (deep folded into sub)
    assert t3.files == [0]
    assert list(t3.children) == ["pkg"]
    assert t3.children["pkg"].files == [1]
    assert list(t3.children["pkg"].children) == ["sub"]
    assert sorted(t3.children["pkg"].children["sub"].files) == [2, 3]
    t1 = build_tree(paths, max_depth=1)          # flat: everything under the root
    assert sorted(t1.files) == [0, 1, 2, 3] and not t1.children


def test_encode_tree_runs_recursively_and_returns_root_reps(tiny_lfm2_model):
    from bgkit2.models.multilevel import encode_tree

    enc = _enc(tiny_lfm2_model, repro_head=True)
    paths = ["a.py", "pkg/b.py", "pkg/c.py", "pkg/sub/d.py"]
    files = [torch.randint(3, 50, (24,)) for _ in paths]
    tok_encode = lambda s: [3 + (ord(ch) % 40) for ch in s]  # noqa: E731 - any int ids
    with torch.no_grad():
        out = encode_tree(enc, files, paths, tok_encode=tok_encode,
                          ratio_leaf=0.5, ratio_node=0.5, max_depth=3)
    assert out.leaf_mask.shape[0] == 4 and out.leaf_reps.shape[0] == 4
    assert out.root.reps.shape[0] == 1                 # one repo -> one root
    assert out.depth == 3 and out.n_nodes == 3         # root, pkg, pkg/sub
    assert out.root.reps.shape[-1] == HIDDEN


def test_encode_tree_depth_one_is_a_single_upper_pass(tiny_lfm2_model):
    from bgkit2.models.multilevel import encode_tree

    enc = _enc(tiny_lfm2_model, repro_head=True)
    paths = ["a.py", "pkg/b.py", "pkg/sub/c.py"]
    files = [torch.randint(3, 50, (16,)) for _ in paths]
    with torch.no_grad():
        out = encode_tree(enc, files, paths, tok_encode=lambda s: [5] * 3,
                          ratio_leaf=0.5, ratio_node=0.5, max_depth=1)
    assert out.depth == 1 and out.n_nodes == 1


def test_encode_tree_leaf_chunks_match_and_grad_levels_bound_the_graph(tiny_lfm2_model):
    """Whole-repo trees: files go through the encoder in chunks (same root as
    one batch), and grad_levels keeps the graph to the top levels -- 0 builds
    none (frozen Stage D), 1 leaves the leaves and directories out."""
    from bgkit2.models.multilevel import encode_tree

    enc = _enc(tiny_lfm2_model, repro_head=True)
    enc.eval()  # deterministic selection for the equality check
    paths = ["a.py", "pkg/b.py", "pkg/c.py", "pkg/sub/d.py", "pkg/sub/e.py"]
    files = [torch.randint(3, 50, (20,)) for _ in paths]
    tok_encode = lambda s: [3 + (ord(ch) % 40) for ch in s]  # noqa: E731
    kw = dict(tok_encode=tok_encode, ratio_leaf=0.5, ratio_node=0.5, max_depth=3)
    with torch.no_grad():
        one = encode_tree(enc, files, paths, leaf_chunk=99, **kw)
        chunked = encode_tree(enc, files, paths, leaf_chunk=2, **kw)
    assert torch.allclose(one.root.reps, chunked.root.reps, atol=1e-4)
    assert torch.equal(one.leaf_mask, chunked.leaf_mask)
    # grad_levels=0: nothing requires grad (a frozen encoder stores no activations)
    out0 = encode_tree(enc, files, paths, grad_levels=0, **kw)
    assert not out0.root.reps.requires_grad
    # grad_levels=1: the root pass has a graph, the leaves do not
    out1 = encode_tree(enc, files, paths, grad_levels=1, **kw)
    assert out1.root.reps.requires_grad and not out1.leaf_reps.requires_grad
    # default: the whole tree is differentiable
    outall = encode_tree(enc, files, paths, **kw)
    assert outall.leaf_reps.requires_grad


def test_encode_tree_leaf_calls_respect_the_token_budget(tiny_lfm2_model):
    """No leaf encoder call exceeds leaf_chunk_tokens padded tokens (files x
    longest), whatever the file count cap says; the root is unchanged."""
    from bgkit2.models.multilevel import encode_tree

    enc = _enc(tiny_lfm2_model, repro_head=True)
    enc.eval()
    paths = [f"pkg/f{i}.py" for i in range(7)]
    files = [torch.randint(3, 50, (n,)) for n in (4, 30, 6, 28, 5, 8, 29)]
    tok_encode = lambda s: [3 + (ord(ch) % 40) for ch in s]  # noqa: E731
    kw = dict(tok_encode=tok_encode, ratio_leaf=0.5, ratio_node=0.5, max_depth=2)
    calls = []
    orig = enc.forward

    def spy(ids, mask, **k):
        if ids is not None:
            calls.append(tuple(ids.shape))
        return orig(ids, mask, **k)

    enc.forward = spy
    with torch.no_grad():
        budget = encode_tree(enc, files, paths, leaf_chunk=99, leaf_chunk_tokens=100, **kw)
    assert len(calls) > 1 and all(b * w <= 100 or b == 1 for b, w in calls)
    calls.clear()
    with torch.no_grad():
        one = encode_tree(enc, files, paths, leaf_chunk=99, leaf_chunk_tokens=10**9, **kw)
    assert len(calls) == 1
    assert torch.allclose(one.root.reps, budget.root.reps, atol=1e-4)


def test_wide_directory_is_split_into_virtual_subdirectories(tiny_lfm2_model):
    """A flat directory's input is the sum of its children's survivors --
    unbounded in real repos (337 files -> one 54k-token sequence -> a 39 GiB
    attention allocation, the 2026-09-17 host OOM). Above node_max_tokens the
    children are grouped into virtual sub-directories, recursively, and no
    directory-level encoder call sees more than the budget (+ its header)."""
    from bgkit2.models.multilevel import encode_tree

    enc = _enc(tiny_lfm2_model, repro_head=True)
    enc.eval()
    paths = [f"flat/f{i:02d}.py" for i in range(14)]
    files = [torch.randint(3, 50, (24,)) for _ in paths]
    tok_encode = lambda s: [3 + (ord(ch) % 40) for ch in s]  # noqa: E731
    kw = dict(tok_encode=tok_encode, ratio_leaf=0.5, ratio_node=0.5, max_depth=3)
    widths = []
    orig = enc.forward

    def spy(ids, mask, **k):
        if ids is None:  # directory-level call (doc_embeds)
            widths.append(int(mask.sum(-1).max()))
        return orig(ids, mask, **k)

    enc.forward = spy
    with torch.no_grad():
        out = encode_tree(enc, files, paths, node_max_tokens=64, **kw)
    hdr = len(tok_encode("# flat\n"))
    assert out.n_virtual > 0 and max(widths) <= 64 + hdr
    assert torch.isfinite(out.root.reps).all() and int(out.root.rep_mask.sum()) > 0
    widths.clear()
    with torch.no_grad():
        big = encode_tree(enc, files, paths, node_max_tokens=10**6, **kw)
    assert big.n_virtual == 0 and max(widths) > 64 + hdr


def test_encode_tree_returns_every_directory_node(tiny_lfm2_model):
    """The path walk reads the tree level by level: every REAL directory's
    decoder-space reps come back keyed by path ("" = root); virtual
    sub-directories stay internal."""
    from bgkit2.models.multilevel import encode_tree

    enc = _enc(tiny_lfm2_model, repro_head=True)
    enc.eval()
    paths = ["a.py", "pkg/b.py", "pkg/c.py", "pkg/sub/d.py", "tests/t.py"]
    files = [torch.randint(3, 50, (20,)) for _ in paths]
    tok_encode = lambda s: [3 + (ord(ch) % 40) for ch in s]  # noqa: E731
    with torch.no_grad():
        out = encode_tree(enc, files, paths, tok_encode=tok_encode, ratio_leaf=0.5,
                          ratio_node=0.5, max_depth=3, node_max_tokens=64)
    assert set(out.nodes) == {"", "pkg", "pkg/sub", "tests"}
    r, m = out.nodes[""]
    assert torch.equal(r, out.root.reps[0]) and torch.equal(m, out.root.rep_mask[0])
    assert all(int(mk.sum()) > 0 and rp.shape[-1] == HIDDEN for rp, mk in out.nodes.values())


def test_walk_steps_follow_the_encoded_tree_including_the_depth_fold():
    from bgkit2.models.multilevel import node_children, walk_steps

    paths = ["a.py", "pkg/b.py", "pkg/sub/d.py", "pkg/sub/deep/e.py"]
    # depth 3: root -> pkg -> pkg/sub; pkg/sub/deep is FOLDED into pkg/sub
    assert walk_steps(paths, "a.py", 3) == [("", "a.py")]
    assert walk_steps(paths, "pkg/b.py", 3) == [("", "pkg/"), ("pkg", "b.py")]
    assert walk_steps(paths, "pkg/sub/deep/e.py", 3) == [
        ("", "pkg/"), ("pkg", "sub/"), ("pkg/sub", "deep/e.py")]
    # depth 1: everything hangs off the root
    assert walk_steps(paths, "pkg/sub/d.py", 1) == [("", "pkg/sub/d.py")]
    kids = node_children(paths, 3)
    assert kids[""] == {"a.py", "pkg/"} and kids["pkg"] == {"b.py", "sub/"}
    assert kids["pkg/sub"] == {"d.py", "deep/e.py"}
    # every step's target is a legal child of its node
    for ans in paths:
        assert all(t in kids[n] for n, t in walk_steps(paths, ans, 3))


def test_a_repo_variant_reuses_the_leaves_of_the_full_pass(tiny_lfm2_model):
    """Variants (a random subset of the files) re-run only the directory levels
    on the full pass's leaves: same result as encoding the subset from scratch,
    and no file goes through the encoder again."""
    from bgkit2.models.multilevel import encode_tree

    enc = _enc(tiny_lfm2_model, repro_head=True)
    enc.eval()
    paths = ["a.py", "pkg/b.py", "pkg/c.py", "pkg/sub/d.py", "tests/t.py"]
    files = [torch.randint(3, 50, (20,)) for _ in paths]
    tok_encode = lambda s: [3 + (ord(ch) % 40) for ch in s]  # noqa: E731
    kw = dict(tok_encode=tok_encode, ratio_leaf=0.5, ratio_node=0.5, max_depth=3)
    with torch.no_grad():
        full = encode_tree(enc, files, paths, **kw)
        keep = [0, 2, 3]
        scratch = encode_tree(enc, [files[i] for i in keep], [paths[i] for i in keep], **kw)
        calls = []
        orig = enc.forward

        def spy(ids, mask, **k):
            calls.append(ids is not None)
            return orig(ids, mask, **k)

        enc.forward = spy
        variant = encode_tree(enc, [files[i] for i in keep], [paths[i] for i in keep],
                              leaves=(full.leaf_reps[keep], full.leaf_mask[keep]), **kw)
    assert not any(calls)                       # directory passes only (doc_embeds)
    assert set(variant.nodes) == {"", "pkg", "pkg/sub"}
    assert torch.allclose(variant.root.reps, scratch.root.reps, atol=1e-4)


def test_whole_files_are_windowed_and_long_files_compress_harder(tiny_lfm2_model):
    """leaf_window: a long file is encoded in windows (none wider than the
    window + header) and its windows' survivors form ONE leaf; leaf_max_reps
    bounds the leaf by raising the compression, not by cutting the file; with
    both off the leaves are the original ones."""
    from bgkit2.models.multilevel import encode_tree

    enc = _enc(tiny_lfm2_model, repro_head=True)
    enc.eval()
    paths = ["a.py", "pkg/long.py", "pkg/c.py"]
    files = [torch.randint(3, 50, (n,)) for n in (12, 100, 20)]
    tok_encode = lambda s: [3 + (ord(ch) % 40) for ch in s]  # noqa: E731
    hdr = max(len(tok_encode(f"# {p}\n")) for p in paths) + 8
    kw = dict(tok_encode=tok_encode, ratio_leaf=0.5, ratio_node=0.5, max_depth=2)
    calls = []
    orig = enc.forward

    def spy(ids, mask, **k):
        if ids is not None:
            calls.append(tuple(ids.shape))
        return orig(ids, mask, **k)

    enc.forward = spy
    with torch.no_grad():
        base = encode_tree(enc, files, paths, **kw)
        calls.clear()
        win = encode_tree(enc, files, paths, leaf_window=32, **kw)
        assert all(w <= 32 + hdr for _b, w in calls)
        capped = encode_tree(enc, files, paths, leaf_window=32, leaf_max_reps=16, **kw)
    assert win.leaf_mask.shape[0] == 3                       # one leaf per FILE
    n_base, n_win, n_cap = (int(o.leaf_mask[1].sum()) for o in (base, win, capped))
    assert n_win >= n_base - 2                                # nothing dropped by windowing
    assert n_cap <= 16 + 4 < n_win                            # the cap compresses harder
    assert int(capped.leaf_mask[0].sum()) == int(base.leaf_mask[0].sum())  # short file as before
    assert capped.root.reps.shape[0] == 1


def test_a_wide_level_is_encoded_in_budgeted_chunks(tiny_lfm2_model):
    """Many directories at one depth (large repos) never go through the encoder
    as one padded batch; the root is the same either way."""
    from bgkit2.models.multilevel import encode_tree

    enc = _enc(tiny_lfm2_model, repro_head=True)
    enc.eval()
    paths = [f"d{i}/f{j}.py" for i in range(9) for j in range(2)]
    files = [torch.randint(3, 50, (10 + (i % 5),)) for i in range(len(paths))]
    tok_encode = lambda s: [3 + (ord(ch) % 40) for ch in s]  # noqa: E731
    kw = dict(tok_encode=tok_encode, ratio_leaf=0.5, ratio_node=0.5, max_depth=2)
    calls = []
    orig = enc.forward

    def spy(ids, mask, **k):
        if ids is None:
            calls.append(tuple(mask.shape))
        return orig(ids, mask, **k)

    enc.forward = spy
    with torch.no_grad():
        one = encode_tree(enc, files, paths, leaf_chunk_tokens=10**9, node_max_tokens=10**9, **kw)
        n_one = len(calls)
        calls.clear()
        many = encode_tree(enc, files, paths, leaf_chunk_tokens=64, node_max_tokens=64, **kw)
    assert n_one == 2 and len(calls) > 2                   # level of 9 dirs + root -> chunks
    assert set(one.nodes) == set(many.nodes) and len(one.nodes) == 10
