"""bgkit_core.segments / bgkit_core.selection: the shared segment math."""

from __future__ import annotations

import torch

from bgkit_core.segments import Segments, pack, segment_zscore, unpack
from bgkit_core.selection import (
    exact_topk,
    head_scale_anchor,
    listwise_kl,
    quota,
    teacher_concentration,
    utility_targets,
    zscore_padded,
)


def test_padded_and_flat_zscores_agree_and_pack_unpack_round_trips():
    torch.manual_seed(0)
    x = torch.randn(3, 7)
    mask = torch.tensor([[1] * 7, [1] * 4 + [0] * 3, [1] * 2 + [0] * 5]).bool()
    zp = zscore_padded(x, mask)
    flat, seg, idx = pack(x, mask)
    assert seg.n_segments == 3 and seg.counts.tolist() == [7, 4, 2]
    zf = segment_zscore(flat, seg)
    assert torch.allclose(unpack(zf, idx, (3, 7)), zp, atol=1e-5)
    cu = torch.tensor([0, 7, 11, 13])
    seg2 = Segments.from_cu(cu, 13)
    assert torch.equal(seg2.seg_ids, seg.seg_ids)


def test_exact_topk_matches_the_old_padded_operator_and_honours_pinned():
    torch.manual_seed(0)
    scores = torch.randn(4, 10)
    mask = torch.ones(4, 10, dtype=torch.bool)
    mask[1, 6:] = False
    keep, k, thr = exact_topk(scores, mask, 0.3)
    assert k.tolist() == [3, 2, 3, 3]  # ceil(0.3 * n_valid)
    for b in range(4):
        top = scores[b].masked_fill(~mask[b], -1e9).topk(int(k[b])).indices
        assert set(top.tolist()) == set(keep[b].nonzero().flatten().tolist())
        assert torch.isclose(
            thr[b], scores[b].masked_fill(~mask[b], -1e9).topk(int(k[b])).values[-1]
        )
    # pinned: a long pinned prefix must not starve the content channel
    pinned = torch.zeros_like(mask)
    pinned[0, :8] = True
    keep_p, k_p, _ = exact_topk(scores, mask, 0.3, pinned=pinned)
    assert keep_p[0, :8].all() and int(k_p[0]) == 8 + 1  # ceil(0.3 * 2 controllable) = 1
    assert torch.equal(keep_p[1:], keep[1:])
    assert quota(mask, 0.3, pinned=pinned)[0] == 9


def test_head_scale_anchor_and_utility_targets_and_listwise_kl():
    torch.manual_seed(0)
    raw = torch.randn(2, 6) * 5.0
    mask = torch.ones(2, 6, dtype=torch.bool)
    loss, std = head_scale_anchor(raw, mask, target_std=1.0, weight=0.5)
    assert loss is not None and float(loss) > 0 and std.shape == (2,)
    none, _ = head_scale_anchor(raw, mask, 1.0, 0.0)
    assert none is None
    g, h = torch.randn(2, 6, 4), torch.randn(2, 6, 4)
    mask[1, 4:] = False
    t = utility_targets(g, h, mask)
    assert (
        torch.allclose(t.sum(-1), torch.ones(2), atol=1e-6) and float(t[1, 4:].abs().sum()) == 0.0
    )
    z = torch.randn(2, 6, requires_grad=True)
    kl = listwise_kl(t, z, mask)
    assert kl is not None and float(kl) >= 0
    kl.backward()
    assert z.grad is not None and float(z.grad[1, 4:].abs().sum()) == 0.0
    conc = teacher_concentration(t, mask)
    assert ((conc > 0) & (conc <= 1)).all()


def test_splice_guard_flags_a_collapsed_or_off_scale_payload():
    from bgkit_core.splice_guard import SpliceGuard

    torch.manual_seed(0)
    g = SpliceGuard()
    mask = torch.ones(2, 8, dtype=torch.bool)
    healthy = torch.randn(2, 8, 16)
    assert (
        g.observe(healthy, mask, {"norm_ratio": 1.0, "eff_rank_within": 6.0, "shared_frac": 0.3})
        == {}
    )
    v = g.observe(healthy, mask, {"norm_ratio": 3.0, "eff_rank_within": 1.2, "shared_frac": 0.99})
    assert set(v) >= {"norm_ratio", "eff_rank_within", "shared_frac"}
    # a constant payload identical to the EMA direction trips the cosine rule
    const = torch.ones(2, 8, 16)
    g2 = SpliceGuard(momentum=1.0)
    g2.observe(const, mask, {})
    v2 = g2.observe(const, mask, {"shared_frac": 0.9})
    assert "cos_to_corpus_ema" in v2
    st = g2.state_dict()
    g3 = SpliceGuard()
    g3.load_state_dict(st)
    assert torch.equal(g3.ema_mean, g2.ema_mean)


def test_splice_guard_rank_check_needs_enough_reps_per_document():
    import torch

    from bgkit_core.splice_guard import SpliceGuard

    g = SpliceGuard(eff_rank_lo=2.0, eff_rank_min_reps=8)
    reps = torch.randn(2, 4, 16)
    few = torch.tensor([[True, True, False, False], [True, False, False, False]])
    assert "eff_rank_within" not in g.observe(reps, few, {"eff_rank_within": 1.2})
    many = torch.ones(2, 4, dtype=torch.bool)
    g2 = SpliceGuard(eff_rank_lo=2.0, eff_rank_min_reps=4)
    assert g2.observe(reps, many, {"eff_rank_within": 1.2})["eff_rank_within"] == 1.2
