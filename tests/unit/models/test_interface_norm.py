"""The interface contract must actually remove the thing it exists to remove.

widenet v8's survivors were 99% one corpus-constant vector; the document
signal sat at ~1% of the norm and the decoder's RMSNorm divided by the
constant's magnitude. These tests pin the two properties that matter: the
constant is gone, and inflating one stops paying.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.models.interface_norm import DecoderInterfaceNorm


def _payload(n=256, d=64, shared_scale=300.0, seed=0):
    torch.manual_seed(seed)
    shared = torch.randn(d) * shared_scale
    return shared + torch.randn(n, d), shared


def test_the_shared_component_is_gone_after_one_update():
    x, _ = _payload()
    norm = DecoderInterfaceNorm(64, target_row_norm=0.64)
    norm.train()
    y = norm(x).detach()
    energy = y.pow(2).sum(dim=-1).mean()
    shared_frac = y.mean(dim=0).pow(2).sum() / energy
    assert float(shared_frac) < 0.05


def test_output_row_norm_matches_the_decoder_embedding_scale():
    x, _ = _payload()
    norm = DecoderInterfaceNorm(64, target_row_norm=0.64)
    norm.train()
    y = norm(x).detach()
    assert float(y.norm(dim=-1).mean()) == pytest.approx(0.64, rel=0.15)


def test_inflating_a_constant_no_longer_changes_what_the_decoder_sees():
    """The point of the contract. If a bigger shared vector still moved the
    output, a training signal could be served by growing one -- which is how
    the collapse happened in the first place."""
    torch.manual_seed(0)
    content = torch.randn(256, 64)
    out = []
    for scale in (0.0, 50.0, 5000.0):
        norm = DecoderInterfaceNorm(64, target_row_norm=0.64)
        norm.train()
        out.append(norm(content + torch.ones(64) * scale))
    torch.testing.assert_close(out[0], out[1], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(out[0], out[2], rtol=2e-3, atol=2e-3)


def test_statistics_are_an_ema_not_the_current_batch():
    """Batch statistics would make what the decoder reads depend on which
    other documents happened to be in the microbatch, and would differ
    between training and eval."""
    x, _ = _payload(seed=1)
    norm = DecoderInterfaceNorm(64)
    norm.train()
    norm(x)
    before = norm.running_mean.clone()
    odd = x + 1000.0
    norm(odd)
    # One outlier batch moves the reference by ~momentum, not to the batch.
    moved = (norm.running_mean - before).abs().mean()
    assert 5.0 < float(moved) < 20.0


def test_eval_mode_does_not_update_a_calibrated_reference():
    x, _ = _payload()
    norm = DecoderInterfaceNorm(64)
    norm.train()
    norm(x)
    ref = norm.running_mean.clone()
    norm.eval()
    norm(x + 500.0)
    torch.testing.assert_close(norm.running_mean, ref)


def test_eval_calibrates_once_when_the_reference_was_never_set():
    """Loading a checkpoint that predates this module leaves the buffers at
    (0, 1). Standardising by those is not a no-op, it is a silent wrong
    normalisation -- so the first eval forward must calibrate."""
    x, _ = _payload(shared_scale=300.0)
    norm = DecoderInterfaceNorm(64)
    norm.eval()
    y = norm(x).detach()
    assert int(norm.num_updates.item()) == 1
    energy = y.pow(2).sum(dim=-1).mean()
    assert float(y.mean(dim=0).pow(2).sum() / energy) < 0.05
    # ...and only once: a second eval batch must not move it.
    ref = norm.running_mean.clone()
    norm(x + 500.0)
    torch.testing.assert_close(norm.running_mean, ref)


def test_train_and_eval_produce_the_same_output_for_the_same_input():
    """No train/eval discrepancy to reconcile: the normaliser is the same
    frozen reference in both, unlike batch normalisation."""
    x, _ = _payload()
    norm = DecoderInterfaceNorm(64)
    norm.train()
    norm(x)
    y_train = norm(x)
    norm.eval()
    torch.testing.assert_close(norm(x), y_train)


def test_first_update_sets_the_reference_rather_than_blending_into_defaults():
    """An EMA started at mean 0 / var 1 takes ~1/momentum steps to arrive,
    and until then the decoder is handed a payload normalised against numbers
    that describe nothing."""
    x, _ = _payload(shared_scale=300.0)
    norm = DecoderInterfaceNorm(64, momentum=0.01)
    norm.train()
    norm(x)
    torch.testing.assert_close(
        norm.running_mean, x.float().mean(dim=0), rtol=1e-5, atol=1e-5,
    )
    assert int(norm.num_updates.item()) == 1


def test_gradient_flows_through_the_input_but_not_the_reference():
    x, _ = _payload()
    x = x.requires_grad_(True)
    norm = DecoderInterfaceNorm(64)
    norm.train()
    norm(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert norm.running_mean.grad is None


def test_empty_payload_passes_through():
    norm = DecoderInterfaceNorm(64)
    empty = torch.zeros(0, 64)
    assert norm(empty).shape == empty.shape
    assert int(norm.num_updates.item()) == 0


def test_a_single_row_does_not_define_a_variance():
    """One vector has zero variance per channel; taking it as the reference
    would divide every later payload by eps."""
    norm = DecoderInterfaceNorm(64)
    norm.train()
    norm(torch.randn(1, 64))
    assert int(norm.num_updates.item()) == 0


def test_wrong_width_is_refused_rather_than_broadcast():
    norm = DecoderInterfaceNorm(64)
    with pytest.raises(ValueError, match="last dim 64"):
        norm(torch.randn(8, 32))


def test_the_contract_is_fixed_by_default():
    """Measured, not stylistic. The first arm ran with a learnable gain and
    shift; over 65 steps the emitted norm ratio climbed 1.30 -> 6.52 and
    shared_frac 0.26 -> 0.94. A trainable shift IS a corpus-constant re-added
    after the standardisation that removed one, and a trainable gain restores
    the free scale direction the contract exists to close."""
    assert not any(p.requires_grad for p in DecoderInterfaceNorm(64).parameters())
    assert list(DecoderInterfaceNorm(64).parameters()) == []


def test_the_affine_can_be_opened_deliberately():
    norm = DecoderInterfaceNorm(64, affine=True)
    names = {n for n, p in norm.named_parameters() if p.requires_grad}
    assert names == {"weight", "bias"}


@pytest.mark.parametrize("bad", [{"dim": 0}, {"momentum": 0.0}, {"momentum": 2.0},
                                 {"target_row_norm": 0.0}])
def test_invalid_configuration_is_refused_at_construction(bad):
    kwargs = {"dim": 64, **bad}
    with pytest.raises(ValueError):
        DecoderInterfaceNorm(**kwargs)
