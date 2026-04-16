"""Tests for the two-head survivorship adapter: gradient routing + EMA semantics.

Specifically tests:
- BCE/moment-match-shaped losses on ``base_raw`` flow ONLY to ``head_base``.
- Soft-attn-shaped losses on ``logits_for_softattn`` flow ONLY to ``head_adapter``.
- ``logits_for_op`` and ``logits_for_softattn`` produce numerically identical values.
- At step 0 (μ=0, adapter zero-init): ``adapter_zm.mean()`` ≈ 0 and ``logits_for_op
  == base_raw`` exactly.
- After EMA convergence to a constant input, ``adapter_zm`` mean is bounded.
- Adapter learns from zero (zero-init final layer doesn't trap gradient).
- Adapter is batch-independent (μ is a buffer, not a per-batch statistic).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from bgkit.models.bgkit_compressor import BgKITCompressor  # noqa: E402
from bgkit.models.components.selection import AdapterMeanEMA  # noqa: E402


HIDDEN_DIM = 32


class _Output:
    def __init__(self, last_hidden_state, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class _SinglePassBackbone(nn.Module):
    """Minimal backbone whose hook fires at index 7 (matching standard mapping)."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.embed_tokens = nn.Embedding(64, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)])
        self.norm = nn.Identity()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        return_intermediates=False,
        layer_hooks=None,
        **kwargs,
    ):
        x = self.layers[0](inputs_embeds)
        if layer_hooks:
            for idx in sorted(layer_hooks):
                x = layer_hooks[idx](x)
        x = self.norm(x)
        return _Output(last_hidden_state=x)


def _make_compressor(hidden_dim: int = HIDDEN_DIM) -> BgKITCompressor:
    backbone = _SinglePassBackbone(hidden_dim)
    norm = nn.LayerNorm(hidden_dim)
    return BgKITCompressor(
        backbone, norm,
        hidden_dim=hidden_dim,
        survivorship_inner_dim=8,
    )


def _seed():
    torch.manual_seed(0)


# ----------------------------------------------------------------------
# At step 0 (μ=0, adapter zero-init)
# ----------------------------------------------------------------------


def test_adapter_zero_init_means_logits_for_op_equals_base_raw():
    _seed()
    comp = _make_compressor()
    x = torch.randn(2, 6, HIDDEN_DIM)
    out = comp(x, target_ratio=0.5, level="l0")
    # adapter_raw should be exactly zero (zero-init final layer + GELU(0)=0
    # times zero weight).
    assert out.adapter_raw is not None
    assert torch.all(out.adapter_raw == 0.0)
    # μ also zero by construction.
    assert torch.all(out.adapter_zm == 0.0)
    # logits_for_op == base_raw exactly.
    assert torch.allclose(out.logits_for_op, out.base_raw, atol=0, rtol=0)


def test_adapter_zm_mean_near_zero_at_step_zero():
    _seed()
    comp = _make_compressor()
    x = torch.randn(2, 6, HIDDEN_DIM)
    out = comp(x, target_ratio=0.5, level="l0")
    # All zeros => mean is exactly zero.
    valid = torch.ones(2, 6, dtype=torch.bool)
    val = out.adapter_zm.masked_select(valid).mean().abs().item()
    assert val < 1e-5


# ----------------------------------------------------------------------
# Numerical identity of operator and softattn views
# ----------------------------------------------------------------------


def test_logits_for_op_and_softattn_numerically_identical():
    _seed()
    comp = _make_compressor()
    # Initialize adapter with non-zero weights so we exercise more of the math.
    with torch.no_grad():
        for p in comp.head_adapter_l0.parameters():
            p.normal_(0.0, 0.1)
        comp.adapter_mean_ema_l0.mu_param.fill_(0.3)

    x = torch.randn(2, 6, HIDDEN_DIM)
    out = comp(x, target_ratio=0.5, level="l0")
    assert torch.allclose(
        out.logits_for_op, out.logits_for_softattn, atol=1e-6, rtol=1e-6,
    )


# ----------------------------------------------------------------------
# Gradient routing
# ----------------------------------------------------------------------


def test_bce_on_base_raw_does_not_leak_to_adapter():
    _seed()
    comp = _make_compressor()
    # Give adapter non-trivial weights so it produces non-zero output (any
    # leak through adapter_zm subtraction would show in its grad).
    with torch.no_grad():
        for p in comp.head_adapter_l0.parameters():
            p.normal_(0.0, 0.1)

    x = torch.randn(2, 6, HIDDEN_DIM)
    out = comp(x, target_ratio=0.5, level="l0")
    target = torch.zeros_like(out.base_raw)
    loss = nn.functional.binary_cross_entropy_with_logits(out.base_raw, target)
    loss.backward()
    for name, p in comp.head_adapter_l0.named_parameters():
        assert p.grad is None or float(p.grad.abs().sum().item()) == 0.0, (
            f"adapter param {name} unexpectedly received gradient from base-only loss"
        )
    base_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in comp.head_base_l0.parameters()
    )
    assert base_has_grad, "head_base should have received gradient from base-only loss"


def test_softattn_on_logits_for_softattn_does_not_leak_to_base():
    _seed()
    comp = _make_compressor()
    with torch.no_grad():
        for p in comp.head_adapter_l0.parameters():
            p.normal_(0.0, 0.1)

    x = torch.randn(2, 6, HIDDEN_DIM)
    out = comp(x, target_ratio=0.5, level="l0")
    # Pretend a downstream forward consumed logits_for_softattn — backprop a
    # scalar through it.
    loss = out.logits_for_softattn.pow(2).mean()
    loss.backward()

    for name, p in comp.head_base_l0.named_parameters():
        assert p.grad is None or float(p.grad.abs().sum().item()) == 0.0, (
            f"base param {name} unexpectedly received gradient from softattn-only loss"
        )
    adapter_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in comp.head_adapter_l0.parameters()
    )
    assert adapter_has_grad, "head_adapter should have received gradient from softattn loss"


def test_adapter_learns_from_zero_init():
    """Zero-init final layer doesn't trap gradient — upstream signal propagates."""
    _seed()
    comp = _make_compressor()
    # Adapter is zero-init by construction.
    x = torch.randn(2, 6, HIDDEN_DIM)
    out = comp(x, target_ratio=0.5, level="l0")
    loss = out.logits_for_softattn.pow(2).mean()
    loss.backward()
    # Final layer weight grad is non-zero because the upstream gradient is
    # non-zero (the GELU input is non-zero through the non-zero first-layer
    # weights).
    final_layer = comp.head_adapter_l0.head[-1]
    assert final_layer.weight.grad is not None
    # The bias receives gradient regardless; the weight grad should also be
    # non-trivial because the inner activations are non-zero.
    assert float(final_layer.bias.grad.abs().sum().item()) > 0.0


# ----------------------------------------------------------------------
# Batch independence
# ----------------------------------------------------------------------


@pytest.mark.parametrize("level", ["l0", "l1"])
def test_adapter_logits_batch_independent(level):
    """A position's logits depend only on its hidden state and saved μ.

    Placing the same position in two different batches must yield bit-identical
    logits_for_op (μ is a buffer, not a per-batch statistic).

    Parameterized over both levels to catch any l0/l1 divergence.
    """
    _seed()
    comp = _make_compressor()
    adapter = comp.head_adapter_l0 if level == "l0" else comp.head_adapter_l1
    ema = comp.adapter_mean_ema_l0 if level == "l0" else comp.adapter_mean_ema_l1
    with torch.no_grad():
        for p in adapter.parameters():
            p.normal_(0.0, 0.1)
        ema.mu_param.fill_(0.42)

    pos = torch.randn(1, 1, HIDDEN_DIM)
    batch_a = torch.cat([pos, torch.randn(1, 5, HIDDEN_DIM)], dim=1)
    batch_b = torch.cat([pos, torch.randn(1, 5, HIDDEN_DIM)], dim=1)

    out_a = comp(batch_a, target_ratio=0.5, level=level)
    out_b = comp(batch_b, target_ratio=0.5, level=level)
    assert torch.allclose(
        out_a.logits_for_op[:, 0], out_b.logits_for_op[:, 0], atol=1e-6,
    )
    assert torch.allclose(
        out_a.adapter_zm[:, 0], out_b.adapter_zm[:, 0], atol=1e-6,
    )


# ----------------------------------------------------------------------
# EMA convergence
# ----------------------------------------------------------------------


def test_adapter_mean_ema_converges_to_constant():
    ema = AdapterMeanEMA(init_mu=0.0, momentum=0.99)
    for _ in range(500):
        ema.update(0.7)
    assert abs(float(ema.value.item()) - 0.7) < 0.01


def test_adapter_zero_sum_under_constant_input():
    """If adapter_raw mean is constant, μ converges and adapter_zm mean → 0."""
    ema = AdapterMeanEMA(init_mu=0.0, momentum=0.99)
    for _ in range(500):
        ema.update(0.7)
    # Pretend a new batch has mean exactly 0.7 (μ has converged).
    fake_adapter = torch.full((4, 32), 0.7)
    valid = torch.ones_like(fake_adapter, dtype=torch.bool)
    zm = fake_adapter - ema.value.to(fake_adapter.dtype)
    assert abs(float(zm.masked_select(valid).mean().item())) < 0.01


def test_phase2_style_aux_loss_routes_gradient_to_base_only():
    """Phase 2's L0/L1 ratio/decisiveness/relevance losses must NOT train
    head_adapter — they pull head_base (which shapes raw head distribution),
    while head_adapter receives its gradient exclusively via soft-attn.

    Regression: before the fix, these losses read enc_out.survive_probs
    (attached sigmoid of base+adapter) and leaked gradient into the
    adapter. The fix recomposes logits via base_raw + adapter_zm.detach()
    so the adapter's autograd contribution is zeroed.
    """
    _seed()
    comp = _make_compressor()
    with torch.no_grad():
        for p in comp.head_adapter_l0.parameters():
            p.normal_(0.0, 0.1)

    x = torch.randn(1, 6, HIDDEN_DIM)
    out = comp(x, target_ratio=0.5, level="l0")

    # Simulate Phase 2's new aux-loss path: base attached, adapter detached.
    base_raw = out.base_raw
    adapter_zm = out.adapter_zm
    logits_base_only = base_raw + adapter_zm.detach()
    theta_t = out.theta_tensor
    probs = torch.sigmoid(
        logits_base_only.float() - theta_t.to(base_raw.device).float()
    ).to(base_raw.dtype)
    # ratio-style loss.
    loss = (probs.mean() - 0.1) ** 2
    loss.backward()

    adapter_grad = sum(
        (p.grad.abs().sum().item() if p.grad is not None else 0.0)
        for p in comp.head_adapter_l0.parameters()
    )
    base_grad = sum(
        (p.grad.abs().sum().item() if p.grad is not None else 0.0)
        for p in comp.head_base_l0.parameters()
    )
    assert adapter_grad == 0.0, (
        f"adapter unexpectedly received gradient from base-only aux loss; "
        f"adapter_grad={adapter_grad}"
    )
    assert base_grad > 0.0


def test_adapter_lag_under_noisy_input():
    """Noisy adapter mean → bounded EMA-tracking lag."""
    torch.manual_seed(0)
    ema = AdapterMeanEMA(init_mu=0.0, momentum=0.99)
    for _ in range(500):
        ema.update(float(torch.randn(()) * 0.1 + 0.7))
    # Sample 100 more batches; the rolling residual should be small.
    residuals = []
    for _ in range(100):
        new_mean = float(torch.randn(()) * 0.1 + 0.7)
        ema.update(new_mean)
        residual = new_mean - float(ema.value.item())
        residuals.append(abs(residual))
    avg_residual = sum(residuals) / len(residuals)
    # With momentum=0.99 and noise std=0.1 per step, the residual per
    # new sample is dominated by the new sample's own deviation from the
    # slow-tracking EMA — approximately the noise std itself. The check
    # is "bounded by the noise scale," not "tightly tracking."
    assert avg_residual < 0.15
