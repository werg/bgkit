"""Unit tests for :mod:`bgkit.utils.cpu_offload_checkpoint`.

Exercised on CPU without a GPU. Since the actual CPU-offload path requires
CUDA, the ``cpu_offload_checkpoint`` helper falls through to a plain
``torch.utils.checkpoint.checkpoint(use_reentrant=False)`` call on CPU
runtimes. We therefore test:

- end-to-end forward / backward equivalence against an un-wrapped forward,
- equivalence against ``torch.utils.checkpoint.checkpoint(use_reentrant=False)``,
- that ``enabled=False`` degenerates to a direct call,
- that non-tensor args (bools, ints, None) pass through untouched,
- that gradient equivalence holds on a 5-layer toy MLP.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from bgkit.utils.cpu_offload_checkpoint import cpu_offload_checkpoint


def _make_toy_mlp(in_dim: int = 8, hidden: int = 16, depth: int = 5) -> nn.Module:
    """A 5-layer MLP with per-layer residual so every layer's grad is non-trivial."""
    layers = [nn.Linear(in_dim, hidden), nn.GELU()]
    for _ in range(depth - 2):
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
    layers += [nn.Linear(hidden, in_dim)]
    return nn.Sequential(*layers)


class TestCpuOffloadCheckpoint:
    def test_enabled_false_passthrough(self):
        """With enabled=False the call must be identical to a plain fn(*args)."""
        torch.manual_seed(0)
        mlp = _make_toy_mlp()
        x = torch.randn(4, 8, requires_grad=True)

        y_ref = mlp(x)
        y_off = cpu_offload_checkpoint(mlp, x, enabled=False)
        assert torch.allclose(y_ref, y_off, atol=1e-6)

    def test_forward_matches_plain_checkpoint(self):
        """End-to-end forward + backward equivalence against the stock
        torch.utils.checkpoint (use_reentrant=False) path."""
        torch.manual_seed(1)
        mlp = _make_toy_mlp()
        x1 = torch.randn(4, 8, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)

        # Reference: plain torch_checkpoint
        y_ref = torch_checkpoint(mlp, x1, use_reentrant=False)
        loss_ref = y_ref.pow(2).sum()
        loss_ref.backward()

        # Wrapped: cpu_offload_checkpoint — on CPU this falls through to the
        # same torch_checkpoint path, so gradients MUST match bit-for-bit.
        y_off = cpu_offload_checkpoint(mlp, x2, enabled=True)
        loss_off = y_off.pow(2).sum()
        loss_off.backward()

        assert torch.allclose(y_ref, y_off, atol=1e-6)
        assert torch.allclose(x1.grad, x2.grad, atol=1e-6)

    def test_grads_match_unwrapped_forward(self):
        """Compared to an un-wrapped fn(*args) forward, the wrapped version
        must produce the same parameter gradients."""
        torch.manual_seed(2)
        mlp_ref = _make_toy_mlp()
        mlp_off = _make_toy_mlp()
        mlp_off.load_state_dict(mlp_ref.state_dict())

        x_ref = torch.randn(4, 8, requires_grad=True)
        x_off = x_ref.detach().clone().requires_grad_(True)

        y_ref = mlp_ref(x_ref)
        y_ref.pow(2).sum().backward()

        y_off = cpu_offload_checkpoint(mlp_off, x_off, enabled=True)
        y_off.pow(2).sum().backward()

        assert torch.allclose(y_ref, y_off, atol=1e-6)
        for p_ref, p_off in zip(
            mlp_ref.parameters(), mlp_off.parameters(), strict=True,
        ):
            assert p_ref.grad is not None and p_off.grad is not None
            assert torch.allclose(p_ref.grad, p_off.grad, atol=1e-5, rtol=1e-4)

    def test_non_tensor_args_passthrough(self):
        """Trailing non-tensor args (bools, ints, None) must reach fn unchanged.

        The CPU-offload path splits args into ``(tensors..., others...)``;
        this test exercises the split boundary and the backward reassembly.
        """
        calls = []

        def fn(x, flag, count, extra):
            calls.append((flag, count, extra))
            if flag:
                return x * count
            return x + count

        x = torch.randn(3, 4, requires_grad=True)
        out = cpu_offload_checkpoint(fn, x, True, 2, None, enabled=True)
        assert torch.allclose(out, x * 2)
        out.sum().backward()
        assert x.grad is not None
        assert calls[0] == (True, 2, None)

    def test_multi_tensor_inputs(self):
        """Multiple leading tensor args must all be offloaded + restored."""
        torch.manual_seed(3)

        def fn(a, b):
            return (a @ b.t()).relu()

        a_ref = torch.randn(4, 8, requires_grad=True)
        b_ref = torch.randn(6, 8, requires_grad=True)
        a_off = a_ref.detach().clone().requires_grad_(True)
        b_off = b_ref.detach().clone().requires_grad_(True)

        y_ref = fn(a_ref, b_ref)
        y_ref.sum().backward()

        y_off = cpu_offload_checkpoint(fn, a_off, b_off, enabled=True)
        y_off.sum().backward()

        assert torch.allclose(y_ref, y_off, atol=1e-6)
        assert torch.allclose(a_ref.grad, a_off.grad, atol=1e-5)
        assert torch.allclose(b_ref.grad, b_off.grad, atol=1e-5)

    def test_five_layer_mlp_end_to_end(self):
        """The advertised use case: a 5-layer MLP whose activations would
        otherwise be held on the GPU between forward and backward. On CPU
        we just verify gradient equivalence against a baseline."""
        torch.manual_seed(4)
        mlp = _make_toy_mlp(in_dim=16, hidden=32, depth=5)
        mlp_baseline = _make_toy_mlp(in_dim=16, hidden=32, depth=5)
        mlp_baseline.load_state_dict(mlp.state_dict())

        x = torch.randn(8, 16, requires_grad=True)
        x_b = x.detach().clone().requires_grad_(True)

        y = cpu_offload_checkpoint(mlp, x, enabled=True)
        y.sum().backward()

        y_b = mlp_baseline(x_b)
        y_b.sum().backward()

        for p, p_b in zip(
            mlp.parameters(), mlp_baseline.parameters(), strict=True,
        ):
            assert torch.allclose(p.grad, p_b.grad, atol=1e-5, rtol=1e-4)
