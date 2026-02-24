"""Muon optimizer — vendored from https://github.com/KellerJordan/Muon

MomentUm Orthogonalized by Newton-Schulz. Single-device variant with
auxiliary AdamW for non-Muon parameter groups (biases, norms, embeddings).

Each param group must have a ``use_muon: bool`` key:
- ``use_muon=True``: Newton-Schulz orthogonalization on 2D+ weight gradients
- ``use_muon=False``: Standard AdamW update

Vendored to avoid an external dependency. Only the single-device variant is
included (DGX Spark trains on one GPU).
"""

from __future__ import annotations

import torch


def _zeropower_via_newtonschulz5(g: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Uses a quintic iteration whose coefficients maximize the slope at zero.
    Produces something like US'V^T where S' ~ Uniform(0.5, 1.5), which works
    as well as exact UV^T in practice.
    """
    assert g.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    if g.size(-2) > g.size(-1):
        x = x.mT

    # Ensure spectral norm is at most 1
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        aa = x @ x.mT
        bb = b * aa + c * aa @ aa
        x = a * x + bb @ x

    if g.size(-2) > g.size(-1):
        x = x.mT
    return x


def _muon_update(
    grad: torch.Tensor,
    momentum_buf: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    """Compute Muon update: momentum + Newton-Schulz orthogonalization."""
    momentum_buf.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum_buf, beta) if nesterov else momentum_buf
    if update.ndim == 4:  # conv filters
        update = update.view(len(update), -1)
    update = _zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def _adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    """Standard Adam update from sufficient statistics."""
    exp_avg.lerp_(grad, 1 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1 - betas[1])
    bc1 = exp_avg / (1 - betas[0] ** step)
    bc2 = exp_avg_sq / (1 - betas[1] ** step)
    return bc1 / (bc2.sqrt() + eps)


class Muon(torch.optim.Optimizer):
    """Muon+AdamW hybrid optimizer (single-device).

    Each param group must include ``use_muon: bool``.

    Muon groups use Newton-Schulz orthogonalization on gradients.
    Non-Muon groups use standard AdamW updates.

    Args:
        param_groups: List of dicts, each with ``params`` and ``use_muon``.
            Muon groups accept: lr, momentum, weight_decay.
            AdamW groups accept: lr, betas, eps, weight_decay.
    """

    def __init__(self, param_groups: list[dict]) -> None:
        for group in param_groups:
            self._apply_group_defaults(group)
        super().__init__(param_groups, dict())

    @staticmethod
    def _apply_group_defaults(group: dict) -> None:
        """Inject defaults based on use_muon flag."""
        if "use_muon" not in group:
            raise ValueError("Each param group must have 'use_muon' key")
        if group["use_muon"]:
            group.setdefault("lr", 0.02)
            group.setdefault("momentum", 0.95)
            group.setdefault("weight_decay", 0)
        else:
            group.setdefault("lr", 3e-4)
            group.setdefault("betas", (0.9, 0.95))
            group.setdefault("eps", 1e-10)
            group.setdefault("weight_decay", 0)

    def add_param_group(self, param_group: dict) -> None:
        """Override to inject Muon/AdamW defaults before adding."""
        self._apply_group_defaults(param_group)
        super().add_param_group(param_group)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = _muon_update(
                        p.grad, state["momentum_buffer"], beta=group["momentum"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = _adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
