"""Muon optimizer — vendored from https://github.com/KellerJordan/Muon

MomentUm Orthogonalized by Newton-Schulz. Single-device variant with
auxiliary AdamW for non-Muon parameter groups (biases, norms, embeddings).

Each param group must have a ``use_muon: bool`` key:
- ``use_muon=True``: Newton-Schulz orthogonalization on 2D+ weight gradients
- ``use_muon=False``: Standard AdamW update

Vendored to avoid an external dependency. Only the single-device variant is
included (DGX Spark trains on one GPU).

FP32 master weights (2026-05-27): for any param whose dtype is not fp32,
the optimizer maintains an fp32 master copy in
``state["master_param"]``. The optimizer step runs entirely in fp32 (grad
upcast, update in fp32, master updated) and then writes the result back
into the low-precision live param. Without this, a per-step update of
magnitude ``lr * adam_update`` smaller than the bf16 increment at the
param's value (~param/128) gets rounded to zero before being added to the
live param. The bug surfaced on the survivorship head (head.head.2.bias
stuck at 0.0625 across 50k+ training steps under asymmetric BCE) but
applies to every bf16 param the optimizer touches.
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


def _ensure_master_param(state: dict, p: torch.Tensor) -> torch.Tensor:
    """Return the fp32 master copy of ``p``, lazily creating it.

    If ``p`` is already fp32 we return ``p`` itself — no master needed.
    Otherwise we materialize an fp32 copy of ``p.data`` on the same
    device. Saved-state checkpoints that predate the master-weight
    refactor have no ``master_param`` key, so a resume initializes the
    master from the bf16 live param's current value — that's the same
    state the optimizer would have continued from anyway.

    The master is stored under ``state["master_param"]`` so the
    name-keyed save/restore path in BaseTrainer serializes it
    automatically.
    """
    if p.dtype == torch.float32:
        return p
    master = state.get("master_param")
    if master is None or master.shape != p.shape or master.device != p.device:
        master = p.detach().to(dtype=torch.float32).clone()
        state["master_param"] = master
    return master


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
            group.setdefault("ns_steps", 5)
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
                    master = _ensure_master_param(state, p)
                    grad32 = p.grad.to(dtype=torch.float32)
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(master)
                    elif state["momentum_buffer"].dtype != torch.float32:
                        # Migrate legacy bf16 momentum buffers to fp32.
                        state["momentum_buffer"] = state["momentum_buffer"].to(
                            dtype=torch.float32
                        )
                    update = _muon_update(
                        grad32,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                    )
                    master.mul_(1 - group["lr"] * group["weight_decay"])
                    master.add_(update.reshape(master.shape), alpha=-group["lr"])
                    if master is not p:
                        p.data.copy_(master)
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    master = _ensure_master_param(state, p)
                    grad32 = p.grad.to(dtype=torch.float32)
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(master)
                        state["exp_avg_sq"] = torch.zeros_like(master)
                        state["step"] = 0
                    else:
                        if state["exp_avg"].dtype != torch.float32:
                            state["exp_avg"] = state["exp_avg"].to(
                                dtype=torch.float32
                            )
                        if state["exp_avg_sq"].dtype != torch.float32:
                            state["exp_avg_sq"] = state["exp_avg_sq"].to(
                                dtype=torch.float32
                            )
                    state["step"] += 1
                    update = _adam_update(
                        grad32,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    master.mul_(1 - group["lr"] * group["weight_decay"])
                    master.add_(update, alpha=-group["lr"])
                    if master is not p:
                        p.data.copy_(master)

        return loss
