"""Regression test for fp32 master weights in the Muon optimizer.

Without fp32 master weights, a bf16 param updated by AdamW with a small
lr silently no-ops: ``p.add_(update, alpha=-lr)`` rounds the update to
zero because the bf16 increment at the param's value exceeds
``lr * |update|``. The bug surfaced on the survivorship head bias
(``encoder.l0.head.head[2].bias`` stuck at 0.0625 across 50k+ training
steps under asymmetric BCE).
"""

from __future__ import annotations

import torch

from bgkit.training.muon import Muon


def _make_param(value: float, dtype: torch.dtype) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.tensor([value], dtype=dtype))


def test_adamw_path_updates_bf16_param_with_small_lr():
    """Bf16 param under AdamW + small lr must accumulate in fp32 master.

    Without master weights, ``p.add_(update, alpha=-lr)`` rounds the
    sub-bf16-increment update to zero every step. With the fp32 master,
    individual updates accumulate exactly in the master, and the live
    bf16 param eventually steps once the master crosses the bf16 grid.

    We run enough steps for the cumulative update to exceed the bf16
    increment at the param's value (~4.88e-4 at 0.0625) and assert both:
    (a) the master moved by roughly N * lr per step
    (b) the live bf16 param eventually reflects the master.
    """
    param = _make_param(0.0625, torch.bfloat16)
    optimizer = Muon([
        {"params": [param], "use_muon": False, "lr": 2e-5},
    ])

    # Simulate a steady negative gradient — like target=1 BCE on a logit
    # sitting at deep negative. Enough steps that cumulative |update| ≈
    # 50 * 2e-5 = 1e-3 exceeds the bf16 increment at 0.0625.
    for _ in range(50):
        param.grad = torch.tensor([-0.74], dtype=torch.bfloat16)
        optimizer.step()

    master = optimizer.state[param]["master_param"]
    assert master.item() > 0.0625, (
        "fp32 master did not accumulate — gradient is not reaching the "
        "master via the AdamW path."
    )
    assert param.item() != 0.0625, (
        "bf16 live param did not pick up the fp32 master after 50 steps — "
        "the .data.copy_ writeback is broken."
    )
    assert param.item() > 0.0625


def test_adamw_path_updates_match_fp32_baseline():
    """Bf16 param with master weights should track an fp32 param closely."""
    p_bf16 = _make_param(0.0625, torch.bfloat16)
    p_fp32 = _make_param(0.0625, torch.float32)
    opt_bf16 = Muon([{"params": [p_bf16], "use_muon": False, "lr": 2e-5}])
    opt_fp32 = Muon([{"params": [p_fp32], "use_muon": False, "lr": 2e-5}])

    for _ in range(20):
        p_bf16.grad = torch.tensor([-0.74], dtype=torch.bfloat16)
        p_fp32.grad = torch.tensor([-0.74], dtype=torch.float32)
        opt_bf16.step()
        opt_fp32.step()

    # bf16 round-tripping through .data.copy_ adds bf16-precision noise but
    # the trajectory should be qualitatively the same.
    drift = abs(p_bf16.item() - p_fp32.item())
    assert drift < 5e-3, (
        f"bf16 trajectory drifted from fp32 baseline by {drift} — "
        "fp32 master not properly tracking"
    )


def test_muon_path_updates_bf16_param():
    """The Newton-Schulz Muon path must also work on bf16 params.

    Same bf16-truncation concern as the AdamW path: each per-step update
    of size ``lr * normalized_ns_update`` (≈ 1) gets rounded to zero if
    the bf16 increment at the param's value exceeds it. The fp32 master
    accumulates correctly and the live bf16 view picks up changes once
    the cumulative master delta crosses the grid.
    """
    weight = torch.nn.Parameter(
        torch.full((8, 8), 0.5, dtype=torch.bfloat16)
    )
    initial = weight.detach().clone()
    optimizer = Muon([
        {"params": [weight], "use_muon": True, "lr": 1e-3, "ns_steps": 1},
    ])

    torch.manual_seed(0)
    for _ in range(20):
        weight.grad = torch.randn(8, 8, dtype=torch.bfloat16)
        optimizer.step()

    state = optimizer.state[weight]
    master = state["master_param"]
    master_delta = (master - initial.float()).abs().max().item()
    assert master_delta > 1e-3, (
        f"fp32 master did not accumulate (max delta {master_delta}) — "
        "Muon NS path not flowing to master."
    )
    delta = (weight.detach() - initial).abs().max().item()
    assert delta > 0, "bf16 Muon param did not pick up the fp32 master"


def test_master_param_lazy_creation_and_state_visibility():
    """Master param appears in optimizer state[p] for bf16 params."""
    p = _make_param(0.5, torch.bfloat16)
    opt = Muon([{"params": [p], "use_muon": False, "lr": 1e-3}])
    p.grad = torch.tensor([-0.5], dtype=torch.bfloat16)
    opt.step()
    state = opt.state[p]
    assert "master_param" in state, "fp32 master not stored in state"
    assert state["master_param"].dtype == torch.float32
    # Saved state should be serializable — the by-name save path copies state
    # dicts via torch.save.
    cloned = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in state.items()}
    assert cloned["master_param"].dtype == torch.float32


def test_fp32_param_does_not_create_redundant_master():
    """If the live param is already fp32, no master copy is created."""
    p = _make_param(0.5, torch.float32)
    opt = Muon([{"params": [p], "use_muon": False, "lr": 1e-3}])
    p.grad = torch.tensor([-0.5], dtype=torch.float32)
    opt.step()
    state = opt.state[p]
    assert "master_param" not in state, (
        "fp32 param should not get a redundant master copy"
    )


def test_resume_recreates_master_from_live_param():
    """Loading a pre-master-weights checkpoint resumes cleanly."""
    p = _make_param(0.0625, torch.bfloat16)
    opt = Muon([{"params": [p], "use_muon": False, "lr": 2e-5}])
    # Simulate old-format state: no master_param key.
    opt.state[p] = {
        "exp_avg": torch.tensor([-0.07], dtype=torch.bfloat16),
        "exp_avg_sq": torch.tensor([0.028], dtype=torch.bfloat16),
        "step": 1246,
    }
    for _ in range(50):
        p.grad = torch.tensor([-0.74], dtype=torch.bfloat16)
        opt.step()
    state = opt.state[p]
    assert "master_param" in state
    assert state["master_param"].dtype == torch.float32
    # exp_avg / exp_avg_sq should also be upcast.
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32
    # And the bf16 param now reflects the accumulated fp32 update.
    assert p.item() != 0.0625
