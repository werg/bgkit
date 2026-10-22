"""bgkit_triton experts (src/bgkit2/models/moe_kernels.py) vs transformers' eager experts on small
random Lfm2Moe shapes. The fp32 eager forward of the same bf16 values is the ground truth; the
Triton path must be no further from it than bf16 eager is (plus a small slack). CUDA only."""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.gpu

# GPU tests run in the NGC container only: the host venv sees CUDA but cannot build Triton's
# launcher (no Python.h), and is CPU-only by project convention.
cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isdir("/workspace/bgkit"),
    reason="needs CUDA inside the GPU container (/workspace/bgkit)",
)


def _experts(impl: str, dtype, hidden=256, inter=200, num_experts=8, topk=3, seed=0):
    from transformers.models.lfm2_moe.configuration_lfm2_moe import Lfm2MoeConfig
    from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeExperts

    import bgkit2.models.moe_kernels  # registers bgkit_triton

    cfg = Lfm2MoeConfig(
        hidden_size=hidden,
        moe_intermediate_size=inter,
        num_experts=num_experts,
        num_experts_per_tok=topk,
    )
    cfg._experts_implementation = impl
    m = Lfm2MoeExperts(cfg)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.gate_up_proj.copy_(torch.randn(m.gate_up_proj.shape, generator=g) * hidden**-0.5)
        m.down_proj.copy_(torch.randn(m.down_proj.shape, generator=g) * inter**-0.5)
    m.to(device="cuda", dtype=torch.bfloat16).to(dtype)  # same bf16 values for every dtype
    m.requires_grad_(False)
    return m


def _inputs(t=333, hidden=256, num_experts=8, topk=3, seed=1, skew=True):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(t, hidden, generator=g).to(torch.bfloat16)
    logits = torch.randn(t, num_experts, generator=g)
    if skew:  # uneven expert load, one expert unused
        logits[:, 0] += 2.0
        logits[:, -1] -= 100.0
    w, idx = torch.topk(logits.sigmoid(), topk, dim=-1)
    w = (w / w.sum(-1, keepdim=True)).to(torch.bfloat16)
    return x.cuda(), idx.cuda(), w.cuda()


def _run(m, x, idx, w, dtype):
    xg = x.to(dtype).clone().requires_grad_(True)
    wg = w.to(dtype).clone().requires_grad_(True)
    out = m(xg, idx, wg)
    gout = torch.randn(out.shape, generator=torch.Generator().manual_seed(7)).to(out.device, dtype)
    (out.float() * gout.float()).sum().backward()
    return out.detach().float(), xg.grad.float(), wg.grad.float()


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@cuda
@pytest.mark.parametrize("t,skew", [(333, True), (64, False), (1, False)])
def test_triton_experts_match_eager(t, skew):
    x, idx, w = _inputs(t=t, skew=skew)
    ref = _run(_experts("eager", torch.float32), x, idx, w, torch.float32)
    eag = _run(_experts("eager", torch.bfloat16), x, idx, w, torch.bfloat16)
    tri = _run(_experts("bgkit_triton", torch.bfloat16), x, idx, w, torch.bfloat16)
    for name, r, e, k in zip(("out", "dx", "dw"), ref, eag, tri, strict=True):
        floor, err = _rel(e, r), _rel(k, r)
        msg = f"{name}: triton {err:.2e} vs bf16-eager floor {floor:.2e}"
        assert err <= 2.0 * floor + 2e-3, msg


@cuda
def test_triton_experts_forward_no_grad_and_weight_grad_fallback():
    x, idx, w = _inputs()
    m = _experts("bgkit_triton", torch.bfloat16)
    with torch.no_grad():
        a = m(x, idx, w)
    e = _experts("eager", torch.bfloat16)
    with torch.no_grad():
        b = e(x, idx, w)
    assert _rel(a.float(), b.float()) < 1e-2
    # expert weights that require grad must route to the grouped_mm path (weight grads computed)
    m.requires_grad_(True)
    m(x, idx, w).float().sum().backward()
    assert m.gate_up_proj.grad is not None and m.down_proj.grad is not None


@cuda
@pytest.mark.parametrize("shape", [(3, 37, 256), (2, 5, 4, 64), (1, 7, 2048)])
def test_fused_rmsnorm_forward_backward_match_reference(shape):
    from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeRMSNorm

    from bgkit2.models.fused_rmsnorm import FusedRMSNorm, fuse_rmsnorm

    g = torch.Generator().manual_seed(0)
    n = shape[-1]
    x = (torch.randn(shape, generator=g) * 3 + 0.5).to(torch.bfloat16).cuda()
    wv = (1 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16)
    gout = torch.randn(shape, generator=g).cuda()

    def run(dtype, fused):
        m = Lfm2MoeRMSNorm(n, eps=1e-5)
        with torch.no_grad():
            m.weight.copy_(wv.float())
        m = m.cuda().to(dtype)
        m.weight.requires_grad_(True)
        if fused:
            assert fuse_rmsnorm(torch.nn.Sequential(m)) == 1 and isinstance(m, FusedRMSNorm)
            assert isinstance(m, Lfm2MoeRMSNorm)
        xg = x.to(dtype).clone().requires_grad_(True)
        y = m(xg)
        (y.float() * gout).sum().backward()
        return y.detach().float(), xg.grad.float(), m.weight.grad.float()

    truth = run(torch.float32, False)
    ref = run(torch.bfloat16, False)
    fus = run(torch.bfloat16, True)
    for name, t, r, f in zip(("y", "dx", "dw"), truth, ref, fus, strict=True):
        floor, err = _rel(r, t), _rel(f, t)
        assert err <= 1.5 * floor + 1e-3, f"{name}: fused {err:.2e} vs bf16 floor {floor:.2e}"
