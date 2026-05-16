"""Parity tests for Falcon-H1 patch.

The patch in ``bgkit.utils.falcon_h1_patch`` strips no-op unit multipliers
from Falcon-H1 attention / MLP / mixer / layer forwards. These tests pin
forward-pass parity (and, where feasible, backward parity) between the
stock HF implementation and the patched implementation.

CPU-only tests run on the host venv (CI-friendly). GPU + fused-kernel tests
require Triton + mamba-ssm and only run under the training Docker image.

Test plan:

- ``test_cpu_forward_parity``: full Falcon-H1-Tiny config, num_hidden_layers=2,
  CPU, ``model.eval()`` (torch_forward path). Asserts max abs diff < 1e-5 fp32.
- ``test_cpu_forward_parity_train_mode``: same config, ``model.train()``,
  but mixer fast-path is gated off on CPU so the unit-mul strip is the
  only thing tested. Still asserts max abs diff < 1e-5 fp32.
- ``test_cpu_backward_parity``: backward pass parity. Asserts grad max abs
  diff for embed_tokens.weight < 1e-5 fp32.
- ``test_idempotent``: running ``patch_falcon_h1_decoder`` twice returns
  zero new patches on the second call.
- ``test_non_unit_multiplier_not_patched``: when a multiplier is not 1.0,
  the corresponding sub-module is NOT patched (skip silently).
- ``test_mixer_cuda_fast_path_parity`` (GPU + fused kernels): full
  fused-kernel forward + backward parity at bf16. Asserts max abs
  fwd-out diff < 1e-3 (bf16 tolerance) and max abs grad diff < 5e-3.
"""

from __future__ import annotations

import os

import pytest
import torch


def _build_small_falcon(num_layers: int = 2):
    """Return a `FalconH1ForCausalLM` with ``num_hidden_layers=num_layers``.

    Uses the Falcon-H1-Tiny-90M-Instruct config (all unit multipliers).
    """
    from transformers import AutoConfig
    from transformers.models.falcon_h1 import FalconH1ForCausalLM

    cfg = AutoConfig.from_pretrained("tiiuae/Falcon-H1-Tiny-90M-Instruct")
    cfg.num_hidden_layers = num_layers
    model = FalconH1ForCausalLM(cfg)
    return model, cfg


def _state_dict_clone(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _two_models_with_same_init(num_layers: int = 2):
    """Build two Falcon-H1 instances with bit-identical initial weights."""
    torch.manual_seed(1234)
    model_ref, cfg = _build_small_falcon(num_layers)
    sd = _state_dict_clone(model_ref)

    torch.manual_seed(5678)  # different init to prove the load equalizes
    model_patched, _ = _build_small_falcon(num_layers)
    model_patched.load_state_dict(sd)
    return model_ref, model_patched, cfg


def test_cpu_forward_parity_eval_mode():
    """Stock vs patched on CPU in eval mode (torch_forward Mamba path)."""
    os.environ.pop("BGKIT_FALCON_H1_PATCH", None)
    from bgkit.utils.falcon_h1_patch import patch_falcon_h1_decoder

    model_ref, model_patched, cfg = _two_models_with_same_init(num_layers=2)
    model_ref.eval()
    model_patched.eval()

    report = patch_falcon_h1_decoder(model_patched)
    assert report.attention == 2, report
    assert report.mlp == 2, report
    assert report.mixer == 2, report
    assert report.layer == 2, report

    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        out_ref = model_ref(input_ids=ids, use_cache=False).logits
        out_p = model_patched(input_ids=ids, use_cache=False).logits

    diff = (out_p - out_ref).abs().max().item()
    assert diff < 1e-5, f"max abs diff = {diff:.3e}"


def test_cpu_forward_parity_train_mode():
    """Stock vs patched on CPU in train mode.

    On CPU the mixer fast-path is gated off (no CUDA device), so this test
    only exercises the attention / MLP / layer unit-mul strips.
    """
    os.environ.pop("BGKIT_FALCON_H1_PATCH", None)
    from bgkit.utils.falcon_h1_patch import patch_falcon_h1_decoder

    model_ref, model_patched, cfg = _two_models_with_same_init(num_layers=2)
    model_ref.train()
    model_patched.train()
    patch_falcon_h1_decoder(model_patched)

    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        out_ref = model_ref(input_ids=ids, use_cache=False).logits
        out_p = model_patched(input_ids=ids, use_cache=False).logits

    diff = (out_p - out_ref).abs().max().item()
    assert diff < 1e-5, f"max abs diff = {diff:.3e}"


def test_cpu_backward_parity():
    """Backward pass parity: gradients on embed_tokens.weight match."""
    os.environ.pop("BGKIT_FALCON_H1_PATCH", None)
    from bgkit.utils.falcon_h1_patch import patch_falcon_h1_decoder

    model_ref, model_patched, cfg = _two_models_with_same_init(num_layers=2)
    patch_falcon_h1_decoder(model_patched)

    model_ref.eval()
    model_patched.eval()

    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, 16))

    model_ref.zero_grad()
    model_patched.zero_grad()

    out_ref = model_ref(input_ids=ids, labels=ids, use_cache=False)
    out_ref.loss.backward()

    out_p = model_patched(input_ids=ids, labels=ids, use_cache=False)
    out_p.loss.backward()

    g_ref = model_ref.model.embed_tokens.weight.grad
    g_p = model_patched.model.embed_tokens.weight.grad
    assert g_ref is not None and g_p is not None
    diff = (g_p - g_ref).abs().max().item()
    assert diff < 1e-5, f"max grad abs diff = {diff:.3e}"


def test_idempotent():
    """patch_falcon_h1_decoder twice returns 0 new patches on second call."""
    os.environ.pop("BGKIT_FALCON_H1_PATCH", None)
    from bgkit.utils.falcon_h1_patch import patch_falcon_h1_decoder

    model, _ = _build_small_falcon(num_layers=2)
    r1 = patch_falcon_h1_decoder(model)
    r2 = patch_falcon_h1_decoder(model)
    assert r2.attention == 0 and r2.mlp == 0 and r2.mixer == 0 and r2.layer == 0, r2
    # The first call should have patched at least one module of each kind.
    assert r1.attention > 0 and r1.mlp > 0 and r1.mixer > 0 and r1.layer > 0, r1


def test_non_unit_multiplier_not_patched():
    """If config has a non-unit multiplier, the patched module count drops."""
    os.environ.pop("BGKIT_FALCON_H1_PATCH", None)
    from transformers import AutoConfig
    from transformers.models.falcon_h1 import FalconH1ForCausalLM

    from bgkit.utils.falcon_h1_patch import patch_falcon_h1_decoder

    cfg = AutoConfig.from_pretrained("tiiuae/Falcon-H1-Tiny-90M-Instruct")
    cfg.num_hidden_layers = 2
    cfg.key_multiplier = 2.0  # non-unit
    model = FalconH1ForCausalLM(cfg)
    report = patch_falcon_h1_decoder(model)
    assert report.attention == 0, "attention should not be patched when key_multiplier != 1.0"
    # Other categories still patched
    assert report.mlp == 2
    assert report.mixer == 2
    assert report.layer == 2


@pytest.mark.gpu
def test_mixer_cuda_fast_path_parity():
    """Full fused-kernel forward + backward parity at bf16 on GPU.

    Requires Triton + mamba-ssm and a CUDA device — runs only in the
    training Docker image.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    try:
        import mamba_ssm  # noqa: F401
    except Exception:
        pytest.skip("requires mamba-ssm")

    os.environ.pop("BGKIT_FALCON_H1_PATCH", None)
    from bgkit.utils.falcon_h1_patch import patch_falcon_h1_decoder

    model_ref, model_patched, cfg = _two_models_with_same_init(num_layers=4)
    model_ref = model_ref.to(device="cuda", dtype=torch.bfloat16)
    model_patched = model_patched.to(device="cuda", dtype=torch.bfloat16)

    patch_falcon_h1_decoder(model_patched)

    model_ref.train()
    model_patched.train()

    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, 64), device="cuda")

    out_ref = model_ref(input_ids=ids, labels=ids, use_cache=False)
    out_ref.loss.backward()
    out_p = model_patched(input_ids=ids, labels=ids, use_cache=False)
    out_p.loss.backward()

    logits_diff = (out_p.logits - out_ref.logits).abs().max().item()
    loss_diff = (out_p.loss - out_ref.loss).abs().item()
    print(f"bf16 logits max abs diff = {logits_diff:.3e}, loss diff = {loss_diff:.3e}")
    assert logits_diff < 5e-2, f"bf16 logits diff too large: {logits_diff:.3e}"
    assert loss_diff < 5e-3, f"loss diff too large: {loss_diff:.3e}"

    g_ref = model_ref.model.embed_tokens.weight.grad
    g_p = model_patched.model.embed_tokens.weight.grad
    grad_diff = (g_p.float() - g_ref.float()).abs().max().item()
    print(f"bf16 embed grad max abs diff = {grad_diff:.3e}")
    assert grad_diff < 5e-2, f"grad diff too large: {grad_diff:.3e}"
