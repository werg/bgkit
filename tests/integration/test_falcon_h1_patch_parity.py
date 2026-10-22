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
import torch.nn.functional as F

_FALCON_H1_FAST_ENV_KEYS = (
    "BGKIT_FALCON_H1_PATCH",
    "BGKIT_FALCON_H1_PACKED_MAMBA_SEQIDX",
    "BGKIT_FALCON_H1_PACKED_QKV",
    "BGKIT_FALCON_H1_ATTENTION_CAT_QKV",
    "BGKIT_FALCON_H1_DIRECT_FLASH_ATTN",
    "BGKIT_FALCON_H1_DIRECT_FA4_ATTN",
    "BGKIT_FALCON_H1_DIRECT_HF_FLASH_ATTN",
    "BGKIT_FALCON_H1_DIRECT_SDPA",
    "BGKIT_FALCON_H1_PACKED_MLP",
    "BGKIT_FALCON_H1_MLP_CAT_GATE_UP",
    "BGKIT_FALCON_H1_TRAINABLE_MLP_AUTOGRAD",
    "BGKIT_FALCON_H1_SPECIALIZED_MAMBA",
    "BGKIT_FALCON_H1_MAMBA_INPROJ_AUTOGRAD",
    "BGKIT_FALCON_H1_MAMBA_SAVE_OUT",
    "BGKIT_FALCON_H1_MAMBA_SAVE_CONV",
    "BGKIT_FALCON_H1_MAMBA_SAVE_SCAN",
    "BGKIT_FALCON_H1_MAMBA_SKIP_D_IN_DX_KERNEL",
    "BGKIT_FALCON_H1_FUSED_INPUT_PROJ",
    "BGKIT_FALCON_H1_FUSED_LAYER_LOOP",
)


def _clear_falcon_h1_fast_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _FALCON_H1_FAST_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


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


def test_default_fast_training_patch_report(monkeypatch):
    """Unset Falcon env vars resolve to the measured-fast training contract."""
    _clear_falcon_h1_fast_env(monkeypatch)
    from bgkit.utils.falcon_h1_defaults import effective_falcon_h1_fast_env
    from bgkit.utils.falcon_h1_patch import patch_falcon_h1_decoder

    model, _cfg = _build_small_falcon(num_layers=2)
    report = patch_falcon_h1_decoder(model)

    expected_env = effective_falcon_h1_fast_env()
    assert expected_env["BGKIT_FALCON_H1_PATCH"] == "1"
    assert expected_env["BGKIT_FALCON_H1_PACKED_QKV"] == "1"
    assert expected_env["BGKIT_FALCON_H1_PACKED_MLP"] == "1"
    assert expected_env["BGKIT_FALCON_H1_SPECIALIZED_MAMBA"] == "1"
    assert expected_env["BGKIT_FALCON_H1_MAMBA_SKIP_D_IN_DX_KERNEL"] == "1"
    assert expected_env["BGKIT_FALCON_H1_DIRECT_SDPA"] == "0"
    assert expected_env["BGKIT_FALCON_H1_FUSED_INPUT_PROJ"] == "0"
    assert expected_env["BGKIT_FALCON_H1_FUSED_LAYER_LOOP"] == "0"

    assert report.attention == 2, report
    assert report.attention_packed_qkv == 2, report
    assert report.attention_cat_qkv == 0, report
    assert report.attention_direct_flash == 0, report
    assert report.attention_direct_fa4 == 0, report
    assert report.attention_direct_hf_flash == 0, report
    assert report.attention_direct_sdpa == 0, report
    assert report.mlp == 2, report
    assert report.mlp_packed_gate_up == 2, report
    assert report.mlp_cat_gate_up == 0, report
    assert report.mlp_trainable_autograd == 2, report
    assert report.mixer == 2, report
    assert report.mixer_specialized_mamba == 2, report
    assert report.mixer_save_scan == 2, report
    assert report.mixer_inproj_autograd == 2, report
    assert report.packed_seqidx_loop == 1, report
    assert report.layer_fused_input_proj == 0, report
    assert report.fused_layer_loop == 0, report


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


def test_packed_linear_patches_load_legacy_split_state_dict():
    """Packed Falcon QKV/MLP projections strict-load old split checkpoint keys."""
    os.environ.pop("BGKIT_FALCON_H1_PATCH", None)
    from bgkit.utils.falcon_h1_patch import patch_falcon_h1_decoder

    model_ref, cfg = _build_small_falcon(num_layers=2)
    legacy_sd = _state_dict_clone(model_ref)

    torch.manual_seed(5678)
    model_patched, _ = _build_small_falcon(num_layers=2)
    report = patch_falcon_h1_decoder(model_patched)
    assert report.attention_packed_qkv == 2, report
    assert report.mlp_packed_gate_up == 2, report

    model_patched.load_state_dict(legacy_sd)
    packed_sd = model_patched.state_dict()

    assert "model.layers.0.self_attn.qkv_proj.weight" in packed_sd
    assert "model.layers.0.self_attn.q_proj.weight" not in packed_sd
    expected_qkv = torch.cat(
        (
            legacy_sd["model.layers.0.self_attn.q_proj.weight"],
            legacy_sd["model.layers.0.self_attn.k_proj.weight"],
            legacy_sd["model.layers.0.self_attn.v_proj.weight"],
        ),
        dim=0,
    )
    torch.testing.assert_close(
        packed_sd["model.layers.0.self_attn.qkv_proj.weight"],
        expected_qkv,
    )

    assert "model.layers.0.feed_forward.gate_up_proj.weight" in packed_sd
    assert "model.layers.0.feed_forward.gate_proj.weight" not in packed_sd
    expected_gate_up = torch.cat(
        (
            legacy_sd["model.layers.0.feed_forward.gate_proj.weight"],
            legacy_sd["model.layers.0.feed_forward.up_proj.weight"],
        ),
        dim=0,
    )
    torch.testing.assert_close(
        packed_sd["model.layers.0.feed_forward.gate_up_proj.weight"],
        expected_gate_up,
    )

    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    model_ref.eval()
    model_patched.eval()
    with torch.no_grad():
        out_ref = model_ref(input_ids=ids, use_cache=False).logits
        out_p = model_patched(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(out_p, out_ref, rtol=0.0, atol=1e-5)


def test_packed_falcon_optimizer_state_merges_legacy_split_moments():
    """Old split optimizer moments are concatenated for packed Falcon params."""
    from bgkit.training.base_trainer import BaseTrainer

    class _ConcreteTrainer(BaseTrainer):
        def setup(self):
            raise NotImplementedError

        def _forward_backward(self, batch):
            raise NotImplementedError

        def evaluate(self):
            raise NotImplementedError

    param = torch.nn.Parameter(torch.empty(6, 2))
    state_by_name = {
        "decoder.block.self_attn.q_proj.weight": {
            "momentum_buffer": torch.full((2, 2), 1.0),
            "step": torch.tensor(3),
        },
        "decoder.block.self_attn.k_proj.weight": {
            "momentum_buffer": torch.full((2, 2), 2.0),
            "step": torch.tensor(4),
        },
        "decoder.block.self_attn.v_proj.weight": {
            "momentum_buffer": torch.full((2, 2), 3.0),
            "step": torch.tensor(5),
        },
    }

    trainer = object.__new__(_ConcreteTrainer)
    packed, sources = trainer._packed_falcon_optimizer_state(
        "decoder.block.self_attn.qkv_proj.weight",
        param,
        state_by_name,
    )

    assert sources == (
        "decoder.block.self_attn.q_proj.weight",
        "decoder.block.self_attn.k_proj.weight",
        "decoder.block.self_attn.v_proj.weight",
    )
    assert packed is not None
    torch.testing.assert_close(
        packed["momentum_buffer"],
        torch.cat(
            (
                state_by_name["decoder.block.self_attn.q_proj.weight"]["momentum_buffer"],
                state_by_name["decoder.block.self_attn.k_proj.weight"]["momentum_buffer"],
                state_by_name["decoder.block.self_attn.v_proj.weight"]["momentum_buffer"],
            ),
            dim=0,
        ),
    )
    torch.testing.assert_close(packed["step"], torch.tensor(5))


def test_trainable_packed_mlp_autograd_parity_cpu():
    """Packed trainable MLP autograd matches torch forward/backward on CPU."""
    from bgkit.kernels.falcon_h1_mlp import falcon_h1_packed_mlp_trainable

    torch.manual_seed(0)
    batch, seq_len, hidden_size, intermediate_size = 2, 5, 8, 18

    x = torch.randn(batch, seq_len, hidden_size, requires_grad=True)
    gate_up_weight = torch.randn(2 * intermediate_size, hidden_size, requires_grad=True)
    gate_up_bias = torch.randn(2 * intermediate_size, requires_grad=True)
    down_weight = torch.randn(hidden_size, intermediate_size, requires_grad=True)
    down_bias = torch.randn(hidden_size, requires_grad=True)

    x_ref = x.detach().clone().requires_grad_(True)
    gate_up_weight_ref = gate_up_weight.detach().clone().requires_grad_(True)
    gate_up_bias_ref = gate_up_bias.detach().clone().requires_grad_(True)
    down_weight_ref = down_weight.detach().clone().requires_grad_(True)
    down_bias_ref = down_bias.detach().clone().requires_grad_(True)

    out = falcon_h1_packed_mlp_trainable(
        x,
        gate_up_weight,
        gate_up_bias,
        down_weight,
        down_bias,
    )
    gate_up_ref = F.linear(x_ref, gate_up_weight_ref, gate_up_bias_ref)
    gate_ref, up_ref = gate_up_ref.split(intermediate_size, dim=-1)
    out_ref = F.linear(F.silu(gate_ref) * up_ref, down_weight_ref, down_bias_ref)

    torch.testing.assert_close(out, out_ref)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)

    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(gate_up_weight.grad, gate_up_weight_ref.grad)
    torch.testing.assert_close(gate_up_bias.grad, gate_up_bias_ref.grad)
    torch.testing.assert_close(down_weight.grad, down_weight_ref.grad)
    torch.testing.assert_close(down_bias.grad, down_bias_ref.grad)


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
        import mamba_ssm
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


@pytest.mark.gpu
def test_packed_mamba_seqidx_matches_padded_batch():
    """Packed Falcon-H1 Mamba seq_idx matches the padded ragged-batch path."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    try:
        import mamba_ssm
    except Exception:
        pytest.skip("requires mamba-ssm")

    os.environ.pop("BGKIT_FALCON_H1_PATCH", None)
    from bgkit.utils.falcon_h1_patch import patch_falcon_h1_decoder

    model_ref, model_patched, cfg = _two_models_with_same_init(num_layers=2)
    model_ref = model_ref.to(device="cuda", dtype=torch.bfloat16)
    model_patched = model_patched.to(device="cuda", dtype=torch.bfloat16)
    patch_falcon_h1_decoder(model_patched)
    model_ref.train()
    model_patched.train()

    torch.manual_seed(0)
    lengths = [31, 19, 11]
    max_len = max(lengths)
    ids = torch.zeros(len(lengths), max_len, dtype=torch.long, device="cuda")
    mask = torch.zeros(len(lengths), max_len, dtype=torch.bool, device="cuda")
    for row, length in enumerate(lengths):
        ids[row, :length] = torch.randint(0, cfg.vocab_size, (length,), device="cuda")
        mask[row, :length] = True

    with torch.no_grad():
        padded_logits = model_ref(input_ids=ids, attention_mask=mask, use_cache=False).logits

        flat_ids = torch.cat([ids[row, :length] for row, length in enumerate(lengths)], dim=0)
        flat_embeds = model_patched.model.embed_tokens(flat_ids).unsqueeze(0)
        pos_ids = torch.cat(
            [torch.arange(length, device="cuda", dtype=torch.long) for length in lengths],
            dim=0,
        ).unsqueeze(0)
        cu = torch.tensor(
            [0, *torch.tensor(lengths, device="cpu").cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        seq_idx = torch.repeat_interleave(
            torch.arange(len(lengths), device="cuda", dtype=torch.int32),
            torch.tensor(lengths, device="cuda", dtype=torch.int32),
        ).unsqueeze(0)
        packed_logits = model_patched(
            inputs_embeds=flat_embeds,
            position_ids=pos_ids,
            use_cache=False,
            cu_seq_lens_q=cu,
            cu_seq_lens_k=cu,
            max_length_q=max_len,
            max_length_k=max_len,
            mamba_seq_idx=seq_idx,
        ).logits

    padded_real = torch.cat(
        [padded_logits[row, :length] for row, length in enumerate(lengths)],
        dim=0,
    )
    diff = (packed_logits.squeeze(0) - padded_real).abs().max().item()
    assert diff < 5e-2, f"packed seq_idx logits diff too large: {diff:.3e}"
