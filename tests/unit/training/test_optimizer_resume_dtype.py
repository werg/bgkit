"""Regression tests for the 2026-06-28 fixes:

1. KRKBTrainer (bf16 model + plain AdamW) resume crash: ``_restore_opt_tensor``
   used to unconditionally upcast saved optimizer moments bf16->fp32, but a
   fresh bf16-AdamW run keeps bf16 moments, so on resume ``exp_avg.lerp_(grad)``
   mixed fp32 state with bf16 grad -> ``RuntimeError: expected dtype float for
   'end'``. The fix restores moments to the dtype a FRESH optimizer would use
   (param.dtype for plain AdamW; fp32 for Muon, which also self-heals).

2. ``_resolve_one_epoch_max_steps``: the ``max_steps: null`` one-epoch path now
   supports a ``training.epochs`` override that divides by accum_steps (TRUE
   epochs), while preserving the legacy (batch-count) value when unset.
"""

import math

import pytest
import torch

from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer


# ---------------------------------------------------------------------------
# FIX 1: optimizer-state restore dtype
# ---------------------------------------------------------------------------

def test_restore_opt_tensor_casts_bf16_to_target():
    dev = torch.device("cpu")
    bf = torch.zeros(3, dtype=torch.bfloat16)
    # default target fp32 (Muon / back-compat)
    assert BaseTrainer._restore_opt_tensor(bf, dev).dtype == torch.float32
    # explicit bf16 target (plain AdamW on a bf16 model)
    assert BaseTrainer._restore_opt_tensor(bf, dev, torch.bfloat16).dtype == torch.bfloat16
    # fp16 also follows the target
    fp16 = torch.zeros(3, dtype=torch.float16)
    assert BaseTrainer._restore_opt_tensor(fp16, dev, torch.bfloat16).dtype == torch.bfloat16


def test_restore_opt_tensor_protects_fp32_scalars_and_nontensors():
    dev = torch.device("cpu")
    # An fp32 buffer (e.g. AdamW's `step` scalar) is NOT in {bf16,fp16}, so it is
    # left untouched even when the target is bf16 — never corrupt the step count.
    step = torch.tensor(7.0, dtype=torch.float32)
    out = BaseTrainer._restore_opt_tensor(step, dev, torch.bfloat16)
    assert out.dtype == torch.float32 and float(out) == 7.0
    # non-tensors pass through
    assert BaseTrainer._restore_opt_tensor(5, dev, torch.bfloat16) == 5


def test_fresh_opt_state_float_dtype_by_optimizer():
    # adamw -> param dtype; anything else (muon/default) -> fp32
    obj = KRKBTrainer.__new__(KRKBTrainer)
    p_bf16 = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    p_fp32 = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))

    obj._optimizer_type = "adamw"
    assert obj._fresh_opt_state_float_dtype(p_bf16) == torch.bfloat16
    assert obj._fresh_opt_state_float_dtype(p_fp32) == torch.float32

    obj._optimizer_type = "muon"
    assert obj._fresh_opt_state_float_dtype(p_bf16) == torch.float32
    assert obj._fresh_opt_state_float_dtype(p_fp32) == torch.float32


def _adamw_state_after_step(dtype):
    """Build a real AdamW per-param state by taking one step on a `dtype` param."""
    p = torch.nn.Parameter(torch.randn(4, 4, dtype=dtype))
    opt = torch.optim.AdamW([p], lr=1e-3)
    p.grad = torch.randn(4, 4, dtype=dtype)
    opt.step()
    return p, opt.state[p]


def test_old_upcast_reproduces_bf16_adamw_crash():
    """Sanity: the OLD unconditional bf16->fp32 upcast crashes bf16 AdamW."""
    p, state = _adamw_state_after_step(torch.bfloat16)
    bad = {
        k: (v.to(torch.float32) if isinstance(v, torch.Tensor)
            and v.is_floating_point() and v.dtype in (torch.bfloat16, torch.float16)
            else v)
        for k, v in state.items()
    }
    opt = torch.optim.AdamW([p], lr=1e-3)
    opt.state[p] = bad
    p.grad = torch.randn(4, 4, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="expected dtype"):
        opt.step()


def test_new_restore_fixes_bf16_adamw_resume():
    """The fix (restore to param.dtype) lets bf16 AdamW resume + step cleanly."""
    p, state = _adamw_state_after_step(torch.bfloat16)
    # round-trip through the save downcast (already bf16 -> stays bf16) + restore
    restored = {
        k: BaseTrainer._restore_opt_tensor(v, p.device, p.dtype)
        for k, v in state.items()
    }
    for k, v in restored.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point() and v.numel() > 1:
            assert v.dtype == torch.bfloat16, f"{k} should match bf16 param"
    opt = torch.optim.AdamW([p], lr=1e-3)
    opt.state[p] = restored
    p.grad = torch.randn(4, 4, dtype=torch.bfloat16)
    opt.step()  # must NOT raise


def test_fp32_adamw_resume_unaffected():
    """fp32 model + AdamW: target fp32, moments stay fp32, step works."""
    p, state = _adamw_state_after_step(torch.float32)
    restored = {
        k: BaseTrainer._restore_opt_tensor(v, p.device, p.dtype)  # target fp32
        for k, v in state.items()
    }
    opt = torch.optim.AdamW([p], lr=1e-3)
    opt.state[p] = restored
    p.grad = torch.randn(4, 4, dtype=torch.float32)
    opt.step()


def test_muon_path_keeps_fp32_master():
    """Muon's fp32 master saved as bf16 must restore to fp32 (default target)."""
    master_bf16 = torch.zeros(4, 4, dtype=torch.bfloat16)  # downcast-on-disk form
    out = BaseTrainer._restore_opt_tensor(master_bf16, torch.device("cpu"))  # default fp32
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# FIX 2: one-epoch / true-epoch max_steps resolution
# ---------------------------------------------------------------------------

def test_one_epoch_legacy_preserved_when_epochs_unset():
    # epochs=None -> exactly n_batches (unchanged legacy behavior)
    assert KRKBTrainer._resolve_one_epoch_max_steps(1163, 4, None) == 1163
    assert KRKBTrainer._resolve_one_epoch_max_steps(3387, 4, None) == 3387


def test_true_epochs_divides_by_accum():
    # epochs set -> epochs * ceil(n_batches / accum_steps)
    assert KRKBTrainer._resolve_one_epoch_max_steps(1163, 4, 1) == math.ceil(1163 / 4)
    assert KRKBTrainer._resolve_one_epoch_max_steps(1163, 4, 3) == 3 * math.ceil(1163 / 4)
    # accum 1 -> epochs * n_batches (no division)
    assert KRKBTrainer._resolve_one_epoch_max_steps(1000, 1, 2) == 2000
    # the legacy value == accum_steps real epochs of the fixed version
    legacy = KRKBTrainer._resolve_one_epoch_max_steps(1163, 4, None)
    one_true = KRKBTrainer._resolve_one_epoch_max_steps(1163, 4, 1)
    assert legacy == 1163 and one_true == 291  # ~4x fewer steps
