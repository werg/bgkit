"""Tests for attention_backend — packed path only.

The old mask-based tests (test_bgkit_fa4_forward_uses_padding_mask_path,
test_bgkit_fa4_forward_rejects_query_specific_masks,
test_bgkit_fa4_forward_keeps_true_gqa_when_sm12x_native_ready) were removed
when the mask path was deleted in the Wave-0 packed migration.

Remaining coverage:
- resolve_attention_implementation (strict FA-only)
- SM12x owned-backend fail-fast guard
- output_attentions rejection
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from bgkit.utils import attention_backend as ab


@pytest.fixture(autouse=True)
def _reset_sm12x_owned_backend_flag():
    """Reset the per-process owned-backend success cache between tests.

    ``require_sm12x_owned_backend`` memoizes its success result so the hot
    per-attention-layer dispatch path is a single attribute read. Tests that
    monkey-patch the backend state need to force a re-probe each call.
    """
    ab._sm12x_owned_backend_ok = False
    yield
    ab._sm12x_owned_backend_ok = False


# ---------------------------------------------------------------------------
# resolve_attention_implementation
# ---------------------------------------------------------------------------


def test_resolve_attention_implementation_auto_prefers_bgkit_fa4():
    with (
        patch.object(ab, "install_bgkit_attention_backend", return_value=True),
        patch.object(ab, "require_sm12x_owned_backend", return_value=None),
    ):
        assert ab.resolve_attention_implementation("auto") == ab.BGKIT_FA4_ATTENTION_IMPL


def test_resolve_attention_implementation_auto_requires_fa4():
    with patch.object(ab, "install_bgkit_attention_backend", return_value=False):
        try:
            ab.resolve_attention_implementation("auto")
        except RuntimeError as exc:
            assert "strict FlashAttention-only" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected strict auto resolution to fail without FA4")


def test_resolve_attention_implementation_explicit_fa4_requires_install():
    with patch.object(ab, "install_bgkit_attention_backend", return_value=False):
        try:
            ab.resolve_attention_implementation("flash_attention_4")
        except RuntimeError as exc:
            assert "flash_attn.cute" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected explicit FA4 request to fail without FA4")


def test_resolve_attention_implementation_unknown_raises():
    try:
        ab.resolve_attention_implementation("some_unknown_backend")
    except ValueError as exc:
        assert "Unsupported attention implementation" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected unknown backend to raise ValueError")


def test_require_sm12x_owned_backend_noops_off_cuda():
    with patch.object(ab.torch.cuda, "is_available", return_value=False):
        ab.require_sm12x_owned_backend()


def test_require_sm12x_owned_backend_rejects_unowned_sm12x_backend():
    with (
        patch.object(ab.torch.cuda, "is_available", return_value=True),
        patch.object(ab.torch.cuda, "get_device_capability", return_value=(12, 1)),
        patch.dict(
            "sys.modules",
            {
                "flash_attn.cute.native_sm12x": SimpleNamespace(
                    native_sm12x_backend_kind=lambda: "aten",
                    native_sm12x_owned_backend_available=lambda: False,
                )
            },
        ),
    ):
        try:
            ab.require_sm12x_owned_backend()
        except RuntimeError as exc:
            assert "backend_kind='aten'" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected unowned SM12x backend to fail fast")


def test_require_sm12x_owned_backend_accepts_owned_sm12x_backend():
    with (
        patch.object(ab.torch.cuda, "is_available", return_value=True),
        patch.object(ab.torch.cuda, "get_device_capability", return_value=(12, 1)),
        patch.dict(
            "sys.modules",
            {
                "flash_attn.cute.native_sm12x": SimpleNamespace(
                    native_sm12x_backend_kind=lambda: "flash_attn",
                    native_sm12x_owned_backend_available=lambda: True,
                )
            },
        ),
    ):
        ab.require_sm12x_owned_backend()


# ---------------------------------------------------------------------------
# bgkit_flash_attention_4_forward — output_attentions rejection
# ---------------------------------------------------------------------------


def test_bgkit_fa4_forward_rejects_output_attentions():
    class _DummyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_attn_implementation=ab.BGKIT_FA4_ATTENTION_IMPL)
            self.is_causal = False

    module = _DummyModule()
    n, h, d = 10, 2, 8
    q = torch.randn(n, h, d)
    k = torch.randn(n, h, d)
    v = torch.randn(n, h, d)
    cu_seqlens = torch.tensor([0, 5, 10], dtype=torch.int32)

    try:
        ab.bgkit_flash_attention_4_forward(
            module,
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=5,
            output_attentions=True,
        )
    except NotImplementedError as exc:
        assert "output_attentions" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected output_attentions to fail fast")
