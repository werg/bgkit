"""Unit tests for :mod:`bgkit.utils.liger_integration`.

These tests run on CPU and force the no-Liger paths where needed. They exercise:

- the no-op behaviour of ``apply_liger_to_qwen35`` when the package is
  absent (should warn once, return 0, never raise);
- the fallback behaviour of ``liger_chunked_ce_loss`` which transparently
  delegates to the existing ``_chunked_lm_ce`` path in decoder.py when the
  Liger fused-CE kernel isn't importable;
- numerical equivalence between the fallback wrapper and a straight
  ``F.cross_entropy`` reference on a small synthetic batch.
"""

from __future__ import annotations

import warnings

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn
import torch.nn.functional as F

from bgkit.utils import liger_integration
from bgkit.utils.liger_integration import (
    apply_liger_to_qwen35,
    is_liger_available,
    liger_chunked_ce_loss,
)


@pytest.fixture(autouse=True)
def _reset_availability_cache():
    """Force ``is_liger_available`` to re-probe between tests so we can
    monkey-patch the import result independently per test."""
    liger_integration._LIGER_AVAILABLE = None
    liger_integration._LIGER_WARNED = False
    yield
    liger_integration._LIGER_AVAILABLE = None
    liger_integration._LIGER_WARNED = False


# ---------------------------------------------------------------------------
# apply_liger_to_qwen35 — no-op fallback
# ---------------------------------------------------------------------------


class TestApplyLigerToQwen35Fallback:
    def test_no_liger_installed_returns_zero(self):
        """Without liger-kernel installed the helper must return 0 cleanly."""
        liger_integration._LIGER_AVAILABLE = False
        assert not is_liger_available()
        model = nn.Linear(4, 4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            n_patched = apply_liger_to_qwen35(model)
        assert n_patched == 0
        # Warning is emitted once per process; first call should produce it.
        assert any("liger-kernel not installed" in str(w.message) for w in caught)

    def test_no_liger_warning_is_one_shot(self):
        """Calling twice without liger should still just return 0 and not
        crash. The warning is throttled after the first emission."""
        liger_integration._LIGER_AVAILABLE = False
        model = nn.Linear(4, 4)
        apply_liger_to_qwen35(model)  # triggers the warning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            n_patched = apply_liger_to_qwen35(model)
        assert n_patched == 0
        # No duplicate warning on the second call (module-level flag is set).
        assert not any(
            "liger-kernel not installed" in str(w.message) for w in caught
        )

    def test_none_model_is_noop(self):
        """Passing None (e.g. no encoder yet) must not raise."""
        assert apply_liger_to_qwen35(None) == 0

    def test_iter_text_backbone_handles_common_wrappers(self):
        """The inner traversal helper should find a text backbone across
        the wrappers the real trainers produce."""
        from bgkit.utils.liger_integration import _iter_text_backbone

        # Plain module passes through unchanged.
        m = nn.Linear(4, 4)
        assert _iter_text_backbone(m) is m

        # ``.backbone`` wrapper (ReconstructionDecoder pattern)
        class Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(4, 4)

        w = Wrap()
        assert _iter_text_backbone(w) is w.backbone

    def test_patch_rmsnorm_default_is_false(self):
        """Regression: apply_liger_to_qwen35's ``patch_rmsnorm`` kwarg must
        default to False. liger-kernel 0.7.x's LigerRMSNorm silently
        corrupts Qwen3.5 backward (decoder loss jumps to the LM prior at
        near-zero LR); the safe default is off. Callers that need the
        kernel must opt in explicitly.

        Flipping this default back to True without independently verifying
        the kernel is good against the current transformers + Qwen3.5
        combination will silently break training. Keep the test — the
        false-positive cost is a one-line flip; the miss cost is another
        24-hour debug session.
        """
        import inspect

        sig = inspect.signature(apply_liger_to_qwen35)
        assert sig.parameters["patch_rmsnorm"].default is False
        # SwiGLU / RoPE are unaffected by the 0.7.x regression — they
        # should still default to on so the throughput win is preserved.
        assert sig.parameters["patch_swiglu"].default is True
        assert sig.parameters["patch_rope"].default is True


# ---------------------------------------------------------------------------
# liger_chunked_ce_loss — fallback path
# ---------------------------------------------------------------------------


def _reference_ce(
    hidden_states: torch.Tensor,
    lm_head: nn.Linear,
    labels: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Reference next-token CE loss using materialized logits."""
    shift_h = hidden_states[:, :-1, :]
    shift_y = labels[:, 1:]
    logits = lm_head(shift_h)
    b, s, v = logits.shape
    losses = F.cross_entropy(
        logits.reshape(b * s, v),
        shift_y.reshape(b * s),
        reduction="none",
        ignore_index=-100,
    ).view(b, s)
    m = mask[:, 1:].float() if mask is not None else torch.ones_like(losses)
    return (losses * m).sum() / m.sum().clamp(min=1)


class TestLigerChunkedCELossFallback:
    def test_fallback_matches_reference_no_mask(self):
        """Without liger installed the wrapper should defer to _chunked_lm_ce
        and produce a loss that matches a plain F.cross_entropy reference."""
        torch.manual_seed(0)
        b, s, d, v = 2, 16, 8, 32
        hidden = torch.randn(b, s, d, requires_grad=True)
        labels = torch.randint(0, v, (b, s))
        head = nn.Linear(d, v, bias=False)

        loss = liger_chunked_ce_loss(
            hidden_states=hidden,
            lm_head_weight=head.weight,
            lm_head_bias=None,
            labels=labels,
            mask=None,
            chunk_size=8,
        )
        ref = _reference_ce(hidden, head, labels, None)
        assert torch.allclose(loss, ref, atol=1e-5, rtol=1e-4), (
            f"liger fallback loss {loss.item()} diverged from reference {ref.item()}"
        )

    def test_fallback_matches_reference_with_mask(self):
        """Masked positions should be excluded from both numerator and
        denominator; wrapper output must track the reference."""
        torch.manual_seed(1)
        b, s, d, v = 3, 12, 6, 24
        hidden = torch.randn(b, s, d, requires_grad=True)
        labels = torch.randint(0, v, (b, s))
        head = nn.Linear(d, v, bias=False)

        mask = torch.ones(b, s, dtype=torch.bool)
        mask[:, :2] = False  # mask first two positions (prompt prefix)
        mask[0, -3:] = False  # and the tail of the first example

        loss = liger_chunked_ce_loss(
            hidden_states=hidden,
            lm_head_weight=head.weight,
            lm_head_bias=None,
            labels=labels,
            mask=mask,
            chunk_size=4,
        )
        ref = _reference_ce(hidden, head, labels, mask)
        assert torch.allclose(loss, ref, atol=1e-5, rtol=1e-4)

    def test_fallback_backward_grads_flow(self):
        """Fallback path must produce non-zero gradients on the hidden
        states — confirming the chunked wrapper actually participates in
        autograd rather than detaching."""
        torch.manual_seed(2)
        b, s, d, v = 1, 10, 4, 16
        hidden = torch.randn(b, s, d, requires_grad=True)
        labels = torch.randint(0, v, (b, s))
        head = nn.Linear(d, v, bias=False)

        loss = liger_chunked_ce_loss(
            hidden_states=hidden,
            lm_head_weight=head.weight,
            lm_head_bias=None,
            labels=labels,
            mask=None,
            chunk_size=4,
        )
        loss.backward()
        assert hidden.grad is not None
        assert hidden.grad.abs().sum().item() > 0

    def test_decoder_hook_enables_liger_ce_flag(self):
        """`ReconstructionDecoder.enable_liger_ce` should toggle the internal
        flag that ``_compute_lm_ce`` consults. Without Liger installed the
        code still executes the fallback path, so this test just asserts the
        attribute flip."""
        from bgkit.models.decoder import ReconstructionDecoder

        dec = ReconstructionDecoder.__new__(ReconstructionDecoder)
        nn.Module.__init__(dec)
        dec._use_liger_ce = False
        dec.enable_liger_ce(True)
        assert dec._use_liger_ce is True
        dec.enable_liger_ce(False)
        assert dec._use_liger_ce is False
