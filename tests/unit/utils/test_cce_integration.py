"""Unit tests for optional cut-cross-entropy decoder integration."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn
import torch.nn.functional as F

from bgkit.utils import cce_integration
from bgkit.utils.cce_integration import cce_labels_from_masks, cut_cross_entropy_lm_ce


def _reference_ce(
    hidden_states: torch.Tensor,
    lm_head: nn.Linear,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor | None,
) -> torch.Tensor:
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
    mask = attention_mask[:, 1:].to(dtype=torch.bool)
    if loss_mask is not None:
        mask = mask & loss_mask[:, 1:].to(dtype=torch.bool)
    weights = mask.to(dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp(min=1)


@pytest.fixture(autouse=True)
def _reset_cce_state():
    cce_integration._CCE_AVAILABLE = None
    cce_integration._CCE_WARNED = False
    cce_integration._CCE_RUNTIME_WARNED = False
    yield
    cce_integration._CCE_AVAILABLE = None
    cce_integration._CCE_WARNED = False
    cce_integration._CCE_RUNTIME_WARNED = False


def test_cce_labels_from_masks_sets_ignore_index_before_shift():
    labels = torch.arange(10).view(2, 5)
    attention_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
        ]
    )
    loss_mask = torch.tensor(
        [
            [False, True, True, True, True],
            [True, False, True, False, True],
        ]
    )

    out = cce_labels_from_masks(labels, attention_mask, loss_mask, ignore_index=-100)

    expected = torch.tensor(
        [
            [-100, 1, 2, -100, -100],
            [5, -100, 7, -100, 9],
        ]
    )
    torch.testing.assert_close(out, expected)


def test_cut_cross_entropy_fallback_matches_reference_on_cpu(monkeypatch):
    monkeypatch.setattr(cce_integration, "_try_import_linear_cross_entropy", lambda: None)
    torch.manual_seed(0)
    b, s, d, v = 2, 12, 8, 32
    hidden = torch.randn(b, s, d, requires_grad=True)
    labels = torch.randint(0, v, (b, s))
    head = nn.Linear(d, v, bias=True)
    attention_mask = torch.ones(b, s, dtype=torch.bool)
    loss_mask = torch.ones(b, s, dtype=torch.bool)
    loss_mask[:, :3] = False
    loss_mask[0, -2:] = False

    loss = cut_cross_entropy_lm_ce(
        hidden_states=hidden,
        lm_head_weight=head.weight,
        lm_head_bias=head.bias,
        labels=labels,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        impl="cce",
        chunk_size=4,
    )
    ref = _reference_ce(hidden, head, labels, attention_mask, loss_mask)
    torch.testing.assert_close(loss, ref, atol=1e-5, rtol=1e-4)


def test_decoder_ce_impl_setter_validates_values():
    from bgkit.models.decoder import ReconstructionDecoder

    dec = ReconstructionDecoder.__new__(ReconstructionDecoder)
    nn.Module.__init__(dec)
    dec.set_lm_ce_impl("cce_exact")
    assert dec._lm_ce_impl == "cce_exact"
    with pytest.raises(ValueError, match="Unsupported decoder CE implementation"):
        dec.set_lm_ce_impl("not-a-ce")


def test_decoder_ce_impl_default_is_cce():
    from bgkit.models.decoder import DEFAULT_LM_CE_IMPL

    assert DEFAULT_LM_CE_IMPL == "cce"
