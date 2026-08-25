"""Held-out plain-text language health metric (2026-08-25).

Exists because the wide-net runs took the decoder from PPL 33 to 2585 with
no in-distribution metric noticing; this is the format-free instrument that
catches it.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from bgkit.training.lm_health import lm_health_metrics


class _FakeOut:
    def __init__(self, logits):
        self.logits = logits


class _FakeLM(torch.nn.Module):
    """Predicts the NEXT id with confidence controlled by ``sharpness``."""

    def __init__(self, vocab: int = 16, sharpness: float = 12.0):
        super().__init__()
        self.vocab = vocab
        self.sharpness = sharpness

    def forward(self, input_ids=None, **kw):
        b, s = input_ids.shape
        logits = torch.zeros(b, s, self.vocab)
        nxt = torch.roll(input_ids, shifts=-1, dims=1)
        logits.scatter_(2, nxt.unsqueeze(-1), self.sharpness)
        return _FakeOut(logits)


def test_metrics_track_prediction_quality_and_expose_perplexity():
    ids = torch.arange(8, dtype=torch.long) % 16
    sharp = lm_health_metrics(_FakeLM(sharpness=12.0), [ids], torch.device("cpu"))
    blunt = lm_health_metrics(_FakeLM(sharpness=0.0), [ids], torch.device("cpu"))
    assert sharp["eval/lm_health/ce"] < blunt["eval/lm_health/ce"]
    # A uniform model over V=16 must sit at ln(16) — the metric is a real CE,
    # not an arbitrary score.
    assert blunt["eval/lm_health/ce"] == pytest.approx(math.log(16), abs=1e-4)
    assert blunt["eval/lm_health/ppl"] == pytest.approx(16.0, rel=1e-3)


def test_empty_chunks_yield_no_metrics():
    assert lm_health_metrics(_FakeLM(), [], torch.device("cpu")) == {}


def test_train_mode_is_restored():
    """Eval must not silently leave the decoder in eval mode — Falcon-H1's
    eval path is numerically different and the trainer relies on train mode."""
    m = _FakeLM()
    m.train()
    lm_health_metrics(m, [torch.arange(8) % 16], torch.device("cpu"))
    assert m.training
    m.eval()
    lm_health_metrics(m, [torch.arange(8) % 16], torch.device("cpu"))
    assert not m.training


def test_accepts_a_reconstruction_decoder_wrapper():
    class _Wrapper:
        def __init__(self, backbone):
            self.backbone = backbone

    out = lm_health_metrics(_Wrapper(_FakeLM()), [torch.arange(8) % 16], torch.device("cpu"))
    assert "eval/lm_health/ce" in out


def test_uses_the_interleaved_forward_when_the_decoder_has_one():
    """bgkit's in-training decoders are patched for FA4 varlen PACKED
    attention with no padded fallback: a bare model(input_ids=(B, L)) hits the
    packed path with no cu_seqlens and triggers a device-side assert, which
    poisons the CUDA context and kills the RUN (2026-08-25, first lrprobe
    launch). The metric must route through the decoder's own primitive."""

    class _Out:
        loss = torch.tensor(2.0)

    class _Dec:
        def __init__(self):
            self.backbone = _FakeLM()
            self.calls = []

        def forward_interleaved_with_loss(self, segments, **kw):
            self.calls.append(segments)
            return _Out()

    dec = _Dec()
    out = lm_health_metrics(dec, [torch.arange(8) % 16], torch.device("cpu"))
    assert len(dec.calls) == 1
    seg = dec.calls[0][0]
    # One all-loss token segment — plain text, nothing spliced.
    assert seg.loss_mask.all() and seg.token_ids.shape[-1] == 8
    assert out["eval/lm_health/ce"] == pytest.approx(2.0)
