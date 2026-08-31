"""Unit tests for the global spliced-rep NORM GUARD in
:meth:`ReconstructionDecoder._concat_segments` (repr-interface hardening,
2026-07-31). The guard is a SAMPLED, LOG-ONLY safety net that warns when the
mean spliced-rep L2-norm leaves a wide band relative to the decoder's
``embed_tokens`` row-norm — the durable detector for the rep-inflation collapse
class. These CPU tests confirm it fires on inflation, stays quiet in-band, never
raises, and respects the sampling cadence + enable flag.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F
from torch import nn

import bgkit.models.decoder as decoder_mod
from bgkit.models.decoder import (
    EmbeddingSegment,
    ReconstructionDecoder,
    TokenSegment,
)

VOCAB_SIZE = 64
HIDDEN_DIM = 16


class _InnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        # Force unit-norm rows so mean embed row-norm == 1 (target band == ratio).
        with torch.no_grad():
            self.embed_tokens.weight.copy_(
                F.normalize(torch.randn(VOCAB_SIZE, HIDDEN_DIM), dim=-1),
            )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _InnerModel()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens


NORM_EVENT = "spliced_rep_norm_out_of_band"


def norm_warnings(rec):
    """Only this guard's warnings. The same sampled block also emits
    ``spliced_rep_rank_collapsed``, which carries its own ``consecutive``
    counter -- collecting both would make these assertions about the norm
    band silently depend on the rank band's threshold."""
    return [(e, kw) for e, kw in rec.warnings if e == NORM_EVENT]


class _RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict]] = []

    def warning(self, event: str, **kw) -> None:
        self.warnings.append((event, kw))

    # Guard-safe no-ops for any other level the code might call.
    def info(self, *a, **k) -> None:  # pragma: no cover - defensive
        pass


def _decoder() -> ReconstructionDecoder:
    torch.manual_seed(11)
    return ReconstructionDecoder(_Backbone(), hidden_dim=HIDDEN_DIM)


def _reps(k: int, norm_val: float) -> torch.Tensor:
    torch.manual_seed(5)
    return (F.normalize(torch.randn(k, HIDDEN_DIM), dim=-1) * norm_val).to(torch.float32)


def _segments(rep_norm: float) -> list:
    ids = torch.arange(4, dtype=torch.long)
    return [
        TokenSegment(token_ids=ids, loss=False),
        EmbeddingSegment(embeddings=_reps(5, rep_norm)),
    ]


@pytest.fixture
def guard_every_1(monkeypatch):
    """Force the guard to sample EVERY call so a single decode triggers it."""
    monkeypatch.setattr(decoder_mod, "_REP_NORM_GUARD_ENABLED", True)
    monkeypatch.setattr(decoder_mod, "_REP_NORM_GUARD_EVERY", 1)
    monkeypatch.setattr(decoder_mod, "_REP_NORM_GUARD_HI", 8.0)


def _run(dec, rep_norm, monkeypatch) -> _RecordingLogger:
    rec = _RecordingLogger()
    monkeypatch.setattr(decoder_mod, "logger", rec)
    embed_fn = dec.backbone.get_input_embeddings()
    dec._concat_segments(_segments(rep_norm), embed_fn)
    return rec


def test_guard_fires_on_inflated_reps(guard_every_1, monkeypatch):
    dec = _decoder()
    rec = _run(dec, rep_norm=20.0, monkeypatch=monkeypatch)  # ratio ~20 > 8
    events = [e for e, _ in rec.warnings]
    assert "spliced_rep_norm_out_of_band" in events
    kw = next(kw for e, kw in rec.warnings if e == "spliced_rep_norm_out_of_band")
    assert kw["ratio"] > 8.0


def test_guard_fires_on_collapsed_reps(guard_every_1, monkeypatch):
    dec = _decoder()
    rec = _run(dec, rep_norm=0.05, monkeypatch=monkeypatch)  # ratio ~0.05 < 1/8
    events = [e for e, _ in rec.warnings]
    assert "spliced_rep_norm_out_of_band" in events


def test_guard_quiet_in_band(guard_every_1, monkeypatch):
    dec = _decoder()
    rec = _run(dec, rep_norm=4.2, monkeypatch=monkeypatch)  # ratio ~4.2 in [1/8, 8]
    events = [e for e, _ in rec.warnings]
    assert "spliced_rep_norm_out_of_band" not in events


def test_guard_respects_sampling_cadence(monkeypatch):
    monkeypatch.setattr(decoder_mod, "_REP_NORM_GUARD_ENABLED", True)
    monkeypatch.setattr(decoder_mod, "_REP_NORM_GUARD_EVERY", 50)
    monkeypatch.setattr(decoder_mod, "_REP_NORM_GUARD_HI", 8.0)
    dec = _decoder()
    rec = _RecordingLogger()
    monkeypatch.setattr(decoder_mod, "logger", rec)
    embed_fn = dec.backbone.get_input_embeddings()
    # 49 calls: below the sampling cadence -> silent even though inflated.
    for _ in range(49):
        dec._concat_segments(_segments(20.0), embed_fn)
    assert not norm_warnings(rec)
    # 50th call trips the sample and warns.
    dec._concat_segments(_segments(20.0), embed_fn)
    assert any(e == NORM_EVENT for e, _ in rec.warnings)


def test_guard_disabled_never_warns(monkeypatch):
    monkeypatch.setattr(decoder_mod, "_REP_NORM_GUARD_ENABLED", False)
    monkeypatch.setattr(decoder_mod, "_REP_NORM_GUARD_EVERY", 1)
    dec = _decoder()
    rec = _run(dec, rep_norm=100.0, monkeypatch=monkeypatch)
    assert not norm_warnings(rec)


def test_guard_raises_after_consecutive_out_of_band(guard_every_1, monkeypatch):
    """Escalation (2026-08-22): N CONSECUTIVE sampled out-of-band checks raise —
    absent / collapsed reps for hundreds of forwards is never a transient
    (the widenet v1→v4 zero-rep runs). An in-band check resets the streak."""
    monkeypatch.setattr(decoder_mod, "_REP_NORM_GUARD_RAISE_AFTER", 3)
    dec = _decoder()
    rec = _RecordingLogger()
    monkeypatch.setattr(decoder_mod, "logger", rec)
    embed_fn = dec.backbone.get_input_embeddings()
    # Two zero-norm splices: warned, not raised.
    dec._concat_segments(_segments(0.0), embed_fn)
    dec._concat_segments(_segments(0.0), embed_fn)
    assert [kw["consecutive"] for _, kw in norm_warnings(rec)] == [1, 2]
    # An in-band splice resets the streak.
    dec._concat_segments(_segments(1.0), embed_fn)
    dec._concat_segments(_segments(0.0), embed_fn)
    assert norm_warnings(rec)[-1][1]["consecutive"] == 1
    dec._concat_segments(_segments(0.0), embed_fn)
    # Third consecutive hit raises with a diagnostic message.
    with pytest.raises(RuntimeError, match="consecutive sampled checks"):
        dec._concat_segments(_segments(0.0), embed_fn)


def test_guard_stands_down_under_expected_degenerate_reps(guard_every_1, monkeypatch):
    """While the trainer flags an ablation that deliberately zeroes reps, the
    guard neither logs nor counts toward the streak; clearing the flag resumes
    counting from the preserved streak."""
    monkeypatch.setattr(decoder_mod, "_REP_NORM_GUARD_RAISE_AFTER", 2)
    dec = _decoder()
    rec = _RecordingLogger()
    monkeypatch.setattr(decoder_mod, "logger", rec)
    embed_fn = dec.backbone.get_input_embeddings()
    dec._concat_segments(_segments(0.0), embed_fn)  # streak 1
    dec._rep_norm_guard_expect_degenerate = True
    for _ in range(10):
        dec._concat_segments(_segments(0.0), embed_fn)  # ablation pass: ignored
    assert len(norm_warnings(rec)) == 1
    dec._rep_norm_guard_expect_degenerate = False
    with pytest.raises(RuntimeError):
        dec._concat_segments(_segments(0.0), embed_fn)  # streak 2 -> raise


def test_guard_reference_ratio_steady_high_is_not_degenerate(guard_every_1, monkeypatch):
    """A checkpoint family whose operating point is far above the nominal band
    (the 51945 Qwen projection: ~38x) is NOT degenerate: the first readable
    sample becomes the decoder's reference, the absolute band warns ONCE, and a
    steady ratio never builds a streak. A >band DRIFT from the reference (the
    4x->320x runaway class) does, and raises."""
    monkeypatch.setattr(decoder_mod, "_REP_NORM_GUARD_RAISE_AFTER", 2)
    dec = _decoder()
    rec = _RecordingLogger()
    monkeypatch.setattr(decoder_mod, "logger", rec)
    embed_fn = dec.backbone.get_input_embeddings()
    for _ in range(6):
        dec._concat_segments(_segments(37.0), embed_fn)  # steady 37x: fine
    assert dec._rep_norm_guard_ref_ratio == pytest.approx(37.0, rel=1e-3)
    assert len(norm_warnings(rec)) == 1  # informational, once
    assert norm_warnings(rec)[0][1]["degenerate"] is False
    assert dec._rep_norm_guard_oob_streak == 0
    dec._concat_segments(_segments(37.0 * 9), embed_fn)  # 9x drift: degenerate
    assert norm_warnings(rec)[-1][1]["degenerate"] is True
    with pytest.raises(RuntimeError, match="degenerate"):
        dec._concat_segments(_segments(37.0 * 9), embed_fn)


def test_guard_raise_disabled_with_zero(guard_every_1, monkeypatch):
    monkeypatch.setattr(decoder_mod, "_REP_NORM_GUARD_RAISE_AFTER", 0)
    dec = _decoder()
    monkeypatch.setattr(decoder_mod, "logger", _RecordingLogger())
    embed_fn = dec.backbone.get_input_embeddings()
    for _ in range(20):
        dec._concat_segments(_segments(0.0), embed_fn)  # never raises


def test_guard_returns_correct_embeds(guard_every_1, monkeypatch):
    """The guard must not alter the concatenated inputs_embeds it observes."""
    dec = _decoder()
    monkeypatch.setattr(decoder_mod, "logger", _RecordingLogger())
    embed_fn = dec.backbone.get_input_embeddings()
    segs = _segments(20.0)
    inputs_embeds, token_ids, loss_mask = dec._concat_segments(segs, embed_fn)
    # 4 token positions + 5 rep positions = 9.
    assert inputs_embeds.shape == (1, 9, HIDDEN_DIM)
    assert token_ids.shape == (1, 9)
    assert loss_mask.shape == (1, 9)
    assert not bool(loss_mask.any())  # loss=False token seg + rep seg
