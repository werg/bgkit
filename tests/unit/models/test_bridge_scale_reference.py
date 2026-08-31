"""A runaway detector whose reference resets each run cannot see a runaway.

The L0->L1 bridge scale went 484 at the Phase-1 base to 845 by widenet v5b,
and the L1-input corpus mean reached 70985 by v8. The guard never fired:
its reference lived in a module global, so every restart re-anchored to
whatever that process saw first, and no single run moved it 2x.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.models import encoder as enc_mod


@pytest.fixture(autouse=True)
def _fresh_guard_state():
    enc_mod._bridge_guard_calls.clear()
    enc_mod._bridge_guard_reference.clear()
    yield
    enc_mod._bridge_guard_calls.clear()
    enc_mod._bridge_guard_reference.clear()


def _bridged(scale: float, d: int = 16) -> torch.Tensor:
    torch.manual_seed(0)
    x = torch.randn(32, d)
    return x / x.norm(dim=-1, keepdim=True) * scale


def _embed(norm: float = 1.0, d: int = 16) -> torch.Tensor:
    torch.manual_seed(1)
    w = torch.randn(50, d)
    return w / w.norm(dim=-1, keepdim=True) * norm


def test_guard_returns_the_measured_ratio_on_sampled_calls():
    ratio = enc_mod.guard_bridge_output_scale(
        _bridged(500.0), _embed(1.0), site="t", every=1,
    )
    assert ratio == pytest.approx(500.0, rel=1e-3)


def test_guard_returns_none_on_skipped_calls():
    assert enc_mod.guard_bridge_output_scale(
        _bridged(500.0), _embed(), site="t", every=10,
    ) is not None
    assert enc_mod.guard_bridge_output_scale(
        _bridged(500.0), _embed(), site="t", every=10,
    ) is None


def test_guard_returns_none_for_an_empty_payload():
    assert enc_mod.guard_bridge_output_scale(
        torch.zeros(0, 16), _embed(), site="t", every=1,
    ) is None


def test_an_explicit_reference_beats_the_process_global(monkeypatch):
    """The whole point: pass the lineage's operating point and a run that
    starts already inflated is measured against where it SHOULD be, not
    against its own first sample."""
    events: list[tuple[str, dict]] = []

    class _Rec:
        def warning(self, event, **kw):
            events.append((event, kw))

        def info(self, event, **kw):
            events.append((event, kw))

    monkeypatch.setattr(enc_mod, "logger", _Rec())
    enc_mod.guard_bridge_output_scale(
        _bridged(5000.0), _embed(), site="t", every=1, reference=500.0,
    )
    assert [e for e, _ in events] == ["bridge_output_scale_out_of_band"]
    assert events[0][1]["drift"] == pytest.approx(10.0, rel=1e-2)


def test_without_a_reference_the_same_inflation_looks_normal(monkeypatch):
    """The failure being replaced, pinned: with the process-global reference
    the guard anchors to the inflated value and reports it in band."""
    events: list[tuple[str, dict]] = []

    class _Rec:
        def warning(self, event, **kw):
            events.append((event, kw))

        def info(self, event, **kw):
            events.append((event, kw))

    monkeypatch.setattr(enc_mod, "logger", _Rec())
    enc_mod.guard_bridge_output_scale(
        _bridged(5000.0), _embed(), site="t", every=1,
    )
    assert [e for e, _ in events] == ["bridge_output_scale"]


def test_reference_is_persisted_on_the_encoder_and_seeded_once():
    class _Encoder(torch.nn.Module):
        bridge_reference = enc_mod.BgKITEncoder.bridge_reference
        observe_bridge_scale = enc_mod.BgKITEncoder.observe_bridge_scale

        def __init__(self):
            super().__init__()
            self.register_buffer("bridge_scale_reference", torch.zeros(()))

    e = _Encoder()
    assert e.bridge_reference() is None
    e.observe_bridge_scale(_bridged(500.0), _embed(), site="t")
    assert e.bridge_reference() == pytest.approx(500.0, rel=1e-3)
    # A later, larger observation must NOT move the anchor -- re-anchoring is
    # exactly the failure this replaces.
    enc_mod._bridge_guard_calls.clear()
    e.observe_bridge_scale(_bridged(50000.0), _embed(), site="t")
    assert e.bridge_reference() == pytest.approx(500.0, rel=1e-3)


def test_the_reference_buffer_rides_in_the_state_dict():
    class _Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("bridge_scale_reference", torch.zeros(()))

    src = _Encoder()
    src.bridge_scale_reference.fill_(484.0)
    dst = _Encoder()
    dst.load_state_dict(src.state_dict())
    assert float(dst.bridge_scale_reference.item()) == pytest.approx(484.0)
