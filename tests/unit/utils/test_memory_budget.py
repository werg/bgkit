"""Unit tests for the scoped memory-budget helper."""

from __future__ import annotations

import pytest

from bgkit.utils.memory_budget import (
    MemoryBudgetExceeded,
    MemoryScopeStats,
    collect_memory_diagnostics,
    memory_budget_scope,
)


def test_collect_memory_diagnostics_returns_host_keys():
    """Host-side fields are always populated (no CUDA required)."""
    out = collect_memory_diagnostics()
    assert "mem/proc_rss_gb" in out
    assert "mem/proc_vsz_gb" in out
    assert "mem/system_used_gb" in out
    assert "mem/system_available_gb" in out
    assert out["mem/proc_rss_gb"] > 0
    assert out["mem/system_used_gb"] > 0


def test_scope_without_cap_logs_and_exits_cleanly():
    with memory_budget_scope("test_noop") as stats:
        assert stats.name == "test_noop"
        assert stats.cap_gb is None
    assert stats.cuda_peak_gb >= 0
    assert stats.system_pre_gb > 0
    assert stats.system_post_gb > 0


def test_scope_populates_stats_on_exit():
    with memory_budget_scope("test_stats", cap_gb=100.0) as stats:
        pass
    assert stats.cap_gb == 100.0
    assert stats.rss_pre_gb > 0
    assert stats.rss_post_gb > 0


def test_scope_nests_independently():
    """Nested scopes each track their own peak via reset."""
    with memory_budget_scope("outer") as outer:
        with memory_budget_scope("inner") as inner:
            pass
        # Inner's peak is measured only during the inner scope.
        assert inner.name == "inner"
    assert outer.name == "outer"


def test_scope_raises_when_cap_exceeded(monkeypatch):
    """Blowing the cap raises MemoryBudgetExceeded with the scope name."""
    # Patch the peak reader to simulate a blown budget without needing
    # a real allocation.
    from bgkit.utils import memory_budget as mb

    monkeypatch.setattr(mb, "_cuda_peak_gb", lambda: 50.0)

    with pytest.raises(MemoryBudgetExceeded) as exc_info:
        with memory_budget_scope("over_cap", cap_gb=10.0):
            pass
    msg = str(exc_info.value)
    assert "over_cap" in msg
    assert "50.00" in msg
    assert "10.00" in msg


def test_scope_exception_in_body_still_reclaims(monkeypatch):
    """Exceptions inside the scope don't skip the reclaim / log path."""
    from bgkit.utils import memory_budget as mb

    reclaim_calls = []

    def fake_reclaim() -> None:
        reclaim_calls.append(True)

    monkeypatch.setattr(mb, "_reclaim", fake_reclaim)

    with pytest.raises(ValueError, match="body"):
        with memory_budget_scope("boom"):
            raise ValueError("body")
    # Both enter and exit reclaim should have fired.
    assert len(reclaim_calls) == 2


def test_scope_reclaim_flags_disable_calls(monkeypatch):
    """reclaim_on_enter / reclaim_on_exit toggles are honored."""
    from bgkit.utils import memory_budget as mb

    reclaim_calls = []
    monkeypatch.setattr(
        mb, "_reclaim", lambda: reclaim_calls.append(True),
    )

    with memory_budget_scope(
        "no_reclaim",
        reclaim_on_enter=False,
        reclaim_on_exit=False,
    ):
        pass
    assert reclaim_calls == []


def test_memory_scope_stats_defaults():
    s = MemoryScopeStats(name="x", cap_gb=None)
    assert s.name == "x"
    assert s.cap_gb is None
    assert s.cuda_pre_gb == 0.0
    assert s.cuda_peak_gb == 0.0
    assert s.extra == {}
