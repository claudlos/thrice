"""Tests for ``cache_optimizer``."""
from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = os.path.normpath(os.path.join(_HERE, "..", "modules"))
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)

from cache_optimizer import (                           # noqa: E402
    CacheTracker,
    PrefixGuard,
)


# ---------------------------------------------------------------------------
# PrefixGuard: dynamic-value detection
# ---------------------------------------------------------------------------

class TestPrefixGuardDynamic:

    def _guard(self) -> PrefixGuard:
        return PrefixGuard()

    def test_clean_system_prompt_is_stable(self):
        v = self._guard().check(system_prompt="You are a helpful assistant.")
        assert v.ok, v.summary()

    def test_timestamp_in_system_prompt_flagged(self):
        v = self._guard().check(
            system_prompt="You are helpful. Today is 2026-04-17T12:30:00.",
        )
        assert not v.ok
        kinds = {b.kind for b in v.breakages}
        assert "timestamp" in kinds

    def test_uuid_in_system_prompt_flagged(self):
        v = self._guard().check(
            system_prompt="Session 550e8400-e29b-41d4-a716-446655440000 active.",
        )
        assert any(b.kind == "uuid" for b in v.breakages)

    def test_epoch_millis_flagged(self):
        v = self._guard().check(system_prompt="time=1712400000000")
        assert any(b.kind == "epoch_millis" for b in v.breakages)

    def test_session_id_flagged(self):
        v = self._guard().check(
            system_prompt="session_id: abc123def456",
        )
        assert any(b.kind == "session_id" for b in v.breakages)

    def test_non_dynamic_numbers_not_flagged(self):
        # Small integers shouldn't look like epoch timestamps.
        v = self._guard().check(system_prompt="max_retries=3, timeout=30")
        assert v.ok, v.summary()


# ---------------------------------------------------------------------------
# PrefixGuard: tool-set stability
# ---------------------------------------------------------------------------

class TestPrefixGuardTools:

    def test_same_tools_twice_is_stable(self):
        g = PrefixGuard()
        tools = [{"name": "edit"}, {"name": "run"}]
        g.check(tools=tools)
        v = g.check(tools=tools)
        assert v.ok

    def test_added_tool_flagged(self):
        g = PrefixGuard()
        g.check(tools=[{"name": "edit"}])
        v = g.check(tools=[{"name": "edit"}, {"name": "run"}])
        assert not v.ok
        assert any(b.kind == "tools_changed" for b in v.breakages)
        assert "run" in v.breakages[0].detail

    def test_removed_tool_flagged(self):
        g = PrefixGuard()
        g.check(tools=[{"name": "a"}, {"name": "b"}])
        v = g.check(tools=[{"name": "a"}])
        assert not v.ok
        assert any("removed" in b.detail for b in v.breakages)

    def test_first_check_is_always_stable(self):
        """No baseline on first call -> can't be a regression."""
        g = PrefixGuard()
        v = g.check(tools=[{"name": "edit"}])
        # First observation defines the baseline.
        assert v.ok


# ---------------------------------------------------------------------------
# PrefixGuard: system-prompt stability
# ---------------------------------------------------------------------------

class TestPrefixGuardSystem:
    def test_system_mutation_flagged(self):
        g = PrefixGuard()
        g.check(system_prompt="You are helpful")
        v = g.check(system_prompt="You are VERY helpful")
        assert any(b.kind == "system_prompt_changed" for b in v.breakages)


# ---------------------------------------------------------------------------
# CacheTracker
# ---------------------------------------------------------------------------

class TestCacheTracker:

    def test_record_basic(self):
        t = CacheTracker()
        rec = t.record(turn=1, input_tokens=1000, cached_tokens=0)
        assert rec.hit_rate == 0.0
        rec = t.record(turn=2, input_tokens=1000, cached_tokens=900)
        assert abs(rec.hit_rate - 0.9) < 1e-6

    def test_hit_rate_clamped(self):
        """cached > input is clamped to input (some providers over-report)."""
        t = CacheTracker()
        rec = t.record(turn=1, input_tokens=1000, cached_tokens=10_000)
        assert rec.cached_tokens == 1000
        assert rec.hit_rate == 1.0

    def test_stats_empty(self):
        s = CacheTracker().stats()
        assert s.turns == 0 and s.savings == 0.0

    def test_stats_computes_savings(self):
        t = CacheTracker()
        # 10 turns, 1000 input each, 800 cached → 80% hit rate
        for i in range(10):
            t.record(turn=i, input_tokens=1000, cached_tokens=800,
                     provider="anthropic")
        s = t.stats()
        assert s.turns == 10
        assert abs(s.mean_hit_rate - 0.8) < 1e-6
        assert s.savings > 0
        # At 80% hit with Anthropic's default 10x cache discount, savings
        # should land in the 60-80% band predicted by the paper.
        assert 0.60 <= s.savings_pct <= 0.80, s

    def test_regression_detection(self):
        t = CacheTracker(regression_threshold=0.20, window=5)
        # First 5 turns: 90% hit rate
        for i in range(5):
            t.record(turn=i, input_tokens=1000, cached_tokens=900)
        # Next 5 turns: 30% hit rate (big regression)
        for i in range(5, 10):
            t.record(turn=i, input_tokens=1000, cached_tokens=300)
        assert t.needs_attention() is True

    def test_no_regression_when_stable(self):
        t = CacheTracker(regression_threshold=0.20, window=5)
        for i in range(15):
            t.record(turn=i, input_tokens=1000, cached_tokens=850)
        assert t.needs_attention() is False

    def test_not_enough_data(self):
        t = CacheTracker(window=5)
        for i in range(3):
            t.record(turn=i, input_tokens=1000, cached_tokens=300)
        assert t.needs_attention() is False


# ---------------------------------------------------------------------------
# Integration: guard + tracker
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_guard_then_tracker_full_turn(self):
        g = PrefixGuard()
        t = CacheTracker()
        v = g.check(
            system_prompt="Be concise.",
            tools=[{"name": "edit"}, {"name": "run"}],
        )
        assert v.ok
        t.record(turn=1, input_tokens=2000, cached_tokens=1800,
                 provider="anthropic")
        s = t.stats()
        assert s.savings > 0
