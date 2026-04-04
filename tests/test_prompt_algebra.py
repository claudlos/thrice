"""Tests for prompt_algebra.py — Formal Prompt Builder Specification.

Includes both unit tests and property-based tests using hypothesis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new-files"))

import pytest
from prompt_algebra import (
    Priority,
    PromptSegment,
    PromptBudget,
    PromptVerifier,
    BudgetExceededError,
    compose,
    truncate,
    prioritize,
    estimate_tokens,
)

try:
    from hypothesis import given, settings, assume
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


# ---------------------------------------------------------------------------
# PromptSegment
# ---------------------------------------------------------------------------

class TestPromptSegment:
    def test_token_count(self):
        seg = PromptSegment(content="Hello world", priority=Priority.MEDIUM)
        assert seg.token_count > 0

    def test_effective_tokens_no_cap(self):
        seg = PromptSegment(content="a" * 100, priority=Priority.MEDIUM)
        assert seg.effective_tokens == seg.token_count

    def test_effective_tokens_with_cap(self):
        seg = PromptSegment(content="a" * 100, priority=Priority.MEDIUM, max_tokens=5)
        assert seg.effective_tokens == 5

    def test_required_flag(self):
        seg = PromptSegment(content="identity", priority=Priority.CRITICAL, required=True)
        assert seg.required is True


# ---------------------------------------------------------------------------
# Algebra Operations
# ---------------------------------------------------------------------------

class TestAlgebraOperations:
    def test_compose_concatenates(self):
        a = PromptSegment(content="Hello", priority=Priority.HIGH, name="a")
        b = PromptSegment(content="World", priority=Priority.LOW, name="b")
        result = compose(a, b)
        assert "Hello" in result.content
        assert "World" in result.content

    def test_compose_takes_higher_priority(self):
        a = PromptSegment(content="A", priority=Priority.HIGH)
        b = PromptSegment(content="B", priority=Priority.LOW)
        result = compose(a, b)
        assert result.priority == Priority.HIGH

    def test_compose_required_propagates(self):
        a = PromptSegment(content="A", priority=Priority.LOW, required=True)
        b = PromptSegment(content="B", priority=Priority.HIGH, required=False)
        result = compose(a, b)
        assert result.required is True

    def test_truncate_within_limit(self):
        seg = PromptSegment(content="Short", priority=Priority.MEDIUM)
        result = truncate(seg, 1000)
        assert result.content == "Short"

    def test_truncate_cuts_content(self):
        seg = PromptSegment(content="a" * 1000, priority=Priority.MEDIUM)
        result = truncate(seg, 10)
        assert result.effective_tokens <= 10

    def test_prioritize_sorts_descending(self):
        segments = [
            PromptSegment(content="low", priority=Priority.LOW),
            PromptSegment(content="critical", priority=Priority.CRITICAL),
            PromptSegment(content="medium", priority=Priority.MEDIUM),
        ]
        result = prioritize(segments)
        assert result[0].priority == Priority.CRITICAL
        assert result[-1].priority == Priority.LOW


# ---------------------------------------------------------------------------
# PromptBudget
# ---------------------------------------------------------------------------

class TestPromptBudget:
    def setup_method(self):
        self.budget = PromptBudget()

    def test_all_fit_within_budget(self):
        segments = [
            PromptSegment(content="a" * 40, priority=Priority.HIGH, name="a"),
            PromptSegment(content="b" * 40, priority=Priority.LOW, name="b"),
        ]
        result = self.budget.allocate(segments, 1000)
        total = sum(s.effective_tokens for s in result)
        assert total <= 1000

    def test_required_always_kept(self):
        segments = [
            PromptSegment(content="identity" * 10, priority=Priority.CRITICAL, required=True, name="sys"),
            PromptSegment(content="optional" * 100, priority=Priority.LOW, name="opt"),
        ]
        result = self.budget.allocate(segments, 50)
        names = [s.name for s in result]
        assert "sys" in names

    def test_higher_priority_kept_first(self):
        segments = [
            PromptSegment(content="x" * 200, priority=Priority.HIGH, name="high"),
            PromptSegment(content="y" * 200, priority=Priority.LOW, name="low"),
        ]
        # Budget enough for only one
        result = self.budget.allocate(segments, 60)
        names = [s.name for s in result]
        assert "high" in names

    def test_budget_exceeded_error(self):
        segments = [
            PromptSegment(content="a" * 400, priority=Priority.CRITICAL, required=True, name="a"),
            PromptSegment(content="b" * 400, priority=Priority.CRITICAL, required=True, name="b"),
        ]
        # Budget too small for both required — should raise or truncate
        # Our impl truncates first, raises if can't fit at all
        with pytest.raises(BudgetExceededError):
            self.budget.allocate(segments, 1)  # Impossibly small

    def test_empty_segments(self):
        result = self.budget.allocate([], 1000)
        assert result == []

    def test_allocate_with_report(self):
        segments = [
            PromptSegment(content="hello" * 20, priority=Priority.HIGH, name="hi"),
            PromptSegment(content="world" * 20, priority=Priority.LOW, name="lo"),
        ]
        result, report = self.budget.allocate_with_report(segments, 1000)
        assert "total_budget" in report
        assert report["tokens_used"] <= report["total_budget"]

    def test_partial_truncation(self):
        segments = [
            PromptSegment(content="a" * 400, priority=Priority.HIGH, name="a"),
        ]
        # Budget smaller than full content but enough for something
        result = self.budget.allocate(segments, 50)
        assert len(result) == 1
        assert result[0].effective_tokens <= 50


# ---------------------------------------------------------------------------
# PromptVerifier
# ---------------------------------------------------------------------------

class TestPromptVerifier:
    def test_verify_budget_ok(self):
        segments = [
            PromptSegment(content="a" * 40, priority=Priority.HIGH),
        ]
        assert PromptVerifier.verify_budget(segments, 100) is True

    def test_verify_budget_exceeded(self):
        segments = [
            PromptSegment(content="a" * 800, priority=Priority.HIGH),
        ]
        assert PromptVerifier.verify_budget(segments, 10) is False

    def test_verify_priority_ordering_ok(self):
        segments = [
            PromptSegment(content="a", priority=Priority.CRITICAL),
            PromptSegment(content="b", priority=Priority.HIGH),
            PromptSegment(content="c", priority=Priority.LOW),
        ]
        assert PromptVerifier.verify_priority_ordering(segments) is True

    def test_verify_priority_ordering_bad(self):
        segments = [
            PromptSegment(content="a", priority=Priority.LOW),
            PromptSegment(content="b", priority=Priority.CRITICAL),
        ]
        assert PromptVerifier.verify_priority_ordering(segments) is False

    def test_verify_completeness_ok(self):
        required = [
            PromptSegment(content="system prompt", priority=Priority.CRITICAL, required=True, name="sys"),
        ]
        composed = [
            PromptSegment(content="system prompt", priority=Priority.CRITICAL, name="sys"),
            PromptSegment(content="other stuff", priority=Priority.LOW, name="other"),
        ]
        assert PromptVerifier.verify_completeness(composed, required) is True

    def test_verify_completeness_missing(self):
        required = [
            PromptSegment(content="system prompt", priority=Priority.CRITICAL, required=True, name="sys"),
        ]
        composed = [
            PromptSegment(content="other stuff", priority=Priority.LOW, name="other"),
        ]
        assert PromptVerifier.verify_completeness(composed, required) is False

    def test_verify_no_higher_priority_dropped(self):
        kept = [PromptSegment(content="a", priority=Priority.HIGH)]
        dropped = [PromptSegment(content="b", priority=Priority.LOW)]
        assert PromptVerifier.verify_no_higher_priority_dropped(kept, dropped) is True

    def test_verify_higher_priority_dropped_fails(self):
        kept = [PromptSegment(content="a", priority=Priority.LOW)]
        dropped = [PromptSegment(content="b", priority=Priority.HIGH)]
        assert PromptVerifier.verify_no_higher_priority_dropped(kept, dropped) is False


# ---------------------------------------------------------------------------
# Integration: Budget + Verifier
# ---------------------------------------------------------------------------

class TestBudgetVerifierIntegration:
    def test_allocated_passes_all_invariants(self):
        segments = [
            PromptSegment(content="System identity and rules " * 5, priority=Priority.CRITICAL, required=True, name="system"),
            PromptSegment(content="Memory context " * 10, priority=Priority.HIGH, name="memory"),
            PromptSegment(content="Conversation history " * 20, priority=Priority.MEDIUM, name="history"),
            PromptSegment(content="Example outputs " * 15, priority=Priority.LOW, name="examples"),
        ]
        budget_val = 200
        allocator = PromptBudget()
        result = allocator.allocate(segments, budget_val)

        # Invariant 1: budget
        assert PromptVerifier.verify_budget(result, budget_val)

        # Invariant 2: priority ordering
        assert PromptVerifier.verify_priority_ordering(result)

        # Invariant 3: completeness (required segments present)
        required = [s for s in segments if s.required]
        assert PromptVerifier.verify_completeness(result, required)


# ---------------------------------------------------------------------------
# Property-based tests (hypothesis)
# ---------------------------------------------------------------------------

if HAS_HYPOTHESIS:
    segment_strategy = st.builds(
        PromptSegment,
        content=st.text(min_size=4, max_size=200),
        priority=st.sampled_from([Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.CRITICAL]),
        max_tokens=st.one_of(st.none(), st.integers(min_value=1, max_value=500)),
        required=st.booleans(),
        name=st.text(min_size=1, max_size=10),
    )

    class TestPropertyBased:
        @given(
            segments=st.lists(segment_strategy, min_size=0, max_size=20),
            budget=st.integers(min_value=1, max_value=10000),
        )
        @settings(max_examples=200)
        def test_budget_never_exceeded(self, segments, budget):
            """Property: allocated segments never exceed budget."""
            allocator = PromptBudget()
            try:
                result = allocator.allocate(segments, budget)
            except BudgetExceededError:
                return  # Valid: required segments too large
            total = sum(s.effective_tokens for s in result)
            assert total <= budget

        @given(
            segments=st.lists(segment_strategy, min_size=0, max_size=20),
            budget=st.integers(min_value=1, max_value=10000),
        )
        @settings(max_examples=200)
        def test_priority_ordering_maintained(self, segments, budget):
            """Property: output is always in non-increasing priority order."""
            allocator = PromptBudget()
            try:
                result = allocator.allocate(segments, budget)
            except BudgetExceededError:
                return
            assert PromptVerifier.verify_priority_ordering(result)

        @given(
            segments=st.lists(segment_strategy, min_size=1, max_size=10),
            budget=st.integers(min_value=500, max_value=10000),
        )
        @settings(max_examples=200)
        def test_required_segments_never_dropped(self, segments, budget):
            """Property: required segments are always present in output."""
            allocator = PromptBudget()
            try:
                result = allocator.allocate(segments, budget)
            except BudgetExceededError:
                return
            required = [s for s in segments if s.required]
            # Each required segment should have a match in result
            for req in required:
                found = any(
                    r.content[:50] == req.content[:50] or
                    (r.name and r.name == req.name)
                    for r in result
                )
                assert found, f"Required segment '{req.name}' was dropped"
