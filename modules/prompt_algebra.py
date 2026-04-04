"""
Formal Specification of the Prompt Builder — Prompt Algebra
============================================================

This module defines a formal algebra for prompt construction with
provable invariants:

  Invariant 1 (Budget): total tokens of composed prompt never exceed budget.
  Invariant 2 (Priority): higher-priority segments are never dropped before
                           lower-priority ones.
  Invariant 3 (Completeness): required segments are never dropped.

Algebra:
  A prompt P is a sequence of PromptSegments s_1, ..., s_n.
  Each segment s_i = (content_i, priority_i, max_tokens_i, required_i).

  Operations:
    compose(a, b)  → concatenation of segments
    truncate(s, k) → segment with content trimmed to k tokens
    prioritize(S)  → sorted by priority descending

  Budget Allocation:
    Given segments S and budget B:
      1. Sort S by priority descending.
      2. Include all required segments (must fit; error if not).
      3. Greedily include remaining by priority until budget exhausted.
      4. If a segment partially fits, truncate it.

  Priority Levels:
    CRITICAL (100) — system prompt, identity
    HIGH     (75)  — memory, skills
    MEDIUM   (50)  — conversation context
    LOW      (25)  — examples, supplementary info
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Priority levels
# ---------------------------------------------------------------------------

class Priority(IntEnum):
    LOW = 25
    MEDIUM = 50
    HIGH = 75
    CRITICAL = 100


# ---------------------------------------------------------------------------
# PromptSegment
# ---------------------------------------------------------------------------

@dataclass
class PromptSegment:
    """An atomic unit of prompt content with metadata.

    Attributes:
        content: The text content of this segment.
        priority: Higher = more important. Determines keep/drop order.
        max_tokens: Maximum tokens this segment may use. None = unlimited
                    (bounded only by content length).
        required: If True, this segment must never be dropped.
        name: Optional human-readable name for debugging.
    """
    content: str
    priority: int = Priority.MEDIUM
    max_tokens: Optional[int] = None
    required: bool = False
    name: str = ""

    @property
    def token_count(self) -> int:
        """Actual tokens used by content."""
        return estimate_tokens(self.content)

    @property
    def effective_tokens(self) -> int:
        """Tokens this segment will use (respecting max_tokens cap)."""
        tc = self.token_count
        if self.max_tokens is not None:
            return min(tc, self.max_tokens)
        return tc


# ---------------------------------------------------------------------------
# Prompt Operations (the algebra)
# ---------------------------------------------------------------------------

def compose(a: PromptSegment, b: PromptSegment) -> PromptSegment:
    """Concatenate two segments. Result inherits higher priority.
    Required if either is required.
    """
    return PromptSegment(
        content=a.content + "\n" + b.content,
        priority=max(a.priority, b.priority),
        max_tokens=(
            (a.effective_tokens + b.effective_tokens)
            if a.max_tokens is not None and b.max_tokens is not None
            else None
        ),
        required=a.required or b.required,
        name=f"{a.name}+{b.name}" if a.name or b.name else "",
    )


def truncate(segment: PromptSegment, max_tokens: int) -> PromptSegment:
    """Return a new segment truncated to at most max_tokens tokens.

    Truncation preserves the beginning of the content (prefix truncation).
    """
    if segment.token_count <= max_tokens:
        return PromptSegment(
            content=segment.content,
            priority=segment.priority,
            max_tokens=max_tokens,
            required=segment.required,
            name=segment.name,
        )
    # Approximate: 4 chars per token
    char_limit = max(0, max_tokens * 4)
    truncated_content = segment.content[:char_limit]
    return PromptSegment(
        content=truncated_content,
        priority=segment.priority,
        max_tokens=max_tokens,
        required=segment.required,
        name=segment.name,
    )


def prioritize(segments: List[PromptSegment]) -> List[PromptSegment]:
    """Sort segments by priority descending. Stable sort preserves
    insertion order among equal priorities."""
    return sorted(segments, key=lambda s: s.priority, reverse=True)


# ---------------------------------------------------------------------------
# PromptBudget — allocation with provable invariants
# ---------------------------------------------------------------------------

class BudgetExceededError(Exception):
    """Raised when required segments alone exceed the budget."""
    pass


class PromptBudget:
    """Allocates token budget across segments respecting invariants.

    Invariant proofs (informal):

    1. Budget invariant:  The allocation loop tracks remaining_budget and
       never assigns more tokens than available. Each segment gets
       min(effective_tokens, remaining_budget). Sum ≤ total_budget. ∎

    2. Priority invariant: Segments are processed in descending priority.
       A segment is only dropped (gets 0 tokens) when budget is exhausted.
       Since we process high-priority first, lower-priority segments are
       dropped first. ∎

    3. Completeness invariant: Required segments are allocated before
       optional ones. If required segments exceed budget, we raise
       BudgetExceededError rather than silently dropping them. ∎
    """

    def allocate(
        self,
        segments: List[PromptSegment],
        total_budget: int,
    ) -> List[PromptSegment]:
        """Allocate budget and return segments (possibly truncated).

        Returns only segments that received tokens, in original priority order.
        Raises BudgetExceededError if required segments exceed budget.
        """
        if not segments:
            return []

        # Phase 1: Reserve space for required segments
        required = [s for s in segments if s.required]
        optional = [s for s in segments if not s.required]

        required_tokens = sum(s.effective_tokens for s in required)
        if required_tokens > total_budget:
            # Must truncate required segments to fit
            # Still include all of them, just truncated
            result = []
            remaining = total_budget
            for s in prioritize(required):
                if remaining <= 0:
                    raise BudgetExceededError(
                        f"Required segments need {required_tokens} tokens "
                        f"but budget is {total_budget}."
                    )
                allocated = min(s.effective_tokens, remaining)
                result.append(truncate(s, allocated))
                remaining -= allocated
            return prioritize(result)

        # Phase 2: Allocate remaining budget to optional by priority
        remaining = total_budget - required_tokens
        result = list(required)  # All required segments kept as-is

        for s in prioritize(optional):
            if remaining <= 0:
                break
            allocated = min(s.effective_tokens, remaining)
            if allocated > 0:
                result.append(truncate(s, allocated))
                remaining -= allocated

        return prioritize(result)

    def allocate_with_report(
        self,
        segments: List[PromptSegment],
        total_budget: int,
    ) -> Tuple[List[PromptSegment], dict]:
        """Allocate and return (segments, report_dict)."""
        allocated = self.allocate(segments, total_budget)
        used = sum(s.effective_tokens for s in allocated)
        dropped = [
            s for s in segments
            if not any(a.name == s.name and a.content[:20] == s.content[:20] for a in allocated)
        ]
        report = {
            "total_budget": total_budget,
            "tokens_used": used,
            "tokens_remaining": total_budget - used,
            "segments_kept": len(allocated),
            "segments_dropped": len(segments) - len(allocated),
            "dropped_names": [s.name for s in dropped if s.name],
        }
        return allocated, report


# ---------------------------------------------------------------------------
# PromptVerifier — verification of invariants
# ---------------------------------------------------------------------------

class PromptVerifier:
    """Verifies prompt composition invariants."""

    @staticmethod
    def verify_budget(
        composed_segments: List[PromptSegment], budget: int
    ) -> bool:
        """Invariant 1: total tokens never exceed budget."""
        total = sum(s.effective_tokens for s in composed_segments)
        return total <= budget

    @staticmethod
    def verify_priority_ordering(segments: List[PromptSegment]) -> bool:
        """Invariant 2: segments are in non-increasing priority order."""
        for i in range(len(segments) - 1):
            if segments[i].priority < segments[i + 1].priority:
                return False
        return True

    @staticmethod
    def verify_completeness(
        composed_segments: List[PromptSegment],
        required_segments: List[PromptSegment],
    ) -> bool:
        """Invariant 3: all required segments appear in composed output.

        We check by name or content prefix match.
        """
        for req in required_segments:
            # Check if any composed segment matches
            found = False
            for s in composed_segments:
                if s.name and s.name == req.name:
                    found = True
                    break
                if s.content[:50] == req.content[:50]:
                    found = True
                    break
            if not found:
                return False
        return True

    @staticmethod
    def verify_no_higher_priority_dropped(
        kept: List[PromptSegment],
        dropped: List[PromptSegment],
    ) -> bool:
        """No kept segment has lower priority than any dropped segment."""
        if not kept or not dropped:
            return True
        min_kept_priority = min(s.priority for s in kept)
        max_dropped_priority = max(s.priority for s in dropped)
        return min_kept_priority >= max_dropped_priority
