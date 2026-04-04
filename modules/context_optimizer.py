"""
Context Window Optimization via Information Theory
====================================================

Mathematical Framework
----------------------

We model context window optimization as a rate-distortion problem from
information theory. Given a conversation history M = {M_1, ..., M_n},
we want to select a subset S ⊆ M that maximizes task performance while
staying within a token budget R.

Definitions:
  - Importance I(M_i) ≈ proxy for mutual information I(M_i; Outcome)
    where Outcome is the quality of the model's next response.
  - Rate R(S) = Σ_{M_i ∈ S} tokens(M_i)  (total tokens used)
  - Distortion D(S) = Σ_{M_i ∉ S} I(M_i)  (importance of dropped messages)

Rate-Distortion Formulation:
  minimize   D(S)  =  Σ_{M_i ∉ S} I(M_i)
  subject to R(S) <= budget

Theorem (Greedy Optimality for Knapsack):
  When message sizes are approximately uniform, the optimal strategy is
  to remove the lowest-importance messages first. This follows from the
  greedy solution to the fractional knapsack problem — sorting by
  importance density (importance / tokens) and removing from the bottom.

  Proof sketch:
    1. D(S) = Σ_all I(M_i) - Σ_{M_i ∈ S} I(M_i)
    2. Minimizing D(S) ⟺ maximizing Σ_{M_i ∈ S} I(M_i)
    3. Subject to Σ_{M_i ∈ S} tokens(M_i) <= budget
    4. This is a 0-1 knapsack; greedy by importance density is optimal
       for the fractional relaxation and near-optimal for integer case.

Importance Estimation:
  We use five proxy signals combined linearly:
    I(M_i) = w_r * recency(i) + w_d * ref_density(i) + w_t * tool_crit(i)
             + w_e * error_signal(i) + w_u * user_signal(i) - w_x * redundancy(i)

  where:
    - recency(i) = exp(-λ * (n - i) / n)          ∈ [0, 1]
    - ref_density(i) = count of later references    normalized to [0, 1]
    - tool_crit(i) = 1.0 if tool result for recent call, else 0.0
    - error_signal(i) = 1.0 if contains error/traceback
    - user_signal(i) = 1.0 if role == "user"
    - redundancy(i) = similarity to later messages  ∈ [0, 1]
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Token estimation utility
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def message_tokens(msg: Dict[str, Any]) -> int:
    """Estimate tokens in a message dict."""
    content = msg.get("content", "")
    if isinstance(content, list):
        # Multi-part content (e.g. tool results with text blocks)
        total = 0
        for part in content:
            if isinstance(part, dict):
                total += estimate_tokens(str(part.get("text", "")))
            else:
                total += estimate_tokens(str(part))
        return max(1, total)
    return estimate_tokens(str(content))


# ---------------------------------------------------------------------------
# Message role / type helpers
# ---------------------------------------------------------------------------

def _get_role(msg: Dict[str, Any]) -> str:
    return msg.get("role", "unknown")


def _get_content_str(msg: Dict[str, Any]) -> str:
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(str(p.get("text", "")))
            else:
                parts.append(str(p))
        return " ".join(parts)
    return str(content)


_ERROR_PATTERNS = re.compile(
    r"(error|exception|traceback|failed|failure|errno|panic|fatal)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# MessageImportance
# ---------------------------------------------------------------------------

class MessageImportance:
    """Estimates importance of each message as a proxy for I(M_i; Outcome).

    Weights are tunable; defaults reflect typical conversational dynamics.
    """

    def __init__(
        self,
        w_recency: float = 0.25,
        w_reference: float = 0.15,
        w_tool: float = 0.20,
        w_error: float = 0.15,
        w_user: float = 0.15,
        w_redundancy: float = 0.10,
        recency_lambda: float = 2.0,
    ):
        self.w_recency = w_recency
        self.w_reference = w_reference
        self.w_tool = w_tool
        self.w_error = w_error
        self.w_user = w_user
        self.w_redundancy = w_redundancy
        self.recency_lambda = recency_lambda

    def estimate_importance(
        self,
        message: Dict[str, Any],
        conversation_history: List[Dict[str, Any]],
    ) -> float:
        """Return importance score in [0, 1] for a single message."""
        if message not in conversation_history:
            return 0.5  # Unknown message, return neutral
        idx = next(i for i, m in enumerate(conversation_history) if m is message)
        n = len(conversation_history)

        recency = self._recency_score(idx, n)
        ref_density = self._reference_density(idx, conversation_history)
        tool_crit = self._tool_criticality(idx, message, conversation_history)
        error_sig = self._error_signal(message)
        user_sig = self._user_signal(message)
        redundancy = self._redundancy_score(idx, message, conversation_history)

        raw = (
            self.w_recency * recency
            + self.w_reference * ref_density
            + self.w_tool * tool_crit
            + self.w_error * error_sig
            + self.w_user * user_sig
            - self.w_redundancy * redundancy
        )
        return max(0.0, min(1.0, raw))

    def score_all(
        self, conversation_history: List[Dict[str, Any]]
    ) -> List[float]:
        """Score every message in the history."""
        return [
            self.estimate_importance(m, conversation_history)
            for m in conversation_history
        ]

    # -- component scores --------------------------------------------------

    def _recency_score(self, idx: int, n: int) -> float:
        if n <= 1:
            return 1.0
        normalized_age = (n - 1 - idx) / (n - 1)  # 0 = newest, 1 = oldest
        return math.exp(-self.recency_lambda * normalized_age)

    def _reference_density(
        self, idx: int, history: List[Dict[str, Any]]
    ) -> float:
        """How many later messages reference content from this message."""
        msg_content = _get_content_str(history[idx]).lower()
        if not msg_content or len(msg_content) < 10:
            return 0.0
        # Extract key terms (words >= 5 chars as rough "important" terms)
        terms = set(
            w for w in re.findall(r"\w+", msg_content) if len(w) >= 5
        )
        if not terms:
            return 0.0
        ref_count = 0
        for later_msg in history[idx + 1 :]:
            later_content = _get_content_str(later_msg).lower()
            overlap = sum(1 for t in terms if t in later_content)
            if overlap >= max(1, len(terms) * 0.2):
                ref_count += 1
        n_later = len(history) - idx - 1
        if n_later == 0:
            return 0.0
        return min(1.0, ref_count / max(1, n_later))

    def _tool_criticality(
        self,
        idx: int,
        message: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> float:
        role = _get_role(message)
        # Tool results are critical if they correspond to recent tool calls
        if role == "tool":
            # Check if there's a tool_use in the preceding assistant message
            if idx > 0 and _get_role(history[idx - 1]) == "assistant":
                return 1.0
            return 0.8
        # Assistant messages with tool_use
        if role == "assistant":
            content = message.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        return 0.9
        return 0.0

    def _error_signal(self, message: Dict[str, Any]) -> float:
        content = _get_content_str(message)
        if _ERROR_PATTERNS.search(content):
            return 1.0
        return 0.0

    def _user_signal(self, message: Dict[str, Any]) -> float:
        return 1.0 if _get_role(message) == "user" else 0.0

    def _redundancy_score(
        self,
        idx: int,
        message: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> float:
        """If message content is substantially repeated later, it's redundant."""
        content = _get_content_str(message).lower()
        if len(content) < 20:
            return 0.0
        content_words = set(re.findall(r"\w+", content))
        if not content_words:
            return 0.0
        max_overlap = 0.0
        for later_msg in history[idx + 1 :]:
            later_words = set(
                re.findall(r"\w+", _get_content_str(later_msg).lower())
            )
            if not later_words:
                continue
            overlap = len(content_words & later_words) / len(content_words)
            max_overlap = max(max_overlap, overlap)
        return max_overlap


# ---------------------------------------------------------------------------
# ContextOptimizer
# ---------------------------------------------------------------------------

@dataclass
class BudgetAllocation:
    """Token budget allocation across prompt regions."""
    system_prompt: int      # 30% — system prompt / identity
    recent_context: int     # 20% — always-keep recent messages
    optimized_history: int  # 50% — scored & pruned history

    @classmethod
    def from_total(cls, total: int) -> "BudgetAllocation":
        system = int(total * 0.30)
        recent = int(total * 0.20)
        history = total - system - recent
        return cls(system_prompt=system, recent_context=recent, optimized_history=history)


class ContextOptimizer:
    """Optimizes conversation context to fit within a token budget.

    Strategy:
      1. Score each message by importance.
      2. Protect constraints (last user message, tool result pairing).
      3. Remove lowest-importance messages first until within budget.
    """

    def __init__(
        self,
        importance_scorer: Optional[MessageImportance] = None,
        recent_window: int = 4,
    ):
        self.scorer = importance_scorer or MessageImportance()
        self.recent_window = recent_window

    def optimize(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: int,
        model: str = "default",
    ) -> List[Dict[str, Any]]:
        """Return optimized message list fitting within target_tokens.

        Invariants preserved:
          - Last user message is never removed.
          - Tool results are never orphaned (kept with their tool_use).
          - Message ordering is preserved.
        """
        if not messages:
            return []

        budget = BudgetAllocation.from_total(target_tokens)
        current_tokens = sum(message_tokens(m) for m in messages)

        if current_tokens <= target_tokens:
            return list(messages)

        n = len(messages)

        # Identify protected indices
        protected = set()

        # Protect recent window
        for i in range(max(0, n - self.recent_window), n):
            protected.add(i)

        # Protect last user message
        for i in range(n - 1, -1, -1):
            if _get_role(messages[i]) == "user":
                protected.add(i)
                break

        # Protect system messages (they use the system budget)
        for i, m in enumerate(messages):
            if _get_role(m) == "system":
                protected.add(i)

        # Build tool pairing map: tool result -> preceding tool_use
        tool_pairs: Dict[int, int] = {}
        for i, m in enumerate(messages):
            if _get_role(m) == "tool" and i > 0:
                tool_pairs[i] = i - 1
                # If tool result is protected, protect the tool_use too
                if i in protected:
                    protected.add(i - 1)

        # Score all messages
        scores = self.scorer.score_all(messages)

        # Build removal candidates sorted by importance (ascending)
        candidates = []
        for i in range(n):
            if i not in protected:
                candidates.append((scores[i], i))
        candidates.sort(key=lambda x: x[0])

        # Remove lowest-importance first until within budget
        removed = set()
        for score, idx in candidates:
            if current_tokens <= budget.optimized_history + budget.recent_context + budget.system_prompt:
                break
            # Don't orphan tool results
            if idx in tool_pairs.values():
                # This is a tool_use; check if its tool result is still present
                result_idx = [k for k, v in tool_pairs.items() if v == idx]
                if result_idx and result_idx[0] not in removed:
                    # Remove both tool_use and result together
                    current_tokens -= message_tokens(messages[idx])
                    current_tokens -= message_tokens(messages[result_idx[0]])
                    removed.add(idx)
                    removed.add(result_idx[0])
                    continue
            if idx in tool_pairs:
                # This is a tool result; remove with its tool_use
                pair_idx = tool_pairs[idx]
                if pair_idx not in protected:
                    current_tokens -= message_tokens(messages[idx])
                    current_tokens -= message_tokens(messages[pair_idx])
                    removed.add(idx)
                    removed.add(pair_idx)
                    continue

            current_tokens -= message_tokens(messages[idx])
            removed.add(idx)

        return [m for i, m in enumerate(messages) if i not in removed]


# ---------------------------------------------------------------------------
# RateDistortionAnalyzer (research / analysis tool)
# ---------------------------------------------------------------------------

class RateDistortionAnalyzer:
    """Analysis utilities for understanding context token distribution
    and information density. Useful for research and tuning."""

    def __init__(self, importance_scorer: Optional[MessageImportance] = None):
        self.scorer = importance_scorer or MessageImportance()

    def compute_token_distribution(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Token counts by role."""
        dist: Dict[str, int] = {}
        for m in messages:
            role = _get_role(m)
            dist[role] = dist.get(role, 0) + message_tokens(m)
        dist["total"] = sum(dist.values())
        return dist

    def compute_information_density(
        self, messages: List[Dict[str, Any]]
    ) -> List[float]:
        """Information density = importance / tokens for each message.

        Higher density means more important per token — these messages
        should be kept preferentially under rate constraints.
        """
        scores = self.scorer.score_all(messages)
        densities = []
        for i, m in enumerate(messages):
            tokens = message_tokens(m)
            densities.append(scores[i] / max(1, tokens) * 100)
        return densities

    def suggest_compression_strategy(
        self, messages: List[Dict[str, Any]]
    ) -> str:
        """Analyze messages and suggest a compression strategy."""
        dist = self.compute_token_distribution(messages)
        total = dist.get("total", 0)
        if total == 0:
            return "No messages to analyze."

        densities = self.compute_information_density(messages)
        scores = self.scorer.score_all(messages)

        # Find low-density, low-importance regions
        n = len(messages)
        low_importance_tokens = 0
        for i, m in enumerate(messages):
            if scores[i] < 0.3:
                low_importance_tokens += message_tokens(m)

        suggestions = []

        tool_tokens = dist.get("tool", 0)
        if tool_tokens > total * 0.5:
            suggestions.append(
                f"Tool results consume {tool_tokens}/{total} tokens "
                f"({100*tool_tokens//total}%). Consider truncating verbose "
                f"tool outputs (e.g., file reads, terminal output)."
            )

        assistant_tokens = dist.get("assistant", 0)
        if assistant_tokens > total * 0.4:
            suggestions.append(
                f"Assistant messages use {assistant_tokens}/{total} tokens "
                f"({100*assistant_tokens//total}%). Consider summarizing "
                f"older assistant reasoning."
            )

        if low_importance_tokens > total * 0.2:
            suggestions.append(
                f"Low-importance messages use ~{low_importance_tokens} tokens "
                f"({100*low_importance_tokens//total}%). These can be safely "
                f"pruned with minimal distortion."
            )

        if n > 50:
            suggestions.append(
                f"Conversation has {n} messages. Consider periodic "
                f"summarization of older exchanges."
            )

        if not suggestions:
            suggestions.append(
                "Context usage looks efficient. No major optimization needed."
            )

        return "\n".join(f"- {s}" for s in suggestions)
