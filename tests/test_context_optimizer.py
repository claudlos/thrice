"""Tests for context_optimizer.py — Information-Theoretic Context Optimization."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new-files"))

import pytest
from context_optimizer import (
    MessageImportance,
    ContextOptimizer,
    RateDistortionAnalyzer,
    BudgetAllocation,
    estimate_tokens,
    message_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_msg(role: str, content: str, **kwargs) -> dict:
    msg = {"role": role, "content": content}
    msg.update(kwargs)
    return msg


def make_conversation(n: int = 10) -> list:
    """Generate a simple alternating user/assistant conversation."""
    msgs = []
    for i in range(n):
        if i % 2 == 0:
            msgs.append(make_msg("user", f"User message number {i}: please help with task {i}"))
        else:
            msgs.append(make_msg("assistant", f"Assistant response number {i}: here is my analysis of the problem"))
    return msgs


def make_tool_conversation() -> list:
    """Conversation with tool use."""
    return [
        make_msg("user", "Read the file config.json"),
        make_msg("assistant", [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "config.json"}}]),
        make_msg("tool", "Contents of config.json: {\"key\": \"value\", \"debug\": true}"),
        make_msg("assistant", "The config file contains a key-value pair and debug is enabled."),
        make_msg("user", "Now update the debug flag to false"),
        make_msg("assistant", [{"type": "tool_use", "id": "t2", "name": "write_file", "input": {"path": "config.json"}}]),
        make_msg("tool", "File written successfully."),
        make_msg("assistant", "Done! I've updated debug to false."),
    ]


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

class TestTokenEstimation:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_string(self):
        assert estimate_tokens("hi") >= 1

    def test_rough_ratio(self):
        text = "a" * 400
        tokens = estimate_tokens(text)
        assert 90 <= tokens <= 110  # ~100 tokens for 400 chars

    def test_message_tokens_string_content(self):
        msg = make_msg("user", "Hello world, this is a test message")
        assert message_tokens(msg) > 0

    def test_message_tokens_list_content(self):
        msg = {"role": "tool", "content": [{"text": "some output here"}]}
        assert message_tokens(msg) > 0


# ---------------------------------------------------------------------------
# MessageImportance
# ---------------------------------------------------------------------------

class TestMessageImportance:
    def setup_method(self):
        self.scorer = MessageImportance()

    def test_user_messages_high_importance(self):
        history = make_conversation(6)
        scores = self.scorer.score_all(history)
        user_scores = [scores[i] for i in range(len(history)) if history[i]["role"] == "user"]
        asst_scores = [scores[i] for i in range(len(history)) if history[i]["role"] == "assistant"]
        # User messages should generally score higher
        assert sum(user_scores) / len(user_scores) > sum(asst_scores) / len(asst_scores)

    def test_recency_effect(self):
        history = make_conversation(10)
        scores = self.scorer.score_all(history)
        # Last message should score higher than first (same role type)
        assert scores[-2] > scores[0]  # Both user messages

    def test_error_messages_high_importance(self):
        history = [
            make_msg("user", "Run the build"),
            make_msg("assistant", "Running build now..."),
            make_msg("tool", "Error: compilation failed with traceback\nFileNotFoundError: missing module"),
            make_msg("user", "Fix the error"),
        ]
        scores = self.scorer.score_all(history)
        error_idx = 2
        # Error message should have high importance
        assert scores[error_idx] > 0.3

    def test_importance_in_range(self):
        history = make_conversation(20)
        scores = self.scorer.score_all(history)
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_single_message(self):
        history = [make_msg("user", "Hello")]
        score = self.scorer.estimate_importance(history[0], history)
        assert 0.0 <= score <= 1.0

    def test_tool_criticality(self):
        history = make_tool_conversation()
        scores = self.scorer.score_all(history)
        tool_idx_1 = 2  # First tool result
        # Tool result should have notable importance
        assert scores[tool_idx_1] > 0.2


# ---------------------------------------------------------------------------
# ContextOptimizer
# ---------------------------------------------------------------------------

class TestContextOptimizer:
    def setup_method(self):
        self.optimizer = ContextOptimizer()

    def test_no_optimization_when_under_budget(self):
        msgs = make_conversation(4)
        total = sum(message_tokens(m) for m in msgs)
        result = self.optimizer.optimize(msgs, total + 100)
        assert len(result) == len(msgs)

    def test_reduces_messages_when_over_budget(self):
        msgs = make_conversation(20)
        total = sum(message_tokens(m) for m in msgs)
        # Ask for half the budget
        result = self.optimizer.optimize(msgs, total // 2)
        assert len(result) < len(msgs)
        assert len(result) > 0

    def test_preserves_last_user_message(self):
        msgs = make_conversation(20)
        total = sum(message_tokens(m) for m in msgs)
        result = self.optimizer.optimize(msgs, total // 3)
        # Find last user message in original
        last_user = None
        for m in reversed(msgs):
            if m["role"] == "user":
                last_user = m
                break
        assert last_user in result

    def test_preserves_message_order(self):
        msgs = make_conversation(20)
        total = sum(message_tokens(m) for m in msgs)
        result = self.optimizer.optimize(msgs, total // 2)
        # Check that result messages appear in same relative order
        result_indices = [msgs.index(m) for m in result]
        assert result_indices == sorted(result_indices)

    def test_tool_results_not_orphaned(self):
        msgs = make_tool_conversation()
        total = sum(message_tokens(m) for m in msgs)
        # Very tight budget — but tool pairs should stay together
        result = self.optimizer.optimize(msgs, total // 2)
        for i, m in enumerate(result):
            if m.get("role") == "tool":
                # There should be an assistant message before it
                assert i > 0

    def test_empty_messages(self):
        result = self.optimizer.optimize([], 1000)
        assert result == []

    def test_system_messages_preserved(self):
        msgs = [
            make_msg("system", "You are a helpful assistant."),
            *make_conversation(10),
        ]
        total = sum(message_tokens(m) for m in msgs)
        result = self.optimizer.optimize(msgs, total // 2)
        system_msgs = [m for m in result if m["role"] == "system"]
        assert len(system_msgs) == 1


# ---------------------------------------------------------------------------
# BudgetAllocation
# ---------------------------------------------------------------------------

class TestBudgetAllocation:
    def test_allocation_sums_to_total(self):
        budget = BudgetAllocation.from_total(10000)
        assert budget.system_prompt + budget.recent_context + budget.optimized_history == 10000

    def test_allocation_ratios(self):
        budget = BudgetAllocation.from_total(10000)
        assert budget.system_prompt == 3000
        assert budget.recent_context == 2000
        assert budget.optimized_history == 5000


# ---------------------------------------------------------------------------
# RateDistortionAnalyzer
# ---------------------------------------------------------------------------

class TestRateDistortionAnalyzer:
    def setup_method(self):
        self.analyzer = RateDistortionAnalyzer()

    def test_token_distribution(self):
        msgs = make_conversation(10)
        dist = self.analyzer.compute_token_distribution(msgs)
        assert "user" in dist
        assert "assistant" in dist
        assert "total" in dist
        assert dist["total"] == dist["user"] + dist["assistant"]

    def test_information_density(self):
        msgs = make_conversation(10)
        densities = self.analyzer.compute_information_density(msgs)
        assert len(densities) == len(msgs)
        for d in densities:
            assert d >= 0

    def test_suggest_compression_strategy(self):
        msgs = make_conversation(10)
        suggestion = self.analyzer.suggest_compression_strategy(msgs)
        assert isinstance(suggestion, str)
        assert len(suggestion) > 0

    def test_tool_heavy_suggestion(self):
        # Create a conversation dominated by tool output
        msgs = [
            make_msg("user", "Run command"),
        ]
        for i in range(10):
            msgs.append(make_msg("assistant", f"Running tool {i}"))
            msgs.append(make_msg("tool", "x" * 2000))  # Huge tool outputs
        suggestion = self.analyzer.suggest_compression_strategy(msgs)
        assert "tool" in suggestion.lower() or "Tool" in suggestion

    def test_empty_messages(self):
        suggestion = self.analyzer.suggest_compression_strategy([])
        assert "No messages" in suggestion
