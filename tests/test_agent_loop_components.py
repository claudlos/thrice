"""
Tests for agent_loop_components.py — Agent Loop Decomposition.

40+ tests covering ToolDispatcher, RetryEngine, MessageProcessor,
IterationTracker, and CostTracker.
"""

import sys
import os
import time

import pytest

# Ensure the new-files directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new-files"))

from agent_loop_components import (
    ToolDispatcher,
    ToolResult,
    RetryEngine,
    RetryConfig,
    ErrorClass,
    MessageProcessor,
    IterationTracker,
    CostTracker,
)


# ===========================================================================
# ToolDispatcher tests
# ===========================================================================

class TestToolResult:
    def test_success_is_truthy(self):
        r = ToolResult(success=True, result="ok", tool_name="t", duration=0.1)
        assert r
        assert r.success is True
        assert r.error is None

    def test_failure_is_falsy(self):
        r = ToolResult(success=False, result="", tool_name="t", duration=0.1, error="boom")
        assert not r
        assert r.error == "boom"


class TestToolDispatcher:
    def setup_method(self):
        self.dispatcher = ToolDispatcher()

    def test_dispatch_success(self):
        tools = {"echo": lambda text="": text}
        result = self.dispatcher.dispatch("echo", {"text": "hello"}, tools)
        assert result.success
        assert result.result == "hello"
        assert result.tool_name == "echo"
        assert result.duration > 0

    def test_dispatch_unknown_tool(self):
        result = self.dispatcher.dispatch("nonexistent", {}, {"echo": lambda: ""})
        assert not result.success
        assert "Unknown tool" in result.error

    def test_dispatch_tool_exception(self):
        def bad_tool():
            raise RuntimeError("kaboom")
        result = self.dispatcher.dispatch("bad", {}, {"bad": bad_tool})
        assert not result.success
        assert "RuntimeError" in result.error
        assert "kaboom" in result.error

    def test_dispatch_repairs_name(self):
        tools = {"terminal": lambda cmd="": f"ran: {cmd}"}
        result = self.dispatcher.dispatch("bash", {"cmd": "ls"}, tools)
        assert result.success
        assert result.tool_name == "terminal"

    def test_repair_exact_match(self):
        assert self.dispatcher.repair_tool_name("foo", ["foo", "bar"]) == "foo"

    def test_repair_case_insensitive(self):
        assert self.dispatcher.repair_tool_name("FOO", ["foo", "bar"]) == "foo"

    def test_repair_alias(self):
        assert self.dispatcher.repair_tool_name("bash", ["terminal", "read_file"]) == "terminal"
        assert self.dispatcher.repair_tool_name("grep", ["search", "read_file"]) == "search"
        assert self.dispatcher.repair_tool_name("cat", ["search", "read_file"]) == "read_file"

    def test_repair_fuzzy_match(self):
        # "termnal" is close to "terminal"
        result = self.dispatcher.repair_tool_name("termnal", ["terminal", "search"])
        assert result == "terminal"

    def test_repair_no_match(self):
        result = self.dispatcher.repair_tool_name("zzzzz", ["terminal", "search"])
        assert result == "zzzzz"  # unchanged

    def test_repair_empty_tools(self):
        assert self.dispatcher.repair_tool_name("foo", []) == "foo"

    def test_validate_args_valid(self):
        schema = {
            "required": ["path"],
            "properties": {
                "path": {"type": "string"},
                "line": {"type": "integer"},
            },
        }
        errors = self.dispatcher.validate_args("read", {"path": "/foo", "line": 10}, schema)
        assert errors == []

    def test_validate_args_missing_required(self):
        schema = {"required": ["path"], "properties": {"path": {"type": "string"}}}
        errors = self.dispatcher.validate_args("read", {}, schema)
        assert any("Missing required" in e for e in errors)

    def test_validate_args_wrong_type(self):
        schema = {"required": [], "properties": {"count": {"type": "integer"}}}
        errors = self.dispatcher.validate_args("t", {"count": "not_int"}, schema)
        assert any("expected integer" in e for e in errors)

    def test_validate_args_unknown_param(self):
        schema = {"required": [], "properties": {"name": {"type": "string"}}}
        errors = self.dispatcher.validate_args("t", {"unknown": "val"}, schema)
        assert any("Unknown parameter" in e for e in errors)

    def test_custom_aliases(self):
        d = ToolDispatcher(aliases={"mycmd": "my_tool"})
        assert d.repair_tool_name("mycmd", ["my_tool", "other"]) == "my_tool"


# ===========================================================================
# RetryEngine tests
# ===========================================================================

class TestRetryConfig:
    def test_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.strategy == "exponential"
        assert ErrorClass.TRANSIENT.value in cfg.retryable_errors

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unknown retry strategy"):
            RetryConfig(strategy="random")


class TestRetryEngine:
    def setup_method(self):
        self.engine = RetryEngine()

    def test_classify_rate_limit(self):
        err = Exception("Rate limit exceeded (429)")
        assert self.engine.classify_error(err) == "rate_limit"

    def test_classify_auth(self):
        err = Exception("401 Unauthorized: invalid API key")
        assert self.engine.classify_error(err) == "auth"

    def test_classify_context_overflow(self):
        err = Exception("context_length exceeded: max_tokens is 200000")
        assert self.engine.classify_error(err) == "context_overflow"

    def test_classify_transient(self):
        err = ConnectionError("Connection refused")
        assert self.engine.classify_error(err) == "transient"

    def test_classify_timeout(self):
        err = TimeoutError("Request timed out")
        assert self.engine.classify_error(err) == "transient"

    def test_classify_permanent(self):
        err = ValueError("Invalid JSON in response")
        assert self.engine.classify_error(err) == "permanent"

    def test_should_retry_transient(self):
        err = ConnectionError("timeout")
        assert self.engine.should_retry(err, attempt=0)
        assert self.engine.should_retry(err, attempt=2)
        assert not self.engine.should_retry(err, attempt=3)

    def test_should_not_retry_permanent(self):
        err = ValueError("bad input")
        assert not self.engine.should_retry(err, attempt=0)

    def test_should_not_retry_auth(self):
        err = Exception("401 Unauthorized")
        assert not self.engine.should_retry(err, attempt=0)

    def test_delay_exponential(self):
        assert self.engine.get_delay(0) == 1.0
        assert self.engine.get_delay(1) == 2.0
        assert self.engine.get_delay(2) == 4.0
        assert self.engine.get_delay(3) == 8.0

    def test_delay_linear(self):
        assert self.engine.get_delay(0, strategy="linear") == 1.0
        assert self.engine.get_delay(1, strategy="linear") == 2.0
        assert self.engine.get_delay(2, strategy="linear") == 3.0

    def test_delay_constant(self):
        assert self.engine.get_delay(0, strategy="constant") == 1.0
        assert self.engine.get_delay(5, strategy="constant") == 1.0

    def test_delay_capped(self):
        cfg = RetryConfig(base_delay=10.0, max_delay=30.0)
        engine = RetryEngine(cfg)
        assert engine.get_delay(5) <= 30.0

    def test_execute_with_retry_success(self):
        calls = []
        def fn():
            calls.append(1)
            return "ok"
        result = self.engine.execute_with_retry(fn)
        assert result == "ok"
        assert len(calls) == 1

    def test_execute_with_retry_transient_then_success(self):
        attempts = []
        def fn():
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("timeout")
            return "recovered"

        cfg = RetryConfig(max_retries=5, base_delay=0.01)
        result = self.engine.execute_with_retry(fn, config=cfg)
        assert result == "recovered"
        assert len(attempts) == 3

    def test_execute_with_retry_exhausted(self):
        def fn():
            raise ConnectionError("always fails")

        cfg = RetryConfig(max_retries=2, base_delay=0.01)
        with pytest.raises(ConnectionError, match="always fails"):
            self.engine.execute_with_retry(fn, config=cfg)

    def test_execute_with_retry_permanent_no_retry(self):
        attempts = []
        def fn():
            attempts.append(1)
            raise ValueError("permanent error")

        cfg = RetryConfig(max_retries=5, base_delay=0.01)
        with pytest.raises(ValueError, match="permanent"):
            self.engine.execute_with_retry(fn, config=cfg)
        assert len(attempts) == 1


# ===========================================================================
# MessageProcessor tests
# ===========================================================================

class TestMessageProcessor:
    def setup_method(self):
        self.proc = MessageProcessor()

    def test_validate_valid_messages(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        assert self.proc.validate_messages(msgs) == []

    def test_validate_empty(self):
        errors = self.proc.validate_messages([])
        assert any("empty" in e.lower() for e in errors)

    def test_validate_missing_role(self):
        errors = self.proc.validate_messages([{"content": "hi"}])
        assert any("missing 'role'" in e for e in errors)

    def test_validate_invalid_role(self):
        errors = self.proc.validate_messages([{"role": "alien", "content": "hi"}])
        assert any("invalid role" in e for e in errors)

    def test_validate_system_not_first(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "sys"},
        ]
        errors = self.proc.validate_messages(msgs)
        assert any("first" in e.lower() for e in errors)

    def test_validate_missing_content(self):
        errors = self.proc.validate_messages([{"role": "user"}])
        assert any("missing 'content'" in e for e in errors)

    def test_trim_within_budget(self):
        msgs = [{"role": "user", "content": "hi"}]
        counter = lambda m: len(m) * 10
        result = self.proc.trim_to_budget(msgs, max_tokens=100, token_counter=counter)
        assert len(result) == 1

    def test_trim_preserves_system(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old msg"},
            {"role": "assistant", "content": "old reply"},
            {"role": "user", "content": "new msg"},
        ]
        # Counter: 10 tokens per message
        counter = lambda m: len(m) * 10
        result = self.proc.trim_to_budget(msgs, max_tokens=25, token_counter=counter)
        assert result[0]["role"] == "system"
        assert len(result) <= 3

    def test_trim_removes_oldest_first(self):
        msgs = [
            {"role": "user", "content": "msg1"},
            {"role": "user", "content": "msg2"},
            {"role": "user", "content": "msg3"},
        ]
        counter = lambda m: len(m) * 10
        result = self.proc.trim_to_budget(msgs, max_tokens=15, token_counter=counter)
        # Should keep most recent
        assert result[-1]["content"] == "msg3"

    def test_trim_empty(self):
        assert self.proc.trim_to_budget([], 100, lambda m: 0) == []

    def test_inject_system_note_existing(self):
        msgs = [
            {"role": "system", "content": "Original."},
            {"role": "user", "content": "hi"},
        ]
        result = self.proc.inject_system_note(msgs, "IMPORTANT NOTE")
        assert "Original." in result[0]["content"]
        assert "IMPORTANT NOTE" in result[0]["content"]
        assert len(result) == 2
        # Original not mutated
        assert "IMPORTANT" not in msgs[0]["content"]

    def test_inject_system_note_none(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = self.proc.inject_system_note(msgs, "System note")
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "System note"
        assert len(result) == 2

    def test_extract_tool_calls_openai(self):
        msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/foo.py"}',
                    },
                }
            ],
        }
        calls = self.proc.extract_tool_calls(msg)
        assert len(calls) == 1
        assert calls[0]["name"] == "read_file"
        assert calls[0]["arguments"] == {"path": "/foo.py"}

    def test_extract_tool_calls_anthropic(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me read that."},
                {
                    "type": "tool_use",
                    "id": "tu_456",
                    "name": "search",
                    "input": {"query": "hello"},
                },
            ],
        }
        calls = self.proc.extract_tool_calls(msg)
        assert len(calls) == 1
        assert calls[0]["name"] == "search"
        assert calls[0]["arguments"] == {"query": "hello"}

    def test_extract_tool_calls_none(self):
        msg = {"role": "assistant", "content": "Just text."}
        assert self.proc.extract_tool_calls(msg) == []

    def test_format_tool_result(self):
        result = self.proc.format_tool_result("call_123", "file contents here")
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert result["content"] == "file contents here"


# ===========================================================================
# IterationTracker tests
# ===========================================================================

class TestIterationTracker:
    def test_start_iteration(self):
        tracker = IterationTracker()
        tracker.start_iteration(0)
        tracker.start_iteration(1)
        stats = tracker.get_stats()
        assert stats["total_iterations"] == 2

    def test_record_tool_call(self):
        tracker = IterationTracker()
        tracker.start_iteration(0)
        tracker.record_tool_call("terminal", {"cmd": "ls"}, "output", 0.5)
        stats = tracker.get_stats()
        assert stats["total_tool_calls"] == 1

    def test_record_api_call(self):
        tracker = IterationTracker()
        tracker.start_iteration(0)
        tracker.record_api_call("claude-sonnet-4", 1000, 500, 1.2)
        stats = tracker.get_stats()
        assert stats["total_api_calls"] == 1
        assert stats["total_input_tokens"] == 1000
        assert stats["total_output_tokens"] == 500
        assert stats["total_tokens"] == 1500

    def test_auto_start_iteration(self):
        tracker = IterationTracker()
        # Should auto-start when recording without explicit start
        tracker.record_tool_call("terminal", {}, "out", 0.1)
        assert tracker.get_stats()["total_iterations"] == 1

    def test_is_stuck_not_enough_data(self):
        tracker = IterationTracker()
        tracker.start_iteration(0)
        tracker.record_tool_call("terminal", {"cmd": "ls"}, "out", 0.1)
        assert not tracker.is_stuck(window=5)

    def test_is_stuck_detected(self):
        tracker = IterationTracker()
        for i in range(6):
            tracker.start_iteration(i)
            tracker.record_tool_call("terminal", {"cmd": "ls"}, "out", 0.1)
        assert tracker.is_stuck(window=5)

    def test_is_stuck_different_tools(self):
        tracker = IterationTracker()
        tools = ["terminal", "read_file", "search", "write_file", "terminal"]
        for i, tool in enumerate(tools):
            tracker.start_iteration(i)
            tracker.record_tool_call(tool, {"arg": str(i)}, "out", 0.1)
        assert not tracker.is_stuck(window=5)

    def test_suggest_escape_no_data(self):
        tracker = IterationTracker()
        suggestion = tracker.suggest_escape()
        assert "No iterations" in suggestion

    def test_suggest_escape_with_data(self):
        tracker = IterationTracker()
        for i in range(5):
            tracker.start_iteration(i)
            tracker.record_tool_call("terminal", {"cmd": "make"}, "error", 0.1)
        suggestion = tracker.suggest_escape()
        assert "terminal" in suggestion
        assert len(suggestion) > 20

    def test_get_stats_empty(self):
        tracker = IterationTracker()
        stats = tracker.get_stats()
        assert stats["total_iterations"] == 0
        assert stats["total_tool_calls"] == 0
        assert stats["total_tokens"] == 0


# ===========================================================================
# CostTracker tests
# ===========================================================================

class TestCostTracker:
    def test_record_returns_cost(self):
        tracker = CostTracker()
        cost = tracker.record("gpt-4o-mini", 1_000_000, 0)
        # gpt-4o-mini input: $0.15/1M
        assert abs(cost - 0.15) < 0.001

    def test_session_cost(self):
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", 1_000_000, 1_000_000)
        # input: 0.15 + output: 0.60 = 0.75
        assert abs(tracker.get_session_cost() - 0.75) < 0.001

    def test_multiple_models(self):
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", 100_000, 50_000)
        tracker.record("claude-sonnet-4", 100_000, 50_000)
        stats = tracker.get_session_stats()
        assert stats["total_calls"] == 2
        assert len(stats["models"]) == 2

    def test_session_stats_structure(self):
        tracker = CostTracker()
        tracker.record("claude-sonnet-4", 10000, 5000)
        stats = tracker.get_session_stats()
        assert "total_calls" in stats
        assert "total_input_tokens" in stats
        assert "total_output_tokens" in stats
        assert "total_tokens" in stats
        assert "total_cost_usd" in stats
        assert "models" in stats

    def test_format_summary_empty(self):
        tracker = CostTracker()
        summary = tracker.format_summary()
        assert "No API calls" in summary

    def test_format_summary_with_data(self):
        tracker = CostTracker()
        tracker.record("claude-sonnet-4", 50000, 10000)
        summary = tracker.format_summary()
        assert "Session Cost:" in summary
        assert "Total Calls:" in summary
        assert "Total Tokens:" in summary

    def test_unknown_model_fallback(self):
        tracker = CostTracker()
        # Should not crash, uses default pricing
        cost = tracker.record("unknown-model-xyz", 1_000_000, 0)
        assert cost > 0  # uses fallback pricing

    def test_prefix_matching(self):
        tracker = CostTracker()
        # "claude-sonnet-4-custom" should match "claude-sonnet-4"
        cost = tracker.record("claude-sonnet-4-custom", 1_000_000, 0)
        assert abs(cost - 3.00) < 0.001

    def test_custom_pricing(self):
        tracker = CostTracker(pricing={"my-model": (1.0, 2.0)})
        cost = tracker.record("my-model", 1_000_000, 1_000_000)
        assert abs(cost - 3.0) < 0.001

    def test_claude_opus_pricing(self):
        tracker = CostTracker()
        cost = tracker.record("claude-opus-4", 1_000_000, 1_000_000)
        # input: 15.00 + output: 75.00 = 90.00
        assert abs(cost - 90.0) < 0.001

    def test_zero_tokens(self):
        tracker = CostTracker()
        cost = tracker.record("gpt-4o", 0, 0)
        assert cost == 0.0
        assert tracker.get_session_cost() == 0.0


# ===========================================================================
# Integration / cross-component tests
# ===========================================================================

class TestIntegration:
    def test_dispatch_and_track(self):
        """ToolDispatcher + IterationTracker working together."""
        dispatcher = ToolDispatcher()
        tracker = IterationTracker()

        tools = {"echo": lambda text="": text}
        tracker.start_iteration(0)

        result = dispatcher.dispatch("echo", {"text": "hello"}, tools)
        tracker.record_tool_call(
            result.tool_name, {"text": "hello"}, result.result, result.duration
        )

        stats = tracker.get_stats()
        assert stats["total_tool_calls"] == 1

    def test_retry_with_cost_tracking(self):
        """RetryEngine + CostTracker working together."""
        cost_tracker = CostTracker()
        engine = RetryEngine()

        attempts = []
        def api_call():
            attempts.append(1)
            cost_tracker.record("gpt-4o-mini", 100, 50)
            if len(attempts) < 2:
                raise ConnectionError("timeout")
            return "success"

        cfg = RetryConfig(max_retries=3, base_delay=0.01)
        result = engine.execute_with_retry(api_call, config=cfg)
        assert result == "success"
        assert cost_tracker.get_session_stats()["total_calls"] == 2
