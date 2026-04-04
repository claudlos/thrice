"""Tests for verified_messages.py — Verified Message Protocol (#14)."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new-files"))

from verified_messages import (
    MessageBuilder, MessageValidator, AutoFixer,
    InvalidMessageSequence, ValidationResult, ValidationError
)


class TestMessageBuilder:
    def test_simple_conversation(self):
        msgs = (MessageBuilder()
                .user("Hello")
                .assistant("Hi there!")
                .user("How are you?")
                .assistant("I'm good!")
                .build())
        assert len(msgs) == 4
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_system_first(self):
        msgs = (MessageBuilder()
                .system("You are helpful.")
                .user("Hello")
                .assistant("Hi!")
                .build())
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"

    def test_system_not_first_raises(self):
        with pytest.raises(InvalidMessageSequence, match="first"):
            MessageBuilder().user("Hi").system("Nope")

    def test_consecutive_user_raises(self):
        with pytest.raises(InvalidMessageSequence):
            MessageBuilder().user("Hi").user("Again")

    def test_consecutive_assistant_raises(self):
        with pytest.raises(InvalidMessageSequence):
            MessageBuilder().user("Hi").assistant("A").assistant("B")

    def test_assistant_first_raises(self):
        with pytest.raises(InvalidMessageSequence, match="first"):
            MessageBuilder().assistant("Hello")

    def test_tool_calls_and_results(self):
        tool_calls = [{"id": "tc1", "function": {"name": "search", "arguments": "{}"}}]
        msgs = (MessageBuilder()
                .user("Search for X")
                .assistant("Let me search.", tool_calls=tool_calls)
                .tool_result("tc1", "Found X")
                .assistant("I found X!")
                .build())
        assert len(msgs) == 4
        assert msgs[2]["role"] == "tool"
        assert msgs[2]["tool_call_id"] == "tc1"

    def test_tool_result_wrong_id_raises(self):
        tool_calls = [{"id": "tc1", "function": {"name": "search"}}]
        with pytest.raises(InvalidMessageSequence, match="not found"):
            (MessageBuilder()
             .user("Hi")
             .assistant("Ok", tool_calls=tool_calls)
             .tool_result("wrong_id", "result"))

    def test_tool_result_without_assistant_raises(self):
        with pytest.raises(InvalidMessageSequence):
            MessageBuilder().user("Hi").tool_result("tc1", "result")

    def test_duplicate_tool_result_raises(self):
        tool_calls = [{"id": "tc1", "function": {"name": "search"}}]
        with pytest.raises(InvalidMessageSequence, match="already"):
            (MessageBuilder()
             .user("Hi")
             .assistant("Ok", tool_calls=tool_calls)
             .tool_result("tc1", "result1")
             .tool_result("tc1", "result2"))

    def test_multiple_tool_calls(self):
        tool_calls = [
            {"id": "tc1", "function": {"name": "search"}},
            {"id": "tc2", "function": {"name": "read"}},
        ]
        msgs = (MessageBuilder()
                .user("Do both")
                .assistant("On it", tool_calls=tool_calls)
                .tool_result("tc1", "search result")
                .tool_result("tc2", "read result")
                .assistant("Done!")
                .build())
        assert len(msgs) == 5

    def test_immutability(self):
        b1 = MessageBuilder().user("Hello")
        b2 = b1.assistant("Hi")
        # b1 should be unmodified
        assert len(b1) == 1
        assert len(b2) == 2

    def test_empty_build_raises(self):
        with pytest.raises(InvalidMessageSequence):
            MessageBuilder().build()


class TestMessageValidator:
    def test_valid_sequence(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = MessageValidator().validate(msgs)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_empty_sequence(self):
        result = MessageValidator().validate([])
        assert result.valid is False

    def test_consecutive_users(self):
        msgs = [
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},
        ]
        result = MessageValidator().validate(msgs)
        assert result.valid is False
        assert any("Consecutive user" in e.message for e in result.errors)

    def test_orphaned_tool_result(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Ok"},
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
        ]
        result = MessageValidator().validate(msgs)
        assert result.valid is False
        assert any("Orphaned" in e.message for e in result.errors)

    def test_missing_tool_result_warning(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Ok",
             "tool_calls": [{"id": "tc1", "function": {"name": "x"}}]},
        ]
        result = MessageValidator().validate(msgs)
        # Missing tool result is a warning, not error
        assert len(result.warnings) > 0
        assert any("no corresponding result" in w.message for w in result.warnings)

    def test_system_not_first(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Be nice"},
        ]
        result = MessageValidator().validate(msgs)
        assert result.valid is False

    def test_invalid_role(self):
        msgs = [{"role": "banana", "content": "Hi"}]
        result = MessageValidator().validate(msgs)
        assert result.valid is False

    def test_validation_result_bool(self):
        good = ValidationResult(valid=True)
        bad = ValidationResult(valid=False, errors=[
            ValidationError(position=0, message="bad", severity="error")
        ])
        assert bool(good) is True
        assert bool(bad) is False


class TestAutoFixer:
    def test_fix_consecutive_users(self):
        msgs = [
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},
        ]
        fixed = AutoFixer().fix(msgs)
        validator = MessageValidator()
        result = validator.validate(fixed)
        assert result.valid is True

    def test_fix_orphaned_tool_result(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Ok"},
            {"role": "tool", "tool_call_id": "orphan", "content": "result"},
        ]
        fixed = AutoFixer().fix(msgs)
        # Orphaned result should be removed
        tool_msgs = [m for m in fixed if m.get("role") == "tool"]
        assert len(tool_msgs) == 0

    def test_fix_system_position(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Be nice"},
        ]
        fixed = AutoFixer().fix(msgs)
        assert fixed[0]["role"] == "system"

    def test_fix_missing_tool_result(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Ok",
             "tool_calls": [{"id": "tc1", "function": {"name": "x"}}]},
            {"role": "user", "content": "What happened?"},
        ]
        fixed = AutoFixer().fix(msgs)
        tool_msgs = [m for m in fixed if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "tc1"

    def test_fix_empty_preserves(self):
        assert AutoFixer().fix([]) == []

    def test_fix_already_valid(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        fixed = AutoFixer().fix(msgs)
        assert len(fixed) == 2
        assert fixed[0]["role"] == "user"
