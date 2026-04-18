"""Tests for debugging_guidance module."""

import os
import sys

import pytest

# Add the new-files directory to path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'new-files'))

from debugging_guidance import (
    CODING_BEST_PRACTICES,
    DEBUGGING_GUIDANCE,
    _extract_text_from_message,
    count_recent_errors,
    get_debugging_prompt,
    is_repeated_failure,
    should_inject_debugging_guidance,
)

# ── Constants tests ──────────────────────────────────────────────────────

class TestConstants:
    def test_debugging_guidance_is_nonempty_string(self):
        assert isinstance(DEBUGGING_GUIDANCE, str)
        assert len(DEBUGGING_GUIDANCE) > 100

    def test_debugging_guidance_contains_key_steps(self):
        assert "error trace" in DEBUGGING_GUIDANCE.lower()
        assert "smallest possible fix" in DEBUGGING_GUIDANCE.lower()
        assert "tests after" in DEBUGGING_GUIDANCE.lower()
        assert "3 failed attempts" in DEBUGGING_GUIDANCE.lower() or "three" in DEBUGGING_GUIDANCE.lower()
        assert "import" in DEBUGGING_GUIDANCE.lower()

    def test_coding_best_practices_is_nonempty_string(self):
        assert isinstance(CODING_BEST_PRACTICES, str)
        assert len(CODING_BEST_PRACTICES) > 50

    def test_coding_best_practices_contains_key_principles(self):
        assert "read before write" in CODING_BEST_PRACTICES.lower()
        assert "verify assumptions" in CODING_BEST_PRACTICES.lower() or "verify" in CODING_BEST_PRACTICES.lower()
        assert "one change at a time" in CODING_BEST_PRACTICES.lower() or "atomic" in CODING_BEST_PRACTICES.lower()


# ── Message text extraction tests ────────────────────────────────────────

class TestExtractText:
    def test_string_content(self):
        msg = {"role": "user", "content": "hello world"}
        assert _extract_text_from_message(msg) == "hello world"

    def test_list_content_with_text_blocks(self):
        msg = {"role": "assistant", "content": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]}
        result = _extract_text_from_message(msg)
        assert "first" in result
        assert "second" in result

    def test_list_content_with_tool_result(self):
        msg = {"role": "user", "content": [
            {"type": "tool_result", "content": "Error: file not found"},
        ]}
        result = _extract_text_from_message(msg)
        assert "Error: file not found" in result

    def test_empty_content(self):
        msg = {"role": "assistant", "content": ""}
        assert _extract_text_from_message(msg) == ""

    def test_missing_content(self):
        msg = {"role": "assistant"}
        assert _extract_text_from_message(msg) == ""

    def test_none_content(self):
        msg = {"role": "assistant", "content": None}
        assert _extract_text_from_message(msg) == ""


# ── Error detection tests ────────────────────────────────────────────────

class TestShouldInjectDebuggingGuidance:
    def test_empty_messages(self):
        assert should_inject_debugging_guidance([]) is False

    def test_no_errors(self):
        messages = [
            {"role": "user", "content": "Write a hello world program"},
            {"role": "assistant", "content": "Here's the program..."},
        ]
        assert should_inject_debugging_guidance(messages) is False

    def test_python_traceback(self):
        messages = [
            {"role": "user", "content": "Run the tests"},
            {"role": "assistant", "content": (
                "Traceback (most recent call last):\n"
                "  File 'test.py', line 5\n"
                "    x = 1/0\n"
                "ZeroDivisionError: division by zero"
            )},
        ]
        assert should_inject_debugging_guidance(messages) is True

    def test_test_failure(self):
        messages = [
            {"role": "user", "content": "Run pytest"},
            {"role": "assistant", "content": "FAILED tests/test_foo.py::test_bar - AssertionError"},
        ]
        assert should_inject_debugging_guidance(messages) is True

    def test_import_error(self):
        messages = [
            {"role": "assistant", "content": "ImportError: No module named 'foo'"},
        ]
        assert should_inject_debugging_guidance(messages) is True

    def test_type_error(self):
        messages = [
            {"role": "assistant", "content": "TypeError: expected str, got int"},
        ]
        assert should_inject_debugging_guidance(messages) is True

    def test_build_failure(self):
        messages = [
            {"role": "assistant", "content": "BUILD FAILED in 5s"},
        ]
        assert should_inject_debugging_guidance(messages) is True

    def test_js_module_error(self):
        messages = [
            {"role": "assistant", "content": "Cannot find module '@/components/Button'"},
        ]
        assert should_inject_debugging_guidance(messages) is True

    def test_exit_code_error(self):
        messages = [
            {"role": "assistant", "content": "command exited with non-zero exit code"},
        ]
        assert should_inject_debugging_guidance(messages) is True

    def test_error_in_tool_result(self):
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "content": "FileNotFoundError: [Errno 2] No such file"},
            ]},
        ]
        assert should_inject_debugging_guidance(messages) is True

    def test_only_scans_recent_messages(self):
        """Errors far back in history shouldn't trigger guidance."""
        old_error = {"role": "assistant", "content": "Traceback (most recent call last):\nValueError: bad"}
        clean_msgs = [{"role": "user", "content": f"message {i}"} for i in range(15)]
        messages = [old_error] + clean_msgs
        # Error is more than 10 messages ago
        assert should_inject_debugging_guidance(messages) is False


# ── Error counting tests ─────────────────────────────────────────────────

class TestCountRecentErrors:
    def test_empty(self):
        assert count_recent_errors([]) == 0

    def test_no_errors(self):
        messages = [{"role": "user", "content": "hello"}]
        assert count_recent_errors(messages) == 0

    def test_single_error(self):
        messages = [
            {"role": "assistant", "content": "TypeError: bad arg"},
        ]
        assert count_recent_errors(messages) == 1

    def test_multiple_errors(self):
        messages = [
            {"role": "assistant", "content": "TypeError: bad arg"},
            {"role": "user", "content": "try again"},
            {"role": "assistant", "content": "ValueError: wrong value"},
            {"role": "user", "content": "hmm"},
            {"role": "assistant", "content": "KeyError: 'missing'"},
        ]
        assert count_recent_errors(messages) == 3

    def test_counts_each_message_once(self):
        """A message with multiple errors should only count as 1."""
        messages = [
            {"role": "assistant", "content": (
                "TypeError: bad\n"
                "ValueError: also bad\n"
                "KeyError: missing too"
            )},
        ]
        assert count_recent_errors(messages) == 1


# ── Repeated failure detection ───────────────────────────────────────────

class TestIsRepeatedFailure:
    def test_empty(self):
        assert is_repeated_failure([]) is False

    def test_explicit_repeated_failure_language(self):
        messages = [
            {"role": "user", "content": "It's still failing with the same error"},
        ]
        assert is_repeated_failure(messages) is True

    def test_not_fixed_language(self):
        messages = [
            {"role": "user", "content": "That's not fixed, same error again"},
        ]
        assert is_repeated_failure(messages) is True

    def test_three_consecutive_errors(self):
        messages = [
            {"role": "assistant", "content": "TypeError: x"},
            {"role": "user", "content": "try something else"},
            {"role": "assistant", "content": "ValueError: y"},
            {"role": "user", "content": "hmm"},
            {"role": "assistant", "content": "KeyError: z"},
        ]
        assert is_repeated_failure(messages) is True

    def test_no_repeated_failure(self):
        messages = [
            {"role": "user", "content": "Can you write a function?"},
            {"role": "assistant", "content": "Sure, here it is."},
        ]
        assert is_repeated_failure(messages) is False


# ── get_debugging_prompt tests ───────────────────────────────────────────

class TestGetDebuggingPrompt:
    def test_no_messages_returns_best_practices_only(self):
        result = get_debugging_prompt()
        assert CODING_BEST_PRACTICES in result
        assert DEBUGGING_GUIDANCE not in result

    def test_no_errors_returns_best_practices_only(self):
        messages = [{"role": "user", "content": "hello"}]
        result = get_debugging_prompt(messages)
        assert CODING_BEST_PRACTICES in result
        assert DEBUGGING_GUIDANCE not in result

    def test_with_errors_includes_debugging_guidance(self):
        messages = [
            {"role": "assistant", "content": "Traceback (most recent call last):\nValueError: oops"},
        ]
        result = get_debugging_prompt(messages)
        assert DEBUGGING_GUIDANCE in result
        assert CODING_BEST_PRACTICES in result

    def test_repeated_failure_includes_escalation(self):
        messages = [
            {"role": "assistant", "content": "TypeError: x"},
            {"role": "user", "content": "try again"},
            {"role": "assistant", "content": "TypeError: x"},
            {"role": "user", "content": "still failing"},
            {"role": "assistant", "content": "TypeError: x"},
        ]
        result = get_debugging_prompt(messages)
        assert "Repeated failure detected" in result
        assert "completely different approach" in result

    def test_returns_string(self):
        result = get_debugging_prompt()
        assert isinstance(result, str)

    def test_none_messages_same_as_no_messages(self):
        result = get_debugging_prompt(None)
        assert CODING_BEST_PRACTICES in result
        assert DEBUGGING_GUIDANCE not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
