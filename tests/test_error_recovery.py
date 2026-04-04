"""
Tests for error_recovery module — Error Classification and Recovery Strategy.

Covers: ErrorCategory, ErrorClassifier, RecoveryEngine, RetryTracker,
        is_same_error, RecoveryStrategy.
"""

import sys
import os
import pytest

# Allow imports from new-files
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new-files"))

from error_recovery import (
    ErrorCategory,
    ErrorClassifier,
    RecoveryEngine,
    RecoveryStrategy,
    RetryTracker,
    is_same_error,
)


# ─── ErrorCategory Tests ──────────────────────────────────────────────────────


class TestErrorCategory:
    def test_all_categories_exist(self):
        expected = {
            "SYNTAX", "IMPORT", "TYPE_ERROR", "ASSERTION", "RUNTIME",
            "TIMEOUT", "PERMISSION", "FILE_NOT_FOUND", "NETWORK",
            "API_ERROR", "TOOL_ERROR", "LOGIC", "UNKNOWN",
        }
        actual = {c.name for c in ErrorCategory}
        assert actual == expected

    def test_category_is_str_enum(self):
        assert isinstance(ErrorCategory.SYNTAX, str)
        assert ErrorCategory.SYNTAX == "syntax"

    def test_category_values_are_lowercase(self):
        for cat in ErrorCategory:
            assert cat.value == cat.value.lower()


# ─── ErrorClassifier.classify Tests ───────────────────────────────────────────


class TestClassify:
    def setup_method(self):
        self.classifier = ErrorClassifier()

    def test_syntax_error(self):
        assert self.classifier.classify("SyntaxError: invalid syntax") == ErrorCategory.SYNTAX

    def test_indentation_error(self):
        assert self.classifier.classify("IndentationError: unexpected indent") == ErrorCategory.SYNTAX

    def test_import_error(self):
        assert self.classifier.classify("ModuleNotFoundError: No module named 'foo'") == ErrorCategory.IMPORT

    def test_import_error_cannot_import(self):
        assert self.classifier.classify("ImportError: cannot import name 'bar' from 'baz'") == ErrorCategory.IMPORT

    def test_type_error(self):
        assert self.classifier.classify("TypeError: expected str, got int") == ErrorCategory.TYPE_ERROR

    def test_assertion_error(self):
        assert self.classifier.classify("AssertionError: assert 1 == 2") == ErrorCategory.ASSERTION

    def test_timeout_error(self):
        assert self.classifier.classify("TimeoutError: operation timed out") == ErrorCategory.TIMEOUT

    def test_permission_error(self):
        assert self.classifier.classify("PermissionError: [Errno 13] Permission denied") == ErrorCategory.PERMISSION

    def test_file_not_found(self):
        assert self.classifier.classify("FileNotFoundError: No such file or directory: 'foo.py'") == ErrorCategory.FILE_NOT_FOUND

    def test_network_error(self):
        assert self.classifier.classify("ConnectionRefusedError: [Errno 111] Connection refused") == ErrorCategory.NETWORK

    def test_api_error_429(self):
        assert self.classifier.classify("HTTPError: 429 Too Many Requests") == ErrorCategory.API_ERROR

    def test_api_error_500(self):
        assert self.classifier.classify("Server returned status code 500") == ErrorCategory.API_ERROR

    def test_tool_error_match(self):
        assert self.classifier.classify("unique match not found for old_string") == ErrorCategory.TOOL_ERROR

    def test_runtime_error_key(self):
        assert self.classifier.classify("KeyError: 'missing_key'") == ErrorCategory.RUNTIME

    def test_runtime_error_index(self):
        assert self.classifier.classify("IndexError: list index out of range") == ErrorCategory.RUNTIME

    def test_logic_error(self):
        assert self.classifier.classify("expected 42 but got 0") == ErrorCategory.LOGIC

    def test_unknown_error(self):
        assert self.classifier.classify("something completely unrecognizable happened") == ErrorCategory.UNKNOWN

    def test_empty_string(self):
        assert self.classifier.classify("") == ErrorCategory.UNKNOWN

    def test_whitespace_only(self):
        assert self.classifier.classify("   \n  ") == ErrorCategory.UNKNOWN

    def test_multiline_traceback(self):
        tb = """Traceback (most recent call last):
  File "test.py", line 10, in main
    import nonexistent_module
ModuleNotFoundError: No module named 'nonexistent_module'"""
        assert self.classifier.classify(tb) == ErrorCategory.IMPORT


# ─── ErrorClassifier.extract_details Tests ────────────────────────────────────


class TestExtractDetails:
    def setup_method(self):
        self.classifier = ErrorClassifier()

    def test_extract_file_and_line(self):
        error = 'File "app/main.py", line 42, in process'
        details = self.classifier.extract_details(error)
        assert details["file"] == "app/main.py"
        assert details["line"] == "42"

    def test_extract_function(self):
        error = """Traceback (most recent call last):
  File "test.py", line 5, in my_function
    raise ValueError"""
        details = self.classifier.extract_details(error)
        assert details["function"] == "my_function"

    def test_extract_variable_name_error(self):
        error = "NameError: name 'undefined_var' is not defined"
        details = self.classifier.extract_details(error)
        assert details["variable"] == "undefined_var"

    def test_extract_module_name(self):
        error = "ModuleNotFoundError: No module named 'requests'"
        details = self.classifier.extract_details(error)
        assert details["module_name"] == "requests"

    def test_extract_all_none_for_garbage(self):
        details = self.classifier.extract_details("generic problem")
        assert all(v is None for v in details.values())

    def test_extract_has_all_keys(self):
        details = self.classifier.extract_details("anything")
        expected_keys = {"file", "line", "function", "variable",
                        "expected_type", "actual_type", "module_name"}
        assert set(details.keys()) == expected_keys


# ─── ErrorClassifier.classify_with_context Tests ─────────────────────────────


class TestClassifyWithContext:
    def setup_method(self):
        self.classifier = ErrorClassifier()

    def test_strong_classification_kept(self):
        """A clearly classified error shouldn't be overridden by context."""
        cat = self.classifier.classify_with_context(
            "SyntaxError: invalid syntax",
            [{"tool": "patch", "success": False}],
        )
        assert cat == ErrorCategory.SYNTAX

    def test_unknown_after_patch_becomes_tool_error(self):
        cat = self.classifier.classify_with_context(
            "something failed weirdly",
            [{"tool": "patch", "success": False}],
        )
        assert cat == ErrorCategory.TOOL_ERROR

    def test_unknown_after_read_file_becomes_file_not_found(self):
        cat = self.classifier.classify_with_context(
            "something not there",
            [{"tool": "read_file", "success": False}],
        )
        assert cat == ErrorCategory.FILE_NOT_FOUND

    def test_empty_context_returns_base(self):
        cat = self.classifier.classify_with_context("SyntaxError: oops", [])
        assert cat == ErrorCategory.SYNTAX

    def test_multiple_failures_becomes_tool_error(self):
        cat = self.classifier.classify_with_context(
            "weird thing",
            [
                {"tool": "terminal", "success": False},
                {"tool": "terminal", "success": False},
            ],
        )
        assert cat == ErrorCategory.TOOL_ERROR


# ─── RecoveryStrategy Tests ──────────────────────────────────────────────────


class TestRecoveryStrategy:
    def test_dataclass_creation(self):
        s = RecoveryStrategy(
            strategy_name="test",
            prompt_injection="do something",
            recommended_tools=["read_file"],
            max_retries=3,
            should_reread_file=True,
            should_search_codebase=False,
            confidence=0.8,
        )
        assert s.strategy_name == "test"
        assert s.confidence == 0.8
        assert s.recommended_tools == ["read_file"]


# ─── RecoveryEngine.get_strategy Tests ───────────────────────────────────────


class TestGetStrategy:
    def setup_method(self):
        self.engine = RecoveryEngine()

    def test_attempt_1_returns_base(self):
        s = self.engine.get_strategy(ErrorCategory.SYNTAX, attempt=1)
        assert s.strategy_name == "fix_syntax"
        assert s.confidence == 0.9

    def test_attempt_2_broader(self):
        s = self.engine.get_strategy(ErrorCategory.SYNTAX, attempt=2)
        assert "broader" in s.strategy_name
        assert s.should_reread_file is True
        assert s.should_search_codebase is True
        assert s.confidence < 0.9

    def test_attempt_3_replan(self):
        s = self.engine.get_strategy(ErrorCategory.IMPORT, attempt=3)
        assert "replan" in s.strategy_name
        assert "re-plan" in s.prompt_injection.lower() or "Re-plan" in s.prompt_injection

    def test_attempt_4_alternative(self):
        s = self.engine.get_strategy(ErrorCategory.TYPE_ERROR, attempt=4)
        assert "alternative" in s.strategy_name
        assert "different approach" in s.prompt_injection.lower()

    def test_attempt_5_still_alternative(self):
        s = self.engine.get_strategy(ErrorCategory.RUNTIME, attempt=5)
        assert "alternative" in s.strategy_name

    def test_confidence_decreases_with_attempts(self):
        c1 = self.engine.get_strategy(ErrorCategory.SYNTAX, 1).confidence
        c2 = self.engine.get_strategy(ErrorCategory.SYNTAX, 2).confidence
        c3 = self.engine.get_strategy(ErrorCategory.SYNTAX, 3).confidence
        c4 = self.engine.get_strategy(ErrorCategory.SYNTAX, 4).confidence
        assert c1 > c2 > c3 > c4

    def test_confidence_never_negative(self):
        s = self.engine.get_strategy(ErrorCategory.UNKNOWN, attempt=10)
        assert s.confidence >= 0.0

    def test_all_categories_have_base_strategy(self):
        for cat in ErrorCategory:
            s = self.engine.get_strategy(cat, attempt=1)
            assert isinstance(s, RecoveryStrategy)
            assert s.strategy_name

    def test_import_strategy_recommends_search(self):
        s = self.engine.get_strategy(ErrorCategory.IMPORT, attempt=1)
        assert "search_files" in s.recommended_tools


# ─── RecoveryEngine.generate_recovery_prompt Tests ───────────────────────────


class TestGenerateRecoveryPrompt:
    def setup_method(self):
        self.engine = RecoveryEngine()

    def test_prompt_contains_error_type(self):
        prompt = self.engine.generate_recovery_prompt(
            "SyntaxError: invalid syntax",
            ErrorCategory.SYNTAX,
            attempt=1,
        )
        assert "syntax" in prompt.lower()
        assert "attempt 1" in prompt.lower()

    def test_prompt_includes_extracted_details(self):
        error = 'File "app/main.py", line 42\nSyntaxError: invalid syntax'
        prompt = self.engine.generate_recovery_prompt(
            error, ErrorCategory.SYNTAX, attempt=1,
        )
        assert "app/main.py" in prompt
        assert "42" in prompt

    def test_prompt_includes_context(self):
        prompt = self.engine.generate_recovery_prompt(
            "ImportError: No module named 'foo'",
            ErrorCategory.IMPORT,
            attempt=1,
            context={"project": "myapp", "venv": "/home/user/.venv"},
        )
        assert "myapp" in prompt
        assert "/home/user/.venv" in prompt

    def test_prompt_includes_recommended_tools(self):
        prompt = self.engine.generate_recovery_prompt(
            "FileNotFoundError", ErrorCategory.FILE_NOT_FOUND, attempt=1,
        )
        assert "search_files" in prompt


# ─── is_same_error Tests ─────────────────────────────────────────────────────


class TestIsSameError:
    def test_identical(self):
        assert is_same_error("KeyError: 'foo'", "KeyError: 'foo'") is True

    def test_different_line_numbers(self):
        e1 = "File 'test.py', line 10: SyntaxError"
        e2 = "File 'test.py', line 25: SyntaxError"
        assert is_same_error(e1, e2) is True

    def test_different_timestamps(self):
        e1 = "2024-01-15T10:30:00Z ConnectionError: refused"
        e2 = "2024-01-15T10:31:05Z ConnectionError: refused"
        assert is_same_error(e1, e2) is True

    def test_completely_different(self):
        assert is_same_error("SyntaxError: invalid", "KeyError: 'foo'") is False

    def test_empty_strings(self):
        assert is_same_error("", "") is True

    def test_similar_with_different_pids(self):
        e1 = "pid=12345 process crashed"
        e2 = "pid=67890 process crashed"
        assert is_same_error(e1, e2) is True


# ─── RetryTracker Tests ──────────────────────────────────────────────────────


class TestRetryTracker:
    def setup_method(self):
        self.tracker = RetryTracker()

    def test_record_returns_attempt_number(self):
        assert self.tracker.record_attempt("t1", "error 1") == 1
        assert self.tracker.record_attempt("t1", "error 2") == 2
        assert self.tracker.record_attempt("t1", "error 3") == 3

    def test_attempt_count(self):
        self.tracker.record_attempt("t1", "err")
        self.tracker.record_attempt("t1", "err")
        assert self.tracker.attempt_count("t1") == 2
        assert self.tracker.attempt_count("nonexistent") == 0

    def test_is_repeating_same_error(self):
        self.tracker.record_attempt("t1", "SyntaxError: line 10")
        self.tracker.record_attempt("t1", "SyntaxError: line 15")
        assert self.tracker.is_repeating("t1") is True

    def test_is_not_repeating_different_errors(self):
        self.tracker.record_attempt("t1", "SyntaxError: missing colon")
        self.tracker.record_attempt("t1", "ImportError: no module named foo")
        assert self.tracker.is_repeating("t1") is False

    def test_is_repeating_single_attempt(self):
        self.tracker.record_attempt("t1", "some error")
        assert self.tracker.is_repeating("t1") is False

    def test_is_progressing(self):
        self.tracker.record_attempt("t1", "SyntaxError: bad syntax")
        self.tracker.record_attempt("t1", "ImportError: missing module")
        assert self.tracker.is_progressing("t1") is True

    def test_not_progressing(self):
        self.tracker.record_attempt("t1", "SyntaxError: line 10")
        self.tracker.record_attempt("t1", "SyntaxError: line 20")
        assert self.tracker.is_progressing("t1") is False

    def test_should_change_strategy_after_repeats(self):
        self.tracker.record_attempt("t1", "SyntaxError: invalid")
        self.tracker.record_attempt("t1", "SyntaxError: invalid")
        assert self.tracker.should_change_strategy("t1") is True

    def test_should_not_change_strategy_early(self):
        self.tracker.record_attempt("t1", "error one")
        assert self.tracker.should_change_strategy("t1") is False

    def test_get_history(self):
        self.tracker.record_attempt("t1", "SyntaxError: oops", ErrorCategory.SYNTAX)
        history = self.tracker.get_history("t1")
        assert len(history) == 1
        assert history[0]["category"] == "syntax"
        assert history[0]["attempt"] == 1

    def test_clear_specific_task(self):
        self.tracker.record_attempt("t1", "err")
        self.tracker.record_attempt("t2", "err")
        self.tracker.clear("t1")
        assert self.tracker.attempt_count("t1") == 0
        assert self.tracker.attempt_count("t2") == 1

    def test_clear_all(self):
        self.tracker.record_attempt("t1", "err")
        self.tracker.record_attempt("t2", "err")
        self.tracker.clear()
        assert self.tracker.attempt_count("t1") == 0
        assert self.tracker.attempt_count("t2") == 0

    def test_separate_task_tracking(self):
        self.tracker.record_attempt("t1", "error A")
        self.tracker.record_attempt("t2", "error B")
        assert self.tracker.attempt_count("t1") == 1
        assert self.tracker.attempt_count("t2") == 1

    def test_progressing_on_first_attempt(self):
        self.tracker.record_attempt("t1", "first error")
        assert self.tracker.is_progressing("t1") is True
