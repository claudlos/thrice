"""Tests for test_lint_loop.py - Iterative Test/Lint Fix Loop."""

import json
import os
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new-files"))

from test_lint_loop import (
    FixIteration,
    FixLoop,
    FixLoopResult,
    Framework,
    LintResult,
    ParsedError,
    TestResult,
    TestRunner,
    detect_framework,
    quick_test,
)


# ---------------------------------------------------------------------------
# Framework detection tests
# ---------------------------------------------------------------------------

class TestFrameworkDetection:
    """Test auto_detect_framework."""

    def setup_method(self):
        self.runner = TestRunner()
        self.tmpdir = tempfile.mkdtemp()

    def _create_file(self, name: str, content: str = ""):
        path = os.path.join(self.tmpdir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

    def test_detect_pytest_conftest(self):
        self._create_file("conftest.py")
        assert self.runner.auto_detect_framework(self.tmpdir) == Framework.PYTEST

    def test_detect_pytest_ini(self):
        self._create_file("pytest.ini")
        assert self.runner.auto_detect_framework(self.tmpdir) == Framework.PYTEST

    def test_detect_pytest_pyproject(self):
        self._create_file("pyproject.toml", "[tool.pytest.ini_options]\naddopts = '-v'")
        assert self.runner.auto_detect_framework(self.tmpdir) == Framework.PYTEST

    def test_detect_cargo(self):
        self._create_file("Cargo.toml", "[package]\nname = 'test'")
        assert self.runner.auto_detect_framework(self.tmpdir) == Framework.CARGO

    def test_detect_go(self):
        self._create_file("go.mod", "module example.com/test")
        assert self.runner.auto_detect_framework(self.tmpdir) == Framework.GO

    def test_detect_npm(self):
        self._create_file("package.json", json.dumps({
            "scripts": {"test": "jest"}
        }))
        assert self.runner.auto_detect_framework(self.tmpdir) == Framework.NPM

    def test_detect_npm_no_test_script(self):
        self._create_file("package.json", json.dumps({"scripts": {}}))
        # No test script -> should not detect as NPM
        result = self.runner.auto_detect_framework(self.tmpdir)
        assert result != Framework.NPM

    def test_detect_make(self):
        self._create_file("Makefile", "test:\n\techo ok")
        assert self.runner.auto_detect_framework(self.tmpdir) == Framework.MAKE

    def test_detect_unknown(self):
        assert self.runner.auto_detect_framework(self.tmpdir) == Framework.UNKNOWN

    def test_detect_pytest_from_test_files(self):
        self._create_file("test_something.py", "def test_foo(): pass")
        assert self.runner.auto_detect_framework(self.tmpdir) == Framework.PYTEST

    def test_priority_pytest_over_make(self):
        """pytest indicators take priority over Makefile."""
        self._create_file("conftest.py")
        self._create_file("Makefile", "test:\n\tpytest")
        assert self.runner.auto_detect_framework(self.tmpdir) == Framework.PYTEST


# ---------------------------------------------------------------------------
# ParsedError tests
# ---------------------------------------------------------------------------

class TestParsedError:
    """Test ParsedError formatting."""

    def test_str_full(self):
        err = ParsedError(
            file="src/main.py", line=10, column=5,
            message="undefined name", severity="error", rule="F821"
        )
        s = str(err)
        assert "src/main.py:10:5" in s
        assert "[error]" in s
        assert "undefined name" in s
        assert "(F821)" in s

    def test_str_minimal(self):
        err = ParsedError(message="something failed")
        s = str(err)
        assert "something failed" in s

    def test_str_no_column(self):
        err = ParsedError(file="test.py", line=5, message="fail")
        s = str(err)
        assert "test.py:5" in s
        assert ":5:" not in s or "test.py:5" in s


# ---------------------------------------------------------------------------
# Error parsing tests
# ---------------------------------------------------------------------------

class TestErrorParsing:
    """Test error parsing from various framework outputs."""

    def setup_method(self):
        self.runner = TestRunner()

    def test_parse_pytest_errors(self):
        output = """FAILED tests/test_foo.py::test_bar - AssertionError: expected 1 got 2
FAILED tests/test_baz.py::test_qux:15 - ValueError: invalid"""
        errors = self.runner.parse_errors(output, Framework.PYTEST)
        assert len(errors) >= 1
        assert any("test_foo.py" in (e.file or "") for e in errors)

    def test_parse_go_errors(self):
        output = """main.go:10:5: undefined: foo
utils.go:25:1: syntax error: unexpected end"""
        errors = self.runner.parse_errors(output, Framework.GO)
        assert len(errors) == 2
        assert errors[0].file == "main.go"
        assert errors[0].line == 10
        assert errors[0].column == 5
        assert "undefined" in errors[0].message

    def test_parse_cargo_errors(self):
        output = """error[E0308]: mismatched types
  --> src/main.rs:10:5
warning[W0001]: unused variable
  --> src/lib.rs:20:9"""
        errors = self.runner.parse_errors(output, Framework.CARGO)
        assert len(errors) == 2
        assert errors[0].file == "src/main.rs"
        assert errors[0].severity == "error"
        assert errors[1].severity == "warning"

    def test_parse_generic_errors(self):
        output = """file.py:10:5: error: something wrong
other.py:20: warning: maybe bad"""
        errors = self.runner.parse_errors(output, Framework.UNKNOWN)
        assert len(errors) == 2
        assert errors[0].severity == "error"
        assert errors[1].severity == "warning"

    def test_parse_empty_output(self):
        errors = self.runner.parse_errors("", Framework.PYTEST)
        assert errors == []


# ---------------------------------------------------------------------------
# TestResult tests
# ---------------------------------------------------------------------------

class TestTestResult:
    """Test TestResult properties."""

    def test_success_when_all_pass(self):
        r = TestResult(passed=5, failed=0, errors=0, return_code=0)
        assert r.success is True

    def test_failure_when_failed(self):
        r = TestResult(passed=3, failed=2, errors=0, return_code=1)
        assert r.success is False

    def test_failure_when_errors(self):
        r = TestResult(passed=3, failed=0, errors=1, return_code=1)
        assert r.success is False

    def test_failure_on_return_code(self):
        r = TestResult(passed=5, failed=0, errors=0, return_code=1)
        assert r.success is False


# ---------------------------------------------------------------------------
# TestRunner run_tests tests (mocked)
# ---------------------------------------------------------------------------

class TestRunTests:
    """Test run_tests with mocked subprocess."""

    def setup_method(self):
        self.runner = TestRunner()
        self.tmpdir = tempfile.mkdtemp()

    @patch.object(TestRunner, "_run_command")
    def test_run_tests_success(self, mock_cmd):
        mock_cmd.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="5 passed in 1.2s\n", stderr=""
        )
        result = self.runner.run_tests(self.tmpdir, Framework.PYTEST)
        assert result.return_code == 0
        assert result.passed == 5
        assert result.framework == Framework.PYTEST

    @patch.object(TestRunner, "_run_command")
    def test_run_tests_failure(self, mock_cmd):
        mock_cmd.return_value = subprocess.CompletedProcess(
            args=[], returncode=1,
            stdout="FAILED tests/test_a.py::test_x - AssertionError: nope\n2 failed, 3 passed\n",
            stderr=""
        )
        result = self.runner.run_tests(self.tmpdir, Framework.PYTEST)
        assert result.return_code == 1
        assert result.failed == 2
        assert result.passed == 3
        assert not result.success

    def test_run_tests_unknown_framework(self):
        result = self.runner.run_tests(self.tmpdir, Framework.UNKNOWN)
        assert result.return_code == -1
        assert "Unknown framework" in result.output


# ---------------------------------------------------------------------------
# FixLoop tests
# ---------------------------------------------------------------------------

class TestFixLoop:
    """Test the iterative fix loop."""

    def test_run_and_collect(self):
        runner = MagicMock(spec=TestRunner)
        runner.run_tests.return_value = TestResult(
            passed=3, failed=1, errors=0, return_code=1,
            output="1 failed, 3 passed",
            parsed_errors=[ParsedError(file="t.py", line=5, message="fail")],
        )
        runner.run_lint.return_value = LintResult(
            errors=0, warnings=0, output="", return_code=0,
        )

        loop = FixLoop(runner=runner, run_lint=True)
        iteration = loop.run_and_collect("/tmp/proj")
        assert iteration.errors_found == 1
        assert iteration.test_result.failed == 1

    def test_run_until_green_immediate_success(self):
        runner = MagicMock(spec=TestRunner)
        runner.auto_detect_framework.return_value = Framework.PYTEST
        runner.run_tests.return_value = TestResult(
            passed=5, failed=0, errors=0, return_code=0,
        )
        runner.run_lint.return_value = LintResult(
            errors=0, warnings=0, output="", return_code=0,
        )

        loop = FixLoop(runner=runner)
        result = loop.run_until_green("/tmp/proj")
        assert result.final_success is True
        assert result.total_iterations == 1

    def test_run_until_green_with_fix_callback(self):
        runner = MagicMock(spec=TestRunner)
        runner.auto_detect_framework.return_value = Framework.PYTEST
        call_count = [0]

        def run_tests_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 3:
                return TestResult(passed=5, failed=0, errors=0, return_code=0)
            return TestResult(
                passed=3, failed=1, errors=0, return_code=1,
                parsed_errors=[ParsedError(message="fail")],
            )

        runner.run_tests.side_effect = run_tests_side_effect
        runner.run_lint.return_value = LintResult(errors=0, warnings=0, output="", return_code=0)

        def fix_callback(iteration):
            return True  # always say we fixed something

        loop = FixLoop(runner=runner, run_lint=True)
        result = loop.run_until_green("/tmp/proj", fix_callback=fix_callback)
        assert result.final_success is True
        assert result.total_iterations == 3

    def test_run_until_green_max_iterations(self):
        runner = MagicMock(spec=TestRunner)
        runner.auto_detect_framework.return_value = Framework.PYTEST
        runner.run_tests.return_value = TestResult(
            passed=0, failed=5, errors=0, return_code=1,
            parsed_errors=[ParsedError(message="fail")],
        )
        runner.run_lint.return_value = LintResult(errors=0, warnings=0, output="", return_code=0)

        loop = FixLoop(runner=runner, max_iterations=3)
        result = loop.run_until_green(
            "/tmp/proj",
            fix_callback=lambda it: True,
            max_iterations=3,
        )
        assert result.final_success is False
        assert result.total_iterations == 3

    def test_run_until_green_callback_aborts(self):
        runner = MagicMock(spec=TestRunner)
        runner.auto_detect_framework.return_value = Framework.PYTEST
        runner.run_tests.return_value = TestResult(
            passed=0, failed=1, errors=0, return_code=1,
            parsed_errors=[ParsedError(message="fail")],
        )
        runner.run_lint.return_value = LintResult(errors=0, return_code=0, output="")

        loop = FixLoop(runner=runner)
        result = loop.run_until_green(
            "/tmp/proj",
            fix_callback=lambda it: False,  # abort immediately
        )
        assert result.total_iterations == 1
        assert not result.final_success

    def test_no_callback_single_iteration(self):
        runner = MagicMock(spec=TestRunner)
        runner.auto_detect_framework.return_value = Framework.PYTEST
        runner.run_tests.return_value = TestResult(
            passed=0, failed=1, errors=0, return_code=1,
            parsed_errors=[ParsedError(message="fail")],
        )
        runner.run_lint.return_value = LintResult(errors=0, return_code=0, output="")

        loop = FixLoop(runner=runner)
        result = loop.run_until_green("/tmp/proj")  # no callback
        assert result.total_iterations == 1


# ---------------------------------------------------------------------------
# format_errors_for_model tests
# ---------------------------------------------------------------------------

class TestFormatErrors:
    """Test error formatting for the model."""

    def test_format_test_failures(self):
        loop = FixLoop()
        iteration = FixIteration(
            iteration=1,
            test_result=TestResult(
                passed=3, failed=2, errors=0, return_code=1,
                framework=Framework.PYTEST,
                parsed_errors=[
                    ParsedError(file="test.py", line=10, message="assert 1 == 2"),
                ],
            ),
        )
        formatted = loop.format_errors_for_model(iteration)
        assert "Test Failures" in formatted
        assert "test.py" in formatted
        assert "assert 1 == 2" in formatted

    def test_format_lint_errors(self):
        loop = FixLoop()
        iteration = FixIteration(
            iteration=2,
            test_result=TestResult(return_code=0),
            lint_result=LintResult(
                errors=1, warnings=1, return_code=1,
                parsed_errors=[
                    ParsedError(file="main.py", line=5, message="unused import", severity="warning"),
                    ParsedError(file="main.py", line=10, message="undefined name", severity="error"),
                ],
            ),
        )
        formatted = loop.format_errors_for_model(iteration)
        assert "Lint Errors" in formatted
        assert "unused import" in formatted

    def test_format_raw_output_fallback(self):
        loop = FixLoop()
        iteration = FixIteration(
            iteration=1,
            test_result=TestResult(
                passed=0, failed=1, errors=0, return_code=1,
                framework=Framework.PYTEST,
                output="some raw test output here\n",
                parsed_errors=[],  # no parsed errors
            ),
        )
        formatted = loop.format_errors_for_model(iteration)
        assert "Raw output" in formatted
        assert "raw test output" in formatted


# ---------------------------------------------------------------------------
# FixLoopResult tests
# ---------------------------------------------------------------------------

class TestFixLoopResult:
    """Test FixLoopResult properties."""

    def test_errors_fixed_calculation(self):
        r = FixLoopResult(initial_errors=5, remaining_errors=2)
        assert r.errors_fixed == 3

    def test_errors_fixed_no_negative(self):
        r = FixLoopResult(initial_errors=2, remaining_errors=5)
        assert r.errors_fixed == 0


# ---------------------------------------------------------------------------
# Convenience function tests
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_detect_framework(self):
        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "conftest.py"), "w") as f:
            f.write("")
        assert detect_framework(tmpdir) == Framework.PYTEST
