"""Comprehensive tests for test_fix_loop.py — the enhanced iterative test-fix loop."""

import os
import sys
import tempfile
import textwrap
from unittest.mock import MagicMock, patch

import pytest

# Ensure the module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new-files"))

from test_fix_loop import (
    FailureParser,
    LintIssue,
    LintOutputParser,
    LintRunResult,
    LoopResult,
    TestDetector,
    TestFailure,
    TestFixConfig,
    TestFixLoop,
    TestRunResult,
)


# ===========================================================================
# TestFixConfig tests
# ===========================================================================

class TestTestFixConfig:
    def test_defaults(self):
        cfg = TestFixConfig()
        assert cfg.max_iterations == 5
        assert cfg.test_command == ""
        assert cfg.lint_command == ""
        assert cfg.auto_detect is True
        assert cfg.run_lint_first is True
        assert cfg.fail_fast is True
        assert cfg.timeout == 120

    def test_custom_values(self):
        cfg = TestFixConfig(
            max_iterations=10,
            test_command="pytest -x",
            lint_command="ruff check .",
            auto_detect=False,
            run_lint_first=False,
            fail_fast=False,
            timeout=60,
        )
        assert cfg.max_iterations == 10
        assert cfg.test_command == "pytest -x"
        assert cfg.lint_command == "ruff check ."
        assert cfg.auto_detect is False


# ===========================================================================
# Data class tests
# ===========================================================================

class TestDataClasses:
    def test_test_failure_summary(self):
        f = TestFailure(
            test_name="test_foo",
            file_path="tests/test_bar.py",
            line_number=42,
            error_message="expected 1 got 2",
            error_type="AssertionError",
        )
        s = f.summary()
        assert "tests/test_bar.py:42" in s
        assert "test_foo" in s
        assert "AssertionError" in s
        assert "expected 1 got 2" in s

    def test_test_failure_summary_no_line(self):
        f = TestFailure(file_path="foo.py", error_message="boom")
        s = f.summary()
        assert "foo.py" in s
        assert ":None" not in s  # line_number is None, shouldn't show

    def test_test_run_result_success(self):
        r = TestRunResult(passed=5, failed=0, errors=0, return_code=0)
        assert r.success is True
        assert r.total == 5

    def test_test_run_result_failure(self):
        r = TestRunResult(passed=3, failed=2, errors=0, return_code=1)
        assert r.success is False
        assert r.total == 5

    def test_test_run_result_errors(self):
        r = TestRunResult(passed=3, failed=0, errors=1, return_code=1)
        assert r.success is False

    def test_lint_issue_summary(self):
        issue = LintIssue(
            file_path="foo.py", line=10, column=5,
            code="E501", message="Line too long",
        )
        s = issue.summary()
        assert "foo.py:10:5" in s
        assert "E501" in s
        assert "Line too long" in s

    def test_lint_run_result_clean(self):
        r = LintRunResult(issues=[], clean=True)
        assert r.error_count == 0
        assert r.warning_count == 0

    def test_lint_run_result_with_issues(self):
        r = LintRunResult(
            issues=[
                LintIssue(severity="error"),
                LintIssue(severity="error"),
                LintIssue(severity="warning"),
            ],
            clean=False,
        )
        assert r.error_count == 2
        assert r.warning_count == 1

    def test_loop_result_defaults(self):
        r = LoopResult()
        assert r.success is False
        assert r.iterations == 0
        assert r.final_test_result is None
        assert r.history == []
        assert r.fix_prompts == []


# ===========================================================================
# TestDetector tests
# ===========================================================================

class TestTestDetector:
    def setup_method(self):
        self.detector = TestDetector()
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_detect_pytest_conftest(self):
        open(os.path.join(self.tmpdir, "conftest.py"), "w").close()
        info = self.detector.detect_test_framework(self.tmpdir)
        assert info["framework"] == "pytest"
        assert "pytest" in info["command"]
        assert info["config_file"] == "conftest.py"

    def test_detect_pytest_ini(self):
        open(os.path.join(self.tmpdir, "pytest.ini"), "w").close()
        info = self.detector.detect_test_framework(self.tmpdir)
        assert info["framework"] == "pytest"

    def test_detect_pytest_pyproject(self):
        with open(os.path.join(self.tmpdir, "pyproject.toml"), "w") as f:
            f.write("[tool.pytest.ini_options]\n")
        info = self.detector.detect_test_framework(self.tmpdir)
        assert info["framework"] == "pytest"

    def test_detect_jest(self):
        import json
        pkg = {"devDependencies": {"jest": "^29.0.0"}, "scripts": {"test": "jest"}}
        with open(os.path.join(self.tmpdir, "package.json"), "w") as f:
            json.dump(pkg, f)
        info = self.detector.detect_test_framework(self.tmpdir)
        assert info["framework"] == "jest"
        assert "jest" in info["command"]

    def test_detect_mocha(self):
        import json
        pkg = {"devDependencies": {"mocha": "^10.0.0"}, "scripts": {"test": "mocha"}}
        with open(os.path.join(self.tmpdir, "package.json"), "w") as f:
            json.dump(pkg, f)
        info = self.detector.detect_test_framework(self.tmpdir)
        assert info["framework"] == "mocha"

    def test_detect_cargo(self):
        with open(os.path.join(self.tmpdir, "Cargo.toml"), "w") as f:
            f.write("[package]\nname = \"test\"\n")
        info = self.detector.detect_test_framework(self.tmpdir)
        assert info["framework"] == "cargo_test"
        assert "cargo test" in info["command"]

    def test_detect_go(self):
        with open(os.path.join(self.tmpdir, "go.mod"), "w") as f:
            f.write("module example.com/test\n")
        info = self.detector.detect_test_framework(self.tmpdir)
        assert info["framework"] == "go_test"
        assert "go test" in info["command"]

    def test_detect_unittest_fallback(self):
        open(os.path.join(self.tmpdir, "test_something.py"), "w").close()
        info = self.detector.detect_test_framework(self.tmpdir)
        assert info["framework"] == "unittest"

    def test_detect_unittest_tests_dir(self):
        tests_dir = os.path.join(self.tmpdir, "tests")
        os.makedirs(tests_dir)
        open(os.path.join(tests_dir, "test_foo.py"), "w").close()
        info = self.detector.detect_test_framework(self.tmpdir)
        assert info["framework"] == "unittest"

    def test_detect_unknown(self):
        info = self.detector.detect_test_framework(self.tmpdir)
        assert info["framework"] == "unknown"
        assert info["command"] == ""

    def test_detect_relevant_tests_python(self):
        # Create test file
        tests_dir = os.path.join(self.tmpdir, "tests")
        os.makedirs(tests_dir)
        open(os.path.join(tests_dir, "test_utils.py"), "w").close()

        result = self.detector.detect_relevant_tests(
            ["src/utils.py"], self.tmpdir
        )
        assert any("test_utils.py" in f for f in result)

    def test_detect_relevant_tests_same_dir(self):
        open(os.path.join(self.tmpdir, "test_foo.py"), "w").close()
        result = self.detector.detect_relevant_tests(
            [os.path.join(self.tmpdir, "foo.py")], self.tmpdir
        )
        assert any("test_foo.py" in f for f in result)

    def test_detect_relevant_tests_js(self):
        src_dir = os.path.join(self.tmpdir, "src")
        tests_dir = os.path.join(src_dir, "__tests__")
        os.makedirs(tests_dir)
        open(os.path.join(tests_dir, "utils.test.js"), "w").close()

        result = self.detector.detect_relevant_tests(
            [os.path.join(src_dir, "utils.js")], self.tmpdir
        )
        assert any("utils.test.js" in f for f in result)

    def test_detect_relevant_tests_go(self):
        open(os.path.join(self.tmpdir, "utils_test.go"), "w").close()
        result = self.detector.detect_relevant_tests(
            [os.path.join(self.tmpdir, "utils.go")], self.tmpdir
        )
        assert any("utils_test.go" in f for f in result)

    def test_detect_relevant_tests_dedup(self):
        open(os.path.join(self.tmpdir, "test_foo.py"), "w").close()
        result = self.detector.detect_relevant_tests(
            [os.path.join(self.tmpdir, "foo.py"),
             os.path.join(self.tmpdir, "foo.py")],
            self.tmpdir,
        )
        # Should not have duplicates
        assert len(result) == len(set(result))

    def test_detect_lint_ruff_toml(self):
        open(os.path.join(self.tmpdir, "ruff.toml"), "w").close()
        info = self.detector.detect_lint_tool(self.tmpdir)
        assert info["tool"] == "ruff"

    def test_detect_lint_flake8(self):
        open(os.path.join(self.tmpdir, ".flake8"), "w").close()
        info = self.detector.detect_lint_tool(self.tmpdir)
        assert info["tool"] == "flake8"

    def test_detect_lint_pylintrc(self):
        open(os.path.join(self.tmpdir, ".pylintrc"), "w").close()
        info = self.detector.detect_lint_tool(self.tmpdir)
        assert info["tool"] == "pylint"

    def test_detect_lint_eslint(self):
        open(os.path.join(self.tmpdir, ".eslintrc.json"), "w").close()
        info = self.detector.detect_lint_tool(self.tmpdir)
        assert info["tool"] == "eslint"

    def test_detect_lint_cargo(self):
        with open(os.path.join(self.tmpdir, "Cargo.toml"), "w") as f:
            f.write("[package]\n")
        info = self.detector.detect_lint_tool(self.tmpdir)
        assert info["tool"] == "rustfmt"

    def test_detect_lint_ruff_in_pyproject(self):
        with open(os.path.join(self.tmpdir, "pyproject.toml"), "w") as f:
            f.write("[tool.ruff]\nline-length = 88\n")
        info = self.detector.detect_lint_tool(self.tmpdir)
        assert info["tool"] == "ruff"

    def test_is_test_file(self):
        assert TestDetector._is_test_file("test_foo.py") is True
        assert TestDetector._is_test_file("foo_test.py") is True
        assert TestDetector._is_test_file("foo.test.js") is True
        assert TestDetector._is_test_file("foo.spec.ts") is True
        assert TestDetector._is_test_file("utils_test.go") is True
        assert TestDetector._is_test_file("foo.py") is False
        assert TestDetector._is_test_file("utils.go") is False


# ===========================================================================
# FailureParser tests
# ===========================================================================

class TestFailureParser:
    def setup_method(self):
        self.parser = FailureParser()

    def test_parse_pytest_failed_line(self):
        output = textwrap.dedent("""\
            FAILED tests/test_foo.py::test_addition - AssertionError: assert 1 == 2
            FAILED tests/test_bar.py::test_string - ValueError: invalid literal
        """)
        failures = self.parser.parse_pytest(output)
        assert len(failures) == 2
        assert failures[0].file_path == "tests/test_foo.py"
        assert failures[0].test_name == "test_addition"
        assert failures[0].error_type == "AssertionError"
        assert "1 == 2" in failures[0].error_message

    def test_parse_pytest_location(self):
        output = "tests/test_foo.py:42: AssertionError: bad value\n"
        failures = self.parser.parse_pytest(output)
        assert len(failures) >= 1
        found = [f for f in failures if f.line_number == 42]
        assert len(found) == 1
        assert found[0].error_type == "AssertionError"

    def test_parse_pytest_empty(self):
        output = "===== 5 passed in 0.5s =====\n"
        failures = self.parser.parse_pytest(output)
        assert len(failures) == 0

    def test_parse_unittest(self):
        output = textwrap.dedent("""\
            ======================================================================
            FAIL: test_add (tests.test_math.TestMath)
            ----------------------------------------------------------------------
            Traceback (most recent call last):
              File "tests/test_math.py", line 15, in test_add
                self.assertEqual(add(1, 2), 4)
            AssertionError: 3 != 4

            ======================================================================
        """)
        failures = self.parser.parse_unittest(output)
        assert len(failures) == 1
        assert failures[0].test_name == "tests.test_math.TestMath.test_add"
        assert failures[0].file_path == "tests/test_math.py"
        assert failures[0].line_number == 15
        assert failures[0].error_type == "AssertionError"
        assert "3 != 4" in failures[0].error_message

    def test_parse_unittest_error(self):
        output = textwrap.dedent("""\
            ======================================================================
            ERROR: test_import (tests.test_mod.TestMod)
            ----------------------------------------------------------------------
            Traceback (most recent call last):
              File "tests/test_mod.py", line 5, in test_import
                import nonexistent
            ModuleNotFoundError: No module named 'nonexistent'

            ======================================================================
        """)
        failures = self.parser.parse_unittest(output)
        assert len(failures) == 1
        assert "ModuleNotFoundError" in failures[0].error_type

    def test_parse_jest(self):
        output = textwrap.dedent("""\
            FAIL src/utils.test.js
              ● add function > should add two numbers

                expect(received).toBe(expected)

                Expected: 4
                Received: 3

                  at Object.<anonymous> (src/utils.test.js:10:5)
        """)
        failures = self.parser.parse_jest(output)
        assert len(failures) >= 1
        assert failures[0].test_name == "add function > should add two numbers"

    def test_parse_cargo_test(self):
        output = textwrap.dedent("""\
            running 2 tests
            test tests::test_add ... ok
            test tests::test_sub ... FAILED

            failures:

            ---- tests::test_sub stdout ----
            thread 'tests::test_sub' panicked at 'assertion failed: 5 == 3', src/lib.rs:10:5

            failures:
                tests::test_sub

            test result: FAILED. 1 passed; 1 failed; 0 ignored
        """)
        failures = self.parser.parse_cargo_test(output)
        assert len(failures) == 1
        assert failures[0].test_name == "tests::test_sub"
        assert "assertion failed" in failures[0].error_message or failures[0].error_type == "panic"

    def test_parse_go_test(self):
        output = textwrap.dedent("""\
            === RUN   TestAdd
            --- PASS: TestAdd (0.00s)
            === RUN   TestSub
                utils_test.go:25: expected 5 but got 3
            --- FAIL: TestSub (0.00s)
            FAIL
        """)
        failures = self.parser.parse_go_test(output)
        assert len(failures) == 1
        assert failures[0].test_name == "TestSub"
        assert failures[0].file_path == "utils_test.go"
        assert failures[0].line_number == 25

    def test_parse_dispatch(self):
        output = "FAILED tests/test_x.py::test_y - AssertionError: nope\n"
        failures = self.parser.parse(output, "pytest")
        assert len(failures) >= 1

    def test_parse_unknown_framework(self):
        output = "foo.py:10:5: error: something broke\n"
        failures = self.parser.parse(output, "unknown_framework")
        assert len(failures) >= 1

    def test_parse_generic(self):
        output = "src/main.py:42:10: error: undefined name 'foo'\n"
        failures = self.parser.parse_generic(output)
        assert len(failures) == 1
        assert failures[0].file_path == "src/main.py"
        assert failures[0].line_number == 42


# ===========================================================================
# LintOutputParser tests
# ===========================================================================

class TestLintOutputParser:
    def setup_method(self):
        self.parser = LintOutputParser()

    def test_parse_ruff(self):
        output = "foo.py:10:5: E501 Line too long (120 > 88 characters)\n"
        issues = self.parser.parse(output, "ruff")
        assert len(issues) == 1
        assert issues[0].code == "E501"
        assert issues[0].line == 10
        assert issues[0].column == 5

    def test_parse_flake8(self):
        output = "bar.py:3:1: F401 'os' imported but unused\n"
        issues = self.parser.parse(output, "flake8")
        assert len(issues) == 1
        assert issues[0].code == "F401"

    def test_parse_pylint(self):
        output = "foo.py:1:0: C0114: Missing module docstring (missing-module-docstring)\n"
        issues = self.parser.parse(output, "pylint")
        assert len(issues) == 1
        assert issues[0].code == "C0114"
        assert issues[0].severity == "warning"  # C-codes are convention

    def test_parse_pylint_error(self):
        output = "foo.py:5:0: E0001: Syntax error in module (syntax-error)\n"
        issues = self.parser.parse(output, "pylint")
        assert len(issues) == 1
        assert issues[0].severity == "error"

    def test_parse_multiple_issues(self):
        output = textwrap.dedent("""\
            foo.py:1:1: E501 line too long
            foo.py:2:1: W291 trailing whitespace
            bar.py:5:10: F821 undefined name
        """)
        issues = self.parser.parse(output, "ruff")
        assert len(issues) == 3

    def test_parse_empty_output(self):
        issues = self.parser.parse("", "ruff")
        assert len(issues) == 0

    def test_parse_rustfmt(self):
        output = textwrap.dedent("""\
            warning[clippy::unused_variable]: unused variable: `x`
             --> src/main.rs:5:9
        """)
        issues = self.parser.parse(output, "rustfmt")
        assert len(issues) == 1
        assert issues[0].file_path == "src/main.rs"
        assert issues[0].severity == "warning"


# ===========================================================================
# TestFixLoop tests
# ===========================================================================

class TestTestFixLoop:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create a conftest so pytest is detected
        open(os.path.join(self.tmpdir, "conftest.py"), "w").close()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_init_auto_detect(self):
        cfg = TestFixConfig(auto_detect=True)
        loop = TestFixLoop(cfg, self.tmpdir)
        assert "pytest" in loop.config.test_command

    def test_init_no_auto_detect(self):
        cfg = TestFixConfig(auto_detect=False, test_command="make test")
        loop = TestFixLoop(cfg, self.tmpdir)
        assert loop.config.test_command == "make test"

    @patch("test_fix_loop.subprocess.run")
    def test_run_tests_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="===== 5 passed in 0.3s =====\n",
            stderr="",
        )
        cfg = TestFixConfig(auto_detect=False, test_command="pytest")
        loop = TestFixLoop(cfg, self.tmpdir)
        result = loop.run_tests()
        assert result.return_code == 0
        assert result.passed == 5

    @patch("test_fix_loop.subprocess.run")
    def test_run_tests_failure(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="FAILED tests/test_foo.py::test_bar - AssertionError: nope\n1 failed, 2 passed\n",
            stderr="",
        )
        cfg = TestFixConfig(auto_detect=False, test_command="pytest")
        loop = TestFixLoop(cfg, self.tmpdir)
        result = loop.run_tests()
        assert result.success is False
        assert result.failed >= 1

    @patch("test_fix_loop.subprocess.run")
    def test_run_tests_timeout(self, mock_run):
        import subprocess as sp
        mock_run.side_effect = sp.TimeoutExpired(cmd="pytest", timeout=120)
        cfg = TestFixConfig(auto_detect=False, test_command="pytest")
        loop = TestFixLoop(cfg, self.tmpdir)
        result = loop.run_tests()
        assert result.success is False
        assert "timed out" in result.output.lower()

    def test_run_tests_no_command(self):
        cfg = TestFixConfig(auto_detect=False, test_command="")
        loop = TestFixLoop(cfg, self.tmpdir)
        result = loop.run_tests()
        assert result.return_code == -1

    @patch("test_fix_loop.subprocess.run")
    def test_run_lint_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="", stderr="",
        )
        cfg = TestFixConfig(auto_detect=False, lint_command="ruff check .")
        loop = TestFixLoop(cfg, self.tmpdir)
        result = loop.run_lint()
        assert result.clean is True

    @patch("test_fix_loop.subprocess.run")
    def test_run_lint_issues(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="foo.py:1:1: E501 Line too long\n",
            stderr="",
        )
        cfg = TestFixConfig(auto_detect=False, lint_command="ruff check .")
        loop = TestFixLoop(cfg, self.tmpdir)
        loop._lint_tool = "ruff"
        result = loop.run_lint()
        assert result.clean is False

    def test_run_lint_no_command(self):
        cfg = TestFixConfig(auto_detect=False, lint_command="")
        loop = TestFixLoop(cfg, self.tmpdir)
        result = loop.run_lint()
        assert result.clean is True  # no lint = clean

    def test_generate_fix_prompt_first_attempt(self):
        cfg = TestFixConfig(auto_detect=False, test_command="pytest")
        loop = TestFixLoop(cfg, self.tmpdir)
        failures = [
            TestFailure(
                test_name="test_add",
                file_path="test_math.py",
                line_number=10,
                error_message="assert 1 == 2",
                error_type="AssertionError",
            )
        ]
        prompt = loop.generate_fix_prompt(failures, attempt=1)
        assert "Tests are failing" in prompt
        assert "test_add" in prompt or "test_math.py" in prompt
        assert "assert 1 == 2" in prompt

    def test_generate_fix_prompt_later_attempt(self):
        cfg = TestFixConfig(auto_detect=False, test_command="pytest")
        loop = TestFixLoop(cfg, self.tmpdir)
        failures = [TestFailure(test_name="test_x", error_message="boom")]
        prompt = loop.generate_fix_prompt(failures, attempt=3)
        assert "attempt 3" in prompt

    def test_generate_fix_prompt_with_lint(self):
        cfg = TestFixConfig(auto_detect=False, test_command="pytest")
        loop = TestFixLoop(cfg, self.tmpdir)
        failures = [TestFailure(test_name="test_x", error_message="fail")]
        lint_issues = [
            LintIssue(file_path="foo.py", line=1, column=1, code="E501",
                       message="Line too long", severity="error"),
        ]
        prompt = loop.generate_fix_prompt(failures, 1, lint_issues)
        assert "Lint Errors" in prompt
        assert "E501" in prompt

    def test_generate_fix_prompt_no_failures(self):
        cfg = TestFixConfig(auto_detect=False, test_command="pytest")
        loop = TestFixLoop(cfg, self.tmpdir)
        prompt = loop.generate_fix_prompt([], attempt=1)
        assert "Fix the code" in prompt

    def test_parse_failures_delegates(self):
        cfg = TestFixConfig(auto_detect=False, test_command="pytest")
        loop = TestFixLoop(cfg, self.tmpdir)
        output = "FAILED tests/test_x.py::test_y - ValueError: bad\n"
        failures = loop.parse_failures(output, "pytest")
        assert len(failures) >= 1

    @patch("test_fix_loop.subprocess.run")
    def test_iterate_all_green(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="===== 3 passed in 0.1s =====\n",
            stderr="",
        )
        cfg = TestFixConfig(
            auto_detect=False,
            test_command="pytest",
            lint_command="",
            run_lint_first=False,
        )
        loop = TestFixLoop(cfg, self.tmpdir)
        result = loop.iterate(changed_files=["foo.py"])
        assert result.success is True
        assert result.iterations == 1
        assert len(result.fix_prompts) == 0

    @patch("test_fix_loop.subprocess.run")
    def test_iterate_max_iterations(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="FAILED tests/test_foo.py::test_bar - AssertionError: nope\n1 failed\n",
            stderr="",
        )
        cfg = TestFixConfig(
            max_iterations=3,
            auto_detect=False,
            test_command="pytest",
            lint_command="",
            run_lint_first=False,
            fail_fast=False,
        )
        loop = TestFixLoop(cfg, self.tmpdir)
        result = loop.iterate(changed_files=["foo.py"])
        assert result.success is False
        assert result.iterations == 3
        assert len(result.history) == 3
        assert len(result.fix_prompts) == 3

    @patch("test_fix_loop.subprocess.run")
    def test_iterate_with_lint(self, mock_run):
        # First call: lint, second call: tests
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="===== 2 passed in 0.1s =====\n",
            stderr="",
        )
        cfg = TestFixConfig(
            auto_detect=False,
            test_command="pytest",
            lint_command="ruff check .",
            run_lint_first=True,
        )
        loop = TestFixLoop(cfg, self.tmpdir)
        result = loop.iterate(changed_files=["foo.py"])
        assert result.success is True
        assert len(result.lint_results) >= 1

    def test_parse_counts_pytest(self):
        cfg = TestFixConfig(auto_detect=False, test_command="pytest")
        loop = TestFixLoop(cfg, self.tmpdir)
        p, f, e = loop._parse_counts("3 failed, 10 passed, 1 error in 2.5s")
        assert p == 10
        assert f == 3
        assert e == 1

    def test_parse_counts_unittest(self):
        cfg = TestFixConfig(auto_detect=False, test_command="pytest")
        loop = TestFixLoop(cfg, self.tmpdir)
        p, f, e = loop._parse_counts(
            "Ran 10 tests in 0.5s\nFAILED (failures=2, errors=1)"
        )
        assert p == 7
        assert f == 2
        assert e == 1

    def test_parse_counts_go(self):
        cfg = TestFixConfig(auto_detect=False, test_command="pytest")
        loop = TestFixLoop(cfg, self.tmpdir)
        output = "--- PASS: TestA\n--- PASS: TestB\n--- FAIL: TestC\n"
        p, f, e = loop._parse_counts(output)
        assert p == 2
        assert f == 1


# ===========================================================================
# Edge case / integration-style tests
# ===========================================================================

class TestEdgeCases:
    def test_empty_changed_files(self):
        tmpdir = tempfile.mkdtemp()
        try:
            detector = TestDetector()
            result = detector.detect_relevant_tests([], tmpdir)
            assert result == []
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_test_failure_no_fields(self):
        f = TestFailure()
        s = f.summary()
        assert isinstance(s, str)

    def test_multiple_frameworks_priority(self):
        """pytest should take priority over unittest when conftest exists."""
        tmpdir = tempfile.mkdtemp()
        try:
            open(os.path.join(tmpdir, "conftest.py"), "w").close()
            open(os.path.join(tmpdir, "test_foo.py"), "w").close()
            detector = TestDetector()
            info = detector.detect_test_framework(tmpdir)
            assert info["framework"] == "pytest"
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_rust_test_file_is_source(self):
        """For Rust, the source file itself contains tests."""
        tmpdir = tempfile.mkdtemp()
        try:
            open(os.path.join(tmpdir, "main.rs"), "w").close()
            detector = TestDetector()
            result = detector.detect_relevant_tests(
                [os.path.join(tmpdir, "main.rs")], tmpdir
            )
            assert any("main.rs" in f for f in result)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_parser_handles_garbage_input(self):
        parser = FailureParser()
        failures = parser.parse_pytest("random garbage with no structure")
        assert isinstance(failures, list)

    def test_lint_parser_handles_empty(self):
        parser = LintOutputParser()
        issues = parser.parse("", "ruff")
        assert issues == []

    def test_loop_result_tracks_history(self):
        r = LoopResult()
        r.history.append(TestRunResult(passed=1, failed=2))
        r.history.append(TestRunResult(passed=2, failed=1))
        assert len(r.history) == 2
        assert r.history[0].failed == 2
        assert r.history[1].failed == 1
