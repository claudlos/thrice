"""Iterative Test/Lint Fix Loop for Hermes Agent.

Provides automatic framework detection, test/lint running, error parsing,
and an iterative fix loop that runs tests, parses errors, and feeds them
back to the model for fixing until all tests pass or max iterations reached.

Integration point: called by the agent loop when the model decides to
run tests, or automatically after code edits.
"""

import os
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class Framework(Enum):
    """Detected test/lint framework."""
    PYTEST = "pytest"
    NPM = "npm"
    CARGO = "cargo"
    GO = "go"
    MAKE = "make"
    UNKNOWN = "unknown"

@dataclass
class ParsedError:
    """A single parsed error from test/lint output."""
    file: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    message: str = ""
    severity: str = "error"  # "error" or "warning"
    rule: Optional[str] = None  # lint rule id if available

    def __str__(self) -> str:
        parts = []
        if self.file:
            loc = self.file
            if self.line is not None:
                loc += f":{self.line}"
                if self.column is not None:
                    loc += f":{self.column}"
            parts.append(loc)
        parts.append(f"[{self.severity}]")
        parts.append(self.message)
        if self.rule:
            parts.append(f"({self.rule})")
        return " ".join(parts)

@dataclass
class TestResult:
    """Result of running tests."""
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    output: str = ""
    return_code: int = 0
    framework: Framework = Framework.UNKNOWN
    parsed_errors: List[ParsedError] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.return_code == 0 and self.failed == 0 and self.errors == 0

@dataclass
class LintResult:
    """Result of running linting."""
    errors: int = 0
    warnings: int = 0
    output: str = ""
    return_code: int = 0
    framework: str = ""
    parsed_errors: List[ParsedError] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.errors == 0

@dataclass
class FixIteration:
    """One iteration of the fix loop."""
    iteration: int
    test_result: Optional[TestResult] = None
    lint_result: Optional[LintResult] = None
    errors_found: int = 0
    errors_fixed: int = 0

@dataclass
class FixLoopResult:
    """Result of the entire fix loop."""
    iterations: List[FixIteration] = field(default_factory=list)
    total_iterations: int = 0
    final_success: bool = False
    initial_errors: int = 0
    remaining_errors: int = 0
    framework: Framework = Framework.UNKNOWN

    @property
    def errors_fixed(self) -> int:
        return max(0, self.initial_errors - self.remaining_errors)


# ---------------------------------------------------------------------------
# Error parsing patterns (per framework)
# ---------------------------------------------------------------------------

# pytest: file.py:10: AssertionError: message
# pytest: FAILED tests/test_foo.py::test_bar - AssertionError: msg
PYTEST_ERROR_RE = re.compile(
    r"^(?:FAILED\s+)?([^\s:]+\.py)(?:::(\w+))?(?::(\d+))?(?:\s*[-:]\s*)(.+)$",
    re.MULTILINE,
)

# pytest summary: "1 failed, 2 passed"
PYTEST_SUMMARY_RE = re.compile(
    r"(\d+)\s+failed.*?(\d+)\s+passed|(\d+)\s+passed.*?(\d+)\s+failed",
)

# npm/jest: FAIL src/foo.test.js
#   ● test name > assertion
#     at Object.<anonymous> (src/foo.test.js:10:5)
NPM_ERROR_RE = re.compile(
    r"^\s+at\s+\S+\s+\(([^:]+):(\d+):(\d+)\)",
    re.MULTILINE,
)

# cargo: error[E0308]: mismatched types
#   --> src/main.rs:10:5
CARGO_ERROR_RE = re.compile(
    r"^\s*--> ([^:]+):(\d+):(\d+)",
    re.MULTILINE,
)
CARGO_MSG_RE = re.compile(
    r"^(error|warning)(?:\[(\w+)\])?: (.+)$",
    re.MULTILINE,
)

# go: file.go:10:5: error message
GO_ERROR_RE = re.compile(
    r"^([^\s:]+\.go):(\d+):(\d+):\s*(.+)$",
    re.MULTILINE,
)

# Generic: file:line: message (covers many linters)
GENERIC_ERROR_RE = re.compile(
    r"^([^\s:]+):(\d+)(?::(\d+))?:\s*(error|warning|Error|Warning)?\s*:?\s*(.+)$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# TestRunner
# ---------------------------------------------------------------------------

class TestRunner:
    """Detects frameworks and runs tests/linting."""

    def __init__(self, timeout: int = 120):
        self.timeout = timeout

    def auto_detect_framework(self, project_dir: str) -> Framework:
        """Detect the test framework for a project directory."""
        def has_file(*names: str) -> bool:
            return any(
                os.path.isfile(os.path.join(project_dir, n)) for n in names
            )

        def has_dir(*names: str) -> bool:
            return any(
                os.path.isdir(os.path.join(project_dir, n)) for n in names
            )

        # Check for pytest indicators
        if has_file("conftest.py", "pytest.ini", "setup.cfg"):
            return Framework.PYTEST
        if has_file("pyproject.toml"):
            try:
                with open(os.path.join(project_dir, "pyproject.toml")) as f:
                    content = f.read()
                if "[tool.pytest" in content or "pytest" in content.lower():
                    return Framework.PYTEST
            except OSError:
                pass

        # Check for Cargo (Rust)
        if has_file("Cargo.toml"):
            return Framework.CARGO

        # Check for Go
        if has_file("go.mod"):
            return Framework.GO

        # Check for npm/node
        if has_file("package.json"):
            try:
                import json
                with open(os.path.join(project_dir, "package.json")) as f:
                    pkg = json.load(f)
                if "test" in pkg.get("scripts", {}):
                    return Framework.NPM
            except (OSError, ValueError):
                pass

        # Check for Makefile
        if has_file("Makefile", "makefile"):
            return Framework.MAKE

        # Fallback: check for any Python test files
        for root, dirs, files in os.walk(project_dir):
            dirs[:] = [d for d in dirs if d not in {"node_modules", ".git", "__pycache__", "venv", ".venv"}]
            for f in files:
                if f.startswith("test_") and f.endswith(".py"):
                    return Framework.PYTEST
            break  # only check top level + 1

        return Framework.UNKNOWN

    def run_tests(self, project_dir: str, framework: Optional[Framework] = None) -> TestResult:
        """Run tests for the detected or specified framework."""
        if framework is None:
            framework = self.auto_detect_framework(project_dir)

        commands = {
            Framework.PYTEST: ["python", "-m", "pytest", "-v", "--tb=short", "--no-header"],
            Framework.NPM: ["npm", "test", "--", "--no-coverage"],
            Framework.CARGO: ["cargo", "test", "--", "--nocapture"],
            Framework.GO: ["go", "test", "./...", "-v"],
            Framework.MAKE: ["make", "test"],
        }

        cmd = commands.get(framework)
        if cmd is None:
            return TestResult(
                output="[Unknown framework - cannot run tests]",
                return_code=-1,
                framework=framework,
            )

        result = self._run_command(cmd, project_dir)
        test_result = TestResult(
            output=result.stdout + result.stderr,
            return_code=result.returncode,
            framework=framework,
        )

        # Parse errors
        test_result.parsed_errors = self.parse_errors(
            test_result.output, framework
        )
        test_result.errors = len([
            e for e in test_result.parsed_errors if e.severity == "error"
        ])

        # Parse pass/fail counts from output
        self._parse_counts(test_result)

        return test_result

    def run_lint(self, project_dir: str, framework: Optional[Framework] = None) -> LintResult:
        """Run linting for the detected or specified framework."""
        if framework is None:
            framework = self.auto_detect_framework(project_dir)

        lint_commands = {
            Framework.PYTEST: ["python", "-m", "ruff", "check", "."],
            Framework.NPM: ["npx", "eslint", ".", "--format", "unix"],
            Framework.CARGO: ["cargo", "clippy", "--", "-D", "warnings"],
            Framework.GO: ["golangci-lint", "run"],
        }

        cmd = lint_commands.get(framework)
        if cmd is None:
            # Try ruff as a universal Python linter fallback
            cmd = ["python", "-m", "ruff", "check", "."]

        result = self._run_command(cmd, project_dir)
        lint_result = LintResult(
            output=result.stdout + result.stderr,
            return_code=result.returncode,
            framework=framework.value if isinstance(framework, Framework) else str(framework),
        )

        lint_result.parsed_errors = self.parse_errors(
            lint_result.output, framework
        )
        lint_result.errors = len([
            e for e in lint_result.parsed_errors if e.severity == "error"
        ])
        lint_result.warnings = len([
            e for e in lint_result.parsed_errors if e.severity == "warning"
        ])

        return lint_result

    def parse_errors(self, output: str, framework: Framework) -> List[ParsedError]:
        """Parse errors from test/lint output for a given framework."""
        parsers = {
            Framework.PYTEST: self._parse_pytest_errors,
            Framework.NPM: self._parse_npm_errors,
            Framework.CARGO: self._parse_cargo_errors,
            Framework.GO: self._parse_go_errors,
        }
        parser = parsers.get(framework, self._parse_generic_errors)
        return parser(output)

    # -- Internal helpers ---------------------------------------------------

    def _run_command(self, cmd: List[str], cwd: str) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        try:
            return subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except FileNotFoundError:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=-1,
                stdout="",
                stderr=f"[Command not found: {cmd[0]}]",
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=-1,
                stdout="",
                stderr=f"[Command timed out after {self.timeout}s]",
            )

    def _parse_counts(self, result: TestResult) -> None:
        """Parse pass/fail counts from test output."""
        output = result.output
        if result.framework == Framework.PYTEST:
            # "1 failed, 2 passed, 1 error"
            failed_m = re.search(r"(\d+) failed", output)
            passed_m = re.search(r"(\d+) passed", output)
            errors_m = re.search(r"(\d+) error", output)
            skipped_m = re.search(r"(\d+) skipped", output)
            if failed_m:
                result.failed = int(failed_m.group(1))
            if passed_m:
                result.passed = int(passed_m.group(1))
            if errors_m:
                result.errors = int(errors_m.group(1))
            if skipped_m:
                result.skipped = int(skipped_m.group(1))
        elif result.framework == Framework.GO:
            result.passed = output.count("--- PASS:")
            result.failed = output.count("--- FAIL:")

    def _parse_pytest_errors(self, output: str) -> List[ParsedError]:
        errors = []
        for m in PYTEST_ERROR_RE.finditer(output):
            filepath = m.group(1)
            line = int(m.group(3)) if m.group(3) else None
            message = m.group(4).strip()
            errors.append(ParsedError(
                file=filepath, line=line, message=message,
            ))
        return errors

    def _parse_npm_errors(self, output: str) -> List[ParsedError]:
        errors = []
        # Capture error messages from jest/npm output
        for m in NPM_ERROR_RE.finditer(output):
            errors.append(ParsedError(
                file=m.group(1),
                line=int(m.group(2)),
                column=int(m.group(3)),
                message="Test failure",
            ))
        return errors

    def _parse_cargo_errors(self, output: str) -> List[ParsedError]:
        errors = []
        # First find all error/warning messages
        messages = list(CARGO_MSG_RE.finditer(output))
        # Then find their locations
        locations = list(CARGO_ERROR_RE.finditer(output))

        for i, loc in enumerate(locations):
            msg = messages[i].group(3) if i < len(messages) else "compile error"
            severity = messages[i].group(1) if i < len(messages) else "error"
            rule = messages[i].group(2) if i < len(messages) else None
            errors.append(ParsedError(
                file=loc.group(1),
                line=int(loc.group(2)),
                column=int(loc.group(3)),
                message=msg,
                severity=severity,
                rule=rule,
            ))
        return errors

    def _parse_go_errors(self, output: str) -> List[ParsedError]:
        errors = []
        for m in GO_ERROR_RE.finditer(output):
            errors.append(ParsedError(
                file=m.group(1),
                line=int(m.group(2)),
                column=int(m.group(3)),
                message=m.group(4).strip(),
            ))
        return errors

    def _parse_generic_errors(self, output: str) -> List[ParsedError]:
        errors = []
        for m in GENERIC_ERROR_RE.finditer(output):
            severity = (m.group(4) or "error").lower()
            errors.append(ParsedError(
                file=m.group(1),
                line=int(m.group(2)),
                column=int(m.group(3)) if m.group(3) else None,
                message=m.group(5).strip(),
                severity=severity,
            ))
        return errors


# ---------------------------------------------------------------------------
# FixLoop
# ---------------------------------------------------------------------------

class FixLoop:
    """Iterative test-fix loop that runs until green or max iterations."""

    def __init__(
        self,
        runner: Optional[TestRunner] = None,
        max_iterations: int = 5,
        run_lint: bool = True,
    ):
        self.runner = runner or TestRunner()
        self.max_iterations = max_iterations
        self.run_lint = run_lint

    def run_and_collect(
        self,
        project_dir: str,
        framework: Optional[Framework] = None,
    ) -> FixIteration:
        """Run tests (and optionally lint) and collect results.

        Returns a FixIteration with the results. This is a single step
        that can be used by the agent loop to decide what to fix.
        """
        iteration = FixIteration(iteration=0)

        test_result = self.runner.run_tests(project_dir, framework)
        iteration.test_result = test_result
        iteration.errors_found += len(test_result.parsed_errors)

        if self.run_lint:
            lint_result = self.runner.run_lint(project_dir, framework)
            iteration.lint_result = lint_result
            iteration.errors_found += lint_result.errors

        return iteration

    def run_until_green(
        self,
        project_dir: str,
        fix_callback: Optional[Callable[[FixIteration], bool]] = None,
        max_iterations: Optional[int] = None,
        framework: Optional[Framework] = None,
    ) -> FixLoopResult:
        """Run the iterative fix loop.

        Args:
            project_dir: Directory to run tests in
            fix_callback: Called after each failing iteration with the
                FixIteration. Should return True if fixes were applied,
                False to abort. If None, the loop just collects results.
            max_iterations: Override the default max iterations
            framework: Override framework detection

        Returns:
            FixLoopResult with all iterations and summary
        """
        max_iter = max_iterations or self.max_iterations
        if framework is None:
            framework = self.runner.auto_detect_framework(project_dir)

        loop_result = FixLoopResult(framework=framework)
        prev_error_count = None

        for i in range(max_iter):
            iteration = self.run_and_collect(project_dir, framework)
            iteration.iteration = i + 1
            loop_result.iterations.append(iteration)
            loop_result.total_iterations = i + 1

            if i == 0:
                loop_result.initial_errors = iteration.errors_found

            # Check if we're green
            test_ok = iteration.test_result is None or iteration.test_result.success
            lint_ok = iteration.lint_result is None or iteration.lint_result.success
            if test_ok and lint_ok:
                loop_result.final_success = True
                loop_result.remaining_errors = 0
                break

            loop_result.remaining_errors = iteration.errors_found

            # Track whether we're making progress
            if prev_error_count is not None and iteration.errors_found >= prev_error_count:
                # Not making progress - might want to try different approach
                pass
            prev_error_count = iteration.errors_found

            # Call fix callback if provided
            if fix_callback is not None:
                fixed = fix_callback(iteration)
                if not fixed:
                    break  # Callback says stop
            else:
                break  # No callback, just report first iteration

        return loop_result

    def format_errors_for_model(self, iteration: FixIteration) -> str:
        """Format errors from an iteration into a prompt-friendly string."""
        lines = []

        if iteration.test_result and not iteration.test_result.success:
            lines.append(f"=== Test Failures (iteration {iteration.iteration}) ===")
            lines.append(f"Framework: {iteration.test_result.framework.value}")
            lines.append(f"Passed: {iteration.test_result.passed}, "
                        f"Failed: {iteration.test_result.failed}, "
                        f"Errors: {iteration.test_result.errors}")
            lines.append("")
            if iteration.test_result.parsed_errors:
                lines.append("Parsed errors:")
                for err in iteration.test_result.parsed_errors:
                    lines.append(f"  {err}")
            else:
                # Include raw output if we couldn't parse specific errors
                lines.append("Raw output (last 50 lines):")
                raw_lines = iteration.test_result.output.strip().split("\n")
                for line in raw_lines[-50:]:
                    lines.append(f"  {line}")

        if iteration.lint_result and not iteration.lint_result.success:
            lines.append("")
            lines.append(f"=== Lint Errors (iteration {iteration.iteration}) ===")
            lines.append(f"Errors: {iteration.lint_result.errors}, "
                        f"Warnings: {iteration.lint_result.warnings}")
            if iteration.lint_result.parsed_errors:
                lines.append("Parsed errors:")
                for err in iteration.lint_result.parsed_errors:
                    lines.append(f"  {err}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def quick_test(project_dir: str) -> TestResult:
    """Run tests once and return the result."""
    runner = TestRunner()
    return runner.run_tests(project_dir)


def quick_lint(project_dir: str) -> LintResult:
    """Run lint once and return the result."""
    runner = TestRunner()
    return runner.run_lint(project_dir)


def detect_framework(project_dir: str) -> Framework:
    """Detect the test framework for a project."""
    runner = TestRunner()
    return runner.auto_detect_framework(project_dir)
