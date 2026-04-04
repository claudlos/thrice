"""Enhanced Iterative Test-Fix Loop for Hermes Agent (Thrice).

Higher-level wiring layer that connects test execution to the agent loop.
Builds on top of test_lint_loop.py with richer detection, parsing, and
prompt generation for the iterative fix cycle.

Flow: detect tests -> run -> parse failures -> generate fix prompt ->
      agent applies fix -> repeat until green or max iterations.
"""

import glob
import os
import re
import subprocess
import shlex
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TestFixConfig:
    """Configuration for the test-fix loop."""
    max_iterations: int = 5
    test_command: str = ""          # auto-detected if empty
    lint_command: str = ""          # optional
    auto_detect: bool = True
    run_lint_first: bool = True
    fail_fast: bool = True
    timeout: int = 120


@dataclass
class TestFailure:
    """A single test failure."""
    test_name: str = ""
    file_path: str = ""
    line_number: Optional[int] = None
    error_message: str = ""
    error_type: str = ""
    full_output: str = ""

    def summary(self) -> str:
        loc = self.file_path
        if self.line_number is not None:
            loc += f":{self.line_number}"
        parts = [loc]
        if self.test_name:
            parts.append(self.test_name)
        if self.error_type:
            parts.append(f"({self.error_type})")
        parts.append(self.error_message)
        return " :: ".join(parts)


@dataclass
class TestRunResult:
    """Result of running a test suite."""
    passed: int = 0
    failed: int = 0
    errors: int = 0
    output: str = ""
    duration: float = 0.0
    failures: List[TestFailure] = field(default_factory=list)
    return_code: int = 0

    @property
    def success(self) -> bool:
        return self.failed == 0 and self.errors == 0 and self.return_code == 0

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors


@dataclass
class LintIssue:
    """A single lint issue."""
    file_path: str = ""
    line: int = 0
    column: int = 0
    code: str = ""
    message: str = ""
    severity: str = "error"  # "error" or "warning"

    def summary(self) -> str:
        return f"{self.file_path}:{self.line}:{self.column} [{self.code}] {self.message}"


@dataclass
class LintRunResult:
    """Result of running a linter."""
    issues: List[LintIssue] = field(default_factory=list)
    clean: bool = True
    output: str = ""
    return_code: int = 0

    @property
    def error_count(self) -> int:
        return len([i for i in self.issues if i.severity == "error"])

    @property
    def warning_count(self) -> int:
        return len([i for i in self.issues if i.severity == "warning"])


@dataclass
class LoopResult:
    """Result of the full iterative test-fix loop."""
    success: bool = False
    iterations: int = 0
    final_test_result: Optional[TestRunResult] = None
    history: List[TestRunResult] = field(default_factory=list)
    lint_results: List[LintRunResult] = field(default_factory=list)
    fix_prompts: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TestDetector — framework / test file / lint tool detection
# ---------------------------------------------------------------------------

class TestDetector:
    """Detects test frameworks, relevant test files, and lint tools."""

    def detect_test_framework(self, project_root: str) -> Dict[str, str]:
        """Detect the test framework for a project.

        Returns dict with keys: framework, command, config_file
        """
        root = project_root

        # pytest
        for cfg in ("pytest.ini", "conftest.py", "setup.cfg"):
            if os.path.isfile(os.path.join(root, cfg)):
                return {
                    "framework": "pytest",
                    "command": "python -m pytest -v --tb=short",
                    "config_file": cfg,
                }
        if os.path.isfile(os.path.join(root, "pyproject.toml")):
            try:
                with open(os.path.join(root, "pyproject.toml")) as f:
                    content = f.read()
                if "[tool.pytest" in content or "pytest" in content.lower():
                    return {
                        "framework": "pytest",
                        "command": "python -m pytest -v --tb=short",
                        "config_file": "pyproject.toml",
                    }
            except OSError:
                pass

        # jest / mocha (node)
        if os.path.isfile(os.path.join(root, "package.json")):
            try:
                import json
                with open(os.path.join(root, "package.json")) as f:
                    pkg = json.load(f)
                deps = {}
                deps.update(pkg.get("devDependencies", {}))
                deps.update(pkg.get("dependencies", {}))
                if "jest" in deps or os.path.isfile(os.path.join(root, "jest.config.js")):
                    return {
                        "framework": "jest",
                        "command": "npx jest --verbose",
                        "config_file": "package.json",
                    }
                if "mocha" in deps:
                    return {
                        "framework": "mocha",
                        "command": "npx mocha --recursive",
                        "config_file": "package.json",
                    }
                # generic npm test
                if "test" in pkg.get("scripts", {}):
                    return {
                        "framework": "jest",
                        "command": "npm test",
                        "config_file": "package.json",
                    }
            except (OSError, ValueError):
                pass

        # cargo test (Rust)
        if os.path.isfile(os.path.join(root, "Cargo.toml")):
            return {
                "framework": "cargo_test",
                "command": "cargo test -- --nocapture",
                "config_file": "Cargo.toml",
            }

        # go test
        if os.path.isfile(os.path.join(root, "go.mod")):
            return {
                "framework": "go_test",
                "command": "go test ./... -v",
                "config_file": "go.mod",
            }

        # unittest fallback — look for test_*.py files
        for entry in os.listdir(root):
            if entry.startswith("test_") and entry.endswith(".py"):
                return {
                    "framework": "unittest",
                    "command": "python -m unittest discover -v",
                    "config_file": "",
                }
        tests_dir = os.path.join(root, "tests")
        if os.path.isdir(tests_dir):
            for entry in os.listdir(tests_dir):
                if entry.startswith("test_") and entry.endswith(".py"):
                    return {
                        "framework": "unittest",
                        "command": "python -m unittest discover -s tests -v",
                        "config_file": "",
                    }

        return {"framework": "unknown", "command": "", "config_file": ""}

    def detect_relevant_tests(
        self, changed_files: List[str], project_root: str
    ) -> List[str]:
        """Find test files relevant to the changed source files.

        Heuristics:
        - foo.py -> test_foo.py, tests/test_foo.py
        - foo.js -> __tests__/foo.test.js, foo.test.js
        - foo.rs -> same file (Rust inline tests)
        """
        test_files: List[str] = []
        seen: set = set()

        for changed in changed_files:
            basename = os.path.basename(changed)
            name, ext = os.path.splitext(basename)
            dirname = os.path.dirname(changed)

            candidates: List[str] = []

            if ext == ".py":
                # test_foo.py in same dir
                candidates.append(os.path.join(dirname, f"test_{name}.py"))
                # tests/test_foo.py relative to project root
                candidates.append(os.path.join(project_root, "tests", f"test_{name}.py"))
                # test_foo.py at project root
                candidates.append(os.path.join(project_root, f"test_{name}.py"))
            elif ext in (".js", ".ts", ".jsx", ".tsx"):
                # __tests__/foo.test.js
                candidates.append(os.path.join(dirname, "__tests__", f"{name}.test{ext}"))
                # foo.test.js in same dir
                candidates.append(os.path.join(dirname, f"{name}.test{ext}"))
                # foo.spec.js in same dir
                candidates.append(os.path.join(dirname, f"{name}.spec{ext}"))
            elif ext == ".rs":
                # Rust: tests are inline, so include the source file itself
                candidates.append(changed)
                # Also check tests/ directory
                candidates.append(os.path.join(project_root, "tests", f"{name}.rs"))
            elif ext == ".go":
                # foo_test.go in same dir
                candidates.append(os.path.join(dirname, f"{name}_test.go"))

            # Also check if the file itself is a test file
            if self._is_test_file(basename):
                candidates.append(changed)

            for candidate in candidates:
                full = os.path.join(project_root, candidate) if not os.path.isabs(candidate) else candidate
                if full not in seen and os.path.isfile(full):
                    seen.add(full)
                    test_files.append(full)

        return test_files

    def detect_lint_tool(self, project_root: str) -> Dict[str, str]:
        """Detect the lint tool for a project.

        Returns dict with keys: tool, command
        """
        root = project_root

        # Python linters
        if os.path.isfile(os.path.join(root, "ruff.toml")) or os.path.isfile(
            os.path.join(root, ".ruff.toml")
        ):
            return {"tool": "ruff", "command": "ruff check ."}

        if os.path.isfile(os.path.join(root, "pyproject.toml")):
            try:
                with open(os.path.join(root, "pyproject.toml")) as f:
                    content = f.read()
                if "[tool.ruff" in content:
                    return {"tool": "ruff", "command": "ruff check ."}
                if "[tool.pylint" in content:
                    return {"tool": "pylint", "command": "pylint ."}
            except OSError:
                pass

        if os.path.isfile(os.path.join(root, ".flake8")):
            return {"tool": "flake8", "command": "flake8 ."}

        if os.path.isfile(os.path.join(root, ".pylintrc")):
            return {"tool": "pylint", "command": "pylint ."}

        # JS linters
        if os.path.isfile(os.path.join(root, ".eslintrc.js")) or os.path.isfile(
            os.path.join(root, ".eslintrc.json")
        ) or os.path.isfile(os.path.join(root, ".eslintrc.yml")):
            return {"tool": "eslint", "command": "npx eslint ."}

        # Rust
        if os.path.isfile(os.path.join(root, "rustfmt.toml")) or os.path.isfile(
            os.path.join(root, ".rustfmt.toml")
        ):
            return {"tool": "rustfmt", "command": "cargo fmt -- --check"}
        if os.path.isfile(os.path.join(root, "Cargo.toml")):
            return {"tool": "rustfmt", "command": "cargo clippy -- -D warnings"}

        # Fallback: check if any Python files exist -> try ruff
        for entry in os.listdir(root):
            if entry.endswith(".py"):
                return {"tool": "ruff", "command": "ruff check ."}

        return {"tool": "unknown", "command": ""}

    @staticmethod
    def _is_test_file(filename: str) -> bool:
        """Check if a filename looks like a test file."""
        name = filename.lower()
        return (
            name.startswith("test_")
            or name.endswith("_test.py")
            or name.endswith(".test.js")
            or name.endswith(".test.ts")
            or name.endswith(".spec.js")
            or name.endswith(".spec.ts")
            or name.endswith("_test.go")
        )


# ---------------------------------------------------------------------------
# FailureParser — framework-specific output parsing
# ---------------------------------------------------------------------------

class FailureParser:
    """Parse test output into structured TestFailure objects."""

    def parse(self, output: str, framework: str) -> List[TestFailure]:
        """Dispatch to framework-specific parser."""
        parsers = {
            "pytest": self.parse_pytest,
            "unittest": self.parse_unittest,
            "jest": self.parse_jest,
            "mocha": self.parse_jest,  # similar output format
            "cargo_test": self.parse_cargo_test,
            "go_test": self.parse_go_test,
        }
        parser = parsers.get(framework, self.parse_generic)
        return parser(output)

    def parse_pytest(self, output: str) -> List[TestFailure]:
        """Parse pytest verbose output."""
        failures: List[TestFailure] = []

        # Pattern: FAILED tests/test_foo.py::test_bar - AssertionError: msg
        failed_re = re.compile(
            r"FAILED\s+([^\s:]+\.py)::(\S+)\s*[-–]\s*(\w+(?:Error|Exception|Failure)?):?\s*(.*)",
            re.MULTILINE,
        )
        for m in failed_re.finditer(output):
            failures.append(TestFailure(
                file_path=m.group(1),
                test_name=m.group(2),
                error_type=m.group(3),
                error_message=m.group(4).strip(),
                full_output=m.group(0),
            ))

        # Pattern: file.py:10: AssertionError
        loc_re = re.compile(
            r"^([^\s:]+\.py):(\d+):\s*(\w+(?:Error|Exception|Failure)?)(?::\s*(.*))?$",
            re.MULTILINE,
        )
        seen_files = {(f.file_path, f.test_name) for f in failures}
        for m in loc_re.finditer(output):
            fpath = m.group(1)
            test_name = ""
            key = (fpath, test_name)
            if key not in seen_files:
                failures.append(TestFailure(
                    file_path=fpath,
                    line_number=int(m.group(2)),
                    error_type=m.group(3),
                    error_message=(m.group(4) or "").strip(),
                    full_output=m.group(0),
                ))
                seen_files.add(key)

        # Parse summary counts (used by the loop)
        return failures

    def parse_unittest(self, output: str) -> List[TestFailure]:
        """Parse unittest verbose output."""
        failures: List[TestFailure] = []

        # Pattern: FAIL: test_name (module.ClassName)
        fail_re = re.compile(
            r"^(FAIL|ERROR): (\S+) \((\S+)\)",
            re.MULTILINE,
        )

        # Split on separator lines to get each failure block
        blocks = re.split(r"^={70}$", output, flags=re.MULTILINE)

        for block in blocks:
            m = fail_re.search(block)
            if m:
                error_type_label = m.group(1)  # FAIL or ERROR
                test_name = m.group(2)
                module = m.group(3)

                # Find the actual error
                tb_match = re.search(
                    r'File "([^"]+)", line (\d+)',
                    block,
                )
                file_path = tb_match.group(1) if tb_match else ""
                line_num = int(tb_match.group(2)) if tb_match else None

                # Error type from last line of traceback
                err_match = re.search(
                    r"^(\w+(?:Error|Exception|Failure)):\s*(.*)",
                    block,
                    re.MULTILINE,
                )
                error_type = err_match.group(1) if err_match else error_type_label
                error_msg = err_match.group(2).strip() if err_match else ""

                failures.append(TestFailure(
                    test_name=f"{module}.{test_name}",
                    file_path=file_path,
                    line_number=line_num,
                    error_type=error_type,
                    error_message=error_msg,
                    full_output=block.strip(),
                ))

        return failures

    def parse_jest(self, output: str) -> List[TestFailure]:
        """Parse jest/mocha output."""
        failures: List[TestFailure] = []

        # Pattern: FAIL src/foo.test.js
        fail_file_re = re.compile(r"^FAIL\s+(\S+)", re.MULTILINE)

        # Pattern: ● test suite > test name
        test_re = re.compile(r"^\s+●\s+(.+)$", re.MULTILINE)

        # Pattern: at Object.<anonymous> (file:line:col)
        loc_re = re.compile(
            r"at\s+\S+\s+\(([^:]+):(\d+):(\d+)\)",
        )

        # Pattern: Error: message or expect(received).toBe(expected)
        msg_re = re.compile(
            r"^\s+((?:Error|TypeError|ReferenceError|expect).+)$",
            re.MULTILINE,
        )

        current_file = ""
        # Split by test marker
        for m in test_re.finditer(output):
            test_name = m.group(1).strip()
            # Look backwards for FAIL file
            before = output[:m.start()]
            file_m = None
            for file_m in fail_file_re.finditer(before):
                pass  # get last one
            if file_m:
                current_file = file_m.group(1)

            # Look forward for location and error
            after = output[m.start():m.start() + 2000]
            loc_m = loc_re.search(after)
            msg_m = msg_re.search(after)

            failures.append(TestFailure(
                test_name=test_name,
                file_path=loc_m.group(1) if loc_m else current_file,
                line_number=int(loc_m.group(2)) if loc_m else None,
                error_type="AssertionError",
                error_message=msg_m.group(1).strip() if msg_m else "",
                full_output=after[:500],
            ))

        return failures

    def parse_cargo_test(self, output: str) -> List[TestFailure]:
        """Parse cargo test output."""
        failures: List[TestFailure] = []

        # Pattern: test module::test_name ... FAILED
        fail_re = re.compile(
            r"^test\s+(\S+)\s+\.\.\.\s+FAILED",
            re.MULTILINE,
        )

        # Failure detail blocks start with "---- test_name stdout ----"
        detail_re = re.compile(
            r"---- (\S+) stdout ----\n(.*?)(?=\n----|\nfailures::\n|\Z)",
            re.DOTALL,
        )
        detail_map: Dict[str, str] = {}
        for m in detail_re.finditer(output):
            detail_map[m.group(1)] = m.group(2).strip()

        for m in fail_re.finditer(output):
            test_name = m.group(1)
            detail = detail_map.get(test_name, "")

            # Try to find file:line in detail
            loc_re2 = re.compile(r"([^\s:]+\.rs):(\d+)(?::(\d+))?")
            loc_m = loc_re2.search(detail)

            # Try to find panic message
            panic_re = re.compile(r"panicked at '([^']+)'")
            panic_m = panic_re.search(detail)

            # Try assertion message
            assert_re = re.compile(r"assertion `[^`]*` failed:?\s*(.*)")
            assert_m = assert_re.search(detail)

            error_msg = ""
            if panic_m:
                error_msg = panic_m.group(1)
            elif assert_m:
                error_msg = assert_m.group(1).strip()

            failures.append(TestFailure(
                test_name=test_name,
                file_path=loc_m.group(1) if loc_m else "",
                line_number=int(loc_m.group(2)) if loc_m else None,
                error_type="panic" if panic_m else "assertion",
                error_message=error_msg,
                full_output=detail[:1000],
            ))

        return failures

    def parse_go_test(self, output: str) -> List[TestFailure]:
        """Parse go test -v output."""
        failures: List[TestFailure] = []

        # Pattern: --- FAIL: TestName (0.00s)
        fail_re = re.compile(
            r"^--- FAIL: (\S+)\s+\(([^)]+)\)",
            re.MULTILINE,
        )

        # Error location: file_test.go:42: message
        loc_re = re.compile(
            r"^\s+([^\s:]+\.go):(\d+):\s*(.+)$",
            re.MULTILINE,
        )

        for m in fail_re.finditer(output):
            test_name = m.group(1)
            # Look backwards and forwards for error details
            before = output[max(0, m.start() - 1000):m.start()]
            after = output[m.start():m.start() + 1000]
            loc_m = loc_re.search(after) or loc_re.search(before)

            failures.append(TestFailure(
                test_name=test_name,
                file_path=loc_m.group(1) if loc_m else "",
                line_number=int(loc_m.group(2)) if loc_m else None,
                error_type="FAIL",
                error_message=loc_m.group(3).strip() if loc_m else "",
                full_output=after[:500],
            ))

        return failures

    def parse_generic(self, output: str) -> List[TestFailure]:
        """Fallback generic parser."""
        failures: List[TestFailure] = []
        generic_re = re.compile(
            r"^([^\s:]+):(\d+)(?::(\d+))?:\s*(error|Error|FAIL)?\s*:?\s*(.+)$",
            re.MULTILINE,
        )
        for m in generic_re.finditer(output):
            failures.append(TestFailure(
                file_path=m.group(1),
                line_number=int(m.group(2)),
                error_type=m.group(4) or "error",
                error_message=m.group(5).strip(),
                full_output=m.group(0),
            ))
        return failures


# ---------------------------------------------------------------------------
# Lint output parser
# ---------------------------------------------------------------------------

class LintOutputParser:
    """Parse lint tool output into LintIssue objects."""

    def parse(self, output: str, tool: str) -> List[LintIssue]:
        """Dispatch to tool-specific parser."""
        parsers = {
            "ruff": self._parse_ruff,
            "flake8": self._parse_flake8,
            "pylint": self._parse_pylint,
            "eslint": self._parse_eslint,
            "rustfmt": self._parse_rustfmt,
        }
        parser = parsers.get(tool, self._parse_generic)
        return parser(output)

    def _parse_ruff(self, output: str) -> List[LintIssue]:
        """Parse ruff output: file.py:10:5: E501 Line too long."""
        issues: List[LintIssue] = []
        pat = re.compile(
            r"^([^\s:]+):(\d+):(\d+):\s+(\S+)\s+(.+)$",
            re.MULTILINE,
        )
        for m in pat.finditer(output):
            issues.append(LintIssue(
                file_path=m.group(1),
                line=int(m.group(2)),
                column=int(m.group(3)),
                code=m.group(4),
                message=m.group(5).strip(),
                severity="error",
            ))
        return issues

    def _parse_flake8(self, output: str) -> List[LintIssue]:
        """Parse flake8 output (same format as ruff)."""
        return self._parse_ruff(output)

    def _parse_pylint(self, output: str) -> List[LintIssue]:
        """Parse pylint output: file.py:10:5: C0114: message (code)."""
        issues: List[LintIssue] = []
        pat = re.compile(
            r"^([^\s:]+):(\d+):(\d+):\s+([A-Z]\d{4}):\s+(.+)$",
            re.MULTILINE,
        )
        for m in pat.finditer(output):
            code = m.group(4)
            severity = "error" if code.startswith(("E", "F")) else "warning"
            issues.append(LintIssue(
                file_path=m.group(1),
                line=int(m.group(2)),
                column=int(m.group(3)),
                code=code,
                message=m.group(5).strip(),
                severity=severity,
            ))
        return issues

    def _parse_eslint(self, output: str) -> List[LintIssue]:
        """Parse eslint unix format: file.js:10:5: message [severity/rule]."""
        issues: List[LintIssue] = []
        pat = re.compile(
            r"^([^\s:]+):(\d+):(\d+):\s+(.+?)(?:\s+\[(\w+)/([^\]]+)\])?$",
            re.MULTILINE,
        )
        for m in pat.finditer(output):
            severity_raw = m.group(5) or "Error"
            issues.append(LintIssue(
                file_path=m.group(1),
                line=int(m.group(2)),
                column=int(m.group(3)),
                code=m.group(6) or "",
                message=m.group(4).strip(),
                severity="warning" if severity_raw.lower() == "warning" else "error",
            ))
        return issues

    def _parse_rustfmt(self, output: str) -> List[LintIssue]:
        """Parse cargo clippy / rustfmt output."""
        issues: List[LintIssue] = []
        # clippy: warning[clippy::rule]: message --> file:line:col
        msg_re = re.compile(
            r"^(warning|error)(?:\[([^\]]+)\])?: (.+)$",
            re.MULTILINE,
        )
        loc_re = re.compile(
            r"^\s+--> ([^:]+):(\d+):(\d+)",
            re.MULTILINE,
        )

        messages = list(msg_re.finditer(output))
        locations = list(loc_re.finditer(output))

        for i, loc in enumerate(locations):
            msg_obj = messages[i] if i < len(messages) else None
            issues.append(LintIssue(
                file_path=loc.group(1),
                line=int(loc.group(2)),
                column=int(loc.group(3)),
                code=msg_obj.group(2) if msg_obj and msg_obj.group(2) else "",
                message=msg_obj.group(3) if msg_obj else "lint issue",
                severity=msg_obj.group(1) if msg_obj else "warning",
            ))

        return issues

    def _parse_generic(self, output: str) -> List[LintIssue]:
        """Generic file:line:col: message parser."""
        issues: List[LintIssue] = []
        pat = re.compile(
            r"^([^\s:]+):(\d+):(\d+):\s+(.+)$",
            re.MULTILINE,
        )
        for m in pat.finditer(output):
            issues.append(LintIssue(
                file_path=m.group(1),
                line=int(m.group(2)),
                column=int(m.group(3)),
                message=m.group(4).strip(),
            ))
        return issues


# ---------------------------------------------------------------------------
# TestFixLoop — the main orchestrator
# ---------------------------------------------------------------------------

class TestFixLoop:
    """Iterative test-fix loop that wires test execution to the agent.

    Usage:
        config = TestFixConfig(max_iterations=3)
        loop = TestFixLoop(config, project_root="/path/to/project")
        result = loop.iterate(changed_files=["src/foo.py"])
        if not result.success:
            # result.fix_prompts[-1] has the prompt for the agent
            pass
    """

    def __init__(self, config: TestFixConfig, project_root: str):
        self.config = config
        self.project_root = project_root
        self.detector = TestDetector()
        self.failure_parser = FailureParser()
        self.lint_parser = LintOutputParser()

        # Auto-detect if needed
        if config.auto_detect and not config.test_command:
            info = self.detector.detect_test_framework(project_root)
            self.config.test_command = info["command"]
            self._framework = info["framework"]
        else:
            self._framework = "unknown"

        if config.auto_detect and not config.lint_command:
            lint_info = self.detector.detect_lint_tool(project_root)
            self.config.lint_command = lint_info["command"]
            self._lint_tool = lint_info["tool"]
        else:
            self._lint_tool = "unknown"

    def run_tests(self, test_files: Optional[List[str]] = None) -> TestRunResult:
        """Run tests and return structured result."""
        cmd = self.config.test_command
        if not cmd:
            return TestRunResult(
                output="[No test command configured]",
                return_code=-1,
            )

        if test_files:
            cmd = cmd + " " + " ".join(test_files)

        start = time.monotonic()
        try:
            proc = subprocess.run(
                shlex.split(cmd),
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )
            elapsed = time.monotonic() - start
            output = proc.stdout + proc.stderr

            # Parse failures
            failures = self.failure_parser.parse(output, self._framework)

            # Parse counts from output
            passed, failed, errors = self._parse_counts(output)

            # If parser found failures but count parsing didn't, use parser count
            if failures and failed == 0:
                failed = len(failures)

            return TestRunResult(
                passed=passed,
                failed=failed,
                errors=errors,
                output=output,
                duration=elapsed,
                failures=failures,
                return_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            return TestRunResult(
                output=f"[Test command timed out after {self.config.timeout}s]",
                duration=elapsed,
                errors=1,
                return_code=-1,
            )
        except Exception as e:
            elapsed = time.monotonic() - start
            return TestRunResult(
                output=f"[Error running tests: {e}]",
                duration=elapsed,
                errors=1,
                return_code=-1,
            )

    def run_lint(self, files: Optional[List[str]] = None) -> LintRunResult:
        """Run linter and return structured result."""
        cmd = self.config.lint_command
        if not cmd:
            return LintRunResult(clean=True)

        if files:
            cmd = cmd + " " + " ".join(files)

        try:
            proc = subprocess.run(
                shlex.split(cmd),
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )
            output = proc.stdout + proc.stderr
            issues = self.lint_parser.parse(output, self._lint_tool)

            return LintRunResult(
                issues=issues,
                clean=len(issues) == 0 and proc.returncode == 0,
                output=output,
                return_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return LintRunResult(
                output=f"[Lint command timed out after {self.config.timeout}s]",
                clean=False,
                return_code=-1,
            )
        except Exception as e:
            return LintRunResult(
                output=f"[Error running lint: {e}]",
                clean=False,
                return_code=-1,
            )

    def parse_failures(self, output: str, framework: str) -> List[TestFailure]:
        """Parse test failures from output."""
        return self.failure_parser.parse(output, framework)

    def generate_fix_prompt(
        self, failures: List[TestFailure], attempt: int,
        lint_issues: Optional[List[LintIssue]] = None,
    ) -> str:
        """Generate a prompt for the agent to fix the failures.

        This is the key integration point — it formats errors into a
        prompt that the agent model can act on.
        """
        lines: List[str] = []

        if attempt == 1:
            lines.append("Tests are failing. Please fix the following issues:\n")
        else:
            lines.append(
                f"Tests are still failing (attempt {attempt}/{self.config.max_iterations}). "
                f"Please fix the remaining issues:\n"
            )

        if failures:
            lines.append(f"## Test Failures ({len(failures)}):\n")
            for i, f in enumerate(failures, 1):
                lines.append(f"{i}. {f.summary()}")
                if f.full_output and len(f.full_output) < 500:
                    lines.append(f"   Output: {f.full_output}")
                lines.append("")

        if lint_issues:
            errors = [i for i in lint_issues if i.severity == "error"]
            if errors:
                lines.append(f"\n## Lint Errors ({len(errors)}):\n")
                for i, issue in enumerate(errors[:20], 1):  # cap at 20
                    lines.append(f"{i}. {issue.summary()}")
                if len(errors) > 20:
                    lines.append(f"   ... and {len(errors) - 20} more")
                lines.append("")

        lines.append(
            "\nFix the code so all tests pass. "
            "Focus on the root cause, not just the symptoms."
        )

        return "\n".join(lines)

    def iterate(self, changed_files: List[str]) -> LoopResult:
        """Main iterative test-fix loop.

        1. Detect relevant tests for changed files
        2. Optionally run lint first
        3. Run tests
        4. Parse failures
        5. Generate fix prompt
        6. Repeat until green or max iterations

        Returns LoopResult with history and fix prompts.
        The caller (agent loop) should apply fixes between iterations.
        """
        result = LoopResult()

        # Detect relevant test files
        test_files = self.detector.detect_relevant_tests(
            changed_files, self.project_root
        )

        for i in range(self.config.max_iterations):
            result.iterations = i + 1

            # Run lint first if configured
            if self.config.run_lint_first and self.config.lint_command:
                lint_result = self.run_lint(
                    files=changed_files if changed_files else None
                )
                result.lint_results.append(lint_result)
            else:
                lint_result = None

            # Run tests
            test_result = self.run_tests(
                test_files=test_files if test_files else None
            )
            result.history.append(test_result)
            result.final_test_result = test_result

            # Check if all green
            if test_result.success and (lint_result is None or lint_result.clean):
                result.success = True
                break

            # Generate fix prompt for the agent
            lint_issues = lint_result.issues if lint_result else None
            prompt = self.generate_fix_prompt(
                test_result.failures, i + 1, lint_issues
            )
            result.fix_prompts.append(prompt)

            # If fail_fast and no failures found to fix, stop
            if self.config.fail_fast and not test_result.failures and test_result.return_code != 0:
                # Tests failed but we can't parse why — still provide output
                prompt = (
                    f"Tests failed (exit code {test_result.return_code}) but "
                    f"failures could not be parsed. Raw output:\n\n"
                    f"{test_result.output[:3000]}"
                )
                result.fix_prompts[-1] = prompt
                break

        return result

    def _parse_counts(self, output: str) -> Tuple[int, int, int]:
        """Parse pass/fail/error counts from test output."""
        passed = failed = errors = 0

        # pytest: "1 failed, 2 passed, 1 error"
        m = re.search(r"(\d+) passed", output)
        if m:
            passed = int(m.group(1))
        m = re.search(r"(\d+) failed", output)
        if m:
            failed = int(m.group(1))
        m = re.search(r"(\d+) error", output)
        if m:
            errors = int(m.group(1))

        # unittest: "Ran 5 tests ... OK / FAILED (failures=2, errors=1)"
        m = re.search(r"Ran (\d+) test", output)
        if m and passed == 0:
            total = int(m.group(1))
            fm = re.search(r"failures=(\d+)", output)
            em = re.search(r"errors=(\d+)", output)
            if fm:
                failed = int(fm.group(1))
            if em:
                errors = int(em.group(1))
            passed = total - failed - errors

        # go test: count PASS/FAIL lines
        go_pass = output.count("--- PASS:")
        go_fail = output.count("--- FAIL:")
        if go_pass or go_fail:
            passed = max(passed, go_pass)
            failed = max(failed, go_fail)

        return passed, failed, errors
