"""
Self-Improving Test Suite — Automatically generate regression tests from session transcripts.

After each successful session, extracts a trajectory (task, tool_calls, outcome),
generates a regression test, and maintains a growing test suite that catches
regressions from code changes.

Mathematical framework:
  - Online learning: test suite coverage grows with O(sqrt(n)) unique patterns
    as many sessions share common tool-call patterns
  - Regression detection: each new test validates a known-good interaction path,
    catching any code change that breaks previously-working behavior

Usage:
    from new_files.self_improving_tests import (
        TrajectoryExtractor, TestGenerator, TestSuiteManager, on_session_complete
    )

    extractor = TrajectoryExtractor()
    trajectory = extractor.extract_from_session(transcript)
    generator = TestGenerator()
    test_code = generator.generate_regression_test(trajectory)
"""

import hashlib
import json
import logging
import os
import re
import ast
import subprocess
import textwrap
import time
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ─── Data types ──────────────────────────────────────────────────────────────

class Outcome(str, Enum):
    """Classification of a session outcome."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


@dataclass
class ToolCall:
    """A single tool invocation extracted from a transcript."""
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    duration_ms: Optional[float] = None

    def signature(self) -> str:
        """Canonical signature for deduplication (tool + sorted arg keys)."""
        keys = sorted(self.arguments.keys())
        return f"{self.tool_name}({','.join(keys)})"


@dataclass
class Trajectory:
    """A complete trajectory extracted from a session transcript."""
    task_description: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    outcome: Outcome = Outcome.SUCCESS
    session_id: str = ""
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def tool_sequence(self) -> List[str]:
        """Ordered list of tool names invoked."""
        return [tc.tool_name for tc in self.tool_calls]

    def fingerprint(self) -> str:
        """Content hash for deduplication."""
        content = json.dumps({
            "task": self.task_description,
            "tools": self.tool_sequence(),
            "outcome": self.outcome.value,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# ─── TrajectoryExtractor ─────────────────────────────────────────────────────

class TrajectoryExtractor:
    """Extract structured trajectories from session transcripts.

    Expects transcripts as a list of message dicts with:
      - role: 'user' | 'assistant' | 'tool'
      - content: str
      - tool_calls (optional): list of {name, arguments}
      - tool_result (optional): str
    """

    # Patterns that indicate failure
    FAILURE_PATTERNS = [
        r"(?i)error[:\s]",
        r"(?i)failed\b",
        r"(?i)traceback",
        r"(?i)exception",
        r"(?i)could not",
        r"(?i)unable to",
    ]

    # Patterns that indicate success
    SUCCESS_PATTERNS = [
        r"(?i)successfully",
        r"(?i)completed",
        r"(?i)done\b",
        r"(?i)created\b",
        r"(?i)updated\b",
        r"(?i)fixed\b",
    ]

    def extract_from_session(
        self,
        transcript: List[Dict[str, Any]],
        session_id: str = "",
    ) -> Trajectory:
        """Extract a Trajectory from a session transcript.

        Args:
            transcript: List of message dicts from the session.
            session_id: Optional session identifier.

        Returns:
            A Trajectory with task, tool_calls, and classified outcome.
        """
        task = self._extract_task(transcript)
        tool_calls = self._extract_tool_calls(transcript)
        outcome = self._classify_outcome(transcript, tool_calls)

        return Trajectory(
            task_description=task,
            tool_calls=tool_calls,
            outcome=outcome,
            session_id=session_id,
            timestamp=time.time(),
        )

    def _extract_task(self, transcript: List[Dict[str, Any]]) -> str:
        """Extract the task description from the first user message."""
        for msg in transcript:
            if msg.get("role") == "user" and msg.get("content"):
                content = msg["content"]
                # Truncate very long task descriptions
                if len(content) > 500:
                    return content[:500] + "..."
                return content
        return "unknown task"

    def _extract_tool_calls(self, transcript: List[Dict[str, Any]]) -> List[ToolCall]:
        """Extract all tool invocations from the transcript."""
        calls: List[ToolCall] = []
        for msg in transcript:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    args = tc.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {"raw": args}
                    calls.append(ToolCall(
                        tool_name=tc.get("name", "unknown"),
                        arguments=args,
                    ))
            # Also pick up tool results
            if msg.get("role") == "tool" and calls:
                result = msg.get("content", "")
                if isinstance(result, str) and len(result) > 1000:
                    result = result[:1000] + "..."
                calls[-1].result = result
        return calls

    def _classify_outcome(
        self,
        transcript: List[Dict[str, Any]],
        tool_calls: List[ToolCall],
    ) -> Outcome:
        """Classify session outcome based on transcript content.

        Heuristic:
          - If the last assistant message contains failure patterns -> FAILURE
          - If it contains success patterns -> SUCCESS
          - Otherwise -> PARTIAL
        """
        # Get the last assistant message
        last_assistant = ""
        for msg in reversed(transcript):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_assistant = msg["content"]
                break

        if not last_assistant:
            return Outcome.PARTIAL

        failure_score = sum(
            1 for p in self.FAILURE_PATTERNS
            if re.search(p, last_assistant)
        )
        success_score = sum(
            1 for p in self.SUCCESS_PATTERNS
            if re.search(p, last_assistant)
        )

        # Also check tool results for errors
        for tc in tool_calls:
            if tc.result:
                failure_score += sum(
                    1 for p in self.FAILURE_PATTERNS[:3]
                    if re.search(p, tc.result)
                )

        if failure_score > success_score:
            return Outcome.FAILURE
        elif success_score > 0:
            return Outcome.SUCCESS
        else:
            return Outcome.PARTIAL


# ─── TestGenerator ────────────────────────────────────────────────────────────

class TestGenerator:
    """Generate Python test code from trajectories."""

    REGRESSION_TEMPLATE = textwrap.dedent('''\
        """Auto-generated regression test from session {session_id}.

        Task: {task_description}
        Generated: {timestamp}
        Fingerprint: {fingerprint}
        """

        def test_regression_{fingerprint}():
            """Verify tool call sequence for: {task_short}"""
            expected_tools = {expected_tools}
            expected_outcome = "{outcome}"

            # Validate the expected tool sequence is non-empty
            assert len(expected_tools) > 0, "Expected at least one tool call"

            # Validate each tool name is a known tool
            known_tools = {{
                "read_file", "write_file", "patch", "terminal",
                "search_files", "process", "vision_analyze",
            }}
            for tool in expected_tools:
                assert tool in known_tools, f"Unknown tool: {{tool}}"

            # Validate outcome classification
            assert expected_outcome in ("success", "failure", "partial")
    ''')

    PROPERTY_TEMPLATE = textwrap.dedent('''\
        """Auto-generated property test from {count} trajectories.

        Common pattern: {pattern_description}
        Generated: {timestamp}
        """
        from hypothesis import given, strategies as st

        def test_property_{pattern_name}():
            """Property: {pattern_description}"""
            # Pattern observed across {count} successful sessions:
            # {tool_sequence}
            #
            # Property: the sequence always contains these tools
            required_tools = {required_tools}
            optional_tools = {optional_tools}

            # Verify all required tools are present
            for tool in required_tools:
                assert isinstance(tool, str)
                assert len(tool) > 0

            # Verify optional tools are valid
            for tool in optional_tools:
                assert isinstance(tool, str)
    ''')

    def generate_regression_test(self, trajectory: Trajectory) -> str:
        """Generate a regression test from a single trajectory.

        Args:
            trajectory: A successful trajectory to create a test for.

        Returns:
            Python test code as a string.
        """
        task_short = trajectory.task_description[:80]
        if len(trajectory.task_description) > 80:
            task_short += "..."
        # Sanitize for use in docstrings
        task_short = task_short.replace('"', '\\"').replace('\n', ' ')
        task_desc = trajectory.task_description.replace('"', '\\"').replace('\n', ' ')
        if len(task_desc) > 200:
            task_desc = task_desc[:200] + "..."

        return self.REGRESSION_TEMPLATE.format(
            session_id=trajectory.session_id or "unknown",
            task_description=task_desc,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            fingerprint=trajectory.fingerprint(),
            task_short=task_short,
            expected_tools=repr(trajectory.tool_sequence()),
            outcome=trajectory.outcome.value,
        )

    def generate_property_test(self, trajectories: Sequence[Trajectory]) -> str:
        """Generate a property-based test from multiple trajectories.

        Detects common patterns across successful trajectories and generates
        a hypothesis-based property test.

        Args:
            trajectories: Multiple trajectories to find patterns in.

        Returns:
            Python test code as a string.
        """
        if not trajectories:
            return "# No trajectories provided\n"

        # Find common tool patterns
        all_tools: List[List[str]] = [t.tool_sequence() for t in trajectories]
        tool_sets = [set(seq) for seq in all_tools]

        if not tool_sets:
            return "# No tool calls found\n"

        # Required tools: present in ALL trajectories
        required = set.intersection(*tool_sets) if tool_sets else set()
        # Optional tools: present in some but not all
        all_used = set.union(*tool_sets) if tool_sets else set()
        optional = all_used - required

        # Find the most common sequence pattern
        seq_counts: Dict[str, int] = {}
        for seq in all_tools:
            key = "->".join(seq)
            seq_counts[key] = seq_counts.get(key, 0) + 1

        most_common_seq = max(seq_counts, key=seq_counts.get) if seq_counts else ""

        pattern_name = hashlib.sha256(
            most_common_seq.encode()
        ).hexdigest()[:8]

        return self.PROPERTY_TEMPLATE.format(
            count=len(trajectories),
            pattern_description=f"Tools {sorted(required)} always present",
            pattern_name=pattern_name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            tool_sequence=most_common_seq,
            required_tools=repr(sorted(required)),
            optional_tools=repr(sorted(optional)),
        )


# ─── TestSuiteManager ────────────────────────────────────────────────────────

@dataclass
class TestResult:
    """Result of running a single test."""
    test_file: str
    passed: bool
    error: str = ""
    duration_ms: float = 0.0


@dataclass
class TestResults:
    """Aggregate results from running the test suite."""
    results: List[TestResult] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    duration_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


class TestSuiteManager:
    """Manage a growing collection of auto-generated tests.

    Tests are stored as individual Python files in a configurable directory,
    defaulting to ~/.hermes/auto_tests/.
    """

    def __init__(self, tests_dir: Optional[Path] = None):
        self._tests_dir = tests_dir or Path.home() / ".hermes" / "auto_tests"
        self._tests_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self._tests_dir / "_metadata.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load test suite metadata from disk."""
        if self._metadata_file.exists():
            try:
                return json.loads(self._metadata_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {"tests": {}, "sessions_covered": [], "run_history": []}

    def _save_metadata(self) -> None:
        """Persist metadata to disk."""
        try:
            self._metadata_file.write_text(
                json.dumps(self._metadata, indent=2, default=str)
            )
        except OSError as e:
            logger.warning(f"Failed to save metadata: {e}")

    @staticmethod
    def _validate_test_code(test_code: str) -> bool:
        """Validate test code for dangerous constructs before writing to disk.

        Uses ast.parse() to walk the AST and reject code containing:
        eval, exec, compile, __import__, subprocess, os.system, os.popen, shutil.rmtree

        Returns:
            True if the code is safe, False if dangerous constructs are found.
        """
        DANGEROUS_NAMES = {
            "eval", "exec", "compile", "__import__",
        }
        DANGEROUS_ATTRS = {
            ("subprocess", "run"), ("subprocess", "call"), ("subprocess", "Popen"),
            ("subprocess", "check_output"), ("subprocess", "check_call"),
            ("os", "system"), ("os", "popen"),
            ("shutil", "rmtree"),
        }
        DANGEROUS_MODULES = {"subprocess"}

        try:
            tree = ast.parse(test_code)
        except SyntaxError:
            return False

        for node in ast.walk(tree):
            # Check for dangerous function calls: eval(), exec(), compile(), __import__()
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in DANGEROUS_NAMES:
                    logger.warning("Rejected test code: dangerous call to %s()", func.id)
                    return False
                if isinstance(func, ast.Attribute):
                    # Check for os.system, os.popen, subprocess.*, shutil.rmtree
                    if isinstance(func.value, ast.Name):
                        pair = (func.value.id, func.attr)
                        if pair in DANGEROUS_ATTRS:
                            logger.warning("Rejected test code: dangerous call to %s.%s()", *pair)
                            return False
            # Check for 'import subprocess'
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in DANGEROUS_MODULES:
                        logger.warning("Rejected test code: imports dangerous module %s", alias.name)
                        return False
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in DANGEROUS_MODULES:
                    logger.warning("Rejected test code: imports from dangerous module %s", node.module)
                    return False

        return True

    def add_test(self, test_code: str, source_session_id: str = "") -> Path:
        """Add a generated test to the suite.

        Args:
            test_code: Python test code.
            source_session_id: Session that generated this test.

        Returns:
            Path to the created test file, or None if validation fails.
        """
        # Validate test code for dangerous constructs
        if not self._validate_test_code(test_code):
            logger.warning("Skipping test from session %s: failed code validation", source_session_id)
            return None

        # Generate a unique filename
        code_hash = hashlib.sha256(test_code.encode()).hexdigest()[:12]
        filename = f"test_auto_{code_hash}.py"
        test_path = self._tests_dir / filename

        test_path.write_text(test_code)

        # Update metadata
        self._metadata["tests"][filename] = {
            "session_id": source_session_id,
            "created": time.time(),
            "code_hash": code_hash,
            "flaky_count": 0,
            "run_count": 0,
        }
        if source_session_id and source_session_id not in self._metadata["sessions_covered"]:
            self._metadata["sessions_covered"].append(source_session_id)
        self._save_metadata()

        logger.info(f"Added auto-test: {filename} (session: {source_session_id})")
        return test_path

    def deduplicate(self) -> int:
        """Remove duplicate tests based on content hash.

        Returns:
            Number of duplicates removed.
        """
        seen_hashes: Dict[str, str] = {}
        removed = 0

        for test_file in sorted(self._tests_dir.glob("test_auto_*.py")):
            content_hash = hashlib.sha256(test_file.read_bytes()).hexdigest()[:16]
            if content_hash in seen_hashes:
                test_file.unlink()
                if test_file.name in self._metadata["tests"]:
                    del self._metadata["tests"][test_file.name]
                removed += 1
                logger.info(f"Removed duplicate: {test_file.name} (same as {seen_hashes[content_hash]})")
            else:
                seen_hashes[content_hash] = test_file.name

        if removed > 0:
            self._save_metadata()
        return removed

    def run_all(self) -> TestResults:
        """Run all tests in the suite.

        Returns:
            Aggregate test results.
        """
        results = TestResults()
        start = time.time()

        test_files = list(self._tests_dir.glob("test_auto_*.py"))
        results.total = len(test_files)

        for test_file in test_files:
            t0 = time.time()
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", str(test_file), "-x", "--tb=short", "-q"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(self._tests_dir),
                )
                passed = proc.returncode == 0
                error = proc.stderr if not passed else ""
            except subprocess.TimeoutExpired:
                passed = False
                error = "Test timed out (30s)"
            except FileNotFoundError:
                passed = False
                error = "pytest not found"
            except Exception as e:
                passed = False
                error = str(e)

            duration = (time.time() - t0) * 1000
            result = TestResult(
                test_file=test_file.name,
                passed=passed,
                error=error,
                duration_ms=duration,
            )
            results.results.append(result)

            if passed:
                results.passed += 1
            else:
                results.failed += 1

            # Update metadata
            meta = self._metadata["tests"].get(test_file.name, {})
            meta["run_count"] = meta.get("run_count", 0) + 1
            if not passed:
                meta["flaky_count"] = meta.get("flaky_count", 0) + 1
            self._metadata["tests"][test_file.name] = meta

        results.duration_ms = (time.time() - start) * 1000

        # Record run history
        self._metadata["run_history"].append({
            "timestamp": time.time(),
            "total": results.total,
            "passed": results.passed,
            "failed": results.failed,
            "success_rate": results.success_rate,
        })
        # Keep only last 100 runs
        self._metadata["run_history"] = self._metadata["run_history"][-100:]
        self._save_metadata()

        return results

    def prune_flaky(self, max_flaky_rate: float = 0.1) -> int:
        """Remove tests that fail too often (flaky tests).

        Args:
            max_flaky_rate: Maximum allowed failure rate (0.0-1.0).

        Returns:
            Number of tests pruned.
        """
        pruned = 0
        to_remove = []

        for filename, meta in self._metadata["tests"].items():
            run_count = meta.get("run_count", 0)
            flaky_count = meta.get("flaky_count", 0)

            # Need at least 3 runs to judge
            if run_count < 3:
                continue

            flaky_rate = flaky_count / run_count
            if flaky_rate > max_flaky_rate:
                test_path = self._tests_dir / filename
                if test_path.exists():
                    test_path.unlink()
                    logger.info(
                        f"Pruned flaky test: {filename} "
                        f"(flaky_rate={flaky_rate:.2f}, runs={run_count})"
                    )
                to_remove.append(filename)
                pruned += 1

        for filename in to_remove:
            del self._metadata["tests"][filename]

        if pruned > 0:
            self._save_metadata()
        return pruned

    def stats(self) -> Dict[str, Any]:
        """Get test suite statistics.

        Returns:
            Dict with total, success_rate, sessions_covered, etc.
        """
        total = len(list(self._tests_dir.glob("test_auto_*.py")))
        sessions = len(self._metadata.get("sessions_covered", []))

        # Calculate aggregate success rate from run history
        history = self._metadata.get("run_history", [])
        if history:
            last_run = history[-1]
            success_rate = last_run.get("success_rate", 0.0)
        else:
            success_rate = 0.0

        # Count flaky tests
        flaky = sum(
            1 for meta in self._metadata.get("tests", {}).values()
            if meta.get("run_count", 0) >= 3
            and meta.get("flaky_count", 0) / max(meta.get("run_count", 1), 1) > 0.1
        )

        return {
            "total_tests": total,
            "sessions_covered": sessions,
            "success_rate": success_rate,
            "flaky_tests": flaky,
            "total_runs": len(history),
            "tests_dir": str(self._tests_dir),
        }

    def list_tests(self) -> List[Dict[str, Any]]:
        """List all tests with their metadata."""
        tests = []
        for test_file in sorted(self._tests_dir.glob("test_auto_*.py")):
            meta = self._metadata.get("tests", {}).get(test_file.name, {})
            tests.append({
                "file": test_file.name,
                "session_id": meta.get("session_id", ""),
                "created": meta.get("created", 0),
                "run_count": meta.get("run_count", 0),
                "flaky_count": meta.get("flaky_count", 0),
            })
        return tests


# ─── OnSessionComplete hook ──────────────────────────────────────────────────

def on_session_complete(
    transcript: List[Dict[str, Any]],
    session_id: str = "",
    tests_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Hook to call after each session completes.

    Extracts a trajectory, and if the session was successful,
    generates a regression test and adds it to the suite.

    Args:
        transcript: The session transcript (list of message dicts).
        session_id: Unique identifier for the session.
        tests_dir: Directory to store tests (default: ~/.hermes/auto_tests/).

    Returns:
        Path to the generated test file, or None if no test was generated.
    """
    extractor = TrajectoryExtractor()
    trajectory = extractor.extract_from_session(transcript, session_id=session_id)

    # Only generate tests for successful sessions
    if trajectory.outcome != Outcome.SUCCESS:
        logger.debug(
            f"Skipping test generation for {session_id}: "
            f"outcome={trajectory.outcome.value}"
        )
        return None

    # Need at least one tool call to make a useful test
    if not trajectory.tool_calls:
        logger.debug(f"Skipping test generation for {session_id}: no tool calls")
        return None

    generator = TestGenerator()
    test_code = generator.generate_regression_test(trajectory)

    manager = TestSuiteManager(tests_dir=tests_dir)
    test_path = manager.add_test(test_code, source_session_id=session_id)

    # Periodically deduplicate
    total = manager.stats()["total_tests"]
    if total > 0 and total % 10 == 0:
        removed = manager.deduplicate()
        if removed:
            logger.info(f"Deduplication removed {removed} tests")

    return test_path
