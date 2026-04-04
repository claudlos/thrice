"""
Error Recovery — Classify errors by type and apply specialized recovery prompts.

Extends the structured_errors.py module with higher-level recovery logic:
  - ErrorCategory enum for broad error classification
  - ErrorClassifier with regex + context-aware classification
  - RecoveryEngine with graduated, per-category recovery strategies
  - RetryTracker to detect repeated errors and forward progress
  - is_same_error() fuzzy matcher for error deduplication

Integration point: agent loop retry logic

Usage:
    from new_files.error_recovery import (
        ErrorClassifier, RecoveryEngine, RetryTracker, is_same_error,
    )

    classifier = ErrorClassifier()
    engine = RecoveryEngine()
    tracker = RetryTracker()

    category = classifier.classify(error_text)
    details = classifier.extract_details(error_text)
    strategy = engine.get_strategy(category, attempt=tracker.attempt_count("task-1"))
    prompt = engine.generate_recovery_prompt(error_text, category, attempt=1)
    tracker.record_attempt("task-1", error_text)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ─── Error Category Enum ──────────────────────────────────────────────────────


class ErrorCategory(str, Enum):
    """Broad categories for error classification."""
    SYNTAX = "syntax"
    IMPORT = "import"
    TYPE_ERROR = "type_error"
    ASSERTION = "assertion"
    RUNTIME = "runtime"
    TIMEOUT = "timeout"
    PERMISSION = "permission"
    FILE_NOT_FOUND = "file_not_found"
    NETWORK = "network"
    API_ERROR = "api_error"
    TOOL_ERROR = "tool_error"
    LOGIC = "logic"
    UNKNOWN = "unknown"


# ─── Recovery Strategy Dataclass ──────────────────────────────────────────────


@dataclass
class RecoveryStrategy:
    """A recovery plan for a classified error."""
    strategy_name: str
    prompt_injection: str
    recommended_tools: List[str]
    max_retries: int
    should_reread_file: bool
    should_search_codebase: bool
    confidence: float  # 0.0 – 1.0


# ─── Classification Patterns ─────────────────────────────────────────────────
# (regex, ErrorCategory) — first match wins

_CATEGORY_PATTERNS: list[tuple[re.Pattern, ErrorCategory]] = [
    # Syntax
    (re.compile(
        r"SyntaxError|IndentationError|TabError|unexpected EOF|"
        r"invalid syntax|unterminated string|unexpected indent",
        re.I,
    ), ErrorCategory.SYNTAX),
    # Import
    (re.compile(
        r"ImportError|ModuleNotFoundError|No module named|cannot import name",
        re.I,
    ), ErrorCategory.IMPORT),
    # Type error
    (re.compile(
        r"TypeError|unsupported operand type|"
        r"not subscriptable|has no attribute.*type|argument.*type",
        re.I,
    ), ErrorCategory.TYPE_ERROR),
    # Assertion
    (re.compile(
        r"AssertionError|assert .* ==|assertion failed",
        re.I,
    ), ErrorCategory.ASSERTION),
    # Timeout
    (re.compile(
        r"TimeoutError|timed out|timeout|deadline exceeded|ETIMEDOUT",
        re.I,
    ), ErrorCategory.TIMEOUT),
    # Permission
    (re.compile(
        r"PermissionError|permission denied|EACCES|not writable|read.only filesystem",
        re.I,
    ), ErrorCategory.PERMISSION),
    # File not found
    (re.compile(
        r"FileNotFoundError|No such file|ENOENT|file not found|"
        r"path.*does not exist|cannot find.*file",
        re.I,
    ), ErrorCategory.FILE_NOT_FOUND),
    # Network
    (re.compile(
        r"ConnectionError|ConnectionRefusedError|ECONNREFUSED|"
        r"socket\.error|network.*unreachable|DNS.*fail|ECONNRESET|"
        r"BrokenPipeError|RemoteDisconnected",
        re.I,
    ), ErrorCategory.NETWORK),
    # API error
    (re.compile(
        r"HTTPError|status.code.(?:4\d{2}|5\d{2})|API.*error|"
        r"rate.limit|\b429\b|\b401\b|\b403\b|\b500\b|\b502\b|\b503\b",
        re.I,
    ), ErrorCategory.API_ERROR),
    # Tool error (Hermes-specific)
    (re.compile(
        r"tool.*error|tool.*fail|patch.*fail|unique match not found|"
        r"old_string.*not found|no match found",
        re.I,
    ), ErrorCategory.TOOL_ERROR),
    # Runtime (broad catch)
    (re.compile(
        r"RuntimeError|ZeroDivisionError|KeyError|IndexError|"
        r"ValueError|AttributeError|NameError|UnboundLocalError|"
        r"StopIteration|RecursionError|OverflowError|OSError",
        re.I,
    ), ErrorCategory.RUNTIME),
    # Logic (test failures, wrong output)
    (re.compile(
        r"expected.*but got|test.*fail|incorrect.*result|"
        r"wrong.*output|mismatch|does not match expected|"
        r"expected.*got",
        re.I,
    ), ErrorCategory.LOGIC),
]

# ─── Detail extraction patterns ──────────────────────────────────────────────

_DETAIL_PATTERNS: dict[str, re.Pattern] = {
    "file": re.compile(
        r'(?:File\s+["\'])([^"\']+)["\']'
        r"|"
        r"(?:in\s+file\s+)(\S+)"
        r"|"
        r"(\S+\.py)(?::\d+)",
    ),
    "line": re.compile(
        r"line\s+(\d+)"
        r"|"
        r":(\d+):",
    ),
    "function": re.compile(
        r"in\s+(\w+)\s*$"
        r"|"
        r"in\s+function\s+['\"]?(\w+)",
        re.M,
    ),
    "variable": re.compile(
        r"name\s+['\"](\w+)['\"].*is not defined"
        r"|"
        r"(?:NameError|KeyError):\s+['\"](\w+)"
        r"|"
        r"has no attribute\s+['\"](\w+)",
    ),
    "expected_type": re.compile(
        r"expected\s+(\w+)"
        r"|"
        r"must be\s+(\w+)",
        re.I,
    ),
    "actual_type": re.compile(
        r"got\s+(\w+)"
        r"|"
        r"not\s+['\"]?(\w+)['\"]?$",
        re.I | re.M,
    ),
    "module_name": re.compile(
        r"No module named\s+['\"]?([.\w]+)['\"]?"
        r"|"
        r"cannot import name\s+['\"]?(\w+)['\"]?\s+from\s+['\"]?([.\w]+)",
        re.I,
    ),
}


# ─── Fuzzy Error Matching ─────────────────────────────────────────────────────

# Patterns stripped for comparison
_NOISE_PATTERNS = re.compile(
    r"line\s+\d+"
    r"|0x[0-9a-fA-F]+"
    r"|\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.\d]*Z?"
    r"|pid\s*=?\s*\d+"
    r"|\b\d{10,}\b"  # long numbers (timestamps, ids)
    r"|#\d+"          # issue/id numbers
)


def is_same_error(err1: str, err2: str) -> bool:
    """Fuzzy-match two error strings, ignoring line numbers, timestamps, pids.

    Returns True if the errors are likely the same underlying issue.
    """
    if err1 == err2:
        return True

    def _normalize(text: str) -> str:
        text = _NOISE_PATTERNS.sub("", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    n1 = _normalize(err1)
    n2 = _normalize(err2)

    if n1 == n2:
        return True

    # Check if one is a substring of the other (>80% overlap)
    if not n1 or not n2:
        return False
    shorter, longer = (n1, n2) if len(n1) <= len(n2) else (n2, n1)
    if shorter in longer:
        return True

    # Token-based similarity
    tokens1 = set(n1.split())
    tokens2 = set(n2.split())
    if not tokens1 or not tokens2:
        return False
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    jaccard = len(intersection) / len(union)
    return jaccard >= 0.75


# ─── Error Classifier ─────────────────────────────────────────────────────────


class ErrorClassifier:
    """Classifies error text into ErrorCategory with detail extraction."""

    def classify(self, error_text: str) -> ErrorCategory:
        """Classify an error string into an ErrorCategory using regex patterns.

        Args:
            error_text: Raw error output (traceback, message, etc.)

        Returns:
            The best-matching ErrorCategory, or UNKNOWN.
        """
        if not error_text or not error_text.strip():
            return ErrorCategory.UNKNOWN

        for pattern, category in _CATEGORY_PATTERNS:
            if pattern.search(error_text):
                return category

        return ErrorCategory.UNKNOWN

    def extract_details(self, error_text: str) -> dict:
        """Extract structured details from an error string.

        Returns dict with keys: file, line, function, variable,
        expected_type, actual_type, module_name.
        Missing values are None.
        """
        details: dict = {}
        for key, pattern in _DETAIL_PATTERNS.items():
            match = pattern.search(error_text)
            if match:
                # Take the first non-None group
                groups = match.groups()
                value = next((g for g in groups if g is not None), None)
                details[key] = value
            else:
                details[key] = None

        # Special handling: module_name from "cannot import name X from Y"
        if details.get("module_name") is None:
            m = _DETAIL_PATTERNS["module_name"].search(error_text)
            if m and m.group(3):
                details["module_name"] = m.group(3)

        return details

    def classify_with_context(
        self,
        error_text: str,
        recent_tool_calls: List[dict],
    ) -> ErrorCategory:
        """Classify using both error text and recent tool call context.

        If the base classification is ambiguous (UNKNOWN or RUNTIME),
        tool call context can disambiguate. For example, if the last tool
        was 'patch' and we got an error, it's likely TOOL_ERROR.

        Args:
            error_text: Raw error output.
            recent_tool_calls: List of dicts with at least 'tool' key,
                optionally 'result', 'success'.

        Returns:
            Refined ErrorCategory.
        """
        base = self.classify(error_text)

        if not recent_tool_calls:
            return base

        last_call = recent_tool_calls[-1] if recent_tool_calls else {}
        last_tool = last_call.get("tool", "")

        # If base classification is strong, keep it
        if base not in (ErrorCategory.UNKNOWN, ErrorCategory.RUNTIME):
            return base

        # Multiple failed tool calls suggest tool error pattern (check first)
        recent_failures = sum(
            1 for tc in recent_tool_calls
            if not tc.get("success", True)
        )
        if recent_failures >= 2 and base in (ErrorCategory.UNKNOWN, ErrorCategory.RUNTIME):
            return ErrorCategory.TOOL_ERROR

        # Context-based refinement
        if last_tool in ("patch", "write_file"):
            # Errors after file edits are often syntax or tool errors
            if any(kw in error_text.lower() for kw in ("syntax", "indent")):
                return ErrorCategory.SYNTAX
            if any(kw in error_text.lower() for kw in ("match", "not found")):
                return ErrorCategory.TOOL_ERROR
            return ErrorCategory.TOOL_ERROR if base == ErrorCategory.UNKNOWN else base

        if last_tool == "terminal":
            # Terminal errors could be anything; keep base unless UNKNOWN
            if base == ErrorCategory.UNKNOWN:
                if "command not found" in error_text.lower():
                    return ErrorCategory.RUNTIME
                if "exit code" in error_text.lower():
                    return ErrorCategory.RUNTIME
            return base

        if last_tool in ("read_file", "search_files"):
            if base == ErrorCategory.UNKNOWN:
                return ErrorCategory.FILE_NOT_FOUND

        return base


# ─── Recovery Engine ──────────────────────────────────────────────────────────

# Base strategies per category (attempt 1)
_BASE_STRATEGIES: dict[ErrorCategory, RecoveryStrategy] = {
    ErrorCategory.SYNTAX: RecoveryStrategy(
        strategy_name="fix_syntax",
        prompt_injection=(
            "The code has a syntax error. Read the file around the error line, "
            "identify the exact syntax issue (missing colon, bracket, indent), "
            "and apply a targeted fix."
        ),
        recommended_tools=["read_file", "patch"],
        max_retries=3,
        should_reread_file=True,
        should_search_codebase=False,
        confidence=0.9,
    ),
    ErrorCategory.IMPORT: RecoveryStrategy(
        strategy_name="resolve_import",
        prompt_injection=(
            "An import failed. Search the codebase for the module/package name "
            "to verify it exists and check the correct import path. "
            "If it's an external dependency, check if it needs to be installed."
        ),
        recommended_tools=["search_files", "terminal"],
        max_retries=3,
        should_reread_file=False,
        should_search_codebase=True,
        confidence=0.85,
    ),
    ErrorCategory.TYPE_ERROR: RecoveryStrategy(
        strategy_name="fix_type",
        prompt_injection=(
            "A type error occurred. Read the file to check the variable types, "
            "function signatures, and expected vs actual types. "
            "Apply appropriate type conversions or fix the call signature."
        ),
        recommended_tools=["read_file", "search_files", "patch"],
        max_retries=3,
        should_reread_file=True,
        should_search_codebase=True,
        confidence=0.8,
    ),
    ErrorCategory.ASSERTION: RecoveryStrategy(
        strategy_name="fix_assertion",
        prompt_injection=(
            "An assertion or test failed. Read the test to understand the expected "
            "behavior, then read the implementation to find the discrepancy. "
            "Fix the implementation (not the test) unless the test is wrong."
        ),
        recommended_tools=["read_file", "search_files", "patch"],
        max_retries=4,
        should_reread_file=True,
        should_search_codebase=True,
        confidence=0.7,
    ),
    ErrorCategory.RUNTIME: RecoveryStrategy(
        strategy_name="fix_runtime",
        prompt_injection=(
            "A runtime error occurred. Read the traceback carefully, identify "
            "the root cause (wrong variable, missing key, index out of bounds), "
            "and fix the code at the point of failure."
        ),
        recommended_tools=["read_file", "patch"],
        max_retries=3,
        should_reread_file=True,
        should_search_codebase=False,
        confidence=0.75,
    ),
    ErrorCategory.TIMEOUT: RecoveryStrategy(
        strategy_name="handle_timeout",
        prompt_injection=(
            "The operation timed out. Consider increasing the timeout, "
            "breaking the operation into smaller steps, or running it "
            "in the background."
        ),
        recommended_tools=["terminal"],
        max_retries=2,
        should_reread_file=False,
        should_search_codebase=False,
        confidence=0.6,
    ),
    ErrorCategory.PERMISSION: RecoveryStrategy(
        strategy_name="handle_permission",
        prompt_injection=(
            "Permission denied. Check file ownership and permissions. "
            "Try writing to a different location or adjusting permissions."
        ),
        recommended_tools=["terminal"],
        max_retries=2,
        should_reread_file=False,
        should_search_codebase=False,
        confidence=0.5,
    ),
    ErrorCategory.FILE_NOT_FOUND: RecoveryStrategy(
        strategy_name="find_file",
        prompt_injection=(
            "File not found. Search the codebase for the correct file path "
            "using search_files with target='files'. Check for typos in the "
            "filename and verify the working directory."
        ),
        recommended_tools=["search_files", "terminal"],
        max_retries=3,
        should_reread_file=False,
        should_search_codebase=True,
        confidence=0.9,
    ),
    ErrorCategory.NETWORK: RecoveryStrategy(
        strategy_name="handle_network",
        prompt_injection=(
            "Network error. Verify the URL/host is correct. Check if the "
            "service is running. Consider retrying after a brief wait."
        ),
        recommended_tools=["terminal"],
        max_retries=2,
        should_reread_file=False,
        should_search_codebase=False,
        confidence=0.5,
    ),
    ErrorCategory.API_ERROR: RecoveryStrategy(
        strategy_name="handle_api",
        prompt_injection=(
            "API error. Check the status code: 401/403 = auth issue, "
            "429 = rate limited (wait and retry), 500+ = server error (retry). "
            "Verify API parameters and authentication."
        ),
        recommended_tools=["terminal"],
        max_retries=3,
        should_reread_file=False,
        should_search_codebase=False,
        confidence=0.6,
    ),
    ErrorCategory.TOOL_ERROR: RecoveryStrategy(
        strategy_name="fix_tool_usage",
        prompt_injection=(
            "Tool usage error. Re-read the target file to see its CURRENT "
            "content before retrying. Ensure old_string matches exactly "
            "(including whitespace). Use search_files if unsure of the path."
        ),
        recommended_tools=["read_file", "search_files"],
        max_retries=3,
        should_reread_file=True,
        should_search_codebase=True,
        confidence=0.85,
    ),
    ErrorCategory.LOGIC: RecoveryStrategy(
        strategy_name="fix_logic",
        prompt_injection=(
            "Logic error — the code runs but produces wrong results. "
            "Re-read the requirements, trace the logic step by step, "
            "and identify where the output diverges from expectations."
        ),
        recommended_tools=["read_file", "search_files", "patch"],
        max_retries=4,
        should_reread_file=True,
        should_search_codebase=True,
        confidence=0.6,
    ),
    ErrorCategory.UNKNOWN: RecoveryStrategy(
        strategy_name="generic_recovery",
        prompt_injection=(
            "An unclassified error occurred. Read the full error message, "
            "search the codebase for relevant context, and try a different "
            "approach if the same error repeats."
        ),
        recommended_tools=["read_file", "search_files", "terminal"],
        max_retries=3,
        should_reread_file=True,
        should_search_codebase=True,
        confidence=0.4,
    ),
}


class RecoveryEngine:
    """Generates graduated recovery strategies for classified errors."""

    def __init__(self) -> None:
        self._strategies = dict(_BASE_STRATEGIES)

    def get_strategy(
        self,
        category: ErrorCategory,
        attempt: int = 1,
    ) -> RecoveryStrategy:
        """Get a recovery strategy, graduated by attempt number.

        Attempt 1: targeted fix (base strategy)
        Attempt 2: broader context gathering
        Attempt 3: re-plan the approach
        Attempt 4+: fundamentally different approach

        Args:
            category: The classified error category.
            attempt: Which retry attempt (1-indexed).

        Returns:
            A RecoveryStrategy appropriate for this attempt.
        """
        base = self._strategies.get(category, self._strategies[ErrorCategory.UNKNOWN])

        if attempt <= 1:
            return base

        if attempt == 2:
            return RecoveryStrategy(
                strategy_name=f"{base.strategy_name}_broader",
                prompt_injection=(
                    f"Previous fix attempt failed. Gather broader context: "
                    f"read surrounding code, search for related definitions, "
                    f"and understand the full picture before retrying. "
                    f"Original guidance: {base.prompt_injection}"
                ),
                recommended_tools=["read_file", "search_files"] + base.recommended_tools,
                max_retries=base.max_retries,
                should_reread_file=True,
                should_search_codebase=True,
                confidence=max(0.0, base.confidence - 0.15),
            )

        if attempt == 3:
            return RecoveryStrategy(
                strategy_name=f"{base.strategy_name}_replan",
                prompt_injection=(
                    f"Two fix attempts failed. STOP and re-plan: "
                    f"1) Re-read all relevant files from scratch. "
                    f"2) List what you know and what you're unsure about. "
                    f"3) Form a new hypothesis about the root cause. "
                    f"4) Try a different fix strategy. "
                    f"Original error type: {category.value}"
                ),
                recommended_tools=["read_file", "search_files", "terminal"],
                max_retries=base.max_retries,
                should_reread_file=True,
                should_search_codebase=True,
                confidence=max(0.0, base.confidence - 0.3),
            )

        # attempt >= 4
        return RecoveryStrategy(
            strategy_name=f"{base.strategy_name}_alternative",
            prompt_injection=(
                f"Multiple fix attempts have failed ({attempt} so far). "
                f"Take a fundamentally different approach: "
                f"1) Consider if the original plan is flawed. "
                f"2) Look for alternative implementations or workarounds. "
                f"3) Check if the problem is elsewhere (wrong file, wrong function). "
                f"4) Consider reverting recent changes and starting fresh. "
                f"Original error type: {category.value}"
            ),
            recommended_tools=["read_file", "search_files", "terminal", "patch"],
            max_retries=base.max_retries,
            should_reread_file=True,
            should_search_codebase=True,
            confidence=max(0.0, base.confidence - 0.4),
        )

    def generate_recovery_prompt(
        self,
        error_text: str,
        category: ErrorCategory,
        attempt: int = 1,
        context: Optional[dict] = None,
    ) -> str:
        """Generate a full recovery prompt combining strategy + error details.

        Args:
            error_text: The raw error text.
            category: Classified error category.
            attempt: Retry attempt number.
            context: Optional dict with extra context (file, function, etc.)

        Returns:
            A prompt string to inject into the conversation.
        """
        strategy = self.get_strategy(category, attempt)
        classifier = ErrorClassifier()
        details = classifier.extract_details(error_text)

        parts = [
            f"[ERROR RECOVERY — attempt {attempt}]",
            f"Error type: {category.value}",
            f"Strategy: {strategy.strategy_name}",
            "",
            strategy.prompt_injection,
        ]

        # Add extracted details
        detail_parts = []
        if details.get("file"):
            detail_parts.append(f"  File: {details['file']}")
        if details.get("line"):
            detail_parts.append(f"  Line: {details['line']}")
        if details.get("function"):
            detail_parts.append(f"  Function: {details['function']}")
        if details.get("variable"):
            detail_parts.append(f"  Variable: {details['variable']}")
        if details.get("module_name"):
            detail_parts.append(f"  Module: {details['module_name']}")

        if detail_parts:
            parts.append("")
            parts.append("Extracted details:")
            parts.extend(detail_parts)

        # Add context
        if context:
            ctx_parts = []
            for k, v in context.items():
                if v is not None:
                    ctx_parts.append(f"  {k}: {v}")
            if ctx_parts:
                parts.append("")
                parts.append("Additional context:")
                parts.extend(ctx_parts)

        # Recommended actions
        parts.append("")
        parts.append(f"Recommended tools: {', '.join(strategy.recommended_tools)}")
        if strategy.should_reread_file:
            parts.append("→ Re-read the target file before making changes.")
        if strategy.should_search_codebase:
            parts.append("→ Search the codebase for related code/definitions.")

        return "\n".join(parts)


# ─── Retry Tracker ────────────────────────────────────────────────────────────


@dataclass
class _AttemptRecord:
    """Internal record of an error attempt."""
    error_text: str
    category: ErrorCategory
    attempt_num: int


class RetryTracker:
    """Tracks retry attempts per task/error, detecting stalls and progress."""

    def __init__(self) -> None:
        self._attempts: Dict[str, List[_AttemptRecord]] = {}

    def record_attempt(
        self,
        task_id: str,
        error_text: str,
        category: Optional[ErrorCategory] = None,
    ) -> int:
        """Record an error attempt for a task.

        Args:
            task_id: Identifier for the task/operation.
            error_text: The error that occurred.
            category: Optionally pre-classified category.

        Returns:
            The current attempt number (1-indexed).
        """
        if category is None:
            category = ErrorClassifier().classify(error_text)

        if task_id not in self._attempts:
            self._attempts[task_id] = []

        attempt_num = len(self._attempts[task_id]) + 1
        self._attempts[task_id].append(_AttemptRecord(
            error_text=error_text,
            category=category,
            attempt_num=attempt_num,
        ))
        return attempt_num

    def attempt_count(self, task_id: str) -> int:
        """Return the number of attempts for a task."""
        return len(self._attempts.get(task_id, []))

    def is_repeating(self, task_id: str) -> bool:
        """Check if the last two errors are the same (stalled).

        Returns True if the task is stuck on the same error.
        """
        attempts = self._attempts.get(task_id, [])
        if len(attempts) < 2:
            return False
        return is_same_error(
            attempts[-1].error_text,
            attempts[-2].error_text,
        )

    def is_progressing(self, task_id: str) -> bool:
        """Check if the error is changing (making progress).

        Returns True if the latest error is different from the previous one,
        suggesting forward movement even though errors persist.
        """
        attempts = self._attempts.get(task_id, [])
        if len(attempts) < 2:
            return True  # First attempt = progressing by default
        return not is_same_error(
            attempts[-1].error_text,
            attempts[-2].error_text,
        )

    def should_change_strategy(self, task_id: str) -> bool:
        """Check if we should escalate to a different strategy.

        True if same error repeated 2+ times in a row.
        """
        attempts = self._attempts.get(task_id, [])
        if len(attempts) < 2:
            return False

        # Count consecutive same errors from the end
        consecutive = 1
        for i in range(len(attempts) - 1, 0, -1):
            if is_same_error(attempts[i].error_text, attempts[i - 1].error_text):
                consecutive += 1
            else:
                break
        return consecutive >= 2

    def get_history(self, task_id: str) -> List[dict]:
        """Return attempt history for a task."""
        return [
            {
                "attempt": rec.attempt_num,
                "category": rec.category.value,
                "error_preview": rec.error_text[:100],
            }
            for rec in self._attempts.get(task_id, [])
        ]

    def clear(self, task_id: Optional[str] = None) -> None:
        """Clear tracking data. If task_id given, clear only that task."""
        if task_id:
            self._attempts.pop(task_id, None)
        else:
            self._attempts.clear()
