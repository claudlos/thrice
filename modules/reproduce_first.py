"""
Reproduce-First Debugging Workflow — Confirm bugs before fixing them.

Implements a structured debugging workflow that prioritizes reproducing the
bug before attempting any fix. This reduces wasted effort on misdiagnosed
issues and ensures fixes are verifiable.

Components:
  - BugType enum for classifying detected bugs
  - ReproductionResult dataclass tracking reproduction outcomes
  - FileLocation dataclass for traceback file/line extraction
  - BugAnalyzer class for parsing errors and classifying bug types
  - ReproduceFirstWorkflow class orchestrating the full workflow
  - detect_bug_report() for heuristic bug detection in user messages
  - generate_debug_prompt() for creating structured LLM debugging prompts

Integration point: agent/run_agent.py or prompt_builder.py

Usage:
    from new_files.reproduce_first import (
        ReproduceFirstWorkflow, BugAnalyzer, detect_bug_report,
        generate_debug_prompt,
    )

    workflow = ReproduceFirstWorkflow()
    if workflow.detect_bug(user_message):
        reproduction = workflow.generate_reproduction(user_message)
        analysis = workflow.analyze_error(user_message)
        approach = workflow.suggest_approach(analysis)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ─── Bug Type Enum ─────────────────────────────────────────────────────────


class BugType(str, Enum):
    """Classification of detected bug types."""
    SYNTAX = "syntax"
    IMPORT = "import"
    TYPE = "type"
    LOGIC = "logic"
    RUNTIME = "runtime"
    TIMEOUT = "timeout"
    ASSERTION = "assertion"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    UNKNOWN = "unknown"


# ─── Dataclasses ───────────────────────────────────────────────────────────


@dataclass
class FileLocation:
    """A file and line number extracted from a traceback."""
    filepath: str
    line_number: int
    function_name: Optional[str] = None
    code_line: Optional[str] = None

    def __str__(self) -> str:
        parts = [f"{self.filepath}:{self.line_number}"]
        if self.function_name:
            parts.append(f" in {self.function_name}")
        return "".join(parts)


@dataclass
class ReproductionResult:
    """Tracks the outcome of a bug reproduction attempt."""
    bug_confirmed: bool
    reproduction_script: str
    error_output: str
    suggested_fix_approach: str
    bug_type: BugType = BugType.UNKNOWN
    file_locations: List[FileLocation] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary of the reproduction result."""
        status = "CONFIRMED" if self.bug_confirmed else "NOT CONFIRMED"
        lines = [
            f"Bug Status: {status}",
            f"Bug Type: {self.bug_type.value}",
            f"Error Output: {self.error_output[:200]}{'...' if len(self.error_output) > 200 else ''}",
        ]
        if self.file_locations:
            lines.append("Locations:")
            for loc in self.file_locations:
                lines.append(f"  - {loc}")
        if self.suggested_fix_approach:
            lines.append(f"Suggested Approach: {self.suggested_fix_approach}")
        return "\n".join(lines)


# ─── Bug Detection Patterns ───────────────────────────────────────────────

# Keywords that suggest a bug report
_BUG_KEYWORDS = [
    "error", "bug", "broken", "fails", "crash", "crashed", "crashing",
    "exception", "traceback", "failing", "failure", "doesn't work",
    "does not work", "not working", "won't work", "stopped working",
    "unexpected", "wrong output", "incorrect", "segfault", "panic",
]

# Patterns that strongly indicate a bug report
_BUG_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"^\w*Error:\s+.+", re.MULTILINE),
    re.compile(r"^\w*Exception:\s+.+", re.MULTILINE),
    re.compile(r"File\s+\"[^\"]+\",\s+line\s+\d+", re.MULTILINE),
    re.compile(r"FAILED\s+[\w/.:]+", re.IGNORECASE),
    re.compile(r"FAIL:\s+test_\w+", re.IGNORECASE),
    re.compile(r"\d+\s+(?:failed|errors?)\b", re.IGNORECASE),
    re.compile(r"exit\s+code\s+[1-9]\d*", re.IGNORECASE),
    re.compile(r"core\s+dump", re.IGNORECASE),
    re.compile(r"stack\s*trace", re.IGNORECASE),
]

# Traceback file/line extraction
_TRACEBACK_FILE_LINE = re.compile(
    r'File\s+"([^"]+)",\s+line\s+(\d+)(?:,\s+in\s+(\w+))?'
)

# Error type classification patterns
_ERROR_TYPE_PATTERNS: List[Tuple[re.Pattern, BugType]] = [
    (re.compile(r"SyntaxError|IndentationError|TabError", re.IGNORECASE), BugType.SYNTAX),
    (re.compile(r"ImportError|ModuleNotFoundError|No module named", re.IGNORECASE), BugType.IMPORT),
    (re.compile(r"TypeError|AttributeError|'[\w]+' object has no attribute", re.IGNORECASE), BugType.TYPE),
    (re.compile(r"AssertionError|assert\s+.+failed", re.IGNORECASE), BugType.ASSERTION),
    (re.compile(r"TimeoutError|Timeout|timed?\s*out", re.IGNORECASE), BugType.TIMEOUT),
    (re.compile(r"PermissionError|Permission denied|EACCES", re.IGNORECASE), BugType.PERMISSION),
    (re.compile(r"FileNotFoundError|No such file|ENOENT", re.IGNORECASE), BugType.NOT_FOUND),
    (re.compile(
        r"RuntimeError|ValueError|KeyError|IndexError|ZeroDivisionError|"
        r"OverflowError|RecursionError|StopIteration|OSError|IOError|"
        r"NameError|UnboundLocalError",
        re.IGNORECASE,
    ), BugType.RUNTIME),
]


# ─── Detection Function ───────────────────────────────────────────────────


def detect_bug_report(message: str) -> bool:
    """Detect whether a user message describes a bug.

    Uses a combination of keyword matching and regex pattern matching
    to determine if the message is reporting a bug or error.

    Args:
        message: The user's message text.

    Returns:
        True if the message likely describes a bug.
    """
    if not message or not message.strip():
        return False

    lower_msg = message.lower()

    # Check for strong pattern matches first (tracebacks, error lines)
    for pattern in _BUG_PATTERNS:
        if pattern.search(message):
            return True

    # Check for keyword matches (need at least one)
    for keyword in _BUG_KEYWORDS:
        if keyword in lower_msg:
            return True

    return False


# ─── BugAnalyzer Class ────────────────────────────────────────────────────


class BugAnalyzer:
    """Analyzes error messages and stack traces to classify bugs.

    Parses tracebacks, extracts file/line information, classifies the
    bug type, and suggests which files should be examined.
    """

    def classify_bug_type(self, error_text: str) -> BugType:
        """Classify the type of bug from error text.

        Args:
            error_text: The error message or traceback text.

        Returns:
            The classified BugType.
        """
        if not error_text or not error_text.strip():
            return BugType.UNKNOWN

        for pattern, bug_type in _ERROR_TYPE_PATTERNS:
            if pattern.search(error_text):
                return bug_type

        # Check for logic errors heuristically
        lower = error_text.lower()
        logic_hints = [
            "wrong output", "incorrect", "expected", "unexpected",
            "should be", "should have", "but got", "instead of",
            "not equal", "mismatch",
        ]
        for hint in logic_hints:
            if hint in lower:
                return BugType.LOGIC

        return BugType.UNKNOWN

    def extract_file_locations(self, error_text: str) -> List[FileLocation]:
        """Extract file paths and line numbers from traceback text.

        Args:
            error_text: The error message or traceback text.

        Returns:
            List of FileLocation objects found in the traceback.
        """
        locations: List[FileLocation] = []
        if not error_text:
            return locations

        for match in _TRACEBACK_FILE_LINE.finditer(error_text):
            filepath = match.group(1)
            line_number = int(match.group(2))
            function_name = match.group(3) if match.group(3) else None
            locations.append(FileLocation(
                filepath=filepath,
                line_number=line_number,
                function_name=function_name,
            ))

        return locations

    def extract_error_message(self, error_text: str) -> str:
        """Extract the main error message from a traceback.

        Looks for the final error line in a Python traceback, which is
        typically the most informative.

        Args:
            error_text: The full error/traceback text.

        Returns:
            The extracted error message, or the original text if no
            specific error line is found.
        """
        if not error_text or not error_text.strip():
            return ""

        # Look for the last ErrorType: message line
        error_line_pattern = re.compile(r"^(\w*(?:Error|Exception)):\s*(.+)$", re.MULTILINE)
        matches = list(error_line_pattern.finditer(error_text))
        if matches:
            last_match = matches[-1]
            return f"{last_match.group(1)}: {last_match.group(2).strip()}"

        # Fallback: return first non-empty line
        for line in error_text.strip().splitlines():
            stripped = line.strip()
            if stripped:
                return stripped

        return error_text.strip()

    def suggest_files_to_examine(self, error_text: str) -> List[str]:
        """Suggest which files should be examined based on the error.

        Extracts unique file paths from tracebacks, filtering out
        standard library and site-packages paths to focus on user code.

        Args:
            error_text: The error message or traceback text.

        Returns:
            List of file paths to examine, with user code prioritized.
        """
        locations = self.extract_file_locations(error_text)
        if not locations:
            return []

        # Filter out stdlib and site-packages, prioritize user code
        user_files: List[str] = []
        other_files: List[str] = []

        seen: set = set()
        for loc in locations:
            fp = loc.filepath
            if fp in seen:
                continue
            seen.add(fp)

            if any(skip in fp for skip in [
                "site-packages", "/lib/python", "<frozen",
                "<string>", "<module>", "importlib",
            ]):
                other_files.append(fp)
            else:
                user_files.append(fp)

        # Return user files first, then others
        return user_files + other_files

    def analyze(self, error_text: str) -> dict:
        """Perform full analysis of an error text.

        Args:
            error_text: The error message or traceback text.

        Returns:
            Dictionary with bug_type, error_message, file_locations,
            and files_to_examine.
        """
        return {
            "bug_type": self.classify_bug_type(error_text),
            "error_message": self.extract_error_message(error_text),
            "file_locations": self.extract_file_locations(error_text),
            "files_to_examine": self.suggest_files_to_examine(error_text),
        }


# ─── Reproduction Helpers ─────────────────────────────────────────────────


def _generate_reproduction_script(error_text: str, bug_type: BugType) -> str:
    """Generate a minimal reproduction script suggestion.

    Args:
        error_text: The error description or traceback.
        bug_type: The classified bug type.

    Returns:
        A string containing a suggested reproduction script template.
    """
    analyzer = BugAnalyzer()
    error_msg = analyzer.extract_error_message(error_text)
    locations = analyzer.extract_file_locations(error_text)

    lines = [
        '"""Reproduction script for the reported bug."""',
        "",
    ]

    # Add imports based on file locations
    if locations:
        lines.append("# Files involved:")
        for loc in locations:
            lines.append(f"#   {loc}")
        lines.append("")

    # Generate type-specific reproduction template
    if bug_type == BugType.IMPORT:
        module_match = re.search(r"No module named ['\"]?(\w[\w.]*)['\"]?", error_text)
        module_name = module_match.group(1) if module_match else "module_name"
        lines.extend([
            "# Reproduce the import error:",
            f"import {module_name}",
        ])
    elif bug_type == BugType.SYNTAX:
        if locations:
            loc = locations[-1]
            lines.extend([
                "# Check the syntax error at:",
                f"#   {loc.filepath}, line {loc.line_number}",
                f"# Run: python -c \"import py_compile; py_compile.compile('{loc.filepath}')\"",
            ])
        else:
            lines.extend([
                "# Run the file with syntax checking:",
                "# python -m py_compile <filename>",
            ])
    elif bug_type == BugType.TYPE:
        lines.extend([
            "# Reproduce the type error:",
            f"# Error: {error_msg}",
            "# Call the function/method with the same arguments",
            "# to trigger the type mismatch.",
        ])
    elif bug_type == BugType.ASSERTION:
        lines.extend([
            "# Reproduce the assertion failure:",
            f"# {error_msg}",
            "# Run the failing test:",
            "# python -m pytest <test_file>::<test_name> -v",
        ])
    elif bug_type == BugType.TIMEOUT:
        lines.extend([
            "# Reproduce the timeout:",
            "import signal",
            "",
            "def timeout_handler(signum, frame):",
            "    raise TimeoutError('Reproduction timed out')",
            "",
            "signal.signal(signal.SIGALRM, timeout_handler)",
            "signal.alarm(30)  # 30 second timeout",
            "",
            "# Call the function that times out here",
        ])
    else:
        lines.extend([
            "# Reproduce the error:",
            f"# {error_msg}",
            "",
            "# TODO: Add the minimal code to trigger this error",
        ])

    if locations:
        last_loc = locations[-1]
        lines.extend([
            "",
            f"# Start by examining: {last_loc.filepath}",
            f"# Around line: {last_loc.line_number}",
        ])

    return "\n".join(lines)


def _suggest_fix_approach(bug_type: BugType, error_text: str) -> str:
    """Suggest a fix approach based on bug type and error text.

    Args:
        bug_type: The classified bug type.
        error_text: The error description or traceback.

    Returns:
        A string describing the suggested debugging approach.
    """
    approaches = {
        BugType.SYNTAX: (
            "Fix the syntax error: check for missing colons, parentheses, "
            "brackets, or incorrect indentation at the reported line. "
            "Run `python -m py_compile <file>` to verify the fix."
        ),
        BugType.IMPORT: (
            "Resolve the import error: check that the module is installed "
            "(`pip list | grep <module>`), verify the import path is correct, "
            "and ensure __init__.py files exist for package imports."
        ),
        BugType.TYPE: (
            "Fix the type error: check the types of arguments being passed, "
            "verify function signatures match their call sites, and add "
            "type checks or conversions where needed."
        ),
        BugType.LOGIC: (
            "Debug the logic error: add print/logging statements to trace "
            "the actual values vs expected values. Write a minimal test case "
            "that demonstrates the incorrect behavior."
        ),
        BugType.RUNTIME: (
            "Fix the runtime error: check for None values, out-of-range "
            "indices, missing dictionary keys, or division by zero. Add "
            "defensive checks at the error location."
        ),
        BugType.TIMEOUT: (
            "Address the timeout: profile the slow code path, check for "
            "infinite loops or excessive recursion, and consider adding "
            "timeouts or pagination for large data operations."
        ),
        BugType.ASSERTION: (
            "Fix the assertion failure: compare actual vs expected values, "
            "check if the test expectations are correct, and trace the code "
            "path that produces the unexpected result."
        ),
        BugType.PERMISSION: (
            "Fix the permission error: check file/directory permissions, "
            "verify the process has the required access rights, and ensure "
            "paths are writable."
        ),
        BugType.NOT_FOUND: (
            "Fix the not-found error: verify the file/path exists, check "
            "for typos in file names, and ensure working directory is correct."
        ),
        BugType.UNKNOWN: (
            "Investigate the error: read the full error message carefully, "
            "search for the error message online, check recent changes that "
            "might have introduced the issue, and write a minimal reproduction."
        ),
    }
    return approaches.get(bug_type, approaches[BugType.UNKNOWN])


# ─── ReproduceFirstWorkflow Class ─────────────────────────────────────────


class ReproduceFirstWorkflow:
    """Orchestrates the Reproduce-First Debugging workflow.

    Workflow steps:
    1. detect_bug() — determine if the user is reporting a bug
    2. generate_reproduction() — create a reproduction plan
    3. analyze_error() — classify and parse the error
    4. suggest_approach() — recommend a fix strategy

    Usage:
        workflow = ReproduceFirstWorkflow()
        if workflow.detect_bug(message):
            result = workflow.run(message)
            print(result.summary())
    """

    def __init__(self) -> None:
        self.analyzer = BugAnalyzer()

    def detect_bug(self, message: str) -> bool:
        """Detect whether the message describes a bug.

        Args:
            message: The user's message text.

        Returns:
            True if the message likely describes a bug.
        """
        return detect_bug_report(message)

    def generate_reproduction(self, error_text: str) -> ReproductionResult:
        """Generate a reproduction plan for the reported bug.

        Args:
            error_text: The error description or traceback.

        Returns:
            A ReproductionResult with reproduction script and analysis.
        """
        bug_type = self.analyzer.classify_bug_type(error_text)
        locations = self.analyzer.extract_file_locations(error_text)
        error_msg = self.analyzer.extract_error_message(error_text)
        script = _generate_reproduction_script(error_text, bug_type)
        approach = _suggest_fix_approach(bug_type, error_text)

        return ReproductionResult(
            bug_confirmed=False,  # Not yet confirmed until reproduction runs
            reproduction_script=script,
            error_output=error_msg,
            suggested_fix_approach=approach,
            bug_type=bug_type,
            file_locations=locations,
        )

    def analyze_error(self, error_text: str) -> dict:
        """Analyze an error message or traceback.

        Args:
            error_text: The error text to analyze.

        Returns:
            Analysis dictionary from BugAnalyzer.analyze().
        """
        return self.analyzer.analyze(error_text)

    def suggest_approach(self, analysis: dict) -> str:
        """Suggest a fix approach based on analysis results.

        Args:
            analysis: The analysis dict from analyze_error().

        Returns:
            A string describing the suggested approach.
        """
        bug_type = analysis.get("bug_type", BugType.UNKNOWN)
        error_msg = analysis.get("error_message", "")
        return _suggest_fix_approach(bug_type, error_msg)

    def run(self, message: str) -> Optional[ReproductionResult]:
        """Run the full workflow on a user message.

        Orchestrates: detect_bug -> generate_reproduction -> analyze ->
        suggest_approach.

        Args:
            message: The user's message text.

        Returns:
            A ReproductionResult if a bug was detected, None otherwise.
        """
        if not self.detect_bug(message):
            return None

        result = self.generate_reproduction(message)
        analysis = self.analyze_error(message)
        result.suggested_fix_approach = self.suggest_approach(analysis)
        return result


# ─── Debug Prompt Generator ───────────────────────────────────────────────


def generate_debug_prompt(
    error_text: str,
    context: Optional[str] = None,
    previous_attempts: Optional[List[str]] = None,
) -> str:
    """Generate a structured debugging prompt for the LLM.

    Creates a prompt that guides the LLM through the reproduce-first
    debugging workflow, providing structured context about the error.

    Args:
        error_text: The error message or traceback.
        context: Optional additional context about the bug.
        previous_attempts: Optional list of previously attempted fixes.

    Returns:
        A structured debugging prompt string.
    """
    analyzer = BugAnalyzer()
    bug_type = analyzer.classify_bug_type(error_text)
    error_msg = analyzer.extract_error_message(error_text)
    locations = analyzer.extract_file_locations(error_text)
    files_to_examine = analyzer.suggest_files_to_examine(error_text)

    sections: List[str] = []

    # Header
    sections.append("# Reproduce-First Debugging Task")
    sections.append("")

    # Error summary
    sections.append("## Error Summary")
    sections.append(f"Bug Type: {bug_type.value}")
    sections.append(f"Error: {error_msg}")
    sections.append("")

    # File locations
    if locations:
        sections.append("## Error Locations")
        for loc in locations:
            sections.append(f"- {loc}")
        sections.append("")

    # Files to examine
    if files_to_examine:
        sections.append("## Files to Examine")
        for fp in files_to_examine:
            sections.append(f"- {fp}")
        sections.append("")

    # Context
    if context:
        sections.append("## Additional Context")
        sections.append(context)
        sections.append("")

    # Previous attempts
    if previous_attempts:
        sections.append("## Previous Fix Attempts (all failed)")
        for i, attempt in enumerate(previous_attempts, 1):
            sections.append(f"{i}. {attempt}")
        sections.append("")
        sections.append(
            "IMPORTANT: The above approaches did not work. "
            "Try a fundamentally different approach."
        )
        sections.append("")

    # Instructions
    sections.append("## Debugging Instructions")
    sections.append("Follow these steps in order:")
    sections.append("1. REPRODUCE: First, reproduce the error to confirm it")
    sections.append("2. ANALYZE: Read the full error trace carefully")
    sections.append("3. LOCATE: Find the exact code causing the error")
    sections.append("4. FIX: Make the smallest possible fix")
    sections.append("5. VERIFY: Run tests to confirm the fix works")
    sections.append("")

    # Approach suggestion
    approach = _suggest_fix_approach(bug_type, error_text)
    sections.append("## Suggested Approach")
    sections.append(approach)

    return "\n".join(sections)
