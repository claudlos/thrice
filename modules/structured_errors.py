"""
Structured Error Types — Smarter retry decisions, fewer error loops.

Instead of returning plain text errors that the model must parse, this module
provides structured error classification with:
  - ErrorType enum for categorical classification
  - ToolError dataclass with error_type, recoverable flag, and hints
  - classify_error() for automatic pattern-based classification
  - format_error_result() for JSON-serialized error output

Integration point: model_tools.py (~line 440)

Usage:
    from new_files.structured_errors import classify_error, format_error_result, ToolError

    try:
        result = execute_tool(...)
    except Exception as e:
        tool_error = classify_error(e, tool_name="patch")
        return format_error_result(tool_error)
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

# ─── Error Type Enum ────────────────────────────────────────────────────────


class ErrorType(str, Enum):
    """Categorical error types for tool results."""
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    SYNTAX_ERROR = "syntax_error"
    TIMEOUT = "timeout"
    VALIDATION_FAILED = "validation_failed"
    MATCH_FAILED = "match_failed"
    NETWORK_ERROR = "network_error"
    RATE_LIMITED = "rate_limited"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    COMMAND_NOT_FOUND = "command_not_found"
    PROCESS_ERROR = "process_error"
    IMPORT_ERROR = "import_error"
    PARSE_ERROR = "parse_error"
    TOO_LARGE = "too_large"
    AUTH_ERROR = "auth_error"
    ENCODING_ERROR = "encoding_error"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


# ─── ToolError Dataclass ───────────────────────────────────────────────────


@dataclass
class ToolError:
    """A structured tool error with classification and recovery guidance."""
    message: str
    error_type: ErrorType
    recoverable: bool
    hint: Optional[str] = None
    tool_name: str = ""
    original_exception: Optional[str] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Serialize to dict, omitting empty optional fields."""
        d = {
            "error": self.message,
            "error_type": self.error_type.value,
            "recoverable": self.recoverable,
        }
        if self.hint:
            d["hint"] = self.hint
        if self.tool_name:
            d["tool_name"] = self.tool_name
        return d

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())


# ─── Error Classification Patterns ─────────────────────────────────────────
# Each entry: (regex_pattern, ErrorType, recoverable, default_hint)

_ERROR_PATTERNS: list[tuple[re.Pattern, ErrorType, bool, str]] = [
    # File / path not found
    (
        re.compile(
            r"(file|path|directory).*not found|no such file|FileNotFoundError|ENOENT|"
            r"does not exist|cannot find",
            re.I,
        ),
        ErrorType.NOT_FOUND,
        True,
        "Use search_files to locate the correct path before retrying.",
    ),
    # Permission denied
    (
        re.compile(r"permission denied|EACCES|PermissionError|not writable|read.only", re.I),
        ErrorType.PERMISSION_DENIED,
        False,
        "This file/directory is not writable. Try a different location or check permissions.",
    ),
    # Unique match not found (patch tool)
    (
        re.compile(
            r"unique match not found|multiple matches|no match found|ambiguous match|"
            r"could not find.*to replace|old_string.*not found",
            re.I,
        ),
        ErrorType.MATCH_FAILED,
        True,
        "Read the file first with read_file to see current content, "
        "then use a more specific/unique search string.",
    ),
    # Syntax error
    (
        re.compile(
            r"syntax error|SyntaxError|IndentationError|unexpected token|"
            r"unexpected end|unterminated|parse error.*syntax",
            re.I,
        ),
        ErrorType.SYNTAX_ERROR,
        True,
        "The edit introduced a syntax error. Read the file around the error line and fix.",
    ),
    # Timeout
    (
        re.compile(r"timeout|timed out|TimeoutError|deadline exceeded|ETIMEDOUT", re.I),
        ErrorType.TIMEOUT,
        True,
        "Operation timed out. Try with a higher timeout, or break into smaller steps.",
    ),
    # Validation failed
    (
        re.compile(r"validation.*fail|invalid.*argument|invalid.*parameter|ValueError", re.I),
        ErrorType.VALIDATION_FAILED,
        True,
        "Invalid input. Check the parameter values and constraints.",
    ),
    # Network error
    (
        re.compile(
            r"connection refused|ConnectionError|network|ECONNREFUSED|DNS|"
            r"resolve.*fail|ECONNRESET|EHOSTUNREACH|socket",
            re.I,
        ),
        ErrorType.NETWORK_ERROR,
        True,
        "Network error. Verify the URL/host is correct and the service is running.",
    ),
    # Rate limit
    (
        re.compile(r"rate limit|429|too many requests|throttl|quota exceeded", re.I),
        ErrorType.RATE_LIMITED,
        True,
        "Rate limited. Wait a moment before retrying.",
    ),
    # Resource exhaustion
    (
        re.compile(
            r"out of memory|MemoryError|OOM|resource exhausted|disk full|"
            r"no space left|ENOMEM|ENOSPC",
            re.I,
        ),
        ErrorType.RESOURCE_EXHAUSTED,
        False,
        "System resource exhausted. Try a smaller operation or free up resources.",
    ),
    # Command not found
    (
        re.compile(r"command not found|not recognized|No such command|not installed", re.I),
        ErrorType.COMMAND_NOT_FOUND,
        True,
        "Command not available. Check spelling or install the required package.",
    ),
    # Process error
    (
        re.compile(
            r"exit code [1-9]|non-zero exit|process.*failed|CalledProcessError|"
            r"returned error|exited with",
            re.I,
        ),
        ErrorType.PROCESS_ERROR,
        True,
        "Command failed. Check the error output and fix the underlying issue.",
    ),
    # Import error
    (
        re.compile(r"ImportError|ModuleNotFoundError|No module named|cannot import", re.I),
        ErrorType.IMPORT_ERROR,
        True,
        "Missing module. Install it: pip install <module> or npm install <package>.",
    ),
    # Parse error (JSON, XML, etc.)
    (
        re.compile(
            r"JSONDecodeError|json.*error|invalid.*json|malformed|decode error|"
            r"XML.*error|YAML.*error",
            re.I,
        ),
        ErrorType.PARSE_ERROR,
        True,
        "Data format error. Check the input format and try again.",
    ),
    # Too large
    (
        re.compile(
            r"file too large|exceeds.*limit|too many (lines|characters|bytes)|"
            r"content.*too long|payload too large",
            re.I,
        ),
        ErrorType.TOO_LARGE,
        True,
        "Content exceeds size limit. Use offset/limit to read a section, or split the operation.",
    ),
    # Auth error
    (
        re.compile(r"401|403|unauthorized|forbidden|authentication.*fail|auth.*error", re.I),
        ErrorType.AUTH_ERROR,
        False,
        "Authentication or authorization failed. Check credentials or permissions.",
    ),
    # Encoding error
    (
        re.compile(r"UnicodeDecodeError|encoding.*error|codec.*can't|binary file", re.I),
        ErrorType.ENCODING_ERROR,
        False,
        "File encoding error. This may be a binary file — use a different approach.",
    ),
    # Conflict (e.g., git merge conflict)
    (
        re.compile(r"conflict|merge.*fail|already exists|EEXIST", re.I),
        ErrorType.CONFLICT,
        True,
        "Resource conflict. Resolve the conflict manually or use a different name/path.",
    ),
]


# ─── Tool-specific hint overrides ──────────────────────────────────────────
_TOOL_HINT_OVERRIDES: dict[str, dict[ErrorType, str]] = {
    "patch": {
        ErrorType.NOT_FOUND: (
            "Use search_files to find the file, then read_file to verify content before patching."
        ),
        ErrorType.MATCH_FAILED: (
            "Read the file with read_file to see its CURRENT content. "
            "Your old_string must match EXACTLY (including whitespace and comments)."
        ),
    },
    "read_file": {
        ErrorType.NOT_FOUND: "Use search_files(target='files', pattern='*filename*') to locate the file.",
    },
    "terminal": {
        ErrorType.COMMAND_NOT_FOUND: (
            "Check if the command is installed: 'which <cmd>' or 'apt list --installed | grep <cmd>'."
        ),
        ErrorType.TIMEOUT: (
            "Command timed out. Increase timeout parameter, or run in background with background=true."
        ),
    },
    "search_files": {
        ErrorType.NOT_FOUND: "Check the directory path. Use target='files' to list available files.",
    },
    "write_file": {
        ErrorType.PERMISSION_DENIED: "Check directory permissions. Try writing to a different location.",
    },
}


# ─── Public API ─────────────────────────────────────────────────────────────


def classify_error(
    error: Union[Exception, str],
    tool_name: str = "",
) -> ToolError:
    """Classify an error into a structured ToolError.

    Pattern-matches the error message against known error types, then
    applies tool-specific hint overrides if applicable.

    Args:
        error: The exception object or error message string.
        tool_name: The tool that produced the error (for context-specific hints).

    Returns:
        ToolError with classified type, recoverability, and hint.
    """
    if isinstance(error, Exception):
        error_message = f"{type(error).__name__}: {error}"
        original_exception = type(error).__name__
    else:
        error_message = str(error)
        original_exception = None

    # Pattern matching
    error_type = ErrorType.UNKNOWN
    recoverable = True
    hint = "Examine the error message. If transient, retry. If structural, try a different approach."

    for pattern, etype, is_recoverable, default_hint in _ERROR_PATTERNS:
        if pattern.search(error_message):
            error_type = etype
            recoverable = is_recoverable
            hint = default_hint
            break

    # Apply tool-specific overrides
    if tool_name and tool_name in _TOOL_HINT_OVERRIDES:
        overrides = _TOOL_HINT_OVERRIDES[tool_name]
        if error_type in overrides:
            hint = overrides[error_type]

    return ToolError(
        message=error_message,
        error_type=error_type,
        recoverable=recoverable,
        hint=hint,
        tool_name=tool_name,
        original_exception=original_exception,
    )


def format_error_result(tool_error: ToolError) -> str:
    """Format a ToolError as a JSON string for tool result output.

    Args:
        tool_error: The classified error.

    Returns:
        JSON string with error, error_type, recoverable, and hint fields.
    """
    return tool_error.to_json()


def enrich_error_result(
    result_str: str,
    tool_name: str = "",
) -> str:
    """Enrich an existing tool result string with structured error fields.

    If the result contains error info (JSON with "error" field, or error keywords),
    adds error_type, recoverable, and hint fields.

    Args:
        result_str: Existing tool result string.
        tool_name: Tool that produced the result.

    Returns:
        Enriched result string (JSON) or original string if not an error.
    """
    # Try parsing as JSON first
    try:
        data = json.loads(result_str)
        if isinstance(data, dict) and "error" in data:
            error_msg = str(data["error"])
            tool_error = classify_error(error_msg, tool_name)
            data["error_type"] = tool_error.error_type.value
            data["recoverable"] = tool_error.recoverable
            if tool_error.hint:
                data["hint"] = tool_error.hint
            return json.dumps(data)
    except (json.JSONDecodeError, TypeError):
        pass

    # Check if plain text looks like an error
    error_keywords = ("error", "failed", "exception", "traceback", "denied", "not found")
    if any(kw in result_str.lower() for kw in error_keywords):
        tool_error = classify_error(result_str, tool_name)
        return format_error_result(tool_error)

    # Not an error — return as-is
    return result_str
