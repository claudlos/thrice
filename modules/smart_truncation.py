"""
Smart Terminal Output Truncation — Keep first 10K + last 40K chars.

Current behavior: terminal output is hard-cut at ~100K characters, losing
all context about what happened at the start of the output.

This module implements a smarter truncation strategy:
  - Keep the first 10,000 characters (command start, initial errors, headers)
  - Keep the last 40,000 characters (final output, results, error traces)
  - Insert a clear truncation marker in between with line/char counts
  - Preserve stderr patterns (tracebacks, error messages) wherever they appear
  - Total: ~50K chars max (well within context budget)

Integration point: run_agent.py (~line 4818)

Usage:
    from new_files.smart_truncation import truncate_output

    result = truncate_output(huge_output)
    # Returns first 10K + marker + last 40K
"""

import re

# ─── Default limits ─────────────────────────────────────────────────────────
DEFAULT_HEAD_CHARS = 10_000
DEFAULT_TAIL_CHARS = 40_000
DEFAULT_MAX_CHARS = 50_000

# ─── Truncation marker template ────────────────────────────────────────────
_TRUNCATION_MARKER = (
    "\n\n"
    "╔══════════════════════════════════════════════════════════════╗\n"
    "║  TRUNCATED: {omitted_lines:,} lines ({omitted_chars:,} chars) omitted   ║\n"
    "║  Showing: first {head_lines:,} lines + last {tail_lines:,} lines"
    " of {total_lines:,} total  ║\n"
    "╚══════════════════════════════════════════════════════════════╝"
    "\n\n"
)

# ─── Stderr / error patterns to preserve ───────────────────────────────────
_STDERR_PATTERNS = [
    re.compile(r"^Traceback \(most recent call last\):$", re.MULTILINE),
    re.compile(r"^\w+Error:.*$", re.MULTILINE),
    re.compile(r"^\w+Exception:.*$", re.MULTILINE),
    re.compile(r"^ERROR[:!].*$", re.MULTILINE),
    re.compile(r"^FATAL[:!].*$", re.MULTILINE),
    re.compile(r"^FAILED.*$", re.MULTILINE),
    re.compile(r"^error\[E\d+\]:.*$", re.MULTILINE),  # Rust errors
    re.compile(r"^\s+at .+:\d+:\d+$", re.MULTILINE),  # JS stack trace lines
    re.compile(r"^panic:.*$", re.MULTILINE),  # Go panics
    re.compile(r"^warning:.*$", re.MULTILINE | re.IGNORECASE),
]


def _find_line_boundary(text: str, pos: int, direction: str = "before") -> int:
    """Find the nearest newline boundary near pos.

    Args:
        text: The full text.
        pos: Character position to search from.
        direction: "before" finds the last newline before pos,
                   "after" finds the first newline after pos.

    Returns:
        Character position of the line boundary.
    """
    if direction == "before":
        idx = text.rfind("\n", 0, pos)
        return idx + 1 if idx >= 0 else 0
    else:
        idx = text.find("\n", pos)
        return idx + 1 if idx >= 0 else len(text)


def _extract_stderr_lines(text: str) -> list[str]:
    """Extract lines matching stderr/error patterns from the middle section.

    Returns at most 50 lines to avoid bloating output.
    """
    error_lines = []
    for pattern in _STDERR_PATTERNS:
        for match in pattern.finditer(text):
            line_start = text.rfind("\n", 0, match.start())
            line_start = line_start + 1 if line_start >= 0 else 0
            line_end = text.find("\n", match.end())
            line_end = line_end if line_end >= 0 else len(text)
            error_lines.append(text[line_start:line_end].strip())

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for line in error_lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    return unique[:50]


def truncate_output(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    *,
    head_chars: int = DEFAULT_HEAD_CHARS,
    tail_chars: int = DEFAULT_TAIL_CHARS,
    preserve_stderr: bool = True,
) -> str:
    """Truncate long output, preserving head, tail, and optionally stderr.

    Args:
        text: The full output string.
        max_chars: Maximum total output size. If text is shorter, returns as-is.
        head_chars: Number of characters to keep from the start.
        tail_chars: Number of characters to keep from the end.
        preserve_stderr: If True, extract and include error patterns from
                         the truncated middle section.

    Returns:
        Truncated output with a marker showing what was omitted,
        or the original text if it's within limits.
    """
    if len(text) <= max_chars:
        return text

    # Find clean line boundaries
    head_end = _find_line_boundary(text, head_chars, "before")
    # Ensure we keep at least half the requested head
    if head_end < head_chars // 2:
        head_end = head_chars

    tail_start = _find_line_boundary(text, len(text) - tail_chars, "after")
    # Ensure we keep at least half the requested tail
    if tail_start > len(text) - tail_chars // 2:
        tail_start = len(text) - tail_chars

    # Prevent overlap
    if head_end >= tail_start:
        return text[:max_chars]

    head_section = text[:head_end]
    tail_section = text[tail_start:]
    middle_section = text[head_end:tail_start]

    # Calculate stats
    omitted_chars = len(middle_section)
    omitted_lines = middle_section.count("\n")
    total_lines = text.count("\n") + 1
    head_lines = head_section.count("\n") + 1
    tail_lines = tail_section.count("\n") + 1

    # Build marker
    marker = _TRUNCATION_MARKER.format(
        omitted_lines=omitted_lines,
        omitted_chars=omitted_chars,
        head_lines=head_lines,
        tail_lines=tail_lines,
        total_lines=total_lines,
    )

    # Optionally extract stderr from the truncated middle
    stderr_section = ""
    if preserve_stderr:
        stderr_lines = _extract_stderr_lines(middle_section)
        if stderr_lines:
            stderr_section = (
                "\n[Preserved errors from truncated section:]\n"
                + "\n".join(stderr_lines)
                + "\n"
            )

    result = head_section + marker + stderr_section + tail_section
    # Final hard cap: ensure output never exceeds max_chars after adding markers
    if len(result) > max_chars:
        result = result[:max_chars]
    return result


def truncate_for_budget(
    text: str,
    available_chars: int,
    *,
    head_ratio: float = 0.2,
    preserve_stderr: bool = True,
) -> str:
    """Truncate output to fit within a specific character budget.

    Adapts head/tail split to available space.

    Args:
        text: The full output string.
        available_chars: Maximum characters allowed.
        head_ratio: Fraction of available space for the head (0.0-0.5).
        preserve_stderr: Whether to preserve stderr patterns.

    Returns:
        Truncated output fitting within available_chars.
    """
    if len(text) <= available_chars:
        return text

    # Reserve space for marker (~300 chars)
    usable = max(available_chars - 300, available_chars // 2)
    head_chars = int(usable * head_ratio)
    tail_chars = usable - head_chars

    return truncate_output(
        text,
        max_chars=available_chars,
        head_chars=head_chars,
        tail_chars=tail_chars,
        preserve_stderr=preserve_stderr,
    )
