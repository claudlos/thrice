"""
Post-Edit Self-Reflection — Review changes for correctness before moving on.

Research from SWE-agent shows that injecting a self-review step after edits
catches 10-20% more bugs. This module generates reflection prompts that
ask the model to review its own changes for:
  - Correctness and logic errors
  - Edge cases and boundary conditions
  - Style consistency with surrounding code
  - Missing imports, declarations, or dependencies
  - Off-by-one errors and typos

Integration point: After successful patch/write_file operations in run_agent.py

Usage:
    from new_files.self_reflection import generate_reflection_prompt, ReflectionResult

    prompt = generate_reflection_prompt(changes=[
        EditChange(file="main.py", before="old code", after="new code", description="fix bug")
    ])
    # Inject prompt into conversation for self-review
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ReviewSeverity(str, Enum):
    """Severity levels for review findings."""
    CRITICAL = "critical"   # Will break functionality
    WARNING = "warning"     # Potential issue
    INFO = "info"           # Style or minor concern
    OK = "ok"               # No issues found


@dataclass
class EditChange:
    """Represents a single file edit for review."""
    file: str
    before: str = ""
    after: str = ""
    description: str = ""
    language: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None

    @property
    def lines_changed(self) -> int:
        """Number of lines in the 'after' section."""
        return self.after.count("\n") + 1 if self.after else 0

    @property
    def is_new_file(self) -> bool:
        """Whether this is a new file creation (no 'before')."""
        return not self.before and bool(self.after)


@dataclass
class ReviewFinding:
    """A single finding from self-review."""
    severity: ReviewSeverity
    category: str       # e.g., "logic", "edge_case", "style", "import", "typo"
    description: str
    file: str = ""
    line: Optional[int] = None
    suggestion: str = ""


@dataclass
class ReflectionResult:
    """Result of a self-reflection review."""
    findings: list[ReviewFinding] = field(default_factory=list)
    overall_severity: ReviewSeverity = ReviewSeverity.OK
    summary: str = ""
    should_revise: bool = False

    @property
    def has_critical(self) -> bool:
        return any(f.severity == ReviewSeverity.CRITICAL for f in self.findings)

    @property
    def has_warnings(self) -> bool:
        return any(f.severity == ReviewSeverity.WARNING for f in self.findings)

    @property
    def finding_count(self) -> int:
        return len(self.findings)

    def to_dict(self) -> dict:
        return {
            "overall_severity": self.overall_severity.value,
            "should_revise": self.should_revise,
            "summary": self.summary,
            "finding_count": self.finding_count,
            "findings": [
                {
                    "severity": f.severity.value,
                    "category": f.category,
                    "description": f.description,
                    "file": f.file,
                    "line": f.line,
                    "suggestion": f.suggestion,
                }
                for f in self.findings
            ],
        }


# ─── Reflection prompt templates ───────────────────────────────────────────

_REFLECTION_HEADER = """Review your recent changes for correctness before proceeding.

CHECK FOR:
1. **Logic errors** — Does the code do what was intended?
2. **Edge cases** — Empty inputs, None values, boundary conditions?
3. **Missing imports** — Are all new symbols properly imported?
4. **Style consistency** — Does it match the surrounding code style?
5. **Off-by-one errors** — Array indices, loop bounds, slicing?
6. **Error handling** — Are new failure modes handled?
7. **Side effects** — Could this break existing functionality?

"""

_SINGLE_FILE_TEMPLATE = """FILE: {file}
{description}
{location}

BEFORE:
```
{before}
```

AFTER:
```
{after}
```
"""

_NEW_FILE_TEMPLATE = """NEW FILE: {file}
{description}

CONTENT:
```
{after}
```
"""

_REFLECTION_FOOTER = """
If you find any issues, fix them now. If everything looks correct, proceed with the task.
Be brief — only mention actual problems, not things that are fine.
"""

_MULTI_EDIT_FOOTER = """
Review the interaction between these {count} changes — could they conflict or
create inconsistencies? Fix any issues before proceeding.
"""


def generate_reflection_prompt(
    changes: list[EditChange],
    task_context: str = "",
    max_display_lines: int = 100,
) -> str:
    """Generate a self-reflection prompt for the model to review its changes.

    Args:
        changes: List of EditChange objects describing what was modified.
        task_context: Optional context about what the task was trying to achieve.
        max_display_lines: Maximum lines to show per change (truncate if larger).

    Returns:
        A prompt string to inject into the conversation.
    """
    if not changes:
        return ""

    parts = [_REFLECTION_HEADER]

    if task_context:
        parts.append(f"TASK CONTEXT: {task_context}\n\n")

    for change in changes:
        before = change.before
        after = change.after

        # Truncate very large diffs
        if before.count("\n") > max_display_lines:
            before_lines = before.splitlines()
            before = "\n".join(before_lines[:max_display_lines // 2])
            before += f"\n... [{len(before_lines) - max_display_lines} lines truncated] ...\n"
            before += "\n".join(before_lines[-max_display_lines // 2:])

        if after.count("\n") > max_display_lines:
            after_lines = after.splitlines()
            after = "\n".join(after_lines[:max_display_lines // 2])
            after += f"\n... [{len(after_lines) - max_display_lines} lines truncated] ...\n"
            after += "\n".join(after_lines[-max_display_lines // 2:])

        description = f"({change.description})" if change.description else ""
        location = ""
        if change.line_start is not None:
            if change.line_end is not None:
                location = f"Lines {change.line_start}-{change.line_end}"
            else:
                location = f"Line {change.line_start}"

        if change.is_new_file:
            parts.append(_NEW_FILE_TEMPLATE.format(
                file=change.file,
                description=description,
                after=after,
            ))
        else:
            parts.append(_SINGLE_FILE_TEMPLATE.format(
                file=change.file,
                description=description,
                location=location,
                before=before,
                after=after,
            ))

    # Footer
    if len(changes) > 1:
        parts.append(_MULTI_EDIT_FOOTER.format(count=len(changes)))
    else:
        parts.append(_REFLECTION_FOOTER)

    return "\n".join(parts)


def should_trigger_reflection(
    changes: list[EditChange],
    min_lines_changed: int = 5,
    always_for_multi_file: bool = True,
) -> bool:
    """Determine if a reflection step should be triggered.

    Small, trivial edits may not need reflection. This heuristic decides
    when the overhead is worth it.

    Args:
        changes: List of changes made.
        min_lines_changed: Minimum total lines changed to trigger.
        always_for_multi_file: Always reflect when multiple files are changed.

    Returns:
        True if reflection should be triggered.
    """
    if not changes:
        return False

    # Always reflect on multi-file edits
    if always_for_multi_file and len(changes) > 1:
        return True

    # Check total lines changed
    total_lines = sum(c.lines_changed for c in changes)
    if total_lines >= min_lines_changed:
        return True

    # Check if any new files were created
    if any(c.is_new_file for c in changes):
        return True

    return False
