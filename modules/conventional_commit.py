"""Conventional Commits validator + generator for Hermes Agent (Thrice).

Conventional Commits spec (v1.0.0, abridged):

    <type>[optional scope][!]: <description>

    [optional body]

    [optional footer(s)]

Valid ``<type>`` values (extended): ``feat``, ``fix``, ``docs``, ``style``,
``refactor``, ``perf``, ``test``, ``build``, ``ci``, ``chore``, ``revert``.

Used by ``auto_commit`` to gate auto-generated commit messages - bad
messages get blocked before they land in history.

Example::

    from conventional_commit import validate, suggest_type

    result = validate("feat(cron): add SM-1 state machine")
    assert result.ok

    # Inspect a diff and propose a type.
    diff = "diff --git a/test_foo.py ..."
    t = suggest_type(diff)  # -> "test"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

VALID_TYPES = frozenset({
    "feat", "fix", "docs", "style", "refactor", "perf",
    "test", "build", "ci", "chore", "revert",
})

# Breaking-change types (always emit `!` or `BREAKING CHANGE:` footer).
_BREAKING_TYPES = frozenset({"feat", "fix", "refactor"})

# <type>(<scope>)!: <description>
_HEADER_RE = re.compile(
    r"""^
    (?P<type>[a-z]+)
    (?:\((?P<scope>[^)]+)\))?
    (?P<bang>!)?
    :\s
    (?P<description>.+)
    $""",
    re.VERBOSE,
)

# Max header length per spec (soft) - we hard-cap at 72 chars to match git conventions.
MAX_HEADER = 72


@dataclass
class ValidationResult:
    """Outcome of validating a commit message."""

    ok: bool
    type: Optional[str] = None
    scope: Optional[str] = None
    breaking: bool = False
    description: Optional[str] = None
    body: Optional[str] = None
    footers: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def format_errors(self) -> str:
        return "; ".join(self.errors)


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate(message: str) -> ValidationResult:
    """Check whether ``message`` follows the Conventional Commits spec."""
    errors: List[str] = []
    if not message or not message.strip():
        return ValidationResult(ok=False, errors=["empty message"])

    lines = message.splitlines()
    header = lines[0]
    rest = lines[1:]

    if len(header) > MAX_HEADER:
        errors.append(f"header too long: {len(header)} > {MAX_HEADER}")

    m = _HEADER_RE.match(header)
    if not m:
        errors.append(
            "header does not match '<type>[(<scope>)][!]: <description>'"
        )
        return ValidationResult(ok=False, errors=errors)

    ctype = m.group("type")
    scope = m.group("scope")
    bang = m.group("bang") == "!"
    description = m.group("description").strip()

    if ctype not in VALID_TYPES:
        errors.append(
            f"unknown type '{ctype}' (expected one of: "
            f"{', '.join(sorted(VALID_TYPES))})"
        )
    if description.endswith("."):
        errors.append("description must not end with a period")
    if description and not description[0].islower():
        errors.append("description should start lowercase (conventional style)")

    # Blank line separates header / body / footer groups.
    body_lines: List[str] = []
    footer_lines: List[str] = []
    if rest:
        if rest[0].strip() != "":
            errors.append("second line must be blank (body/footer separator)")
        # Strip the (expected) single leading blank, then split the rest into
        # paragraphs.  If the LAST paragraph is all "Token: value" / "Token
        # #ref" lines, it's the footer block.
        after_blank = rest[1:] if rest and rest[0].strip() == "" else rest
        paragraphs = _split_paragraphs(after_blank)
        if paragraphs and _looks_like_footers(paragraphs[-1]):
            footer_lines = paragraphs.pop()
        body_lines = _join_paragraphs(paragraphs)

    body_text: Optional[str] = "\n".join(body_lines).strip() or None
    breaking = bang or any(
        fl.startswith("BREAKING CHANGE:") or fl.startswith("BREAKING-CHANGE:")
        for fl in footer_lines
    )
    if bang and ctype not in _BREAKING_TYPES:
        errors.append(
            f"'{ctype}' cannot be marked breaking with '!' "
            f"(use one of: {', '.join(sorted(_BREAKING_TYPES))})"
        )

    return ValidationResult(
        ok=not errors,
        type=ctype,
        scope=scope,
        breaking=breaking,
        description=description,
        body=body_text,
        footers=footer_lines,
        errors=errors,
    )


def _split_paragraphs(lines: Sequence[str]) -> List[List[str]]:
    paragraphs: List[List[str]] = [[]]
    for line in lines:
        if line.strip() == "":
            if paragraphs[-1]:
                paragraphs.append([])
        else:
            paragraphs[-1].append(line)
    if paragraphs and not paragraphs[-1]:
        paragraphs.pop()
    return paragraphs


def _join_paragraphs(paragraphs: Sequence[List[str]]) -> List[str]:
    out: List[str] = []
    for i, para in enumerate(paragraphs):
        if i > 0:
            out.append("")
        out.extend(para)
    return out


# "Token: value" (tokens may not have spaces) *or* the two spec-defined
# special tokens "BREAKING CHANGE" / "BREAKING-CHANGE".
_FOOTER_TOKEN_RE = re.compile(
    r"^(?:BREAKING[ -]CHANGE|[A-Z][A-Za-z-]+):\s"
)
_FOOTER_ISSUE_RE = re.compile(r"^[A-Z][A-Za-z-]+ #")


def _looks_like_footers(lines: List[str]) -> bool:
    if not lines:
        return False
    return all(
        _FOOTER_TOKEN_RE.match(line) or _FOOTER_ISSUE_RE.match(line)
        for line in lines
    )


# ---------------------------------------------------------------------------
# Type suggestion
# ---------------------------------------------------------------------------

def suggest_type(diff: str, paths: Optional[Sequence[str]] = None) -> str:
    """Heuristically infer the best Conventional Commits type from a diff.

    Priority rules (first match wins):

    1. Any path under ``.github/workflows/`` or ``.circleci/`` -> ``ci``.
    2. Any path under ``tests/`` or starting with ``test_`` -> ``test``.
    3. Only ``.md`` / ``.rst`` / ``.txt`` touched -> ``docs``.
    4. Only ``requirements*``, ``pyproject.toml``, ``package.json``,
       ``Cargo.toml`` touched -> ``build``.
    5. Diff hunks touch tests AND source -> ``fix``.
    6. New files only (``new file mode``) with non-test paths -> ``feat``.
    7. Else ``refactor``.
    """
    raw_paths = list(paths) if paths else _paths_from_diff(diff)
    # Normalize separators so Windows-style backslashes don't defeat the
    # ``startswith("tests/")`` checks below.
    paths = [p.replace("\\", "/") for p in raw_paths]

    def any_under(prefix: str) -> bool:
        return any(p.startswith(prefix) or "/" + prefix in p for p in paths)

    def all_suffix(suffixes: Sequence[str]) -> bool:
        return bool(paths) and all(
            any(p.endswith(s) for s in suffixes) for p in paths
        )

    if any_under(".github/workflows") or any_under(".circleci"):
        return "ci"

    test_touch = any(
        p.startswith("tests/") or "/tests/" in p or p.split("/")[-1].startswith("test_")
        for p in paths
    )
    src_touch = any(
        not (p.startswith("tests/") or "/tests/" in p or
             p.split("/")[-1].startswith("test_"))
        for p in paths
    )

    if test_touch and not src_touch:
        return "test"
    if all_suffix((".md", ".rst", ".txt")):
        return "docs"
    if all_suffix(("requirements.txt", "pyproject.toml", "package.json",
                   "Cargo.toml", "go.mod")):
        return "build"
    if test_touch and src_touch:
        return "fix"
    if _has_new_file(diff) and not test_touch:
        return "feat"
    return "refactor"


_DIFF_PATH_RE = re.compile(r"^diff --git a/(?P<a>\S+) b/(?P<b>\S+)", re.MULTILINE)


def _paths_from_diff(diff: str) -> List[str]:
    return sorted({m.group("b") for m in _DIFF_PATH_RE.finditer(diff)})


def _has_new_file(diff: str) -> bool:
    return "new file mode" in diff


# ---------------------------------------------------------------------------
# Message generation
# ---------------------------------------------------------------------------

def generate(
    ctype: str,
    description: str,
    *,
    scope: Optional[str] = None,
    breaking: bool = False,
    body: Optional[str] = None,
    footers: Optional[Sequence[str]] = None,
) -> str:
    """Build a Conventional Commits message.  Does NOT validate; pair with
    ``validate()`` to enforce rules."""
    header = ctype
    if scope:
        header += f"({scope})"
    if breaking:
        header += "!"
    header += f": {description.strip()}"
    parts = [header]
    if body:
        parts += ["", body.strip()]
    if footers:
        parts += ["", "\n".join(f.strip() for f in footers)]
    return "\n".join(parts)
